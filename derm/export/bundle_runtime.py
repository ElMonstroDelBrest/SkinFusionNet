"""Deployed-bundle inference: conv nets on the RTX 50 GPU, LightGBM on CPU.

U-Net and the two EfficientNetV2-S extractors are the ONNX graphs converted
with onnx2torch and executed by PyTorch 2.11+cu128 (native sm_120 / cuDNN 9
kernels, TF32 off). Tree heads stay on ONNX Runtime CPU — they are faster
there and keep the baked-in StandardScaler.

Decode, DullRazor and handcraft run on a CPU thread pool so the GPU is not
waiting on JPEG/inpaint.

`eval_external.sess` remains CPU-only for the byte-exact parity gate.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from derm.export.cuda_libs import configure_nvidia_libs

configure_nvidia_libs()

import cv2
import numpy as np
import onnxruntime as ort

from derm.export.eval_external import (
    C,
    MEAN,
    STD,
    U,
    WORK,
    decode_bgr,
    dehair_bgr,
    handcraft18,
)


def _session_options(*, intra_op: int) -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_mem_pattern = True
    options.enable_cpu_mem_arena = True
    options.intra_op_num_threads = max(1, intra_op)
    options.inter_op_num_threads = 1
    return options


def make_lgbm_session(model_path: Path) -> ort.InferenceSession:
    threads = min(8, os.cpu_count() or 8)
    return ort.InferenceSession(
        str(model_path),
        _session_options(intra_op=threads),
        providers=["CPUExecutionProvider"],
    )


def _unet_nchw(bgr512: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr512, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (U, U), interpolation=cv2.INTER_LINEAR)
    return (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)


def _cnn_nchw(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (C, C), interpolation=cv2.INTER_LINEAR)
    scaled = resized.astype(np.float32) / 255.0
    return ((scaled - MEAN) / STD).transpose(2, 0, 1).astype(np.float32)


def _masks_from_logits(logits: np.ndarray) -> np.ndarray:
    probs = 1.0 / (1.0 + np.exp(-logits[:, 0]))
    binary = (probs > 0.5).astype(np.uint8) * 255
    masks = np.empty((binary.shape[0], WORK, WORK), dtype=np.uint8)
    for index, mask in enumerate(binary):
        masks[index] = cv2.resize(mask, (WORK, WORK), interpolation=cv2.INTER_NEAREST)
    return masks


def _decode_work(path: Path) -> np.ndarray | None:
    image = decode_bgr(path)
    if image is None:
        return None
    return cv2.resize(image, (WORK, WORK), interpolation=cv2.INTER_LINEAR)


def _cpu_views(bgr512: np.ndarray, mask: np.ndarray) -> dict:
    coverage = float((mask > 0).mean())
    binary01 = (mask > 0).astype(np.uint8)
    lesion = cv2.bitwise_and(bgr512, bgr512, mask=mask)
    dehair = dehair_bgr(bgr512)
    handcraft = np.asarray(handcraft18(binary01, lesion), dtype=np.float32)
    return {
        "coverage": coverage,
        "handcraft": handcraft,
        "lesion_nchw": _cnn_nchw(lesion),
        "dehair_nchw": _cnn_nchw(dehair),
    }


class _TorchOnnxConv:
    """Run exported ONNX conv nets through PyTorch CUDA (sm_120 cuDNN)."""

    def __init__(
        self, models_dir: Path, batch_size: int, *, load_cnn: bool = True
    ) -> None:
        import torch
        from onnx2torch import convert

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        self.torch = torch
        self.device = torch.device("cuda")
        self.load_cnn = load_cnn
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        self.unet = convert(str(models_dir / "unet.onnx")).to(self.device).eval()
        self.cnn_lesion = None
        self.cnn_dehair = None
        if load_cnn:
            self.cnn_lesion = (
                convert(str(models_dir / "cnn_lesion_feats.onnx")).to(self.device).eval()
            )
            self.cnn_dehair = (
                convert(str(models_dir / "cnn_dehair_feats.onnx")).to(self.device).eval()
            )
        self._warmup(batch_size)

    def _warmup(self, batch_size: int) -> None:
        torch = self.torch
        n = max(1, min(batch_size, 8))
        with torch.inference_mode():
            self.unet(torch.zeros(n, 3, U, U, device=self.device))
            if self.load_cnn:
                dummy = torch.zeros(n, 3, C, C, device=self.device)
                self.cnn_lesion(dummy)
                self.cnn_dehair(dummy)
        torch.cuda.synchronize()

    def _forward(self, module, nchw: np.ndarray) -> np.ndarray:
        torch = self.torch
        tensor = torch.from_numpy(np.ascontiguousarray(nchw)).to(
            self.device, non_blocking=True
        )
        with torch.inference_mode():
            output = module(tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
            host = output.detach().float().cpu().numpy()
        return host

    def unet_logits(self, nchw: np.ndarray) -> np.ndarray:
        return self._forward(self.unet, nchw)

    def lesion_features(self, nchw: np.ndarray) -> np.ndarray:
        if self.cnn_lesion is None:
            raise RuntimeError("CNN extractors were not loaded")
        return self._forward(self.cnn_lesion, nchw)

    def dehair_features(self, nchw: np.ndarray) -> np.ndarray:
        if self.cnn_dehair is None:
            raise RuntimeError("CNN extractors were not loaded")
        return self._forward(self.cnn_dehair, nchw)


class BundleEngine:
    """Batched deployed-bundle inference."""

    def __init__(
        self,
        models_dir: Path,
        *,
        batch_size: int = 16,
        workers: int = 8,
        device: str = "auto",
        load_cnn: bool = True,
    ) -> None:
        cv2.setNumThreads(1)
        self.batch_size = max(1, batch_size)
        self.workers = max(1, workers)
        self.load_cnn = load_cnn
        configure_nvidia_libs()
        want_gpu = device in {"cuda", "auto"}
        self.conv = None
        self.device = "cpu"
        self.conv_backend = "ort-cpu"
        if want_gpu:
            try:
                self.conv = _TorchOnnxConv(
                    models_dir, self.batch_size, load_cnn=load_cnn
                )
                self.device = "cuda"
                self.conv_backend = "pytorch-cu128-sm120"
            except Exception as exc:
                print(f"GPU conv backend unavailable ({exc!r}); using CPU ORT", flush=True)
        if self.conv is None:
            self.unet = make_lgbm_session(models_dir / "unet.onnx")
            if load_cnn:
                self.cnn_lesion = make_lgbm_session(models_dir / "cnn_lesion_feats.onnx")
                self.cnn_dehair = make_lgbm_session(models_dir / "cnn_dehair_feats.onnx")
        self.lgbm_a = None
        self.lgbm_c = None
        if load_cnn:
            self.lgbm_a = make_lgbm_session(models_dir / "lgbm_a.onnx")
            self.lgbm_c = make_lgbm_session(models_dir / "lgbm_c.onnx")
        self._pool = ThreadPoolExecutor(max_workers=self.workers)
        if load_cnn:
            dummy_t = np.zeros((min(self.batch_size, 4), 1298), np.float32)
            self.lgbm_a.run(None, {"input": dummy_t})
            self.lgbm_c.run(None, {"input": dummy_t})

    def describe(self) -> str:
        cnn = "deployed" if self.load_cnn else "unet-only"
        lgbm = "CPUExecutionProvider" if self.load_cnn else "skipped"
        return (
            f"device={self.device}  backend={self.conv_backend}  "
            f"batch={self.batch_size}  workers={self.workers}  "
            f"cnn={cnn}  lgbm={lgbm}"
        )

    def close(self) -> None:
        self._pool.shutdown(wait=False)

    def _run_unet(self, nchw: np.ndarray) -> np.ndarray:
        if self.conv is not None:
            return self.conv.unet_logits(nchw)
        return self.unet.run(None, {"input": nchw})[0]

    def _run_cnn_lesion(self, nchw: np.ndarray) -> np.ndarray:
        if self.conv is not None:
            return self.conv.lesion_features(nchw)
        return self.cnn_lesion.run(None, {"input": nchw})[0]

    def _run_cnn_dehair(self, nchw: np.ndarray) -> np.ndarray:
        if self.conv is not None:
            return self.conv.dehair_features(nchw)
        return self.cnn_dehair.run(None, {"input": nchw})[0]

    def infer_paths(self, paths: list[Path]) -> list[dict | BaseException]:
        if not self.load_cnn:
            raise RuntimeError("infer_paths requires load_cnn=True")
        results: list[dict | BaseException] = []
        for start in range(0, len(paths), self.batch_size):
            results.extend(self._infer_chunk(paths[start : start + self.batch_size]))
        return results

    def extract_paths(self, paths: list[Path]) -> list[dict | BaseException]:
        """Same preprocess/CNN path, returns 18-d handcraft + two 1280-d embeddings."""
        if not self.load_cnn:
            raise RuntimeError("extract_paths requires load_cnn=True")
        results: list[dict | BaseException] = []
        for start in range(0, len(paths), self.batch_size):
            results.extend(self._extract_chunk(paths[start : start + self.batch_size]))
        return results

    def preprocess_paths(self, paths: list[Path]) -> list[dict | BaseException]:
        """U-Net + DullRazor + handcraft + CNN-ready nchw tensors. No deployed CNN."""
        results: list[dict | BaseException] = []
        for start in range(0, len(paths), self.batch_size):
            results.extend(self._preprocess_chunk(paths[start : start + self.batch_size]))
        return results

    def _infer_chunk(self, paths: list[Path]) -> list[dict | BaseException]:
        decoded = list(self._pool.map(_decode_work, paths))
        valid_index = [i for i, image in enumerate(decoded) if image is not None]
        out: list[dict | BaseException] = [
            RuntimeError("decode_none") if image is None else {}
            for image in decoded
        ]
        if not valid_index:
            return out
        images = [decoded[i] for i in valid_index]
        unet_batch = np.stack([_unet_nchw(image) for image in images], axis=0)
        logits = self._run_unet(unet_batch)
        masks = _masks_from_logits(logits)
        views = list(self._pool.map(lambda pair: _cpu_views(*pair), zip(images, masks)))
        lesion_batch = np.ascontiguousarray(
            np.stack([row["lesion_nchw"] for row in views], axis=0), dtype=np.float32
        )
        dehair_batch = np.ascontiguousarray(
            np.stack([row["dehair_nchw"] for row in views], axis=0), dtype=np.float32
        )
        feat_l = self._run_cnn_lesion(lesion_batch)
        feat_d = self._run_cnn_dehair(dehair_batch)
        handcraft = np.stack([row["handcraft"] for row in views], axis=0)
        x_a = np.concatenate([handcraft, feat_l], axis=1).astype(np.float32)
        x_c = np.concatenate([handcraft, feat_d], axis=1).astype(np.float32)
        p_a = np.asarray(self.lgbm_a.run(None, {"input": x_a})[1], dtype=np.float64)
        p_c = np.asarray(self.lgbm_c.run(None, {"input": x_c})[1], dtype=np.float64)
        ensemble = (p_a + p_c) / 2.0
        for local, source in enumerate(valid_index):
            probs = ensemble[local]
            out[source] = {
                "p_nevus": float(probs[0]),
                "p_mel": float(probs[1]),
                "p_atyp": float(probs[2]),
                "argmax": int(probs.argmax()),
                "coverage": views[local]["coverage"],
                "probability_sum": float(probs.sum()),
            }
        return out

    def _extract_chunk(self, paths: list[Path]) -> list[dict | BaseException]:
        decoded = list(self._pool.map(_decode_work, paths))
        valid_index = [i for i, image in enumerate(decoded) if image is not None]
        out: list[dict | BaseException] = [
            RuntimeError("decode_none") if image is None else {}
            for image in decoded
        ]
        if not valid_index:
            return out
        images = [decoded[i] for i in valid_index]
        unet_batch = np.stack([_unet_nchw(image) for image in images], axis=0)
        logits = self._run_unet(unet_batch)
        masks = _masks_from_logits(logits)
        views = list(self._pool.map(lambda pair: _cpu_views(*pair), zip(images, masks)))
        lesion_batch = np.ascontiguousarray(
            np.stack([row["lesion_nchw"] for row in views], axis=0), dtype=np.float32
        )
        dehair_batch = np.ascontiguousarray(
            np.stack([row["dehair_nchw"] for row in views], axis=0), dtype=np.float32
        )
        feat_l = self._run_cnn_lesion(lesion_batch)
        feat_d = self._run_cnn_dehair(dehair_batch)
        for local, source in enumerate(valid_index):
            out[source] = {
                "handcraft": views[local]["handcraft"].astype(np.float32),
                "lesion": np.asarray(feat_l[local], dtype=np.float32),
                "dehair": np.asarray(feat_d[local], dtype=np.float32),
                "coverage": views[local]["coverage"],
            }
        return out

    def _preprocess_chunk(self, paths: list[Path]) -> list[dict | BaseException]:
        decoded = list(self._pool.map(_decode_work, paths))
        valid_index = [i for i, image in enumerate(decoded) if image is not None]
        out: list[dict | BaseException] = [
            RuntimeError("decode_none") if image is None else {}
            for image in decoded
        ]
        if not valid_index:
            return out
        images = [decoded[i] for i in valid_index]
        unet_batch = np.stack([_unet_nchw(image) for image in images], axis=0)
        logits = self._run_unet(unet_batch)
        masks = _masks_from_logits(logits)
        views = list(self._pool.map(lambda pair: _cpu_views(*pair), zip(images, masks)))
        for local, source in enumerate(valid_index):
            out[source] = {
                "handcraft": views[local]["handcraft"].astype(np.float32),
                "coverage": views[local]["coverage"],
                "lesion_nchw": views[local]["lesion_nchw"],
                "dehair_nchw": views[local]["dehair_nchw"],
            }
        return out
