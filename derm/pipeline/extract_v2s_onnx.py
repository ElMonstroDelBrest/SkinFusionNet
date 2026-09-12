"""Extract canonical V2S single-pass features with deployed ONNX encoders.

This is the CPU-only extractor for the repository's canonical model:
EfficientNetV2-S+CBAM, lesion/dehair views, no TTA.

Outputs are aligned row-for-row with ``artifacts/image_metadata.csv`` and keep
the filename on each row. Downstream code should select columns named
``cnn_*`` rather than assuming that every non-Class column is numeric.

The dehair view has known missing files. For those rows, this extractor applies
the same DullRazor operation on the raw image on the fly and marks
``view_source=dehair_fallback_raw``.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd

from derm import paths
from derm.analysis import internal_validity_m1 as iv


IMG_SIZE = 224
BATCH = 64
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
HAIR_K = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))


def log(msg: str) -> None:
    print(msg, flush=True)


def load_metadata() -> pd.DataFrame:
    p = paths.ARTIFACTS / "image_metadata.csv"
    if p.exists():
        return pd.read_csv(p)
    meta, _ = iv.build_image_metadata()
    return meta


def dehair_bgr(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, HAIR_K)
    _, hair = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(bgr, hair, 1, cv2.INPAINT_TELEA)


def raw_image_path(row: pd.Series) -> Path:
    stem = Path(str(row["file"])).stem
    root = paths.MERGED / str(row["split_original"]) / str(row["class_name"])
    for ext in (".jpg", ".jpeg", ".png"):
        p = root / f"{stem}{ext}"
        if p.exists():
            return p
    matches = sorted(root.glob(stem + ".*"))
    if not matches:
        raise FileNotFoundError(f"raw image missing for {row['file']}")
    return matches[0]


def image_for_row(row: pd.Series, view: str) -> tuple[np.ndarray, str]:
    stem = Path(str(row["file"])).stem
    split = str(row["split_original"])
    cls = str(row["class_name"])
    if view == "lesion":
        p = paths.LESION / split / cls / f"{stem}.png"
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"lesion image missing/unreadable: {p}")
        return bgr, "lesion"
    if view != "dehair":
        raise ValueError(view)

    p = paths.DEHAIR / split / cls / f"{stem}.jpg"
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is not None:
        return bgr, "dehair_file"

    raw = cv2.imread(str(raw_image_path(row)), cv2.IMREAD_COLOR)
    if raw is None:
        raise FileNotFoundError(f"raw fallback unreadable for {row['file']}")
    return dehair_bgr(raw), "dehair_fallback_raw"


def preprocess(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    x = (resized.astype(np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1), dtype=np.float32)


def write_view(view: str, out_csv: Path, model_path: Path, batch_size: int) -> None:
    meta = load_metadata()
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    rows: list[dict[str, object]] = []
    batch: list[np.ndarray] = []
    batch_meta: list[tuple[pd.Series, str]] = []
    n_fallback = 0
    n_dim: int | None = None

    out_csv.parent.mkdir(exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer: csv.DictWriter | None = None

        def flush() -> None:
            nonlocal batch, batch_meta, writer, n_dim
            if not batch:
                return
            x = np.stack(batch, axis=0)
            feats = sess.run(None, {"input": x})[0]
            if n_dim is None:
                n_dim = int(feats.shape[1])
                fieldnames = [
                    "row_index",
                    "file",
                    "split_original",
                    "class_name",
                    "view_source",
                ] + [f"cnn_{i}" for i in range(n_dim)] + ["Class"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            assert writer is not None
            for feat, (row, source) in zip(feats, batch_meta):
                rec: dict[str, object] = {
                    "row_index": int(row["row_index"]),
                    "file": row["file"],
                    "split_original": row["split_original"],
                    "class_name": row["class_name"],
                    "view_source": source,
                    "Class": int(row["class_id"]),
                }
                for j, val in enumerate(feat):
                    rec[f"cnn_{j}"] = f"{float(val):.6f}"
                writer.writerow(rec)
            batch = []
            batch_meta = []

        for i, row in meta.iterrows():
            bgr, source = image_for_row(row, view)
            if source == "dehair_fallback_raw":
                n_fallback += 1
            batch.append(preprocess(bgr))
            batch_meta.append((row, source))
            if len(batch) >= batch_size:
                flush()
            if (i + 1) % 1000 == 0:
                log(f"{view}: {i + 1}/{len(meta)} fallback={n_fallback}")
        flush()

    log(f"wrote {out_csv} rows={len(meta)} dim={n_dim} fallback={n_fallback}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--view", choices=["lesion", "dehair", "both"], default="both")
    p.add_argument("--batch-size", type=int, default=BATCH)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.view in ("lesion", "both"):
        write_view(
            "lesion",
            paths.CNN_V2S_FEATS_CSV,
            paths.MODELS / "cnn_lesion_feats.onnx",
            args.batch_size,
        )
    if args.view in ("dehair", "both"):
        write_view(
            "dehair",
            paths.CNN_V2S_DEHAIR_CSV,
            paths.MODELS / "cnn_dehair_feats.onnx",
            args.batch_size,
        )


if __name__ == "__main__":
    main()
