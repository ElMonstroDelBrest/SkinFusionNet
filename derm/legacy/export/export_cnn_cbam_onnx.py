"""
Export the two ResNet18+CBAM feature extractors to ONNX (512-d, fc=Identity).

Reuses the exact architecture from cnn.py (ResNet18CBAM) so the weights load
cleanly. Produces:
  models/legacy/cnn_lesion_feats_512.onnx  <- cnn_best.pth
  models/legacy/cnn_dehair_feats_512.onnx  <- cnn_dehair_best.pth

Input  : 1x3x224x224 float32 (RGB, ImageNet-normalised on the app side)
Output : 1x512       penultimate features (cnn_0 .. cnn_511)

Run from the repository root: python3 -m derm.legacy.export.export_cnn_cbam_onnx
"""
import sys
from pathlib import Path


import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from derm.legacy.pipeline import cnn  # ResNet18CBAM, make_model
from derm import paths

IMG = 224
JOBS = [
    (str(paths.CNN_BEST), str(paths.LEGACY_MODELS / "cnn_lesion_feats_512.onnx")),
    (str(paths.CNN_DEHAIR_BEST), str(paths.LEGACY_MODELS / "cnn_dehair_feats_512.onnx")),
]


def export_one(weights, out):
    model = cnn.make_model()
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.fc = nn.Identity()  # 512-d penultimate features (matches cnn.py:194)
    model.eval()

    dummy = torch.zeros(1, 3, IMG, IMG)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, out,
        opset_version=17,
        input_names=["input"], output_names=["features"],
        dynamic_axes={"input": {0: "batch"}, "features": {0: "batch"}},
    )
    with torch.no_grad():
        ref = model(dummy).numpy()
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input": dummy.numpy()})[0]
    err = float(np.abs(ref - got).max())
    print(f"wrote {out}  shape={got.shape}  max abs diff vs torch = {err:.2e}")
    assert err < 1e-3, f"{out} diverges from PyTorch"


if __name__ == "__main__":
    for w, o in JOBS:
        export_one(w, o)
