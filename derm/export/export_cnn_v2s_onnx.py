"""
Export the two EfficientNetV2-S+CBAM feature extractors to ONNX (1280-d).

Overwrites the SAME asset filenames the app already uses, so no Dart change is
needed (the pipeline builds the feature vector with dynamic length 18+1280):
  export/cnn_lesion_feats.onnx  <- cnn_v2s_best.pth
  export/cnn_dehair_feats.onnx  <- cnn_v2s_dehair_best.pth

Run from the repository root: python3 -m derm.export.export_cnn_v2s_onnx
"""
import sys
from pathlib import Path


import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from derm.pipeline import cnn_v2s  # defines EfficientNetV2SCBAM, registers v2s globals on cnn
from derm import paths

IMG = 224
JOBS = [
    (str(paths.CNN_V2S_BEST), str(paths.MODELS / "cnn_lesion_feats.onnx")),
    (str(paths.CNN_V2S_DEHAIR_BEST), str(paths.MODELS / "cnn_dehair_feats.onnx")),
]


def export_one(weights, out):
    model = cnn_v2s.EfficientNetV2SCBAM(n_classes=3)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.fc = nn.Identity()  # 1280-d penultimate features
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
