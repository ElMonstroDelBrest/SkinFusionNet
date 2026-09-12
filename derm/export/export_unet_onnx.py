"""
Export the trained U-Net (smp.Unet, EfficientNet-B0 encoder) to ONNX.

Mirrors the model construction of infer_unet.py:48-50.
Input  : 1x3x384x384  float32 (RGB, /255 done on the app side)
Output : 1x1x384x384  raw logits  (apply sigmoid + >0.5 threshold downstream)

Run from C:\\Projet\\Stage :  python export/export_unet_onnx.py
"""
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from derm import paths

WEIGHTS = paths.UNET_BEST
OUT = paths.MODELS / "unet.onnx"
IMG_SIZE = 384


def main():
    model = smp.Unet(encoder_name="efficientnet-b0",
                     encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.eval()

    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(OUT),
        opset_version=17,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )

    # sanity check against the PyTorch output
    import onnxruntime as ort
    import numpy as np
    with torch.no_grad():
        ref = model(dummy).numpy()
    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input": dummy.numpy()})[0]
    err = float(np.abs(ref - got).max())
    print(f"wrote {OUT}  (max abs diff vs torch = {err:.2e})")
    # Zero-input is degenerate; small logit drift is fine (mask uses sigmoid>0.5).
    # Real-image parity is checked in verify_onnx.py.
    assert err < 1e-2, "ONNX U-Net diverges from PyTorch"


if __name__ == "__main__":
    main()
