"""
End-to-end evaluation of the deployed ensemble from raw JPEG images using
ONNX Runtime with preprocessing matching the Flutter pipeline:

  jpg ─┬─ U-Net(onnx) ─► mask ─► lesion-only ─┐
       │                                       ├─ 18 handcraft features ┐
       └─ DullRazor ─► dehair ─────────────────┘                        │
                              ├ CNN_lesion(onnx) 1280 → [18|1280] → LGBM_A(onnx) → P_A
                              └ CNN_dehair(onnx) 1280 → [18|1280] → LGBM_C(onnx) → P_C
                                                          mean(P_A,P_C) → 3 probs

Writes local cross-runtime golden JSON records per image (excluded from Git).

Run from the repository root:
  python3 -m derm.export.verify_serious
"""
import json
from pathlib import Path


import cv2
import numpy as np
import onnxruntime as ort

from derm.preprocess import features as F  # asymmetry_A4, border_diameter, lab_features, morpho_fractal, build_header
from derm import paths
from derm.export.model_manifest import validate_manifest

SRC = paths.MERGED / "test"
OUT_DIR = Path(__file__).resolve().parent / "golden"
CLASS_NAMES = ["nevus", "melanoma", "atypical"]
PER_CLASS = 5
U, C = 384, 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
HAIR_K = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))


def sess(p):
    return ort.InferenceSession(p, providers=["CPUExecutionProvider"])


def unet_mask(unet, rgb):
    x = cv2.resize(rgb, (U, U)).astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None]
    logits = unet.run(None, {"input": x})[0][0, 0]
    m = ((1 / (1 + np.exp(-logits))) > 0.5).astype(np.uint8)
    h, w = rgb.shape[:2]
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)


def dehair_bgr(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, HAIR_K)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(bgr, mask, 1, cv2.INPAINT_TELEA)


def cnn_feats(cnnsess, bgr_img):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (C, C)).astype(np.float32) / 255.0
    x = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None]
    return cnnsess.run(None, {"input": x})[0].ravel()


def handcraft_18(binary, lesion_bgr):
    A4 = F.asymmetry_A4(binary)
    B, D = F.border_diameter(binary)
    a_kurt, b_std, a_std, b_skew = F.lab_features(lesion_bgr, binary)
    morpho = F.morpho_fractal(binary)
    return [A4, B, D, a_kurt, b_std, a_std, b_skew] + morpho


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = paths.APP_MODELS
    manifest = validate_manifest(model_dir / "model_manifest.json", model_dir)
    unet = sess(str(model_dir / "unet.onnx"))
    cnnL = sess(str(model_dir / "cnn_lesion_feats.onnx"))
    cnnD = sess(str(model_dir / "cnn_dehair_feats.onnx"))
    lgbmA = sess(str(model_dir / "lgbm_a.onnx"))
    lgbmC = sess(str(model_dir / "lgbm_c.onnx"))

    correct = total = 0
    for cid, cname in enumerate(CLASS_NAMES):
        for ip in sorted((SRC / cname).glob("*.jpg"))[:PER_CLASS]:
            bgr = cv2.imread(str(ip))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            mask = unet_mask(unet, rgb)
            binary = (mask > 0).astype(np.uint8)
            lesion = bgr.copy()
            lesion[mask == 0] = 0
            dehair = dehair_bgr(bgr)

            hc = handcraft_18(binary, lesion)
            fL = cnn_feats(cnnL, lesion)
            fD = cnn_feats(cnnD, dehair)
            xA = np.array(hc + fL.tolist(), np.float32)[None]
            xC = np.array(hc + fD.tolist(), np.float32)[None]

            pA = np.array(lgbmA.run(None, {"input": xA})[1]).ravel()
            pC = np.array(lgbmC.run(None, {"input": xC})[1]).ravel()
            pe = (pA + pC) / 2.0
            pred = int(pe.argmax())
            correct += int(pred == cid)
            total += 1
            print(f"{cname:>9}/{ip.stem:<16} pred={CLASS_NAMES[pred]:<9} "
                  f"P={[round(float(p),3) for p in pe]}")

            payload = {
                "model_bundle_id": manifest["bundle_id"],
                "model_bundle_sha256": manifest["bundle_sha256"],
                "preprocessing_version": manifest["preprocessing"]["version"],
                "decision_policy_version": manifest["decision_policy"]["version"],
                "source_image": ip.relative_to(paths.DATA).as_posix(),
                "class_true": cid,
                "handcraft18": [float(v) for v in hc],
                "probs_A": [float(v) for v in pA],
                "probs_C": [float(v) for v in pC],
                "probs_ensemble": [float(v) for v in pe],
                "pred": pred,
            }
            (OUT_DIR / f"{cname}_{ip.stem}.json").write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )

    print(f"\nensemble accuracy on sample = {correct}/{total} = {correct / total:.2%}")
    print(f"feature order (18) = {F.build_header()[:-1]}")


if __name__ == "__main__":
    main()
