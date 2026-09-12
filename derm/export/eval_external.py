"""
External (hospital) validation of the DEPLOYED ensemble, fully from raw images,
mirroring exactly what the Flutter app runs (cv_pipeline.dart +
classifier_service.dart):

  decode -> resize 512 (work)
  U-Net(384) -> sigmoid>0.5 -> mask(512, NEAREST) -> lesion-only (bitwise_and)
  DullRazor (blackhat CROSS 17, thr 10, inpaint TELEA r=1) -> dehair
  18 handcraft (features.py, computed on the 512 mask/lesion)
  EfficientNetV2-S+CBAM feats, single-pass (1280) for lesion and dehair
  xA=[hc|fL]  xC=[hc|fD]  (1298, raw; StandardScaler baked into the ONNX)
  P_A=lgbm_a(xA)  P_C=lgbm_c(xC)  ensemble = (P_A+P_C)/2

Readout = 3 normalized multiclass probabilities with class-specific thresholds
loaded from the canonical runtime model manifest.

The external set has 2 labelled classes (Nevi=nevus, Melanom=melanoma); the
model still emits 3 probabilities summing to one. We evaluate the melanoma and
nevus class probabilities as one-vs-rest scores, report the atypical flag rate,
and provide the 3-class argmax confusion.

Run from the repository root: python3 -m derm.export.eval_external
"""
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# HEIC support
import pillow_heif
from PIL import Image
pillow_heif.register_heif_opener()

from derm import paths
from derm.export.model_manifest import validate_manifest
from derm.preprocess import features as F

ROOT = paths.ROOT
DATA = paths.EXTERNAL_VAL
MODELS = paths.APP_MODELS
MANIFEST = validate_manifest(MODELS / "model_manifest.json", MODELS)
OUT_CSV = DATA / "per_image.csv"
OUT_TXT = DATA / "summary.txt"
PREVIEW_DIR = DATA / "previews"

WORK, U, C = 512, 384, 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
HAIR_K = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))

CLASS_NAMES = ["nevus", "melanoma", "atypical"]
# folder -> true class id
FOLDERS = {"Nevi": 0, "Melanom": 1}
# Release-scoped readout; never duplicate threshold values in evaluation code.
T = {
    key: float(value)
    for key, value in MANIFEST["decision_policy"]["thresholds"].items()
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


def sess(p):
    return ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])


def decode_bgr(path: Path):
    """Decode any supported image to BGR uint8 (HEIC via pillow-heif)."""
    if path.suffix.lower() in (".heic", ".heif"):
        img = Image.open(path).convert("RGB")
        rgb = np.array(img)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return bgr


def unet_mask(unet, bgr512):
    rgb = cv2.cvtColor(bgr512, cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb, (U, U), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None]
    logits = unet.run(None, {"input": x})[0][0, 0]
    m = ((1.0 / (1.0 + np.exp(-logits))) > 0.5).astype(np.uint8) * 255
    return cv2.resize(m, (WORK, WORK), interpolation=cv2.INTER_NEAREST)


def dehair_bgr(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, HAIR_K)
    _, hair = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(bgr, hair, 1, cv2.INPAINT_TELEA)


def d4(img):
    """8 dihedral-group orientations of an HxWxC array."""
    rots = [np.rot90(img, k) for k in range(4)]
    return rots + [np.fliplr(r) for r in rots]


def cnn_feats_single(cnnsess, bgr):
    """Single-pass 1280-d V2S features (canonical deployed path)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    base = cv2.resize(rgb, (C, C), interpolation=cv2.INTER_LINEAR)
    x = ((base.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
    return cnnsess.run(None, {"input": np.ascontiguousarray(x, np.float32)})[0].ravel()


def handcraft18(binary01, lesion_bgr):
    A4 = F.asymmetry_A4(binary01)
    B, D = F.border_diameter(binary01)
    a_kurt, b_std, a_std, b_skew = F.lab_features(lesion_bgr, binary01)
    morpho = F.morpho_fractal(binary01)
    return [A4, B, D, a_kurt, b_std, a_std, b_skew] + morpho


def wilson(k, n, z=1.96):
    """Wilson score 95% CI for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def roc_auc_binary(scores, ytrue):
    """ROC AUC via Mann-Whitney U (ytrue in {0,1})."""
    scores = np.asarray(scores, float)
    ytrue = np.asarray(ytrue, int)
    pos, neg = scores[ytrue == 1], scores[ytrue == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    sr = scores[order]
    i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r_pos = ranks[ytrue == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    auc = (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def main():
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    unet = sess(MODELS / "unet.onnx")
    cnnL = sess(MODELS / "cnn_lesion_feats.onnx")
    cnnD = sess(MODELS / "cnn_dehair_feats.onnx")
    lgbmA = sess(MODELS / "lgbm_a.onnx")
    lgbmC = sess(MODELS / "lgbm_c.onnx")

    rows = []
    failures = []
    preview_budget = {0: 4, 1: 4}

    for folder, cid in FOLDERS.items():
        paths = sorted(p for p in (DATA / folder).iterdir()
                       if p.suffix.lower() in IMG_EXTS)
        print(f"\n=== {folder} (true={CLASS_NAMES[cid]})  n={len(paths)} ===")
        for ip in paths:
            try:
                bgr = decode_bgr(ip)
                if bgr is None:
                    failures.append((ip.name, "decode_none"))
                    continue
                bgr = cv2.resize(bgr, (WORK, WORK), interpolation=cv2.INTER_LINEAR)
                mask = unet_mask(unet, bgr)
                coverage = float((mask > 0).mean())
                binary01 = (mask > 0).astype(np.uint8)
                lesion = cv2.bitwise_and(bgr, bgr, mask=mask)
                dehair = dehair_bgr(bgr)

                hc = handcraft18(binary01, lesion)
                fL = cnn_feats_single(cnnL, lesion)
                fD = cnn_feats_single(cnnD, dehair)
                xA = np.array(hc + fL.tolist(), np.float32)[None]
                xC = np.array(hc + fD.tolist(), np.float32)[None]

                pA = np.array(lgbmA.run(None, {"input": xA})[1]).ravel()
                pC = np.array(lgbmC.run(None, {"input": xC})[1]).ravel()
                pe = (pA + pC) / 2.0
                pred = int(pe.argmax())

                rows.append({
                    "file": ip.name, "folder": folder, "true": cid,
                    "p_nevus": float(pe[0]), "p_mel": float(pe[1]),
                    "p_atyp": float(pe[2]), "argmax": pred,
                    "coverage": coverage,
                })
                if preview_budget.get(cid, 0) > 0:
                    cv2.imwrite(str(PREVIEW_DIR / f"{folder}_{ip.stem}_lesion.png"), lesion)
                    preview_budget[cid] -= 1
                mark = "" if pred == cid else "  <-- argmax miss"
                print(f"  {ip.name[:34]:<34} P={np.round(pe,3).tolist()} cov={coverage:.2f}{mark}")
            except Exception as e:
                failures.append((ip.name, repr(e)[:80]))
                print(f"  {ip.name[:34]:<34} FAILED: {repr(e)[:60]}")

    # ---- write per-image csv ----
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "folder", "true", "p_nevus",
                                          "p_mel", "p_atyp", "argmax", "coverage"])
        w.writeheader()
        w.writerows(rows)

    # ---- metrics ----
    true = np.array([r["true"] for r in rows])
    p_mel = np.array([r["p_mel"] for r in rows])
    p_nev = np.array([r["p_nevus"] for r in rows])
    p_atyp = np.array([r["p_atyp"] for r in rows])
    argmax = np.array([r["argmax"] for r in rows])
    cov = np.array([r["coverage"] for r in rows])
    n_mel = int((true == 1).sum())
    n_nev = int((true == 0).sum())

    lines = []
    def pr(s=""):
        print(s); lines.append(s)

    pr("\n" + "=" * 70)
    pr("EXTERNAL (HOSPITAL) VALIDATION — deployed EfficientNetV2-S+CBAM single-pass ensemble")
    pr("=" * 70)
    pr(f"bundle_id = {MANIFEST['bundle_id']}")
    pr(f"bundle_sha256 = {MANIFEST['bundle_sha256']}")
    pr(f"preprocessing = {MANIFEST['preprocessing']['version']}")
    pr(f"decision_policy = {MANIFEST['decision_policy']['version']}")
    pr(f"n = {len(rows)}   (nevus={n_nev}, melanoma={n_mel})   failures={len(failures)}")
    pr(f"U-Net mask coverage: median={np.median(cov):.2f}  "
       f"empty(<1%)={int((cov<0.01).sum())}  full(>85%)={int((cov>0.85).sum())}")

    # --- melanoma class probability as an OvR score ---
    y_mel = (true == 1).astype(int)
    auc_mel = roc_auc_binary(p_mel, y_mel)
    tp = int(((p_mel >= T["melanoma"]) & (true == 1)).sum())
    fn = n_mel - tp
    tn = int(((p_mel < T["melanoma"]) & (true == 0)).sum())
    fp = n_nev - tn
    sens, sl, sh = wilson(tp, n_mel)
    spec, pl, ph = wilson(tn, n_nev)
    pr("\n--- Melanoma multiclass probability used one-vs-rest, score = P(melanoma) ---")
    pr(f"  ROC AUC (melanoma vs nevus) = {auc_mel:.4f}")
    pr(f"  @ policy T*={T['melanoma']:.3f} :")
    pr(f"    Sensibilité = {sens:.3f}  [{sl:.3f}, {sh:.3f}]   ({tp}/{n_mel} mélanomes détectés, {fn} ratés)")
    pr(f"    Spécificité = {spec:.3f}  [{pl:.3f}, {ph:.3f}]   ({tn}/{n_nev} nevi corrects, {fp} faux positifs)")

    # --- nevus class probability as an OvR score ---
    y_nev = (true == 0).astype(int)
    auc_nev = roc_auc_binary(p_nev, y_nev)
    tp_n = int(((p_nev >= T["nevus"]) & (true == 0)).sum())
    tn_n = int(((p_nev < T["nevus"]) & (true == 1)).sum())
    sens_n, _, _ = wilson(tp_n, n_nev)
    spec_n, _, _ = wilson(tn_n, n_mel)
    pr("\n--- Nevus multiclass probability used one-vs-rest, score = P(nevus) ---")
    pr(f"  ROC AUC = {auc_nev:.4f}")
    pr(f"  @ T*={T['nevus']:.3f} : Sensibilité(nevus)={sens_n:.3f}  Spécificité={spec_n:.3f}")

    # --- atypical flag rate (informational; class is a catch-all) ---
    flag_atyp_nev = int(((p_atyp >= T["atypical"]) & (true == 0)).sum())
    flag_atyp_mel = int(((p_atyp >= T["atypical"]) & (true == 1)).sum())
    pr(f"\n--- Atypical multiclass probability (flag rate @ T*={T['atypical']:.3f}) ---")
    pr(f"  nevi flagués atypical    : {flag_atyp_nev}/{n_nev}")
    pr(f"  mélanomes flagués atypical: {flag_atyp_mel}/{n_mel}")

    # --- 'not-benign' framing (mel OR atypical) for melanoma recall ---
    not_benign_mel = int((((p_mel >= T["melanoma"]) | (p_atyp >= T["atypical"])) & (true == 1)).sum())
    nb_sens, nbl, nbh = wilson(not_benign_mel, n_mel)
    pr("\n--- 'Not-benign' (mélanome sorti melanoma OU atypical) ---")
    pr(f"  Sensibilité mélanome = {nb_sens:.3f}  [{nbl:.3f}, {nbh:.3f}]  ({not_benign_mel}/{n_mel})")

    # --- 3-class argmax confusion ---
    pr("\n--- 3-class argmax confusion (rows=true, cols=pred) ---")
    pr(f"            {'nevus':>8}{'melanoma':>10}{'atypical':>10}")
    for c in (0, 1):
        r = [int(((true == c) & (argmax == k)).sum()) for k in (0, 1, 2)]
        pr(f"  {CLASS_NAMES[c]:<9}{r[0]:>8}{r[1]:>10}{r[2]:>10}")
    acc2 = float((argmax == true).mean())  # note: 'atypical' argmax always counts as miss here
    pr(f"  (argmax exact-match acc, atypical=miss = {acc2:.3f})")

    if failures:
        pr("\n--- FAILURES ---")
        for nm, err in failures:
            pr(f"  {nm}: {err}")

    OUT_TXT.write_text("\n".join(lines))
    pr(f"\nwrote {OUT_CSV}  and  {OUT_TXT}")
    pr(f"previews -> {PREVIEW_DIR}")


if __name__ == "__main__":
    main()
