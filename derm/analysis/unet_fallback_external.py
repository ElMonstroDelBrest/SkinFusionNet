"""Evaluate fallback masks for low-coverage external U-Net segmentations.

Fit one three-class LightGBM ensemble on internal features using the original
U-Net masks. On external cases below a coverage threshold, replace the mask with
a central disk or the full frame. Recompute lesion and handcrafted features;
the dehair branch is independent of the mask. Compare melanoma probabilities
and ROC-AUC with the unchanged-mask condition. A central disk assumes a centered
lesion, and any observed improvement requires further validation.

Run: python -m derm.analysis.unet_fallback_external [N_internal]"""
import sys
from pathlib import Path

import cv2
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from derm.export import eval_external as E
from derm import paths

RNG = np.random.RandomState(0)
INT_ROOT = paths.MERGED / "train"
EXT = paths.EXTERNAL_VAL
EXT_FOLDERS = {"Nevi": 0, "Melanom": 1}
N_INTERNAL = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
FRAC = {"nevus": 0.40, "melanoma": 0.35, "atypical": 0.25}
T_MEL = 0.188
W = E.WORK                                   # 512


def log(*a):
    print(*a, flush=True)


def cnn_single(sess, bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    r = cv2.resize(rgb, (E.C, E.C), interpolation=cv2.INTER_LINEAR)
    x = ((r.astype(np.float32) / 255.0 - E.MEAN) / E.STD).transpose(2, 0, 1)[None]
    return sess.run(None, {"input": np.ascontiguousarray(x, np.float32)})[0].ravel()


def disk_mask(frac_r=0.30):
    m = np.zeros((W, W), np.uint8)
    cv2.circle(m, (W // 2, W // 2), int(frac_r * W), 255, -1)
    return m


def feats_for_mask(bgr, mask, cnnL):
    """Return 18 handcrafted and 1280 lesion CNN features for the supplied mask."""
    binary01 = (mask > 0).astype(np.uint8)
    lesion = cv2.bitwise_and(bgr, bgr, mask=mask)
    hc = np.array(E.handcraft18(binary01, lesion), np.float32)
    fL = cnn_single(cnnL, lesion).astype(np.float32)
    return hc, fL


def featurize_internal(items, unet, cnnL, cnnD):
    HC, FL, FD, ok = [], [], [], []
    for i, p in enumerate(items):
        bgr = E.decode_bgr(Path(p))
        if bgr is None:
            ok.append(False); continue
        bgr = cv2.resize(bgr, (W, W), interpolation=cv2.INTER_LINEAR)
        mask = E.unet_mask(unet, bgr)
        hc, fL = feats_for_mask(bgr, mask, cnnL)
        fD = cnn_single(cnnD, E.dehair_bgr(bgr)).astype(np.float32)
        HC.append(hc); FL.append(fL); FD.append(fD); ok.append(True)
        if (i + 1) % 250 == 0:
            log(f"  int {i+1}/{len(items)}")
    return np.array(HC), np.array(FL), np.array(FD), np.array(ok)


def featurize_external(items, unet, cnnL, cnnD, variants):
    """Return mask-independent dehair features and per-variant handcrafted/lesion features.
    Each variant is (name, cov_min, kind), with kind in {'unet', 'disk', 'frame'}."""
    out = {v[0]: {"HC": [], "FL": []} for v in variants}
    FD, COV, ok = [], [], []
    disk = disk_mask()
    for i, p in enumerate(items):
        bgr = E.decode_bgr(Path(p))
        if bgr is None:
            ok.append(False); continue
        bgr = cv2.resize(bgr, (W, W), interpolation=cv2.INTER_LINEAR)
        umask = E.unet_mask(unet, bgr)
        cov = float((umask > 0).mean())
        FD.append(cnn_single(cnnD, E.dehair_bgr(bgr)).astype(np.float32))
        COV.append(cov); ok.append(True)
        for name, cov_min, kind in variants:
            if cov >= cov_min:
                mask = umask
            elif kind == "disk":
                mask = disk
            elif kind == "frame":
                mask = np.full((W, W), 255, np.uint8)
            else:
                mask = umask
            hc, fL = feats_for_mask(bgr, mask, cnnL)
            out[name]["HC"].append(hc); out[name]["FL"].append(fL)
        if (i + 1) % 25 == 0:
            log(f"  ext {i+1}/{len(items)}")
    return out, np.array(FD), np.array(COV), np.array(ok)


def collect_internal():
    items, y = [], []
    cmap = {"nevus": 0, "melanoma": 1, "atypical": 2}
    for cname, frac in FRAC.items():
        pool = sorted((INT_ROOT / cname).glob("*.jpg"))
        k = min(int(N_INTERNAL * frac), len(pool))
        for i in RNG.choice(len(pool), k, replace=False):
            items.append(pool[i]); y.append(cmap[cname])
    return items, np.array(y)


def mk():
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                          random_state=42, n_jobs=-1, verbose=-1)


def main():
    log(f"N_internal={N_INTERNAL}")
    unet = E.sess(E.MODELS / "unet.onnx")
    cnnL = E.sess(E.MODELS / "cnn_lesion_feats.onnx")
    cnnD = E.sess(E.MODELS / "cnn_dehair_feats.onnx")

    ip, iy = collect_internal()
    iHC, iFL, iFD, iok = featurize_internal(ip, unet, cnnL, cnnD)
    iy = iy[iok]
    log(f"internal featurized: {len(iy)}")
    mA = mk().fit(np.concatenate([iHC, iFL], 1), iy)
    mC = mk().fit(np.concatenate([iHC, iFD], 1), iy)

    ep, ey = [], []
    for folder, cid in EXT_FOLDERS.items():
        for p in sorted((EXT / folder).iterdir()):
            if p.suffix.lower() in E.IMG_EXTS:
                ep.append(p); ey.append(cid)
    ey = np.array(ey)

    variants = [("U-Net (baseline)", 0.0, "unet"),
                ("disque @cov<0.02", 0.02, "disk"),
                ("disque @cov<0.05", 0.05, "disk"),
                ("image @cov<0.05", 0.05, "frame")]
    feats, eFD, eCOV, eok = featurize_external(ep, unet, cnnL, cnnD, variants)
    ey = ey[eok]
    yb = (ey == 1).astype(int)
    nmel, nnev = int((ey == 1).sum()), int((ey == 0).sum())
    n_trig = {v[0]: int((eCOV < v[1]).sum()) for v in variants}
    log(f"\nexterne: {len(ey)} (nevus={nnev}, mel={nmel}) ; "
        f"cov<0.02={int((eCOV<0.02).sum())}  cov<0.05={int((eCOV<0.05).sum())}\n")

    log(f"{'variante':<20} {'cas-fallback':>12} {'AUC':>8} {'sens@0.188':>11} {'spec':>7}")
    base_pmel = None
    for name, cov_min, kind in variants:
        HC = np.array(feats[name]["HC"])[eok]
        FL = np.array(feats[name]["FL"])[eok]
        P = (mA.predict_proba(np.concatenate([HC, FL], 1)) +
             mC.predict_proba(np.concatenate([HC, eFD], 1))) / 2.0
        pmel = P[:, 1]
        if base_pmel is None:
            base_pmel = pmel
        auc = roc_auc_score(yb, pmel)
        tp = int(((pmel >= T_MEL) & (ey == 1)).sum())
        tn = int(((pmel < T_MEL) & (ey == 0)).sum())
        log(f"{name:<20} {n_trig[name]:>12} {auc:>8.4f} {f'{tp}/{nmel}':>11} {tn/nnev:>7.3f}")

    # Inspect probability changes for melanomas with near-zero mask coverage.
    log("\n=== mélanomes à couverture < 0.05 : p_mel baseline -> meilleure variante ===")
    files = [ep[i].name for i in range(len(ep)) if eok[i]]
    best = np.array(feats["disque @cov<0.05"]["HC"])[eok], np.array(feats["disque @cov<0.05"]["FL"])[eok]
    Pb = (mA.predict_proba(np.concatenate([best[0], best[1]], 1)) +
          mC.predict_proba(np.concatenate([best[0], eFD], 1))) / 2.0
    for i in np.where((ey == 1) & (eCOV < 0.05))[0]:
        log(f"  {files[i][:30]:<30} cov={eCOV[i]:.3f}  p_mel {base_pmel[i]:.3f} -> {Pb[i,1]:.3f} (disque)")


if __name__ == "__main__":
    main()
