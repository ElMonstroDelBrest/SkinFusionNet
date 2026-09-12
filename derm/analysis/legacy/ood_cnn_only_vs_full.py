"""Compare full and CNN-only classifiers under external distribution shift.

Regenerate single-pass 1280-D CNN and handcrafted features on an internal sample
and the local external images. Train two LightGBM ensembles and compare external
melanoma ROC-AUC and sensitivity at the historical threshold of 0.188.
Both conditions share preprocessing and feature extraction. These refitted
models are experimental comparisons, not the archived TTA release.

Run: python -m derm.analysis.legacy.ood_cnn_only_vs_full [N_internal]"""
import sys
from pathlib import Path

import cv2
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from derm.export import eval_external as E  # decode_bgr, unet_mask, dehair_bgr, handcraft18, sess, MEAN, STD, WORK, C
from derm import paths

RNG = np.random.RandomState(0)
INT_ROOT = paths.MERGED / "train"
EXT = paths.EXTERNAL_VAL
EXT_FOLDERS = {"Nevi": 0, "Melanom": 1}
N_INTERNAL = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
N_PER = {"nevus": 0.40, "melanoma": 0.35, "atypical": 0.25}  # fractions of N_INTERNAL
T_MEL = 0.188


def log(*a): print(*a, flush=True)


def cnn_single(sess, bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    r = cv2.resize(rgb, (E.C, E.C), interpolation=cv2.INTER_LINEAR)
    x = ((r.astype(np.float32) / 255.0 - E.MEAN) / E.STD).transpose(2, 0, 1)[None]
    return sess.run(None, {"input": np.ascontiguousarray(x, np.float32)})[0].ravel()


def process(path, unet, cnnL, cnnD):
    bgr = E.decode_bgr(Path(path))
    if bgr is None:
        return None
    bgr = cv2.resize(bgr, (E.WORK, E.WORK), interpolation=cv2.INTER_LINEAR)
    mask = E.unet_mask(unet, bgr)
    binary01 = (mask > 0).astype(np.uint8)
    lesion = cv2.bitwise_and(bgr, bgr, mask=mask)
    dehair = E.dehair_bgr(bgr)
    hc = E.handcraft18(binary01, lesion)
    fL = cnn_single(cnnL, lesion)
    fD = cnn_single(cnnD, dehair)
    return np.array(hc, np.float32), fL.astype(np.float32), fD.astype(np.float32)


def collect_internal():
    paths, y = [], []
    cmap = {"nevus": 0, "melanoma": 1, "atypical": 2}
    for cname, frac in N_PER.items():
        pool = sorted((INT_ROOT / cname).glob("*.jpg"))
        k = min(int(N_INTERNAL * frac), len(pool))
        for i in RNG.choice(len(pool), k, replace=False):
            paths.append(pool[i]); y.append(cmap[cname])
    return paths, np.array(y)


def featurize(paths, unet, cnnL, cnnD, tag):
    HC, FL, FD, ok = [], [], [], []
    for i, p in enumerate(paths):
        r = process(p, unet, cnnL, cnnD)
        if r is None:
            ok.append(False); continue
        HC.append(r[0]); FL.append(r[1]); FD.append(r[2]); ok.append(True)
        if (i + 1) % 250 == 0:
            log(f"  {tag} {i+1}/{len(paths)}")
    return np.array(HC), np.array(FL), np.array(FD), np.array(ok)


def fit_ensemble(hc, fL, fD, y, use_hc):
    def mk():
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                              random_state=42, n_jobs=-1, verbose=-1, device="gpu")
    XA = np.concatenate([hc, fL], 1) if use_hc else fL
    XC = np.concatenate([hc, fD], 1) if use_hc else fD
    return mk().fit(XA, y), mk().fit(XC, y)


def predict_ens(mA, mC, hc, fL, fD, use_hc):
    XA = np.concatenate([hc, fL], 1) if use_hc else fL
    XC = np.concatenate([hc, fD], 1) if use_hc else fD
    return (mA.predict_proba(XA) + mC.predict_proba(XC)) / 2.0


def main():
    log(f"N_internal={N_INTERNAL}")
    unet = E.sess(E.MODELS / "unet.onnx")
    cnnL = E.sess(E.MODELS / "cnn_lesion_feats.onnx")
    cnnD = E.sess(E.MODELS / "cnn_dehair_feats.onnx")

    ip, iy = collect_internal()
    log(f"internal sample: {len(iy)} (nevus={int((iy==0).sum())}, mel={int((iy==1).sum())}, atyp={int((iy==2).sum())})")
    iHC, iFL, iFD, iok = featurize(ip, unet, cnnL, cnnD, "int")
    iy = iy[iok]
    log(f"internal featurized: {len(iy)}")

    ep, ey = [], []
    for folder, cid in EXT_FOLDERS.items():
        for p in sorted((EXT / folder).iterdir()):
            if p.suffix.lower() in E.IMG_EXTS:
                ep.append(p); ey.append(cid)
    ey = np.array(ey)
    eHC, eFL, eFD, eok = featurize(ep, unet, cnnL, cnnD, "ext")
    ey = ey[eok]
    log(f"external featurized: {len(ey)} (nevus={int((ey==0).sum())}, mel={int((ey==1).sum())})")

    log("\n=== OOD melanoma detection (P_mel), full vs cnn_only ===")
    yb = (ey == 1).astype(int)
    for use_hc, name in ((True, "full  [handcraft|CNN]"), (False, "cnn_only [CNN]")):
        mA, mC = fit_ensemble(iHC, iFL, iFD, iy, use_hc)
        P = predict_ens(mA, mC, eHC, eFL, eFD, use_hc)
        pmel = P[:, 1]
        auc = roc_auc_score(yb, pmel)
        tp = int(((pmel >= T_MEL) & (ey == 1)).sum()); nmel = int((ey == 1).sum())
        tn = int(((pmel < T_MEL) & (ey == 0)).sum()); nnev = int((ey == 0).sum())
        log(f"  {name:<22} AUC={auc:.4f}  sens@0.188={tp/nmel:.3f} ({tp}/{nmel})  "
            f"spec@0.188={tn/nnev:.3f} ({tn}/{nnev})")


if __name__ == "__main__":
    main()
