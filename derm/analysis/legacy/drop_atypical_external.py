"""Compare binary and three-class training on an external evaluation set.

Extract handcrafted and 1280-D CNN features with the deployed single-pass ONNX
models. Fit both LightGBM ensembles from the same internal sample, excluding
atypical cases only for binary training. Compare external melanoma-versus-nevus
ROC-AUC and sensitivity. All input images remain local.

Run: python -m derm.analysis.legacy.drop_atypical_external [N_internal]"""
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
N_INTERNAL = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
FRAC = {"nevus": 0.40, "melanoma": 0.35, "atypical": 0.25}
T_MEL = 0.188


def log(*a):
    print(*a, flush=True)


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


def featurize(items, unet, cnnL, cnnD, tag):
    HC, FL, FD, ok = [], [], [], []
    for i, p in enumerate(items):
        r = process(p, unet, cnnL, cnnD)
        if r is None:
            ok.append(False); continue
        HC.append(r[0]); FL.append(r[1]); FD.append(r[2]); ok.append(True)
        if (i + 1) % 250 == 0:
            log(f"  {tag} {i+1}/{len(items)}")
    return np.array(HC), np.array(FL), np.array(FD), np.array(ok)


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


def fit_full(hc, fL, fD, y):
    return (mk().fit(np.concatenate([hc, fL], 1), y),
            mk().fit(np.concatenate([hc, fD], 1), y))


def pmel_ext(mA, mC, hc, fL, fD):
    P = (mA.predict_proba(np.concatenate([hc, fL], 1)) +
         mC.predict_proba(np.concatenate([hc, fD], 1))) / 2.0
    return P[:, 1]                      # Class column 1 is melanoma in both conditions.


def youden(yb, p):
    bt, bj = 0.5, -1
    for t in np.unique(p):
        se = ((p >= t) & (yb == 1)).sum() / max((yb == 1).sum(), 1)
        sp = ((p < t) & (yb == 0)).sum() / max((yb == 0).sum(), 1)
        if se + sp - 1 > bj:
            bj, bt = se + sp - 1, t
    se = ((p >= bt) & (yb == 1)).sum() / (yb == 1).sum()
    sp = ((p < bt) & (yb == 0)).sum() / (yb == 0).sum()
    return bt, se, sp


def main():
    log(f"N_internal={N_INTERNAL}")
    unet = E.sess(E.MODELS / "unet.onnx")
    cnnL = E.sess(E.MODELS / "cnn_lesion_feats.onnx")
    cnnD = E.sess(E.MODELS / "cnn_dehair_feats.onnx")

    ip, iy = collect_internal()
    log(f"internal: {len(iy)} (nevus={int((iy==0).sum())}, mel={int((iy==1).sum())}, atyp={int((iy==2).sum())})")
    iHC, iFL, iFD, iok = featurize(ip, unet, cnnL, cnnD, "int")
    iy = iy[iok]

    ep, ey = [], []
    for folder, cid in EXT_FOLDERS.items():
        for p in sorted((EXT / folder).iterdir()):
            if p.suffix.lower() in E.IMG_EXTS:
                ep.append(p); ey.append(cid)
    ey = np.array(ey)
    eHC, eFL, eFD, eok = featurize(ep, unet, cnnL, cnnD, "ext")
    ey = ey[eok]
    yb = (ey == 1).astype(int)
    nmel, nnev = int((ey == 1).sum()), int((ey == 0).sum())
    log(f"external: {len(ey)} (nevus={nnev}, mel={nmel})\n")

    log(f"{'setup':<26} {'AUC mél-vs-nevus':>17} {'sens@0.188':>11} {'sens@Youden':>12} {'spec':>7}")
    results = {}
    # Three-class condition: use the full sample.
    mA, mC = fit_full(iHC, iFL, iFD, iy)
    p3 = pmel_ext(mA, mC, eHC, eFL, eFD)
    # Binary condition: use the nevus/melanoma subset of the same sample.
    m = iy != 2
    mA2, mC2 = fit_full(iHC[m], iFL[m], iFD[m], iy[m])
    p2 = pmel_ext(mA2, mC2, eHC, eFL, eFD)

    for name, p in (("3-classes (nevus/mel/atyp)", p3), ("2-classes (sans atypical)", p2)):
        auc = roc_auc_score(yb, p)
        tp = int(((p >= T_MEL) & (ey == 1)).sum())
        ty, sey, spy = youden(yb, p)
        log(f"{name:<26} {auc:>17.4f} {f'{tp}/{nmel}':>11} {sey:>12.3f} {spy:>7.3f}")
        results[name] = auc

    d = results["2-classes (sans atypical)"] - results["3-classes (nevus/mel/atyp)"]
    log(f"\nΔ AUC OOD (2cl − 3cl) = {d:+.4f}")
    log("single-pass sans TTA (≈ déployé à -1 pt) ; n=19 mél -> Δ à lire avec prudence.")


if __name__ == "__main__":
    main()
