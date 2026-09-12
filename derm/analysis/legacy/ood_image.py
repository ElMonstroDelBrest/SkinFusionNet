"""Compare silhouette and image-derived descriptors on an external dataset.

Fit four internal classifiers: CNN-only, CNN plus 18 silhouette descriptors,
CNN plus 16 image descriptors, and CNN plus both descriptor groups. Use the same
single-pass pipeline and internal seed as ood_cnn_only_vs_full. Report melanoma
ROC-AUC, sensitivity and specificity at the historical threshold of 0.188.

Run: python -m derm.analysis.legacy.ood_image [N_internal]"""
import sys
from pathlib import Path

import cv2
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from derm.export import eval_external as E
from derm.preprocess import image_features as IMG
from derm import paths

RNG = np.random.RandomState(0)
INT_ROOT = paths.MERGED / "train"
EXT = paths.EXTERNAL_VAL
EXT_FOLDERS = {"Nevi": 0, "Melanom": 1}
N_INTERNAL = int(sys.argv[1]) if len(sys.argv) > 1 else 4500
FRAC = {"nevus": 0.40, "melanoma": 0.35, "atypical": 0.25}
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
    hc = np.array(E.handcraft18(binary01, lesion), np.float32)
    img = np.array(IMG.image_block(dehair, mask), np.float32)   # texture on dehaired lesion
    fL = cnn_single(cnnL, lesion).astype(np.float32)
    fD = cnn_single(cnnD, dehair).astype(np.float32)
    return hc, img, fL, fD


def collect_internal():
    paths, y = [], []
    cmap = {"nevus": 0, "melanoma": 1, "atypical": 2}
    for cname, frac in FRAC.items():
        pool = sorted((INT_ROOT / cname).glob("*.jpg"))
        k = min(int(N_INTERNAL * frac), len(pool))
        for i in RNG.choice(len(pool), k, replace=False):
            paths.append(pool[i]); y.append(cmap[cname])
    return paths, np.array(y)


def featurize(paths, unet, cnnL, cnnD, tag):
    HC, IM, FL, FD, ok = [], [], [], [], []
    for i, p in enumerate(paths):
        r = process(p, unet, cnnL, cnnD)
        if r is None:
            ok.append(False); continue
        HC.append(r[0]); IM.append(r[1]); FL.append(r[2]); FD.append(r[3]); ok.append(True)
        if (i + 1) % 250 == 0:
            log(f"  {tag} {i+1}/{len(paths)}")
    return np.array(HC), np.array(IM), np.array(FL), np.array(FD), np.array(ok)


def mk():
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                          random_state=42, n_jobs=-1, verbose=-1, device="gpu")


def ens_eval(parts_int, parts_ext, y_int, ey):
    """parts_* : list of arrays to hstack for view A and view B."""
    (AiL, AiD), (AeL, AeD) = parts_int, parts_ext
    mA = mk().fit(AiL, y_int); mC = mk().fit(AiD, y_int)
    P = (mA.predict_proba(AeL) + mC.predict_proba(AeD)) / 2.0
    return P[:, 1]


def report(name, pmel, ey):
    yb = (ey == 1).astype(int)
    auc = roc_auc_score(yb, pmel)
    tp = int(((pmel >= T_MEL) & (ey == 1)).sum()); nmel = int((ey == 1).sum())
    tn = int(((pmel < T_MEL) & (ey == 0)).sum()); nnev = int((ey == 0).sum())
    log(f"  {name:<28} AUC={auc:.4f}  sens@0.188={tp/nmel:.3f} ({tp}/{nmel})  spec={tn/nnev:.3f} ({tn}/{nnev})")
    return auc


def main():
    log(f"N_internal={N_INTERNAL}")
    unet = E.sess(E.MODELS / "unet.onnx")
    cnnL = E.sess(E.MODELS / "cnn_lesion_feats.onnx")
    cnnD = E.sess(E.MODELS / "cnn_dehair_feats.onnx")

    ip, iy = collect_internal()
    log(f"internal: {len(iy)} (nevus={int((iy==0).sum())}, mel={int((iy==1).sum())}, atyp={int((iy==2).sum())})")
    iHC, iIM, iFL, iFD, iok = featurize(ip, unet, cnnL, cnnD, "int"); iy = iy[iok]
    log(f"internal featurized: {len(iy)}")

    ep, ey = [], []
    for folder, cid in EXT_FOLDERS.items():
        for p in sorted((EXT / folder).iterdir()):
            if p.suffix.lower() in E.IMG_EXTS:
                ep.append(p); ey.append(cid)
    ey = np.array(ey)
    eHC, eIM, eFL, eFD, eok = featurize(ep, unet, cnnL, cnnD, "ext"); ey = ey[eok]
    log(f"external featurized: {len(ey)} (nevus={int((ey==0).sum())}, mel={int((ey==1).sum())})")

    # diagnostic: univariate melanoma AUC of each image feature (internal)
    log("\n=== image features : AUC univariée melanome-vs-reste (interne) ===")
    yb_i = (iy == 1).astype(int)
    for j, nm in enumerate(IMG.NAMES):
        a = roc_auc_score(yb_i, iIM[:, j])
        log(f"  {nm:<16} AUC={a:.4f}  |dev|={abs(a-0.5):.4f}")

    log("\n=== OOD melanoma detection : full vs cnn_only ===")
    cat = np.concatenate
    res = {}
    res["cnn_only [CNN]"] = ens_eval((iFL, iFD), (eFL, eFD), iy, ey)
    res["full_sil [CNN|silhouette]"] = ens_eval(
        (cat([iHC, iFL], 1), cat([iHC, iFD], 1)),
        (cat([eHC, eFL], 1), cat([eHC, eFD], 1)), iy, ey)
    res["full_img [CNN|image]"] = ens_eval(
        (cat([iIM, iFL], 1), cat([iIM, iFD], 1)),
        (cat([eIM, eFL], 1), cat([eIM, eFD], 1)), iy, ey)
    res["full_all [CNN|sil|image]"] = ens_eval(
        (cat([iHC, iIM, iFL], 1), cat([iHC, iIM, iFD], 1)),
        (cat([eHC, eIM, eFL], 1), cat([eHC, eIM, eFD], 1)), iy, ey)
    # image-block ONLY (no CNN) for reference
    res["image_only [no CNN]"] = ens_eval((iIM, iIM), (eIM, eIM), iy, ey)
    for name, pmel in res.items():
        report(name, pmel, ey)


if __name__ == "__main__":
    main()
