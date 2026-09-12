"""Test handcrafted features under binary and three-class training.

Reuse single-pass feature extraction from drop_atypical_external. Compare full
[handcrafted | CNN] and CNN-only models with and without atypical training cases,
using the same images and features. Evaluate external melanoma detection.

Run: python -m derm.analysis.legacy.handcraft_value_2cls [N_internal]"""
import numpy as np
from sklearn.metrics import roc_auc_score

from derm.export import eval_external as E
from derm import paths
from derm.analysis.legacy.drop_atypical_external import (
    featurize, collect_internal, mk, EXT_FOLDERS, youden,
)

EXT = paths.EXTERNAL_VAL
T_MEL = 0.188


def log(*a):
    print(*a, flush=True)


def fit_pair(hc, fL, fD, y, use_hc):
    XA = np.concatenate([hc, fL], 1) if use_hc else fL
    XC = np.concatenate([hc, fD], 1) if use_hc else fD
    return mk().fit(XA, y), mk().fit(XC, y)


def pmel(mA, mC, hc, fL, fD, use_hc):
    XA = np.concatenate([hc, fL], 1) if use_hc else fL
    XC = np.concatenate([hc, fD], 1) if use_hc else fD
    return ((mA.predict_proba(XA) + mC.predict_proba(XC)) / 2.0)[:, 1]


def main():
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

    log(f"{'setup':<34} {'AUC OOD':>8} {'sens@0.188':>11} {'sens@Youden':>12} {'spec@Y':>7}")
    res = {}
    for cls_name, mask in (("3-classes", np.ones(len(iy), bool)), ("2-classes", iy != 2)):
        for use_hc, fname in ((True, "full [hc|CNN]"), (False, "cnn_only [CNN]")):
            mA, mC = fit_pair(iHC[mask], iFL[mask], iFD[mask], iy[mask], use_hc)
            p = pmel(mA, mC, eHC, eFL, eFD, use_hc)
            auc = roc_auc_score(yb, p)
            tp = int(((p >= T_MEL) & (ey == 1)).sum())
            ty, sey, spy = youden(yb, p)
            name = f"{cls_name} {fname}"
            log(f"{name:<34} {auc:>8.4f} {f'{tp}/{nmel}':>11} {sey:>12.3f} {spy:>7.3f}")
            res[(cls_name, use_hc)] = auc

    log("\n=== Valeur ajoutée des handcraft (Δ AUC = full − cnn_only) ===")
    for cls_name in ("3-classes", "2-classes"):
        d = res[(cls_name, True)] - res[(cls_name, False)]
        log(f"  {cls_name:<12} Δ = {d:+.4f}")
    log("\nLecture : si Δ(2-classes) > Δ(3-classes), retirer atypical redonne de la valeur aux ABCD.")
    log("Caveat : single-pass sans TTA ; n=19 mél -> Δ de 1-2 cas, suggestif.")


if __name__ == "__main__":
    main()
