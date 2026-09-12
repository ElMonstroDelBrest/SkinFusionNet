"""Compare a learned atypical class with binary classification and abstention.

Train three-class and binary LightGBM ensembles on the same internal sample,
using features extracted with the deployed single-pass ONNX pipeline.
The three-class readout flags melanoma at p_mel >= 0.188, then abstains at
p_atyp >= 0.412. The binary readout uses an external Youden threshold and
progressively defers cases nearest that threshold to form a risk-coverage curve.

Report coverage, sensitivity and specificity among decided cases, and the
fraction of melanomas either flagged or deferred. External threshold selection
makes this a descriptive analysis, not an independent validation.

Run: python -m derm.analysis.abstention_external [N_internal]"""
import numpy as np
from sklearn.metrics import roc_auc_score

from derm.export import eval_external as E
from derm import paths
from derm.analysis.legacy.drop_atypical_external import (
    cnn_single, process, featurize, collect_internal, mk, EXT_FOLDERS,
)

EXT = paths.EXTERNAL_VAL
T_MEL, T_ATYP = 0.188, 0.412


def log(*a):
    print(*a, flush=True)


def youden(yb, p):
    bt, bj = 0.5, -1
    for t in np.unique(p):
        se = ((p >= t) & (yb == 1)).sum() / max((yb == 1).sum(), 1)
        sp = ((p < t) & (yb == 0)).sum() / max((yb == 0).sum(), 1)
        if se + sp - 1 > bj:
            bj, bt = se + sp - 1, t
    return bt


def selective(ey, p, T, abstain):
    """Return coverage, decided-case sensitivity/specificity and flagged-or-deferred sensitivity."""
    mel, nev = ey == 1, ey == 0
    decided = ~abstain
    pred_mel = p >= T
    dmel, dnev = decided & mel, decided & nev
    sens = (pred_mel & dmel).sum() / max(dmel.sum(), 1)
    spec = (~pred_mel & dnev).sum() / max(dnev.sum(), 1)
    safety = (~(decided & ~pred_mel) & mel).sum() / mel.sum()   # Melanomas flagged or deferred rather than classified as benign.
    return decided.mean(), sens, spec, safety


def main():
    unet = E.sess(E.MODELS / "unet.onnx")
    cnnL = E.sess(E.MODELS / "cnn_lesion_feats.onnx")
    cnnD = E.sess(E.MODELS / "cnn_dehair_feats.onnx")

    ip, iy = collect_internal()
    log(f"internal: {len(iy)} (nevus={int((iy==0).sum())}, mel={int((iy==1).sum())}, atyp={int((iy==2).sum())})")
    iHC, iFL, iFD, iok = featurize(ip, unet, cnnL, cnnD, "int")
    iy = iy[iok]

    # Three-class sample and its matched nevus/melanoma subset.
    mA3 = mk().fit(np.concatenate([iHC, iFL], 1), iy)
    mC3 = mk().fit(np.concatenate([iHC, iFD], 1), iy)
    m = iy != 2
    mA2 = mk().fit(np.concatenate([iHC[m], iFL[m]], 1), iy[m])
    mC2 = mk().fit(np.concatenate([iHC[m], iFD[m]], 1), iy[m])

    ep, ey = [], []
    for folder, cid in EXT_FOLDERS.items():
        for p in sorted((EXT / folder).iterdir()):
            if p.suffix.lower() in E.IMG_EXTS:
                ep.append(p); ey.append(cid)
    ey = np.array(ey)
    eHC, eFL, eFD, eok = featurize(ep, unet, cnnL, cnnD, "ext")
    ey = ey[eok]
    nmel = int((ey == 1).sum())

    P3 = (mA3.predict_proba(np.concatenate([eHC, eFL], 1)) +
          mC3.predict_proba(np.concatenate([eHC, eFD], 1))) / 2.0
    P2 = (mA2.predict_proba(np.concatenate([eHC, eFL], 1)) +
          mC2.predict_proba(np.concatenate([eHC, eFD], 1))) / 2.0
    p_mel3, p_atyp3 = P3[:, 1], P3[:, 2]
    p2 = P2[:, 1]
    log(f"externe: {len(ey)} (mel={nmel}) ; AUC 3cl={roc_auc_score((ey==1), p_mel3):.4f}  AUC 2cl={roc_auc_score((ey==1), p2):.4f}\n")

    # (a) Three-class threshold readout at one operating point.
    alert = p_mel3 >= T_MEL
    abstain_a = (~alert) & (p_atyp3 >= T_ATYP)
    cov_a, sens_a, spec_a, safe_a = selective(ey, p_mel3, T_MEL, abstain_a)
    log("(a) 3-classes 'atypical' (seuils cliniques) — un seul point :")
    log(f"    abstention={1-cov_a:.2f}  sens(décidés)={sens_a:.3f}  spec(décidés)={spec_a:.3f}  "
        f"SÉCURITÉ(malin-ou-abst)={safe_a:.3f}\n")

    # (b) Binary classification with abstention near the external Youden threshold.
    T2 = youden((ey == 1).astype(int), p2)
    margin = np.abs(p2 - T2)                      # Confidence proxy: distance from the decision threshold.
    order = np.argsort(margin)                    # Defer the least confident cases first.
    log(f"(b) 2-classes + abstention (frontière Youden externe T={T2:.3f}) — courbe risque-couverture :")
    log(f"    {'abstention':>10} {'couverture':>10} {'sens(déc)':>10} {'spec(déc)':>10} {'SÉCURITÉ':>9}")
    n = len(ey)
    for r in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 1 - cov_a):
        k = int(round(r * n))
        abst = np.zeros(n, bool); abst[order[:k]] = True
        cov, se, sp, safe = selective(ey, p2, T2, abst)
        tag = "  <- même abst. que (a)" if abs(r - (1 - cov_a)) < 1e-9 else ""
        log(f"    {1-cov:>10.2f} {cov:>10.2f} {se:>10.3f} {sp:>10.3f} {safe:>9.3f}{tag}")

    log("\nLecture : à abstention égale, (b) atteint-il une meilleure sécurité / sélectivité que (a) ?")
    log("Caveat : T2 = Youden externe (in-sample, n=19) ; c'est la FORME de la courbe qui compte, pas l'absolu.")


if __name__ == "__main__":
    main()
