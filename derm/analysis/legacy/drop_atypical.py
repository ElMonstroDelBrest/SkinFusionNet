"""Compare binary and three-class training with the historical 512-D baseline.

Use ten-fold cross-validation and evaluate melanoma-versus-nevus ROC-AUC on the
same nevus and melanoma observations. The binary model excludes atypical cases
from training and evaluation. Binary accuracy is not directly comparable with
three-class accuracy because the evaluated classification task changes.

Run: python -m derm.analysis.legacy.drop_atypical"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from derm import paths


def model():
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                          random_state=42, n_jobs=-1, verbose=-1)


def oof_ensemble(XA, XC, y):
    """Probas out-of-fold, ensemble = moyenne des 2 vues. Renvoie (n, n_classes)."""
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    nc = len(np.unique(y))
    P = np.zeros((len(y), nc))
    for tr, te in cv.split(XA, y):
        pa = model().fit(XA[tr], y[tr]).predict_proba(XA[te])
        pc = model().fit(XC[tr], y[tr]).predict_proba(XC[te])
        P[te] = (pa + pc) / 2.0
    return P


def youden(yb, p):
    best_t, best_j = 0.5, -1
    for t in np.unique(p):
        se = ((p >= t) & (yb == 1)).sum() / max((yb == 1).sum(), 1)
        sp = ((p < t) & (yb == 0)).sum() / max((yb == 0).sum(), 1)
        if se + sp - 1 > best_j:
            best_j, best_t = se + sp - 1, t
    se = ((p >= best_t) & (yb == 1)).sum() / (yb == 1).sum()
    sp = ((p < best_t) & (yb == 0)).sum() / (yb == 0).sum()
    return best_t, se, sp


def main():
    hc = pd.read_csv(paths.FEATURES_CSV)
    cnn_a = pd.read_csv(paths.CNN_FEATS_CSV).drop(columns=["Class"])
    cnn_c = pd.read_csv(paths.CNN_DEHAIR_CSV).drop(columns=["Class"])
    y = hc["Class"].values                          # 0=nevus 1=mel 2=atyp
    Xhc = hc.iloc[:, :-1].values
    from collections import Counter
    print("composition :", dict(Counter(y)), "(0=nevus 1=mel 2=atyp)\n")

    # Three-class baseline: scaler fitted before cross-validation (leakage limitation).
    XA = StandardScaler().fit_transform(np.concatenate([Xhc, cnn_a.values], 1))
    XC = StandardScaler().fit_transform(np.concatenate([Xhc, cnn_c.values], 1))
    P3 = oof_ensemble(XA, XC, y)
    acc3 = accuracy_score(y, P3.argmax(1))
    m = y != 2                                       # lignes nevus+mel
    auc3 = roc_auc_score((y[m] == 1).astype(int), P3[m, 1])
    t3, se3, sp3 = youden((y[m] == 1).astype(int), P3[m, 1])

    # Binary condition: exclude atypical cases from training and evaluation.
    y2 = y[m]                                        # 0/1
    XA2 = StandardScaler().fit_transform(np.concatenate([Xhc[m], cnn_a.values[m]], 1))
    XC2 = StandardScaler().fit_transform(np.concatenate([Xhc[m], cnn_c.values[m]], 1))
    P2 = oof_ensemble(XA2, XC2, y2)
    acc2 = accuracy_score(y2, P2.argmax(1))
    auc2 = roc_auc_score((y2 == 1).astype(int), P2[:, 1])
    t2, se2, sp2 = youden((y2 == 1).astype(int), P2[:, 1])

    print(f"{'setup':<24} {'acc':>7} {'AUC mél-vs-nevus':>18} {'sens@Youden':>12} {'spec':>7}")
    print(f"{'3-classes (p_mel)':<24} {acc3:>7.4f} {auc3:>18.4f} {se3:>12.3f} {sp3:>7.3f}")
    print(f"{'2-classes (sans atyp)':<24} {acc2:>7.4f} {auc2:>18.4f} {se2:>12.3f} {sp2:>7.3f}")
    print(f"\nΔ AUC mél-vs-nevus (2cl − 3cl) = {auc2 - auc3:+.4f}")
    print("(l'acc 2-classes est gonflée mécaniquement : classe dure retirée — comparer l'AUC, pas l'acc)")


if __name__ == "__main__":
    main()
