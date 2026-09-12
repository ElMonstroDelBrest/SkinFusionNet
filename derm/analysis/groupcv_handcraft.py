"""Evaluate handcrafted features under mixed-source and held-out-source splits.

Compare full V2S single-pass features with CNN-only features on identical splits.
Use stratified cross-validation for mixed-source evaluation and leave-one-source-
out evaluation for source shift. Report melanoma average precision, one-vs-rest
ROC-AUC, argmax recall and three-class balanced accuracy with paired bootstrap
comparisons. The experiment tests whether handcrafted features remain useful
when the evaluation source was absent from training; it does not establish why
any source-dependent performance difference occurs."""
import warnings
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             balanced_accuracy_score, recall_score)
import lightgbm as lgb

from derm import paths as RP

warnings.filterwarnings("ignore")

HC = ['A4', 'B', 'D', 'Lab_a_kurt', 'Lab_b_std', 'Lab_a_std', 'Lab_b_skew',
      'internal_R1', 'internal_R2', 'internal_R4', 'internal_R8', 'internal_R16',
      'external_R1', 'external_R2', 'external_R4', 'external_R8', 'external_R16',
      'fractal_D']
LESION = RP.LESION
CLASS_MAP = {"nevus": 0, "melanoma": 1, "atypical": 2}
SRC = {'isic2019': 'ISIC2019', 'ham': 'HAM10000', 's1': 'ISIC2017', 's2': 'Kaggle9'}


def lgbm():
    return lgb.LGBMClassifier(objective='multiclass', num_class=3,
                              n_estimators=300, learning_rate=0.05,
                              num_leaves=63, n_jobs=-1, verbosity=-1)


def metrics(y, p1, pred):
    """p1 = P(melanome), pred = argmax 3-classes."""
    yb = (y == 1).astype(int)
    return dict(
        auprc_mel=average_precision_score(yb, p1),
        auc_mel=roc_auc_score(yb, p1),
        mel_recall=recall_score(y, pred, labels=[1], average='macro'),
        bal_acc=balanced_accuracy_score(y, pred),
    )


def oof_predict(X, y, splitter, groups=None):
    """Predictions out-of-fold ; renvoie p1 (proba melanome) et pred argmax."""
    p1 = np.zeros(len(y))
    pred = np.zeros(len(y), dtype=int)
    for tr, te in splitter.split(X, y, groups):
        m = lgbm().fit(X[tr], y[tr])
        proba = m.predict_proba(X[te])
        p1[te] = proba[:, 1]
        pred[te] = proba.argmax(1)
    return p1, pred


def paired_bootstrap(y, p1_full, p1_cnn, n=1000, seed=0):
    """Bootstrap 95% intervals for full-minus-CNN-only melanoma AUPRC and AUC."""
    rng = np.random.default_rng(seed)
    yb = (y == 1).astype(int)
    idx = np.arange(len(y))
    d_ap, d_auc = [], []
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        if yb[b].sum() == 0 or yb[b].sum() == len(b):
            continue
        d_ap.append(average_precision_score(yb[b], p1_full[b]) -
                    average_precision_score(yb[b], p1_cnn[b]))
        d_auc.append(roc_auc_score(yb[b], p1_full[b]) -
                     roc_auc_score(yb[b], p1_cnn[b]))
    q = lambda a: (np.mean(a), np.percentile(a, 2.5), np.percentile(a, 97.5))
    return q(d_ap), q(d_auc)


def main():
    cnn = pd.read_csv(RP.CNN_V2S_FEATS_CSV)
    cnn_cols = [c for c in cnn.columns if c.startswith("cnn_")]
    feats = pd.read_csv(RP.FEATURES_CSV, usecols=HC + ['Class'])
    assert (cnn['Class'].values == feats['Class'].values).all()
    y = cnn['Class'].values
    Z = cnn[cnn_cols].values.astype('float32')
    H = feats[HC].values.astype('float32')
    H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
    Xcnn = Z
    Xfull = np.hstack([Z, H])

    # source par ligne (ordre reconstruit + verifie)
    paths = []
    for split in ("train", "valid", "test"):
        paths += sorted((LESION / split).rglob("*.png"))
    assert len(paths) == len(y)
    src = np.array([SRC[re.match(r'^([a-zA-Z0-9]+?)_', p.name).group(1)] for p in paths])
    assert (np.array([CLASS_MAP[p.parent.name] for p in paths]) == y).all(), "ordre KO"

    print("=== composition par source (classes 0=nevus 1=mel 2=atyp) ===")
    for s in sorted(set(src)):
        c = Counter(y[src == s])
        print(f"  {s:9s} n={int((src==s).sum()):>6}  "
              f"nevus={c[0]:>5} mel={c[1]:>5} atyp={c[2]:>5}")

    def report(tag, p1f, predf, p1c, predc, yy):
        mf, mc = metrics(yy, p1f, predf), metrics(yy, p1c, predc)
        print(f"\n--- {tag} ---")
        print(f"  {'metric':12s}  {'cnn_only':>10}  {'full(+18)':>10}  {'delta':>8}")
        for k in ('auprc_mel', 'auc_mel', 'mel_recall', 'bal_acc'):
            print(f"  {k:12s}  {mc[k]:10.4f}  {mf[k]:10.4f}  {mf[k]-mc[k]:+8.4f}")
        return mf, mc

    # 1. Mixed-source stratified five-fold evaluation.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    p1f, predf = oof_predict(Xfull, y, skf)
    p1c, predc = oof_predict(Xcnn, y, skf)
    report("INTRA  (Stratified 5-fold, sources mélangées)", p1f, predf, p1c, predc, y)
    (ap_m, ap_lo, ap_hi), (au_m, au_lo, au_hi) = paired_bootstrap(y, p1f, p1c)
    print(f"  bootstrap Δ AUPRC = {ap_m:+.4f} [{ap_lo:+.4f}, {ap_hi:+.4f}]")
    print(f"  bootstrap Δ AUC   = {au_m:+.4f} [{au_lo:+.4f}, {au_hi:+.4f}]")

    # ---------- 2. SHIFT : Leave-One-Source-Out (4 folds) ----------
    logo = LeaveOneGroupOut()
    p1f, predf = oof_predict(Xfull, y, logo, groups=src)
    p1c, predc = oof_predict(Xcnn, y, logo, groups=src)
    print("\n=== LEAVE-ONE-SOURCE-OUT : par source tenue à l'écart ===")
    deltas_ap = []
    for s in sorted(set(src)):
        msk = src == s
        if (y[msk] == 1).sum() == 0:
            print(f"  [{s}] pas de mélanome -> AUC indéfinie, sauté")
            continue
        mf = metrics(y[msk], p1f[msk], predf[msk])
        mc = metrics(y[msk], p1c[msk], predc[msk])
        deltas_ap.append(mf['auprc_mel'] - mc['auprc_mel'])
        print(f"  [{s:9s}] AUPRC_mel cnn={mc['auprc_mel']:.3f} full={mf['auprc_mel']:.3f} "
              f"Δ={mf['auprc_mel']-mc['auprc_mel']:+.3f} | "
              f"AUC_mel cnn={mc['auc_mel']:.3f} full={mf['auc_mel']:.3f} "
              f"Δ={mf['auc_mel']-mc['auc_mel']:+.3f}")
    report("SHIFT  (LOSO, prédictions poolées)", p1f, predf, p1c, predc, y)
    print(f"  Δ AUPRC moyen par-source = {np.mean(deltas_ap):+.4f}")
    (ap_m, ap_lo, ap_hi), (au_m, au_lo, au_hi) = paired_bootstrap(y, p1f, p1c)
    print(f"  bootstrap Δ AUPRC (poolé) = {ap_m:+.4f} [{ap_lo:+.4f}, {ap_hi:+.4f}]")
    print(f"  bootstrap Δ AUC   (poolé) = {au_m:+.4f} [{au_lo:+.4f}, {au_hi:+.4f}]")


if __name__ == '__main__':
    main()
