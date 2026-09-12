"""Evaluate image-derived descriptors alongside historical 512-D CNN features.

Use out-of-fold regression to estimate how well CNN embeddings predict each
descriptor. Compare CNN-only and CNN-plus-image classifiers under five-fold and
leave-one-source-out splits. Low probing R-squared and a positive ablation delta
are evidence of potential complementary value under these protocols, rather
than proof that the CNN contains no equivalent information."""
import warnings, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (cross_val_predict, KFold, StratifiedKFold,
                                     LeaveOneGroupOut)
from sklearn.metrics import (r2_score, roc_auc_score, average_precision_score,
                             balanced_accuracy_score, recall_score)
import lightgbm as lgb
warnings.filterwarnings("ignore")

from derm import paths as RP

CNN = [f'cnn_{i}' for i in range(512)]
LES = RP.LESION
CLASS_MAP = {"nevus": 0, "melanoma": 1, "atypical": 2}
SRC = {'isic2019': 'ISIC2019', 'ham': 'HAM10000', 's1': 'ISIC2017', 's2': 'Kaggle9'}


def lgbm_clf():
    return lgb.LGBMClassifier(objective='multiclass', num_class=3, n_estimators=300,
                              learning_rate=0.05, num_leaves=63, n_jobs=-1, verbosity=-1)


def metrics(y, p1, pred):
    yb = (y == 1).astype(int)
    return dict(auprc_mel=average_precision_score(yb, p1),
                auc_mel=roc_auc_score(yb, p1),
                mel_recall=recall_score(y, pred, labels=[1], average='macro'),
                bal_acc=balanced_accuracy_score(y, pred))


def oof(X, y, splitter, groups=None):
    p1 = np.zeros(len(y)); pred = np.zeros(len(y), dtype=int)
    for tr, te in splitter.split(X, y, groups):
        m = lgbm_clf().fit(X[tr], y[tr]); pr = m.predict_proba(X[te])
        p1[te] = pr[:, 1]; pred[te] = pr.argmax(1)
    return p1, pred


def boot(y, pf, pc, n=1000, seed=0):
    rng = np.random.default_rng(seed); yb = (y == 1).astype(int); idx = np.arange(len(y))
    da, du = [], []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if 0 < yb[b].sum() < len(b):
            da.append(average_precision_score(yb[b], pf[b]) - average_precision_score(yb[b], pc[b]))
            du.append(roc_auc_score(yb[b], pf[b]) - roc_auc_score(yb[b], pc[b]))
    f = lambda a: (np.mean(a), np.percentile(a, 2.5), np.percentile(a, 97.5))
    return f(da), f(du)


def per_class_auc(h, y, c):
    m = np.isfinite(h); a = roc_auc_score((y[m] == c).astype(int), h[m]); return max(a, 1 - a)


def main():
    cnn = pd.read_csv(RP.CNN_FEATS_CSV, usecols=CNN + ['Class'])
    img = pd.read_csv(RP.IMAGE_FEATS_CSV)
    assert (cnn['Class'].values == img['Class'].values).all(), "désaligné!"
    y = cnn['Class'].values
    Z = cnn[CNN].values.astype('float32')
    cols = [c for c in img.columns if c != 'Class']
    I = np.nan_to_num(img[cols].values.astype('float32'), nan=0.0, posinf=0.0, neginf=0.0)

    paths = []
    for s in ("train", "valid", "test"):
        paths += sorted((LES / s).rglob("*.png"))
    src = np.array([SRC[re.match(r'^([a-zA-Z0-9]+?)_', p.name).group(1)] for p in paths])

    # ---------- 1. PROBING ----------
    print("=" * 92)
    print("PROBING R²(descripteur image | embedding CNN 512-d)  +  AUC OvR par classe")
    print("=" * 92)
    kf = KFold(5, shuffle=True, random_state=0)
    rows = []
    for j, c in enumerate(cols):
        h = I[:, j]
        r2l = r2_score(h, cross_val_predict(make_pipeline(StandardScaler(), Ridge(10.0)), Z, h, cv=kf, n_jobs=5))
        r2n = r2_score(h, cross_val_predict(lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                       num_leaves=63, n_jobs=-1, verbosity=-1), Z, h, cv=kf, n_jobs=1))
        rows.append(dict(feature=c, r2_lin=r2l, r2_nl=r2n,
                         auc_nevus=per_class_auc(h, y, 0), auc_mel=per_class_auc(h, y, 1),
                         auc_atyp=per_class_auc(h, y, 2)))
    P = pd.DataFrame(rows).sort_values('r2_nl')
    with pd.option_context('display.width', 200):
        print(P.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    RP.PROBING_IMAGE_FEATS_CSV.parent.mkdir(parents=True, exist_ok=True)
    P.to_csv(RP.PROBING_IMAGE_FEATS_CSV, index=False)

    nonred = P[P.r2_nl < 0.20].feature.tolist()
    print(f"\nNON encodés (R²_nl < 0.20) = candidats info NEUVE : {nonred}")
    disc_new = P[(P.r2_nl < 0.20) & (P[['auc_nevus','auc_mel','auc_atyp']].max(axis=1) >= 0.60)].feature.tolist()
    print(f"NON encodés ET discriminants (AUC≥0.60 sur une classe) : {disc_new}")

    # ---------- 2. ABLATION ----------
    Xcnn = Z
    Ximg = np.hstack([Z, I])
    idx_nr = [cols.index(c) for c in nonred] if nonred else []
    Xnr = np.hstack([Z, I[:, idx_nr]]) if idx_nr else None

    def block(tag, splitter, groups=None):
        print(f"\n===== {tag} =====")
        p1c, prc = oof(Xcnn, y, splitter, groups)
        p1i, pri = oof(Ximg, y, splitter, groups)
        mc, mi = metrics(y, p1c, prc), metrics(y, p1i, pri)
        print(f"  {'metric':12s} {'cnn_only':>10} {'cnn+image':>10} {'Δ':>9}")
        for k in ('auprc_mel', 'auc_mel', 'mel_recall', 'bal_acc'):
            print(f"  {k:12s} {mc[k]:10.4f} {mi[k]:10.4f} {mi[k]-mc[k]:+9.4f}")
        (ap, apl, aph), (au, aul, auh) = boot(y, p1i, p1c)
        print(f"  bootstrap Δ AUPRC = {ap:+.4f} [{apl:+.4f}, {aph:+.4f}]")
        print(f"  bootstrap Δ AUC   = {au:+.4f} [{aul:+.4f}, {auh:+.4f}]")
        if Xnr is not None:
            p1n, prn = oof(Xnr, y, splitter, groups)
            mn = metrics(y, p1n, prn)
            print(f"  [cnn + non-encodés({len(idx_nr)})]  AUPRC={mn['auprc_mel']:.4f} "
                  f"(Δ{mn['auprc_mel']-mc['auprc_mel']:+.4f})  AUC={mn['auc_mel']:.4f} "
                  f"(Δ{mn['auc_mel']-mc['auc_mel']:+.4f})")

    block("INTRA  (Stratified 5-fold, sources mélangées)", StratifiedKFold(5, shuffle=True, random_state=0))
    block("SHIFT  (Leave-One-Source-Out, 4 folds)", LeaveOneGroupOut(), src)

    # Reference: image descriptors only, evaluated with melanoma AUC.
    p1, _ = oof(I, y, StratifiedKFold(5, shuffle=True, random_state=0))
    print(f"\n(réf) image-only sans CNN : AUPRC_mel={average_precision_score((y==1),p1):.4f} "
          f"AUC_mel={roc_auc_score((y==1),p1):.4f}")


if __name__ == "__main__":
    main()
