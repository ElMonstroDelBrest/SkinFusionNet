"""Ablate handcrafted features in the historical 512-dimensional CNN baseline.

Compare full [18 handcrafted | 512 CNN], CNN-only, handcrafted-only, four Lab
colour features and eleven morphology/fractal features using ten-fold cross-
validation. Average the A and C branch predictions except in the handcrafted-
only condition. Report accuracy, melanoma recall and macro Dice.
Feature CSV inputs are local artifacts and are not distributed with the source."""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from derm import paths

hc = pd.read_csv(paths.FEATURES_CSV)
cnn_a = pd.read_csv(paths.CNN_FEATS_CSV).drop(columns=["Class"])
cnn_c = pd.read_csv(paths.CNN_DEHAIR_CSV).drop(columns=["Class"])
y = hc["Class"].values
hc_cols = list(hc.columns[:-1])
X_hc = hc.iloc[:, :-1].values
COLOUR = ["Lab_a_kurt", "Lab_b_std", "Lab_a_std", "Lab_b_skew"]
MORPHO = [c for c in hc_cols if c.startswith(("internal_R", "external_R"))] + ["fractal_D", "B", "D"]
idx_colour = [hc_cols.index(c) for c in COLOUR]
idx_morpho = [hc_cols.index(c) for c in MORPHO]
CLASSES = [0, 1, 2]


def make():
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                          random_state=42, n_jobs=-1, verbose=-1, device="gpu")


def dice_macro(yt, yp):
    out = []
    for c in CLASSES:
        TP = int(((yp == c) & (yt == c)).sum()); FP = int(((yp == c) & (yt != c)).sum())
        FN = int(((yp != c) & (yt == c)).sum())
        out.append((2*TP)/(2*TP+FP+FN) if (2*TP+FP+FN) else 0.0)
    return np.mean(out)


# build the feature matrices for each config (A view = lesion, C view = dehair)
def mats(kind):
    if kind == "full":
        return np.c_[X_hc, cnn_a.values], np.c_[X_hc, cnn_c.values]
    if kind == "cnn_only":
        return cnn_a.values, cnn_c.values
    if kind == "hc_only":
        return X_hc, X_hc            # both views share handcraft -> identical
    if kind == "colour":
        return X_hc[:, idx_colour], X_hc[:, idx_colour]
    if kind == "morpho":
        return X_hc[:, idx_morpho], X_hc[:, idx_morpho]


cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
CONFIGS = ["full", "cnn_only", "hc_only", "colour", "morpho"]
res = {k: {"acc": [], "mel_recall": [], "dice": []} for k in CONFIGS}

for kind in CONFIGS:
    XA, XC = mats(kind)
    XA = StandardScaler().fit_transform(XA)
    XC = StandardScaler().fit_transform(XC)
    single = kind in ("hc_only", "colour", "morpho")  # A==C, train one model
    for tr, te in cv.split(XA, y):
        mA = make().fit(XA[tr], y[tr]); pA = mA.predict_proba(XA[te])
        if single:
            pe = pA
        else:
            mC = make().fit(XC[tr], y[tr]); pe = (pA + mC.predict_proba(XC[te])) / 2
        yhat = pe.argmax(1); yt = y[te]
        res[kind]["acc"].append(accuracy_score(yt, yhat))
        res[kind]["mel_recall"].append(recall_score(yt, yhat, labels=[1], average="macro"))
        res[kind]["dice"].append(dice_macro(yt, yhat))
    print(f"done: {kind}", flush=True)

print(f"\n{'config':<10}{'n_feat':>8}{'acc':>16}{'mel_recall':>16}{'dice_macro':>16}")
NFEAT = {"full": X_hc.shape[1]+512, "cnn_only": 512, "hc_only": 18, "colour": 4, "morpho": 14}
for k in CONFIGS:
    a, m, d = res[k]["acc"], res[k]["mel_recall"], res[k]["dice"]
    print(f"{k:<10}{NFEAT[k]:>8}{np.mean(a):>10.4f}±{np.std(a):.3f}"
          f"{np.mean(m):>10.4f}±{np.std(m):.3f}{np.mean(d):>10.4f}±{np.std(d):.3f}")

base = np.mean(res["full"]["acc"]); cnn = np.mean(res["cnn_only"]["acc"])
print(f"\nApport des 18 handcraft (full - cnn_only) : {100*(base-cnn):+.2f} pt d'accuracy")
print(f"Apport sur recall mélanome : "
      f"{100*(np.mean(res['full']['mel_recall'])-np.mean(res['cnn_only']['mel_recall'])):+.2f} pt")
