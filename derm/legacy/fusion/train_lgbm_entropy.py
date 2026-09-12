"""
Entropy-weighted fusion of A + C with macro Dice score.

Per sample :
  w_A = (1/H_A) / (1/H_A + 1/H_C)
  w_C = (1/H_C) / (1/H_A + 1/H_C)
  P_E = w_A * P_A + w_C * P_C

Dice (multi-class) = macro F1
  Dice_c = 2*TP_c / (2*TP_c + FP_c + FN_c)   (per class)
  Dice   = mean over classes
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from derm import paths

EPS = 1e-12

hc = pd.read_csv(paths.FEATURES_CSV)
cnn_a = pd.read_csv(paths.CNN_FEATS_CSV).drop(columns=["Class"])
cnn_c = pd.read_csv(paths.CNN_DEHAIR_CSV).drop(columns=["Class"])

X_hc = hc.iloc[:, :-1].values
y = hc["Class"].values
X_A = np.concatenate([X_hc, cnn_a.values], axis=1)
X_C = np.concatenate([X_hc, cnn_c.values], axis=1)
X_A = StandardScaler().fit_transform(X_A)
X_C = StandardScaler().fit_transform(X_C)

CLASSES = sorted(set(y))


def shannon(P):
    return -(P * np.log(P + EPS)).sum(axis=1)


def entropy_fuse(P_A, P_C):
    """Per-sample weights inversely proportional to entropy."""
    H_A = shannon(P_A) + EPS
    H_C = shannon(P_C) + EPS
    inv_A, inv_C = 1.0 / H_A, 1.0 / H_C
    w_A = (inv_A / (inv_A + inv_C))[:, None]
    w_C = (inv_C / (inv_A + inv_C))[:, None]
    return w_A * P_A + w_C * P_C


def dice_per_class(y_true, y_pred):
    """Dice = 2 TP / (2 TP + FP + FN) per class."""
    out = []
    for c in CLASSES:
        TP = int(((y_pred == c) & (y_true == c)).sum())
        FP = int(((y_pred == c) & (y_true != c)).sum())
        FN = int(((y_pred != c) & (y_true == c)).sum())
        d = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) else 0.0
        out.append(d)
    return out


def make_model():
    return LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        random_state=42, n_jobs=-1, verbose=-1,
    )


cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
rows = {k: [] for k in ("A_acc", "C_acc", "Eavg_acc", "Eent_acc",
                        "A_auc", "C_auc", "Eavg_auc", "Eent_auc",
                        "A_dice", "C_dice", "Eavg_dice", "Eent_dice")}
per_class_dice = {k: [[], [], []] for k in ("A", "C", "Eavg", "Eent")}

for fold, (tr, te) in enumerate(cv.split(X_A, y), 1):
    m_A = make_model().fit(X_A[tr], y[tr])
    m_C = make_model().fit(X_C[tr], y[tr])
    p_A = m_A.predict_proba(X_A[te])
    p_C = m_C.predict_proba(X_C[te])
    p_avg = (p_A + p_C) / 2.0
    p_ent = entropy_fuse(p_A, p_C)
    yt = y[te]

    for label, P in (("A", p_A), ("C", p_C), ("Eavg", p_avg), ("Eent", p_ent)):
        yhat = P.argmax(axis=1)
        rows[f"{label}_acc"].append(accuracy_score(yt, yhat))
        rows[f"{label}_auc"].append(roc_auc_score(yt, P, multi_class="ovr"))
        d_pc = dice_per_class(yt, yhat)
        rows[f"{label}_dice"].append(np.mean(d_pc))
        for c, dv in enumerate(d_pc):
            per_class_dice[label][c].append(dv)

    print(f"  fold {fold:>2}/10  acc A={rows['A_acc'][-1]:.4f}  "
          f"C={rows['C_acc'][-1]:.4f}  Eavg={rows['Eavg_acc'][-1]:.4f}  "
          f"Eent={rows['Eent_acc'][-1]:.4f}")


def mn(v):
    return f"{np.mean(v):.4f} ± {np.std(v):.4f}"


print("\n=== 10-fold CV summary ===")
print(f"{'metric':<15} {'A':>20} {'C':>20} {'E avg':>20} {'E entropy':>20}")
for m in ("acc", "auc", "dice"):
    print(f"{m:<15} "
          + " ".join(f"{mn(rows[f'{k}_{m}']):>20}" for k in ("A", "C", "Eavg", "Eent")))

print("\n=== Dice per class (mean) ===")
labels = {0: "nevus", 1: "melanoma", 2: "atypical"}
for c in (0, 1, 2):
    print(f"  {labels[c]:<10} "
          + " ".join(f"{np.mean(per_class_dice[k][c]):.4f}" for k in ("A", "C", "Eavg", "Eent")))
