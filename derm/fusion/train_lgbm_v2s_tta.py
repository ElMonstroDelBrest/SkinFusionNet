"""Same as train_lgbm_v2s.py but uses the TTA-extracted CNN features."""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from derm import paths

EPS = 1e-12

hc = pd.read_csv(paths.FEATURES_CSV)
cnn_a = pd.read_csv(paths.CNN_V2S_TTA_CSV).drop(columns=["Class"])
cnn_c = pd.read_csv(paths.CNN_V2S_DEHAIR_TTA_CSV).drop(columns=["Class"])
y = hc["Class"].values
X_A = StandardScaler().fit_transform(np.c_[hc.iloc[:, :-1].values, cnn_a.values])
X_C = StandardScaler().fit_transform(np.c_[hc.iloc[:, :-1].values, cnn_c.values])
print(f"X_A: {X_A.shape}   X_C: {X_C.shape}")
CLASSES = sorted(set(y))


def shannon(P):
    return -(P * np.log(P + EPS)).sum(axis=1)


def entropy_fuse(P_A, P_C):
    H_A = shannon(P_A) + EPS
    H_C = shannon(P_C) + EPS
    inv_A, inv_C = 1 / H_A, 1 / H_C
    w_A = (inv_A / (inv_A + inv_C))[:, None]
    return w_A * P_A + (1 - w_A) * P_C


def dice_per_class(y_true, y_pred):
    out = []
    for c in CLASSES:
        TP = int(((y_pred == c) & (y_true == c)).sum())
        FP = int(((y_pred == c) & (y_true != c)).sum())
        FN = int(((y_pred != c) & (y_true == c)).sum())
        d = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) else 0.0
        out.append(d)
    return out


def make_model():
    return LGBMClassifier(n_estimators=300, learning_rate=0.05,
                         num_leaves=63, random_state=42,
                         n_jobs=-1, verbose=-1)


cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
rows = {k: [] for k in ("A_acc", "C_acc", "Eavg_acc",
                        "A_auc", "C_auc", "Eavg_auc",
                        "A_dice", "C_dice", "Eavg_dice")}
per_class_dice = {k: [[], [], []] for k in ("A", "C", "Eavg")}

for fold, (tr, te) in enumerate(cv.split(X_A, y), 1):
    m_A = make_model().fit(X_A[tr], y[tr])
    m_C = make_model().fit(X_C[tr], y[tr])
    p_A = m_A.predict_proba(X_A[te])
    p_C = m_C.predict_proba(X_C[te])
    p_avg = (p_A + p_C) / 2.0
    yt = y[te]

    for label, P in (("A", p_A), ("C", p_C), ("Eavg", p_avg)):
        yhat = P.argmax(axis=1)
        rows[f"{label}_acc"].append(accuracy_score(yt, yhat))
        rows[f"{label}_auc"].append(roc_auc_score(yt, P, multi_class="ovr"))
        d_pc = dice_per_class(yt, yhat)
        rows[f"{label}_dice"].append(np.mean(d_pc))
        for c, dv in enumerate(d_pc):
            per_class_dice[label][c].append(dv)

    print(f"  fold {fold:>2}/10  acc A={rows['A_acc'][-1]:.4f}  "
          f"C={rows['C_acc'][-1]:.4f}  Eavg={rows['Eavg_acc'][-1]:.4f}")


def mn(v):
    return f"{np.mean(v):.4f} +- {np.std(v):.4f}"


print("\n=== V2-S + TTA x8 10-fold CV ===")
print(f"{'metric':<15} {'V_A':>22} {'V_C':>22} {'E avg':>22}")
for m in ("acc", "auc", "dice"):
    print(f"{m:<15} "
          + " ".join(f"{mn(rows[f'{k}_{m}']):>22}" for k in ("A", "C", "Eavg")))

print("\n=== Dice per class (mean) ===")
labels = {0: "nevus", 1: "melanoma", 2: "atypical"}
for c in (0, 1, 2):
    print(f"  {labels[c]:<10} "
          + " ".join(f"{np.mean(per_class_dice[k][c]):.4f}" for k in ("A", "C", "Eavg")))
