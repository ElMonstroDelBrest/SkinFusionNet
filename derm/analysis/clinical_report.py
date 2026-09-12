"""
Clinical analysis report on the ensemble LightGBM outputs.

For each class (nevus / melanoma / atypical), use its normalized multiclass
probability as a one-vs-rest score and produce :
  - calibration curve  (reliability diagram)
  - ROC curve + AUC + optimal Youden threshold
  - histogram of P(class) per true class
  - confusion matrix at default (argmax) and tuned thresholds

Plus :
  - global confusion matrix
  - decision curve analysis (clinical net benefit) for melanoma
  - per-sample entropy histogram (model confidence)

All saved in clinical_report/ as PNGs + summary.txt.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (auc, confusion_matrix, f1_score, precision_score,
                              recall_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from derm import paths

OUT = paths.DOCS_CLINICAL
OUT.mkdir(parents=True, exist_ok=True)
MODEL_MANIFEST = json.loads((paths.APP_MODELS / "model_manifest.json").read_text())
LABELS = {0: "nevus", 1: "melanoma", 2: "atypical"}
COLORS = {0: "#2ca02c", 1: "#d62728", 2: "#ff7f0e"}

# --- load canonical V2S single-pass features ----------------------------------
hc = pd.read_csv(paths.FEATURES_CSV)
cnn_a_raw = pd.read_csv(paths.CNN_V2S_FEATS_CSV)
cnn_c_raw = pd.read_csv(paths.CNN_V2S_DEHAIR_CSV)
cnn_cols = [c for c in cnn_a_raw.columns if c.startswith("cnn_")]
if cnn_cols != [c for c in cnn_c_raw.columns if c.startswith("cnn_")]:
    raise ValueError("CNN feature columns differ between lesion and dehair")
cnn_a = cnn_a_raw[cnn_cols]
cnn_c = cnn_c_raw[cnn_cols]
y = hc["Class"].values
if not np.array_equal(y, cnn_a_raw["Class"].to_numpy(int)):
    raise ValueError("Class mismatch features.csv vs V2S lesion CSV")
if not np.array_equal(y, cnn_c_raw["Class"].to_numpy(int)):
    raise ValueError("Class mismatch features.csv vs V2S dehair CSV")
X_A = np.c_[hc.iloc[:, :-1].values, cnn_a.values]
X_C = np.c_[hc.iloc[:, :-1].values, cnn_c.values]
print(f"X_A: {X_A.shape}   X_C: {X_C.shape}")


def make_model():
    # Fit the scaler on each training fold, not on held-out observations.
    return make_pipeline(
        StandardScaler(),
        LGBMClassifier(n_estimators=300, learning_rate=0.05,
                       num_leaves=63, random_state=42,
                       n_jobs=-1, verbose=-1),
    )


# --- 10-fold CV to gather out-of-fold probabilities ---------------------------
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
oof_probs = np.zeros((len(y), 3))  # ensemble averaged probas

for fold, (tr, te) in enumerate(cv.split(X_A, y), 1):
    m_A = make_model().fit(X_A[tr], y[tr])
    m_C = make_model().fit(X_C[tr], y[tr])
    oof_probs[te] = (m_A.predict_proba(X_A[te]) + m_C.predict_proba(X_C[te])) / 2
    print(f"  fold {fold}/10 done")

# argmax predictions
y_pred = oof_probs.argmax(axis=1)
print(f"\nGlobal accuracy: {(y_pred == y).mean():.4f}")

# =============================================================================
#  Reliability diagram per class
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for c, ax in zip((0, 1, 2), axes):
    y_bin = (y == c).astype(int)
    p = oof_probs[:, c]
    fop, mpv = calibration_curve(y_bin, p, n_bins=10, strategy="quantile")
    ax.plot([0, 1], [0, 1], "k:", label="Perfect calibration")
    ax.plot(mpv, fop, "o-", color=COLORS[c], lw=2, ms=8, label=LABELS[c])
    ax.set_xlabel("Predicted class probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration — {LABELS[c]}")
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "calibration.png", dpi=120)
plt.close()
print("saved calibration.png")

# =============================================================================
#  ROC curves with Youden optimal threshold
# =============================================================================
optimal_thresholds = {}
fig, ax = plt.subplots(figsize=(8, 7))
for c in (0, 1, 2):
    y_bin = (y == c).astype(int)
    p = oof_probs[:, c]
    fpr, tpr, thr = roc_curve(y_bin, p)
    a = auc(fpr, tpr)
    youden = tpr - fpr
    j = youden.argmax()
    t_opt = thr[j]
    optimal_thresholds[c] = (float(t_opt), float(tpr[j]), float(1 - fpr[j]))
    ax.plot(fpr, tpr, color=COLORS[c], lw=2,
            label=f"{LABELS[c]}  AUC={a:.3f}  T*={t_opt:.2f}")
    ax.plot(fpr[j], tpr[j], "o", color=COLORS[c], ms=10, mew=2,
            mec="black", label=f"  Sens={tpr[j]:.2f}, Spec={1-fpr[j]:.2f}")
ax.plot([0, 1], [0, 1], "k:")
ax.set_xlabel("1 - Specificity (FPR)")
ax.set_ylabel("Sensitivity (TPR)")
ax.set_title("One-vs-rest ROC curves and Youden thresholds")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "roc_per_class.png", dpi=120)
plt.close()
print("saved roc_per_class.png")

# =============================================================================
#  Probability histograms
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for c, ax in zip((0, 1, 2), axes):
    for true_c in (0, 1, 2):
        mask = (y == true_c)
        ax.hist(oof_probs[mask, c], bins=40, alpha=0.5,
                color=COLORS[true_c], label=f"True class = {LABELS[true_c]}")
    ax.axvline(optimal_thresholds[c][0], color="black", ls="--",
               label=f"T* = {optimal_thresholds[c][0]:.2f}")
    ax.set_xlabel(f"P({LABELS[c]}) predicted")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of P({LABELS[c]}) by true class")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
plt.tight_layout()
plt.savefig(OUT / "proba_histograms.png", dpi=120)
plt.close()
print("saved proba_histograms.png")

# =============================================================================
#  Global confusion matrix (argmax)
# =============================================================================
cm = confusion_matrix(y, y_pred, normalize="true")
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        col = "white" if cm[i, j] > 0.5 else "black"
        ax.text(j, i, f"{cm[i, j]:.3f}", ha="center", va="center", color=col)
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(["nevus", "melanoma", "atypical"])
ax.set_yticklabels(["nevus", "melanoma", "atypical"])
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Normalized confusion matrix (argmax)")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(OUT / "confusion_argmax.png", dpi=120)
plt.close()
print("saved confusion_argmax.png")

# =============================================================================
#  Decision curve analysis for melanoma
#  Net benefit = TP/N - FP/N * (pt / (1-pt))
#  pt = threshold probability (clinical preference)
# =============================================================================
y_mel = (y == 1).astype(int)
p_mel = oof_probs[:, 1]
N = len(y)
pts = np.linspace(0.01, 0.99, 99)

net_benefit_model = []
for pt in pts:
    pred_pos = p_mel >= pt
    TP = ((pred_pos) & (y_mel == 1)).sum()
    FP = ((pred_pos) & (y_mel == 0)).sum()
    nb = TP / N - FP / N * pt / (1 - pt)
    net_benefit_model.append(nb)

net_benefit_all = y_mel.mean() - (1 - y_mel.mean()) * pts / (1 - pts)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(pts, net_benefit_model, color="#d62728", lw=2,
        label="Ensemble model")
ax.plot(pts, net_benefit_all,
        color="gray", lw=1, ls="--", label="Treat all")
ax.axhline(0, color="black", lw=1, label="Treat none")
ax.set_xlabel("Threshold probability (pt)")
ax.set_ylabel("Net benefit")
ax.set_title("Decision curve analysis - melanoma")
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim(-0.05, max(net_benefit_model) * 1.2)
plt.tight_layout()
plt.savefig(OUT / "decision_curve_melanoma.png", dpi=120)
plt.close()
print("saved decision_curve_melanoma.png")

# =============================================================================
#  Entropy histogram (model confidence)
# =============================================================================
H = -(oof_probs * np.log(oof_probs + 1e-12)).sum(axis=1)
correct = (y_pred == y)
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(H[correct], bins=50, alpha=0.6, color="green", label="Correct predictions")
ax.hist(H[~correct], bins=50, alpha=0.6, color="red", label="Incorrect predictions")
ax.set_xlabel("Shannon entropy (3 classes)")
ax.set_ylabel("Count")
ax.set_title("Prediction entropy")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "entropy_confidence.png", dpi=120)
plt.close()
print("saved entropy_confidence.png")

# =============================================================================
#  Summary text
# =============================================================================
lines = []
lines.append("=" * 72)
lines.append(
    "MODEL EVALUATION REPORT - " + MODEL_MANIFEST["bundle_id"]
    + " (handcraft + 2 × V2S+CBAM)"
)
lines.append("=" * 72)
lines.append(
    "Youden thresholds are selected using the same out-of-fold predictions; "
    "results are descriptive, not a prospective calibration."
)
lines.append(
    "The scaler and classifier are fitted within each fold. CNN features are "
    "precomputed; this is not an end-to-end held-out evaluation."
)
lines.append("")
lines.append(f"Total observations : {len(y)}")
for c in (0, 1, 2):
    lines.append(f"  Class {c} = {LABELS[c]:<10} n = {(y == c).sum()}")
lines.append("")
lines.append("--- Argmax classification metrics ---")
lines.append(f"  Accuracy : {(y_pred == y).mean():.4f}")
for c in (0, 1, 2):
    p = precision_score(y, y_pred, labels=[c], average="micro")
    r = recall_score(y, y_pred, labels=[c], average="micro")
    f = f1_score(y, y_pred, labels=[c], average="micro")
    lines.append(f"  {LABELS[c]:<10}  precision={p:.3f}  recall(sens)={r:.3f}  f1={f:.3f}")
lines.append("")
lines.append("--- Thresholds on multiclass probabilities used as one-vs-rest scores ---")
lines.append("  Method: Youden index = sensitivity + specificity - 1")
lines.append("")
for c in (0, 1, 2):
    t, sens, spec = optimal_thresholds[c]
    lines.append(f"  {LABELS[c]:<10}  T* = {t:.3f}   Sensitivity = {sens:.3f}   Specificity = {spec:.3f}")
lines.append("")
lines.append("--- Melanoma operating point ---")
t_mel, sens_mel, spec_mel = optimal_thresholds[1]
lines.append(f"  At threshold T*({LABELS[1]}) = {t_mel:.3f} : ")
lines.append(f"    - Sensitivity = {sens_mel:.3f}  -> detected {sens_mel*100:.1f} % of true melanomas")
lines.append(f"    - Specificity = {spec_mel:.3f}  -> {(1-spec_mel)*100:.1f} % false positives")
lines.append("")
lines.append("  Lower thresholds trade specificity for sensitivity; higher thresholds do the reverse.")
lines.append("  Quantify this trade-off from the evaluated predictions before choosing an operating point.")
lines.append("")
lines.append("--- Calibration ---")
lines.append("  In a reliability diagram, a curve below the diagonal indicates overconfidence;")
lines.append("  a curve above the diagonal indicates underconfidence.")
lines.append("  Recalibration methods require a separate evaluation; improvement is not guaranteed.")
lines.append("")
lines.append("--- Generated files in the local clinical report directory ---")
lines.append("  - calibration.png             (3 reliability diagrams)")
lines.append("  - roc_per_class.png           (3 ROC + Youden thresholds)")
lines.append("  - proba_histograms.png        (Probability distribution by class)")
lines.append("  - confusion_argmax.png        (Normalized confusion)")
lines.append("  - decision_curve_melanoma.png (decision curve analysis)")
lines.append("  - entropy_confidence.png      (Prediction entropy)")
lines.append("  - summary.txt                 (This file)")

summary = "\n".join(lines)
(OUT / "summary.txt").write_text(summary, encoding="utf-8")
print()
print(summary)
