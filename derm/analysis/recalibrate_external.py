"""Explore melanoma threshold recalibration on local external predictions.

Read external_val/per_image.csv from derm.export.eval_external, including true
class, melanoma probability and segmentation coverage. Separate errors related
to the inherited decision threshold from errors associated with low mask
coverage. Write console output and a local recalibration CSV.
Thresholds selected on this evaluation set are descriptive, not independently
validated operating points.

Run: python -m derm.analysis.recalibrate_external"""
import csv

import numpy as np
from sklearn.metrics import roc_auc_score

from derm import paths

T_CLINICAL = 0.188        # Inherited deployed threshold selected using internal cross-validation.
TARGET_SENS = 0.90        # Target sensitivity.
COV_FAIL = 0.05           # Near-empty mask: possible segmentation failure.


def load():
    rows = list(csv.DictReader(open(paths.EXTERNAL_VAL / "per_image.csv")))
    y = np.array([int(r["true"]) for r in rows])          # 1 = melanoma, 0 = nevus
    p = np.array([float(r["p_mel"]) for r in rows])
    cov = np.array([float(r["coverage"]) for r in rows])
    files = [r["file"] for r in rows]
    return y, p, cov, files


def sens_spec(y, p, t):
    pred = p >= t
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return sens, spec, tp, fn, tn, fp


def youden_threshold(y, p):
    ts = np.unique(p)
    best_t, best_j = 0.5, -1.0
    for t in ts:
        se, sp, *_ = sens_spec(y, p, t)
        j = se + sp - 1
        if j > best_j:
            best_j, best_t = j, t
    return best_t, best_j


def threshold_for_sensitivity(y, p, target):
    """Return the highest threshold meeting the target sensitivity."""
    ts = np.unique(p)
    best = 0.0
    for t in ts:
        se, *_ = sens_spec(y, p, t)
        if se >= target:
            best = max(best, t)
    return best


def main():
    y, p, cov, files = load()
    n_mel, n_nev = int((y == 1).sum()), int((y == 0).sum())
    auc = roc_auc_score(y, p)
    print(f"n = {len(y)}  (mélanomes={n_mel}, nevus={n_nev})")
    print(f"AUC mélanome OvR (OOD) = {auc:.4f}\n")

    t_you, j = youden_threshold(y, p)
    t_s90 = threshold_for_sensitivity(y, p, TARGET_SENS)

    print(f"{'seuil':<26} {'T*':>7} {'sens':>7} {'spec':>7} {'TP/FN':>8} {'FP':>5}")
    out_rows = []
    for name, t in (("clinique (déployé)", T_CLINICAL),
                    ("Youden externe", t_you),
                    (f"cible sens>={TARGET_SENS:.2f}", t_s90)):
        se, sp, tp, fn, tn, fp = sens_spec(y, p, t)
        print(f"{name:<26} {t:>7.3f} {se:>7.3f} {sp:>7.3f} {f'{tp}/{fn}':>8} {fp:>5}")
        out_rows.append(dict(seuil=name, T=round(t, 4), sens=round(se, 4),
                             spec=round(sp, 4), TP=tp, FN=fn, FP=fp))

    # Bootstrap recalibrated-threshold variability on the small external sample.
    rng = np.random.RandomState(0)
    idx = np.arange(len(y))
    tlist, selist, splist = [], [], []
    for _ in range(2000):
        b = rng.choice(idx, len(idx), replace=True)
        if 0 < y[b].sum() < len(b):
            tb, _ = youden_threshold(y[b], p[b])
            se, sp, *_ = sens_spec(y, p, tb)   # Evaluate on the full sample.
            tlist.append(tb); selist.append(se); splist.append(sp)
    q = lambda a: (np.percentile(a, 2.5), np.percentile(a, 97.5))
    print(f"\nBootstrap (2000) du Youden externe :")
    print(f"  T* médian = {np.median(tlist):.3f}  IC95 [{q(tlist)[0]:.3f}, {q(tlist)[1]:.3f}]  <- très large = seuil instable à n={n_mel}")
    print(f"  sens @ T*boot : médiane {np.median(selist):.3f}  IC95 [{q(selist)[0]:.3f}, {q(selist)[1]:.3f}]")

    # Separate threshold-related errors from low-coverage cases.
    print(f"\n=== Mélanomes RATÉS au seuil clinique (p_mel < {T_CLINICAL}) ===")
    print(f"{'file':<28} {'p_mel':>7} {'coverage':>9}  cause probable")
    seg_fail = recovered = confident = 0
    for i in np.where((y == 1) & (p < T_CLINICAL))[0]:
        c = cov[i]
        if c < COV_FAIL:
            cause = "SEGMENTATION (masque ~vide)"; seg_fail += 1
        elif p[i] >= t_you:
            cause = "seuil (rattrapé par recalibration)"; recovered += 1
        else:
            cause = "erreur modèle confiante"; confident += 1
        print(f"{files[i]:<28} {p[i]:>7.3f} {c:>9.3f}  {cause}")
    print(f"\nRésumé des ratés cliniques : segmentation={seg_fail}  rattrapés-par-seuil={recovered}  erreurs-modèle={confident}")
    print(f"=> recalibrer le seuil ne corrige QUE les {recovered} ; les {seg_fail} exigent de durcir le U-Net.")

    out = paths.RECALIBRATION_EXTERNAL_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
