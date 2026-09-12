"""Compare fractal estimators and morphology-ring radii on local lesion masks.

Evaluate box-counting and Minkowski-Bouligand descriptors using univariate
melanoma-versus-rest ROC-AUC and five-fold ROC-AUC of a LightGBM descriptor block.
Rank configurations on masks downsampled to 512 pixels. These exploratory
rankings require separate validation before selecting a production descriptor.

Run: python -m derm.analysis.legacy.optimize_border_fractal"""
import sys
from pathlib import Path

import cv2
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from derm.preprocess import border_fractal as bf
from derm.preprocess import features as F
from derm import paths

MASK_ROOT = paths.UNET_MASKS
CLASS_MAP = {"nevus": 0, "melanoma": 1, "atypical": 2}
RNG = np.random.RandomState(42)
N_PER = {"melanoma": 1200, "nevus": 800, "atypical": 400}  # 1200 mel vs 1200 rest
SEARCH_RES = 512

FRACTAL_CFG = {
    "minkowski_baseline":     None,
    "box_dyadic_2_32":        dict(box_sizes=(2, 4, 8, 16, 32), n_offsets=1),
    "box_dyadic_2_64":        dict(box_sizes=(2, 4, 8, 16, 32, 64), n_offsets=1),
    "box_fine_off3":          dict(box_sizes=(2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64), n_offsets=3),
    "box_norm_off1":          dict(box_sizes=(1/2, 1/4, 1/8, 1/16, 1/32, 1/64), n_offsets=1, normalize=True),
    "box_norm_off4":          dict(box_sizes=(1/2, 1/4, 1/8, 1/16, 1/32, 1/64), n_offsets=4, normalize=True),
    "box_norm_drop_big_off4": dict(box_sizes=(1/4, 1/8, 1/16, 1/32, 1/64), n_offsets=4, normalize=True),
}
RADII_CFG = {
    "baseline_1_16": (1, 2, 4, 8, 16),
    "fib_1_13":      (1, 2, 3, 5, 8, 13),
    "wide_2_32":     (2, 4, 8, 16, 32),
    "deep_1_64":     (1, 2, 4, 8, 16, 32, 64),
}


def log(*a):
    print(*a, flush=True)


def sample_masks():
    paths, labels = [], []
    for cname, cid in CLASS_MAP.items():
        pool = sorted((MASK_ROOT / "train" / cname).glob("*.png"))
        k = min(N_PER[cname], len(pool))
        for i in RNG.choice(len(pool), k, replace=False):
            paths.append(pool[i]); labels.append(cid)
    return paths, np.array(labels)


def load(paths):
    masks = []
    for p in paths:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is not None and (m.shape[0] != SEARCH_RES):
            m = cv2.resize(m, (SEARCH_RES, SEARCH_RES), interpolation=cv2.INTER_NEAREST)
        masks.append(None if m is None else (m > 0).astype(np.uint8))
    return masks


def fractal_vec(masks, cfg):
    out = np.zeros(len(masks))
    for i, m in enumerate(masks):
        if m is None or m.sum() == 0:
            continue
        out[i] = F.morpho_fractal(m)[-1] if cfg is None else bf.boxcount_D(m, **cfg)
    return out


def precompute_BD_rings(masks, radii):
    """B, D once + rings for this radii set."""
    n = len(masks)
    B = np.zeros(n); D = np.zeros(n)
    ins = np.zeros((n, len(radii))); ext = np.zeros((n, len(radii)))
    for i, m in enumerate(masks):
        if m is None or m.sum() == 0:
            continue
        B[i], D[i] = F.border_diameter(m)
        a, b = bf.morpho_rings(m, radii)
        ins[i] = a; ext[i] = b
    return B, D, ins, ext


def block_auc(X, y, n_splits=5):
    yb = (y == 1).astype(int)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    aucs = []
    for tr, te in cv.split(X, yb):
        m = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                           random_state=0, n_jobs=-1, verbose=-1).fit(X[tr], yb[tr])
        aucs.append(roc_auc_score(yb[te], m.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs))


def main():
    paths, y = sample_masks()
    log(f"sample: n={len(y)}  mel={int((y==1).sum())}  rest={int((y!=1).sum())}  res={SEARCH_RES}")
    masks = load(paths)
    log("masks loaded")

    log("\n=== fractal_D univariate melanoma-vs-rest AUC ===")
    frac = {}
    for name, cfg in FRACTAL_CFG.items():
        s = fractal_vec(masks, cfg); frac[name] = s
        a = roc_auc_score((y == 1).astype(int), s)
        log(f"  {name:<24} AUC={a:.4f}  |dev|={abs(a-0.5):.4f}   "
            f"(mel meanD={s[y==1].mean():.3f}  rest meanD={s[y!=1].mean():.3f})")
    best_frac = max(frac, key=lambda k: abs(roc_auc_score((y == 1).astype(int), frac[k]) - 0.5))
    log(f"  -> best fractal: {best_frac}")

    log("\n=== border/fractal BLOCK melanoma-vs-rest AUC (5-fold LGBM) ===")
    results = {}
    for rname, radii in RADII_CFG.items():
        B, D, ins, ext = precompute_BD_rings(masks, radii)
        for fname in ("minkowski_baseline", best_frac):
            X = np.column_stack([B, D, ins, ext, frac[fname]])
            a = block_auc(X, y)
            results[(fname, rname)] = a
            log(f"  fractal={fname:<24} radii={rname:<14} blockAUC={a:.4f}")
    base = results[("minkowski_baseline", "baseline_1_16")]
    best_key = max(results, key=results.get)
    log(f"\n  BASELINE (minkowski + radii 1-16)   blockAUC={base:.4f}")
    log(f"  BEST     ({best_key[0]} + {best_key[1]})   blockAUC={results[best_key]:.4f}")
    log(f"  gain = {results[best_key]-base:+.4f} AUC")


if __name__ == "__main__":
    main()
