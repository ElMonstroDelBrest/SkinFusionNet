"""Evaluate alternative contour and texture descriptors for melanoma separation.

Compare the baseline Minkowski fractal descriptor with solidity, convexity,
normalized convexity-defect depth, defect count, radial Fourier power in low
and high harmonics, contour turning-angle variance, and within-mask Canny
box-counting dimension. Report univariate and combined-block ROC-AUC.
These exploratory comparisons do not establish external robustness.

Run: python -m derm.analysis.legacy.optimize_border_v2"""
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
LES_ROOT = paths.LESION
CLASS_MAP = {"nevus": 0, "melanoma": 1, "atypical": 2}
RNG = np.random.RandomState(42)
N_PER = {"melanoma": 1200, "nevus": 800, "atypical": 400}
RES = 512


def log(*a): print(*a, flush=True)


def sample():
    paths, y = [], []
    for c, cid in CLASS_MAP.items():
        pool = sorted((MASK_ROOT / "train" / c).glob("*.png"))
        for i in RNG.choice(len(pool), min(N_PER[c], len(pool)), replace=False):
            paths.append(pool[i]); y.append(cid)
    return paths, np.array(y)


def largest_contour(m):
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    return max(cs, key=cv2.contourArea)


def radial_signature(c, n=256):
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    pts = c.reshape(-1, 2).astype(np.float64)
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    rad = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    order = np.argsort(ang)
    ang, rad = ang[order], rad[order]
    grid = np.linspace(-np.pi, np.pi, n, endpoint=False)
    r = np.interp(grid, ang, rad, period=2 * np.pi)
    if r.mean() == 0:
        return None
    return r / r.mean()  # scale-invariant


def descriptors(m, lesion=None):
    d = {}
    c = largest_contour(m)
    if c is None or len(c) < 8 or cv2.contourArea(c) < 20:
        return None
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    hull = cv2.convexHull(c)
    harea = cv2.contourArea(hull)
    hperi = cv2.arcLength(hull, True)

    d["fractal_mink"] = F.morpho_fractal(m)[-1]
    d["solidity"] = area / harea if harea > 0 else 1.0
    d["convexity"] = hperi / peri if peri > 0 else 1.0

    # convexity defects
    try:
        hidx = cv2.convexHull(c, returnPoints=False)
        defs = cv2.convexityDefects(c, hidx)
    except cv2.error:
        defs = None
    if defs is not None:
        depths = defs[:, 0, 3] / 256.0
        norm = np.sqrt(area)
        d["defect_depth"] = float(depths.mean() / norm) if norm > 0 else 0.0
        d["defect_count"] = float((depths > 0.05 * norm).sum())
    else:
        d["defect_depth"] = 0.0; d["defect_count"] = 0.0

    # radial-signature FFT
    r = radial_signature(c)
    if r is not None:
        P = np.abs(np.fft.rfft(r - r.mean())) ** 2
        tot = P[1:].sum()
        d["radial_fft_high"] = float(P[6:].sum() / tot) if tot > 0 else 0.0
        d["radial_fft_2_5"] = float(P[2:6].sum() / tot) if tot > 0 else 0.0
    else:
        d["radial_fft_high"] = 0.0; d["radial_fft_2_5"] = 0.0

    # curvature variance (turning angle)
    pts = c.reshape(-1, 2).astype(np.float64)
    if len(pts) > 12:
        step = max(1, len(pts) // 128)
        p = pts[::step]
        v = np.diff(np.vstack([p, p[:1]]), axis=0)
        ang = np.arctan2(v[:, 1], v[:, 0])
        dang = np.diff(np.concatenate([ang, ang[:1]]))
        dang = (dang + np.pi) % (2 * np.pi) - np.pi
        d["curv_var"] = float(np.var(dang))
    else:
        d["curv_var"] = 0.0

    # canny-fractal inside mask (needs lesion image)
    if lesion is not None:
        gray = cv2.cvtColor(lesion, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        edges = cv2.bitwise_and(edges, edges, mask=m)
        d["canny_fractal"] = bf.boxcount_D(edges, box_sizes=(2, 4, 8, 16, 32, 64), n_offsets=2)
    return d


def main():
    paths, y = sample()
    log(f"sample n={len(y)}  mel={int((y==1).sum())}  rest={int((y!=1).sum())}  res={RES}")
    keys = None
    rows = []
    keep = []
    for k, p in enumerate(paths):
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape[0] != RES:
            m = cv2.resize(m, (RES, RES), interpolation=cv2.INTER_NEAREST)
        m = (m > 0).astype(np.uint8)
        lp = LES_ROOT / p.relative_to(MASK_ROOT)
        les = cv2.imread(str(lp), cv2.IMREAD_COLOR)
        if les is not None and les.shape[0] != RES:
            les = cv2.resize(les, (RES, RES), interpolation=cv2.INTER_LINEAR)
        d = descriptors(m, les)
        if d is None:
            continue
        if keys is None:
            keys = list(d.keys())
        rows.append([d[kk] for kk in keys]); keep.append(y[k])
        if (k + 1) % 600 == 0:
            log(f"  {k+1}/{len(paths)}")
    X = np.array(rows); yy = np.array(keep)
    yb = (yy == 1).astype(int)
    log(f"computed {len(yy)} samples, {len(keys)} descriptors\n")

    log("=== AUC univariée melanome-vs-reste (orientée, |dev| vs 0.5) ===")
    aucs = {}
    for j, kk in enumerate(keys):
        a = roc_auc_score(yb, X[:, j])
        aucs[kk] = a
        log(f"  {kk:<18} AUC={a:.4f}   |dev|={abs(a-0.5):.4f}")

    # block of the NEW shape descriptors (exclude fractal_mink to isolate the gain)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    def block(cols):
        Xs = X[:, [keys.index(c) for c in cols]]
        s = []
        for tr, te in cv.split(Xs, yb):
            mdl = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                                 random_state=0, n_jobs=-1, verbose=-1).fit(Xs[tr], yb[tr])
            s.append(roc_auc_score(yb[te], mdl.predict_proba(Xs[te])[:, 1]))
        return float(np.mean(s))

    new_shape = ["solidity", "convexity", "defect_depth", "defect_count",
                 "radial_fft_high", "radial_fft_2_5", "curv_var"]
    log("\n=== AUC bloc (LGBM 5-fold) ===")
    log(f"  fractal_mink seul                 : {block(['fractal_mink']):.4f}")
    log(f"  nouveaux descripteurs de forme    : {block(new_shape):.4f}")
    if "canny_fractal" in keys:
        log(f"  + canny_fractal                   : {block(new_shape + ['canny_fractal']):.4f}")
    log(f"  TOUT (forme + fractal + canny)    : {block(keys):.4f}")
    log(f"\n  (rappel : bloc bord/fractale d'origine ~0.64, fractal univarié 0.58)")


if __name__ == "__main__":
    main()
