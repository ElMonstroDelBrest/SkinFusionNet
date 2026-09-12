"""
Image-derived (NOT mask-silhouette) tabular features for a lesion.

All computed on the *pixels* inside the mask of the (dehaired) lesion — texture
and colour — so they degrade gracefully when the U-Net mask is imperfect OOD,
unlike the silhouette features (B, D, fractal, morpho rings).

block (16 features):
  canny_fractal                         box-counting D of Canny edges intra-lesion
  glcm_contrast/dissim/homog/energy/corr 16-level GLCM, offsets (0,1)+(1,0)
  lbp_entropy, lbp_uniformity           8-neighbour LBP histogram summary
  lap_var                               variance of the Laplacian (edge energy)
  grad_mean, grad_std                   Sobel gradient magnitude stats
  hue_circstd, sat_mean, sat_std, val_std  HSV colour spread
  n_colors                              # of populated 3-bit/channel colour bins
"""
import cv2
import numpy as np

from derm.preprocess import border_fractal as bf
from derm import paths

NAMES = ["canny_fractal", "glcm_contrast", "glcm_dissim", "glcm_homog",
         "glcm_energy", "glcm_corr", "lbp_entropy", "lbp_uniformity",
         "lap_var", "grad_mean", "grad_std",
         "hue_circstd", "sat_mean", "sat_std", "val_std", "n_colors"]


def _glcm(gray, m, levels=16):
    g = (gray.astype(np.int64) * levels // 256).clip(0, levels - 1)
    P = np.zeros((levels, levels), np.float64)
    for (sa, sb, ma_, mb_) in (
        (g[:, :-1], g[:, 1:], m[:, :-1], m[:, 1:]),
        (g[:-1, :], g[1:, :], m[:-1, :], m[1:, :]),
    ):
        v = (ma_ > 0) & (mb_ > 0)
        np.add.at(P, (sa[v], sb[v]), 1.0)
    P = P + P.T
    s = P.sum()
    if s == 0:
        return [0.0] * 5
    P /= s
    i = np.arange(levels)
    I, J = np.meshgrid(i, i, indexing="ij")
    contrast = float((P * (I - J) ** 2).sum())
    dissim = float((P * np.abs(I - J)).sum())
    homog = float((P / (1.0 + (I - J) ** 2)).sum())
    energy = float(np.sqrt((P ** 2).sum()))
    mu_i = (P * I).sum(); mu_j = (P * J).sum()
    si = np.sqrt((P * (I - mu_i) ** 2).sum()); sj = np.sqrt((P * (J - mu_j) ** 2).sum())
    corr = float((P * (I - mu_i) * (J - mu_j)).sum() / (si * sj)) if si > 0 and sj > 0 else 0.0
    return [contrast, dissim, homog, energy, corr]


def _lbp(gray, m):
    g = gray.astype(np.int32)
    gp = np.pad(g, 1, mode="edge")
    H, W = g.shape
    c = gp[1:-1, 1:-1]
    code = np.zeros_like(c)
    neigh = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for k, (dy, dx) in enumerate(neigh):
        nb = gp[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        code |= ((nb >= c).astype(np.int32) << k)
    vals = code[m > 0]
    if vals.size == 0:
        return [0.0, 0.0]
    hist = np.bincount(vals, minlength=256).astype(np.float64)
    hist /= hist.sum()
    nz = hist[hist > 0]
    entropy = float(-(nz * np.log2(nz)).sum())
    uniformity = float((hist ** 2).sum())
    return [entropy, uniformity]


def _colour(bgr, m):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sel = m > 0
    if sel.sum() == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    hue = hsv[..., 0][sel].astype(np.float64) * 2.0 * np.pi / 180.0  # 0..2pi
    sat = hsv[..., 1][sel].astype(np.float64)
    val = hsv[..., 2][sel].astype(np.float64)
    R = np.hypot(np.cos(hue).mean(), np.sin(hue).mean())
    hue_circstd = float(np.sqrt(-2.0 * np.log(R))) if R > 1e-9 else 0.0
    # quantised colour count (3 bits / channel)
    b = (bgr[sel].astype(np.int64) >> 5)
    key = b[:, 0] * 64 + b[:, 1] * 8 + b[:, 2]
    cnt = np.bincount(key, minlength=512)
    n_colors = float((cnt > 0.01 * sel.sum()).sum())
    return [hue_circstd, float(sat.mean()), float(sat.std()), float(val.std()), n_colors]


def image_block(bgr, mask):
    """bgr : dehaired lesion BGR (full frame) ; mask : 0/255 lesion mask."""
    m = (mask > 0).astype(np.uint8)
    if m.sum() < 32:
        return [0.0] * len(NAMES)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.bitwise_and(cv2.Canny(gray, 40, 120), m * 255)
    canny_fd = bf.boxcount_D(edges, box_sizes=(2, 4, 8, 16, 32, 64), n_offsets=2)
    glcm = _glcm(gray, m)
    lbp = _lbp(gray, m)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F)[m > 0].var())
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gmag = np.hypot(gx, gy)[m > 0]
    grad = [float(gmag.mean()), float(gmag.std())]
    col = _colour(bgr, m)
    return [canny_fd] + glcm + lbp + [lap] + grad + col


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m derm.preprocess.image_features IMAGE_PATH")
    p = sys.argv[1]
    bgr = cv2.imread(p)
    bgr = cv2.resize(bgr, (512, 512))
    mask = np.zeros((512, 512), np.uint8)
    cv2.circle(mask, (256, 256), 150, 255, -1)
    v = image_block(bgr, mask)
    for n, x in zip(NAMES, v):
        print(f"  {n:<16} {x:.4f}")
