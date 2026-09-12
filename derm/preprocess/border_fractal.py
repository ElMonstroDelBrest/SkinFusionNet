"""
Border / fractal descriptors for a binary lesion mask, box-counting flavour.

fractal_D (box-counting) : cover the lesion *boundary* with a grid of boxes of
side eps ; the number of occupied boxes N(eps) scales as N ~ eps^(-D), so
  D = slope of  log N(eps)  vs  log(1/eps).
- smooth curve (circle, square)  -> D ~ 1
- jagged / dentate border         -> 1 < D < 2
- space-filling                   -> D -> 2

morpho rings : internal/external pixel counts at a configurable set of disk
radii (kept for compatibility / as extra border-roughness features).

Run from the repository root: python3 -m derm.preprocess.border_fractal
"""
import cv2
import numpy as np

DEFAULT_BOXES = (2, 4, 8, 16, 32, 64)
DEFAULT_RADII = (1, 2, 4, 8, 16)


def _boundary(mask):
    """1-px inner boundary of the binary mask."""
    m = (mask > 0).astype(np.uint8)
    er = cv2.erode(m, np.ones((3, 3), np.uint8))
    return m - er


def boxcount_D(mask, box_sizes=DEFAULT_BOXES, n_offsets=1, normalize=False):
    """Box-counting fractal dimension of the lesion boundary.

    box_sizes : box sides in pixels (or, if normalize=True, fractions of the
                lesion bounding-box max side).
    n_offsets : average occupied-box counts over this many grid origins
                (reduces grid-placement quantisation noise).
    """
    b = _boundary(mask)
    pts = np.argwhere(b > 0).astype(np.int64)
    if len(pts) < 16:
        return 0.0
    H, W = mask.shape[:2]
    if normalize:
        y0, x0 = pts.min(0)
        side = int(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))) + 1
        pts = pts - [y0, x0]
        sizes = [max(1, int(round(f * side))) for f in box_sizes]
        span = side
    else:
        sizes = [int(s) for s in box_sizes]
        span = max(H, W)

    logx, logy = [], []
    for eps in sizes:
        if eps < 1 or eps > span:
            continue
        ncols = span // eps + 2
        counts = []
        n_off = max(1, n_offsets)
        for k in range(n_off):
            off = (eps * k) // n_off
            q = (pts + off) // eps
            ids = q[:, 0] * ncols + q[:, 1]
            counts.append(np.unique(ids).size)
        N = float(np.mean(counts))
        logx.append(np.log(1.0 / eps))
        logy.append(np.log(max(N, 1.0)))
    if len(logx) < 2:
        return 0.0
    return float(np.polyfit(logx, logy, 1)[0])


def morpho_rings(mask, radii=DEFAULT_RADII):
    """internal_R + external_R pixel counts for each radius (border roughness)."""
    m = (mask > 0).astype(np.uint8)
    internals, externals = [], []
    for R in radii:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * R + 1, 2 * R + 1))
        internals.append(float(int((m - cv2.erode(m, k)).sum())))
        externals.append(float(int((cv2.dilate(m, k) - m).sum())))
    return internals, externals


# ---------------------------------------------------------------------------
#  self-test on synthetic shapes : box-counting D should be ~1 for smooth,
#  >1 for jagged. (mirrors the README's Minkowski validation table)
# ---------------------------------------------------------------------------
def _synthetic():
    S = 512
    shapes = {}

    disk = np.zeros((S, S), np.uint8)
    cv2.circle(disk, (S // 2, S // 2), 180, 255, -1)
    shapes["disk"] = disk

    sq = np.zeros((S, S), np.uint8)
    cv2.rectangle(sq, (100, 100), (412, 412), 255, -1)
    shapes["square"] = sq

    star = np.zeros((S, S), np.uint8)
    cx = cy = S // 2
    pts = []
    for i in range(10):
        r = 200 if i % 2 == 0 else 80
        a = np.pi * i / 5
        pts.append([cx + r * np.cos(a), cy + r * np.sin(a)])
    cv2.fillPoly(star, [np.array(pts, np.int32)], 255)
    shapes["star5"] = star

    # spiky / dentate circle (high-frequency radial noise on the boundary)
    spiky = np.zeros((S, S), np.uint8)
    angs = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    rng = np.random.RandomState(0)
    rad = 160 + 35 * np.sin(angs * 24) + rng.uniform(-12, 12, angs.size)
    poly = np.stack([cx + rad * np.cos(angs), cy + rad * np.sin(angs)], 1).astype(np.int32)
    cv2.fillPoly(spiky, [poly], 255)
    shapes["spiky"] = spiky

    print(f"{'shape':<10}{'boxD':>8}{'boxD(off=4)':>14}{'boxD(norm)':>14}")
    for name, m in shapes.items():
        d1 = boxcount_D(m)
        d2 = boxcount_D(m, n_offsets=4)
        d3 = boxcount_D(m, box_sizes=[1/2, 1/4, 1/8, 1/16, 1/32, 1/64], normalize=True)
        print(f"{name:<10}{d1:>8.3f}{d2:>14.3f}{d3:>14.3f}")


if __name__ == "__main__":
    _synthetic()
