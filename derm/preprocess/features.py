"""
Lean feature set per lesion image, with Fourier rotation-invariant block.

Inputs :
  Skin_Cancer_Merged_unet/   (binary lesion masks)
  Skin_Cancer_Merged_lesion/ (RGB lesion-only images)

Output : features.csv  -- 18 features + Class
  3   geometry      : A4, B, D
  4   colour        : Lab_a_kurt, Lab_b_std, Lab_a_std, Lab_b_skew
  11  morpho        : internal_R{1,2,4,8,16}, external_R{1,2,4,8,16}, fractal_D
"""
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import kurtosis, skew

from derm import paths

MASK_ROOT = paths.UNET_MASKS
LESION_ROOT = paths.LESION
OUT = paths.FEATURES_CSV
CLASS_MAP = {"nevus": 0, "melanoma": 1, "atypical": 2}
MORPHO_RADII = (1, 2, 4, 8, 16)


# ---------------------------------------------------------------------------
#  A4 -- Pearson correlation of row vs column projections of the mask
# ---------------------------------------------------------------------------
def asymmetry_A4(binary):
    if binary.sum() == 0:
        return 1.0
    H = binary.sum(axis=1).astype(float)
    V = binary.sum(axis=0).astype(float)
    n = min(len(H), len(V))
    H, V = H[:n], V[:n]
    if H.std() == 0 or V.std() == 0:
        return 1.0
    return float(np.corrcoef(H, V)[0, 1])


# ---------------------------------------------------------------------------
#  B = P^2 / (4 pi A)        D = Feret diameter (max distance on convex hull)
# ---------------------------------------------------------------------------
def border_diameter(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0, 0.0
    c = max(contours, key=cv2.contourArea)
    A = cv2.contourArea(c)
    P = cv2.arcLength(c, True)
    if A == 0:
        return 0.0, 0.0
    B = (P * P) / (4 * np.pi * A)
    hull = cv2.convexHull(c).reshape(-1, 2)
    if len(hull) < 2:
        return float(B), 0.0
    d = hull[:, None, :] - hull[None, :, :]
    D = float(np.sqrt((d ** 2).sum(axis=-1)).max())
    return float(B), D


# ---------------------------------------------------------------------------
#  Morpho fractal : Minkowski-Bouligand dimension via erosion / dilation rings
# ---------------------------------------------------------------------------
#  For each radius R in MORPHO_RADII we build a structuring disk of size R and :
#     internal_R = | binary  -  erode(binary, disk(R)) |    inner ring
#     external_R = | dilate(binary, disk(R))  -  binary |   outer ring
#     boundary_R = internal_R + external_R                 thick contour
#
#  Minkowski-Bouligand dimension is defined by how fast boundary_R grows with R :
#     - smooth 1D contour in 2D     -> boundary_R ~ R   (linear, D = 1)
#     - jagged / fractal contour    -> boundary_R ~ R^(2 - D)   with 1 < D < 2
#     - space-filling shape         -> boundary_R = constant (D = 2)
#
#  Take the log of both sides :
#     log boundary_R = (2 - D) * log R + cst
#  A linear fit on (log R, log boundary_R) gives a slope = (2 - D),
#  hence  D = 2 - slope.
#
#  Practical interpretation :
#     D ~ 1.0  ->  perfect disk / square (smooth)
#     D ~ 1.1  ->  regular nevus
#     D ~ 1.3  ->  melanoma with irregular border
#     D >= 1.5 ->  very dentelled / spiky shape
# ---------------------------------------------------------------------------
def morpho_fractal(binary):
    if binary.sum() == 0:
        return [0.0] * (2 * len(MORPHO_RADII) + 1)

    internals, externals, boundary = [], [], []
    for R in MORPHO_RADII:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * R + 1, 2 * R + 1))
        eroded = cv2.erode(binary, k)
        dilated = cv2.dilate(binary, k)
        i = int((binary - eroded).sum())
        e = int((dilated - binary).sum())
        internals.append(float(i))
        externals.append(float(e))
        boundary.append(i + e)

    # Linear fit log boundary_R = slope * log R + cst  -> D = 2 - slope
    R_log = np.log(MORPHO_RADII)
    b_log = np.log([max(b, 1) for b in boundary])
    slope = float(np.polyfit(R_log, b_log, 1)[0])
    fractal_D = 2.0 - slope
    return internals + externals + [fractal_D]


# ---------------------------------------------------------------------------
#  4 Lab features on the lesion pixels
# ---------------------------------------------------------------------------
def lab_features(img_bgr, mask):
    fg = mask > 0
    if fg.sum() == 0:
        return 0.0, 0.0, 0.0, 0.0
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    a = lab[..., 1][fg].astype(np.float64)
    b = lab[..., 2][fg].astype(np.float64)
    a_kurt = float(kurtosis(a)) if a.std() > 0 else 0.0
    a_std = float(a.std())
    b_std = float(b.std())
    b_skew = float(skew(b)) if b.std() > 0 else 0.0
    return a_kurt, b_std, a_std, b_skew


# ---------------------------------------------------------------------------
#  Per-image pipeline
# ---------------------------------------------------------------------------
def process_one(mask_path_str):
    mask_path = Path(mask_path_str)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    binary = (mask > 0).astype(np.uint8)

    lesion_path = LESION_ROOT / mask_path.relative_to(MASK_ROOT)
    img = cv2.imread(str(lesion_path), cv2.IMREAD_COLOR)
    if img is None:
        return None

    A4 = asymmetry_A4(binary)
    B, D = border_diameter(binary)
    a_kurt, b_std, a_std, b_skew = lab_features(img, binary)
    morpho = morpho_fractal(binary)
    cls = CLASS_MAP[mask_path.parent.name]
    return [A4, B, D, a_kurt, b_std, a_std, b_skew] + morpho + [cls]


def build_header():
    cols = ["A4", "B", "D",
            "Lab_a_kurt", "Lab_b_std", "Lab_a_std", "Lab_b_skew"]
    cols += [f"internal_R{R}" for R in MORPHO_RADII]
    cols += [f"external_R{R}" for R in MORPHO_RADII]
    cols += ["fractal_D", "Class"]
    return cols


if __name__ == "__main__":
    paths = [str(p) for split in ("train", "valid", "test")
                    for p in sorted((MASK_ROOT / split).rglob("*.png"))]
    print(f"{len(paths)} images")

    header = build_header()
    with open(OUT, "w", newline="") as f, ProcessPoolExecutor() as pool:
        w = csv.writer(f)
        w.writerow(header)
        for i, r in enumerate(pool.map(process_one, paths, chunksize=64), 1):
            if r is None:
                continue
            w.writerow([f"{v:.6f}" if isinstance(v, float) else v for v in r])
            if i % 1000 == 0:
                print(f"  {i} / {len(paths)}")
    print(f"done -> {OUT}  ({len(header)} columns)")
