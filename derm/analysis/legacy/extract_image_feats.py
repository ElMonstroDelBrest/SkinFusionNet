"""Extract 23 image and contour descriptors aligned with local CNN feature rows.

Follow the historical extraction order: train, validation and test, with sorted
lesion-image paths within each split. Combine 16 texture/colour descriptors from
image_features.image_block with seven contour descriptors from optimize_border_v2.
The resulting local feature matrix supports probing and incremental ablations;
it is excluded from Git."""
import csv
from pathlib import Path
from multiprocessing import Pool

import cv2
import numpy as np

from derm.preprocess import image_features as IF
from derm.analysis.legacy import optimize_border_v2 as OB
from derm import paths as RP

LES = RP.LESION
UN = RP.UNET_MASKS
CLASS_MAP = {"nevus": 0, "melanoma": 1, "atypical": 2}
RES = 512

IMG_NAMES = IF.NAMES                       # 16
SHAPE_NAMES = ["solidity", "convexity", "defect_depth", "defect_count",
               "radial_fft_high", "radial_fft_2_5", "curv_var"]   # Seven descriptors, excluding duplicated fractal and Canny features.
NAMES = IMG_NAMES + SHAPE_NAMES


def canonical_paths():
    paths = []
    for s in ("train", "valid", "test"):
        paths += sorted((LES / s).rglob("*.png"))
    return paths


def one(p):
    cls = CLASS_MAP[p.parent.name]
    les = cv2.imread(str(p), cv2.IMREAD_COLOR)
    m = cv2.imread(str(UN / p.relative_to(LES)), cv2.IMREAD_GRAYSCALE)
    if les is None or m is None:
        return [0.0] * len(NAMES) + [cls]
    if les.shape[0] != RES:
        les = cv2.resize(les, (RES, RES), interpolation=cv2.INTER_LINEAR)
    if m.shape[0] != RES:
        m = cv2.resize(m, (RES, RES), interpolation=cv2.INTER_NEAREST)
    mb = (m > 0).astype(np.uint8)

    img = IF.image_block(les, mb * 255)                 # 16
    d = OB.descriptors(mb, les)                          # forme
    if d is None:
        shape = [0.0] * len(SHAPE_NAMES)
    else:
        shape = [float(d.get(k, 0.0)) for k in SHAPE_NAMES]
    row = [float(x) for x in img] + shape + [cls]
    return row


def main():
    paths = canonical_paths()
    print(f"extracting {len(paths)} rows × {len(NAMES)} image-derived descriptors")
    rows = []
    with Pool() as pool:
        for i, r in enumerate(pool.imap(one, paths, chunksize=32)):
            rows.append(r)
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(paths)}", flush=True)

    out = RP.IMAGE_FEATS_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(NAMES + ["Class"])
        w.writerows(rows)
    print(f"done -> {out}  ({len(rows)} rows)")

    # Check label alignment with cnn_features.csv.
    import pandas as pd
    a = pd.read_csv(out)['Class'].values
    b = pd.read_csv(RP.CNN_FEATS_CSV, usecols=['Class'])['Class'].values
    print("aligné avec cnn_features.csv :", len(a) == len(b) and bool((a == b).all()))


if __name__ == "__main__":
    main()
