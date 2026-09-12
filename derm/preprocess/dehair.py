"""
DullRazor : hair removal on dermoscopic images (Lee, Gallagher, Coldman, McLean 1997).

For each .jpg in Skin_Cancer_Merged/ :
  1. grayscale + morphological blackhat (cross 17x17) -> dark thin structures
  2. threshold -> hair mask
  3. inpaint the hair pixels in the colour image

Output : Skin_Cancer_Merged_dehair/<split>/<class>/<file>.jpg
"""
import cv2
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from derm import paths

SRC = paths.MERGED
DST = paths.DEHAIR
KERNEL = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))


def dehair(src_str):
    src = Path(src_str)
    dst = DST / src.relative_to(SRC)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, KERNEL)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    clean = cv2.inpaint(img, mask, 1, cv2.INPAINT_TELEA)
    cv2.imwrite(str(dst), clean)


if __name__ == "__main__":
    paths = [str(p) for split in ("train", "valid", "test")
                    for p in sorted((SRC / split).rglob("*.jpg"))]
    print(f"{len(paths)} images")
    with ProcessPoolExecutor() as pool:
        for i, _ in enumerate(pool.map(dehair, paths, chunksize=64), 1):
            if i % 1000 == 0:
                print(f"  {i} / {len(paths)}")
    print(f"done -> {DST}")
