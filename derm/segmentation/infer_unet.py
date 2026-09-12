"""
Apply the trained U-Net to every image of Skin_Cancer_Merged.

For each image we save :
  - the binary mask              -> Skin_Cancer_Merged_unet/<split>/<class>/<id>.png
  - the lesion-only RGB image    -> Skin_Cancer_Merged_lesion/<split>/<class>/<id>.png
    (everything outside the mask is set to black)
"""
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset

from derm import paths

SRC = paths.MERGED
DST_MASK = paths.UNET_MASKS
DST_LESION = paths.LESION
WEIGHTS = str(paths.UNET_BEST)
IMG_SIZE = 384
BATCH = 16
THRESHOLD = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LesionImages(Dataset):
    """Returns (resized tensor, path) -- we re-read the original image
    later to know its exact size and write the mask back."""

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(str(self.paths[i]), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        return tensor, str(self.paths[i])


def main():
    print(f"device: {device}")
    model = smp.Unet(encoder_name="efficientnet-b0",
                     encoder_weights=None, in_channels=3, classes=1).to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.eval()

    paths = [p for split in ("train", "valid", "test")
               for p in sorted((SRC / split).rglob("*.jpg"))]
    print(f"{len(paths)} images")

    loader = DataLoader(LesionImages(paths), batch_size=BATCH,
                        shuffle=False, num_workers=4, pin_memory=True)

    done = 0
    with torch.no_grad():
        for tensors, paths_b in loader:
            tensors = tensors.to(device, non_blocking=True)
            preds = torch.sigmoid(model(tensors)).cpu().numpy()  # (B, 1, IMG_SIZE, IMG_SIZE)

            for pred, p in zip(preds, paths_b):
                src_path = Path(p)
                orig = cv2.imread(p, cv2.IMREAD_COLOR)
                h, w = orig.shape[:2]

                mask = (pred[0] > THRESHOLD).astype(np.uint8) * 255
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

                rel = src_path.relative_to(SRC).with_suffix(".png")
                (DST_MASK / rel).parent.mkdir(parents=True, exist_ok=True)
                (DST_LESION / rel).parent.mkdir(parents=True, exist_ok=True)

                cv2.imwrite(str(DST_MASK / rel), mask)
                lesion = orig.copy()
                lesion[mask == 0] = 0
                cv2.imwrite(str(DST_LESION / rel), lesion)

                done += 1
                if done % 500 == 0:
                    print(f"  {done}/{len(paths)}")

    print(f"done. masks -> {DST_MASK}, lesion-only RGB -> {DST_LESION}")


if __name__ == "__main__":
    main()
