"""
Test-Time Augmentation feature extraction for a trained EfficientNetV2-S+CBAM
model. For each image we run the model on the 8 dihedral-group orientations
(D4 = 4 rotations x 2 flips) and average the 1280-d penultimate features.

Outputs (depending on the input dir) :
  cnn_v2s_tta_features.csv          (from Skin_Cancer_Merged_lesion)
  cnn_v2s_dehair_tta_features.csv   (from Skin_Cancer_Merged_dehair)

Usage : python extract_tta.py {lesion|dehair}
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from derm.legacy.pipeline import cnn
from derm.pipeline import cnn_v2s  # registers ResNet18CBAM and EfficientNetV2SCBAM
from derm import paths


def parse_target():
    if len(sys.argv) < 2:
        print("usage: python extract_tta.py {lesion|dehair}")
        sys.exit(1)
    t = sys.argv[1]
    if t == "lesion":
        return (paths.LESION, "*.png",
                paths.CNN_V2S_BEST, paths.CNN_V2S_TTA_CSV)
    if t == "dehair":
        return (paths.DEHAIR, "*.jpg",
                paths.CNN_V2S_DEHAIR_BEST,
                paths.CNN_V2S_DEHAIR_TTA_CSV)
    print(f"unknown target: {t}")
    sys.exit(1)


SRC, EXT, MODEL_PATH, OUT_CSV = parse_target()
CLASS_MAP = {"nevus": 0, "melanoma": 1, "atypical": 2}
IMG_SIZE = 224
BATCH = 32
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PlainDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.tf(img), CLASS_MAP[p.parent.name]


def d4_augment(x):
    """Apply the 8 dihedral group transforms on a (B, C, H, W) batch.
    Returns a list of 8 (B, C, H, W) tensors."""
    # 4 rotations
    r0 = x
    r1 = torch.rot90(x, 1, dims=(2, 3))
    r2 = torch.rot90(x, 2, dims=(2, 3))
    r3 = torch.rot90(x, 3, dims=(2, 3))
    # + horizontal flip of each
    flips = [r0, r1, r2, r3,
             torch.flip(r0, dims=(3,)),
             torch.flip(r1, dims=(3,)),
             torch.flip(r2, dims=(3,)),
             torch.flip(r3, dims=(3,))]
    return flips


def main():
    paths = []
    for split in ("train", "valid", "test"):
        paths.extend(sorted((SRC / split).rglob(EXT)))
    print(f"{len(paths)} images   model={MODEL_PATH}")

    model = cnn.make_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.fc = nn.Identity()
    model.eval()

    loader = DataLoader(PlainDataset(paths), batch_size=BATCH,
                        shuffle=False, num_workers=4, pin_memory=True)

    feats_all, classes_all = [], []
    with torch.no_grad():
        for b_idx, (x, y) in enumerate(loader, 1):
            x = x.to(device, non_blocking=True)
            augs = d4_augment(x)                          # list of 8 tensors
            f_sum = None
            for ax in augs:
                f = model(ax)                             # (B, 1280)
                f_sum = f if f_sum is None else f_sum + f
            f_mean = (f_sum / len(augs)).cpu().numpy()
            feats_all.append(f_mean)
            classes_all.extend(y.tolist())
            if b_idx % 50 == 0:
                done = b_idx * BATCH
                print(f"  {done}/{len(paths)}")

    feats_all = np.concatenate(feats_all, axis=0)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"cnn_{i}" for i in range(feats_all.shape[1])] + ["Class"])
        for row, cls in zip(feats_all, classes_all):
            w.writerow([f"{v:.6f}" for v in row] + [cls])
    print(f"done -> {OUT_CSV}  ({feats_all.shape[0]} rows, "
          f"{feats_all.shape[1]} features, TTA x 8)")


if __name__ == "__main__":
    main()
