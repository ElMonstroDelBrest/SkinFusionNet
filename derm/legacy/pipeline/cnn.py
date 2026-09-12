"""
ResNet18 + CBAM (Woo 2018) for skin lesion 3-class classification.
The CBAM block (channel + spatial attention) is inserted between the last
residual stage (layer4) and the global average pool, so the 512-d
penultimate features are attention-weighted.

Outputs :
  cnn_best.pth      state_dict of the best epoch (by val accuracy)
  cnn_features.csv  512 CNN features + Class, same row order as features.csv
"""
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from derm import paths

LESION = paths.LESION
OUT_MODEL = paths.CNN_BEST
OUT_FEATS = paths.CNN_FEATS_CSV
CLASS_MAP = {"nevus": 0, "melanoma": 1, "atypical": 2}

IMG_SIZE = 224
BATCH = 64
EPOCHS = 12
LR = 1e-4
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------

class LesionDataset(Dataset):
    def __init__(self, paths, augment=False):
        self.paths = paths
        steps = [transforms.ToPILImage(),
                 transforms.Resize((IMG_SIZE, IMG_SIZE))]
        if augment:
            steps += [transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip(),
                      transforms.RandomRotation(15),
                      transforms.ColorJitter(0.1, 0.1, 0.1)]
        steps += [transforms.ToTensor(),
                  transforms.Normalize(MEAN, STD)]
        self.tf = transforms.Compose(steps)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.tf(img), CLASS_MAP[p.parent.name]


def collect(split):
    """Match the iteration order of features.py."""
    return sorted((LESION / split).rglob("*.png"))


# ---------------------------------------------------------------------------
#  CBAM = channel attention then spatial attention
# ---------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        red = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, red, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(red, channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1).view(b, c))
        mx  = self.mlp(F.adaptive_max_pool2d(x, 1).view(b, c))
        return x * torch.sigmoid(avg + mx).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel = ChannelAttention(channels)
        self.spatial = SpatialAttention()

    def forward(self, x):
        return self.spatial(self.channel(x))


# ---------------------------------------------------------------------------

class ResNet18CBAM(nn.Module):
    """ResNet18 backbone + CBAM block before global avg pool + FC head."""

    def __init__(self, n_classes=3):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.cbam = CBAM(512)
        self.avgpool = base.avgpool
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def make_model():
    return ResNet18CBAM(n_classes=3)


# ---------------------------------------------------------------------------

def train():
    train_paths = collect("train")
    val_paths = collect("valid")
    print(f"train: {len(train_paths)}, val: {len(val_paths)}")

    train_loader = DataLoader(LesionDataset(train_paths, augment=True),
                              batch_size=BATCH, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(LesionDataset(val_paths, augment=False),
                            batch_size=BATCH, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = make_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(train_loader.dataset)
        sch.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        v_acc = correct / total

        print(f"epoch {epoch:>2}  loss={tr_loss:.4f}  val_acc={v_acc:.4f}")
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), OUT_MODEL)
            print(f"  -> saved (best acc {best_acc:.4f})")

    print(f"best val acc = {best_acc:.4f}")


def extract():
    """Compute 512-d penultimate features for every image."""
    model = make_model().to(device)
    model.load_state_dict(torch.load(OUT_MODEL, map_location=device))
    model.fc = nn.Identity()
    model.eval()

    paths = []
    for split in ("train", "valid", "test"):
        paths.extend(collect(split))
    print(f"extracting from {len(paths)} images")

    loader = DataLoader(LesionDataset(paths, augment=False),
                        batch_size=BATCH, shuffle=False,
                        num_workers=4, pin_memory=True)

    feats, classes = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            feats.append(model(x).cpu().numpy())
            classes.extend(y.tolist())
    feats = np.concatenate(feats, axis=0)

    with open(OUT_FEATS, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"cnn_{i}" for i in range(feats.shape[1])] + ["Class"])
        for row, cls in zip(feats, classes):
            w.writerow([f"{v:.6f}" for v in row] + [cls])
    print(f"done -> {OUT_FEATS}  ({feats.shape[0]} rows, {feats.shape[1]} CNN features)")


if __name__ == "__main__":
    print(f"device: {device}")
    if not OUT_MODEL.exists():
        train()
    extract()
