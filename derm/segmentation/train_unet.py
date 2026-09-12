"""
Train a U-Net for skin lesion segmentation on ISIC 2018 Task 1.

Expected layout :
  isic2018/
    images/   ISIC_xxxxxxx.jpg                (2594 files)
    masks/    ISIC_xxxxxxx_segmentation.png   (2594 files)

Output : unet_best.pth (state_dict of the best model by validation Dice)
"""
import random
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset
from derm import paths

ROOT = paths.ISIC2018
IMG_SIZE = 384
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4
VAL_FRAC = 0.15
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


class ISICDataset(Dataset):
    def __init__(self, pairs, augment=False):
        self.pairs = pairs
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img_path, mask_path = self.pairs[i]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if random.random() < 0.5:
                img, mask = np.fliplr(img).copy(), np.fliplr(mask).copy()
            if random.random() < 0.5:
                img, mask = np.flipud(img).copy(), np.flipud(mask).copy()
            k = random.randint(0, 3)
            if k:
                img = np.rot90(img, k).copy()
                mask = np.rot90(mask, k).copy()

        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = (mask > 127).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask


def dice_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    return (2 * inter + eps) / (pred.sum() + target.sum() + eps)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    # Build (image, mask) pairs
    imgs = sorted((ROOT / "images").glob("*.jpg"))
    pairs = []
    for img in imgs:
        m = ROOT / "masks" / f"{img.stem}_segmentation.png"
        if m.exists():
            pairs.append((img, m))
    print(f"{len(pairs)} image+mask pairs")

    random.shuffle(pairs)
    n_val = int(len(pairs) * VAL_FRAC)
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    print(f"train: {len(train_pairs)}, val: {len(val_pairs)}")

    train_loader = DataLoader(ISICDataset(train_pairs, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(ISICDataset(val_pairs, augment=False),
                            batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    # U-Net with EfficientNet-B0 encoder pretrained on ImageNet
    model = smp.Unet(encoder_name="efficientnet-b0",
                     encoder_weights="imagenet",
                     in_channels=3, classes=1).to(device)

    loss_fn = smp.losses.DiceLoss(mode="binary", from_logits=True)
    bce = torch.nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=LR)

    best_dice = 0.0
    for epoch in range(1, EPOCHS + 1):
        # train
        model.train()
        tr_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = loss_fn(logits, y) + bce(logits, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(train_loader.dataset)

        # validate
        model.eval()
        dices = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                pred = torch.sigmoid(model(x))
                for p, t in zip(pred, y):
                    dices.append(dice_score(p, t).item())
        v_dice = float(np.mean(dices))

        print(f"epoch {epoch:>2}  train_loss={tr_loss:.4f}  val_dice={v_dice:.4f}")

        if v_dice > best_dice:
            best_dice = v_dice
            torch.save(model.state_dict(), str(paths.UNET_BEST))
            print(f"  -> saved unet_best.pth (dice {best_dice:.4f})")

    print(f"done. best val dice = {best_dice:.4f}")


if __name__ == "__main__":
    main()
