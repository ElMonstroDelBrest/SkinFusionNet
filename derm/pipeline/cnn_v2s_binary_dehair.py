"""Train EfficientNetV2-S+CBAM as nevus-vs-melanoma on DullRazor views.

Does not overwrite the 3-class dehair checkpoint or the lesion 2-class weights.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from derm import paths
from derm.legacy.pipeline import cnn
from derm.pipeline.cnn_v2s import EfficientNetV2SCBAM

OUT_DIR = paths.MODELS / "binary"
OUT_MODEL = OUT_DIR / "cnn_v2s_binary_dehair.pth"
EPOCHS = 12


def collect_binary(split: str) -> list[Path]:
    root = paths.DEHAIR / split
    paths_ = []
    for cls in ("nevus", "melanoma"):
        folder = root / cls
        if folder.exists():
            paths_.extend(sorted(folder.glob("*.jpg")))
    return paths_


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cnn.LESION = paths.DEHAIR
    cnn.OUT_MODEL = OUT_MODEL
    cnn.BATCH = 32
    cnn.EPOCHS = EPOCHS
    cnn.CLASS_MAP = {"nevus": 0, "melanoma": 1}
    cnn.make_model = lambda: EfficientNetV2SCBAM(n_classes=2)

    train_paths = collect_binary("train")
    val_paths = collect_binary("valid")
    print(f"device={cnn.device}  train={len(train_paths)}  val={len(val_paths)}", flush=True)

    train_loader = DataLoader(
        cnn.LesionDataset(train_paths, augment=True),
        batch_size=cnn.BATCH,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        cnn.LesionDataset(val_paths, augment=False),
        batch_size=cnn.BATCH,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    model = cnn.make_model().to(cnn.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cnn.LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0.0
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(cnn.device, non_blocking=True), y.to(cnn.device, non_blocking=True)
            loss = loss_fn(model(x), y)
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
                x, y = x.to(cnn.device, non_blocking=True), y.to(cnn.device, non_blocking=True)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        acc = correct / total
        history.append((epoch, tr_loss, acc))
        star = ""
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), OUT_MODEL)
            star = "  *best*"
        print(f"epoch {epoch:02d}/{EPOCHS}  loss={tr_loss:.4f}  val_acc={acc:.4f}{star}", flush=True)
    print(f"wrote {OUT_MODEL}  best_val_acc={best_acc:.4f}", flush=True)
    (OUT_DIR / "cnn_v2s_binary_dehair_log.txt").write_text(
        "epoch,train_loss,val_acc\n"
        + "\n".join(f"{e},{loss:.6f},{acc:.6f}" for e, loss, acc in history)
        + f"\n# best_val_acc={best_acc:.6f}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
