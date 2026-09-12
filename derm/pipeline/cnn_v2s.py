"""
EfficientNetV2-S + CBAM for skin lesion 3-class classification.
Same recipe as cnn.py, just swap the backbone (12M -> 22M params,
output dim 512 -> 1280).

Outputs :
  cnn_v2s_best.pth
  cnn_v2s_features.csv   (1280 features + Class)
"""
from pathlib import Path

import torch.nn as nn
import torch
from torchvision import models

from derm.legacy.pipeline import cnn
from derm import paths


class EfficientNetV2SCBAM(nn.Module):
    """EfficientNetV2-S backbone + CBAM block before global avg pool + FC head."""

    def __init__(self, n_classes=3):
        super().__init__()
        base = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
        self.features = base.features          # all conv stages, ends at 1280 ch
        self.cbam = cnn.CBAM(1280)
        self.avgpool = base.avgpool
        self.fc = nn.Linear(1280, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# Override cnn module globals before running
cnn.LESION = paths.LESION
cnn.OUT_MODEL = paths.CNN_V2S_BEST
cnn.OUT_FEATS = paths.CNN_V2S_FEATS_CSV
cnn.make_model = lambda: EfficientNetV2SCBAM(n_classes=3)
cnn.BATCH = 32      # smaller batch -- V2-S is heavier than ResNet18 on 8GB VRAM


if __name__ == "__main__":
    print(f"device: {cnn.device}")
    if not cnn.OUT_MODEL.exists():
        cnn.train()
    cnn.extract()
