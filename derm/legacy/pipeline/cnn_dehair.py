"""
Same as cnn.py but trained on the DullRazor-dehaired raw images.
For comparing : CNN(raw) vs CNN(dehair) impact on the downstream LightGBM.

Outputs :
  cnn_dehair_best.pth
  cnn_dehair_features.csv
"""
from derm.legacy.pipeline import cnn
from derm import paths

cnn.LESION = paths.DEHAIR
cnn.OUT_MODEL = paths.CNN_DEHAIR_BEST
cnn.OUT_FEATS = paths.CNN_DEHAIR_CSV


def collect(split):
    return sorted((cnn.LESION / split).rglob("*.jpg"))

cnn.collect = collect


if __name__ == "__main__":
    print(f"device: {cnn.device}")
    if not cnn.OUT_MODEL.exists():
        cnn.train()
    cnn.extract()
