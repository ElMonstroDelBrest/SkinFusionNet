"""EfficientNetV2-S + CBAM on dehaired images (DullRazor)."""
from derm.legacy.pipeline import cnn
from derm.pipeline import cnn_v2s  # registers the V2S architecture in cnn.make_model
from derm import paths

cnn.LESION = paths.DEHAIR
cnn.OUT_MODEL = paths.CNN_V2S_DEHAIR_BEST
cnn.OUT_FEATS = paths.CNN_V2S_DEHAIR_CSV


def collect(split):
    return sorted((cnn.LESION / split).rglob("*.jpg"))

cnn.collect = collect


if __name__ == "__main__":
    print(f"device: {cnn.device}")
    if not cnn.OUT_MODEL.exists():
        cnn.train()
    cnn.extract()
