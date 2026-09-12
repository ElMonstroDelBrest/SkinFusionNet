"""
Export and synchronize the deployed EfficientNetV2-S model bundle.

Produces in models/ and syncs the app runtime assets in app/assets/models/:
  cnn_lesion_feats.onnx   <- cnn_v2s_best.pth        (1280-d, fc=Identity)
  cnn_dehair_feats.onnx   <- cnn_v2s_dehair_best.pth
  lgbm_a.onnx, lgbm_c.onnx          (non-TTA, on cnn_v2s_*features.csv)
  lgbm_a_tta.onnx, lgbm_c_tta.onnx  (TTA, on cnn_v2s_*tta_features.csv)
  model_manifest.json

Both LGBM variants reuse the SAME 2 CNN ONNX. The canonical app bundle is
single-pass; TTA remains an optional archived/experimental export.

Run from the repository root:
  python3 -m derm.export.export_all_v2s
"""
import argparse
import shutil
from pathlib import Path


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from derm.pipeline import cnn_v2s  # EfficientNetV2SCBAM
from derm import paths
from derm.export.model_manifest import (
    refresh_manifest,
    write_compatibility_pointer,
)

OUT = paths.MODELS
OUT.mkdir(exist_ok=True)
APP_OUT = paths.APP_MODELS
IMG = 224


# ---------------------------------------------------------------- CNN -> ONNX
def export_cnn(weights, out):
    m = cnn_v2s.EfficientNetV2SCBAM(n_classes=3)
    m.load_state_dict(torch.load(weights, map_location="cpu"))
    m.fc = nn.Identity()
    m.eval()
    dummy = torch.zeros(1, 3, IMG, IMG)
    torch.onnx.export(
        m,
        dummy,
        str(out),
        opset_version=17,
        dynamo=False,
        input_names=["input"],
        output_names=["features"],
        dynamic_axes={"input": {0: "batch"}, "features": {0: "batch"}},
    )
    import onnxruntime as ort
    with torch.no_grad():
        ref = m(dummy).numpy()
    got = ort.InferenceSession(
        str(out), providers=["CPUExecutionProvider"]
    ).run(None, {"input": dummy.numpy()})[0]
    print(f"  {out.name}: shape={got.shape} diff={float(np.abs(ref-got).max()):.2e}")


# ------------------------------------------------------------- LGBM -> ONNX
def make_lgbm():
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                          random_state=42, n_jobs=-1, verbose=-1)


def to_onnx(pipe, n, out):
    from skl2onnx import convert_sklearn, update_registered_converter
    from skl2onnx.common.data_types import FloatTensorType
    from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
    from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
    update_registered_converter(LGBMClassifier, "LightGbmLGBMClassifier",
                                calculate_linear_classifier_output_shapes, convert_lightgbm,
                                options={"nocl": [True, False], "zipmap": [True, False]})
    onx = convert_sklearn(pipe, "lgbm",
                          initial_types=[("input", FloatTensorType([None, n]))],
                          target_opset={"": 17, "ai.onnx.ml": 3}, options={"zipmap": False})
    Path(out).write_bytes(onx.SerializeToString())


def build_lgbm(hc_df, cnn_a_csv, cnn_c_csv, suffix):
    cnn_a_raw = pd.read_csv(cnn_a_csv)
    cnn_c_raw = pd.read_csv(cnn_c_csv)
    cnn_cols = [c for c in cnn_a_raw.columns if c.startswith("cnn_")]
    if cnn_cols != [c for c in cnn_c_raw.columns if c.startswith("cnn_")]:
        raise ValueError("CNN feature columns differ between lesion and dehair")
    cnn_a = cnn_a_raw[cnn_cols]
    cnn_c = cnn_c_raw[cnn_cols]
    assert len(hc_df) == len(cnn_a) == len(cnn_c), "row mismatch"
    X_hc = hc_df.iloc[:, :-1].values.astype(np.float64)
    y = hc_df["Class"].values.astype(int)
    X_A = np.concatenate([X_hc, cnn_a.values], axis=1)
    X_C = np.concatenate([X_hc, cnn_c.values], axis=1)
    n = X_A.shape[1]

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
    pa = Pipeline([("s", StandardScaler()), ("c", make_lgbm())]).fit(X_A[tr], y[tr])
    pc = Pipeline([("s", StandardScaler()), ("c", make_lgbm())]).fit(X_C[tr], y[tr])
    PE = (pa.predict_proba(X_A[te]) + pc.predict_proba(X_C[te])) / 2
    acc = accuracy_score(y[te], PE.argmax(1))
    auc = roc_auc_score(y[te], PE, multi_class="ovr")
    print(f"[{suffix or 'no-tta'}] n={n}  holdout ensemble acc={acc:.4f} auc={auc:.4f}")

    for tag, X in (("a", X_A), ("c", X_C)):
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", make_lgbm())]).fit(X, y)
        to_onnx(pipe, n, OUT / f"lgbm_{tag}{suffix}.onnx")
    return int(n), float(acc), float(auc)


def sync_app_assets():
    """Copy the deployed ONNX bundle into the Flutter asset directory."""
    APP_OUT.mkdir(parents=True, exist_ok=True)
    for name in (
        "unet.onnx",
        "cnn_lesion_feats.onnx",
        "cnn_dehair_feats.onnx",
        "lgbm_a.onnx",
        "lgbm_c.onnx",
    ):
        src = OUT / name
        if not src.exists():
            raise FileNotFoundError(f"missing deployed asset: {src}")
        shutil.copy2(src, APP_OUT / name)
    print(f"synced app assets -> {APP_OUT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-bundle-change",
        action="store_true",
        help="Allow changed ONNX bytes only with a new bundle ID and release date.",
    )
    parser.add_argument(
        "--new-bundle-id",
        help="New immutable bundle ID; required when exported ONNX bytes change.",
    )
    parser.add_argument(
        "--release-date",
        help="New release date (YYYY-MM-DD); required when ONNX bytes change.",
    )
    args = parser.parse_args()
    hc = pd.read_csv(paths.FEATURES_CSV)
    hc_cols = list(hc.columns[:-1])
    print(f"features.csv handcraft = {len(hc_cols)} : {hc_cols}")

    print("== CNN V2-S -> ONNX ==")
    export_cnn(str(paths.CNN_V2S_BEST), OUT / "cnn_lesion_feats.onnx")
    export_cnn(str(paths.CNN_V2S_DEHAIR_BEST), OUT / "cnn_dehair_feats.onnx")

    print("== LGBM (non-TTA) ==")
    n0, a0, u0 = build_lgbm(hc, paths.CNN_V2S_FEATS_CSV, paths.CNN_V2S_DEHAIR_CSV, "")
    variants = {
        "single_pass": {"n_features": n0, "holdout_acc": a0, "holdout_auc": u0},
    }
    if paths.CNN_V2S_TTA_CSV.exists() and paths.CNN_V2S_DEHAIR_TTA_CSV.exists():
        print("== LGBM (optional TTA x8) ==")
        n1, a1, u1 = build_lgbm(hc, paths.CNN_V2S_TTA_CSV, paths.CNN_V2S_DEHAIR_TTA_CSV, "_tta")
        variants["tta8_optional"] = {"n_features": n1, "holdout_acc": a1, "holdout_auc": u1}
    else:
        variants["tta8_optional"] = {
            "status": "blocked_missing_data",
            "missing": [
                paths.CNN_V2S_TTA_CSV.name,
                paths.CNN_V2S_DEHAIR_TTA_CSV.name,
            ],
        }

    manifest = refresh_manifest(
        paths.APP_MODELS / "model_manifest.json",
        OUT,
        [OUT / "model_manifest.json"],
        export_validation={"handcraft_cols": hc_cols, "variants": variants},
        allow_bundle_change=args.allow_bundle_change,
        new_bundle_id=args.new_bundle_id,
        release_date=args.release_date,
    )
    sync_app_assets()
    shutil.copy2(OUT / "model_manifest.json", paths.APP_MODELS / "model_manifest.json")
    write_compatibility_pointer(
        Path(__file__).with_name("serious_manifest.json"), manifest
    )
    print(
        "done -> canonical single-pass ONNX bundle + model_manifest.json "
        f"({manifest['bundle_id']})"
    )


if __name__ == "__main__":
    main()
