"""
Train + save the two LightGBM models for the EfficientNetV2-S ensemble, export ONNX.

A = handcraft(18) + V2S(lesion 1280)   -> scaler_A + LGBM_A
C = handcraft(18) + V2S(dehair 1280)   -> scaler_C + LGBM_C
Ensemble (app) = (P_A + P_C) / 2

Overwrites models/lgbm_a.onnx and models/lgbm_c.onnx.
Input per model = 1298 raw features [18 handcraft | 1280 V2S], z-scored inside ONNX.

This focused tool writes an export report, not the canonical bundle manifest.
Use ``python3 -m derm.export.export_all_v2s`` for a versioned app release.

Run from the repository root: python3 -m derm.export.train_export_lgbm_v2s
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from derm import paths

OUT = paths.MODELS


def make():
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                          random_state=42, n_jobs=-1, verbose=-1)


def to_onnx(pipe, n, out_path):
    from skl2onnx import convert_sklearn, update_registered_converter
    from skl2onnx.common.data_types import FloatTensorType
    from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
    from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
    update_registered_converter(
        LGBMClassifier, "LightGbmLGBMClassifier",
        calculate_linear_classifier_output_shapes, convert_lightgbm,
        options={"nocl": [True, False], "zipmap": [True, False]})
    onx = convert_sklearn(
        pipe, "lgbm",
        initial_types=[("input", FloatTensorType([None, n]))],
        target_opset={"": 17, "ai.onnx.ml": 3}, options={"zipmap": False})
    Path(out_path).write_bytes(onx.SerializeToString())


def main():
    OUT.mkdir(exist_ok=True)
    hc = pd.read_csv(paths.FEATURES_CSV)
    cnn_a_raw = pd.read_csv(paths.CNN_V2S_FEATS_CSV)
    cnn_c_raw = pd.read_csv(paths.CNN_V2S_DEHAIR_CSV)
    cnn_cols = [c for c in cnn_a_raw.columns if c.startswith("cnn_")]
    if cnn_cols != [c for c in cnn_c_raw.columns if c.startswith("cnn_")]:
        raise ValueError("CNN feature columns differ between lesion and dehair")
    cnn_a = cnn_a_raw[cnn_cols]
    cnn_c = cnn_c_raw[cnn_cols]
    assert len(hc) == len(cnn_a) == len(cnn_c), "row mismatch"

    hc_cols = list(hc.columns[:-1])
    X_hc = hc.iloc[:, :-1].values.astype(np.float64)
    y = hc["Class"].values.astype(int)
    X_A = np.concatenate([X_hc, cnn_a.values], axis=1)
    X_C = np.concatenate([X_hc, cnn_c.values], axis=1)
    n = X_A.shape[1]
    print(f"rows={len(y)}  handcraft={len(hc_cols)}  cnn={cnn_a.shape[1]}  total={n}")

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
    pa = Pipeline([("s", StandardScaler()), ("c", make())]).fit(X_A[tr], y[tr])
    pc = Pipeline([("s", StandardScaler()), ("c", make())]).fit(X_C[tr], y[tr])
    PA, PC = pa.predict_proba(X_A[te]), pc.predict_proba(X_C[te])
    PE = (PA + PC) / 2.0
    yt = y[te]
    holdout = {}
    for name, P in (("A", PA), ("C", PC), ("Ensemble", PE)):
        acc = float(accuracy_score(yt, P.argmax(1)))
        auc = float(roc_auc_score(yt, P, multi_class="ovr"))
        holdout[name.lower()] = {"accuracy": acc, "auc_macro_ovr": auc}
        print(f"  holdout {name:<9} acc={acc:.4f}  auc={auc:.4f}")

    for tag, X in (("a", X_A), ("c", X_C)):
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", make())]).fit(X, y)
        joblib.dump(pipe, OUT / f"lgbm_{tag}.pkl")
        out_onnx = OUT / f"lgbm_{tag}.onnx"
        to_onnx(pipe, n, out_onnx)
        import onnxruntime as ort
        rng = np.random.RandomState(0)
        Xs = X[rng.choice(len(X), 200, replace=False)].astype(np.float32)
        ref = pipe.predict_proba(Xs)
        sess = ort.InferenceSession(str(out_onnx), providers=["CPUExecutionProvider"])
        probs = np.array(sess.run(None, {"input": Xs})[1])
        print(f"  LGBM_{tag.upper()} -> {out_onnx.name}  onnx vs sklearn diff = "
              f"{float(np.abs(ref - probs).max()):.2e}")

    report = {
        "report_type": "lgbm-export-validation-not-a-bundle-manifest",
        "backbone": "EfficientNetV2-S+CBAM",
        "handcraft_cols": hc_cols,
        "n_handcraft": len(hc_cols),
        "n_cnn": int(cnn_a.shape[1]),
        "n_features": int(n),
        "classes": ["nevus", "melanoma", "atypical"],
        "ensemble": "mean(P_A, P_C)",
        "holdout": holdout,
        "release_command": "python3 -m derm.export.export_all_v2s",
    }
    (OUT / "lgbm_export_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote lgbm_a.onnx, lgbm_c.onnx, *.pkl, lgbm_export_report.json")


if __name__ == "__main__":
    main()
