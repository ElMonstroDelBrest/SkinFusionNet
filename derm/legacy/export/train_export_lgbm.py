"""
Train + save the two LightGBM models of the serious ensemble, then export to ONNX.

A = handcraft(18) + CNN(lesion)   -> scaler_A + LGBM_A
C = handcraft(18) + CNN(dehair)   -> scaler_C + LGBM_C
Ensemble (deployed in the app)    = (P_A + P_C) / 2

train_lgbm_entropy.py only did CV (saved nothing); here we fit on ALL data for
deployment and also print an 80/20 holdout accuracy as a sanity estimate.

Each ONNX model takes the RAW 530-vector [18 handcraft | 512 CNN] and applies the
z-score internally (StandardScaler baked into the pipeline) -> [P0,P1,P2].

Run with the isolated venv:  C:\\pcv\\Scripts\\python.exe export\\train_export_lgbm.py
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

OUT = paths.LEGACY_MODELS


def make():
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                          random_state=42, n_jobs=-1, verbose=-1)


def to_onnx(pipe, n_features, out_path):
    from skl2onnx import convert_sklearn, update_registered_converter
    from skl2onnx.common.data_types import FloatTensorType
    from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
    from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm

    update_registered_converter(
        LGBMClassifier, "LightGbmLGBMClassifier",
        calculate_linear_classifier_output_shapes, convert_lightgbm,
        options={"nocl": [True, False], "zipmap": [True, False]},
    )
    onx = convert_sklearn(
        pipe, "lgbm",
        initial_types=[("input", FloatTensorType([None, n_features]))],
        target_opset={"": 17, "ai.onnx.ml": 3},
        options={"zipmap": False},
    )
    Path(out_path).write_bytes(onx.SerializeToString())


def main():
    hc = pd.read_csv(paths.FEATURES_CSV)
    cnn_a = pd.read_csv(paths.CNN_FEATS_CSV).drop(columns=["Class"])
    cnn_c = pd.read_csv(paths.CNN_DEHAIR_CSV).drop(columns=["Class"])
    assert len(hc) == len(cnn_a) == len(cnn_c), "row mismatch"

    hc_cols = list(hc.columns[:-1])
    X_hc = hc.iloc[:, :-1].values.astype(np.float64)
    y = hc["Class"].values.astype(int)
    X_A = np.concatenate([X_hc, cnn_a.values], axis=1)
    X_C = np.concatenate([X_hc, cnn_c.values], axis=1)
    n = X_A.shape[1]
    print(f"rows={len(y)}  handcraft={len(hc_cols)}  total per model={n}")

    # ---- holdout sanity (same split for A and C) ----
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
    pa = Pipeline([("s", StandardScaler()), ("c", make())]).fit(X_A[tr], y[tr])
    pc = Pipeline([("s", StandardScaler()), ("c", make())]).fit(X_C[tr], y[tr])
    PA, PC = pa.predict_proba(X_A[te]), pc.predict_proba(X_C[te])
    PE = (PA + PC) / 2.0
    yt = y[te]
    for name, P in (("A", PA), ("C", PC), ("Ensemble", PE)):
        acc = accuracy_score(yt, P.argmax(1))
        auc = roc_auc_score(yt, P, multi_class="ovr")
        print(f"  holdout {name:<9} acc={acc:.4f}  auc={auc:.4f}")

    # ---- final models on ALL data (deployed) ----
    results = {}
    for tag, X in (("a", X_A), ("c", X_C)):
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", make())]).fit(X, y)
        joblib.dump(pipe, OUT / f"lgbm_{tag}.pkl")
        out_onnx = OUT / f"lgbm_{tag}.onnx"
        to_onnx(pipe, n, out_onnx)

        # parity check onnx vs sklearn on a random sample
        import onnxruntime as ort
        rng = np.random.RandomState(0)
        Xs = X[rng.choice(len(X), 200, replace=False)].astype(np.float32)
        ref = pipe.predict_proba(Xs)
        sess = ort.InferenceSession(str(out_onnx), providers=["CPUExecutionProvider"])
        o = sess.run(None, {"input": Xs})
        probs = np.array(o[1])
        err = float(np.abs(ref - probs).max())
        print(f"  LGBM_{tag.upper()} -> {out_onnx.name}  onnx vs sklearn max diff = {err:.2e}")
        results[tag] = err

    manifest = {
        "handcraft_cols": hc_cols,
        "n_handcraft": len(hc_cols),
        "n_cnn": 512,
        "n_features": n,
        "classes": ["nevus", "melanoma", "atypical"],
        "ensemble": "mean(P_A, P_C)",
        "inputs": {
            "lgbm_a": "[18 handcraft | 512 cnn_lesion]",
            "lgbm_c": "[18 handcraft | 512 cnn_dehair]",
        },
    }
    (OUT / "serious_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote export/lgbm_a.onnx, export/lgbm_c.onnx, *.pkl, serious_manifest.json")


if __name__ == "__main__":
    main()
