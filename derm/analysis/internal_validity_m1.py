"""M1 internal-validity audit for the canonical V2S single-pass model.

CPU scope: Phase 0, quick-win 1a, C0, C1 and C2. C3 requires GPU CNN
retraining/extraction and is reported as blocked_needs_gpu.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from scipy.stats import norm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from derm import paths


CLASS_MAP = {"nevus": 0, "melanoma": 1, "atypical": 2}
CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}
SPLITS = ("train", "valid", "test")
V2S_SINGLE = (paths.CNN_V2S_FEATS_CSV, paths.CNN_V2S_DEHAIR_CSV)
V2S_TTA = (paths.CNN_V2S_TTA_CSV, paths.CNN_V2S_DEHAIR_TTA_CSV)
OUT_METADATA = paths.INTERNAL_METADATA
OUT_DECOMP = paths.INTERNAL_DECOMPOSITION
OUT_PREDICTIONS = paths.INTERNAL_PREDICTIONS
OUT_EXPERIMENTS = paths.INTERNAL_EXPERIMENT_REGISTRY
OUT_REPORT = paths.INTERNAL_VALIDITY_REPORT
OUT_LOG = paths.INTERNAL_VALIDITY_LOG
N_JOBS = int(os.environ.get("DERM_LGBM_JOBS", "4"))
CV_JOBS = int(os.environ.get("DERM_CV_JOBS", "1"))
LGBM_DEVICE = os.environ.get("DERM_LGBM_DEVICE", "cpu").lower()


def lgbm_device_kwargs() -> dict[str, int | str]:
    if LGBM_DEVICE == "gpu":
        return {
            "device_type": "gpu",
            "gpu_platform_id": int(os.environ.get("DERM_LGBM_GPU_PLATFORM_ID", "0")),
            "gpu_device_id": int(os.environ.get("DERM_LGBM_GPU_DEVICE_ID", "0")),
        }
    return {}


@dataclass(frozen=True)
class ImageRow:
    row_index: int
    file: str
    image_id: str
    source: str
    lesion_id: str
    patient_id: str
    split_original: str
    class_name: str
    class_id: int
    group_key: str


def log(msg: str) -> None:
    print(msg, flush=True)


def infer_source(stem: str) -> str:
    if stem.startswith("ham_"):
        return "HAM10000"
    if stem.startswith("isic2019_"):
        return "ISIC2019"
    if stem.startswith("s1_"):
        return "ISIC2017"
    if stem.startswith("s2_"):
        return "Kaggle9"
    return "unknown"


def canonical_image_id(stem: str) -> str:
    for prefix in ("ham_", "isic2019_", "s1_", "s2_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def metadata_candidates() -> list[Path]:
    patterns = (
        "*HAM10000*metadata*.csv",
        "*metadata*.csv",
        "*groundtruth*.csv",
        "*ground_truth*.csv",
        "*ISIC*.csv",
    )
    found: set[Path] = set()
    for root in (paths.DATA, paths.ROOT):
        if not root.exists():
            continue
        for pat in patterns:
            found.update(root.rglob(pat))
    return sorted(
        p for p in found
        if p.is_file()
        and p.resolve() != OUT_METADATA.resolve()
        and paths.RESULTS not in p.parents
    )


def load_metadata_maps() -> tuple[dict[str, str], dict[str, str], list[Path]]:
    lesion_by_id: dict[str, str] = {}
    patient_by_id: dict[str, str] = {}
    used: list[Path] = []

    for meta in metadata_candidates():
        try:
            df = pd.read_csv(meta, dtype=str, low_memory=False)
        except Exception:
            continue
        cols = {c.lower(): c for c in df.columns}
        id_col = next(
            (cols[c] for c in ("image_id", "image", "isic_id", "name") if c in cols),
            None,
        )
        lesion_col = next(
            (cols[c] for c in ("lesion_id", "lesion", "lesionid") if c in cols),
            None,
        )
        patient_col = next(
            (cols[c] for c in ("patient_id", "patient", "patientid") if c in cols),
            None,
        )
        if id_col is None or (lesion_col is None and patient_col is None):
            continue
        added = 0
        for _, r in df.iterrows():
            image_id = str(r.get(id_col, "")).strip()
            if not image_id:
                continue
            image_id = Path(image_id).stem
            if lesion_col is not None:
                lesion = str(r.get(lesion_col, "")).strip()
                if lesion and lesion.lower() != "nan":
                    lesion_by_id[image_id] = lesion
                    added += 1
            if patient_col is not None:
                patient = str(r.get(patient_col, "")).strip()
                if patient and patient.lower() != "nan":
                    patient_by_id[image_id] = patient
                    added += 1
        if added:
            used.append(meta)
    return lesion_by_id, patient_by_id, used


def collect_feature_order() -> list[Path]:
    files: list[Path] = []
    for split in SPLITS:
        files.extend(sorted((paths.UNET_MASKS / split).rglob("*.png")))
    return files


def build_image_metadata() -> tuple[pd.DataFrame, list[Path]]:
    lesion_by_id, patient_by_id, used_metadata = load_metadata_maps()
    rows: list[ImageRow] = []
    for i, p in enumerate(collect_feature_order()):
        split = p.relative_to(paths.UNET_MASKS).parts[0]
        cls = p.parent.name
        stem = p.stem
        image_id = canonical_image_id(stem)
        lesion_id = lesion_by_id.get(stem) or lesion_by_id.get(image_id) or ""
        patient_id = patient_by_id.get(stem) or patient_by_id.get(image_id) or ""
        group_key = lesion_id or patient_id or p.name
        rows.append(
            ImageRow(
                row_index=i,
                file=p.name,
                image_id=image_id,
                source=infer_source(stem),
                lesion_id=lesion_id,
                patient_id=patient_id,
                split_original=split,
                class_name=cls,
                class_id=CLASS_MAP[cls],
                group_key=group_key,
            )
        )
    df = pd.DataFrame([r.__dict__ for r in rows])
    paths.ARTIFACTS.mkdir(exist_ok=True)
    df.to_csv(OUT_METADATA, index=False)
    return df, used_metadata


def verify_mapping(meta: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    hc = pd.read_csv(paths.FEATURES_CSV, usecols=["Class"])
    if len(hc) != len(meta):
        issues.append(f"features.csv rows={len(hc)} vs metadata rows={len(meta)}")
    else:
        bad = int((hc["Class"].to_numpy() != meta["class_id"].to_numpy()).sum())
        if bad:
            issues.append(f"features.csv class mismatch rows={bad}")

    lesion_count = sum(1 for split in SPLITS for _ in (paths.LESION / split).rglob("*.png"))
    dehair_count = sum(1 for split in SPLITS for _ in (paths.DEHAIR / split).rglob("*.jpg"))
    if lesion_count != len(meta):
        issues.append(f"lesion PNG count={lesion_count} vs metadata rows={len(meta)}")
    if dehair_count != len(meta):
        issues.append(f"dehair JPG count={dehair_count} vs metadata rows={len(meta)}")
    return issues


def coverage_table(meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, g in meta.groupby("source", sort=True):
        n = len(g)
        rows.append(
            {
                "source": source,
                "n_images": n,
                "lesion_id_resolved_pct": 100 * (g["lesion_id"] != "").mean(),
                "patient_id_resolved_pct": 100 * (g["patient_id"] != "").mean(),
            }
        )
    return pd.DataFrame(rows)


def duplication_table(meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, g in meta.groupby("source", sort=True):
        counts = g.groupby("group_key").size()
        n = len(g)
        multi_images = int(g["group_key"].map(counts).gt(1).sum())
        rows.append(
            {
                "source": source,
                "n_images": n,
                "n_groups": int(counts.size),
                "max_images_per_group": int(counts.max()),
                "mean_images_per_group": float(counts.mean()),
                "pct_images_in_multi_image_group": 100 * multi_images / n if n else math.nan,
            }
        )
    counts = meta.groupby("group_key").size()
    rows.append(
        {
            "source": "ALL",
            "n_images": len(meta),
            "n_groups": int(counts.size),
            "max_images_per_group": int(counts.max()),
            "mean_images_per_group": float(counts.mean()),
            "pct_images_in_multi_image_group": 100
            * int(meta["group_key"].map(counts).gt(1).sum())
            / len(meta),
        }
    )
    return pd.DataFrame(rows)


def split_audit(meta: pd.DataFrame) -> dict[str, object]:
    train_valid = set(meta.loc[meta["split_original"].isin(["train", "valid"]), "group_key"])
    test = set(meta.loc[meta["split_original"] == "test", "group_key"])
    overlap = sorted(train_valid & test)

    # Supplemental exact-image-ID audit; this can catch same ISIC image reused under
    # different source prefixes, but it is not a lesion-disjoint proof.
    tv_img = set(meta.loc[meta["split_original"].isin(["train", "valid"]), "image_id"])
    test_img = set(meta.loc[meta["split_original"] == "test", "image_id"])
    image_overlap = sorted(tv_img & test_img)
    return {
        "group_overlap_count": len(overlap),
        "group_overlap_examples": overlap[:20],
        "image_id_overlap_count": len(image_overlap),
        "image_id_overlap_examples": image_overlap[:20],
    }


def make_model() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        n_jobs=N_JOBS,
        verbose=-1,
        **lgbm_device_kwargs(),
    )


def dice_per_class(y_true: np.ndarray, y_pred: np.ndarray) -> list[float]:
    out = []
    for c in (0, 1, 2):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        out.append((2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return math.nan, math.nan, math.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def delong_auc_ci(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float, float]:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return math.nan, math.nan, math.nan
    tx = compute_midrank(pos)
    ty = compute_midrank(neg)
    tz = compute_midrank(np.r_[pos, neg])
    auc = (tz[:m].sum() / m - (m + 1) / 2) / n
    v01 = (tz[:m] - tx) / n
    v10 = 1 - (tz[m:] - ty) / m
    sx = np.cov(v01, ddof=1) if m > 1 else 0.0
    sy = np.cov(v10, ddof=1) if n > 1 else 0.0
    var = float(sx / m + sy / n)
    se = math.sqrt(max(var, 0.0))
    lo = max(0.0, auc - norm.ppf(0.975) * se)
    hi = min(1.0, auc + norm.ppf(0.975) * se)
    return float(auc), lo, hi


def youden_threshold(y_binary: np.ndarray, scores: np.ndarray) -> float:
    y_binary = np.asarray(y_binary, dtype=int)
    scores = np.asarray(scores, dtype=float)
    best_t = float(scores.min())
    best_j = -np.inf
    for t in np.unique(scores):
        pred = scores >= t
        tp = int((pred & (y_binary == 1)).sum())
        fn = int((~pred & (y_binary == 1)).sum())
        tn = int((~pred & (y_binary == 0)).sum())
        fp = int((pred & (y_binary == 0)).sum())
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        j = sens + spec - 1
        if j > best_j:
            best_j = j
            best_t = float(t)
    return best_t


def summarize_predictions(
    condition: str,
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold_mel: float | np.ndarray,
    status: str = "ok",
    reason: str = "",
    fold_metric: dict[str, list[float]] | None = None,
) -> dict[str, object]:
    y_pred = proba.argmax(axis=1)
    y_mel = (y_true == 1).astype(int)
    p_mel = proba[:, 1]
    auc_mel, auc_lo, auc_hi = delong_auc_ci(y_mel, p_mel)
    threshold_arr = np.asarray(threshold_mel, dtype=float)
    pred_mel = p_mel >= threshold_arr
    tp = int((pred_mel & (y_true == 1)).sum())
    fn = int((~pred_mel & (y_true == 1)).sum())
    tn = int((~pred_mel & (y_true != 1)).sum())
    fp = int((pred_mel & (y_true != 1)).sum())
    sens, sens_lo, sens_hi = wilson(tp, tp + fn)
    spec, spec_lo, spec_hi = wilson(tn, tn + fp)
    dpc = dice_per_class(y_true, y_pred)
    out: dict[str, object] = {
        "condition": condition,
        "status": status,
        "reason": reason,
        "n_eval": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "auc_macro_ovr": roc_auc_score(y_true, proba, multi_class="ovr"),
        "auc_melanoma_ovr": auc_mel,
        "auc_melanoma_ci95_low": auc_lo,
        "auc_melanoma_ci95_high": auc_hi,
        "threshold_melanoma": float(np.mean(threshold_arr)),
        "threshold_melanoma_std": float(np.std(threshold_arr)),
        "sensitivity_melanoma": sens,
        "sensitivity_melanoma_ci95_low": sens_lo,
        "sensitivity_melanoma_ci95_high": sens_hi,
        "specificity_melanoma": spec,
        "specificity_melanoma_ci95_low": spec_lo,
        "specificity_melanoma_ci95_high": spec_hi,
        "macro_dice": float(np.mean(dpc)),
        "dice_nevus": dpc[0],
        "dice_melanoma": dpc[1],
        "dice_atypical": dpc[2],
        "tp_mel": tp,
        "fn_mel": fn,
        "tn_mel": tn,
        "fp_mel": fp,
    }
    if fold_metric:
        for key, vals in fold_metric.items():
            out[f"{key}_fold_mean"] = float(np.mean(vals))
            out[f"{key}_fold_std"] = float(np.std(vals))
    return out


def required_csvs(deep_source: str) -> tuple[Path, Path]:
    if deep_source == "deployed_single":
        return V2S_SINGLE
    if deep_source == "deployed_tta":
        return V2S_TTA
    raise ValueError(deep_source)


def load_v2s_matrices(
    deep_source: str = "deployed_single",
    meta: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    required = required_csvs(deep_source)
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing required V2S CSV(s) for {deep_source}: " + ", ".join(missing))
    hc = pd.read_csv(paths.FEATURES_CSV)
    cnn_a = pd.read_csv(required[0])
    cnn_c = pd.read_csv(required[1])
    if not (len(hc) == len(cnn_a) == len(cnn_c)):
        raise ValueError(f"row mismatch hc={len(hc)} a={len(cnn_a)} c={len(cnn_c)}")
    y = hc["Class"].to_numpy(int)
    if not np.array_equal(y, cnn_a["Class"].to_numpy(int)):
        raise ValueError("Class mismatch features.csv vs CNN lesion CSV")
    if not np.array_equal(y, cnn_c["Class"].to_numpy(int)):
        raise ValueError("Class mismatch features.csv vs CNN dehair CSV")
    cols_a = [c for c in cnn_a.columns if c.startswith("cnn_")]
    cols_c = [c for c in cnn_c.columns if c.startswith("cnn_")]
    if cols_a != cols_c:
        raise ValueError("CNN feature columns differ between lesion and dehair CSVs")
    audit_columns = ["row_index", "file", "split_original", "class_name"]
    if all(column in cnn_a.columns for column in audit_columns):
        if not cnn_a[audit_columns].equals(cnn_c[audit_columns]):
            raise ValueError("CNN lesion/dehair observation metadata differ")
        expected_rows = np.arange(len(cnn_a), dtype=int)
        if not np.array_equal(cnn_a["row_index"].to_numpy(int), expected_rows):
            raise ValueError("CNN row_index is not the canonical contiguous row order")
        if meta is not None:
            for column in audit_columns:
                left = cnn_a[column].astype(str).to_numpy()
                right = meta[column].astype(str).to_numpy()
                if not np.array_equal(left, right):
                    raise ValueError(f"CNN feature metadata mismatch for {column}")
    elif meta is not None:
        raise ValueError(
            "auditable internal export requires row_index/file/split_original/class_name "
            "in both CNN feature CSVs"
        )
    x_hc = hc.drop(columns=["Class"]).to_numpy(float)
    x_a = np.c_[x_hc, cnn_a[cols_a].to_numpy(float)]
    x_c = np.c_[x_hc, cnn_c[cols_c].to_numpy(float)]
    return x_a, x_c, y, hc


def prediction_frame(
    condition: str,
    meta: pd.DataFrame,
    indices: np.ndarray,
    y_true: np.ndarray,
    proba: np.ndarray,
    folds: np.ndarray,
    thresholds: float | np.ndarray,
) -> pd.DataFrame:
    """Create the observation-level handoff table for one evaluation condition."""
    indices = np.asarray(indices, dtype=int)
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    folds = np.asarray(folds, dtype=int)
    threshold_array = np.asarray(thresholds, dtype=float)
    if threshold_array.ndim == 0:
        threshold_array = np.full(len(indices), float(threshold_array))
    if not (
        len(indices)
        == len(y_true)
        == len(proba)
        == len(folds)
        == len(threshold_array)
    ):
        raise ValueError(f"prediction export length mismatch for {condition}")
    if proba.shape != (len(indices), 3):
        raise ValueError(f"unexpected probability shape for {condition}: {proba.shape}")
    if not np.isfinite(proba).all() or not np.isfinite(threshold_array).all():
        raise ValueError(f"non-finite prediction values for {condition}")
    probability_sum = proba.sum(axis=1)
    if not np.allclose(probability_sum, 1.0, atol=1e-6):
        raise ValueError(
            f"multiclass probabilities do not sum to one for {condition}: "
            f"min={probability_sum.min():.9f} max={probability_sum.max():.9f}"
        )
    selected = meta.iloc[indices].reset_index(drop=True)
    if not np.array_equal(selected["class_id"].to_numpy(int), y_true):
        raise ValueError(f"metadata/target mismatch in prediction export for {condition}")
    argmax = proba.argmax(axis=1).astype(int)
    return pd.DataFrame(
        {
            "condition": condition,
            "row_index": selected["row_index"].to_numpy(int),
            "file": selected["file"].astype(str),
            "image_id": selected["image_id"].astype(str),
            "source": selected["source"].astype(str),
            "split_original": selected["split_original"].astype(str),
            "class_name": selected["class_name"].astype(str),
            "class_id": y_true,
            "group_key": selected["group_key"].astype(str),
            "fold": folds,
            "p_nevus": proba[:, 0],
            "p_melanoma": proba[:, 1],
            "p_atypical": proba[:, 2],
            "probability_sum": probability_sum,
            "argmax_class_id": argmax,
            "argmax_class_name": [CLASS_NAMES[value] for value in argmax],
            "threshold_melanoma": threshold_array,
            "predicted_melanoma_at_threshold": (
                proba[:, 1] >= threshold_array
            ).astype(int),
        }
    )


def experiment_row(
    condition: str,
    *,
    status: str,
    n_predictions: int,
    fitting_data: str,
    evaluation_data: str,
    splitter: str,
    threshold_provenance: str,
    representation_provenance: str,
    caveat: str,
) -> dict[str, object]:
    return {
        "condition": condition,
        "status": status,
        "n_predictions": n_predictions,
        "evaluation_unit": "image/file row; patient and lesion IDs unresolved",
        "fitting_data": fitting_data,
        "evaluation_data": evaluation_data,
        "splitter": splitter,
        "threshold_provenance": threshold_provenance,
        "representation_provenance": representation_provenance,
        "score_semantics": (
            "three normalized multiclass probabilities in order "
            "[nevus, melanoma, atypical]; class scores may be evaluated one-vs-rest"
        ),
        "caveat": caveat,
    }


def run_phase_1a_outputs(
    meta: pd.DataFrame, deep_source: str
) -> tuple[dict[str, object], pd.DataFrame, dict[str, object]]:
    x_a, x_c, y, _ = load_v2s_matrices(deep_source, meta)
    train_mask = meta["split_original"].isin(["train", "valid"]).to_numpy()
    valid_mask = (meta["split_original"] == "valid").to_numpy()
    test_mask = (meta["split_original"] == "test").to_numpy()

    def fit_branch(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", make_model())])
        pipe.fit(matrix[train_mask], y[train_mask])
        return pipe.predict_proba(matrix[valid_mask]), pipe.predict_proba(
            matrix[test_mask]
        )

    branch_outputs = Parallel(n_jobs=min(max(CV_JOBS, 1), 2))(
        delayed(fit_branch)(matrix) for matrix in (x_a, x_c)
    )
    p_valid = (branch_outputs[0][0] + branch_outputs[1][0]) / 2
    p_test = (branch_outputs[0][1] + branch_outputs[1][1]) / 2
    thr_mel = youden_threshold((y[valid_mask] == 1).astype(int), p_valid[:, 1])
    summary = summarize_predictions("1a", y[test_mask], p_test, thr_mel)
    test_indices = np.flatnonzero(test_mask)
    predictions = prediction_frame(
        "1a",
        meta,
        test_indices,
        y[test_mask],
        p_test,
        np.zeros(len(test_indices), dtype=int),
        thr_mel,
    )
    registry = experiment_row(
        "1a",
        status="current-rerun-heldout-configuration",
        n_predictions=len(predictions),
        fitting_data="original train+valid image rows",
        evaluation_data="original test image rows",
        splitter="fixed original split; fold=0 means not cross-validated",
        threshold_provenance=(
            "Youden threshold evaluated on valid predictions after the heads were "
            "fit on train+valid; threshold operating-point estimates are not independent"
        ),
        representation_provenance=(
            "current V2S matrices; evaluated configuration, not byte-identical final ONNX heads"
        ),
        caveat=(
            "AUC/argmax metrics use held-out test rows, but patient/lesion independence "
            "and exact historical input provenance are unresolved"
        ),
    )
    return summary, predictions, registry


def run_phase_1a(meta: pd.DataFrame, deep_source: str) -> dict[str, object]:
    """Compatibility wrapper used by the independent paper-resource verifier."""
    summary, _, _ = run_phase_1a_outputs(meta, deep_source)
    return summary


def c0_fold_output(
    fold: int,
    tr: np.ndarray,
    te: np.ndarray,
    x_a: np.ndarray,
    x_c: np.ndarray,
    y: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray, float, float, float]:
    p = fit_predict_ensemble(x_a, x_c, y, tr, te)
    yt = y[te]
    yhat = p.argmax(axis=1)
    return (
        fold,
        te,
        p,
        accuracy_score(yt, yhat),
        roc_auc_score(yt, p, multi_class="ovr"),
        float(np.mean(dice_per_class(yt, yhat))),
    )


def run_c0_outputs(
    meta: pd.DataFrame, deep_source: str
) -> tuple[dict[str, object], pd.DataFrame, dict[str, object]]:
    x_a, x_c, y, _ = load_v2s_matrices(deep_source, meta)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    proba = np.full((len(y), 3), np.nan, dtype=float)
    folds = np.zeros(len(y), dtype=int)
    fold_metric = {"accuracy": [], "auc_macro_ovr": [], "macro_dice": []}
    splits = [
        (fold, tr, te) for fold, (tr, te) in enumerate(cv.split(x_a, y), 1)
    ]
    outputs = Parallel(n_jobs=max(CV_JOBS, 1))(
        delayed(c0_fold_output)(fold, tr, te, x_a, x_c, y)
        for fold, tr, te in splits
    )
    for fold, te, p, accuracy, auc_macro, macro_dice in outputs:
        fold_metric["accuracy"].append(accuracy)
        fold_metric["auc_macro_ovr"].append(auc_macro)
        fold_metric["macro_dice"].append(macro_dice)
        proba[te] = p
        folds[te] = fold
        log(f"C0 fold {fold}/10 acc={fold_metric['accuracy'][-1]:.4f}")
    if not np.isfinite(proba).all() or np.any(folds == 0):
        raise RuntimeError("C0 did not produce exactly one prediction per row")
    thr_mel = youden_threshold((y == 1).astype(int), proba[:, 1])
    row = summarize_predictions("C0", y, proba, thr_mel, fold_metric=fold_metric)
    if deep_source == "deployed_tta":
        target_acc, target_auc, target_dice = 0.9528, 0.9919, 0.9437
    else:
        # Canonical deployed model: V2S single-pass 1280-d, app ensembling
        # mean(P_A, P_C). The older 1344-d log is archived as irreproducible.
        target_acc, target_auc, target_dice = 0.9424, 0.9886, 0.9312
    if (
        abs(row["accuracy_fold_mean"] - target_acc) > 0.01
        or abs(row["auc_macro_ovr_fold_mean"] - target_auc) > 0.005
        or abs(row["macro_dice_fold_mean"] - target_dice) > 0.01
    ):
        row["status"] = "failed_sanity_gate"
        row["reason"] = (
            "C0 does not reproduce expected deployed-source baseline within tolerance: "
            f"acc={row['accuracy_fold_mean']:.4f}, "
            f"auc={row['auc_macro_ovr_fold_mean']:.4f}, "
            f"dice={row['macro_dice_fold_mean']:.4f}"
        )
    predictions = prediction_frame(
        "C0", meta, np.arange(len(y)), y, proba, folds, thr_mel
    )
    registry = experiment_row(
        "C0",
        status=str(row.get("status", "ok")),
        n_predictions=len(predictions),
        fitting_data="nine image-level folds for each outer evaluation fold",
        evaluation_data="one image-level outer fold at a time; all 25,903 rows receive OOF scores",
        splitter="StratifiedKFold(n_splits=10, shuffle=True, random_state=42)",
        threshold_provenance=(
            "single pooled Youden threshold selected on all OOF predictions; descriptive, "
            "not nested for threshold operating-point estimates"
        ),
        representation_provenance=(
            "partially in-sample CNN representations; 82.77% of rows belonged to CNN "
            "training or checkpoint-selection splits"
        ),
        caveat=(
            "LightGBM/scaler are fold-local here, but representation leakage and image-level "
            "rather than patient/lesion-level splitting remain"
        ),
    )
    return row, predictions, registry


def run_c0(deep_source: str, meta: pd.DataFrame | None = None) -> dict[str, object]:
    if meta is None:
        meta, _ = build_image_metadata()
    row, _, _ = run_c0_outputs(meta, deep_source)
    return row


def fit_predict_ensemble(
    x_a: np.ndarray,
    x_c: np.ndarray,
    y: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
) -> np.ndarray:
    pipe_a = Pipeline([("scaler", StandardScaler()), ("clf", make_model())])
    pipe_c = Pipeline([("scaler", StandardScaler()), ("clf", make_model())])
    pipe_a.fit(x_a[tr], y[tr])
    pipe_c.fit(x_c[tr], y[tr])
    return (pipe_a.predict_proba(x_a[te]) + pipe_c.predict_proba(x_c[te])) / 2


def nested_threshold(
    x_a: np.ndarray,
    x_c: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    tr: np.ndarray,
) -> float:
    y_tr = y[tr]
    g_tr = groups[tr]
    n_splits = min(5, int(np.bincount(y_tr, minlength=3).min()))
    if n_splits < 2:
        p_train = fit_predict_ensemble(x_a, x_c, y, tr, tr)
        return youden_threshold((y[tr] == 1).astype(int), p_train[:, 1])
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    inner_tr_rel, cal_rel = next(splitter.split(np.zeros(len(tr)), y_tr, g_tr))
    inner_tr = tr[inner_tr_rel]
    cal = tr[cal_rel]
    p_cal = fit_predict_ensemble(x_a, x_c, y, inner_tr, cal)
    return youden_threshold((y[cal] == 1).astype(int), p_cal[:, 1])


def grouped_fold_output(
    condition: str,
    fold: int,
    tr: np.ndarray,
    te: np.ndarray,
    x_a: np.ndarray,
    x_c: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    nested: bool,
) -> tuple[int, np.ndarray, np.ndarray, float, float, float, float | None]:
    overlap = set(groups[tr]) & set(groups[te])
    if overlap:
        raise RuntimeError(
            f"group leakage in {condition} fold {fold}: {list(overlap)[:5]}"
        )
    p = fit_predict_ensemble(x_a, x_c, y, tr, te)
    yt = y[te]
    yhat = p.argmax(axis=1)
    threshold = nested_threshold(x_a, x_c, y, groups, tr) if nested else None
    return (
        fold,
        te,
        p,
        accuracy_score(yt, yhat),
        roc_auc_score(yt, p, multi_class="ovr"),
        float(np.mean(dice_per_class(yt, yhat))),
        threshold,
    )


def run_grouped_cv_outputs(
    condition: str,
    meta: pd.DataFrame,
    deep_source: str,
    nested: bool,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, object]]:
    x_a, x_c, y, _ = load_v2s_matrices(deep_source, meta)
    groups = meta["group_key"].to_numpy(str)
    cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    proba = np.zeros((len(y), 3), dtype=float)
    thresholds = np.zeros(len(y), dtype=float)
    folds = np.zeros(len(y), dtype=int)
    fold_metric = {"accuracy": [], "auc_macro_ovr": [], "macro_dice": []}
    splits = [
        (fold, tr, te)
        for fold, (tr, te) in enumerate(
            cv.split(np.zeros(len(y)), y, groups), 1
        )
    ]
    outputs = Parallel(n_jobs=max(CV_JOBS, 1))(
        delayed(grouped_fold_output)(
            condition, fold, tr, te, x_a, x_c, y, groups, nested
        )
        for fold, tr, te in splits
    )
    for fold, te, p, accuracy, auc_macro, macro_dice, threshold in outputs:
        proba[te] = p
        folds[te] = fold
        fold_metric["accuracy"].append(accuracy)
        fold_metric["auc_macro_ovr"].append(auc_macro)
        fold_metric["macro_dice"].append(macro_dice)
        if nested:
            if threshold is None:
                raise RuntimeError(f"missing nested threshold for {condition} fold {fold}")
            thresholds[te] = threshold
        log(f"{condition} fold {fold}/10 acc={fold_metric['accuracy'][-1]:.4f}")
    if not nested:
        thresholds[:] = youden_threshold((y == 1).astype(int), proba[:, 1])
    if not np.isfinite(proba).all() or np.any(folds == 0):
        raise RuntimeError(f"{condition} did not produce exactly one prediction per row")
    summary = summarize_predictions(
        condition, y, proba, thresholds, fold_metric=fold_metric
    )
    predictions = prediction_frame(
        condition, meta, np.arange(len(y)), y, proba, folds, thresholds
    )
    threshold_text = (
        "outer-fold-specific threshold selected on one group-stratified calibration fold "
        "inside the corresponding training partition"
        if nested
        else "single pooled Youden threshold selected on all OOF predictions; descriptive, not nested"
    )
    registry = experiment_row(
        condition,
        status=str(summary.get("status", "ok")),
        n_predictions=len(predictions),
        fitting_data="outer training folds",
        evaluation_data="one grouped outer fold at a time; all 25,903 rows receive OOF scores",
        splitter="StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)",
        threshold_provenance=threshold_text,
        representation_provenance="partially in-sample CNN representations",
        caveat=(
            "group_key currently falls back to one unique file per row because lesion_id and "
            "patient_id are unresolved; grouped CV therefore does not establish biological independence"
        ),
    )
    return summary, predictions, registry


def run_grouped_cv(
    condition: str,
    meta: pd.DataFrame,
    deep_source: str,
    nested: bool,
) -> dict[str, object]:
    summary, _, _ = run_grouped_cv_outputs(
        condition, meta, deep_source, nested
    )
    return summary


def blocked_row(condition: str, reason: str) -> dict[str, object]:
    return {"condition": condition, "status": "blocked_missing_data", "reason": reason}


def write_decomposition(rows: list[dict[str, object]]) -> None:
    OUT_DECOMP.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(OUT_DECOMP, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_observation_outputs(
    prediction_frames: list[pd.DataFrame],
    experiment_rows: list[dict[str, object]],
) -> None:
    if prediction_frames:
        predictions = pd.concat(prediction_frames, ignore_index=True)
        predictions = predictions.sort_values(
            ["condition", "row_index"], kind="stable"
        ).reset_index(drop=True)
        if predictions.duplicated(["condition", "row_index"]).any():
            raise ValueError("duplicate condition/row_index in internal prediction export")
        predictions.to_csv(
            OUT_PREDICTIONS, index=False, float_format="%.10g"
        )
    elif OUT_PREDICTIONS.exists():
        raise RuntimeError(
            "no predictions were produced; refusing to overwrite or silently reuse "
            f"{OUT_PREDICTIONS}"
        )
    pd.DataFrame(experiment_rows).to_csv(OUT_EXPERIMENTS, index=False)


def markdown_table(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    if df.empty:
        return "_empty_"
    rows = []
    cols = [str(c) for c in df.columns]
    rows.append("| " + " | ".join(cols) + " |")
    rows.append("| " + " | ".join("---" for _ in cols) + " |")
    for _, r in df.iterrows():
        vals = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, float):
                vals.append(format(v, floatfmt))
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def write_report(
    meta: pd.DataFrame,
    used_metadata: list[Path],
    mapping_issues: list[str],
    decomp_rows: list[dict[str, object]],
    deep_source: str,
) -> None:
    cov = coverage_table(meta)
    dup = duplication_table(meta)
    audit = split_audit(meta)
    source_counts = meta["source"].value_counts().sort_index()
    class_counts = meta.groupby(["split_original", "class_name"]).size().unstack(fill_value=0)
    multi_pct_all = float(dup.loc[dup["source"] == "ALL", "pct_images_in_multi_image_group"].iloc[0])

    lines = [
        "# Validite interne M1",
        "",
        "## Faisabilite et mapping",
        "",
        "- Voie retenue : reconstruction deterministe ligne<->fichier depuis l'ordre d'extraction existant (`train`, `valid`, `test`, puis `sorted(rglob)`).",
        f"- Deep source : `{deep_source}`.",
        "- Justification : les CSV de features historiques ne contenaient pas de filename ; les CSV V2S single-pass regeneres conservent `file`/`row_index` et restent alignes sur `artifacts/image_metadata.csv`.",
        f"- Metadata originelles trouvees : {len(used_metadata)} fichier(s).",
    ]
    if used_metadata:
        lines.extend(f"  - `{p}`" for p in used_metadata)
    else:
        lines.append("  - Aucune metadata source (`HAM10000_metadata.csv`, ISIC metadata) trouvee sous le depot.")
    lines.append(f"- Issues de mapping : {mapping_issues if mapping_issues else 'aucune sur features.csv/lesion PNG'}")
    lines.append("")
    lines.append("## Couverture source")
    lines.append("")
    lines.append(markdown_table(cov, ".2f"))
    lines.append("")
    lines.append("## Surface de duplication par groupe")
    lines.append("")
    lines.append(markdown_table(dup, ".2f"))
    lines.append("")
    lines.append("## Splits")
    lines.append("")
    lines.append("Images par source :")
    lines.append("")
    lines.append(markdown_table(source_counts.rename_axis("source").reset_index(name="n")))
    lines.append("")
    lines.append("Classes par split :")
    lines.append("")
    lines.append(markdown_table(class_counts.reset_index()))
    lines.append("")
    lines.append(f"- Groupes test chevauchant train+valid : {audit['group_overlap_count']}")
    lines.append(f"- Image IDs test chevauchant train+valid : {audit['image_id_overlap_count']}")
    if audit["image_id_overlap_examples"]:
        lines.append(f"- Exemples image_id overlap : {audit['image_id_overlap_examples']}")
    lines.append("")
    lines.append("## Resultats M1")
    lines.append("")
    lines.append(markdown_table(pd.DataFrame(decomp_rows), ".4f"))
    lines.append("")
    lines.append(
        f"Predictions par observation : `{OUT_PREDICTIONS.relative_to(paths.ROOT)}`."
    )
    lines.append(
        f"Registre des protocoles et caveats : `{OUT_EXPERIMENTS.relative_to(paths.ROOT)}`."
    )
    lines.append("")
    lines.append("## Lecture")
    lines.append("")
    if used_metadata:
        lines.append(f"- {multi_pct_all:.2f}% des images appartiennent a un groupe multi-image selon les IDs resolus.")
    else:
        lines.append("- Les metadata lesion/patient sont absentes ; la cle de groupe retombe sur `file`, donc la duplication lesion est non mesurable ici. Le 0.00% multi-image observe est un fallback technique, pas une preuve d'absence de doublons lesion.")
    lines.append(
        "- Pour 1a, les tetes sont ajustees sur train+valid puis le seuil de Youden "
        "est choisi sur valid. Les metriques argmax/AUC du test restent held-out, mais "
        "les metriques au seuil ne sont pas une estimation independante."
    )
    missing = [str(p) for p in required_csvs(deep_source) if not p.exists()]
    if missing:
        lines.append("- Les conditions CPU sont bloquees : CSV requis absents, aucun chiffre n'a ete invente.")
        lines.append(f"- Fichiers manquants : {missing}")
    if any(r.get("condition") == "C3" for r in decomp_rows):
        lines.append("- C3 est laisse en TODO : representation CNN propre = reentrainement/extraction GPU hors perimetre CPU.")
    lines.append("")
    OUT_REPORT.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--deep-source",
        choices=["deployed_single", "deployed_tta"],
        default="deployed_single",
    )
    p.add_argument(
        "--tta",
        action="store_const",
        const="deployed_tta",
        dest="deep_source",
        help="Use optional V2S TTA CSVs instead of canonical single-pass features.",
    )
    p.add_argument(
        "--observation-handoff-fast",
        action="store_true",
        help=(
            "Recompute and export held-out 1a observation predictions only. Preserve "
            "the recorded aggregate C0/C1/C2 table and register their missing OOF rows "
            "instead of spending hours recomputing CPU folds."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths.ARTIFACTS.mkdir(exist_ok=True)
    paths.RESULTS_INTERNAL.mkdir(parents=True, exist_ok=True)
    log("== Phase 0: image metadata ==")
    meta, used_metadata = build_image_metadata()
    mapping_issues = verify_mapping(meta)
    log(f"wrote {OUT_METADATA} rows={len(meta)}")
    if mapping_issues:
        log("mapping issues: " + "; ".join(mapping_issues))

    decomp_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    experiment_rows: list[dict[str, object]] = []
    missing = [str(p) for p in required_csvs(args.deep_source) if not p.exists()]
    if missing:
        reason = f"missing required V2S CSV(s) for {args.deep_source}: " + ", ".join(missing)
        log("BLOCKED CPU conditions: " + reason)
        decomp_rows.append(blocked_row("1a", reason))
        decomp_rows.append(blocked_row("C0", reason))
        decomp_rows.append(blocked_row("C1", reason))
        decomp_rows.append(blocked_row("C2", reason))
    else:
        log("== Phase 1a ==")
        phase_1a, phase_1a_predictions, phase_1a_registry = run_phase_1a_outputs(
            meta, args.deep_source
        )
        decomp_rows.append(phase_1a)
        prediction_frames.append(phase_1a_predictions)
        experiment_rows.append(phase_1a_registry)
        if args.observation_handoff_fast:
            for condition, threshold_text in (
                (
                    "C0",
                    "pooled threshold on OOF predictions in the recorded run",
                ),
                (
                    "C1",
                    "pooled threshold on grouped OOF predictions in the recorded run",
                ),
                (
                    "C2",
                    "nested thresholds in the recorded run",
                ),
            ):
                experiment_rows.append(
                    experiment_row(
                        condition,
                        status="aggregate-only-observation-predictions-not-retained",
                        n_predictions=0,
                        fitting_data="see recorded aggregate experiment and current feature matrices",
                        evaluation_data="all 25,903 image rows in the recorded CV run",
                        splitter=f"see {paths.relative(OUT_DECOMP)} and protocol documentation",
                        threshold_provenance=threshold_text,
                        representation_provenance="partially in-sample CNN representations",
                        caveat=(
                            "The historical run retained aggregate metrics but not OOF "
                            "observation predictions/fold assignments. Recompute with the full "
                            "command in a GPU-enabled or long-running environment if needed."
                        ),
                    )
                )
            experiment_rows.append(
                experiment_row(
                    "C3",
                    status="blocked_needs_gpu",
                    n_predictions=0,
                    fitting_data="not run",
                    evaluation_data="not run",
                    splitter="planned patient/lesion-disjoint evaluation",
                    threshold_provenance="not available",
                    representation_provenance="requires clean CNN retraining/extraction",
                    caveat="blocked; no observation-level predictions exist",
                )
            )
            write_observation_outputs(prediction_frames, experiment_rows)
            log(f"wrote {OUT_PREDICTIONS}")
            log(f"wrote {OUT_EXPERIMENTS}")
            log(
                "preserved recorded aggregate decomposition/report; C0/C1/C2 OOF "
                "predictions were not retained by the historical run"
            )
            return
        log("== C0 sanity gate ==")
        c0, c0_predictions, c0_registry = run_c0_outputs(meta, args.deep_source)
        decomp_rows.append(c0)
        prediction_frames.append(c0_predictions)
        experiment_rows.append(c0_registry)
        if c0.get("status") == "failed_sanity_gate":
            log("C0 sanity gate failed; stopping before any later condition.")
        else:
            log("== C1 grouped CV, pooled threshold ==")
            c1, c1_predictions, c1_registry = run_grouped_cv_outputs(
                "C1", meta, args.deep_source, nested=False
            )
            decomp_rows.append(c1)
            prediction_frames.append(c1_predictions)
            experiment_rows.append(c1_registry)
            log("== C2 grouped CV, nested threshold ==")
            c2, c2_predictions, c2_registry = run_grouped_cv_outputs(
                "C2", meta, args.deep_source, nested=True
            )
            decomp_rows.append(c2)
            prediction_frames.append(c2_predictions)
            experiment_rows.append(c2_registry)
        decomp_rows.append(
            {
                "condition": "C3",
                "status": "blocked_needs_gpu",
                "reason": "requires lesion-disjoint CNN retraining/extraction; out of CPU scope",
            }
        )
        experiment_rows.append(
            experiment_row(
                "C3",
                status="blocked_needs_gpu",
                n_predictions=0,
                fitting_data="not run",
                evaluation_data="not run",
                splitter="planned patient/lesion-disjoint evaluation",
                threshold_provenance="not available",
                representation_provenance="requires clean CNN retraining/extraction",
                caveat="blocked; no observation-level predictions exist",
            )
        )

    write_decomposition(decomp_rows)
    if not missing:
        write_observation_outputs(prediction_frames, experiment_rows)
    write_report(meta, used_metadata, mapping_issues, decomp_rows, args.deep_source)
    log(f"wrote {OUT_DECOMP}")
    if not missing:
        log(f"wrote {OUT_PREDICTIONS}")
        log(f"wrote {OUT_EXPERIMENTS}")
    log(f"wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
