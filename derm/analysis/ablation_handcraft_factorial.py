"""Factorial ABCD x deep ablation on the clean CPU harness.

Primary pre-registered metric: clean internal melanoma OvR AUC.

Default mode is strict about the deployed canonical representation: V2S
single-pass feature CSVs produced by ``derm.pipeline.extract_v2s_onnx``. TTA is
kept as an explicit option and, if requested without its CSVs, remains blocked
instead of falling back to the older 512-d baseline.

CPU scope only:
  - C1/C2-style internal grouped CV on existing features.
  - External OOD exploratory evaluation from cached/generated single-pass ONNX
    features.
  - B-block confound scaffold for Otsu masks.

TODO(C3): rerun the same factorial grid on out-of-sample CNN representations
after GPU retraining/extraction.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import binomtest, norm
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from derm import paths
from derm.analysis import internal_validity_m1 as iv


SEED = 42
CLASSES = (0, 1, 2)
CLASS_NAMES = {0: "nevus", 1: "melanoma", 2: "atypical"}
PRIMARY_METRIC = "auc_melanoma_ovr"
N_JOBS = int(os.environ.get("DERM_LGBM_JOBS", "4"))
LGBM_DEVICE = os.environ.get("DERM_LGBM_DEVICE", "cpu").lower()
FOLD_WORKERS = int(os.environ.get("DERM_ABLATION_FOLD_WORKERS", "1"))


def lgbm_device_kwargs() -> dict[str, int | str]:
    if LGBM_DEVICE == "gpu":
        return {
            "device_type": "gpu",
            "gpu_platform_id": int(os.environ.get("DERM_LGBM_GPU_PLATFORM_ID", "0")),
            "gpu_device_id": int(os.environ.get("DERM_LGBM_GPU_DEVICE_ID", "0")),
        }
    return {}

OUT_FACTORIAL = paths.ABLATION_HANDCRAFT_FACTORIAL
OUT_EFFECTS = paths.ABLATION_HANDCRAFT_EFFECTS
OUT_CONFOUNDB = paths.ABLATION_HANDCRAFT_CONFOUNDB
OUT_DOC = paths.ABLATION_HANDCRAFT_DOC
EXTERNAL_CACHE = paths.ARTIFACTS / "external_ablation_handcraft_singlepass_v2s.csv"
INTERNAL_B_OTSU_CACHE = paths.ARTIFACTS / "features_B_otsu.csv"
PRED_CACHE_DIR = paths.ABLATION_HANDCRAFT_CACHE

HC_BLOCKS: dict[str, list[str]] = {
    "A": ["A4"],
    "B": [
        "B",
        "internal_R1",
        "internal_R2",
        "internal_R4",
        "internal_R8",
        "internal_R16",
        "external_R1",
        "external_R2",
        "external_R4",
        "external_R8",
        "external_R16",
        "fractal_D",
    ],
    "C": ["Lab_a_kurt", "Lab_a_std", "Lab_b_std", "Lab_b_skew"],
    "D": ["D"],
}
HC_FEATURE_ORDER = [
    "A4",
    "B",
    "D",
    "Lab_a_kurt",
    "Lab_b_std",
    "Lab_a_std",
    "Lab_b_skew",
    "internal_R1",
    "internal_R2",
    "internal_R4",
    "internal_R8",
    "internal_R16",
    "external_R1",
    "external_R2",
    "external_R4",
    "external_R8",
    "external_R16",
    "fractal_D",
]


@dataclass(frozen=True)
class Condition:
    blocks: tuple[str, ...]
    deep: bool

    @property
    def id(self) -> str:
        prefix = "Deep" if self.deep else "HC"
        suffix = "".join(self.blocks) if self.blocks else "none"
        return f"{prefix}_{suffix}"

    @property
    def label(self) -> str:
        pieces = []
        if self.deep:
            pieces.append("Deep")
        pieces.extend(self.blocks)
        return "+".join(pieces)


@dataclass
class PredictionSet:
    front: str
    condition: Condition
    y: np.ndarray
    proba: np.ndarray
    pred_mel: np.ndarray
    threshold_mel: np.ndarray


class MissingData(RuntimeError):
    pass


def log(msg: str) -> None:
    print(msg, flush=True)


def build_conditions() -> list[Condition]:
    out: list[Condition] = []
    blocks = tuple(HC_BLOCKS)
    for deep in (False, True):
        for r in range(0, len(blocks) + 1):
            for subset in itertools.combinations(blocks, r):
                if not deep and not subset:
                    continue
                out.append(Condition(tuple(subset), deep))
    return out


def selected_hc_columns(condition: Condition) -> list[str]:
    cols: list[str] = []
    for block in condition.blocks:
        cols.extend(HC_BLOCKS[block])
    return cols


def make_model() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        random_state=SEED,
        n_jobs=N_JOBS,
        verbose=-1,
        **lgbm_device_kwargs(),
    )


def pipe() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("clf", make_model())])


def proba3(model: Pipeline, x: np.ndarray) -> np.ndarray:
    p = model.predict_proba(x)
    classes = list(model.named_steps["clf"].classes_)
    out = np.zeros((len(x), 3), dtype=float)
    for j, cls in enumerate(classes):
        out[:, int(cls)] = p[:, j]
    row_sum = out.sum(axis=1)
    bad = row_sum <= 0
    if np.any(bad):
        out[bad] = 1.0 / 3.0
        row_sum = out.sum(axis=1)
    return out / row_sum[:, None]


def fit_condition(
    xa: np.ndarray,
    xc: np.ndarray | None,
    y: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
) -> np.ndarray:
    ma = pipe().fit(xa[tr], y[tr])
    pa = proba3(ma, xa[te])
    if xc is None:
        return pa
    mc = pipe().fit(xc[tr], y[tr])
    pc = proba3(mc, xc[te])
    return (pa + pc) / 2.0


def make_condition_matrices(
    condition: Condition,
    hc: pd.DataFrame,
    deep_a: np.ndarray | None,
    deep_c: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, int, int]:
    hc_cols = selected_hc_columns(condition)
    x_hc = hc[hc_cols].to_numpy(float) if hc_cols else None
    n_hc = 0 if x_hc is None else x_hc.shape[1]
    if condition.deep:
        if deep_a is None or deep_c is None:
            raise MissingData("deep features are required for this condition")
        xa = deep_a if x_hc is None else np.c_[x_hc, deep_a]
        xc = deep_c if x_hc is None else np.c_[x_hc, deep_c]
        return xa, xc, n_hc, deep_a.shape[1] + deep_c.shape[1]
    if x_hc is None:
        raise ValueError("empty handcraft-only condition is not useful")
    return x_hc, None, n_hc, 0


def midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    z = x[order]
    ranks = np.zeros(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and z[j] == z[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(len(x), dtype=float)
    out[order] = ranks
    return out


def delong_components(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return math.nan, np.array([]), np.array([])
    tx = midrank(pos)
    ty = midrank(neg)
    tz = midrank(np.r_[pos, neg])
    auc = (tz[:m].sum() / m - (m + 1) / 2) / n
    v01 = (tz[:m] - tx) / n
    v10 = 1 - (tz[m:] - ty) / m
    return float(auc), v01, v10


def delong_auc_ci(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float, float]:
    auc, v01, v10 = delong_components(y_true, scores)
    if not np.isfinite(auc):
        return math.nan, math.nan, math.nan
    m, n = len(v01), len(v10)
    sx = float(np.var(v01, ddof=1)) if m > 1 else 0.0
    sy = float(np.var(v10, ddof=1)) if n > 1 else 0.0
    se = math.sqrt(max(sx / max(m, 1) + sy / max(n, 1), 0.0))
    z = norm.ppf(0.975)
    return auc, max(0.0, auc - z * se), min(1.0, auc + z * se)


def delong_paired(
    y_true: np.ndarray,
    scores_base: np.ndarray,
    scores_new: np.ndarray,
) -> tuple[float, float, float, float]:
    auc_a, a01, a10 = delong_components(y_true, scores_base)
    auc_b, b01, b10 = delong_components(y_true, scores_new)
    if not (np.isfinite(auc_a) and np.isfinite(auc_b)):
        return math.nan, math.nan, math.nan, math.nan
    d01 = b01 - a01
    d10 = b10 - a10
    var = 0.0
    if len(d01) > 1:
        var += float(np.var(d01, ddof=1) / len(d01))
    if len(d10) > 1:
        var += float(np.var(d10, ddof=1) / len(d10))
    delta = auc_b - auc_a
    if var <= 0:
        p = 1.0 if abs(delta) < 1e-15 else 0.0
        return delta, delta, delta, p
    se = math.sqrt(var)
    z = norm.ppf(0.975)
    p = 2 * (1 - norm.cdf(abs(delta / se)))
    return delta, delta - z * se, delta + z * se, p


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return math.nan, math.nan, math.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def safe_div(num: int, den: int) -> float:
    return float(num / den) if den else math.nan


def youden_threshold(y_binary: np.ndarray, scores: np.ndarray) -> float:
    y_binary = np.asarray(y_binary, dtype=int)
    scores = np.asarray(scores, dtype=float)
    best_t = float(scores.min()) if len(scores) else 0.5
    best_j = -np.inf
    for t in np.unique(scores):
        pred = scores >= t
        tp = int((pred & (y_binary == 1)).sum())
        fn = int((~pred & (y_binary == 1)).sum())
        tn = int((~pred & (y_binary == 0)).sum())
        fp = int((pred & (y_binary == 0)).sum())
        sens = safe_div(tp, tp + fn)
        spec = safe_div(tn, tn + fp)
        j = sens + spec - 1
        if j > best_j:
            best_j = j
            best_t = float(t)
    return best_t


def calibration_threshold(
    xa: np.ndarray,
    xc: np.ndarray | None,
    y: np.ndarray,
    groups: np.ndarray,
    tr: np.ndarray,
    mode: str,
) -> float:
    if mode == "outer_train":
        p_train = fit_condition(xa, xc, y, tr, tr)
        return youden_threshold((y[tr] == 1).astype(int), p_train[:, 1])

    y_tr = y[tr]
    g_tr = groups[tr]
    n_splits = min(5, np.bincount(y_tr, minlength=3).min())
    if n_splits < 2:
        p_train = fit_condition(xa, xc, y, tr, tr)
        return youden_threshold((y[tr] == 1).astype(int), p_train[:, 1])
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    inner_tr_rel, cal_rel = next(splitter.split(np.zeros(len(tr)), y_tr, g_tr))
    inner_tr = tr[inner_tr_rel]
    cal = tr[cal_rel]
    p_cal = fit_condition(xa, xc, y, inner_tr, cal)
    return youden_threshold((y[cal] == 1).astype(int), p_cal[:, 1])


def dice_per_class(y_true: np.ndarray, y_pred: np.ndarray) -> dict[int, float]:
    out: dict[int, float] = {}
    for c in CLASSES:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        out[c] = safe_div(2 * tp, 2 * tp + fp + fn)
    return out


def ece_multiclass(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> float:
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y_true).astype(float)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            m = (conf >= lo) & (conf <= hi)
        else:
            m = (conf >= lo) & (conf < hi)
        if not np.any(m):
            continue
        ece += (m.mean()) * abs(float(correct[m].mean()) - float(conf[m].mean()))
    return float(ece)


def brier_multiclass(y_true: np.ndarray, proba: np.ndarray) -> float:
    oh = np.zeros_like(proba, dtype=float)
    oh[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((proba - oh) ** 2, axis=1)))


def auc_per_class(y_true: np.ndarray, proba: np.ndarray) -> dict[int, float]:
    out: dict[int, float] = {}
    for c in CLASSES:
        yb = (y_true == c).astype(int)
        if len(np.unique(yb)) < 2:
            out[c] = math.nan
        else:
            out[c] = float(roc_auc_score(yb, proba[:, c]))
    return out


def summarize_predictions(
    pred: PredictionSet,
    n_hc: int,
    n_deep: int,
    deep_source: str,
    probe: dict[str, float] | None = None,
    status: str = "ok",
    reason: str = "",
) -> dict[str, object]:
    y = pred.y
    p = pred.proba
    y_argmax = p.argmax(axis=1)
    y_mel = (y == 1).astype(int)
    auc_mel, auc_lo, auc_hi = delong_auc_ci(y_mel, p[:, 1])
    aucs = auc_per_class(y, p)
    defined_auc = [v for v in aucs.values() if np.isfinite(v)]
    pred_mel = pred.pred_mel.astype(bool)

    tp = int((pred_mel & (y == 1)).sum())
    fn = int((~pred_mel & (y == 1)).sum())
    tn = int((~pred_mel & (y != 1)).sum())
    fp = int((pred_mel & (y != 1)).sum())
    sens, sens_lo, sens_hi = wilson(tp, tp + fn)
    spec, spec_lo, spec_hi = wilson(tn, tn + fp)
    ppv, ppv_lo, ppv_hi = wilson(tp, tp + fp)
    npv, npv_lo, npv_hi = wilson(tn, tn + fn)
    f1_mel = safe_div(2 * tp, 2 * tp + fp + fn)
    dpc = dice_per_class(y, y_argmax)
    precision, recall, _, _ = precision_recall_fscore_support(
        y,
        y_argmax,
        labels=list(CLASSES),
        zero_division=0,
    )
    row: dict[str, object] = {
        "front": pred.front,
        "condition_id": pred.condition.id,
        "condition": pred.condition.label,
        "deep": pred.condition.deep,
        "blocks": "".join(pred.condition.blocks) or "none",
        "status": status,
        "reason": reason,
        "primary_metric": PRIMARY_METRIC if pred.front == "internal_clean" else "exploratory",
        "deep_source": deep_source,
        "n_eval": len(y),
        "n_features_hc": n_hc,
        "n_features_deep_total": n_deep,
        "threshold_melanoma_mean": float(np.mean(pred.threshold_mel)),
        "threshold_melanoma_std": float(np.std(pred.threshold_mel)),
        "auc_melanoma_ovr": auc_mel,
        "auc_melanoma_ci95_low": auc_lo,
        "auc_melanoma_ci95_high": auc_hi,
        "auc_nevus_ovr": aucs[0],
        "auc_atypical_ovr": aucs[2],
        "auc_macro_ovr": float(np.mean(defined_auc)) if defined_auc else math.nan,
        "sensitivity_melanoma": sens,
        "sensitivity_melanoma_ci95_low": sens_lo,
        "sensitivity_melanoma_ci95_high": sens_hi,
        "specificity_melanoma": spec,
        "specificity_melanoma_ci95_low": spec_lo,
        "specificity_melanoma_ci95_high": spec_hi,
        "ppv_melanoma": ppv,
        "ppv_melanoma_ci95_low": ppv_lo,
        "ppv_melanoma_ci95_high": ppv_hi,
        "npv_melanoma": npv,
        "npv_melanoma_ci95_low": npv_lo,
        "npv_melanoma_ci95_high": npv_hi,
        "f1_melanoma_threshold": f1_mel,
        "balanced_accuracy_melanoma_threshold": (sens + spec) / 2,
        "youden_j_melanoma": sens + spec - 1,
        "tp_mel": tp,
        "fn_mel": fn,
        "tn_mel": tn,
        "fp_mel": fp,
        "accuracy": float(accuracy_score(y, y_argmax)),
        "macro_f1": float(f1_score(y, y_argmax, labels=list(CLASSES), average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y, y_argmax, labels=list(CLASSES))),
        "balanced_accuracy_argmax": float(balanced_accuracy_score(y, y_argmax)),
        "macro_dice": float(np.mean(list(dpc.values()))),
        "ece": ece_multiclass(y, p),
        "brier": brier_multiclass(y, p),
    }
    for c in CLASSES:
        name = CLASS_NAMES[c]
        row[f"dice_{name}"] = dpc[c]
        row[f"recall_{name}"] = float(recall[c])
        row[f"precision_{name}"] = float(precision[c])
    for block in HC_BLOCKS:
        row[f"probe_r2_{block}"] = math.nan if probe is None else probe.get(block, math.nan)
    return row


def run_internal_condition(
    condition: Condition,
    hc: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    deep_a: np.ndarray | None,
    deep_c: np.ndarray | None,
    calibration: str,
    cache_key: str | None = None,
) -> tuple[PredictionSet, int, int]:
    xa, xc, n_hc, n_deep = make_condition_matrices(condition, hc, deep_a, deep_c)
    n_splits = 10
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    proba = np.zeros((len(y), 3), dtype=float)
    pred_mel = np.zeros(len(y), dtype=bool)
    thresholds = np.zeros(len(y), dtype=float)
    splits: list[tuple[int, np.ndarray, np.ndarray]] = []
    for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups), 1):
        overlap = set(groups[tr]) & set(groups[te])
        if overlap:
            raise RuntimeError(f"group leakage in fold {fold}: {list(overlap)[:5]}")
        splits.append((fold, tr, te))

    cached_results: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, float]] = []
    todo: list[tuple[int, np.ndarray, np.ndarray]] = []
    if cache_key is not None:
        for fold, tr, te in splits:
            cached = load_cached_internal_fold(cache_key, condition, fold)
            if cached is None:
                todo.append((fold, tr, te))
            else:
                cached_results.append(cached)
        if cached_results:
            log(f"  fold cache hits {condition.label}: {len(cached_results)}/{len(splits)}")
    else:
        todo = splits

    def run_fold(fold: int, tr: np.ndarray, te: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, float]:
        thr = calibration_threshold(xa, xc, y, groups, tr, calibration)
        p = fit_condition(xa, xc, y, tr, te)
        result = (fold, te, p, p[:, 1] >= thr, thr)
        if cache_key is not None:
            save_cached_internal_fold(cache_key, condition, *result)
        return result

    if FOLD_WORKERS > 1:
        new_results = Parallel(n_jobs=FOLD_WORKERS, prefer="threads")(
            delayed(run_fold)(fold, tr, te) for fold, tr, te in todo
        )
    else:
        new_results = [run_fold(fold, tr, te) for fold, tr, te in todo]

    results = cached_results + new_results
    if len(results) != len(splits):
        raise RuntimeError(f"incomplete fold cache/results for {condition.label}: {len(results)}/{len(splits)}")
    for _fold, te, p, pred, thr in results:
        proba[te] = p
        pred_mel[te] = pred
        thresholds[te] = thr
    return PredictionSet("internal_clean", condition, y, proba, pred_mel, thresholds), n_hc, n_deep


def run_external_condition(
    condition: Condition,
    hc_int: pd.DataFrame,
    y_int: np.ndarray,
    deep_a_int: np.ndarray | None,
    deep_c_int: np.ndarray | None,
    ext: pd.DataFrame,
    calibration: str,
) -> tuple[PredictionSet, int, int]:
    train_condition = condition
    xa_int, xc_int, n_hc, n_deep = make_condition_matrices(
        train_condition,
        hc_int,
        deep_a_int,
        deep_c_int,
    )
    ext_hc = ext[[c for c in hc_int.columns if c != "Class"]].copy()
    deep_a_ext = ext.filter(regex=r"^cnnL_").to_numpy(float) if condition.deep else None
    deep_c_ext = ext.filter(regex=r"^cnnD_").to_numpy(float) if condition.deep else None
    xa_ext, xc_ext, _, _ = make_condition_matrices(condition, ext_hc, deep_a_ext, deep_c_ext)

    # Threshold is tuned on internal validation only, never on OOD samples.
    meta = load_or_build_metadata()
    valid = (meta["split_original"].to_numpy() == "valid")
    train = (meta["split_original"].to_numpy() == "train")
    if calibration == "inner":
        p_valid = fit_condition(xa_int, xc_int, y_int, np.flatnonzero(train), np.flatnonzero(valid))
        thr = youden_threshold((y_int[valid] == 1).astype(int), p_valid[:, 1])
    else:
        thr = calibration_threshold(
            xa_int,
            xc_int,
            y_int,
            meta["group_key"].to_numpy(str),
            np.arange(len(y_int)),
            "outer_train",
        )
    # Refit on all internal data and predict external with the same scaler/model family.
    ma = pipe().fit(xa_int, y_int)
    pa = proba3(ma, xa_ext)
    if xc_int is None or xc_ext is None:
        p = pa
    else:
        mc = pipe().fit(xc_int, y_int)
        p = (pa + proba3(mc, xc_ext)) / 2.0
    y_ext = ext["true"].to_numpy(int)
    return (
        PredictionSet(
            "external_ood_exploratory",
            condition,
            y_ext,
            p,
            p[:, 1] >= thr,
            np.full(len(y_ext), thr, dtype=float),
        ),
        n_hc,
        n_deep,
    )


def load_or_build_metadata() -> pd.DataFrame:
    if (paths.ARTIFACTS / "image_metadata.csv").exists():
        return pd.read_csv(paths.ARTIFACTS / "image_metadata.csv")
    meta, _ = iv.build_image_metadata()
    return meta


def load_internal_features(deep_source: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None, np.ndarray | None]:
    hc = pd.read_csv(paths.FEATURES_CSV)
    missing_cols = sorted(set(itertools.chain.from_iterable(HC_BLOCKS.values())) - set(hc.columns))
    if missing_cols:
        raise MissingData(f"missing handcraft columns: {missing_cols}")
    y = hc["Class"].to_numpy(int)
    hc_no_class = hc.drop(columns=["Class"])

    if deep_source == "deployed_single":
        required = [paths.CNN_V2S_FEATS_CSV, paths.CNN_V2S_DEHAIR_CSV]
    elif deep_source == "deployed_tta":
        required = [paths.CNN_V2S_TTA_CSV, paths.CNN_V2S_DEHAIR_TTA_CSV]
    elif deep_source == "baseline512":
        required = [paths.CNN_FEATS_CSV, paths.CNN_DEHAIR_CSV]
    else:
        raise ValueError(deep_source)
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise MissingData("missing required deep feature CSV(s): " + ", ".join(missing))

    da = pd.read_csv(required[0])
    dc = pd.read_csv(required[1])
    if not (len(hc) == len(da) == len(dc)):
        raise MissingData(f"row mismatch hc={len(hc)} deep_a={len(da)} deep_c={len(dc)}")
    if not np.array_equal(y, da["Class"].to_numpy(int)):
        raise MissingData("Class mismatch features.csv vs deep A CSV")
    if not np.array_equal(y, dc["Class"].to_numpy(int)):
        raise MissingData("Class mismatch features.csv vs deep C CSV")
    cols_a = [c for c in da.columns if c.startswith("cnn_")]
    cols_c = [c for c in dc.columns if c.startswith("cnn_")]
    if cols_a != cols_c:
        raise MissingData("deep A/C feature columns differ")
    return hc_no_class, y, da[cols_a].to_numpy(float), dc[cols_c].to_numpy(float)


def load_external_cache(generate: bool) -> pd.DataFrame:
    if EXTERNAL_CACHE.exists():
        df = pd.read_csv(EXTERNAL_CACHE)
        missing_otsu = [f"otsu_{c}" for c in HC_BLOCKS["B"] if f"otsu_{c}" not in df.columns]
        if not missing_otsu:
            return df
        if not generate:
            raise MissingData(f"external cache lacks Otsu B columns: {missing_otsu}")
        EXTERNAL_CACHE.unlink()
    if not generate:
        raise MissingData(f"missing external feature cache: {EXTERNAL_CACHE}")
    return generate_external_cache()


def generate_external_cache() -> pd.DataFrame:
    try:
        import cv2
        from derm.export import eval_external as E
    except Exception as exc:  # pragma: no cover - depends on export venv
        raise MissingData(f"external feature generation requires cv2/onnxruntime/pillow_heif: {exc}") from exc

    def cnn_single(sess, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        r = cv2.resize(rgb, (E.C, E.C), interpolation=cv2.INTER_LINEAR)
        x = ((r.astype(np.float32) / 255.0 - E.MEAN) / E.STD).transpose(2, 0, 1)[None]
        return sess.run(None, {"input": np.ascontiguousarray(x, np.float32)})[0].ravel()

    unet = E.sess(E.MODELS / "unet.onnx")
    cnn_l = E.sess(E.MODELS / "cnn_lesion_feats.onnx")
    cnn_d = E.sess(E.MODELS / "cnn_dehair_feats.onnx")
    rows: list[dict[str, object]] = []
    for folder, cid in E.FOLDERS.items():
        imgs = sorted(p for p in (paths.EXTERNAL_VAL / folder).iterdir() if p.suffix.lower() in E.IMG_EXTS)
        for i, ip in enumerate(imgs, 1):
            bgr = E.decode_bgr(ip)
            if bgr is None:
                continue
            bgr = cv2.resize(bgr, (E.WORK, E.WORK), interpolation=cv2.INTER_LINEAR)
            mask = E.unet_mask(unet, bgr)
            binary = (mask > 0).astype(np.uint8)
            lesion = cv2.bitwise_and(bgr, bgr, mask=mask)
            dehair = E.dehair_bgr(bgr)
            hc = E.handcraft18(binary, lesion)
            otsu_b = b_block_from_bgr_otsu(bgr)
            fl = cnn_single(cnn_l, lesion)
            fd = cnn_single(cnn_d, dehair)
            row: dict[str, object] = {"file": ip.name, "folder": folder, "true": cid}
            for col, val in zip(HC_FEATURE_ORDER, hc):
                row[col] = float(val)
            for col, val in zip(HC_BLOCKS["B"], otsu_b):
                row[f"otsu_{col}"] = float(val)
            for j, val in enumerate(fl):
                row[f"cnnL_{j}"] = float(val)
            for j, val in enumerate(fd):
                row[f"cnnD_{j}"] = float(val)
            rows.append(row)
            if i % 25 == 0:
                log(f"external cache {folder}: {i}/{len(imgs)}")
    df = pd.DataFrame(rows)
    paths.ARTIFACTS.mkdir(exist_ok=True)
    df.to_csv(EXTERNAL_CACHE, index=False)
    log(f"wrote {EXTERNAL_CACHE} rows={len(df)}")
    return df


def largest_component(binary: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise MissingData(f"OpenCV required for Otsu mask computation: {exc}") from exc
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    if n <= 1:
        return binary.astype(np.uint8)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == idx).astype(np.uint8)


def b_block_from_binary(binary: np.ndarray) -> list[float]:
    from derm.preprocess import features as F

    binary = binary.astype(np.uint8)
    b, _diameter = F.border_diameter(binary)
    morpho = F.morpho_fractal(binary)
    return [float(b)] + [float(v) for v in morpho]


def b_block_from_bgr_otsu(bgr: np.ndarray) -> list[float]:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise MissingData(f"OpenCV required for Otsu mask computation: {exc}") from exc
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _t, inv = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = inv.astype(np.uint8)
    if binary.mean() > 0.70:
        binary = 1 - binary
    binary = largest_component(binary)
    return b_block_from_binary(binary)


def raw_image_path(row: pd.Series) -> Path | None:
    stem = Path(str(row["file"])).stem
    base = paths.MERGED / str(row["split_original"]) / str(row["class_name"])
    for ext in (".jpg", ".jpeg", ".png"):
        p = base / f"{stem}{ext}"
        if p.exists():
            return p
    matches = sorted(base.glob(stem + ".*"))
    return matches[0] if matches else None


def load_or_generate_internal_b_otsu(meta: pd.DataFrame) -> pd.DataFrame:
    if INTERNAL_B_OTSU_CACHE.exists():
        return pd.read_csv(INTERNAL_B_OTSU_CACHE)
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise MissingData(f"OpenCV required for internal B confound: {exc}") from exc
    rows = []
    for i, row in meta.iterrows():
        ip = raw_image_path(row)
        if ip is None:
            raise MissingData(f"raw image not found for {row['file']}")
        bgr = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        if bgr is None:
            raise MissingData(f"failed to read raw image {ip}")
        vals = b_block_from_bgr_otsu(bgr)
        out = {"row_index": int(row["row_index"]), "Class": int(row["class_id"])}
        for col, val in zip(HC_BLOCKS["B"], vals):
            out[col] = val
        rows.append(out)
        if (i + 1) % 2500 == 0:
            log(f"internal Otsu B {i+1}/{len(meta)}")
    df = pd.DataFrame(rows).sort_values("row_index")
    df.to_csv(INTERNAL_B_OTSU_CACHE, index=False)
    log(f"wrote {INTERNAL_B_OTSU_CACHE} rows={len(df)}")
    return df


def probe_r2_internal(
    hc: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    deep_a: np.ndarray | None,
    deep_c: np.ndarray | None,
) -> dict[str, float]:
    if deep_a is None or deep_c is None:
        return {b: math.nan for b in HC_BLOCKS}
    z = np.c_[deep_a, deep_c]
    out: dict[str, float] = {}
    cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    for block, cols in HC_BLOCKS.items():
        target = hc[cols].to_numpy(float)
        yp = np.zeros_like(target, dtype=float)
        for tr, te in cv.split(np.zeros(len(y)), y, groups):
            model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
            model.fit(z[tr], target[tr])
            yp_fold = np.asarray(model.predict(z[te]), dtype=float)
            if yp_fold.ndim == 1:
                yp_fold = yp_fold[:, None]
            yp[te] = yp_fold
        vals = r2_score(target, yp, multioutput="raw_values")
        out[block] = float(np.mean(vals))
    return out


def mcnemar_p(y_true: np.ndarray, pred_base: np.ndarray, pred_new: np.ndarray) -> tuple[int, int, float]:
    correct_base = pred_base == (y_true == 1)
    correct_new = pred_new == (y_true == 1)
    b = int((correct_base & ~correct_new).sum())
    c = int((~correct_base & correct_new).sum())
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(binomtest(min(b, c), b + c, 0.5).pvalue)


def add_unet_delta(row: dict[str, object], base: PredictionSet, new: PredictionSet) -> None:
    yb = (base.y == 1).astype(int)
    delta, lo, hi, p = delong_paired(yb, base.proba[:, 1], new.proba[:, 1])
    b, c, pm = mcnemar_p(base.y, base.pred_mel, new.pred_mel)
    row["delta_auc_mel_vs_unet"] = delta
    row["delta_auc_mel_vs_unet_ci95_low"] = lo
    row["delta_auc_mel_vs_unet_ci95_high"] = hi
    row["delong_p_vs_unet"] = p
    row["mcnemar_b_unet_only_correct"] = b
    row["mcnemar_c_otsu_only_correct"] = c
    row["mcnemar_p_vs_unet"] = pm


def hc_with_otsu_b(hc: pd.DataFrame, b_otsu: pd.DataFrame) -> pd.DataFrame:
    out = hc.copy()
    for col in HC_BLOCKS["B"]:
        out[col] = b_otsu[col].to_numpy(float)
    return out


def external_with_otsu_b(ext: pd.DataFrame) -> pd.DataFrame:
    out = ext.copy()
    missing = [f"otsu_{c}" for c in HC_BLOCKS["B"] if f"otsu_{c}" not in out.columns]
    if missing:
        raise MissingData(f"external cache lacks Otsu B columns: {missing}")
    for col in HC_BLOCKS["B"]:
        out[col] = out[f"otsu_{col}"].to_numpy(float)
    return out


def run_confound_b(
    hc: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    deep_a: np.ndarray | None,
    deep_c: np.ndarray | None,
    preds: dict[tuple[str, str], PredictionSet],
    ext_df: pd.DataFrame | None,
    args: argparse.Namespace,
    probe: dict[str, float],
) -> list[dict[str, object]]:
    meta = load_or_build_metadata()
    b_otsu = load_or_generate_internal_b_otsu(meta)
    if len(b_otsu) != len(hc):
        raise MissingData(f"Otsu B rows={len(b_otsu)} vs hc rows={len(hc)}")
    hc_otsu = hc_with_otsu_b(hc, b_otsu)
    rows: list[dict[str, object]] = []
    for cond in (Condition(("B",), False), Condition(("B",), True)):
        unet = preds.get(("internal_clean", cond.id))
        if unet is not None:
            n_hc = len(selected_hc_columns(cond))
            n_deep = 0 if not cond.deep or deep_a is None or deep_c is None else deep_a.shape[1] + deep_c.shape[1]
            r = summarize_predictions(unet, n_hc, n_deep, args.deep_source, probe)
            r["mask_source"] = "unet"
            r["confound_comparison"] = "B_unet"
            rows.append(r)
        otsu, n_hc, n_deep = run_internal_condition(cond, hc_otsu, y, groups, deep_a, deep_c, args.calibration)
        r = summarize_predictions(otsu, n_hc, n_deep, args.deep_source, probe)
        r["mask_source"] = "otsu"
        r["confound_comparison"] = "B_otsu"
        if unet is not None:
            add_unet_delta(r, unet, otsu)
        rows.append(r)

        if ext_df is not None:
            ext_otsu = external_with_otsu_b(ext_df)
            ext_unet = preds.get(("external_ood_exploratory", cond.id))
            if ext_unet is not None:
                r = summarize_predictions(ext_unet, n_hc, n_deep, args.deep_source, probe)
                r["mask_source"] = "unet"
                r["confound_comparison"] = "B_unet"
                rows.append(r)
            ext_pred, n_hc, n_deep = run_external_condition(
                cond,
                hc_otsu,
                y,
                deep_a,
                deep_c,
                ext_otsu,
                args.calibration,
            )
            r = summarize_predictions(ext_pred, n_hc, n_deep, args.deep_source, probe)
            r["mask_source"] = "otsu"
            r["confound_comparison"] = "B_otsu"
            if ext_unet is not None:
                add_unet_delta(r, ext_unet, ext_pred)
            rows.append(r)
    return rows


def bh_fdr(rows: list[dict[str, object]], p_col: str = "p_value") -> None:
    indexed = [(i, float(r[p_col])) for i, r in enumerate(rows) if np.isfinite(r.get(p_col, math.nan))]
    if not indexed:
        return
    indexed.sort(key=lambda x: x[1])
    m = len(indexed)
    qvals = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        idx, p = indexed[rank - 1]
        running = min(running, p * m / rank)
        qvals[rank - 1] = running
    for (idx, _), q in zip(indexed, qvals):
        rows[idx]["q_value_bh_fdr"] = min(1.0, q)
        rows[idx]["expected_false_positive_family"] = float(rows[idx]["q_value_bh_fdr"])


def direct_effect_row(
    front: str,
    effect_type: str,
    block: str,
    base: PredictionSet,
    new: PredictionSet,
) -> dict[str, object]:
    yb = (base.y == 1).astype(int)
    delta, lo, hi, p = delong_paired(yb, base.proba[:, 1], new.proba[:, 1])
    b, c, pm = mcnemar_p(base.y, base.pred_mel, new.pred_mel)
    return {
        "front": front,
        "effect_type": effect_type,
        "block": block,
        "comparison": f"{new.condition.label} - {base.condition.label}",
        "metric": PRIMARY_METRIC,
        "delta": delta,
        "delta_ci95_low": lo,
        "delta_ci95_high": hi,
        "test": "paired DeLong",
        "p_value": p,
        "mcnemar_b_base_only_correct": b,
        "mcnemar_c_new_only_correct": c,
        "mcnemar_p_value": pm,
        "family": f"{front}:{PRIMARY_METRIC}:factorial_blocks",
    }


def build_effects(preds: dict[tuple[str, str], PredictionSet]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for front in sorted({k[0] for k in preds}):
        def get(blocks: Iterable[str], deep: bool = True) -> PredictionSet:
            return preds[(front, Condition(tuple(blocks), deep).id)]

        deep = get(())
        full = get(("A", "B", "C", "D"))
        for block in HC_BLOCKS:
            rows.append(direct_effect_row(front, "add_one", block, deep, get((block,))))
            loo_blocks = tuple(b for b in HC_BLOCKS if b != block)
            rows.append(direct_effect_row(front, "leave_one_out", block, get(loo_blocks), full))

            deltas = []
            for r in range(0, 4):
                for subset in itertools.combinations([b for b in HC_BLOCKS if b != block], r):
                    base = get(subset)
                    new = get(tuple(sorted((*subset, block), key=list(HC_BLOCKS).index)))
                    deltas.append(
                        roc_auc_score((base.y == 1).astype(int), new.proba[:, 1])
                        - roc_auc_score((base.y == 1).astype(int), base.proba[:, 1])
                    )
            rows.append(
                {
                    "front": front,
                    "effect_type": "main_effect_factorial",
                    "block": block,
                    "comparison": f"mean paired add({block}) over 8 Deep-on subsets",
                    "metric": PRIMARY_METRIC,
                    "delta": float(np.mean(deltas)),
                    "delta_ci95_low": math.nan,
                    "delta_ci95_high": math.nan,
                    "test": "descriptive contrast",
                    "p_value": math.nan,
                    "family": f"{front}:{PRIMARY_METRIC}:factorial_blocks",
                }
            )

        for b1, b2 in itertools.combinations(HC_BLOCKS, 2):
            vals = []
            others = [b for b in HC_BLOCKS if b not in (b1, b2)]
            for r in range(0, len(others) + 1):
                for subset in itertools.combinations(others, r):
                    s = tuple(subset)
                    sx = tuple(sorted((*s, b1), key=list(HC_BLOCKS).index))
                    sy = tuple(sorted((*s, b2), key=list(HC_BLOCKS).index))
                    sxy = tuple(sorted((*s, b1, b2), key=list(HC_BLOCKS).index))
                    p0 = get(s)
                    px = get(sx)
                    py = get(sy)
                    pxy = get(sxy)
                    yb = (p0.y == 1).astype(int)
                    vals.append(
                        roc_auc_score(yb, pxy.proba[:, 1])
                        - roc_auc_score(yb, px.proba[:, 1])
                        - roc_auc_score(yb, py.proba[:, 1])
                        + roc_auc_score(yb, p0.proba[:, 1])
                    )
            rows.append(
                {
                    "front": front,
                    "effect_type": "pair_interaction_factorial",
                    "block": f"{b1}{b2}",
                    "comparison": f"mean interaction {b1}:{b2} over Deep-on subsets",
                    "metric": PRIMARY_METRIC,
                    "delta": float(np.mean(vals)),
                    "delta_ci95_low": math.nan,
                    "delta_ci95_high": math.nan,
                    "test": "descriptive contrast",
                    "p_value": math.nan,
                    "family": f"{front}:{PRIMARY_METRIC}:factorial_blocks",
                }
            )
    for family in sorted({r["family"] for r in rows}):
        fam = [r for r in rows if r["family"] == family and np.isfinite(r.get("p_value", math.nan))]
        bh_fdr(fam)
    return rows


def blocked_factorial(reason: str, deep_source: str) -> None:
    paths.RESULTS_ABLATION_HANDCRAFT.mkdir(parents=True, exist_ok=True)
    conditions = build_conditions()
    rows = []
    for front in ("internal_clean", "external_ood_exploratory"):
        for c in conditions:
            rows.append(
                {
                    "front": front,
                    "condition_id": c.id,
                    "condition": c.label,
                    "deep": c.deep,
                    "blocks": "".join(c.blocks) or "none",
                    "status": "blocked_missing_data",
                    "reason": reason,
                    "primary_metric": PRIMARY_METRIC if front == "internal_clean" else "exploratory",
                    "deep_source": deep_source,
                }
            )
    pd.DataFrame(rows).to_csv(OUT_FACTORIAL, index=False)
    pd.DataFrame(
        [
            {
                "status": "blocked_missing_data",
                "reason": reason,
                "family": "internal_clean:auc_melanoma_ovr:factorial_blocks",
            }
        ]
    ).to_csv(OUT_EFFECTS, index=False)
    pd.DataFrame(
        [
            {
                "status": "blocked_missing_data",
                "reason": reason,
                "comparison": "B U-Net vs B Otsu",
            }
        ]
    ).to_csv(OUT_CONFOUNDB, index=False)
    write_doc(pd.DataFrame(rows), pd.DataFrame(), pd.DataFrame(), reason)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def cache_path(cache_key: str, front: str, condition: Condition) -> Path:
    safe_front = front.replace("/", "_")
    return PRED_CACHE_DIR / f"{cache_key}__{safe_front}__{condition.id}.npz"


def save_cached_prediction(cache_key: str, pred: PredictionSet, n_hc: int, n_deep: int) -> None:
    PRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path(cache_key, pred.front, pred.condition),
        y=pred.y,
        proba=pred.proba,
        pred_mel=pred.pred_mel.astype(np.uint8),
        threshold_mel=pred.threshold_mel,
        n_hc=np.array([n_hc], dtype=np.int32),
        n_deep=np.array([n_deep], dtype=np.int32),
    )


def load_cached_prediction(cache_key: str, front: str, condition: Condition) -> tuple[PredictionSet, int, int] | None:
    p = cache_path(cache_key, front, condition)
    if not p.exists():
        return None
    z = np.load(p)
    pred = PredictionSet(
        front,
        condition,
        z["y"].astype(int),
        z["proba"].astype(float),
        z["pred_mel"].astype(bool),
        z["threshold_mel"].astype(float),
    )
    return pred, int(z["n_hc"][0]), int(z["n_deep"][0])


def internal_fold_cache_path(cache_key: str, condition: Condition, fold: int) -> Path:
    return PRED_CACHE_DIR / f"{cache_key}__internal_clean__{condition.id}__fold{fold:02d}.npz"


def save_cached_internal_fold(
    cache_key: str,
    condition: Condition,
    fold: int,
    te: np.ndarray,
    proba: np.ndarray,
    pred_mel: np.ndarray,
    threshold_mel: float,
) -> None:
    PRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = internal_fold_cache_path(cache_key, condition, fold)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        fold=np.array([fold], dtype=np.int16),
        te=te.astype(np.int32),
        proba=proba.astype(np.float32),
        pred_mel=pred_mel.astype(np.uint8),
        threshold_mel=np.array([threshold_mel], dtype=np.float64),
    )
    tmp.replace(out)


def load_cached_internal_fold(
    cache_key: str,
    condition: Condition,
    fold: int,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, float] | None:
    p = internal_fold_cache_path(cache_key, condition, fold)
    if not p.exists():
        return None
    try:
        z = np.load(p)
        return (
            int(z["fold"][0]),
            z["te"].astype(int),
            z["proba"].astype(float),
            z["pred_mel"].astype(bool),
            float(z["threshold_mel"][0]),
        )
    except Exception:
        p.unlink(missing_ok=True)
        return None


def md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
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
                vals.append(format(v, floatfmt) if np.isfinite(v) else "nan")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def write_doc(
    factorial: pd.DataFrame,
    effects: pd.DataFrame,
    confound: pd.DataFrame,
    note: str = "",
) -> None:
    lines = [
        "# Ablation factorielle ABCD x deep",
        "",
        "- Metrique primaire pre-enregistree : AUC melanome OvR interne propre.",
        "- Externe OOD : exploratoire seulement ; n=19 melanomes, IC a reporter sans conclusion forte sur 1-2 cas.",
        "- C3 representation propre GPU : TODO, hors perimetre CPU de ce run.",
        "- Correction multiple : Benjamini-Hochberg/FDR sur la famille des comparaisons directes.",
        "",
    ]
    if note:
        lines.extend(["## Statut", "", note, ""])
    if not factorial.empty and "status" in factorial:
        lines.extend(["## Synthese conditions cles", ""])
        keep = factorial[factorial["condition"].isin(["Deep", "Deep+A", "Deep+B", "Deep+C", "Deep+D", "Deep+A+B+C+D"])]
        cols = [
            c
            for c in [
                "front",
                "condition",
                "status",
                "auc_melanoma_ovr",
                "auc_melanoma_ci95_low",
                "auc_melanoma_ci95_high",
                "sensitivity_melanoma",
                "specificity_melanoma",
                "macro_f1",
                "ece",
                "brier",
                "reason",
            ]
            if c in keep.columns
        ]
        lines.append(md_table(keep[cols] if cols else keep.head(12)))
        lines.append("")
    if not effects.empty:
        lines.extend(["## Effets derives", ""])
        cols = [c for c in ["front", "effect_type", "block", "delta", "delta_ci95_low", "delta_ci95_high", "p_value", "q_value_bh_fdr"] if c in effects.columns]
        lines.append(md_table(effects[cols].head(60) if cols else effects.head(60)))
        lines.append("")
    if not confound.empty:
        lines.extend(["## Confound bloc B", ""])
        lines.append(md_table(confound.head(30)))
        lines.append("")
    lines.extend(
        [
            "## Verdict par bloc",
            "",
            "A renseigner apres run complet non bloque. Ne pas conclure sur l'OOD sans tenir compte de n=19 melanomes.",
            "",
        ]
    )
    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    OUT_DOC.write_text("\n".join(lines))


def run(args: argparse.Namespace) -> None:
    conditions = build_conditions()
    try:
        hc, y, deep_a, deep_c = load_internal_features(args.deep_source)
    except MissingData as exc:
        reason = str(exc)
        log("BLOCKED: " + reason)
        if args.missing_policy == "fail":
            raise
        blocked_factorial(reason, args.deep_source)
        return

    meta = load_or_build_metadata()
    if len(meta) != len(y):
        raise MissingData(f"metadata rows={len(meta)} vs feature rows={len(y)}")
    if not np.array_equal(meta["class_id"].to_numpy(int), y):
        raise MissingData("metadata class_id does not match features.csv Class")
    groups = meta["group_key"].to_numpy(str)

    probe = {} if args.skip_probing else probe_r2_internal(hc, y, groups, deep_a, deep_c)
    factor_rows: list[dict[str, object]] = []
    preds: dict[tuple[str, str], PredictionSet] = {}
    nfeat: dict[tuple[str, str], tuple[int, int]] = {}
    cache_key = f"{args.deep_source}_{args.calibration}"

    for i, cond in enumerate(conditions, 1):
        t0 = time.time()
        log(f"internal {i}/{len(conditions)} {cond.label}")
        cached = load_cached_prediction(cache_key, "internal_clean", cond)
        if cached is None:
            pred, n_hc, n_deep = run_internal_condition(
                cond,
                hc,
                y,
                groups,
                deep_a,
                deep_c,
                args.calibration,
                cache_key=cache_key,
            )
            save_cached_prediction(cache_key, pred, n_hc, n_deep)
        else:
            pred, n_hc, n_deep = cached
            log(f"  cache hit {cond.label}")
        preds[(pred.front, cond.id)] = pred
        nfeat[(pred.front, cond.id)] = (n_hc, n_deep)
        factor_rows.append(summarize_predictions(pred, n_hc, n_deep, args.deep_source, probe))
        write_csv(OUT_FACTORIAL, factor_rows)
        log(f"  done {cond.label} in {(time.time() - t0) / 60:.1f} min")

    ext_df: pd.DataFrame | None = None
    if args.external:
        try:
            ext_df = load_external_cache(generate=not args.no_generate_external_cache)
            for i, cond in enumerate(conditions, 1):
                if cond.deep and args.deep_source not in ("deployed_single", "deployed_tta"):
                    raise MissingData("external deep evaluation is only implemented for deployed V2S ONNX features")
                log(f"external {i}/{len(conditions)} {cond.label}")
                cached = load_cached_prediction(cache_key, "external_ood_exploratory", cond)
                if cached is None:
                    pred, n_hc, n_deep = run_external_condition(cond, hc, y, deep_a, deep_c, ext_df, args.calibration)
                    save_cached_prediction(cache_key, pred, n_hc, n_deep)
                else:
                    pred, n_hc, n_deep = cached
                    log(f"  cache hit external {cond.label}")
                preds[(pred.front, cond.id)] = pred
                nfeat[(pred.front, cond.id)] = (n_hc, n_deep)
                factor_rows.append(summarize_predictions(pred, n_hc, n_deep, args.deep_source, probe))
                write_csv(OUT_FACTORIAL, factor_rows)
        except MissingData as exc:
            reason = str(exc)
            for cond in conditions:
                factor_rows.append(
                    {
                        "front": "external_ood_exploratory",
                        "condition_id": cond.id,
                        "condition": cond.label,
                        "deep": cond.deep,
                        "blocks": "".join(cond.blocks) or "none",
                        "status": "blocked_missing_data",
                        "reason": reason,
                        "primary_metric": "exploratory",
                        "deep_source": args.deep_source,
                    }
                )

    effect_rows = build_effects(preds) if all(("internal_clean", c.id) in preds for c in conditions) else []
    if args.confound_b:
        try:
            confound_rows = run_confound_b(hc, y, groups, deep_a, deep_c, preds, ext_df, args, probe)
        except MissingData as exc:
            confound_rows = [
                {
                    "status": "blocked_missing_data",
                    "comparison": "B U-Net vs B Otsu",
                    "reason": str(exc),
                }
            ]
    else:
        confound_rows = [
            {
                "status": "not_run",
                "comparison": "B U-Net vs B Otsu",
                "reason": "Use --confound-b to recompute Otsu B features and compare B/Deep+B.",
            }
        ]
    write_csv(OUT_FACTORIAL, factor_rows)
    write_csv(OUT_EFFECTS, effect_rows)
    write_csv(OUT_CONFOUNDB, confound_rows)
    write_doc(pd.DataFrame(factor_rows), pd.DataFrame(effect_rows), pd.DataFrame(confound_rows))
    log(f"wrote {OUT_FACTORIAL}")
    log(f"wrote {OUT_EFFECTS}")
    log(f"wrote {OUT_CONFOUNDB}")
    log(f"wrote {OUT_DOC}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--deep-source",
        choices=["deployed_single", "deployed_tta", "baseline512"],
        default="deployed_single",
    )
    p.add_argument(
        "--tta",
        action="store_const",
        const="deployed_tta",
        dest="deep_source",
        help="Use optional V2S TTA CSVs instead of canonical single-pass features.",
    )
    p.add_argument("--missing-policy", choices=["blocked", "fail"], default="blocked")
    p.add_argument("--calibration", choices=["inner", "outer_train"], default="inner")
    p.add_argument("--external", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--no-generate-external-cache", action="store_true")
    p.add_argument("--skip-probing", action="store_true")
    p.add_argument("--confound-b", action="store_true")
    return p.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
