"""Render paper-oriented tables and figures from frozen result artifacts.

This script is intentionally read-only with respect to models, datasets and the
Flutter application. It only writes generated resources under ``docs/paper``.

Run from the repository root with the analysis environment::

    MPLCONFIGDIR=/tmp/matplotlib .venv_extval/bin/python \
      -m derm.analysis.render_paper_resources
"""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

from derm import paths


CURRENT_ID = "skinfusionnet-v2s3c-singlepass-1.0.0"
CURRENT_BUNDLE_SHA = (
    "24d6d97423c08b2864be590b07cadff027d1210ae28d3e76d7d7c85292833fc3"
)
TTA8_ID = "efficientnetv2s-cbam-ensemble+tta8 (2026-06-04)"
TTA8_RETROSPECTIVE_SHA = (
    "5b3023cf340d8369afa9c412b8bd9df7f242630bdf95803fce160150c79f7129"
)

CURRENT_CSV = paths.EXTERNAL_PARITY_CSV
TTA8_CSV = paths.TTA8_EXTERNAL_CSV
INTERNAL_CSV = paths.INTERNAL_DECOMPOSITION
CV_LOG = paths.INTERNAL_CV_LOG

OUT = paths.PAPER
GENERATED = OUT / "generated"
FIGURES = OUT / "figures"

T_MEL = 0.188
T_ATYP = 0.412
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 42

plt.rcParams["svg.hashsalt"] = "skinfusionnet-paper-resources-v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.zeros(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and sorted_values[end] == sorted_values[index]:
            end += 1
        ranks[index:end] = 0.5 * (index + end - 1) + 1
        index = end
    result = np.empty(len(values), dtype=float)
    result[order] = ranks
    return result


def delong_components(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    positive = scores[y_true == 1]
    negative = scores[y_true == 0]
    n_positive, n_negative = len(positive), len(negative)
    rank_positive = midrank(positive)
    rank_negative = midrank(negative)
    rank_all = midrank(np.r_[positive, negative])
    auc = (
        rank_all[:n_positive].sum() / n_positive - (n_positive + 1) / 2
    ) / n_negative
    v_positive = (rank_all[:n_positive] - rank_positive) / n_negative
    v_negative = 1 - (
        rank_all[n_positive:] - rank_negative
    ) / n_positive
    return float(auc), v_positive, v_negative


def delong_ci(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[float, float, float]:
    auc, v_positive, v_negative = delong_components(y_true, scores)
    variance = np.var(v_positive, ddof=1) / len(v_positive)
    variance += np.var(v_negative, ddof=1) / len(v_negative)
    half_width = norm.ppf(0.975) * math.sqrt(max(float(variance), 0.0))
    return auc, max(0.0, auc - half_width), min(1.0, auc + half_width)


def paired_delong(
    y_true: np.ndarray, scores_old: np.ndarray, scores_new: np.ndarray
) -> tuple[float, float, float]:
    auc_old, old_positive, old_negative = delong_components(y_true, scores_old)
    auc_new, new_positive, new_negative = delong_components(y_true, scores_new)
    variance = np.var(new_positive - old_positive, ddof=1) / len(old_positive)
    variance += np.var(new_negative - old_negative, ddof=1) / len(old_negative)
    if variance <= 0:
        p_value = 1.0 if auc_old == auc_new else 0.0
    else:
        z_score = (auc_new - auc_old) / math.sqrt(float(variance))
        p_value = float(2 * norm.sf(abs(z_score)))
    return auc_old, auc_new, p_value


def wilson(successes: int, total: int) -> tuple[float, float, float]:
    if total == 0:
        return math.nan, math.nan, math.nan
    z = 1.96
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    half_width = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return proportion, max(0.0, centre - half_width), min(1.0, centre + half_width)


def external_metrics(
    data: pd.DataFrame,
    *,
    model_id: str,
    bundle_sha256: str,
    status: str,
    source: Path,
) -> dict[str, object]:
    y_true = (data["true"].to_numpy(int) == 1).astype(int)
    score_melanoma = data["p_mel"].to_numpy(float)
    score_atypical = data["p_atyp"].to_numpy(float)
    auc, auc_low, auc_high = delong_ci(y_true, score_melanoma)
    threshold_prediction = score_melanoma >= T_MEL
    not_benign_prediction = threshold_prediction | (score_atypical >= T_ATYP)

    tp = int((threshold_prediction & (y_true == 1)).sum())
    fn = int((~threshold_prediction & (y_true == 1)).sum())
    tn = int((~threshold_prediction & (y_true == 0)).sum())
    fp = int((threshold_prediction & (y_true == 0)).sum())
    sensitivity, sensitivity_low, sensitivity_high = wilson(tp, tp + fn)
    specificity, specificity_low, specificity_high = wilson(tn, tn + fp)

    nb_tp = int((not_benign_prediction & (y_true == 1)).sum())
    nb_fn = int((~not_benign_prediction & (y_true == 1)).sum())
    nb_tn = int((~not_benign_prediction & (y_true == 0)).sum())
    nb_fp = int((not_benign_prediction & (y_true == 0)).sum())
    nb_sensitivity, nb_sensitivity_low, nb_sensitivity_high = wilson(
        nb_tp, nb_tp + nb_fn
    )
    nb_specificity, nb_specificity_low, nb_specificity_high = wilson(
        nb_tn, nb_tn + nb_fp
    )

    return {
        "model_id": model_id,
        "bundle_sha256": bundle_sha256,
        "status": status,
        "source": source.relative_to(paths.ROOT).as_posix(),
        "source_sha256": file_sha256(source),
        "n": len(data),
        "n_melanoma": int(y_true.sum()),
        "n_nevus": int((y_true == 0).sum()),
        "auc_ci_method": "DeLong 95%",
        "proportion_ci_method": "Wilson 95%",
        "auc_melanoma": auc,
        "auc_melanoma_ci95_low": auc_low,
        "auc_melanoma_ci95_high": auc_high,
        "average_precision_melanoma": average_precision_score(
            y_true, score_melanoma
        ),
        "brier_melanoma": brier_score_loss(y_true, score_melanoma),
        "threshold_melanoma": T_MEL,
        "sensitivity_melanoma": sensitivity,
        "sensitivity_melanoma_ci95_low": sensitivity_low,
        "sensitivity_melanoma_ci95_high": sensitivity_high,
        "specificity_melanoma": specificity,
        "specificity_melanoma_ci95_low": specificity_low,
        "specificity_melanoma_ci95_high": specificity_high,
        "tp_melanoma": tp,
        "fn_melanoma": fn,
        "tn_melanoma": tn,
        "fp_melanoma": fp,
        "not_benign_sensitivity": nb_sensitivity,
        "not_benign_sensitivity_ci95_low": nb_sensitivity_low,
        "not_benign_sensitivity_ci95_high": nb_sensitivity_high,
        "not_benign_specificity": nb_specificity,
        "not_benign_specificity_ci95_low": nb_specificity_low,
        "not_benign_specificity_ci95_high": nb_specificity_high,
        "not_benign_tp": nb_tp,
        "not_benign_fn": nb_fn,
        "not_benign_tn": nb_tn,
        "not_benign_fp": nb_fp,
        "argmax_exact_accuracy": float(
            (data["argmax"].to_numpy(int) == data["true"].to_numpy(int)).mean()
        ),
        "mask_coverage_median": float(data["coverage"].median()),
        "mask_coverage_below_1pct": int((data["coverage"] < 0.01).sum()),
    }


def paired_bootstrap_delta(
    merged: pd.DataFrame, repetitions: int = BOOTSTRAP_REPETITIONS
) -> tuple[float, float, float]:
    y_true = (merged["true"].to_numpy(int) == 1).astype(int)
    old_score = merged["p_mel_tta8"].to_numpy(float)
    new_score = merged["p_mel_singlepass"].to_numpy(float)
    positive = np.flatnonzero(y_true == 1)
    negative = np.flatnonzero(y_true == 0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sample = np.r_[
            rng.choice(positive, len(positive), replace=True),
            rng.choice(negative, len(negative), replace=True),
        ]
        deltas[index] = roc_auc_score(y_true[sample], new_score[sample])
        deltas[index] -= roc_auc_score(y_true[sample], old_score[sample])
    return (
        float(np.mean(deltas)),
        float(np.quantile(deltas, 0.025)),
        float(np.quantile(deltas, 0.975)),
    )


def threshold_for_min_specificity(
    y_true: np.ndarray, scores: np.ndarray, target: float
) -> tuple[float, dict[str, int | float]]:
    best: tuple[float, float, float] | None = None
    for threshold in np.r_[np.inf, np.sort(np.unique(scores))[::-1], -np.inf]:
        prediction = scores >= threshold
        tp = int((prediction & (y_true == 1)).sum())
        fn = int((~prediction & (y_true == 1)).sum())
        tn = int((~prediction & (y_true == 0)).sum())
        fp = int((prediction & (y_true == 0)).sum())
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        candidate = (sensitivity, specificity, float(threshold))
        if specificity >= target and (best is None or candidate > best):
            best = candidate
    if best is None:
        raise ValueError(f"no threshold reaches specificity {target}")
    sensitivity, specificity, threshold = best
    prediction = scores >= threshold
    return threshold, {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": int((prediction & (y_true == 1)).sum()),
        "fn": int((~prediction & (y_true == 1)).sum()),
        "tn": int((~prediction & (y_true == 0)).sum()),
        "fp": int((prediction & (y_true == 0)).sum()),
    }


def write_posthoc_readout(current: pd.DataFrame) -> None:
    y_true = (current["true"].to_numpy(int) == 1).astype(int)
    score_original = current["p_mel"].to_numpy(float)
    denominator = score_original + current["p_nevus"].to_numpy(float)
    score_readout = np.divide(
        score_original,
        denominator,
        out=np.zeros_like(score_original),
        where=denominator > 1e-12,
    )
    original_auc, readout_auc, p_value = paired_delong(
        y_true, score_original, score_readout
    )
    readout_auc, readout_low, readout_high = delong_ci(y_true, score_readout)
    threshold, operating = threshold_for_min_specificity(
        y_true, score_readout, 0.90
    )
    pd.DataFrame(
        [
            {
                "analysis": "posthoc two-class readout on current external file set",
                "n": len(current),
                "auc_original_p_mel": original_auc,
                "auc_readout": readout_auc,
                "auc_readout_ci95_low": readout_low,
                "auc_readout_ci95_high": readout_high,
                "paired_delong_p": p_value,
                "target_specificity": 0.90,
                "threshold_selected_on_same_external_set": threshold,
                **operating,
                "evidence_grade": "E-hypothesis-only-resubstitution-threshold",
                "warning": "threshold selected and evaluated on the same 107 labelled files; not a validation result",
            }
        ]
    ).to_csv(
        GENERATED / "external_posthoc_two_class_readout.csv",
        index=False,
        float_format="%.6f",
    )


def parse_cv_metrics() -> dict[str, tuple[float, float]]:
    text = CV_LOG.read_text(encoding="utf-8")
    result: dict[str, tuple[float, float]] = {}
    for metric in ("acc", "auc", "dice"):
        match = re.search(rf"^{metric}\s+(.+)$", text, flags=re.MULTILINE)
        if not match:
            raise ValueError(f"missing {metric} row in {CV_LOG}")
        numbers = [float(value) for value in re.findall(r"0\.\d+", match.group(1))]
        if len(numbers) != 8:
            raise ValueError(f"unexpected {metric} row in {CV_LOG}: {numbers}")
        result[metric] = (numbers[4], numbers[5])
    return result


def write_internal_table() -> pd.DataFrame:
    cv = parse_cv_metrics()
    held_out = pd.read_csv(INTERNAL_CSV).set_index("condition").loc["1a"]
    rows = [
        {
            "evaluation_id": "internal-image-cv-in-sample-representations",
            "model_scope": "single-pass 1280-d configuration",
            "protocol": "10-fold image-level CV; CNN representations partly in-sample; scaler fit globally before folds",
            "n": 25_903,
            "accuracy": cv["acc"][0],
            "accuracy_uncertainty": cv["acc"][1],
            "auc_macro_ovr": cv["auc"][0],
            "auc_macro_ovr_uncertainty": cv["auc"][1],
            "auc_melanoma_ovr": math.nan,
            "auc_melanoma_ci95_low": math.nan,
            "auc_melanoma_ci95_high": math.nan,
            "macro_dice": cv["dice"][0],
            "macro_dice_uncertainty": cv["dice"][1],
            "uncertainty_definition": "standard deviation across 10 folds",
            "evidence_grade": "D-descriptive-optimistic",
            "source": CV_LOG.relative_to(paths.ROOT).as_posix(),
            "source_sha256": file_sha256(CV_LOG),
        },
        {
            "evaluation_id": "internal-fixed-test-configuration",
            "model_scope": "heads retrained on train+valid; not exact deployed head bytes",
            "protocol": "original fixed test split; no resolved lesion/patient grouping",
            "n": int(held_out["n_eval"]),
            "accuracy": held_out["accuracy"],
            "accuracy_uncertainty": math.nan,
            "auc_macro_ovr": held_out["auc_macro_ovr"],
            "auc_macro_ovr_uncertainty": math.nan,
            "auc_melanoma_ovr": held_out["auc_melanoma_ovr"],
            "auc_melanoma_ci95_low": held_out["auc_melanoma_ci95_low"],
            "auc_melanoma_ci95_high": held_out["auc_melanoma_ci95_high"],
            "macro_dice": held_out["macro_dice"],
            "macro_dice_uncertainty": math.nan,
            "uncertainty_definition": "no interval available except melanoma AUC DeLong 95% CI",
            "evidence_grade": "B-recorded-held-out-with-provenance-and-reproduction-gaps",
            "source": INTERNAL_CSV.relative_to(paths.ROOT).as_posix(),
            "source_sha256": file_sha256(INTERNAL_CSV),
        },
    ]
    table = pd.DataFrame(rows)
    table.to_csv(GENERATED / "internal_performance.csv", index=False, float_format="%.6f")
    return table


def write_performance_registry(
    internal: pd.DataFrame,
    external: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    rows: list[dict[str, object]] = []

    def add(
        *,
        result_id: str,
        model_identity: str,
        scope: str,
        metric: str,
        estimate: float,
        evidence_grade: str,
        source: str,
        uncertainty_value: float = math.nan,
        ci_low: float = math.nan,
        ci_high: float = math.nan,
        uncertainty_type: str = "none",
        note: str = "",
    ) -> None:
        rows.append(
            {
                "result_id": result_id,
                "model_identity": model_identity,
                "scope": scope,
                "metric": metric,
                "estimate": estimate,
                "uncertainty_value": uncertainty_value,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "uncertainty_type": uncertainty_type,
                "evidence_grade": evidence_grade,
                "source": source,
                "note": note,
            }
        )

    cv, held_out = internal.iloc[0], internal.iloc[1]
    for metric, value_key, uncertainty_key in (
        ("accuracy", "accuracy", "accuracy_uncertainty"),
        ("auc_macro_ovr", "auc_macro_ovr", "auc_macro_ovr_uncertainty"),
        ("macro_dice", "macro_dice", "macro_dice_uncertainty"),
    ):
        add(
            result_id=str(cv["evaluation_id"]),
            model_identity=str(cv["model_scope"]),
            scope="internal image-level CV",
            metric=metric,
            estimate=float(cv[value_key]),
            uncertainty_value=float(cv[uncertainty_key]),
            uncertainty_type="standard deviation across 10 folds",
            evidence_grade=str(cv["evidence_grade"]),
            source=str(cv["source"]),
            note="partially in-sample CNN representations; scaler fit before folds",
        )
    for metric in ("accuracy", "auc_macro_ovr", "macro_dice"):
        add(
            result_id=str(held_out["evaluation_id"]),
            model_identity=str(held_out["model_scope"]),
            scope="internal original held-out image split",
            metric=metric,
            estimate=float(held_out[metric]),
            evidence_grade=str(held_out["evidence_grade"]),
            source=str(held_out["source"]),
            note="recorded result; patient/lesion independence unresolved; current-input rerun differs slightly",
        )
    add(
        result_id=str(held_out["evaluation_id"]),
        model_identity=str(held_out["model_scope"]),
        scope="internal original held-out image split",
        metric="auc_melanoma_ovr",
        estimate=float(held_out["auc_melanoma_ovr"]),
        ci_low=float(held_out["auc_melanoma_ci95_low"]),
        ci_high=float(held_out["auc_melanoma_ci95_high"]),
        uncertainty_type="DeLong 95% CI",
        evidence_grade=str(held_out["evidence_grade"]),
        source=str(held_out["source"]),
        note="recorded result; patient/lesion independence unresolved; current-input rerun differs slightly",
    )

    external_metric_specs = (
        (
            "auc_melanoma",
            "auc_melanoma_ci95_low",
            "auc_melanoma_ci95_high",
            "DeLong 95% CI",
        ),
        ("average_precision_melanoma", None, None, "none"),
        ("brier_melanoma", None, None, "none"),
        (
            "sensitivity_melanoma_at_0.188",
            "sensitivity_melanoma_ci95_low",
            "sensitivity_melanoma_ci95_high",
            "Wilson 95% CI",
        ),
        (
            "specificity_melanoma_at_0.188",
            "specificity_melanoma_ci95_low",
            "specificity_melanoma_ci95_high",
            "Wilson 95% CI",
        ),
        (
            "not_benign_sensitivity",
            "not_benign_sensitivity_ci95_low",
            "not_benign_sensitivity_ci95_high",
            "Wilson 95% CI",
        ),
        (
            "not_benign_specificity",
            "not_benign_specificity_ci95_low",
            "not_benign_specificity_ci95_high",
            "Wilson 95% CI",
        ),
        ("argmax_exact_accuracy", None, None, "none"),
    )
    value_alias = {
        "sensitivity_melanoma_at_0.188": "sensitivity_melanoma",
        "specificity_melanoma_at_0.188": "specificity_melanoma",
    }
    for _, row in external.iterrows():
        grade = (
            "C-file-level-external-exact-bundle"
            if row["status"] == "current-exploratory-external"
            else "D-archived-file-level-external"
        )
        for metric, low_key, high_key, uncertainty in external_metric_specs:
            value_key = value_alias.get(metric, metric)
            add(
                result_id=f"external-{row['status']}",
                model_identity=str(row["model_id"]),
                scope="technical external file-level analysis; 107 labelled files; 19 melanoma-labelled files",
                metric=metric,
                estimate=float(row[value_key]),
                ci_low=float(row[low_key]) if low_key else math.nan,
                ci_high=float(row[high_key]) if high_key else math.nan,
                uncertainty_type=uncertainty,
                evidence_grade=grade,
                source=str(row["source"]),
                note="per-file analysis; near-duplicates, clustering and reference-standard provenance unresolved",
            )

    delta = comparison.iloc[0]
    add(
        result_id="paired-singlepass-minus-tta8",
        model_identity="paired current single-pass versus archived TTA8",
        scope="same 107 external labelled files",
        metric="delta_auc_melanoma",
        estimate=float(delta["delta_auc_observed"]),
        ci_low=float(delta["delta_auc_bootstrap_ci95_low"]),
        ci_high=float(delta["delta_auc_bootstrap_ci95_high"]),
        uncertainty_type="paired stratified bootstrap 95% CI",
        evidence_grade="C-small-paired-external",
        source=paths.relative(paths.PAPER_EXTERNAL_PAIRED),
        note=f"paired DeLong p={delta['paired_delong_p']:.6f}; no superiority demonstrated",
    )
    pd.DataFrame(rows).to_csv(
        GENERATED / "performance_registry.csv",
        index=False,
        float_format="%.6f",
    )


def plot_external_roc(current: pd.DataFrame, tta8: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(6.4, 5.2))
    colours = {"Single-pass 1.0.0": "#0072B2", "Archived TTA×8": "#D55E00"}
    for label, data in (("Single-pass 1.0.0", current), ("Archived TTA×8", tta8)):
        y_true = (data["true"].to_numpy(int) == 1).astype(int)
        scores = data["p_mel"].to_numpy(float)
        auc, low, high = delong_ci(y_true, scores)
        fpr, tpr, _ = roc_curve(y_true, scores)
        axis.plot(
            fpr,
            tpr,
            lw=2.2,
            color=colours[label],
            label=f"{label}: AUC {auc:.3f} (95% CI {low:.3f}–{high:.3f})",
        )
    axis.plot([0, 1], [0, 1], "--", color="#777777", lw=1)
    axis.set(
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        title="External melanoma discrimination",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="lower right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    save_figure(fig, "external_roc_singlepass_vs_tta8")
    plt.close(fig)


def plot_external_operating_points(metrics: pd.DataFrame) -> None:
    labels = ["Single-pass 1.0.0", "Archived TTA×8"]
    rows = [metrics.iloc[0], metrics.iloc[1]]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
    panels = (
        (
            "Melanoma score at T=0.188",
            "sensitivity_melanoma",
            "specificity_melanoma",
            "sensitivity_melanoma_ci95_low",
            "sensitivity_melanoma_ci95_high",
            "specificity_melanoma_ci95_low",
            "specificity_melanoma_ci95_high",
        ),
        (
            "Composite not-benign rule",
            "not_benign_sensitivity",
            "not_benign_specificity",
            "not_benign_sensitivity_ci95_low",
            "not_benign_sensitivity_ci95_high",
            "not_benign_specificity_ci95_low",
            "not_benign_specificity_ci95_high",
        ),
    )
    x = np.arange(len(labels))
    width = 0.34
    for axis, panel in zip(axes, panels):
        title, sens_key, spec_key, sens_low, sens_high, spec_low, spec_high = panel
        sensitivity = np.array([row[sens_key] for row in rows], dtype=float)
        specificity = np.array([row[spec_key] for row in rows], dtype=float)
        sens_error = np.array(
            [
                sensitivity - np.array([row[sens_low] for row in rows]),
                np.array([row[sens_high] for row in rows]) - sensitivity,
            ]
        )
        spec_error = np.array(
            [
                specificity - np.array([row[spec_low] for row in rows]),
                np.array([row[spec_high] for row in rows]) - specificity,
            ]
        )
        axis.bar(
            x - width / 2,
            sensitivity,
            width,
            yerr=sens_error,
            capsize=3,
            label="Sensitivity",
            color="#0072B2",
        )
        axis.bar(
            x + width / 2,
            specificity,
            width,
            yerr=spec_error,
            capsize=3,
            label="Specificity",
            color="#E69F00",
        )
        axis.set_title(title, fontsize=10)
        axis.set_xticks(x, labels, rotation=15, ha="right")
        axis.set_ylim(0, 1.08)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Proportion (Wilson 95% CI)")
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "external_operating_points")
    plt.close(fig)


def plot_internal_gap(internal: pd.DataFrame) -> None:
    metrics = ["accuracy", "auc_macro_ovr", "macro_dice"]
    names = ["Accuracy", "Macro AUC", "Macro Dice"]
    cv = internal.iloc[0][metrics].to_numpy(float)
    held_out = internal.iloc[1][metrics].to_numpy(float)
    x = np.arange(len(metrics))
    width = 0.34
    fig, axis = plt.subplots(figsize=(7.1, 4.5))
    axis.bar(
        x - width / 2,
        cv,
        width,
        label="Image-level CV (optimistic)",
        color="#56B4E9",
    )
    axis.bar(
        x + width / 2,
        held_out,
        width,
        label="Original held-out test split",
        color="#009E73",
    )
    axis.set_xticks(x, names)
    axis.set_ylim(0.70, 1.01)
    axis.set_ylabel("Score")
    axis.set_title("Internal evaluation gap across protocols")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, fontsize=9)
    for index, values in enumerate((cv, held_out)):
        offset = -width / 2 if index == 0 else width / 2
        for position, value in zip(x + offset, values):
            axis.text(position, value + 0.008, f"{value:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    save_figure(fig, "internal_protocol_gap")
    plt.close(fig)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.svg", metadata={"Date": None})
    fig.savefig(
        FIGURES / f"{stem}.png",
        dpi=300,
        metadata={"Software": "SkinFusionNet paper resource generator"},
    )


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    current = pd.read_csv(CURRENT_CSV)
    tta8 = pd.read_csv(TTA8_CSV)
    key = ["file", "folder", "true"]
    merged = current.merge(
        tta8,
        on=key,
        suffixes=("_singlepass", "_tta8"),
        validate="one_to_one",
    )
    if len(merged) != len(current) or len(merged) != len(tta8):
        raise ValueError("external single-pass and TTA8 cases are not aligned")

    external = pd.DataFrame(
        [
            external_metrics(
                current,
                model_id=CURRENT_ID,
                bundle_sha256=CURRENT_BUNDLE_SHA,
                status="current-exploratory-external",
                source=CURRENT_CSV,
            ),
            external_metrics(
                tta8,
                model_id=TTA8_ID,
                bundle_sha256=TTA8_RETROSPECTIVE_SHA,
                status="archived-exploratory-external",
                source=TTA8_CSV,
            ),
        ]
    )
    external.to_csv(
        GENERATED / "external_performance.csv", index=False, float_format="%.6f"
    )
    write_posthoc_readout(current)

    y_true = (merged["true"].to_numpy(int) == 1).astype(int)
    auc_old, auc_new, p_value = paired_delong(
        y_true,
        merged["p_mel_tta8"].to_numpy(float),
        merged["p_mel_singlepass"].to_numpy(float),
    )
    delta_mean, delta_low, delta_high = paired_bootstrap_delta(merged)
    comparison = pd.DataFrame(
        [
            {
                "comparison": "single-pass minus archived TTA8",
                "n_paired": len(merged),
                "auc_tta8": auc_old,
                "auc_singlepass": auc_new,
                "delta_auc_observed": auc_new - auc_old,
                "delta_auc_bootstrap_mean": delta_mean,
                "delta_auc_bootstrap_ci95_low": delta_low,
                "delta_auc_bootstrap_ci95_high": delta_high,
                "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_design": "paired stratified resampling within class",
                "paired_delong_p": p_value,
                "interpretation": "no evidence of superiority; exploratory small-n paired comparison",
            }
        ]
    )
    comparison.to_csv(
        GENERATED / "external_paired_comparison.csv",
        index=False,
        float_format="%.6f",
    )

    internal = write_internal_table()
    write_performance_registry(internal, external, comparison)
    plot_external_roc(current, tta8)
    plot_external_operating_points(external)
    plot_internal_gap(internal)
    print(f"wrote paper resources -> {OUT}")


if __name__ == "__main__":
    main()
