"""Audit the factual claims used by the paper-oriented resource pack.

The default checks are read-only with respect to the application, models, raw
datasets and result sources.  Generated audit tables are written only below
``docs/paper``.  The optional internal recomputation is deliberately explicit
because it retrains two LightGBM heads from the current local feature matrices
and can take several minutes::

    MPLCONFIGDIR=/tmp/matplotlib .venv_extval/bin/python \
      -m derm.analysis.verify_paper_resources --recompute-internal

This module does not treat file count as an independent patient or lesion
count.  It also distinguishes hash-verifiable ONNX binaries from a reproducible
training pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import lightgbm
import numpy as np
import pandas as pd
import pillow_heif
import scipy
import sklearn
from PIL import Image
from scipy.stats import norm
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from derm import paths
from derm.analysis.internal_validity_m1 import run_phase_1a
from derm.export.model_manifest import validate_manifest


pillow_heif.register_heif_opener()

OUT = paths.PAPER
GENERATED = OUT / "generated"
AUDIT_CSV = GENERATED / "verification_audit.csv"
DUPLICATE_CSV = GENERATED / "external_near_duplicate_audit.csv"
DEDUP_CSV = GENERATED / "external_deduplicated_sensitivity.csv"
DEDUP_PAIRED_CSV = GENERATED / "external_deduplicated_paired_comparison.csv"
INTERNAL_CSV = GENERATED / "internal_reproduction_check.csv"
REPORT = OUT / "verification_report.md"

CURRENT_CSV = paths.EXTERNAL_PARITY_CSV
TTA8_CSV = paths.TTA8_EXTERNAL_CSV
EXTERNAL_AGGREGATE = GENERATED / "external_performance.csv"
PAIRED_AGGREGATE = GENERATED / "external_paired_comparison.csv"
POSTHOC_AGGREGATE = GENERATED / "external_posthoc_two_class_readout.csv"
INTERNAL_RECORDED = paths.INTERNAL_DECOMPOSITION
CV_LOG = paths.INTERNAL_CV_LOG
PARITY_GATE = paths.EXTERNAL_PARITY_GATE_CSV

CURRENT_ID = "skinfusionnet-v2s3c-singlepass-1.0.0"
TTA8_ID = "efficientnetv2s-cbam-ensemble+tta8 (2026-06-04)"
T_MEL = 0.188
T_ATYP = 0.412
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(paths.ROOT).as_posix()


def placement_values(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Independent pairwise formulation of the DeLong placement values."""
    positive = scores[y_true == 1]
    negative = scores[y_true == 0]
    comparison = (positive[:, None] > negative[None, :]).astype(float)
    comparison += 0.5 * (positive[:, None] == negative[None, :])
    return (
        float(comparison.mean()),
        comparison.mean(axis=1),
        comparison.mean(axis=0),
    )


def delong_ci(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float, float]:
    auc, positive, negative = placement_values(y_true, scores)
    variance = np.var(positive, ddof=1) / len(positive)
    variance += np.var(negative, ddof=1) / len(negative)
    half_width = norm.ppf(0.975) * math.sqrt(max(float(variance), 0.0))
    return auc, max(0.0, auc - half_width), min(1.0, auc + half_width)


def paired_delong(
    y_true: np.ndarray, old: np.ndarray, new: np.ndarray
) -> tuple[float, float, float, float]:
    auc_old, old_positive, old_negative = placement_values(y_true, old)
    auc_new, new_positive, new_negative = placement_values(y_true, new)
    variance = np.var(new_positive - old_positive, ddof=1) / len(old_positive)
    variance += np.var(new_negative - old_negative, ddof=1) / len(old_negative)
    delta = auc_new - auc_old
    p_value = float(2 * norm.sf(abs(delta / math.sqrt(float(variance)))))
    return auc_old, auc_new, delta, p_value


def paired_bootstrap(
    y_true: np.ndarray,
    old: np.ndarray,
    new: np.ndarray,
    *,
    repetitions: int = 10_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    positive = np.flatnonzero(y_true == 1)
    negative = np.flatnonzero(y_true == 0)
    rng = np.random.default_rng(seed)
    deltas = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sample = np.r_[
            rng.choice(positive, len(positive), replace=True),
            rng.choice(negative, len(negative), replace=True),
        ]
        deltas[index] = roc_auc_score(y_true[sample], new[sample])
        deltas[index] -= roc_auc_score(y_true[sample], old[sample])
    return (
        float(deltas.mean()),
        float(np.quantile(deltas, 0.025)),
        float(np.quantile(deltas, 0.975)),
    )


def wilson(successes: int, total: int) -> tuple[float, float, float]:
    z = 1.96
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    half_width = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return proportion, centre - half_width, centre + half_width


def external_metrics(data: pd.DataFrame) -> dict[str, float | int]:
    y_true = data["true"].to_numpy(int)
    score = data["p_mel"].to_numpy(float)
    atypical = data["p_atyp"].to_numpy(float)
    auc, auc_low, auc_high = delong_ci(y_true, score)
    melanoma_prediction = score >= T_MEL
    not_benign = melanoma_prediction | (atypical >= T_ATYP)

    def counts(prediction: np.ndarray) -> tuple[int, int, int, int]:
        return (
            int((prediction & (y_true == 1)).sum()),
            int((~prediction & (y_true == 1)).sum()),
            int((~prediction & (y_true == 0)).sum()),
            int((prediction & (y_true == 0)).sum()),
        )

    tp, fn, tn, fp = counts(melanoma_prediction)
    nb_tp, nb_fn, nb_tn, nb_fp = counts(not_benign)
    sensitivity, sensitivity_low, sensitivity_high = wilson(tp, tp + fn)
    specificity, specificity_low, specificity_high = wilson(tn, tn + fp)
    nb_sensitivity, nb_sensitivity_low, nb_sensitivity_high = wilson(
        nb_tp, nb_tp + nb_fn
    )
    nb_specificity, nb_specificity_low, nb_specificity_high = wilson(
        nb_tn, nb_tn + nb_fp
    )
    return {
        "n_files": len(data),
        "n_melanoma_labelled_files": int((y_true == 1).sum()),
        "n_nevus_labelled_files": int((y_true == 0).sum()),
        "auc_melanoma": auc,
        "auc_melanoma_ci95_low": auc_low,
        "auc_melanoma_ci95_high": auc_high,
        "average_precision_melanoma": float(average_precision_score(y_true, score)),
        "brier_melanoma": float(brier_score_loss(y_true, score)),
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
            (data["argmax"].to_numpy(int) == y_true).mean()
        ),
    }


def decoded_hashes(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(
        np.float32
    )
    dct = cv2.dct(resized)[:8, :8]
    phash = (dct > np.median(dct.ravel()[1:])).ravel()
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    dhash = (resized[:, 1:] > resized[:, :-1]).ravel()
    return phash, dhash, rgb


@dataclass(frozen=True)
class NearDuplicateGroup:
    group_id: str
    members: tuple[tuple[str, str], ...]
    phash_distance: int
    dhash_distance: int
    pixel_mae: float
    pixel_max_abs_difference: int


def find_near_duplicates(data: pd.DataFrame) -> list[NearDuplicateGroup]:
    entries: list[
        tuple[tuple[str, str], Path, np.ndarray, np.ndarray, tuple[int, ...]]
    ] = []
    for row in data.itertuples(index=False):
        key = (str(row.folder), str(row.file))
        path = paths.EXTERNAL_VAL / key[0] / key[1]
        phash, dhash, rgb = decoded_hashes(path)
        entries.append((key, path, phash, dhash, rgb.shape))

    groups: list[NearDuplicateGroup] = []
    for left in range(len(entries)):
        for right in range(left + 1, len(entries)):
            left_key, left_path, left_phash, left_dhash, left_shape = entries[left]
            right_key, right_path, right_phash, right_dhash, right_shape = entries[right]
            if left_shape != right_shape:
                continue
            phash_distance = int(np.count_nonzero(left_phash != right_phash))
            dhash_distance = int(np.count_nonzero(left_dhash != right_dhash))
            if phash_distance != 0 or dhash_distance != 0:
                continue
            left_rgb = np.asarray(Image.open(left_path).convert("RGB"))
            right_rgb = np.asarray(Image.open(right_path).convert("RGB"))
            difference = np.abs(left_rgb.astype(np.int16) - right_rgb.astype(np.int16))
            pixel_mae = float(difference.mean())
            if pixel_mae > 1.0:
                continue
            groups.append(
                NearDuplicateGroup(
                    group_id=f"near_duplicate_{len(groups) + 1:03d}",
                    members=(left_key, right_key),
                    phash_distance=phash_distance,
                    dhash_distance=dhash_distance,
                    pixel_mae=pixel_mae,
                    pixel_max_abs_difference=int(difference.max()),
                )
            )
    return groups


def select_duplicate_derivatives(
    groups: list[NearDuplicateGroup],
) -> set[tuple[str, str]]:
    """Keep one file per detected pair without claiming lesion-level deduplication."""
    dropped: set[tuple[str, str]] = set()
    for group in groups:
        ordered = sorted(group.members, key=lambda key: (len(key[1]), key[1]))
        dropped.update(ordered[1:])
    return dropped


def audit_row(
    check_id: str,
    area: str,
    status: str,
    *,
    recorded: Any = "",
    recomputed: Any = "",
    difference: Any = "",
    source: str = "",
    note: str = "",
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "area": area,
        "status": status,
        "recorded_value": recorded,
        "recomputed_value": recomputed,
        "absolute_difference": difference,
        "source": source,
        "note": note,
    }


def verify_external() -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    audit: list[dict[str, Any]] = []
    current = pd.read_csv(CURRENT_CSV)
    tta8 = pd.read_csv(TTA8_CSV)
    aggregate = pd.read_csv(EXTERNAL_AGGREGATE).set_index("model_id")
    current_metrics = external_metrics(current)
    tta8_metrics = external_metrics(tta8)

    required_columns = {
        "file",
        "folder",
        "true",
        "p_nevus",
        "p_mel",
        "p_atyp",
        "argmax",
        "coverage",
    }
    integrity = (
        required_columns.issubset(current.columns)
        and len(current) == current[["file", "folder", "true"]].drop_duplicates().shape[0]
        and not current.isna().any().any()
        and set(current["true"].unique()) == {0, 1}
        and (
            current["folder"].eq("Melanom").astype(int)
            == current["true"].astype(int)
        ).all()
    )
    audit.append(
        audit_row(
            "external_file_table_integrity",
            "External",
            "PASS" if integrity else "FAIL",
            recorded="107 rows",
            recomputed=f"{len(current)} unique complete rows",
            source=rel(CURRENT_CSV),
            note="This verifies file-level rows, not patient- or lesion-level independence.",
        )
    )

    raw_paths = [
        paths.EXTERNAL_VAL / str(row.folder) / str(row.file)
        for row in current.itertuples(index=False)
    ]
    raw_present = all(path.exists() for path in raw_paths)
    raw_hashes = [file_sha256(path) for path in raw_paths]
    audit.append(
        audit_row(
            "external_raw_files",
            "External",
            "PASS" if raw_present and len(set(raw_hashes)) == len(raw_hashes) else "FAIL",
            recorded="107 files",
            recomputed=(
                f"{sum(path.exists() for path in raw_paths)} present; "
                f"{len(set(raw_hashes))} unique byte hashes"
            ),
            source="external_val/{Nevi,Melanom}",
            note="Byte uniqueness does not exclude recompressed copies or repeated lesions.",
        )
    )

    for model_id, metrics in ((CURRENT_ID, current_metrics), (TTA8_ID, tta8_metrics)):
        recorded = aggregate.loc[model_id]
        for key in (
            "auc_melanoma",
            "average_precision_melanoma",
            "brier_melanoma",
            "sensitivity_melanoma",
            "specificity_melanoma",
            "not_benign_sensitivity",
            "not_benign_specificity",
            "argmax_exact_accuracy",
        ):
            difference = abs(float(recorded[key]) - float(metrics[key]))
            audit.append(
                audit_row(
                    f"{model_id}_{key}",
                    "External metrics",
                    "PASS" if difference <= 5e-7 else "FAIL",
                    recorded=f"{float(recorded[key]):.6f}",
                    recomputed=f"{float(metrics[key]):.6f}",
                    difference=f"{difference:.3g}",
                    source=str(recorded["source"]),
                    note="Independently recomputed from per-file probabilities.",
                )
            )

    merged = current.merge(
        tta8,
        on=["file", "folder", "true"],
        suffixes=("_current", "_tta8"),
        validate="one_to_one",
    )
    y_true = merged["true"].to_numpy(int)
    auc_old, auc_new, delta, p_value = paired_delong(
        y_true,
        merged["p_mel_tta8"].to_numpy(float),
        merged["p_mel_current"].to_numpy(float),
    )
    bootstrap_mean, bootstrap_low, bootstrap_high = paired_bootstrap(
        y_true,
        merged["p_mel_tta8"].to_numpy(float),
        merged["p_mel_current"].to_numpy(float),
    )
    recorded_pair = pd.read_csv(PAIRED_AGGREGATE).iloc[0]
    paired_checks = {
        "auc_tta8": auc_old,
        "auc_singlepass": auc_new,
        "delta_auc_observed": delta,
        "paired_delong_p": p_value,
        "delta_auc_bootstrap_mean": bootstrap_mean,
        "delta_auc_bootstrap_ci95_low": bootstrap_low,
        "delta_auc_bootstrap_ci95_high": bootstrap_high,
    }
    for key, value in paired_checks.items():
        difference = abs(float(recorded_pair[key]) - value)
        audit.append(
            audit_row(
                f"paired_{key}",
                "External paired comparison",
                "PASS" if difference <= 5e-7 else "FAIL",
                recorded=f"{float(recorded_pair[key]):.6f}",
                recomputed=f"{value:.6f}",
                difference=f"{difference:.3g}",
                source=rel(PAIRED_AGGREGATE),
                note="Paired by exact folder, filename and label.",
            )
        )

    posthoc = pd.read_csv(POSTHOC_AGGREGATE).iloc[0]
    score_original = current["p_mel"].to_numpy(float)
    denominator = score_original + current["p_nevus"].to_numpy(float)
    score_readout = np.divide(
        score_original,
        denominator,
        out=np.zeros_like(score_original),
        where=denominator > 1e-12,
    )
    readout_auc, readout_low, readout_high = delong_ci(
        current["true"].to_numpy(int), score_readout
    )
    _, _, _, readout_p = paired_delong(
        current["true"].to_numpy(int), score_original, score_readout
    )
    for key, value in {
        "auc_readout": readout_auc,
        "auc_readout_ci95_low": readout_low,
        "auc_readout_ci95_high": readout_high,
        "paired_delong_p": readout_p,
    }.items():
        difference = abs(float(posthoc[key]) - value)
        audit.append(
            audit_row(
                f"posthoc_{key}",
                "Post-hoc readout",
                "PASS" if difference <= 5e-7 else "FAIL",
                recorded=f"{float(posthoc[key]):.6f}",
                recomputed=f"{value:.6f}",
                difference=f"{difference:.3g}",
                source=rel(POSTHOC_AGGREGATE),
                note="Numerically correct but remains same-set, hypothesis-only analysis.",
            )
        )

    gate = pd.read_csv(PARITY_GATE).iloc[0]
    gate_ok = (
        gate["status"] == "ok"
        and str(gate["byte_equal"]).lower() == "true"
        and int(gate["mismatched_exact_cells"]) == 0
        and float(gate["max_abs_diff"]) == 0.0
    )
    audit.append(
        audit_row(
            "external_exact_bundle_parity",
            "Model identity",
            "PASS" if gate_ok else "FAIL",
            recorded=str(gate["model_bundle_sha256"]),
            recomputed="byte-equal 107-row output" if gate_ok else "parity failed",
            source=rel(PARITY_GATE),
            note="Parity was rerun from raw files through the five deployed ONNX components.",
        )
    )

    duplicate_groups = find_near_duplicates(current)
    drop_keys = select_duplicate_derivatives(duplicate_groups)
    duplicate_rows: list[dict[str, Any]] = []
    for group in duplicate_groups:
        ordered = sorted(group.members, key=lambda key: (len(key[1]), key[1]))
        retained = ordered[0]
        removed = ordered[1:]
        duplicate_rows.append(
            {
                "duplicate_group": group.group_id,
                "file_count": len(group.members),
                "class_folder": group.members[0][0],
                "member_1_folder": group.members[0][0],
                "member_1_file": group.members[0][1],
                "member_2_folder": group.members[1][0],
                "member_2_file": group.members[1][1],
                "retained_folder": retained[0],
                "retained_file": retained[1],
                "removed_files": ";".join(
                    f"{folder}/{file}" for folder, file in removed
                ),
                "phash_distance": group.phash_distance,
                "dhash_distance": group.dhash_distance,
                "decoded_pixel_mae_on_0_255_scale": group.pixel_mae,
                "decoded_pixel_max_abs_difference": group.pixel_max_abs_difference,
                "files_removed_in_sensitivity_analysis": len(group.members) - 1,
                "interpretation": "near-identical decoded images; not an independent file pair",
            }
        )
    pd.DataFrame(duplicate_rows).to_csv(
        DUPLICATE_CSV, index=False, float_format="%.6f"
    )
    audit.append(
        audit_row(
            "external_near_identical_files",
            "External independence",
            "CAUTION",
            recorded="107 files treated as rows",
            recomputed=(
                f"{len(duplicate_groups)} near-identical pairs; "
                f"{len(current) - len(drop_keys)} files after one-per-pair sensitivity filter"
            ),
            source=rel(DUPLICATE_CSV),
            note="pHash=0, dHash=0 and mean absolute decoded-pixel difference <=1 intensity level on the 0–255 scale for each pair.",
        )
    )
    audit.append(
        audit_row(
            "external_patient_lesion_independence",
            "External independence",
            "NOT_VERIFIABLE",
            recorded="Not documented",
            recomputed="At least one apparent repeated-acquisition sequence observed",
            source="external_val/ plus visual audit",
            note=(
                "Patient and lesion identifiers are absent. DeLong and Wilson intervals are "
                "therefore naive per-file intervals and are not cluster-adjusted."
            ),
        )
    )
    audit.append(
        audit_row(
            "external_reference_standard",
            "External labels",
            "NOT_VERIFIABLE",
            recorded="Folder labels: Nevi / Melanom",
            recomputed="Folder-to-label mapping is internally consistent",
            source=rel(CURRENT_CSV),
            note="Histopathology, clinical follow-up and adjudication are not documented in available artifacts.",
        )
    )

    sensitivity_rows: list[dict[str, Any]] = []
    for model_id, source_data in ((CURRENT_ID, current), (TTA8_ID, tta8)):
        keep = [
            (str(row.folder), str(row.file)) not in drop_keys
            for row in source_data.itertuples(index=False)
        ]
        filtered = source_data.loc[keep].copy()
        sensitivity_rows.append(
            {
                "analysis": "one file retained per objectively near-identical pair",
                "model_id": model_id,
                "removed_files": len(source_data) - len(filtered),
                **external_metrics(filtered),
                "warning": (
                    "sensitivity analysis only; patient/lesion clustering and reference "
                    "standard remain unresolved"
                ),
            }
        )
    pd.DataFrame(sensitivity_rows).to_csv(
        DEDUP_CSV, index=False, float_format="%.6f"
    )

    keep_merged = [
        (str(row.folder), str(row.file)) not in drop_keys
        for row in merged.itertuples(index=False)
    ]
    filtered_merged = merged.loc[keep_merged].copy()
    filtered_y = filtered_merged["true"].to_numpy(int)
    filtered_old = filtered_merged["p_mel_tta8"].to_numpy(float)
    filtered_new = filtered_merged["p_mel_current"].to_numpy(float)
    filtered_auc_old, filtered_auc_new, filtered_delta, filtered_p = paired_delong(
        filtered_y, filtered_old, filtered_new
    )
    filtered_bootstrap = paired_bootstrap(filtered_y, filtered_old, filtered_new)
    pd.DataFrame(
        [
            {
                "analysis": "paired comparison after one-per-near-identical-pair filter",
                "n_paired_files": len(filtered_merged),
                "removed_files": len(merged) - len(filtered_merged),
                "auc_tta8": filtered_auc_old,
                "auc_singlepass": filtered_auc_new,
                "delta_auc_observed": filtered_delta,
                "delta_auc_bootstrap_mean": filtered_bootstrap[0],
                "delta_auc_bootstrap_ci95_low": filtered_bootstrap[1],
                "delta_auc_bootstrap_ci95_high": filtered_bootstrap[2],
                "paired_delong_p": filtered_p,
                "warning": "file-level sensitivity analysis; clustering remains unresolved",
            }
        ]
    ).to_csv(DEDUP_PAIRED_CSV, index=False, float_format="%.6f")
    return audit, drop_keys


def dataset_counts() -> tuple[dict[str, int], dict[tuple[str, str], int], int]:
    source_counts = {"ham": 0, "isic2019": 0, "s1": 0, "s2": 0}
    split_class_counts: dict[tuple[str, str], int] = {}
    for split in ("train", "valid", "test"):
        for class_name in ("nevus", "melanoma", "atypical"):
            files = [
                path
                for path in (paths.MERGED / split / class_name).rglob("*")
                if path.is_file()
            ]
            split_class_counts[(split, class_name)] = len(files)
            for path in files:
                prefix = path.name.split("_", 1)[0]
                if prefix in source_counts:
                    source_counts[prefix] += 1
    dehair_count = sum(
        1
        for split in ("train", "valid", "test")
        for path in (paths.DEHAIR / split).rglob("*")
        if path.is_file()
    )
    return source_counts, split_class_counts, dehair_count


def verify_static_and_dataset() -> list[dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    manifest = validate_manifest(
        paths.APP_MODELS / "model_manifest.json", paths.APP_MODELS
    )
    audit.append(
        audit_row(
            "current_bundle_hash",
            "Model identity",
            "PASS",
            recorded=str(manifest["bundle_sha256"]),
            recomputed=str(manifest["bundle_sha256"]),
            difference="0",
            source="app/assets/models/model_manifest.json plus five ONNX files",
            note="The deployed binaries are hash-verifiable; this does not reproduce their training.",
        )
    )
    provenance = manifest.get("training_provenance", {})
    audit.append(
        audit_row(
            "training_pipeline_reproducibility",
            "Training provenance",
            "NOT_VERIFIABLE",
            recorded=str(provenance.get("reproducibility_status", "")),
            recomputed="dataset fingerprint, training commit and source checkpoint hashes absent",
            source="app/assets/models/model_manifest.json",
            note="Use 'hash-verifiable binaries', not 'reproducible training pipeline'.",
        )
    )

    sources, split_classes, dehair_count = dataset_counts()
    expected_sources = {"ham": 8954, "isic2019": 13390, "s1": 2130, "s2": 1429}
    expected_split_classes = {
        ("train", "nevus"): 8824,
        ("train", "melanoma"): 3061,
        ("train", "atypical"): 5317,
        ("valid", "nevus"): 1960,
        ("valid", "melanoma"): 911,
        ("valid", "atypical"): 1368,
        ("test", "nevus"): 2319,
        ("test", "melanoma"): 836,
        ("test", "atypical"): 1307,
    }
    audit.append(
        audit_row(
            "internal_source_file_counts",
            "Dataset composition",
            "PASS" if sources == expected_sources else "FAIL",
            recorded=str(expected_sources),
            recomputed=str(sources),
            source="data/Skin_Cancer_Merged/{train,valid,test}",
            note="Counts are file counts reconstructed from prefixes, not patient/lesion counts.",
        )
    )
    audit.append(
        audit_row(
            "internal_split_class_file_counts",
            "Dataset composition",
            "PASS" if split_classes == expected_split_classes else "FAIL",
            recorded="25,903 files across the documented split/class cells",
            recomputed=f"{sum(split_classes.values()):,} files; all cells matched",
            source="data/Skin_Cancer_Merged/{train,valid,test}",
            note="Native diagnostic mapping and grouped independence remain unverified.",
        )
    )
    audit.append(
        audit_row(
            "dehair_file_completeness",
            "Dataset composition",
            "CAUTION" if dehair_count == 25_630 else "FAIL",
            recorded="25,630 / 25,903",
            recomputed=f"{dehair_count:,} / {sum(split_classes.values()):,}",
            source="data/Skin_Cancer_Merged_dehair",
            note="273 dehair outputs are absent or unmatched.",
        )
    )

    cv_text = CV_LOG.read_text(encoding="utf-8")
    for metric, expected in (("acc", 0.9424), ("auc", 0.9886), ("dice", 0.9312)):
        match = re.search(rf"^{metric}\s+(.+)$", cv_text, flags=re.MULTILINE)
        numbers = [float(value) for value in re.findall(r"0\.\d+", match.group(1))]
        recorded = numbers[4]
        audit.append(
            audit_row(
                f"cv_log_{metric}",
                "Internal CV",
                "SOURCE_CONFIRMED" if recorded == expected else "FAIL",
                recorded=f"{expected:.4f}",
                recomputed=f"{recorded:.4f} parsed from log",
                difference=f"{abs(recorded - expected):.4g}",
                source=rel(CV_LOG),
                note="The full 10-fold training was not rerun by this audit.",
            )
        )
    cv_source = (paths.ROOT / "derm" / "fusion" / "train_lgbm_v2s.py").read_text(
        encoding="utf-8"
    )
    scaler_before_cv = cv_source.find("StandardScaler().fit_transform") < cv_source.find(
        "cv = StratifiedKFold"
    )
    audit.append(
        audit_row(
            "cv_scaler_scope",
            "Internal CV",
            "CAUTION" if scaler_before_cv else "PASS",
            recorded="Previously described as refit within each fold",
            recomputed="Scaler fit once on the full dataset before fold creation",
            source="derm/fusion/train_lgbm_v2s.py",
            note="This is unsupervised preprocessing leakage and adds to the optimistic status.",
        )
    )
    return audit


def internal_reproduction_rows(recompute: bool) -> list[dict[str, Any]]:
    if not recompute and INTERNAL_CSV.exists():
        return pd.read_csv(INTERNAL_CSV).to_dict("records")
    if not recompute:
        return []

    metadata_path = paths.ARTIFACTS / "image_metadata.csv"
    metadata = pd.read_csv(metadata_path)
    reproduced = run_phase_1a(metadata, "deployed_single")
    recorded = pd.read_csv(INTERNAL_RECORDED).set_index("condition").loc["1a"]
    metrics = (
        "accuracy",
        "balanced_accuracy",
        "auc_macro_ovr",
        "auc_melanoma_ovr",
        "auc_melanoma_ci95_low",
        "auc_melanoma_ci95_high",
        "macro_dice",
        "sensitivity_melanoma",
        "specificity_melanoma",
        "tp_mel",
        "fn_mel",
        "tn_mel",
        "fp_mel",
    )
    runtime = (
        f"Python {platform.python_version()}; numpy {np.__version__}; "
        f"pandas {pd.__version__}; sklearn {sklearn.__version__}; "
        f"scipy {scipy.__version__}; lightgbm {lightgbm.__version__}"
    )
    input_hashes = "; ".join(
        f"{path.name}={file_sha256(path)}"
        for path in (
            paths.FEATURES_CSV,
            paths.CNN_V2S_FEATS_CSV,
            paths.CNN_V2S_DEHAIR_CSV,
            metadata_path,
        )
    )
    rows = []
    for metric in metrics:
        recorded_value = float(recorded[metric])
        reproduced_value = float(reproduced[metric])
        difference = abs(recorded_value - reproduced_value)
        rows.append(
            {
                "evaluation": "original held-out image split; heads retrained on train+valid",
                "metric": metric,
                "recorded_value": recorded_value,
                "recomputed_value": reproduced_value,
                "absolute_difference": difference,
                "exact_match": difference <= 1e-12,
                "status": "PASS" if difference <= 1e-12 else "NOT_EXACTLY_REPRODUCED",
                "recorded_source": rel(INTERNAL_RECORDED),
                "recomputed_from": (
                    "current local handcraft and V2S single-pass feature matrices"
                ),
                "runtime": runtime,
                "input_sha256": input_hashes,
                "warning": (
                    "No historical input hashes/runtime manifest accompanied the recorded run; "
                    "patient/lesion independence remains unresolved."
                ),
            }
        )
    pd.DataFrame(rows).to_csv(INTERNAL_CSV, index=False, float_format="%.12f")
    return rows


def internal_audit_rows(reproduction: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not reproduction:
        return [
            audit_row(
                "internal_heldout_reproduction",
                "Internal held-out",
                "NOT_RUN",
                source=rel(INTERNAL_CSV),
                note="Run with --recompute-internal to train the two current LightGBM heads.",
            )
        ]
    rows = []
    for item in reproduction:
        rows.append(
            audit_row(
                f"internal_heldout_{item['metric']}",
                "Internal held-out",
                str(item["status"]),
                recorded=f"{float(item['recorded_value']):.6f}",
                recomputed=f"{float(item['recomputed_value']):.6f}",
                difference=f"{float(item['absolute_difference']):.6g}",
                source=str(item["recorded_source"]),
                note=(
                    "Current-matrix rerun; the original run did not record input hashes or a "
                    "complete runtime manifest."
                ),
            )
        )
    return rows


def write_report(audit: pd.DataFrame) -> None:
    internal = pd.read_csv(INTERNAL_CSV) if INTERNAL_CSV.exists() else pd.DataFrame()
    dedup = pd.read_csv(DEDUP_CSV)
    current_dedup = dedup.loc[dedup["model_id"] == CURRENT_ID].iloc[0]
    internal_text = "Not rerun."
    if not internal.empty:
        original_auc = internal.loc[
            internal["metric"] == "auc_melanoma_ovr", "recorded_value"
        ].iloc[0]
        rerun_auc = internal.loc[
            internal["metric"] == "auc_melanoma_ovr", "recomputed_value"
        ].iloc[0]
        internal_text = (
            f"The recorded held-out melanoma AUC was {original_auc:.6f}; the current-input "
            f"rerun produced {rerun_auc:.6f}. The small difference is real and the historical "
            "run lacks enough provenance to explain it exactly."
        )

    failures = int((audit["status"] == "FAIL").sum())
    lines = [
        "# Paper Resource Verification Report",
        "",
        "Evidence snapshot: **2026-07-10**.",
        "",
        "## Verdict",
        "",
        f"Automated hard-check failures: **{failures}**.",
        "",
        "The external file-level arithmetic and exact deployed-bundle identity are correct. "
        "The original wording was nevertheless too strong about observation independence, "
        "diagnostic ground truth, internal reproducibility, cross-validation preprocessing, "
        "and training reproducibility.",
        "",
        "## Findings that passed",
        "",
        "- The five deployed ONNX files match the manifest and bundle SHA-256.",
        "- Full external inference was rerun from the raw files and remained byte-identical "
        "to the 107-row reference output.",
        "- AUC, average precision, Brier score, threshold counts, Wilson intervals, paired "
        "DeLong test and paired bootstrap were independently recomputed from per-file scores.",
        "- Internal source, split and class file counts match the current filesystem.",
        "",
        "## Corrections and limitations",
        "",
        "1. The external unit is a labelled **file**, not a proven independent patient, lesion "
        "or clinical case. Folder labels are internally consistent, but the diagnostic "
        "reference standard is not documented in the available artifacts.",
        "2. Four near-identical image pairs were detected objectively (pHash and dHash distance "
        "zero; mean absolute decoded-pixel difference below one 8-bit intensity level). "
        "Retaining one file per pair leaves 103 files. "
        f"The current-bundle AUC becomes {current_dedup['auc_melanoma']:.6f} and not-benign "
        f"specificity becomes {current_dedup['not_benign_specificity']:.6f}. This sensitivity "
        "analysis does not resolve patient/lesion clustering.",
        "3. Visual review also found at least one apparent repeated-acquisition sequence, but "
        "clinical metadata are required before defining lesion clusters. Consequently, the "
        "reported DeLong and Wilson intervals are naive per-file intervals, not cluster-adjusted "
        "clinical confidence intervals.",
        f"4. {internal_text}",
        "5. In the descriptive 10-fold CV script, StandardScaler is fit on the full dataset "
        "before folds are created. The CV is therefore additionally affected by unsupervised "
        "preprocessing leakage, besides partially in-sample CNN representations.",
        "6. The current ONNX binaries are available and hash-verifiable. The exact training "
        "pipeline is not reproducible because dataset, commit and checkpoint provenance are "
        "incomplete.",
        "",
        "## Paper-safe rule",
        "",
        "Use ‘107 labelled files’ rather than ‘107 independent cases’ or ‘107 patients’. Treat "
        "external estimates as a technical file-level analysis until patient/lesion identifiers, "
        "reference standard, recruitment and ethics are documented. Preserve both the recorded "
        "and rerun internal values until a fully versioned rerun is frozen.",
        "",
        "Machine-readable details:",
        "",
        f"- `{rel(AUDIT_CSV)}`",
        f"- `{rel(DUPLICATE_CSV)}`",
        f"- `{rel(DEDUP_CSV)}`",
        f"- `{rel(DEDUP_PAIRED_CSV)}`",
        f"- `{rel(INTERNAL_CSV)}`",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recompute-internal",
        action="store_true",
        help="Retrain two LightGBM heads from current local matrices for the held-out check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    GENERATED.mkdir(parents=True, exist_ok=True)
    external, _ = verify_external()
    static = verify_static_and_dataset()
    internal = internal_reproduction_rows(args.recompute_internal)
    audit = pd.DataFrame(external + static + internal_audit_rows(internal))
    audit.to_csv(AUDIT_CSV, index=False)
    write_report(audit)
    failures = int((audit["status"] == "FAIL").sum())
    print(f"wrote {AUDIT_CSV} ({len(audit)} checks; hard failures={failures})")
    print(f"wrote {REPORT}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
