"""Build and verify the experiment-data handoff package.

The handoff manifest inventories immutable inputs and observation-level outputs.
Raw-image fingerprints describe the files currently present on disk; they do not
retroactively prove that the same bytes trained the deployed ONNX bundle.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from derm import paths
from derm.export.model_manifest import validate_manifest


INTERNAL_RAW_MANIFEST = paths.INTERNAL_RAW_MANIFEST
EXTERNAL_RAW_MANIFEST = paths.EXTERNAL_RAW_MANIFEST
INTERNAL_METADATA = paths.INTERNAL_METADATA
INTERNAL_PREDICTIONS = paths.INTERNAL_PREDICTIONS
INTERNAL_EXPERIMENTS = paths.INTERNAL_EXPERIMENT_REGISTRY
INTERNAL_SUMMARY = paths.INTERNAL_DECOMPOSITION
ISIC2019_PREDICTIONS = paths.ISIC2019_PREDICTIONS
ISIC2019_METRICS = paths.ISIC2019_METRICS
ISIC2019_AUDIT = paths.ISIC2019_AUDIT
ISIC2019_WORKBOOK = paths.ISIC2019_WORKBOOK
CURRENT_ONNX_IMPORTANCE = paths.CURRENT_ONNX_IMPORTANCE
EXTERNAL_PREDICTIONS = paths.EXTERNAL_PARITY_CSV
EXTERNAL_GATE = paths.EXTERNAL_PARITY_GATE_CSV
DUPLICATE_AUDIT = paths.PAPER_NEAR_DUPLICATE_AUDIT
VERIFICATION_AUDIT = paths.PAPER_VERIFICATION_AUDIT
HANDOFF_MANIFEST = paths.HANDOFF_MANIFEST

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
EXPECTED_INTERNAL_ROWS = 25_903
EXPECTED_EXTERNAL_ROWS = 107
EXPECTED_INTERNAL_PREDICTIONS = 4_462
EXPECTED_ISIC2019_HELDOUT = 2_239


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(paths.ROOT).as_posix()


def source_from_name(name: str) -> str:
    stem = Path(name).stem
    if stem.startswith("ham_"):
        return "HAM10000"
    if stem.startswith("isic2019_"):
        return "ISIC2019"
    if stem.startswith("s1_"):
        return "ISIC2017"
    if stem.startswith("s2_"):
        return "Kaggle9"
    return "unknown"


def canonical_image_id(name: str) -> str:
    stem = Path(name).stem
    for prefix in ("ham_", "isic2019_", "s1_", "s2_"):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return stem


def aggregate_fingerprint(rows: list[dict[str, Any]]) -> str:
    """Hash ordered path, size and raw SHA-256 digest for a file collection."""
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(item["relative_path"])):
        digest.update(str(row["relative_path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(str(row["sha256"])))
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_internal_raw_manifest() -> tuple[int, str]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "valid", "test"):
        for class_name in ("nevus", "melanoma", "atypical"):
            class_dir = paths.MERGED / split / class_name
            for path in sorted(class_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                rows.append(
                    {
                        "relative_path": relative(path),
                        "file": path.name,
                        "image_id": canonical_image_id(path.name),
                        "source_reconstructed": source_from_name(path.name),
                        "split_original": split,
                        "class_harmonized": class_name,
                        "bytes": path.stat().st_size,
                        "sha256": file_sha256(path),
                    }
                )
    if len(rows) != EXPECTED_INTERNAL_ROWS:
        raise ValueError(
            f"internal raw image count changed: {len(rows)} != {EXPECTED_INTERNAL_ROWS}"
        )
    write_csv(INTERNAL_RAW_MANIFEST, rows)
    return len(rows), aggregate_fingerprint(rows)


def build_external_raw_manifest() -> tuple[int, str]:
    rows: list[dict[str, Any]] = []
    for folder, class_id, class_name in (
        ("Nevi", 0, "nevus"),
        ("Melanom", 1, "melanoma"),
    ):
        for path in sorted((paths.EXTERNAL_VAL / folder).iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            rows.append(
                {
                    "relative_path": relative(path),
                    "folder_label": folder,
                    "class_id_folder_derived": class_id,
                    "class_name_folder_derived": class_name,
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                    "reference_standard": "not documented; label derived from folder",
                }
            )
    if len(rows) != EXPECTED_EXTERNAL_ROWS:
        raise ValueError(
            f"external raw image count changed: {len(rows)} != {EXPECTED_EXTERNAL_ROWS}"
        )
    write_csv(EXTERNAL_RAW_MANIFEST, rows)
    return len(rows), aggregate_fingerprint(rows)


def fingerprint_from_csv(path: Path) -> tuple[int, str]:
    data = pd.read_csv(path)
    required = {"relative_path", "bytes", "sha256"}
    if not required.issubset(data.columns):
        raise ValueError(f"invalid raw manifest schema: {path}")
    rows = data.to_dict(orient="records")
    return len(rows), aggregate_fingerprint(rows)


def check(condition: bool, check_id: str, detail: str) -> dict[str, str]:
    if not condition:
        raise ValueError(f"handoff check failed [{check_id}]: {detail}")
    return {"check_id": check_id, "status": "PASS", "detail": detail}


def verify_internal_predictions() -> list[dict[str, str]]:
    data = pd.read_csv(INTERNAL_PREDICTIONS)
    meta = pd.read_csv(INTERNAL_METADATA)
    checks = [
        check(
            len(data) == EXPECTED_INTERNAL_PREDICTIONS,
            "internal_prediction_row_count",
            f"rows={len(data)} expected={EXPECTED_INTERNAL_PREDICTIONS}",
        ),
        check(
            set(data["condition"]) == {"1a"},
            "internal_prediction_conditions",
            (
                f"conditions={sorted(data['condition'].unique())}; C0/C1/C2 historical "
                "runs did not retain observation-level OOF outputs"
            ),
        ),
        check(
            not data.duplicated(["condition", "row_index"]).any(),
            "internal_prediction_keys",
            "condition,row_index keys are unique",
        ),
    ]
    sums = data[["p_nevus", "p_melanoma", "p_atypical"]].sum(axis=1)
    checks.append(
        check(
            np.allclose(sums, 1.0, atol=1e-6),
            "internal_probability_normalization",
            f"min_sum={sums.min():.9f} max_sum={sums.max():.9f}",
        )
    )
    aligned = data.merge(
        meta[["row_index", "file", "class_id", "split_original"]],
        on="row_index",
        suffixes=("_prediction", "_metadata"),
        validate="many_to_one",
    )
    alignment_ok = (
        (aligned["file_prediction"] == aligned["file_metadata"]).all()
        and (aligned["class_id_prediction"] == aligned["class_id_metadata"]).all()
        and (
            aligned["split_original_prediction"]
            == aligned["split_original_metadata"]
        ).all()
    )
    checks.append(
        check(alignment_ok, "internal_metadata_alignment", "all prediction rows align")
    )
    phase_1a = data[data["condition"] == "1a"]
    checks.append(
        check(
            set(phase_1a["split_original"]) == {"test"}
            and set(phase_1a["fold"]) == {0},
            "internal_phase_1a_scope",
            "1a contains original test rows only; fold=0 denotes fixed split",
        )
    )
    registry = pd.read_csv(INTERNAL_EXPERIMENTS)
    checks.append(
        check(
            set(registry["condition"]) == {"1a", "C0", "C1", "C2", "C3"},
            "internal_experiment_registry",
            "registry includes completed conditions and blocked C3",
        )
    )
    aggregate_only = registry[registry["condition"].isin(["C0", "C1", "C2"])]
    checks.append(
        check(
            set(aggregate_only["status"])
            == {"aggregate-only-observation-predictions-not-retained"},
            "internal_missing_oof_disclosure",
            "C0/C1/C2 missing observation-level OOF outputs are explicitly registered",
        )
    )
    return checks


def verify_external_predictions() -> list[dict[str, str]]:
    data = pd.read_csv(EXTERNAL_PREDICTIONS)
    checks = [
        check(
            len(data) == EXPECTED_EXTERNAL_ROWS,
            "external_prediction_row_count",
            f"rows={len(data)} expected={EXPECTED_EXTERNAL_ROWS}",
        ),
        check(
            not data.duplicated(["folder", "file"]).any(),
            "external_prediction_keys",
            "folder,file keys are unique",
        ),
    ]
    sums = data[["p_nevus", "p_mel", "p_atyp"]].sum(axis=1)
    checks.append(
        check(
            np.allclose(sums, 1.0, atol=1e-6),
            "external_probability_normalization",
            f"min_sum={sums.min():.9f} max_sum={sums.max():.9f}",
        )
    )
    duplicates = pd.read_csv(DUPLICATE_AUDIT)
    filename_columns = {
        "member_1_folder",
        "member_1_file",
        "member_2_folder",
        "member_2_file",
        "retained_file",
        "removed_files",
    }
    checks.append(
        check(
            len(duplicates) == 4 and filename_columns.issubset(duplicates.columns),
            "external_duplicate_traceability",
            "four near-identical pairs include member and sensitivity-filter filenames",
        )
    )
    return checks


def verify_isic2019_heldout() -> list[dict[str, str]]:
    """Verify the dedicated ISIC2019/test scientific handoff."""
    predictions = pd.read_csv(ISIC2019_PREDICTIONS)
    metrics = pd.read_csv(ISIC2019_METRICS)
    audit = pd.read_csv(ISIC2019_AUDIT)
    importance = pd.read_csv(CURRENT_ONNX_IMPORTANCE)
    raw = pd.read_csv(INTERNAL_RAW_MANIFEST)
    checks = [
        check(
            len(predictions) == EXPECTED_ISIC2019_HELDOUT,
            "isic2019_heldout_row_count",
            f"rows={len(predictions)} expected={EXPECTED_ISIC2019_HELDOUT}",
        ),
        check(
            predictions["image_id"].nunique() == len(predictions)
            and predictions["raw_sha256"].nunique() == len(predictions),
            "isic2019_heldout_unique_images",
            "image_id and raw SHA-256 are unique within evaluation",
        ),
        check(
            set(predictions["evaluation_cohort"]) == {"ISIC2019/test held-out"},
            "isic2019_heldout_scope",
            "all rows are explicitly scoped to ISIC2019/test",
        ),
        check(
            predictions["true_class"].value_counts().to_dict()
            == {"nevus": 894, "atypical": 803, "melanoma": 542},
            "isic2019_heldout_class_counts",
            "nevus=894 atypical=803 melanoma=542",
        ),
    ]
    sums = predictions[["p_nevus", "p_melanoma", "p_atypical"]].sum(axis=1)
    checks.append(
        check(
            np.allclose(sums, 1.0, atol=1e-6),
            "isic2019_probability_normalization",
            f"min_sum={sums.min():.9f} max_sum={sums.max():.9f}",
        )
    )
    fitting = raw[raw["split_original"].isin(["train", "valid"])]
    image_overlap = set(predictions["image_id"]) & set(fitting["image_id"])
    hash_overlap = set(predictions["raw_sha256"]) & set(fitting["sha256"])
    checks.extend(
        [
            check(
                not image_overlap,
                "isic2019_image_id_overlap_vs_train_valid",
                f"overlap={len(image_overlap)}",
            ),
            check(
                not hash_overlap,
                "isic2019_sha256_overlap_vs_train_valid",
                f"overlap={len(hash_overlap)}",
            ),
            check(
                not (audit["status"] == "FAIL").any(),
                "isic2019_dedicated_audit",
                f"statuses={audit['status'].value_counts().to_dict()}",
            ),
            check(
                set(metrics["scope"])
                == {
                    "global_argmax",
                    "calibration",
                    "per_class_argmax_ovr",
                    "melanoma_fixed_threshold",
                },
                "isic2019_metric_scopes",
                "global, calibration, per-class and melanoma-threshold metrics present",
            ),
            check(
                ISIC2019_WORKBOOK.stat().st_size > 100_000,
                "isic2019_workbook_present",
                f"bytes={ISIC2019_WORKBOOK.stat().st_size}",
            ),
            check(
                len(importance) == 1_298
                and importance["feature"].nunique() == 1_298,
                "current_onnx_importance_feature_count",
                f"rows={len(importance)} unique_features={importance['feature'].nunique()}",
            ),
            check(
                int(importance["splits_combined"].sum()) == 111_600,
                "current_onnx_importance_split_count",
                f"combined_branch_splits={int(importance['splits_combined'].sum())}",
            ),
        ]
    )
    return checks


def artifact_record(path: Path, role: str, status: str, count_rows: bool = True) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing handoff artifact: {path}")
    rows: int | None = None
    if count_rows and path.suffix.lower() == ".csv":
        with path.open("rb") as stream:
            rows = max(sum(chunk.count(b"\n") for chunk in iter(lambda: stream.read(1024 * 1024), b"")) - 1, 0)
    return {
        "path": relative(path),
        "role": role,
        "status": status,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "data_rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-raw-manifests",
        action="store_true",
        help="Re-hash all current internal and external raw images before building the handoff.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_manifest(
        paths.APP_MODELS / "model_manifest.json", paths.APP_MODELS
    )
    if args.refresh_raw_manifests:
        internal_count, internal_fingerprint = build_internal_raw_manifest()
        external_count, external_fingerprint = build_external_raw_manifest()
    else:
        if not INTERNAL_RAW_MANIFEST.exists() or not EXTERNAL_RAW_MANIFEST.exists():
            raise FileNotFoundError(
                "raw manifests are missing; rerun with --refresh-raw-manifests"
            )
        internal_count, internal_fingerprint = fingerprint_from_csv(
            INTERNAL_RAW_MANIFEST
        )
        external_count, external_fingerprint = fingerprint_from_csv(
            EXTERNAL_RAW_MANIFEST
        )
    checks = (
        verify_internal_predictions()
        + verify_isic2019_heldout()
        + verify_external_predictions()
    )
    checks.extend(
        [
            check(
                internal_count == EXPECTED_INTERNAL_ROWS,
                "internal_raw_manifest_count",
                f"rows={internal_count}",
            ),
            check(
                external_count == EXPECTED_EXTERNAL_ROWS,
                "external_raw_manifest_count",
                f"rows={external_count}",
            ),
        ]
    )

    artifacts = [
        artifact_record(paths.ROOT / "HANDOFF.md", "human-readable handoff contract and data dictionary", "current"),
        artifact_record(paths.ROOT / "derm" / "analysis" / "build_handoff.py", "handoff builder and integrity checks", "current"),
        artifact_record(paths.ROOT / "derm" / "analysis" / "internal_validity_m1.py", "internal evaluation and observation export implementation", "current"),
        artifact_record(paths.ROOT / "derm" / "analysis" / "build_isic2019_eval_workbook.py", "ISIC2019/test held-out workbook builder", "current"),
        artifact_record(paths.ROOT / "derm" / "analysis" / "extract_onnx_feature_importance.py", "deployed LightGBM ONNX split-frequency extractor", "current"),
        artifact_record(paths.ROOT / "derm" / "analysis" / "verify_paper_resources.py", "external arithmetic and duplicate audit implementation", "current"),
        artifact_record(paths.APP_MODELS / "model_manifest.json", "deployed bundle identity", "current"),
        artifact_record(INTERNAL_RAW_MANIFEST, "current internal raw-image byte manifest", "current-filesystem-snapshot"),
        artifact_record(INTERNAL_METADATA, "internal observation metadata reconstructed from file layout", "current-with-missing-patient-lesion-ids"),
        artifact_record(paths.FEATURES_CSV, "18 handcrafted feature matrix", "current-input", count_rows=False),
        artifact_record(paths.CNN_V2S_FEATS_CSV, "V2S lesion-view feature matrix", "current-input", count_rows=False),
        artifact_record(paths.CNN_V2S_DEHAIR_CSV, "V2S dehair-view feature matrix", "current-input", count_rows=False),
        artifact_record(INTERNAL_PREDICTIONS, "current held-out 1a observation-level predictions", "current-auditable-output"),
        artifact_record(ISIC2019_PREDICTIONS, "ISIC2019/test held-out observation-level predictions", "current-primary-evaluation"),
        artifact_record(ISIC2019_METRICS, "ISIC2019/test recalculated metrics", "generated-derivative"),
        artifact_record(ISIC2019_AUDIT, "ISIC2019/test overlap and integrity audit", "current-audit"),
        artifact_record(CURRENT_ONNX_IMPORTANCE, "deployed ONNX input split-frequency inspection", "current-model-inspection"),
        artifact_record(ISIC2019_WORKBOOK, "ISIC2019/test complete scientific handoff workbook", "generated-derivative", count_rows=False),
        artifact_record(INTERNAL_EXPERIMENTS, "internal experiment protocol/caveat registry", "current-auditable-output"),
        artifact_record(INTERNAL_SUMMARY, "internal aggregate metrics", "current-summary"),
        artifact_record(EXTERNAL_RAW_MANIFEST, "external raw-file byte manifest", "current-folder-labelled-snapshot"),
        artifact_record(EXTERNAL_PREDICTIONS, "external exact-bundle observation-level predictions", "current-auditable-output"),
        artifact_record(EXTERNAL_GATE, "external Python repeatability gate", "current-not-device-parity"),
        artifact_record(DUPLICATE_AUDIT, "external near-identical-file audit", "current-sensitivity-audit"),
        artifact_record(VERIFICATION_AUDIT, "independent arithmetic and provenance checks", "current-audit"),
        artifact_record(paths.PAPER_INTERNAL_REPRODUCTION, "recorded-versus-current held-out comparison", "current-audit"),
        artifact_record(paths.PAPER_EXTERNAL_PERFORMANCE, "external aggregate performance table", "generated-derivative"),
        artifact_record(paths.PAPER_EXTERNAL_PAIRED, "paired single-pass versus TTA8 comparison", "generated-derivative"),
        artifact_record(paths.PAPER_PERFORMANCE_REGISTRY, "cross-protocol performance registry", "generated-derivative"),
        artifact_record(paths.PAPER_WORKBOOK, "generated writing workbook", "generated-derivative"),
        artifact_record(paths.TTA8_EXTERNAL_CSV, "archived paired TTA8 predictions", "historical"),
        artifact_record(paths.INTERNAL_CV_LOG, "recorded optimistic image-level CV log", "historical-descriptive"),
    ]
    manifest = {
        "schema_version": 1,
        "handoff_date": dt.date.today().isoformat(),
        "scope": "experiment data and provenance; not a scientific or clinical report",
        "score_semantics": (
            "normalized multiclass probabilities summing to one in class order "
            "[nevus, melanoma, atypical]; individual class probabilities may be evaluated one-vs-rest"
        ),
        "dataset_snapshots": {
            "internal_current_filesystem": {
                "n_files": internal_count,
                "aggregate_sha256": internal_fingerprint,
                "manifest": relative(INTERNAL_RAW_MANIFEST),
                "provenance_warning": (
                    "This fingerprints the files currently present. It does not prove that the "
                    "same byte set trained the deployed bundle."
                ),
            },
            "external_folder_labelled": {
                "n_files": external_count,
                "aggregate_sha256": external_fingerprint,
                "manifest": relative(EXTERNAL_RAW_MANIFEST),
                "provenance_warning": (
                    "Labels are derived from folders; diagnostic reference standard and "
                    "patient/lesion independence are not documented."
                ),
            },
        },
        "checks": checks,
        "artifacts": artifacts,
        "known_limitations": [
            "patient_id and lesion_id are unresolved for the internal merged dataset",
            "the external unit is a folder-labelled file, not a proven independent clinical case",
            "four external near-identical file pairs are retained in the primary table and audited separately",
            "condition 1a calibrates its threshold on valid after fitting heads on train+valid; its threshold metrics are not independent",
            "C0/C1/C2 use partially in-sample CNN representations",
            "the historical C0/C1/C2 runs retained aggregate metrics but not observation-level OOF predictions or fold assignments",
            "the exact training dataset, commit and source CNN checkpoint hashes of the deployed bundle were not recorded",
            "Python repeatability against a prior CSV is not end-to-end Flutter/Android parity",
            "the external data contain no labelled atypical cases and do not validate the full three-class task",
            "ONNX input split frequency is not gain importance, SHAP importance, marginal value or causal effect",
        ],
    }
    HANDOFF_MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"handoff ok: checks={len(checks)} artifacts={len(artifacts)} "
        f"-> {HANDOFF_MANIFEST}"
    )


if __name__ == "__main__":
    main()
