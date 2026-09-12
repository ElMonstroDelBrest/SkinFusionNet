"""Deployment parity gate for the canonical V2S single-pass ONNX bundle.

Runs the full deployed external pipeline without overwriting
``external_val/per_image.csv``, then compares the generated per-image outputs
against that reference.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from derm import paths
from derm.export import eval_external as E


OUT_CSV = paths.EXTERNAL_PARITY_GATE_CSV
OUT_MD = paths.EXTERNAL_PARITY_GATE_MD
GENERATED_CSV = paths.EXTERNAL_PARITY_CSV
GENERATED_TXT = paths.EXTERNAL_PARITY_SUMMARY
GENERATED_PREVIEWS = paths.EXTERNAL_PARITY_PREVIEWS
REF_CSV = paths.EXTERNAL_VAL / "per_image.csv"


NUMERIC_COLS = ["p_nevus", "p_mel", "p_atyp", "coverage"]
EXACT_COLS = ["file", "folder", "true", "argmax"]


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(paths.ROOT).as_posix()
    except ValueError:
        return str(path)


def compare(
    ref: Path, got: Path, manifest: dict[str, object]
) -> dict[str, object]:
    ref_bytes = ref.read_bytes()
    got_bytes = got.read_bytes()
    ref_df = pd.read_csv(ref)
    got_df = pd.read_csv(got)

    row: dict[str, object] = {
        "status": "ok",
        "model_bundle_id": manifest["bundle_id"],
        "model_bundle_sha256": manifest["bundle_sha256"],
        "preprocessing_version": manifest["preprocessing"]["version"],
        "decision_policy_version": manifest["decision_policy"]["version"],
        "reference": portable_path(ref),
        "generated": portable_path(got),
        "byte_equal": ref_bytes == got_bytes,
        "n_reference": len(ref_df),
        "n_generated": len(got_df),
        "max_abs_diff": 0.0,
        "mismatched_exact_cells": 0,
        "reason": "",
    }

    if list(ref_df.columns) != list(got_df.columns):
        row["status"] = "failed"
        row["reason"] = "column mismatch"
        return row
    if len(ref_df) != len(got_df):
        row["status"] = "failed"
        row["reason"] = "row-count mismatch"
        return row

    exact_mismatch = 0
    for col in EXACT_COLS:
        exact_mismatch += int((ref_df[col].to_numpy() != got_df[col].to_numpy()).sum())
    row["mismatched_exact_cells"] = exact_mismatch

    max_diff = 0.0
    for col in NUMERIC_COLS:
        diff = np.abs(ref_df[col].to_numpy(float) - got_df[col].to_numpy(float))
        if len(diff):
            max_diff = max(max_diff, float(np.max(diff)))
        row[f"max_abs_diff_{col}"] = float(np.max(diff)) if len(diff) else 0.0
    row["max_abs_diff"] = max_diff

    if exact_mismatch or max_diff != 0.0:
        row["status"] = "failed"
        row["reason"] = f"exact_mismatch={exact_mismatch}, max_abs_diff={max_diff:.3g}"
    return row


def write_outputs(row: dict[str, object]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)

    OUT_MD.write_text(
        "\n".join([
            "# External Deployment Parity Gate",
            "",
            f"Status: `{row['status']}`",
            "",
            f"- Bundle: `{row['model_bundle_id']}`",
            f"- Bundle SHA-256: `{row['model_bundle_sha256']}`",
            f"- Preprocessing: `{row['preprocessing_version']}`",
            f"- Decision policy: `{row['decision_policy_version']}`",
            f"- Reference: `{row['reference']}`",
            f"- Generated: `{row['generated']}`",
            f"- Rows: reference={row['n_reference']}, generated={row['n_generated']}",
            f"- Byte equal: `{row['byte_equal']}`",
            f"- Mismatched exact cells: `{row['mismatched_exact_cells']}`",
            f"- Max abs diff: `{row['max_abs_diff']}`",
            f"- Reason: `{row['reason']}`",
            "",
            "This gate exercises the deployed ONNX path: U-Net, V2S lesion/dehair "
            "single-pass 1280-d feature extractors, and LightGBM A/C averaged "
            "probabilities.",
        ])
        + "\n"
    )


def main() -> None:
    if not REF_CSV.exists():
        raise FileNotFoundError(f"missing external reference CSV: {REF_CSV}")
    manifest = E.MANIFEST
    E.OUT_CSV = GENERATED_CSV
    E.OUT_TXT = GENERATED_TXT
    E.PREVIEW_DIR = GENERATED_PREVIEWS
    E.main()
    row = compare(REF_CSV, E.OUT_CSV, manifest)
    write_outputs(row)
    print(f"parity status={row['status']} byte_equal={row['byte_equal']} max_abs_diff={row['max_abs_diff']}")
    print(f"wrote {OUT_CSV} and {OUT_MD}")
    if row["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
