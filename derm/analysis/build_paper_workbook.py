"""Build the English paper-data workbook from frozen project resources.

The workbook has two layers. The aggregate layer holds the reading views used
for drafting. The raw layer holds the underlying per-observation and technical
tables, so that every reported number can be recomputed from the workbook
alone, without the reader having to obtain the CSV files separately.

The raw layer does carry per-file identifiers and per-image SHA-256 digests. It
still contains no raw images, and the two CNN embedding matrices are out of
scope: at 25 903 x 1280 each they do not fit a spreadsheet and remain
transfer-only artifacts.

Adding the raw layer relaxes the earlier rule that this workbook stayed
aggregate-only. The rule it does not relax: an aggregate view never overrides
the per-observation table it is derived from. Where both are present, the raw
sheet is authoritative.

Run from the repository root::

    .venv_extval/bin/python -m derm.analysis.build_paper_workbook
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.drawing.image import Image
from openpyxl.chart import BarChart, Reference
from openpyxl.chart.error_bar import ErrorBars
from openpyxl.chart.data_source import NumDataSource, NumRef
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.hyperlink import Hyperlink
from openpyxl.worksheet.table import Table, TableStyleInfo

from derm import paths


PAPER = paths.PAPER
GENERATED = PAPER / "generated"
FIGURES = PAPER / "figures"
OUTPUT = PAPER / "SkinFusionNet_Paper_Data.xlsx"

SNAPSHOT_DATE = "2026-07-10"
CURRENT_ID = "skinfusionnet-v2s3c-singlepass-1.0.0"

NAVY = "17365D"
BLUE = "1F4E78"
LIGHT_BLUE = "D9EAF7"
TEAL = "0F6B78"
LIGHT_TEAL = "DDEBF7"
GREEN = "E2F0D9"
YELLOW = "FFF2CC"
ORANGE = "FCE4D6"
RED = "F4CCCC"
GREY = "E7E6E6"
WHITE = "FFFFFF"
TEXT = "1F1F1F"

THIN_GREY = Side(style="thin", color="D9E1F2")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(paths.ROOT).as_posix()


def clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def table_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not sanitized[0].isalpha():
        sanitized = "T_" + sanitized
    return sanitized[:240]


def add_title(ws, title: str, subtitle: str, *, end_col: int = 12) -> int:
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=end_col)
    cell = ws.cell(1, 1, title)
    cell.font = Font(size=20, bold=True, color=WHITE)
    cell.fill = PatternFill("solid", fgColor=NAVY)
    cell.alignment = Alignment(vertical="center")
    ws.row_dimensions[1].height = 32
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=end_col)
    sub = ws.cell(2, 1, subtitle)
    sub.font = Font(size=10, italic=True, color="404040")
    sub.fill = PatternFill("solid", fgColor=LIGHT_BLUE)
    sub.alignment = Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[2].height = 30
    return 4


def add_section(ws, row: int, title: str, *, start_col: int = 1, end_col: int = 12) -> int:
    ws.merge_cells(
        start_row=row,
        start_column=start_col,
        end_row=row,
        end_column=end_col,
    )
    cell = ws.cell(row, start_col, title)
    cell.font = Font(size=12, bold=True, color=WHITE)
    cell.fill = PatternFill("solid", fgColor=TEAL)
    cell.alignment = Alignment(vertical="center")
    ws.row_dimensions[row].height = 22
    return row + 1


def add_key_value(ws, row: int, key: str, value: Any, *, note: str = "") -> int:
    ws.cell(row, 1, key)
    ws.cell(row, 2, clean(value))
    ws.cell(row, 1).font = Font(bold=True, color=NAVY)
    ws.cell(row, 1).fill = PatternFill("solid", fgColor=LIGHT_TEAL)
    ws.cell(row, 2).alignment = Alignment(wrap_text=True, vertical="top")
    if note:
        ws.merge_cells(start_row=row, start_column=3, end_row=row, end_column=8)
        ws.cell(row, 3, note)
        ws.cell(row, 3).font = Font(italic=True, color="666666")
        ws.cell(row, 3).alignment = Alignment(wrap_text=True, vertical="top")
    return row + 1


def add_table(
    ws,
    headers: list[str],
    rows: Iterable[Iterable[Any]],
    *,
    start_row: int,
    start_col: int = 1,
    name: str,
    style: str = "TableStyleMedium2",
) -> tuple[int, int]:
    materialized = [[clean(value) for value in row] for row in rows]
    for offset, header in enumerate(headers):
        cell = ws.cell(start_row, start_col + offset, header)
        cell.font = Font(bold=True, color=WHITE)
        cell.fill = PatternFill("solid", fgColor=BLUE)
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        cell.border = Border(bottom=THIN_GREY)
    ws.row_dimensions[start_row].height = 32

    for row_offset, values in enumerate(materialized, start=1):
        for col_offset, value in enumerate(values):
            cell = ws.cell(start_row + row_offset, start_col + col_offset, value)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            cell.border = Border(bottom=THIN_GREY)
            header = headers[col_offset].lower()
            if isinstance(value, float):
                if any(token in header for token in ("auc", "accuracy", "dice", "sensitivity", "specificity", "precision", "brier", "estimate", "ci")):
                    cell.number_format = "0.0000"
                else:
                    cell.number_format = "0.000000"
            elif isinstance(value, int):
                cell.number_format = "#,##0"

    end_row = start_row + len(materialized)
    end_col = start_col + len(headers) - 1
    if materialized:
        ref = (
            f"{get_column_letter(start_col)}{start_row}:"
            f"{get_column_letter(end_col)}{end_row}"
        )
        table = Table(displayName=table_name(name), ref=ref)
        table.tableStyleInfo = TableStyleInfo(
            name=style,
            showFirstColumn=False,
            showLastColumn=False,
            showRowStripes=True,
            showColumnStripes=False,
        )
        ws.add_table(table)
    return end_row + 2, end_col


def fit_columns(ws, *, min_width: int = 10, max_width: int = 48) -> None:
    for column_cells in ws.columns:
        letter = get_column_letter(column_cells[0].column)
        width = min_width
        for cell in column_cells:
            if cell.value is not None:
                lines = str(cell.value).splitlines() or [""]
                longest_line = max(len(line) for line in lines)
                width = max(width, min(longest_line + 2, max_width))
        ws.column_dimensions[letter].width = width


def style_evidence(ws, header_row: int, first_data_row: int, last_data_row: int) -> None:
    evidence_col = None
    for cell in ws[header_row]:
        if str(cell.value).lower() in {"evidence grade", "evidence_grade"}:
            evidence_col = cell.column
            break
    if evidence_col is None:
        return
    fills = {
        "B": PatternFill("solid", fgColor=GREEN),
        "C": PatternFill("solid", fgColor=YELLOW),
        "D": PatternFill("solid", fgColor=ORANGE),
        "E": PatternFill("solid", fgColor=RED),
    }
    for row in range(first_data_row, last_data_row + 1):
        cell = ws.cell(row, evidence_col)
        prefix = str(cell.value or "")[:1]
        if prefix in fills:
            cell.fill = fills[prefix]
            cell.font = Font(bold=True, color=TEXT)


def style_status(ws, header_row: int, first_data_row: int, last_data_row: int) -> None:
    status_col = None
    for cell in ws[header_row]:
        if str(cell.value).lower() == "status":
            status_col = cell.column
            break
    if status_col is None:
        return
    fills = {
        "PASS": GREEN,
        "SOURCE_CONFIRMED": LIGHT_BLUE,
        "CAUTION": YELLOW,
        "NOT_EXACTLY_REPRODUCED": ORANGE,
        "NOT_VERIFIABLE": RED,
        "FAIL": RED,
    }
    for row in range(first_data_row, last_data_row + 1):
        cell = ws.cell(row, status_col)
        colour = fills.get(str(cell.value or ""))
        if colour:
            cell.fill = PatternFill("solid", fgColor=colour)
            cell.font = Font(bold=True, color=TEXT)


def dataframe_rows(data: pd.DataFrame, columns: list[str]) -> list[list[Any]]:
    return [[row[column] for column in columns] for _, row in data.iterrows()]


def build_readme(wb: Workbook, manifest: dict[str, Any]) -> None:
    ws = wb.create_sheet("README")
    ws.sheet_properties.tabColor = NAVY
    row = add_title(
        ws,
        "SkinFusionNet — Paper Data Workbook",
        "English evidence pack for future manuscript preparation, in two layers: aggregate reading views, then the raw "
        "per-observation and technical tables behind them. External results are file-level. No raw images are included.",
        end_col=8,
    )
    row = add_section(ws, row, "Workbook identity", end_col=8)
    row = add_key_value(ws, row, "Evidence snapshot", SNAPSHOT_DATE)
    row = add_key_value(ws, row, "Current model bundle", manifest["bundle_id"])
    row = add_key_value(ws, row, "Bundle SHA-256", manifest["bundle_sha256"])
    row = add_key_value(
        ws,
        row,
        "Intended use",
        "Research documentation and future manuscript drafting",
        note="Not a medical device dossier and not evidence of clinical safety.",
    )
    row = add_key_value(
        ws,
        row,
        "Primary narrative",
        "Evaluation honesty and exact model-bundle traceability",
        note="The evidence supports a technical reproducibility case study, not a clinical-performance or state-of-the-art claim.",
    )

    row += 1
    row = add_section(ws, row, "Navigation", end_col=8)
    aggregate = [
        ("Verification Audit", "Independent arithmetic checks and near-duplicate sensitivity"),
        ("Key Results", "The headline estimates, with their caveats"),
        ("External Performance", "107 labelled files, deployed bundle"),
        ("Internal Performance", "Held-out and cross-validation decomposition"),
        ("Model Versions", "Lineages and what each one may be cited for"),
        ("Dataset Composition", "Sources, splits and class counts"),
        ("Claims Matrix", "Claim -> evidence grade -> safe wording"),
        ("Study Gaps", "What blocks a stronger claim, by priority"),
        ("Artifact Audit", "Which historical reports are current, archived or excluded"),
        ("Figures", "Embedded figure previews"),
        ("Charts", "Live Excel charts driven by the data below"),
        ("Source Registry", "Provenance and SHA-256 of every source file"),
    ]
    for sheet, purpose in aggregate:
        link_to_sheet(ws.cell(row, 1), sheet)
        ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=8)
        ws.cell(row, 2, purpose).alignment = Alignment(vertical="center")
        row += 1
    link_to_sheet(ws.cell(row, 1), "Raw Data Index")
    ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=8)
    ws.cell(row, 2, f"Entry point to the raw layer — {len(RAW_SHEETS) + 1} verbatim source tables").font = Font(bold=True)
    row += 1

    row += 1
    row = add_section(
        ws,
        row,
        "Project-specific evidence labels — not a validated grading scale",
        end_col=8,
    )
    evidence = [
        ["B", "Recorded internal held-out result with provenance, grouping and exact-reproduction gaps"],
        ["C", "Exact-bundle technical external file set; small and non-independent structure unresolved"],
        ["D", "Optimistic, historical, or descriptive result"],
        ["E", "Post-hoc hypothesis, stale output, or incomplete experiment"],
    ]
    row, _ = add_table(
        ws,
        ["Grade", "Meaning"],
        evidence,
        start_row=row,
        name="ReadmeEvidenceGrades",
        style="TableStyleMedium4",
    )

    row = add_section(ws, row, "Critical interpretation notes", end_col=8)
    notes = [
        "The external unit is 107 folder-labelled files, not 107 proven independent patients, lesions or clinical cases. The diagnostic reference standard is not documented in available artifacts.",
        "Four objectively near-identical file pairs were detected; a one-file-per-pair sensitivity analysis leaves 103 files. At least one additional apparent repeated-acquisition sequence requires metadata confirmation.",
        "External DeLong and Wilson intervals are naive per-file intervals. They are not cluster-adjusted because patient and lesion identifiers are unavailable.",
        "The 94.24% accuracy is descriptive image-level cross-validation with partially in-sample CNN representations and a scaler fit before folds; it is optimistic, not an external generalization estimate.",
        "The recorded held-out melanoma AUC is 0.892755; a current-input rerun produced 0.893546. Preserve both until a fully versioned rerun is frozen.",
        "Single-pass was numerically higher than archived TTA×8 in file-level external AUC, but superiority was not demonstrated (paired DeLong p=0.305).",
        "For the 107-file analysis, the not-benign rule flagged 17/19 melanoma-labelled files and rejected 73/88 nevus-labelled files. After the near-identical-pair filter, specificity is 69/84.",
        "The two-class readout threshold was selected and evaluated on the same 107 labelled files and is hypothesis-generating only.",
        "The ONNX binaries are available and hash-verifiable; regenerating them from the exact training pipeline is not currently reproducible.",
        "Handcrafted-feature marginal value above the current deep representation remains untested because Deep+ABCD conditions are incomplete.",
        "Sheets prefixed 'Raw' reproduce their source table verbatim and are authoritative over any aggregate view derived from them. Start at the Raw Data Index; each raw sheet also repeats its own caveat.",
    ]
    for note in notes:
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=8)
        ws.cell(row, 1, "• " + note)
        ws.cell(row, 1).alignment = Alignment(wrap_text=True, vertical="top")
        ws.cell(row, 1).fill = PatternFill("solid", fgColor="F8F9FA")
        ws.row_dimensions[row].height = 34
        row += 1

    row += 1
    row = add_section(ws, row, "Regeneration", end_col=8)
    row = add_key_value(
        ws,
        row,
        "Paper resources",
        "MPLCONFIGDIR=/tmp/matplotlib .venv_extval/bin/python -m derm.analysis.render_paper_resources",
    )
    row = add_key_value(
        ws,
        row,
        "Verification audit",
        "MPLCONFIGDIR=/tmp/matplotlib .venv_extval/bin/python -m derm.analysis.verify_paper_resources --recompute-internal",
    )
    add_key_value(
        ws,
        row,
        "This workbook",
        ".venv_extval/bin/python -m derm.analysis.build_paper_workbook",
    )
    ws.freeze_panes = "A4"
    fit_columns(ws, max_width=62)
    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 58


def build_verification(
    wb: Workbook,
    audit: pd.DataFrame,
    duplicates: pd.DataFrame,
    deduplicated: pd.DataFrame,
    deduplicated_paired: pd.DataFrame,
    internal_recheck: pd.DataFrame,
) -> None:
    ws = wb.create_sheet("Verification Audit")
    ws.sheet_properties.tabColor = "C00000"
    row = add_title(
        ws,
        "Verification Audit",
        "Independent arithmetic checks, exact-bundle parity, near-duplicate sensitivity and internal rerun status. A PASS verifies the stated source-level calculation, not clinical validity.",
        end_col=12,
    )

    row = add_section(ws, row, "Complete check register", end_col=12)
    audit_header = row
    audit_columns = [
        "check_id",
        "area",
        "status",
        "recorded_value",
        "recomputed_value",
        "absolute_difference",
        "source",
        "note",
    ]
    row, _ = add_table(
        ws,
        [column.replace("_", " ").title() for column in audit_columns],
        dataframe_rows(audit, audit_columns),
        start_row=row,
        name="VerificationCheckRegister",
        style="TableStyleMedium2",
    )
    style_status(ws, audit_header, audit_header + 1, row - 2)

    row = add_section(
        ws,
        row,
        "Objectively near-identical external file pairs",
        end_col=12,
    )
    duplicate_columns = [
        "duplicate_group",
        "file_count",
        "class_folder",
        "phash_distance",
        "dhash_distance",
        "decoded_pixel_mae_on_0_255_scale",
        "decoded_pixel_max_abs_difference",
        "files_removed_in_sensitivity_analysis",
        "interpretation",
    ]
    row, _ = add_table(
        ws,
        [column.replace("_", " ").title() for column in duplicate_columns],
        dataframe_rows(duplicates, duplicate_columns),
        start_row=row,
        name="ExternalNearDuplicateAudit",
        style="TableStyleMedium7",
    )

    row = add_section(
        ws,
        row,
        "One-file-per-near-identical-pair sensitivity analysis",
        end_col=12,
    )
    dedup_columns = [
        "model_id",
        "removed_files",
        "n_files",
        "n_melanoma_labelled_files",
        "n_nevus_labelled_files",
        "auc_melanoma",
        "auc_melanoma_ci95_low",
        "auc_melanoma_ci95_high",
        "sensitivity_melanoma",
        "specificity_melanoma",
        "not_benign_sensitivity",
        "not_benign_specificity",
        "warning",
    ]
    row, _ = add_table(
        ws,
        [column.replace("_", " ").title() for column in dedup_columns],
        dataframe_rows(deduplicated, dedup_columns),
        start_row=row,
        name="ExternalDeduplicatedSensitivity",
        style="TableStyleMedium9",
    )

    row = add_section(
        ws,
        row,
        "Paired model comparison after the same file filter",
        end_col=12,
    )
    paired_columns = [
        "n_paired_files",
        "removed_files",
        "auc_tta8",
        "auc_singlepass",
        "delta_auc_observed",
        "delta_auc_bootstrap_ci95_low",
        "delta_auc_bootstrap_ci95_high",
        "paired_delong_p",
        "warning",
    ]
    row, _ = add_table(
        ws,
        [column.replace("_", " ").title() for column in paired_columns],
        dataframe_rows(deduplicated_paired, paired_columns),
        start_row=row,
        name="ExternalDeduplicatedPaired",
        style="TableStyleMedium9",
    )

    row = add_section(ws, row, "Internal held-out reproduction check", end_col=12)
    internal_header = row
    internal_columns = [
        "metric",
        "recorded_value",
        "recomputed_value",
        "absolute_difference",
        "exact_match",
        "status",
        "runtime",
        "warning",
    ]
    row, _ = add_table(
        ws,
        [column.replace("_", " ").title() for column in internal_columns],
        dataframe_rows(internal_recheck, internal_columns),
        start_row=row,
        name="InternalReproductionCheck",
        style="TableStyleMedium4",
    )
    style_status(ws, internal_header, internal_header + 1, row - 2)
    ws.freeze_panes = "A5"
    fit_columns(ws, max_width=74)


def build_key_results(
    wb: Workbook,
    internal: pd.DataFrame,
    external: pd.DataFrame,
    comparison: pd.Series,
    posthoc: pd.Series,
    deduplicated: pd.DataFrame,
    internal_recheck: pd.DataFrame,
) -> None:
    ws = wb.create_sheet("Key Results")
    ws.sheet_properties.tabColor = TEAL
    start = add_title(
        ws,
        "Key Results",
        "Headline estimates with the evidence language that must accompany them.",
        end_col=12,
    )
    cv, held = internal.iloc[0], internal.iloc[1]
    current = external.iloc[0]
    dedup_current = deduplicated.loc[deduplicated["model_id"] == CURRENT_ID].iloc[0]
    recheck = internal_recheck.set_index("metric")
    rerun_auc = recheck.loc["auc_melanoma_ovr"]
    rerun_auc_low = recheck.loc["auc_melanoma_ci95_low", "recomputed_value"]
    rerun_auc_high = recheck.loc["auc_melanoma_ci95_high", "recomputed_value"]
    rows = [
        ["Internal", "Recorded image-level 10-fold CV", "Single-pass 1280-d configuration", 25903, 4808, "Accuracy", cv.accuracy, f"± {cv.accuracy_uncertainty:.4f} fold SD", "", cv.evidence_grade, "Optimistic source-confirmed run: CNN representations partly in-sample and scaler fit before folds.", cv.source],
        ["Internal", "Recorded image-level 10-fold CV", "Single-pass 1280-d configuration", 25903, 4808, "Macro AUC OvR", cv.auc_macro_ovr, f"± {cv.auc_macro_ovr_uncertainty:.4f} fold SD", "", cv.evidence_grade, "Do not present as independent validation; full CV was not rerun by the verification audit.", cv.source],
        ["Internal", "Recorded original held-out image split", "Fusion heads retrained on train+valid", int(held.n), 836, "Accuracy", held.accuracy, "No interval available", "Exact current-input rerun match", held.evidence_grade, "Configuration-level file estimate; patient/lesion grouping unresolved.", held.source],
        ["Internal", "Recorded original held-out image split", "Fusion heads retrained on train+valid", int(held.n), 836, "Recorded melanoma AUC OvR", held.auc_melanoma_ovr, f"95% CI {held.auc_melanoma_ci95_low:.4f}–{held.auc_melanoma_ci95_high:.4f}", "Current-input rerun differs", held.evidence_grade, "Recorded artifact, not exactly reproduced from current matrices.", held.source],
        ["Internal", "Current-input rerun of held-out split", "Fusion heads retrained on train+valid", int(held.n), 836, "Recomputed melanoma AUC OvR", rerun_auc.recomputed_value, f"95% CI {rerun_auc_low:.4f}–{rerun_auc_high:.4f}", f"Δ vs recorded = {rerun_auc.absolute_difference:.6f}", "B-current-input-rerun-with-grouping-gaps", "Preserve alongside the recorded value until a fully versioned rerun is frozen.", paths.relative(paths.PAPER_INTERNAL_REPRODUCTION)],
        ["External", "Exact-bundle technical file-level analysis", CURRENT_ID, int(current.n), int(current.n_melanoma), "Melanoma AUC", current.auc_melanoma, f"Naive per-file 95% CI {current.auc_melanoma_ci95_low:.4f}–{current.auc_melanoma_ci95_high:.4f}", "", "C-file-level-exact-bundle", "107 folder-labelled files; reference standard and clustering are unresolved.", current.source],
        ["External sensitivity", "One file per near-identical pair", CURRENT_ID, int(dedup_current.n_files), int(dedup_current.n_melanoma_labelled_files), "Melanoma AUC", dedup_current.auc_melanoma, f"Naive per-file 95% CI {dedup_current.auc_melanoma_ci95_low:.4f}–{dedup_current.auc_melanoma_ci95_high:.4f}", "4 derivative files removed", "C-sensitivity-analysis-only", "Does not resolve repeated lesions or diagnostic-reference provenance.", paths.relative(paths.PAPER_DEDUPLICATED_SENS)],
        ["External", "File-level analysis at T=0.188", CURRENT_ID, int(current.n), int(current.n_melanoma), "Melanoma sensitivity", current.sensitivity_melanoma, f"Naive per-file 95% CI {current.sensitivity_melanoma_ci95_low:.4f}–{current.sensitivity_melanoma_ci95_high:.4f}", f"TP/FN files = {int(current.tp_melanoma)}/{int(current.fn_melanoma)}", "C-file-level-exact-bundle", "Always report with specificity and the non-independence caveat.", current.source],
        ["External", "File-level analysis at T=0.188", CURRENT_ID, int(current.n), int(current.n_melanoma), "Melanoma specificity", current.specificity_melanoma, f"Naive per-file 95% CI {current.specificity_melanoma_ci95_low:.4f}–{current.specificity_melanoma_ci95_high:.4f}", f"TN/FP files = {int(current.tn_melanoma)}/{int(current.fp_melanoma)}", "C-file-level-exact-bundle", "Inherited threshold; not independently calibrated for this release.", current.source],
        ["External", "Composite not-benign file rule", CURRENT_ID, int(current.n), int(current.n_melanoma), "Sensitivity", current.not_benign_sensitivity, f"Naive per-file 95% CI {current.not_benign_sensitivity_ci95_low:.4f}–{current.not_benign_sensitivity_ci95_high:.4f}", f"TP/FN files = {int(current.not_benign_tp)}/{int(current.not_benign_fn)}", "C-file-level-exact-bundle", "File-level result only; higher sensitivity comes with lower specificity.", current.source],
        ["External", "Composite not-benign file rule", CURRENT_ID, int(current.n), int(current.n_melanoma), "Specificity", current.not_benign_specificity, f"Naive per-file 95% CI {current.not_benign_specificity_ci95_low:.4f}–{current.not_benign_specificity_ci95_high:.4f}", f"TN/FP files = {int(current.not_benign_tn)}/{int(current.not_benign_fp)}", "C-file-level-exact-bundle", "73/88 in the 107-file table; 69/84 after the near-identical-pair filter.", current.source],
        ["Paired external", "Single-pass minus archived TTA×8", "Same 107 labelled files", int(comparison.n_paired), 19, "Delta melanoma AUC", comparison.delta_auc_observed, f"Per-file bootstrap 95% CI {comparison.delta_auc_bootstrap_ci95_low:.4f}–{comparison.delta_auc_bootstrap_ci95_high:.4f}", f"Paired DeLong p={comparison.paired_delong_p:.4f}", "C-file-level-paired", "No evidence of superiority; clustering remains unresolved.", paths.relative(paths.PAPER_EXTERNAL_PAIRED)],
        ["Post-hoc", "Two-class probability readout", CURRENT_ID, int(posthoc.n), 19, "Readout melanoma AUC", posthoc.auc_readout, f"Naive per-file 95% CI {posthoc.auc_readout_ci95_low:.4f}–{posthoc.auc_readout_ci95_high:.4f}", f"Paired DeLong p={posthoc.paired_delong_p:.4f}", posthoc.evidence_grade, "Hypothesis only; threshold selected and tested on the same 107 labelled files.", paths.relative(paths.PAPER_POSTHOC_READOUT)],
    ]
    headers = [
        "Domain",
        "Evaluation",
        "Model / Configuration",
        "N files / images",
        "Positive-labelled files / images",
        "Metric",
        "Estimate",
        "Interval / Variability",
        "Counts / Test",
        "Evidence Grade",
        "Paper-safe Interpretation",
        "Source",
    ]
    end_row, _ = add_table(
        ws,
        headers,
        rows,
        start_row=start,
        name="KeyResultsTable",
        style="TableStyleMedium2",
    )
    style_evidence(ws, start, start + 1, end_row - 2)
    ws.freeze_panes = f"A{start + 1}"
    fit_columns(ws, max_width=54)


def build_external(
    wb: Workbook, external: pd.DataFrame, deduplicated: pd.DataFrame
) -> None:
    ws = wb.create_sheet("External Performance")
    ws.sheet_properties.tabColor = "5B9BD5"
    start = add_title(
        ws,
        "External Performance",
        "Primary table: 107 folder-labelled files (19 melanoma, 88 nevus), not proven independent cases. Intervals are naive per-file DeLong/Wilson intervals and are not cluster-adjusted.",
        end_col=26,
    )
    columns = [
        "model_id",
        "status",
        "n",
        "n_melanoma",
        "n_nevus",
        "auc_melanoma",
        "auc_melanoma_ci95_low",
        "auc_melanoma_ci95_high",
        "average_precision_melanoma",
        "brier_melanoma",
        "threshold_melanoma",
        "sensitivity_melanoma",
        "sensitivity_melanoma_ci95_low",
        "sensitivity_melanoma_ci95_high",
        "specificity_melanoma",
        "specificity_melanoma_ci95_low",
        "specificity_melanoma_ci95_high",
        "tp_melanoma",
        "fn_melanoma",
        "tn_melanoma",
        "fp_melanoma",
        "not_benign_sensitivity",
        "not_benign_specificity",
        "argmax_exact_accuracy",
        "source",
        "source_sha256",
    ]
    headers = [column.replace("_", " ").title() for column in columns]
    headers[2] = "N Labelled Files"
    headers[3] = "N Melanoma-labelled Files"
    headers[4] = "N Nevus-labelled Files"
    row, _ = add_table(
        ws,
        headers,
        dataframe_rows(external, columns),
        start_row=start,
        name="ExternalPerformanceTable",
        style="TableStyleMedium9",
    )
    row = add_section(
        ws,
        row,
        "Sensitivity analysis — one file retained per objectively near-identical pair",
        end_col=26,
    )
    dedup_columns = [
        "model_id",
        "removed_files",
        "n_files",
        "n_melanoma_labelled_files",
        "n_nevus_labelled_files",
        "auc_melanoma",
        "auc_melanoma_ci95_low",
        "auc_melanoma_ci95_high",
        "average_precision_melanoma",
        "brier_melanoma",
        "sensitivity_melanoma",
        "specificity_melanoma",
        "not_benign_sensitivity",
        "not_benign_specificity",
        "argmax_exact_accuracy",
        "warning",
    ]
    add_table(
        ws,
        [column.replace("_", " ").title() for column in dedup_columns],
        dataframe_rows(deduplicated, dedup_columns),
        start_row=row,
        name="ExternalFilteredSensitivityTable",
        style="TableStyleMedium9",
    )
    ws.freeze_panes = f"F{start + 1}"
    fit_columns(ws, max_width=42)


def build_internal(
    wb: Workbook, internal: pd.DataFrame, internal_recheck: pd.DataFrame
) -> None:
    ws = wb.create_sheet("Internal Performance")
    ws.sheet_properties.tabColor = "70AD47"
    start = add_title(
        ws,
        "Internal Performance",
        "Protocol-specific recorded estimates. The held-out row was rerun from current matrices and did not reproduce every metric exactly; both values are retained below.",
        end_col=len(internal.columns),
    )
    columns = list(internal.columns)
    headers = [column.replace("_", " ").title() for column in columns]
    row, _ = add_table(
        ws,
        headers,
        dataframe_rows(internal, columns),
        start_row=start,
        name="InternalPerformanceTable",
        style="TableStyleMedium4",
    )
    style_evidence(ws, start, start + 1, row - 2)
    row = add_section(ws, row, "Current-input held-out rerun", end_col=17)
    check_header = row
    check_columns = [
        "metric",
        "recorded_value",
        "recomputed_value",
        "absolute_difference",
        "exact_match",
        "status",
        "runtime",
        "warning",
    ]
    row, _ = add_table(
        ws,
        [column.replace("_", " ").title() for column in check_columns],
        dataframe_rows(internal_recheck, check_columns),
        start_row=row,
        name="InternalHeldoutReproductionTable",
        style="TableStyleMedium4",
    )
    style_status(ws, check_header, check_header + 1, row - 2)
    ws.freeze_panes = f"E{start + 1}"
    fit_columns(ws, max_width=48)


def build_model_versions(
    wb: Workbook, internal: pd.DataFrame, external: pd.DataFrame
) -> None:
    ws = wb.create_sheet("Model Versions")
    ws.sheet_properties.tabColor = "8064A2"
    start = add_title(
        ws,
        "Model Version Registry",
        "Current and historical lineages. Historical cross-validation values are for chronology only.",
        end_col=13,
    )
    current_cv = internal.iloc[0]
    current_ext, tta_ext = external.iloc[0], external.iloc[1]
    rows = [
        [CURRENT_ID, "Current", "V2S+CBAM", "Single-pass", "1280 deep + 18 handcrafted", "24d6d97423c08b2864be590b07cadff027d1210ae28d3e76d7d7c85292833fc3", current_cv.accuracy, current_cv.auc_macro_ovr, current_cv.macro_dice, current_ext.auc_melanoma, "Current application; technical file-level external analysis", "CV is optimistic; external labels, duplicates and clustering unresolved", current_ext.source],
        ["efficientnetv2s-cbam-ensemble+tta8 (2026-06-04)", "Former deployed", "V2S+CBAM", "TTA×8", "1280 deep + 18 handcrafted", "5b3023cf340d8369afa9c412b8bd9df7f242630bdf95803fce160150c79f7129", 0.9528, 0.9919, 0.9437, tta_ext.auc_melanoma, "Historical paired file-level comparison only", "Archived; CV optimistic; external clustering unresolved", tta_ext.source],
        ["V2S-1344 historical", "Irreproducible", "V2S historical", "Single-pass", "1344 deep + 18 handcrafted", None, 0.9470, 0.9909, 0.9370, None, "Project chronology only", "Extractor and feature CSV unavailable", "results/legacy/lgbm_v2s_log.txt"],
        ["ResNet18-CBAM-512 historical", "Legacy", "ResNet18+CBAM", "Single-pass", "512 deep + 18 handcrafted", None, 0.9205, 0.9819, 0.9038, None, "Historical baseline only", "Different lineage; image-level CV", "results/legacy/lgbm_entropy_log.txt"],
    ]
    headers = [
        "Model Identity",
        "Status",
        "Backbone",
        "Inference Variant",
        "Feature Dimension",
        "Bundle SHA-256",
        "Internal CV Accuracy",
        "Internal CV Macro AUC",
        "Internal CV Macro Dice",
        "External Melanoma AUC — file-level",
        "Allowed Use",
        "Main Caveat",
        "Source",
    ]
    add_table(
        ws,
        headers,
        rows,
        start_row=start,
        name="ModelVersionRegistry",
        style="TableStyleMedium5",
    )
    ws.freeze_panes = f"A{start + 1}"
    fit_columns(ws, max_width=48)


def build_dataset(wb: Workbook) -> None:
    ws = wb.create_sheet("Dataset Composition")
    ws.sheet_properties.tabColor = "A5A5A5"
    row = add_title(
        ws,
        "Dataset Composition",
        "File counts reconstructed from current artifacts. They are not patient or lesion counts; grouped independence remains unresolved.",
        end_col=12,
    )
    row = add_section(ws, row, "Internal source composition", end_col=5)
    source_rows = [
        ["ISIC 2019", 13390, "Reconstructed from filename prefixes"],
        ["HAM10000", 8954, "Reconstructed from filename prefixes"],
        ["ISIC 2017", 2130, "Reconstructed from filename prefixes"],
        ["Kaggle 9-class", 1429, "Reconstructed from filename prefixes"],
        ["Total", 25903, "Four-source harmonized file set"],
    ]
    row, _ = add_table(
        ws,
        ["Source", "Files", "Provenance Note"],
        source_rows,
        start_row=row,
        name="DatasetSources",
        style="TableStyleMedium3",
    )
    row = add_section(ws, row, "Class and split composition", end_col=6)
    class_rows = [
        ["Nevus", 8824, 1960, 2319, 13103],
        ["Melanoma", 3061, 911, 836, 4808],
        ["Atypical", 5317, 1368, 1307, 7992],
        ["Total", 17202, 4239, 4462, 25903],
    ]
    row, _ = add_table(
        ws,
        ["Harmonized Class", "Train", "Validation", "Test", "Total"],
        class_rows,
        start_row=row,
        name="DatasetClasses",
        style="TableStyleMedium3",
    )
    row = add_section(ws, row, "External pilot", end_col=6)
    external_rows = [
        ["Nevus", 88, 84, "Folder label; reference standard and lesion count not documented"],
        ["Melanoma", 19, 19, "Folder label; reference standard and lesion count not documented"],
        ["Total", 107, 103, "Technical file-level set; four near-identical derivatives removed in sensitivity analysis"],
    ]
    row, _ = add_table(
        ws,
        ["Folder-derived Class Label", "Primary-table Files", "After Near-identical-pair Filter", "Study-design Note"],
        external_rows,
        start_row=row,
        name="ExternalComposition",
        style="TableStyleMedium3",
    )
    row = add_section(ws, row, "Known data-provenance gaps", end_col=8)
    gaps = [
        ["Dataset fingerprint", "Not recorded", "Create a manifest and hashes for every source and merged split"],
        ["Patient ID", "Unresolved", "Required for patient-disjoint evaluation"],
        ["Lesion ID", "Unresolved", "Required for lesion-disjoint evaluation"],
        ["Dehair completeness", "25,630 files for 25,903 rows", "Investigate 273 missing or mismatched images"],
        ["Native-label mapping", "Partially documented", "Review source diagnosis to harmonized class mapping"],
        ["External near-identical files", "Four pairs objectively detected", "Resolve derivatives before any inferential analysis"],
        ["External repeated acquisitions", "At least one apparent sequence; metadata absent", "Recover patient and lesion identifiers and use grouped analysis"],
        ["External reference standard", "Not documented in available artifacts", "Document histopathology, follow-up or adjudication for every label"],
        ["External study design", "Not documented in available artifacts", "Document recruitment, device, ethics, consent and missing-data handling"],
    ]
    add_table(
        ws,
        ["Item", "Current Status", "Required Action"],
        gaps,
        start_row=row,
        name="DatasetGaps",
        style="TableStyleMedium7",
    )
    ws.freeze_panes = "A4"
    fit_columns(ws, max_width=58)


def build_claims(wb: Workbook) -> None:
    ws = wb.create_sheet("Claims Matrix")
    ws.sheet_properties.tabColor = "FFC000"
    start = add_title(
        ws,
        "Claims-to-Evidence Matrix",
        "Use the paper-safe wording verbatim unless stronger evidence is added.",
        end_col=7,
    )
    rows = [
        ["The current bundle produced file-level external melanoma AUC 0.9318", "C", "Allowed with strict caveat", "Across 107 folder-labelled files, file-level AUC was 0.932 (naive 95% CI 0.864–0.999); independence and reference standard are unresolved", "External exact-bundle CSV + verification audit"],
        ["The external set contains 107 independent cases or patients", "None", "Not supported", "Use ‘107 labelled files’; do not infer patient, lesion or case count", "Verification audit"],
        ["Threshold 0.188 flagged 12/19 melanoma-labelled files", "C", "Allowed as file-level arithmetic", "Report with 86/88 nevus-labelled files below threshold, naive Wilson intervals and clustering caveat", "External exact-bundle CSV"],
        ["The not-benign rule flagged 17/19 melanoma-labelled files", "C", "Allowed as file-level arithmetic", "Also report 73/88 file-level specificity; after near-identical-pair filtering it is 69/84", "External CSV + duplicate sensitivity"],
        ["External confidence intervals quantify patient-level uncertainty", "None", "Not supported", "Intervals are naive per-file intervals and are not cluster-adjusted", "Verification audit"],
        ["Single-pass is superior to TTA×8", "C", "Not supported", "Numerically higher file-level AUC only; delta CI crosses zero and paired DeLong p=0.305", "Paired comparison CSV"],
        ["Model accuracy is 94.24%", "D", "Misleading alone", "Specify recorded image-level CV with partially in-sample CNN representations and globally pre-fit scaler", "Internal CV log and source audit"],
        ["Held-out internal melanoma AUC is exactly reproducible at 0.8928", "B", "Not supported", "Recorded AUC 0.892755; current-input rerun 0.893546. Report both until a fully versioned rerun is frozen", "Internal reproduction check"],
        ["Handcrafted features add no value to the current CNN", "E", "Not established", "Deep+handcrafted factorial conditions are incomplete", "Handcraft factorial artifacts"],
        ["The atypical class is harmful", "E", "Not established", "Two-class readout is a post-hoc hypothesis requiring separate calibration and validation", "Post-hoc readout CSV"],
        ["The training pipeline is fully reproducible", "D/E", "False", "Current ONNX binaries are available and hash-verifiable; exact training regeneration is not reproducible", "Model manifest"],
        ["The system is clinically safe or diagnostic-ready", "None", "Prohibited", "Research prototype; no clinical safety claim", "Study gaps"],
    ]
    headers = ["Potential Claim", "Evidence Grade", "Verdict", "Paper-safe Wording", "Evidence Source"]
    end_row, _ = add_table(
        ws,
        headers,
        rows,
        start_row=start,
        name="ClaimsEvidenceMatrix",
        style="TableStyleMedium6",
    )
    style_evidence(ws, start, start + 1, end_row - 2)
    ws.freeze_panes = f"A{start + 1}"
    fit_columns(ws, max_width=72)


def build_gaps(wb: Workbook) -> None:
    ws = wb.create_sheet("Study Gaps")
    ws.sheet_properties.tabColor = "C00000"
    start = add_title(
        ws,
        "Study Gaps and Next Experiments",
        "Priority order for converting the current evidence pack into a defensible manuscript.",
        end_col=7,
    )
    rows = [
        ["P0", "Recover patient and lesion identifiers", "Study integrity", "File-level rows cannot establish independent clinical units or support cluster-adjusted inference", "Blocking", "Metadata audit, deterministic group mapping and grouped analysis"],
        ["P0", "Resolve external derivatives and acquisition clusters", "Study integrity", "Four near-identical pairs are confirmed and at least one repeated-acquisition sequence is apparent", "Blocking", "Curated cohort manifest with one declared unit per analysis"],
        ["P0", "Freeze dataset and native-label mapping", "Reproducibility", "Current merged dataset has no recorded fingerprint", "Blocking", "Dataset manifest, file hashes and mapping review"],
        ["P0", "Document external reference standard and ethics", "Clinical reporting", "Histopathology/follow-up, recruitment, approvals and consent were not found in available artifacts", "Blocking", "Cohort description aligned with the target journal's reporting guidance"],
        ["P1", "Train representation without evaluation images", "Internal validity", "Removes representation leakage", "High", "Lesion-disjoint training and held-out extraction"],
        ["P1", "Refit all preprocessing within CV folds", "Internal validity", "The descriptive CV scaler is currently fit before folds", "High", "Leakage-free grouped cross-validation pipeline"],
        ["P1", "Independent threshold calibration", "Operating point", "Current thresholds are inherited; post-hoc alternatives reuse the test set", "High", "Calibration set plus untouched external test"],
        ["P1", "Second external cohort", "Generalization", "Current set has only 19 melanoma-labelled files and fewer independent lesions may be present", "High", "Power calculation and independent grouped cohort"],
        ["P2", "Complete Deep + ABCD factorial", "Scientific hypothesis", "Marginal handcrafted-feature value is not measured", "Medium", "Paired effects, confidence intervals, FDR"],
        ["P2", "Validate two-class readout", "Scientific hypothesis", "Current result is post-hoc and resubstitution-biased", "Medium", "Predefined threshold and untouched test set"],
        ["P2", "Analyze segmentation failures and abstention", "Safety mechanism", "One of two not-benign melanoma-file misses had 1.3% mask coverage; no association test is established", "Medium", "Predefined quality gate and grouped failure analysis"],
        ["P3", "Measure device latency, RAM, energy and failures", "Systems paper", "On-device claims currently lack performance measurements", "Medium", "Repeated measurements on target tablet"],
    ]
    add_table(
        ws,
        ["Priority", "Task", "Evidence Area", "Why It Matters", "Blocking Level", "Expected Deliverable"],
        rows,
        start_row=start,
        name="StudyGapPlan",
        style="TableStyleMedium10",
    )
    ws.freeze_panes = f"A{start + 1}"
    fit_columns(ws, max_width=60)


def build_artifacts(wb: Workbook) -> None:
    ws = wb.create_sheet("Artifact Audit")
    ws.sheet_properties.tabColor = "7F6000"
    start = add_title(
        ws,
        "Artifact Audit",
        "Prevents reuse of attractive but stale or model-mismatched numbers.",
        end_col=6,
    )
    rows = [
        [paths.relative(paths.EXTERNAL_PARITY_CSV), "Current file-level source", "Exact-bundle arithmetic", "Folder labels, four near-identical pairs, clustering and reference standard unresolved", "Use only with file-level grade C caveats and sensitivity table"],
        [paths.relative(paths.PAPER_DEDUPLICATED_SENS), "Current sensitivity analysis", "Effect of four objectively near-identical pairs", "Still not patient/lesion-disjoint", "Report alongside the 107-file table"],
        [f"{paths.relative(paths.INTERNAL_DECOMPOSITION)} — row 1a", "Recorded; not exactly reproduced", "Internal held-out configuration result", "Small current-matrix drift and grouping provenance unresolved", "Preserve recorded and rerun values together"],
        [paths.relative(paths.PAPER_INTERNAL_REPRODUCTION), "Current verification", "Held-out rerun and exact input hashes", "Historical run lacked matching hashes/runtime manifest", "Use to document reproducibility gap"],
        [paths.relative(paths.INTERNAL_CV_LOG), "Descriptive source-confirmed", "Current CV chronology", "Partially in-sample representations and scaler fit before folds", "Never call independent validation"],
        [paths.relative(paths.PAPER_VERIFICATION_AUDIT), "Current verification", "Calculation and provenance check register", "Does not create missing clinical metadata", "Use as audit trail"],
        [paths.relative(paths.TTA8_EXTERNAL_CSV), "Archive", "Paired historical comparison", "Former model path", "Use only as archived comparator"],
        [f"{paths.relative(paths.DOCS_CLINICAL)}/*.png", "Do not cite", "Historical visualization", "ResNet18 lineage from 2026-06-05", "Regenerate on a valid current protocol"],
        [paths.relative(paths.DOCS_SLIDES / "comparaison_modeles.xlsx"), "Do not cite", "Project chronology", "Mixes lineages, shortcuts and evaluation regimes", "Rebuild from this workbook"],
        [paths.relative(paths.DOCS_SLIDES / "etat_art.xlsx"), "Verify first", "Literature notes", "Time-sensitive and heterogeneous claims", "Recheck every row against primary sources"],
        [f"{paths.relative(paths.DOCS_ANALYSES / 'ablations.xlsx')} / {paths.relative(paths.DOCS_ANALYSES / 'ablations.md')}", "Do not cite", "Exploratory ablation summary", "External atypical snapshot predates current exact-bundle CSV", "Use paper pack instead"],
        [f"{paths.relative(paths.RESULTS_ABLATION_ATYPICAL)}/ablation_atypical*", "Stale external section", "Hypothesis generation", "Old probability snapshot and same-set threshold selection", "Rerun with separated calibration/test"],
        [paths.relative(paths.ABLATION_HANDCRAFT_FACTORIAL), "Incomplete", "Handcrafted-only signal", "Deep and external conditions missing", "Complete factorial before a marginal-value claim"],
        [f"{paths.relative(paths.RESULTS_FEATURES)}/probing_handcraft.*", "Version mismatch risk", "Redundancy hypothesis", "Log says 512-d while current script targets V2S", "Rerun with input hashes"],
    ]
    add_table(
        ws,
        ["Artifact", "Status", "Potential Use", "Problem", "Required Action"],
        rows,
        start_row=start,
        name="ArtifactAuditTable",
        style="TableStyleMedium8",
    )
    ws.freeze_panes = f"A{start + 1}"
    fit_columns(ws, max_width=68)


def build_figures(wb: Workbook) -> None:
    ws = wb.create_sheet("Figures")
    ws.sheet_properties.tabColor = "4472C4"
    add_title(
        ws,
        "Paper Figures",
        "Embedded PNG previews. External panels are descriptive per-file plots and do not resolve near-duplicates or clustering. Use SVG sources for layout.",
        end_col=12,
    )
    entries = [
        (
            "Figure 1 — Internal evaluation gap across protocols",
            FIGURES / "internal_protocol_gap.png",
            "Vector source: docs/paper/figures/internal_protocol_gap.svg",
            4,
        ),
        (
            "Figure 2 — External file-level ROC: current single-pass versus archived TTA×8",
            FIGURES / "external_roc_singlepass_vs_tta8.png",
            "Vector source: docs/paper/figures/external_roc_singlepass_vs_tta8.svg",
            33,
        ),
        (
            "Figure 3 — External file-level operating points",
            FIGURES / "external_operating_points.png",
            "Vector source: docs/paper/figures/external_operating_points.svg",
            67,
        ),
    ]
    for title, path, source, row in entries:
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=12)
        ws.cell(row, 1, title)
        ws.cell(row, 1).font = Font(size=13, bold=True, color=WHITE)
        ws.cell(row, 1).fill = PatternFill("solid", fgColor=TEAL)
        ws.merge_cells(start_row=row + 1, start_column=1, end_row=row + 1, end_column=12)
        ws.cell(row + 1, 1, source)
        ws.cell(row + 1, 1).font = Font(italic=True, color="666666")
        image = Image(path)
        target_width = 820
        image.height = int(image.height * target_width / image.width)
        image.width = target_width
        ws.add_image(image, f"A{row + 2}")
    for column in range(1, 13):
        ws.column_dimensions[get_column_letter(column)].width = 12
    ws.sheet_view.showGridLines = False
    ws.page_setup.orientation = "landscape"


def build_charts(wb: Workbook, ablation: pd.DataFrame, external: pd.DataFrame) -> None:
    """Live Excel charts, driven by data tables written on this same sheet.

    Both value axes start at zero. A truncated axis would make the ablation
    deltas look like something, and the whole point is that they are not.
    """
    ws = wb.create_sheet("Charts")
    ws.sheet_properties.tabColor = TEAL
    row = add_title(
        ws,
        "Charts",
        "Native Excel charts. They recalculate from the tables on this sheet, which are extracted from the "
        "same sources as the raw layer. Value axes start at zero on purpose.",
        end_col=10,
    )

    # ── 1. Ablation ────────────────────────────────────────────────────────
    row = add_section(ws, row, "Melanoma OvR AUC by ablation condition", end_col=10)
    data = ablation.sort_values("auc_melanoma_ovr", ascending=False)
    head = row
    for offset, label in enumerate(["Condition", "AUC melanoma", "CI minus", "CI plus"], start=1):
        cell = ws.cell(head, offset, label)
        cell.font = Font(bold=True, color=WHITE)
        cell.fill = PatternFill("solid", fgColor=BLUE)
    row += 1
    first = row
    for _, record in data.iterrows():
        auc = float(record["auc_melanoma_ovr"])
        # ``condition`` is already the full label ("Deep+A", "A+B+C+D"); do not
        # prefix it again.
        ws.cell(row, 1, str(record["condition"]))
        ws.cell(row, 2, auc).number_format = "0.0000"
        ws.cell(row, 3, round(auc - float(record["auc_melanoma_ci95_low"]), 6))
        ws.cell(row, 4, round(float(record["auc_melanoma_ci95_high"]) - auc, 6))
        row += 1
    last = row - 1

    chart = BarChart()
    chart.type = "col"
    chart.title = "Melanoma OvR AUC by condition (internal, partially in-sample representation)"
    chart.y_axis.title = "AUC"
    chart.x_axis.title = "Condition"
    chart.y_axis.scaling.min = 0
    chart.y_axis.scaling.max = 1
    chart.height, chart.width = 10, 30
    chart.legend = None
    chart.add_data(Reference(ws, min_col=2, min_row=head, max_row=last), titles_from_data=True)
    chart.set_categories(Reference(ws, min_col=1, min_row=first, max_row=last))
    chart.series[0].errBars = ErrorBars(
        errDir="y",
        errValType="cust",
        plus=NumDataSource(numRef=NumRef(f="'Charts'!$D${first}:$D${last}")),
        minus=NumDataSource(numRef=NumRef(f="'Charts'!$C${first}:$C${last}")),
    )
    ws.add_chart(chart, f"F{head}")

    row = max(row, head + 22) + 2

    # ── 2. External model comparison ───────────────────────────────────────
    row = add_section(ws, row, "Deployed single-pass vs archived TTA x8, external file level", end_col=10)
    metrics = [
        ("Melanoma AUC", "auc_melanoma"),
        ("Sensitivity @0.188", "sensitivity_melanoma"),
        ("Specificity @0.188", "specificity_melanoma"),
        ("Not-benign sensitivity", "not_benign_sensitivity"),
        ("Not-benign specificity", "not_benign_specificity"),
    ]
    current = external.iloc[0]
    archived = external.iloc[1] if len(external) > 1 else None
    head2 = row
    for offset, label in enumerate(["Metric", "Single-pass (deployed)", "TTA x8 (archived)"], start=1):
        cell = ws.cell(head2, offset, label)
        cell.font = Font(bold=True, color=WHITE)
        cell.fill = PatternFill("solid", fgColor=BLUE)
    row += 1
    first2 = row
    for label, column in metrics:
        ws.cell(row, 1, label)
        ws.cell(row, 2, round(float(current[column]), 6)).number_format = "0.0000"
        if archived is not None:
            ws.cell(row, 3, round(float(archived[column]), 6)).number_format = "0.0000"
        row += 1
    last2 = row - 1

    chart2 = BarChart()
    chart2.type = "col"
    chart2.grouping = "clustered"
    chart2.title = "External file-level metrics, 107 labelled files (19 melanoma)"
    chart2.y_axis.title = "Value"
    chart2.y_axis.scaling.min = 0
    chart2.y_axis.scaling.max = 1
    chart2.height, chart2.width = 10, 24
    chart2.add_data(Reference(ws, min_col=2, max_col=3, min_row=head2, max_row=last2), titles_from_data=True)
    chart2.set_categories(Reference(ws, min_col=1, min_row=first2, max_row=last2))
    ws.add_chart(chart2, f"F{head2}")

    row = max(row, head2 + 22) + 1
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=10)
    ws.cell(row, 1,
            "Reading note: the single-pass / TTA x8 difference is not a demonstrated superiority "
            "(paired DeLong p = 0.305). Ablation deltas above must be read against a CI width near 0.0037.")
    ws.cell(row, 1).alignment = Alignment(wrap_text=True, vertical="top")
    ws.cell(row, 1).fill = PatternFill("solid", fgColor=YELLOW)
    row += 2
    add_back_link(ws, row, "README")

    for column, width in (("A", 26), ("B", 22), ("C", 20), ("D", 12)):
        ws.column_dimensions[column].width = width


def build_sources(wb: Workbook) -> None:
    ws = wb.create_sheet("Source Registry")
    ws.sheet_properties.tabColor = "264478"
    start = add_title(
        ws,
        "Source Registry",
        "File-level provenance for every aggregate source used in this workbook.",
        end_col=5,
    )
    sources = [
        (paths.APP_MODELS / "model_manifest.json", "Current bundle identity and model contracts"),
        (paths.EXTERNAL_PARITY_CSV, "Current external per-file predictions"),
        (paths.TTA8_EXTERNAL_CSV, "Archived TTA×8 per-file predictions"),
        (paths.EXTERNAL_PARITY_GATE_CSV, "Rerun exact-bundle external parity gate"),
        (paths.INTERNAL_DECOMPOSITION, "Recorded internal held-out and validity decomposition"),
        (paths.INTERNAL_CV_LOG, "Current configuration image-level CV"),
        (GENERATED / "external_performance.csv", "External aggregate statistics"),
        (GENERATED / "external_paired_comparison.csv", "Paired AUC comparison"),
        (GENERATED / "external_posthoc_two_class_readout.csv", "Post-hoc two-class readout hypothesis"),
        (GENERATED / "external_near_duplicate_audit.csv", "Near-identical external file audit without identifiers"),
        (GENERATED / "external_deduplicated_sensitivity.csv", "One-file-per-near-identical-pair sensitivity analysis"),
        (GENERATED / "external_deduplicated_paired_comparison.csv", "Filtered paired model comparison"),
        (GENERATED / "internal_performance.csv", "Internal aggregate statistics"),
        (GENERATED / "internal_reproduction_check.csv", "Current-input internal held-out rerun"),
        (GENERATED / "performance_registry.csv", "Long-form evidence registry"),
        (GENERATED / "verification_audit.csv", "Complete factual verification register"),
        (PAPER / "verification_report.md", "Human-readable verification verdict"),
        (paths.ROOT / "derm" / "analysis" / "render_paper_resources.py", "Paper table and figure generator"),
        (paths.ROOT / "derm" / "analysis" / "verify_paper_resources.py", "Independent verification generator"),
        (paths.ROOT / "derm" / "analysis" / "build_paper_workbook.py", "Workbook generator"),
    ]
    # Raw-layer sources, minus the ones already listed above.
    listed = {path.resolve() for path, _ in sources}
    sources += [
        (spec["path"], spec["role"])
        for spec in RAW_SHEETS
        if spec["path"].resolve() not in listed
    ]

    rows = []
    for path, role in sources:
        rows.append(
            [
                relative(path),
                file_sha256(path),
                path.stat().st_size,
                role,
                "Present",
            ]
        )
    add_table(
        ws,
        ["Source File", "SHA-256", "Bytes", "Role", "Status"],
        rows,
        start_row=start,
        name="WorkbookSourceRegistry",
        style="TableStyleMedium2",
    )
    ws.freeze_panes = f"A{start + 1}"
    fit_columns(ws, max_width=78)


# ── Raw layer ───────────────────────────────────────────────────────────────
#
# Each entry becomes one sheet holding a source table verbatim: same rows, same
# columns, same order. ``caveat`` is what a reader must know before reusing the
# numbers, and is printed on the sheet itself rather than only in the index, so
# a single exported tab still carries its own limits.

RAW_PURPOSE = "Verbatim source table. Rows, columns and order are unchanged."

RAW_SHEETS: list[dict[str, Any]] = [
    {
        "sheet": "Raw External Predictions",
        "path": paths.EXTERNAL_PARITY_CSV,
        "role": "Per-file predictions of the deployed bundle on the external set",
        "caveat": (
            "107 folder-labelled files, not 107 proven independent patients or lesions. "
            "Four near-identical pairs are retained here; see Verification Audit. "
            "p_nevus + p_mel + p_atyp = 1; coverage is the U-Net mask fraction."
        ),
    },
    {
        "sheet": "Raw External File Manifest",
        "path": paths.EXTERNAL_RAW_MANIFEST,
        "role": "SHA-256 and byte size of each external file",
        "caveat": (
            "Labels are derived from the containing folder. No histopathology, "
            "adjudication or reference standard is documented for these files."
        ),
    },
    {
        "sheet": "Raw External Recalibration",
        "path": paths.RECALIBRATION_EXTERNAL_CSV,
        "role": "Melanoma threshold recalibrated on the external set",
        "caveat": (
            "Thresholds are fitted and evaluated on the same 107 files. "
            "Hypothesis-generating only; column headers are French in the source."
        ),
    },
    {
        "sheet": "Raw ISIC2019 Predictions",
        "path": paths.ISIC2019_PREDICTIONS,
        "role": "Held-out ISIC2019/test predictions with raw-file digests",
        "caveat": (
            "Primary held-out population: 2 239 images absent from train and from "
            "validation, verified by identifier and by SHA-256. The melanoma "
            "threshold was selected on validation, so threshold-level numbers are "
            "descriptive and not independently calibrated."
        ),
    },
    {
        "sheet": "Raw ISIC2019 Metrics",
        "path": paths.ISIC2019_METRICS,
        "role": "Global, per-class, calibration and threshold metrics",
        "caveat": (
            "Fully recomputable from Raw ISIC2019 Predictions. Intervals are naive "
            "image-level intervals; patient and lesion clustering is unavailable."
        ),
    },
    {
        "sheet": "Raw ISIC2019 Audit",
        "path": paths.ISIC2019_AUDIT,
        "role": "Cohort composition and train/validation overlap checks",
        "caveat": "0 shared identifier and 0 shared SHA-256 with train and with train+valid.",
    },
    {
        "sheet": "Raw Internal Predictions",
        "path": paths.INTERNAL_PREDICTIONS,
        "role": "Per-observation held-out predictions for condition 1a",
        "caveat": (
            "Heads fitted on train+valid, predictions on test. group_key falls back "
            "to the file because patient_id and lesion_id are unavailable. The OOF "
            "outputs of C0/C1/C2 were not preserved historically."
        ),
    },
    {
        "sheet": "Raw Internal Decomposition",
        "path": paths.INTERNAL_DECOMPOSITION,
        "role": "Aggregate metrics per internal condition (1a, C0, C1, C2, C3)",
        "caveat": (
            "C0/C1/C2 are image-level CV with partially in-sample CNN "
            "representations and a scaler fit before folds: optimistic. "
            "C3 is blocked, it needs lesion-disjoint CNN retraining on GPU."
        ),
    },
    {
        "sheet": "Raw Experiment Registry",
        "path": paths.INTERNAL_EXPERIMENT_REGISTRY,
        "role": "Protocol definition and caveats of each internal condition",
        "caveat": "Read this before comparing any two internal numbers.",
    },
    {
        "sheet": "Raw Handcraft Ablation",
        "path": paths.ABLATION_HANDCRAFT_FACTORIAL,
        "role": "Factorial ABCD ablation, 59 columns per condition",
        "caveat": (
            "Grid incomplete: 18 of 31 conditions. The 15 handcraft-only conditions "
            "are complete, and 3 of the 16 Deep conditions have run (Deep, Deep+A, "
            "Deep+B). The remaining Deep conditions, the external front and the FDR "
            "step are still missing, so no corrected significance can be claimed."
        ),
    },
    {
        "sheet": "Raw Atypical Ablation",
        "path": paths.ABLATION_ATYPICAL_CSV,
        "role": "Learned atypical class versus binary readout, specificity-matched",
        "caveat": (
            "The external B_retrain arm is blocked: the 512-d baseline features of "
            "the 107 files were not available for re-extraction."
        ),
    },
    {
        "sheet": "Raw ONNX Feature Splits",
        "path": paths.CURRENT_ONNX_IMPORTANCE,
        "role": "Split frequency of the 1 298 inputs in both deployed ONNX heads",
        "caveat": (
            "Split counts only. Not gain, not SHAP, not a causal effect, and not "
            "evidence that a feature carries clinical meaning."
        ),
    },
    {
        "sheet": "Raw Legacy Feature Ranks",
        "path": paths.FEATURE_IMPORTANCE_CSV,
        "role": "Feature importance of the archived 512-d ResNet18 lineage",
        "caveat": "Historical lineage, not the deployed bundle. Kept for comparison only.",
    },
    {
        "sheet": "Raw Performance Registry",
        "path": GENERATED / "performance_registry.csv",
        "role": "Long-form registry of every reported estimate",
        "caveat": "Each row carries its own model identity, protocol, population and evidence grade.",
    },
    {
        "sheet": "Raw Image Metadata",
        "path": paths.ARTIFACTS / "image_metadata.csv",
        "role": "Deterministic mapping row_index <-> file <-> source/split/class",
        "caveat": (
            "This is the key that makes row_index in the prediction sheets "
            "interpretable. lesion_id and patient_id are empty: that absence is the "
            "P0 limitation of the whole study, since it forbids grouped splits."
        ),
    },
    {
        "sheet": "Raw Dataset Manifest",
        "path": paths.ARTIFACTS / "internal_dataset_manifest.csv",
        "role": "SHA-256 and byte size of each of the 25 903 internal images",
        "caveat": (
            "Snapshot of the files present on disk now. It does not prove they match "
            "the historical training snapshot byte for byte."
        ),
    },
    {
        "sheet": "Raw Handcraft Features",
        "path": paths.ARTIFACTS / "features.csv",
        "role": "The 18 handcrafted ABCD descriptors, model input, one row per image",
        "caveat": (
            "Row order matches Raw Image Metadata. These 18 values plus a 1280-d CNN "
            "embedding form the 1 298 inputs of each head. The embedding matrices "
            "themselves are too large for a spreadsheet and stay transfer-only."
        ),
    },
]


LINK_BLUE = "0563C1"


def link_to_sheet(cell, sheet: str, text: str | None = None) -> None:
    """Turn ``cell`` into an internal link to ``sheet``!A1.

    ``location`` rather than ``target`` is what makes Excel treat the hyperlink
    as internal; a target would send the reader out of the workbook.
    """
    if text is not None:
        cell.value = text
    cell.hyperlink = Hyperlink(ref=cell.coordinate, location=f"'{sheet}'!A1")
    cell.font = Font(color=LINK_BLUE, underline="single")


def add_back_link(ws, row: int, to: str = "Raw Data Index") -> int:
    cell = ws.cell(row, 1)
    link_to_sheet(cell, to, f"↑ {to}")
    return row + 1


def add_raw_table(
    ws,
    headers: list[str],
    rows: list[list[Any]],
    *,
    start_row: int,
    name: str,
) -> int:
    """Write a large table fast: style the header, leave data cells unstyled.

    ``add_table`` builds an Alignment and a Border per cell, which is fine for a
    30-row reading view and far too slow for 25 903 rows. Banding is delegated to
    the Excel table style instead of per-cell fills.
    """
    for offset, header in enumerate(headers, start=1):
        cell = ws.cell(start_row, offset, header)
        cell.font = Font(bold=True, color=WHITE)
        cell.fill = PatternFill("solid", fgColor=BLUE)
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[start_row].height = 30

    for row_offset, values in enumerate(rows, start=start_row + 1):
        for col_offset, value in enumerate(values, start=1):
            ws.cell(row_offset, col_offset, value)

    end_row = start_row + len(rows)
    if rows:
        ref = f"A{start_row}:{get_column_letter(len(headers))}{end_row}"
        table = Table(displayName=table_name(name), ref=ref)
        table.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2",
            showFirstColumn=False,
            showLastColumn=False,
            showRowStripes=True,
            showColumnStripes=False,
        )
        ws.add_table(table)
    return end_row


def fit_columns_sampled(
    ws, header_row: int, last_row: int, *, sample: int = 200, max_width: int = 44
) -> None:
    """Width from the header plus the first ``sample`` data rows.

    Scanning every cell of a 25 903-row sheet costs far more than it improves the
    layout, and one outlier row should not widen a column for the whole sheet.
    ``last_row`` caps the scan: iterating past the real data would materialize
    empty cells and inflate the sheet's stored dimension.
    """
    stop = min(header_row + sample, last_row)
    for column_cells in ws.iter_cols(min_row=header_row, max_row=stop):
        letter = get_column_letter(column_cells[0].column)
        width = 10
        for cell in column_cells:
            if cell.value is not None:
                width = max(width, min(len(str(cell.value)) + 2, max_width))
        ws.column_dimensions[letter].width = width


def build_raw_sheet(wb: Workbook, spec: dict[str, Any]) -> dict[str, Any]:
    path: Path = spec["path"]
    # Default type inference on purpose: numeric columns must land in Excel as
    # numbers, otherwise nothing in the raw layer can be recomputed in place.
    # Verified against the sources that no column carries leading zeros or
    # identifier-like digits that inference would mangle; the only differences
    # are textual (trailing zeros, int-as-float), not numeric.
    data = pd.read_csv(path)
    headers = [str(column) for column in data.columns]
    rows = [[clean(value) for value in record] for record in data.itertuples(index=False)]

    ws = wb.create_sheet(spec["sheet"])
    ws.sheet_properties.tabColor = TEAL
    row = add_title(
        ws,
        spec["sheet"].replace("Raw ", "Raw data — "),
        f"{spec['role']}. {RAW_PURPOSE}",
        end_col=min(len(headers), 12),
    )
    row = add_key_value(ws, row, "Source", relative(path))
    row = add_key_value(ws, row, "SHA-256", file_sha256(path))
    row = add_key_value(ws, row, "Rows x columns", f"{len(rows):,} x {len(headers)}")
    row = add_key_value(ws, row, "Read before reuse", spec["caveat"])
    row = add_back_link(ws, row)
    row += 1

    header_row = row
    end_row = add_raw_table(ws, headers, rows, start_row=header_row, name=spec["sheet"])
    ws.freeze_panes = f"A{header_row + 1}"
    fit_columns_sampled(ws, header_row, end_row)
    ws.column_dimensions["A"].width = max(ws.column_dimensions["A"].width or 10, 22)
    return {
        "sheet": spec["sheet"],
        "path": path,
        "role": spec["role"],
        "caveat": spec["caveat"],
        "rows": len(rows),
        "columns": len(headers),
        "last_row": end_row,
    }


def build_raw_model_contract(wb: Workbook, manifest: dict[str, Any]) -> dict[str, Any]:
    """Flatten model_manifest.json: the runtime contract the app actually enforces."""
    ws = wb.create_sheet("Raw Model Contract")
    ws.sheet_properties.tabColor = TEAL
    manifest_path = paths.APP_MODELS / "model_manifest.json"
    row = add_title(
        ws,
        "Raw data — Model Contract",
        "Flattened runtime manifest: bundle identity, preprocessing, decision policy and per-component ONNX contracts. "
        "The application hashes every binary against this manifest before creating its inference session.",
        end_col=8,
    )
    row = add_key_value(ws, row, "Source", relative(manifest_path))
    row = add_key_value(ws, row, "SHA-256", file_sha256(manifest_path))
    row += 1

    row = add_section(ws, row, "Bundle identity", end_col=8)
    for key in ("schema_version", "bundle_id", "release_date", "status", "variant",
                "bundle_sha256", "bundle_hash_algorithm", "ensemble"):
        if key in manifest:
            row = add_key_value(ws, row, key, manifest[key])
    row = add_key_value(ws, row, "classes", ", ".join(manifest.get("classes", [])))
    row += 1

    row = add_section(ws, row, "Preprocessing", end_col=8)
    for key, value in manifest.get("preprocessing", {}).items():
        row = add_key_value(ws, row, key, ", ".join(map(str, value)) if isinstance(value, list) else value)
    row += 1

    policy = manifest.get("decision_policy", {})
    row = add_section(ws, row, "Decision policy", end_col=8)
    for key, value in policy.items():
        if key == "thresholds":
            for name, threshold in value.items():
                row = add_key_value(
                    ws, row, f"threshold[{name}]", threshold,
                    note="Legacy Youden value; recalibration for the single-pass bundle is still pending.",
                )
        else:
            row = add_key_value(ws, row, key, ", ".join(map(str, value)) if isinstance(value, list) else value)
    row += 1

    row = add_section(ws, row, "Components", end_col=8)
    components = manifest.get("components", [])
    header_row = row
    row, _ = add_table(
        ws,
        ["Name", "Role", "Architecture", "SHA-256", "Input", "Output", "Opset", "Exporter"],
        [
            [
                component.get("name"),
                component.get("role"),
                component.get("architecture"),
                component.get("sha256"),
                component.get("input"),
                component.get("output"),
                component.get("onnx_opset"),
                component.get("exporter"),
            ]
            for component in components
        ],
        start_row=row,
        name="RawModelComponents",
    )
    fit_columns(ws, max_width=62)
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 46
    return {
        "sheet": "Raw Model Contract",
        "path": manifest_path,
        "role": "Runtime contract of the deployed bundle",
        "caveat": (
            "Hash-verifiable identity of the five deployed ONNX files. It does not make "
            "the training pipeline reproducible: dataset, commit and checkpoint "
            "provenance are incomplete."
        ),
        "rows": len(components),
        "columns": 8,
        "last_row": header_row,
    }


def build_raw_index(wb: Workbook, entries: list[dict[str, Any]]) -> None:
    """Index sheet: what is raw, what is aggregated, and which one wins."""
    ws = wb.create_sheet("Raw Data Index")
    ws.sheet_properties.tabColor = NAVY
    row = add_title(
        ws,
        "Raw Data Index",
        "The technical layer of this workbook. Every sheet below is a verbatim source table, "
        "carried so that each reported number can be recomputed from the workbook alone.",
        end_col=7,
    )

    row = add_section(ws, row, "Precedence rule", end_col=7)
    for note in (
        "Where an aggregate sheet and a raw sheet disagree, the raw sheet is authoritative and the aggregate view must be corrected.",
        "Raw sheets carry per-file identifiers and per-image SHA-256 digests; the aggregate sheets deliberately do not.",
        "No raw images are included. The two 25 903 x 1280 CNN embedding matrices do not fit a spreadsheet and remain transfer-only artifacts listed in HANDOFF.md.",
        "Nothing in this layer upgrades the evidence grade of a result. A complete raw table can still describe an incomplete experiment.",
        "Numeric columns are stored as numbers so they can be recomputed in place. Spreadsheet storage keeps about 16 significant digits, so a few probabilities below 1e-4 differ from the CSV in their last digit (relative error near 1e-16). For bit-exact values, use the source CSV listed against each sheet.",
    ):
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=7)
        ws.cell(row, 1, "• " + note)
        ws.cell(row, 1).alignment = Alignment(wrap_text=True, vertical="top")
        ws.cell(row, 1).fill = PatternFill("solid", fgColor="F8F9FA")
        ws.row_dimensions[row].height = 30
        row += 1
    row += 1

    row = add_section(ws, row, "Raw sheets", end_col=7)
    header_row = row
    row, _ = add_table(
        ws,
        ["Sheet", "Contents", "Rows", "Columns", "Source File", "SHA-256", "Read before reuse"],
        [
            [
                entry["sheet"],
                entry["role"],
                entry["rows"],
                entry["columns"],
                relative(entry["path"]),
                file_sha256(entry["path"]),
                entry["caveat"],
            ]
            for entry in entries
        ],
        start_row=row,
        name="RawDataIndexTable",
        style="TableStyleMedium4",
    )
    # Column A of the table becomes a clickable jump to each sheet: 31 tabs is
    # well past what a tab bar makes findable.
    for offset, entry in enumerate(entries, start=1):
        link_to_sheet(ws.cell(header_row + offset, 1), entry["sheet"])
    ws.freeze_panes = f"A{header_row + 1}"
    fit_columns(ws, max_width=54)
    ws.column_dimensions["A"].width = 27
    ws.column_dimensions["B"].width = 46
    ws.column_dimensions["G"].width = 62


def build_workbook() -> Path:
    required = [
        GENERATED / "external_performance.csv",
        GENERATED / "external_paired_comparison.csv",
        GENERATED / "external_posthoc_two_class_readout.csv",
        GENERATED / "external_near_duplicate_audit.csv",
        GENERATED / "external_deduplicated_sensitivity.csv",
        GENERATED / "external_deduplicated_paired_comparison.csv",
        GENERATED / "internal_performance.csv",
        GENERATED / "internal_reproduction_check.csv",
        GENERATED / "performance_registry.csv",
        GENERATED / "verification_audit.csv",
        PAPER / "verification_report.md",
        FIGURES / "external_roc_singlepass_vs_tta8.png",
        FIGURES / "external_operating_points.png",
        FIGURES / "internal_protocol_gap.png",
    ]
    required += [spec["path"] for spec in RAW_SHEETS]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Generate paper resources first; missing: " + ", ".join(missing)
        )

    manifest = json.loads(
        (paths.APP_MODELS / "model_manifest.json").read_text(encoding="utf-8")
    )
    external = pd.read_csv(GENERATED / "external_performance.csv")
    internal = pd.read_csv(GENERATED / "internal_performance.csv")
    comparison = pd.read_csv(GENERATED / "external_paired_comparison.csv").iloc[0]
    posthoc = pd.read_csv(GENERATED / "external_posthoc_two_class_readout.csv").iloc[0]
    audit = pd.read_csv(GENERATED / "verification_audit.csv")
    duplicates = pd.read_csv(GENERATED / "external_near_duplicate_audit.csv")
    deduplicated = pd.read_csv(GENERATED / "external_deduplicated_sensitivity.csv")
    deduplicated_paired = pd.read_csv(
        GENERATED / "external_deduplicated_paired_comparison.csv"
    )
    internal_recheck = pd.read_csv(GENERATED / "internal_reproduction_check.csv")

    wb = Workbook()
    wb.remove(wb.active)
    wb.properties.title = "SkinFusionNet Paper Data Workbook"
    wb.properties.subject = "Aggregate performance evidence and manuscript resources"
    wb.properties.creator = "SkinFusionNet research project"
    wb.properties.keywords = "dermoscopy, AI, model versioning, external validation"
    wb.properties.description = (
        "English aggregate evidence pack generated from versioned project artifacts."
    )

    build_readme(wb, manifest)
    build_verification(
        wb,
        audit,
        duplicates,
        deduplicated,
        deduplicated_paired,
        internal_recheck,
    )
    build_key_results(
        wb,
        internal,
        external,
        comparison,
        posthoc,
        deduplicated,
        internal_recheck,
    )
    build_external(wb, external, deduplicated)
    build_internal(wb, internal, internal_recheck)
    build_model_versions(wb, internal, external)
    build_dataset(wb)
    build_claims(wb)
    build_gaps(wb)
    build_artifacts(wb)
    build_figures(wb)
    build_charts(wb, pd.read_csv(paths.ABLATION_HANDCRAFT_FACTORIAL), external)
    build_sources(wb)

    # Raw layer last, so the aggregate sheet order stays stable for anyone who
    # already cites a tab by position.
    raw_entries = [build_raw_model_contract(wb, manifest)]
    raw_entries += [build_raw_sheet(wb, spec) for spec in RAW_SHEETS]
    build_raw_index(wb, raw_entries)
    wb.move_sheet("Raw Data Index", offset=-len(raw_entries))

    for ws in wb.worksheets:
        ws.sheet_view.showGridLines = False
        ws.sheet_properties.pageSetUpPr.fitToPage = True
        ws.page_setup.fitToWidth = 1
        ws.page_setup.fitToHeight = 0
        ws.oddFooter.center.text = "SkinFusionNet paper resource — evidence snapshot 2026-07-10"
        ws.oddFooter.right.text = "Page &P of &N"
    wb.calculation.fullCalcOnLoad = True
    wb.calculation.forceFullCalc = True
    wb.calculation.calcMode = "auto"
    wb.save(OUTPUT)
    return OUTPUT


def validate_workbook(path: Path) -> None:
    wb = load_workbook(path, read_only=False, data_only=True)
    expected_sheets = [
        "README",
        "Verification Audit",
        "Key Results",
        "External Performance",
        "Internal Performance",
        "Model Versions",
        "Dataset Composition",
        "Claims Matrix",
        "Study Gaps",
        "Artifact Audit",
        "Figures",
        "Charts",
        "Source Registry",
        "Raw Data Index",
        "Raw Model Contract",
    ] + [spec["sheet"] for spec in RAW_SHEETS]
    if wb.sheetnames != expected_sheets:
        raise ValueError(f"unexpected workbook sheets: {wb.sheetnames}")
    if len(wb["Figures"]._images) != 3:
        raise ValueError("the Figures sheet must contain three embedded previews")

    # The raw layer is only worth carrying if it is complete: every sheet must
    # hold exactly as many data rows as its source CSV.
    for spec in RAW_SHEETS:
        source_rows = sum(1 for _ in spec["path"].open(encoding="utf-8")) - 1
        ws = wb[spec["sheet"]]
        header_row = next(
            (
                cell.row
                for row in ws.iter_rows(min_col=1, max_col=1)
                for cell in row
                if cell.value == pd.read_csv(spec["path"], nrows=0).columns[0]
            ),
            None,
        )
        if header_row is None:
            raise ValueError(f"no header row found in raw sheet: {spec['sheet']}")
        written = ws.max_row - header_row
        if written != source_rows:
            raise ValueError(
                f"raw sheet {spec['sheet']} holds {written} rows, "
                f"source {relative(spec['path'])} holds {source_rows}"
            )

    key_values = [
        cell.value
        for row in wb["Key Results"].iter_rows()
        for cell in row
        if isinstance(cell.value, (int, float))
    ]
    for expected in (
        0.931818,
        0.930451,
        0.892755,
        0.893546,
        0.631579,
        0.829545,
        0.029306,
    ):
        if not any(abs(float(value) - expected) < 1e-6 for value in key_values):
            raise ValueError(f"missing key estimate in workbook: {expected}")
    verification_text = " ".join(
        str(cell.value)
        for row in wb["Verification Audit"].iter_rows()
        for cell in row
        if cell.value is not None
    )
    for expected_text in ("NOT_EXACTLY_REPRODUCED", "near-identical", "NOT_VERIFIABLE"):
        if expected_text not in verification_text:
            raise ValueError(f"missing verification warning in workbook: {expected_text}")
    wb.close()


def main() -> None:
    path = build_workbook()
    validate_workbook(path)
    print(f"wrote and validated {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
