"""Repeat internal 1a / 10-fold CV with recovered lesion_id as the group key.

Uses the deployed 3-class V2S embeddings (not the 2-class CNN lineage).
Does not overwrite internal_validity_predictions.csv or the handoff metadata.

Unresolved lesion_id falls back to the filename (same as the original M1
fallback, so ISIC 2017 remains image-level).

Usage, from the repo root:

    .venv_train/bin/python -m derm.analysis.suite_lesion_grouped
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from derm import paths
from derm.analysis.internal_validity_m1 import (
    load_v2s_matrices,
    run_grouped_cv_outputs,
    run_phase_1a_outputs,
    youden_threshold,
)
from derm.analysis.retrain_binary_heads import metrics_row

OUT = paths.RESULTS_SUITE
JOINED = paths.INTERNAL_METADATA_JOINED


def leak_audit(meta: pd.DataFrame) -> dict:
    resolved = meta["lesion_id_resolved"].astype(str).to_numpy()
    train = meta["split_original"].isin(["train", "valid"]).to_numpy()
    test = (meta["split_original"] == "test").to_numpy()
    train_ids = set(resolved[train & (resolved != "")].tolist())
    leaked_mask = test & (resolved != "") & np.isin(resolved, list(train_ids))
    n_test = int(test.sum())
    n_leaked = int(leaked_mask.sum())
    return {
        "n_test": n_test,
        "n_leaked_test_images": n_leaked,
        "n_clean_test_images": n_test - n_leaked,
        "n_leaked_lesions": int(pd.unique(resolved[leaked_mask]).size),
        "frac_leaked": n_leaked / n_test if n_test else float("nan"),
        "leaked_on_test_rows": leaked_mask[test],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    joined = pd.read_csv(JOINED)
    meta = pd.read_csv(paths.INTERNAL_METADATA)
    if not np.array_equal(joined["file"].astype(str), meta["file"].astype(str)):
        raise ValueError("joined metadata is not row-aligned with image_metadata.csv")

    resolved = (
        joined["lesion_id_resolved"].fillna("").astype(str).replace({"nan": "", "None": ""})
    )
    fallback = int((resolved == "").sum())
    grouped_meta = meta.copy()
    grouped_meta["group_key"] = resolved.where(resolved != "", meta["file"].astype(str))
    grouped_meta["lesion_id_resolved"] = resolved.to_numpy()

    audit = leak_audit(grouped_meta)
    print(
        f"1a leak: {audit['n_leaked_test_images']}/{audit['n_test']} test images "
        f"({100 * audit['frac_leaked']:.1f}%) share a lesion_id with train+valid "
        f"({audit['n_leaked_lesions']} lesions). fallback_unresolved={fallback}",
        flush=True,
    )

    print("== 1a original split ==", flush=True)
    _summary_1a, pred_1a, _ = run_phase_1a_outputs(grouped_meta, "deployed_single")
    pred_1a.to_csv(OUT / "lesion_grouped_1a_predictions.csv", index=False)

    y_test = pred_1a["class_id"].to_numpy(int)
    p_test = pred_1a["p_melanoma"].to_numpy(float)
    t_1a = float(pred_1a["threshold_melanoma"].iloc[0])
    y_bin = (y_test == 1).astype(int)
    clean_mask = ~audit["leaked_on_test_rows"]
    if len(clean_mask) != len(y_bin):
        raise ValueError("clean mask / 1a test length mismatch")

    rows = [
        metrics_row("internal_test_original", "1a_mel_vs_rest", y_bin, p_test, t_1a),
        metrics_row(
            "internal_test_lesion_clean",
            "1a_mel_vs_rest",
            y_bin[clean_mask],
            p_test[clean_mask],
            t_1a,
        ),
    ]
    print(
        f"1a melanoma AUC original={rows[0]['auc']:.4f}  "
        f"clean={rows[1]['auc']:.4f}  n_clean={rows[1]['n']}",
        flush=True,
    )

    print("== grouped 10-fold CV (C1-style, lesion_id else file) ==", flush=True)
    summary_c1, pred_c1, _ = run_grouped_cv_outputs(
        "C1_lesion", grouped_meta, "deployed_single", nested=False
    )
    pred_c1.to_csv(OUT / "lesion_grouped_cv_predictions.csv", index=False)

    _, _, y, _ = load_v2s_matrices("deployed_single", grouped_meta)
    p_oof = pred_c1["p_melanoma"].to_numpy(float)
    if len(p_oof) != len(y):
        raise ValueError("grouped CV prediction length mismatch")
    t_oof = youden_threshold((y == 1).astype(int), p_oof)
    y_bin_all = (y == 1).astype(int)
    rows.append(metrics_row("internal_all_oof", "C1_lesion_mel_vs_rest", y_bin_all, p_oof, t_oof))
    resolved_mask = (resolved != "").to_numpy()
    rows.append(
        metrics_row(
            "internal_resolved_oof",
            "C1_lesion_mel_vs_rest",
            y_bin_all[resolved_mask],
            p_oof[resolved_mask],
            t_oof,
        )
    )

    pd.DataFrame(rows).to_csv(OUT / "lesion_grouped_metrics.csv", index=False)

    acc_mean = float(summary_c1.get("accuracy_fold_mean", float("nan")))
    acc_std = float(summary_c1.get("accuracy_fold_std", float("nan")))
    auc_mean = float(summary_c1.get("auc_macro_ovr_fold_mean", float("nan")))
    auc_std = float(summary_c1.get("auc_macro_ovr_fold_std", float("nan")))
    auc_mel = float(summary_c1.get("auc_melanoma_ovr", float("nan")))

    lines = [
        "# Lesion-grouped internal evaluation",
        "",
        "Deployed 3-class V2S embeddings. Group key = resolved `lesion_id`, else filename.",
        f"Unresolved (filename fallback): **{fallback}/{len(meta)}** "
        f"({100 * fallback / len(meta):.1f}%), mostly ISIC 2017.",
        "",
        "## Original split leak",
        "",
        f"- Test images whose lesion also appears in train+valid: "
        f"**{audit['n_leaked_test_images']}/{audit['n_test']}** "
        f"({100 * audit['frac_leaked']:.1f}%), {audit['n_leaked_lesions']} lesions.",
        f"- Clean test images: **{audit['n_clean_test_images']}**.",
        "- Image IDs were already disjoint; the leak is **same lesion, different images**.",
        "",
        "## 1a (heads on train+valid, original test vs lesion-clean test)",
        "",
        "| set | n | pos | melanoma AUC | AP | T | sens | spec |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["dataset"] in {"internal_test_original", "internal_test_lesion_clean"}:
            lines.append(
                f"| {row['dataset']} | {row['n']} | {row['n_pos']} | {row['auc']:.4f} | "
                f"{row['ap']:.4f} | {row['threshold']:.3f} | "
                f"{row['sens']:.3f} ({row['tp']}/{row['n_pos']}) | "
                f"{row['spec']:.3f} ({row['tn']}/{row['n_neg']}) |"
            )
    lines += [
        "",
        "## 10-fold StratifiedGroupKFold (CNN still partly in-sample)",
        "",
        f"- Accuracy fold mean: {acc_mean:.4f} ± {acc_std:.4f}",
        f"- Macro OvR AUC fold mean: {auc_mean:.4f} ± {auc_std:.4f}",
        f"- Melanoma OvR AUC (pooled OOF): {auc_mel:.4f}",
        "- Image-level C0 reference: acc 0.9424 / macro-AUC 0.9886 / melanoma AUC ~0.983 (leaky).",
        "",
        "C3 (lesion-disjoint **CNN** retrain) is still not run. This script only regroups the heads.",
        "",
        "| set | n | pos | melanoma AUC | AP |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["dataset"] in {"internal_all_oof", "internal_resolved_oof"}:
            lines.append(
                f"| {row['dataset']} | {row['n']} | {row['n_pos']} | {row['auc']:.4f} | {row['ap']:.4f} |"
            )
    (OUT / "lesion_grouped_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
