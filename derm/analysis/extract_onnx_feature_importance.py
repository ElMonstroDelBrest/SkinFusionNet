"""Extract auditable split-frequency importance from deployed LightGBM ONNX heads.

This script requires ``onnx`` (available in the export/parity environment).  It
counts branch nodes that use each input feature.  Split frequency is a model
inspection statistic, not gain importance, SHAP importance, or causal effect.

Run from the repository root::

    .parity_venv/bin/python -m derm.analysis.extract_onnx_feature_importance
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import onnx

from derm import paths


OUTPUT = paths.CURRENT_ONNX_IMPORTANCE
HEADS = (
    ("lesion_head_A", paths.APP_MODELS / "lgbm_a.onnx"),
    ("dehair_head_C", paths.APP_MODELS / "lgbm_c.onnx"),
)


def handcraft_block(feature: str) -> str:
    if feature == "A4":
        return "A_asymmetry"
    if feature == "D":
        return "D_diameter"
    if feature.startswith("Lab_"):
        return "C_colour"
    return "B_border"


def feature_names() -> list[str]:
    with paths.FEATURES_CSV.open(newline="", encoding="utf-8") as stream:
        header = next(csv.reader(stream))
    handcraft = [name for name in header if name != "Class"]
    if len(handcraft) != 18:
        raise ValueError(f"Expected 18 handcrafted features, found {len(handcraft)}")
    return handcraft + [f"cnn_{index}" for index in range(1_280)]


def branch_counts(path: Path) -> Counter[int]:
    model = onnx.load(path)
    ensemble_nodes = [
        node for node in model.graph.node if node.op_type == "TreeEnsembleClassifier"
    ]
    if len(ensemble_nodes) != 1:
        raise ValueError(f"Expected one TreeEnsembleClassifier in {path}")
    attributes = {attribute.name: attribute for attribute in ensemble_nodes[0].attribute}
    feature_ids = list(attributes["nodes_featureids"].ints)
    modes = [value.decode("utf-8") for value in attributes["nodes_modes"].strings]
    if len(feature_ids) != len(modes):
        raise ValueError(f"ONNX node attribute length mismatch in {path}")
    return Counter(
        feature_id
        for feature_id, mode in zip(feature_ids, modes)
        if mode != "LEAF"
    )


def main() -> None:
    names = feature_names()
    counts = {head: branch_counts(path) for head, path in HEADS}
    totals = {head: sum(values.values()) for head, values in counts.items()}
    combined = Counter()
    for values in counts.values():
        combined.update(values)
    ranked = sorted(range(len(names)), key=lambda index: (-combined[index], index))
    fields = [
        "rank_combined",
        "feature_index",
        "feature",
        "feature_family",
        "clinical_block",
        "splits_lesion_head_A",
        "splits_dehair_head_C",
        "splits_combined",
        "share_all_branch_splits",
        "importance_semantics",
        "source_models",
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        total_combined = sum(combined.values())
        for rank, feature_index in enumerate(ranked, start=1):
            feature = names[feature_index]
            is_handcraft = feature_index < 18
            writer.writerow(
                {
                    "rank_combined": rank,
                    "feature_index": feature_index,
                    "feature": feature,
                    "feature_family": "handcrafted" if is_handcraft else "CNN_embedding",
                    "clinical_block": handcraft_block(feature) if is_handcraft else "deep_latent",
                    "splits_lesion_head_A": counts["lesion_head_A"][feature_index],
                    "splits_dehair_head_C": counts["dehair_head_C"][feature_index],
                    "splits_combined": combined[feature_index],
                    "share_all_branch_splits": combined[feature_index] / total_combined,
                    "importance_semantics": (
                        "tree split frequency (branch-node count); not gain, SHAP, or causal importance"
                    ),
                    "source_models": "app/assets/models/lgbm_a.onnx + lgbm_c.onnx",
                }
            )
    print(
        f"Wrote {OUTPUT} with {len(names)} features; "
        f"branch splits A={totals['lesion_head_A']:,}, "
        f"C={totals['dehair_head_C']:,}, combined={sum(combined.values()):,}"
    )


if __name__ == "__main__":
    main()
