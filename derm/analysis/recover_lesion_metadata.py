"""Join public HAM10000 / ISIC 2019 metadata onto the merged image table.

Does not overwrite artifacts/image_metadata.csv (handoff-pinned).
Writes artifacts/image_metadata_joined.csv plus a coverage report.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from derm import paths

META_DIR = paths.DATA / "metadata"
OUT_CSV = paths.ARTIFACTS / "image_metadata_joined.csv"
OUT_REPORT = paths.RESULTS / "suite" / "metadata_join_report.md"


def main() -> None:
    internal = pd.read_csv(paths.INTERNAL_METADATA)
    ham = pd.read_csv(META_DIR / "HAM10000_metadata.csv")
    isic2019 = pd.read_csv(META_DIR / "ISIC_2019_Training_Metadata.csv")

    ham = ham.rename(columns={"image_id": "image_id", "lesion_id": "lesion_id_ham"})
    ham["image_id"] = ham["image_id"].astype(str)
    isic2019 = isic2019.rename(
        columns={"image": "image_id", "lesion_id": "lesion_id_isic2019"}
    )
    isic2019["image_id"] = isic2019["image_id"].astype(str)

    merged = internal.merge(ham[["image_id", "lesion_id_ham", "dx", "dx_type", "age", "sex", "localization"]], on="image_id", how="left")
    merged = merged.merge(
        isic2019[["image_id", "lesion_id_isic2019", "age_approx", "anatom_site_general", "sex"]].rename(
            columns={"sex": "sex_isic2019"}
        ),
        on="image_id",
        how="left",
    )
    merged["lesion_id_resolved"] = merged["lesion_id_ham"].fillna(merged["lesion_id_isic2019"]).fillna("")
    merged["lesion_id_resolved"] = merged["lesion_id_resolved"].astype(str).replace({"nan": "", "None": ""})
    merged["patient_or_lesion_key"] = merged["lesion_id_resolved"].where(
        merged["lesion_id_resolved"] != "", merged["file"]
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_CSV, index=False)

    lines = ["# Lesion / patient metadata join", "", f"Rows: {len(merged)}", ""]
    lines.append("| source | n | lesion_id resolved | multi-image lesions | max images/lesion |")
    lines.append("|---|---:|---:|---:|---:|")
    for source, g in merged.groupby("source"):
        resolved = (g["lesion_id_resolved"] != "").mean()
        keyed = g[g["lesion_id_resolved"] != ""]
        if len(keyed):
            counts = keyed.groupby("lesion_id_resolved").size()
            multi = int((counts > 1).sum())
            mx = int(counts.max())
        else:
            multi, mx = 0, 1
        lines.append(
            f"| {source} | {len(g)} | {100 * resolved:.1f}% | {multi} | {mx} |"
        )
    resolved_all = (merged["lesion_id_resolved"] != "").mean()
    keyed = merged[merged["lesion_id_resolved"] != ""]
    counts = keyed.groupby("lesion_id_resolved").size() if len(keyed) else pd.Series(dtype=int)
    lines += [
        "",
        f"Overall lesion_id resolved: **{100 * resolved_all:.1f}%** ({int((merged.lesion_id_resolved != '').sum())}/{len(merged)}).",
        f"Resolved lesions: {counts.size if len(counts) else 0}; multi-image lesions: {int((counts > 1).sum()) if len(counts) else 0}.",
        "",
        "Kaggle9 / ISIC2017 still have no public lesion_id in this join.",
        f"Wrote `{paths.relative(OUT_CSV)}`. Did not overwrite the handoff `image_metadata.csv`.",
        "",
    ]
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
