"""Prepare the ISIC 2020 OOD cohort: download, overlap-filter, layout.

The merged training set is ISIC 2017 + ISIC 2019 + HAM10000 + Kaggle 9-class.
ISIC 2020 is a later challenge. Identifier overlap against the 25,903 merged
files is expected to be zero; this script still checks IDs and SHA-256.

Usage, from the repo root:

    .venv_extval/bin/python -m derm.analysis.prepare_ood_isic2020
    .venv_extval/bin/python -m derm.analysis.prepare_ood_isic2020 --skip-download
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

from derm import paths

GT_URL = "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_GroundTruth.csv"
DUP_URL = "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_Duplicates.csv"
ZIP_URL = "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_JPEG.zip"
ZIP_NAME = "ISIC_2020_Training_JPEG.zip"
EXPECTED_ZIP_BYTES = 24_707_698_022


def log(message: str) -> None:
    print(message, flush=True)


def download(url: str, dest: Path, expected_bytes: int | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        if expected_bytes is None or dest.stat().st_size == expected_bytes:
            log(f"already present {dest} ({dest.stat().st_size} bytes)")
            return
        log(f"size mismatch for {dest.name}: {dest.stat().st_size} vs {expected_bytes}, resuming")
    log(f"downloading {url} -> {dest}")
    cmd = [
        "curl", "-C", "-", "-L", "--retry", "5", "--retry-delay", "5",
        "-o", str(dest), url,
    ]
    subprocess.run(cmd, check=True)
    if expected_bytes is not None and dest.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"{dest.name} size {dest.stat().st_size} != expected {expected_bytes}"
        )


def download_small(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        log(f"already present {dest}")
        return
    log(f"downloading {url}")
    with urllib.request.urlopen(url) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def union_find(pairs: list[tuple[str, str]]) -> dict[str, list[str]]:
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, right in pairs:
        union(left, right)
    groups: dict[str, list[str]] = defaultdict(list)
    for node in parent:
        groups[find(node)].append(node)
    return groups


def find_jpegs(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}:
            found[path.stem] = path
    return found


def link_or_copy(src: Path, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        dest.hardlink_to(src)
        return "hardlink"
    except OSError:
        try:
            dest.symlink_to(src.resolve())
            return "symlink"
        except OSError:
            shutil.copy2(src, dest)
            return "copy"


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    marker = out_dir / ".extract_ok"
    if marker.exists() and any(out_dir.rglob("*.jpg")):
        log(f"already extracted under {out_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"extracting {zip_path} -> {out_dir}")
    unzip = shutil.which("unzip")
    if unzip:
        subprocess.run(
            [unzip, "-q", "-o", str(zip_path), "-d", str(out_dir)],
            check=True,
        )
    else:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(out_dir)
    marker.write_text("ok\n", encoding="utf-8")


def write_sources(n_kept: int, n_mel: int, n_neg: int, n_patients: int) -> None:
    text = f"""ISIC 2020 — OOD evaluation cohort (held out of Skin_Cancer_Merged)
================================================================

Kept after official intra-2020 duplicate filter and overlap audit:
  files     : {n_kept}
  melanoma  : {n_mel}
  non-melanoma : {n_neg}
  patients  : {n_patients}

Source: ISIC 2020 SIIM-ISIC Melanoma Classification training set
  33,126 labelled JPEG images, 2,056 patients, 584 melanomas (raw).
  Rotemberg et al., Sci Data 2021.
  https://challenge.isic-archive.com/data/#2020
  License: CC BY-NC 4.0

This dataset is NOT part of the fused training set
(ISIC 2017 + ISIC 2019 + HAM10000 + Kaggle 9-class). Identifier overlap
with the 25,903 merged files is audited in
results/ood_isic2020/ood_isic2020_overlap_audit.csv.

Folders under eval_root/:
  Melanoma/     target=1
  NonMelanoma/  target=0  (nevus, seborrheic keratosis, unknown, …)
                Do not describe these files as nevi.

Layout created by: python -m derm.analysis.prepare_ood_isic2020
"""
    (paths.OOD_ISIC2020 / "SOURCES.txt").write_text(text, encoding="utf-8")


def prepare(skip_download: bool, skip_extract: bool) -> None:
    root = paths.OOD_ISIC2020
    raw = paths.OOD_ISIC2020_RAW
    eval_root = paths.OOD_ISIC2020_EVAL
    results = paths.RESULTS_OOD_ISIC2020
    root.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    gt_path = paths.OOD_ISIC2020_GT
    dup_path = paths.OOD_ISIC2020_DUPLICATES
    zip_path = raw / ZIP_NAME
    jpeg_dir = raw / "jpeg"

    if not skip_download:
        download_small(GT_URL, gt_path)
        download_small(DUP_URL, dup_path)
        download(ZIP_URL, zip_path, EXPECTED_ZIP_BYTES)
    for required in (gt_path, dup_path):
        if not required.exists():
            raise FileNotFoundError(required)
    if not skip_extract:
        if not zip_path.exists():
            raise FileNotFoundError(zip_path)
        extract_zip(zip_path, jpeg_dir)

    gt = pd.read_csv(gt_path)
    dups = pd.read_csv(dup_path)
    metadata = pd.read_csv(paths.INTERNAL_METADATA)
    manifest = pd.read_csv(paths.INTERNAL_RAW_MANIFEST)
    internal_ids = set(metadata["image_id"].dropna().astype(str))
    internal_ids |= set(manifest["image_id"].dropna().astype(str))
    internal_hashes = set(manifest["sha256"].dropna().astype(str).str.lower())
    external_hashes: set[str] = set()
    if paths.EXTERNAL_RAW_MANIFEST.exists():
        external = pd.read_csv(paths.EXTERNAL_RAW_MANIFEST)
        hash_col = "sha256" if "sha256" in external.columns else None
        if hash_col:
            external_hashes = set(external[hash_col].dropna().astype(str).str.lower())

    groups = union_find(
        [(str(a), str(b)) for a, b in dups.itertuples(index=False)]
    )
    drop_official: set[str] = set()
    official_rows: list[dict[str, str]] = []
    for members in groups.values():
        ordered = sorted(members)
        kept = ordered[0]
        for removed in ordered[1:]:
            drop_official.add(removed)
            official_rows.append(
                {
                    "image_name": removed,
                    "check": "official_intra_2020_duplicate",
                    "status": "EXCLUDED",
                    "detail": f"pair_kept={kept}",
                }
            )

    id_overlap = sorted(set(gt["image_name"].astype(str)) & internal_ids)
    audit_rows = list(official_rows)
    for image_name in id_overlap:
        audit_rows.append(
            {
                "image_name": image_name,
                "check": "isic_id_overlap_internal",
                "status": "EXCLUDED",
                "detail": "image_name present in merged 25,903",
            }
        )

    jpegs = find_jpegs(jpeg_dir)
    if not jpegs:
        raise FileNotFoundError(f"no JPEG files under {jpeg_dir}")
    log(f"found {len(jpegs)} JPEG files under {jpeg_dir}")

    missing_files = 0
    sha_internal = 0
    sha_external = 0
    kept_rows: list[dict[str, object]] = []
    link_mode = "hardlink"

    eval_mel = eval_root / "Melanoma"
    eval_neg = eval_root / "NonMelanoma"
    if eval_root.exists():
        shutil.rmtree(eval_root)
    eval_mel.mkdir(parents=True)
    eval_neg.mkdir(parents=True)

    drop_ids = drop_official | set(id_overlap)
    for row in gt.itertuples(index=False):
        image_name = str(row.image_name)
        if image_name in drop_ids:
            continue
        src = jpegs.get(image_name)
        if src is None:
            missing_files += 1
            audit_rows.append(
                {
                    "image_name": image_name,
                    "check": "jpeg_present",
                    "status": "EXCLUDED",
                    "detail": "ground-truth row without extracted JPEG",
                }
            )
            continue
        digest = file_sha256(src)
        if digest in internal_hashes:
            sha_internal += 1
            audit_rows.append(
                {
                    "image_name": image_name,
                    "check": "sha256_overlap_internal",
                    "status": "EXCLUDED",
                    "detail": digest,
                }
            )
            continue
        if digest in external_hashes:
            sha_external += 1
            audit_rows.append(
                {
                    "image_name": image_name,
                    "check": "sha256_overlap_hospital_external",
                    "status": "EXCLUDED",
                    "detail": digest,
                }
            )
            continue
        target = int(row.target)
        folder = "Melanoma" if target == 1 else "NonMelanoma"
        dest = (eval_mel if target == 1 else eval_neg) / src.name
        link_mode = link_or_copy(src, dest)
        kept_rows.append(
            {
                "image_name": image_name,
                "file": src.name,
                "folder": folder,
                "true": target,
                "diagnosis": row.diagnosis,
                "benign_malignant": row.benign_malignant,
                "patient_id": row.patient_id,
                "sex": row.sex,
                "age_approx": row.age_approx,
                "anatom_site_general_challenge": row.anatom_site_general_challenge,
                "bytes": src.stat().st_size,
                "sha256": digest,
                "source_path": src.as_posix(),
            }
        )

    audit_rows.append(
        {
            "image_name": "*",
            "check": "isic_id_overlap_internal",
            "status": "PASS" if not id_overlap else "FAIL",
            "detail": f"overlap={len(id_overlap)}",
        }
    )
    audit_rows.append(
        {
            "image_name": "*",
            "check": "sha256_overlap_internal",
            "status": "PASS" if sha_internal == 0 else "FAIL",
            "detail": f"overlap={sha_internal}",
        }
    )
    audit_rows.append(
        {
            "image_name": "*",
            "check": "sha256_overlap_hospital_external",
            "status": "PASS" if sha_external == 0 else "FAIL",
            "detail": f"overlap={sha_external}",
        }
    )
    audit_rows.append(
        {
            "image_name": "*",
            "check": "official_intra_2020_duplicate",
            "status": "PASS",
            "detail": f"excluded={len(drop_official)} from {len(groups)} pairs",
        }
    )
    audit_rows.append(
        {
            "image_name": "*",
            "check": "jpeg_present",
            "status": "PASS" if missing_files == 0 else "CAUTION",
            "detail": f"missing={missing_files}",
        }
    )

    kept = pd.DataFrame(kept_rows)
    kept.to_csv(paths.OOD_ISIC2020_KEPT, index=False)
    with paths.OOD_ISIC2020_OVERLAP.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["image_name", "check", "status", "detail"]
        )
        writer.writeheader()
        writer.writerows(audit_rows)

    n_mel = int((kept["true"] == 1).sum()) if len(kept) else 0
    n_neg = int((kept["true"] == 0).sum()) if len(kept) else 0
    n_patients = int(kept["patient_id"].nunique()) if len(kept) else 0
    write_sources(len(kept), n_mel, n_neg, n_patients)

    log(
        f"kept {len(kept)} files "
        f"(melanoma={n_mel}, non-melanoma={n_neg}, patients={n_patients})"
    )
    log(f"layout {eval_root} via {link_mode}")
    log(f"wrote {paths.OOD_ISIC2020_KEPT}")
    log(f"wrote {paths.OOD_ISIC2020_OVERLAP}")
    if id_overlap or sha_internal or sha_external:
        print("WARNING: overlap detected; see the audit CSV", file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="use files already under data/ood_isic2020/",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="assume JPEG files are already extracted under raw/jpeg/",
    )
    args = parser.parse_args()
    prepare(skip_download=args.skip_download, skip_extract=args.skip_extract)


if __name__ == "__main__":
    main()
