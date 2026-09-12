"""Validate and refresh the canonical deployed-model manifest.

The human-maintained release metadata lives in
``app/assets/models/model_manifest.json``. This module verifies its component
hashes and computes the deterministic bundle fingerprint used by the app audit.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from derm import paths


DEPLOYED_COMPONENTS = (
    "unet.onnx",
    "cnn_lesion_feats.onnx",
    "cnn_dehair_feats.onnx",
    "lgbm_a.onnx",
    "lgbm_c.onnx",
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_sha256(component_hashes: list[tuple[str, str]]) -> str:
    """Hash ordered component names plus their raw SHA-256 digests."""
    digest = hashlib.sha256()
    for name, component_hash in component_hashes:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(component_hash))
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"manifest must be an object: {path}")
    required = {
        "schema_version",
        "bundle_id",
        "bundle_sha256",
        "variant",
        "preprocessing",
        "decision_policy",
        "components",
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise ValueError(f"manifest is missing required fields: {missing}")
    if raw["schema_version"] != 1:
        raise ValueError(f"unsupported manifest schema: {raw['schema_version']}")
    if raw["variant"] != "single_pass":
        raise ValueError(f"unsupported deployed variant: {raw['variant']}")
    return raw


def computed_identity(asset_dir: Path, manifest: dict[str, Any]) -> tuple[list[tuple[str, str]], str]:
    components = manifest.get("components")
    if not isinstance(components, list):
        raise ValueError("manifest components must be a list")
    if any(not isinstance(component, dict) for component in components):
        raise ValueError("manifest components must be objects")
    names = [component.get("name") for component in components]
    if tuple(names) != DEPLOYED_COMPONENTS:
        raise ValueError(
            f"deployed component order mismatch: expected={DEPLOYED_COMPONENTS}, got={names}"
        )
    hashes = []
    for name in names:
        path = asset_dir / name
        if not path.exists():
            raise FileNotFoundError(f"missing deployed component: {path}")
        hashes.append((name, file_sha256(path)))
    return hashes, bundle_sha256(hashes)


def validate_manifest(manifest_path: Path, asset_dir: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    if not SHA256_PATTERN.fullmatch(str(manifest.get("bundle_sha256", ""))):
        raise ValueError("manifest bundle_sha256 is not a lowercase SHA-256")
    for component in manifest["components"]:
        if not SHA256_PATTERN.fullmatch(str(component.get("sha256", ""))):
            raise ValueError(
                f"invalid component SHA-256 for {component.get('name')}"
            )
    hashes, bundle_hash = computed_identity(asset_dir, manifest)
    declared = {
        component["name"]: component["sha256"] for component in manifest["components"]
    }
    mismatches = [
        f"{name}: declared={declared.get(name)} actual={actual}"
        for name, actual in hashes
        if declared.get(name) != actual
    ]
    if manifest.get("bundle_sha256") != bundle_hash:
        mismatches.append(
            f"bundle: declared={manifest.get('bundle_sha256')} actual={bundle_hash}"
        )
    if mismatches:
        raise ValueError("model manifest hash mismatch:\n  " + "\n  ".join(mismatches))
    return manifest


def refresh_manifest(
    template_path: Path,
    model_dir: Path,
    output_paths: list[Path],
    *,
    export_validation: dict[str, Any] | None = None,
    allow_bundle_change: bool = False,
    new_bundle_id: str | None = None,
    release_date: str | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(template_path)
    hashes, bundle_hash = computed_identity(model_dir, manifest)
    old_bundle_hash = manifest.get("bundle_sha256")
    bundle_changed = bool(old_bundle_hash and old_bundle_hash != bundle_hash)
    if bundle_changed:
        if not allow_bundle_change:
            raise RuntimeError(
                "Exported model bytes changed. Rerun with --allow-bundle-change, "
                "--new-bundle-id and --release-date after reviewing provenance. "
                f"old={old_bundle_hash} new={bundle_hash}"
            )
        old_bundle_id = str(manifest.get("bundle_id", ""))
        if not new_bundle_id or new_bundle_id == old_bundle_id:
            raise RuntimeError(
                "Changed ONNX bytes require a non-empty --new-bundle-id that "
                f"differs from {old_bundle_id!r}"
            )
        if not release_date:
            raise RuntimeError("Changed ONNX bytes require --release-date YYYY-MM-DD")
        try:
            dt.date.fromisoformat(release_date)
        except ValueError as exc:
            raise RuntimeError("--release-date must use YYYY-MM-DD") from exc
        manifest["supersedes"] = old_bundle_id
        manifest["bundle_id"] = new_bundle_id
        manifest["release_date"] = release_date
    elif new_bundle_id or release_date:
        raise RuntimeError(
            "--new-bundle-id/--release-date were supplied but the ONNX bundle "
            "fingerprint did not change"
        )
    by_name = {name: digest for name, digest in hashes}
    for component in manifest["components"]:
        component["sha256"] = by_name[component["name"]]
    manifest["bundle_sha256"] = bundle_hash
    if export_validation is not None:
        manifest["latest_export_validation"] = export_validation
    payload = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    for output in output_paths:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    return manifest


def write_compatibility_pointer(path: Path, manifest: dict[str, Any]) -> None:
    """Update the deprecated legacy pointer without duplicating model metadata."""
    payload = {
        "deprecated": True,
        "reason": "Replaced by the versioned runtime model manifest.",
        "canonical_manifest": "../../app/assets/models/model_manifest.json",
        "bundle_id": manifest["bundle_id"],
        "bundle_sha256": manifest["bundle_sha256"],
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=paths.APP_MODELS / "model_manifest.json")
    parser.add_argument("--assets", type=Path, default=paths.APP_MODELS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = validate_manifest(args.manifest, args.assets)
    print(
        f"model manifest ok: bundle_id={manifest['bundle_id']} "
        f"sha256={manifest['bundle_sha256']}"
    )


if __name__ == "__main__":
    main()
