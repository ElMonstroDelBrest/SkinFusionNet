"""Evaluate the deployed ONNX bundle on the ISIC 2020 OOD cohort.

Same inference path as `derm.export.eval_external` and the Flutter app.
Does not retrain. Thresholds come from the runtime manifest and are not
fitted on ISIC 2020.

Usage, from the repo root:

    .venv_train/bin/python -m derm.export.eval_shifted
    .venv_train/bin/python -m derm.export.eval_shifted --subset full
    .venv_train/bin/python -m derm.export.eval_shifted --resume --limit 50
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from derm import paths
from derm.export.cuda_libs import configure_nvidia_libs

configure_nvidia_libs()

from derm.export.bundle_runtime import BundleEngine
from derm.export.eval_external import CLASS_NAMES, T, wilson
from derm.export.model_manifest import validate_manifest

FOLDERS = {"NonMelanoma": 0, "Melanoma": 1}
POWERED_NEG_PER_POS = 3
POWERED_SEED = 42


def select_subset(kept: pd.DataFrame, subset: str) -> pd.DataFrame:
    if subset == "full":
        return kept.copy()
    if subset != "powered":
        raise ValueError(f"unknown subset {subset}")
    melanoma_patients = set(kept.loc[kept["true"] == 1, "patient_id"].astype(str))
    positives = kept[kept["true"] == 1].copy()
    benign_only = kept[
        (kept["true"] == 0) & ~kept["patient_id"].astype(str).isin(melanoma_patients)
    ].copy()
    n_neg = min(len(benign_only), POWERED_NEG_PER_POS * len(positives))
    negatives = benign_only.sample(n=n_neg, random_state=POWERED_SEED)
    selected = pd.concat([positives, negatives], ignore_index=True)
    return selected.sort_values(["folder", "file"]).reset_index(drop=True)


def patient_table(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    grouped = frame.groupby("patient_id", sort=False)
    return pd.DataFrame(
        {
            "patient_id": grouped.size().index,
            "n_images": grouped.size().to_numpy(),
            "true": grouped["true"].max().to_numpy(),
            "p_mel": grouped["p_mel"].max().to_numpy(),
            "p_nevus": grouped["p_nevus"].min().to_numpy(),
            "p_atyp": grouped["p_atyp"].max().to_numpy(),
            "coverage": grouped["coverage"].median().to_numpy(),
        }
    )


def metric_block(y: np.ndarray, p_mel: np.ndarray, p_atyp: np.ndarray) -> dict[str, float]:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    pred_mel = p_mel >= T["melanoma"]
    tp = int(((y == 1) & pred_mel).sum())
    tn = int(((y == 0) & ~pred_mel).sum())
    not_benign = (p_mel >= T["melanoma"]) | (p_atyp >= T["atypical"])
    nb_tp = int(((y == 1) & not_benign).sum())
    nb_tn = int(((y == 0) & ~not_benign).sum())
    sens, sl, sh = wilson(tp, n_pos)
    spec, pl, ph = wilson(tn, n_neg)
    nb_sens, nbl, nbh = wilson(nb_tp, n_pos)
    nb_spec, nbpl, nbph = wilson(nb_tn, n_neg)
    return {
        "n": float(len(y)),
        "n_pos": float(n_pos),
        "n_neg": float(n_neg),
        "auc": float(roc_auc_score(y, p_mel)) if n_pos and n_neg else float("nan"),
        "ap": float(average_precision_score(y, p_mel)) if n_pos and n_neg else float("nan"),
        "brier": float(brier_score_loss(y, p_mel)),
        "tp": float(tp),
        "fn": float(n_pos - tp),
        "tn": float(tn),
        "fp": float(n_neg - tn),
        "sens": sens,
        "sens_low": sl,
        "sens_high": sh,
        "spec": spec,
        "spec_low": pl,
        "spec_high": ph,
        "nb_tp": float(nb_tp),
        "nb_fn": float(n_pos - nb_tp),
        "nb_tn": float(nb_tn),
        "nb_fp": float(n_neg - nb_tn),
        "nb_sens": nb_sens,
        "nb_sens_low": nbl,
        "nb_sens_high": nbh,
        "nb_spec": nb_spec,
        "nb_spec_low": nbpl,
        "nb_spec_high": nbph,
    }


def write_metrics_csv(path: Path, image: dict, patient: dict) -> None:
    rows = []
    for unit, block in (("file", image), ("patient", patient)):
        for key, value in block.items():
            rows.append({"unit": unit, "metric": key, "estimate": value})
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=("powered", "full"), default="powered")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="defaults to results/ood_isic2020/",
    )
    args = parser.parse_args()

    if not paths.OOD_ISIC2020_KEPT.exists():
        sys.exit(
            "missing data/ood_isic2020/kept_manifest.csv — "
            "run python -m derm.analysis.prepare_ood_isic2020 first"
        )

    models = paths.APP_MODELS
    manifest = validate_manifest(models / "model_manifest.json", models)
    kept = pd.read_csv(paths.OOD_ISIC2020_KEPT)
    selected = select_subset(kept, args.subset)
    selected = selected.set_index("file", drop=False)
    out_dir = args.output_dir or paths.RESULTS_OOD_ISIC2020
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "ood_isic2020_predictions.csv"
    metrics_path = out_dir / "ood_isic2020_metrics.csv"
    summary_path = out_dir / "ood_isic2020_summary.txt"

    done: set[str] = set()
    rows: list[dict] = []
    if args.resume and pred_path.exists():
        previous = pd.read_csv(pred_path)
        rows = previous.to_dict("records")
        done = set(previous["file"].astype(str))
        print(f"resume: {len(done)} rows already written", flush=True)

    todo = [
        row
        for row in selected.itertuples()
        if str(row.file) not in done
    ]
    if args.limit:
        todo = todo[: args.limit]
    print(
        f"subset={args.subset}  queued={len(todo)}  "
        f"selected={len(selected)}  kept={len(kept)}",
        flush=True,
    )

    engine = BundleEngine(
        models,
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device,
    )
    runtime_desc = engine.describe()
    print(runtime_desc, flush=True)
    failures: list[tuple[str, str]] = []
    started = time.time()
    fieldnames = [
        "file", "folder", "true", "patient_id", "diagnosis",
        "p_nevus", "p_mel", "p_atyp", "argmax", "coverage", "probability_sum",
    ]

    image_paths = [
        paths.OOD_ISIC2020_EVAL / str(item.folder) / item.file for item in todo
    ]
    processed = 0
    batch = args.batch_size
    try:
        for start in range(0, len(todo), batch):
            items = todo[start : start + batch]
            chunk_paths = image_paths[start : start + batch]
            outputs = engine.infer_paths(chunk_paths)
            for item, output in zip(items, outputs):
                processed += 1
                if isinstance(output, BaseException):
                    failures.append((str(item.file), repr(output)[:120]))
                    print(f"  FAIL {item.file}: {output!r}"[:120], flush=True)
                    continue
                meta = selected.loc[item.file]
                row = {
                    "file": item.file,
                    "folder": str(item.folder),
                    "true": int(item.true),
                    "patient_id": str(meta["patient_id"]),
                    "diagnosis": str(meta["diagnosis"]),
                    **output,
                }
                rows.append(row)
            elapsed = time.time() - started
            rate = processed / max(elapsed, 1e-6)
            last = rows[-1] if rows else None
            extra = (
                f"  p_mel={last['p_mel']:.3f} cov={last['coverage']:.2f}"
                if last
                else ""
            )
            print(
                f"  {processed}/{len(todo)}{extra}  {rate:.2f} img/s",
                flush=True,
            )
            if processed % 64 == 0 or processed == len(todo):
                pd.DataFrame(rows).to_csv(pred_path, index=False, columns=fieldnames)
    finally:
        engine.close()

    pd.DataFrame(rows).to_csv(pred_path, index=False, columns=fieldnames)
    if not rows:
        sys.exit("no predictions written")

    frame = pd.DataFrame(rows)
    y = frame["true"].to_numpy(int)
    p_mel = frame["p_mel"].to_numpy(float)
    p_atyp = frame["p_atyp"].to_numpy(float)
    image_metrics = metric_block(y, p_mel, p_atyp)
    patients = patient_table(rows)
    patient_metrics = metric_block(
        patients["true"].to_numpy(int),
        patients["p_mel"].to_numpy(float),
        patients["p_atyp"].to_numpy(float),
    )
    write_metrics_csv(metrics_path, image_metrics, patient_metrics)

    lines: list[str] = []

    def pr(text: str = "") -> None:
        print(text)
        lines.append(text)

    pr("=" * 70)
    pr("OOD EVALUATION — ISIC 2020, deployed single-pass bundle")
    pr("=" * 70)
    pr(f"bundle_id = {manifest['bundle_id']}")
    pr(f"bundle_sha256 = {manifest['bundle_sha256']}")
    pr(f"subset = {args.subset}")
    pr(f"runtime = {runtime_desc}")
    pr(
        f"files = {int(image_metrics['n'])}  "
        f"(melanoma={int(image_metrics['n_pos'])}, "
        f"non-melanoma={int(image_metrics['n_neg'])})  "
        f"failures={len(failures)}"
    )
    pr(
        f"patients = {int(patient_metrics['n'])}  "
        f"(melanoma-positive={int(patient_metrics['n_pos'])})"
    )
    pr("NonMelanoma is target=0, not a proven nevus class.")
    pr("Thresholds are the legacy policy; they were not fit on ISIC 2020.")
    pr("")
    pr("--- File-level (naive; multiple images per patient) ---")
    pr(f"  AUC melanoma = {image_metrics['auc']:.4f}")
    pr(f"  AP  melanoma = {image_metrics['ap']:.4f}")
    pr(f"  Brier        = {image_metrics['brier']:.4f}")
    pr(
        f"  @ p_mel>={T['melanoma']:.3f}  "
        f"sens={image_metrics['sens']:.3f} "
        f"[{image_metrics['sens_low']:.3f},{image_metrics['sens_high']:.3f}]  "
        f"({int(image_metrics['tp'])}/{int(image_metrics['n_pos'])})  "
        f"spec={image_metrics['spec']:.3f} "
        f"[{image_metrics['spec_low']:.3f},{image_metrics['spec_high']:.3f}]  "
        f"({int(image_metrics['tn'])}/{int(image_metrics['n_neg'])})"
    )
    pr(
        f"  not-benign   "
        f"sens={image_metrics['nb_sens']:.3f} "
        f"({int(image_metrics['nb_tp'])}/{int(image_metrics['n_pos'])})  "
        f"spec={image_metrics['nb_spec']:.3f} "
        f"({int(image_metrics['nb_tn'])}/{int(image_metrics['n_neg'])})"
    )
    pr("")
    pr("--- Patient-level (true=any melanoma image; score=max p_mel) ---")
    pr(f"  AUC melanoma = {patient_metrics['auc']:.4f}")
    pr(f"  AP  melanoma = {patient_metrics['ap']:.4f}")
    pr(
        f"  @ p_mel>={T['melanoma']:.3f}  "
        f"sens={patient_metrics['sens']:.3f} "
        f"({int(patient_metrics['tp'])}/{int(patient_metrics['n_pos'])})  "
        f"spec={patient_metrics['spec']:.3f} "
        f"({int(patient_metrics['tn'])}/{int(patient_metrics['n_neg'])})"
    )
    argmax = frame["argmax"].to_numpy(int)
    pr("")
    pr("--- 3-class argmax (atypical has no 2020 label; it counts as a miss) ---")
    pr(f"            {CLASS_NAMES[0]:>8}{CLASS_NAMES[1]:>10}{CLASS_NAMES[2]:>10}")
    for cid, name in ((0, "non-mel"), (1, "melanoma")):
        counts = [int(((y == cid) & (argmax == k)).sum()) for k in (0, 1, 2)]
        pr(f"  {name:<9}{counts[0]:>8}{counts[1]:>10}{counts[2]:>10}")
    if failures:
        pr("")
        pr("--- FAILURES ---")
        for name, err in failures:
            pr(f"  {name}: {err}")
    pr("")
    pr(f"wrote {pred_path}")
    pr(f"wrote {metrics_path}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pr(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
