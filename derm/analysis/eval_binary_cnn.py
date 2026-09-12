"""2-class CNN lineage: extract embeddings, fit binary heads, frozen-threshold OOD.

Does not overwrite the deployed 3-class bundle, its ONNX graphs, or the
canonical V2S feature CSVs. New artifacts live under artifacts/binary/ and
results/ablations/binary_cnn/.

Protocol:
  - LightGBM heads fit on train nevus+melanoma only (nested) and on
    train+valid (1a-style, comparable to B_retrain).
  - Youden / spec>=0.90 chosen on internal valid only, then locked.
  - ISIC 2020 and the hospital set are never used to pick a threshold.

Usage, from the repo root:

    .venv_train/bin/python -m derm.analysis.eval_binary_cnn
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from derm import paths
from derm.export.cuda_libs import configure_nvidia_libs
from derm.pipeline.cnn_v2s import EfficientNetV2SCBAM
from derm.pipeline.extract_v2s_onnx import image_for_row, preprocess as preprocess_bgr

configure_nvidia_libs()

T_LEGACY = 0.188
OUT = paths.RESULTS_BINARY_CNN
CNN_COLS = [f"cnn_{i}" for i in range(1280)]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return math.nan, math.nan, math.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def youden(y: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, scores)
    return float(thr[int(np.argmax(tpr - fpr))])


def threshold_for_spec(y: np.ndarray, scores: np.ndarray, target: float) -> float:
    fpr, tpr, thr = roc_curve(y, scores)
    spec = 1.0 - fpr
    ok = np.where(spec >= target)[0]
    if len(ok) == 0:
        return float(thr[-1])
    return float(thr[ok[int(np.argmax(tpr[ok]))]])


def metrics_row(dataset: str, arm: str, y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    pred = scores >= threshold
    tp = int(((y == 1) & pred).sum())
    tn = int(((y == 0) & ~pred).sum())
    sens, sl, sh = wilson(tp, n_pos)
    spec, pl, ph = wilson(tn, n_neg)
    auc = float(roc_auc_score(y, scores)) if n_pos and n_neg else math.nan
    ap = float(average_precision_score(y, scores)) if n_pos and n_neg else math.nan
    return {
        "dataset": dataset,
        "arm": arm,
        "n": int(len(y)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "auc": auc,
        "ap": ap,
        "threshold": float(threshold),
        "sens": sens,
        "sens_ci_low": sl,
        "sens_ci_high": sh,
        "spec": spec,
        "spec_ci_low": pl,
        "spec_ci_high": ph,
        "tp": tp,
        "fn": n_pos - tp,
        "tn": tn,
        "fp": n_neg - tn,
        "median_p_mel_pos": float(np.median(scores[y == 1])) if n_pos else math.nan,
        "frac_p_mel_lt_0_01_pos": float(np.mean(scores[y == 1] < 0.01)) if n_pos else math.nan,
    }


def make_lgbm() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        n_jobs=4,
        verbose=-1,
        objective="binary",
    )


def fit_heads(x_a: np.ndarray, x_c: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[Pipeline, Pipeline]:
    pipe_a = Pipeline([("scaler", StandardScaler()), ("clf", make_lgbm())])
    pipe_c = Pipeline([("scaler", StandardScaler()), ("clf", make_lgbm())])
    yt = y[mask]
    print(f"fitting binary CNN heads n={int(mask.sum())}  mel={int((yt == 1).sum())}", flush=True)
    pipe_a.fit(x_a[mask], yt)
    pipe_c.fit(x_c[mask], yt)
    return pipe_a, pipe_c


def score_heads(pipe_a: Pipeline, pipe_c: Pipeline, x_a: np.ndarray, x_c: np.ndarray) -> np.ndarray:
    return (pipe_a.predict_proba(x_a)[:, 1] + pipe_c.predict_proba(x_c)[:, 1]) / 2.0


def load_binary_cnn(weights: Path, device: torch.device) -> EfficientNetV2SCBAM:
    model = EfficientNetV2SCBAM(n_classes=2)
    state = torch.load(weights, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


@torch.inference_mode()
def forward_feat_prob(
    model: EfficientNetV2SCBAM, nchw: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(np.ascontiguousarray(nchw)).to(device, non_blocking=True)
    x = model.features(tensor)
    x = model.cbam(x)
    x = model.avgpool(x)
    feat = torch.flatten(x, 1)
    logits = model.fc(feat)
    prob = torch.softmax(logits, dim=1)[:, 1]
    return feat.float().cpu().numpy(), prob.float().cpu().numpy()


def extract_internal_view(
    view: str,
    weights: Path,
    out_csv: Path,
    meta: pd.DataFrame,
    device: torch.device,
    batch_size: int,
) -> None:
    if out_csv.exists():
        print(f"reusing {out_csv}", flush=True)
        return
    print(f"extracting internal {view} from {weights.name}", flush=True)
    model = load_binary_cnn(weights, device)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_fallback = 0
    started = time.time()
    with open(out_csv, "w", newline="") as handle:
        writer: csv.DictWriter | None = None
        batch: list[np.ndarray] = []
        batch_meta: list[tuple[pd.Series, str]] = []

        def flush() -> None:
            nonlocal batch, batch_meta, writer
            if not batch:
                return
            feats, probs = forward_feat_prob(model, np.stack(batch, axis=0), device)
            if writer is None:
                fieldnames = [
                    "row_index",
                    "file",
                    "split_original",
                    "class_name",
                    "view_source",
                    "p_mel_cnn",
                ] + CNN_COLS + ["Class"]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
            for feat, prob, (row, source) in zip(feats, probs, batch_meta):
                rec = {
                    "row_index": int(row["row_index"]),
                    "file": row["file"],
                    "split_original": row["split_original"],
                    "class_name": row["class_name"],
                    "view_source": source,
                    "p_mel_cnn": f"{float(prob):.6f}",
                    "Class": int(row["class_id"]),
                }
                for index, value in enumerate(feat):
                    rec[f"cnn_{index}"] = f"{float(value):.6f}"
                writer.writerow(rec)
            batch = []
            batch_meta = []

        for i, row in meta.iterrows():
            bgr, source = image_for_row(row, view)
            if source == "dehair_fallback_raw":
                n_fallback += 1
            batch.append(preprocess_bgr(bgr))
            batch_meta.append((row, source))
            if len(batch) >= batch_size:
                flush()
            if (i + 1) % 1000 == 0:
                rate = (i + 1) / max(time.time() - started, 1e-6)
                print(
                    f"  {view} {i + 1}/{len(meta)}  {rate:.1f} img/s  fallback={n_fallback}",
                    flush=True,
                )
        flush()
    del model
    torch.cuda.empty_cache()
    print(f"wrote {out_csv}  fallback={n_fallback}", flush=True)


def matrices_from_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    hc = pd.read_csv(paths.FEATURES_CSV)
    if len(frame) != len(hc):
        raise ValueError(f"row mismatch {path.name}={len(frame)} features.csv={len(hc)}")
    if not np.array_equal(frame["Class"].to_numpy(int), hc["Class"].to_numpy(int)):
        raise ValueError(f"Class mismatch {path.name} vs features.csv")
    x_hc = hc.drop(columns=["Class"]).to_numpy(float)
    x = np.c_[x_hc, frame[CNN_COLS].to_numpy(float)]
    p_cnn = frame["p_mel_cnn"].to_numpy(float)
    y = frame["Class"].to_numpy(int)
    return x, p_cnn, y


def extract_ood_npz(
    items: list[tuple[Path, str, int]],
    npz_path: Path,
    engine,
    model_l: EfficientNetV2SCBAM,
    model_d: EfficientNetV2SCBAM,
    device: torch.device,
) -> dict[str, np.ndarray]:
    if npz_path.exists():
        print(f"reusing {npz_path}", flush=True)
        loaded = np.load(npz_path, allow_pickle=True)
        return {key: loaded[key] for key in loaded.files}
    print(f"extracting {len(items)} OOD images -> {npz_path}", flush=True)
    started = time.time()
    files, folders, labels = [], [], []
    handcraft, lesion, dehair, coverage = [], [], [], []
    p_lesion, p_dehair = [], []
    failures = 0
    batch = engine.batch_size
    for start in range(0, len(items), batch):
        chunk = items[start : start + batch]
        outputs = engine.preprocess_paths([item[0] for item in chunk])
        kept = []
        nchw_l, nchw_d = [], []
        for item, output in zip(chunk, outputs):
            if isinstance(output, BaseException) or not output:
                failures += 1
                continue
            kept.append(item)
            nchw_l.append(output["lesion_nchw"])
            nchw_d.append(output["dehair_nchw"])
            handcraft.append(output["handcraft"])
            coverage.append(output["coverage"])
            files.append(item[0].name)
            folders.append(item[1])
            labels.append(item[2])
        if kept:
            feat_l, prob_l = forward_feat_prob(model_l, np.stack(nchw_l, axis=0), device)
            feat_d, prob_d = forward_feat_prob(model_d, np.stack(nchw_d, axis=0), device)
            lesion.append(feat_l)
            dehair.append(feat_d)
            p_lesion.append(prob_l)
            p_dehair.append(prob_d)
        done = min(start + batch, len(items))
        rate = done / max(time.time() - started, 1e-6)
        print(f"  {done}/{len(items)}  {rate:.1f} img/s  fail={failures}", flush=True)
    payload = {
        "filenames": np.array(files),
        "folder": np.array(folders),
        "true": np.array(labels, dtype=np.int8),
        "handcraft": np.stack(handcraft).astype(np.float32),
        "lesion": np.concatenate(lesion, axis=0).astype(np.float32),
        "dehair": np.concatenate(dehair, axis=0).astype(np.float32),
        "coverage": np.array(coverage, dtype=np.float32),
        "p_cnn_lesion": np.concatenate(p_lesion, axis=0).astype(np.float32),
        "p_cnn_dehair": np.concatenate(p_dehair, axis=0).astype(np.float32),
    }
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **payload)
    print(f"wrote {npz_path}  kept={len(files)} fail={failures}", flush=True)
    return payload


def matrices_from_npz(blob: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    x_a = np.concatenate([blob["handcraft"], blob["lesion"]], axis=1)
    x_c = np.concatenate([blob["handcraft"], blob["dehair"]], axis=1)
    return x_a, x_c


def write_report(rows: list[dict], notes: list[str]) -> None:
    lines = [
        "# Binary CNN lineage — nevus vs melanoma",
        "",
        "New EfficientNetV2-S+CBAM trained on nevus+melanoma only. LightGBM heads",
        "on 18 handcraft + 1280-d embeddings. Does **not** replace the deployed 3-class bundle.",
        "",
        "Thresholds: nested = train-only fit, valid-only Youden / spec>=0.90, then locked.",
        "`1a` = train+valid fit (comparable to B_retrain). Hospital / ISIC 2020 never used to pick T.",
        "",
    ]
    lines.extend(notes)
    lines += [
        "",
        "| dataset | arm | n | pos | AUC | AP | T | sens | spec | med p_mel pos |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['arm']} | {row['n']} | {row['n_pos']} | "
            f"{row['auc']:.4f} | {row['ap']:.4f} | {row['threshold']:.3f} | "
            f"{row['sens']:.3f} ({row['tp']}/{row['n_pos']}) | "
            f"{row['spec']:.3f} ({row['tn']}/{row['n_neg']}) | "
            f"{row['median_p_mel_pos']:.3f} |"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    paths.BINARY_CNN_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ood-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-isic2020", action="store_true")
    args = parser.parse_args()

    if not paths.CNN_V2S_BINARY_LESION.exists():
        raise FileNotFoundError(paths.CNN_V2S_BINARY_LESION)
    if not paths.CNN_V2S_BINARY_DEHAIR.exists():
        raise FileNotFoundError(
            f"missing dehair 2-class CNN {paths.CNN_V2S_BINARY_DEHAIR}; "
            "train with python -m derm.pipeline.cnn_v2s_binary_dehair"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("eval_binary_cnn expects CUDA")
    meta = pd.read_csv(paths.INTERNAL_METADATA)
    extract_internal_view(
        "lesion",
        paths.CNN_V2S_BINARY_LESION,
        paths.CNN_V2S_BINARY_LESION_CSV,
        meta,
        device,
        args.batch_size,
    )
    extract_internal_view(
        "dehair",
        paths.CNN_V2S_BINARY_DEHAIR,
        paths.CNN_V2S_BINARY_DEHAIR_CSV,
        meta,
        device,
        args.batch_size,
    )

    x_a, p_cnn_a, y = matrices_from_csv(paths.CNN_V2S_BINARY_LESION_CSV)
    x_c, p_cnn_c, y_c = matrices_from_csv(paths.CNN_V2S_BINARY_DEHAIR_CSV)
    if not np.array_equal(y, y_c):
        raise ValueError("lesion/dehair Class mismatch")
    p_cnn = (p_cnn_a + p_cnn_c) / 2.0
    y_bin = (y == 1).astype(int)
    nm = np.isin(y, (0, 1))
    train_only = (meta["split_original"] == "train").to_numpy() & nm
    train_valid = meta["split_original"].isin(["train", "valid"]).to_numpy() & nm
    valid_mask = (meta["split_original"] == "valid").to_numpy() & nm
    test_mask = (meta["split_original"] == "test").to_numpy()
    test_nm = test_mask & nm

    nested_a, nested_c = fit_heads(x_a, x_c, y_bin, train_only)
    p_valid_nested = score_heads(nested_a, nested_c, x_a[valid_mask], x_c[valid_mask])
    t_youden = youden(y_bin[valid_mask], p_valid_nested)
    t_spec90 = threshold_for_spec(y_bin[valid_mask], p_valid_nested, 0.90)
    print(f"nested Youden T={t_youden:.4f}  spec90 T={t_spec90:.4f}", flush=True)

    one_a, one_c = fit_heads(x_a, x_c, y_bin, train_valid)
    p_valid_1a = score_heads(one_a, one_c, x_a[valid_mask], x_c[valid_mask])
    t_youden_1a = youden(y_bin[valid_mask], p_valid_1a)
    print(f"1a-style Youden on valid T={t_youden_1a:.4f}", flush=True)

    p_test_nested = score_heads(nested_a, nested_c, x_a[test_mask], x_c[test_mask])
    p_test_1a = score_heads(one_a, one_c, x_a[test_mask], x_c[test_mask])
    p_test_cnn = p_cnn[test_mask]
    y_test = y_bin[test_mask]
    y_test_nm = y_bin[test_nm]
    p_test_nested_nm = score_heads(nested_a, nested_c, x_a[test_nm], x_c[test_nm])
    p_test_1a_nm = score_heads(one_a, one_c, x_a[test_nm], x_c[test_nm])

    rows = [
        metrics_row("internal_test_mel_vs_rest", "cnn_softmax", y_test, p_test_cnn, T_LEGACY),
        metrics_row("internal_test_mel_vs_rest", "B_cnn_nested@Youden", y_test, p_test_nested, t_youden),
        metrics_row("internal_test_mel_vs_rest", "B_cnn_nested@spec90", y_test, p_test_nested, t_spec90),
        metrics_row("internal_test_mel_vs_rest", "B_cnn_1a@0.188", y_test, p_test_1a, T_LEGACY),
        metrics_row("internal_test_mel_vs_rest", "B_cnn_1a@Youden", y_test, p_test_1a, t_youden_1a),
        metrics_row("internal_test_mel_vs_nevus", "cnn_softmax", y_test_nm, p_cnn[test_nm], T_LEGACY),
        metrics_row("internal_test_mel_vs_nevus", "B_cnn_nested@Youden", y_test_nm, p_test_nested_nm, t_youden),
        metrics_row("internal_test_mel_vs_nevus", "B_cnn_1a@0.188", y_test_nm, p_test_1a_nm, T_LEGACY),
    ]

    atyp_mask = test_mask & (y == 2)
    p_atyp = score_heads(one_a, one_c, x_a[atyp_mask], x_c[atyp_mask])
    to_mel = int((p_atyp >= 0.5).sum())
    to_nev = int((p_atyp < 0.5).sum())
    notes = [
        f"Lesion 2-class CNN: `{paths.relative(paths.CNN_V2S_BINARY_LESION)}`.",
        f"Dehair 2-class CNN: `{paths.relative(paths.CNN_V2S_BINARY_DEHAIR)}`.",
        (
            f"True-atypical on internal test under B_cnn_1a @0.5: "
            f"{to_nev}/{len(p_atyp)} -> nevus ({100 * to_nev / len(p_atyp):.1f}%), "
            f"{to_mel}/{len(p_atyp)} -> melanoma ({100 * to_mel / len(p_atyp):.1f}%)."
        ),
        f"Frozen nested thresholds from internal valid: Youden={t_youden:.4f}, spec90={t_spec90:.4f}.",
    ]
    print(notes[-2], flush=True)
    pd.DataFrame(
        {
            "file": meta.loc[atyp_mask, "file"].to_numpy(),
            "p_melanoma_binary": p_atyp,
            "pred_at_0_5": np.where(p_atyp >= 0.5, "melanoma", "nevus"),
        }
    ).to_csv(OUT / "binary_cnn_internal_atypical_trace.csv", index=False)

    from derm.export.bundle_runtime import BundleEngine
    from derm.export.eval_external import IMG_EXTS

    model_l = load_binary_cnn(paths.CNN_V2S_BINARY_LESION, device)
    model_d = load_binary_cnn(paths.CNN_V2S_BINARY_DEHAIR, device)
    engine = BundleEngine(
        paths.APP_MODELS,
        batch_size=args.ood_batch_size,
        workers=args.workers,
        device="cuda",
        load_cnn=False,
    )
    print(engine.describe(), flush=True)

    hospital_items: list[tuple[Path, str, int]] = []
    for folder, cid in (("Nevi", 0), ("Melanom", 1)):
        for path in sorted((paths.EXTERNAL_VAL / folder).iterdir()):
            if path.suffix.lower() in IMG_EXTS:
                hospital_items.append((path, folder, cid))
    hosp = extract_ood_npz(
        hospital_items,
        OUT / "features_hospital.npz",
        engine,
        model_l,
        model_d,
        device,
    )
    hx_a, hx_c = matrices_from_npz(hosp)
    h_y = hosp["true"].astype(int)
    h_nested = score_heads(nested_a, nested_c, hx_a, hx_c)
    h_1a = score_heads(one_a, one_c, hx_a, hx_c)
    h_cnn = (hosp["p_cnn_lesion"] + hosp["p_cnn_dehair"]) / 2.0
    rows += [
        metrics_row("hospital", "cnn_softmax", h_y, h_cnn, T_LEGACY),
        metrics_row("hospital", "B_cnn_nested@Youden", h_y, h_nested, t_youden),
        metrics_row("hospital", "B_cnn_nested@spec90", h_y, h_nested, t_spec90),
        metrics_row("hospital", "B_cnn_1a@0.188", h_y, h_1a, T_LEGACY),
    ]
    pd.DataFrame(
        {
            "file": hosp["filenames"],
            "true": h_y,
            "coverage": hosp["coverage"],
            "p_cnn_softmax": h_cnn,
            "p_nested": h_nested,
            "p_1a": h_1a,
        }
    ).to_csv(OUT / "binary_cnn_hospital.csv", index=False)

    if not args.skip_isic2020:
        kept = pd.read_csv(paths.OOD_ISIC2020_KEPT)
        ood_items = [
            (
                paths.OOD_ISIC2020_EVAL / str(row.folder) / str(row.file),
                str(row.folder),
                int(row.true),
            )
            for row in kept.itertuples()
        ]
        ood = extract_ood_npz(
            ood_items,
            OUT / "features_isic2020.npz",
            engine,
            model_l,
            model_d,
            device,
        )
        ox_a, ox_c = matrices_from_npz(ood)
        o_y = ood["true"].astype(int)
        o_nested = score_heads(nested_a, nested_c, ox_a, ox_c)
        o_1a = score_heads(one_a, one_c, ox_a, ox_c)
        o_cnn = (ood["p_cnn_lesion"] + ood["p_cnn_dehair"]) / 2.0
        rows += [
            metrics_row("isic2020_full", "cnn_softmax", o_y, o_cnn, T_LEGACY),
            metrics_row("isic2020_full", "B_cnn_nested@Youden", o_y, o_nested, t_youden),
            metrics_row("isic2020_full", "B_cnn_nested@spec90", o_y, o_nested, t_spec90),
            metrics_row("isic2020_full", "B_cnn_1a@0.188", o_y, o_1a, T_LEGACY),
            metrics_row("isic2020_full", "B_cnn_1a@Youden", o_y, o_1a, t_youden_1a),
        ]
        pd.DataFrame(
            {
                "file": ood["filenames"],
                "true": o_y,
                "coverage": ood["coverage"],
                "p_cnn_softmax": o_cnn,
                "p_nested": o_nested,
                "p_1a": o_1a,
            }
        ).to_csv(OUT / "binary_cnn_isic2020.csv", index=False)
        nevus_files = set(kept.loc[kept["diagnosis"] == "nevus", "file"].astype(str))
        mel_files = set(kept.loc[kept["true"] == 1, "file"].astype(str))
        labelled = np.isin(ood["filenames"].astype(str), list(nevus_files | mel_files))
        rows += [
            metrics_row(
                "isic2020_mel_vs_labelled_nevus",
                "cnn_softmax",
                o_y[labelled],
                o_cnn[labelled],
                T_LEGACY,
            ),
            metrics_row(
                "isic2020_mel_vs_labelled_nevus",
                "B_cnn_nested@Youden",
                o_y[labelled],
                o_nested[labelled],
                t_youden,
            ),
            metrics_row(
                "isic2020_mel_vs_labelled_nevus",
                "B_cnn_1a@0.188",
                o_y[labelled],
                o_1a[labelled],
                T_LEGACY,
            ),
        ]

    engine.close()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(paths.BINARY_CNN_METRICS, index=False)
    write_report(rows, notes)
    print(f"wrote {paths.BINARY_CNN_METRICS}", flush=True)


if __name__ == "__main__":
    main()
