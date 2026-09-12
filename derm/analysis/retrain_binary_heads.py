"""Retrain LightGBM heads as nevus-vs-melanoma (atypical dropped from fit).

CNN embeddings stay frozen. Protocol mirrors internal 1a: scaler + two
LightGBM branches (lesion / dehair), mean of P(melanoma), fitted on
train+valid rows with class in {nevus, melanoma}.

Usage, from the repo root:

    .venv_train/bin/python -m derm.analysis.retrain_binary_heads
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from derm import paths
from derm.analysis.internal_validity_m1 import load_v2s_matrices
from derm.export.cuda_libs import configure_nvidia_libs

configure_nvidia_libs()

T_MEL = 0.188
OUT = paths.RESULTS_BINARY_RETRAIN


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


def readout(p_mel: np.ndarray, p_nev: np.ndarray) -> np.ndarray:
    den = p_mel + p_nev
    out = np.zeros_like(p_mel, dtype=float)
    np.divide(p_mel, den, out=out, where=den > 1e-12)
    return out


def make_model() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        n_jobs=4,
        verbose=-1,
        objective="binary",
    )


def fit_binary_heads(
    x_a: np.ndarray, x_c: np.ndarray, y: np.ndarray, train_mask: np.ndarray
) -> tuple[Pipeline, Pipeline]:
    pipe_a = Pipeline([("scaler", StandardScaler()), ("clf", make_model())])
    pipe_c = Pipeline([("scaler", StandardScaler()), ("clf", make_model())])
    yt = y[train_mask]
    print(f"fitting binary heads on n={train_mask.sum()}  mel={(yt == 1).sum()}", flush=True)
    pipe_a.fit(x_a[train_mask], yt)
    pipe_c.fit(x_c[train_mask], yt)
    return pipe_a, pipe_c


def score_binary(pipe_a: Pipeline, pipe_c: Pipeline, x_a: np.ndarray, x_c: np.ndarray) -> np.ndarray:
    p_a = pipe_a.predict_proba(x_a)[:, 1]
    p_c = pipe_c.predict_proba(x_c)[:, 1]
    return (p_a + p_c) / 2.0


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
    }


def extract_cohort(
    engine,
    items: list[tuple[Path, str, int]],
    npz_path: Path,
) -> dict[str, np.ndarray]:
    if npz_path.exists():
        print(f"reusing {npz_path}", flush=True)
        loaded = np.load(npz_path, allow_pickle=True)
        return {key: loaded[key] for key in loaded.files}
    paths_list = [item[0] for item in items]
    print(f"extracting {len(paths_list)} images -> {npz_path}", flush=True)
    started = time.time()
    handcraft, lesion, dehair, coverage = [], [], [], []
    files, folders, labels = [], [], []
    failures = 0
    batch = 256
    for start in range(0, len(paths_list), batch):
        chunk_items = items[start : start + batch]
        chunk_paths = [item[0] for item in chunk_items]
        outputs = engine.extract_paths(chunk_paths)
        for item, output in zip(chunk_items, outputs):
            if isinstance(output, BaseException) or not output:
                failures += 1
                continue
            files.append(item[0].name)
            folders.append(item[1])
            labels.append(item[2])
            handcraft.append(output["handcraft"])
            lesion.append(output["lesion"])
            dehair.append(output["dehair"])
            coverage.append(output["coverage"])
        done = min(start + batch, len(paths_list))
        rate = done / max(time.time() - started, 1e-6)
        print(f"  {done}/{len(paths_list)}  {rate:.1f} img/s  fail={failures}", flush=True)
    payload = {
        "filenames": np.array(files),
        "folder": np.array(folders),
        "true": np.array(labels, dtype=np.int8),
        "handcraft": np.stack(handcraft).astype(np.float32),
        "lesion": np.stack(lesion).astype(np.float32),
        "dehair": np.stack(dehair).astype(np.float32),
        "coverage": np.array(coverage, dtype=np.float32),
    }
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **payload)
    print(f"wrote {npz_path}  kept={len(files)} fail={failures}", flush=True)
    return payload


def matrices_from_npz(blob: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    x_a = np.concatenate([blob["handcraft"], blob["lesion"]], axis=1)
    x_c = np.concatenate([blob["handcraft"], blob["dehair"]], axis=1)
    return x_a, x_c


def write_report(rows: list[dict], atypical_trace: str) -> None:
    lines = [
        "# B_retrain — têtes LightGBM 2-classes (nævus vs mélanome)",
        "",
        "CNN gelé. Atypical retiré du **fit** (train+valid). Recette 1a :",
        "StandardScaler + LightGBM 300/0.05/63, branches lésion et dehair, moyenne.",
        "",
        atypical_trace,
        "",
        "| dataset | arm | n | pos | AUC | AP | T | sens | spec |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['arm']} | {row['n']} | {row['n_pos']} | "
            f"{row['auc']:.4f} | {row['ap']:.4f} | {row['threshold']:.3f} | "
            f"{row['sens']:.3f} ({row['tp']}/{row['n_pos']}) | "
            f"{row['spec']:.3f} ({row['tn']}/{row['n_neg']}) |"
        )
    lines += [
        "",
        "`3cl_p_mel` = bundle actuel. `3cl_readout` = p_mel/(p_mel+p_nevus).",
        "`B_retrain` = têtes binaire réentraînées. Le seuil 0.188 est hérité ;",
        "`B_retrain@Youden` est calé sur valid interne binaire, donc descriptif OOD.",
        "",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    paths.BINARY_RETRAIN_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-isic2020", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(paths.INTERNAL_METADATA)
    x_a, x_c, y, _ = load_v2s_matrices("deployed_single", meta)
    y_bin = (y == 1).astype(int)
    nm = np.isin(y, (0, 1))
    train_mask = meta["split_original"].isin(["train", "valid"]).to_numpy() & nm
    valid_mask = (meta["split_original"] == "valid").to_numpy() & nm
    test_mask = (meta["split_original"] == "test").to_numpy()
    test_nm = test_mask & nm

    pipe_a, pipe_c = fit_binary_heads(x_a, x_c, y_bin, train_mask)
    p_valid = score_binary(pipe_a, pipe_c, x_a[valid_mask], x_c[valid_mask])
    thr_youden = youden(y_bin[valid_mask], p_valid)
    print(f"Youden binary on valid nm: T={thr_youden:.4f}", flush=True)

    p_test = score_binary(pipe_a, pipe_c, x_a[test_mask], x_c[test_mask])
    internal_pred = pd.read_csv(paths.INTERNAL_PREDICTIONS)
    internal_pred = internal_pred[internal_pred["condition"] == "1a"].copy()
    if len(internal_pred) != int(test_mask.sum()):
        raise ValueError("internal 1a row count mismatch")
    s3 = internal_pred["p_melanoma"].to_numpy(float)
    s_read = readout(s3, internal_pred["p_nevus"].to_numpy(float))
    y_test = y_bin[test_mask]
    y_test_nm = y_bin[test_nm]
    # align 1a scores to test_nm via class filter on the 1a table
    one_a_nm = internal_pred[internal_pred["class_name"].isin(["nevus", "melanoma"])]
    s3_nm = one_a_nm["p_melanoma"].to_numpy(float)
    s_read_nm = readout(s3_nm, one_a_nm["p_nevus"].to_numpy(float))
    p_test_nm = score_binary(pipe_a, pipe_c, x_a[test_nm], x_c[test_nm])

    rows = [
        metrics_row("internal_test_mel_vs_rest", "3cl_p_mel", y_test, s3, T_MEL),
        metrics_row("internal_test_mel_vs_rest", "3cl_readout", y_test, s_read, T_MEL),
        metrics_row("internal_test_mel_vs_rest", "B_retrain@0.188", y_test, p_test, T_MEL),
        metrics_row("internal_test_mel_vs_rest", "B_retrain@Youden", y_test, p_test, thr_youden),
        metrics_row("internal_test_mel_vs_nevus", "3cl_p_mel", y_test_nm, s3_nm, T_MEL),
        metrics_row("internal_test_mel_vs_nevus", "3cl_readout", y_test_nm, s_read_nm, T_MEL),
        metrics_row("internal_test_mel_vs_nevus", "B_retrain@0.188", y_test_nm, p_test_nm, T_MEL),
        metrics_row("internal_test_mel_vs_nevus", "B_retrain@Youden", y_test_nm, p_test_nm, thr_youden),
    ]

    atyp_mask = test_mask & (y == 2)
    p_atyp = score_binary(pipe_a, pipe_c, x_a[atyp_mask], x_c[atyp_mask])
    to_mel = int((p_atyp >= 0.5).sum())
    to_nev = int((p_atyp < 0.5).sum())
    atypical_trace = (
        f"True-atypical on internal test under B_retrain @0.5: "
        f"{to_nev}/{len(p_atyp)} -> nevus ({100 * to_nev / len(p_atyp):.1f}%), "
        f"{to_mel}/{len(p_atyp)} -> melanoma ({100 * to_mel / len(p_atyp):.1f}%)."
    )
    print(atypical_trace, flush=True)
    pd.DataFrame(
        {
            "file": meta.loc[atyp_mask, "file"].to_numpy(),
            "p_melanoma_binary": p_atyp,
            "pred_at_0_5": np.where(p_atyp >= 0.5, "melanoma", "nevus"),
        }
    ).to_csv(OUT / "binary_retrain_internal_atypical_trace.csv", index=False)

    from derm.export.bundle_runtime import BundleEngine
    from derm.export.eval_external import IMG_EXTS

    engine = BundleEngine(
        paths.APP_MODELS,
        batch_size=args.batch_size,
        workers=args.workers,
        device="cuda",
    )
    print(engine.describe(), flush=True)

    hospital_items: list[tuple[Path, str, int]] = []
    for folder, cid in (("Nevi", 0), ("Melanom", 1)):
        for path in sorted((paths.EXTERNAL_VAL / folder).iterdir()):
            if path.suffix.lower() in IMG_EXTS:
                hospital_items.append((path, folder, cid))
    hosp = extract_cohort(engine, hospital_items, OUT / "features_hospital.npz")
    hx_a, hx_c = matrices_from_npz(hosp)
    h_score = score_binary(pipe_a, pipe_c, hx_a, hx_c)
    hosp_ref = pd.read_csv(paths.EXTERNAL_PARITY_CSV)
    hosp_ref = hosp_ref.set_index("file").loc[list(hosp["filenames"])]
    h_s3 = hosp_ref["p_mel"].to_numpy(float)
    h_read = readout(h_s3, hosp_ref["p_nevus"].to_numpy(float))
    h_y = hosp["true"].astype(int)
    rows += [
        metrics_row("hospital", "3cl_p_mel", h_y, h_s3, T_MEL),
        metrics_row("hospital", "3cl_readout", h_y, h_read, T_MEL),
        metrics_row("hospital", "B_retrain@0.188", h_y, h_score, T_MEL),
        metrics_row("hospital", "B_retrain@Youden", h_y, h_score, thr_youden),
    ]
    pd.DataFrame(
        {
            "file": hosp["filenames"],
            "true": h_y,
            "p_mel_3cl": h_s3,
            "p_mel_readout": h_read,
            "p_mel_retrain": h_score,
        }
    ).to_csv(OUT / "binary_retrain_hospital.csv", index=False)

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
        ood = extract_cohort(engine, ood_items, OUT / "features_isic2020.npz")
        ox_a, ox_c = matrices_from_npz(ood)
        o_score = score_binary(pipe_a, pipe_c, ox_a, ox_c)
        full = pd.read_csv(paths.RESULTS_OOD_ISIC2020 / "full" / "ood_isic2020_predictions.csv")
        full = full.set_index("file").loc[list(ood["filenames"])]
        o_s3 = full["p_mel"].to_numpy(float)
        o_read = readout(o_s3, full["p_nevus"].to_numpy(float))
        o_y = ood["true"].astype(int)
        rows += [
            metrics_row("isic2020_full", "3cl_p_mel", o_y, o_s3, T_MEL),
            metrics_row("isic2020_full", "3cl_readout", o_y, o_read, T_MEL),
            metrics_row("isic2020_full", "B_retrain@0.188", o_y, o_score, T_MEL),
            metrics_row("isic2020_full", "B_retrain@Youden", o_y, o_score, thr_youden),
        ]
        pd.DataFrame(
            {
                "file": ood["filenames"],
                "true": o_y,
                "coverage": ood["coverage"],
                "p_mel_3cl": o_s3,
                "p_mel_readout": o_read,
                "p_mel_retrain": o_score,
            }
        ).to_csv(OUT / "binary_retrain_isic2020.csv", index=False)
        nevus_files = set(kept.loc[kept["diagnosis"] == "nevus", "file"].astype(str))
        mel_files = set(kept.loc[kept["true"] == 1, "file"].astype(str))
        labelled = np.isin(ood["filenames"].astype(str), list(nevus_files | mel_files))
        rows += [
            metrics_row(
                "isic2020_mel_vs_labelled_nevus",
                "3cl_p_mel",
                o_y[labelled],
                o_s3[labelled],
                T_MEL,
            ),
            metrics_row(
                "isic2020_mel_vs_labelled_nevus",
                "3cl_readout",
                o_y[labelled],
                o_read[labelled],
                T_MEL,
            ),
            metrics_row(
                "isic2020_mel_vs_labelled_nevus",
                "B_retrain@0.188",
                o_y[labelled],
                o_score[labelled],
                T_MEL,
            ),
        ]

    engine.close()
    pd.DataFrame(rows).to_csv(paths.BINARY_RETRAIN_METRICS, index=False)
    write_report(rows, atypical_trace)
    print(f"wrote {paths.BINARY_RETRAIN_METRICS}", flush=True)


if __name__ == "__main__":
    main()
