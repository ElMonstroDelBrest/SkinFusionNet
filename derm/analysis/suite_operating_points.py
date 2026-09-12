"""Abstention by U-Net coverage and frozen internal operating points on OOD.

Thresholds are chosen on internal validation only, then applied untouched
to ISIC 2020 and the hospital set. Coverage gates are pre-specified.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from derm import paths
from derm.analysis.internal_validity_m1 import load_v2s_matrices

OUT = paths.RESULTS / "suite"
T_LEGACY = 0.188
COVERAGE_CUTS = (0.02, 0.05, 0.10, 0.20)


def wilson(k: int, n: int) -> tuple[float, float, float]:
    if n <= 0:
        return math.nan, math.nan, math.nan
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def youden(y: np.ndarray, s: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, s)
    return float(thr[int(np.argmax(tpr - fpr))])


def threshold_for_spec(y: np.ndarray, s: np.ndarray, target: float) -> float:
    fpr, tpr, thr = roc_curve(y, s)
    spec = 1.0 - fpr
    # highest sensitivity among points with spec >= target
    ok = np.where(spec >= target)[0]
    if len(ok) == 0:
        return float(thr[-1])
    idx = ok[int(np.argmax(tpr[ok]))]
    return float(thr[idx])


def row(dataset: str, arm: str, y: np.ndarray, s: np.ndarray, t: float, *, n_abstain: int = 0) -> dict:
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    pred = s >= t
    tp = int(((y == 1) & pred).sum())
    tn = int(((y == 0) & ~pred).sum())
    se, sl, sh = wilson(tp, n_pos)
    sp, pl, ph = wilson(tn, n_neg)
    auc = float(roc_auc_score(y, s)) if n_pos and n_neg else math.nan
    ap = float(average_precision_score(y, s)) if n_pos and n_neg else math.nan
    n = int(len(y))
    return {
        "dataset": dataset,
        "arm": arm,
        "n_scored": n,
        "n_abstain": int(n_abstain),
        "coverage_kept": n / (n + n_abstain) if (n + n_abstain) else math.nan,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "auc": auc,
        "ap": ap,
        "threshold": float(t),
        "sens": se,
        "sens_ci_low": sl,
        "sens_ci_high": sh,
        "spec": sp,
        "spec_ci_low": pl,
        "spec_ci_high": ph,
        "tp": tp,
        "fn": n_pos - tp,
        "tn": tn,
        "fp": n_neg - tn,
        "mel_abstained": math.nan,
    }


def make_lgbm(*, binary: bool) -> LGBMClassifier:
    kwargs = dict(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        n_jobs=4,
        verbose=-1,
    )
    if binary:
        kwargs["objective"] = "binary"
    else:
        kwargs["objective"] = "multiclass"
        kwargs["num_class"] = 3
    return LGBMClassifier(**kwargs)


def fit_ensemble(x_a, x_c, y, mask, *, binary: bool) -> tuple[Pipeline, Pipeline]:
    pa = Pipeline([("scaler", StandardScaler()), ("clf", make_lgbm(binary=binary))])
    pc = Pipeline([("scaler", StandardScaler()), ("clf", make_lgbm(binary=binary))])
    pa.fit(x_a[mask], y[mask])
    pc.fit(x_c[mask], y[mask])
    return pa, pc


def score_mel(pa: Pipeline, pc: Pipeline, x_a, x_c, *, binary: bool) -> np.ndarray:
    a = pa.predict_proba(x_a)
    c = pc.predict_proba(x_c)
    p = (a + c) / 2.0
    return p[:, 1]


def abstention_table(dataset: str, arm: str, y, s, cov, t_dec: float) -> list[dict]:
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    cov = np.asarray(cov, float)
    rows = []
    rows.append(row(dataset, f"{arm} / no-abstention", y, s, t_dec, n_abstain=0))
    rows[-1]["mel_abstained"] = 0
    for cut in COVERAGE_CUTS:
        keep = cov >= cut
        n_abs = int((~keep).sum())
        mel_abs = int(((~keep) & (y == 1)).sum())
        r = row(
            dataset,
            f"{arm} / abstain cov<{cut:.2f}",
            y[keep],
            s[keep],
            t_dec,
            n_abstain=n_abs,
        )
        r["mel_abstained"] = mel_abs
        rows.append(r)
        # treat abstain as review (positive for detection)
        pred_review = (s >= t_dec) | (~keep)
        tp = int(((y == 1) & pred_review).sum())
        tn = int(((y == 0) & ~pred_review).sum())
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        se, sl, sh = wilson(tp, n_pos)
        sp, pl, ph = wilson(tn, n_neg)
        rows.append(
            {
                "dataset": dataset,
                "arm": f"{arm} / abstain-as-review cov<{cut:.2f}",
                "n_scored": int(len(y)),
                "n_abstain": n_abs,
                "coverage_kept": float(keep.mean()),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "auc": math.nan,
                "ap": math.nan,
                "threshold": float(t_dec),
                "sens": se,
                "sens_ci_low": sl,
                "sens_ci_high": sh,
                "spec": sp,
                "spec_ci_low": pl,
                "spec_ci_high": ph,
                "tp": tp,
                "fn": n_pos - tp,
                "tn": tn,
                "fp": n_neg - tn,
                "mel_abstained": mel_abs,
            }
        )
    return rows


def load_npz(path: Path) -> dict:
    blob = np.load(path, allow_pickle=True)
    return {k: blob[k] for k in blob.files}


def matrices(blob: dict) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([blob["handcraft"], blob["lesion"]], axis=1),
        np.concatenate([blob["handcraft"], blob["dehair"]], axis=1),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(paths.INTERNAL_METADATA)
    x_a, x_c, y3, _ = load_v2s_matrices("deployed_single", meta)
    y_bin = (y3 == 1).astype(int)
    train = meta["split_original"].eq("train").to_numpy()
    valid = meta["split_original"].eq("valid").to_numpy()
    test = meta["split_original"].eq("test").to_numpy()
    nm = np.isin(y3, (0, 1))

    print("fitting 3-class heads on TRAIN only", flush=True)
    pa3, pc3 = fit_ensemble(x_a, x_c, y3, train, binary=False)
    p_valid_3 = score_mel(pa3, pc3, x_a[valid], x_c[valid], binary=False)
    t_youden_3 = youden(y_bin[valid], p_valid_3)
    t_spec90_3 = threshold_for_spec(y_bin[valid], p_valid_3, 0.90)
    t_spec95_3 = threshold_for_spec(y_bin[valid], p_valid_3, 0.95)
    print(
        f"3cl TRAIN-only  Youden={t_youden_3:.4f}  spec90={t_spec90_3:.4f}  spec95={t_spec95_3:.4f}",
        flush=True,
    )

    print("fitting binary heads on TRAIN nm only", flush=True)
    pa2, pc2 = fit_ensemble(x_a, x_c, y_bin, train & nm, binary=True)
    p_valid_2 = score_mel(pa2, pc2, x_a[valid & nm], x_c[valid & nm], binary=True)
    t_youden_2 = youden(y_bin[valid & nm], p_valid_2)
    t_spec90_2 = threshold_for_spec(y_bin[valid & nm], p_valid_2, 0.90)
    t_spec95_2 = threshold_for_spec(y_bin[valid & nm], p_valid_2, 0.95)
    print(
        f"2cl TRAIN-only  Youden={t_youden_2:.4f}  spec90={t_spec90_2:.4f}  spec95={t_spec95_2:.4f}",
        flush=True,
    )

    p_test_3 = score_mel(pa3, pc3, x_a[test], x_c[test], binary=False)
    p_test_2 = score_mel(pa2, pc2, x_a[test], x_c[test], binary=True)
    y_test = y_bin[test]
    y_test_nm = y_bin[test & nm]
    p_test_3_nm = score_mel(pa3, pc3, x_a[test & nm], x_c[test & nm], binary=False)
    p_test_2_nm = score_mel(pa2, pc2, x_a[test & nm], x_c[test & nm], binary=True)

    rows: list[dict] = []
    for name, y, s, t in (
        ("internal_test_3cl_heads / legacy 0.188", y_test, p_test_3, T_LEGACY),
        ("internal_test_3cl_heads / Youden-valid", y_test, p_test_3, t_youden_3),
        ("internal_test_3cl_heads / spec90-valid", y_test, p_test_3, t_spec90_3),
        ("internal_test_3cl_heads / spec95-valid", y_test, p_test_3, t_spec95_3),
        ("internal_test_2cl_heads / legacy 0.188", y_test, p_test_2, T_LEGACY),
        ("internal_test_2cl_heads / Youden-valid", y_test, p_test_2, t_youden_2),
        ("internal_test_2cl_heads / spec90-valid", y_test, p_test_2, t_spec90_2),
        ("internal_test_nm_3cl / Youden-valid", y_test_nm, p_test_3_nm, t_youden_3),
        ("internal_test_nm_2cl / Youden-valid", y_test_nm, p_test_2_nm, t_youden_2),
    ):
        dataset, arm = name.split(" / ", 1)
        rows.append(row(dataset, arm, y, s, t))

    ood = load_npz(paths.RESULTS_BINARY_RETRAIN / "features_isic2020.npz")
    ox_a, ox_c = matrices(ood)
    o_y = ood["true"].astype(int)
    o_cov = ood["coverage"].astype(float)
    o_p3 = score_mel(pa3, pc3, ox_a, ox_c, binary=False)
    o_p2 = score_mel(pa2, pc2, ox_a, ox_c, binary=True)

    deployed = pd.read_csv(paths.RESULTS_OOD_ISIC2020 / "full" / "ood_isic2020_predictions.csv")
    deployed = deployed.set_index("file").loc[list(ood["filenames"])]
    d_p3 = deployed["p_mel"].to_numpy(float)
    d_cov = deployed["coverage"].to_numpy(float)
    retrain = pd.read_csv(paths.RESULTS_BINARY_RETRAIN / "binary_retrain_isic2020.csv")
    retrain = retrain.set_index("file").loc[list(ood["filenames"])]
    r_p2 = retrain["p_mel_retrain"].to_numpy(float)

    for dataset, arm, y, s, t, cov in (
        ("isic2020", "deployed_3cl / legacy 0.188", o_y, d_p3, T_LEGACY, d_cov),
        ("isic2020", "deployed_3cl / Youden-valid (train-only 3cl heads T)", o_y, d_p3, t_youden_3, d_cov),
        ("isic2020", "nested_3cl_heads / Youden-valid", o_y, o_p3, t_youden_3, o_cov),
        ("isic2020", "nested_3cl_heads / spec90-valid", o_y, o_p3, t_spec90_3, o_cov),
        ("isic2020", "nested_3cl_heads / spec95-valid", o_y, o_p3, t_spec95_3, o_cov),
        ("isic2020", "nested_2cl_heads / Youden-valid", o_y, o_p2, t_youden_2, o_cov),
        ("isic2020", "nested_2cl_heads / spec90-valid", o_y, o_p2, t_spec90_2, o_cov),
        ("isic2020", "B_retrain_tv / Youden-valid-nm", o_y, r_p2, 0.5993142922697512, o_cov),
        ("isic2020", "B_retrain_tv / legacy 0.188", o_y, r_p2, T_LEGACY, o_cov),
    ):
        rows.append(row(dataset, arm, y, s, t))

    rows += abstention_table("isic2020", "deployed_3cl@0.188", o_y, d_p3, d_cov, T_LEGACY)
    rows += abstention_table("isic2020", "nested_2cl@Youden", o_y, o_p2, o_cov, t_youden_2)
    rows += abstention_table("isic2020", "B_retrain@0.188", o_y, r_p2, o_cov, T_LEGACY)

    hosp = load_npz(paths.RESULTS_BINARY_RETRAIN / "features_hospital.npz")
    hx_a, hx_c = matrices(hosp)
    h_y = hosp["true"].astype(int)
    h_cov = hosp["coverage"].astype(float)
    h_p3 = score_mel(pa3, pc3, hx_a, hx_c, binary=False)
    h_p2 = score_mel(pa2, pc2, hx_a, hx_c, binary=True)
    href = pd.read_csv(paths.EXTERNAL_PARITY_CSV).set_index("file").loc[list(hosp["filenames"])]
    hd = href["p_mel"].to_numpy(float)
    rows.append(row("hospital", "deployed_3cl / legacy 0.188", h_y, hd, T_LEGACY))
    rows.append(row("hospital", "nested_3cl_heads / Youden-valid", h_y, h_p3, t_youden_3))
    rows.append(row("hospital", "nested_2cl_heads / Youden-valid", h_y, h_p2, t_youden_2))
    rows += abstention_table("hospital", "deployed_3cl@0.188", h_y, hd, h_cov, T_LEGACY)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "operating_points.csv", index=False)

    lines = [
        "# Frozen operating points and coverage abstention",
        "",
        "Thresholds below were fit on **internal valid only** (heads fit on **train only**).",
        "ISIC 2020 and the hospital set were not used to choose a threshold.",
        "",
        f"- 3-class Youden (valid, mel vs rest): `{t_youden_3:.4f}`",
        f"- 3-class spec≥0.90: `{t_spec90_3:.4f}`",
        f"- 3-class spec≥0.95: `{t_spec95_3:.4f}`",
        f"- 2-class Youden (valid, nevus+melanoma only): `{t_youden_2:.4f}`",
        f"- 2-class spec≥0.90: `{t_spec90_2:.4f}`",
        f"- 2-class spec≥0.95: `{t_spec95_2:.4f}`",
        "",
        "Full table: `results/suite/operating_points.csv`.",
        "",
    ]
    (OUT / "operating_points.md").write_text("\n".join(lines), encoding="utf-8")
    print(df.to_string(index=False))
    print("wrote", OUT / "operating_points.csv")


if __name__ == "__main__":
    main()
