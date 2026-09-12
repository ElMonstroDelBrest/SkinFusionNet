"""M1 ablation of the learned `atypical` class.

Scope:
  - external B_readout from external_val/per_image.csv
  - internal A / B_readout / B_retrain on baseline 512-d features
  - no C arm, no GPU, no external B_retrain re-extraction
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from derm import paths


T_MEL = 0.188
T_ATYP = 0.412
EPS = 1e-12
TARGET_SPECS = (0.80, 0.90)
OUT_SUMMARY = paths.ABLATION_ATYPICAL_CSV
OUT_CASES = paths.ABLATION_ATYPICAL_CASES
OUT_REPORT = paths.ABLATION_ATYPICAL_REPORT


@dataclass(frozen=True)
class ArmScores:
    arm: str
    dataset: str
    y_binary: np.ndarray
    score: np.ndarray
    status: str = "ok"
    reason: str = ""


def make_model() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return math.nan, math.nan, math.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def delong_components(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    m, n = len(pos), len(neg)
    tx = midrank(pos)
    ty = midrank(neg)
    tz = midrank(np.r_[pos, neg])
    auc = (tz[:m].sum() / m - (m + 1) / 2) / n
    v01 = (tz[:m] - tx) / n
    v10 = 1 - (tz[m:] - ty) / m
    return float(auc), v01, v10


def delong_ci(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float, float]:
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return math.nan, math.nan, math.nan
    auc, v01, v10 = delong_components(y_true, scores)
    m, n = len(v01), len(v10)
    sx = np.var(v01, ddof=1) if m > 1 else 0.0
    sy = np.var(v10, ddof=1) if n > 1 else 0.0
    se = math.sqrt(max(float(sx / m + sy / n), 0.0))
    z = norm.ppf(0.975)
    return auc, max(0.0, auc - z * se), min(1.0, auc + z * se)


def delong_paired_test(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> tuple[float, float, float]:
    auc_a, a01, a10 = delong_components(y_true, scores_a)
    auc_b, b01, b10 = delong_components(y_true, scores_b)
    m, n = len(a01), len(a10)
    diff01 = b01 - a01
    diff10 = b10 - a10
    var = 0.0
    if m > 1:
        var += float(np.var(diff01, ddof=1) / m)
    if n > 1:
        var += float(np.var(diff10, ddof=1) / n)
    if var <= 0:
        p = 1.0 if abs(auc_b - auc_a) < 1e-15 else 0.0
    else:
        z = (auc_b - auc_a) / math.sqrt(var)
        p = 2 * (1 - norm.cdf(abs(z)))
    return auc_a, auc_b, p


def threshold_for_min_specificity(
    y_binary: np.ndarray,
    score: np.ndarray,
    target_spec: float,
) -> tuple[float, float]:
    y_binary = np.asarray(y_binary, dtype=int)
    score = np.asarray(score, dtype=float)
    neg = score[y_binary == 0]
    thresholds = np.r_[np.inf, np.sort(np.unique(score))[::-1], -np.inf]
    best_t = thresholds[0]
    best_sens = -1.0
    best_spec = math.nan
    for t in thresholds:
        pred = score >= t
        tn = int(((~pred) & (y_binary == 0)).sum())
        fp = int((pred & (y_binary == 0)).sum())
        tp = int((pred & (y_binary == 1)).sum())
        fn = int(((~pred) & (y_binary == 1)).sum())
        spec = tn / (tn + fp) if (tn + fp) else math.nan
        sens = tp / (tp + fn) if (tp + fn) else math.nan
        if spec >= target_spec and sens > best_sens:
            best_t = float(t)
            best_sens = float(sens)
            best_spec = float(spec)
    return best_t, best_spec


def metrics_at_threshold(
    y_binary: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    pred = score >= threshold
    tp = int((pred & (y_binary == 1)).sum())
    fn = int(((~pred) & (y_binary == 1)).sum())
    tn = int(((~pred) & (y_binary == 0)).sum())
    fp = int((pred & (y_binary == 0)).sum())
    sens, sens_lo, sens_hi = wilson(tp, tp + fn)
    spec, spec_lo, spec_hi = wilson(tn, tn + fp)
    return {
        "threshold": threshold,
        "sensitivity": sens,
        "sensitivity_ci95_low": sens_lo,
        "sensitivity_ci95_high": sens_hi,
        "specificity": spec,
        "specificity_ci95_low": spec_lo,
        "specificity_ci95_high": spec_hi,
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
    }


def summarize_arm(arm_scores: ArmScores, target_spec: float) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset": arm_scores.dataset,
        "arm": arm_scores.arm,
        "target_specificity": target_spec,
        "status": arm_scores.status,
        "reason": arm_scores.reason,
    }
    if arm_scores.status != "ok":
        return row
    auc, lo, hi = delong_ci(arm_scores.y_binary, arm_scores.score)
    threshold, achieved = threshold_for_min_specificity(
        arm_scores.y_binary,
        arm_scores.score,
        target_spec,
    )
    m = metrics_at_threshold(arm_scores.y_binary, arm_scores.score, threshold)
    row.update(
        {
            "n": len(arm_scores.y_binary),
            "n_melanoma": int((arm_scores.y_binary == 1).sum()),
            "n_nevus": int((arm_scores.y_binary == 0).sum()),
            "auc_mel_vs_nevus": auc,
            "auc_ci95_low": lo,
            "auc_ci95_high": hi,
            "threshold": threshold,
            "achieved_specificity": achieved,
            **m,
        }
    )
    return row


def a_score_from_probs(p_nevus: np.ndarray, p_mel: np.ndarray, p_atyp: np.ndarray) -> np.ndarray:
    return np.maximum(p_mel / T_MEL, p_atyp / T_ATYP)


def b_readout_score(p_nevus: np.ndarray, p_mel: np.ndarray) -> np.ndarray:
    denom = p_nevus + p_mel
    return np.divide(p_mel, denom, out=np.zeros_like(p_mel, dtype=float), where=denom > EPS)


def external_scores() -> tuple[list[ArmScores], pd.DataFrame, dict[str, object]]:
    df = pd.read_csv(paths.EXTERNAL_VAL / "per_image.csv")
    y = (df["true"].to_numpy(int) == 1).astype(int)
    p_nev = df["p_nevus"].to_numpy(float)
    p_mel = df["p_mel"].to_numpy(float)
    p_atyp = df["p_atyp"].to_numpy(float)
    score_a = a_score_from_probs(p_nev, p_mel, p_atyp)
    score_b = b_readout_score(p_nev, p_mel)
    clinical_a = (p_mel >= T_MEL) | (p_atyp >= T_ATYP)
    clinical_a_m = metrics_at_threshold(y, clinical_a.astype(float), 0.5)
    arms = [
        ArmScores("A", "external", y, score_a),
        ArmScores("B_readout", "external", y, score_b),
        ArmScores(
            "B_retrain",
            "external",
            y,
            np.zeros_like(y, dtype=float),
            "blocked_missing_features",
            "external canonical V2S features for 107 images were not present; re-extraction is M1.5",
        ),
    ]
    extra = {
        "A_not_benign_clinical_sensitivity": clinical_a_m["sensitivity"],
        "A_not_benign_clinical_specificity": clinical_a_m["specificity"],
        "A_not_benign_clinical_tp": clinical_a_m["tp"],
        "A_not_benign_clinical_fn": clinical_a_m["fn"],
        "A_not_benign_clinical_tn": clinical_a_m["tn"],
        "A_not_benign_clinical_fp": clinical_a_m["fp"],
    }
    return arms, df, extra


def load_internal_features() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hc = pd.read_csv(paths.FEATURES_CSV)
    cnn_a = pd.read_csv(paths.CNN_V2S_FEATS_CSV)
    cnn_c = pd.read_csv(paths.CNN_V2S_DEHAIR_CSV)
    if not (len(hc) == len(cnn_a) == len(cnn_c)):
        raise ValueError(f"row mismatch hc={len(hc)} a={len(cnn_a)} c={len(cnn_c)}")
    y = hc["Class"].to_numpy(int)
    if not np.array_equal(y, cnn_a["Class"].to_numpy(int)):
        raise ValueError("Class mismatch features.csv vs V2S lesion CSV")
    if not np.array_equal(y, cnn_c["Class"].to_numpy(int)):
        raise ValueError("Class mismatch features.csv vs V2S dehair CSV")
    x_hc = hc.drop(columns=["Class"]).to_numpy(float)
    cols_a = [c for c in cnn_a.columns if c.startswith("cnn_")]
    cols_c = [c for c in cnn_c.columns if c.startswith("cnn_")]
    if cols_a != cols_c:
        raise ValueError("CNN feature columns differ between lesion and dehair")
    x_a = np.c_[x_hc, cnn_a[cols_a].to_numpy(float)]
    x_c = np.c_[x_hc, cnn_c[cols_c].to_numpy(float)]
    return x_a, x_c, y


def fit_predict_fold(
    x_a: np.ndarray,
    x_c: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    train_mask: np.ndarray | None = None,
) -> np.ndarray:
    if train_mask is None:
        tr = train_idx
    else:
        tr = train_idx[train_mask[train_idx]]
    pa = Pipeline([("scaler", StandardScaler()), ("clf", make_model())])
    pc = Pipeline([("scaler", StandardScaler()), ("clf", make_model())])
    pa.fit(x_a[tr], y[tr])
    pc.fit(x_c[tr], y[tr])
    return (pa.predict_proba(x_a[test_idx]) + pc.predict_proba(x_c[test_idx])) / 2.0


def internal_scores() -> tuple[list[ArmScores], dict[str, object], pd.DataFrame]:
    x_a, x_c, y = load_internal_features()
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    p3 = np.zeros((len(y), 3), dtype=float)
    p2 = np.full(len(y), np.nan, dtype=float)
    pred_atyp_b = np.full(len(y), -1, dtype=int)

    for fold, (tr, te) in enumerate(cv.split(x_a, y), 1):
        p3[te] = fit_predict_fold(x_a, x_c, y, tr, te)
        train_nm = y != 2
        p2_fold = fit_predict_fold(x_a, x_c, y, tr, te, train_mask=train_nm)
        classes = np.array([0, 1])
        # LGBM keeps columns in sorted class order for this binary fit.
        p2[te] = p2_fold[:, list(classes).index(1)]
        atyp_te = te[y[te] == 2]
        if len(atyp_te):
            pred_atyp_b[atyp_te] = (p2[atyp_te] >= 0.5).astype(int)
        print(f"internal fold {fold}/10", flush=True)

    nm = y != 2
    y_nm = (y[nm] == 1).astype(int)
    p_nev = p3[nm, 0]
    p_mel = p3[nm, 1]
    p_atyp = p3[nm, 2]
    score_a = a_score_from_probs(p_nev, p_mel, p_atyp)
    score_b_readout = b_readout_score(p_nev, p_mel)
    score_b_retrain = p2[nm]

    auc_a, auc_b, p_delong = delong_paired_test(y_nm, p_mel, score_b_retrain)
    atyp = y == 2
    atyp_scores = p2[atyp]
    atyp_pred = (atyp_scores >= 0.5).astype(int)
    atyp_summary = {
        "internal_delong_auc_A_pmel": auc_a,
        "internal_delong_auc_B_retrain": auc_b,
        "internal_delong_p_A_vs_B_retrain": p_delong,
        "true_atypical_n": int(atyp.sum()),
        "true_atypical_to_nevus_at_0_5": int((atyp_pred == 0).sum()),
        "true_atypical_to_melanoma_at_0_5": int((atyp_pred == 1).sum()),
        "true_atypical_p_mel_median": float(np.median(atyp_scores)),
        "true_atypical_p_mel_p90": float(np.quantile(atyp_scores, 0.90)),
    }
    atyp_by_fold = pd.DataFrame(
        {
            "p_mel_B_retrain": atyp_scores,
            "pred_B_retrain_at_0_5": np.where(atyp_pred == 1, "melanoma", "nevus"),
        }
    )
    arms = [
        ArmScores("A", "internal", y_nm, score_a),
        ArmScores("B_readout", "internal", y_nm, score_b_readout),
        ArmScores("B_retrain", "internal", y_nm, score_b_retrain),
    ]
    return arms, atyp_summary, atyp_by_fold


def build_external_cases(df: pd.DataFrame) -> pd.DataFrame:
    mel = df[df["true"] == 1].copy()
    p_nev = mel["p_nevus"].to_numpy(float)
    p_mel = mel["p_mel"].to_numpy(float)
    p_atyp = mel["p_atyp"].to_numpy(float)
    score_b = b_readout_score(p_nev, p_mel)
    clinical_a_alert = p_mel >= T_MEL
    clinical_a_watch = (~clinical_a_alert) & (p_atyp >= T_ATYP)
    clinical_a_not_benign = clinical_a_alert | clinical_a_watch

    # Thresholds on the full external set at S*=0.90, used as the case-level B readout.
    y_all = (df["true"].to_numpy(int) == 1).astype(int)
    b_all = b_readout_score(df["p_nevus"].to_numpy(float), df["p_mel"].to_numpy(float))
    t_b90, spec_b90 = threshold_for_min_specificity(y_all, b_all, 0.90)
    out = mel[["file", "folder", "p_nevus", "p_mel", "p_atyp", "argmax", "coverage"]].copy()
    out["p_mel_readout2"] = score_b
    out["A_alert_mel"] = clinical_a_alert
    out["A_watch_atypical"] = clinical_a_watch
    out["A_not_benign"] = clinical_a_not_benign
    out["B_readout_alert_at_S90"] = score_b >= t_b90
    out["B_readout_threshold_S90"] = t_b90
    out["B_readout_spec_S90"] = spec_b90
    out["case_bucket"] = np.where(
        (p_atyp >= T_ATYP) & (p_mel < T_MEL),
        "atypical_well",
        np.where((p_nev > 0.997) & (p_mel < T_MEL), "nevus_ceiling", "other"),
    )
    return out


def markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_empty_"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        vals = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, float):
                vals.append(format(v, floatfmt))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(
    summary: pd.DataFrame,
    cases: pd.DataFrame,
    external_extra: dict[str, object],
    internal_extra: dict[str, object],
) -> None:
    b_ext_90 = summary[
        (summary["dataset"] == "external")
        & (summary["arm"] == "B_readout")
        & (summary["target_specificity"] == 0.90)
    ].iloc[0]
    a_ext_90 = summary[
        (summary["dataset"] == "external")
        & (summary["arm"] == "A")
        & (summary["target_specificity"] == 0.90)
    ].iloc[0]
    b_ext_80 = summary[
        (summary["dataset"] == "external")
        & (summary["arm"] == "B_readout")
        & (summary["target_specificity"] == 0.80)
    ].iloc[0]
    auc_delta = (
        internal_extra["internal_delong_auc_B_retrain"]
        - internal_extra["internal_delong_auc_A_pmel"]
    )
    p_delong = internal_extra["internal_delong_p_A_vs_B_retrain"]
    p_text = "<1e-300" if p_delong == 0 else f"{p_delong:.3g}"
    atyp_nev = internal_extra["true_atypical_to_nevus_at_0_5"]
    atyp_mel = internal_extra["true_atypical_to_melanoma_at_0_5"]
    atyp_n = internal_extra["true_atypical_n"]
    well = cases[cases["case_bucket"] == "atypical_well"]
    ceiling = cases[cases["case_bucket"] == "nevus_ceiling"]
    lines = [
        "# Ablation atypical M1",
        "",
        "## Decision externe",
        "",
        f"- A not-benign clinique : {int(external_extra['A_not_benign_clinical_tp'])}/19 melanomes, specificite={external_extra['A_not_benign_clinical_specificity']:.3f}.",
        f"- B_readout @S*=0.90 : {int(b_ext_90['tp'])}/19 melanomes, specificite={b_ext_90['specificity']:.3f}.",
        f"- B_readout @S*=0.80 : {int(b_ext_80['tp'])}/19 melanomes, specificite={b_ext_80['specificity']:.3f}.",
        f"- A composite @S*=0.90 : {int(a_ext_90['tp'])}/19 melanomes, specificite={a_ext_90['specificity']:.3f}.",
        "",
        "## Puits et plafond",
        "",
        f"- Cas puits atypical detectes par B_readout @S*=0.90 : {int(well['B_readout_alert_at_S90'].sum())}/{len(well)}.",
        f"- Cas plafond nevus detectes par B_readout @S*=0.90 : {int(ceiling['B_readout_alert_at_S90'].sum())}/{len(ceiling)}.",
        "",
        "## Interne baseline single-pass",
        "",
        f"- AUC A p_mel={internal_extra['internal_delong_auc_A_pmel']:.4f}; AUC B_retrain={internal_extra['internal_delong_auc_B_retrain']:.4f}; delta={auc_delta:+.4f}; p DeLong apparie={p_text}.",
        f"- True-atypical sous B_retrain a 0.5 : {atyp_nev} vers nevus ({100 * atyp_nev / atyp_n:.1f}%), {atyp_mel} vers melanoma ({100 * atyp_mel / atyp_n:.1f}%), n={atyp_n}.",
        "",
        "## Tableau principal",
        "",
        markdown_table(summary),
        "",
        "## Cas externes melanomes",
        "",
        markdown_table(cases),
        "",
        "Note : B_retrain externe est hors M1, faute de features baseline externes pre-extraites.",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n")


def main() -> None:
    paths.RESULTS_ABLATION_ATYPICAL.mkdir(parents=True, exist_ok=True)
    ext_arms, ext_df, ext_extra = external_scores()
    int_arms, int_extra, atyp_desc = internal_scores()

    rows = []
    for arm in ext_arms + int_arms:
        for target in TARGET_SPECS:
            rows.append(summarize_arm(arm, target))

    # Add clinical fixed A-not-benign external reference rows at target_specificity=-1.
    clin = {
        "dataset": "external",
        "arm": "A_not_benign_clinical",
        "target_specificity": -1,
        "status": "ok",
        "reason": "fixed thresholds p_mel>=0.188 OR p_atyp>=0.412",
        "n": len(ext_df),
        "n_melanoma": int((ext_df["true"] == 1).sum()),
        "n_nevus": int((ext_df["true"] == 0).sum()),
        **ext_extra,
    }
    rows.append(clin)

    summary = pd.DataFrame(rows)
    summary["delong_p_A_vs_B_retrain_internal"] = np.nan
    mask_b_int = (summary["dataset"] == "internal") & (summary["arm"] == "B_retrain")
    summary.loc[mask_b_int, "delong_p_A_vs_B_retrain_internal"] = int_extra[
        "internal_delong_p_A_vs_B_retrain"
    ]
    summary.to_csv(OUT_SUMMARY, index=False)

    cases = build_external_cases(ext_df)
    cases.to_csv(OUT_CASES, index=False)

    # The true-atypical distribution is descriptive; keep counts in the main CSV
    # and write the detailed distribution as a small companion block in the report.
    atyp_desc.to_csv(paths.ABLATION_ATYPICAL_TRUE_INTERNAL, index=False)
    write_report(summary, cases, ext_extra, int_extra)
    print(f"wrote {OUT_SUMMARY}")
    print(f"wrote {OUT_CASES}")
    print(f"wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
