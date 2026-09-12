"""Estimate how well CNN embeddings predict the 18 handcrafted descriptors.

Fit Ridge and LightGBM regressors with five-fold out-of-fold predictions from
the deployed V2S lesion and dehair feature blocks. High R-squared indicates that
a descriptor is predictable from these embeddings. Low R-squared indicates
limited recoverability by the tested regressors; it does not prove absence of
the information or lack of predictive value for classification.

Read aligned local cnn_v2s_features.csv, cnn_v2s_dehair_features.csv and
features.csv artifacts. Interpret probing together with classifier ablations."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score, roc_auc_score
import lightgbm as lgb

from derm import paths

HC = ['A4', 'B', 'D',
      'Lab_a_kurt', 'Lab_b_std', 'Lab_a_std', 'Lab_b_skew',
      'internal_R1', 'internal_R2', 'internal_R4', 'internal_R8', 'internal_R16',
      'external_R1', 'external_R2', 'external_R4', 'external_R8', 'external_R16',
      'fractal_D']

def univariate_auc_mel(h, y):
    """Return univariate melanoma-versus-rest AUC, oriented to be at least 0.5."""
    yb = (y == 1).astype(int)
    m = np.isfinite(h)
    a = roc_auc_score(yb[m], h[m])
    return max(a, 1 - a)


def main():
    print("loading...")
    cnn_l = pd.read_csv(paths.CNN_V2S_FEATS_CSV)
    cnn_d = pd.read_csv(paths.CNN_V2S_DEHAIR_CSV)
    cnn_cols = [c for c in cnn_l.columns if c.startswith("cnn_")]
    if cnn_cols != [c for c in cnn_d.columns if c.startswith("cnn_")]:
        raise ValueError("CNN feature columns differ between lesion and dehair")
    feats = pd.read_csv(paths.FEATURES_CSV, usecols=HC + ['Class'])
    assert len(cnn_l) == len(cnn_d) == len(feats)
    assert (cnn_l['Class'].values == feats['Class'].values).all(), "lesion rows not aligned!"
    assert (cnn_d['Class'].values == feats['Class'].values).all(), "dehair rows not aligned!"
    y = cnn_l['Class'].values
    Z = np.c_[cnn_l[cnn_cols].values, cnn_d[cnn_cols].values].astype('float32')
    print(f"n={len(Z)}  embedding={Z.shape[1]}-d  handcraft={len(HC)}")

    # Historical 512-D model importance ranks for comparison.
    try:
        imp = pd.read_csv(paths.FEATURE_IMPORTANCE_CSV).set_index('feature')['rank'].to_dict()
    except Exception:
        imp = {}

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    rows = []
    for col in HC:
        h = feats[col].values.astype('float64')
        mask = np.isfinite(h)
        if mask.sum() < len(h):
            print(f"  {col}: {len(h)-mask.sum()} non-finite values skipped")
        Zc, hc = Z[mask], h[mask]

        lin = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        pred_lin = cross_val_predict(lin, Zc, hc, cv=kf, n_jobs=5)
        r2_lin = r2_score(hc, pred_lin)

        gbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                num_leaves=63, n_jobs=-1, verbosity=-1)
        pred_gbm = cross_val_predict(gbm, Zc, hc, cv=kf, n_jobs=1)
        r2_gbm = r2_score(hc, pred_gbm)

        auc = univariate_auc_mel(h, y)
        rows.append(dict(feature=col, rank=imp.get(col, np.nan),
                         auc_mel=auc, r2_linear=r2_lin, r2_nonlinear=r2_gbm,
                         residual=1 - max(r2_gbm, 0)))
        print(f"  {col:12s}  rank={imp.get(col,'?'):>4}  AUC_mel={auc:.3f}  "
              f"R2_lin={r2_lin:6.3f}  R2_nl={r2_gbm:6.3f}  resid={1-max(r2_gbm,0):.3f}")

    df = pd.DataFrame(rows).sort_values('r2_nonlinear', ascending=False)

    def verdict(r):
        if r.r2_nonlinear >= 0.5:
            return "Predictable from CNN features with the tested regressor"
        if r.r2_nonlinear >= 0.2:
            return "Partially predictable from CNN features"
        # Compare descriptors poorly predicted by the probing models.
        if r.auc_mel >= 0.60:
            return "Low probing R2 and univariate discrimination; test complementary value"
        return "Low probing R2 and weak univariate discrimination"
    df['verdict'] = df.apply(verdict, axis=1)

    print("\n" + "=" * 78)
    print("PROBING R2(handcrafted | V2S lesion + dehair embeddings), sorted by nonlinear R2")
    print("=" * 78)
    with pd.option_context('display.width', 200, 'display.max_colwidth', 60):
        print(df[['feature', 'rank', 'auc_mel', 'r2_linear',
                  'r2_nonlinear', 'verdict']].to_string(index=False))

    paths.PROBING_HANDCRAFT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.PROBING_HANDCRAFT_CSV, index=False)
    print(f"\nsaved -> {paths.relative(paths.PROBING_HANDCRAFT_CSV)}")

    # Descriptor-block summary.
    color = df[df.feature.str.startswith('Lab')]
    morpho = df[~df.feature.str.startswith('Lab')]
    print("\n--- moyennes par bloc ---")
    print(f"  couleur (Lab, 4)   : R2_nl moyen = {color.r2_nonlinear.mean():.3f}  "
          f"AUC_mel moyen = {color.auc_mel.mean():.3f}")
    print(f"  morpho/forme (14)  : R2_nl moyen = {morpho.r2_nonlinear.mean():.3f}  "
          f"AUC_mel moyen = {morpho.auc_mel.mean():.3f}")


if __name__ == '__main__':
    main()
