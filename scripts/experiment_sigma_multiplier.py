"""Quick experiment: compute MAE/σ after applying a sigma multiplier

Usage:
    python -m scripts.experiment_sigma_multiplier

Prints Mean MAE/σ for a set of multipliers and writes results to data/sigma_multiplier_experiment.csv
"""
from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic

BASE = os.path.dirname(__file__) or '.'
DATA_PATH = os.path.join(BASE, '..', 'data', 'ml_training_data.csv')
MODEL_DIR = os.path.join(BASE, '..', 'data')
FEATURES_PATH = os.path.join(BASE, '..', 'data', 'nba_features_v20.pkl')
OUT_CSV = os.path.join(BASE, '..', 'data', 'sigma_multiplier_experiment.csv')

if __name__ == '__main__':
    multipliers = [1.0, 1.25, 1.5, 1.6, 1.8, 2.0]

    df = pd.read_csv(DATA_PATH)
    features = joblib.load(FEATURES_PATH)
    X = df[features].fillna(0.0)

    model_mean = joblib.load(os.path.join(MODEL_DIR, 'nba_model_scoring.pkl'))
    model_var = joblib.load(os.path.join(MODEL_DIR, 'nba_variance_scoring.pkl'))

    pred_mean = model_mean.predict(X)

    # select variance features defensively (same logic as validate script)
    var_feature_names = None
    if hasattr(model_var, 'feature_names_in_'):
        var_feature_names = list(model_var.feature_names_in_)
    elif hasattr(model_var, 'get_booster'):
        try:
            var_feature_names = list(model_var.get_booster().feature_names)
        except Exception:
            var_feature_names = None

    if var_feature_names:
        X_var = df[[c for c in var_feature_names if c in df.columns]].fillna(0.0)
    else:
        candidate = ['feat_cv', 'feat_min_volatility', 'feat_ema', 'feat_avg_minutes', 'feat_usage_rate', 'feat_days_rest', 'feat_spread', 'feat_games_played']
        X_var = df[[c for c in candidate if c in df.columns]].fillna(0.0)

    pred_log_var = model_var.predict(X_var)
    base_sigma = np.sqrt(np.expm1(pred_log_var))
    residuals = np.abs(df['actual_value'].to_numpy() - pred_mean)

    rows = []
    for m in multipliers:
        sigma = base_sigma * float(m)
        # bin by predicted sigma quantiles
        valid_sigma = np.where(sigma == 0, np.nan, sigma)
        valid_sigma = np.where(np.isnan(valid_sigma), np.nanmedian(valid_sigma), valid_sigma)
        try:
            bins = pd.qcut(valid_sigma, 10, duplicates='drop')
            gb = pd.DataFrame({'sigma': sigma, 'residual': residuals, 'bin': bins}).dropna()
            bin_stats = gb.groupby('bin').agg({'sigma': 'mean', 'residual': 'mean'})
            ratio = (bin_stats['residual'] / bin_stats['sigma']).replace([np.inf, -np.inf], np.nan).dropna()
            mean_ratio = float(ratio.mean())
        except Exception:
            mean_ratio = float((residuals / sigma).mean())

        rows.append({'multiplier': m, 'mean_mae_over_sigma': mean_ratio, 'n_rows': int(len(residuals))})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(out.to_string(index=False))
    print(f"Wrote: {OUT_CSV}")
