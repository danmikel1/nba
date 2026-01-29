"""Calibration & variance forensic for V20 models

Produces:
 - data/calibration_variance.png
 - data/calibration_pit.png

Run locally before deploying risk-pricing defaults.
"""
from __future__ import annotations
import os
import sys
import math
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False
from scipy.stats import norm

BASE = os.path.dirname(__file__) or '.'
DATA_PATH = os.path.join(BASE, '..', 'data', 'ml_training_data.csv')
MODEL_DIR = os.path.join(BASE, '..', 'data')
FEATURES_PATH = os.path.join(BASE, '..', 'data', 'nba_features_v20.pkl')
OUT_VAR = os.path.join(BASE, '..', 'data', 'calibration_variance.png')
OUT_PIT = os.path.join(BASE, '..', 'data', 'calibration_pit.png')


def validate_calibration(markets=None):
    print("🚀 Starting Forensic Calibration Check...")

    if not os.path.exists(DATA_PATH):
        raise SystemExit(f"Training CSV not found: {DATA_PATH}")
    if not os.path.exists(MODEL_DIR):
        raise SystemExit(f"Model directory not found: {MODEL_DIR}")

    df = pd.read_csv(DATA_PATH)

    if markets is None:
        markets = ['PTS']

    # Try to load feature order (best-effort)
    features = None
    if os.path.exists(FEATURES_PATH):
        try:
            features = joblib.load(FEATURES_PATH)
            print(f"Loaded feature list ({len(features)} cols) from {FEATURES_PATH}")
        except Exception as e:
            print(f"Could not load feature list: {e}")
            features = None

    # Load models
    try:
        model_mean = joblib.load(os.path.join(MODEL_DIR, 'nba_model_scoring.pkl'))
        model_var = joblib.load(os.path.join(MODEL_DIR, 'nba_variance_scoring.pkl'))
    except Exception as e:
        raise SystemExit(f"Failed to load models: {e}")

    # Filter
    df = df[df['market'].isin(['PTS', 'PRA', 'PR', 'PA', 'PTS'])].copy()
    if df.empty:
        raise SystemExit("No rows for selected markets in training CSV")

    # Build X
    if features is None:
        # try to infer columns from dataframe (feat_ prefix)
        features = [c for c in df.columns if c.startswith('feat_')]
        if not features:
            raise SystemExit("No feature columns found to run inference")
    X = df[features].fillna(0.0)

    print(f"📊 Generating predictions for {len(X)} rows...")
    pred_mean = model_mean.predict(X)

    # Variance model may have been trained on a compact feature subset — select accordingly
    var_feature_names = None
    # sklearn API
    if hasattr(model_var, 'feature_names_in_'):
        var_feature_names = list(model_var.feature_names_in_)
    # xgboost sklearn wrapper
    elif hasattr(model_var, 'get_booster'):
        try:
            var_feature_names = list(model_var.get_booster().feature_names)
        except Exception:
            var_feature_names = None

    if var_feature_names:
        missing = [c for c in var_feature_names if c not in df.columns]
        if missing:
            print(f"warning: variance-model expects columns not present in CSV: {missing[:6]}{'...' if len(missing)>6 else ''}")
        X_var = df[[c for c in var_feature_names if c in df.columns]].fillna(0.0)
    else:
        # fallback: attempt to find a compact subset that matches common VARIANCE_FEATURES
        candidate = ['feat_cv', 'feat_min_volatility', 'feat_ema', 'feat_avg_minutes', 'feat_usage_rate', 'feat_days_rest', 'feat_spread', 'feat_games_played']
        X_var = df[[c for c in candidate if c in df.columns]].fillna(0.0)
        print(f"info: using fallback variance features: {list(X_var.columns)}")

    try:
        pred_log_var = model_var.predict(X_var)
    except Exception as e:
        raise SystemExit(f"Variance model predict failed after selecting subset ({list(X_var.columns)[:8]}...): {e}")

    pred_sigma = np.sqrt(np.expm1(pred_log_var))
    residuals = np.abs(df['actual_value'].to_numpy() - pred_mean)

    # VARIANCE CALIBRATION PLOT
    os.makedirs(os.path.dirname(OUT_VAR), exist_ok=True)
    results = pd.DataFrame({'sigma': pred_sigma, 'residual': residuals})
    # Bin by predicted sigma quantiles
    results['sigma_bin'] = pd.qcut(results['sigma'].replace(0, np.nan).fillna(results['sigma'].median()), 10, duplicates='drop')
    bin_stats = results.groupby('sigma_bin').agg({'sigma': 'mean', 'residual': 'mean'}).dropna()

    plt.figure(figsize=(9, 6))
    x = bin_stats['sigma'].values
    y = bin_stats['residual'].values
    if _HAS_SEABORN:
        sns.regplot(x=x, y=y, scatter_kws={'s': 80}, ci=None)
    else:
        plt.scatter(x, y, s=80, alpha=0.9)
        # simple linear fit for visual guidance
        if len(x) >= 2:
            coeffs = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            plt.plot(xs, np.polyval(coeffs, xs), color='C1', alpha=0.8)

    maxx = max(bin_stats['sigma'].max() * 1.05, 1.0)
    plt.plot([0, maxx], [0, maxx * 0.8], 'r--', label='Ideal (MAE ≈ 0.8*σ)')
    plt.title('Variance Calibration: Predicted σ vs Actual MAE')
    plt.xlabel('Predicted Volatility (σ)')
    plt.ylabel('Actual Error (MAE)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_VAR)
    plt.close()
    print(f"✅ Saved {OUT_VAR}")

    # PIT Histogram
    z_scores = (df['actual_value'].to_numpy() - pred_mean) / np.where(pred_sigma == 0, 1e-6, pred_sigma)
    p_values = norm.cdf(z_scores)

    plt.figure(figsize=(9, 6))
    plt.hist(p_values, bins=20, density=True, alpha=0.75, color='purple')
    plt.axhline(1.0, color='k', linestyle='--', label='Perfect Calibration')
    plt.title('Probability Integral Transform (PIT) Histogram')
    plt.xlabel('CDF Value of Actual Result')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PIT)
    plt.close()
    print(f"✅ Saved {OUT_PIT}")

    # Quick numeric sanity checks
    mae_by_bin = bin_stats['residual']
    sigma_by_bin = bin_stats['sigma']
    ratio = (mae_by_bin / sigma_by_bin).replace([np.inf, -np.inf], np.nan).dropna()
    mean_ratio = float(ratio.mean())
    print(f"Mean MAE/σ across bins: {mean_ratio:.3f} (ideal ≈ 0.8)")
    return mean_ratio


if __name__ == '__main__':
    validate_calibration()
