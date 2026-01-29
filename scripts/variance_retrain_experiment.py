"""Variance-model retrain experiment (pilot)

- Purpose: run a fast, reproducible pilot retrain of the variance model (log1p(var) target)
  on a sampled subset to identify promising hyperparameters and feature subsets.
- Output:
  - data/variance_retrain_pilot_summary.csv
  - data/nba_variance_scoring_pilot.pkl (best pilot model)

Usage:
    python -m scripts.variance_retrain_experiment --quick

Notes:
- Quick mode uses a random sample + small grid to keep runtime short.
- Full mode (no --quick) will run a larger grid and save additional diagnostics.
"""
from __future__ import annotations
import argparse
import os
import joblib
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

BASE = os.path.dirname(__file__) or '.'
DATA_CSV = os.path.join(BASE, '..', 'data', 'ml_training_data.csv')
FEATURES_PKL = os.path.join(BASE, '..', 'data', 'nba_features_v20.pkl')
OUT_SUM = os.path.join(BASE, '..', 'data', 'variance_retrain_pilot_summary.csv')
OUT_MODEL = os.path.join(BASE, '..', 'data', 'nba_variance_scoring_pilot.pkl')

# Candidate variance features (expandable)
CANDIDATES = [
    'feat_cv', 'feat_min_volatility', 'feat_ema', 'feat_avg_minutes', 'feat_usage_rate',
    'feat_days_rest', 'feat_spread', 'feat_games_played', 'feat_hit_rate_l10', 'feat_hit_rate_l20',
    'feat_opp_rank_vs_pos'
]


def build_target(df, pred_mean_col='predicted_mean'):
    # target = log1p((y - pred_mean)^2)
    resid = df['actual_value'].to_numpy() - df[pred_mean_col].to_numpy()
    return np.log1p(resid ** 2)


def pilot_run(sample_frac=0.12, random_state=42):
    if XGBRegressor is None:
        raise SystemExit('xgboost not available in environment')

    df = pd.read_csv(DATA_CSV)

    # quick sanity: ensure actual_value present; compute predicted_mean from saved mean-model if missing
    if 'actual_value' not in df.columns:
        raise SystemExit('Training CSV must include actual_value for pilot')

    if 'predicted_mean' not in df.columns:
        # try to compute predicted_mean using the committed mean model + saved feature order
        try:
            feat_order = joblib.load(FEATURES_PKL)
            model_mean = joblib.load(os.path.join(os.path.dirname(FEATURES_PKL), 'nba_model_scoring.pkl'))
            X_all = df[[c for c in feat_order if c in df.columns]].fillna(0.0)
            df['predicted_mean'] = model_mean.predict(X_all)
            print('Computed `predicted_mean` from saved mean model')
        except Exception as e:
            raise SystemExit(f'Could not compute predicted_mean from model/artifact: {e}')

    # sample for speed
    df = df.sample(frac=sample_frac, random_state=random_state)

    # choose features (intersection)
    features = [c for c in CANDIDATES if c in df.columns]
    if not features:
        raise SystemExit('No candidate variance features found in CSV')

    X = df[features].fillna(0.0)
    y = build_target(df)

    X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=0.2, random_state=random_state)

    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 6],
        'learning_rate': [0.05, 0.1],
        'reg_lambda': [1.0, 5.0]
    }

    model = XGBRegressor(objective='reg:squarederror', verbosity=0, n_jobs=4, random_state=random_state)
    gs = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=2, verbose=1)

    t0 = time.time()
    gs.fit(X_train, y_train)
    dt = time.time() - t0

    best = gs.best_estimator_
    pred_logvar = best.predict(X_hold)
    pred_sigma = np.sqrt(np.expm1(pred_logvar))

    # compute metrics
    # MAE on log-var
    mae_logvar = mean_absolute_error(y_hold, pred_logvar)
    # recovery metrics: compare actual residuals to predicted sigma
    actual_resid = np.abs(np.expm1(y_hold) ** 0.5)  # sqrt(expm1(log1p(resid^2))) == abs(resid)
    mae_sigma = mean_absolute_error(actual_resid, pred_sigma)
    # mean MAE/σ by bin (robust)
    dfm = pd.DataFrame({'sigma': pred_sigma, 'resid': actual_resid})
    dfm['sigma'] = np.where(dfm['sigma'] <= 0, np.nanmedian(dfm['sigma']), dfm['sigma'])
    try:
        bins = pd.qcut(dfm['sigma'], 8, duplicates='drop')
        gb = dfm.groupby(bins).agg({'sigma': 'mean', 'resid': 'mean'}).dropna()
        ratio = (gb['resid'] / gb['sigma']).mean()
    except Exception:
        ratio = float((dfm['resid'] / dfm['sigma']).mean())

    # save model + summary
    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
    joblib.dump(best, OUT_MODEL)

    row = {
        'n_rows': len(df),
        'features': ','.join(features),
        'best_params': str(gs.best_params_),
        'mae_logvar': float(mae_logvar),
        'mae_sigma': float(mae_sigma),
        'mean_mae_over_sigma': float(ratio),
        'fit_seconds': float(dt)
    }
    pd.DataFrame([row]).to_csv(OUT_SUM, index=False)
    return row


def main(quick: bool):
    if quick:
        print('Running quick pilot retrain (small sample, small grid)...')
        r = pilot_run()
        print('\nPilot results:')
        print(r)
        print('\nSaved model:', OUT_MODEL)
        print('Saved summary:', OUT_SUM)
    else:
        print('Full retrain not yet implemented in this script. Use pilot then iterate.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='run a short pilot')
    args = ap.parse_args()
    main(args.quick)
