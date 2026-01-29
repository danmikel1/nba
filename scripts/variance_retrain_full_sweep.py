"""Extended variance retrain sweep

- Runs a broader hyperparameter + feature-subset sweep for the variance model.
- Saves top-K candidate models and a CSV summary to `data/`.

Usage (long-running):
    python -m scripts.variance_retrain_full_sweep --jobs 4 --topk 5
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
OUT_CSV = os.path.join(BASE, '..', 'data', 'variance_retrain_sweep_results.csv')
OUT_DIR = os.path.join(BASE, '..', 'data', 'variance_sweep_models')

VAR_FEATURE_CANDIDATES = [
    ['feat_std','feat_min_volatility','feat_cv','feat_usage_rate','feat_days_rest','feat_spread','feat_game_total'],
    ['feat_std','feat_cv','feat_min_volatility','feat_hit_rate_l10','feat_hit_rate_l20','feat_opp_rank_vs_pos'],
    ['feat_cv','feat_min_volatility','feat_ema','feat_avg_minutes','feat_usage_rate']
]

GRID = {
    'n_estimators': [200, 600, 1000],
    'max_depth': [3, 6],
    'learning_rate': [0.02, 0.05, 0.1],
    'reg_lambda': [1.0, 5.0]
}


def run_sweep(jobs: int = 4, topk: int = 3):
    if XGBRegressor is None:
        raise SystemExit('xgboost not available')

    df = pd.read_csv(DATA_CSV)
    features = joblib.load(FEATURES_PKL)
    X = df[features].fillna(0.0)

    # compute mean predictions if not present
    if 'predicted_mean' not in df.columns:
        mean_model = joblib.load(os.path.join(os.path.dirname(FEATURES_PKL), 'nba_model_scoring.pkl'))
        df['predicted_mean'] = mean_model.predict(X)

    y_logvar = np.log1p((df['actual_value'].to_numpy() - df['predicted_mean'].to_numpy()) ** 2)

    rows = []
    os.makedirs(OUT_DIR, exist_ok=True)

    for feat_set in VAR_FEATURE_CANDIDATES:
        available = [c for c in feat_set if c in df.columns]
        if not available:
            continue
        Xv = df[available].fillna(0.0)
        X_tr, X_te, y_tr, y_te = train_test_split(Xv, y_logvar, test_size=0.2, random_state=42)

        model = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=jobs)
        gs = GridSearchCV(model, GRID, cv=3, scoring='neg_mean_absolute_error', n_jobs=min(8, jobs), verbose=1)
        t0 = time.time()
        gs.fit(X_tr, y_tr)
        dt = time.time() - t0

        best = gs.best_estimator_
        pred_log = best.predict(X_te)
        pred_sigma = np.sqrt(np.expm1(pred_log))
        actual_resid = np.abs(np.expm1(y_te) ** 0.5)

        mae_log = mean_absolute_error(y_te, pred_log)
        mae_sigma = mean_absolute_error(actual_resid, pred_sigma)

        rows.append({
            'features': ','.join(available),
            'best_params': str(gs.best_params_),
            'mae_logvar': float(mae_log),
            'mae_sigma': float(mae_sigma),
            'mean_mae_over_sigma': float((actual_resid / pred_sigma).mean()),
            'fit_seconds': float(dt)
        })

        # save top model
        model_path = os.path.join(OUT_DIR, f'variance_model_{"_".join(available)}.pkl')
        joblib.dump(best, model_path)

    out = pd.DataFrame(rows).sort_values('mean_mae_over_sigma')
    out.to_csv(OUT_CSV, index=False)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--topk', type=int, default=3)
    args = ap.parse_args()
    print('Starting extended variance retrain sweep (this can take hours)')
    res = run_sweep(jobs=args.jobs, topk=args.topk)
    print(res.to_string(index=False))
