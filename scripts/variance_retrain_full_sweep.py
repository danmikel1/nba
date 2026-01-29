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
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
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
    # compact volatility drivers (baseline)
    ['feat_std','feat_min_volatility','feat_cv','feat_usage_rate','feat_days_rest','feat_spread','feat_game_total'],
    # add hit-rate and opponent-vs-pos signals (recent behavior + matchup)
    ['feat_std','feat_cv','feat_min_volatility','feat_hit_rate_l10','feat_hit_rate_l20','feat_opp_rank_vs_pos','feat_days_rest'],
    # contextual features that explain volatility (usage, opponent pace, minutes)
    ['feat_cv','feat_min_volatility','feat_ema','feat_avg_minutes','feat_usage_rate','feat_opponent_pace','feat_team_pace'],
    # richer set including absence / lineup pressure signals
    ['feat_usage_rate','feat_days_rest','feat_team_out_ppg','feat_team_out_count','feat_opp_out_ppg','feat_opp_out_count','feat_min_volatility']
]

GRID = {
    'n_estimators': [200, 600, 1000],
    'max_depth': [3, 6],
    'learning_rate': [0.02, 0.05, 0.1],
    'reg_lambda': [1.0, 5.0]
}


def run_sweep(jobs: int = 4, topk: int = 3, validation_window: int = 80, top_n_players: int = 100, recent_only: bool = False, dry_run: bool = False):
    """Run an extended sweep.

    - Training always uses the full CSV (max data).
    - Validation may be restricted to a recent temporal window and the top-N players
      to provide a fast, representative evaluation slice (stars + rotation players).
    - dry_run: print candidate plan and validation strategy, then exit (no training).
    """
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

    # Build a representative validation index (optional, recent + top-N players)
    val_idx = None
    if recent_only:
        # require a player identifier and a date-like column for time-based validation
        player_cols = [c for c in ('player_name', 'player', 'player_id', 'feat_player_id') if c in df.columns]
        date_cols = [c for c in ('game_date', 'date', 'game_dt', 'datetime') if c in df.columns]
        if not player_cols or not date_cols:
            raise SystemExit(
                "FATAL: dataset lacks required columns for time-series validation.\n"
                "Required: one of ['player_name','player','player_id'] AND one of ['game_date','date','game_dt'].\n"
                "Fix: add/rename columns in ml_training_data.csv (e.g. 'player' and 'date') and retry."
            )

        # canonical names we'll use
        player_col = player_cols[0]
        date_col = date_cols[0]

        # ensure date parsing and temporal ordering
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception as e:
            raise SystemExit(f"FATAL: could not parse {date_col} as dates: {e}")

        # select top-N players (by appearance) and take their most recent `validation_window` rows
        top_players = df[player_col].value_counts().head(top_n_players).index.tolist()
        recent_slice = df[df[player_col].isin(top_players)].sort_values(date_col).groupby(player_col).tail(validation_window)
        if recent_slice.empty:
            raise SystemExit('FATAL: recent validation slice is empty; check `validation_window` and data coverage')

        val_idx = recent_slice.index
        print(f'Validation slice: recent {validation_window} rows for top {len(top_players)} players (by {player_col}) -> {len(val_idx)} rows')

    rows = []
    os.makedirs(OUT_DIR, exist_ok=True)

    # Filter candidate feature-sets to those present in the CSV (require at least half the features)
    filtered_candidates = []
    for feat_set in VAR_FEATURE_CANDIDATES:
        present = [c for c in feat_set if c in df.columns]
        if len(present) >= max(1, int(0.5 * len(feat_set))):
            filtered_candidates.append((feat_set, present))
        else:
            print(f"Skipping candidate (missing columns): {feat_set} -> present: {present}")

    if not filtered_candidates:
        raise SystemExit('No variance feature candidates have sufficient coverage in the training CSV.\n'
                         'Ensure the CSV contains the new context features (days_rest, opponent_pace, usage_rate, etc.).')

    # Ensure at least one candidate includes the new contextual features (sanity check)
    new_feature_keys = {'feat_days_rest','feat_opponent_pace','feat_usage_rate','feat_team_out_ppg','feat_hit_rate_l10'}
    if not any(new_feature_keys.intersection(present) for _, present in filtered_candidates):
        raise SystemExit('No candidate includes the new contextual features (days_rest, opponent_pace, usage_rate, ...).\n'
                         'Retrain will be uninformative without these features present in the CSV.')

    # DRY-RUN: print plan and exit before any training
    if dry_run:
        print('\n=== VARIANCE SWEEP DRY-RUN ===')
        print(f'Number of feature-candidates: {len(filtered_candidates)}')
        for i, (orig, present) in enumerate(filtered_candidates, start=1):
            print(f'  Candidate {i}: will test {len(present)}/{len(orig)} features -> {present}')
        if val_idx is not None:
            print('  Validation strategy: recent-only (TimeSeriesSplit will be used for CV)')
            print(f'  Validation rows: {len(val_idx)}')
        else:
            print('  Validation strategy: random split (train_test_split)')
        print('  Grid search hyperparams:', GRID)
        print('=== END DRY-RUN ===\n')
        return pd.DataFrame([{'status': 'dry-run', 'n_candidates': len(filtered_candidates)}])

    for feat_set, available in filtered_candidates:
        # log exact feature-set under test
        print(f"Testing feature-set {available} (orig: {feat_set})...")
        if not available:
            continue
        Xv = df[available].fillna(0.0)

        # If we have a targeted validation slice (recent-only requested), ALWAYS use it as the test set
        # — do not fall back to random split (avoids leakage and ensures realistic validation).
        if val_idx is not None:
            X_tr = Xv.drop(index=val_idx)
            X_te = Xv.loc[val_idx]
            y_tr = y_logvar[X_tr.index]
            y_te = y_logvar[X_te.index]
            print(f'Using targeted temporal validation for feature-set {available[:3]}... train={len(X_tr)} test={len(X_te)}')
        else:
            X_tr, X_te, y_tr, y_te = train_test_split(Xv, y_logvar, test_size=0.2, random_state=42)
            print(f'Using random split for feature-set {available[:3]}... train={len(X_tr)} test={len(X_te)}')

        model = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=jobs)

        # Choose CV strategy: use TimeSeriesSplit on temporal/ recent mode to avoid leakage
        if val_idx is not None:
            print('Using TimeSeriesSplit (temporal CV) on training set to avoid leakage')
            tscv = TimeSeriesSplit(n_splits=3)
            cv_strategy = tscv
        else:
            cv_strategy = 3

        gs = GridSearchCV(model, GRID, cv=cv_strategy, scoring='neg_mean_absolute_error', n_jobs=min(8, jobs), verbose=1)
        t0 = time.time()
        gs.fit(X_tr, y_tr)
        dt = time.time() - t0

        best = gs.best_estimator_
        pred_log = best.predict(X_te)
        pred_sigma = np.sqrt(np.expm1(pred_log))
        actual_resid = np.abs(np.expm1(y_te) ** 0.5)

        mae_log = mean_absolute_error(y_te, pred_log)
        mae_sigma = mean_absolute_error(actual_resid, pred_sigma)

        # feature importances (top 5) — robust to XGBoost wrapper
        try:
            fmap = dict(zip(X_tr.columns, best.feature_importances_.tolist()))
        except Exception:
            try:
                fmap = best.get_booster().get_score(importance_type='gain')
            except Exception:
                fmap = {c: 0.0 for c in X_tr.columns}

        # sort and keep top 5
        top_feats = sorted(fmap.items(), key=lambda x: x[1], reverse=True)[:5]
        top_feats_str = ';'.join([f'{k}:{v:.4f}' for k, v in top_feats])

        rows.append({
            'features': ','.join(available),
            'best_params': str(gs.best_params_),
            'mae_logvar': float(mae_log),
            'mae_sigma': float(mae_sigma),
            'mean_mae_over_sigma': float((actual_resid / pred_sigma).mean()),
            'fit_seconds': float(dt),
            'n_train': int(len(X_tr)),
            'n_test': int(len(X_te)),
            'top_features': top_feats_str
        })

        # save top model and its feature importances
        model_path = os.path.join(OUT_DIR, f'variance_model_{"_".join(available)}.pkl')
        joblib.dump(best, model_path)
        # write a small JSON/CSV with importances for auditing
        fi_path = os.path.join(OUT_DIR, f'variance_model_{"_".join(available)}.feats.csv')
        pd.DataFrame(list(fmap.items()), columns=['feature', 'importance']).sort_values('importance', ascending=False).to_csv(fi_path, index=False)

    out = pd.DataFrame(rows).sort_values('mean_mae_over_sigma')
    out.to_csv(OUT_CSV, index=False)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--topk', type=int, default=3)
    ap.add_argument('--validation-window', type=int, default=80, help='Number of recent rows per player to include in validation')
    ap.add_argument('--top-n-players', type=int, default=100, help='Top N players (by appearances) to include in recent validation slice')
    ap.add_argument('--recent-only', action='store_true', help='Use recent+top-N players slice for validation instead of random split')
    ap.add_argument('--dry-run', action='store_true', help='Print candidate feature-sets and validation strategy, then exit')
    args = ap.parse_args()
    print('Starting extended variance retrain sweep (this can take minutes->hours depending on grid)')
    res = run_sweep(jobs=args.jobs, topk=args.topk, validation_window=args.validation_window, top_n_players=args.top_n_players, recent_only=args.recent_only, dry_run=args.dry_run)
    print(res.to_string(index=False))
