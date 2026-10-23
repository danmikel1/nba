"""
tune_models_gpu.py — Fast GPU Hyperparameter Optimization for NBA Quantile Models
Optimized for NVIDIA GeForce RTX 5060 Ti 16GB (CUDA).

Explores tree depth, learning rate, regularisation (alpha, lambda, gamma),
subsample, and colsample parameters using Time-Series Cross-Validation on GPU.
Saves the best parameters per market into config/model_hyperparams.json.

Usage:
    python scripts/tune_models_gpu.py                      # Tune all standard markets (15 trials each)
    python scripts/tune_models_gpu.py --markets PTS REB    # Specific markets
    python scripts/tune_models_gpu.py --quick              # Fast 5-trial search
    python scripts/tune_models_gpu.py --trials 30          # Deep 30-trial search
"""

import sys
import os
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from hardware_utils import check_cuda_availability, get_xgboost_hardware_params
from persistence_utils import load_ml_training_data
from nba_prediction import TRAINING_FEATURE_COLUMNS, ML_MARKET_GROUPS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
CONFIG_DIR = Path("config")
OUTPUT_PATH = CONFIG_DIR / "model_hyperparams.json"

# Search parameter distribution space
PARAM_CANDIDATES = {
    'learning_rate': [0.02, 0.03, 0.05, 0.08, 0.12],
    'max_depth': [3, 4, 5, 6],
    'min_child_weight': [2, 4, 6, 8],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'reg_alpha': [0.5, 1.0, 2.0, 5.0],
    'reg_lambda': [1.0, 2.0, 5.0, 10.0],
    'gamma': [0.0, 0.1, 0.2],
}

BASELINE_PARAMS = {
    'learning_rate': 0.05,
    'max_depth': 4,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
    'gamma': 0.1,
}


def sample_hyperparameters(rng: np.random.Generator) -> Dict[str, Any]:
    """Sample one candidate parameter set from the search space."""
    return {
        param: float(rng.choice(values)) if isinstance(values[0], float) else int(rng.choice(values))
        for param, values in PARAM_CANDIDATES.items()
    }


def evaluate_candidate(
    X: pd.DataFrame,
    y: pd.Series,
    params: Dict[str, Any],
    n_splits: int = 5,
) -> float:
    """
    Evaluate candidate hyperparameters on GPU via TimeSeriesSplit CV.
    Returns the mean cross-validation MAE on q50.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_maes = []

    full_params = get_xgboost_hardware_params({
        'n_estimators': 400,
        'objective': 'reg:quantileerror',
        'quantile_alpha': 0.50,
        'eval_metric': 'mae',
        'early_stopping_rounds': 20,
        'random_state': 42,
        'verbosity': 0,
        **params,
    })

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBRegressor(**full_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_tr, y_tr), (X_val, y_val)],
            verbose=False,
        )

        preds = model.predict(X_val)
        fold_mae = mean_absolute_error(y_val, preds)
        fold_maes.append(fold_mae)

    return float(np.mean(fold_maes))


def tune_market(
    market: str,
    market_df: pd.DataFrame,
    feature_cols: List[str],
    n_trials: int = 15,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Runs GPU-accelerated hyperparameter tuning for a single market.
    """
    logger.info(f"--- Tuning {market} ({len(market_df):,} samples, {n_trials} trials) ---")
    X = market_df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = market_df['actual_value'].astype(float)

    rng = np.random.default_rng(seed)

    # 1. Evaluate baseline
    t0 = time.time()
    baseline_mae = evaluate_candidate(X, y, BASELINE_PARAMS, n_splits=n_splits)
    logger.info(f"  Baseline MAE: {baseline_mae:.4f} ({time.time() - t0:.2f}s)")

    best_params = BASELINE_PARAMS.copy()
    best_mae = baseline_mae
    trial_history = [{"trial": 0, "params": BASELINE_PARAMS, "mae": baseline_mae}]

    for trial in range(1, n_trials + 1):
        candidate = sample_hyperparameters(rng)
        t_start = time.time()
        mae = evaluate_candidate(X, y, candidate, n_splits=n_splits)
        elapsed = time.time() - t_start

        trial_history.append({"trial": trial, "params": candidate, "mae": round(mae, 4)})

        is_new_best = mae < best_mae
        marker = "🔥 NEW BEST" if is_new_best else ""
        logger.info(f"  Trial {trial:02d}/{n_trials}: MAE = {mae:.4f} (took {elapsed:.2f}s) {marker}")

        if is_new_best:
            best_mae = mae
            best_params = candidate

    improvement_pct = ((baseline_mae - best_mae) / baseline_mae) * 100.0
    logger.info(
        f"  Result for {market}: Baseline {baseline_mae:.4f} -> Best {best_mae:.4f} "
        f"({improvement_pct:+.2f}% improvement)"
    )

    return {
        "best_params": best_params,
        "best_mae": round(best_mae, 4),
        "baseline_mae": round(baseline_mae, 4),
        "improvement_pct": round(improvement_pct, 2),
        "sample_count": len(market_df),
        "trials_run": n_trials,
    }


def main():
    parser = argparse.ArgumentParser(description="GPU Hyperparameter Tuner for NBA Quantile Models")
    parser.add_argument("--markets", nargs="*", default=None, help="Markets to tune (default: standard ML groups)")
    parser.add_argument("--trials", type=int, default=15, help="Number of random parameter trials per market")
    parser.add_argument("--quick", action="store_true", help="Quick mode (5 trials per market)")
    parser.add_argument("--splits", type=int, default=5, help="Number of TimeSeriesSplit folds")
    parser.add_argument("--min-samples", type=int, default=200, help="Minimum sample count to tune a market")
    args = parser.parse_args()

    n_trials = 5 if args.quick else args.trials

    # Verify Hardware
    has_cuda, gpu_name = check_cuda_availability()
    logger.info("=" * 60)
    logger.info("NBA QUANTILE REGRESSION GPU HYPERPARAMETER TUNER")
    if has_cuda:
        logger.info(f"⚡ GPU Acceleration Active: {gpu_name} (device='cuda')")
    else:
        logger.warning("⚠️ No CUDA GPU detected, falling back to CPU")
    logger.info("=" * 60)

    # Load Training Data
    try:
        df = load_ml_training_data(DATA_DIR)
    except FileNotFoundError:
        logger.error("Dataset not found in %s", DATA_DIR)
        sys.exit(1)

    logger.info(f"Loaded {len(df):,} training rows across {df['market'].nunique()} markets")

    target_markets = args.markets if args.markets else [m for m in ML_MARKET_GROUPS if m in df['market'].unique()]
    logger.info(f"Selected markets for tuning: {target_markets}")

    # Load existing hyperparameter file if available to preserve untuned markets
    tuned_results = {}
    if OUTPUT_PATH.exists():
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                tuned_results = json.load(f)
        except Exception:
            tuned_results = {}

    start_time = time.time()

    for market in target_markets:
        market_df = df[df['market'] == market]
        if len(market_df) < args.min_samples:
            logger.warning(f"Skipping {market}: {len(market_df)} < {args.min_samples} samples")
            continue

        res = tune_market(
            market=market,
            market_df=market_df,
            feature_cols=TRAINING_FEATURE_COLUMNS,
            n_trials=n_trials,
            n_splits=args.splits,
        )

        tuned_results[market] = res["best_params"]
        tuned_results[f"_{market}_meta"] = {
            "best_mae": res["best_mae"],
            "baseline_mae": res["baseline_mae"],
            "improvement_pct": res["improvement_pct"],
            "samples": res["sample_count"],
            "tuned_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    # Save to JSON
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(tuned_results, f, indent=2)

    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"✅ Tuning complete in {total_time:.1f}s! Saved to: {OUTPUT_PATH}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
