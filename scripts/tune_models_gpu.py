"""
tune_models_gpu.py — Optuna + CUDA Bayesian Hyperparameter Optimization for NBA Quantile Models
Optimized for NVIDIA GeForce RTX 5060 Ti 16GB (CUDA, ~2 GB VRAM).

Uses Optuna TPESampler (Bayesian Tree-structured Parzen Estimator) + MedianPruner
with pre-built XGBoost QuantileDMatrix folds on GPU to optimize Multi-Quantile
Pinball Loss (q05, q50, q95) and q50 MAE via Time-Series Cross-Validation.

Saves the best parameters per market into config/model_hyperparams.json.

Usage:
    python scripts/tune_models_gpu.py                         # 500 total trials across all markets (~2.5 min)
    python scripts/tune_models_gpu.py --total-trials 500      # 500 total trials across selected markets
    python scripts/tune_models_gpu.py --markets PTS REB       # Specific markets
    python scripts/tune_models_gpu.py --trials 60             # Explicit trials per market
    python scripts/tune_models_gpu.py --quick                 # Fast 10-trial search per market
"""

import sys
import os
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from xgboost import XGBRegressor

from hardware_utils import check_cuda_availability, get_xgboost_hardware_params
from persistence_utils import load_ml_training_data
from core_models import TRAINING_FEATURE_COLUMNS, ML_MARKET_GROUPS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = Path("data")
CONFIG_DIR = Path("config")
OUTPUT_PATH = CONFIG_DIR / "model_hyperparams.json"

CORE_PROP_MARKETS = ["PTS", "REB", "AST", "PRA", "PR", "PA", "RA", "3PM"]
QUANTILE_ALPHAS = [0.05, 0.50, 0.95]

BASELINE_PARAMS: Dict[str, Any] = {
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "gamma": 0.1,
}


def compute_pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Exact quantile pinball (check) loss for quantile alpha in (0, 1)."""
    err = y_true - y_pred
    return float(np.maximum(alpha * err, (alpha - 1.0) * err).mean())


def build_cv_dmatrices(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 4,
) -> List[Tuple[xgb.QuantileDMatrix, xgb.QuantileDMatrix, np.ndarray]]:
    """
    Pre-build XGBoost QuantileDMatrix folds once per market so GPU histogram bins
    are reused across all Optuna trials without redundant CPU->GPU PCIe copies.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        dtrain = xgb.QuantileDMatrix(X_tr, y_tr)
        dval = xgb.QuantileDMatrix(X_val, y_val, ref=dtrain)
        folds.append((dtrain, dval, y_val))
    return folds


def evaluate_candidate_dmatrix(
    folds: List[Tuple[xgb.QuantileDMatrix, xgb.QuantileDMatrix, np.ndarray]],
    params: Dict[str, Any],
    trial: Optional[optuna.Trial] = None,
    num_boost_round: int = 200,
    early_stopping_rounds: int = 15,
) -> Tuple[float, float]:
    """
    Evaluate candidate hyperparameters on GPU across pre-built QuantileDMatrix folds.
    Uses multi-quantile vector leaf (`quantile_alpha=[0.05, 0.50, 0.95]`) to optimize
    joint pinball loss across the entire (q05, q50, q95) distribution while also tracking q50 MAE.
    Supports Optuna intermediate reporting and MedianPruner early termination.
    """
    hw_params = get_xgboost_hardware_params({
        "objective": "reg:quantileerror",
        "quantile_alpha": QUANTILE_ALPHAS,
        "eval_metric": "quantile",
        "seed": 42,
        "verbosity": 0,
        **params,
    })
    # Remove sklearn-only keys if present
    hw_params.pop("n_estimators", None)
    hw_params.pop("random_state", None)

    fold_pinballs = []
    fold_maes = []

    for step_idx, (dtrain, dval, y_val) in enumerate(folds):
        bst = xgb.train(
            hw_params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dval, "val")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        preds = bst.predict(dval)
        # Multi-quantile output has shape (n_samples, 3) corresponding to [q05, q50, q95]
        if preds.ndim == 2 and preds.shape[1] == 3:
            pb_05 = compute_pinball_loss(y_val, preds[:, 0], 0.05)
            pb_50 = compute_pinball_loss(y_val, preds[:, 1], 0.50)
            pb_95 = compute_pinball_loss(y_val, preds[:, 2], 0.95)
            joint_pinball = (pb_05 + pb_50 + pb_95) / 3.0
            q50_mae = float(np.abs(y_val - preds[:, 1]).mean())
        else:
            joint_pinball = compute_pinball_loss(y_val, preds, 0.50)
            q50_mae = float(np.abs(y_val - preds).mean())

        fold_pinballs.append(joint_pinball)
        fold_maes.append(q50_mae)

        if trial is not None:
            running_loss = float(np.mean(fold_pinballs))
            trial.report(running_loss, step=step_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return float(np.mean(fold_pinballs)), float(np.mean(fold_maes))


def evaluate_candidate(
    X: pd.DataFrame,
    y: pd.Series,
    params: Dict[str, Any],
    n_splits: int = 4,
) -> float:
    """
    Backward-compatible helper: evaluate candidate parameters on (X, y) DataFrame/Series
    and return mean cross-validation q50 MAE.
    """
    X_arr = X.to_numpy(dtype=np.float32, copy=False)
    y_arr = y.to_numpy(dtype=np.float32, copy=False)
    folds = build_cv_dmatrices(X_arr, y_arr, n_splits=n_splits)
    _, q50_mae = evaluate_candidate_dmatrix(folds, params, trial=None)
    return q50_mae


def suggest_optuna_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample hyperparameters using Optuna Bayesian TPE search space."""
    return {
        "learning_rate": round(trial.suggest_float("learning_rate", 0.015, 0.12, log=True), 4),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 20),
        "subsample": round(trial.suggest_float("subsample", 0.65, 1.0, step=0.05), 2),
        "colsample_bytree": round(trial.suggest_float("colsample_bytree", 0.60, 1.0, step=0.05), 2),
        "reg_alpha": round(trial.suggest_float("reg_alpha", 0.05, 10.0, log=True), 4),
        "reg_lambda": round(trial.suggest_float("reg_lambda", 0.5, 15.0, log=True), 4),
        "gamma": round(trial.suggest_float("gamma", 0.0, 0.5, step=0.05), 2),
    }


def tune_market(
    market: str,
    market_df: pd.DataFrame,
    feature_cols: List[str],
    n_trials: int = 60,
    n_splits: int = 4,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Runs Optuna Bayesian TPE + MedianPruner GPU hyperparameter tuning for a single market.
    """
    logger.info(f"--- Optuna + CUDA Tuning: {market} ({len(market_df):,} samples, {n_trials} trials) ---")

    if "date" in market_df.columns:
        market_df = market_df.sort_values("date")

    X_df = market_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_df = X_df.fillna(X_df.median()).fillna(0.0)
    y_ser = pd.to_numeric(market_df["actual_value"], errors="coerce")
    valid = ~y_ser.isna()
    X_arr = X_df.loc[valid].to_numpy(dtype=np.float32)
    y_arr = y_ser.loc[valid].to_numpy(dtype=np.float32)

    t_build = time.perf_counter()
    folds = build_cv_dmatrices(X_arr, y_arr, n_splits=n_splits)
    logger.info(f"  Built {n_splits} GPU QuantileDMatrix folds in {(time.perf_counter() - t_build)*1000:.0f} ms")

    # 1. Evaluate Baseline
    t0 = time.perf_counter()
    baseline_pinball, baseline_mae = evaluate_candidate_dmatrix(folds, BASELINE_PARAMS, trial=None)
    logger.info(
        f"  Baseline: Pinball={baseline_pinball:.4f} | q50 MAE={baseline_mae:.4f} "
        f"({time.perf_counter() - t0:.2f}s)"
    )

    sampler = TPESampler(seed=seed, multivariate=True)
    pruner = MedianPruner(n_startup_trials=max(5, n_trials // 8), n_warmup_steps=1)
    study = optuna.create_study(
        study_name=f"nba_quantile_{market}",
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )
    # Always enqueue baseline as Trial 0 so Optuna never regresses below baseline
    study.enqueue_trial(BASELINE_PARAMS)

    best_mae_holder = {"mae": baseline_mae, "pinball": baseline_pinball}

    def objective(trial: optuna.Trial) -> float:
        candidate = suggest_optuna_params(trial)
        pinball, q50_mae = evaluate_candidate_dmatrix(folds, candidate, trial=trial)
        trial.set_user_attr("q50_mae", round(q50_mae, 4))
        if pinball < best_mae_holder["pinball"]:
            best_mae_holder["pinball"] = pinball
            best_mae_holder["mae"] = q50_mae
            logger.info(
                f"  [Trial {trial.number:03d}/{n_trials}] 🔥 NEW BEST Pinball={pinball:.4f} "
                f"(q50 MAE={q50_mae:.4f}) | depth={candidate['max_depth']} lr={candidate['learning_rate']:.4f}"
            )
        return pinball

    t_study = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    study_elapsed = time.perf_counter() - t_study

    best_trial = study.best_trial
    best_pinball = float(best_trial.value)
    best_mae = float(best_trial.user_attrs.get("q50_mae", best_mae_holder["mae"]))

    # Format best params cleanly
    best_params = {
        "learning_rate": round(float(best_trial.params["learning_rate"]), 4),
        "max_depth": int(best_trial.params["max_depth"]),
        "min_child_weight": int(best_trial.params["min_child_weight"]),
        "subsample": round(float(best_trial.params["subsample"]), 2),
        "colsample_bytree": round(float(best_trial.params["colsample_bytree"]), 2),
        "reg_alpha": round(float(best_trial.params["reg_alpha"]), 4),
        "reg_lambda": round(float(best_trial.params["reg_lambda"]), 4),
        "gamma": round(float(best_trial.params["gamma"]), 2),
    }

    pruned_count = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    completed_count = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)

    pinball_imp_pct = ((baseline_pinball - best_pinball) / max(baseline_pinball, 1e-6)) * 100.0
    mae_imp_pct = ((baseline_mae - best_mae) / max(baseline_mae, 1e-6)) * 100.0

    # Compute hyperparameter importance if enough completed trials
    importances = {}
    if completed_count >= 6:
        try:
            imp = optuna.importance.get_param_importances(study)
            importances = {k: round(float(v), 4) for k, v in list(imp.items())[:5]}
        except Exception:
            importances = {}

    logger.info(
        f"  ✅ {market} Complete in {study_elapsed:.1f}s ({completed_count} complete, {pruned_count} pruned): "
        f"Pinball {baseline_pinball:.4f} -> {best_pinball:.4f} ({pinball_imp_pct:+.2f}%) | "
        f"MAE {baseline_mae:.4f} -> {best_mae:.4f} ({mae_imp_pct:+.2f}%)"
    )

    return {
        "best_params": best_params,
        "best_pinball": round(best_pinball, 4),
        "baseline_pinball": round(baseline_pinball, 4),
        "pinball_improvement_pct": round(pinball_imp_pct, 2),
        "best_mae": round(best_mae, 4),
        "baseline_mae": round(baseline_mae, 4),
        "improvement_pct": round(mae_imp_pct, 2),
        "sample_count": int(len(y_arr)),
        "trials_run": int(len(study.trials)),
        "trials_completed": int(completed_count),
        "trials_pruned": int(pruned_count),
        "elapsed_sec": round(study_elapsed, 2),
        "param_importances": importances,
    }


def main():
    parser = argparse.ArgumentParser(description="Optuna + CUDA GPU Hyperparameter Tuner for NBA Quantile Models")
    parser.add_argument("--markets", nargs="*", default=None, help="Markets to tune (default: 8 core prop markets)")
    parser.add_argument("--total-trials", type=int, default=500, help="Total Optuna trials across all markets (default: 500)")
    parser.add_argument("--trials", type=int, default=None, help="Explicit number of trials per market (overrides --total-trials)")
    parser.add_argument("--quick", action="store_true", help="Quick mode (10 trials per market)")
    parser.add_argument("--splits", type=int, default=4, help="Number of TimeSeriesSplit folds")
    parser.add_argument("--min-samples", type=int, default=200, help="Minimum sample count to tune a market")
    args = parser.parse_args()

    has_cuda, gpu_name = check_cuda_availability()
    logger.info("=" * 70)
    logger.info("⚡ OPTUNA + CUDA BAYESIAN HYPERPARAMETER OPTIMIZER (NBA QUANTILE ML)")
    if has_cuda:
        logger.info(f"🚀 GPU Acceleration Active: {gpu_name} (device='cuda', tree_method='hist')")
    else:
        logger.warning("⚠️ No CUDA GPU detected, falling back to CPU")
    logger.info("=" * 70)

    try:
        df = load_ml_training_data(DATA_DIR)
    except FileNotFoundError:
        logger.error("Dataset not found in %s", DATA_DIR)
        sys.exit(1)

    available = set(df["market"].unique())
    if args.markets:
        target_markets = [m for m in args.markets if m in available]
    else:
        target_markets = [m for m in CORE_PROP_MARKETS if m in available]

    if not target_markets:
        logger.error("No valid markets found to tune.")
        sys.exit(1)

    if args.quick:
        trials_per_market = 10
    elif args.trials is not None:
        trials_per_market = args.trials
    else:
        trials_per_market = max(10, int(round(args.total_trials / len(target_markets))))

    total_planned = trials_per_market * len(target_markets)
    logger.info(
        f"Markets ({len(target_markets)}): {target_markets} | "
        f"{trials_per_market} trials/market ({total_planned} total trials)"
    )

    tuned_results: Dict[str, Any] = {}
    if OUTPUT_PATH.exists():
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                tuned_results = json.load(f)
        except Exception:
            tuned_results = {}

    start_time = time.perf_counter()
    summary_rows = []

    for market in target_markets:
        market_df = df[df["market"] == market]
        if len(market_df) < args.min_samples:
            logger.warning(f"Skipping {market}: {len(market_df)} < {args.min_samples} samples")
            continue

        res = tune_market(
            market=market,
            market_df=market_df,
            feature_cols=TRAINING_FEATURE_COLUMNS,
            n_trials=trials_per_market,
            n_splits=args.splits,
        )

        tuned_results[market] = res["best_params"]
        tuned_results[f"_{market}_meta"] = {
            "best_pinball": res["best_pinball"],
            "baseline_pinball": res["baseline_pinball"],
            "pinball_improvement_pct": res["pinball_improvement_pct"],
            "best_mae": res["best_mae"],
            "baseline_mae": res["baseline_mae"],
            "improvement_pct": res["improvement_pct"],
            "samples": res["sample_count"],
            "trials_run": res["trials_run"],
            "trials_completed": res["trials_completed"],
            "trials_pruned": res["trials_pruned"],
            "elapsed_sec": res["elapsed_sec"],
            "param_importances": res["param_importances"],
            "optimizer": "Optuna TPESampler + MedianPruner (CUDA)",
            "tuned_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        summary_rows.append((market, res))

        # Incremental save after each market completes
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(tuned_results, f, indent=2)

    total_time = time.perf_counter() - start_time
    logger.info("=" * 70)
    logger.info(f"✅ Optuna + CUDA Tuning Complete in {total_time:.1f}s! Saved to: {OUTPUT_PATH}")
    logger.info("=" * 70)

    print(f"\n{'Market':<8} {'Samples':>8} {'Trials':>7} {'Pruned':>7} {'Base PB':>9} {'Best PB':>9} {'PB Imp%':>8} {'Base MAE':>9} {'Best MAE':>9}")
    print("-" * 82)
    for mkt, r in summary_rows:
        print(
            f"{mkt:<8} {r['sample_count']:>8,} {r['trials_run']:>7} {r['trials_pruned']:>7} "
            f"{r['baseline_pinball']:>9.4f} {r['best_pinball']:>9.4f} {r['pinball_improvement_pct']:>+7.2f}% "
            f"{r['baseline_mae']:>9.4f} {r['best_mae']:>9.4f}"
        )


if __name__ == "__main__":
    main()
