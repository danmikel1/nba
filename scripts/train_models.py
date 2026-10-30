"""
Headless model training script — replicates the Streamlit 'Train Model' tab logic.

Usage:
    python scripts/train_models.py                   # per-market (default)
    python scripts/train_models.py --mode universal   # universal only
    python scripts/train_models.py --markets PTS REB  # specific markets
"""

import sys, os, logging, argparse, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy import stats
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.isotonic import IsotonicRegression
from model_engine import SigmoidCalibrator
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")

# Canonical feature columns — must match nba_prediction.py TRAINING_FEATURE_COLUMNS
from core_models import TRAINING_FEATURE_COLUMNS, ML_MARKET_GROUPS

from hardware_utils import get_xgboost_hardware_params

# XGBoost base params (V21.6 regularisation + CUDA GPU acceleration)
BASE_PARAMS = get_xgboost_hardware_params({
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 4,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
    'gamma': 0.1,
    'random_state': 42,
    'verbosity': 0,
})


def get_tuned_market_params(market: str = None) -> dict:
    """Load market-specific tuned hyperparameters from config/model_hyperparams.json if available."""
    hp_path = Path("config/model_hyperparams.json")
    if hp_path.exists():
        try:
            with open(hp_path, "r", encoding="utf-8") as f:
                hp = json.load(f)
            if market and market in hp and isinstance(hp[market], dict):
                logger.info(f"Loaded tuned GPU hyperparameters for {market}: {hp[market]}")
                return hp[market]
            if "default" in hp and isinstance(hp["default"], dict):
                return hp["default"]
        except Exception as e:
            logger.warning(f"Could not load tuned hyperparameters: {e}")
    return {}

QUANTILES = [0.05, 0.50, 0.95]
QUANTILE_NAMES = ['q05', 'q50', 'q95']


def _interpolate_prob(line, q_low, q50, q_high):
    """V22: Split-normal CDF."""
    z_val = 1.645
    if q_low >= q_high or q_high - q_low < 0.1:
        return 0.50
    sigma_lower = max((q50 - q_low) / z_val, 0.1)
    sigma_upper = max((q_high - q50) / z_val, 0.1)
    if line <= q50:
        z = (line - q50) / sigma_lower
    else:
        z = (line - q50) / sigma_upper
    p_over = 1.0 - stats.norm.cdf(z)
    return float(max(0.02, min(0.98, p_over)))


def train_quantile_models(X_train, y_train, X_test, y_test, model_name,
                          lines_train=None, lines_test=None):
    """Train q05/q50/q95 models + optional isotonic calibrator."""
    quantile_models = {}
    best_iters = {}
    market_params = get_tuned_market_params(model_name)
    merged_base = {**BASE_PARAMS, **market_params}

    for q, name in zip(QUANTILES, QUANTILE_NAMES):
        params = {
            **merged_base,
            'objective': 'reg:quantileerror',
            'quantile_alpha': q,
            'eval_metric': 'quantile',
            'early_stopping_rounds': 25,
        }
        use_early_stop = True

        model = XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False,
        )
        quantile_models[name] = model
        best_iter = getattr(model, 'best_iteration', merged_base['n_estimators'])
        best_iters[name] = best_iter
        logger.info(f"  {model_name}/{name}: {'early stop @' if use_early_stop else 'ran'} iter {best_iter}")

    quantile_models['_wide_quantile'] = True

    # Evaluate
    pred_05 = quantile_models['q05'].predict(X_test)
    pred_50 = quantile_models['q50'].predict(X_test)
    pred_95 = quantile_models['q95'].predict(X_test)

    mae = mean_absolute_error(y_test, pred_50)
    r2 = r2_score(y_test, pred_50)
    in_interval = ((y_test >= pred_05) & (y_test <= pred_95)).mean()
    coverage = in_interval * 100
    avg_width = (pred_95 - pred_05).mean()

    # Time-series CV (5 folds) on q50
    cv_mae_folds = []
    try:
        tscv = TimeSeriesSplit(n_splits=5)
        X_cv = pd.concat([X_train, X_test])
        y_cv = pd.concat([y_train, y_test])
        for fold_train_idx, fold_val_idx in tscv.split(X_cv):
            fold_model = XGBRegressor(
                **merged_base,
                objective='reg:quantileerror',
                quantile_alpha=0.50,
                eval_metric='mae',
                early_stopping_rounds=25,
            )
            Xf_tr, Xf_val = X_cv.iloc[fold_train_idx], X_cv.iloc[fold_val_idx]
            yf_tr, yf_val = y_cv.iloc[fold_train_idx], y_cv.iloc[fold_val_idx]
            fold_model.fit(Xf_tr, yf_tr, eval_set=[(Xf_val, yf_val)], verbose=False)
            fold_pred = fold_model.predict(Xf_val)
            cv_mae_folds.append(mean_absolute_error(yf_val, fold_pred))
    except Exception as e:
        logger.warning(f"CV failed for {model_name}: {e}")

    cv_mae_mean = float(np.mean(cv_mae_folds)) if cv_mae_folds else None
    cv_mae_std = float(np.std(cv_mae_folds)) if cv_mae_folds else None

    # Calibration
    calibrator = None
    cal_delta = 0.0

    raw_probs_test = None
    actual_over_test = None
    if lines_test is not None and len(lines_test) == len(y_test):
        lines_arr_test = np.asarray(lines_test, dtype=float)
        raw_probs_test = np.array([
            _interpolate_prob(l, q05, q50, q95)
            for l, q05, q50, q95 in zip(lines_arr_test, pred_05, pred_50, pred_95)
        ])
        actual_over_test = (np.asarray(y_test, dtype=float) > lines_arr_test).astype(float)

    if lines_train is not None and len(lines_train) == len(y_train):
        val_size = max(1, int(len(y_train) * 0.25))
        val_start = len(y_train) - val_size

        X_val = X_train.iloc[val_start:]
        y_val = y_train.iloc[val_start:]
        lines_val = np.asarray(lines_train[val_start:], dtype=float)

        pred_05_val = quantile_models['q05'].predict(X_val)
        pred_50_val = quantile_models['q50'].predict(X_val)
        pred_95_val = quantile_models['q95'].predict(X_val)

        raw_probs_val = np.array([
            _interpolate_prob(l, q05, q50, q95)
            for l, q05, q50, q95 in zip(lines_val, pred_05_val, pred_50_val, pred_95_val)
        ])
        actual_over_val = (np.asarray(y_val, dtype=float) > lines_val).astype(float)

        try:
            cal = SigmoidCalibrator(clip_min=0.01, clip_max=0.99)
            cal.fit(raw_probs_val, actual_over_val)
            calibrator = cal
            if raw_probs_test is not None and actual_over_test is not None:
                cal_probs = cal.predict(raw_probs_test)
                cal_delta = abs(cal_probs.mean() - actual_over_test.mean()) - \
                            abs(raw_probs_test.mean() - actual_over_test.mean())
        except Exception as e:
            logger.warning(f"Calibration fitting failed for {model_name}: {e}")

    metrics = {
        'name': model_name,
        'samples': len(y_train) + len(y_test),
        'mae': mae,
        'r2': r2,
        'coverage': coverage,
        'avg_width': avg_width,
        'calibrated': calibrator is not None,
        'cal_delta': round(cal_delta, 4),
        'cv_mae_mean': round(cv_mae_mean, 4) if cv_mae_mean is not None else None,
        'cv_mae_std': round(cv_mae_std, 4) if cv_mae_std is not None else None,
    }
    return quantile_models, calibrator, metrics, best_iters


def main():
    parser = argparse.ArgumentParser(description="Train NBA quantile regression models")
    parser.add_argument("--mode", choices=["per_market", "universal"], default="per_market")
    parser.add_argument("--markets", nargs="*", default=None,
                        help="Markets to train (default: all available in data)")
    parser.add_argument("--min-samples", type=int, default=100)
    args = parser.parse_args()

    logger.info("Loading dataset (preferring high-speed Parquet)...")
    from persistence_utils import load_ml_training_data
    try:
        df = load_ml_training_data(DATA_DIR)
    except FileNotFoundError:
        logger.error("Training data not found in %s", DATA_DIR)
        sys.exit(1)
    logger.info(f"  {len(df):,} rows, {df['player'].nunique()} players, {df['market'].nunique()} markets")

    # Schema validation
    V20_REQUIRED_FEATURES = TRAINING_FEATURE_COLUMNS.copy()
    missing = [f for f in V20_REQUIRED_FEATURES if f not in df.columns]
    if missing:
        logger.error("Missing features: %s", missing)
        sys.exit(1)
    if 'actual_value' not in df.columns:
        logger.error("Missing target column 'actual_value'")
        sys.exit(1)
    logger.info(f"  Schema OK: {len(V20_REQUIRED_FEATURES)} features")

    all_metrics = []
    start = time.time()

    if args.mode == "per_market":
        available_markets = sorted(df['market'].unique())
        selected = args.markets if args.markets else available_markets
        logger.info(f"Per-market training for: {selected}")

        trained = []
        skipped = []

        for market in selected:
            market_df = df[df['market'] == market]
            if len(market_df) < args.min_samples:
                logger.warning(f"  {market}: skipping ({len(market_df)} < {args.min_samples} samples)")
                skipped.append(market)
                continue

            logger.info(f"  {market}: {len(market_df):,} samples")

            X = market_df[V20_REQUIRED_FEATURES].apply(pd.to_numeric, errors='coerce')
            y = pd.to_numeric(market_df['actual_value'], errors='coerce')
            X = X.fillna(X.median())
            valid_mask = ~y.isna()
            X = X[valid_mask]
            y = y[valid_mask]

            if len(y) < args.min_samples:
                skipped.append(market)
                continue

            # Lines for calibration
            if 'feat_line' in market_df.columns:
                market_lines = pd.to_numeric(market_df['feat_line'], errors='coerce').fillna(0)
            elif 'line' in market_df.columns:
                market_lines = pd.to_numeric(market_df['line'], errors='coerce').fillna(0)
            else:
                market_lines = pd.Series(0, index=market_df.index)
            market_lines = market_lines[valid_mask]

            # Time-aware sort
            if 'date' in market_df.columns:
                ordered_idx = market_df.loc[valid_mask].sort_values('date').index
                X = X.loc[ordered_idx]
                y = y.loc[ordered_idx]
                market_lines = market_lines.loc[ordered_idx]

            X_train, X_test, y_train, y_test, lines_train, lines_test = train_test_split(
                X, y, market_lines, test_size=0.2, shuffle=False
            )

            qm, calibrator, metrics, best_iters = train_quantile_models(
                X_train, y_train, X_test, y_test, market,
                lines_train=lines_train.values, lines_test=lines_test.values
            )
            all_metrics.append(metrics)

            # Save
            model_path = DATA_DIR / f"nba_quantile_{market}.pkl"
            joblib.dump(qm, model_path)

            if calibrator is not None:
                group = ML_MARKET_GROUPS.get(market, 'universal')
                cal_path = DATA_DIR / f"nba_calibrator_{group}.pkl"
                joblib.dump(calibrator, cal_path)

            logger.info(f"    MAE={metrics['mae']:.2f}  R²={metrics['r2']:.3f}  "
                         f"Coverage={metrics['coverage']:.1f}%  Width={metrics['avg_width']:.1f}  "
                         f"CV_MAE={metrics['cv_mae_mean']}±{metrics['cv_mae_std']}  "
                         f"Cal={'yes' if metrics['calibrated'] else 'no'}")
            trained.append(market)

        # Universal fallback for skipped markets
        if skipped:
            logger.info("Training universal fallback (for skipped markets)...")
            X_all = df[V20_REQUIRED_FEATURES].apply(pd.to_numeric, errors='coerce').fillna(
                df[V20_REQUIRED_FEATURES].median())
            y_all = pd.to_numeric(df['actual_value'], errors='coerce')
            valid_mask = ~y_all.isna()
            X_all = X_all[valid_mask]
            y_all = y_all[valid_mask]

            if 'feat_line' in df.columns:
                all_lines = pd.to_numeric(df['feat_line'], errors='coerce').fillna(0)[valid_mask]
            elif 'line' in df.columns:
                all_lines = pd.to_numeric(df['line'], errors='coerce').fillna(0)[valid_mask]
            else:
                all_lines = pd.Series(0, index=X_all.index)

            if 'date' in df.columns:
                ordered_idx = df[valid_mask].sort_values('date').index
                X_all = X_all.loc[ordered_idx]
                y_all = y_all.loc[ordered_idx]
                all_lines = all_lines.loc[ordered_idx]

            X_train, X_test, y_train, y_test, lines_train, lines_test = train_test_split(
                X_all, y_all, all_lines, test_size=0.2, shuffle=False
            )
            qm, calibrator, metrics, _ = train_quantile_models(
                X_train, y_train, X_test, y_test, "universal",
                lines_train=lines_train.values, lines_test=lines_test.values
            )
            all_metrics.append(metrics)
            joblib.dump(qm, DATA_DIR / "nba_quantile_universal.pkl")
            if calibrator is not None:
                joblib.dump(calibrator, DATA_DIR / "nba_calibrator_universal.pkl")
            logger.info(f"  Universal: MAE={metrics['mae']:.2f}  R²={metrics['r2']:.3f}")

        logger.info(f"Trained: {trained}")
        if skipped:
            logger.info(f"Skipped (universal fallback): {skipped}")

    else:
        # Universal only
        logger.info("Training universal model on all markets...")
        X = df[V20_REQUIRED_FEATURES].apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(df['actual_value'], errors='coerce')
        X = X.fillna(X.median())
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]

        if 'feat_line' in df.columns:
            uni_lines = pd.to_numeric(df['feat_line'], errors='coerce').fillna(0)[valid_mask]
        elif 'line' in df.columns:
            uni_lines = pd.to_numeric(df['line'], errors='coerce').fillna(0)[valid_mask]
        else:
            uni_lines = pd.Series(0, index=X.index)

        if 'date' in df.columns:
            ordered_idx = df[valid_mask].sort_values('date').index
            X = X.loc[ordered_idx]
            y = y.loc[ordered_idx]
            uni_lines = uni_lines.loc[ordered_idx]

        X_train, X_test, y_train, y_test, lines_train, lines_test = train_test_split(
            X, y, uni_lines, test_size=0.2, shuffle=False
        )
        qm, calibrator, metrics, _ = train_quantile_models(
            X_train, y_train, X_test, y_test, "universal",
            lines_train=lines_train.values, lines_test=lines_test.values
        )
        all_metrics.append(metrics)
        joblib.dump(qm, DATA_DIR / "nba_quantile_universal.pkl")
        if calibrator is not None:
            joblib.dump(calibrator, DATA_DIR / "nba_calibrator_universal.pkl")

    # Save feature list
    joblib.dump(V20_REQUIRED_FEATURES, DATA_DIR / "nba_features_v21.pkl")

    elapsed = time.time() - start
    logger.info(f"\n{'='*60}")
    logger.info(f"Training complete in {elapsed:.0f}s")
    logger.info(f"{'='*60}")

    # Summary table
    print(f"\n{'Model':<12} {'Samples':>8} {'MAE':>6} {'R²':>6} {'Cov%':>6} {'Width':>6} {'CV MAE':>10} {'Cal':>4}")
    print("-" * 68)
    for m in all_metrics:
        cv = f"{m['cv_mae_mean']:.3f}" if m['cv_mae_mean'] is not None else "n/a"
        print(f"{m['name']:<12} {m['samples']:>8,} {m['mae']:>6.2f} {m['r2']:>6.3f} "
              f"{m['coverage']:>5.1f}% {m['avg_width']:>6.1f} {cv:>10} "
              f"{'yes' if m['calibrated'] else 'no':>4}")


if __name__ == "__main__":
    main()
