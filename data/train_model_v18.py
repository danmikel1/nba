"""
NBA Prediction Model Training Script v18
=========================================
Implements three major improvements:

1. PROBABILITY CALIBRATION
   - Uses Isotonic Regression to transform raw XGBoost outputs into calibrated probabilities
   - Raw: 55% ML → Actual 59% wins becomes Calibrated: 59% displayed
   - Saves calibrator alongside model for inference

2. FEATURE IMPORTANCE ANALYSIS  
   - Analyzes XGBoost feature importances
   - Identifies top predictive features
   - Prunes low-importance features (< 1% contribution)
   - Generates feature importance report

3. MARKET-SPECIFIC MODELS
   - Trains separate models for each market group:
     * scoring: PTS (high volume, high variance)
     * counting: REB, AST (medium volume counting stats)
     * combo: PRA, PR, PA, RA (combined markets)
     * rare: 3PM, STL, BLK (low count, Poisson-like)
   - Each model optimized for its market characteristics

Usage:
    python train_model_v18.py

Outputs:
    - nba_model_scoring.pkl + nba_calibrator_scoring.pkl
    - nba_model_counting.pkl + nba_calibrator_counting.pkl
    - nba_model_combo.pkl + nba_calibrator_combo.pkl
    - nba_model_rare.pkl + nba_calibrator_rare.pkl
    - nba_model_universal.pkl + nba_calibrator_universal.pkl (fallback)
    - feature_importance_report.json
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    brier_score_loss, log_loss, roc_auc_score
)
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

CSV_FILE = 'ml_training_data.csv'
MODEL_DIR = '.'

# Market groups matching MARKET_GROUPS in nba_prediction.py
MARKET_GROUPS = {
    'scoring': ['PTS'],
    'counting': ['REB', 'AST'],
    'combo': ['PRA', 'PR', 'PA', 'RA'],
    'rare': ['3PM', 'STL', 'BLK', 'FG3M']  # FG3M is alias for 3PM
}

# Feature columns (must match TRAINING_FEATURE_COLUMNS in nba_prediction.py)
FEATURE_COLUMNS = [
    'feat_ema', 'feat_std', 'feat_sma_5', 'feat_sma_10', 'feat_trend',
    'feat_avg_minutes', 'feat_mins_trend',
    'feat_hit_l5', 'feat_hit_l10', 'feat_hit_l15', 'feat_hit_season',
    'feat_pace_mult', 'feat_def_mult', 'feat_position_mult',
    'feat_base_matchup_mult', 'feat_combined_matchup_mult',
    'feat_split_factor', 'feat_rest_factor', 'feat_blowout_factor', 'feat_usage_mult',
    'feat_is_home', 'feat_is_b2b', 'feat_spread', 'feat_games_played', 'feat_days_rest',
    'feat_game_total', 'feat_opp_drtg_season', 'feat_opp_drtg_l5',
    'feat_blowout_prob', 'feat_personal_fatigue_factor', 'feat_b2b_games_in_sample',
    'feat_dynamic_std_mult', 'feat_coef_variation',
    'feat_odds_decimal', 'feat_usg_season', 'feat_clv',
    'feat_market_scoring', 'feat_market_counting', 'feat_market_combo', 'feat_market_rare'
]

# Minimum feature importance threshold (features below this are candidates for pruning)
MIN_FEATURE_IMPORTANCE = 0.01  # 1%

# XGBoost hyperparameters by market type
XGBOOST_PARAMS = {
    'scoring': {
        'n_estimators': 150,
        'learning_rate': 0.05,
        'max_depth': 4,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
    },
    'counting': {
        'n_estimators': 120,
        'learning_rate': 0.05,
        'max_depth': 3,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
    },
    'combo': {
        'n_estimators': 100,
        'learning_rate': 0.05,
        'max_depth': 3,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.2,
        'reg_lambda': 1.0,
    },
    'rare': {
        # More conservative for low-count stats
        'n_estimators': 80,
        'learning_rate': 0.03,
        'max_depth': 2,
        'min_child_weight': 10,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
        'reg_alpha': 0.3,
        'reg_lambda': 2.0,
    },
    'universal': {
        'n_estimators': 100,
        'learning_rate': 0.05,
        'max_depth': 3,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
    }
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_market_group(market: str) -> str:
    """Map a market to its group."""
    for group_name, markets in MARKET_GROUPS.items():
        if market in markets:
            return group_name
    return 'universal'


def analyze_feature_importance(
    model: xgb.XGBClassifier,
    feature_names: List[str],
    market_group: str
) -> Dict:
    """
    Analyze feature importances from XGBoost model.
    Returns dict with rankings and pruning recommendations.
    """
    importances = model.feature_importances_
    
    # Create sorted list of (feature, importance)
    feature_importance = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Identify features to prune (below threshold)
    total_importance = sum(importances)
    prunable = []
    important = []
    
    for feat, imp in feature_importance:
        normalized = float(imp) / total_importance if total_importance > 0 else 0
        if normalized < MIN_FEATURE_IMPORTANCE:
            prunable.append({'feature': feat, 'importance': round(float(normalized), 4)})
        else:
            important.append({'feature': feat, 'importance': round(float(normalized), 4)})
    
    return {
        'market_group': market_group,
        'total_features': len(feature_names),
        'important_features': important,
        'prunable_features': prunable,
        'top_10': [f['feature'] for f in important[:10]]
    }


def train_calibrator(
    model: xgb.XGBClassifier,
    X_calib: pd.DataFrame,
    y_calib: pd.Series
) -> IsotonicRegression:
    """
    Train an Isotonic Regression calibrator on held-out data.
    
    Isotonic Regression is non-parametric and works well for:
    - XGBoost's tendency to output compressed probabilities
    - Non-linear probability mappings
    - Preserving probability ordering
    """
    # Get raw probabilities from model
    raw_probs = model.predict_proba(X_calib)[:, 1]
    
    # Fit isotonic regression: raw_prob -> actual_outcome
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(raw_probs, y_calib)
    
    return calibrator


def evaluate_calibration(
    y_true: np.ndarray,
    y_prob_raw: np.ndarray,
    y_prob_calibrated: np.ndarray
) -> Dict:
    """
    Compare calibration metrics before and after calibration.
    """
    # Brier score (lower is better)
    brier_raw = brier_score_loss(y_true, y_prob_raw)
    brier_cal = brier_score_loss(y_true, y_prob_calibrated)
    
    # Log loss (lower is better)
    log_raw = log_loss(y_true, y_prob_raw)
    log_cal = log_loss(y_true, y_prob_calibrated)
    
    # Calibration curve (reliability diagram data)
    prob_true_raw, prob_pred_raw = calibration_curve(y_true, y_prob_raw, n_bins=10)
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, y_prob_calibrated, n_bins=10)
    
    # Calculate calibration error (mean absolute difference from diagonal)
    calib_error_raw = np.mean(np.abs(prob_true_raw - prob_pred_raw))
    calib_error_cal = np.mean(np.abs(prob_true_cal - prob_pred_cal))
    
    return {
        'brier_raw': round(brier_raw, 4),
        'brier_calibrated': round(brier_cal, 4),
        'brier_improvement': round((brier_raw - brier_cal) / brier_raw * 100, 1),
        'log_loss_raw': round(log_raw, 4),
        'log_loss_calibrated': round(log_cal, 4),
        'calibration_error_raw': round(calib_error_raw, 4),
        'calibration_error_calibrated': round(calib_error_cal, 4),
    }


def train_market_model(
    df: pd.DataFrame,
    market_group: str,
    feature_cols: List[str]
) -> Tuple[xgb.XGBClassifier, IsotonicRegression, Dict, Dict]:
    """
    Train a model for a specific market group with calibration.
    
    Returns:
        - Trained XGBoost model
        - Calibrator (IsotonicRegression)
        - Feature importance analysis
        - Performance metrics
    """
    print(f"\n{'='*60}")
    print(f"  Training: {market_group.upper()} Model")
    print(f"{'='*60}")
    
    # Filter data for this market group
    if market_group == 'universal':
        df_market = df.copy()
    else:
        market_col = f'feat_market_{market_group}'
        if market_col in df.columns:
            df_market = df[df[market_col] == 1].copy()
        else:
            # Fallback: use market column directly
            markets = MARKET_GROUPS.get(market_group, [])
            df_market = df[df['market'].isin(markets)].copy()
    
    print(f"  📊 Dataset size: {len(df_market):,} rows")
    
    if len(df_market) < 1000:
        print(f"  ⚠️  Insufficient data for {market_group} (need 1000+), skipping...")
        return None, None, None, None
    
    # Prepare features
    available_features = [c for c in feature_cols if c in df_market.columns]
    X = df_market[available_features].copy()
    y = df_market['hit'].astype(int)
    
    # Handle missing values
    X = X.fillna(0)
    
    print(f"  📋 Features: {len(available_features)}")
    
    # Split: 70% train, 15% calibration, 15% test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train_full, y_train_full, test_size=0.176, random_state=42, stratify=y_train_full
    )
    # 0.176 of 85% ≈ 15% of total
    
    print(f"  📦 Train: {len(X_train):,} | Calibration: {len(X_calib):,} | Test: {len(X_test):,}")
    
    # Get hyperparameters for this market
    params = XGBOOST_PARAMS.get(market_group, XGBOOST_PARAMS['universal'])
    
    # Train XGBoost
    print(f"  🧠 Training XGBoost (depth={params['max_depth']}, trees={params['n_estimators']})...")
    
    model = xgb.XGBClassifier(
        **params,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Train calibrator on held-out calibration set
    print(f"  🎯 Training Isotonic Regression Calibrator...")
    calibrator = train_calibrator(model, X_calib, y_calib)
    
    # Evaluate on test set
    raw_probs = model.predict_proba(X_test)[:, 1]
    calibrated_probs = calibrator.predict(raw_probs)
    predictions = (calibrated_probs >= 0.5).astype(int)
    
    # Metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    auc = roc_auc_score(y_test, calibrated_probs)
    
    print(f"\n  📈 TEST SET RESULTS ({market_group}):")
    print(f"     Accuracy:  {accuracy:.1%}")
    print(f"     Precision: {precision:.1%} (Win rate on predictions)")
    print(f"     Recall:    {recall:.1%}")
    print(f"     F1 Score:  {f1:.3f}")
    print(f"     AUC-ROC:   {auc:.3f}")
    
    # Calibration comparison
    calib_metrics = evaluate_calibration(y_test.values, raw_probs, calibrated_probs)
    print(f"\n  🎯 CALIBRATION IMPROVEMENT:")
    print(f"     Brier Score: {calib_metrics['brier_raw']:.4f} → {calib_metrics['brier_calibrated']:.4f} ({calib_metrics['brier_improvement']:+.1f}%)")
    print(f"     Calib Error: {calib_metrics['calibration_error_raw']:.4f} → {calib_metrics['calibration_error_calibrated']:.4f}")
    
    # Feature importance
    importance_analysis = analyze_feature_importance(model, available_features, market_group)
    print(f"\n  🔍 TOP 10 FEATURES:")
    for i, feat in enumerate(importance_analysis['top_10'], 1):
        imp = next(f['importance'] for f in importance_analysis['important_features'] if f['feature'] == feat)
        print(f"     {i:2}. {feat.replace('feat_', '')}: {imp:.1%}")
    
    if importance_analysis['prunable_features']:
        print(f"\n  ⚠️  {len(importance_analysis['prunable_features'])} low-importance features (< 1%):")
        for feat in importance_analysis['prunable_features'][:5]:
            print(f"     - {feat['feature'].replace('feat_', '')}: {feat['importance']:.2%}")
    
    # Performance dict
    performance = {
        'market_group': market_group,
        'dataset_size': len(df_market),
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'auc_roc': round(auc, 4),
        'calibration': calib_metrics
    }
    
    return model, calibrator, importance_analysis, performance


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def run_temporal_validation(df: pd.DataFrame, feature_cols: List[str]) -> Dict:
    """
    Run temporal holdout validation to detect overfitting.
    
    Splits data by date:
    - Train on older data (first 80%)
    - Test on recent data (last 20%)
    
    This simulates real-world performance where we train on historical data
    and bet on future games.
    """
    print("\n" + "=" * 60)
    print("  TEMPORAL HOLDOUT VALIDATION")
    print("  Training on older data, testing on recent data")
    print("=" * 60)
    
    # Sort by date
    if 'date' not in df.columns:
        print("  ⚠️  No 'date' column found, skipping temporal validation")
        return {}
    
    df_sorted = df.sort_values('date').reset_index(drop=True)
    
    # 80/20 temporal split
    split_idx = int(len(df_sorted) * 0.80)
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    
    train_dates = f"{train_df['date'].min()} to {train_df['date'].max()}"
    test_dates = f"{test_df['date'].min()} to {test_df['date'].max()}"
    
    print(f"  📅 Train period: {train_dates} ({len(train_df):,} samples)")
    print(f"  📅 Test period:  {test_dates} ({len(test_df):,} samples)")
    
    # Prepare features
    available_features = [c for c in feature_cols if c in df_sorted.columns]
    X_train = train_df[available_features].fillna(0)
    y_train = train_df['hit'].astype(int)
    X_test = test_df[available_features].fillna(0)
    y_test = test_df['hit'].astype(int)
    
    # Train model
    print(f"  🧠 Training on historical data...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate on future data
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    
    # Train set performance (for comparison)
    train_probs = model.predict_proba(X_train)[:, 1]
    train_preds = (train_probs >= 0.5).astype(int)
    train_acc = accuracy_score(y_train, train_preds)
    
    # Test set performance
    test_acc = accuracy_score(y_test, preds)
    test_precision = precision_score(y_test, preds, zero_division=0)
    test_auc = roc_auc_score(y_test, probs)
    test_brier = brier_score_loss(y_test, probs)
    
    # Calculate overfit ratio
    overfit_ratio = train_acc / test_acc if test_acc > 0 else 999
    
    print(f"\n  📈 TEMPORAL VALIDATION RESULTS:")
    print(f"     Train Accuracy:  {train_acc:.1%}")
    print(f"     Test Accuracy:   {test_acc:.1%}")
    print(f"     Test Precision:  {test_precision:.1%}")
    print(f"     Test AUC-ROC:    {test_auc:.3f}")
    print(f"     Test Brier:      {test_brier:.4f}")
    print(f"     Overfit Ratio:   {overfit_ratio:.2f}x")
    
    if overfit_ratio > 1.10:
        print(f"\n  ⚠️  WARNING: Model may be overfitting (train {train_acc:.1%} >> test {test_acc:.1%})")
        print(f"     Consider: reducing tree depth, increasing regularization, or adding more data")
    elif overfit_ratio < 1.03:
        print(f"\n  ✅ Model generalizes well to future data!")
    else:
        print(f"\n  ➖ Mild overfit detected. Monitor on live bets.")
    
    return {
        'train_accuracy': round(train_acc, 4),
        'test_accuracy': round(test_acc, 4),
        'test_precision': round(test_precision, 4),
        'test_auc': round(test_auc, 4),
        'test_brier': round(test_brier, 4),
        'overfit_ratio': round(overfit_ratio, 3),
        'train_period': train_dates,
        'test_period': test_dates,
        'train_samples': len(train_df),
        'test_samples': len(test_df)
    }


def main():
    print("=" * 70)
    print("  NBA PREDICTION MODEL TRAINING v18")
    print("  • Probability Calibration (Isotonic Regression)")
    print("  • Feature Importance Analysis")
    print("  • Market-Specific Models")
    print("  • Temporal Holdout Validation")
    print("=" * 70)
    
    # Load data
    print(f"\n📂 Loading data from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    
    # Clean data
    df = df.dropna(subset=['hit'])
    df['hit'] = df['hit'].astype(int)
    
    print(f"✅ Loaded {len(df):,} training samples")
    
    # Check market distribution
    if 'market' in df.columns:
        print(f"\n📊 Market Distribution:")
        market_counts = df['market'].value_counts()
        for market, count in market_counts.items():
            group = get_market_group(market)
            print(f"   {market}: {count:,} ({group})")
    
    # Run temporal validation first
    temporal_results = run_temporal_validation(df, FEATURE_COLUMNS)
    
    # Train models for each market group + universal
    all_results = {}
    feature_importance_report = {}
    
    market_groups = ['scoring', 'counting', 'combo', 'rare', 'universal']
    
    for group in market_groups:
        model, calibrator, importance, performance = train_market_model(
            df, group, FEATURE_COLUMNS
        )
        
        if model is not None:
            # Save model
            model_path = os.path.join(MODEL_DIR, f'nba_model_{group}.pkl')
            joblib.dump(model, model_path)
            print(f"\n  💾 Model saved: {model_path}")
            
            # Save calibrator
            calib_path = os.path.join(MODEL_DIR, f'nba_calibrator_{group}.pkl')
            joblib.dump(calibrator, calib_path)
            print(f"  💾 Calibrator saved: {calib_path}")
            
            all_results[group] = performance
            feature_importance_report[group] = importance
    
    # Also save universal as the default fallback (nba_model.pkl)
    if 'universal' in all_results:
        universal_model_path = os.path.join(MODEL_DIR, 'nba_model_universal.pkl')
        if os.path.exists(universal_model_path):
            # Copy to default name for backwards compatibility
            import shutil
            shutil.copy(universal_model_path, os.path.join(MODEL_DIR, 'nba_model.pkl'))
            shutil.copy(
                os.path.join(MODEL_DIR, 'nba_calibrator_universal.pkl'),
                os.path.join(MODEL_DIR, 'nba_calibrator.pkl')
            )
            print(f"\n  📋 Universal model copied to nba_model.pkl (backwards compatible)")
    
    # Save feature importance report
    report_path = os.path.join(MODEL_DIR, 'feature_importance_report.json')
    report = {
        'generated_at': datetime.now().isoformat(),
        'version': 'v18',
        'total_samples': len(df),
        'temporal_validation': temporal_results,
        'market_groups': feature_importance_report,
        'model_performance': all_results
    }
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n📊 Feature importance report saved: {report_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE - SUMMARY")
    print("=" * 70)
    
    for group, perf in all_results.items():
        print(f"\n  {group.upper()}:")
        print(f"    Samples: {perf['dataset_size']:,}")
        print(f"    Precision: {perf['precision']:.1%} | AUC: {perf['auc_roc']:.3f}")
        print(f"    Brier Improvement: {perf['calibration']['brier_improvement']:+.1f}%")
    
    print("\n" + "=" * 70)
    print("  ✅ All models trained and saved!")
    print("  📁 Models: nba_model_[scoring|counting|combo|rare|universal].pkl")
    print("  📁 Calibrators: nba_calibrator_[...].pkl")
    print("  📁 Report: feature_importance_report.json")
    print("=" * 70)


if __name__ == '__main__':
    main()
