"""
model_engine.py — Quantile Regression Inference & Calibration Engine

Loads per-market and group XGBoost quantile models (q05, q50, q95),
runs split-normal CDF probability interpolation, applies isotonic calibration,
and produces ProjectionResult outputs.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from core_models import (
    FeatureVector,
    ProjectionResult,
    ML_MARKET_GROUPS,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"


class MLModelRequiredError(Exception):
    """Raised when an ML model is required for a market but unavailable."""
    pass


class SigmoidCalibrator:
    """
    Platt Scaling (Logistic Sigmoid) probability calibrator.
    Maps uncalibrated probabilities to well-calibrated probabilities
    using a smooth, strictly monotonic sigmoid curve, eliminating the
    step-function plateaus of IsotonicRegression.
    """
    def __init__(self, clip_min: float = 0.01, clip_max: float = 0.99):
        self.clip_min = clip_min
        self.clip_max = clip_max
        from sklearn.linear_model import LogisticRegression
        self.lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
        self._fitted = False

    def fit(self, raw_probs, y_true):
        from scipy.special import logit
        p = np.clip(np.asarray(raw_probs, dtype=float), 1e-4, 1.0 - 1e-4)
        z = logit(p).reshape(-1, 1)
        y = np.asarray(y_true, dtype=float).astype(int)
        self.lr.fit(z, y)
        self._fitted = True
        return self

    def predict(self, raw_probs) -> np.ndarray:
        if not self._fitted:
            return np.asarray(raw_probs, dtype=float)
        from scipy.special import logit
        p = np.clip(np.asarray(raw_probs, dtype=float), 1e-4, 1.0 - 1e-4)
        z = logit(p).reshape(-1, 1)
        calibrated = self.lr.predict_proba(z)[:, 1]
        return np.clip(calibrated, self.clip_min, self.clip_max)


class ModelEngine:
    """
    ML Model Engine (Quantile Regression & Calibration).
    
    Architecture:
    - XGBRegressor predicts expected stat value E[stat] (median / q50)
    - Low (q05) and High (q95) quantiles measure asymmetric distribution spread
    - Split-normal CDF computes exact P(Over) without Gaussian assumptions
    - Isotonic calibration curves correct sportsbook probability biases
    """
    def __init__(self, config: Any = None, tracker: Any = None, data_dir: Optional[Path] = None):
        self.config = config
        self.tracker = tracker
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        
        # Per-market quantile models: {market: ({'q05': model, 'q50': model, 'q95': model}, feature_names)}
        self.ml_models: Dict[str, Tuple[Dict[str, Any], List[str]]] = {}
        # Calibration curves: {market_or_group: IsotonicRegression}
        self.calibrators: Dict[str, Any] = {}
        self.ml_model = None  # Legacy fallback
        self.model_features: List[str] = []
        
        self._load_ml_models()
        self._load_calibrators()

    def _load_ml_models(self):
        """
        Load quantile regression models with priority:
        1. Per-market models: nba_quantile_{MARKET}.pkl (e.g. PTS, REB, AST)
        2. Group models: nba_quantile_{GROUP}.pkl (scoring, counting, combo, rare)
        3. Universal model: nba_quantile_universal.pkl
        4. Legacy models: nba_model_{group}.pkl (wrapped as pseudo-quantile)
        """
        individual_markets = ['PTS', 'REB', 'AST', 'PRA', 'PR', 'PA', 'RA', '3PM', 'FG3M', 'STL', 'BLK']
        market_groups = ['scoring', 'counting', 'combo', 'rare', 'universal']
        
        loaded_count = 0
        per_market_count = 0
        group_count = 0
        
        def _extract_features_from_model(model) -> List[str]:
            if model and hasattr(model, 'feature_names_in_'):
                return [str(f) for f in model.feature_names_in_]
            elif model and hasattr(model, 'get_booster'):
                booster_names = model.get_booster().feature_names
                return [str(f) for f in booster_names] if booster_names else []
            return []
        
        def _load_quantile_model(key: str, path: Path) -> bool:
            nonlocal loaded_count
            if not path.exists():
                return False
            try:
                quantile_dict = joblib.load(path)
                if isinstance(quantile_dict, dict) and 'q50' in quantile_dict:
                    for _qk, _qm in quantile_dict.items():
                        if hasattr(_qm, 'set_params'):
                            try:
                                _qm.set_params(device='cpu')
                                _qm.get_booster().set_param({'device': 'cpu'})
                            except Exception:
                                pass
                    is_wide_quantile = 'q05' in quantile_dict or 'q95' in quantile_dict
                    quantile_dict['_wide_quantile'] = is_wide_quantile
                    
                    feature_names = _extract_features_from_model(quantile_dict.get('q50'))
                    self.ml_models[key] = (quantile_dict, feature_names)
                    loaded_count += 1
                    
                    if not self.model_features and feature_names:
                        self.model_features = feature_names
                    return True
                else:
                    logger.warning(f"Invalid quantile model format for {key}")
            except Exception as e:
                logger.warning(f"Failed to load {key} quantile model: {e}")
            return False
        
        def _load_legacy_model(key: str, path: Path) -> bool:
            nonlocal loaded_count
            if not path.exists():
                return False
            try:
                model = joblib.load(path)
                feature_names = _extract_features_from_model(model)
                legacy_quantile = {'q50': model, 'q10': None, 'q90': None, '_legacy': True}
                self.ml_models[key] = (legacy_quantile, feature_names)
                loaded_count += 1
                if not self.model_features and feature_names:
                    self.model_features = feature_names
                return True
            except Exception as e:
                logger.warning(f"Failed to load {key} legacy model: {e}")
            return False
        
        # 1. Load per-market models first
        for market in individual_markets:
            quantile_path = self.data_dir / f"nba_quantile_{market}.pkl"
            if _load_quantile_model(market, quantile_path):
                per_market_count += 1
        
        # 2. Load group models
        for group in market_groups:
            quantile_path = self.data_dir / f"nba_quantile_{group}.pkl"
            legacy_path = self.data_dir / f"nba_model_{group}.pkl"
            
            if _load_quantile_model(group, quantile_path):
                group_count += 1
            elif _load_legacy_model(group, legacy_path):
                group_count += 1
        
        if loaded_count > 0:
            logger.info(f"ML MODELS: {per_market_count} per-market, {group_count} group models loaded from {self.data_dir}")
        else:
            logger.warning(f"No ML models found in {self.data_dir}")

    def _load_calibrators(self):
        """Load isotonic regression calibrators for probability correction."""
        market_groups = ['scoring', 'counting', 'combo', 'rare', 'universal']
        loaded_count = 0
        
        for group in market_groups:
            calibrator_path = self.data_dir / f"nba_calibrator_{group}.pkl"
            if calibrator_path.exists():
                try:
                    calibrator = joblib.load(calibrator_path)
                    self.calibrators[group] = calibrator
                    loaded_count += 1
                except Exception as e:
                    logger.warning(f"Failed to load {group} calibrator: {e}")
        
        if loaded_count > 0:
            logger.info(f"Loaded {loaded_count} calibration curves")

    def _get_model_for_market(self, market: str) -> Tuple[Dict[str, Any], List[str]]:
        """Get the appropriate quantile models and feature names for a market."""
        if market in self.ml_models:
            return self.ml_models[market]
        
        group = ML_MARKET_GROUPS.get(market, 'universal')
        if group in self.ml_models:
            return self.ml_models[group]
        
        if 'universal' in self.ml_models:
            return self.ml_models['universal']
        
        if self.ml_model is not None:
            return ({'q50': self.ml_model, 'q10': None, 'q90': None, '_legacy': True}, self.model_features)
        
        return ({}, self.model_features)

    def _extract_features_for_model(self, features: FeatureVector, expected_features: List[str]) -> np.ndarray:
        """Extract feature values in the exact order the model expects."""
        feature_map = {
            'feat_avg_minutes': features.avg_minutes,
            'feat_ema': features.ema,
            'feat_std': features.std,
            'feat_opp_drtg_season': features.opponent_drtg_season,
            'feat_opponent_drtg_season': features.opponent_drtg_season,
            'feat_line': features.line,
            'feat_spread': features.spread,
            'feat_game_total': features.game_total,
            'feat_days_rest': float(features.days_rest),
            'feat_is_home': float(features.is_home),
            'feat_is_b2b': float(features.is_b2b),
            'feat_games_played': float(features.games_played),
            'feat_opponent_pace': features.opponent_pace,
            'feat_team_pace': features.team_pace,
            'feat_trend_5g': features.trend_5g,
            'feat_home_avg': features.home_avg,
            'feat_away_avg': features.away_avg,
            'feat_ts_pct': features.feat_ts_pct,
            'feat_ts_pct_delta': features.feat_ts_pct_delta,
            'feat_team_out_ppg': features.team_out_ppg,
            'feat_team_out_count': float(features.team_out_count),
            'feat_opp_out_ppg': features.opp_out_ppg,
            'feat_opp_out_count': float(features.opp_out_count),
            'feat_market_scoring': float(features.market_scoring),
            'feat_market_counting': float(features.market_counting),
            'feat_market_combo': float(features.market_combo),
            'feat_min_volatility': float(features.feat_min_volatility),
            'feat_foul_rate': float(features.feat_foul_rate),
            'feat_cv': float(features.feat_cv),
            'feat_usage_rate': float(getattr(features, 'feat_usage_rate', 0.0)),
            'feat_h2h_avg': float(getattr(features, 'feat_h2h_avg', 0.0)),
            'feat_season_avg': float(getattr(features, 'feat_season_avg', 0.0)),
            'feat_l10_avg': float(getattr(features, 'feat_l10_avg', 0.0)),
            'feat_hit_rate_l10': float(getattr(features, 'feat_hit_rate_l10', 0.0)),
            'feat_hit_rate_l20': float(getattr(features, 'feat_hit_rate_l20', 0.0)),
            'feat_opp_rank_vs_pos': float(getattr(features, 'feat_opp_rank_vs_pos', 15.0)),
            'feat_depth_rank': float(getattr(features, 'feat_depth_rank', 4.0)),
            'feat_miles_96h': float(getattr(features, 'feat_miles_96h', 0.0)),
            'feat_altitude_next': float(getattr(features, 'feat_altitude_next', 0)),
            'feat_eff_per_min': float(getattr(features, 'feat_eff_per_min', 0.0)),
            'feat_fatigue_load': float(getattr(features, 'feat_fatigue_load', 0.0)),
            'feat_form_gap': float(getattr(features, 'feat_form_gap', 0.0)),
            'feat_market_rare': float(getattr(features, 'market_rare', 0.0)),
        }

        lstm_mom = float(getattr(features, 'feat_lstm_momentum', 0.0) or 0.0)
        lstm_fg = float(getattr(features, 'feat_lstm_fatigue_gate', 0.5) if getattr(features, 'feat_lstm_fatigue_gate', None) is not None else 0.5)
        if lstm_mom == 0.0 and abs(lstm_fg - 0.5) < 1e-6 and (
            'feat_lstm_momentum' in expected_features or 'feat_lstm_fatigue_gate' in expected_features
        ):
            try:
                if not hasattr(self, '_lstm_scout') or self._lstm_scout is None:
                    from lstm_feature_scout import LSTMFeatureScoutEngine
                    self._lstm_scout = LSTMFeatureScoutEngine()
                mkt = getattr(features, 'market', None) or 'PTS'
                lstm_mom, lstm_fg = self._lstm_scout.predict_live_features(features, mkt)
            except Exception:
                pass
        feature_map['feat_lstm_momentum'] = lstm_mom
        feature_map['feat_lstm_fatigue_gate'] = lstm_fg
        
        values = []
        for feat_name in expected_features:
            values.append(feature_map.get(feat_name, 0.0))
        return np.array(values)

    def get_ml_prediction(self, features: FeatureVector, raw_only: bool = False) -> Optional[float]:
        """Get P(Over) probability via quantile interpolation."""
        regression = self.get_ml_regression_output(features)
        if not regression.get('has_model'):
            return None
        return regression.get('p_over')

    def get_ml_regression_output(self, features: FeatureVector) -> Dict[str, Any]:
        """Get predictions from q05, q50, q95 quantile models."""
        market = getattr(features, 'market', None) or 'PTS'
        quantile_dict, expected_features = self._get_model_for_market(market)
        
        result = {
            'predicted_value': None,
            'predicted_std': None,
            'p_over': None,
            'p_under': None,
            'confidence_interval': (None, None),
            'has_model': bool(quantile_dict and quantile_dict.get('q50')),
            'is_regression': True,
            'is_quantile': True,
            'quantiles': {'q_low': None, 'q50': None, 'q_high': None},
            'is_wide_quantile': bool(quantile_dict and quantile_dict.get('_wide_quantile', False)),
        }
        
        if not quantile_dict or not quantile_dict.get('q50'):
            return result
        
        try:
            if expected_features:
                X = self._extract_features_for_model(features, expected_features).reshape(1, -1)
            else:
                X = features.to_ml_array().reshape(1, -1)
            
            q50_model = quantile_dict.get('q50')
            if hasattr(q50_model, 'predict_proba'):
                raw_prob = float(q50_model.predict_proba(X)[0][1])
                result['p_over'] = max(0.01, min(0.99, raw_prob))
                result['p_under'] = 1.0 - result['p_over']
                result['is_regression'] = False
                result['is_quantile'] = False
                result['predicted_value'] = features.ema
                result['predicted_std'] = features.std
                return result
            
            pred_50 = float(q50_model.predict(X)[0])
            result['predicted_value'] = pred_50
            result['quantiles']['q50'] = pred_50
            
            is_wide = quantile_dict.get('_wide_quantile', False)
            is_legacy = quantile_dict.get('_legacy', False)
            
            if is_wide:
                q_low_model = quantile_dict.get('q05')
                q_high_model = quantile_dict.get('q95')
                z_divisor = 3.29
                fallback_z = 1.645
            else:
                q_low_model = quantile_dict.get('q10')
                q_high_model = quantile_dict.get('q90')
                z_divisor = 2.56
                fallback_z = 1.28
            
            if q_low_model is not None and q_high_model is not None and not is_legacy:
                pred_low = float(q_low_model.predict(X)[0])
                pred_high = float(q_high_model.predict(X)[0])
            else:
                emp_std = features.std if (features.std is not None and features.std > 0) else 1.0
                pred_low = pred_50 - fallback_z * emp_std
                pred_high = pred_50 + fallback_z * emp_std
            
            # Widen interval if quantile crossing occurs
            if pred_low > pred_50 or pred_high < pred_50:
                emp_std = features.std if (features.std is not None and features.std > 0) else 3.0
                fallback_z_cross = z_divisor / 2
                pred_low = pred_50 - fallback_z_cross * emp_std
                pred_high = pred_50 + fallback_z_cross * emp_std

            pred_low = max(0.0, pred_low)
            pred_high = max(pred_low + 0.1, pred_high)
            
            result['quantiles']['q_low'] = pred_low
            result['quantiles']['q_high'] = pred_high
            result['is_wide_quantile'] = is_wide
            
            spread = pred_high - pred_low
            predicted_std = spread / z_divisor if spread > 0 else 1.0
            result['predicted_std'] = predicted_std
            result['confidence_interval'] = (max(0.0, pred_low), pred_high)
            
            line = features.line
            p_over = self._interpolate_probability_from_quantiles(
                line, pred_low, pred_50, pred_high, is_wide
            )
            result['p_over'] = float(max(0.01, min(0.99, p_over)))
            result['p_under'] = 1.0 - result['p_over']
            
        except Exception as e:
            logger.warning(f"ML Quantile Regression failed: {e}")
        
        return result

    def _interpolate_probability_from_quantiles(
        self, 
        line: float, 
        q_low: float, 
        q50: float, 
        q_high: float,
        is_wide: bool = True
    ) -> float:
        """Split-normal CDF probability interpolation from 3 quantiles."""
        z_val = 1.645 if is_wide else 1.28
        
        if q_low >= q_high or q_high - q_low < 0.1:
            return 0.50
        
        sigma_lower = max((q50 - q_low) / z_val, 0.1)
        sigma_upper = max((q_high - q50) / z_val, 0.1)
        
        if line <= q50:
            z_score = (line - q50) / sigma_lower
        else:
            z_score = (line - q50) / sigma_upper
        
        p_over = 1.0 - stats.norm.cdf(z_score)
        return float(max(0.02, min(0.98, p_over)))

    def generate_projection(self, features: FeatureVector) -> ProjectionResult:
        """Generate projection using quantile ML models."""
        market = getattr(features, 'market', None)
        regression = self.get_ml_regression_output(features)
        
        if not regression['has_model']:
            raise MLModelRequiredError(
                f"No ML model available for market '{market}'."
            )
        
        ml_proj = regression['predicted_value']
        ml_std = regression['predicted_std']
        ml_prob = regression['p_over']
        
        # Apply isotonic calibration if enabled
        calib_enabled = getattr(self.config, 'ML_CALIBRATION_ENABLED', True) if self.config else True
        if calib_enabled and hasattr(self, 'calibrators'):
            calibrated_prob = self._apply_calibration(ml_prob, market)
            if calibrated_prob != ml_prob:
                ml_prob = calibrated_prob
        
        ci_low, ci_high = regression['confidence_interval']
        ml_std_safe = ml_std if (ml_std is not None and ml_std > 0) else (features.std if (features.std is not None and features.std > 0) else 1.0)
        ci = (
            ci_low if ci_low is not None else max(0.0, float(ml_proj) - ml_std_safe * 1.645),
            ci_high if ci_high is not None else float(ml_proj) + ml_std_safe * 1.645
        )
        
        return ProjectionResult(
            base_projection=float(ml_proj),
            final_projection=float(ml_proj),
            confidence_interval=ci,
            adjustments={
                'predicted_std': float(ml_std_safe),
                'model_type': 'regression',
                'calibration_applied': calib_enabled,
            },
            ml_prob=ml_prob
        )

    def _apply_calibration(self, raw_prob: float, market: str) -> float:
        """Apply isotonic regression calibration curve to raw probability."""
        group = ML_MARKET_GROUPS.get(market, 'universal')
        calibrator = self.calibrators.get(group)
        if calibrator is None:
            return raw_prob
        
        try:
            calibrated = float(calibrator.predict([raw_prob])[0])
            return max(0.01, min(0.99, calibrated))
        except Exception as e:
            logger.warning(f"Calibration failed: {e}")
            return raw_prob
