"""
core_models.py — Domain Data Models, Schemas & Feature Definitions

Defines canonical dataclasses, feature schemas, enums, and dynamic
extraction routines for the NBA prediction pipeline.
"""

from dataclasses import dataclass, field, asdict, fields
from enum import Enum
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# MARKET CONSTANTS & GROUPS
# =============================================================================

MARKET_GROUPS = {
    'scoring': ['PTS'],                  # High volume, high variance
    'counting': ['REB', 'AST'],          # Medium volume counting stats
    'combo': ['PRA', 'PR', 'PA', 'RA'],  # Combined markets
}

ML_MARKET_GROUPS = {
    'PTS': 'scoring',
    'REB': 'counting',
    'AST': 'counting',
    'PRA': 'combo',
    'PR': 'combo',
    'PA': 'combo',
    'RA': 'combo',
    '3PM': 'rare',
    'FG3M': 'rare',
    'STL': 'rare',
    'BLK': 'rare',
}

VALID_MARKETS = list(ML_MARKET_GROUPS.keys())


# =============================================================================
# RESULT & DATA QUALITY
# =============================================================================

class ResultQuality(Enum):
    """
    Result quality categories for ML training.
    Differentiates between bad beats and bad reads.
    """
    # Losses
    BAD_BEAT = "bad_beat"          # Lost by 0.5-1.5 (unlucky)
    CLOSE_LOSS = "close_loss"      # Lost by 1.5-3.5 (competitive)
    CLEAR_LOSS = "clear_loss"      # Lost by 3.5-7.5 (missed read)
    BAD_READ = "bad_read"          # Lost by 7.5+ (completely wrong)
    
    # Wins
    SWEAT_WIN = "sweat_win"        # Won by 0.5-1.5 (lucky)
    CLOSE_WIN = "close_win"        # Won by 1.5-3.5 (competitive)
    SOLID_WIN = "solid_win"        # Won by 3.5-7.5 (good read)
    BLOWOUT_WIN = "blowout_win"    # Won by 7.5+ (excellent read)
    
    # Special
    PUSH = "push"                  # Exactly on line
    PENDING = "pending"            # Not yet resolved


@dataclass
class DataQuality:
    """
    Tracks data quality and fallbacks used during analysis.
    Helps users understand prediction reliability.
    """
    # Fallback flags
    used_default_pace: bool = False
    used_default_def_rating: bool = False
    used_fallback_std: bool = False
    used_fallback_minutes: bool = False
    used_fallback_split: bool = False
    missing_team_stats: bool = False
    low_sample_size: bool = False  # < 10 games
    
    # Cache age tracking
    team_stats_age_hours: float = 0.0
    
    # Warnings list
    warnings: List[str] = field(default_factory=list)
    
    @property
    def score(self) -> float:
        """Calculate data quality score (0-100)."""
        penalties = {
            'used_default_pace': 10,
            'used_default_def_rating': 10,
            'used_fallback_std': 5,
            'used_fallback_minutes': 5,
            'used_fallback_split': 5,
            'missing_team_stats': 20,
            'low_sample_size': 15,
        }
        total_penalty = sum(
            penalty for flag, penalty in penalties.items() 
            if getattr(self, flag, False)
        )
        return max(0, 100 - total_penalty)
    
    @property
    def grade(self) -> str:
        """Letter grade for data quality."""
        s = self.score
        if s >= 90: return 'A'
        if s >= 75: return 'B'
        if s >= 60: return 'C'
        if s >= 40: return 'D'
        return 'F'
    
    @property
    def has_issues(self) -> bool:
        """Returns True if any fallback was used."""
        return self.score < 100
    
    def add_warning(self, msg: str):
        """Add a warning message."""
        if msg not in self.warnings:
            self.warnings.append(msg)


# =============================================================================
# FEATURE VECTOR
# =============================================================================

@dataclass
class FeatureVector:
    """
    V20.3 EMPIRICAL: Pure observable feature vector with absence-awareness.
    
    DESIGN PRINCIPLES:
    - Raw observables only (no computed multipliers)
    - Timestamp-valid (frozen before game tip-off)
    - Non-decisional (no belief-encoded adjustments)
    - Let ML learn all relationships from data
    """
    # === IDENTIFIERS (not used in ML, for tracking only) ===
    player_id: int
    player_name: str
    opponent_abbrev: str
    market: str  # PTS, REB, AST, PRA, 3PM, STL, BLK, etc.
    
    # === RAW OBSERVABLES (all known before tip-off) ===
    line: float              # Vegas prop line
    avg_minutes: float       # Recent average minutes
    ema: float               # Exponential moving average of stat
    std: float               # Standard deviation of stat
    opponent_drtg_season: float  # Opponent defensive rating (season-to-date)
    spread: float            # Vegas spread
    game_total: float        # Vegas O/U
    days_rest: int           # Days since last game (raw integer)
    is_home: bool            # Home/away
    is_b2b: bool             # Back-to-back flag
    games_played: int        # Sample size
    
    # === V20.2 FEATURES (raw observables) ===
    opponent_pace: float = 100.0
    team_pace: float = 100.0
    trend_5g: float = 0.0
    home_avg: float = 0.0
    away_avg: float = 0.0

    # === V20.3: TRUE-SHOOTING% FEATURES ===
    feat_ts_pct: float = -1.0
    feat_ts_pct_delta: float = -1.0
    
    # === V20.3: ABSENCE-AWARE FEATURES ===
    team_out_ppg: float = -1.0
    team_out_count: int = -1
    opp_out_ppg: float = -1.0
    opp_out_count: int = -1

    # === V20.4: ADVANCED CONTEXT ===
    feat_usage_rate: float = -1.0
    feat_h2h_avg: float = -1.0
    feat_season_avg: float = 0.0
    feat_l10_avg: float = 0.0
    feat_eff_per_min: float = 0.0
    feat_fatigue_load: float = 0.0
    feat_form_gap: float = 0.0

    # === V21.6: TRAVEL / ALTITUDE FEATURES ===
    feat_miles_96h: float = 0.0
    feat_altitude_next: int = 0

    # === V20.5: HIT RATE FEATURES ===
    feat_hit_rate_l10: float = 0.0
    feat_hit_rate_l20: float = 0.0

    # === V20.5: DEFENSE VS POSITION ===
    feat_opp_rank_vs_pos: int = 15

    # === V21.1: DEPTH CHART RANK ===
    feat_depth_rank: int = 4

    # === MARKET IDENTITY (one-hot encoded) ===
    market_scoring: int = 0
    market_counting: int = 0
    market_combo: int = 0

    # === BEHAVIOR & RISK FEATURES ===
    feat_min_volatility: float = -1.0
    feat_foul_rate: float = -1.0
    feat_cv: float = -1.0

    # === V27: 4-GATE LSTM SEQUENTIAL FEATURE SCOUT (Tool #4) ===
    feat_lstm_momentum: float = 0.0
    feat_lstm_fatigue_gate: float = 0.5
    
    # === DATA QUALITY (audit only, not used in ML) ===
    data_quality: DataQuality = field(default_factory=DataQuality)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for ML pipeline."""
        return asdict(self)

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Return canonical training feature column names."""
        identifiers = {'player_id', 'player_name', 'opponent_abbrev', 'market', 'data_quality'}

        legacy = [
            'avg_minutes', 'ema', 'std', 'opponent_drtg_season', 'line', 'spread',
            'game_total', 'days_rest', 'is_home', 'is_b2b', 'games_played',
            'opponent_pace', 'team_pace', 'trend_5g', 'home_avg', 'away_avg'
        ]

        all_attrs = [f.name for f in fields(cls) if f.name not in identifiers]

        ordered: List[str] = []
        for name in legacy:
            if name in all_attrs and name not in ordered:
                ordered.append(name)

        for name in all_attrs:
            if name not in ordered:
                ordered.append(name)

        feat_columns: List[str] = []
        for name in ordered:
            feat_columns.append(name if name.startswith('feat_') else f'feat_{name}')

        feat_columns = [c for c in feat_columns if c not in ('feat_hit_rate_l10', 'feat_hit_rate_l20')]
        return feat_columns

    @property
    def feat_fields(self) -> List[str]:
        """Instance-level canonical ML column names."""
        return self.__class__.get_feature_names()

    def to_ml_array(self) -> np.ndarray:
        """Convert feature values to a numeric numpy array matching canonical feature order."""
        feature_names = self.__class__.get_feature_names()
        arr: List[float] = []

        for col in feature_names:
            attr = col[len('feat_'):]
            raw = getattr(self, attr, None) if hasattr(self, attr) else getattr(self, col, None)

            if col.endswith('_count'):
                sentinel = -1
            elif col in ('feat_is_home', 'feat_is_b2b'):
                sentinel = 0
            else:
                sentinel = -1.0

            if raw is None:
                arr.append(float(sentinel))
                continue

            try:
                if isinstance(raw, bool):
                    arr.append(1.0 if raw else 0.0)
                else:
                    arr.append(float(raw))
            except Exception:
                arr.append(float(sentinel))

        return np.array(arr)


# Canonical training & export schemas
TRAINING_FEATURE_COLUMNS = [c for c in FeatureVector.get_feature_names() if c not in ('feat_hit_rate_l10', 'feat_hit_rate_l20')]

ML_EXPORT_METADATA = [
    'date', 'player', 'opponent', 'market', 'position',
    'line', 'predicted_side', 'predicted_prob', 'predicted_ev', 'projected_value',
    'result', 'hit', 'actual_value', 'margin', 'margin_pct', 'result_quality',
    'tag', 'grade',
]

ML_EXPORT_SCHEMA = ML_EXPORT_METADATA + [c for c in TRAINING_FEATURE_COLUMNS.copy() if c not in ('feat_hit_rate_l10', 'feat_hit_rate_l20')]


def get_market_group_features(market: str) -> Dict[str, int]:
    """Convert market to one-hot encoded group features."""
    features = {
        'feat_market_scoring': 0,
        'feat_market_counting': 0,
        'feat_market_combo': 0
    }
    for group_name, markets in MARKET_GROUPS.items():
        if market in markets:
            features[f'feat_market_{group_name}'] = 1
            break
    return features


def extract_features_dynamically(features_obj: Any, market: Optional[str] = None) -> Dict[str, Any]:
    """Dynamically extract ML feature keys based on canonical ML_EXPORT_SCHEMA."""
    result: Dict[str, Any] = {}
    resolved_market = market or getattr(features_obj, 'market', None) or 'PTS'

    for col in ML_EXPORT_SCHEMA:
        if not col.startswith('feat_') or col.startswith('feat_market_'):
            continue

        attr_name = col[len('feat_'):]
        candidates = [attr_name, f'feat_{attr_name}']
        found_attr = None
        raw_val = None
        for cand in candidates:
            if hasattr(features_obj, cand):
                found_attr = cand
                raw_val = getattr(features_obj, cand)
                break

        if found_attr is not None:
            try:
                if col.endswith('_count'):
                    val = int(raw_val)
                elif col in ('feat_is_home', 'feat_is_b2b'):
                    val = 1 if bool(raw_val) else 0
                else:
                    val = float(raw_val) if raw_val is not None else -1.0
            except Exception:
                val = -1 if col.endswith('_count') else 0 if col in ('feat_is_home', 'feat_is_b2b') else -1.0
        else:
            val = -1 if col.endswith('_count') else 0 if col in ('feat_is_home', 'feat_is_b2b') else -1.0

        result[col] = val

    result.update(get_market_group_features(resolved_market))
    return result


# =============================================================================
# PREDICTION & DECISION MODELS
# =============================================================================

@dataclass
class Projection:
    """Model projection output."""
    base_projection: float
    final_projection: float
    confidence_interval: Tuple[float, float]
    adjustments: Dict[str, float]
    ml_prob: Optional[float] = None


@dataclass
class ProjectionResult:
    """Holds the output of the projection model engine."""
    base_projection: float
    final_projection: float
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    adjustments: Dict[str, Any] = field(default_factory=dict)
    ml_prob: Optional[float] = None



@dataclass
class SimulationResult:
    """Monte Carlo simulation output."""
    over_prob: float
    under_prob: float
    median: float
    ci_10: float
    ci_90: float
    simulations: np.ndarray


@dataclass
class BetDecision:
    """Pure statistical output and Kelly stake recommendation."""
    recommended_side: str
    probability: float
    expected_value: float
    predicted_mean: float
    predicted_std: float
    kelly_stake: float
    kelly_fraction: float
    quality_score: float = 0.0
    is_bet: bool = True
    rejection_reason: Optional[str] = None


@dataclass
class AnalysisResult:
    """Complete analysis container."""
    success: bool
    error: Optional[str] = None
    player_name: Optional[str] = None
    player_id: Optional[int] = None
    opponent_name: Optional[str] = None
    opponent_id: Optional[int] = None
    market: Optional[str] = None
    line: Optional[float] = None
    odds: Optional[float] = None
    player_position: Optional[str] = None
    features: Optional[FeatureVector] = None
    projection: Optional[Projection] = None
    simulation: Optional[SimulationResult] = None
    decision: Optional[BetDecision] = None
    game_logs: Optional[pd.DataFrame] = None
    ml_details: Optional[Dict[str, Any]] = None


@dataclass
class BacktestResult:
    """Single backtest prediction result."""
    date: str
    player_name: str
    market: str
    line: float
    predicted_side: str
    predicted_prob: float
    predicted_ev: float
    actual_value: float
    hit: bool
    grade: str
    opponent: str = ''
    position: str = ''
    is_home: bool = True
    features: Dict[str, Any] = field(default_factory=dict)
    snapshot_date: str = ''
    snapshot_season: str = ''


@dataclass
class BacktestSummary:
    """Aggregated backtest metrics."""
    total_predictions: int
    wins: int
    losses: int
    win_rate: float
    roi: float
    brier_score: float
    calibration_by_bucket: Dict[str, Dict[str, float]]
    grade_performance: Dict[str, Dict[str, float]]
    results_df: pd.DataFrame
