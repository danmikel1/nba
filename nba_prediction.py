import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, commonplayerinfo, leaguedashplayerstats, leaguegamelog
from nba_api.stats.static import players, teams
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Any, Protocol
from abc import ABC, abstractmethod
import time
import json
import unicodedata
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from enum import Enum
import warnings
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(page_title="NBA Stat Prediction", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
    /* Mobile-first responsive design */
    @media (max-width: 768px) {
        .stColumns > div { flex-direction: column !important; }
        .stMetric { padding: 0.3rem !important; font-size: 0.9rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 2px; }
        .stTabs [data-baseweb="tab"] { padding: 8px 12px; font-size: 12px; }
        section[data-testid="stSidebar"] { width: 280px !important; }
    }
    /* Clean minimal styling */
    .stButton > button { width: 100%; border-radius: 8px; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* Compact metrics */
    [data-testid="stMetricValue"] { font-size: 1.5rem; }
    [data-testid="stMetricDelta"] { font-size: 0.8rem; }
    /* Card styling */
    .card { background: rgba(60,60,60,0.3); padding: 1rem; border-radius: 8px; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CURRENT_VERSION = 'v15.0'


@dataclass(frozen=True)
class Config:
    """Immutable configuration constants."""
    CURRENT_SEASON: str = "2025-26"
    PREV_SEASON: str = "2024-25"
    API_DELAY: float = 1.0
    API_MAX_RETRIES:  int = 3
    DEFAULT_PACE: float = 100.0
    DEFAULT_DEF_RATING: float = 115.0
    CACHE_TTL_TEAM_STATS: int = 3600
    CACHE_TTL_PLAYER_IDS: int = 86400
    CACHE_TTL_GAME_LOGS: int = 1800
    CACHE_TTL_POSITION_DEF: int = 14400
    MONTE_CARLO_SIMS: int = 10000
    B2B_PENALTY:  float = 0.95
    MATCHUP_MULT_MIN: float = 0.85
    MATCHUP_MULT_MAX: float = 1.15
    POSITION_DEF_WEIGHT: float = 0.4
    BASE_MATCHUP_WEIGHT: float = 0.6
    BLOWOUT_SPREAD_THRESHOLD_1: float = 8.0
    BLOWOUT_SPREAD_THRESHOLD_2: float = 12.0
    BLOWOUT_SPREAD_THRESHOLD_3: float = 16.0
    BLOWOUT_PENALTY_1: float = 0.95
    BLOWOUT_PENALTY_2: float = 0.88  # Was 0.90
    BLOWOUT_PENALTY_3: float = 0.80  # Was 0.85 (20% reduction for massive spreads)
    STAR_MINUTES_THRESHOLD: float = 32.0
    STAR_BLOWOUT_EXTRA_PENALTY: float = 0.03
    ROLLOVER_MIN_PROB: float = 0.65
    ROLLOVER_MIN_HIT_RATE: float = 0.60
    ROLLOVER_MAX_STD_RATIO: float = 0.25
    ROLLOVER_MIN_GAMES: int = 10
    MAX_PARLAY_LEGS: int = 10
    PARLAY_KELLY_FRACTION: float = 0.10
    # Grade-based flat stakes (professional approach)
    STAKE_GRADE_A: float = 0.10  # 10% for Grade A bets
    STAKE_GRADE_B: float = 0.05  # 5% for Grade B bets  
    STAKE_GRADE_C: float = 0.03  # 3% for Grade C bets
    BACKTEST_DEFAULT_DAYS: int = 30
    BACKTEST_MIN_GAMES: int = 5
    # New feature configs
    REST_BONUS_2_DAYS: float = 1.02  # 2 days rest bonus
    REST_BONUS_3_PLUS: float = 1.03  # 3+ days rest bonus
    GAME_TOTAL_NEUTRAL: float = 225.0  # Neutral game total baseline
    GAME_TOTAL_WEIGHT: float = 0.15  # Weight for game total adjustment
    DRTG_L5_WEIGHT: float = 0.3  # Weight for recent DRTG vs season
    MIXTURE_BLOWUP_PROB: float = 0.12  # Probability of blow-up game
    MIXTURE_BLOWUP_MULT: float = 1.35  # Multiplier for blow-up games
    MIXTURE_DUD_PROB: float = 0.10  # Probability of dud game
    MIXTURE_DUD_MULT: float = 0.60  # Multiplier for dud games
    # ML Integration
    ML_PROJECTION_NUDGE: float = 0.35  # Strength of ML nudge on projection (0.0-1.0)
    ML_BLEND_WEIGHT: float = 0.0  # DISABLED - ML now applied at projection level via nudge
    
    # V15 Intelligent Models
    # Blowout Risk Model (Logistic Regression)
    BLOWOUT_MODEL_LOW_MINS_THRESHOLD: float = 25.0  # Minutes threshold for "low minutes" game
    BLOWOUT_PROB_SEVERE_THRESHOLD: float = 0.40  # P(low_mins) > 40% = severe penalty
    BLOWOUT_PROB_MODERATE_THRESHOLD: float = 0.20  # P(low_mins) > 20% = moderate penalty
    BLOWOUT_PROB_SLIGHT_THRESHOLD: float = 0.12  # P(low_mins) > 12% = slight penalty
    
    # Player-Specific Fatigue Profiling
    FATIGUE_MIN_B2B_GAMES: int = 3  # Minimum B2B games needed to calculate personal split
    FATIGUE_DEFAULT_PENALTY: float = 0.95  # Default if insufficient B2B data
    FATIGUE_DAMPEN_FACTOR: float = 0.7  # Dampen extreme splits (0.7 = 70% of raw delta)
    
    # Dynamic CV Simulation Width
    CV_MIN_STD_MULT: float = 0.70  # Minimum std multiplier for very consistent players
    CV_MAX_STD_MULT: float = 1.50  # Maximum std multiplier for volatile players
    CV_BASELINE: float = 0.25  # "Average" CV baseline for scaling
    
    # Low-Count Stats Configuration (3PM, STL, BLK)
    # These stats behave differently - they're discrete, low-frequency events
    LOW_COUNT_STATS: tuple = ('3PM', 'FG3M', 'STL', 'BLK', 'TOV')  # Treat these as Poisson
    LOW_COUNT_MAX_MULTIPLIER: float = 2.0  # Cap upside at 2x the mean (tighter than before)
    # When a player's average is below this threshold for ANY stat, treat as low-count
    LOW_AVERAGE_THRESHOLD: float = 3.0  # Derik Queen averaging 0.5 AST = treat as low-count
    # Maximum probability cap - no prediction should be more confident than this
    MAX_PROBABILITY_CAP: float = 0.75  # 75% max - prevents 99-100% hallucinations
    MIN_PROBABILITY_FLOOR: float = 0.15  # 15% min - even bad bets have some chance
    # Realistic MEAN caps (no player averages more than these)
    LOW_COUNT_MEAN_CAPS: dict = field(default_factory=lambda: {
        '3PM': 6.0,   # Steph Curry averages ~5-6, cap at 6
        'FG3M': 6.0,
        'STL': 3.0,   # Elite defenders average 2-2.5, cap at 3
        'BLK': 4.0,   # Elite blockers average 2-3, cap at 4
        'TOV': 5.0    # High usage players average 3-4
    })
    # Realistic OUTPUT caps (single-game maximums)
    LOW_COUNT_ABSOLUTE_CAPS: dict = field(default_factory=lambda: {
        '3PM': 10,    # 10+ threes is historic (Klay had 14 once)
        'FG3M': 10,
        'STL': 5,     # 5+ steals is extremely rare
        'BLK': 6,     # 6+ blocks is very rare
        'TOV': 8      # Cap turnovers
    })
    # Low-count margin thresholds for result quality (tighter than high-count)
    LOW_COUNT_MARGIN_SWEAT: float = 0.5   # Won/Lost by 0.5 (1 stat)
    LOW_COUNT_MARGIN_CLOSE: float = 1.5   # Won/Lost by 1-1.5 (1-2 stats)
    LOW_COUNT_MARGIN_SOLID: float = 2.5   # Won/Lost by 2-2.5 (2-3 stats)
    # High-count margin thresholds (existing behavior, now explicit)
    HIGH_COUNT_MARGIN_SWEAT: float = 1.5
    HIGH_COUNT_MARGIN_CLOSE: float = 3.5
    HIGH_COUNT_MARGIN_SOLID: float = 7.5


CONFIG = Config()

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
TRACKER_FILE = DATA_DIR / "bet_tracker.json"
PARLAY_FILE = DATA_DIR / "parlay_tracker.json"
POSITION_DEF_CACHE_FILE = DATA_DIR / "position_defense_cache.json"
WATCHLIST_FILE = DATA_DIR / "watchlist.json"
FATIGUE_PROFILE_CACHE_FILE = DATA_DIR / "fatigue_profiles.json"


# =============================================================================
# INTELLIGENT MODELS (V15 Upgrades)
# =============================================================================

class BlowoutPredictor:
    """
    Predicts blowout risk to adjust minute projections.
    """
    def __init__(self, config: Config = CONFIG):
        self.config = config
        self.intercept = -3.5  
        self.coef_abs_spread = 0.12  
        self.coef_avg_minutes = 0.04  
        self.coef_is_home_favorite = 0.25  
        self.coef_spread_x_star = 0.03  
        self.coef_opponent_weak = 0.15  
    
    def predict_low_minutes_prob(self, spread, avg_minutes, is_home, opponent_drtg=115.0, league_avg_drtg=115.0):
        abs_spread = abs(spread)
        is_favorite = spread < 0 if is_home else spread > 0
        is_home_favorite = is_home and is_favorite
        is_star = avg_minutes >= self.config.STAR_MINUTES_THRESHOLD
        opponent_weak = opponent_drtg > league_avg_drtg + 3
        
        log_odds = self.intercept
        log_odds += self.coef_abs_spread * abs_spread
        log_odds += self.coef_avg_minutes * (avg_minutes - 28)
        log_odds += self.coef_is_home_favorite * float(is_home_favorite)
        log_odds += self.coef_spread_x_star * abs_spread * float(is_star)
        log_odds += self.coef_opponent_weak * float(opponent_weak)
        
        prob = 1.0 / (1.0 + np.exp(-log_odds))
        return float(np.clip(prob, 0.01, 0.95))
    
    def get_blowout_factor(self, spread, avg_minutes, is_home, opponent_drtg=115.0, league_avg_drtg=115.0):
        prob = self.predict_low_minutes_prob(spread, avg_minutes, is_home, opponent_drtg, league_avg_drtg)
        if prob >= self.config.BLOWOUT_PROB_SEVERE_THRESHOLD:
            factor = max(0.70, 0.85 - (prob - 0.40) * 0.25)
            risk_level = BlowoutRisk.HIGH
        elif prob >= self.config.BLOWOUT_PROB_MODERATE_THRESHOLD:
            factor = 0.92 - (prob - 0.25) * 0.27
            risk_level = BlowoutRisk.MODERATE
        elif prob >= self.config.BLOWOUT_PROB_SLIGHT_THRESHOLD:
            factor = 0.97 - (prob - 0.15) * 0.20
            risk_level = BlowoutRisk.SLIGHT
        else:
            factor = 1.0
            risk_level = BlowoutRisk.NONE
        return float(factor), risk_level, prob

class PlayerFatigueProfiler:
    """
    [FIXED v15.1] Correctly caches nested dictionaries to prevent KeyError.
    """
    def __init__(self, config: Config = CONFIG):
        self.config = config
        self._profile_cache: Dict[int, Dict[str, Any]] = {}
        self._load_cache()

    def _load_cache(self):
        """Load cached fatigue profiles from disk."""
        try:
            if FATIGUE_PROFILE_CACHE_FILE.exists():
                with open(FATIGUE_PROFILE_CACHE_FILE, 'r') as f:
                    raw_cache = json.load(f)
                    # Convert string keys back to int for player IDs
                    self._profile_cache = {int(k): v for k, v in raw_cache.items()}
        except (json.JSONDecodeError, IOError):
            self._profile_cache = {}

    def _save_cache(self):
        """Save fatigue profiles to disk."""
        try:
            with open(FATIGUE_PROFILE_CACHE_FILE, 'w') as f:
                json.dump(self._profile_cache, f)
        except IOError as e:
            logger.warning(f"Failed to save fatigue cache: {e}")

    def calculate_fatigue_profile(self, df: pd.DataFrame, stat_col: str, player_id: int) -> Dict[str, float]:
        """
        Get fatigue profile, ensuring nested [player][stat] structure is preserved.
        """
        # 1. Check nested cache: [player_id][stat_col]
        if player_id in self._profile_cache:
            if stat_col in self._profile_cache[player_id]:
                return self._profile_cache[player_id][stat_col]
        
        # 2. Calculate fresh
        profile = self._compute_fatigue_splits(df, stat_col)
        
        # 3. Initialize player level if missing
        if player_id not in self._profile_cache:
            self._profile_cache[player_id] = {}
        
        # 4. Save to cache structure
        self._profile_cache[player_id][stat_col] = profile
        self._save_cache()
        
        return profile

    def _compute_fatigue_splits(self, df: pd.DataFrame, stat_col: str) -> Dict[str, float]:
        """Compute fatigue splits from game log data."""
        if len(df) < 5 or stat_col not in df.columns or 'GAME_DATE' not in df.columns:
            return self._default_profile()
        
        df = df.copy()
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df.sort_values('GAME_DATE', ascending=True)
        df['days_rest'] = df['GAME_DATE'].diff().dt.days.fillna(2)
        
        b2b_games = df[df['days_rest'] <= 1]
        non_b2b = df[df['days_rest'] > 1]
        
        baseline = df[stat_col].mean() if len(non_b2b) == 0 else non_b2b[stat_col].mean()
        if baseline == 0 or pd.isna(baseline): return self._default_profile()
        
        # B2B Factor
        b2b_count = len(b2b_games)
        if b2b_count >= self.config.FATIGUE_MIN_B2B_GAMES:
            b2b_avg = b2b_games[stat_col].mean()
            raw_delta = (b2b_avg / baseline) - 1
            dampened_delta = max(-0.20, min(0.10, raw_delta * self.config.FATIGUE_DAMPEN_FACTOR))
            b2b_factor = 1.0 + dampened_delta
        else:
            b2b_factor = self.config.FATIGUE_DEFAULT_PENALTY
            
        # Rest Factors (simplified for robustness)
        rest_3plus_games = df[df['days_rest'] >= 3]
        rest_3_factor = min(1.08, rest_3plus_games[stat_col].mean() / baseline) if len(rest_3plus_games) >= 3 else self.config.REST_BONUS_3_PLUS
        
        return {
            'b2b_factor': round(b2b_factor, 3),
            'rest_2_factor': 1.0, 
            'rest_3plus_factor': round(rest_3_factor, 3),
            'b2b_games': b2b_count
        }

    def _default_profile(self) -> Dict[str, float]:
        return {
            'b2b_factor': self.config.FATIGUE_DEFAULT_PENALTY,
            'rest_2_factor': 1.0,
            'rest_3plus_factor': self.config.REST_BONUS_3_PLUS,
            'b2b_games': 0
        }

    def get_rest_factor(self, days_rest: int, df: pd.DataFrame, stat_col: str, player_id: int) -> Tuple[float, Dict[str, float]]:
        """Public API to get the specific factor for today's rest."""
        profile = self.calculate_fatigue_profile(df, stat_col, player_id)
        
        if days_rest <= 0:
            return profile['b2b_factor'], profile
        elif days_rest == 1:
            return 1.0, profile
        elif days_rest == 2:
            return profile['rest_2_factor'], profile
        else:
            return profile['rest_3plus_factor'], profile
        
        
# =============================================================================
# DATA STRUCTURES
# =============================================================================

class BetGrade(Enum):
    """Bet quality grades."""
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


class BlowoutRisk(Enum):
    """Blowout risk levels."""
    NONE = "none"
    SLIGHT = "slight"
    MODERATE = "moderate"
    HIGH = "high"


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
    used_neutral_position_defense: bool = False
    used_fallback_std: bool = False
    used_fallback_minutes: bool = False
    used_fallback_split: bool = False
    missing_team_stats: bool = False
    low_sample_size: bool = False  # < 10 games
    
    # Warnings list
    warnings: List[str] = field(default_factory=list)
    
    @property
    def score(self) -> float:
        """
        Calculate data quality score (0-100).
        100 = perfect data, lower = more fallbacks used.
        """
        penalties = {
            'used_default_pace': 10,
            'used_default_def_rating': 10,
            'used_neutral_position_defense': 15,
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


@dataclass
class FeatureVector:
    """
    Standardized feature vector for ML compatibility.
    All features needed for prediction in one place.
    """
    # Player identifiers
    player_id: int
    player_name: str
    player_position: str
    
    # Game context
    opponent_id: int
    opponent_abbrev: str
    is_home: bool
    is_b2b: bool
    spread: float
    days_rest: int  # Days since last game (0=B2B, 1=normal, 2+=well-rested)
    game_total: float  # Vegas O/U total for the game
    market: str  # The stat market being analyzed (PTS, REB, 3PM, STL, etc.)
    line: float  # Vegas betting line for the stat market
    
    # Opponent recent form
    opponent_drtg_season: float  # Opponent defensive rating (season)
    opponent_drtg_l5: float  # Opponent defensive rating (last 5 games)
    
    # Statistical features
    ema:  float
    std:  float
    sma_5: float
    sma_10: float
    trend:  float
    avg_minutes: float
    mins_trend: float
    
    # Hit rates
    hit_rate_l5: float
    hit_rate_l10: float
    hit_rate_l15: float
    hit_rate_season: float
    hit_rate_weighted: float  # Recency-weighted hit rate
    
    # Volatility
    coef_variation: float  # std/mean - normalized volatility measure
    
    # Multipliers
    pace_mult: float
    def_mult:  float
    position_mult: float
    base_matchup_mult: float
    combined_matchup_mult: float
    split_factor: float
    rest_factor: float
    blowout_factor:  float
    usage_mult: float
    
    # Derived
    games_played: int
    
    # V15 Intelligent Model Outputs
    blowout_prob: float = 0.0  # P(minutes < 25) from BlowoutPredictor
    personal_fatigue_factor: float = 1.0  # Player-specific B2B/rest factor
    b2b_games_in_sample: int = 0  # Number of B2B games used for fatigue calc
    dynamic_std_mult: float = 1.0  # CV-based std multiplier for simulation
    
    # V16 Market & Role Context
    odds_decimal: float = 1.91  # Market efficiency signal (default -110)
    usg_season: float = 0.0     # Season usage baseline (USG%)
    clv: float = 0.0            # Closing Line Value (closing_line - line)
    
    # Data quality tracking
    data_quality: DataQuality = field(default_factory=DataQuality)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for ML pipeline."""
        return asdict(self)
    
    def to_ml_array(self) -> np.ndarray:
        """Convert numeric features to numpy array for ML models."""
        numeric_features = [
            self.ema, self.std, self.sma_5, self.sma_10, self.trend,
            self.avg_minutes, self.mins_trend,
            self.hit_rate_l5, self.hit_rate_l10, self.hit_rate_l15, self.hit_rate_season,
            self.pace_mult, self.def_mult, self.position_mult, self.base_matchup_mult,
            self.combined_matchup_mult, self.split_factor, self.rest_factor,
            self.blowout_factor, self.usage_mult,
            float(self.is_home), float(self.is_b2b), self.spread, float(self.games_played),
            float(self.days_rest), self.game_total, self.opponent_drtg_season,
            self.opponent_drtg_l5,
            # V15 features
            self.blowout_prob, self.personal_fatigue_factor, 
            float(self.b2b_games_in_sample), self.dynamic_std_mult, self.coef_variation,
            # V16 market & role context
            self.odds_decimal, self.usg_season, self.clv
        ]
        return np.array(numeric_features)


@dataclass
class Projection:
    """Model projection output."""
    base_projection: float
    final_projection: float
    confidence_interval:  Tuple[float, float]
    adjustments: Dict[str, float]
    context: str
    ml_prob: Optional[float] = None  # ML model win probability (if available)


@dataclass
class SimulationResult:
    """Monte Carlo simulation output."""
    over_prob: float
    under_prob: float
    median:  float
    ci_10: float
    ci_90: float
    simulations:  np.ndarray


@dataclass
class BetDecision:
    """Final bet decision with all metadata."""
    recommended_side: str  # "OVER" or "UNDER"
    probability: float
    expected_value: float
    grade: BetGrade
    kelly_stake: float
    kelly_fraction: float
    rollover_suitable: bool
    rollover_score: float
    reasons_good: List[str]
    reasons_bad: List[str]
    confidence_warning: Optional[str] = None  # Warning for low sample size, high variance, etc.


@dataclass
class AnalysisResult: 
    """Complete analysis result container."""
    success: bool
    error:  Optional[str] = None
    
    # Inputs
    player_name: Optional[str] = None
    player_id: Optional[int] = None
    opponent_name: Optional[str] = None
    opponent_id: Optional[int] = None
    market: Optional[str] = None
    line: Optional[float] = None
    odds: Optional[float] = None
    
    # Components
    features: Optional[FeatureVector] = None
    projection: Optional[Projection] = None
    simulation:  Optional[SimulationResult] = None
    decision: Optional[BetDecision] = None
    
    # Raw data for UI
    game_logs: Optional[pd.DataFrame] = None
    hit_rates: Optional[Dict[str, float]] = None
    
    # Injury-aware usage boost info
    injured_teammates: List[str] = field(default_factory=list)  # e.g., ["Tyrese Maxey (28.1%)"]
    injury_usage_boost: float = 1.0  # e.g., 1.10 = +10% boost


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
    hit:  bool
    grade: str
    # Additional context for ML training
    opponent: str = ''  # Opponent team abbreviation
    position: str = ''  # Player position
    is_home: bool = True  # Was player at home
    # ML training features
    features: Dict[str, Any] = field(default_factory=dict)


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


# =============================================================================
# UTILITIES
# =============================================================================

def normalize_name(name: str) -> str:
    """Normalize player/team names for matching."""
    if not name:
        return ""
    nfkd_form = unicodedata.normalize('NFKD', name)
    return "".join([c for c in nfkd_form if not unicodedata.category(c) == 'Mn']).lower().strip()


def safe_divide(numerator: float, denominator: float, default: float = 1.0) -> float:
    """Safe division with fallback."""
    if denominator == 0 or pd.isna(denominator):
        return default
    result = numerator / denominator
    return default if pd.isna(result) else result


def validate_inputs(line: float, odds:  float) -> Tuple[bool, str]:
    """Validate bet inputs."""
    if line <= 0:
        return False, "Line must be positive"
    if odds < 1.01: 
        return False, "Odds must be at least 1.01"
    if odds > 100: 
        return False, "Odds seem unrealistic (>100)"
    return True, ""


def american_to_decimal(odds: float) -> float:
    """Convert odds to decimal format for ML features.
    Handles both American (-110, +150) and already-decimal (1.91, 2.50) formats.
    """
    if odds == 0:
        return 1.91  # Default -110
    elif odds >= 1.0 and odds < 50:
        # Already in decimal format (e.g., 1.91, 2.50)
        return odds
    elif odds > 0:
        # American positive (+150)
        return (odds / 100) + 1
    else:
        # American negative (-110)
        return (100 / abs(odds)) + 1


# =============================================================================
# LAYER 1: DATA LOADER
# =============================================================================

class DataLoaderError(Exception):
    """Custom exception for data loading failures."""
    pass


class DataLoader: 
    """
    Strictly responsible for fetching API data and managing caches.
    Fails hard with clear errors if data is missing. 
    """
    
    def __init__(self, config: Config = CONFIG):
        self.config = config
        self._position_cache: Dict[int, str] = {}
        self._team_stats_cache: Optional[Tuple[pd.DataFrame, float, float]] = None
        self._team_stats_cache_time: float = 0
    
    def _api_call_with_retry(self, func, description: str = "API call"):
        """Execute API call with retry logic. Exponential backoff for rate limits."""
        import random
        last_exception = None
        cooldown_count = 0
        MAX_COOLDOWNS = 5  # More patient - 5 cooldowns before failing
        
        max_retries = self.config.API_MAX_RETRIES + 4  # More retries total
        
        for attempt in range(max_retries):
            try:
                # Add base delay + small jitter to look less robotic
                time.sleep(self.config.API_DELAY + random.uniform(0.1, 0.5))
                return func()
            except Exception as e: 
                last_exception = e
                error_str = str(e).lower()
                
                # CHECK FOR TIMEOUTS / RATE LIMITS
                if "timed out" in error_str or "timeout" in error_str or "429" in error_str or "read timeout" in error_str:
                    cooldown_count += 1
                    if cooldown_count >= MAX_COOLDOWNS:
                        logger.error(f"🛑 {MAX_COOLDOWNS} consecutive timeouts. API is blocked. Failing fast.")
                        raise DataLoaderError(f"{description} failed: API rate limited after {cooldown_count} timeouts")
                    # Exponential backoff: 30s, 60s, 90s, 120s
                    wait_time = 30 * cooldown_count
                    logger.warning(f"⚠️ API Cooldown {cooldown_count}/{MAX_COOLDOWNS}. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # Standard error (e.g. JSON decode), short wait
                    delay = self.config.API_DELAY * (attempt + 1)
                    logger.warning(f"{description} attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
        
        raise DataLoaderError(f"{description} failed after {max_retries} attempts: {last_exception}")
    
    def get_player_and_team(self, player_name: str, opponent_name: str) -> Tuple[Dict, Dict]:
        """
        Resolve player and team from names.
        Raises DataLoaderError if not found.
        """
        clean_player = normalize_name(player_name)
        clean_opp = normalize_name(opponent_name)
        
        all_players = players. get_players()
        p_obj = next((p for p in all_players if normalize_name(p['full_name']) == clean_player), None)
        if not p_obj: 
            p_obj = next((p for p in all_players if clean_player in normalize_name(p['full_name'])), None)
        
        if not p_obj:
            raise DataLoaderError(f"Player not found: '{player_name}'.  Check spelling.")
        
        all_teams = teams. get_teams()
        t_obj = next((t for t in all_teams
                      if normalize_name(t['full_name']) == clean_opp
                      or normalize_name(t['abbreviation']) == clean_opp
                      or normalize_name(t['nickname']) == clean_opp), None)
        
        if not t_obj:
            raise DataLoaderError(f"Team not found: '{opponent_name}'. Check spelling.")
        
        return p_obj, t_obj
    
    @st.cache_data(ttl=CONFIG.CACHE_TTL_PLAYER_IDS)
    def get_player_position(_self, player_id:  int) -> str:
        """Fetch player's primary position."""
        try:
            def api_call():
                return commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
            
            df = _self._api_call_with_retry(api_call, f"Get position for player {player_id}")
            
            if df is None or len(df) == 0:
                return 'SF'
            
            position = df['POSITION'].iloc[0] if 'POSITION' in df.columns else ''
            position = str(position).upper().strip()
            
            position_map = {
                'GUARD': 'PG', 'POINT GUARD': 'PG', 'G': 'PG', 'PG':  'PG',
                'SHOOTING GUARD': 'SG', 'SG': 'SG', 'G-F': 'SG',
                'FORWARD': 'SF', 'SMALL FORWARD': 'SF', 'SF':  'SF', 'F': 'SF', 'F-G': 'SF',
                'POWER FORWARD': 'PF', 'PF': 'PF', 'F-C': 'PF',
                'CENTER': 'C', 'C': 'C', 'C-F': 'C',
            }
            
            for key, val in position_map. items():
                if key in position: 
                    return val
            
            if '-' in position: 
                first_pos = position.split('-')[0].strip()
                for key, val in position_map.items():
                    if key in first_pos: 
                        return val
            
            return 'SF'
        except Exception as e: 
            logger.error(f"Failed to get player position: {e}")
            return 'SF'
    
    @st.cache_data(ttl=CONFIG.CACHE_TTL_GAME_LOGS)
    def fetch_game_logs(_self, player_id: int, season: str = None) -> pd.DataFrame:
        """
        Fetch player game logs for a season.
        Returns empty DataFrame if no data (not an error for new players).
        """
        if season is None:
            season = _self.config. CURRENT_SEASON
        
        try: 
            def api_call():
                return playergamelog. PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
            
            df = _self._api_call_with_retry(api_call, f"Fetch game logs for player {player_id}")
            
            if df is None or len(df) == 0:
                return pd.DataFrame()
            
            # Process the dataframe
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            df = df.sort_values('GAME_DATE')
            
            # Create combo stats
            df['PRA'] = df['PTS'] + df['REB'] + df['AST']
            df['PA'] = df['PTS'] + df['AST']
            df['PR'] = df['PTS'] + df['REB']
            df['RA'] = df['REB'] + df['AST']
            df['IS_HOME'] = df['MATCHUP'].str.contains('vs. ', na=False)
            
            if 'FG3M' in df.columns:
                df['3PM'] = df['FG3M']
            
            df['DAYS_REST'] = df['GAME_DATE']. diff().dt.days - 1
            df['DAYS_REST'] = df['DAYS_REST'].fillna(3).clip(lower=0)
            
            # Parse minutes
            if df['MIN'].dtype == object:
                def parse_minutes(min_val):
                    try:
                        if pd.isna(min_val):
                            return 0.0
                        if isinstance(min_val, str):
                            if ': ' in min_val:
                                parts = min_val. split(':')
                                return float(parts[0]) + float(parts[1]) / 60
                            return float(min_val)
                        return float(min_val)
                    except (ValueError, IndexError, TypeError):
                        return 0.0
                df['MIN_FLOAT'] = df['MIN'].apply(parse_minutes)
            else:
                df['MIN_FLOAT'] = pd.to_numeric(df['MIN'], errors='coerce').fillna(0)
            
            # Per-minute stats
            target_stats = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M']
            for stat in target_stats:
                if stat in df.columns:
                    df[f'{stat}_PER_MIN'] = df[stat] / df['MIN_FLOAT']
                    df[f'{stat}_PER_MIN'] = df[f'{stat}_PER_MIN'].fillna(0.0)
            return df
            
        except DataLoaderError: 
            raise
        except Exception as e:
            logger.error(f"Failed to fetch game logs for player {player_id}: {e}")
            return pd. DataFrame()
    
    def fetch_multi_season_logs(self, player_id: int) -> pd.DataFrame:
        """Fetch logs from current and previous season."""
        try:
            df_current = self.fetch_game_logs(player_id, self.config.CURRENT_SEASON)
            df_prev = self.fetch_game_logs(player_id, self.config.PREV_SEASON)
            frames = [df for df in [df_current, df_prev] if len(df) > 0]
            return pd. concat(frames, ignore_index=True) if frames else pd. DataFrame()
        except Exception as e: 
            logger.error(f"Failed to fetch multi-season logs:  {e}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=CONFIG.CACHE_TTL_TEAM_STATS)
    def fetch_team_stats(_self) -> Tuple[pd.DataFrame, float, float]:
        """Fetch league-wide team stats."""
        try:
            def api_call():
                return leaguedashteamstats.LeagueDashTeamStats(
                    season=_self.config. CURRENT_SEASON,
                    measure_type_detailed_defense='Advanced',
                    per_mode_detailed='PerGame'
                ).get_data_frames()[0]
            
            stats = _self._api_call_with_retry(api_call, "Fetch team stats")
            
            if stats is None or len(stats) == 0:
                logger.warning("No team stats available, using defaults")
                return pd.DataFrame(), _self.config.DEFAULT_PACE, _self.config.DEFAULT_DEF_RATING
            
            if 'PACE' not in stats.columns:
                stats['PACE'] = _self.config.DEFAULT_PACE
            if 'DEF_RATING' not in stats.columns:
                stats['DEF_RATING'] = _self.config. DEFAULT_DEF_RATING
            
            return stats. set_index('TEAM_ID'), stats['PACE']. mean(), stats['DEF_RATING'].mean()
            
        except Exception as e:
            logger.error(f"Failed to fetch team stats: {e}")
            return pd.DataFrame(), _self.config.DEFAULT_PACE, _self.config.DEFAULT_DEF_RATING
    
    def fetch_opponent_stats(self, season: str = None) -> pd.DataFrame:
        """Fetch opponent (defensive) stats for position defense calculation."""
        if season is None:
            season = self.config.CURRENT_SEASON
        try:
            def api_call():
                return leaguedashteamstats.LeagueDashTeamStats(
                    season=season,
                    measure_type_detailed_defense='Opponent',
                    per_mode_detailed='PerGame'
                ).get_data_frames()[0]
            
            return self._api_call_with_retry(api_call, f"Fetch opponent stats ({season})")
        except Exception as e:
            logger.error(f"Failed to fetch opponent stats for {season}: {e}")
            return pd.DataFrame()
    
    def fetch_position_defense(self, season: str = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate position-based defense multipliers from real data.
        Uses per-season caching to avoid repeated API calls.
        """
        if season is None:
            season = self.config.CURRENT_SEASON
        
        # Season-specific cache file
        cache_file = DATA_DIR / f"position_defense_cache_{season.replace('-', '_')}.json"
        
        # Check file cache
        try:
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
                    if time.time() - cache.get('timestamp', 0) < self.config.CACHE_TTL_POSITION_DEF:
                        logger.info(f"Using cached position defense data for {season}")
                        return cache.get('data', {})
        except (json.JSONDecodeError, IOError):
            pass
        
        logger.info(f"Fetching position defense data for {season} from NBA API...")
        
        opp_stats = self.fetch_opponent_stats(season=season)
        if opp_stats is None or len(opp_stats) == 0:
            return self._get_neutral_position_multipliers()
        
        team_position_mult = self._calculate_position_multipliers(opp_stats)
        
        # Save to season-specific cache
        try:
            cache = {'timestamp': time.time(), 'season': season, 'data': team_position_mult}
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save position defense cache for {season}: {e}")
        
        return team_position_mult
    
    def _calculate_position_multipliers(self, opp_stats:  pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate position multipliers from opponent stats."""
        all_teams = teams.get_teams()
        team_id_to_abbrev = {t['id']: t['abbreviation'] for t in all_teams}
        
        available_cols = opp_stats.columns.tolist()
        
        fg3_col = 'OPP_FG3_PCT' if 'OPP_FG3_PCT' in available_cols else 'FG3_PCT'
        pts_col = 'OPP_PTS' if 'OPP_PTS' in available_cols else 'PTS'
        reb_col = 'OPP_REB' if 'OPP_REB' in available_cols else 'REB'
        ast_col = 'OPP_AST' if 'OPP_AST' in available_cols else 'AST'
        
        avg_fg3_pct = opp_stats[fg3_col].mean() if fg3_col in available_cols else 0.36
        avg_pts = opp_stats[pts_col]. mean() if pts_col in available_cols else 115.0
        avg_reb = opp_stats[reb_col].mean() if reb_col in available_cols else 44.0
        avg_ast = opp_stats[ast_col].mean() if ast_col in available_cols else 25.0
        
        team_position_mult = {}
        
        for _, row in opp_stats.iterrows():
            team_id = row. get('TEAM_ID')
            abbrev = team_id_to_abbrev.get(team_id)
            
            if not abbrev:
                continue
            
            team_fg3_pct = row. get(fg3_col, avg_fg3_pct)
            team_pts = row. get(pts_col, avg_pts)
            team_reb = row.get(reb_col, avg_reb)
            team_ast = row.get(ast_col, avg_ast)
            
            fg3_mult = team_fg3_pct / avg_fg3_pct if avg_fg3_pct > 0 else 1.0
            pts_mult = team_pts / avg_pts if avg_pts > 0 else 1.0
            reb_mult = team_reb / avg_reb if avg_reb > 0 else 1.0
            ast_mult = team_ast / avg_ast if avg_ast > 0 else 1.0
            
            def clamp(v):
                return round(max(self.config. MATCHUP_MULT_MIN, min(self.config. MATCHUP_MULT_MAX, v)), 3)
            
            team_position_mult[abbrev] = {
                'PG': clamp(0.35 * fg3_mult + 0.30 * ast_mult + 0.35 * pts_mult),
                'SG': clamp(0.45 * fg3_mult + 0.20 * ast_mult + 0.35 * pts_mult),
                'SF': clamp(0.30 * fg3_mult + 0.20 * reb_mult + 0.50 * pts_mult),
                'PF': clamp(0.15 * fg3_mult + 0.40 * reb_mult + 0.45 * pts_mult),
                'C': clamp(0.10 * fg3_mult + 0.50 * reb_mult + 0.40 * pts_mult),
            }
        
        return team_position_mult
    
    def _get_neutral_position_multipliers(self) -> Dict[str, Dict[str, float]]: 
        """Return neutral multipliers as fallback."""
        all_teams = teams. get_teams()
        return {
            t['abbreviation']:  {'PG': 1.0, 'SG': 1.0, 'SF': 1.0, 'PF': 1.0, 'C': 1.0}
            for t in all_teams
        }


# =============================================================================
# LAYER 2:  FEATURE ENGINEER
# =============================================================================

class InjuryManager:
    """
    [UPGRADED] Injury-Aware Usage Redistribution Engine.
    
    When a high-usage teammate is OUT, automatically boosts the analyzed player's
    projected usage based on the missing player's usage share.
    
    Logic:
    1. Fetch team's top 5 usage players from NBA API
    2. Check injury report for who's OUT/DOUBTFUL
    3. Redistribute ~60% of missing player's usage to remaining starters
    4. Return multiplier for the analyzed player
    
    Example:
    - Tyrese Maxey (USG 28%) is OUT
    - Kyle Lowry gets ~10% usage boost (28% * 0.6 / 3 teammates)
    - Returns usage_mult = 1.10
    """
    def __init__(self, rapid_api_key: str = None):
        self.rapid_api_key = rapid_api_key
        # Cache for injury reports (1 hour TTL)
        self._injury_cache: Dict[str, Any] = {}
        # Cache for team usage data (24 hour TTL - doesn't change often)
        self._usage_cache: Dict[str, Any] = {}

    def get_injury_usage_boost(self, player_name: str, team_id: int) -> Tuple[float, List[str]]:
        """
        Calculate usage boost for a player based on injured teammates.
        
        Returns:
            Tuple of (usage_multiplier, list_of_injured_high_usage_players)
        """
        try:
            # 1. Get team's top usage players
            team_usage = self._get_team_usage_leaders(team_id)
            if not team_usage:
                return 1.0, []
            
            # 2. Get injury report for team
            injured_players = self._fetch_team_injuries(team_id)
            if not injured_players:
                return 1.0, []
            
            # 3. Find injured high-usage players (not the analyzed player)
            injured_high_usage = []
            missing_usage_total = 0.0
            
            for inj in injured_players:
                inj_name = inj.get('name', '').lower()
                inj_status = inj.get('status', '').upper()
                
                # Skip if this is the player we're analyzing
                if player_name.lower() in inj_name or inj_name in player_name.lower():
                    continue
                
                # Check if injured player is a high-usage player
                for usage_player in team_usage:
                    if usage_player['name'].lower() in inj_name or inj_name in usage_player['name'].lower():
                        if inj_status in ['OUT', 'INACTIVE']:
                            missing_usage_total += usage_player['usg_pct']
                            injured_high_usage.append(f"{usage_player['name']} ({usage_player['usg_pct']:.1%})")
                        elif inj_status == 'DOUBTFUL':
                            # 50% weight for doubtful players
                            missing_usage_total += usage_player['usg_pct'] * 0.5
                            injured_high_usage.append(f"{usage_player['name']} ({usage_player['usg_pct']:.1%}) [GTD]")
                        break
            
            if missing_usage_total == 0:
                return 1.0, []
            
            # 4. Calculate usage boost
            # Redistribute 60% of missing usage, split among ~3 remaining starters
            redistributed = missing_usage_total * 0.60
            boost_per_player = redistributed / 3.0  # Assume 3 other starters benefit
            
            # Convert to multiplier (e.g., 0.10 boost = 1.10 multiplier)
            # Cap at 1.30 (30% max boost) to prevent explosions
            usage_mult = min(1.30, 1.0 + boost_per_player)
            
            return usage_mult, injured_high_usage
            
        except Exception as e:
            logger.warning(f"Injury usage boost calculation failed: {e}")
            return 1.0, []

    def _get_team_usage_leaders(self, team_id: int) -> List[Dict]:
        """
        Fetch top 5 usage players for a team from NBA API.
        Cached for 24 hours since usage doesn't change game-to-game.
        """
        cache_key = f"usage_{team_id}"
        
        # Check cache (24 hour TTL)
        if cache_key in self._usage_cache:
            cached = self._usage_cache[cache_key]
            if (datetime.now() - cached['time']).seconds < 86400:
                return cached['data']
        
        try:
            from nba_api.stats.endpoints import leaguedashplayerstats
            import time
            
            # Get all players' usage stats for current season
            time.sleep(0.6)  # Rate limit
            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season='2025-26',
                per_mode_detailed='PerGame',
                measure_type_detailed_defense='Base'
            ).get_data_frames()[0]
            
            # Filter to just this team
            team_players = stats[stats['TEAM_ID'] == team_id].copy()
            
            if len(team_players) == 0:
                return []
            
            # Sort by usage, get top 5
            team_players = team_players.nlargest(5, 'USG_PCT')
            
            leaders = []
            for _, row in team_players.iterrows():
                leaders.append({
                    'name': row['PLAYER_NAME'],
                    'player_id': row['PLAYER_ID'],
                    'usg_pct': row['USG_PCT'] / 100.0,  # Convert to decimal
                    'min': row['MIN']
                })
            
            self._usage_cache[cache_key] = {'time': datetime.now(), 'data': leaders}
            return leaders
            
        except Exception as e:
            logger.warning(f"Failed to fetch team usage leaders: {e}")
            return []

    def _fetch_team_injuries(self, team_id: int) -> List[Dict]:
        """
        Fetch injury list for a team.
        Uses NBA API CommonTeamRoster for free injury data.
        """
        cache_key = f"injuries_{team_id}"
        
        # Check cache (1 hour TTL)
        if cache_key in self._injury_cache:
            cached = self._injury_cache[cache_key]
            if (datetime.now() - cached['time']).seconds < 3600:
                return cached['data']

        data = []
        
        try:
            from nba_api.stats.endpoints import commonteamroster
            import time
            
            time.sleep(0.6)  # Rate limit
            roster = commonteamroster.CommonTeamRoster(team_id=team_id).get_data_frames()[0]
            
            # Check for HOW_ACQUIRED column which sometimes has injury info
            # Also check PLAYER_STATUS if available
            for _, player in roster.iterrows():
                status = 'ACTIVE'
                
                # Check various status columns
                if 'PLAYER_STATUS' in roster.columns:
                    ps = str(player.get('PLAYER_STATUS', '')).upper()
                    if 'OUT' in ps or 'INJ' in ps:
                        status = 'OUT'
                    elif 'DOUBT' in ps or 'GTD' in ps:
                        status = 'DOUBTFUL'
                
                # Also check HOW_ACQUIRED for "Injured" designation
                if 'HOW_ACQUIRED' in roster.columns:
                    ha = str(player.get('HOW_ACQUIRED', '')).upper()
                    if 'INJ' in ha:
                        status = 'OUT'
                
                if status != 'ACTIVE':
                    data.append({
                        'name': player.get('PLAYER', ''),
                        'status': status
                    })
                    
        except Exception as e:
            logger.debug(f"Roster fetch failed: {e}")
            
        # Also try to get today's injury report if available
        try:
            # Attempt to scrape or use alternative source
            # For now, rely on roster data above
            pass
        except Exception:
            pass

        self._injury_cache[cache_key] = {'time': datetime.now(), 'data': data}
        return data
    
    # Keep legacy method for backward compatibility
    def get_injury_impact(self, player_id: int, team_id: int) -> float:
        """Legacy method - returns simple boost based on injured count."""
        team_injuries = self._fetch_team_injuries(team_id)
        if not team_injuries:
            return 1.0
        
        usage_bump = 0.0
        for p in team_injuries:
            status = p.get('status', '').upper()
            if status in ['OUT', 'INACTIVE']:
                usage_bump += 0.05
            elif status == 'DOUBTFUL':
                usage_bump += 0.02
        
        return min(1.25, 1.0 + usage_bump)


class FeatureEngineer: 
    """
    Calculates EMA, Volatility, Pace Adjustments, Position Factors. 
    Outputs standardized FeatureVector for ML compatibility.
    
    V15 Upgrades:
    - Uses BlowoutPredictor (logistic regression) for blowout risk
    - Uses PlayerFatigueProfiler for player-specific rest factors
    - Uses InjuryManager for dynamic usage adjustments
    - Calculates dynamic CV-based simulation width
    """
    
    def __init__(self, config: Config = CONFIG):
        self.config = config
        # Initialize V15 intelligent models
        self.blowout_predictor = BlowoutPredictor(config)
        self.fatigue_profiler = PlayerFatigueProfiler(config)
        self.injury_manager = InjuryManager() # [UPGRADED] Smart Injury-Aware Usage
    
    def get_injury_usage_boost(self, player_name: str, team_id: int) -> Tuple[float, List[str]]:
        """
        [NEW] Get usage boost from injured high-usage teammates.
        Returns (multiplier, list_of_injured_stars)
        """
        return self.injury_manager.get_injury_usage_boost(player_name, team_id)
    
    def get_season_usage_pct(self, player_id: int) -> float:
        """
        [V16] Fetch season USG% for role context.
        Returns 0.0 if unavailable (caches for performance).
        """
        cache_key = f"usg_season_{player_id}_{CONFIG.CURRENT_SEASON}"
        if hasattr(self, '_usg_cache') and cache_key in self._usg_cache:
            return self._usg_cache[cache_key]
        
        if not hasattr(self, '_usg_cache'):
            self._usg_cache = {}
        
        try:
            # Fetch season stats for player
            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=CONFIG.CURRENT_SEASON,
                player_id_nullable=player_id,
                per_mode_detailed='PerGame',
                timeout=30
            )
            df = stats.get_data_frames()[0]
            if len(df) > 0 and 'USG_PCT' in df.columns:
                usg_pct = float(df.iloc[0]['USG_PCT'])
                self._usg_cache[cache_key] = usg_pct
                return usg_pct
        except Exception as e:
            logger.debug(f"Failed to fetch USG% for player {player_id}: {e}")
        
        self._usg_cache[cache_key] = 0.0
        return 0.0
    
    def calculate_composite_usage(self, player_id: int, team_id: int, manual_adj_percent: float) -> float:
        """
        [LEGACY] Combines automated injury data with manual slider input.
        Use get_injury_usage_boost() for the new smart method.
        """
        # 1. Get Auto-Impact from API (e.g., 1.10)
        auto_impact = self.injury_manager.get_injury_impact(player_id, team_id)
        
        # 2. Get Manual Adjustment (e.g., 1.05 or 0.90)
        manual_impact = 1.0 + (manual_adj_percent / 100.0)
        
        # 3. Combine
        return auto_impact * manual_impact

    def calculate_statistical_features(
        self, 
        df: pd.DataFrame, 
        stat_col: str, 
        lookback:  int = 15,
        data_quality: DataQuality = None
    ) -> Dict[str, float]: 
        """Calculate core statistical features from game logs."""
        if data_quality is None:
            data_quality = DataQuality()
            
        if len(df) == 0 or stat_col not in df.columns:
            raise ValueError(f"Cannot calculate features:  missing data for {stat_col}")
        
        recent = df.tail(lookback).copy()
        if len(recent) == 0:
            raise ValueError("No recent games available for analysis")
        
        # Track low sample size
        if len(df) < 10:
            data_quality.low_sample_size = True
            data_quality.add_warning(f"Low sample size: only {len(df)} games available")
        
        # EMA (Exponential Moving Average)
        ema = recent[stat_col].ewm(span=len(recent), adjust=False).mean().iloc[-1]
        
        # Standard Deviation
        std_dev = recent[stat_col].std()
        used_fallback_std = False
        if pd.isna(std_dev) or std_dev == 0:
            mean_val = recent[stat_col].mean()
            std_dev = mean_val * 0.2 if not pd.isna(mean_val) and mean_val > 0 else 1.0
            used_fallback_std = True
            data_quality.used_fallback_std = True
            data_quality.add_warning("Using estimated std (20% of mean) due to insufficient variance")
        
        # Simple Moving Averages
        sma_5 = df.tail(5)[stat_col].mean() if len(df) >= 5 else df[stat_col].mean()
        sma_10 = df.tail(10)[stat_col].mean() if len(df) >= 10 else df[stat_col].mean()
        
        # Trend (comparing first half vs second half of recent games)
        if len(recent) >= 5:
            first_half = recent.head(len(recent) // 2)[stat_col].mean()
            second_half = recent.tail(len(recent) // 2)[stat_col].mean()
            trend = safe_divide(second_half, first_half, 1.0) - 1.0
        else:
            trend = 0.0
        
        # Minutes trend
        used_fallback_minutes = False
        min_minutes_l10 = 0.0
        floor_violation = False
        
        if 'MIN_FLOAT' in recent.columns and len(recent) > 0:
            # [RESTORED MISSING LOGIC] Calculate Average Minutes & Trend
            avg_minutes = recent['MIN_FLOAT'].mean()
            recent_mins = recent.tail(3)['MIN_FLOAT'].mean() if len(recent) >= 3 else avg_minutes
            mins_trend = safe_divide(recent_mins, avg_minutes, 1.0)
            
            # [STRATEGY 1: FLOOR GENERAL]
            # Get the lowest minutes played in last 10 games
            min_minutes_l10 = recent.tail(10)['MIN_FLOAT'].min()
            
            # If their floor is scary low (<15 mins), flag it
            if min_minutes_l10 < 15.0:
                floor_violation = True
                data_quality.add_warning(f"RISK: Player had a recent game with only {min_minutes_l10:.1f} minutes.")
        else: 
            avg_minutes = 30.0
            mins_trend = 1.0
            used_fallback_minutes = True
            data_quality.used_fallback_minutes = True
            data_quality.add_warning("Using default 30 minutes (minutes data unavailable)")
        
        # Usage Trend - SNIPER MODE: Dynamic Usage Engine
        usage_trend_mult = 1.0
        usg_season = 0.0
        usg_l5 = 0.0
        if 'USG_PCT' in df.columns:
            usg_season = df['USG_PCT'].mean()
            usg_l5 = recent['USG_PCT'].mean() if len(recent) > 0 else usg_season
            
            # Calculate Trend (Damped 50%)
            if usg_l5 > 0 and usg_season > 0:
                raw_trend = usg_l5 / usg_season
                usage_trend_mult = 1.0 + ((raw_trend - 1.0) * 0.5)
            else:
                usage_trend_mult = 1.0
        
        return {
            'ema': ema,
            'std':  std_dev,
            'sma_5': sma_5,
            'sma_10':  sma_10,
            'trend':  trend,
            'avg_minutes': avg_minutes,
            'mins_trend': mins_trend,
            'usage_trend_mult': usage_trend_mult,
            'usg_season': usg_season,
            'usg_l5': usg_l5,
            'games_played': len(df),
            'used_fallback_std': used_fallback_std,
            'used_fallback_minutes': used_fallback_minutes,
            'min_minutes_l10': min_minutes_l10,
            'floor_violation': floor_violation
        }
    
    def calculate_hit_rates(
        self, 
        df: pd.DataFrame, 
        stat_col: str, 
        line: float
    ) -> Dict[str, float]: 
        """Calculate historical hit rates at various lookback windows with recency weighting."""
        l5 = (df.tail(5)[stat_col] > line).mean() if len(df) >= 5 else 0
        l10 = (df.tail(10)[stat_col] > line).mean() if len(df) >= 10 else 0
        l15 = (df.tail(15)[stat_col] > line).mean() if len(df) >= 15 else 0
        season = (df[stat_col] > line).mean() if len(df) > 0 else 0
        
        # Recency-weighted hit rate using exponential decay
        # Recent games weighted more heavily than older games
        weighted = 0.0
        if len(df) >= 5:
            recent = df.tail(15).copy()
            n = len(recent)
            # Exponential weights: most recent game = 1.0, decays by ~0.9 per game
            decay_rate = 0.1
            weights = np.exp(-decay_rate * np.arange(n))[::-1]  # Reverse so recent = higher
            hits = (recent[stat_col] > line).astype(float).values
            weighted = np.sum(hits * weights) / np.sum(weights)
        else:
            weighted = season
        
        return {
            'l5': l5,
            'l10': l10,
            'l15': l15,
            'season': season,
            'weighted': weighted
        }
    
    def calculate_matchup_multipliers(
        self,
        team_stats: pd.DataFrame,
        opponent_id: int,
        opponent_abbrev: str,
        player_position: str,
        position_defense: Dict[str, Dict[str, float]],
        avg_pace: float,
        avg_def:  float,
        market:  str,
        data_quality: DataQuality = None
    ) -> Dict[str, float]:
        """Calculate matchup-based multipliers."""
        if data_quality is None:
            data_quality = DataQuality()
            
        used_default_pace = False
        used_default_def = False
        missing_team_stats = False
        used_neutral_position = False
        
        # Base matchup from team stats
        if len(team_stats) == 0 or opponent_id not in team_stats.index:
            pace_mult = 1.0
            def_mult = 1.0
            pace = avg_pace
            drtg = avg_def
            missing_team_stats = True
            data_quality.missing_team_stats = True
            data_quality.add_warning(f"Team stats unavailable for opponent - using neutral matchup")
        else: 
            opp = team_stats.loc[opponent_id]
            pace = opp.get('PACE', avg_pace)
            drtg = opp.get('DEF_RATING', avg_def)
            
            safe_avg_p = avg_pace if avg_pace > 0 else self.config.DEFAULT_PACE
            safe_avg_d = avg_def if avg_def > 0 else self.config.DEFAULT_DEF_RATING
            
            if avg_pace <= 0:
                used_default_pace = True
                data_quality.used_default_pace = True
                data_quality.add_warning("Using default pace (100.0) - league average unavailable")
            if avg_def <= 0:
                used_default_def = True
                data_quality.used_default_def_rating = True
                data_quality.add_warning("Using default def rating (115.0) - league average unavailable")
            
            pace_mult = pace / safe_avg_p
            def_mult = drtg / safe_avg_d
        
        # Market-specific weighting
        if market in ['PTS', 'PRA', 'PA', 'PR', '3PM']:
            base_mult = (pace_mult * 0.4) + (def_mult * 0.6)
        elif market in ['REB']: 
            base_mult = (pace_mult * 0.6) + (def_mult * 0.4)
        elif market in ['AST']:
            base_mult = (pace_mult * 0.5) + (def_mult * 0.5)
        else:
            base_mult = (pace_mult + def_mult) / 2
        
        # Position-specific adjustment
        position_mult = 1.0
        if opponent_abbrev in position_defense:
            pos_data = position_defense[opponent_abbrev]
            if player_position in pos_data:
                position_mult = pos_data[player_position]
        else:
            # Check if using neutral multipliers (all 1.0)
            used_neutral_position = True
            data_quality.used_neutral_position_defense = True
            data_quality.add_warning(f"Position defense data unavailable for {opponent_abbrev} - using neutral")
        
        # Combine multipliers
        base_weight = self.config.BASE_MATCHUP_WEIGHT
        pos_weight = self.config.POSITION_DEF_WEIGHT
        combined_mult = (base_mult * base_weight) + (position_mult * pos_weight)
        combined_mult = max(self.config.MATCHUP_MULT_MIN, min(self.config.MATCHUP_MULT_MAX, combined_mult))
        
        return {
            'pace_mult': pace_mult,
            'def_mult':  def_mult,
            'base_matchup_mult': base_mult,
            'position_mult':  position_mult,
            'combined_matchup_mult': combined_mult,
            'pace':  pace,
            'drtg': drtg,
            'used_default_pace': used_default_pace,
            'used_default_def': used_default_def,
            'missing_team_stats': missing_team_stats,
            'used_neutral_position': used_neutral_position
        }
    
    def calculate_split_factor(
        self, 
        df:  pd.DataFrame, 
        stat_col: str, 
        is_home: bool,
        data_quality: DataQuality = None
    ) -> Tuple[float, bool]:
        """
        Calculate home/away split factor.
        Returns (factor, used_fallback).
        """
        if data_quality is None:
            data_quality = DataQuality()
            
        used_fallback = False
        overall_avg = df[stat_col].mean()
        if overall_avg == 0 or pd.isna(overall_avg):
            used_fallback = True
            data_quality.used_fallback_split = True
            data_quality.add_warning("Cannot calculate split factor - overall average is 0 or NaN")
            return 1.0, used_fallback
        
        split_df = df[df['IS_HOME'] == is_home]
        if len(split_df) == 0:
            used_fallback = True
            data_quality.used_fallback_split = True
            location = "home" if is_home else "away"
            data_quality.add_warning(f"No {location} games found - using neutral split factor")
            return 1.0, used_fallback
        
        split_avg = split_df[stat_col].mean()
        raw_factor = safe_divide(split_avg, overall_avg, 1.0)
        
        # Dampen the effect
        return 1 + (raw_factor - 1) * 0.5, used_fallback
    
    def calculate_rest_factor(self, days_rest: int) -> float:
        """
        Calculate rest/fatigue factor based on days since last game.
        More granular than simple B2B flag.
        
        Args:
            days_rest: Days since last game (0=B2B, 1=normal, 2+=well-rested)
        """
        if days_rest <= 0:
            return self.config.B2B_PENALTY  # B2B penalty
        elif days_rest == 1:
            return 1.0  # Normal rest
        elif days_rest == 2:
            return self.config.REST_BONUS_2_DAYS  # Slight bonus
        else:
            return self.config.REST_BONUS_3_PLUS  # Well-rested bonus
    
    def calculate_game_total_factor(self, game_total: float, market: str) -> float:
        """
        Calculate adjustment based on Vegas game total.
        Higher totals = more scoring opportunities.
        """
        if game_total <= 0:
            return 1.0
        
        baseline = self.config.GAME_TOTAL_NEUTRAL
        deviation = (game_total - baseline) / baseline
        
        # Weight based on market type
        if market in ['PTS', 'PRA', 'PA', '3PM']:
            weight = self.config.GAME_TOTAL_WEIGHT
        elif market in ['AST']:
            weight = self.config.GAME_TOTAL_WEIGHT * 0.8  # Slightly less correlated
        elif market in ['REB']:
            weight = self.config.GAME_TOTAL_WEIGHT * 0.5  # Rebounds less correlated with pace
        else:
            weight = self.config.GAME_TOTAL_WEIGHT * 0.3
        
        return 1.0 + (deviation * weight)
    
    def calculate_opponent_form_factor(
        self, 
        drtg_season: float, 
        drtg_l5: float, 
        avg_def: float
    ) -> float:
        """
        Calculate opponent recent form factor.
        Weights recent defensive performance vs season average.
        """
        if drtg_season <= 0 or drtg_l5 <= 0:
            return 1.0
        
        # Blend recent and season DRTG
        l5_weight = self.config.DRTG_L5_WEIGHT
        blended_drtg = (drtg_l5 * l5_weight) + (drtg_season * (1 - l5_weight))
        
        # Compare to league average
        factor = blended_drtg / avg_def if avg_def > 0 else 1.0
        
        # Cap the adjustment
        return max(0.92, min(1.08, factor))
    
    def calculate_blowout_factor(
        self, 
        spread: float, 
        avg_minutes: float,
        is_home: bool = True,
        opponent_drtg: float = 115.0,
        avg_def: float = 115.0
    ) -> Tuple[float, BlowoutRisk, float]:
        """
        V15: Calculate blowout risk using logistic regression model.
        
        Returns:
            Tuple of (blowout_factor, risk_level, blowout_probability)
        """
        return self.blowout_predictor.get_blowout_factor(
            spread=spread,
            avg_minutes=avg_minutes,
            is_home=is_home,
            opponent_drtg=opponent_drtg,
            league_avg_drtg=avg_def
        )
    
    def calculate_dynamic_std_multiplier(self, coef_variation: float) -> float:
        """
        V15: Calculate dynamic standard deviation multiplier based on player's CV.
        
        Instead of using static variance scaling, let the player's own volatility
        dictate how wide the simulation distribution should be.
        
        Args:
            coef_variation: Player's coefficient of variation (std/mean)
            
        Returns:
            Multiplier to apply to base std in simulation (0.7 to 1.5)
        """
        baseline_cv = self.config.CV_BASELINE  # 0.25 = "average" volatility
        min_mult = self.config.CV_MIN_STD_MULT  # 0.70
        max_mult = self.config.CV_MAX_STD_MULT  # 1.50
        
        # Linear interpolation based on CV
        # CV < baseline → tighter distribution (mult < 1.0)
        # CV > baseline → wider distribution (mult > 1.0)
        if coef_variation <= 0:
            return 1.0
        
        # Calculate ratio to baseline
        cv_ratio = coef_variation / baseline_cv
        
        # Map ratio to multiplier range
        # cv_ratio of 0.5 (very consistent) → min_mult
        # cv_ratio of 1.0 (average) → 1.0
        # cv_ratio of 2.0 (very volatile) → max_mult
        if cv_ratio <= 1.0:
            # Consistent player: interpolate between min_mult and 1.0
            mult = min_mult + (1.0 - min_mult) * cv_ratio
        else:
            # Volatile player: interpolate between 1.0 and max_mult
            # Cap cv_ratio at 2.0 for calculation
            capped_ratio = min(cv_ratio, 2.0)
            mult = 1.0 + (max_mult - 1.0) * (capped_ratio - 1.0)
        
        return float(np.clip(mult, min_mult, max_mult))
    
    def build_feature_vector(
        self,
        player_id: int,
        player_name: str,
        player_position: str,
        opponent_id: int,
        opponent_abbrev: str,
        is_home: bool,
        is_b2b: bool,
        spread: float,
        usage_mult: float,
        df: pd.DataFrame,
        stat_col: str,
        line: float,
        lookback: int,
        team_stats: pd.DataFrame,
        position_defense: Dict[str, Dict[str, float]],
        avg_pace: float,
        avg_def: float,
        market: str,
        days_rest: int = 1,  # New: days since last game
        game_total: float = 0.0,  # New: Vegas O/U total
        opponent_drtg_l5: float = 0.0  # New: Opponent recent DRTG
    ) -> FeatureVector: 
        """
        Build complete feature vector for prediction.
        
        V15 Upgrades:
        - Uses BlowoutPredictor (logistic regression) instead of static thresholds
        - Uses PlayerFatigueProfiler for player-specific rest factors
        - Calculates dynamic CV-based simulation width multiplier
        """
        
        # Initialize data quality tracker
        data_quality = DataQuality()
        
        # Statistical features (with data quality tracking)
        stats = self.calculate_statistical_features(df, stat_col, lookback, data_quality)
        
        # Hit rates
        hit_rates = self.calculate_hit_rates(df, stat_col, line)
        
        # Matchup multipliers (with data quality tracking)
        matchup = self.calculate_matchup_multipliers(
            team_stats, opponent_id, opponent_abbrev, player_position,
            position_defense, avg_pace, avg_def, market, data_quality
        )
        
        # Split factor (with data quality tracking)
        split_factor, _ = self.calculate_split_factor(df, stat_col, is_home, data_quality)
        
        # Opponent form factor
        opponent_drtg_season = matchup.get('drtg', avg_def)
        opp_drtg_l5 = opponent_drtg_l5 if opponent_drtg_l5 > 0 else opponent_drtg_season
        
        # =================================================================
        # V15 UPGRADE #1: Blowout Risk Model (Logistic Regression)
        # Uses P(minutes < 25) instead of static spread thresholds
        # =================================================================
        blowout_factor, blowout_risk, blowout_prob = self.calculate_blowout_factor(
            spread=spread,
            avg_minutes=stats['avg_minutes'],
            is_home=is_home,
            opponent_drtg=opp_drtg_l5,
            avg_def=avg_def
        )
        
        # =================================================================
        # V15 UPGRADE #2: Player-Specific Fatigue Response
        # Uses player's actual B2B performance instead of static 5% penalty
        # =================================================================
        personal_rest_factor, fatigue_profile = self.fatigue_profiler.get_rest_factor(
            days_rest=days_rest if days_rest >= 0 else (0 if is_b2b else 1),
            df=df,
            stat_col=stat_col,
            player_id=player_id
        )
        b2b_games_count = fatigue_profile.get('b2b_games', 0)
        
        # Use personal factor if we have enough data, else fall back to static
        if b2b_games_count >= self.config.FATIGUE_MIN_B2B_GAMES:
            rest_factor = personal_rest_factor
        else:
            # Fall back to static calculation
            rest_factor = self.calculate_rest_factor(
                days_rest if days_rest >= 0 else (0 if is_b2b else 1)
            )
        
        # =================================================================
        # V15 UPGRADE #3: Dynamic CV-Based Simulation Width
        # Let player's volatility dictate the simulation variance
        # =================================================================
        coef_variation = stats['std'] / stats['ema'] if stats['ema'] > 0 else 0.5
        dynamic_std_mult = self.calculate_dynamic_std_multiplier(coef_variation)
        
        # ============================================================
        # APPLY STRATEGY 1 PENALTY
        # ============================================================
        if stats.get('floor_violation', False):
            # Force wider variance. If they are risky, we need a wider range of outcomes.
            # This makes it harder for the simulation to be 70% confident.
            dynamic_std_mult = max(dynamic_std_mult, 1.25)
        
        # Game total factor (applied in model engine, stored here)
        game_total_factor = self.calculate_game_total_factor(game_total, market)
        
        return FeatureVector(
            player_id=player_id,
            player_name=player_name,
            player_position=player_position,
            opponent_id=opponent_id,
            opponent_abbrev=opponent_abbrev,
            is_home=is_home,
            is_b2b=is_b2b,
            spread=spread,
            days_rest=days_rest,
            game_total=game_total,
            market=market,  # Added for low-count stat handling
            line=line,  # Added for Vegas line sanity checks
            opponent_drtg_season=opponent_drtg_season,
            opponent_drtg_l5=opp_drtg_l5,
            ema=stats['ema'],
            std=stats['std'],
            sma_5=stats['sma_5'],
            sma_10=stats['sma_10'],
            trend=stats['trend'],
            avg_minutes=stats['avg_minutes'],
            mins_trend=stats['mins_trend'],
            hit_rate_l5=hit_rates['l5'],
            hit_rate_l10=hit_rates['l10'],
            hit_rate_l15=hit_rates['l15'],
            hit_rate_season=hit_rates['season'],
            hit_rate_weighted=hit_rates['weighted'],
            coef_variation=coef_variation,
            pace_mult=matchup['pace_mult'],
            def_mult=matchup['def_mult'],
            position_mult=matchup['position_mult'],
            base_matchup_mult=matchup['base_matchup_mult'],
            combined_matchup_mult=matchup['combined_matchup_mult'],
            split_factor=split_factor,
            rest_factor=rest_factor,
            blowout_factor=blowout_factor,
            usage_mult=usage_mult * stats.get('usage_trend_mult', 1.0),
            games_played=stats['games_played'],
            # V15 new fields
            blowout_prob=blowout_prob,
            personal_fatigue_factor=personal_rest_factor,
            b2b_games_in_sample=b2b_games_count,
            dynamic_std_mult=dynamic_std_mult,
            # V16 market & role context (defaults, populated at bet time)
            odds_decimal=1.91,  # Will be updated with actual odds
            usg_season=self.get_season_usage_pct(player_id),
            clv=0.0,  # Calculated after closing line known
            data_quality=data_quality
        )

@dataclass
class ProjectionResult:
    """
    Holds the output of the projection model.
    """
    base_projection: float
    final_projection: float
    confidence_score: float
    contributing_factors: Dict[str, float]
    # Added fields to satisfy DecisionPolicy and UI requirements
    ml_prob: Optional[float] = None
    context: str = ""
    
# =============================================================================
# LAYER 3: MODEL ENGINE (SMART ENSEMBLE)
# =============================================================================

class ModelEngine:
    """
    Generates projections using weighted averages + [Strategy 3] Bad Beat Fade.
    Now includes REAL ML integration (loads nba_model.pkl).
    """
    def __init__(self, config: Config = CONFIG, tracker: 'Tracker' = None):
        self.config = config
        self.tracker = tracker
        self.ml_model = None
        self._load_ml_model()

    def _load_ml_model(self):
        """Try to load the trained XGBoost model."""
        model_path = DATA_DIR / "nba_model.pkl"
        if model_path.exists():
            try:
                self.ml_model = joblib.load(model_path)
                # logger.info("🤖 ML Model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load ML model: {e}")

    def get_ml_prediction(self, features: FeatureVector) -> Optional[float]:
        """Get win probability from the ML model."""
        if self.ml_model is None:
            return None
        
        try:
            # Convert feature vector to the exact array format the model expects
            # We must ensure this matches the training order exactly.
            # Using the same method from FeatureVector ensures consistency.
            X = features.to_ml_array().reshape(1, -1)
            
            # Predict probability of class 1 (Win)
            prob = self.ml_model.predict_proba(X)[0][1]
            return float(prob)
        except Exception as e:
            logger.warning(f"ML Prediction failed: {e}")
            return None

    def apply_bad_beat_penalty(self, player_name: str, current_projection: float) -> float:
        """[STRATEGY 3] Check for recent 'Bad Beats' and apply frustration penalty."""
        if self.tracker is None:
            return current_projection

        history = self.tracker.get_player_history(player_name)
        if not history:
            return current_projection

        # Count "Bad Beats" (losses < 1.5 pts) in last 5 bets
        recent_bets = history[-5:]
        bad_beat_count = 0
        
        for bet in recent_bets:
            quality = bet.get('result_quality', '')
            if quality in ['bad_beat', 'close_loss', 'bad_read']:
                bad_beat_count += 1
        
        # If they burned us 2+ times recently, fade them
        if bad_beat_count >= 2:
            penalty = 0.96 if bad_beat_count == 2 else 0.94
            logger.info(f"📉 FADING {player_name}: {bad_beat_count} recent losses. Penalty: {penalty}")
            return current_projection * penalty
            
        return current_projection

    def generate_projection(self, features: FeatureVector) -> ProjectionResult:
        """Generate weighted projection and integrate ML insights."""
        # 1. Base Weighted Average
        w_ema = 0.35
        w_sma10 = 0.25
        w_sma5 = 0.15
        w_trend = 0.10
        w_l10_hit = 0.15
        
        base_proj = (
            (features.ema * w_ema) +
            (features.sma_10 * w_sma10) +
            (features.sma_5 * w_sma5) +
            (features.avg_minutes * (1 + features.trend) * w_trend) + 
            (features.ema * features.hit_rate_weighted * w_l10_hit)
        )
        
        # 2. Apply Multipliers
        adjusted_proj = base_proj * features.combined_matchup_mult
        adjusted_proj *= features.rest_factor
        adjusted_proj *= features.split_factor
        adjusted_proj *= features.usage_mult
        adjusted_proj *= features.blowout_factor
        
        # 3. Apply Game Total Adjustment
        if hasattr(features, 'game_total_factor'):
            adjusted_proj *= features.game_total_factor

        # 4. [STRATEGY 3] Apply Bad Beat Fade
        bad_beat_factor = 1.0
        if self.tracker:
            temp_proj = adjusted_proj
            adjusted_proj = self.apply_bad_beat_penalty(features.player_name, adjusted_proj)
            if temp_proj > 0:
                bad_beat_factor = adjusted_proj / temp_proj
        
        # 5. [LOW-COUNT STAT CAP] Cap projection for discrete stats like STL, BLK, 3PM
        # This prevents unrealistic projections like "6.3 steals"
        low_count_capped = False
        market = getattr(features, 'market', None)
        if market and market in CONFIG.LOW_COUNT_STATS:
            mean_cap = CONFIG.LOW_COUNT_MEAN_CAPS.get(market, 5.0)
            if adjusted_proj > mean_cap:
                adjusted_proj = mean_cap
                low_count_capped = True
        
        # 5b. [LOW-COUNT VEGAS SANITY CHECK] For low-count stats, projection should not
        # be absurdly far from the Vegas line. If line is 1.5 and we project 6.0,
        # that's a 4x difference which indicates model failure, not genius insight.
        # Cap low-count projections to max 2.5x the Vegas line (e.g., line 1.5 -> max 3.75)
        line_value = features.line
        if market and market in CONFIG.LOW_COUNT_STATS and line_value and line_value > 0:
            max_reasonable = line_value * 2.5  # Max 2.5x Vegas line for low-count
            min_reasonable = line_value * 0.4  # Min 40% of Vegas line for low-count
            if adjusted_proj > max_reasonable:
                adjusted_proj = max_reasonable
                low_count_capped = True
            elif adjusted_proj < min_reasonable and line_value >= 1.0:
                adjusted_proj = min_reasonable
                low_count_capped = True

        # 6. Get ML Prediction
        ml_prob = self.get_ml_prediction(features)

        # 7. Generate Context String
        context_parts = []
        if low_count_capped: context_parts.append(f"🎯 Capped ({market} max)")
        if features.combined_matchup_mult > 1.05: context_parts.append("✅ Great Matchup")
        elif features.combined_matchup_mult < 0.95: context_parts.append("❌ Tough Defense")
        if features.rest_factor < 1.0: context_parts.append("⚠️ Fatigue Risk")
        if features.blowout_prob > 0.25: context_parts.append("💨 Blowout Risk")
        if bad_beat_factor < 1.0: context_parts.append("📉 Bad Beat Fade")
        if ml_prob and ml_prob > 0.60: context_parts.append("🤖 ML Likes Over")
        
        context_str = " | ".join(context_parts) if context_parts else "Standard projection based on recent form."

        return ProjectionResult(
            base_projection=float(base_proj),
            final_projection=float(adjusted_proj),
            confidence_score=0.75, 
            contributing_factors={
                'matchup': features.combined_matchup_mult,
                'rest': features.rest_factor,
                'usage': features.usage_mult,
                'blowout': features.blowout_factor,
                'recency_bias': features.def_mult,
                'bad_beat_fade': bad_beat_factor
            },
            ml_prob=ml_prob,  # <--- Now passing the real probability
            context=context_str
        )

# =============================================================================
# LAYER 4: SIMULATION ENGINE (Feature-Aware)
# =============================================================================

class SimulationEngine:
    """
    Runs Monte Carlo simulations with feature-aware adjustments.
    
    The "smart" simulation incorporates:
    - Historical hit rates to calibrate the distribution
    - Trend data to shift mixture probabilities
    - Matchup quality to adjust variance and component weights
    - Blowout risk to increase dud probability
    - Rest/B2B to adjust performance expectations
    """
    
    def __init__(self, config: Config = CONFIG):
        self.config = config
    
    def run_simulation(
        self, 
        mean: float, 
        std: float, 
        line: float, 
        market: str,
        features: 'FeatureVector' = None,  # NEW: Optional feature vector
        simulations: int = None,
        use_mixture: bool = True
    ) -> SimulationResult: 
        """
        Run Monte Carlo simulation with optional feature-aware mixture model.
        
        When features are provided, the simulation becomes "smart":
        - Adjusts mixture component probabilities based on context
        - Calibrates mean based on historical hit rates
        - Scales variance based on matchup and rest factors
        """
        if simulations is None: 
            simulations = self.config.MONTE_CARLO_SIMS
        
        # Determine if this should be treated as low-count:
        # 1. Explicitly low-count stat (3PM, STL, BLK, TOV)
        # 2. Any stat where the player's average is below threshold (e.g., role player AST)
        is_low_count_stat = market in self.config.LOW_COUNT_STATS
        is_low_average = mean < self.config.LOW_AVERAGE_THRESHOLD
        use_low_count_simulation = is_low_count_stat or is_low_average
        
        if use_low_count_simulation: 
            # Use Poisson for low-count stats with strict upside caps
            sims = self._generate_low_count_samples(mean, line, market, features, simulations)
        else:
            if use_mixture and features is not None:
                # SMART simulation with feature-aware mixture
                sims = self._generate_smart_mixture_samples(mean, std, line, features, simulations)
            elif use_mixture:
                # Legacy mixture (no features)
                sims = self._generate_mixture_samples(mean, std, simulations)
            else:
                # Simple normal distribution
                adjusted_std = max(0.1, std) * 1.10
                sims = np.random.normal(mean, adjusted_std, simulations)
        
        # Ensure non-negative
        sims = np.maximum(sims, 0)
        
        over_rate = (sims > line).mean()
        under_rate = (sims <= line).mean()
        
        # =================================================================
        # CRITICAL: Cap probabilities to prevent 99-100% hallucinations
        # No bet in sports is ever truly 99%+ - there's always variance
        # =================================================================
        max_prob = self.config.MAX_PROBABILITY_CAP
        min_prob = self.config.MIN_PROBABILITY_FLOOR
        
        over_rate = float(np.clip(over_rate, min_prob, max_prob))
        under_rate = float(np.clip(under_rate, min_prob, max_prob))
        
        return SimulationResult(
            over_prob=over_rate,
            under_prob=under_rate,
            median=np.median(sims),
            ci_10=np.percentile(sims, 10),
            ci_90=np.percentile(sims, 90),
            simulations=sims
        )
    
    def _apply_feature_adjustments_to_mean(
        self, 
        mean: float, 
        line: float, 
        features: 'FeatureVector'
    ) -> float:
        """
        Apply small adjustments to simulation mean based on features.
        This helps calibrate the simulation to historical performance.
        """
        adjusted = mean
        
        # Calibration based on weighted hit rate vs expected
        # If historical hit rate is high (>60%), player consistently beats similar lines
        # Nudge the mean slightly toward the line direction
        gap_from_line = mean - line
        
        if features.hit_rate_weighted > 0.65 and gap_from_line > 0:
            # Strong OVER history + projection above line = reinforce
            adjusted += min(0.5, gap_from_line * 0.1)
        elif features.hit_rate_weighted < 0.35 and gap_from_line < 0:
            # Strong UNDER history + projection below line = reinforce
            adjusted -= min(0.5, abs(gap_from_line) * 0.1)
        elif features.hit_rate_weighted > 0.60 and gap_from_line < 0:
            # Good hit rate but projection is under line - slight upward correction
            adjusted += min(0.3, abs(gap_from_line) * 0.05)
        elif features.hit_rate_weighted < 0.40 and gap_from_line > 0:
            # Poor hit rate but projection over line - slight downward correction
            adjusted -= min(0.3, gap_from_line * 0.05)
        
        # Trend adjustment (capped to avoid overreaction)
        if abs(features.trend) > 0.05:
            trend_adjustment = features.trend * mean * 0.02  # Max ~2% shift
            adjusted += max(-0.5, min(0.5, trend_adjustment))
        
        return adjusted
    
    def _calculate_dynamic_mixture_probs(
        self, 
        features: 'FeatureVector'
    ) -> Tuple[float, float, float]:
        """
        Calculate dynamic mixture probabilities based on features.
        
        Returns: (normal_prob, blowup_prob, dud_prob)
        """
        # Start with base probabilities
        blowup_base = self.config.MIXTURE_BLOWUP_PROB  # 0.12
        dud_base = self.config.MIXTURE_DUD_PROB        # 0.10
        
        blowup_prob = blowup_base
        dud_prob = dud_base
        
        # === TREND ADJUSTMENTS ===
        # Strong positive trend → more likely to blow up
        if features.trend > 0.10:
            blowup_prob += 0.04
            dud_prob -= 0.02
        elif features.trend > 0.05:
            blowup_prob += 0.02
        # Strong negative trend → more likely to dud
        elif features.trend < -0.10:
            dud_prob += 0.04
            blowup_prob -= 0.02
        elif features.trend < -0.05:
            dud_prob += 0.02
        
        # === MATCHUP ADJUSTMENTS ===
        # Favorable matchup (high combined multiplier) → more blowups
        if features.combined_matchup_mult > 1.08:
            blowup_prob += 0.03
            dud_prob -= 0.01
        elif features.combined_matchup_mult > 1.04:
            blowup_prob += 0.02
        # Tough matchup → more duds
        elif features.combined_matchup_mult < 0.92:
            dud_prob += 0.03
            blowup_prob -= 0.02
        elif features.combined_matchup_mult < 0.96:
            dud_prob += 0.02
        
        # === BLOWOUT RISK (V15: Use logistic regression probability) ===
        # High blowout probability → significantly more duds (early benching)
        if features.blowout_prob >= 0.40:
            dud_prob += 0.10
            blowup_prob -= 0.05
        elif features.blowout_prob >= 0.30:
            dud_prob += 0.07
            blowup_prob -= 0.03
        elif features.blowout_prob >= 0.20:
            dud_prob += 0.04
            blowup_prob -= 0.02
        elif features.blowout_prob >= 0.15:
            dud_prob += 0.02
        
        # === REST/FATIGUE (V15: Use player-specific fatigue profile) ===
        # Check the actual personal fatigue factor instead of just B2B flag
        if features.personal_fatigue_factor < 0.90:
            # Player historically struggles with fatigue
            dud_prob += 0.05
            blowup_prob -= 0.02
        elif features.personal_fatigue_factor < 0.95:
            dud_prob += 0.02
            blowup_prob -= 0.01
        elif features.personal_fatigue_factor > 1.02:
            # Player is well rested and historically performs better
            blowup_prob += 0.02
        
        # Also check raw B2B status for context
        if features.is_b2b or features.days_rest <= 0:
            dud_prob += 0.02  # Additional B2B penalty on top of personal factor
        
        # === VOLATILITY (CV) ===
        # High volatility player → wider tails
        if features.coef_variation > 0.40:
            blowup_prob += 0.03
            dud_prob += 0.03
        elif features.coef_variation > 0.30:
            blowup_prob += 0.01
            dud_prob += 0.01
        
        # === HIT RATE CONSISTENCY ===
        # Very consistent hitter → fewer duds
        if features.hit_rate_weighted > 0.70:
            dud_prob -= 0.03
        elif features.hit_rate_weighted > 0.60:
            dud_prob -= 0.01
        # Inconsistent → more duds
        elif features.hit_rate_weighted < 0.35:
            dud_prob += 0.02
        
        # Clamp probabilities to valid range
        blowup_prob = max(0.02, min(0.25, blowup_prob))
        dud_prob = max(0.02, min(0.25, dud_prob))
        
        # Ensure they sum correctly
        total_extremes = blowup_prob + dud_prob
        if total_extremes > 0.40:
            # Scale down proportionally
            scale = 0.40 / total_extremes
            blowup_prob *= scale
            dud_prob *= scale
        
        normal_prob = 1.0 - blowup_prob - dud_prob
        
        return (normal_prob, blowup_prob, dud_prob)
    
    def _generate_smart_mixture_samples(
        self, 
        mean: float, 
        std: float, 
        line: float,
        features: 'FeatureVector',
        n_samples: int
    ) -> np.ndarray:
        """
        Generate samples from a feature-aware 3-component mixture distribution.
        
        V15 Key improvements:
        1. Dynamic component probabilities based on context
        2. Calibrated mean based on historical hit rates
        3. Dynamic CV-based variance scaling (player's own volatility dictates width)
        4. Blowout probability from logistic regression model
        """
        # Get dynamic mixture probabilities
        normal_prob, blowup_prob, dud_prob = self._calculate_dynamic_mixture_probs(features)
        
        # Calibrate the mean based on features
        calibrated_mean = self._apply_feature_adjustments_to_mean(mean, line, features)
        
        # =================================================================
        # V15 UPGRADE #3: Dynamic CV-Based Simulation Width
        # Use the pre-calculated dynamic_std_mult from FeatureVector
        # This replaces the old static CV thresholds
        # =================================================================
        # CRITICAL: Minimum std floor based on mean to prevent over-confidence
        # For a player averaging 20 PTS, std should be at least 3-4
        # For a player averaging 2 AST, std should be at least 1
        min_std_floor = max(0.5, mean * 0.15)  # At least 15% of the mean
        base_std = max(min_std_floor, std)
        
        # Use the player's personal volatility multiplier (0.7 to 1.5)
        # This was calculated in FeatureEngineer based on their CV
        variance_mult = features.dynamic_std_mult
        
        # Matchup affects variance slightly (on top of personal volatility)
        if features.combined_matchup_mult > 1.05 or features.combined_matchup_mult < 0.95:
            variance_mult *= 1.03  # More extreme matchups = slightly more variance
        
        adjusted_std = base_std * variance_mult
        
        # Determine which component each sample comes from
        component = np.random.choice(
            ['normal', 'blowup', 'dud'],
            size=n_samples,
            p=[normal_prob, blowup_prob, dud_prob]
        )
        
        # Generate samples
        sims = np.zeros(n_samples)
        
        # Normal games - centered at calibrated mean
        normal_mask = component == 'normal'
        sims[normal_mask] = np.random.normal(calibrated_mean, adjusted_std, normal_mask.sum())
        
        # Blow-up games - adjust multiplier based on matchup quality
        blowup_mask = component == 'blowup'
        blowup_mult = self.config.MIXTURE_BLOWUP_MULT  # 1.35
        if features.combined_matchup_mult > 1.05:
            blowup_mult *= 1.05  # Even bigger blowups vs bad defense
        blowup_mean = calibrated_mean * blowup_mult
        blowup_std = adjusted_std * 1.3
        sims[blowup_mask] = np.random.normal(blowup_mean, blowup_std, blowup_mask.sum())
        
        # Dud games - adjust based on blowout probability (V15)
        dud_mask = component == 'dud'
        dud_mult = self.config.MIXTURE_DUD_MULT  # 0.60
        
        # V15: Use blowout probability from logistic regression for more precise adjustment
        if features.blowout_prob >= 0.35:
            dud_mult *= 0.80  # Severe blowout risk = even worse duds
        elif features.blowout_prob >= 0.25:
            dud_mult *= 0.85  # Moderate blowout risk
        elif features.blowout_prob >= 0.15:
            dud_mult *= 0.90  # Slight blowout risk
            
        dud_mean = calibrated_mean * dud_mult
        dud_std = adjusted_std * 0.7
        sims[dud_mask] = np.random.normal(dud_mean, dud_std, dud_mask.sum())
        
        return sims
    
    def _generate_low_count_samples(
        self,
        mean: float,
        line: float,
        market: str,
        features: 'FeatureVector',
        n_samples: int
    ) -> np.ndarray:
        """
        Generate samples for low-count stats (3PM, STL, BLK) using Poisson distribution.
        
        Key differences from high-count stats:
        1. Uses Poisson distribution (discrete, non-negative)
        2. Caps the MEAN itself to realistic ranges (no one averages 6+ steals)
        3. Applies strict output caps to prevent impossible single-game values
        4. Adjusts lambda based on features while respecting stat reality
        """
        # CRITICAL: Cap the mean to realistic ranges FIRST
        # This prevents projections like "6.3 steals" which are impossible
        mean_cap = self.config.LOW_COUNT_MEAN_CAPS.get(market, 5.0)
        capped_mean = min(mean, mean_cap)
        
        # Start with the capped mean (lambda for Poisson)
        adjusted_lambda = max(0.1, capped_mean)
        
        # Apply feature adjustments if available (more conservative for low-count)
        if features is not None:
            # Trend adjustment (halved for low-count to prevent overreaction)
            if abs(features.trend) > 0.05:
                trend_adj = features.trend * capped_mean * 0.01  # Only 1% vs 2% for high-count
                adjusted_lambda += max(-0.2, min(0.2, trend_adj))
            
            # Matchup adjustment (capped more tightly)
            matchup_effect = (features.combined_matchup_mult - 1.0) * capped_mean * 0.3
            adjusted_lambda += max(-0.3, min(0.3, matchup_effect))
            
            # Blowout penalty (low-count stats suffer more from reduced minutes)
            if features.blowout_prob >= 0.30:
                adjusted_lambda *= 0.80
            elif features.blowout_prob >= 0.20:
                adjusted_lambda *= 0.90
            
            # B2B/fatigue penalty
            if features.is_b2b:
                adjusted_lambda *= 0.92
        
        # Ensure lambda is positive and doesn't exceed the cap
        adjusted_lambda = max(0.1, min(adjusted_lambda, mean_cap))
        
        # =================================================================
        # CRITICAL FIX: Add uncertainty to Poisson to prevent hallucinations
        # Pure Poisson with lambda=0.5 vs line=2.0 gives ~91% UNDER
        # But in reality, role players have high game-to-game variance
        # =================================================================
        # Add lambda uncertainty: lambda varies game-to-game
        # This models that a player averaging 0.5 AST might have λ=0.3 or λ=1.0 on any given night
        lambda_uncertainty = max(0.3, adjusted_lambda * 0.5)  # 50% uncertainty, min 0.3
        varied_lambdas = np.random.normal(adjusted_lambda, lambda_uncertainty, n_samples)
        varied_lambdas = np.maximum(0.1, varied_lambdas)  # Keep positive
        
        # Generate Poisson samples with varied lambdas
        sims = np.array([np.random.poisson(lam) for lam in varied_lambdas]).astype(float)
        
        # Apply output caps (single-game maximums)
        # Cap 1: Relative to capped mean (prevent 2x+ blowups)
        relative_cap = capped_mean * self.config.LOW_COUNT_MAX_MULTIPLIER
        
        # Cap 2: Absolute cap based on stat type (real-world single-game limits)
        absolute_cap = self.config.LOW_COUNT_ABSOLUTE_CAPS.get(market, 8)
        
        # Use the more restrictive cap
        final_cap = min(relative_cap, absolute_cap)
        
        # Apply cap
        sims = np.minimum(sims, final_cap)
        
        return sims
    
    def _generate_mixture_samples(
        self, 
        mean: float, 
        std: float, 
        n_samples: int
    ) -> np.ndarray:
        """
        Legacy: Generate samples from static 3-component mixture distribution.
        Used when features are not provided.
        """
        blowup_prob = self.config.MIXTURE_BLOWUP_PROB
        dud_prob = self.config.MIXTURE_DUD_PROB
        normal_prob = 1.0 - blowup_prob - dud_prob
        
        component = np.random.choice(
            ['normal', 'blowup', 'dud'],
            size=n_samples,
            p=[normal_prob, blowup_prob, dud_prob]
        )
        
        adjusted_std = max(0.1, std) * 1.10
        sims = np.zeros(n_samples)
        
        # Normal games
        normal_mask = component == 'normal'
        sims[normal_mask] = np.random.normal(mean, adjusted_std, normal_mask.sum())
        
        # Blow-up games
        blowup_mask = component == 'blowup'
        blowup_mean = mean * self.config.MIXTURE_BLOWUP_MULT
        blowup_std = adjusted_std * 1.3
        sims[blowup_mask] = np.random.normal(blowup_mean, blowup_std, blowup_mask.sum())
        
        # Dud games
        dud_mask = component == 'dud'
        dud_mean = mean * self.config.MIXTURE_DUD_MULT
        dud_std = adjusted_std * 0.7
        sims[dud_mask] = np.random.normal(dud_mean, dud_std, dud_mask.sum())
        
        return sims


# =============================================================================
# LAYER 5: DECISION POLICY
# =============================================================================

class DecisionPolicy:
    """
    Applies Kelly Criterion and assigns grades to bets.
    Makes final bet recommendations. 
    """
    
    def __init__(self, config: Config = CONFIG):
        self.config = config
    
    def calculate_ev(self, prob: float, odds: float) -> float:
        """Calculate expected value."""
        return (prob * (odds - 1)) - (1 - prob)
    
    def calculate_flat_stake(
        self, 
        grade: BetGrade,
        bankroll: float
    ) -> Tuple[float, float]:
        """
        Calculate flat stake based on bet grade.
        Professional approach: consistent position sizing by confidence.
        """
        # Grade-based stake percentages
        stake_map = {
            BetGrade.A: self.config.STAKE_GRADE_A,  # 5%
            BetGrade.B: self.config.STAKE_GRADE_B,  # 3%
            BetGrade.C: self.config.STAKE_GRADE_C,  # 1%
            BetGrade.D: 0.0,  # No bet
            BetGrade.F: 0.0,  # No bet
        }
        
        stake_pct = stake_map.get(grade, 0.0)
        return stake_pct * bankroll, stake_pct

    # --- HELPER ALIAS FOR COMPATIBILITY ---
    def calculate_stake(self, ev: float, prob: float, bankroll: float, grade: BetGrade) -> Tuple[float, float]:
        """Wrapper to allow calling calculate_flat_stake with more args."""
        return self.calculate_flat_stake(grade, bankroll)
    
    def assign_grade(self, ev: float, win_prob: float) -> BetGrade: 
        """
        Assign letter grade based on EV and Probability.
        """
        # Logic for "S-Tier" / "A+" bets (High EV + High Prob)
        if ev > 0.10 and win_prob > 0.60:
            return BetGrade.A
        elif ev > 0.05:
            return BetGrade.A
        elif ev > 0.02:
            return BetGrade.B
        elif ev > 0:
            return BetGrade.C
        elif ev > -0.05:
            return BetGrade.D
        else: 
            return BetGrade.F
    
    def make_decision(
        self, 
        projection: Projection, 
        features: FeatureVector, 
        simulation: SimulationResult, 
        line: float, 
        odds: float = 1.91, 
        bankroll: float = 10000
    ) -> BetDecision:
        """
        Determines bet side, stake, and grade based on EV and ML validation.
        FIXED: Removed 'confidence_score' to fix TypeError.
        FIXED: Now properly calculates rollover_score.
        """
        # 1. Use Simulation Probabilities (The "Smart" Source)
        prob_over = simulation.over_prob
        prob_under = simulation.under_prob

        # 2. Calculate EV using actual odds
        decimal_odds = odds

        ev_over = self.calculate_ev(prob_over, decimal_odds)
        ev_under = self.calculate_ev(prob_under, decimal_odds)

        # 3. Determine Recommended Side
        if ev_over > ev_under:
            side = "OVER"
            ev = ev_over
            prob = prob_over
        else:
            side = "UNDER"
            ev = ev_under
            prob = prob_under

        # 4. Assign Grade (A, B, C, D, F)
        grade = self.assign_grade(ev, prob)

        # 5. Calculate Stake
        stake, fraction = self.calculate_flat_stake(grade, bankroll)

        # 6. Generate Warnings
        confidence_warning = self._generate_confidence_warning(features, prob, projection, side)
        
        # 7. Calculate Rollover/Parlay Score (0-5 scale)
        rollover_score = self._calculate_rollover_score(features, prob, ev, grade)
        
        # 8. Determine Rollover Suitability
        rollover_suitable = rollover_score >= 3.0

        return BetDecision(
            recommended_side=side,
            probability=prob,
            expected_value=ev,
            kelly_stake=stake,
            kelly_fraction=fraction,
            grade=grade,
            confidence_warning=confidence_warning,
            rollover_suitable=rollover_suitable,
            reasons_good=[], 
            reasons_bad=[],
            rollover_score=rollover_score
        )
    
    def _calculate_rollover_score(
        self,
        features: FeatureVector,
        probability: float,
        ev: float,
        grade: BetGrade
    ) -> float:
        """
        Calculate parlay/rollover suitability score (0-5 scale).
        
        Higher scores = better for parlays. Considers:
        1. Win probability (higher = better)
        2. Player consistency (lower CV = better)
        3. Sample size (more games = more reliable)
        4. Hit rate history (consistent hitter = better)
        5. Blowout risk (lower = better for parlays)
        """
        score = 0.0
        
        # --- Component 1: Probability (0-1.5 points) ---
        # Sweet spot is 60-70%, above 70% is great
        if probability >= 0.70:
            score += 1.5
        elif probability >= 0.65:
            score += 1.25
        elif probability >= 0.60:
            score += 1.0
        elif probability >= 0.55:
            score += 0.5
        # Below 55% gets 0
        
        # --- Component 2: Consistency/CV (0-1.0 points) ---
        # Lower CV = more predictable = better for parlays
        cv = features.coef_variation
        if cv <= 0.15:
            score += 1.0  # Very consistent
        elif cv <= 0.20:
            score += 0.75
        elif cv <= 0.25:
            score += 0.5
        elif cv <= 0.30:
            score += 0.25
        # Above 30% CV = too volatile for parlays
        
        # --- Component 3: Sample Size (0-0.75 points) ---
        games = features.games_played
        if games >= 20:
            score += 0.75
        elif games >= 15:
            score += 0.5
        elif games >= 10:
            score += 0.25
        # Below 10 games = insufficient data
        
        # --- Component 4: Hit Rate History (0-1.0 points) ---
        # Use weighted hit rate (recency-weighted)
        hit_rate = features.hit_rate_weighted
        if hit_rate >= 0.70:
            score += 1.0
        elif hit_rate >= 0.60:
            score += 0.75
        elif hit_rate >= 0.50:
            score += 0.5
        elif hit_rate >= 0.40:
            score += 0.25
        
        # --- Component 5: Blowout Risk Penalty (0 to -0.75 points) ---
        # High blowout risk = dangerous for parlays
        if features.blowout_prob >= 0.35:
            score -= 0.75
        elif features.blowout_prob >= 0.25:
            score -= 0.5
        elif features.blowout_prob >= 0.15:
            score -= 0.25
        
        # --- Bonus: Grade A gets a boost ---
        if grade == BetGrade.A:
            score += 0.5
        elif grade == BetGrade.B:
            score += 0.25
        
        # Clamp to 0-5 range
        return max(0.0, min(5.0, score))

    def _generate_confidence_warning(
        self, 
        features: FeatureVector, 
        probability: float, 
        projection: Projection = None,
        side: str = "OVER"
    ) -> Optional[str]:
        """
        Generate warning message for low-confidence or conflicting predictions.
        """
        warnings = []
        
        # 1. Sample Size Warning
        if features.games_played < 10:
            warnings.append(f"⚠️ Low sample ({features.games_played} gms)")
        
        # 2. Volatility Warning
        if features.coef_variation > 0.35:
            warnings.append(f"⚠️ High Volatility (CV:{features.coef_variation:.0%})")
        elif features.coef_variation > 0.25:
            warnings.append(f"📊 Mod Volatility (CV:{features.coef_variation:.0%})")
        
        # 3. ML vs Sim Logic
        if projection and projection.ml_prob is not None:
            ml_prob = projection.ml_prob # Probability of WINNING the bet
            sim_prob = probability       # Probability of WINNING the bet
            
            # Scenario A: Civil War (Opposite Sides)
            # ML thinks we lose (< 45%), Sim thinks we win (> 50%)
            if ml_prob < 0.45:
                opp_side = "UNDER" if side == "OVER" else "OVER"
                warnings.append(f"⚠️ CONFLICT: Sim likes {side}, ML leans {opp_side}")
                
            # Scenario B: Confidence Gap (Same Side, Different Confidence)
            elif abs(sim_prob - ml_prob) > 0.15:
                warnings.append(
                    f"ℹ️ GAP: Sim {sim_prob:.0%} vs ML {ml_prob:.0%} (Both Like {side})"
                )

        return " | ".join(warnings) if warnings else None


# =============================================================================
# LAYER 6: BACKTESTER
# =============================================================================

class Backtester:
    """
    Walk-forward backtesting engine.
    Evaluates model performance without future data leakage.
    """
    
    def __init__(
        self,
        data_loader: DataLoader,
        feature_engineer: FeatureEngineer,
        model_engine: ModelEngine,
        simulation_engine: SimulationEngine,
        decision_policy: DecisionPolicy,
        config: Config = CONFIG
    ):
        self.data_loader = data_loader
        self. feature_engineer = feature_engineer
        self.model_engine = model_engine
        self.simulation_engine = simulation_engine
        self.decision_policy = decision_policy
        self.config = config
        self._rng = np.random.default_rng(42)  # Reproducible randomness for CLV
    
    def _generate_synthetic_clv(self, ev: float, prob: float, line: float, hit: bool) -> float:
        """
        Generate realistic synthetic CLV for backtest data.
        
        In real markets:
        - Sharp bets (high EV, high prob) see lines move toward them (+CLV)
        - Square/public bets see adverse movement (-CLV)
        - Winning bets correlate with positive CLV (market was right)
        - Random noise simulates market uncertainty
        
        Returns CLV in points (e.g., +0.5 means line moved 0.5 toward your bet)
        """
        # Base CLV from EV signal (sharp money moves lines)
        # EV of +5% → expect ~0.3 pts CLV, EV of -5% → expect ~-0.3 pts CLV
        ev_component = ev * 6.0  # Scale EV to reasonable CLV range
        
        # Probability component (high confidence → market likely agrees)
        prob_component = (prob - 0.5) * 1.0  # 60% prob → +0.1, 40% → -0.1
        
        # Outcome component (winners had sharper reads on average)
        outcome_component = 0.2 if hit else -0.1
        
        # Random market noise (-0.5 to +0.5 pts typical)
        noise = self._rng.normal(0, 0.3)
        
        # Combine components
        raw_clv = ev_component + prob_component + outcome_component + noise
        
        # Scale by line (bigger lines have bigger absolute movements)
        # A 25.5 PTS line moves more than a 1.5 STL line
        scale_factor = max(0.5, min(2.0, line / 15.0))
        
        # Clamp to realistic range (-3 to +3 pts for most props)
        clv = max(-3.0, min(3.0, raw_clv * scale_factor))
        
        return round(clv, 2)
    
    def run_backtest(
        self,
        player_id: int,
        player_name: str,
        market: str,
        lookback:  int = 15,
        test_days: int = 30,
        line_offset: float = 0.0,  # Test at actual average ± offset
        fixed_spread: float = 0.0,
        progress_callback=None,
        preloaded_df: pd.DataFrame = None  # Optional: use pre-loaded game logs
    ) -> Optional[BacktestSummary]:
        """
        Run walk-forward backtest.
        
        If preloaded_df is provided, uses that instead of fetching from API.
        For each day in the test period:
        1. Use only data available up to that day
        2. Generate prediction
        3. Compare to actual result
        """
        
        # Fetch all available data (or use preloaded)
        if preloaded_df is not None and len(preloaded_df) > 0:
            df = preloaded_df.copy()
        else:
            df = self.data_loader.fetch_multi_season_logs(player_id)
        # NEW FLEXIBLE LOGIC (Harvests all valid data)
        min_required = self.config.BACKTEST_MIN_GAMES
        
        # If they don't even have enough for features (e.g. < 10 games), skip them.
        if len(df) <= min_required:
            return None
            
        # Calculate how many games we can actually test
        available_test_games = len(df) - min_required
        
        # We want 'test_days' amount, but we will settle for what they have
        actual_test_days = min(test_days, available_test_games)
        
        if actual_test_days <= 0:
            return None
            
        # Adjust test start index dynamically
        total_games = len(df)
        test_start_idx = total_games - actual_test_days
        
        # Get player position
        player_position = self.data_loader.get_player_position(player_id)
        
        # Get team stats (use current - this is a simplification)
        team_stats, avg_pace, avg_def = self.data_loader.fetch_team_stats()
        
        # Pre-load position defense for both seasons to avoid repeated API calls
        position_defense_cache = {
            self.config.CURRENT_SEASON: self.data_loader.fetch_position_defense(self.config.CURRENT_SEASON),
            self.config.PREV_SEASON: self.data_loader.fetch_position_defense(self.config.PREV_SEASON)
        }
        
        # Sort by date
        df = df.sort_values('GAME_DATE').reset_index(drop=True)
        
        # NOTE: test_start_idx already calculated above using actual_test_days
        # (flexible logic that uses available games even if less than desired)
        
        results = []
        
        for i in range(test_start_idx, total_games):
            if progress_callback:
                progress_callback((i - test_start_idx) / (total_games - test_start_idx))
            
            # Data available up to (but not including) this game
            historical_df = df. iloc[: i]. copy()
            
            # The game we're predicting
            target_game = df.iloc[i]
            
            if len(historical_df) < self.config. BACKTEST_MIN_GAMES: 
                continue
            
            # Determine line (use rolling average as proxy)
            rolling_avg = historical_df[market].tail(lookback).mean()
            line = rolling_avg + line_offset
            
            # Extract game context
            matchup = target_game. get('MATCHUP', '')
            is_home = 'vs.' in matchup
            
            # Determine opponent (simplified - extract from matchup string)
            if 'vs.' in matchup:
                opp_abbrev = matchup.split('vs.')[-1].strip()
            elif '@' in matchup: 
                opp_abbrev = matchup.split('@')[-1].strip()
            else:
                opp_abbrev = 'UNK'
            
            # Check if B2B and calculate days_rest robustly
            if i > 0:
                try:
                    prev_date = pd.to_datetime(df.iloc[i-1]['GAME_DATE'])
                    curr_date = pd.to_datetime(target_game['GAME_DATE'])
                    days_rest = max(0, (curr_date - prev_date).days - 1)
                    is_b2b = days_rest == 0
                except Exception:
                    days_rest = 1  # Default to 1 day rest
                    is_b2b = False
            else:
                days_rest = 3  # First game of dataset, assume well-rested
                is_b2b = False
            
            # Determine which season this game belongs to for position defense
            try:
                game_date = pd.to_datetime(target_game['GAME_DATE'])
                # NBA season runs Oct-Jun. If month >= 10 (Oct), it's the new season year.
                # e.g., Oct 2024 = "2024-25" season, Jan 2025 = still "2024-25"
                if game_date.month >= 10:
                    game_season = f"{game_date.year}-{str(game_date.year + 1)[-2:]}"
                else:
                    game_season = f"{game_date.year - 1}-{str(game_date.year)[-2:]}"
                
                # Use cached position defense for this season
                position_defense = position_defense_cache.get(
                    game_season, 
                    position_defense_cache.get(self.config.CURRENT_SEASON, {})
                )
            except Exception:
                position_defense = position_defense_cache.get(self.config.CURRENT_SEASON, {})
            
            # Find opponent ID (if possible)
            all_teams = teams. get_teams()
            opp_team = next((t for t in all_teams if t['abbreviation'] == opp_abbrev), None)
            opponent_id = int(opp_team['id']) if opp_team else 0
            
            # =========================================================================
            # SYNTHETIC VEGAS LINES CALCULATION
            # Since historical Vegas odds aren't available via API, we calculate
            # synthetic spread and game total using team Net Rating and Pace.
            # =========================================================================
            try:
                # Get player's team ID from the game log
                player_team_id = target_game.get('Team_ID', None)
                
                # Fallback: try to infer from matchup string
                if player_team_id is None:
                    matchup_str = target_game.get('MATCHUP', '')
                    if 'vs.' in matchup_str:
                        player_team_abbrev = matchup_str.split('vs.')[0].strip().split()[-1]
                    elif '@' in matchup_str:
                        player_team_abbrev = matchup_str.split('@')[0].strip().split()[-1]
                    else:
                        player_team_abbrev = None
                    
                    if player_team_abbrev:
                        player_team_match = next(
                            (t for t in all_teams if t['abbreviation'] == player_team_abbrev), 
                            None
                        )
                        player_team_id = int(player_team_match['id']) if player_team_match else None
                
                # Force integer types for all IDs to prevent type mismatch KeyErrors
                if player_team_id is not None:
                    player_team_id = int(player_team_id)
                opponent_id = int(opponent_id) if opponent_id else 0
                
                # Ensure team_stats index is integer type for consistent lookups
                if len(team_stats) > 0 and team_stats.index.dtype != 'int64':
                    team_stats.index = team_stats.index.astype(int)
                
                # Calculate synthetic spread and total if we have team stats
                HOME_ADVANTAGE = 3.0  # Standard home court advantage in NBA
                
                if (player_team_id is not None and 
                    opponent_id != 0 and 
                    len(team_stats) > 0 and
                    player_team_id in team_stats.index and 
                    opponent_id in team_stats.index):
                    
                    # Get team statistics
                    player_team_stats = team_stats.loc[player_team_id]
                    opp_team_stats = team_stats.loc[opponent_id]
                    
                    # Extract Net Rating (OFF_RATING - DEF_RATING if NET_RATING not available)
                    if 'NET_RATING' in team_stats.columns:
                        player_net_rtg = player_team_stats['NET_RATING']
                        opp_net_rtg = opp_team_stats['NET_RATING']
                    else:
                        # Calculate from OFF and DEF ratings if available
                        player_off = player_team_stats.get('OFF_RATING', 110.0)
                        player_def = player_team_stats.get('DEF_RATING', 110.0)
                        opp_off = opp_team_stats.get('OFF_RATING', 110.0)
                        opp_def = opp_team_stats.get('DEF_RATING', 110.0)
                        player_net_rtg = player_off - player_def
                        opp_net_rtg = opp_off - opp_def
                    
                    # Extract Pace
                    player_pace = player_team_stats.get('PACE', avg_pace)
                    opp_pace = opp_team_stats.get('PACE', avg_pace)
                    
                    # Calculate Synthetic Spread
                    # Spread = -((PlayerTeam_NetRtg - OppTeam_NetRtg) + HomeAdvantage)
                    # Negative spread means player's team is favored
                    net_rtg_diff = player_net_rtg - opp_net_rtg
                    home_adj = HOME_ADVANTAGE if is_home else -HOME_ADVANTAGE
                    synthetic_spread = -((net_rtg_diff + home_adj) / 2.5)  # Scale to points
                    
                    # Calculate Synthetic Total
                    # Total ≈ (PlayerTeam_Pace + OppTeam_Pace) * 1.15
                    # 1.15 is a rough scaling factor to convert pace to expected points
                    synthetic_total = (player_pace + opp_pace) * 1.15
                    
                    # Clamp to reasonable ranges
                    synthetic_spread = max(-25.0, min(25.0, synthetic_spread))
                    synthetic_total = max(200.0, min(260.0, synthetic_total))
                    
                else:
                    # Fallback if team stats are missing
                    synthetic_spread = 0.0
                    synthetic_total = 225.0
                    
            except Exception as e:
                print(f"[DEBUG] Synthetic Vegas calculation failed: {e}")
                logger.debug(f"Synthetic Vegas calculation failed: {e}")
                synthetic_spread = 0.0
                synthetic_total = 225.0
            
            # Use fixed_spread if provided (non-zero), otherwise use synthetic
            backtest_spread = fixed_spread if fixed_spread != 0.0 else synthetic_spread
            backtest_game_total = synthetic_total
            
            try:
                # Build feature vector using historical data only
                features = self.feature_engineer. build_feature_vector(
                    player_id=player_id,
                    player_name=player_name,
                    player_position=player_position,
                    opponent_id=opponent_id,
                    opponent_abbrev=opp_abbrev,
                    is_home=is_home,
                    is_b2b=is_b2b,
                    spread=backtest_spread,  # Synthetic or user-defined spread
                    usage_mult=1.0,
                    df=historical_df,
                    stat_col=market,
                    line=line,
                    lookback=lookback,
                    team_stats=team_stats,
                    position_defense=position_defense,
                    avg_pace=avg_pace,
                    avg_def=avg_def,
                    market=market,
                    days_rest=days_rest,  # Properly calculated rest days
                    game_total=backtest_game_total  # Synthetic game total
                )
                
                # Generate projection
                projection = self.model_engine. generate_projection(features)
                
                # Run simulation (with features for smart mixture model)
                simulation = self.simulation_engine.run_simulation(
                    mean=projection.final_projection,
                    std=features.std,
                    line=line,
                    market=market,
                    features=features  # Enable smart simulation
                )
                
                # Make decision
                decision = self.decision_policy.make_decision(
                    features=features,
                    projection=projection,
                    simulation=simulation,
                    line=line,
                    odds=1.91,  # Standard -110 odds
                    bankroll=1000
                )
                
                # Get actual result
                actual_value = target_game[market]
                hit = (decision.recommended_side == "OVER" and actual_value > line) or \
                      (decision.recommended_side == "UNDER" and actual_value <= line)
                
                # Extract features for ML training
                # MUST MATCH FeatureVector.to_ml_array() EXACTLY in content and order
                feature_dict = {
                    'ema': features.ema,
                    'std': features.std,
                    'sma_5': features.sma_5,
                    'sma_10': features.sma_10,
                    'trend': features.trend,
                    'avg_minutes': features.avg_minutes,
                    'mins_trend': features.mins_trend,
                    # Use shortened names to match tracker export format
                    'hit_l5': features.hit_rate_l5,
                    'hit_l10': features.hit_rate_l10,
                    'hit_l15': features.hit_rate_l15,
                    'hit_season': features.hit_rate_season,
                    'pace_mult': features.pace_mult,
                    'def_mult': features.def_mult,
                    'position_mult': features.position_mult,
                    'base_matchup_mult': features.base_matchup_mult,
                    'combined_matchup_mult': features.combined_matchup_mult,
                    'split_factor': features.split_factor,
                    'rest_factor': features.rest_factor,
                    'blowout_factor': features.blowout_factor,
                    'usage_mult': features.usage_mult,
                    'is_home': int(features.is_home),
                    'is_b2b': int(features.is_b2b),
                    'spread': features.spread,
                    'games_played': features.games_played,
                    'days_rest': features.days_rest,
                    'game_total': features.game_total,
                    # Use shortened names to match tracker export format
                    'opp_drtg_season': features.opponent_drtg_season,
                    'opp_drtg_l5': features.opponent_drtg_l5,
                    # V15 NEW FEATURES
                    'blowout_prob': features.blowout_prob,
                    'personal_fatigue_factor': features.personal_fatigue_factor,
                    'b2b_games_in_sample': features.b2b_games_in_sample,
                    'dynamic_std_mult': features.dynamic_std_mult,
                    'coef_variation': features.coef_variation,
                    # V16 MARKET & ROLE CONTEXT
                    'odds_decimal': 1.91,  # -110 odds in backtest
                    'usg_season': features.usg_season,
                    # Synthetic CLV: Simulate realistic line movement
                    # Sharp bets (high EV, high prob) tend to see favorable CLV
                    # Add noise to simulate market uncertainty
                    'clv': self._generate_synthetic_clv(
                        ev=decision.expected_value,
                        prob=decision.probability,
                        line=line,
                        hit=hit
                    )
                }
                
                results.append(BacktestResult(
                    date=target_game['GAME_DATE']. strftime('%Y-%m-%d'),
                    player_name=player_name,
                    market=market,
                    line=line,
                    predicted_side=decision.recommended_side,
                    predicted_prob=decision.probability,
                    predicted_ev=decision. expected_value,
                    actual_value=actual_value,
                    hit=hit,
                    grade=decision.grade. value,
                    opponent=opp_abbrev,
                    position=player_position,
                    is_home=is_home,
                    features=feature_dict
                ))
                
            except Exception as e:
                logger. warning(f"Backtest error for game {i}: {e}")
                continue
        
        if not results:
            return None
        
        return self._calculate_summary(results)
    
    def _calculate_summary(self, results: List[BacktestResult]) -> BacktestSummary:
        """Calculate backtest summary metrics."""
        
        total = len(results)
        wins = sum(1 for r in results if r.hit)
        losses = total - wins
        win_rate = wins / total if total > 0 else 0
        
        # ROI calculation (assuming flat betting at -110 odds)
        odds = 1.91
        roi = (wins * (odds - 1) - losses) / total if total > 0 else 0
        
        # Brier Score (calibration metric)
        brier_sum = 0
        for r in results:
            outcome = 1.0 if r. hit else 0.0
            brier_sum += (r.predicted_prob - outcome) ** 2
        brier_score = brier_sum / total if total > 0 else 1.0
        
        # Calibration by probability bucket
        calibration = {}
        buckets = [(0.5, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 1.0)]
        for low, high in buckets:
            bucket_results = [r for r in results if low <= r.predicted_prob < high]
            if bucket_results: 
                bucket_wins = sum(1 for r in bucket_results if r.hit)
                bucket_total = len(bucket_results)
                calibration[f"{int(low*100)}-{int(high*100)}%"] = {
                    'predicted':  (low + high) / 2,
                    'actual': bucket_wins / bucket_total,
                    'count': bucket_total
                }
        
        # Performance by grade
        grade_performance = {}
        for grade in ['A', 'B', 'C', 'D', 'F']:
            grade_results = [r for r in results if r. grade == grade]
            if grade_results: 
                grade_wins = sum(1 for r in grade_results if r.hit)
                grade_total = len(grade_results)
                grade_roi = (grade_wins * (odds - 1) - (grade_total - grade_wins)) / grade_total
                grade_performance[grade] = {
                    'count': grade_total,
                    'win_rate': grade_wins / grade_total,
                    'roi': grade_roi
                }
        
        # Create results dataframe with flattened features for ML
        # MUST MATCH TRACKER EXPORT SCHEMA EXACTLY
        rows = []
        for r in results:
            # Calculate margin and margin_pct
            if r.predicted_side == 'OVER':
                margin = r.actual_value - r.line
            else:
                margin = r.line - r.actual_value
            
            margin_pct = ((r.actual_value - r.line) / r.line * 100) if r.line > 0 else 0
            if r.predicted_side == 'UNDER':
                margin_pct = -margin_pct
            
            # Determine result quality based on margin
            abs_margin = abs(margin)
            if r.hit:
                if abs_margin <= 1.5:
                    result_quality = 'sweat_win'
                elif abs_margin <= 3.5:
                    result_quality = 'close_win'
                elif abs_margin <= 7.5:
                    result_quality = 'solid_win'
                else:
                    result_quality = 'blowout_win'
            else:
                if abs_margin <= 1.5:
                    result_quality = 'bad_beat'
                elif abs_margin <= 3.5:
                    result_quality = 'close_loss'
                elif abs_margin <= 7.5:
                    result_quality = 'clear_loss'
                else:
                    result_quality = 'bad_read'
            
            row = {
                # Match tracker export column order exactly
                'date': r.date,
                'player': r.player_name,
                'opponent': r.opponent,  # Now populated from BacktestResult
                'market': r.market,
                'position': r.position,  # Now populated from BacktestResult
                'line': r.line,
                'predicted_side': r.predicted_side,
                'predicted_prob': r.predicted_prob,
                'predicted_ev': r.predicted_ev,
                'projected_value': r.features.get('ema', 0),
                'result': 'Win' if r.hit else 'Loss',
                'hit': 1 if r.hit else 0,
                'actual_value': r.actual_value,
                'margin': margin,
                'margin_pct': margin_pct,
                'result_quality': result_quality,
                # Snapshot fields (use feature values as point-in-time snapshots)
                'snapshot_l5_hit_rate': r.features.get('hit_l5', 0),
                'snapshot_days_rest': r.features.get('days_rest', 1),
                'snapshot_def_rank': int(r.features.get('opp_drtg_season', 115)),
                'tag': 'backtest',  # Tag for filtering
                'grade': r.grade,
            }
            # Flatten features with 'feat_' prefix
            for feat_name, feat_value in r.features.items():
                row[f'feat_{feat_name}'] = feat_value
            # Add missing tracker columns with defaults
            row['closing_line'] = 0
            row['clv'] = 0
            # feat_usg_season and feat_clv already set from features dict above
            row['feat_usg_trend'] = row.get('feat_usage_mult', 1.0)
            rows.append(row)
        
        results_df = pd.DataFrame(rows)
        
        return BacktestSummary(
            total_predictions=total,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            roi=roi,
            brier_score=brier_score,
            calibration_by_bucket=calibration,
            grade_performance=grade_performance,
            results_df=results_df
        )


# =============================================================================
# ML DATA GENERATOR
# =============================================================================

# G-League team IDs start at this threshold (NBA teams are 1610612737-1610612766)
GLEAGUE_TEAM_ID_THRESHOLD = 1610612900

# NBA Division to Team mapping (team abbreviations)
NBA_DIVISIONS = {
    'Atlantic': ['BOS', 'BKN', 'NYK', 'PHI', 'TOR'],
    'Central': ['CHI', 'CLE', 'DET', 'IND', 'MIL'],
    'Southeast': ['ATL', 'CHA', 'MIA', 'ORL', 'WAS'],
    'Northwest': ['DEN', 'MIN', 'OKC', 'POR', 'UTA'],
    'Pacific': ['GSW', 'LAC', 'LAL', 'PHX', 'SAC'],
    'Southwest': ['DAL', 'HOU', 'MEM', 'NOP', 'SAS']
}

# Canonical feature list used for model training (exact order, 36 numeric features)
TRAINING_FEATURE_COLUMNS = [
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
    # Market & Role Context (V16 additions)
    'feat_odds_decimal', 'feat_usg_season', 'feat_clv'
]

def get_top_active_players(limit: int = 50, min_games: int = 10) -> List[Dict]:
    """
    Get top active NBA players by games played this season.
    
    OPTIMIZED: Uses leaguedashplayerstats endpoint for a single API call
    instead of fetching individual game logs (100x faster).
    
    Excludes G-League players by checking team ID.
    Returns list of player dicts with 'id', 'full_name', 'team', and 'division'.
    """
    logger.info(f"Fetching top {limit} active players (min {min_games} GP, excluding G-League)...")
    
    try:
        # Single API call to get ALL active players with their stats
        time.sleep(CONFIG.API_DELAY)
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=CONFIG.CURRENT_SEASON,
            per_mode_detailed='PerGame',
            timeout=60
        )
        df = stats.get_data_frames()[0]
        
        if df is None or len(df) == 0:
            logger.warning("leaguedashplayerstats returned no data")
            return []
        
        logger.info(f"  ✓ Fetched {len(df)} players from NBA API in 1 request")
        
    except Exception as e:
        logger.error(f"Failed to fetch player stats: {e}")
        return []
    
    # Build mappings
    all_teams = teams.get_teams()
    team_id_to_abbrev = {t['id']: t['abbreviation'] for t in all_teams}
    
    abbrev_to_division = {}
    for division, team_abbrevs in NBA_DIVISIONS.items():
        for abbrev in team_abbrevs:
            abbrev_to_division[abbrev] = division
    
    # Filter and process
    player_list = []
    division_counts = {div: 0 for div in NBA_DIVISIONS.keys()}
    
    for _, row in df.iterrows():
        player_id = row['PLAYER_ID']
        player_name = row['PLAYER_NAME']
        team_id = row['TEAM_ID']
        games_played = row['GP']
        
        # Skip players with too few games
        if games_played < min_games:
            continue
        
        # Skip G-League players
        if team_id > GLEAGUE_TEAM_ID_THRESHOLD:
            continue
        
        # Get team abbreviation and division
        team_abbrev = team_id_to_abbrev.get(team_id)
        if not team_abbrev:
            continue
        
        division = abbrev_to_division.get(team_abbrev)
        if not division:
            continue  # Not an NBA team we recognize
        
        player_list.append({
            'id': player_id,
            'full_name': player_name,
            'games': games_played,
            'team': team_abbrev,
            'division': division
        })
        division_counts[division] += 1
    
    # Log division breakdown
    for division in NBA_DIVISIONS.keys():
        logger.info(f"  ✓ {division}: {division_counts[division]} players")
    
    # Sort by games played (descending) and return top N
    player_list.sort(key=lambda x: x['games'], reverse=True)
    total_found = len(player_list)
    logger.info(f"Found {total_found} NBA players, returning top {min(limit, total_found)}")
    
    return player_list[:limit]


class BulkGameLogLoader:
    """
    Downloads ALL player game logs for the entire league in 2 API calls.
    Enables offline backtesting without per-player API calls.
    """
    
    def __init__(self, config: Config = CONFIG):
        self.config = config
        self._cache: Dict[int, pd.DataFrame] = {}  # player_id -> game_logs
        self._loaded = False
    
    def load_all_game_logs(self, progress_callback=None) -> bool:
        """
        Fetch all game logs for current + previous season in 2 API calls.
        Returns True if successful, False otherwise.
        """
        if self._loaded:
            return True
        
        all_logs = []
        seasons = [self.config.CURRENT_SEASON, self.config.PREV_SEASON]
        
        for i, season in enumerate(seasons):
            try:
                logger.info(f"📥 Fetching ALL game logs for {season} (API call {i+1}/2)...")
                time.sleep(self.config.API_DELAY * 2)  # Extra delay for large request
                
                logs = leaguegamelog.LeagueGameLog(
                    season=season,
                    player_or_team_abbreviation='P',  # P = Players
                    timeout=120  # Large request needs more time
                )
                df = logs.get_data_frames()[0]
                
                if df is not None and len(df) > 0:
                    df['SEASON'] = season
                    all_logs.append(df)
                    logger.info(f"  ✓ {season}: {len(df):,} game log entries")
                else:
                    logger.warning(f"  ✗ {season}: No data returned")
                    
                if progress_callback:
                    progress_callback((i + 1) / len(seasons) * 0.3)  # 30% for loading
                    
            except Exception as e:
                logger.error(f"  ✗ Failed to fetch {season} logs: {e}")
                # If current season fails, we can still use previous season
                if season == self.config.CURRENT_SEASON:
                    return False
        
        if not all_logs:
            logger.error("No game logs fetched from either season")
            return False
        
        # Combine all seasons
        combined = pd.concat(all_logs, ignore_index=True)
        logger.info(f"📊 Total: {len(combined):,} game log entries across {len(all_logs)} seasons")
        
        # Parse and organize by player
        self._organize_by_player(combined)
        self._loaded = True
        
        return True
    
    def _organize_by_player(self, df: pd.DataFrame):
        """Organize the bulk data by player ID for fast lookup."""
        # Convert column names to match playergamelog format
        # leaguegamelog returns slightly different column names
        column_map = {
            'PLAYER_ID': 'PLAYER_ID',
            'PLAYER_NAME': 'PLAYER_NAME',
            'TEAM_ID': 'TEAM_ID',
            'TEAM_ABBREVIATION': 'TEAM_ABBREVIATION',
            'TEAM_NAME': 'TEAM_NAME',
            'GAME_ID': 'GAME_ID',
            'GAME_DATE': 'GAME_DATE',
            'MATCHUP': 'MATCHUP',
            'WL': 'WL',
            'MIN': 'MIN',
            'FGM': 'FGM',
            'FGA': 'FGA',
            'FG_PCT': 'FG_PCT',
            'FG3M': 'FG3M',
            'FG3A': 'FG3A',
            'FG3_PCT': 'FG3_PCT',
            'FTM': 'FTM',
            'FTA': 'FTA',
            'FT_PCT': 'FT_PCT',
            'OREB': 'OREB',
            'DREB': 'DREB',
            'REB': 'REB',
            'AST': 'AST',
            'STL': 'STL',
            'BLK': 'BLK',
            'TOV': 'TOV',
            'PF': 'PF',
            'PTS': 'PTS',
            'PLUS_MINUS': 'PLUS_MINUS'
        }
        
        # Only keep columns that exist
        existing_cols = [c for c in column_map.keys() if c in df.columns]
        df = df[existing_cols].copy()
        
        # Add derived columns
        if 'PTS' in df.columns and 'REB' in df.columns and 'AST' in df.columns:
            df['PRA'] = df['PTS'] + df['REB'] + df['AST']
        if 'PTS' in df.columns and 'REB' in df.columns:
            df['PR'] = df['PTS'] + df['REB']
        if 'PTS' in df.columns and 'AST' in df.columns:
            df['PA'] = df['PTS'] + df['AST']
        if 'REB' in df.columns and 'AST' in df.columns:
            df['RA'] = df['REB'] + df['AST']
        if 'FG3M' in df.columns:
            df['3PM'] = df['FG3M']
        
        # Parse MIN to float
        if 'MIN' in df.columns:
            df['MIN_FLOAT'] = pd.to_numeric(df['MIN'], errors='coerce').fillna(0)
        
        # Derive IS_HOME from MATCHUP column
        # "MIL vs. CHI" = home game (contains "vs.")
        # "MIL @ CHI" = away game (contains "@")
        if 'MATCHUP' in df.columns:
            df['IS_HOME'] = df['MATCHUP'].str.contains('vs.', case=False, na=False)
        else:
            df['IS_HOME'] = True  # Default to home if no matchup info
        
        # Sort by date (most recent first)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df.sort_values('GAME_DATE', ascending=False)
        
        # Group by player
        for player_id, player_df in df.groupby('PLAYER_ID'):
            self._cache[player_id] = player_df.reset_index(drop=True)
        
        logger.info(f"  ✓ Organized data for {len(self._cache):,} unique players")
    
    def get_player_logs(self, player_id: int) -> pd.DataFrame:
        """Get game logs for a specific player from cache."""
        if not self._loaded:
            return pd.DataFrame()
        return self._cache.get(player_id, pd.DataFrame())
    
    def has_player(self, player_id: int) -> bool:
        """Check if we have data for a player."""
        return player_id in self._cache
    
    @property
    def player_count(self) -> int:
        """Number of players in cache."""
        return len(self._cache)
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded


def generate_ml_training_data(
    output_file: str = "ml_training_data.csv",
    num_players: int = 50,
    markets: List[str] = None,
    test_days: int = 60,
    lookback: int = 15,
    progress_callback=None
) -> pd.DataFrame:
    """
    Generate ML training dataset by running backtests on multiple players.
    
    OPTIMIZED: Uses bulk data loading (2 API calls total) instead of per-player calls.
    """
    if markets is None:
        markets = ['PTS', 'REB', 'AST', 'PRA']
    
    logger.info(f"Starting ML data generation for {num_players} players...")
    logger.info(f"Markets: {markets}")
    
    config = CONFIG
    data_loader = DataLoader(config)
    feature_engineer = FeatureEngineer(config)
    model_engine = ModelEngine(config)
    simulation_engine = SimulationEngine(config)
    decision_policy = DecisionPolicy(config)
    
    backtester = Backtester(data_loader, feature_engineer, model_engine, simulation_engine, decision_policy, config)
    
    # =========================================================================
    # STEP 1: Bulk load ALL game logs (2 API calls instead of 500+)
    # =========================================================================
    bulk_loader = BulkGameLogLoader(config)
    
    if progress_callback:
        progress_callback(0.05)  # 5%
    
    if not bulk_loader.load_all_game_logs(progress_callback):
        logger.error("🛑 Failed to bulk load game logs. Aborting.")
        return pd.DataFrame()
    
    logger.info(f"✅ Bulk data loaded: {bulk_loader.player_count:,} players available offline")
    
    if progress_callback:
        progress_callback(0.35)  # 35% done with loading
    
    # =========================================================================
    # STEP 2: Get top players (1 API call)
    # =========================================================================
    logger.info("Fetching top active players...")
    top_players = get_top_active_players(num_players)
    
    if progress_callback:
        progress_callback(0.40)  # 40%
    
    # =========================================================================
    # STEP 3: Run backtests using LOCAL data (no more API calls!)
    # =========================================================================
    all_results = []
    total_tasks = len(top_players) * len(markets)
    completed = 0
    skipped_no_data = 0
    
    logger.info(f"🎯 Starting offline backtests for {len(top_players)} players x {len(markets)} markets = {total_tasks} tasks")
    
    for player in top_players:
        player_id = player['id']
        player_name = player['full_name']
        
        # Get pre-loaded data for this player
        player_df = bulk_loader.get_player_logs(player_id)
        
        if len(player_df) < 15:  # Need minimum data
            skipped_no_data += 1
            completed += len(markets)
            if progress_callback:
                progress_callback(0.40 + 0.55 * (completed / total_tasks))
            continue
        
        for market in markets:
            try:
                summary = backtester.run_backtest(
                    player_id=player_id,
                    player_name=player_name,
                    market=market,
                    lookback=lookback,
                    test_days=test_days,
                    line_offset=0.0,
                    fixed_spread=0.0,
                    preloaded_df=player_df  # USE LOCAL DATA
                )
                
                if summary is not None and len(summary.results_df) > 0:
                    summary.results_df['player_id'] = player_id
                    all_results.append(summary.results_df)
            
            except Exception as e:
                logger.warning(f"  ✗ {player_name} - {market}: {e}")
            
            completed += 1
            if progress_callback:
                progress_callback(0.40 + 0.55 * (completed / total_tasks))
    
    logger.info(f"Skipped {skipped_no_data} players with insufficient data")

    if not all_results:
        if progress_callback:
            progress_callback(1.0)  # Complete the progress bar
        return pd.DataFrame()
    
    combined_df = pd.concat(all_results, ignore_index=True)
    output_path = DATA_DIR / output_file
    
    # TRAINING_FEATURE_COLUMNS is defined at module scope

    # Canonical column order - MUST MATCH TRACKER EXPORT EXACTLY
    canonical_columns = [
        'date', 'player', 'opponent', 'market', 'position',
        'line', 'predicted_side', 'predicted_prob', 'predicted_ev', 'projected_value',
        'result', 'hit', 'actual_value', 'margin', 'margin_pct', 'result_quality',
        # Snapshot: Point-in-time frozen stats (time capsule for Robot training)
        'snapshot_l5_hit_rate', 'snapshot_days_rest', 'snapshot_def_rank',
        # Tag: Bet source for filtering training data
        'tag',
        'feat_ema', 'feat_std', 'feat_sma_5', 'feat_sma_10', 'feat_trend',
        'feat_avg_minutes', 'feat_mins_trend',
        'feat_hit_l5', 'feat_hit_l10', 'feat_hit_l15', 'feat_hit_season',
        'feat_pace_mult', 'feat_def_mult', 'feat_position_mult',
        'feat_base_matchup_mult', 'feat_combined_matchup_mult',
        'feat_split_factor', 'feat_rest_factor', 'feat_blowout_factor', 'feat_usage_mult',
        'feat_spread', 'feat_is_home', 'feat_is_b2b', 'feat_days_rest',
        'feat_game_total', 'feat_opp_drtg_season', 'feat_opp_drtg_l5',
        'feat_games_played', 'closing_line', 'clv',
        'feat_blowout_prob', 'feat_personal_fatigue_factor', 'feat_b2b_games_in_sample', 
        'feat_dynamic_std_mult', 'feat_coef_variation',
        # V16 market & role context
        'feat_odds_decimal', 'feat_usg_season', 'feat_clv',
        'feat_usg_trend', 'grade'
    ]
    
    # Ensure all canonical columns exist (fill missing with defaults)
    for col in canonical_columns:
        if col not in combined_df.columns:
            combined_df[col] = 0 if col.startswith('feat_') else ''
    
    # Read-Merge-Write: Append to existing data and deduplicate
    # CRITICAL: Manual tracked bets (with real snapshots/tags) must be PRESERVED
    # Backtest data should only fill gaps, not overwrite your actual betting decisions
    if output_path.exists():
        try:
            existing_df = pd.read_csv(output_path)
            logger.info(f"Found existing data: {len(existing_df)} rows")
            
            # Ensure existing data has all canonical columns too
            for col in canonical_columns:
                if col not in existing_df.columns:
                    existing_df[col] = 0 if col.startswith('feat_') else ''
            
            # Count manual bets (non-backtest tags) before merge
            manual_bets_before = len(existing_df[existing_df['tag'] != 'backtest']) if 'tag' in existing_df.columns else 0
            
            # Merge: existing (your bets) goes AFTER new (backtest), then keep='last'
            # This ensures YOUR manual bets overwrite backtest data for same player/date/market
            combined_df = pd.concat([combined_df, existing_df], ignore_index=True)
            
            # Deduplicate: A player only plays one game per market per day
            # keep='last' ensures manual tracked bets (with real snapshots) are PRESERVED over backtest
            before_dedup = len(combined_df)
            combined_df = combined_df.drop_duplicates(
                subset=['date', 'player', 'market'], 
                keep='last'
            )
            duplicates_removed = before_dedup - len(combined_df)
            
            if duplicates_removed > 0:
                logger.info(f"Removed {duplicates_removed} duplicate entries (date/player/market)")
            
            # Verify manual bets were preserved
            manual_bets_after = len(combined_df[combined_df['tag'] != 'backtest']) if 'tag' in combined_df.columns else 0
            logger.info(f"✅ Manual bets preserved: {manual_bets_after} (was {manual_bets_before})")
            
            logger.info(f"Merged total: {len(combined_df)} rows")
        except Exception as e:
            logger.warning(f"Could not read existing data, overwriting: {e}")
    
    # Reorder columns to canonical order and save
    # Keep only canonical columns (drop any extras like 'grade' duplicates)
    final_columns = [c for c in canonical_columns if c in combined_df.columns]
    combined_df = combined_df[final_columns]
    combined_df.to_csv(output_path, index=False)
    logger.info(f"✓ ML training data saved to {output_path}")
    
    if progress_callback:
        progress_callback(1.0)  # Complete the progress bar
    
    return combined_df


def generate_ml_data_streamlit():
    """Streamlit UI wrapper for ML data generation."""
    st.markdown("### 🤖 Generate ML Training Data")
    
    st.info("""
    **OPTIMIZED**: Downloads ALL game logs in just 2 API calls, then runs backtests offline.
    - Step 1: Bulk download (2 API calls) - ~30 seconds
    - Step 2: Get player list (1 API call) - ~2 seconds  
    - Step 3: Offline backtesting - depends on # players/markets
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        num_players = st.slider("Number of Players", 10, 600, 50, 10)
        test_days = st.slider("Games per Player", 20, 80, 60, 10)
    
    with col2:
        markets = st.multiselect(
            "Markets",
            ['PTS', 'REB', 'AST', 'PRA', 'PA', 'PR', '3PM', 'STL', 'BLK'],
            default=['PTS', 'REB', 'AST', 'PRA']
        )
        output_file = st.text_input("Output Filename", "ml_training_data.csv")
    
    if st.button("🚀 Generate Dataset", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(pct):
            progress_bar.progress(pct)
            status_text.text(f"Progress: {pct:.0%}")
        
        with st.spinner("Generating ML training data..."):
            df = generate_ml_training_data(
                output_file=output_file,
                num_players=num_players,
                markets=markets,
                test_days=test_days,
                progress_callback=update_progress
            )
        
        if len(df) > 0:
            st.success(f"✅ Generated {len(df)} training samples!")
            
            # Show summary stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Samples", len(df))
            col2.metric("Win Rate", f"{df['hit'].mean():.1%}")
            col3.metric("Unique Players", df['player'].nunique())
            
            # Preview
            st.dataframe(df.head(20), width="stretch")
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                output_file,
                "text/csv",
                key='download-ml-data'
            )
        else:
            st.error("Failed to generate training data. Check logs for errors.")


# =============================================================================
# ORCHESTRATOR (Replaces EliteModel)
# =============================================================================

class PredictionOrchestrator: 
    """
    Orchestrates the entire prediction pipeline. 
    Coordinates all layers to produce final analysis results.
    """
    
    def __init__(self, config: Config = CONFIG):
        self.config = config
        self.tracker = Tracker()
        self.data_loader = DataLoader(config)
        self.feature_engineer = FeatureEngineer(config)
        self.model_engine = ModelEngine(config, tracker=self.tracker)
        self.simulation_engine = SimulationEngine(config)
        self.decision_policy = DecisionPolicy(config) 
        self.simulation_engine = SimulationEngine(config)
        self.backtester = Backtester(
            self.data_loader,
            self.feature_engineer,
            self.model_engine,
            self. simulation_engine,
            self.decision_policy,
            config
        )
        
        # Pre-load shared data
        self._team_stats = None
        self._avg_pace = None
        self._avg_def = None
        self._position_defense = None
    
    def _ensure_team_data_loaded(self):
        """
        Lazy load team data with L10 Recency Fix.
        """
        if self._team_stats is None:
            try:
                # 1. Attempt to fetch Last 10 Games data (Recency Bias)
                from nba_api.stats.endpoints import leaguedashteamstats
                
                stats_l10 = leaguedashteamstats.LeagueDashTeamStats(
                    last_n_games=10,
                    measure_type_detailed_defense='Base', # Standard stats
                    measure_type='Advanced',              # <--- FIX: Request Advanced stats to get PACE
                    per_mode_detailed='PerGame',
                    season=self.config.CURRENT_SEASON
                ).get_data_frames()[0]
                
                if not stats_l10.empty:
                    stats_l10.set_index('TEAM_ID', inplace=True)
                    self._team_stats = stats_l10
                    self._avg_pace = stats_l10['PACE'].mean()
                    self._avg_def = stats_l10['DEF_RATING'].mean()
                else:
                    raise ValueError("Empty L10 data")

            except Exception as e:
                # Fallback to season-long data if API fails
                # print(f"⚠️ Recency fetch failed ({e}), falling back to full season data.")
                # Use standard data loader fallback
                self._team_stats, self._avg_pace, self._avg_def = self.data_loader.fetch_team_stats()

            # Load position defense
            self._position_defense = self.data_loader.fetch_position_defense()
    
    def run_analysis(
        self,
        player_name: str,
        opponent_name: str,
        market: str,
        line: float,
        odds: float,
        is_home: bool,
        is_b2b: bool,
        lookback: int,
        usage_mult: float,
        spread: float,
        bankroll: float,
        days_rest: int = 1,  # New: days since last game
        game_total: float = 0.0  # New: Vegas O/U total
    ) -> AnalysisResult: 
        """
        Run complete analysis pipeline. 
        
        Returns AnalysisResult with all components or error message.
        """
        try:
            # Step 1: Resolve player and team
            p_obj, t_obj = self.data_loader. get_player_and_team(player_name, opponent_name)
            
            # Step 2: Get player position
            player_position = self.data_loader.get_player_position(p_obj['id'])
            
            # Step 3:  Fetch game logs
            df = self.data_loader.fetch_game_logs(p_obj['id'])
            if len(df) == 0:
                return AnalysisResult(
                    success=False,
                    error='No game logs found for this player.'
                )
            
            # Step 4: Ensure team data is loaded
            self._ensure_team_data_loaded()
            
            # Step 4.5: [INJURY-AWARE] Get player's team and calculate usage boost
            # The player's team is in their game logs (most recent)
            player_team_id = None
            injured_teammates = []
            injury_boost = 1.0
            try:
                if 'TEAM_ID' in df.columns and len(df) > 0:
                    player_team_id = int(df.iloc[-1]['TEAM_ID'])
                    injury_boost, injured_teammates = self.feature_engineer.get_injury_usage_boost(
                        player_name=p_obj['full_name'],
                        team_id=player_team_id
                    )
                    if injured_teammates:
                        logger.info(f"🏥 Injury boost for {p_obj['full_name']}: {injury_boost:.0%} due to {injured_teammates}")
            except Exception as e:
                logger.debug(f"Injury boost calculation skipped: {e}")
            
            # Apply injury boost to usage_mult
            effective_usage_mult = usage_mult * injury_boost
            
            # Step 5: Build feature vector
            features = self.feature_engineer.build_feature_vector(
                player_id=p_obj['id'],
                player_name=p_obj['full_name'],
                player_position=player_position,
                opponent_id=t_obj['id'],
                opponent_abbrev=t_obj['abbreviation'],
                is_home=is_home,
                is_b2b=is_b2b,
                spread=spread,
                usage_mult=effective_usage_mult,
                df=df,
                stat_col=market,
                line=line,
                lookback=lookback,
                team_stats=self._team_stats,
                position_defense=self._position_defense,
                avg_pace=self._avg_pace,
                avg_def=self._avg_def,
                market=market,
                days_rest=days_rest,
                game_total=game_total
            )
            
            # Step 6: Generate projection
            projection = self.model_engine. generate_projection(features)
            
            # Step 7: Run simulation (with features for smart mixture model)
            simulation = self.simulation_engine.run_simulation(
                mean=projection. final_projection,
                std=features. std,
                line=line,
                market=market,
                features=features  # Enable smart simulation
            )
            
            # Step 8: Make decision
            decision = self.decision_policy.make_decision(
                features=features,
                projection=projection,
                simulation=simulation,
                line=line,
                odds=odds,
                bankroll=bankroll
            )
            
            # Step 9: Calculate hit rates for UI
            hit_rates = self.feature_engineer.calculate_hit_rates(df, market, line)
            
            return AnalysisResult(
                success=True,
                player_name=p_obj['full_name'],
                player_id=p_obj['id'],
                opponent_name=t_obj['abbreviation'],
                opponent_id=t_obj['id'],
                market=market,
                line=line,
                odds=odds,
                features=features,
                projection=projection,
                simulation=simulation,
                decision=decision,
                game_logs=df,
                hit_rates=hit_rates,
                injured_teammates=injured_teammates,
                injury_usage_boost=injury_boost
            )
            
        except DataLoaderError as e:
            return AnalysisResult(success=False, error=str(e))
        except ValueError as e:
            return AnalysisResult(success=False, error=str(e))
        except Exception as e: 
            logger.error(f"Analysis failed: {e}")
            return AnalysisResult(success=False, error=f"Analysis failed: {str(e)}")
    
    def run_backtest(
        self,
        player_name:  str,
        market: str,
        lookback: int = 15,
        test_days: int = 30,
        line_offset: float = 0.0,
        fixed_spread: float = 0.0,
        progress_callback=None
    ) -> Optional[BacktestSummary]:
        """Run backtest for a player."""
        try:
            # Resolve player
            all_players = players. get_players()
            clean_name = normalize_name(player_name)
            p_obj = next((p for p in all_players if normalize_name(p['full_name']) == clean_name), None)
            if not p_obj: 
                p_obj = next((p for p in all_players if clean_name in normalize_name(p['full_name'])), None)
            
            if not p_obj:
                logger.error(f"Player not found for backtest: {player_name}")
                return None
            
            return self.backtester.run_backtest(
                player_id=p_obj['id'],
                player_name=p_obj['full_name'],
                market=market,
                lookback=lookback,
                test_days=test_days,
                line_offset=line_offset,
                fixed_spread=fixed_spread,
                progress_callback=progress_callback
            )
        except Exception as e: 
            logger.error(f"Backtest failed: {e}")
            return None


# =============================================================================
# TRACKERS (Bet & Parlay)
# =============================================================================

class Tracker:
    """Tracks individual bets with ML feature storage."""
    
    def __init__(self, file_path: Path = TRACKER_FILE):
        self.file = file_path
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        if not self.file.exists():
            self._save([])

    def log_bet(self, bet_data: Dict):
        """Alias for save_bet to prevent 'AttributeError'."""
        self.save_bet(bet_data)

    def save_bet(self, bet_data: Dict):
        import time
        from datetime import datetime
        
        bets = self.get_bets()
        
        # 1. Generate ID if missing
        if 'id' not in bet_data:
            bet_data['id'] = int(time.time() * 1000)
            
        # 2. Generate DATE if missing (Format: YYYY-MM-DD)
        if 'date' not in bet_data or not bet_data['date']:
            bet_data['date'] = datetime.now().strftime("%Y-%m-%d")
            
        # 3. FIX: Ensure Result defaults to 'Pending' (Fixes the "1 vs 5" bug)
        if 'result' not in bet_data:
            bet_data['result'] = 'Pending'
            
        bets.append(bet_data)
        self._save(bets)

    def get_bets(self) -> list: 
        """Loads bets and auto-fixes any legacy data."""
        try:
            if self.file.exists():
                with open(self.file, 'r') as f:
                    content = f.read()
                    data = json.loads(content) if content else []
                
                # Auto-Heal Logic
                if data:
                    modified = False
                    import time
                    from datetime import datetime
                    
                    for i, bet in enumerate(data):
                        # Fix ID
                        if 'id' not in bet:
                            bet['id'] = int((time.time() + i) * 1000)
                            modified = True
                        # Fix Date
                        if 'date' not in bet or not bet['date']:
                            bet['date'] = datetime.now().strftime("%Y-%m-%d")
                            modified = True
                        # Fix Missing Result (The logic bug)
                        if 'result' not in bet:
                            bet['result'] = 'Pending'
                            modified = True
                            
                    if modified:
                        self._save(data)
                        
                return data
            return []
        except Exception:
            return []

    def _save(self, bets: list):
        # --- FIX: Custom Encoder for NumPy Types (Fixes JSON Error) ---
        def numpy_converter(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        try:
            with open(self.file, 'w') as f:
                json.dump(bets, f, indent=2, default=numpy_converter)
        except Exception as e:
            logging.error(f"Failed to save bets: {e}")

    def get_player_history(self, player_name: str) -> List[Dict]:
        all_bets = self.get_bets()
        return [bet for bet in all_bets if bet.get('player_name') == player_name]

    def update_result(self, bet_id: int, new_status: str, closing_line: float = None, actual_value: float = None):
        current_bets = self.get_bets()
        for bet in current_bets:
            if bet.get('id') == bet_id: 
                bet['result'] = new_status
                
                if actual_value is not None:
                    bet['actual_value'] = actual_value
                    line = bet.get('line', 0)
                    side = bet.get('side', 'OVER')
                    market = bet.get('market', 'PTS')  # Get market for result categorization
                    
                    if side == 'OVER':
                        bet['margin'] = actual_value - line
                    else:
                        bet['margin'] = line - actual_value
                    
                    # Calculate margin_pct: Standardizes "luck" vs "lock" across stat types
                    # +1.7% margin on PTS is barely a win, +23.8% on REB is dominant
                    if line > 0:
                        bet['margin_pct'] = ((actual_value - line) / line) * 100
                        if side == 'UNDER':  # Flip sign for unders
                            bet['margin_pct'] = -bet['margin_pct']
                    else:
                        bet['margin_pct'] = 0.0
                    
                    # Pass market to use appropriate thresholds (low-count vs high-count)
                    bet['result_quality'] = self._categorize_result(new_status, bet['margin'], market)
                
                if closing_line is not None:
                    bet['closing_line'] = closing_line
                    opening_line = bet.get('line', 0)
                    side = bet.get('side', 'OVER')
                    if side == 'OVER':
                        bet['clv'] = closing_line - opening_line
                    else:
                        bet['clv'] = opening_line - closing_line
                break
        self._save(current_bets)
    
    def _categorize_result(self, result: str, margin: float, market: str = None) -> str:
        """
        Categorize result quality with different thresholds for low-count vs high-count stats.
        
        Low-count stats (3PM, STL, BLK): Tighter margins because 1 stat = significant swing
        High-count stats (PTS, PRA): Wider margins because variance is expected
        """
        abs_margin = abs(margin)
        
        # Determine if this is a low-count stat
        is_low_count = market in CONFIG.LOW_COUNT_STATS if market else False
        
        # Set thresholds based on stat type
        if is_low_count:
            sweat_threshold = CONFIG.LOW_COUNT_MARGIN_SWEAT    # 0.5
            close_threshold = CONFIG.LOW_COUNT_MARGIN_CLOSE    # 1.5
            solid_threshold = CONFIG.LOW_COUNT_MARGIN_SOLID    # 2.5
        else:
            sweat_threshold = CONFIG.HIGH_COUNT_MARGIN_SWEAT   # 1.5
            close_threshold = CONFIG.HIGH_COUNT_MARGIN_CLOSE   # 3.5
            solid_threshold = CONFIG.HIGH_COUNT_MARGIN_SOLID   # 7.5
        
        if result == 'Push' or abs_margin < 0.5:
            return ResultQuality.PUSH.value
        elif result == 'Win':
            if abs_margin <= sweat_threshold: return ResultQuality.SWEAT_WIN.value
            elif abs_margin <= close_threshold: return ResultQuality.CLOSE_WIN.value
            elif abs_margin <= solid_threshold: return ResultQuality.SOLID_WIN.value
            else: return ResultQuality.BLOWOUT_WIN.value
        elif result == 'Loss':
            if abs_margin <= sweat_threshold: return ResultQuality.BAD_BEAT.value
            elif abs_margin <= close_threshold: return ResultQuality.CLOSE_LOSS.value
            elif abs_margin <= solid_threshold: return ResultQuality.CLEAR_LOSS.value
            else: return ResultQuality.BAD_READ.value
        else:
            return ResultQuality.PENDING.value

    def delete_bet(self, bet_id: int):
        current_bets = self.get_bets()
        current_bets = [b for b in current_bets if b.get('id') != bet_id]
        self._save(current_bets)

    def clear_history(self):
        self._save([])

    def get_stats(self) -> dict:
        bets = self.get_bets()
        # FIX: Count None/Missing as 'Pending'
        wins = len([b for b in bets if b.get('result') == 'Win'])
        losses = len([b for b in bets if b.get('result') == 'Loss'])
        pushes = len([b for b in bets if b.get('result') == 'Push'])
        pending = len([b for b in bets if b.get('result', 'Pending') == 'Pending'])
        
        total_decided = wins + losses
        win_rate = wins / total_decided if total_decided > 0 else 0
        total_profit = 0
        for bet in bets: 
            if bet.get('result') == 'Win':
                total_profit += bet.get('stake', 0) * (bet.get('odds', 1) - 1)
            elif bet.get('result') == 'Loss':
                total_profit -= bet.get('stake', 0)
        
        bets_with_clv = [b for b in bets if b.get('clv') is not None]
        clv_count = len(bets_with_clv)
        avg_clv = sum(b['clv'] for b in bets_with_clv) / clv_count if clv_count > 0 else 0
        positive_clv = len([b for b in bets_with_clv if b['clv'] > 0])
        clv_positive_rate = positive_clv / clv_count if clv_count > 0 else 0
        
        bets_with_margin = [b for b in bets if b.get('margin') is not None]
        margin_count = len(bets_with_margin)
        avg_margin = sum(b['margin'] for b in bets_with_margin) / margin_count if margin_count > 0 else 0
        
        quality_counts = {}
        for rq in ResultQuality:
            quality_counts[rq.value] = len([b for b in bets if b.get('result_quality') == rq.value])
        
        bad_beats = quality_counts.get(ResultQuality.BAD_BEAT.value, 0)
        bad_reads = quality_counts.get(ResultQuality.BAD_READ.value, 0)
        bad_beat_rate = bad_beats / losses if losses > 0 else 0
        bad_read_rate = bad_reads / losses if losses > 0 else 0
        
        sweat_wins = quality_counts.get(ResultQuality.SWEAT_WIN.value, 0)
        solid_wins = quality_counts.get(ResultQuality.SOLID_WIN.value, 0) + quality_counts.get(ResultQuality.BLOWOUT_WIN.value, 0)
        sweat_win_rate = sweat_wins / wins if wins > 0 else 0
        
        return {
            'wins': wins, 'losses': losses, 'pushes': pushes, 'pending': pending,
            'total_decided': total_decided, 'win_rate': win_rate, 'total_profit': total_profit,
            'clv_count': clv_count, 'avg_clv': avg_clv, 'clv_positive_rate': clv_positive_rate,
            'margin_count': margin_count, 'avg_margin': avg_margin,
            'quality_counts': quality_counts,
            'bad_beat_rate': bad_beat_rate, 'bad_read_rate': bad_read_rate,
            'sweat_win_rate': sweat_win_rate, 'solid_win_rate': solid_wins / wins if wins > 0 else 0
        }

    def export_training_data(self) -> pd.DataFrame:
        bets = self.get_bets()
        training_bets = [
            b for b in bets
            if b.get('result') in ['Win', 'Loss'] and b.get('feat_ema') is not None
        ]
        if not training_bets:
            return pd.DataFrame()
        df = pd.DataFrame(training_bets)
        df['target'] = (df['result'] == 'Win').astype(int)
        
        if 'margin' in df.columns:
            df['target_margin'] = df['margin'].fillna(0)
            quality_order = {
                ResultQuality.BAD_READ.value: 0,
                ResultQuality.CLEAR_LOSS.value: 1,
                ResultQuality.CLOSE_LOSS.value: 2,
                ResultQuality.BAD_BEAT.value: 3,
                ResultQuality.PUSH.value: 4,
                ResultQuality.SWEAT_WIN.value: 5,
                ResultQuality.CLOSE_WIN.value: 6,
                ResultQuality.SOLID_WIN.value: 7,
                ResultQuality.BLOWOUT_WIN.value: 8,
                ResultQuality.PENDING.value: -1
            }
            df['target_quality'] = df['result_quality'].map(quality_order).fillna(-1).astype(int)
        
        return df

    def export_bets_to_training_csv(self, output_file: str = "ml_training_data.csv") -> Tuple[int, str]:
        """
        Export tracked bets to ML training CSV.
        FIXED: Reads existing CSV header first to match column structure exactly.
        """
        import csv
        bets = self.get_bets()
        output_path = DATA_DIR / output_file
        
        # Default fieldnames (full schema for new files) - MUST match canonical_columns
        default_fieldnames = [
            'date', 'player', 'opponent', 'market', 'position',
            'line', 'predicted_side', 'predicted_prob', 'predicted_ev', 'projected_value',
            'result', 'hit', 'actual_value', 'margin', 'margin_pct', 'result_quality',
            # Snapshot: Point-in-time frozen stats (time capsule for Robot training)
            'snapshot_l5_hit_rate', 'snapshot_days_rest', 'snapshot_def_rank',
            # Tag: Bet source for filtering training data
            'tag',
            'feat_ema', 'feat_std', 'feat_sma_5', 'feat_sma_10', 'feat_trend',
            'feat_avg_minutes', 'feat_mins_trend',
            'feat_hit_l5', 'feat_hit_l10', 'feat_hit_l15', 'feat_hit_season',
            'feat_pace_mult', 'feat_def_mult', 'feat_position_mult',
            'feat_base_matchup_mult', 'feat_combined_matchup_mult',
            'feat_split_factor', 'feat_rest_factor', 'feat_blowout_factor', 'feat_usage_mult',
            'feat_spread', 'feat_is_home', 'feat_is_b2b', 'feat_days_rest',
            'feat_game_total', 'feat_opp_drtg_season', 'feat_opp_drtg_l5',
            'feat_games_played', 'closing_line', 'clv',
            'feat_blowout_prob', 'feat_personal_fatigue_factor', 'feat_b2b_games_in_sample', 
            'feat_dynamic_std_mult', 'feat_coef_variation',
            # V16 market & role context
            'feat_odds_decimal', 'feat_usg_season', 'feat_clv',
            'feat_usg_trend', 'grade'
        ]
        
        # Check if file exists and read its header to match column structure
        file_exists = output_path.exists()
        if file_exists:
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    existing_header = next(reader, None)
                    if existing_header:
                        # Use existing header to maintain consistency
                        fieldnames = existing_header
                    else:
                        fieldnames = default_fieldnames
            except Exception:
                fieldnames = default_fieldnames
        else:
            fieldnames = default_fieldnames
        
        exported_count = 0
        bets_modified = False
        
        with open(output_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if not file_exists:
                writer.writeheader()
            
            for bet in bets:
                if (bet.get('result') in ['Win', 'Loss'] and 
                    bet.get('actual_value') is not None and 
                    not bet.get('exported_to_csv', False)):
                    
                    # Build row with all possible fields - DictWriter will ignore extras
                    row = {
                        'date': str(bet.get('date', '2024-01-01')).split(' ')[0],
                        'player': bet.get('player', ''),
                        'opponent': bet.get('opponent', ''),
                        'market': bet.get('market', ''),
                        'position': bet.get('feat_position', ''),
                        'line': bet.get('line', 0),
                        'predicted_side': bet.get('side', ''),
                        'predicted_prob': bet.get('prob', 0),
                        'predicted_ev': bet.get('ev', 0),
                        'projected_value': bet.get('proj', 0),
                        'result': bet.get('result', ''),
                        'hit': 1 if bet.get('result') == 'Win' else 0,
                        'actual_value': bet.get('actual_value', 0),
                        'margin': bet.get('margin', 0),
                        'margin_pct': bet.get('margin_pct', 0),  # Standardized margin across stat types
                        'result_quality': bet.get('result_quality', 'legacy'),
                        # Snapshot: Frozen point-in-time stats (Time Capsule)
                        'snapshot_l5_hit_rate': bet.get('snapshot_l5_hit_rate', bet.get('feat_hit_l5', 0)),
                        'snapshot_days_rest': bet.get('snapshot_days_rest', bet.get('feat_days_rest', 1)),
                        'snapshot_def_rank': bet.get('snapshot_def_rank', 0),
                        # Tag: Bet source for filtering training data
                        'tag': bet.get('tag', 'legacy'),
                        'feat_ema': bet.get('feat_ema', 0),
                        'feat_std': bet.get('feat_std', 0),
                        'feat_sma_5': bet.get('feat_sma_5', 0),
                        'feat_sma_10': bet.get('feat_sma_10', 0),
                        'feat_trend': bet.get('feat_trend', 0),
                        'feat_avg_minutes': bet.get('feat_avg_minutes', 0),
                        'feat_mins_trend': bet.get('feat_mins_trend', 0),
                        'feat_hit_l5': bet.get('feat_hit_l5', 0),
                        'feat_hit_l10': bet.get('feat_hit_l10', 0),
                        'feat_hit_l15': bet.get('feat_hit_l15', 0),
                        'feat_hit_season': bet.get('feat_hit_season', 0),
                        'feat_pace_mult': bet.get('feat_pace_mult', 1),
                        'feat_def_mult': bet.get('feat_def_mult', 1),
                        'feat_position_mult': bet.get('feat_position_mult', 1),
                        'feat_base_matchup_mult': bet.get('feat_base_matchup_mult', 1),
                        'feat_combined_matchup_mult': bet.get('feat_combined_matchup_mult', 1),
                        'feat_split_factor': bet.get('feat_split_factor', 1),
                        'feat_rest_factor': bet.get('feat_rest_factor', 1),
                        'feat_blowout_factor': bet.get('feat_blowout_factor', 1),
                        'feat_usage_mult': bet.get('feat_usage_mult', 1),
                        'feat_spread': bet.get('feat_spread', 0),
                        'feat_is_home': 1 if bet.get('feat_is_home', False) else 0,
                        'feat_is_b2b': 1 if bet.get('feat_is_b2b', False) else 0,
                        'feat_days_rest': bet.get('feat_days_rest', 1),
                        'feat_game_total': bet.get('feat_game_total', 0),
                        'feat_opp_drtg_season': bet.get('feat_opp_drtg_season', 0),
                        'feat_opp_drtg_l5': bet.get('feat_opp_drtg_l5', 0),
                        'feat_games_played': bet.get('feat_games_played', 0),
                        'closing_line': bet.get('closing_line', 0),
                        'clv': bet.get('clv', 0),
                        'feat_blowout_prob': bet.get('blowout_prob', 0),
                        'feat_personal_fatigue_factor': bet.get('personal_fatigue_factor', 1),
                        'feat_b2b_games_in_sample': bet.get('b2b_games_in_sample', 0),
                        'feat_dynamic_std_mult': bet.get('dynamic_std_mult', 1),
                        'feat_coef_variation': bet.get('coef_variation', 0),
                        # V16 market & role context
                        'feat_odds_decimal': bet.get('odds_decimal', american_to_decimal(bet.get('odds', -110))),
                        'feat_usg_season': bet.get('usg_season', 0),
                        'feat_clv': (bet.get('closing_line', 0) - bet.get('line', 0)) if bet.get('closing_line') else 0,
                        'feat_usg_trend': bet.get('feat_usg_trend', 1.0),
                        # Backtest-style column aliases (for compatibility)
                        'player_name': bet.get('player', ''),
                        'grade': 'A' if bet.get('ev', 0) > 0.05 else 'B',
                    }
                    
                    writer.writerow(row)
                    bet['exported_to_csv'] = True
                    bets_modified = True
                    exported_count += 1
        
        if bets_modified:
            self._save(bets)
        
        return exported_count, str(output_path)

    def get_exportable_count(self) -> int:
        bets = self.get_bets()
        return len([
            b for b in bets 
            if b.get('result') in ['Win', 'Loss'] 
            and b.get('actual_value') is not None 
            and not b.get('exported_to_csv', False)
        ])

    def get_feature_stats(self) -> dict:
        bets = self.get_bets()
        total_with_features = len([b for b in bets if b.get('feat_ema') is not None])
        decided_with_features = len([
            b for b in bets
            if b.get('result') in ['Win', 'Loss'] and b.get('feat_ema') is not None
        ])
        exportable = self.get_exportable_count()
        return {
            'total_with_features': total_with_features,
            'decided_with_features': decided_with_features,
            'pending_with_features': total_with_features - decided_with_features,
            'exportable_to_csv': exportable
        }


class ParlayTracker: 
    """Tracks parlay bets."""
    
    def __init__(self, file_path: Path = PARLAY_FILE):
        self.file = file_path
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        if not self.file.exists():
            self._save([])

    def get_parlays(self) -> list:
        try:
            with open(self.file, 'r') as f:
                content = f.read()
                return json. loads(content) if content else []
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to read parlay tracker:  {e}")
            return []

    def create_parlay(self, legs: list, stake: float, name: str = "") -> dict:
        parlays = self.get_parlays()
        combined_odds = 1.0
        combined_prob = 1.0
        for leg in legs: 
            combined_odds *= leg. get('odds', 1.0)
            combined_prob *= leg. get('prob', 0.5)
        parlay = {
            'id': int(time. time() * 1000),
            'name': name or f"Parlay #{len(parlays) + 1}",
            'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'legs': legs,
            'num_legs': len(legs),
            'stake': stake,
            'combined_odds': round(combined_odds, 2),
            'combined_prob':  round(combined_prob, 4),
            'potential_payout': round(stake * combined_odds, 2),
            'potential_profit': round(stake * (combined_odds - 1), 2),
            'result': 'Pending',
            'legs_hit': 0,
            'legs_decided': 0,
            'actual_payout': 0.0,
        }
        parlays.append(parlay)
        self._save(parlays)
        return parlay

    def update_leg_result(self, parlay_id:  int, leg_index: int, result:  str):
        parlays = self.get_parlays()
        for parlay in parlays:
            if parlay['id'] == parlay_id:
                if 0 <= leg_index < len(parlay['legs']):
                    parlay['legs'][leg_index]['result'] = result
                    self._recalculate_parlay_status(parlay)
                break
        self._save(parlays)

    def _recalculate_parlay_status(self, parlay:  dict):
        legs_hit = legs_lost = legs_push = legs_pending = 0
        for leg in parlay['legs']: 
            result = leg.get('result', 'Pending')
            if result == 'Win':
                legs_hit += 1
            elif result == 'Loss':
                legs_lost += 1
            elif result == 'Push': 
                legs_push += 1
            else:
                legs_pending += 1
        parlay['legs_hit'] = legs_hit
        parlay['legs_decided'] = legs_hit + legs_lost + legs_push
        if legs_lost > 0:
            parlay['result'] = 'Loss'
            parlay['actual_payout'] = 0.0
        elif legs_pending == 0:
            if legs_hit == len(parlay['legs']):
                parlay['result'] = 'Win'
                parlay['actual_payout'] = parlay['potential_payout']
            elif legs_hit + legs_push == len(parlay['legs']):
                adjusted_odds = 1.0
                for leg in parlay['legs']:
                    if leg.get('result') == 'Win':
                        adjusted_odds *= leg. get('odds', 1.0)
                parlay['result'] = 'Win'
                parlay['actual_payout'] = round(parlay['stake'] * adjusted_odds, 2)
        else:
            parlay['result'] = 'Pending'

    def delete_parlay(self, parlay_id: int):
        parlays = self.get_parlays()
        parlays = [p for p in parlays if p['id'] != parlay_id]
        self._save(parlays)

    def get_stats(self) -> dict:
        parlays = self.get_parlays()
        total_parlays = len(parlays)
        wins = len([p for p in parlays if p['result'] == 'Win'])
        losses = len([p for p in parlays if p['result'] == 'Loss'])
        pending = len([p for p in parlays if p['result'] == 'Pending'])
        decided_parlays = [p for p in parlays if p['result'] != 'Pending']
        total_staked = sum(p['stake'] for p in decided_parlays)
        total_returned = sum(p['actual_payout'] for p in decided_parlays)
        total_profit = total_returned - total_staked
        roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
        avg_legs = sum(p['num_legs'] for p in parlays) / total_parlays if total_parlays > 0 else 0
        total_legs = sum(p['legs_decided'] for p in parlays)
        total_legs_hit = sum(p['legs_hit'] for p in parlays)
        leg_hit_rate = total_legs_hit / total_legs if total_legs > 0 else 0
        return {
            'total_parlays': total_parlays, 'wins': wins, 'losses':  losses, 'pending': pending,
            'win_rate': wins / (wins + losses) if (wins + losses) > 0 else 0,
            'total_staked': total_staked, 'total_returned': total_returned,
            'total_profit': total_profit, 'roi':  roi, 'avg_legs': avg_legs,
            'leg_hit_rate':  leg_hit_rate, 'total_legs': total_legs, 'total_legs_hit': total_legs_hit
        }

    def clear_history(self):
        self._save([])

    def _save(self, data: list):
        try:
            with open(self. file, 'w') as f:
                json.dump(data, f, indent=4)
        except IOError as e:
            logger. error(f"Failed to save parlay tracker: {e}")


# =============================================================================
# UI COMPONENTS
# =============================================================================

def check_bet_rules(result: AnalysisResult) -> Tuple[bool, Dict[str, Tuple[bool, str, str]]]:
    """
    Check if a bet passes all 5 mandatory rules.
    Returns (all_passed, {rule_name: (passed, description, explanation)})
    
    THE 5 RULES:
    1. Negative Offset Mandate - Only bet when Model Avg > Vegas Line
    2. Probability Goldilocks Zone - Win Prob between 60% and 72%
    3. Grade A Filter - Only Grade A bets (EV > 5%)
    4. Blowout/Spread Cap - No OVERs on Stars (>32 min) if Spread > 12
    5. Stability Check - CV must be ≤ 30%
    """
    decision = result.decision
    projection = result.projection
    features = result.features
    
    # Calculate offset (Model Avg - Vegas Line)
    # Negative offset = Model predicts OVER the line = good for OVER bets
    offset = result.line - projection.final_projection  # Positive when line > projection
    
    rules = {}
    
    # ==========================================================================
    # RULE 1: Negative Offset Mandate
    # "Never bet on a player to beat their Ceiling. Only bet to beat their Floor."
    # For OVER: Model Avg must be > Vegas Line (offset is negative/green)
    # For UNDER: Model Avg must be < Vegas Line (offset is positive)
    # ==========================================================================
    if decision.recommended_side == "OVER":
        # For OVER bets, we want projection > line (negative offset)
        offset_passed = projection.final_projection > result.line
        offset_val = result.line - projection.final_projection  # Negative is good
        offset_desc = f"Proj {projection.final_projection:.1f} vs Line {result.line} = {offset_val:+.1f}"
        offset_explain = "Model Avg must exceed Vegas Line for OVER bets"
    else:
        # For UNDER bets, we want projection < line (positive offset)
        offset_passed = projection.final_projection < result.line
        offset_val = result.line - projection.final_projection  # Positive is good for UNDER
        offset_desc = f"Proj {projection.final_projection:.1f} vs Line {result.line} = {offset_val:+.1f}"
        offset_explain = "Model Avg must be below Vegas Line for UNDER bets"
    rules['1_offset'] = (offset_passed, offset_desc, offset_explain)
    
    # ==========================================================================
    # RULE 2: Probability "Goldilocks" Zone (60% - 72%)
    # "Your model lies when it is arrogant."
    # >75% predictions often ignore blowout risk (actual win rate ~25%)
    # 65-70% is the sweet spot (predicted 68%, actual 83%)
    # ==========================================================================
    prob_pct = decision.probability * 100
    prob_passed = 60.0 <= prob_pct <= 72.0
    if prob_pct > 72.0:
        prob_desc = f"{prob_pct:.1f}% ⚠️ TOO HIGH (max 72%)"
        prob_explain = "Model is overconfident - likely ignoring blowout/variance risk"
    elif prob_pct < 60.0:
        prob_desc = f"{prob_pct:.1f}% ⚠️ TOO LOW (min 60%)"
        prob_explain = "Edge is too thin to justify the variance"
    else:
        prob_desc = f"{prob_pct:.1f}% ✓ Sweet Spot"
        prob_explain = "In the Goldilocks zone where model is most accurate"
    rules['2_prob'] = (prob_passed, prob_desc, prob_explain)
    
    # ==========================================================================
    # RULE 3: Grade A Filter
    # "Ignore the Noise."
    # Grade A = EV > 5% = only bets that justify variance risk
    # Grade B/C/D/F have negative ROI historically
    # ==========================================================================
    grade_passed = decision.grade == BetGrade.A
    grade_desc = f"Grade {decision.grade.value} (EV: {decision.expected_value:+.1%})"
    if grade_passed:
        grade_explain = "EV > 5% justifies the variance risk"
    else:
        grade_explain = "Only Grade A bets have positive ROI historically"
    rules['3_grade'] = (grade_passed, grade_desc, grade_explain)
    
    # ==========================================================================
    # RULE 4: Blowout/Spread Cap
    # "Stars sit during blowouts."
    # No OVERs on Star Players (Avg Min > 32) if Spread > 12
    # Even with blowout_factor adjustment, real-world benching is unpredictable
    # ==========================================================================
    abs_spread = abs(features.spread)
    is_star = features.avg_minutes >= 32.0
    is_over = decision.recommended_side == "OVER"
    
    # Rule only applies to OVER bets on star players
    if is_over and is_star:
        spread_passed = abs_spread < 12.0
        spread_desc = f"Star ({features.avg_minutes:.0f}min) + Spread {abs_spread:.1f}"
        if spread_passed:
            spread_explain = "Spread is acceptable for star player OVER"
        else:
            spread_explain = "Star players get benched in blowouts - SKIP"
    else:
        spread_passed = True  # Rule doesn't apply
        if not is_over:
            spread_desc = f"UNDER bet - rule N/A"
            spread_explain = "Blowout rule only applies to OVER bets"
        else:
            spread_desc = f"Non-star ({features.avg_minutes:.0f}min) - rule N/A"
            spread_explain = "Role players often get garbage time minutes"
    rules['4_blowout'] = (spread_passed, spread_desc, spread_explain)
    
    # ==========================================================================
    # RULE 5: Stability Check (Coefficient of Variation)
    # "Avoid chaos."
    # CV > 30% = high variance player = model's average-based logic breaks down
    # High CV players (like Jimmy Butler) wreck predictions
    # ==========================================================================
    cv_pct = features.coef_variation * 100
    cv_passed = features.coef_variation <= 0.30
    cv_desc = f"CV: {cv_pct:.0f}% (max 30%)"
    if cv_passed:
        cv_explain = "Consistent player - model predictions reliable"
    else:
        cv_explain = "High variance player - average-based model unreliable"
    rules['5_stability'] = (cv_passed, cv_desc, cv_explain)
    
    all_passed = all(r[0] for r in rules.values())
    
    return all_passed, rules


def render_bet_rules_card(result: AnalysisResult):
    """Render compact bet rules validation card."""
    all_passed, rules = check_bet_rules(result)
    
    # Compact rule display
    rule_icons = {'1_offset': '📊', '2_prob': '🎯', '3_grade': '🏆', '4_blowout': '💨', '5_stability': '📉'}
    
    if all_passed:
        st.success(f"✅ **BET APPROVED** — Place ₱{result.decision.kelly_stake:.0f}")
    else:
        failed = sum(1 for r in rules.values() if not r[0])
        st.error(f"❌ **SKIP** — {failed} rule(s) failed")
    
    # Compact expander for rules
    with st.expander("📋 View Rules" if all_passed else "⚠️ View Failed Rules", expanded=not all_passed):
        for rule_key in sorted(rules.keys()):
            passed, desc, explain = rules[rule_key]
            icon = "✅" if passed else "❌"
            emoji = rule_icons.get(rule_key, '')
            color = "green" if passed else "red"
            st.markdown(f"{icon} {emoji} :{color}[{desc}]")
    
    return all_passed


def render_data_quality_card(result: AnalysisResult):
    """Render data quality indicator with fallback warnings."""
    features = result.features
    dq = features.data_quality
    
    # Color code by grade
    grade_colors = {'A': 'green', 'B': 'green', 'C': 'orange', 'D': 'red', 'F': 'red'}
    color = grade_colors.get(dq.grade, 'gray')
    
    if dq.has_issues:
        with st.expander(f"📡 Data Quality: :{color}[{dq.grade}] ({dq.score:.0f}/100)", expanded=False):
            st.caption("Some data fallbacks were used which may affect prediction accuracy:")
            for warning in dq.warnings:
                st.markdown(f"⚠️ {warning}")
            
            # Show specific flags
            flags = []
            if dq.missing_team_stats:
                flags.append("❌ Team stats unavailable")
            if dq.used_neutral_position_defense:
                flags.append("❌ Position defense data missing")
            if dq.used_default_pace:
                flags.append("⚠️ Using default pace (100.0)")
            if dq.used_default_def_rating:
                flags.append("⚠️ Using default def rating (115.0)")
            if dq.low_sample_size:
                flags.append(f"⚠️ Low sample size ({features.games_played} games)")
            if dq.used_fallback_std:
                flags.append("⚠️ Estimated std deviation")
            if dq.used_fallback_minutes:
                flags.append("⚠️ Default minutes (30)")
            if dq.used_fallback_split:
                flags.append("⚠️ Neutral home/away split")
            
            if flags:
                st.markdown("**Fallbacks used:**")
                for flag in flags:
                    st.markdown(f"  {flag}")
    else:
        st.success(f"📡 Data Quality: **{dq.grade}** (100/100) - All data available")


def render_ticket_card(result: 'AnalysisResult', bankroll: float):
    """
    Render a betting ticket-style card with Sniper Mode safety features.
    Includes: Rust Alert, Injury News Link, Evidence Row.
    """
    decision = result.decision
    projection = result.projection
    features = result.features

    # 1. Color & Badge Logic
    grade_colors = {'A': '#00c853', 'B': '#4caf50', 'C': '#ff9800', 'D': '#f44336', 'F': '#b71c1c'}
    grade_bg = {'A': 'rgba(0,200,83,0.15)', 'B': 'rgba(76,175,80,0.15)', 'C': 'rgba(255,152,0,0.15)', 'D': 'rgba(244,67,54,0.15)', 'F': 'rgba(183,28,28,0.15)'}
    
    grade_val = decision.grade.value
    grade_color = grade_colors.get(grade_val, '#888')
    bg_color = grade_bg.get(grade_val, 'rgba(100,100,100,0.15)')

    is_over = decision.recommended_side == 'OVER'
    side_color = '#00c853' if is_over else '#f44336'
    side_icon = '📈' if is_over else '📉'

    robot_prob = projection.ml_prob if projection.ml_prob is not None else 0.0
    robot_approved = robot_prob >= 0.60
    robot_badge = '🤖✓' if robot_approved else ('🤖✗' if projection.ml_prob is not None else '')

    # Conflict Logic
    is_conflict = (projection.ml_prob is not None) and (projection.ml_prob < 0.50) and (grade_val in ['A', 'B'])
    conflict_div = f"""<div style="background: rgba(255,152,0,0.2); border: 1px solid #ff9800; border-radius: 6px; padding: 8px; margin-top: 12px; text-align: center;"><span style="color: #ff9800; font-weight: bold; font-size: 12px;">⚠️ ROBOT DISAGREES ({robot_prob:.1%})</span></div>""" if is_conflict else ""

    # --- SNIPER MODE: Rust Alert ---
    rest_days = getattr(features, 'days_rest', 1)
    rust_alert = ""
    if rest_days > 7 and rest_days < 100:
        rust_alert = f'<span style="color: #ff9800; font-weight: bold; margin-right: 10px;">⚠️ {int(rest_days)} Days Rust</span>'
    
    # --- SNIPER MODE: Injury News Link ---
    clean_name = result.player_name.replace(" ", "+")
    news_url = f"https://www.google.com/search?q={clean_name}+injury+status+nba&tbm=nws"

    # --- Evidence Row: Form, Matchup, Fatigue ---
    l5_hit = result.hit_rates.get('l5', 0) if hasattr(result, 'hit_rates') and result.hit_rates else 0
    if l5_hit >= 0.80:
        form_label, form_color = "🔥 HOT", "#00c853"
    elif l5_hit >= 0.60:
        form_label, form_color = "✅ WARM", "#4caf50"
    elif l5_hit >= 0.40:
        form_label, form_color = "😐 COLD", "#ff9800"
    else:
        form_label, form_color = "❄️ ICE", "#f44336"
    
    def_mult = features.def_mult if features.def_mult else 1.0
    if def_mult >= 1.10:
        matchup_label, matchup_color = "🧈 SOFT", "#00c853"
    elif def_mult >= 1.02:
        matchup_label, matchup_color = "✅ GOOD", "#4caf50"
    elif def_mult >= 0.95:
        matchup_label, matchup_color = "😐 AVG", "#ff9800"
    else:
        matchup_label, matchup_color = "🧱 TOUGH", "#f44336"
    
    is_b2b = features.is_b2b if hasattr(features, 'is_b2b') else False
    if is_b2b:
        rest_label, rest_color = "😴 B2B", "#f44336"
    elif rest_days >= 3:
        rest_label, rest_color = "💪 RESTED", "#00c853"
    elif rest_days == 2:
        rest_label, rest_color = "✅ FRESH", "#4caf50"
    else:
        rest_label, rest_color = "😐 STD", "#ff9800"

    # Build Evidence Row HTML
    evidence_row = f"""<div style="display: flex; justify-content: space-around; text-align: center; margin-top: 12px; padding-top: 12px; border-top: 1px solid #333;">
<div>
<div style="font-size: 14px; font-weight: 700; color: {form_color};">{form_label}</div>
<div style="font-size: 10px; color: #888;">L5: {l5_hit:.0%}</div>
<div style="font-size: 9px; color: #666; text-transform: uppercase;">Form</div>
</div>
<div style="border-left: 1px solid #444; padding-left: 15px;">
<div style="font-size: 14px; font-weight: 700; color: {matchup_color};">{matchup_label}</div>
<div style="font-size: 10px; color: #888;">×{def_mult:.2f}</div>
<div style="font-size: 9px; color: #666; text-transform: uppercase;">Matchup</div>
</div>
<div style="border-left: 1px solid #444; padding-left: 15px;">
<div style="font-size: 14px; font-weight: 700; color: {rest_color};">{rest_label}</div>
<div style="font-size: 10px; color: #888;">{rest_days}d rest</div>
<div style="font-size: 9px; color: #666; text-transform: uppercase;">Fatigue</div>
</div>
</div>"""

    # Safety Lock Row (Rust + Injury Link)
    safety_row = f"""<div style="text-align: center; margin-top: 10px; font-size: 12px;">
{rust_alert}
<a href="{news_url}" target="_blank" style="color: #4dabf7; text-decoration: none; font-weight: bold; border: 1px solid #4dabf7; padding: 4px 10px; border-radius: 12px;">🏥 Check Injury Status</a>
</div>"""

    # --- INJURY-AWARE: Show teammate injuries boosting this player's usage ---
    injury_boost_row = ""
    if hasattr(result, 'injured_teammates') and result.injured_teammates:
        injured_list = ", ".join(result.injured_teammates)
        boost_pct = (result.injury_usage_boost - 1.0) * 100
        injury_boost_row = f"""<div style="background: rgba(76, 175, 80, 0.15); border: 1px solid #4caf50; border-radius: 6px; padding: 8px; margin-top: 10px; text-align: center;">
<span style="color: #4caf50; font-weight: bold; font-size: 12px;">📈 USAGE BOOST +{boost_pct:.0f}%</span>
<div style="color: #888; font-size: 10px; margin-top: 4px;">Teammate(s) OUT: {injured_list}</div>
</div>"""

    # 2. HTML Construction (FLUSH LEFT)
    ticket_html = f"""<div style="background: linear-gradient(135deg, {bg_color} 0%, rgba(30,30,30,0.9) 100%); border-left: 5px solid {side_color}; border-radius: 12px; padding: 16px; margin: 10px 0; font-family: sans-serif; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
<div>
<div style="font-size: 10px; color: #888; text-transform: uppercase; letter-spacing: 1px;">NBA Prop</div>
<div style="font-size: 20px; font-weight: 700; color: white;">{result.player_name}</div>
<div style="color: #aaa; font-size: 12px;">vs {result.opponent_name}</div>
</div>
<div style="text-align: right;">
<div style="background: {grade_color}; color: white; padding: 4px 12px; border-radius: 16px; font-weight: 700; font-size: 14px;">Grade {grade_val}</div>
<div style="margin-top: 4px; font-size: 12px;">{robot_badge}</div>
</div>
</div>
<div style="background: rgba(255,255,255,0.05); border-radius: 8px; padding: 12px; margin: 10px 0; text-align: center;">
<div style="font-size: 12px; color: #888;">{result.market}</div>
<div style="font-size: 28px; font-weight: 800; color: {side_color};">{side_icon} {decision.recommended_side} {result.line}</div>
<div style="font-size: 12px; color: #aaa;">Proj: {projection.final_projection:.1f} | Odds: {result.odds}</div>
</div>
<div style="display: flex; justify-content: space-between; text-align: center; margin-bottom: 12px; font-size: 14px;">
<div><span style="color: {'#00c853' if decision.probability >= 0.6 else '#ff9800'}; font-weight: 700;">{decision.probability:.0%}</span> <span style="font-size: 10px; color: #888;">Win</span></div>
<div><span style="color: {'#00c853' if decision.expected_value > 0 else '#f44336'}; font-weight: 700;">{decision.expected_value:+.1%}</span> <span style="font-size: 10px; color: #888;">EV</span></div>
<div><span style="color: {'#00c853' if robot_prob >= 0.6 else '#ff9800' if robot_prob >= 0.5 else '#f44336'}; font-weight: 700;">{robot_prob:.0%}</span> <span style="font-size: 10px; color: #888;">🤖 ML</span></div>
<div><span style="color: white; font-weight: 700;">₱{decision.kelly_stake:.0f}</span> <span style="font-size: 10px; color: #888;">Stake</span></div>
</div>
{evidence_row}
{injury_boost_row}
{safety_row}
{conflict_div}
</div>"""

    # 3. Render
    st.markdown(ticket_html, unsafe_allow_html=True)
    
    # ML Signal expander
    if projection.ml_prob is not None:
        with st.expander("🧠 ML Decoder", expanded=False):
            render_ml_decoder(result)


def render_ml_decoder(result: AnalysisResult):
    """Render the ML signal decoder panel."""
    features = result.features
    projection = result.projection
    
    # ROW 1: RISK FACTORS
    st.caption("🛡️ **Risk Profile**")
    r1, r2, r3 = st.columns(3)
    
    with r1:  # Blowout
        prob = features.blowout_prob * 100
        if prob >= 35: 
            label, color = "HIGH", "red"
        elif prob >= 20: 
            label, color = "MED", "orange"
        else: 
            label, color = "LOW", "green"
        st.metric("Blowout Risk", f"{prob:.0f}%", label, delta_color="inverse" if prob >= 20 else "normal")
    
    with r2:  # Volatility
        cv = features.coef_variation * 100
        if cv >= 30: 
            label, color = "HIGH", "red"
        elif cv >= 20: 
            label, color = "MED", "orange"
        else: 
            label, color = "LOW", "green"
        st.metric("Volatility (CV)", f"{cv:.0f}%", label, delta_color="inverse" if cv >= 30 else "normal")
    
    with r3:  # Fatigue
        fatigue = features.personal_fatigue_factor
        if fatigue < 0.92: 
            label = "FATIGUED"
        elif fatigue > 1.02: 
            label = "RESTED"
        else: 
            label = "NORMAL"
        st.metric("Fatigue Factor", f"{fatigue:.2f}", label)
    
    # ROW 2: EDGE FACTORS
    st.caption("📊 **Edge Factors**")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Pace", f"{features.pace_mult:.2f}")
    e2.metric("Defense", f"{features.def_mult:.2f}")
    e3.metric("Position", f"{features.position_mult:.2f}")
    e4.metric("Matchup", f"{features.combined_matchup_mult:.2f}")


def render_recommendation_card(result: AnalysisResult, bankroll: float):
    """
    Render clean, mobile-friendly recommendation card.

    FIXED: explicit labels to prevent 'Over/Under' confusion.
    """
    decision = result.decision
    projection = result.projection
    features = result.features
    simulation = result.simulation
    
    # Grade colors
    grade_colors = {'A': 'green', 'B': 'green', 'C': 'orange', 'D': 'red', 'F': 'red'}
    grade_color = grade_colors.get(decision.grade.value, 'gray')
    
    # --- 1. HEADER & CONFLICT LOGIC ---
    
    # Detect Conflict: Sim recommends a side, but ML predicts a Loss (<50%)
    is_conflict = projection.ml_prob is not None and projection.ml_prob < 0.50
    
    # Check blowout risk for the lightning bolt icon
    blowout_warn = " ⚡" if features.blowout_prob >= 0.25 else ""
    
    if is_conflict:
        # CONFLICT MODE: Big Warning Banner
        opposing_side = "UNDER" if decision.recommended_side == "OVER" else "OVER"
        st.error(f"⚠️ **CONFLICT:** Sim wants {decision.recommended_side}, but ML leans {opposing_side}")
        # Orange question mark header to signify doubt
        st.markdown(f"### ❓ {decision.recommended_side} {result.line} — :{grade_color}[Grade {decision.grade.value}]")
    else:
        # NORMAL MODE: Standard Green/Clean Header
        st.markdown(f"### {decision.recommended_side} {result.line}{blowout_warn} — :{grade_color}[Grade {decision.grade.value}]")
    
    # --- 2. KEY METRICS ---
    
    if projection.ml_prob is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Win Prob", f"{decision.probability:.0%}")
        c2.metric("EV", f"{decision.expected_value:+.1%}")
        c3.metric("Stake", f"₱{decision.kelly_stake:.0f}", f"{decision.kelly_fraction:.1%}")
        
        # --- FIXED: DYNAMIC LABELING ---
        ml_pct = projection.ml_prob
        side = decision.recommended_side
        
        if ml_pct > 0.55:
            # High prob = ML Agrees with Sim
            ml_label = f"ML Likes {side}"
            ml_color = "normal" # Metric handles green automatically for positive delta
        elif ml_pct < 0.45:
             # Low prob = ML Disagrees (Likes the other side)
            opp_side = "UNDER" if side == "OVER" else "OVER"
            ml_label = f"ML Likes {opp_side}"
            # Invert the delta display so it makes sense (shows how much it hates the pick)
            ml_pct = 1.0 - ml_pct 
        else:
            ml_label = "ML Neutral"

        ml_delta = f"{(projection.ml_prob - 0.5) * 100:+.0f}pp"
        c4.metric(ml_label, f"{projection.ml_prob:.0%}", ml_delta)
        
    else:
        # Fallback if no ML model loaded
        c1, c2, c3 = st.columns(3)
        c1.metric("Win Prob", f"{decision.probability:.0%}")
        c2.metric("EV", f"{decision.expected_value:+.1%}")
        c3.metric("Stake", f"₱{decision.kelly_stake:.0f}", f"{decision.kelly_fraction:.1%}")
    
    # Context Text
    if projection.context:
        st.caption(projection.context)
    
    # --- 3. V15 SIGNAL DECODER (THE DASHBOARD) ---
    
    with st.expander("🧠 ML Signal Decoder", expanded=False):
        # ROW 1: RISK FACTORS (The "Killers")
        st.caption("🛡️ **Risk Profile**")
        r1, r2, r3 = st.columns(3)
        
        with r1: # Blowout
            prob = features.blowout_prob * 100
            if prob >= 35: 
                label, color = "HIGH", "red"
            elif prob >= 20: 
                label, color = "MOD", "orange"
            else: 
                label, color = "LOW", "green"
            st.markdown(f"**Blowout Risk:** :{color}[{label}] ({prob:.0f}%)")
            
        with r2: # Fatigue
            fatigue = features.personal_fatigue_factor
            if fatigue < 0.95: 
                val, col = "Fatigued", "red"
            elif fatigue > 1.02: 
                val, col = "Rested", "green"
            else: 
                val, col = "Normal", "gray"
            st.markdown(f"**Fatigue:** :{col}[{val}] ({(fatigue-1)*100:+.0f}%)")
            
        with r3: # Volatility (CV)
            cv = features.coef_variation * 100
            if cv < 20: 
                val, col = "Steady", "green"
            elif cv > 35: 
                val, col = "Chaos", "red"
            else: 
                val, col = "Normal", "gray"
            st.markdown(f"**Variance:** :{col}[{val}] ({cv:.0f}%)")

        st.divider()

        # ROW 2: PERFORMANCE DRIVERS (The "Boosters")
        st.caption("🚀 **Edge Drivers**")
        d1, d2, d3 = st.columns(3)
        
        with d1: # Matchup
            mult = features.combined_matchup_mult
            delta = (mult - 1.0) * 100
            if mult >= 1.05: 
                st.metric("Matchup", "Great", f"{delta:+.1f}%")
            elif mult <= 0.95: 
                st.metric("Matchup", "Tough", f"{delta:+.1f}%")
            else: 
                st.metric("Matchup", "Neutral", f"{delta:+.1f}%")
            
        with d2: # Form
            trend = features.trend * 100
            if trend >= 10: 
                st.metric("Form", "Heating Up", f"{trend:+.1f}%")
            elif trend <= -10: 
                st.metric("Form", "Cold", f"{trend:+.1f}%")
            else: 
                st.metric("Form", "Stable", f"{trend:+.1f}%")
            
        with d3: # Game Context
            total = features.game_total
            if total >= 235: 
                st.metric("Game Env", "Shootout", f"{total:.0f}")
            elif total <= 215: 
                st.metric("Game Env", "Grind", f"{total:.0f}")
            else: 
                st.metric("Game Env", "Standard", f"{total:.0f}")

    # --- 4. FOOTER & ALERTS ---
    
    render_data_quality_card(result)
    
    if decision.confidence_warning:
        st.warning(f"⚠️ {decision.confidence_warning}")
    
    if decision.rollover_suitable:
        st.success(f"🎲 **Parlay OK** — Score: {decision.rollover_score:.1f}/5")


def render_distribution_chart(result: AnalysisResult):
    """Render simulation distribution chart."""
    simulation = result.simulation
    projection = result.projection

    fig, ax = plt.subplots(figsize=(10, 2.5))
    n, bins, patches = ax.hist(simulation.simulations, bins=50, color='skyblue', alpha=0.7, density=True)
    
    for i, patch in enumerate(patches):
        if bins[i] > result.line:
            patch.set_facecolor('#28a745')
            patch.set_alpha(0.6)
        else:
            patch.set_facecolor('#dc3545')
            patch.set_alpha(0.6)
    
    ax.axvline(result. line, color='white', linestyle='--', linewidth=2, label=f"Line ({result.line})")
    ax.axvline(projection.final_projection, color='yellow', linestyle='-', linewidth=2, label=f"Proj ({projection.final_projection:.1f})")
    ax.axvspan(simulation.ci_10, simulation.ci_90, alpha=0.2, color='white', label='90% CI')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel(result.market)
    ax.set_title('Distribution of 10,000 Simulated Games', fontsize=10, pad=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_blowout_info(result: AnalysisResult):
    """Render blowout protection info with V15 probability model."""
    features = result.features
    
    # V15: Use blowout probability instead of factor threshold
    if features.blowout_prob < 0.10:
        return  # No significant blowout risk
    
    # Color code based on probability
    if features.blowout_prob >= 0.35:
        risk_icon = "🔴"
        risk_level = "HIGH"
    elif features.blowout_prob >= 0.25:
        risk_icon = "🟠"
        risk_level = "MODERATE"
    elif features.blowout_prob >= 0.15:
        risk_icon = "🟡"
        risk_level = "SLIGHT"
    else:
        risk_icon = "⚪"
        risk_level = "MINIMAL"
    
    pct_reduction = (1 - features.blowout_factor) * 100
    is_star = features.avg_minutes >= CONFIG.STAR_MINUTES_THRESHOLD
    star_text = " (Star)" if is_star else ""
    prob_pct = features.blowout_prob * 100
    
    st.warning(f"{risk_icon} **{risk_level} Blowout Risk{star_text}** — P(<25min): {prob_pct:.0f}% | Projection reduced {pct_reduction:.0f}% (Spread: {features.spread:+.1f})")


def render_backtest_tab(orchestrator: PredictionOrchestrator):
    """Render compact backtesting tab."""
    st.markdown("### 📊 Model Backtest")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        all_players = players.get_players()
        player_names = sorted([p['full_name'] for p in all_players if p. get('is_active', True)])
        bt_player = st.selectbox("Player", player_names, index=None, placeholder="Search...", key="bt_player")
    with col2:
        bt_market = st.selectbox("Market", ["PTS", "REB", "AST", "PRA", "3PM"], key="bt_market")
    with col3:
        bt_days = st.number_input("Games", 10, 50, 30, key="bt_days")
    
    with st.expander("⚙️ Advanced", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            bt_lookback = st.slider("Lookback", 5, 20, 15, key="bt_lookback")
        with c2:
            bt_offset = st.number_input("Line Offset", -5.0, 5.0, 0.0, 0.5, key="bt_offset")
        with c3:
            bt_spread = st.number_input("Spread", -20.0, 20.0, 0.0, 0.5, key="bt_spread")
    
    if st.button("🚀 Run Backtest", type="primary", disabled=not bt_player):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(pct):
            progress_bar.progress(pct)
            status_text.text(f"Processing... {pct:.0%}")
        
        with st.spinner("Running backtest..."):
            summary = orchestrator.run_backtest(
                player_name=bt_player,
                market=bt_market,
                lookback=bt_lookback,
                test_days=bt_days,
                line_offset=bt_offset,
                fixed_spread=bt_spread,
                progress_callback=update_progress
            )
        
        progress_bar.empty()
        status_text.empty()
        
        if summary is None:
            st.error("Backtest failed - insufficient data.")
            return
        
        # Compact results display
        st.success(f"✅ {summary.total_predictions} predictions analyzed")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Win Rate", f"{summary.win_rate:.0%}")
        c2.metric("ROI", f"{summary.roi:+.1%}")
        c3.metric("Brier", f"{summary.brier_score:.3f}")
        c4.metric("Record", f"{summary.wins}-{summary.losses}")
        
        # Results in expander
        with st.expander("📊 Detailed Results", expanded=True):
            if summary.calibration_by_bucket:
                st.markdown("**Calibration:**")
                cal_data = []
                for bucket, data in summary.calibration_by_bucket.items():
                    cal_data.append({
                        'Bucket': bucket,
                        'Pred': f"{data['predicted']:.0%}",
                        'Actual': f"{data['actual']:.0%}",
                        'N': data['count'],
                    })
                st.dataframe(pd.DataFrame(cal_data), hide_index=True, width="stretch")
            
            if summary.grade_performance:
                st.markdown("**By Grade:**")
                grade_data = []
                for grade, data in summary.grade_performance.items():
                    grade_data.append({
                        'Grade': grade,
                        'N': data['count'],
                        'Win%': f"{data['win_rate']:.0%}",
                        'ROI': f"{data['roi']:+.1%}"
                    })
                st.dataframe(pd.DataFrame(grade_data), hide_index=True, width="stretch")
            
            st.dataframe(summary.results_df, hide_index=True, width="stretch")
            
            csv = summary.results_df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, f"backtest_{bt_player.replace(' ', '_')}.csv", "text/csv")

def load_watchlist():
    if WATCHLIST_FILE.exists():
        try:
            with open(WATCHLIST_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_watchlist(data):
    try:
        with open(WATCHLIST_FILE, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        st.error(f"Failed to save watchlist: {e}")

def render_watchlist_tab(orchestrator):
    """
    Renders the watchlist tab with Line Input and auto-calculated Offsets (EMA-based).
    """
    st.markdown("### 👀 Watchlist")
    
    # 1. Load Watchlist
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = load_watchlist()

    # 2. Add New Player Row
    with st.container(border=True):
        # Columns: Player(3) | Stat(1) | Line(1) | Button(1)
        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        
        with c1:
            all_players = players.get_players()
            active_players = [p['full_name'] for p in all_players if p.get('is_active', True)]
            target_player = st.selectbox(
                "Add Player", 
                active_players, 
                key="wl_player_select", 
                index=None, 
                placeholder="Search Name..."
            )
        
        with c2:
            target_market = st.selectbox(
                "Stat", 
                ["PTS", "REB", "AST", "PRA", "3PM"], 
                key="wl_market_select"
            )
            
        with c3:
            # Line Input for initial add
            target_line = st.number_input(
                "Line", 
                min_value=0.5, 
                max_value=100.0, 
                value=15.5, 
                step=0.5, 
                key="wl_line_input"
            )

        with c4:
            st.write("") # Spacer to align button
            if st.button("Add ➕", use_container_width=True) and target_player:
                # Add to session state if not duplicate
                if not any(x['player'] == target_player and x['market'] == target_market for x in st.session_state.watchlist):
                    import uuid
                    st.session_state.watchlist.append({
                        'id': str(uuid.uuid4())[:8],
                        'player': target_player, 
                        'market': target_market, 
                        'line': target_line,  # Save the line
                        'added_date': datetime.now().strftime("%Y-%m-%d")
                    })
                    save_watchlist(st.session_state.watchlist)
                    st.rerun()

    # 3. Display Watchlist Items
    if not st.session_state.watchlist:
        st.info("Your watchlist is empty. Add players above to calculate offsets.")
    else:
        st.write("---")
        # Header Row
        h1, h2, h3, h4 = st.columns([2, 1, 2, 0.5])
        h1.caption("Player")
        h2.caption("Target Line")
        h3.caption("Trends & Offset")
        
        for i, item in enumerate(st.session_state.watchlist):
            item_id = item.get('id')
            if not item_id:
                import uuid
                item_id = str(uuid.uuid4())[:8]
                st.session_state.watchlist[i]['id'] = item_id
            
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([2, 1, 2, 0.5])
                
                with c1:
                    st.subheader(f"{item['player']}")
                    st.caption(f"Market: **{item['market']}**")
                
                with c2:
                    # Editable Line Input
                    current_line = st.number_input(
                        "Line", 
                        value=float(item.get('line', 10.5)), 
                        step=0.5, 
                        key=f"line_{item_id}",
                        label_visibility="collapsed"
                    )
                    
                    # Update line in storage if changed
                    if current_line != item.get('line'):
                        st.session_state.watchlist[i]['line'] = current_line
                        save_watchlist(st.session_state.watchlist)

                with c3:
                    # Calculate Stats & Offset
                    try:
                        # Find player ID safely
                        p_obj = next((p for p in all_players if p['full_name'].lower() == item['player'].lower()), None)
                        
                        if p_obj:
                            df = orchestrator.data_loader.fetch_game_logs(p_obj['id'])
                            if not df.empty and item['market'] in df.columns:
                                # Calculate EMA (Span 10 for quick trend)
                                ema = df[item['market']].ewm(span=10).mean().iloc[-1]
                                
                                # Calculate Line Offset
                                # Positive = Player is projecting OVER the line
                                offset = ema - current_line
                                
                                offset_color = "green" if offset > 0 else "red"
                                offset_icon = "🔥" if offset > 0 else "❄️"
                                
                                # Display EMA (Not L5)
                                st.markdown(f"**EMA:** {ema:.1f}")
                                st.markdown(f"**Offset:** :{offset_color}[{offset:+.1f}] {offset_icon}")
                            else:
                                st.caption("No recent data")
                        else:
                            st.caption("Player not found")
                    except Exception:
                        st.caption("Stats unavailable")

                with c4:
                    if st.button("🗑️", key=f"del_{item_id}"):
                        st.session_state.watchlist.pop(i)
                        save_watchlist(st.session_state.watchlist)
                        st.rerun()

def render_parlay_tab(parlay_tracker: ParlayTracker, bankroll: float):
    """Render compact parlay tab."""
    st.markdown("### 🎲 Parlay Builder")
    
    builder = st.session_state.get('parlay_builder', [])
    
    if builder:
        combined_odds, combined_prob = 1.0, 1.0
        for leg in builder: 
            combined_odds *= leg['odds']
            combined_prob *= leg['prob']
        
        for i, leg in enumerate(builder):
            with st.container(border=True):
                c1, c2 = st.columns([5, 1])
                c1.markdown(f"**{leg['player']}** {leg['side']} {leg['line']} @ {leg['odds']:.2f}")
                if c2.button("❌", key=f"remove_leg_{i}"):
                    st.session_state.parlay_builder.pop(i)
                    st.rerun()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Odds", f"{combined_odds:.2f}")
        c2.metric("Prob", f"{combined_prob:.1%}")
        c3.metric("Payout", f"₱{50 * combined_odds:,.0f}")
        
        c1, c2 = st.columns(2)
        with c1:
            parlay_stake = st.number_input("Stake", 10.0, float(bankroll), 50.0, 10.0, key="parlay_stake")
        with c2:
            parlay_name = st.text_input("Name", placeholder="Optional...", key="parlay_name")
        
        c1, c2 = st.columns(2)
        if c1.button("✅ Create", type="primary", use_container_width=True):
            parlay_tracker.create_parlay(legs=builder.copy(), stake=parlay_stake, name=parlay_name)
            st.session_state.parlay_builder = []
            st.toast("Parlay created!", icon="🎲")
            st.rerun()
        if c2.button("🗑️ Clear", use_container_width=True):
            st.session_state.parlay_builder = []
            st.rerun()
    else:
        st.info("Add legs from Analyze tab.")
    
    # History
    with st.expander("📜 History", expanded=False):
        stats = parlay_tracker.get_stats()
        if stats['total_parlays'] > 0:
            c1, c2, c3 = st.columns(3)
            c1.metric("Win%", f"{stats['win_rate']:.0%}")
            c2.metric("P/L", f"₱{stats['total_profit']:+,.0f}")
            c3.metric("Legs", f"{stats['total_legs_hit']}/{stats['total_legs']}")
        
        parlays = parlay_tracker.get_parlays()
        for parlay in sorted(parlays, key=lambda x: x['id'], reverse=True):
            result = parlay['result']
            icon = "✅" if result == 'Win' else "❌" if result == 'Loss' else "⏳"
            
            with st.container(border=True):
                col1, col2, col3 = st.columns([3, 1, 0.5])
                with col1:
                    parlay_name = parlay.get('name') or f"#{parlay['id']}"
                    st.markdown(f"{icon} **{parlay_name}** ({parlay['num_legs']})")
                    st.caption(f"₱{parlay['stake']:.0f} @ {parlay['combined_odds']:.2f}")
                with col2:
                    if result == 'Win': 
                        st.markdown(f"**+₱{parlay['actual_payout'] - parlay['stake']:,.0f}**")
                    elif result == 'Loss':
                        st.markdown(f"**-₱{parlay['stake']:,.0f}**")
                    else:
                        st.markdown(f"*₱{parlay['potential_profit']:,.0f}*")
                with col3:
                    if st.button("🗑️", key=f"del_parlay_{parlay['id']}"):
                        parlay_tracker.delete_parlay(parlay['id'])
                        st.rerun()
                
                with st.expander(f"Legs ({parlay['legs_hit']}/{parlay['legs_decided']})"):
                    for i, leg in enumerate(parlay['legs']):
                        leg_result = leg.get('result', 'Pending')
                        leg_icon = "✅" if leg_result == 'Win' else "❌" if leg_result == 'Loss' else "🔄" if leg_result == 'Push' else "⏳"
                        
                        c1, c2 = st.columns([4, 1])
                        c1.markdown(f"{leg_icon} {leg['player']} {leg['side']} {leg['line']}")
                        new_result = c2.selectbox(
                            "Result", ["Pending", "Win", "Loss", "Push"],
                            index=["Pending", "Win", "Loss", "Push"].index(leg_result),
                            key=f"leg_{parlay['id']}_{i}",
                            label_visibility="collapsed"
                        )
                        if new_result != leg_result:
                            parlay_tracker.update_leg_result(parlay['id'], i, new_result)
                            st.rerun()
        
        if st.button("🗑️ Clear All", key="clear_parlays"):
            parlay_tracker.clear_history()
            st.rerun()


def render_ml_data_tab(tracker: Tracker):
    """Render enhanced ML data tab with better preview."""
    st.markdown("### 🤖 ML Training Data")
    
    csv_path = DATA_DIR / "ml_training_data.csv"
    
    # =========================================================================
    # SECTION 1: THEORETICAL PERFORMANCE (The Math - from CSV)
    # =========================================================================
    with st.expander("📐 Theoretical Performance (The Math)", expanded=False):
        st.caption("Based on simulation data from ml_training_data.csv")
        
        if csv_path.exists():
            try:
                csv_df = pd.read_csv(csv_path, on_bad_lines='skip')
                skipped_warning = False
                
                # Quick check for corruption
                try:
                    with open(csv_path, 'r', encoding='utf-8') as f:
                        raw_line_count = sum(1 for _ in f)
                    if raw_line_count > len(csv_df) + 10:
                        skipped_warning = True
                        st.warning(f"⚠️ CSV has {raw_line_count - len(csv_df) - 1} corrupted rows. Click 'Repair CSV' below.")
                except:
                    pass
                
                if 'hit' in csv_df.columns:
                    total_sim_bets = len(csv_df)
                    sim_wins = int(csv_df['hit'].sum())
                    sim_losses = total_sim_bets - sim_wins
                    sim_win_rate = sim_wins / total_sim_bets if total_sim_bets > 0 else 0
                    
                    # --- Theoretical Win Rate (Raw) ---
                    # Standard -110 odds = 1.91 decimal
                    sim_units_won = sim_wins * 0.91
                    sim_units_lost = sim_losses * 1.0
                    sim_net_units = sim_units_won - sim_units_lost
                    sim_roi = (sim_net_units / total_sim_bets * 100) if total_sim_bets > 0 else 0
                    
                    # --- Quality Units (Only bets where predicted_prob > 60%) ---
                    quality_df = csv_df[csv_df['predicted_prob'] > 0.60] if 'predicted_prob' in csv_df.columns else pd.DataFrame()
                    if len(quality_df) > 0:
                        quality_total = len(quality_df)
                        quality_wins = int(quality_df['hit'].sum())
                        quality_losses = quality_total - quality_wins
                        quality_win_rate = quality_wins / quality_total if quality_total > 0 else 0
                        quality_units_won = quality_wins * 0.91
                        quality_units_lost = quality_losses * 1.0
                        quality_net_units = quality_units_won - quality_units_lost
                        quality_roi = (quality_net_units / quality_total * 100) if quality_total > 0 else 0
                    else:
                        quality_total = 0
                        quality_win_rate = 0
                        quality_net_units = 0
                        quality_roi = 0
                    
                    # Display: Theoretical (Raw) vs Quality (Filtered)
                    st.markdown("**📊 Theoretical Win Rate (All Sim Bets)**")
                    th_c1, th_c2, th_c3, th_c4 = st.columns(4)
                    th_c1.metric("Total Samples", f"{total_sim_bets:,}")
                    th_c2.metric("Win Rate", f"{sim_win_rate:.1%}")
                    if sim_net_units >= 0:
                        th_c3.metric("Net Units", f"+{sim_net_units:.1f}u")
                    else:
                        th_c3.metric("Net Units", f"{sim_net_units:.1f}u")
                    th_c4.metric("ROI", f"{sim_roi:+.1f}%")
                    
                    st.markdown("---")
                    st.markdown("**🎯 Quality Units (Only Prob > 60%)**")
                    st.caption("What would happen if you only bet the strong signals")
                    q_c1, q_c2, q_c3, q_c4 = st.columns(4)
                    q_c1.metric("Filtered Bets", f"{quality_total:,}")
                    q_c2.metric("Win Rate", f"{quality_win_rate:.1%}")
                    if quality_net_units >= 0:
                        q_c3.metric("Quality Units", f"+{quality_net_units:.1f}u", delta="Strong signals only")
                    else:
                        q_c3.metric("Quality Units", f"{quality_net_units:.1f}u", delta="Strong signals only", delta_color="inverse")
                    q_c4.metric("ROI", f"{quality_roi:+.1f}%")
                else:
                    st.info("CSV missing 'hit' column. Cannot calculate theoretical performance.")
                
                # Repair button
                if skipped_warning:
                    if st.button("🔧 Repair CSV", type="primary"):
                        clean_df = pd.read_csv(csv_path, on_bad_lines='skip')
                        backup_path = DATA_DIR / "ml_training_data_backup.csv"
                        import shutil
                        shutil.copy(csv_path, backup_path)
                        clean_df.to_csv(csv_path, index=False)
                        st.success(f"✅ Repaired! Kept {len(clean_df):,} valid rows. Backup saved to {backup_path.name}")
                        st.rerun()
                    
            except Exception as e:
                st.warning(f"Could not read CSV file: {e}")
        else:
            st.info("📂 No training data file found. Generate data using backtest or track bets to create one.")
    
    # =========================================================================
    # SECTION 2: REAL PERFORMANCE (The Human - from tracker.history ONLY)
    # =========================================================================
    with st.expander("🧑 Real Performance (Your Actual Bets)", expanded=True):
        st.caption("Based on YOUR tracked bets in bet_tracker.json - this is what actually happened")
        
        tracker_bets = tracker.get_bets()
        decided_bets = [b for b in tracker_bets if b.get('result') in ['Win', 'Loss']]
        
        if len(decided_bets) > 0:
            real_wins = len([b for b in decided_bets if b.get('result') == 'Win'])
            real_losses = len([b for b in decided_bets if b.get('result') == 'Loss'])
            real_total = real_wins + real_losses
            real_win_rate = real_wins / real_total if real_total > 0 else 0
            
            # Calculate real units (using actual odds from tracked bets)
            real_units_profit = 0
            for bet in decided_bets:
                odds = bet.get('odds', 1.91)  # Default to -110 if missing
                if bet.get('result') == 'Win':
                    real_units_profit += (odds - 1)  # Win pays (odds - 1) units
                else:
                    real_units_profit -= 1  # Loss costs 1 unit
            
            real_roi = (real_units_profit / real_total * 100) if real_total > 0 else 0
            
            # Display real performance metrics
            r_c1, r_c2, r_c3, r_c4 = st.columns(4)
            r_c1.metric("Your Bets", f"{real_total:,}")
            r_c2.metric("Win Rate", f"{real_win_rate:.1%}")
            if real_units_profit >= 0:
                r_c3.metric("Net Units", f"+{real_units_profit:.1f}u", delta="Real $$$")
            else:
                r_c3.metric("Net Units", f"{real_units_profit:.1f}u", delta="Real $$$", delta_color="inverse")
            r_c4.metric("ROI", f"{real_roi:+.1f}%")
            
            # Show pending count
            pending_bets = [b for b in tracker_bets if b.get('result', 'Pending') == 'Pending']
            if len(pending_bets) > 0:
                st.caption(f"📋 {len(pending_bets)} pending bets awaiting results")
            
            # Real performance chart (cumulative units over time)
            st.markdown("**📈 Cumulative Units Over Time**")
            
            # Sort by date and calculate cumulative units
            chart_data = []
            cumulative = 0
            sorted_bets = sorted(decided_bets, key=lambda x: x.get('date', ''))
            
            for bet in sorted_bets:
                odds = bet.get('odds', 1.91)
                if bet.get('result') == 'Win':
                    cumulative += (odds - 1)
                else:
                    cumulative -= 1
                chart_data.append({
                    'date': bet.get('date', 'Unknown'),
                    'units': cumulative,
                    'player': bet.get('player_name', bet.get('player', 'Unknown')),
                    'result': bet.get('result')
                })
            
            if chart_data:
                chart_df = pd.DataFrame(chart_data)
                # Color based on positive/negative
                line_color = "#00ff00" if cumulative >= 0 else "#ff4444"
                st.line_chart(chart_df.set_index('date')['units'], color=line_color)
        else:
            st.info("📋 No decided bets yet. Track bets and update their results to see your real performance.")
    
    st.markdown("---")
    
    feature_stats = tracker.get_feature_stats()
    training_df = tracker.export_training_data()
    
    # Top-level stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples", feature_stats['total_with_features'])
    c2.metric("Ready (W/L)", feature_stats['decided_with_features'])
    c3.metric("Pending", feature_stats['pending_with_features'])
    
    if len(training_df) > 0:
        win_rate = training_df['target'].mean() if 'target' in training_df.columns else 0
        c4.metric("Win Rate", f"{win_rate:.0%}")
        
        # Data quality summary
        with st.expander("📊 Data Quality Summary", expanded=False):
            sq1, sq2, sq3 = st.columns(3)
            
            # Feature columns count
            feat_cols = [c for c in training_df.columns if c.startswith('feat_')]
            sq1.metric("Feature Columns", len(feat_cols))
            
            # Date range
            if 'timestamp' in training_df.columns:
                try:
                    dates = pd.to_datetime(training_df['timestamp'])
                    sq2.metric("Date Range", f"{dates.min().strftime('%m/%d')} - {dates.max().strftime('%m/%d')}")
                except:
                    sq2.metric("Date Range", "N/A")
            else:
                sq2.metric("Date Range", "N/A")
            
            # Avg margin if available
            if 'margin' in training_df.columns:
                avg_margin = training_df['margin'].mean()
                sq3.metric("Avg Margin", f"{avg_margin:+.1f}")
            else:
                sq3.metric("Avg Margin", "N/A")
            
            # Result quality breakdown
            if 'result_quality' in training_df.columns:
                st.markdown("**Result Quality Breakdown:**")
                quality_counts = training_df['result_quality'].value_counts()
                
                # Losses
                loss_cols = st.columns(4)
                loss_cols[0].write(f"💔 Bad Beat: {quality_counts.get('bad_beat', 0)}")
                loss_cols[1].write(f"😤 Close Loss: {quality_counts.get('close_loss', 0)}")
                loss_cols[2].write(f"📉 Clear Loss: {quality_counts.get('clear_loss', 0)}")
                loss_cols[3].write(f"🚫 Bad Read: {quality_counts.get('bad_read', 0)}")
                
                # Wins
                win_cols = st.columns(4)
                win_cols[0].write(f"😅 Sweat Win: {quality_counts.get('sweat_win', 0)}")
                win_cols[1].write(f"✌️ Close Win: {quality_counts.get('close_win', 0)}")
                win_cols[2].write(f"💪 Solid Win: {quality_counts.get('solid_win', 0)}")
                win_cols[3].write(f"🔥 Blowout Win: {quality_counts.get('blowout_win', 0)}")
        
        # Tabbed data views
        view_tab1, view_tab2, view_tab3 = st.tabs(["📋 Summary", "🔢 Features", "📄 Raw Data"])
        
        with view_tab1:
            # Key columns for quick overview
            summary_cols = ['player', 'market', 'line', 'side', 'result', 'margin', 'result_quality', 'feat_prob', 'feat_ev']
            display_cols = [c for c in summary_cols if c in training_df.columns]
            if display_cols:
                st.dataframe(training_df[display_cols].tail(20), hide_index=True,width="stretch")
            else:
                st.info("No summary columns available")
        
        with view_tab2:
            # All feature columns
            feat_cols = [c for c in training_df.columns if c.startswith('feat_')]
            if feat_cols:
                # Add identifier columns
                id_cols = ['player', 'market', 'result']
                show_cols = [c for c in id_cols if c in training_df.columns] + feat_cols
                st.caption(f"Showing {len(feat_cols)} feature columns")
                st.dataframe(training_df[show_cols].tail(20), hide_index=True, width="stretch")
            else:
                st.info("No feature columns found")
        
        with view_tab3:
            # Full raw data with column count
            st.caption(f"All {len(training_df.columns)} columns × {len(training_df)} rows")
            st.dataframe(training_df.tail(30), hide_index=True, width="stretch")
        
        # Download buttons
        st.markdown("---")
        dc1, dc2, dc3 = st.columns(3)
        csv_data = training_df.to_csv(index=False)
        dc1.download_button("📥 Download CSV", csv_data, "ml_training_data.csv", "text/csv", use_container_width=True)
        json_data = training_df.to_json(orient='records', indent=2)
        dc2.download_button("📥 Download JSON", json_data, "ml_training_data.json", "application/json", use_container_width=True)
        
        # Show column list
        with dc3:
            if st.button("📋 Show Columns", use_container_width=True):
                st.session_state['show_ml_cols'] = not st.session_state.get('show_ml_cols', False)
        
        if st.session_state.get('show_ml_cols', False):
            with st.expander("Available Columns", expanded=True):
                cols_by_type = {
                    'Identifiers': [c for c in training_df.columns if c in ['id', 'player', 'market', 'opponent', 'timestamp']],
                    'Bet Info': [c for c in training_df.columns if c in ['line', 'odds', 'side', 'stake', 'result', 'closing_line', 'actual_value', 'margin', 'result_quality']],
                    'Targets': [c for c in training_df.columns if c.startswith('target')],
                    'Features': [c for c in training_df.columns if c.startswith('feat_')]
                }
                for cat, cols in cols_by_type.items():
                    if cols:
                        st.markdown(f"**{cat}:** `{'`, `'.join(cols)}`")
        
        st.caption(f"💡 Need 100-200+ samples for XGBoost. Currently: {feature_stats['decided_with_features']}")
    else: 
        c4.metric("Win Rate", "N/A")
        st.info("Track bets to collect training data. Log bets with the 📝 Track button after analysis.")


def render_train_model_tab():
    """Streamlit UI for Training the V15 Model."""
    st.markdown("Train V15 Model")
    st.info("This will use 'ml_training_data.csv' to retrain the brain (nba_model.pkl).")
    
    # Check if data exists
    data_path = DATA_DIR / "ml_training_data.csv"
    
    if not data_path.exists():
        st.warning("Training data not found. Go to 'Generate Dataset' tab first.")
        return

    if st.button("🏋️ Start Training", type="primary"):
        status = st.status("Training in progress...", expanded=True)
        try:
            # 1. Load Data
            status.write("Loading dataset...")
            df = pd.read_csv(data_path)
            
            # Use canonical, ordered, numeric training features (36 exactly)
            try:
                # Access constant defined in ML generator section
                feature_cols = TRAINING_FEATURE_COLUMNS  # type: ignore[name-defined]
            except NameError:
                # Fallback in case of refactor – define inline (must match module-level constant)
                feature_cols = [
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
                    # V16 market & role context
                    'feat_odds_decimal', 'feat_usg_season', 'feat_clv'
                ]
            target_col = 'hit'

            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                status.update(label="❌ Error", state="error")
                st.error("Training data missing required features. Please regenerate dataset.")
                st.caption(f"Missing: {', '.join(missing)}")
                return

            # 2. Prepare Data
            status.write(f"Processing {len(df)} samples with {len(feature_cols)} features...")
            # Coerce to numeric to avoid dtype issues from mixed sources
            X = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
            y = pd.to_numeric(df[target_col], errors='coerce').fillna(0).astype(int)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 3. Train Model
            status.write("Feeding the XGBoost brain...")
            model = XGBClassifier(
                n_estimators=500, learning_rate=0.05, max_depth=4,
                subsample=0.8, colsample_bytree=0.8, eval_metric='logloss'
            )
            model.fit(X_train, y_train)
            
            # 4. Evaluate
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds)
            
            # 5. Save
            status.write("💾 Saving nba_model.pkl...")
            joblib.dump(model, DATA_DIR / "nba_model.pkl")
            
            status.update(label="✅ Training Complete!", state="complete")
            
            # Show Results
            st.success("Brain Updated Successfully!")
            c1, c2 = st.columns(2)
            c1.metric("Precision (Win%)", f"{prec:.1%}")
            c2.metric("Accuracy", f"{acc:.1%}")
            
            st.balloons()
            
        except Exception as e:
            status.update(label="❌ Training Failed", state="error")
            st.error(f"Error: {str(e)}")

def render_guide_tab():
    """Renders the comprehensive user guide."""
    st.markdown("### 📘 User Guide & Strategy Primer")
    st.caption(f"App Version: {CONFIG.CURRENT_SEASON} | {CURRENT_VERSION}")

    # --- SECTION 1: CORE WORKFLOW ---
    with st.expander("🚀 Quick Start: The Core Workflow", expanded=True):
        st.markdown("""
        1. **Analyze:** Go to **Analyze Tab**, pick a player/market, and hit "Run Analysis".
        2. **Adjust:** Use the **Usage Slider** in the sidebar if there are injuries (see rules below).
        3. **Decide:** Look for **Grade A** bets where both the **Sim** (Math) and **ML** (Robot) agree.
        4. **Track:** Click **"💾 Track"** to save the bet to your history.
        5. **Resolve:** Go to **Bets Tab** the next day, mark it Win/Loss, and enter the actual score.
        6. **Train:** Once you have 100+ bets, go to **ML Tab** to retrain the brain.
        """)

    # --- SECTION 2: THE USAGE SLIDER ---
    with st.expander("⚙️ The Usage Slider (Crucial)", expanded=True):
        st.info("The model knows averages, not news. You must tell it about injuries.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 📉 NEGATIVE ADJ (Injuries)")
            st.markdown("""
            * **-5% (Standard D2D):** Player is probable but has a minor ailment.
            * **-10% (GTD):** High risk of playing "decoy" or limited burst.
            * **-15% (Minutes Limit):** Strict cap (e.g., returning from 2-week absence).
            * **-20% (Rust):** First game back from major injury (>1 month).
            """)
        
        with c2:
            st.markdown("#### 📈 POSITIVE ADJ (Opportunity)")
            st.markdown("""
            * **+15% (Backup ➡️ Starter):** Backup PG/Center promoted to starter.
            * **+10% (Star Teammate OUT):** "Vacuum Effect" (e.g., Luka out, Kyrie +10%).
            * **+5% (Starter OUT):** "Trickle Down" to other starters.
            * **+5% (Must Win):** Playoff push or rivalry game (tighter rotation).
            """)

    # --- SECTION 3: INTERPRETING SIGNALS ---
    with st.expander("🧠 Robot vs. Math (Sim vs. ML)"):
        st.markdown("""
        **The Simulation (The Math):**
        * Runs 10,000 games using normal distributions, mixtures, and blowouts.
        * **Strength:** Great at finding "fair value" based on averages.
        * **Weakness:** Doesn't "learn" from specific losing patterns.

        **The ML Model (The Robot):**
        * An XGBoost Brain trained on YOUR past bets.
        * **Strength:** Detects traps (e.g., "Sim loves Over, but every time CV > 30% we lose").
        * **Weakness:** Needs data (100+ bets) to get smart.

        **⚠️ CONFLICT WARNING:**
        If you see **"⚠️ CONFLICT: Sim likes OVER, ML leans UNDER"**, it means the Math found an edge, but the Robot recognizes a losing pattern. **SKIP THE BET.**
        """)

    # --- SECTION 4: GRADING SYSTEM ---
    with st.expander("🏆 The Grading System"):
        st.markdown("""
        * **Grade A (5u):** EV > 5% **AND** Win Prob > 60%. (The "Slam Dunk").
        * **Grade B (3u):** EV > 2%. Solid value, standard play.
        * **Grade C (1u):** EV > 0%. Thin edge, only bet if you love the spot.
        * **Grade D/F (Pass):** Negative EV. The House has the edge. Do not bet.
        """)

    # --- SECTION 5: ML TRAINING LOOP ---
    with st.expander("🤖 How to Train the Brain"):
        st.markdown("""
        The app gets smarter the more you use it.
        1. **Log Everything:** Track every bet, even the losses.
        2. **Update Results:** In **Bets Tab**, be honest. Mark "Bad Beats" vs "Bad Reads".
        3. **Generate Data:** Go to **ML Tab** -> **Generate Dataset**. This converts your history into training rows.
        4. **Train:** Click **Start Training**.
        5. **Result:** The "Robot" will now give better advice on the Analyze tab based on your actual history.
        """)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    st.title("🏀 NBA Elite V15")
    
    # Initialize session state
    if 'version' not in st. session_state or st.session_state['version'] != CURRENT_VERSION: 
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state['version'] = CURRENT_VERSION
    
    if 'analysis_result' not in st. session_state: 
        st.session_state. analysis_result = None
    if 'parlay_builder' not in st. session_state: 
        st.session_state.parlay_builder = []
    if 'usage_mult' not in st. session_state: 
        st.session_state.usage_mult = 1.0
    
    # Initialize components
    orchestrator = PredictionOrchestrator()
    tracker = Tracker()
    parlay_tracker = ParlayTracker()
    
    # Compact Sidebar
    with st. sidebar:
        st.markdown("### ⚙️ Settings")
        bankroll = st.number_input("💰 Bankroll (₱)", 100, 1000000, 600)
        
        # Unit Calculator (1 unit = 1%)
        unit_size = bankroll * 0.01
        st.caption(f"1u = ₱{unit_size:,.0f} | 5u = ₱{unit_size*5:,.0f}")
        st.caption("Stakes: A=5u | B=3u | C=1u")
        
        # Sniper Mode: Usage is now AUTO-calculated from L5 trend
        usage_mult = 1.0  # Base - will be auto-adjusted by stats engine
        st.session_state.usage_mult = usage_mult
        
        # Quick stats with P/L in units
        stats = tracker.get_stats()
        if stats['total_decided'] > 0:
            st.markdown("---")
            profit_color = "green" if stats['total_profit'] >= 0 else "red"
            profit_units = stats['total_profit'] / unit_size if unit_size > 0 else 0
            st.markdown(f"**Record:** {stats['wins']}-{stats['losses']} ({stats['win_rate']:.0%})")
            st.markdown(f"**P/L:** :{profit_color}[₱{stats['total_profit']:+,.0f}] ({profit_units:+.1f}u)")
        
        if st.session_state.parlay_builder:
            st.markdown(f"🎲 Parlay: {len(st.session_state.parlay_builder)} legs")
        
        st.caption(f"{CURRENT_VERSION}")
    
    # Main tabs - simplified names for mobile

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "Analyze", "Backtest", "Watchlist", "H2H", "Splits", "Bets", "Parlays", "ML", "Guide"
    ])
    
    # Tab 1: Analyzer
    with tab1:
        # Mobile-friendly stacked layout with expander for advanced options
        all_players = players.get_players()
        player_names = sorted([p['full_name'] for p in all_players if p.get('is_active', True)])
        
        # Primary inputs - always visible
        col1, col2 = st.columns(2)
        with col1:
            player_in = st.selectbox("🏃 Player", player_names, index=None, placeholder="Search...")
        with col2:
            nba_teams = teams.get_teams()
            team_opts = sorted([t['abbreviation'] for t in nba_teams])
            try:
                def_idx = team_opts.index('LAL')
            except ValueError:
                def_idx = 0
            opp_in = st.selectbox("🎯 vs", team_opts, index=def_idx)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            market = st.selectbox("📊 Market", ["PTS", "REB", "AST", "PRA", "3PM", "PA", "PR", "RA", "STL", "BLK"])
        with col2:
            line_in = st.number_input("🎯 Line", 0.5, 100.0, 25.5, 0.5)
        with col3:
            odds_in = st.number_input("💲 Odds", 1.01, 10.00, 1.85, 0.01)
        
        # Advanced options in expander
        with st.expander("Advanced Options", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                is_home = st.checkbox("Home", True)
            with c2:
                days_rest = st.selectbox("Rest", [0, 1, 2, 3], index=1,
                                         format_func=lambda x: "B2B" if x == 0 else f"{x}d")
            with c3:
                spread = st.number_input("Spread", -25.0, 25.0, 0.0, 0.5)
            with c4:
                game_total = st.number_input("Total", 180.0, 280.0, 225.0, 0.5)
            lookback = st.slider("Lookback Games", 5, 30, 15)
        
        is_b2b = days_rest == 0
        
        is_valid, error_msg = validate_inputs(line_in, odds_in)
        if not is_valid: 
            st.error(error_msg)
        
        if st.button("Run Analysis", type="primary", disabled=not player_in or not is_valid):
            with st.spinner(f"Analyzing {player_in} vs {opp_in}..."):
                result = orchestrator.run_analysis(
                    player_name=player_in,
                    opponent_name=opp_in,
                    market=market,
                    line=line_in,
                    odds=odds_in,
                    is_home=is_home,
                    is_b2b=is_b2b,
                    lookback=lookback,
                    usage_mult=usage_mult,
                    spread=spread,
                    bankroll=bankroll,
                    days_rest=days_rest,
                    game_total=game_total
                )
                
                if not result.success:
                    st.error(result.error)
                else: 
                    st.session_state.analysis_result = result
        
        # Display results
        result = st.session_state.analysis_result
        if result and result.success:
            st.markdown("---")
            
            # Sniper Mode Toggle
            sniper_mode = st.toggle("🎯 Sniper Mode", value=False, help="Hide non-Grade A picks")
            
            # Check if this bet passes Sniper Mode criteria
            is_sniper_worthy = (
                result.decision.grade == BetGrade.A and
                (result.projection.ml_prob is None or result.projection.ml_prob >= 0.50)
            )
            
            if sniper_mode and not is_sniper_worthy:
                st.info("🔇 **Filtered Out** — This pick doesn't meet Sniper Mode criteria (Grade A + Robot Approved)")
                st.caption(f"Grade: {result.decision.grade.value} | ML: {result.projection.ml_prob:.0%}" if result.projection.ml_prob else f"Grade: {result.decision.grade.value}")
            else:
                # Main recommendation - Ticket Style Card
                render_ticket_card(result, bankroll)
                
                # Bet Rules Validation
                render_bet_rules_card(result)
            
                # Action buttons
                col_track, col_parlay = st.columns(2)
                # Tag selection for bet source tracking
                tag_options = ["Sniper", "Robot_Top_Pick", "Gut_Feel", "Live_Bet"]
                # Auto-suggest tag based on context
                default_tag = "Sniper" if (result.decision.grade == BetGrade.A and 
                    (result.projection.ml_prob is None or result.projection.ml_prob >= 0.50)) else "Gut_Feel"
                selected_tag = st.selectbox("Bet Source Tag:", tag_options, 
                    index=tag_options.index(default_tag), key="bet_tag_select",
                    help="Tag for filtering training data - 'Sniper' = disciplined picks")
                
                with col_track: 
                    if st.button("💾 Track", use_container_width=True):
                        try:
                            features = result.features
                            tracker.log_bet({
                                "player": result.player_name,
                                "opponent": result.opponent_name,
                                "market": result.market,
                                "line": result.line,
                                "side": result.decision.recommended_side,
                                "odds": result.odds,
                                "ev": result.decision.expected_value,
                                "proj": result.projection.final_projection,
                                "stake": result.decision.kelly_stake,
                                "prob": result.decision.probability,
                                "rollover_suitable": result.decision.rollover_suitable,
                                "rollover_score": result.decision.rollover_score,
                                # === SNAPSHOT: Freeze point-in-time stats ===
                                "snapshot_l5_hit_rate": float(features.hit_rate_l5),
                                "snapshot_days_rest": int(features.days_rest),
                                "snapshot_def_rank": int(round(features.opponent_drtg_season)),  # Approx rank
                                # === TAG: Bet source for filtering ===
                                "tag": selected_tag,
                                # === FEATURES ===
                                "feat_ema": float(features.ema),
                                "feat_std": float(features.std),
                                "feat_sma_5": float(features.sma_5),
                                "feat_sma_10": float(features.sma_10),
                                "feat_trend": float(features.trend),
                                "feat_avg_minutes": float(features.avg_minutes),
                                "feat_mins_trend": float(features.mins_trend),
                                "feat_hit_l5": float(features.hit_rate_l5),
                                "feat_hit_l10": float(features.hit_rate_l10),
                                "feat_hit_l15": float(features.hit_rate_l15),
                                "feat_hit_season": float(features.hit_rate_season),
                                "feat_pace_mult": float(features.pace_mult),
                                "feat_def_mult": float(features.def_mult),
                                "feat_position_mult": float(features.position_mult),
                                "feat_base_matchup_mult": float(features.base_matchup_mult),
                                "feat_combined_matchup_mult": float(features.combined_matchup_mult),
                                "feat_split_factor": float(features.split_factor),
                                "feat_rest_factor": float(features.rest_factor),
                                "feat_blowout_factor": float(features.blowout_factor),
                                "feat_usage_mult": float(features.usage_mult),
                                "feat_spread": float(features.spread),
                                "feat_is_home": bool(features.is_home),
                                "feat_is_b2b": bool(features.is_b2b),
                                "feat_position": features.player_position,
                                "feat_games_played": int(features.games_played),
                                "feat_days_rest": int(features.days_rest),
                                "feat_game_total": float(features.game_total),
                                "feat_opp_drtg_season": float(features.opponent_drtg_season),
                                "feat_opp_drtg_l5": float(features.opponent_drtg_l5),
                                "feat_proj": float(result.projection.final_projection),
                                "feat_ev": float(result.decision.expected_value),
                                "feat_prob": float(result.decision.probability),
                                # V15 raw fields to fully populate training features
                                "blowout_prob": float(getattr(features, 'blowout_prob', 0.0)),
                                "personal_fatigue_factor": float(getattr(features, 'personal_fatigue_factor', 1.0)),
                                "b2b_games_in_sample": float(getattr(features, 'b2b_games_in_sample', 0.0)),
                                "dynamic_std_mult": float(getattr(features, 'dynamic_std_mult', 1.0)),
                                "coef_variation": float(getattr(features, 'coef_variation', 0.0)),
                                # V16 market & role context
                                "odds_decimal": american_to_decimal(result.odds) if result.odds else 1.91,
                                "usg_season": float(getattr(features, 'usg_season', 0.0)),
                                "clv": 0.0,  # Populated later when closing line known
                                # Sniper Mode: Usage trend features for ML
                                "feat_usg_season": 0.0,  # Raw USG_PCT requires log access
                                "feat_usg_trend": float(features.usage_mult),  # Already incorporates damped trend
                            })
                            st.toast(f"Bet tracked! [{selected_tag}]", icon="💾")
                        except Exception as e:
                            st.error(f"Failed to track bet: {e}")
                            logger.error(f"Bet tracking failed: {e}")
                
                with col_parlay: 
                    if result.decision.rollover_suitable:
                        if st.button("Parlay", use_container_width=True):
                            leg = {
                                'player': result.player_name,
                                'opponent': result.opponent_name,
                                'market': result.market,
                                'line': result.line,
                                'side': result.decision.recommended_side,
                                'odds': result.odds,
                                'prob': result.decision.probability,
                                'proj': result.projection.final_projection,
                                'ev': result.decision.expected_value,
                                'position': result.features.player_position,
                                'result': 'Pending'
                            }
                            existing = [l for l in st.session_state.parlay_builder 
                                        if l['player'] == leg['player'] and l['market'] == leg['market']]
                            if existing:
                                st.warning("Already in parlay!")
                            elif len(st.session_state.parlay_builder) >= CONFIG.MAX_PARLAY_LEGS: 
                                st.warning(f"Max {CONFIG.MAX_PARLAY_LEGS} legs!")
                            else:
                                st.session_state.parlay_builder.append(leg)
                                st.toast("Added!", icon="🎲")
                                st.rerun()
                    else: 
                        st.button("Parlay", disabled=True, use_container_width=True,
                                 help="Not suitable for parlay")
            
                # Details in expander to reduce clutter
                with st.expander("Details & Chart", expanded=False):
                    render_blowout_info(result)
                    
                    # Hit rates
                    st.markdown("**Hit Rates:**")
                    cols = st.columns(4)
                    for i, (label, rate) in enumerate([
                        ("L5", result.hit_rates['l5']), 
                        ("L10", result.hit_rates['l10']),
                        ("L15", result.hit_rates['l15']), 
                        ("Season", result.hit_rates['season'])
                    ]):
                        color = "green" if rate >= 0.6 else "orange" if rate >= 0.4 else "red"
                        cols[i].metric(label, f"{rate:.0%}")
                    
                    st.caption(result.projection.context)
                    render_distribution_chart(result)
    
    # Tab 2: Backtest
    with tab2:
        render_backtest_tab(orchestrator)
    with tab3:
        render_watchlist_tab(orchestrator)
    # Tab 4: H2H
    with tab4:
        result = st.session_state.analysis_result
        if result and result.success:
            st.markdown(f"{result.player_name} vs {result.opponent_name}")
            with st.spinner("Loading..."):
                full_history = orchestrator.data_loader.fetch_multi_season_logs(result.player_id)
            
            if len(full_history) > 0:
                opp_code = result.opponent_name
                h2h_df = full_history[full_history['MATCHUP'].str.contains(opp_code, case=False)].copy()
                
                if len(h2h_df) > 0:
                    h2h_df = h2h_df.sort_values('GAME_DATE', ascending=False)
                    c1, c2, c3 = st.columns(3)
                    avg_h2h = h2h_df[result.market].mean()
                    h2h_hit_rate = (h2h_df[result.market] > result.line).mean()
                    c1.metric(f"Avg {result.market}", f"{avg_h2h:.1f}")
                    c2.metric("Games", len(h2h_df))
                    c3.metric(f"Hit%", f"{h2h_hit_rate:.0%}")
                    
                    display_cols = ['GAME_DATE', 'WL', 'MIN', result.market]
                    st.dataframe(h2h_df[display_cols].head(8), hide_index=True, width="stretch")
                else:
                    st.info(f"No games vs {opp_code}")
            else: 
                st.info("No history available")
        else:
            st.info("Run analysis first")
    
    # Tab 5: Splits
    with tab5:
        result = st.session_state.analysis_result
        if result and result.success:
            df = result.game_logs
            market = result.market
            line = result.line
            
            st.markdown(f"{result.player_name} Splits")
            
            splits = {
                "L15": df.tail(15),
                "Home": df[df['IS_HOME'] == True].tail(15),
                "Away": df[df['IS_HOME'] == False].tail(15),
                "W": df[df['WL'] == 'W'].tail(15),
                "L": df[df['WL'] == 'L'].tail(15),
            }
            
            if 'DAYS_REST' in df.columns:
                splits["Rest"] = df[df['DAYS_REST'] >= 3].tail(15)
                splits[" B2B"] = df[df['DAYS_REST'] <= 1].tail(15)
            
            data = []
            for k, v in splits.items():
                if len(v) >= 3:
                    hit_rate = (v[market] > line).mean()
                    data.append({
                        "Split": k,
                        "N": len(v),
                        "Avg": f"{v[market].mean():.1f}",
                        "Hit%": f"{hit_rate:.0%}",
                    })
            
            if data:
                st.dataframe(pd.DataFrame(data), hide_index=True, width="stretch")
            else:
                st.info("Not enough data")
        else:
            st.info("Run analysis first")
    
    # Tab 6: Bets
    with tab6:
        st.markdown("Bet Tracker")
        
        stats = tracker.get_stats()
        all_bets = tracker.get_bets()
        
        if stats['total_decided'] > 0:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Win%", f"{stats['win_rate']:.0%}")
            c2.metric("Record", f"{stats['wins']}-{stats['losses']}")
            c3.metric("P/L", f"₱{stats['total_profit']:+,.0f}")
            c4.metric("Pending", stats['pending'])
            
            # =========================================================================
            # EQUITY CURVE - Cumulative Profit Chart
            # =========================================================================
            decided_bets = [b for b in all_bets if b.get('result') in ['Win', 'Loss']]
            if len(decided_bets) >= 2:
                # Sort by date
                decided_bets.sort(key=lambda x: x.get('date', '2024-01-01'))
                
                # Calculate cumulative P/L (assuming 1 unit per bet, -110 odds)
                cumulative = []
                running_total = 0
                for bet in decided_bets:
                    if bet.get('result') == 'Win':
                        running_total += 0.91  # Win at -110 odds
                    else:
                        running_total -= 1.0   # Lose 1 unit
                    cumulative.append({
                        'Bet #': len(cumulative) + 1,
                        'Date': bet.get('date', '')[:10] if bet.get('date') else '',
                        'Player': bet.get('player', '')[:15],
                        'P/L': running_total
                    })
                
                equity_df = pd.DataFrame(cumulative)
                
                # Create the chart
                with st.expander("📈 Equity Curve", expanded=True):
                    # Use matplotlib for a clean line chart
                    fig, ax = plt.subplots(figsize=(10, 3))
                    
                    # Plot the line
                    ax.plot(equity_df['Bet #'], equity_df['P/L'], linewidth=2.5, color='#00c853' if running_total >= 0 else '#f44336')
                    ax.fill_between(equity_df['Bet #'], 0, equity_df['P/L'], 
                                   where=(equity_df['P/L'] >= 0), alpha=0.3, color='#00c853', interpolate=True)
                    ax.fill_between(equity_df['Bet #'], 0, equity_df['P/L'], 
                                   where=(equity_df['P/L'] < 0), alpha=0.3, color='#f44336', interpolate=True)
                    
                    # Styling
                    ax.axhline(y=0, color='white', linestyle='--', linewidth=0.8, alpha=0.5)
                    ax.set_xlabel('Bet #', fontsize=10)
                    ax.set_ylabel('Units', fontsize=10)
                    ax.set_title(f'Cumulative P/L: {running_total:+.2f}u ({stats["wins"]}-{stats["losses"]})', fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.2)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    
                    # Set y-axis limits with padding
                    y_max = max(abs(equity_df['P/L'].min()), abs(equity_df['P/L'].max())) * 1.2
                    ax.set_ylim(-max(y_max, 1), max(y_max, 1))
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Quick stats below chart
                    eq_c1, eq_c2, eq_c3 = st.columns(3)
                    peak = equity_df['P/L'].max()
                    trough = equity_df['P/L'].min()
                    max_dd = peak - trough if peak > trough else 0
                    eq_c1.metric("Peak", f"+{peak:.2f}u" if peak > 0 else f"{peak:.2f}u")
                    eq_c2.metric("Trough", f"{trough:.2f}u")
                    eq_c3.metric("Max Drawdown", f"{max_dd:.2f}u")
            
            # Score Box Summary (if we have margin data)
            if stats.get('margin_count', 0) > 0:
                with st.expander("Score Box Analysis", expanded=False):
                    sb1, sb2, sb3, sb4 = st.columns(4)
                    sb1.metric("Avg Margin", f"{stats['avg_margin']:+.1f}", 
                              help="Average margin on decided bets (+ = favorable)")
                    sb2.metric("Bad Beats", f"{stats['bad_beat_rate']:.0%}", 
                              help="% of losses by ≤1.5 pts")
                    sb3.metric("Bad Reads", f"{stats['bad_read_rate']:.0%}", 
                              help="% of losses by 7.5+ pts")
                    sb4.metric("Solid Wins", f"{stats.get('solid_win_rate', 0):.0%}", 
                              help="% of wins by 3.5+ pts")
                    
                    # Quality breakdown
                    qc = stats.get('quality_counts', {})
                    if any(qc.values()):
                        st.caption("**Result Quality Breakdown:**")
                        loss_cols = st.columns(4)
                        loss_cols[0].write(f"💔 Bad Beat: {qc.get('bad_beat', 0)}")
                        loss_cols[1].write(f"😤 Close Loss: {qc.get('close_loss', 0)}")
                        loss_cols[2].write(f"📉 Clear Loss: {qc.get('clear_loss', 0)}")
                        loss_cols[3].write(f"🚫 Bad Read: {qc.get('bad_read', 0)}")
                        
                        win_cols = st.columns(4)
                        win_cols[0].write(f"😅 Sweat Win: {qc.get('sweat_win', 0)}")
                        win_cols[1].write(f"✌️ Close Win: {qc.get('close_win', 0)}")
                        win_cols[2].write(f"💪 Solid Win: {qc.get('solid_win', 0)}")
                        win_cols[3].write(f"🔥 Blowout Win: {qc.get('blowout_win', 0)}")
        
        # Export and download controls
        col_export, col_dl, col_clear = st.columns([2, 2, 1])
        
        with col_export:
            exportable = tracker.get_exportable_count()
            if exportable > 0:
                if st.button(f"📤 Export {exportable} to ML CSV", type="primary", key="export_ml_csv"):
                    count, path = tracker.export_bets_to_training_csv()
                    st.success(f"✓ Exported {count} bets to {path}")
                    st.rerun()
            else:
                st.button("📤 Export to ML CSV", disabled=True, 
                         help="No new completed bets with scores to export")
        
        with col_dl:
            if TRACKER_FILE.exists():
                with open(TRACKER_FILE, "r") as f:
                    st.download_button("📥 Download JSON", f.read(), "nba_bet_history.json", "application/json", key="dl_bet_history")
        
        with col_clear: 
            if st.button("🗑️ Clear All", type="secondary", key="clear_bets"):
                tracker.clear_history()
                st.rerun()
        
        st.divider()
        
        if all_bets: 
            # Tag filter for training data purification
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                filter_opts = ["Pending", "Win", "Loss", "Push"]
                selected_filters = st.multiselect("Filter by Result:", options=filter_opts, default=["Pending", "Win", "Loss"])
            with filter_col2:
                tag_opts = list(set(b.get('tag', 'legacy') for b in all_bets))
                tag_opts.sort()
                selected_tags = st.multiselect("Filter by Tag:", options=tag_opts, default=tag_opts,
                                              help="Filter training data: 'Sniper' = disciplined bets only")
            
            filtered_bets = [b for b in all_bets 
                            if b.get('result', 'Pending') in selected_filters 
                            and b.get('tag', 'legacy') in selected_tags]
            
            sort_map = {'Pending': 1, 'Win': 2, 'Loss': 3, 'Push': 4}
            filtered_bets.sort(key=lambda x: (sort_map.get(x.get('result', 'Pending'), 99), -x.get('id', 0)))
            
            if not filtered_bets: 
                st.info("No bets match these filters.")
            
            for bet in filtered_bets:
                with st.container(border=True):
                    bet_res = bet.get('result', 'Pending')
                    bet_tag = bet.get('tag', 'legacy')
                    tag_emoji = {'Sniper': '🎯', 'Robot_Top_Pick': '🤖', 'Gut_Feel': '🎲', 'Live_Bet': '⚡'}.get(bet_tag, '📋')
                    icon = {"Win": "✅", "Loss": "❌", "Push": "🔄"}.get(bet_res, "⏳")
                    color = {"Win": "green", "Loss": "red", "Push": "blue"}.get(bet_res, "gray")
                    
                    # Compact header line with tag
                    st.markdown(f"{icon} {tag_emoji} :{color}[**{bet['player']}**] {bet['side']} {bet['line']} ({bet['market']})")
                    st.caption(f"vs {bet.get('opponent')} | Proj: {bet.get('proj', 0):.1f} | EV: {bet.get('ev', 0):+.1%} | {bet.get('date', 'N/A')} | Tag: {bet_tag}")
                    
                    # Column labels row
                    lbl1, lbl2, lbl3, lbl4 = st.columns([1.5, 1.5, 1.5, 0.5])
                    lbl1.caption("**Result**")
                    lbl2.caption("**Actual Score**")
                    lbl3.caption("**Closing Line**")
                    lbl4.caption("")
                    
                    # Controls row - Result, Actual Value, Closing Line
                    c1, c2, c3, c4 = st.columns([1.5, 1.5, 1.5, 0.5])
                    opts = ["Pending", "Win", "Loss", "Push"]
                    curr_idx = opts.index(bet_res) if bet_res in opts else 0
                    new = c1.selectbox("Result", opts, index=curr_idx, key=f"s_{bet['id']}", label_visibility="collapsed")
                    
                    # Actual value input for score box tracking
                    current_actual = bet.get('actual_value', bet.get('line', 0))
                    actual_value = c2.number_input("Actual Score", value=float(current_actual), step=0.5,
                                                   key=f"av_{bet['id']}", label_visibility="collapsed",
                                                   placeholder="Player's stat",
                                                   help="Player's actual stat value (e.g., 25 points)")
                    
                    current_closing = bet.get('closing_line', bet.get('line', 0))
                    closing_line = c3.number_input("Closing Line", value=float(current_closing), step=0.5, 
                                                   key=f"cl_{bet['id']}", label_visibility="collapsed",
                                                   placeholder="Final line",
                                                   help="The line at game start (for CLV tracking)")
                    
                    # Check for any changes
                    value_changed = (actual_value != current_actual) and new != "Pending"
                    result_changed = new != bet_res
                    close_changed = (closing_line != current_closing) and new != "Pending"
                    
                    if result_changed or value_changed or close_changed:
                        tracker.update_result(
                            bet['id'], 
                            new, 
                            closing_line if new != "Pending" else None,
                            actual_value if new != "Pending" else None
                        )
                        st.rerun()
                    
                    if c4.button("🗑️", key=f"d_{bet['id']}"):
                        tracker.delete_bet(bet['id'])
                        st.rerun()
                    
                    # Show score box info if available
                    if bet.get('margin') is not None:
                        margin = bet['margin']
                        margin_pct = bet.get('margin_pct', 0)
                        quality = bet.get('result_quality', 'unknown')
                        quality_emoji = {
                            'bad_beat': '💔', 'close_loss': '😤', 'clear_loss': '📉', 'bad_read': '🚫',
                            'sweat_win': '😅', 'close_win': '✌️', 'solid_win': '💪', 'blowout_win': '🔥',
                            'push': '🔄'
                        }.get(quality, '❓')
                        margin_color = 'green' if margin > 0 else ('red' if margin < 0 else 'gray')
                        # Show both raw margin and percentage (standardized across stat types)
                        st.caption(f"{quality_emoji} :{margin_color}[Margin: {margin:+.1f} ({margin_pct:+.1f}%)] | Quality: {quality.replace('_', ' ').title()}")
        else:
            st.info("No bets tracked yet. Run an analysis and click 'Track' to save a bet.")
    
    # Tab 7: Parlays
    with tab7:
        render_parlay_tab(parlay_tracker, bankroll)
    
    # Tab 8: ML Data (UPDATED)
    with tab8:
        # We now have 3 sub-tabs
        ml_subtab1, ml_subtab2, ml_subtab3 = st.tabs(["Training Data", "Generate Dataset", "Train Brain"])
        
        with ml_subtab1:
            render_ml_data_tab(tracker)
        
        with ml_subtab2:
            generate_ml_data_streamlit()
            
        with ml_subtab3:
            render_train_model_tab()

    with tab9:
        render_guide_tab()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__": 
    main()
                
