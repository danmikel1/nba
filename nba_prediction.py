from pyexpat import features
import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, commonplayerinfo
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
logger = logging. getLogger(__name__)

CURRENT_VERSION = 'v15.0'


@dataclass(frozen=True)
class Config:
    """Immutable configuration constants."""
    CURRENT_SEASON: str = "2025-26"
    PREV_SEASON: str = "2024-25"
    API_DELAY: float = 0.6
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
            float(self.b2b_games_in_sample), self.dynamic_std_mult, self.coef_variation
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
        """Execute API call with retry logic."""
        last_exception = None
        for attempt in range(self.config.API_MAX_RETRIES):
            try:
                delay = self.config. API_DELAY * (attempt + 1)
                time.sleep(delay)
                return func()
            except Exception as e: 
                last_exception = e
                logger.warning(f"{description} attempt {attempt + 1}/{self.config.API_MAX_RETRIES} failed: {e}")
        
        raise DataLoaderError(f"{description} failed after {self.config.API_MAX_RETRIES} attempts:  {last_exception}")
    
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
    
    def fetch_opponent_stats(self) -> pd.DataFrame:
        """Fetch opponent (defensive) stats for position defense calculation."""
        try:
            def api_call():
                return leaguedashteamstats.LeagueDashTeamStats(
                    season=self.config.CURRENT_SEASON,
                    measure_type_detailed_defense='Opponent',
                    per_mode_detailed='PerGame'
                ).get_data_frames()[0]
            
            return self._api_call_with_retry(api_call, "Fetch opponent stats")
        except Exception as e:
            logger.error(f"Failed to fetch opponent stats: {e}")
            return pd.DataFrame()
    
    def fetch_position_defense(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate position-based defense multipliers from real data.
        Uses caching to avoid repeated API calls.
        """
        # Check file cache
        try:
            if POSITION_DEF_CACHE_FILE. exists():
                with open(POSITION_DEF_CACHE_FILE, 'r') as f:
                    cache = json.load(f)
                    if time.time() - cache. get('timestamp', 0) < self. config.CACHE_TTL_POSITION_DEF: 
                        logger.info("Using cached position defense data")
                        return cache. get('data', {})
        except (json.JSONDecodeError, IOError):
            pass
        
        logger.info("Fetching real position defense data from NBA API...")
        
        opp_stats = self.fetch_opponent_stats()
        if opp_stats is None or len(opp_stats) == 0:
            return self._get_neutral_position_multipliers()
        
        team_position_mult = self._calculate_position_multipliers(opp_stats)
        
        # Save to cache
        try:
            cache = {'timestamp': time.time(), 'data': team_position_mult}
            with open(POSITION_DEF_CACHE_FILE, 'w') as f:
                json. dump(cache, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save position defense cache: {e}")
        
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
    [NEW] Handles fetching injury data to drive dynamic usage adjustments.
    """
    def __init__(self, rapid_api_key: str = None):
        self.rapid_api_key = rapid_api_key
        # Cache to store injury reports for 1 hour to save API calls
        self._cache: Dict[str, Any] = {}

    def get_injury_impact(self, player_id: int, team_id: int) -> float:
        """
        Calculates usage multiplier based on missing teammates.
        """
        # 1. Fetch Data
        team_injuries = self._fetch_team_injuries(team_id)
        if not team_injuries:
            return 1.0

        # 2. Calculate Impact
        usage_bump = 0.0
        # Simple heuristic: +5% usage for every "OUT" rotation player
        for p in team_injuries:
            status = p.get('status', '').upper()
            # Ensure we aren't counting the player themselves
            # (Requires name matching, skipping for now to keep it simple/fast)
            
            if status in ['OUT', 'INACTIVE']:
                usage_bump += 0.05
            elif status == 'DOUBTFUL':
                usage_bump += 0.02
        
        # Cap the auto-boost at +25% to prevent explosions
        return min(1.25, 1.0 + usage_bump)

    def _fetch_team_injuries(self, team_id: int) -> List[Dict]:
        """Fetch injury list. Defaults to NBA_API (Free) if no RapidKey provided."""
        cache_key = f"injuries_{team_id}"
        # Check cache (1 hour TTL)
        if cache_key in self._cache:
             if (datetime.now() - self._cache[cache_key]['time']).seconds < 3600:
                 return self._cache[cache_key]['data']

        data = []
        
        # A. Try RapidAPI (Tank01) if key is provided
        if self.rapid_api_key:
            try:
                # Placeholder for Tank01 logic
                pass 
            except Exception as e:
                logger.warning(f"RapidAPI failed: {e}")

        # B. Fallback to Native NBA_API (CommonTeamRoster) - Completely Free
        if not data:
            try:
                from nba_api.stats.endpoints import commonteamroster
                # commonteamroster requires team_id
                roster = commonteamroster.CommonTeamRoster(team_id=team_id).get_data_frames()[0]
                
                if 'STATUS' in roster.columns:
                    # Filter for INACTIVE players
                    inactive = roster[roster['STATUS'] == 'INACTIVE']
                    for _, player in inactive.iterrows():
                        data.append({
                            'name': player['PLAYER'],
                            'status': 'OUT', 
                            'role': 'ROTATION'
                        })
            except Exception:
                # Fail silently to avoid breaking the app during internet hiccups
                pass

        self._cache[cache_key] = {'time': datetime.now(), 'data': data}
        return data

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
        self.injury_manager = InjuryManager() # [NEW] Dynamic Injury API
    
    def calculate_composite_usage(self, player_id: int, team_id: int, manual_adj_percent: float) -> float:
        """
        [NEW] Combines automated injury data with manual slider input.
        Result is passed to build_feature_vector.
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
        
        # Usage Trend
        usage_trend_mult = 1.0
        if 'USG_PCT' in df.columns:
            recent_usg = recent['USG_PCT'].mean()
            season_usg = df['USG_PCT'].mean()
            usage_trend = safe_divide(recent_usg, season_usg, 1.0)
            
            if usage_trend > 1.10:
                usage_trend_mult = 1.05
        
        return {
            'ema': ema,
            'std':  std_dev,
            'sma_5': sma_5,
            'sma_10':  sma_10,
            'trend':  trend,
            'avg_minutes': avg_minutes,
            'mins_trend': mins_trend,
            'usage_trend_mult': usage_trend_mult, # Ensure this is returned!
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

        # 5. Get ML Prediction
        ml_prob = self.get_ml_prediction(features)

        # 6. Generate Context String
        context_parts = []
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
        
        low_count_stats = ['STL', 'BLK', '3PM', 'FG3M']
        
        if market in low_count_stats: 
            # Use Poisson for low-count stats
            # For Poisson, adjust lambda based on features if available
            adjusted_mean = mean
            if features is not None:
                adjusted_mean = self._apply_feature_adjustments_to_mean(mean, line, features)
            sims = np.random.poisson(max(0.1, adjusted_mean), simulations).astype(float)
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
        base_std = max(0.1, std)
        
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
    
    def __init__(self, config:  Config = CONFIG):
        self.config = config
    
    def calculate_ev(self, prob:  float, odds: float) -> float:
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
    
    def assign_grade(self, ev: float, win_prob: float) -> BetGrade: 
        """
        Assign letter grade based on EV and Probability.
        """
        # Logic for "S-Tier" / "A+" bets (High EV + High Prob)
        # We map this to BetGrade.A since the Enum doesn't have A+, 
        # but it ensures these get the maximum stake.
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
    
    def assess_rollover_quality(
        self, 
        features: FeatureVector,
        projection:  Projection,
        simulation:  SimulationResult,
        line: float,
        best_side: str,
        best_prob: float
    ) -> Tuple[bool, float, List[str], List[str]]: 
        """Assess suitability for rollover/parlay."""
        reasons_good = []
        reasons_bad = []
        score = 0
        max_score = 5
        
        # Probability check
        if best_prob >= self.config.ROLLOVER_MIN_PROB: 
            score += 1.5
            reasons_good.append(f"High probability ({best_prob:.0%})")
        elif best_prob >= 0.55:
            score += 0.5
            reasons_bad.append(f"Moderate probability ({best_prob:.0%})")
        else:
            reasons_bad.append(f"Low probability ({best_prob:.0%})")
        
        # Consistency check
        std_ratio = features.std / projection. final_projection if projection.final_projection > 0 else 1
        if std_ratio <= self.config.ROLLOVER_MAX_STD_RATIO:
            score += 1
            reasons_good.append(f"Consistent performer (±{std_ratio:.0%} variance)")
        else:
            reasons_bad.append(f"High variance (±{std_ratio:.0%})")
        
        # Hit rate check
        if features.hit_rate_l10 >= self.config.ROLLOVER_MIN_HIT_RATE: 
            score += 1
            reasons_good.append(f"Strong L10 hit rate ({features.hit_rate_l10:.0%})")
        elif features.hit_rate_l10 >= 0.50:
            score += 0.5
            reasons_bad.append(f"Average L10 hit rate ({features.hit_rate_l10:.0%})")
        else:
            reasons_bad.append(f"Poor L10 hit rate ({features.hit_rate_l10:.0%})")

        # Sample size check
        if features.games_played >= self.config.ROLLOVER_MIN_GAMES:
            score += 0.5
            reasons_good.append(f"Good sample size ({features.games_played} games)")
        else:
            reasons_bad.append(f"Small sample ({features.games_played} games)")
        
        # Edge buffer check
        if best_side == "OVER":
            buffer = (projection.final_projection - line) / line if line > 0 else 0
        else:
            buffer = (line - projection.final_projection) / line if line > 0 else 0
        
        if buffer >= 0.10:
            score += 1
            reasons_good.append(f"Large edge ({buffer:.0%} buffer)")
        elif buffer >= 0.05:
            score += 0.5
            reasons_good.append(f"Decent edge ({buffer:.0%} buffer)")
        else:
            reasons_bad.append(f"Thin edge ({buffer:.0%} buffer)")
        
        # Trend alignment
        if features.trend > 0.05 and best_side == "OVER":
            score += 0.25
            reasons_good.append("📈 Trending up")
        elif features.trend < -0.05 and best_side == "UNDER":
            score += 0.25
            reasons_good.append("📉 Trending down")
        
        # Blowout risk penalty
        if features. blowout_factor < 0.90:
            score -= 0.5
            reasons_bad.append("🔴 High blowout risk")
        elif features. blowout_factor < 0.95:
            score -= 0.25
            reasons_bad.append("🟠 Moderate blowout risk")
        
        score = max(0, score)
        score_pct = score / max_score
        suitable = score_pct >= 0.55
        
        return suitable, score, reasons_good, reasons_bad
    
    def _generate_confidence_warning(
        self, 
        features: FeatureVector, 
        probability: float
    ) -> Optional[str]:
        """
        Generate warning message for low-confidence predictions.
        Helps user understand when to be cautious.
        """
        warnings = []
        
        # Low sample size warning
        if features.games_played < 10:
            warnings.append(f"⚠️ Low sample ({features.games_played} games)")
        elif features.games_played < 15:
            warnings.append(f"📊 Limited sample ({features.games_played} games)")
        
        # High volatility warning (CV > 0.35 means std is >35% of mean)
        if features.coef_variation > 0.40:
            warnings.append(f"📈 High volatility (CV: {features.coef_variation:.0%})")
        elif features.coef_variation > 0.30:
            warnings.append(f"📊 Moderate volatility (CV: {features.coef_variation:.0%})")
        
        # Edge case: probability very close to 50% (coin flip)
        if 0.48 <= probability <= 0.52:
            warnings.append("🎲 Near coin-flip probability")
        
        # Conflicting signals: weighted hit rate differs significantly from season
        if abs(features.hit_rate_weighted - features.hit_rate_season) > 0.20:
            if features.hit_rate_weighted > features.hit_rate_season:
                warnings.append("📈 Recent form better than season avg")
            else:
                warnings.append("📉 Recent form worse than season avg")
        
        return " | ".join(warnings) if warnings else None
    
    def make_decision(
        self,
        features: FeatureVector,
        projection: Projection,
        simulation:  SimulationResult,
        line: float,
        odds: float,
        bankroll: float
    ) -> BetDecision:
        """Make final bet decision. ML influence is already baked into projection."""
        
        # Get simulation probabilities (already reflects ML nudge via projection)
        over_prob = simulation.over_prob
        under_prob = simulation.under_prob
        
        # Calculate EVs
        ev_over = self.calculate_ev(over_prob, odds)
        ev_under = self.calculate_ev(under_prob, odds)
        
        # Determine best side
        if ev_over > ev_under:
            best_side = "OVER"
            best_ev = ev_over
            best_prob = over_prob
        else:
            best_side = "UNDER"
            best_ev = ev_under
            best_prob = under_prob
        
        # Assign grade first (needed for stake calculation)
        # [FIXED] Now passing both EV and Probability
        grade = self.assign_grade(best_ev, best_prob)
        
        # Calculate flat stake based on grade
        stake, stake_pct = self.calculate_flat_stake(grade, bankroll)
        
        # Assess rollover suitability
        suitable, score, reasons_good, reasons_bad = self.assess_rollover_quality(
            features, projection, simulation, line, best_side, best_prob
        )
        
        # Generate confidence warning if applicable
        confidence_warning = self._generate_confidence_warning(features, best_prob)
        
        # Add ML disagreement warning if simulation and ML have opposite opinions
        if projection.ml_prob is not None:
            ml_favors_over = projection.ml_prob > 0.5
            sim_favors_over = over_prob > under_prob
            
            if sim_favors_over != ml_favors_over:
                ml_warning = f"⚠️ ML-Sim tension (ML: {projection.ml_prob:.0%}, Sim: {over_prob:.0%} OVER)"
                if confidence_warning:
                    confidence_warning = f"{confidence_warning} | {ml_warning}"
                else:
                    confidence_warning = ml_warning
        
        return BetDecision(
            recommended_side=best_side,
            probability=best_prob,
            expected_value=best_ev,
            grade=grade,
            kelly_stake=stake,
            kelly_fraction=stake_pct,
            rollover_suitable=suitable,
            rollover_score=score,
            reasons_good=reasons_good,
            reasons_bad=reasons_bad,
            confidence_warning=confidence_warning
        )


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
    
    def run_backtest(
        self,
        player_id: int,
        player_name: str,
        market: str,
        lookback:  int = 15,
        test_days: int = 30,
        line_offset: float = 0.0,  # Test at actual average ± offset
        fixed_spread: float = 0.0,
        progress_callback=None
    ) -> Optional[BacktestSummary]:
        """
        Run walk-forward backtest.
        
        For each day in the test period:
        1. Use only data available up to that day
        2. Generate prediction
        3. Compare to actual result
        """
        
        # Fetch all available data
        df = self.data_loader.fetch_multi_season_logs(player_id)
        if len(df) < self.config.BACKTEST_MIN_GAMES + test_days:
            logger.warning(f"Insufficient data for backtest: {len(df)} games")
            return None
        
        # Get player position
        player_position = self.data_loader.get_player_position(player_id)
        
        # Get team stats (use current - this is a simplification)
        team_stats, avg_pace, avg_def = self.data_loader. fetch_team_stats()
        position_defense = self.data_loader. fetch_position_defense()
        
        # Sort by date
        df = df.sort_values('GAME_DATE').reset_index(drop=True)
        
        # Determine test period
        total_games = len(df)
        test_start_idx = max(self.config.BACKTEST_MIN_GAMES, total_games - test_days)
        
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
                    'hit_rate_l5': features.hit_rate_l5,
                    'hit_rate_l10': features.hit_rate_l10,
                    'hit_rate_l15': features.hit_rate_l15,
                    'hit_rate_season': features.hit_rate_season,
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
                    'opponent_drtg_season': features.opponent_drtg_season,
                    'opponent_drtg_l5': features.opponent_drtg_l5,
                    # V15 NEW FEATURES
                    'blowout_prob': features.blowout_prob,
                    'personal_fatigue_factor': features.personal_fatigue_factor,
                    'b2b_games_in_sample': features.b2b_games_in_sample,
                    'dynamic_std_mult': features.dynamic_std_mult,
                    'coef_variation': features.coef_variation
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
        rows = []
        for r in results:
            row = {
                'date': r.date,
                'player': r.player_name,
                'market': r.market,
                'line': r.line,
                'predicted_side': r.predicted_side,
                'predicted_prob': r.predicted_prob,
                'predicted_ev': r.predicted_ev,
                'actual_value': r.actual_value,
                'hit': 1 if r.hit else 0,  # Binary for ML
                'grade': r.grade,
            }
            # Flatten features with 'feat_' prefix
            for feat_name, feat_value in r.features.items():
                row[f'feat_{feat_name}'] = feat_value
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

def get_top_active_players(limit: int = 50) -> List[Dict]:
    """
    Get top active NBA players by games played this season.
    Returns list of player dicts with 'id' and 'full_name'.
    """
    all_players = players.get_active_players()
    
    # Sort by most recognizable (those with most data typically)
    # We'll verify by checking game logs
    player_game_counts = []
    data_loader = DataLoader()
    
    logger.info(f"Scanning for top {limit} active players...")
    
    for p in all_players[:200]:  # Check first 200 active players
        try:
            df = data_loader.fetch_game_logs(p['id'])
            if len(df) >= 10:  # Minimum games threshold
                player_game_counts.append({
                    'id': p['id'],
                    'full_name': p['full_name'],
                    'games': len(df)
                })
        except Exception:
            continue
        
        if len(player_game_counts) >= limit * 2:  # Get more than needed to filter
            break
    
    # Sort by games played and return top N
    player_game_counts.sort(key=lambda x: x['games'], reverse=True)
    return player_game_counts[:limit]


def generate_ml_training_data(
    output_file: str = "ml_training_data.csv",
    num_players: int = 50,
    markets: List[str] = None,
    test_days: int = 60,  # More data for training
    lookback: int = 15,
    progress_callback=None
) -> pd.DataFrame:
    """
    Generate ML training dataset by running backtests on multiple players.
    
    Args:
        output_file: Output CSV filename
        num_players: Number of top players to include
        markets: List of markets to backtest (default: PTS, REB, AST, PRA)
        test_days: Number of games to backtest per player
        lookback: Lookback window for features
        progress_callback: Optional callback for progress updates
    
    Returns:
        Combined DataFrame with all backtest results
    """
    if markets is None:
        markets = ['PTS', 'REB', 'AST', 'PRA']
    
    logger.info(f"Starting ML data generation for {num_players} players...")
    
    # Initialize components
    config = CONFIG
    data_loader = DataLoader(config)
    feature_engineer = FeatureEngineer(config)
    model_engine = ModelEngine(config)
    simulation_engine = SimulationEngine(config)
    decision_policy = DecisionPolicy(config)
    
    backtester = Backtester(
        data_loader=data_loader,
        feature_engineer=feature_engineer,
        model_engine=model_engine,
        simulation_engine=simulation_engine,
        decision_policy=decision_policy,
        config=config
    )
    
    # Get top players
    logger.info("Fetching top active players...")
    top_players = get_top_active_players(num_players)
    logger.info(f"Found {len(top_players)} players with sufficient data")
    
    all_results = []
    total_tasks = len(top_players) * len(markets)
    completed = 0
    
    for player in top_players:
        player_id = player['id']
        player_name = player['full_name']
        
        for market in markets:
            try:
                logger.info(f"Backtesting {player_name} - {market}...")
                
                summary = backtester.run_backtest(
                    player_id=player_id,
                    player_name=player_name,
                    market=market,
                    lookback=lookback,
                    test_days=test_days,
                    line_offset=0.0,
                    fixed_spread=0.0
                )
                
                if summary is not None and len(summary.results_df) > 0:
                    # Add player_id column
                    summary.results_df['player_id'] = player_id
                    all_results.append(summary.results_df)
                    logger.info(f"  ✓ {len(summary.results_df)} predictions added")
                else:
                    logger.warning(f"  ✗ No results for {player_name} - {market}")
                    
            except Exception as e:
                logger.error(f"  ✗ Error for {player_name} - {market}: {e}")
                continue
            
            completed += 1
            if progress_callback:
                progress_callback(completed / total_tasks)
    
    if not all_results:
        logger.error("No backtest results generated!")
        return pd.DataFrame()
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save to CSV
    output_path = DATA_DIR / output_file
    combined_df.to_csv(output_path, index=False)
    
    logger.info(f"✓ ML training data saved to {output_path}")
    logger.info(f"  Total samples: {len(combined_df)}")
    logger.info(f"  Columns: {list(combined_df.columns)}")
    logger.info(f"  Win rate: {combined_df['hit'].mean():.1%}")
    
    return combined_df


def generate_ml_data_streamlit():
    """Streamlit UI wrapper for ML data generation."""
    st.markdown("### 🤖 Generate ML Training Data")
    
    st.info("""
    This will backtest the top NBA players across multiple markets to create 
    a training dataset for machine learning models.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        num_players = st.slider("Number of Players", 10, 100, 50, 10)
        test_days = st.slider("Games per Player", 20, 100, 60, 10)
    
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
        self.data_loader = DataLoader(config)
        self.feature_engineer = FeatureEngineer(config)
        self.model_engine = ModelEngine(config)
        self.simulation_engine = SimulationEngine(config)
        self.decision_policy = DecisionPolicy(config)
        self.decision_policy = DecisionPolicy(config)
        self.tracker = Tracker()

        self.data_loader = DataLoader(config)
        self.feature_engineer = FeatureEngineer(config)
        self.model_engine = ModelEngine(config, tracker=self.tracker) 
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
                usage_mult=usage_mult,
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
                hit_rates=hit_rates
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

    def get_bets(self) -> list: 
        try:
            if self.file.exists():
                with open(self.file, 'r') as f:
                    content = f.read()
                    return json.loads(content) if content else []
            return []
        except Exception:
            return []

    def save_bet(self, bet_data: Dict):
        bets = self.get_bets()
        bets.append(bet_data)
        self._save(bets)

    def _save(self, bets: list):
        try:
            with open(self.file, 'w') as f:
                json.dump(bets, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save bets: {e}")

    def get_player_history(self, player_name: str) -> List[Dict]:
        """
        Retrieve all past bets for a specific player.
        FIX: Uses get_bets() to safely load data from disk.
        """
        all_bets = self.get_bets()
        # Filter bets for this player
        return [bet for bet in all_bets if bet.get('player_name') == player_name]

    def update_result(self, bet_id: int, new_status: str, closing_line: float = None, actual_value: float = None):
        """Update bet result with actual value tracking for ML score box analysis."""
        current_bets = self.get_bets()
        for bet in current_bets:
            if bet.get('id') == bet_id: 
                bet['result'] = new_status
                
                # Store actual value and calculate margin
                if actual_value is not None:
                    bet['actual_value'] = actual_value
                    line = bet.get('line', 0)
                    side = bet.get('side', 'OVER')
                    
                    # Calculate margin (positive = favorable for bet side)
                    if side == 'OVER':
                        # For OVER: margin = actual - line (positive = hit)
                        bet['margin'] = actual_value - line
                    else:
                        # For UNDER: margin = line - actual (positive = hit)
                        bet['margin'] = line - actual_value
                    
                    # Categorize result quality
                    bet['result_quality'] = self._categorize_result(new_status, bet['margin'])
                
                if closing_line is not None:
                    bet['closing_line'] = closing_line
                    # Calculate CLV (positive = got better line than close)
                    opening_line = bet.get('line', 0)
                    side = bet.get('side', 'OVER')
                    if side == 'OVER':
                        # For OVER, lower closing line = CLV positive
                        bet['clv'] = closing_line - opening_line
                    else:
                        # For UNDER, higher closing line = CLV positive
                        bet['clv'] = opening_line - closing_line
                break
        self._save(current_bets)
    
    def _categorize_result(self, result: str, margin: float) -> str:
        """
        Categorize result quality based on margin.
        Helps ML differentiate bad beats from bad reads.
        
        Args:
            result: Win, Loss, or Push
            margin: Positive = favorable for bet side
        """
        abs_margin = abs(margin)
        
        if result == 'Push' or abs_margin < 0.5:
            return ResultQuality.PUSH.value
        elif result == 'Win':
            if abs_margin <= 1.5:
                return ResultQuality.SWEAT_WIN.value
            elif abs_margin <= 3.5:
                return ResultQuality.CLOSE_WIN.value
            elif abs_margin <= 7.5:
                return ResultQuality.SOLID_WIN.value
            else:
                return ResultQuality.BLOWOUT_WIN.value
        elif result == 'Loss':
            if abs_margin <= 1.5:
                return ResultQuality.BAD_BEAT.value
            elif abs_margin <= 3.5:
                return ResultQuality.CLOSE_LOSS.value
            elif abs_margin <= 7.5:
                return ResultQuality.CLEAR_LOSS.value
            else:
                return ResultQuality.BAD_READ.value
        else:
            return ResultQuality.PENDING.value

    def delete_bet(self, bet_id: int):
        current_bets = self. get_bets()
        current_bets = [b for b in current_bets if b.get('id') != bet_id]
        self._save(current_bets)

    def clear_history(self):
        self._save([])

    def _save(self, data:  list):
        try:
            with open(self.file, 'w') as f:
                json.dump(data, f, indent=4)
        except IOError as e: 
            logger.error(f"Failed to save bet tracker: {e}")
            st.error("Failed to save bet data")

    def get_stats(self) -> dict:
        bets = self.get_bets()
        wins = len([b for b in bets if b.get('result') == 'Win'])
        losses = len([b for b in bets if b.get('result') == 'Loss'])
        pushes = len([b for b in bets if b.get('result') == 'Push'])
        pending = len([b for b in bets if b.get('result') == 'Pending'])
        total_decided = wins + losses
        win_rate = wins / total_decided if total_decided > 0 else 0
        total_profit = 0
        for bet in bets: 
            if bet.get('result') == 'Win':
                total_profit += bet.get('stake', 0) * (bet.get('odds', 1) - 1)
            elif bet.get('result') == 'Loss':
                total_profit -= bet.get('stake', 0)
        
        # CLV Stats
        bets_with_clv = [b for b in bets if b.get('clv') is not None]
        clv_count = len(bets_with_clv)
        avg_clv = sum(b['clv'] for b in bets_with_clv) / clv_count if clv_count > 0 else 0
        positive_clv = len([b for b in bets_with_clv if b['clv'] > 0])
        clv_positive_rate = positive_clv / clv_count if clv_count > 0 else 0
        
        # Score Box Stats (Result Quality Breakdown)
        bets_with_margin = [b for b in bets if b.get('margin') is not None]
        margin_count = len(bets_with_margin)
        avg_margin = sum(b['margin'] for b in bets_with_margin) / margin_count if margin_count > 0 else 0
        
        # Count by result quality
        quality_counts = {}
        for rq in ResultQuality:
            quality_counts[rq.value] = len([b for b in bets if b.get('result_quality') == rq.value])
        
        # Bad beat rate (losses by <= 1.5)
        bad_beats = quality_counts.get(ResultQuality.BAD_BEAT.value, 0)
        bad_reads = quality_counts.get(ResultQuality.BAD_READ.value, 0)
        bad_beat_rate = bad_beats / losses if losses > 0 else 0
        bad_read_rate = bad_reads / losses if losses > 0 else 0
        
        # Close win rate (wins by <= 1.5)
        sweat_wins = quality_counts.get(ResultQuality.SWEAT_WIN.value, 0)
        solid_wins = quality_counts.get(ResultQuality.SOLID_WIN.value, 0) + quality_counts.get(ResultQuality.BLOWOUT_WIN.value, 0)
        sweat_win_rate = sweat_wins / wins if wins > 0 else 0
        
        return {
            'wins': wins, 'losses': losses, 'pushes': pushes, 'pending': pending,
            'total_decided': total_decided, 'win_rate': win_rate, 'total_profit': total_profit,
            'clv_count': clv_count, 'avg_clv': avg_clv, 'clv_positive_rate': clv_positive_rate,
            # Score Box Stats
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
        
        # Add margin-based targets for regression/nuanced classification
        if 'margin' in df.columns:
            # Continuous target: margin (can be used for regression)
            df['target_margin'] = df['margin'].fillna(0)
            
            # Quality-based target: 0=bad_read, 1=clear_loss, 2=close_loss, 3=bad_beat, 4=push, 5=sweat_win, 6=close_win, 7=solid_win, 8=blowout_win
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
        Export completed bets (with actual scores) to ML training CSV.
        Only exports bets that haven't been exported yet.
        
        Returns:
            Tuple of (num_exported, output_path)
        """
        import csv
        
        bets = self.get_bets()
        output_path = DATA_DIR / output_file
        
        # Define all feature columns
        fieldnames = [
            # Identifiers
            'date', 'player', 'opponent', 'market', 'position',
            # Bet info
            'line', 'predicted_side', 'predicted_prob', 'predicted_ev', 'projected_value',
            # Outcomes
            'result', 'hit', 'actual_value', 'margin', 'result_quality',
            # Core features
            'feat_ema', 'feat_std', 'feat_sma_5', 'feat_sma_10', 'feat_trend',
            'feat_avg_minutes', 'feat_mins_trend',
            # Hit rates
            'feat_hit_l5', 'feat_hit_l10', 'feat_hit_l15', 'feat_hit_season',
            # Matchup multipliers
            'feat_pace_mult', 'feat_def_mult', 'feat_position_mult',
            'feat_base_matchup_mult', 'feat_combined_matchup_mult',
            # Context factors
            'feat_split_factor', 'feat_rest_factor', 'feat_blowout_factor', 'feat_usage_mult',
            # Game context
            'feat_spread', 'feat_is_home', 'feat_is_b2b', 'feat_days_rest',
            'feat_game_total', 'feat_opp_drtg_season', 'feat_opp_drtg_l5',
            # Meta
            'feat_games_played', 'closing_line', 'clv'
        ]
        
        # Check if file exists to determine if we need headers
        file_exists = output_path.exists()
        
        exported_count = 0
        bets_modified = False
        
        with open(output_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            
            # Write header only if new file
            if not file_exists:
                writer.writeheader()
            
            for bet in bets:
                # Only export completed bets with actual values that haven't been exported
                if (bet.get('result') in ['Win', 'Loss'] and 
                    bet.get('actual_value') is not None and 
                    not bet.get('exported_to_csv', False)):
                    
                    # Build the row
                    row = {
                        'date': bet['date'].split(' ')[0],
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
                        'result_quality': bet.get('result_quality', ''),
                        # Features
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
                        'clv': bet.get('clv', 0)
                    }
                    
                    writer.writerow(row)
                    bet['exported_to_csv'] = True
                    bets_modified = True
                    exported_count += 1
        
        # Save the exported flags
        if bets_modified:
            self._save(bets)
        
        return exported_count, str(output_path)

    def get_exportable_count(self) -> int:
        """Count bets ready to export (completed with actual values, not yet exported)."""
        bets = self.get_bets()
        return len([
            b for b in bets 
            if b.get('result') in ['Win', 'Loss'] 
            and b.get('actual_value') is not None 
            and not b.get('exported_to_csv', False)
        ])

    def get_feature_stats(self) -> dict:
        bets = self.get_bets()
        total_with_features = len([b for b in bets if b. get('feat_ema') is not None])
        decided_with_features = len([
            b for b in bets
            if b.get('result') in ['Win', 'Loss'] and b.get('feat_ema') is not None
        ])
        exportable = self.get_exportable_count()
        return {
            'total_with_features': total_with_features,
            'decided_with_features': decided_with_features,
            'pending_with_features':  total_with_features - decided_with_features,
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


def render_recommendation_card(result: AnalysisResult, bankroll: float):
    """Render clean, mobile-friendly recommendation card with V15 insights."""
    decision = result.decision
    projection = result.projection
    features = result.features
    simulation = result.simulation
    
    # Grade colors
    grade_colors = {'A': 'green', 'B': 'green', 'C': 'orange', 'D': 'red', 'F': 'red'}
    grade_color = grade_colors.get(decision.grade.value, 'gray')
    
    # V15: Blowout warning based on probability
    blowout_warn = ""
    if features.blowout_prob >= 0.35:
        blowout_warn = " 🔴"  # High risk
    elif features.blowout_prob >= 0.25:
        blowout_warn = " ⚠️"  # Moderate risk  
    elif features.blowout_prob >= 0.15:
        blowout_warn = " ⚡"  # Slight risk
    
    # Data quality warning icon
    dq = features.data_quality
    dq_warn = "" if dq.score >= 75 else (" 📡" if dq.score >= 50 else " 📡⚠️")
    
    # Header with recommendation
    st.markdown(f"### {decision.recommended_side} {result.line}{blowout_warn}{dq_warn} — :{grade_color}[Grade {decision.grade.value}]")
    
    # Key metrics in columns (add ML if available)
    if projection.ml_prob is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Win Prob", f"{decision.probability:.0%}")
        c2.metric("EV", f"{decision.expected_value:+.1%}")
        c3.metric("Stake", f"₱{decision.kelly_stake:.0f}", f"{decision.kelly_fraction:.1%}")
        
        # ML Brain indicator with LOVE/LIKE/NEUTRAL/FADE label
        ml_pct = projection.ml_prob
        if ml_pct > 0.60:
            ml_label = "Robot LOVE"
        elif ml_pct > 0.55:
            ml_label = "Robot LIKE"
        elif ml_pct < 0.45:
            ml_label = "Robot FADE"
        else:
            ml_label = "Robot Neutral"
        ml_delta = f"{(ml_pct - 0.5) * 100:+.0f}pp"
        c4.metric(ml_label, f"{ml_pct:.0%}", ml_delta)
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Win Prob", f"{decision.probability:.0%}")
        c2.metric("EV", f"{decision.expected_value:+.1%}")
        c3.metric("Stake", f"₱{decision.kelly_stake:.0f}", f"{decision.kelly_fraction:.1%}")
    
    # Show context (includes matchup, B2B, blowout risk, ML confidence)
    if projection.context:
        st.caption(projection.context)
    
    # Compact info line with V15 dynamic std multiplier
    st.caption(f"📊 Proj: {projection.final_projection:.1f} | 90% CI: [{simulation.ci_10:.1f}-{simulation.ci_90:.1f}] | CV: {features.coef_variation:.0%} | Sim Width: {features.dynamic_std_mult:.2f}x")
    
    # V15 Intelligent Model Insights (compact expander)
    with st.expander("V15 Intelligence", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        # Blowout Risk Model
        with col1:
            prob_pct = features.blowout_prob * 100
            if prob_pct >= 35:
                color = "🔴"
                label = "HIGH"
            elif prob_pct >= 25:
                color = "🟠"
                label = "MOD"
            elif prob_pct >= 15:
                color = "🟡"
                label = "SLIGHT"
            else:
                color = "🟢"
                label = "LOW"
            st.metric(f"{color} Blowout Risk", f"{prob_pct:.0f}%", label)
        
        # Fatigue Profile
        with col2:
            fatigue_pct = (1 - features.personal_fatigue_factor) * 100
            b2b_count = features.b2b_games_in_sample
            if fatigue_pct > 5:
                color = "🔴"
                label = f"-{fatigue_pct:.0f}% B2B"
            elif fatigue_pct > 0:
                color = "🟡"
                label = f"-{fatigue_pct:.0f}% B2B"
            elif fatigue_pct < -2:
                color = "🟢"
                label = f"+{-fatigue_pct:.0f}% B2B"
            else:
                color = "⚪"
                label = "Neutral"
            st.metric(f"{color} Fatigue Profile", label, f"({b2b_count} B2B games)")
        
        # Volatility Profile
        with col3:
            cv_pct = features.coef_variation * 100
            std_mult = features.dynamic_std_mult
            if cv_pct < 20:
                color = "🟢"
                label = "Consistent"
            elif cv_pct < 30:
                color = "🟡"
                label = "Average"
            else:
                color = "🔴"
                label = "Volatile"
            st.metric(f"{color} Volatility", f"{cv_pct:.0f}% CV", f"Sim: {std_mult:.2f}x width")
    
    # Data quality card
    render_data_quality_card(result)
    
    # Confidence warning
    if decision.confidence_warning:
        st.warning(f"⚠️ {decision.confidence_warning}", icon=None)
    
    # Compact rollover badge
    if decision.rollover_suitable:
        st.success(f"🎲 **Parlay OK** — Score: {decision.rollover_score:.1f}/5")
    else:
        st.info(f"🎲 Not for parlays — Score: {decision.rollover_score:.1f}/5")


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
    st.markdown("###Train V15 Model")
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
            
            # Filter for feature columns (must start with 'feat_')
            feature_cols = [c for c in df.columns if c.startswith('feat_')]
            target_col = 'hit'
            
            if len(feature_cols) != 33:
                status.update(label="❌ Error", state="error")
                st.error(f"Shape Mismatch! Found {len(feature_cols)} features, expected 33. Regenerate your dataset.")
                return

            # 2. Prepare Data
            status.write(f"Processing {len(df)} samples with {len(feature_cols)} features...")
            X = df[feature_cols]
            y = df[target_col]
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
        
        usage_bump = st.slider("Usage Adj %", -30, 30, 0, 5)
        usage_mult = 1 + (usage_bump / 100.0)
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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Analyze", "📈 Backtest", "👁️ Watch", "⚔️ H2H", "⚖️ Splits", "📝 Bets", "🎲 Parlays", "🤖 ML"
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
        with st.expander("⚙️ Advanced Options", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                is_home = st.checkbox("🏠 Home", True)
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
        
        if st.button("🔍 Run Analysis", type="primary", disabled=not player_in or not is_valid):
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
            
            # Main recommendation at top
            render_recommendation_card(result, bankroll)
            
            # Bet Rules Validation
            render_bet_rules_card(result)
            
            # Action buttons
            col_track, col_parlay = st.columns(2)
            with col_track: 
                if st.button("💾 Track", use_container_width=True):
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
                    })
                    st.toast("Bet tracked!", icon="💾")
                
            with col_parlay: 
                if result.decision.rollover_suitable:
                    if st.button("🎲 Parlay", use_container_width=True):
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
                    st.button("🎲 Parlay", disabled=True, use_container_width=True,
                             help="Not suitable for parlay")
            
            # Details in expander to reduce clutter
            with st.expander("📊 Details & Chart", expanded=False):
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
            st.markdown(f"### ⚔️ {result.player_name} vs {result.opponent_name}")
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
            
            st.markdown(f"### ⚖️ {result.player_name} Splits")
            
            splits = {
                "📊 L15": df.tail(15),
                "🏠 Home": df[df['IS_HOME'] == True].tail(15),
                "✈️ Away": df[df['IS_HOME'] == False].tail(15),
                "✅ W": df[df['WL'] == 'W'].tail(15),
                "❌ L": df[df['WL'] == 'L'].tail(15),
            }
            
            if 'DAYS_REST' in df.columns:
                splits["😴 Rest"] = df[df['DAYS_REST'] >= 3].tail(15)
                splits["🏃 B2B"] = df[df['DAYS_REST'] <= 1].tail(15)
            
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
        st.markdown("### 📝 Bet Tracker")
        
        stats = tracker.get_stats()
        
        if stats['total_decided'] > 0:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Win%", f"{stats['win_rate']:.0%}")
            c2.metric("Record", f"{stats['wins']}-{stats['losses']}")
            c3.metric("P/L", f"₱{stats['total_profit']:+,.0f}")
            c4.metric("Pending", stats['pending'])
            
            # Score Box Summary (if we have margin data)
            if stats.get('margin_count', 0) > 0:
                with st.expander("📊 Score Box Analysis", expanded=False):
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
        
        all_bets = tracker.get_bets()
        if all_bets: 
            filter_opts = ["Pending", "Win", "Loss", "Push"]
            selected_filters = st.multiselect("Filter by Result:", options=filter_opts, default=["Pending", "Win", "Loss"])
            filtered_bets = [b for b in all_bets if b.get('result', 'Pending') in selected_filters]
            
            sort_map = {'Pending': 1, 'Win': 2, 'Loss': 3, 'Push': 4}
            filtered_bets.sort(key=lambda x: (sort_map.get(x.get('result', 'Pending'), 99), -x.get('id', 0)))
            
            if not filtered_bets: 
                st.info("No bets match these filters.")
            
            for bet in filtered_bets:
                with st.container(border=True):
                    bet_res = bet.get('result', 'Pending')
                    icon = {"Win": "✅", "Loss": "❌", "Push": "🔄"}.get(bet_res, "⏳")
                    color = {"Win": "green", "Loss": "red", "Push": "blue"}.get(bet_res, "gray")
                    
                    # Compact header line
                    st.markdown(f"{icon} :{color}[**{bet['player']}**] {bet['side']} {bet['line']} ({bet['market']})")
                    st.caption(f"vs {bet.get('opponent')} | Proj: {bet.get('proj', 0):.1f} | EV: {bet.get('ev', 0):+.1%} | {bet.get('date', 'N/A')}")
                    
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
                        quality = bet.get('result_quality', 'unknown')
                        quality_emoji = {
                            'bad_beat': '💔', 'close_loss': '😤', 'clear_loss': '📉', 'bad_read': '🚫',
                            'sweat_win': '😅', 'close_win': '✌️', 'solid_win': '💪', 'blowout_win': '🔥',
                            'push': '🔄'
                        }.get(quality, '❓')
                        margin_color = 'green' if margin > 0 else ('red' if margin < 0 else 'gray')
                        st.caption(f"{quality_emoji} :{margin_color}[Margin: {margin:+.1f}] | Quality: {quality.replace('_', ' ').title()}")
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


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__": 
    main()
                
