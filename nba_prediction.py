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
from xgboost import XGBRegressor  # V19: Regression instead of Classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats  # V19: For CDF-based probability calculation
import requests
from bs4 import BeautifulSoup
from data_engine import DataEngine

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Guard Streamlit UI initialization - only run in main process (not worker subprocesses)
# On Windows, ProcessPoolExecutor uses "spawn" which re-imports the module
import multiprocessing
import sys

def _is_main_streamlit_process() -> bool:
    """Check if we're in the main Streamlit process (not a worker subprocess)."""
    # Check 1: multiprocessing process name
    if multiprocessing.current_process().name != 'MainProcess':
        return False
    # Check 2: On Windows spawned processes, __spec__ is different
    if hasattr(sys.modules[__name__], '__mp_main__'):
        return False
    # Check 3: Check if parent is a spawn worker
    if multiprocessing.parent_process() is not None:
        return False
    return True

_IN_MAIN_PROCESS = _is_main_streamlit_process()

if _IN_MAIN_PROCESS:
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

# Conditional cache decorator - no-op in worker processes to avoid Streamlit warnings
def _conditional_cache(ttl: int):
    """Returns st.cache_data in main process, identity decorator in workers."""
    if _IN_MAIN_PROCESS:
        return st.cache_data(ttl=ttl)
    else:
        # No-op decorator for worker processes
        def identity_decorator(func):
            return func
        return identity_decorator

# Initialize Data Engine singleton with st.cache_resource
if _IN_MAIN_PROCESS:
    @st.cache_resource
    def get_data_engine(data_dir: Path, season: str) -> DataEngine:
        """Singleton provider for DataEngine using Streamlit cache."""
        return DataEngine(data_dir=data_dir, season=season)

logging.basicConfig(level=logging.INFO if _IN_MAIN_PROCESS else logging.WARNING)
logger = logging.getLogger(__name__)

CURRENT_VERSION = 'v20.3'


@dataclass(frozen=True)
class Config:
    """
    V20 EMPIRICAL CONFIG: No beliefs, only observables.
    
    All heuristic multipliers, penalties, thresholds REMOVED.
    Model learns relationships from data, not hardcoded assumptions.
    """
    # === IDENTIFIERS ===
    CURRENT_SEASON: str = "2025-26"
    PREV_SEASON: str = "2024-25"
    
    # === MULTI-SEASON TRAINING (V20.1) ===
    # More data = better generalization
    TRAINING_SEASONS: tuple = ("2025-26", "2024-25", "2023-24", "2022-23")
    
    # === API / CACHE ===
    API_DELAY: float = 1.0
    API_MAX_RETRIES: int = 3
    CACHE_TTL_TEAM_STATS: int = 3600
    CACHE_TTL_PLAYER_IDS: int = 86400
    CACHE_TTL_GAME_LOGS: int = 1800
    
    # === ML COMMITMENT ===
    # ML is the SOLE prediction engine. No fallbacks.
    ML_REQUIRE_MODEL: bool = True
    ML_CALIBRATION_ENABLED: bool = True
    
    # === DATA DEFAULTS (not heuristics - just fallbacks for missing data) ===
    DEFAULT_PACE: float = 100.0
    DEFAULT_DEF_RATING: float = 115.0
    DEFAULT_GAME_TOTAL: float = 225.0
    
    # === TRACKING LIMITS ===
    MAX_PARLAY_LEGS: int = 10
    BACKTEST_DEFAULT_DAYS: int = 30
    BACKTEST_MIN_GAMES: int = 5
    
    # === LOW COUNT STAT IDENTIFICATION (for Poisson modeling, not heuristic adjustment) ===
    LOW_COUNT_STATS: tuple = ('3PM', 'FG3M', 'STL', 'BLK', 'TOV')


CONFIG = Config()

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
TRACKER_FILE = DATA_DIR / "bet_tracker.json"
PARLAY_FILE = DATA_DIR / "parlay_tracker.json"
WATCHLIST_FILE = DATA_DIR / "watchlist.json"

        
# =============================================================================
# DATA STRUCTURES
# =============================================================================

# V20 EMPIRICAL: No grades, no risk categories - just data
# ResultQuality kept for audit trail only (post-hoc analysis)


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
    team_stats_age_hours: float = 0.0  # How old is the team stats cache
    
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
    V20.3 EMPIRICAL: Pure observable feature vector with absence-awareness.
    
    DESIGN PRINCIPLES (per refactor directive):
    - Raw observables only (no computed multipliers)
    - Timestamp-valid (frozen before game tip-off)
    - Non-decisional (no belief-encoded adjustments)
    - Let ML learn all relationships from data
    
    V20.3 NEW FEATURES (absence-aware from game logs):
    - team_out_ppg: Total PPG of teammates who didn't play (ground truth)
    - team_out_count: Count of teammates who didn't play
    - opp_out_ppg: Total PPG of opponents who didn't play
    - opp_out_count: Count of opponents who didn't play
    
    -1.0 sentinel = unknown (for live predictions or missing data)
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
    opponent_pace: float = 100.0    # Opponent pace (possessions/game)
    team_pace: float = 100.0        # Player's team pace
    trend_5g: float = 0.0           # Slope of last 5 games (momentum)
    home_avg: float = 0.0           # Player's home game average
    away_avg: float = 0.0           # Player's away game average

    # === V20.3 NEW: TRUE-SHOOTING% FEATURES ===
    # Rolling TS% (fraction 0-1) over lookback window; -1.0 sentinel if unavailable
    feat_ts_pct: float = -1.0
    # Delta between rolling TS% and season TS%
    feat_ts_pct_delta: float = -1.0
    
    # === V20.3 NEW: ABSENCE-AWARE FEATURES ===
    # Computed from game logs (ground truth) during training
    # -1.0 = unknown (for live predictions using injury report proxy)
    team_out_ppg: float = -1.0      # PPG of teammates who didn't play
    team_out_count: int = -1        # Count of teammates out
    opp_out_ppg: float = -1.0       # PPG of opponents who didn't play
    opp_out_count: int = -1         # Count of opponents out
    
    # === MARKET IDENTITY (one-hot encoded) ===
    market_scoring: int = 0  # PTS
    market_counting: int = 0 # REB, AST  
    market_combo: int = 0    # PRA, PR, PA, RA
    market_rare: int = 0     # 3PM, STL, BLK

    # === BEHAVIOR & RISK FEATURES ===
    # Rolling std dev of minutes (risk)
    feat_min_volatility: float = -1.0
    # Rolling mean of personal fouls (behavior)
    feat_foul_rate: float = -1.0
    # Coefficient of variation for target stat (std / ema)
    feat_cv: float = -1.0
    
    # === DATA QUALITY (audit only, not used in ML) ===
    data_quality: DataQuality = field(default_factory=DataQuality)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for ML pipeline."""
        return asdict(self)
    
    def to_ml_array(self) -> np.ndarray:
        """
        Convert to numpy array for ML models.
        V20.3: 24 raw observable features (was 20 in V20.2).
        """
        numeric_features = [
            # Original 15 features
            self.avg_minutes,
            self.ema,
            self.std,
            self.opponent_drtg_season,
            self.line,
            self.spread,
            self.game_total,
            float(self.days_rest),
            float(self.is_home),
            float(self.is_b2b),
            float(self.games_played),
            # V20.2: 5 additional features
            self.opponent_pace,
            self.team_pace,
            self.trend_5g,
            self.home_avg,
            self.away_avg,
            # V20.3 NEW: TS% features
            self.feat_ts_pct,
            self.feat_ts_pct_delta,
            # V20.3 NEW: 4 absence-aware features
            self.team_out_ppg,
            float(self.team_out_count),
            self.opp_out_ppg,
            float(self.opp_out_count),
            # Market identity (4)
            float(self.market_scoring),
            float(self.market_counting),
            float(self.market_combo),
            float(self.market_rare),
            # V20.3 NEW: Behavior & Risk features (added to end of feature vector)
            float(self.feat_min_volatility),
            float(self.feat_foul_rate),
            float(self.feat_cv),
        ]
        return np.array(numeric_features)


@dataclass
class Projection:
    """
    V19 REFACTOR: Model projection output.
    
    Outputs only empirically meaningful values:
    - predicted_mean: E[stat] from regression model
    - predicted_std: σ[stat] uncertainty
    - ml_prob: P(Over) from CDF
    """
    base_projection: float
    final_projection: float
    confidence_interval:  Tuple[float, float]
    adjustments: Dict[str, float]
    ml_prob: Optional[float] = None  # P(Over) from CDF


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
    """
    V19 REFACTOR: Pure statistical output only.
    
    Removed: grade, rollover_suitable, rollover_score, confidence_warning
    (These were "fake certainty" - making guesses look like facts)
    
    Standard output: EV, predicted_mean (via projection), predicted_std, win_prob
    """
    recommended_side: str  # "OVER" or "UNDER"
    probability: float     # P(winning the recommended side)
    expected_value: float  # EV at given odds
    predicted_mean: float  # E[stat] from model
    predicted_std: float   # σ[stat] uncertainty
    kelly_stake: float     # Suggested stake (EV-based)
    kelly_fraction: float  # Stake as fraction of bankroll


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
    
    # V18: ML Calibration Details
    ml_details: Optional[Dict[str, Any]] = None  # Raw vs calibrated prob, model group


@dataclass
class BacktestResult:
    """
    Single backtest prediction result with TEMPORAL INTEGRITY.
    
    V19: All features are frozen snapshots from prediction time,
    ensuring honest walk-forward evaluation.
    """
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
    # Additional context for ML training
    opponent: str = ''  # Opponent team abbreviation
    position: str = ''  # Player position
    is_home: bool = True  # Was player at home
    # ML training features (FROZEN at prediction time)
    features: Dict[str, Any] = field(default_factory=dict)
    # V20 TEMPORAL AUDIT FIELDS (metadata only, no decay applied)
    snapshot_date: str = ''  # Date when prediction was made
    snapshot_season: str = ''  # Season context (metadata)


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


class MLModelRequiredError(Exception):
    """
    V19 ML COMMITMENT: Raised when ML model is required but unavailable.
    
    This enforces the commitment to ML as the sole prediction engine.
    If you see this error, you need to:
    1. Train models using data/train_model_v19.py
    2. Ensure .pkl files are in the data/ directory
    3. Or set CONFIG.ML_REQUIRE_MODEL = False (not recommended)
    """
    pass


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
    
    def __init__(self, config: Config = CONFIG, data_engine: DataEngine = None):
        self.config = config
        self._position_cache: Dict[int, str] = {}
        self._team_stats_cache: Optional[Tuple[pd.DataFrame, float, float]] = None
        self._team_stats_cache_time: float = 0
        # Position Usage Map: {team_id: {position: mean_usg_pct}}
        self._pos_usage_map: Optional[Dict[int, Dict[str, float]]] = None

        # Integrate DataEngine
        if data_engine:
            self.data_engine = data_engine
        elif _IN_MAIN_PROCESS:
            # Use singleton if available
            self.data_engine = get_data_engine(DATA_DIR, config.CURRENT_SEASON)
        else:
            # Create fresh instance for worker process
            # Note: Workers will share disk cache (parquet) so this is efficient
            self.data_engine = DataEngine(DATA_DIR, config.CURRENT_SEASON)
    
    def _api_call_with_retry(self, func, description: str = "API call"):
        """Execute API call with retry logic. Exponential backoff for rate limits."""
        import multiprocessing
        import random
        last_exception = None
        cooldown_count = 0
        MAX_COOLDOWNS = 5  # More patient - 5 cooldowns before failing
        
        current_process = multiprocessing.current_process()
        if current_process.name != 'MainProcess':
            error_msg = (
                f"🚨 API CALL BLOCKED IN WORKER PROCESS! 🚨\n"
                f"  Process: {current_process.name}\n"
                f"  Description: {description}\n"
                f"  This indicates data was not properly pre-loaded.\n"
                f"  Stack trace will show which function tried to call API."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

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
    
    @_conditional_cache(ttl=CONFIG.CACHE_TTL_PLAYER_IDS)
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
    
    @_conditional_cache(ttl=CONFIG.CACHE_TTL_GAME_LOGS)
    def fetch_game_logs(self, player_id: int, season: str = None) -> pd.DataFrame:
        """
        Fetch player game logs for a season.
        Uses DataEngine for bulk fetching and caching.
        """
        if season is None:
            season = self.config.CURRENT_SEASON
        
        try: 
            # Use DataEngine to get player logs from cached master file
            df = self.data_engine.get_player_logs(player_id, season)
            
            if df.empty:
                return pd.DataFrame()
            
            # === Feature Engineering ===
            # Replicate the logic that was previously done after API fetch

            # Ensure sorting
            if 'GAME_DATE' not in df.columns:
                # Should have been handled by DataEngine, but double check
                df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

            df = df.sort_values('GAME_DATE')
            
            # DataEngine already handles numeric optimization for basic stats
            # (PTS, REB, AST, etc.)

            # Create combo stats (these are derived, not in raw logs)
            df['PRA'] = df['PTS'] + df['REB'] + df['AST']
            df['PA'] = df['PTS'] + df['AST']
            df['PR'] = df['PTS'] + df['REB']
            df['RA'] = df['REB'] + df['AST']
            
            # Home/Away logic
            if 'MATCHUP' in df.columns:
                df['IS_HOME'] = df['MATCHUP'].str.contains('vs. ', na=False)

            # 3PM alias
            if 'FG3M' in df.columns:
                df['3PM'] = df['FG3M']
            
            # Days Rest
            df['DAYS_REST'] = df['GAME_DATE'].diff().dt.days - 1
            df['DAYS_REST'] = df['DAYS_REST'].fillna(3).clip(lower=0) 
            
            # Minutes parsing is handled by DataEngine (MIN_FLOAT created there)
            # If for some reason MIN_FLOAT is missing (e.g. old cache), fallback
            if 'MIN_FLOAT' not in df.columns:
                if 'MIN' in df.columns:
                    # Quick parse fallback
                    def parse_minutes(min_val):
                        try:
                            if pd.isna(min_val): return 0.0
                            if isinstance(min_val, (float, int)): return float(min_val)
                            if isinstance(min_val, str):
                                if ':' in min_val:
                                    parts = min_val.split(':')
                                    return float(parts[0]) + float(parts[1]) / 60
                                return float(min_val)
                            return float(min_val)
                        except: return 0.0
                    df['MIN_FLOAT'] = df['MIN'].apply(parse_minutes)
                else:
                    df['MIN_FLOAT'] = 0.0
            
            # Per-minute stats
            target_stats = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M']
            for stat in target_stats:
                if stat in df.columns:
                    df[f'{stat}_PER_MIN'] = df[stat] / df['MIN_FLOAT']
                    df[f'{stat}_PER_MIN'] = df[f'{stat}_PER_MIN'].fillna(0.0)

            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch game logs for player {player_id}: {e}")
            return pd.DataFrame()
    
    def fetch_multi_season_logs(self, player_id: int) -> pd.DataFrame:
        """Fetch logs from current and previous season."""
        try:
            # DataEngine makes this fast - hits local parquet for both seasons
            df_current = self.fetch_game_logs(player_id, self.config.CURRENT_SEASON)
            df_prev = self.fetch_game_logs(player_id, self.config.PREV_SEASON)

            frames = [df for df in [df_current, df_prev] if len(df) > 0]
            if not frames:
                return pd.DataFrame()

            combined = pd.concat(frames, ignore_index=True)
            return combined.sort_values('GAME_DATE').reset_index(drop=True)

        except Exception as e: 
            logger.error(f"Failed to fetch multi-season logs: {e}")
            return pd.DataFrame()
    
    @_conditional_cache(ttl=CONFIG.CACHE_TTL_TEAM_STATS)
    def fetch_team_stats(_self, season: str = None) -> Tuple[pd.DataFrame, float, float]:
        """Fetch league-wide team stats for a specific season."""
        if season is None:
            season = _self.config.CURRENT_SEASON
        try:
            def api_call():
                return leaguedashteamstats.LeagueDashTeamStats(
                    season=season,
                    measure_type_detailed_defense='Advanced',
                    per_mode_detailed='PerGame'
                ).get_data_frames()[0]
            
            stats = _self._api_call_with_retry(api_call, f"Fetch team stats ({season})")
            
            if stats is None or len(stats) == 0:
                logger.warning(f"No team stats available for {season}, using defaults")
                return pd.DataFrame(), _self.config.DEFAULT_PACE, _self.config.DEFAULT_DEF_RATING
            
            if 'PACE' not in stats.columns:
                stats['PACE'] = _self.config.DEFAULT_PACE
            if 'DEF_RATING' not in stats.columns:
                stats['DEF_RATING'] = _self.config.DEFAULT_DEF_RATING
            
            return stats.set_index('TEAM_ID'), stats['PACE'].mean(), stats['DEF_RATING'].mean()
            
        except Exception as e:
            logger.error(f"Failed to fetch team stats for {season}: {e}")
            return pd.DataFrame(), _self.config.DEFAULT_PACE, _self.config.DEFAULT_DEF_RATING
    
    def fetch_all_seasons_team_stats(self, seasons: List[str] = None) -> Dict[str, Tuple[pd.DataFrame, float, float]]:
        """
        Fetch team stats for ALL training seasons.
        
        V20.3 TEMPORAL FIX: Each historical game uses that season's actual team stats,
        not current season stats (which would be temporal leakage).
        
        Returns: Dict[season, (team_stats_df, avg_pace, avg_def)]
        """
        if seasons is None:
            seasons = list(self.config.TRAINING_SEASONS)
        
        all_stats = {}
        for season in seasons:
            logger.info(f"📥 Fetching team stats for {season}...")
            time.sleep(self.config.API_DELAY)  # Rate limit protection
            stats_df, avg_pace, avg_def = self.fetch_team_stats(season)
            all_stats[season] = (stats_df, avg_pace, avg_def)
            logger.info(f"  ✓ {season}: {len(stats_df)} teams, pace={avg_pace:.1f}, drtg={avg_def:.1f}")
        
        return all_stats
    
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


# =============================================================================
# LAYER 2:  FEATURE ENGINEER
# =============================================================================

class InjuryManager:
    """
    V20 EMPIRICAL: Raw injury data fetcher.
    
    Provides observational injury status (who is OUT/DOUBTFUL).
    Does NOT calculate usage boosts or defensive impacts (belief-based).
    """
    def __init__(self, rapid_api_key: str = None):
        self.rapid_api_key = rapid_api_key
        # Cache for injury reports (30 min TTL)
        self._injury_cache: Dict[str, Any] = {}

    def _fetch_injury_report_espn(self, team_abbrev: str) -> List[Dict]:
        """
        Fetch REAL injury data from ESPN (publicly accessible).
        Returns list of injured players with status for a specific team.
        
        ESPN URL format: https://www.espn.com/nba/team/injuries/_/name/{abbrev}/{team-name}
        """
        cache_key = f"espn_injuries_{team_abbrev}"
        
        # Check cache (30 min TTL - injuries update frequently)
        if cache_key in self._injury_cache:
            cached = self._injury_cache[cache_key]
            if (datetime.now() - cached['time']).seconds < 1800:
                return cached['data']
        
        injuries = []
        
        # ESPN team name slugs
        espn_team_slugs = {
            'ATL': 'atlanta-hawks', 'BOS': 'boston-celtics', 'BKN': 'brooklyn-nets',
            'CHA': 'charlotte-hornets', 'CHI': 'chicago-bulls', 'CLE': 'cleveland-cavaliers',
            'DAL': 'dallas-mavericks', 'DEN': 'denver-nuggets', 'DET': 'detroit-pistons',
            'GSW': 'golden-state-warriors', 'HOU': 'houston-rockets', 'IND': 'indiana-pacers',
            'LAC': 'la-clippers', 'LAL': 'los-angeles-lakers', 'MEM': 'memphis-grizzlies',
            'MIA': 'miami-heat', 'MIL': 'milwaukee-bucks', 'MIN': 'minnesota-timberwolves',
            'NOP': 'new-orleans-pelicans', 'NYK': 'new-york-knicks', 'OKC': 'oklahoma-city-thunder',
            'ORL': 'orlando-magic', 'PHI': 'philadelphia-76ers', 'PHX': 'phoenix-suns',
            'POR': 'portland-trail-blazers', 'SAC': 'sacramento-kings', 'SAS': 'san-antonio-spurs',
            'TOR': 'toronto-raptors', 'UTA': 'utah-jazz', 'WAS': 'washington-wizards'
        }
        
        slug = espn_team_slugs.get(team_abbrev)
        if not slug:
            return []
        
        try:
            import re
            url = f"https://www.espn.com/nba/team/injuries/_/name/{team_abbrev.lower()}/{slug}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(' ', strip=True)
            
            # Use regex to find injury patterns: "PlayerName Position Status StatusType"
            # Pattern matches: "Bruce Brown G Status Day-to-day" or "Nikola Jokic C Status Out"
            pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z\'-]+)+)\s+([GFC])\s+Status\s+(Out|Day-to-day|Doubtful|Questionable|Probable)'
            matches = re.findall(pattern, text)
            
            seen_names = set()
            for match in matches:
                player_name = match[0].strip()
                position = match[1]
                status_raw = match[2]
                
                # Avoid duplicates
                if player_name in seen_names:
                    continue
                seen_names.add(player_name)
                
                # Normalize status
                status_map = {
                    'Out': 'OUT',
                    'Day-to-day': 'DAY-TO-DAY',
                    'Doubtful': 'DOUBTFUL',
                    'Questionable': 'QUESTIONABLE',
                    'Probable': 'PROBABLE'
                }
                status = status_map.get(status_raw, status_raw.upper())
                
                # Try to extract injury description from nearby text
                injury_desc = ''
                # Look for text after the player name containing common injury terms
                injury_pattern = rf'{re.escape(player_name)}.*?(?:due to|with|for)\s+([^.]+)'
                injury_match = re.search(injury_pattern, text, re.IGNORECASE)
                if injury_match:
                    injury_desc = injury_match.group(1)[:50].strip()
                
                injuries.append({
                    'name': player_name,
                    'status': status,
                    'position': position,
                    'injury': injury_desc,
                    'source': 'ESPN'
                })
            
            logger.info(f"Fetched {len(injuries)} injuries for {team_abbrev} from ESPN")
            
        except Exception as e:
            logger.warning(f"ESPN injury fetch failed for {team_abbrev}: {e}")
        
        self._injury_cache[cache_key] = {'time': datetime.now(), 'data': injuries}
        return injuries
    
    def _team_name_to_abbrev(self, team_name: str) -> Optional[str]:
        """Convert full team name to abbreviation."""
        name_map = {
            'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
            'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
            'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
            'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
            'LA Clippers': 'LAC', 'Los Angeles Clippers': 'LAC', 'LA Lakers': 'LAL',
            'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA',
            'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NOP',
            'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL',
            'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX', 'Portland Trail Blazers': 'POR',
            'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'TOR',
            'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
        }
        return name_map.get(team_name)

    def _fetch_team_injuries(self, team_id: int) -> List[Dict]:
        """
        Fetch injury list for a team.
        Uses ESPN for real injury data, falls back to NBA API roster.
        """
        # First try to get the team abbreviation
        team_abbrev = None
        try:
            team_info = teams.find_team_by_id(team_id)
            if team_info:
                team_abbrev = team_info.get('abbreviation')
        except:
            pass
        
        # Try ESPN first (real injury data)
        if team_abbrev:
            espn_data = self._fetch_injury_report_espn(team_abbrev)
            if espn_data:
                return espn_data
        
        # Fallback to roster-based detection
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
            
            for _, player in roster.iterrows():
                status = 'ACTIVE'
                
                if 'PLAYER_STATUS' in roster.columns:
                    ps = str(player.get('PLAYER_STATUS', '')).upper()
                    if 'OUT' in ps or 'INJ' in ps:
                        status = 'OUT'
                    elif 'DOUBT' in ps or 'GTD' in ps:
                        status = 'DOUBTFUL'
                
                if 'HOW_ACQUIRED' in roster.columns:
                    ha = str(player.get('HOW_ACQUIRED', '')).upper()
                    if 'INJ' in ha:
                        status = 'OUT'
                
                if status != 'ACTIVE':
                    data.append({
                        'name': player.get('PLAYER', ''),
                        'status': status,
                        'source': 'NBA Roster'
                    })
                    
        except Exception as e:
            logger.debug(f"Roster fetch failed: {e}")

        self._injury_cache[cache_key] = {'time': datetime.now(), 'data': data}
        return data
    
    def get_team_injury_report(self, team_abbrev: str) -> List[Dict]:
        """
        Public method to get injury report for a team by abbreviation.
        Used by UI to display injury status (raw observational data only).
        """
        return self._fetch_injury_report_espn(team_abbrev)
    
    def compute_absence_features_from_injuries(
        self,
        player_team_abbrev: str,
        opponent_abbrev: str,
        player_name: str,
        data_loader: 'DataLoader' = None
    ) -> Dict[str, float]:
        """
        V20.3: Compute absence features from injury reports for LIVE predictions.
        
        Uses injury report as PROXY for who will be absent (not ground truth).
        Only counts players with status 'OUT' or 'DOUBTFUL' (high confidence of absence).
        
        Args:
            player_team_abbrev: The player's team abbreviation (e.g., 'LAL')
            opponent_abbrev: Opponent team abbreviation (e.g., 'BOS')
            player_name: The player being analyzed (exclude from team injuries)
            data_loader: DataLoader instance for fetching player PPG
        
        Returns:
            Dict with team_out_ppg, team_out_count, opp_out_ppg, opp_out_count
            Returns -1.0 sentinel values if unable to fetch injury data
        """
        result = {
            'team_out_ppg': -1.0,
            'team_out_count': -1,
            'opp_out_ppg': -1.0,
            'opp_out_count': -1
        }
        
        try:
            # Fetch injury reports for both teams
            team_injuries = self.get_team_injury_report(player_team_abbrev)
            opp_injuries = self.get_team_injury_report(opponent_abbrev)
            
            # Filter to only OUT and DOUBTFUL (high confidence of absence)
            out_statuses = {'OUT', 'DOUBTFUL'}
            
            # Process teammate injuries (exclude the player being analyzed)
            team_out_ppg = 0.0
            team_out_count = 0
            normalized_player_name = normalize_name(player_name)
            
            for injury in team_injuries:
                if injury.get('status') in out_statuses:
                    injured_name = injury.get('name', '')
                    # Skip the player being analyzed
                    if normalize_name(injured_name) == normalized_player_name:
                        continue
                    
                    # Look up PPG for this player
                    ppg = self._get_player_season_ppg(injured_name, data_loader)
                    if ppg >= 5.0:  # Only count meaningful contributors (5+ PPG)
                        team_out_ppg += ppg
                        team_out_count += 1
            
            # Process opponent injuries
            opp_out_ppg = 0.0
            opp_out_count = 0
            
            for injury in opp_injuries:
                if injury.get('status') in out_statuses:
                    injured_name = injury.get('name', '')
                    ppg = self._get_player_season_ppg(injured_name, data_loader)
                    if ppg >= 5.0:  # Only count meaningful contributors (5+ PPG)
                        opp_out_ppg += ppg
                        opp_out_count += 1
            
            result = {
                'team_out_ppg': round(team_out_ppg, 1),
                'team_out_count': team_out_count,
                'opp_out_ppg': round(opp_out_ppg, 1),
                'opp_out_count': opp_out_count
            }
            
            logger.info(f"Absence features from injuries: team={team_out_count} out ({team_out_ppg:.1f} PPG), "
                       f"opp={opp_out_count} out ({opp_out_ppg:.1f} PPG)")
            
        except Exception as e:
            logger.warning(f"Failed to compute absence features from injuries: {e}")
        
        return result
    
    def _get_player_season_ppg(self, player_name: str, data_loader: 'DataLoader' = None) -> float:
        """
        Get a player's season PPG by looking up their game logs.
        
        Returns 0.0 if player not found or no games.
        Uses cache to avoid repeated API calls for the same player.
        """
        cache_key = f"ppg_{normalize_name(player_name)}"
        
        # Check cache first
        if cache_key in self._injury_cache:
            cached = self._injury_cache[cache_key]
            if (datetime.now() - cached['time']).seconds < 3600:  # 1 hour cache
                return cached['data']
        
        ppg = 0.0
        
        try:
            # Find player ID
            all_players = players.get_players()
            normalized = normalize_name(player_name)
            p_obj = next((p for p in all_players if normalize_name(p['full_name']) == normalized), None)
            
            if not p_obj:
                # Try partial match
                p_obj = next((p for p in all_players if normalized in normalize_name(p['full_name'])), None)
            
            if p_obj and data_loader:
                # Fetch game logs
                df = data_loader.fetch_game_logs(p_obj['id'])
                if len(df) >= 5:  # Need at least 5 games for meaningful PPG
                    ppg = df['PTS'].mean()
            
        except Exception as e:
            logger.debug(f"Could not get PPG for {player_name}: {e}")
        
        self._injury_cache[cache_key] = {'time': datetime.now(), 'data': ppg}
        return ppg


class FeatureEngineer: 
    """
    V20 EMPIRICAL: Pure observable feature engineering.
    
    DESIGN PRINCIPLES (per refactor directive):
    - Raw observables only (no computed multipliers)
    - Timestamp-valid (frozen before game tip-off)
    - Non-decisional (no belief-encoded adjustments)
    - Let ML learn all relationships from data
    
    REMOVED:
    - BlowoutPredictor (belief-based logistic regression)
    - PlayerFatigueProfiler (belief-based rest effects)
    - calculate_rest_factor (heuristic multiplier)
    - calculate_game_total_factor (heuristic multiplier)
    - calculate_blowout_factor (heuristic multiplier)
    - calculate_opponent_form_factor (heuristic multiplier)
    - calculate_dynamic_std_multiplier (heuristic multiplier)
    - calculate_matchup_multipliers (heuristic multiplier)
    - Injury usage boost calculations (belief-based)
    
    KEPT:
    - calculate_statistical_features (raw statistics from game logs)
    - build_feature_vector (now outputs raw observables only)
    """
    
    def __init__(self, config: Config = CONFIG):
        self.config = config
        # V20: Only InjuryManager kept for roster status (raw observable: is player out?)
        self.injury_manager = InjuryManager()
    
    def calculate_statistical_features(
        self, 
        df: pd.DataFrame, 
        stat_col: str, 
        lookback: int = 15,
        data_quality: DataQuality = None
    ) -> Dict[str, float]: 
        """
        Calculate core statistical features from game logs.
        
        V20.2: Returns raw observable statistics including:
        - ema, std, avg_minutes, games_played (original)
        - trend_5g: Linear slope of last 5 games (NEW)
        - home_avg, away_avg: Home/away splits (NEW)
        """
        if data_quality is None:
            data_quality = DataQuality()
            
        if len(df) == 0 or stat_col not in df.columns:
            raise ValueError(f"Cannot calculate features: missing data for {stat_col}")
        
        recent = df.tail(lookback).copy()
        if len(recent) == 0:
            raise ValueError("No recent games available for analysis")
        
        # Track low sample size
        if len(df) < 10:
            data_quality.low_sample_size = True
            data_quality.add_warning(f"Low sample size: only {len(df)} games available")
        
        # EMA (Exponential Moving Average) - raw statistic
        ema = recent[stat_col].ewm(span=len(recent), adjust=False).mean().iloc[-1]
        
        # Standard Deviation - raw statistic
        std_dev = recent[stat_col].std()
        if pd.isna(std_dev) or std_dev == 0:
            mean_val = recent[stat_col].mean()
            std_dev = mean_val * 0.2 if not pd.isna(mean_val) and mean_val > 0 else 1.0
            data_quality.used_fallback_std = True
            data_quality.add_warning("Using estimated std (20% of mean) due to insufficient variance")
        
        # Average minutes - raw statistic
        if 'MIN_FLOAT' in recent.columns and len(recent) > 0:
            avg_minutes = recent['MIN_FLOAT'].mean()
        else:
            avg_minutes = 30.0
            data_quality.used_fallback_minutes = True
            data_quality.add_warning("Using default 30 minutes (minutes data unavailable)")
        
        # === V20.2 NEW: TREND (slope of last 5 games) ===
        # Raw observable: positive slope = improving, negative = declining
        trend_5g = 0.0
        last_5 = df.tail(5)
        if len(last_5) >= 3:  # Need at least 3 games for meaningful slope
            try:
                x = np.arange(len(last_5))
                y = last_5[stat_col].values
                # Linear regression slope
                if len(x) > 1 and not np.all(y == y[0]):  # Avoid division by zero
                    slope, _ = np.polyfit(x, y, 1)
                    trend_5g = float(slope)
            except Exception:
                trend_5g = 0.0
        
        # === V20.2 NEW: HOME/AWAY SPLITS ===
        # Raw observables: player's average in home vs away games
        home_avg = ema  # Default to EMA if no split data
        away_avg = ema
        
        if 'MATCHUP' in df.columns and len(df) >= 5:
            # Home games have '@' NOT in matchup (e.g., "LAL vs BOS")
            # Away games have '@' in matchup (e.g., "LAL @ BOS")
            home_games = df[~df['MATCHUP'].str.contains('@', na=False)]
            away_games = df[df['MATCHUP'].str.contains('@', na=False)]
            
            if len(home_games) >= 2:
                home_avg = home_games[stat_col].mean()
            if len(away_games) >= 2:
                away_avg = away_games[stat_col].mean()
        
        # === V20.3 NEW: Behavior & Risk features ===
        # Rolling std dev of minutes (min_volatility) and foul rate (PF mean)
        min_volatility = 0.0
        if 'MIN_FLOAT' in recent.columns and len(recent) > 0:
            try:
                mv = float(recent['MIN_FLOAT'].std())
                if not pd.isna(mv):
                    min_volatility = mv
            except Exception:
                min_volatility = 0.0

        foul_rate = 0.0
        if 'PF' in recent.columns and len(recent) > 0:
            try:
                fr = float(recent['PF'].mean())
                if not pd.isna(fr):
                    foul_rate = fr
            except Exception:
                foul_rate = 0.0

        # Coefficient of variation (std / ema) - treat very small ema as unstable
        cv = 0.0
        try:
            if (not pd.isna(ema)) and ema >= 0.5 and not pd.isna(std_dev) and std_dev >= 0.0:
                cv = float(std_dev / ema) if ema != 0 else 0.0
        except Exception:
            cv = 0.0

        return {
            'ema': ema,
            'std': std_dev,
            'avg_minutes': avg_minutes,
            'games_played': len(df),
            # V20.2 NEW
            'trend_5g': trend_5g,
            'home_avg': home_avg,
            'away_avg': away_avg,
            # V20.3 NEW: Behavior & Risk
            'min_volatility': min_volatility,
            'foul_rate': foul_rate,
            'cv': cv,
        }
    
    def _calculate_ts_pct(self, df: pd.DataFrame, lookback: int = 15) -> Tuple[float, float]:
        """
        Calculate rolling TS% and season TS% using rolling sums (no per-game averages).
        Both values are returned as fractions (0-1). If insufficient data or
        denominator == 0, returns -1.0 sentinel for unavailable.
        Uses only raw game logs present in `df` (no external data).
        """
        # Require PTS, FGA, FTA present
        for col in ['PTS', 'FGA', 'FTA']:
            if col not in df.columns:
                return -1.0, -1.0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        if len(df) == 0:
            return -1.0, -1.0
        
        recent = df.tail(lookback)
        sum_pts = recent['PTS'].sum()
        sum_fga = recent['FGA'].sum()
        sum_fta = recent['FTA'].sum()
        denom = 2.0 * (sum_fga + 0.44 * sum_fta)
        rolling_ts = -1.0 if denom == 0 or pd.isna(denom) else float(sum_pts / denom)
        
        sum_pts_s = df['PTS'].sum()
        sum_fga_s = df['FGA'].sum()
        sum_fta_s = df['FTA'].sum()
        denom_s = 2.0 * (sum_fga_s + 0.44 * sum_fta_s)
        season_ts = -1.0 if denom_s == 0 or pd.isna(denom_s) else float(sum_pts_s / denom_s)
        
        return rolling_ts, season_ts
    
    def build_feature_vector(
        self,
        player_id: int,
        player_name: str,
        opponent_id: int,
        opponent_abbrev: str,
        is_home: bool,
        is_b2b: bool,
        spread: float,
        df: pd.DataFrame,
        stat_col: str,
        line: float,
        lookback: int,
        team_stats: pd.DataFrame,
        avg_def: float,
        market: str,
        days_rest: int = 1,
        game_total: float = 0.0,
        player_team_id: int = None,  # V20.2: For team pace lookup
        player_team_abbrev: str = None,  # V20.3: For injury-based absence features
        data_loader: 'DataLoader' = None  # V20.3: For PPG lookups
    ) -> FeatureVector: 
        """
        Build pure empirical feature vector for ML models.
        
        V20.3 EMPIRICAL: 24 raw observables known before tip-off.
        
        NO MULTIPLIERS. NO HEURISTICS. Let ML learn all relationships.
        
        V20.2 Features:
        - opponent_pace: Opponent's pace factor
        - team_pace: Player's team pace factor
        - trend_5g: Slope of last 5 games
        - home_avg: Player's home game average
        - away_avg: Player's away game average
        
        V20.3 NEW (Absence-aware from injury reports):
        - team_out_ppg: PPG of teammates listed as OUT/DOUBTFUL
        - team_out_count: Count of teammates out
        - opp_out_ppg: PPG of opponents listed as OUT/DOUBTFUL
        - opp_out_count: Count of opponents out
        
        Note: During training, absence features come from game logs (ground truth).
        For live predictions, injury reports are used as proxy (-1 if unavailable).
        """
        
        # Initialize data quality tracker
        data_quality = DataQuality()
        
        # === CORE STATISTICAL FEATURES (raw from game logs) ===
        stats = self.calculate_statistical_features(df, stat_col, lookback, data_quality)
        
        # === OPPONENT DEFENSIVE RATING (raw season-to-date statistic) ===
        opponent_drtg_season = avg_def  # Default to league average
        opponent_pace = 100.0  # Default pace
        team_pace = 100.0  # Default pace
        
        if len(team_stats) > 0:
            # Opponent stats
            if opponent_id in team_stats.index:
                opp = team_stats.loc[opponent_id]
                opponent_drtg_season = opp.get('DEF_RATING', avg_def)
                opp_pace_val = opp.get('PACE', None)
                if opp_pace_val is not None:
                    opponent_pace = opp_pace_val
                else:
                    data_quality.used_default_pace = True
                    data_quality.add_warning("Using default opponent pace (PACE column missing)")
            else:
                data_quality.used_default_def_rating = True
                data_quality.used_default_pace = True
                data_quality.add_warning(f"Opponent {opponent_id} not in team_stats - using defaults")
            
            # Player's team stats (V20.2 NEW)
            if player_team_id and player_team_id in team_stats.index:
                player_team = team_stats.loc[player_team_id]
                team_pace_val = player_team.get('PACE', None)
                if team_pace_val is not None:
                    team_pace = team_pace_val
                else:
                    data_quality.used_default_pace = True
                    data_quality.add_warning("Using default team pace (PACE column missing)")
            elif player_team_id:
                data_quality.used_default_pace = True
                data_quality.add_warning(f"Player team {player_team_id} not in team_stats - using default pace")
        
        if len(team_stats) == 0:
            data_quality.missing_team_stats = True
            data_quality.used_default_def_rating = True
            data_quality.used_default_pace = True
            data_quality.add_warning("Team stats unavailable - using defaults for DRTG and pace")
        
        # === DAYS REST (raw integer, not a computed factor) ===
        actual_days_rest = days_rest if days_rest >= 0 else (0 if is_b2b else 1)
        
        # === V20.3 NEW: ABSENCE FEATURES FROM INJURY REPORT ===
        # For live predictions, use injury report as proxy for who will be out
        team_out_ppg = -1.0  # Sentinel: unknown
        team_out_count = -1
        opp_out_ppg = -1.0
        opp_out_count = -1
        
        if player_team_abbrev and opponent_abbrev:
            absence_features = self.injury_manager.compute_absence_features_from_injuries(
                player_team_abbrev=player_team_abbrev,
                opponent_abbrev=opponent_abbrev,
                player_name=player_name,
                data_loader=data_loader
            )
            team_out_ppg = absence_features['team_out_ppg']
            team_out_count = absence_features['team_out_count']
            opp_out_ppg = absence_features['opp_out_ppg']
            opp_out_count = absence_features['opp_out_count']
        
        # === V20.3 NEW: TRUE-SHOOTING% (TS%) FEATURES ===
        # Safe admissible metric derived from raw game logs using rolling sums
        feat_ts_pct = -1.0
        feat_ts_pct_delta = -1.0
        try:
            rolling_ts, season_ts = self._calculate_ts_pct(df, lookback)
            if rolling_ts != -1.0:
                feat_ts_pct = rolling_ts
            if season_ts != -1.0 and feat_ts_pct != -1.0:
                feat_ts_pct_delta = feat_ts_pct - season_ts
            elif season_ts != -1.0:
                feat_ts_pct_delta = 0.0
        except Exception:
            # Non-fatal: keep sentinel values
            feat_ts_pct = feat_ts_pct
            feat_ts_pct_delta = feat_ts_pct_delta
        
        # === BUILD V20.3 FEATURE VECTOR ===
        return FeatureVector(
            # Identifiers (not used in ML)
            player_id=player_id,
            player_name=player_name,
            opponent_abbrev=opponent_abbrev,
            market=market,
            # Raw observables only - NO MULTIPLIERS
            line=line,
            avg_minutes=stats['avg_minutes'],
            ema=stats['ema'],
            std=stats['std'],
            opponent_drtg_season=opponent_drtg_season,
            spread=spread,
            game_total=game_total if game_total > 0 else 225.0,
            days_rest=actual_days_rest,
            is_home=is_home,
            is_b2b=is_b2b or actual_days_rest == 0,
            games_played=stats['games_played'],
            # V20.2: Pace context
            opponent_pace=opponent_pace,
            team_pace=team_pace,
            # V20.2: Momentum & splits
            trend_5g=stats['trend_5g'],
            home_avg=stats['home_avg'],
            away_avg=stats['away_avg'],
            # V20.3 NEW: True-Shooting features
            feat_ts_pct=feat_ts_pct,
            feat_ts_pct_delta=feat_ts_pct_delta,
            # V20.3 NEW: Absence-aware features (from injury report)
            team_out_ppg=team_out_ppg,
            team_out_count=team_out_count,
            opp_out_ppg=opp_out_ppg,
            opp_out_count=opp_out_count,
            # V20.3 NEW: Behavior & Risk features (stat-derived)
            feat_min_volatility=stats.get('min_volatility', 0.0),
            feat_foul_rate=stats.get('foul_rate', 0.0),
            feat_cv=stats.get('cv', 0.0),
            # Market identity (one-hot encoded)
            market_scoring=1 if market == 'PTS' else 0,
            market_counting=1 if market in ['REB', 'AST'] else 0,
            market_combo=1 if market in ['PRA', 'PR', 'PA', 'RA'] else 0,
            market_rare=1 if market in ['3PM', 'STL', 'BLK'] else 0,
            # Data quality
            data_quality=data_quality
        )

@dataclass
class ProjectionResult:
    """
    Holds the output of the projection model.
    
    V19 Fields:
    - base_projection: Raw model output
    - final_projection: Adjusted prediction
    - confidence_interval: Tuple of (lower, upper) bounds
    - adjustments: Dict with 'predicted_std', 'model_type', etc.
    - ml_prob: CDF-based probability (0-1)
    """
    base_projection: float
    final_projection: float
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    adjustments: Dict[str, Any] = field(default_factory=dict)
    ml_prob: Optional[float] = None
    
# =============================================================================
# LAYER 3: MODEL ENGINE (SMART ENSEMBLE)
# =============================================================================

# Market group mapping for model selection
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


class ModelEngine:
    """
    V19 ML COMMITMENT ENGINE: ML is the SOLE prediction source.
    
    PHILOSOPHY: No heuristic fallbacks. If ML model is unavailable, prediction FAILS.
    This forces proper model training and prevents silent degradation.
    
    Key Architecture:
    - XGBRegressor predicts expected stat value E[stat]
    - Variance model predicts σ[stat] (uncertainty)
    - Probability via CDF: P(Over) = 1 - Φ((line - μ) / σ)
    - Calibration curves correct probability biases
    - Ablation mode logs feature importance
    
    Model Files (in data/ directory):
    - nba_model_{group}.pkl: Mean regression model
    - nba_variance_{group}.pkl: Variance model  
    - nba_calibrator_{group}.pkl: Isotonic calibration curve
    """
    def __init__(self, config: Config = CONFIG, tracker: 'Tracker' = None):
        self.config = config
        self.tracker = tracker
        # Market-specific models: {group_name: (mean_model, variance_model, feature_names)}
        self.ml_models: Dict[str, Tuple[Any, Any, List[str]]] = {}
        # Calibration curves: {group_name: IsotonicRegression}
        self.calibrators: Dict[str, Any] = {}
        self.ml_model = None  # Legacy fallback
        self.variance_model = None  # Variance prediction model
        self.model_features: List[str] = []
        self._load_ml_models()
        self._load_calibrators()

    def _load_ml_models(self):
        """
        Load market-specific regression models.
        V19 COMMITMENT: Models are REQUIRED, not optional.
        """
        market_groups = ['scoring', 'counting', 'combo', 'rare', 'universal']
        loaded_count = 0
        
        for group in market_groups:
            model_path = DATA_DIR / f"nba_model_{group}.pkl"
            var_model_path = DATA_DIR / f"nba_variance_{group}.pkl"
            
            if model_path.exists():
                try:
                    model = joblib.load(model_path)
                    var_model = None
                    if var_model_path.exists():
                        var_model = joblib.load(var_model_path)
                    
                    # Extract feature names from model
                    feature_names = []
                    if hasattr(model, 'feature_names_in_'):
                        feature_names = [str(f) for f in model.feature_names_in_]
                    elif hasattr(model, 'get_booster'):
                        booster_names = model.get_booster().feature_names
                        feature_names = [str(f) for f in booster_names] if booster_names else []
                    
                    self.ml_models[group] = (model, var_model, feature_names)
                    loaded_count += 1
                    
                    if not self.model_features and feature_names:
                        self.model_features = feature_names
                        
                except Exception as e:
                    logger.warning(f"Failed to load {group} model: {e}")
        
        if loaded_count > 0:
            logger.info(f"🤖 ML COMMITMENT: Loaded {loaded_count} regression models")
            if self.model_features:
                logger.info(f"   Model expects {len(self.model_features)} features")
        else:
            # V19 COMMITMENT: Warn loudly if no models found
            logger.error("⚠️ NO ML MODELS FOUND - Run data/train_model_v19.py first!")
            if self.config.ML_REQUIRE_MODEL:
                logger.error("   Predictions will FAIL until models are trained.")
        
        # Legacy fallback (only if no group models found)
        if not self.ml_models:
            model_path = DATA_DIR / "nba_model.pkl"
            var_path = DATA_DIR / "nba_variance.pkl"
            if model_path.exists():
                try:
                    self.ml_model = joblib.load(model_path)
                    if var_path.exists():
                        self.variance_model = joblib.load(var_path)
                    if hasattr(self.ml_model, 'feature_names_in_'):
                        self.model_features = list(self.ml_model.feature_names_in_)
                    logger.info("🤖 Loaded legacy universal ML model")
                except Exception as e:
                    logger.error(f"Failed to load ML model: {e}")
    
    def _load_calibrators(self):
        """
        Load isotonic regression calibrators for probability correction.
        
        Calibration curves are fitted on validation data to correct systematic
        biases in raw model probabilities. They improve Brier score and reliability.
        """
        market_groups = ['scoring', 'counting', 'combo', 'rare', 'universal']
        loaded_count = 0
        
        for group in market_groups:
            calibrator_path = DATA_DIR / f"nba_calibrator_{group}.pkl"
            
            if calibrator_path.exists():
                try:
                    calibrator = joblib.load(calibrator_path)
                    self.calibrators[group] = calibrator
                    loaded_count += 1
                except Exception as e:
                    logger.warning(f"Failed to load {group} calibrator: {e}")
        
        if loaded_count > 0:
            logger.info(f"📊 Loaded {loaded_count} calibration curves")
        else:
            logger.info("📊 No calibration curves found (using raw probabilities)")

    def _get_model_for_market(self, market: str) -> Tuple[Any, Any, List[str]]:
        """
        Get the appropriate mean model, variance model, and feature names for a market.
        Returns (mean_model, variance_model, feature_names) tuple.
        """
        group = ML_MARKET_GROUPS.get(market, 'universal')
        
        if group in self.ml_models:
            return self.ml_models[group]
        
        if 'universal' in self.ml_models:
            return self.ml_models['universal']
        
        return (self.ml_model, self.variance_model, self.model_features)

    def _extract_features_for_model(self, features: FeatureVector, expected_features: List[str]) -> np.ndarray:
        """
        Extract feature values in the exact order the model expects.
        Maps FeatureVector attributes to model feature names.
        
        V20.3 EMPIRICAL: 24 raw observables.
        """
        # Create mapping from feature name to FeatureVector attribute
        # V20.3: Pure raw observables - no computed multipliers
        feature_map = {
            # Original features
            'feat_avg_minutes': features.avg_minutes,
            'feat_ema': features.ema,
            'feat_std': features.std,
            'feat_opp_drtg_season': features.opponent_drtg_season,
            'feat_opponent_drtg_season': features.opponent_drtg_season,  # Alias
            'feat_line': features.line,
            'feat_spread': features.spread,
            'feat_game_total': features.game_total,
            'feat_days_rest': float(features.days_rest),
            'feat_is_home': float(features.is_home),
            'feat_is_b2b': float(features.is_b2b),
            'feat_games_played': float(features.games_played),
            # V20.2: Pace context
            'feat_opponent_pace': features.opponent_pace,
            'feat_team_pace': features.team_pace,
            # V20.2: Momentum & splits
            'feat_trend_5g': features.trend_5g,
            'feat_home_avg': features.home_avg,
            'feat_away_avg': features.away_avg,
            # V20.3 NEW: True Shooting (2)
            'feat_ts_pct': features.feat_ts_pct,
            'feat_ts_pct_delta': features.feat_ts_pct_delta,
            # V20.3 NEW: Absence-aware features
            'feat_team_out_ppg': features.team_out_ppg,
            'feat_team_out_count': float(features.team_out_count),
            'feat_opp_out_ppg': features.opp_out_ppg,
            'feat_opp_out_count': float(features.opp_out_count),
            # Market identity
            'feat_market_scoring': float(features.market_scoring),
            'feat_market_counting': float(features.market_counting),
            'feat_market_combo': float(features.market_combo),
            'feat_market_rare': float(features.market_rare),
            # V20.3 NEW: Behavior & Risk features
            'feat_min_volatility': float(features.feat_min_volatility),
            'feat_foul_rate': float(features.feat_foul_rate),
            'feat_cv': float(features.feat_cv),
        }
        
        # Extract values in the exact order expected by the model
        values = []
        for feat_name in expected_features:
            if feat_name in feature_map:
                values.append(feature_map[feat_name])
            else:
                # Unknown feature - use 0 as default
                logger.warning(f"Unknown feature requested by model: {feat_name}")
                values.append(0.0)
        
        return np.array(values)

    def get_ml_prediction(self, features: FeatureVector, raw_only: bool = False) -> Optional[float]:
        """
        V19 REGRESSION: Get predicted stat value and convert to P(Over) via CDF.
        
        The model predicts E[stat], then we calculate:
        P(Over) = 1 - Φ((line - predicted_mean) / predicted_std)
        
        Returns: P(Over) probability (for compatibility with existing code)
        """
        market = getattr(features, 'market', None) or 'PTS'
        mean_model, var_model, expected_features = self._get_model_for_market(market)
        
        if mean_model is None:
            return None
        
        try:
            # Extract features
            if expected_features:
                X = self._extract_features_for_model(features, expected_features).reshape(1, -1)
            else:
                X = features.to_ml_array().reshape(1, -1)
            
            # Check if this is a regressor or classifier (for backwards compatibility)
            if hasattr(mean_model, 'predict_proba'):
                # Old classifier model - use legacy method
                raw_prob = mean_model.predict_proba(X)[0][1]
                return float(max(0.01, min(0.99, raw_prob)))
            
            # V19 REGRESSION: Predict expected stat value
            predicted_mean = float(mean_model.predict(X)[0])
            
            # Get variance estimate
            if var_model is not None:
                # Use variance model if available
                predicted_std = float(np.sqrt(max(0.1, var_model.predict(X)[0])))
            else:
                # V20: Use player's empirical std only - no fabricated minimums
                predicted_std = features.std if features.std > 0 else 1.0
            
            # Calculate P(Over) using normal CDF
            # P(X > line) = 1 - Φ((line - μ) / σ)
            line = features.line
            if predicted_std > 0:
                z_score = (line - predicted_mean) / predicted_std
                p_over = 1.0 - stats.norm.cdf(z_score)
            else:
                # Edge case: no variance
                p_over = 1.0 if predicted_mean > line else 0.0
            
            # Clamp to valid probability range
            p_over = float(max(0.01, min(0.99, p_over)))
            
            return p_over
            
        except Exception as e:
            logger.warning(f"ML Prediction failed: {e}")
            return None
    
    def get_ml_regression_output(self, features: FeatureVector) -> Dict[str, Any]:
        """
        V19 NEW: Get full regression output including predicted value and uncertainty.
        
        Returns dict with:
        - predicted_value: E[stat] from model
        - predicted_std: σ[stat] (uncertainty)
        - p_over: P(stat > line)
        - p_under: P(stat <= line)
        - confidence_interval: (5th percentile, 95th percentile)
        """
        market = getattr(features, 'market', None) or 'PTS'
        mean_model, var_model, expected_features = self._get_model_for_market(market)
        
        result = {
            'predicted_value': None,
            'predicted_std': None,
            'p_over': None,
            'p_under': None,
            'confidence_interval': (None, None),
            'has_model': mean_model is not None,
            'is_regression': False,
        }
        
        if mean_model is None:
            return result
        
        try:
            if expected_features:
                X = self._extract_features_for_model(features, expected_features).reshape(1, -1)
            else:
                X = features.to_ml_array().reshape(1, -1)
            
            # Check model type
            if hasattr(mean_model, 'predict_proba'):
                # Old classifier - return probability only
                raw_prob = float(mean_model.predict_proba(X)[0][1])
                result['p_over'] = max(0.01, min(0.99, raw_prob))
                result['p_under'] = 1.0 - result['p_over']
                result['is_regression'] = False
                # Estimate predicted value from EMA and probability direction
                result['predicted_value'] = features.ema
                result['predicted_std'] = features.std
                return result
            
            # V19 REGRESSION
            result['is_regression'] = True
            predicted_mean = float(mean_model.predict(X)[0])
            result['predicted_value'] = predicted_mean
            
            # Get variance
            if var_model is not None:
                predicted_var = float(max(0.1, var_model.predict(X)[0]))
                predicted_std = np.sqrt(predicted_var)
            else:
                # V20: Use player's empirical std only - no fabricated minimums
                predicted_std = features.std if features.std > 0 else 1.0
            result['predicted_std'] = predicted_std
            
            # Calculate probabilities
            line = features.line
            if predicted_std > 0:
                z_score = (line - predicted_mean) / predicted_std
                p_over = 1.0 - stats.norm.cdf(z_score)
            else:
                p_over = 1.0 if predicted_mean > line else 0.0
            
            result['p_over'] = float(max(0.01, min(0.99, p_over)))
            result['p_under'] = 1.0 - result['p_over']
            
            # 90% confidence interval
            ci_low = predicted_mean - 1.645 * predicted_std
            ci_high = predicted_mean + 1.645 * predicted_std
            result['confidence_interval'] = (max(0, ci_low), ci_high)
            
        except Exception as e:
            logger.warning(f"ML Regression failed: {e}")
        
        return result
    
    def get_ml_prediction_details(self, features: FeatureVector) -> Dict[str, Any]:
        """
        V19 UPDATED: Get detailed ML prediction info for UI display.
        Now returns regression outputs instead of classification.
        """
        market = getattr(features, 'market', None) or 'PTS'
        group = ML_MARKET_GROUPS.get(market, 'universal')
        mean_model, var_model, expected_features = self._get_model_for_market(market)
        
        # Check if calibrator exists for this market group
        has_calibrator = group in self.calibrators if hasattr(self, 'calibrators') else False
        
        # Get regression output
        regression = self.get_ml_regression_output(features)
        
        details = {
            'market': market,
            'model_group': group,
            'has_model': mean_model is not None,
            'has_variance_model': var_model is not None,
            'has_calibrator': has_calibrator,
            'is_regression': regression.get('is_regression', False),
            'num_features': len(expected_features) if expected_features else 0,
            # V19 Regression outputs
            'predicted_value': regression.get('predicted_value'),
            'predicted_std': regression.get('predicted_std'),
            'p_over': regression.get('p_over'),
            'p_under': regression.get('p_under'),
            'confidence_interval': regression.get('confidence_interval'),
            # Legacy compatibility
            'raw_prob': regression.get('p_over'),
            'calibrated_prob': regression.get('p_over'),
            'calibration_delta': 0.0,
        }
        
        return details

    def generate_projection(self, features: FeatureVector) -> ProjectionResult:
        """
        V20 EMPIRICAL: Generate projection using ML ONLY.
        
        ENFORCEMENT: ML model is REQUIRED. No fallback, no exceptions.
        If model is unavailable, raises MLModelRequiredError.
        
        The ML model predicts E[stat] directly, which is the projection.
        Probability calculations use CDF: P(Over) = 1 - Φ((line - μ) / σ)
        
        Calibration curves are applied when available to improve probability accuracy.
        """
        market = getattr(features, 'market', None)
        
        # 1. GET ML REGRESSION OUTPUT (MANDATORY)
        regression = self.get_ml_regression_output(features)
        
        # V20 ENFORCEMENT: ML model is REQUIRED - no fallback
        if not regression['has_model']:
            raise MLModelRequiredError(
                f"No ML model available for market '{market}'. "
                f"Run train_model_v20.py to train models."
            )
        
        if not regression['is_regression']:
            # Old classifier detected - treat as error
            raise MLModelRequiredError(
                f"Classifier model detected for '{market}'. "
                f"V20 requires regression models. Retrain with train_model_v20.py."
            )
        
        # 2. EXTRACT ML OUTPUTS
        ml_proj = regression['predicted_value']
        ml_std = regression['predicted_std']
        ml_prob = regression['p_over']
        
        # 3. APPLY CALIBRATION (if enabled and available)
        if self.config.ML_CALIBRATION_ENABLED and hasattr(self, 'calibrators'):
            calibrated_prob = self._apply_calibration(ml_prob, market)
            if calibrated_prob != ml_prob:
                logger.debug(f"Calibration: {ml_prob:.3f} -> {calibrated_prob:.3f}")
                ml_prob = calibrated_prob
        
        # 4. V20 EMPIRICAL: NO CAPS, NO ADJUSTMENTS
        # Let ML learn appropriate ranges from training data
        adjusted_proj = ml_proj
        
        # Get confidence interval
        ci_low, ci_high = regression['confidence_interval']
        ci = (ci_low if ci_low is not None else adjusted_proj - ml_std * 1.645,
              ci_high if ci_high is not None else adjusted_proj + ml_std * 1.645)
        
        return ProjectionResult(
            base_projection=float(ml_proj),
            final_projection=float(adjusted_proj),
            confidence_interval=ci,
            adjustments={
                'predicted_std': ml_std,
                'model_type': 'regression',
                'calibration_applied': self.config.ML_CALIBRATION_ENABLED,
            },
            ml_prob=ml_prob
        )
    
    def _apply_calibration(self, raw_prob: float, market: str) -> float:
        """
        Apply isotonic regression calibration curve to raw probability.
        
        Calibration improves probability accuracy by correcting systematic biases
        learned during training. Uses isotonic regression fitted on validation data.
        """
        group = ML_MARKET_GROUPS.get(market, 'universal')
        calibrator = self.calibrators.get(group) if hasattr(self, 'calibrators') else None
        
        if calibrator is None:
            return raw_prob
        
        try:
            # Isotonic regression expects 2D input
            calibrated = calibrator.predict([[raw_prob]])[0]
            return float(max(0.01, min(0.99, calibrated)))
        except Exception as e:
            logger.warning(f"Calibration failed: {e}")
            return raw_prob
    
    def _log_ablation_metrics(self, features: FeatureVector, regression: Dict, market: str) -> None:
        """
        Log feature ablation metrics for model interpretability.
        
        Used during ablation studies to understand feature importance.
        """
        try:
            ablation_log = {
                'market': market,
                'predicted_value': regression['predicted_value'],
                'predicted_std': regression['predicted_std'],
                'features': {
                    'ema': features.ema,
                    'std': features.std,
                    'line': features.line,
                    'days_rest': features.days_rest,
                    'is_b2b': features.is_b2b,
                    'opp_drtg': features.opponent_drtg_season,
                    'is_home': features.is_home,
                    'spread': features.spread,
                    'game_total': features.game_total,
                    'games_played': features.games_played,
                },
                'timestamp': pd.Timestamp.now().isoformat()
            }
            logger.info(f"ABLATION: {ablation_log}")
        except Exception as e:
            logger.debug(f"Ablation logging failed: {e}")

# =============================================================================
# LAYER 4: SIMULATION ENGINE (Feature-Aware)
# =============================================================================

class SimulationEngine:
    """
    V20 EMPIRICAL: Pure CDF-based probability calculation.
    
    DESIGN PRINCIPLES (per refactor directive):
    - Pure CDF calculation: P(Over) = 1 - Φ((line - μ) / σ)
    - NO mixture models (belief-based)
    - NO variance adjustments based on spread/rest (heuristics)
    - NO probability caps (let ML calibration handle this)
    - Uncertainty comes from variance model, not hardcoded adjustments
    
    REMOVED:
    - _generate_smart_mixture_samples (belief-based)
    - _generate_low_count_samples (heuristic caps and adjustments)
    - _calculate_dynamic_mixture_probs (belief-based)
    - _apply_feature_adjustments_to_mean (heuristic)
    - All probability caps (MAX_PROBABILITY_CAP, MIN_PROBABILITY_FLOOR)
    - All variance multipliers based on spread/CV
    """
    
    def __init__(self, config: Config = CONFIG):
        self.config = config
    
    def compute_cdf_probability(
        self,
        mean: float,
        std: float,
        line: float,
        market: str = None,
        features: 'FeatureVector' = None
    ) -> SimulationResult:
        """
        V20 EMPIRICAL: Pure CDF probability calculation.
        
        Formula: P(Over) = 1 - Φ((line - μ) / σ)
        
        NO ADJUSTMENTS. The ML model's mean and std predictions are used as-is.
        If the model is uncertain, that's reflected in a larger std.
        """
        # Handle edge cases
        if std <= 0:
            std = 1.0  # V20: Minimal fallback - do not fabricate variance
        
        # V20: NO variance adjustments - use model's predicted std directly
        adjusted_std = std
        
        # === CORE CDF CALCULATION ===
        z_score = (line - mean) / adjusted_std
        p_under = stats.norm.cdf(z_score)
        p_over = 1.0 - p_under
        
        # V20: NO probability caps - let calibration handle extreme cases
        # The model should learn when to output extreme probabilities
        
        # Calculate confidence intervals analytically
        ci_10 = mean - 1.28 * adjusted_std  # 10th percentile
        ci_90 = mean + 1.28 * adjusted_std  # 90th percentile
        
        # Minimal placeholder for compatibility
        sims = np.array([mean])
        
        return SimulationResult(
            over_prob=float(max(0.001, min(0.999, p_over))),  # Technical bounds only
            under_prob=float(max(0.001, min(0.999, p_under))),
            median=mean,
            ci_10=max(0, ci_10),
            ci_90=ci_90,
            simulations=sims
        )
    
    def run_simulation(
        self, 
        mean: float, 
        std: float, 
        line: float, 
        market: str,
        features: 'FeatureVector' = None,
        simulations: int = 1000,
        use_mixture: bool = False  # V20: mixture disabled by default
    ) -> SimulationResult: 
        """
        V20 EMPIRICAL: Simple normal-distribution Monte Carlo.
        
        For most use cases, prefer compute_cdf_probability() which is deterministic.
        This method is kept for generating sample distributions for histograms.
        
        NO mixture models, NO heuristic adjustments.
        """
        # Handle edge cases
        if std <= 0:
            std = 1.0  # V20: Minimal fallback
        
        # V20: Pure normal distribution - no mixture, no adjustments
        sims = np.random.normal(mean, std, simulations)
        
        # Ensure non-negative (physical constraint, not heuristic)
        sims = np.maximum(sims, 0)
        
        over_rate = float((sims > line).mean())
        under_rate = float((sims <= line).mean())
        
        return SimulationResult(
            over_prob=max(0.001, min(0.999, over_rate)),  # Technical bounds
            under_prob=max(0.001, min(0.999, under_rate)),
            median=float(np.median(sims)),
            ci_10=float(np.percentile(sims, 10)),
            ci_90=float(np.percentile(sims, 90)),
            simulations=sims
        )
    
    def run_cdf_simulation(
        self, 
        predicted_mean: float, 
        predicted_std: float, 
        line: float, 
        market: str
    ) -> SimulationResult:
        """
        V20: Alias for compute_cdf_probability for backwards compatibility.
        """
        return self.compute_cdf_probability(
            mean=predicted_mean,
            std=predicted_std,
            line=line,
            market=market
        )


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
    
    def calculate_ev_stake(
        self, 
        ev: float,
        bankroll: float
    ) -> Tuple[float, float]:
        """
        V20 EMPIRICAL: Pure EV-proportional staking.
        
        NO hardcoded EV tiers. Stake is linear with EV.
        stake_pct = min(EV * 2, 10%)
        
        This is mathematically equivalent to fractional Kelly.
        """
        if ev <= 0:
            return 0.0, 0.0
        
        # V20: Linear EV-proportional staking, capped at 10%
        stake_pct = min(ev * 2.0, 0.10)
        
        return stake_pct * bankroll, stake_pct
    
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
        V19 REFACTOR: Pure statistical decision output.
        
        REMOVED: Grade system, narratives, rollover scores, confidence warnings
        OUTPUT: EV, Predicted Mean, Std Dev, Win Probability
        
        Formula: P(Over) = 1 - Φ((line - μ) / σ)
        """
        # 1. Get CDF-computed probability from simulation
        prob_over_cdf = simulation.over_prob
        prob_under_cdf = simulation.under_prob
        
        # 2. Determine side based on which has higher probability
        if prob_over_cdf > prob_under_cdf:
            side = "OVER"
            cdf_prob = prob_over_cdf
        else:
            side = "UNDER"
            cdf_prob = prob_under_cdf
        
        # 3. Use ML probability if available (from regression model's CDF)
        ml_prob_over = projection.ml_prob if projection.ml_prob is not None else None
        
        # 4. Select best probability estimate
        if ml_prob_over is not None:
            if side == "OVER":
                prob = ml_prob_over
            else:
                prob = 1.0 - ml_prob_over
        else:
            prob = cdf_prob

        # 5. Calculate EV
        ev = self.calculate_ev(prob, odds)

        # 6. Calculate Stake (EV-based, no grades)
        stake, fraction = self.calculate_ev_stake(ev, bankroll)
        
        # 7. Get predicted mean and std from projection
        predicted_mean = projection.final_projection
        predicted_std = projection.adjustments.get('predicted_std', features.std)

        return BetDecision(
            recommended_side=side,
            probability=prob,
            expected_value=ev,
            predicted_mean=predicted_mean,
            predicted_std=predicted_std,
            kelly_stake=stake,
            kelly_fraction=fraction
        )


# =============================================================================
# LAYER 6: BACKTESTER
# =============================================================================

class Backtester:
    """
    Walk-forward backtesting engine with STRICT TEMPORAL INTEGRITY.
    
    V20 EMPIRICAL:
    - Walk-forward validation ONLY (no look-ahead bias)
    - Features are FROZEN at prediction time (point-in-time snapshots)
    - Season context is RECORDED as metadata, NOT used for decay
    - Let ML learn cross-season relationships from data
    
    REMOVED (V20):
    - TEMPORAL_DECAY_PER_SEASON (hardcoded belief)
    - _calculate_temporal_weight (heuristic decay function)
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
        self.feature_engineer = feature_engineer
        self.model_engine = model_engine
        self.simulation_engine = simulation_engine
        self.decision_policy = decision_policy
        self.config = config
        self._rng = np.random.default_rng(42)  # Reproducible randomness for CLV
    
    def _get_season_for_date(self, game_date: pd.Timestamp) -> str:
        """
        Determine NBA season string for a given date.
        NBA season runs Oct-Jun. Oct 2024 → "2024-25", Jan 2025 → "2024-25"
        
        V20: Used for METADATA only, not for decay calculations.
        """
        if game_date.month >= 10:  # Oct-Dec
            return f"{game_date.year}-{str(game_date.year + 1)[-2:]}"
        else:  # Jan-Sep
            return f"{game_date.year - 1}-{str(game_date.year)[-2:]}"
    
    def _freeze_feature_snapshot(
        self, 
        features: 'FeatureVector', 
        prediction_date: pd.Timestamp
    ) -> Dict[str, Any]:
        """
        Create a frozen snapshot of features at prediction time.
        
        This captures EXACTLY what was known at prediction time,
        preventing any post-hoc feature drift.
        
        V20.3: 24 raw observables including absence-aware features.
        """
        return {
            'snapshot_date': prediction_date.strftime('%Y-%m-%d'),
            'snapshot_season': self._get_season_for_date(prediction_date),
            # Core features (frozen at prediction time) - V20.3 raw observables
            'avg_minutes': features.avg_minutes,
            'ema': features.ema,
            'std': features.std,
            'opponent_drtg_season': features.opponent_drtg_season,
            'line': features.line,
            'spread': features.spread,
            'game_total': features.game_total,
            'days_rest': features.days_rest,
            'is_home': int(features.is_home),
            'is_b2b': int(features.is_b2b),
            'games_played': features.games_played,
            # V20.2: Pace context
            'opponent_pace': features.opponent_pace,
            'team_pace': features.team_pace,
            # V20.2: Momentum & splits
            'trend_5g': features.trend_5g,
            'home_avg': features.home_avg,
            'away_avg': features.away_avg,
            # V20.3 NEW: True-Shooting features
            'feat_ts_pct': features.feat_ts_pct,
            'feat_ts_pct_delta': features.feat_ts_pct_delta,
            # V20.3 NEW: Behavior & Risk features
            'feat_min_volatility': features.feat_min_volatility,
            'feat_foul_rate': features.feat_foul_rate,
            'feat_cv': features.feat_cv,
            # V20.3 NEW: Absence-aware features
            'team_out_ppg': features.team_out_ppg,
            'team_out_count': features.team_out_count,
            'opp_out_ppg': features.opp_out_ppg,
            'opp_out_count': features.opp_out_count,
            # Market identity
            'market_scoring': features.market_scoring,
            'market_counting': features.market_counting,
            'market_combo': features.market_combo,
            'market_rare': features.market_rare,
            # Data quality flags (for audit trail)
            'had_warnings': len(features.data_quality.warnings) > 0 if features.data_quality else False
        }
    
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
        lookback: int = 15,
        test_days: int = 30,
        line_offset: float = 0.0,  # Test at actual average ± offset
        fixed_spread: float = 0.0,
        progress_callback=None,
        preloaded_df: pd.DataFrame = None,  # Optional: use pre-loaded game logs
        bulk_loader: 'BulkGameLogLoader' = None,  # V20.3: For absence feature computation
        preloaded_team_stats_by_season: Dict[str, Tuple[pd.DataFrame, float, float]] = None,  # V20.3: Per-season team stats
        preloaded_position: str = None  # V20.3: Pre-fetched player position (avoids API call in workers)
    ) -> Optional[BacktestSummary]:
        """
        Run STRICTLY walk-forward backtest with TEMPORAL INTEGRITY.
        
        V20.3 EMPIRICAL:
        - Uses ONLY data available at each prediction time (no future leakage)
        - Features are FROZEN at prediction time (point-in-time snapshots)
        - Season context is RECORDED as metadata (no decay applied)
        - Absence features computed from game logs (who didn't play)
        - Team stats (DRTG, Pace) use the CORRECT season's data (temporal fix)
        
        For each game in the test period:
        1. FREEZE: Capture only data available up to that game
        2. PREDICT: Generate prediction using frozen features
        3. EVALUATE: Compare to actual result (ground truth)
        4. LOG: Store frozen snapshot for audit trail
        """
        
        # Fetch all available data (or use preloaded)
        if preloaded_df is not None and len(preloaded_df) > 0:
            df = preloaded_df.copy()
        else:
            df = self.data_loader.fetch_multi_season_logs(player_id)
        
        # TEMPORAL INTEGRITY: Ensure data is sorted chronologically
        if 'GAME_DATE' not in df.columns:
            logger.error("Cannot run walk-forward backtest without GAME_DATE column")
            return None
        df = df.sort_values('GAME_DATE').reset_index(drop=True)
        
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
        
        # Get player position (use pre-loaded if available to avoid API calls in workers)
        if preloaded_position is not None:
            player_position = preloaded_position
            logger.debug(f"Using pre-loaded position: {player_position}")
        else:
            player_position = self.data_loader.get_player_position(player_id)
            logger.debug(f"Fetched position from API: {player_position}")
        
        # V20.3 TEMPORAL FIX: Per-season team stats for correct historical context
        # If pre-loaded dict provided, we look up by season during the loop
        # If not provided, fetch current season only (for live predictions / single-season backtest)
        if preloaded_team_stats_by_season is None:
            # Fallback: single season mode (live predictions or legacy)
            current_stats = self.data_loader.fetch_team_stats()
            preloaded_team_stats_by_season = {self.config.CURRENT_SEASON: current_stats}
            logger.debug("Using current season team stats only (single-season mode)")
        else:
            logger.debug(f"Using per-season team stats ({len(preloaded_team_stats_by_season)} seasons)")
        
        # Sort by date
        df = df.sort_values('GAME_DATE').reset_index(drop=True)
        
        # NOTE: test_start_idx already calculated above using actual_test_days
        # (flexible logic that uses available games even if less than desired)
        
        results = []
        
        for i in range(test_start_idx, total_games):
            if progress_callback:
                progress_callback((i - test_start_idx) / (total_games - test_start_idx))
            
            # =====================================================================
            # TEMPORAL INTEGRITY: STRICT WALK-FORWARD DATA SLICING
            # We use ONLY games that occurred BEFORE the prediction date
            # This is the ONLY way to get honest backtest metrics
            # =====================================================================
            historical_df = df.iloc[:i].copy()  # STRICTLY before index i
            
            # The game we're predicting (our target/ground truth)
            target_game = df.iloc[i]
            prediction_date = pd.to_datetime(target_game['GAME_DATE'])
            
            if len(historical_df) < self.config.BACKTEST_MIN_GAMES:
                continue
            
            # =====================================================================
            # V20.3 TEMPORAL FIX: Look up team stats for the CORRECT season
            # This prevents using 2025-26 team stats for a 2022-23 game
            # =====================================================================
            game_season = self._get_season_for_date(prediction_date)
            if game_season in preloaded_team_stats_by_season:
                team_stats, avg_pace, avg_def = preloaded_team_stats_by_season[game_season]
            else:
                # Fallback: use most recent available season's stats
                fallback_season = max(preloaded_team_stats_by_season.keys())
                team_stats, avg_pace, avg_def = preloaded_team_stats_by_season[fallback_season]
                logger.debug(f"Season {game_season} stats not pre-loaded, using {fallback_season}")
            
            # =====================================================================
            # V20 EMPIRICAL: Record season context as METADATA only
            # No decay factors applied - let ML learn cross-season patterns
            # =====================================================================
            current_season_games = 0
            prev_season_games = 0
            for _, row in historical_df.iterrows():
                game_dt = pd.to_datetime(row['GAME_DATE'])
                if self._get_season_for_date(game_dt) == self._get_season_for_date(prediction_date):
                    current_season_games += 1
                else:
                    prev_season_games += 1
            
            # Log for awareness only (no decay applied)
            if prev_season_games > 0:
                logger.debug(f"Using {current_season_games} current + {prev_season_games} prev-season games")
            
            # Determine line (use rolling average as proxy for what line would have been)
            rolling_avg = historical_df[market].tail(lookback).mean()
            line = rolling_avg + line_offset
            
            # Extract game context
            matchup = target_game.get('MATCHUP', '')
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
                    curr_date = prediction_date
                    days_rest = max(0, (curr_date - prev_date).days - 1)
                    is_b2b = days_rest == 0
                except Exception:
                    days_rest = 1  # Default to 1 day rest
                    is_b2b = False
            else:
                days_rest = 3  # First game of dataset, assume well-rested
                is_b2b = False
            
            # Find opponent ID (if possible)
            all_teams = teams.get_teams()
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
            
            # =========================================================
            # V20.3: COMPUTE ABSENCE FEATURES FROM GAME LOGS
            # Ground truth: who actually played vs who didn't
            # =========================================================
            absence_features = {
                'team_out_ppg': -1.0,
                'team_out_count': -1,
                'opp_out_ppg': -1.0,
                'opp_out_count': -1
            }
            if bulk_loader is not None and player_team_id is not None and opponent_id != 0:
                try:
                    absence_features = bulk_loader.compute_absence_features(
                        game_date=prediction_date,
                        player_team_id=player_team_id,
                        opponent_team_id=opponent_id,
                        player_id=player_id
                    )
                except Exception as e:
                    logger.debug(f"Absence feature computation failed: {e}")
            
            try:
                # Build feature vector using historical data only
                features = self.feature_engineer.build_feature_vector(
                    player_id=player_id,
                    player_name=player_name,
                    opponent_id=opponent_id,
                    opponent_abbrev=opp_abbrev,
                    is_home=is_home,
                    is_b2b=is_b2b,
                    spread=backtest_spread,
                    df=historical_df,
                    stat_col=market,
                    line=line,
                    lookback=lookback,
                    team_stats=team_stats,
                    avg_def=avg_def,
                    market=market,
                    days_rest=days_rest,
                    game_total=backtest_game_total,
                    player_team_id=player_team_id,
                    # FIX: Set to None to prevent workers from hitting ESPN
                    player_team_abbrev=None,  
                    data_loader=None
                )
                
                # V20.3: Inject absence features into feature vector
                features.team_out_ppg = absence_features['team_out_ppg']
                features.team_out_count = absence_features['team_out_count']
                features.opp_out_ppg = absence_features['opp_out_ppg']
                features.opp_out_count = absence_features['opp_out_count']
                
                # =========================================================
                # TEMPORAL INTEGRITY: FREEZE FEATURE SNAPSHOT
                # Capture EXACTLY what was known at prediction time
                # This is our audit trail for walk-forward integrity
                # =========================================================
                frozen_snapshot = self._freeze_feature_snapshot(features, prediction_date)
                
                # Generate projection
                projection = self.model_engine.generate_projection(features)
                
                # V19: Use CDF-based probability (deterministic, no Monte Carlo)
                simulation = self.simulation_engine.compute_cdf_probability(
                    mean=projection.final_projection,
                    std=features.std,
                    line=line,
                    market=market,
                    features=features
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
                
                # =========================================================
                # V19 TEMPORAL INTEGRITY: Use FROZEN snapshot for features
                # NOT the potentially-mutated features object
                # V20: Raw observables only (no usage_mult, rest_factor)
                # =========================================================
                feature_dict = {
                    'avg_minutes': frozen_snapshot['avg_minutes'],
                    'opponent_drtg_season': frozen_snapshot['opponent_drtg_season'],
                    'line': frozen_snapshot['line'],
                    'ema': frozen_snapshot['ema'],
                    'std': frozen_snapshot['std'],
                    'is_home': frozen_snapshot['is_home'],
                    'spread': frozen_snapshot['spread'],
                    'game_total': frozen_snapshot['game_total'],
                    'games_played': frozen_snapshot['games_played'],
                    'days_rest': frozen_snapshot['days_rest'],
                    'is_b2b': frozen_snapshot['is_b2b'],
                    # V20.2: Pace context
                    'opponent_pace': frozen_snapshot['opponent_pace'],
                    'team_pace': frozen_snapshot['team_pace'],
                    # V20.2: Momentum & splits
                    'trend_5g': frozen_snapshot['trend_5g'],
                    'home_avg': frozen_snapshot['home_avg'],
                    'away_avg': frozen_snapshot['away_avg'],
                    # V20.3 NEW: True-Shooting features
                    'feat_ts_pct': frozen_snapshot.get('feat_ts_pct', -1.0),
                    'feat_ts_pct_delta': frozen_snapshot.get('feat_ts_pct_delta', -1.0),
                    # V20.3 NEW: Behavior & Risk features
                    'feat_min_volatility': frozen_snapshot.get('feat_min_volatility', -1.0),
                    'feat_foul_rate': frozen_snapshot.get('feat_foul_rate', -1.0),
                    'feat_cv': frozen_snapshot.get('feat_cv', -1.0),
                    # V20.3 NEW: Absence-aware features
                    'team_out_ppg': frozen_snapshot['team_out_ppg'],
                    'team_out_count': frozen_snapshot['team_out_count'],
                    'opp_out_ppg': frozen_snapshot['opp_out_ppg'],
                    'opp_out_count': frozen_snapshot['opp_out_count'],
                    # Market identity (one-hot encoded)
                    'market_scoring': frozen_snapshot['market_scoring'],
                    'market_counting': frozen_snapshot['market_counting'],
                    'market_combo': frozen_snapshot['market_combo'],
                    'market_rare': frozen_snapshot['market_rare'],
                    # Temporal audit trail
                    '_snapshot_date': frozen_snapshot['snapshot_date'],
                    '_snapshot_season': frozen_snapshot['snapshot_season'],
                    # V20.3: Data quality flag for filtering
                    '_had_warnings': frozen_snapshot['had_warnings'],
                }
                
                results.append(BacktestResult(
                    date=target_game['GAME_DATE'].strftime('%Y-%m-%d'),
                    player_name=player_name,
                    market=market,
                    line=line,
                    predicted_side=decision.recommended_side,
                    predicted_prob=decision.probability,
                    predicted_ev=decision.expected_value,
                    actual_value=actual_value,
                    hit=hit,
                    grade='EV+' if decision.expected_value > 0 else 'EV-',  # V19: EV-based label
                    opponent=opp_abbrev,
                    position=player_position,
                    is_home=is_home,
                    features=feature_dict,
                    # V19: Temporal audit trail
                    snapshot_date=frozen_snapshot['snapshot_date'],
                    snapshot_season=frozen_snapshot['snapshot_season']
                ))
                
            except Exception as e:
                logger.warning(f"Backtest error for game {i}: {e}")
                continue
        
        if not results:
            return None
        
        return self._calculate_summary(results)
    
    def _calculate_summary(self, results: List[BacktestResult]) -> BacktestSummary:
        """
        Calculate backtest summary metrics with TEMPORAL INTEGRITY reporting.
        
        V19: Includes season boundary crossings and data quality indicators.
        """
        
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
                # V19 TEMPORAL INTEGRITY: Frozen snapshot fields
                'snapshot_date': r.snapshot_date,
                'snapshot_season': r.snapshot_season,
                'snapshot_days_rest': r.features.get('days_rest', 0),
                'snapshot_def_rank': int(r.features.get('opponent_drtg_season', 115)),
                'tag': 'backtest',  # Tag for filtering
                'grade': r.grade,
            }
            # Flatten features with 'feat_' prefix
            for feat_name, feat_value in r.features.items():
                # Skip internal audit fields (start with _)
                if str(feat_name).startswith('_'):
                    continue
                # Preserve existing 'feat_' prefix if present to avoid 'feat_feat_*' duplication
                if str(feat_name).startswith('feat_'):
                    col_name = feat_name
                else:
                    col_name = f'feat_{feat_name}'
                row[col_name] = feat_value
            # Add missing tracker columns with defaults
            row['closing_line'] = 0
            row['clv'] = 0
            rows.append(row)
        
        results_df = pd.DataFrame(rows)
        
        # V20: Log season span for awareness (no decay applied)
        seasons_in_backtest = results_df['snapshot_season'].nunique() if 'snapshot_season' in results_df.columns else 1
        if seasons_in_backtest > 1:
            logger.info(f"Backtest spans {seasons_in_backtest} seasons (raw data, no decay)")
        
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

# Market types for one-hot encoding (V17: Market Identity)
# Groups markets by statistical similarity to reduce feature explosion
MARKET_GROUPS = {
    'scoring': ['PTS'],           # High volume, high variance
    'counting': ['REB', 'AST'],   # Medium volume counting stats
    'combo': ['PRA', 'PR', 'PA', 'RA'],  # Combined markets
    'rare': ['3PM', 'STL', 'BLK'] # Low count, Poisson-like
}

# Canonical feature list used for model training (exact order - canonical numeric features)
# V20.3: Base V20 features plus absence-awareness and TS%
TRAINING_FEATURE_COLUMNS = [
    # V20.3 EMPIRICAL: Pure raw observables only.
    # Order MUST match FeatureVector.to_ml_array() exactly.
    # Statistical baseline (3)
    'feat_avg_minutes',
    'feat_ema',
    'feat_std',
    # Opponent context (1)
    'feat_opponent_drtg_season',
    # Line and game context (3)
    'feat_line',
    'feat_spread',
    'feat_game_total',
    # Rest observables (3) - raw values, not multipliers
    'feat_days_rest',
    'feat_is_home',
    'feat_is_b2b',
    # Sample size (1)
    'feat_games_played',
    # V20.2: Pace context (2)
    'feat_opponent_pace',
    'feat_team_pace',
    # V20.2: Momentum & splits (3)
    'feat_trend_5g',
    'feat_home_avg',
    'feat_away_avg',
    # V20.3 NEW: True Shooting (2)
    'feat_ts_pct',
    'feat_ts_pct_delta',
    # V20.3 NEW: Absence-aware (4)
    'feat_team_out_ppg',
    'feat_team_out_count',
    'feat_opp_out_ppg',
    'feat_opp_out_count',
    # Market identity (4)
    'feat_market_scoring',
    'feat_market_counting',
    'feat_market_combo',
    'feat_market_rare',
    # V20.3 NEW: Behavior & Risk features
    'feat_min_volatility',
    'feat_foul_rate',
    'feat_cv',
]

# Canonical export schema used for writing ML training CSVs (single source of truth)
ML_EXPORT_METADATA = [
    'date', 'player', 'opponent', 'market', 'position',
    'line', 'predicted_side', 'predicted_prob', 'predicted_ev', 'projected_value',
    'result', 'hit', 'actual_value', 'margin', 'margin_pct', 'result_quality',
    'tag', 'grade',
]

# The full ordered schema for CSV export: metadata followed by canonical training features
ML_EXPORT_SCHEMA = ML_EXPORT_METADATA + TRAINING_FEATURE_COLUMNS.copy()

def get_market_group_features(market: str) -> Dict[str, int]:
    """
    Convert market to one-hot encoded group features.
    Returns dict with feat_market_* keys (0 or 1).
    """
    features = {
        'feat_market_scoring': 0,
        'feat_market_counting': 0,
        'feat_market_combo': 0,
        'feat_market_rare': 0
    }
    for group_name, markets in MARKET_GROUPS.items():
        if market in markets:
            features[f'feat_market_{group_name}'] = 1
            break
    return features


def extract_features_dynamically(features_obj: Any, market: Optional[str] = None) -> Dict[str, Any]:
    """Dynamically extract ML feature keys based on the canonical `ML_EXPORT_SCHEMA`.

    - Iterates over `ML_EXPORT_SCHEMA` looking for keys starting with `feat_`.
    - For each, derives the attribute name on the `features_obj` by stripping the `feat_` prefix.
    - If attribute exists, converts to an appropriate type and returns it.
    - If attribute is missing, uses a sensible sentinel and logs a debug message.
    - Market identity features (feat_market_*) are handled by merging `get_market_group_features`.
    """
    result: Dict[str, Any] = {}

    # Resolve market for market-derived features
    resolved_market = market or getattr(features_obj, 'market', None) or 'PTS'

    for col in ML_EXPORT_SCHEMA:
        if not col.startswith('feat_'):
            continue

        # Market identity handled separately
        if col.startswith('feat_market_'):
            continue

        attr_name = col[len('feat_'):]

        # Try multiple candidate attribute names to support both 'avg_minutes' and 'feat_ts_pct' style attributes
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
                    # Generic numeric features -> float
                    val = float(raw_val) if raw_val is not None else -1.0
            except Exception:
                # Fallback to sentinel on any conversion error
                if col.endswith('_count'):
                    val = -1
                elif col in ('feat_is_home', 'feat_is_b2b'):
                    val = 0
                else:
                    val = -1.0
                logger.debug(f"Failed to coerce feature '{found_attr}' value; using sentinel {val}")
        else:
            # Missing attribute -> sentinel
            if col.endswith('_count'):
                val = -1
            elif col in ('feat_is_home', 'feat_is_b2b'):
                val = 0
            else:
                val = -1.0
            logger.debug(f"Feature '{attr_name}' missing on features object; using sentinel {val}")

        result[col] = val

    # Merge market identity one-hot features
    result.update(get_market_group_features(resolved_market))

    return result


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
    V20.3 OPTIMIZED: Bulk loader with High-Speed Indexing & Roster Caching.
    
    Fixes the CPU bottleneck by caching team rosters and start dates,
    reducing 7.5 million dataframe operations to near zero.
    """
    
    def __init__(self, config: Config = CONFIG):
        self.config = config
        self._cache: Dict[int, pd.DataFrame] = {}  # player_id -> game_logs
        self._loaded = False
        self._seasons_loaded: List[str] = []
        
        # === OPTIMIZATION INDEXES ===
        self._indexes_built = False
        self._roster_index: Dict[int, set] = {} # team_id -> set(player_ids)
        self._daily_index: Dict[str, set] = {}  # "YYYY-MM-DD_TeamID" -> set(player_ids)
        
        # === NEW: CACHE ROSTERS TO PREVENT RE-CALCULATION ===
        self._roster_cache: Dict[int, Dict] = {} # team_id -> cached_roster_dict
    
    def load_all_game_logs(self, progress_callback=None, seasons: List[str] = None) -> bool:
        if self._loaded: return True
        
        all_logs = []
        seasons = seasons or list(self.config.TRAINING_SEASONS)
        total_seasons = len(seasons)
        
        logger.info(f"📥 Loading {total_seasons} seasons of data for ML training...")
        
        for i, season in enumerate(seasons):
            try:
                logger.info(f"📥 Fetching ALL game logs for {season} (API call {i+1}/{total_seasons})...")
                time.sleep(self.config.API_DELAY * 2)
                
                logs = leaguegamelog.LeagueGameLog(season=season, player_or_team_abbreviation='P', timeout=120)
                df = logs.get_data_frames()[0]
                
                if df is not None and len(df) > 0:
                    df['SEASON'] = season
                    all_logs.append(df)
                    self._seasons_loaded.append(season)
                    logger.info(f"  ✓ {season}: {len(df):,} game log entries")
                else:
                    logger.warning(f"  ✗ {season}: No data returned")
                    
                if progress_callback: progress_callback((i + 1) / total_seasons * 0.3)
                    
            except Exception as e:
                logger.error(f"  ✗ Failed to fetch {season} logs: {e}")
                if season == self.config.CURRENT_SEASON: return False
        
        if not all_logs: return False
        
        combined = pd.concat(all_logs, ignore_index=True)
        self._organize_by_player(combined)
        self._loaded = True
        return True
    
    def _organize_by_player(self, df: pd.DataFrame):    
        """
        Organize the bulk data by player ID.
        """
        # 1. Clean Column Names & Filter - Do NOT aggressively drop unknown columns
        # 2. Force numeric conversion on a wider set of box-score columns so we keep FGA/FTA
        numeric_cols = [
            'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
            'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
            'PLUS_MINUS'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        # Calculated composite stats (safe - require numeric inputs)
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

        # 3. Parse minutes robustly
        if 'MIN' in df.columns:
            def parse_min(x):
                try:
                    if pd.isna(x):
                        return 0.0
                    x_str = str(x)
                    if ':' in x_str:
                        parts = x_str.split(':')
                        return float(parts[0]) + float(parts[1]) / 60.0
                    return float(x)
                except Exception:
                    return 0.0
            df['MIN_FLOAT'] = df['MIN'].apply(parse_min)

        # 4. Metadata
        if 'MATCHUP' in df.columns:
            df['IS_HOME'] = df['MATCHUP'].str.contains('vs.', case=False, na=False)
        else:
            df['IS_HOME'] = True

        if 'GAME_DATE' in df.columns:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

        # 5. DAYS_REST (needs sorting by player + date)
        if 'PLAYER_ID' in df.columns and 'GAME_DATE' in df.columns:
            df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])
            df['DAYS_REST'] = df.groupby('PLAYER_ID')['GAME_DATE'].diff().dt.days - 1
            df['DAYS_REST'] = df['DAYS_REST'].fillna(3).clip(lower=0)

        # 6. Final ordering: most recent first for cache
        if 'GAME_DATE' in df.columns:
            df = df.sort_values('GAME_DATE', ascending=False)

        # 7. Cache by player
        if 'PLAYER_ID' in df.columns:
            for player_id, player_df in df.groupby('PLAYER_ID'):
                self._cache[player_id] = player_df.reset_index(drop=True)

        logger.info(f"  ✓ Organized data for {len(self._cache):,} unique players (with computed stats)")

    def _build_optimization_indexes(self):
        """
        CRITICAL OPTIMIZATION: Build lookup tables for O(1) access.
        This prevents scanning 844 dataframes for every single backtest day.
        """
        if self._indexes_built: return
        
        # logger.info("Building optimization indexes...")
        
        for player_id, df in self._cache.items():
            if 'TEAM_ID' not in df.columns or 'GAME_DATE' not in df.columns:
                continue
            
            # 1. Update Roster Index (Who played for which team)
            teams_played_for = df['TEAM_ID'].unique()
            for tid in teams_played_for:
                if tid not in self._roster_index:
                    self._roster_index[tid] = set()
                self._roster_index[tid].add(player_id)
            
            # 2. Update Daily Index (Who played on Date X for Team Y)
            # Use vectorization for speed
            dates = df['GAME_DATE'].dt.strftime('%Y-%m-%d').values
            teams = df['TEAM_ID'].values
            
            for dt, tid in zip(dates, teams):
                key = f"{dt}_{tid}"
                if key not in self._daily_index:
                    self._daily_index[key] = set()
                self._daily_index[key].add(player_id)
                
        self._indexes_built = True

    def get_player_logs(self, player_id: int) -> pd.DataFrame:
        if not self._loaded: return pd.DataFrame()
        return self._cache.get(player_id, pd.DataFrame())
    
    def get_top_players_from_cache(self, limit: int = 50, min_games: int = 15, current_season: str = None) -> List[Dict]:
        if not self._loaded: return []
        if current_season is None: current_season = CONFIG.CURRENT_SEASON
        
        player_list = []
        for player_id, df in self._cache.items():
            current_season_games = df[df.get('SEASON', '') == current_season] if 'SEASON' in df.columns else df
            if len(current_season_games) < min_games: continue
            
            latest = df.iloc[0]
            player_list.append({
                'id': int(player_id),
                'full_name': latest.get('PLAYER_NAME', f'Player_{player_id}'),
                'team': latest.get('TEAM_ABBREVIATION', 'UNK'),
                'games': len(current_season_games),
                'position': 'SF'
            })
        
        player_list.sort(key=lambda x: x['games'], reverse=True)
        return player_list[:limit]
    
    def get_team_roster(self, team_id: int) -> Dict[int, Dict]:
        """
        Get all players for a team using the optimized index + CACHING.
        Cache results to prevent filtering 7.5 million times.
        """
        if not self._loaded: return {}
        
        # 1. CHECK CACHE (Fastest)
        if team_id in self._roster_cache:
            return self._roster_cache[team_id]
            
        if not self._indexes_built: self._build_optimization_indexes()
        
        # 2. COMPUTE (Only done once per team)
        player_ids = self._roster_index.get(team_id, set())
        
        roster = {}
        for pid in player_ids:
            df = self._cache.get(pid)
            # Filter for this team's games
            team_games = df[df['TEAM_ID'] == team_id]
            
            if len(team_games) > 0:
                ppg = team_games['PTS'].mean() if 'PTS' in team_games.columns else 0.0
                name = team_games['PLAYER_NAME'].iloc[0] if 'PLAYER_NAME' in team_games.columns else 'Unknown'
                # Pre-calculate min_date and max_date to avoid expensive dataframe scans in hot loop
                min_date = team_games['GAME_DATE'].min()
                # FIX: Track last game date for recency-based "ghost" filtering
                max_date = team_games['GAME_DATE'].max()
                
                roster[pid] = {
                    'name': name, 
                    'ppg': ppg, 
                    'games': len(team_games),
                    'min_date': min_date,  # STORE THIS FOR SPEED
                    'max_date': max_date
                }
        
        # 3. SAVE TO CACHE
        self._roster_cache[team_id] = roster
        return roster
    
    def get_players_who_played_on_date(self, game_date: pd.Timestamp, team_id: int) -> set:
        """Get set of player_ids who played on date using optimized index."""
        if not self._loaded: return set()
        if not self._indexes_built: self._build_optimization_indexes()
        
        # INSTANT LOOKUP (O(1)) instead of scanning 844 dataframes
        key = f"{game_date.strftime('%Y-%m-%d')}_{team_id}"
        return self._daily_index.get(key, set())
    
    def compute_absence_features(self, game_date: pd.Timestamp, player_team_id: int, opponent_team_id: int, player_id: int) -> Dict[str, float]:
        """V20.3 Optimized Absence Calculation with Roster Caching."""
        if not self._loaded: return {'team_out_ppg': -1.0, 'team_out_count': -1, 'opp_out_ppg': -1.0, 'opp_out_count': -1}
        
        # Ensure indexes are built (only happens once per worker)
        if not self._indexes_built: self._build_optimization_indexes()
        
        # 1. Get full rosters (Instantly from cache after first run)
        team_roster = self.get_team_roster(player_team_id)
        opp_roster = self.get_team_roster(opponent_team_id)
        
        # 2. Get active players for this specific game (Fast dict lookup)
        team_played = self.get_players_who_played_on_date(game_date, player_team_id)
        opp_played = self.get_players_who_played_on_date(game_date, opponent_team_id)
        
        # 3. Calculate missing production
        team_out_ppg = 0.0
        team_out_count = 0
        
        # Define ghost cutoff: players who haven't played in the last 60 days are treated as gone
        ghost_cutoff = game_date - pd.Timedelta(days=60)
        
        # Iterate roster (~15 items) - Super fast now
        for pid, info in team_roster.items():
            if pid == player_id: continue
            # Check if they are NOT playing today
            if pid not in team_played:
                # Filter 1: Must have played at least 5 games
                if info.get('games', 0) < 5:
                    continue
                # Filter 2: Must have joined the team BEFORE today
                if info.get('min_date') >= game_date:
                    continue
                # Filter 3: Ghost player filter - if last game < ghost_cutoff, skip
                if info.get('max_date') < ghost_cutoff:
                    continue
                # If all filters pass, count them as "out"
                team_out_ppg += info.get('ppg', 0.0)
                team_out_count += 1
        
        opp_out_ppg = 0.0
        opp_out_count = 0
        for pid, info in opp_roster.items():
            if pid not in opp_played:
                if info.get('games', 0) < 5:
                    continue
                if info.get('min_date') >= game_date:
                    continue
                if info.get('max_date') < ghost_cutoff:
                    continue
                opp_out_ppg += info.get('ppg', 0.0)
                opp_out_count += 1
        
        return {
            'team_out_ppg': round(team_out_ppg, 1),
            'team_out_count': team_out_count,
            'opp_out_ppg': round(opp_out_ppg, 1),
            'opp_out_count': opp_out_count
        }
    
    @property
    def player_count(self) -> int:
        return len(self._cache)


# =============================================================================
# GLOBAL SHARED MEMORY (Workers Only)
# =============================================================================
# These variables hold the read-only data in each worker process
# preventing the need to re-pickle/re-transmit 50MB+ for every task.
_worker_bulk_loader = None
_worker_team_stats = None
_worker_config = None

def _init_worker(bulk_cache_dict, team_stats_dict, config_obj):
    """
    Initialize worker process with shared data ONE TIME.
    This eliminates overhead of sending data for every task.
    """
    global _worker_bulk_loader, _worker_team_stats, _worker_config
    
    # 1. Store Config
    _worker_config = config_obj
    
    # 2. Reconstruct BulkLoader (Expensive Step - done once per worker)
    _worker_bulk_loader = BulkGameLogLoader(_worker_config)
    # Load raw data into cache
    for pid, records in bulk_cache_dict.items():
        if records:
            df = pd.DataFrame(records)
            # Ensure DateTime
            if 'GAME_DATE' in df.columns:
                df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            
            # V20.3 FIX: Ensure composite stats exist in worker cache
            # This handles cases where data might be passed without pre-calc
            if 'PRA' not in df.columns and {'PTS','REB','AST'}.issubset(df.columns):
                cols = ['PTS', 'REB', 'AST']
                df[cols] = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0)
                df['PRA'] = df['PTS'] + df['REB'] + df['AST']
                df['PA'] = df['PTS'] + df['AST']
                df['PR'] = df['PTS'] + df['REB']
                df['RA'] = df['REB'] + df['AST']
            if 'FG3M' in df.columns and '3PM' not in df.columns:
                df['3PM'] = df['FG3M']
                
            _worker_bulk_loader._cache[int(pid)] = df
            
    _worker_bulk_loader._loaded = True
    
    # 3. Build Optimization Indexes (Expensive Step - done once per worker)
    # This makes absence checks instant (O(1))
    _worker_bulk_loader._build_optimization_indexes()
    
    # 4. Reconstruct Team Stats
    _worker_team_stats = {}
    for season, (records, avg_pace, avg_def) in team_stats_dict.items():
        if records:
            df = pd.DataFrame(records)
            if 'TEAM_ID' in df.columns:
                df = df.set_index('TEAM_ID')
        else:
            df = pd.DataFrame()
        _worker_team_stats[season] = (df, avg_pace, avg_def)


def _backtest_worker_task(args: Tuple) -> Optional[pd.DataFrame]:
    """
    Worker function for parallel backtest execution.
    V20.3 OPTIMIZED: Uses pre-initialized global data.
    """
    import os
    
    # Unpack lightweight task arguments (No massive data blobs here!)
    (player_id, player_name, market, lookback, test_days, 
     player_df_dict, player_position, task_idx, total_tasks) = args
    
    # Access shared global data
    global _worker_bulk_loader, _worker_team_stats, _worker_config
    
    worker_id = os.getpid()
    
    # [LOG 1] STARTING TASK
    logger.info(f"🔄 [Worker {worker_id}] STARTING Task {task_idx}/{total_tasks}: {player_name} ({market})")
    
    try:
        # Reconstruct player DataFrame from dict (Tiny, just one player)
        player_df = pd.DataFrame(player_df_dict)
        if 'GAME_DATE' in player_df.columns:
            player_df['GAME_DATE'] = pd.to_datetime(player_df['GAME_DATE'])
        
        # Create fresh instances for this worker using cached config
        data_loader = DataLoader(_worker_config)
        feature_engineer = FeatureEngineer(_worker_config)
        model_engine = ModelEngine(_worker_config)
        simulation_engine = SimulationEngine(_worker_config)
        decision_policy = DecisionPolicy(_worker_config)
        
        backtester = Backtester(
            data_loader, feature_engineer, model_engine, 
            simulation_engine, decision_policy, _worker_config
        )
        
        # Run Backtest using GLOBAL bulk_loader and team_stats
        summary = backtester.run_backtest(
            player_id=player_id,
            player_name=player_name,
            market=market,
            lookback=lookback,
            test_days=test_days,
            line_offset=0.0,
            fixed_spread=0.0,
            preloaded_df=player_df,
            bulk_loader=_worker_bulk_loader,  # Use the globally cached loader
            preloaded_team_stats_by_season=_worker_team_stats, # Use globally cached stats
            preloaded_position=player_position
        )
        
        # [LOG 3] COMPLETION
        if summary is not None and len(summary.results_df) > 0:
            result_df = summary.results_df.copy()
            result_df['player_id'] = player_id
            logger.info(f"✅ [Worker {worker_id}] FINISHED Task {task_idx}/{total_tasks}: Generated {len(result_df)} samples")
            return result_df
        
        logger.info(f"⚠️ [Worker {worker_id}] FINISHED Task {task_idx}/{total_tasks}: No results generated")
        return None
        
    except Exception as e:
        logger.error(f"❌ [Worker {worker_id}] CRASHED Task {task_idx}/{total_tasks} ({player_name}): {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def generate_ml_training_data(
    output_file: str = "ml_training_data.csv",
    num_players: int = 50,
    markets: List[str] = None,
    test_days: int = 60,
    lookback: int = 15,
    progress_callback=None
) -> pd.DataFrame:
    """
    V20.3 FINAL: High-Performance Parallel Generation.
    Uses 'Initializer' pattern to load data once per worker.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    if markets is None:
        markets = ['PTS', 'REB', 'AST', 'PRA', 'RA', 'PA', 'PR']
    
    logger.info(f"Starting ML data generation for {num_players} players...")
    logger.info(f"Markets: {markets}")
    
    config = CONFIG
    data_loader = DataLoader(config)
    
    # =========================================================================
    # STEP 1: Bulk load ALL game logs (Main Process)
    # =========================================================================
    bulk_loader = BulkGameLogLoader(config)
    if progress_callback: progress_callback(0.05)
    
    if not bulk_loader.load_all_game_logs(progress_callback):
        logger.error("🛑 Failed to bulk load game logs. Aborting.")
        return pd.DataFrame()
    
    logger.info(f"✅ Bulk data loaded: {bulk_loader.player_count:,} players available offline")
    if progress_callback: progress_callback(0.35)
    
    # =========================================================================
    # STEP 2: Extract top players & Stats
    # =========================================================================
    logger.info(f"Extracting top {num_players} players from bulk cache...")
    top_players = bulk_loader.get_top_players_from_cache(
        limit=num_players, 
        min_games=15, 
        current_season=config.CURRENT_SEASON
    )
    
    if len(top_players) == 0:
        logger.error("🛑 No players found in bulk data with sufficient games")
        return pd.DataFrame()

    logger.info(f"Pre-fetching team stats for {len(config.TRAINING_SEASONS)} seasons...")
    all_season_team_stats = data_loader.fetch_all_seasons_team_stats(list(config.TRAINING_SEASONS))
    
    # Serialize team stats for init
    team_stats_serialized = {}
    for season, (stats_df, avg_pace, avg_def) in all_season_team_stats.items():
        records = stats_df.reset_index().to_dict('records') if len(stats_df) > 0 else []
        team_stats_serialized[season] = (records, avg_pace, avg_def)

    # Serialize bulk logs for init
    logger.info("📦 Serializing bulk data for workers (One-time cost)...")
    bulk_cache_serialized = {}
    for pid, pdf in bulk_loader._cache.items():
        bulk_cache_serialized[pid] = pdf.to_dict('records')

    # =========================================================================
    # STEP 3: Build Lightweight Task List
    # =========================================================================
    tasks = []
    skipped_no_data = 0
    
    for player in top_players:
        player_id = player['id']
        player_name = player['full_name']
        player_df = bulk_loader.get_player_logs(player_id)
        
        if len(player_df) < 15: 
            skipped_no_data += 1
            continue
            
        # We still pass specific player data, but NOT the full bulk loader
        player_df_dict = player_df.to_dict('records')
        
        # Position optimization (Default to SF to skip API)
        position = 'SF' 
        
        for market in markets:
            task_idx = len(tasks) + 1
            # Note: We do NOT pass bulk_cache_serialized here anymore!
            tasks.append((
                player_id, player_name, market, lookback, test_days,
                player_df_dict, position, task_idx, 0 # total updated below
            ))
            
    total_tasks = len(tasks)
    tasks = [(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], total_tasks) for t in tasks]
    
    logger.info(f"🎯 Prepared {total_tasks} tasks. Starting execution...")
    
    if total_tasks == 0:
        return pd.DataFrame()

    # =========================================================================
    # STEP 4: Execute with Initializer (The Speed Fix)
    # =========================================================================
    max_workers = max(1, os.cpu_count() - 1)
    logger.info(f"🚀 Launching {max_workers} workers with shared memory init...")
    
    all_results = []
    completed = 0

    # =========================================================================
    # STEP 4: Execute with Initializer (The Speed Fix)
    # Use guarded execution to avoid Windows 'spawn' pickling issues when the
    # module has been re-imported (Streamlit reloads can cause this).
    # If we're not in the main Streamlit process, or if the ProcessPool fails
    # to initialize due to pickling, fall back to a sequential execution.
    # =========================================================================
    if _IN_MAIN_PROCESS:
        try:
            logger.info(f"🚀 Launching {max_workers} workers with shared memory init...")
            # Pass the heavy data ONCE via initializer
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_worker,
                initargs=(bulk_cache_serialized, team_stats_serialized, config)
            ) as executor:

                futures = {executor.submit(_backtest_worker_task, task): task for task in tasks}

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result is not None:
                            all_results.append(result)
                    except Exception as e:
                        logger.warning(f"Task failed: {e}")

                    completed += 1
                    if progress_callback:
                        # Progress from 40% to 100%
                        progress_callback(0.40 + 0.60 * (completed / total_tasks))
        except Exception as e:
            logger.warning("Parallel workers failed (pickling/initializer issue). Falling back to sequential execution.")
            logger.debug(str(e))
            # Sequential fallback
            for task in tasks:
                try:
                    result = _backtest_worker_task(task)
                    if result is not None:
                        all_results.append(result)
                except Exception as e2:
                    logger.warning(f"Sequential task failed: {e2}")
                completed += 1
                if progress_callback:
                    progress_callback(0.40 + 0.60 * (completed / total_tasks))
    else:
        logger.info("Not in main Streamlit process - running backtests sequentially inside worker.")
        for task in tasks:
            try:
                result = _backtest_worker_task(task)
                if result is not None:
                    all_results.append(result)
            except Exception as e:
                logger.warning(f"Sequential task failed: {e}")
            completed += 1
            if progress_callback:
                progress_callback(0.40 + 0.60 * (completed / total_tasks))

    if not all_results:
        if progress_callback: progress_callback(1.0)
        return pd.DataFrame()
    
    # Merge and Save
    combined_df = pd.concat(all_results, ignore_index=True)
    
    output_path = DATA_DIR / output_file
    # Use the module-level single source of truth for schema
    canonical_columns = ML_EXPORT_SCHEMA

    # Fill missing columns in the newly generated dataframe with explicit sentinels
    for col in canonical_columns:
        if col not in combined_df.columns:
            if col in TRAINING_FEATURE_COLUMNS:
                # integer counts -> -1
                if col.endswith('_count') or col == 'feat_games_played':
                    combined_df[col] = -1
                # boolean / one-hot features -> 0
                elif col in ('feat_is_home', 'feat_is_b2b') or col.startswith('feat_market_'):
                    combined_df[col] = 0
                # float features (including TS% and other metrics) -> -1.0
                else:
                    combined_df[col] = -1.0
            else:
                # metadata
                combined_df[col] = ''

    # If an existing file exists, inspect its header BEFORE loading the full file
    existing_df = None
    if output_path.exists():
        import csv, shutil
        try:
            with open(output_path, 'r', encoding='utf-8') as fh:
                existing_header = next(csv.reader(fh), None)
        except Exception as e:
            logger.exception(f"Failed to read existing CSV header: {e}")
            existing_header = None

        # If header doesn't match exactly, archive and start fresh
        if existing_header != ML_EXPORT_SCHEMA:
            ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
            archive_path = output_path.with_name(f"{output_path.stem}.archived.{ts}{output_path.suffix}")
            try:
                shutil.move(str(output_path), str(archive_path))
                logger.warning(f"Archived mismatched ML CSV to {archive_path}; creating fresh file.")
            except Exception as e:
                logger.exception(f"Failed to archive mismatched CSV: {e}")
                raise
            existing_df = None
        else:
            # Safe to load the existing DataFrame and normalize columns
            existing_df = pd.read_csv(output_path)
            for col in canonical_columns:
                if col not in existing_df.columns:
                    if col in TRAINING_FEATURE_COLUMNS:
                        if col.endswith('_count') or col == 'feat_games_played':
                            existing_df[col] = -1
                        elif col in ('feat_is_home', 'feat_is_b2b') or col.startswith('feat_market_'):
                            existing_df[col] = 0
                        else:
                            existing_df[col] = -1.0
                    else:
                        existing_df[col] = ''

    # Merge: existing first then new so that newer records take precedence when dropping duplicates
    if existing_df is not None:
        merged = pd.concat([existing_df, combined_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=['date', 'player', 'market'], keep='last')
        final_df = merged
    else:
        final_df = combined_df

    # Reorder and select final columns
    final_columns = [c for c in canonical_columns if c in final_df.columns]
    final_df = final_df[final_columns]

    # Atomic write to temp file and replace
    tmp_path = output_path.with_suffix(output_path.suffix + '.tmp')
    try:
        final_df.to_csv(tmp_path, index=False)
        os.replace(str(tmp_path), str(output_path))
    except Exception as e:
        logger.exception(f"Failed to write ML training CSV atomically: {e}")
        if tmp_path.exists():
            try: tmp_path.unlink()
            except: pass
        raise

    logger.info(f"✓ ML training data saved to {output_path}")
    if progress_callback: progress_callback(1.0)

    return final_df

def generate_ml_data_streamlit():
    """Streamlit UI wrapper for ML data generation."""
    st.markdown("### 🤖 Generate ML Training Data")
    
    st.info(f"""
    **V20.3 OPTIMIZED**: Only **8 API calls** total, regardless of player count!
    - Seasons: {', '.join(CONFIG.TRAINING_SEASONS)}
    - API Calls: **8 fixed** (4 game logs + 4 team stats per season)
    - Player list extracted from bulk data (0 extra calls)
    - Step 1: Bulk download (~60-90 seconds)
    - Step 2: Parallel backtesting (100% offline, uses all CPU cores)
    - **Temporal Fix**: Each game uses its season's actual DRTG/Pace
    """)
    
    with st.form(key='ml_gen_form'):
        col1, col2 = st.columns(2)
        with col1:
            num_players = st.slider("Number of Players", 10, 600, 50, 10)
            test_days = st.slider("Games per Player", 30, 300, 150, 10,
                                  help="4 seasons = ~328 games max per player")

        with col2:
            markets = st.multiselect(
                "Markets",
                ['PTS', 'REB', 'AST', 'PRA', 'RA', 'PA', 'PR', '3PM', 'STL', 'BLK'],
                default=['PTS', 'REB', 'AST', 'PRA', 'RA', 'PA', 'PR']
            )
            output_file = st.text_input("Output Filename", "ml_training_data.csv")

        submitted = st.form_submit_button("🚀 Generate Training Data", type="primary")
    
    if submitted:
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
        self.injury_manager = InjuryManager()  # V18: For UI injury report
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
        spread: float,
        bankroll: float,
        days_rest: int = 1,
        game_total: float = 0.0
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
            
            # Step 4.5: Extract player's team from game logs (V20.3)
            # Game logs have MATCHUP like "LAL vs. BOS" or "LAL @ BOS"
            player_team_abbrev = None
            player_team_id = None
            if len(df) > 0 and 'MATCHUP' in df.columns:
                latest_matchup = df.iloc[-1]['MATCHUP']  # Most recent game
                # Extract team abbreviation (first 3 chars before space)
                player_team_abbrev = latest_matchup.split()[0] if latest_matchup else None
                # Try to get team ID from static data
                if player_team_abbrev:
                    all_teams = teams.get_teams()
                    team_obj = next((t for t in all_teams if t['abbreviation'] == player_team_abbrev), None)
                    if team_obj:
                        player_team_id = team_obj['id']
            
            # Step 5: Build feature vector (V20.3: with absence features)
            features = self.feature_engineer.build_feature_vector(
                player_id=p_obj['id'],
                player_name=p_obj['full_name'],
                opponent_id=t_obj['id'],
                opponent_abbrev=t_obj['abbreviation'],
                is_home=is_home,
                is_b2b=is_b2b,
                spread=spread,
                df=df,
                stat_col=market,
                line=line,
                lookback=lookback,
                team_stats=self._team_stats,
                avg_def=self._avg_def,
                market=market,
                days_rest=days_rest,
                game_total=game_total,
                player_team_id=player_team_id,  # V20.2: For team pace
                player_team_abbrev=player_team_abbrev,  # V20.3: For injury-based absence
                data_loader=self.data_loader  # V20.3: For PPG lookups
            )
            
            # Step 6: Generate projection
            projection = self.model_engine. generate_projection(features)
            
            # Step 7: V19 CDF-based probability (deterministic, no Monte Carlo)
            simulation = self.simulation_engine.compute_cdf_probability(
                mean=projection.final_projection,
                std=features.std,
                line=line,
                market=market,
                features=features
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
            
            # Step 9: Get ML calibration details (V18)
            ml_details = self.model_engine.get_ml_prediction_details(features)
            
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
                ml_details=ml_details
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
    
    def _calculate_grade(self, ev: float, win_prob: float) -> str:
        """
        V20 EMPIRICAL: Grades are deprecated - return EV sign only.
        
        The ML model's EV is the only meaningful signal.
        Letter grades were belief-encoded heuristics.
        """
        # V20: Simple EV classification for legacy compatibility
        return 'EV+' if ev > 0 else 'EV-'

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
            
        # 4. Ensure all canonical ML feature columns exist on saved bets to prevent schema drift.
        # This stores explicit sentinel values for any missing V20.3 features so older records
        # remain compatible with downstream diagnostics and export code.
        for col in TRAINING_FEATURE_COLUMNS:
            if col not in bet_data:
                bet_data[col] = self._ml_column_sentinel(col)

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
        V20 EMPIRICAL: Categorize result for AUDIT TRAIL only.
        
        These categories are POST-HOC labels for understanding what happened.
        They do NOT affect predictions or decisions.
        """
        abs_margin = abs(margin)
        
        # V20: Simple fixed thresholds for audit labels (not decision-making)
        sweat_threshold = 1.5
        close_threshold = 3.5
        solid_threshold = 7.5
        
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
            # Get decimal odds - check both 'odds_decimal' (new) and 'odds' (legacy)
            odds = bet.get('odds_decimal', bet.get('odds', 1.91))
            # Handle American format if needed
            if odds < 0 or odds >= 100:
                odds = american_to_decimal(odds)
            if bet.get('result') == 'Win':
                total_profit += bet.get('stake', 0) * (odds - 1)
            elif bet.get('result') == 'Loss':
                total_profit -= bet.get('stake', 0)
        
        bets_with_clv = [b for b in bets if b.get('clv') is not None and b.get('clv') != 0]
        clv_count = len(bets_with_clv)
        avg_clv = sum(b['clv'] for b in bets_with_clv) / clv_count if clv_count > 0 else 0
        positive_clv = len([b for b in bets_with_clv if b['clv'] > 0])
        clv_positive_rate = positive_clv / clv_count if clv_count > 0 else 0
        
        # V18: CLV-Win Correlation Analysis (CLV is best predictor of long-term edge)
        clv_win_correlation = 0.0
        clv_positive_wins = 0
        clv_positive_decided = 0
        clv_negative_wins = 0
        clv_negative_decided = 0
        
        for b in bets_with_clv:
            if b.get('result') not in ['Win', 'Loss']:
                continue
            is_win = b.get('result') == 'Win'
            clv = b.get('clv', 0)
            
            if clv > 0:
                clv_positive_decided += 1
                if is_win:
                    clv_positive_wins += 1
            elif clv < 0:
                clv_negative_decided += 1
                if is_win:
                    clv_negative_wins += 1
        
        clv_positive_win_rate = clv_positive_wins / clv_positive_decided if clv_positive_decided > 0 else 0
        clv_negative_win_rate = clv_negative_wins / clv_negative_decided if clv_negative_decided > 0 else 0
        
        # CLV Edge: Difference in win rate between +CLV and -CLV bets
        clv_edge = clv_positive_win_rate - clv_negative_win_rate
        
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
        
        # V18: Grade Performance Analysis (data-driven grade evaluation)
        grade_stats = {}
        for grade in ['A', 'B', 'C', 'D', 'F']:
            grade_bets = [b for b in bets if b.get('grade') == grade and b.get('result') in ['Win', 'Loss']]
            grade_wins = len([b for b in grade_bets if b.get('result') == 'Win'])
            grade_total = len(grade_bets)
            if grade_total > 0:
                grade_win_rate = grade_wins / grade_total
                grade_profit = sum(0.91 if b.get('result') == 'Win' else -1.0 for b in grade_bets)
                grade_roi = grade_profit / grade_total
                grade_stats[grade] = {
                    'count': grade_total,
                    'wins': grade_wins,
                    'win_rate': grade_win_rate,
                    'roi': grade_roi
                }
        
        return {
            'wins': wins, 'losses': losses, 'pushes': pushes, 'pending': pending,
            'total_decided': total_decided, 'win_rate': win_rate, 'total_profit': total_profit,
            'clv_count': clv_count, 'avg_clv': avg_clv, 'clv_positive_rate': clv_positive_rate,
            # V18: New CLV metrics
            'clv_positive_win_rate': clv_positive_win_rate,
            'clv_negative_win_rate': clv_negative_win_rate,
            'clv_positive_decided': clv_positive_decided,
            'clv_negative_decided': clv_negative_decided,
            'clv_edge': clv_edge,
            'margin_count': margin_count, 'avg_margin': avg_margin,
            'quality_counts': quality_counts,
            'bad_beat_rate': bad_beat_rate, 'bad_read_rate': bad_read_rate,
            'sweat_win_rate': sweat_win_rate, 'solid_win_rate': solid_wins / wins if wins > 0 else 0,
            # V18: Grade analysis
            'grade_stats': grade_stats
        }

    def optimize_grade_thresholds(self) -> Dict[str, Any]:
        """
        Analyze historical bets to find optimal EV thresholds for grade assignment.
        Uses actual win rates and ROI to suggest data-driven thresholds.
        
        Returns: Dict with current thresholds, optimal thresholds, and analysis.
        """
        bets = self.get_bets()
        decided = [b for b in bets if b.get('result') in ['Win', 'Loss']]
        
        if len(decided) < 30:
            return {
                'status': 'insufficient_data',
                'message': f'Need at least 30 decided bets (have {len(decided)})',
                'current_thresholds': {'A': 0.05, 'B': 0.02, 'C': 0.0}
            }
        
        # Build EV -> outcome dataset
        ev_outcomes = []
        for bet in decided:
            ev = bet.get('ev', bet.get('predicted_ev', 0))
            won = bet.get('result') == 'Win'
            odds = bet.get('odds', 1.91)
            profit = (odds - 1) if won else -1
            
            ev_outcomes.append({
                'ev': ev,
                'won': won,
                'profit': profit,
                'odds': odds
            })
        
        df = pd.DataFrame(ev_outcomes)
        
        # Find optimal thresholds using grid search
        ev_bins = np.arange(-0.10, 0.20, 0.01)
        results = []
        
        for threshold in ev_bins:
            above = df[df['ev'] >= threshold]
            if len(above) >= 5:  # Minimum sample
                win_rate = above['won'].mean()
                roi = above['profit'].sum() / len(above) * 100
                results.append({
                    'threshold': round(threshold, 3),
                    'count': len(above),
                    'win_rate': round(win_rate, 3),
                    'roi': round(roi, 2)
                })
        
        # Current thresholds
        current = {'A': 0.05, 'B': 0.02, 'C': 0.0}
        
        # Find optimal thresholds (maximize ROI while keeping sample size reasonable)
        optimal = {'A': 0.05, 'B': 0.02, 'C': 0.0}
        
        # Grade A: Find EV threshold where ROI > 10% AND win rate > 55%
        for r in sorted(results, key=lambda x: x['threshold'], reverse=True):
            if r['roi'] > 10 and r['win_rate'] > 0.55 and r['count'] >= 10:
                optimal['A'] = r['threshold']
                break
        
        # Grade B: Find EV threshold where ROI > 0% AND win rate > 52%
        for r in sorted(results, key=lambda x: x['threshold'], reverse=True):
            if r['roi'] > 0 and r['win_rate'] > 0.52 and r['count'] >= 15:
                if r['threshold'] < optimal['A']:
                    optimal['B'] = r['threshold']
                    break
        
        # Grade C: Find EV threshold where ROI > -5%
        for r in sorted(results, key=lambda x: x['threshold'], reverse=True):
            if r['roi'] > -5 and r['count'] >= 20:
                if r['threshold'] < optimal['B']:
                    optimal['C'] = r['threshold']
                    break
        
        # Calculate performance at each threshold level
        grade_performance = {}
        for grade, thresh in [('A', optimal['A']), ('B', optimal['B']), ('C', optimal['C'])]:
            above = df[df['ev'] >= thresh]
            if grade == 'A':
                filtered = above
            elif grade == 'B':
                filtered = df[(df['ev'] >= thresh) & (df['ev'] < optimal['A'])]
            else:
                filtered = df[(df['ev'] >= thresh) & (df['ev'] < optimal['B'])]
            
            if len(filtered) > 0:
                grade_performance[grade] = {
                    'count': len(filtered),
                    'win_rate': round(filtered['won'].mean(), 3),
                    'roi': round(filtered['profit'].sum() / len(filtered) * 100, 2)
                }
        
        return {
            'status': 'success',
            'total_bets': len(decided),
            'current_thresholds': current,
            'optimal_thresholds': optimal,
            'threshold_changed': current != optimal,
            'grade_performance': grade_performance,
            'ev_analysis': results[:10],  # Top 10 thresholds
            'recommendation': self._generate_threshold_recommendation(current, optimal, grade_performance)
        }
    
    def _generate_threshold_recommendation(self, current: Dict, optimal: Dict, performance: Dict) -> str:
        """Generate human-readable recommendation for threshold changes."""
        changes = []
        
        for grade in ['A', 'B', 'C']:
            if abs(current[grade] - optimal[grade]) > 0.005:
                direction = "raise" if optimal[grade] > current[grade] else "lower"
                changes.append(f"{direction} Grade {grade} threshold from {current[grade]*100:.1f}% to {optimal[grade]*100:.1f}%")
        
        if not changes:
            return "Current thresholds are well-calibrated. No changes needed."
        
        perf_notes = []
        for grade, stats in performance.items():
            if stats['roi'] < 0:
                perf_notes.append(f"Grade {grade} bets have negative ROI ({stats['roi']:.1f}%)")
            elif stats['roi'] > 15:
                perf_notes.append(f"Grade {grade} bets are highly profitable ({stats['roi']:.1f}% ROI)")
        
        return "Suggested changes: " + "; ".join(changes) + (". " + " ".join(perf_notes) if perf_notes else "")

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

    def _map_bet_to_csv_row(self, bet: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Map a bet dict to a CSV row that strictly follows `ML_EXPORT_SCHEMA`.
        Returns None if mandatory V20.2 core features are missing (no fabrication).
        """
        # Required core features (V20.2) - must be present
        required_core = ['feat_opponent_pace', 'feat_team_pace', 'feat_trend_5g', 'feat_home_avg', 'feat_away_avg']
        missing = [f for f in required_core if bet.get(f) is None]
        if missing:
            # Caller will log; we simply indicate missing required fields by returning None
            return None

        # Parse date robustly
        date_val = bet.get('date', None)
        date_iso = None
        if isinstance(date_val, (datetime, pd.Timestamp)):
            date_iso = date_val.date().isoformat()
        else:
            try:
                parsed = pd.to_datetime(date_val, errors='coerce')
                if not pd.isna(parsed):
                    date_iso = parsed.date().isoformat()
                else:
                    date_iso = str(date_val).split(' ')[0]
            except Exception:
                date_iso = str(date_val).split(' ')[0]

        # Compute hit flag
        result = bet.get('result', '')
        hit_flag = 1 if result == 'Win' else 0

        # Build the canonical row
        row: Dict[str, Any] = {
            'date': date_iso,
            'player': bet.get('player', ''),
            'opponent': bet.get('opponent', ''),
            'market': bet.get('market', ''),
            'position': bet.get('feat_position', bet.get('position', '')),
            'line': float(bet.get('line', 0) or 0),
            'predicted_side': bet.get('predicted_side', bet.get('side', '')),
            'predicted_prob': float(bet.get('predicted_prob', bet.get('prob', bet.get('win_prob', 0))) or 0),
            'predicted_ev': float(bet.get('predicted_ev', bet.get('ev', 0) or 0)),
            'projected_value': float(bet.get('projected_value', bet.get('proj', 0) or 0)),
            'result': result,
            'hit': int(hit_flag),
            'actual_value': float(bet.get('actual_value', 0) or 0),
            'margin': float(bet.get('margin', 0) or 0),
            'margin_pct': float(bet.get('margin_pct', 0) or 0),
            'result_quality': bet.get('result_quality', 'legacy'),
            'tag': bet.get('tag', 'legacy'),
            'grade': self._calculate_grade(bet.get('ev', bet.get('predicted_ev', 0)), bet.get('win_prob', bet.get('predicted_prob', 0.5))),

            # V20.3 Required Features (canonical names)
            'feat_avg_minutes': float(bet.get('feat_avg_minutes', bet.get('avg_minutes', 0) or 0)),
            'feat_ema': float(bet.get('feat_ema', 0) or 0),
            'feat_std': float(bet.get('feat_std', 0) or 0),
            'feat_opponent_drtg_season': float(bet.get('feat_opponent_drtg_season', bet.get('feat_opp_drtg_season', 0) or 0)),
            'feat_line': float(bet.get('line', 0) or 0),
            'feat_spread': float(bet.get('feat_spread', 0) or 0),
            'feat_game_total': float(bet.get('feat_game_total', 0) or 0),
            'feat_days_rest': int(bet.get('feat_days_rest', 0) or 0),
            'feat_is_home': 1 if bet.get('feat_is_home', False) else 0,
            'feat_is_b2b': 1 if bet.get('feat_is_b2b', False) else 0,
            'feat_games_played': int(bet.get('feat_games_played', 0) or 0),

            # V20.2: Pace and trend features
            'feat_opponent_pace': float(bet.get('feat_opponent_pace')),
            'feat_team_pace': float(bet.get('feat_team_pace')),
            'feat_trend_5g': float(bet.get('feat_trend_5g')),
            'feat_home_avg': float(bet.get('feat_home_avg')),
            'feat_away_avg': float(bet.get('feat_away_avg')),

            # V20.3: True-Shooting features
            'feat_ts_pct': float(bet.get('feat_ts_pct', -1.0)),
            'feat_ts_pct_delta': float(bet.get('feat_ts_pct_delta', -1.0)),

            # V20.3 NEW: Absence-aware features
            'feat_team_out_ppg': float(bet.get('feat_team_out_ppg', -1.0)),
            'feat_team_out_count': int(bet.get('feat_team_out_count', -1) or -1),
            'feat_opp_out_ppg': float(bet.get('feat_opp_out_ppg', -1.0)),
            'feat_opp_out_count': int(bet.get('feat_opp_out_count', -1) or -1),

            # V20.3 NEW: Behavior & Risk features
            'feat_min_volatility': float(bet.get('feat_min_volatility', -1.0)),
            'feat_foul_rate': float(bet.get('feat_foul_rate', -1.0)),
            'feat_cv': float(bet.get('feat_cv', -1.0)),
        }

        # Market identity features
        row.update(get_market_group_features(bet.get('market', 'PTS')))

        return row

    def _ml_column_sentinel(self, col: str):
        """Return an appropriate sentinel for a given ML export column.
        - Metadata strings -> ''
        - Numeric metadata -> -1.0
        - Integer counters -> -1
        - Float features -> -1.0
        - One-hot / boolean features -> 0
        """
        # Numeric metadata
        numeric_meta = {'line', 'predicted_prob', 'predicted_ev', 'projected_value', 'actual_value', 'margin', 'margin_pct'}
        if col in ML_EXPORT_METADATA:
            if col in numeric_meta:
                return -1.0
            if col == 'hit':
                return -1
            return ''

        # Feature columns
        if col.startswith('feat_'):
            if col.endswith('_count'):
                return -1
            if col in ('feat_is_home', 'feat_is_b2b', 'feat_market_scoring', 'feat_market_counting', 'feat_market_combo', 'feat_market_rare'):
                return 0
            return -1.0

        # Fallback
        return ''

    def _migrate_ml_csv_schema_if_needed(self, output_path: Path) -> None:
        """Migrate an existing ML CSV in-place to include any missing columns from
        `ML_EXPORT_SCHEMA` using sentinel backfilling. The migration writes atomically
        and preserves existing records and any extra legacy columns (they are appended).
        """
        import os
        try:
            if not output_path.exists():
                return

            # Read existing CSV (skip corrupted lines)
            try:
                df = pd.read_csv(output_path, on_bad_lines='skip')
            except Exception as e:
                logger.exception(f"Failed to read existing ML CSV for migration: {e}")
                raise

            existing_cols = df.columns.tolist()

            # Quick check: if canonical columns already exist in canonical order, nothing to do
            if existing_cols[:len(ML_EXPORT_SCHEMA)] == ML_EXPORT_SCHEMA:
                logger.debug("ML CSV already matches canonical schema order. No migration needed.")
                return

            missing = [c for c in ML_EXPORT_SCHEMA if c not in existing_cols]
            if missing:
                logger.info(f"Migrating existing ML CSV: adding missing canonical columns: {missing}")
            else:
                logger.info("Existing ML CSV contains canonical columns but ordering differs; reordering to canonical order.")

            # Add missing canonical columns using appropriate sentinels
            for col in missing:
                df[col] = self._ml_column_sentinel(col)

            # Coerce canonical columns to sentinel-aware dtypes and fill NaNs where appropriate
            for col in ML_EXPORT_SCHEMA:
                if col in df.columns:
                    s = self._ml_column_sentinel(col)
                    if isinstance(s, float):
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(s)
                    elif isinstance(s, int):
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(s).astype(int)
                    else:
                        df[col] = df[col].fillna(s).astype(str)

            # Reorder: canonical columns first, then preserve any legacy extras
            final_cols = ML_EXPORT_SCHEMA + [c for c in df.columns if c not in ML_EXPORT_SCHEMA]
            df = df[final_cols]

            # Atomic write to temp file and replace
            tmp_path = output_path.with_suffix(output_path.suffix + '.migrate.tmp')
            df.to_csv(tmp_path, index=False)
            os.replace(str(tmp_path), str(output_path))

            logger.info(f"Migrated ML CSV in-place to canonical schema at {output_path}")
        except Exception as e:
            logger.exception(f"Failed to migrate ML CSV schema: {e}")
            raise

    def export_bets_to_training_csv(self, output_file: str = "ml_training_data.csv") -> Tuple[int, str]:
        """
        Export tracked bets to ML training CSV with in-place schema migration and sentinel backfilling.

        Behavior changes (Schema Migration):
        - If an existing CSV header does not match `ML_EXPORT_SCHEMA`, do not archive or delete it.
        - Instead, migrate the file in-place by adding missing canonical columns with sentinel values
          and reordering so canonical columns appear first. Preserve legacy extra columns after canonical.
        - After migration (or if the file did not exist), append new bets using the canonical schema.
        """
        import csv
        import os

        bets = self.get_bets()
        output_path = DATA_DIR / output_file

        exported_count = 0
        bets_modified = False

        # If the file exists, migrate it in-place to ensure canonical schema is present first
        if output_path.exists():
            try:
                self._migrate_ml_csv_schema_if_needed(output_path)
            except Exception:
                logger.exception("Schema migration failed; aborting export to avoid corruption.")
                raise

        # If the file does not exist after migration, we'll need to write a canonical header
        need_write_header = not output_path.exists()

        with open(output_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=ML_EXPORT_SCHEMA, extrasaction='ignore')
            if need_write_header:
                writer.writeheader()

            for bet in bets:
                if not (bet.get('result') in ['Win', 'Loss'] and bet.get('actual_value') is not None and not bet.get('exported_to_csv', False)):
                    continue

                row = self._map_bet_to_csv_row(bet)
                if row is None:
                    # Missing required core features - log and skip
                    logger.warning(f"Skipping bet {bet.get('id')} - missing required V20.2 core features.")
                    continue

                # Ensure row contains all canonical columns; fill missing keys with sentinels
                for col in ML_EXPORT_SCHEMA:
                    if col not in row:
                        row[col] = self._ml_column_sentinel(col)

                # All good - write row and mark exported
                writer.writerow(row)
                bet['exported_to_csv'] = True
                bets_modified = True
                exported_count += 1

        if bets_modified:
            self._save(bets)

        logger.info(f"Exported {exported_count} bets to {output_path}")
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

def render_data_quality_card(result: AnalysisResult):
    """Render data quality indicator with fallback warnings."""
    features = result.features
    dq = features.data_quality
    
    # Color code by grade
    grade_colors = {'A': 'green', 'B': 'green', 'C': 'orange', 'D': 'red', 'F': 'red'}
    color = grade_colors.get(dq.grade, 'gray')
    
    # Always show data sources panel for transparency
    with st.expander(f"📡 Data Quality: :{color}[{dq.grade}] ({dq.score:.0f}/100)", expanded=False):
        # Data Sources Status
        st.markdown("**📊 Data Sources:**")
        col1, col2 = st.columns(2)
        
        with col1:
            # Team Stats
            if dq.missing_team_stats:
                st.markdown("❌ **Team Stats:** Unavailable")
            elif dq.team_stats_age_hours > 24:
                st.markdown(f"⚠️ **Team Stats:** {dq.team_stats_age_hours:.0f}h old")
            else:
                st.markdown("✅ **Team Stats:** Fresh (<24h)")
        
        with col2:
            # Sample Size
            if features.games_played >= 15:
                st.markdown(f"✅ **Sample Size:** {features.games_played} games")
            elif features.games_played >= 10:
                st.markdown(f"⚠️ **Sample Size:** {features.games_played} games")
            else:
                st.markdown(f"❌ **Sample Size:** {features.games_played} games (small)")
        
        # Show warnings if any
        if dq.has_issues:
            st.markdown("---")
            st.caption("**⚠️ Fallbacks Used:**")
            
            flags = []
            if dq.used_default_pace:
                flags.append("Default pace (100.0)")
            if dq.used_default_def_rating:
                flags.append("Default def rating (115.0)")
            if dq.used_fallback_std:
                flags.append("Estimated std deviation")
            if dq.used_fallback_minutes:
                flags.append("Default minutes (30)")
            if dq.used_fallback_split:
                flags.append("Neutral home/away split")
            
            if flags:
                st.write(", ".join(flags))
            
            for warning in dq.warnings:
                st.markdown(f"• {warning}")


def render_ticket_card(result: 'AnalysisResult', bankroll: float, bankroll_enabled: bool = True):
    """
    V19 REFACTOR: Clean statistical output card.
    
    Standard Output Only:
    - Expected Value (EV)
    - Predicted Mean
    - Uncertainty (Std Dev)
    - Win Probability
    
    REMOVED: Grade badges, narratives, rollover scores, ML badges
    """
    decision = result.decision
    projection = result.projection
    features = result.features

    # V20 EMPIRICAL: Color based on EV sign only (math, not belief)
    ev = decision.expected_value
    if ev > 0:
        bg_color = 'rgba(0,200,83,0.15)'
        border_color = '#00c853'
    else:
        bg_color = 'rgba(244,67,54,0.15)'
        border_color = '#f44336'

    is_over = decision.recommended_side == 'OVER'
    side_color = '#00c853' if is_over else '#f44336'
    side_icon = '📈' if is_over else '📉'
    
    # Get predicted values
    predicted_mean = decision.predicted_mean
    predicted_std = decision.predicted_std
    prob = decision.probability
    
    # Confidence interval
    ci_low, ci_high = projection.confidence_interval

    # Build clean HTML card
    ticket_html = f"""<div style="background: linear-gradient(135deg, {bg_color} 0%, rgba(30,30,30,0.9) 100%); border-left: 5px solid {border_color}; border-radius: 12px; padding: 16px; margin: 10px 0; font-family: sans-serif; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
<div>
<div style="font-size: 10px; color: #888; text-transform: uppercase; letter-spacing: 1px;">NBA Prop Analysis</div>
<div style="font-size: 20px; font-weight: 700; color: white;">{result.player_name}</div>
<div style="color: #aaa; font-size: 12px;">vs {result.opponent_name}</div>
</div>
<div style="text-align: right;">
<div style="font-size: 24px; font-weight: 800; color: {'#00c853' if ev > 0 else '#f44336'};">{ev:+.1%}</div>
<div style="font-size: 10px; color: #888;">Expected Value</div>
</div>
</div>
<div style="background: rgba(255,255,255,0.05); border-radius: 8px; padding: 12px; margin: 10px 0; text-align: center;">
<div style="font-size: 12px; color: #888;">{result.market}</div>
<div style="font-size: 28px; font-weight: 800; color: {side_color};">{side_icon} {decision.recommended_side} {result.line}</div>
<div style="font-size: 12px; color: #aaa;">Odds: {result.odds}</div>
</div>
<div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 10px; text-align: center; margin: 15px 0;">
<div style="background: rgba(255,255,255,0.05); border-radius: 8px; padding: 10px;">
<div style="font-size: 18px; font-weight: 700; color: white;">{prob:.1%}</div>
<div style="font-size: 10px; color: #888;">Win Prob</div>
</div>
<div style="background: rgba(255,255,255,0.05); border-radius: 8px; padding: 10px;">
<div style="font-size: 18px; font-weight: 700; color: white;">{predicted_mean:.1f}</div>
<div style="font-size: 10px; color: #888;">Predicted</div>
</div>
<div style="background: rgba(255,255,255,0.05); border-radius: 8px; padding: 10px;">
<div style="font-size: 18px; font-weight: 700; color: #aaa;">±{predicted_std:.1f}</div>
<div style="font-size: 10px; color: #888;">Std Dev</div>
</div>
<div style="background: rgba(255,255,255,0.05); border-radius: 8px; padding: 10px;">
<div style="font-size: 18px; font-weight: 700; color: white;">{"₱" + f"{decision.kelly_stake:.0f}" if bankroll_enabled else "—"}</div>
<div style="font-size: 10px; color: #888;">{"Stake" if bankroll_enabled else "N/A"}</div>
</div>
</div>
<div style="text-align: center; font-size: 11px; color: #666; border-top: 1px solid #333; padding-top: 10px;">
90% CI: [{ci_low:.1f}, {ci_high:.1f}] | Games: {features.games_played}
</div>
</div>"""

    st.markdown(ticket_html, unsafe_allow_html=True)
    if projection.ml_prob is not None:
        with st.expander("🧠 ML Decoder", expanded=False):
            render_ml_decoder(result)


def render_ml_decoder(result: AnalysisResult):
    """Render the ML signal decoder panel with V18 calibration info."""
    features = result.features
    projection = result.projection
    # Safe getter for ml_details (backwards compatibility with cached results)
    ml_details = getattr(result, 'ml_details', None)
    
    # ROW 0: ML MODEL INFO (V18 - Calibration Details)
    if ml_details and ml_details.get('has_model'):
        st.caption("🤖 **ML Model Info**")
        m1, m2, m3, m4 = st.columns(4)
        
        model_group = ml_details.get('model_group', 'universal')
        group_emoji = {'scoring': '🏀', 'counting': '📊', 'combo': '🔗', 'rare': '💎', 'universal': '🌐'}
        m1.metric("Model", f"{group_emoji.get(model_group, '🤖')} {model_group.title()}")
        
        raw_prob = ml_details.get('raw_prob')
        if raw_prob is not None:
            m2.metric("Raw ML", f"{raw_prob:.1%}")
        
        calibrated_prob = ml_details.get('calibrated_prob')
        if calibrated_prob is not None:
            delta = ml_details.get('calibration_delta', 0)
            delta_str = f"{delta:+.1%}" if delta else None
            m3.metric("Calibrated", f"{calibrated_prob:.1%}", delta_str)
            
        has_calib = ml_details.get('has_calibrator', False)
        m4.metric("Calibrator", "✅ Active" if has_calib else "❌ None")
        
        st.divider()
    
    # ROW 1: KEY STATS
    st.caption("📊 **Key Stats**")
    r1, r2, r3 = st.columns(3)
    
    with r1:  # Volatility (CV from std/ema)
        cv = (features.std / features.ema * 100) if features.ema > 0 else 30
        st.metric("Volatility (CV)", f"{cv:.0f}%")
    
    with r2:  # Days rest (V20: raw observable)
        days_rest = features.days_rest
        st.metric("Days Rest", f"{days_rest}")
    
    with r3:  # Spread (raw observable)
        st.metric("Spread", f"{features.spread:.1f}")
    
    # ROW 2: KEY FACTORS (V20 EMPIRICAL)
    st.caption("📊 **Key Factors**")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Avg Min", f"{features.avg_minutes:.1f}")
    e2.metric("Opp DRTG", f"{features.opponent_drtg_season:.1f}")
    e3.metric("Games", f"{features.games_played}")
    e4.metric("Game Total", f"{features.game_total:.1f}")


def render_recommendation_card(result: AnalysisResult, bankroll: float):
    """
    Render clean, mobile-friendly recommendation card.

    V19: Removed grades - uses EV-based coloring.
    """
    decision = result.decision
    projection = result.projection
    features = result.features
    simulation = result.simulation
    
    # V20 EMPIRICAL: Color based on EV sign only
    ev = decision.expected_value
    if ev > 0:
        ev_color = 'green'
    else:
        ev_color = 'red'
    
    # --- 1. HEADER ---
    
    # V20 EMPIRICAL: Display ML recommendation directly
    st.markdown(f"### {decision.recommended_side} {result.line} — :{ev_color}[EV: {ev:+.1%}]")
    
    # --- 2. KEY METRICS ---
    
    if projection.ml_prob is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Win Prob", f"{decision.probability:.0%}")
        c2.metric("EV", f"{decision.expected_value:+.1%}")
        c3.metric("Stake", f"₱{decision.kelly_stake:.0f}", f"{decision.kelly_fraction:.1%}")
        
        # V20 EMPIRICAL: Display raw probability - no categorical labels
        c4.metric("ML P(Over)", f"{projection.ml_prob:.0%}")
        
    else:
        # Fallback if no ML model loaded
        c1, c2, c3 = st.columns(3)
        c1.metric("Win Prob", f"{decision.probability:.0%}")
        c2.metric("EV", f"{decision.expected_value:+.1%}")
        c3.metric("Stake", f"₱{decision.kelly_stake:.0f}", f"{decision.kelly_fraction:.1%}")
    
    # --- 3. V19 SIGNAL DECODER (STREAMLINED) ---
    
    with st.expander("🧠 ML Signal Decoder", expanded=False):
        # V20.2 EMPIRICAL: Display raw observables only - NO categorical labels
        # The ML model interprets these values; humans should not add labels
        
        st.caption("📊 **Raw Observables (Model Inputs)**")
        r1, r2, r3 = st.columns(3)
        
        with r1:
            st.metric("Spread", f"{features.spread:+.1f}")
            
        with r2:
            st.metric("Days Rest", f"{features.days_rest}")
            
        with r3:
            cv = (features.std / features.ema * 100) if features.ema > 0 else 0
            st.metric("CV %", f"{cv:.0f}%")

        st.divider()

        # V20: Raw numeric values only - no categorical interpretation
        st.caption("📈 **Feature Values**")
        d1, d2, d3 = st.columns(3)
        
        with d1:
            st.metric("Avg Minutes", f"{features.avg_minutes:.1f}")
            
        with d2:
            st.metric("Opp DRTG", f"{features.opponent_drtg_season:.1f}")
            
        with d3:
            st.metric("Game Total", f"{features.game_total:.0f}")
        
        # V20.2 NEW: Pace and trend features
        st.divider()
        st.caption("🆕 **V20.2 Features**")
        p1, p2, p3 = st.columns(3)
        
        with p1:
            st.metric("Opp Pace", f"{features.opponent_pace:.1f}")
            
        with p2:
            st.metric("Team Pace", f"{features.team_pace:.1f}")
            
        with p3:
            trend_color = "🔺" if features.trend_5g > 0 else "🔻" if features.trend_5g < 0 else "➡️"
            st.metric("Trend 5G", f"{trend_color} {features.trend_5g:+.2f}")
        
        h1, h2 = st.columns(2)
        with h1:
            st.metric("Home Avg", f"{features.home_avg:.1f}")
        with h2:
            st.metric("Away Avg", f"{features.away_avg:.1f}")

    # --- 4. FOOTER ---
    
    render_data_quality_card(result)


def render_distribution_chart(result: AnalysisResult):
    """
    V20 EMPIRICAL: Render probability distribution chart.
    
    Uses ML model's predicted mean and std directly - NO heuristic adjustments.
    The chart is for visualization only; actual probabilities come from CDF.
    """
    simulation = result.simulation
    projection = result.projection
    features = result.features

    # V20: Use ML model's predictions directly - NO heuristic adjustments
    mean = projection.final_projection
    std = features.std if features.std > 0 else 1.0  # Use actual std, fallback only if zero
    
    # Generate 10,000 samples for smooth histogram
    viz_samples = np.random.normal(mean, std, 10000)
    viz_samples = np.maximum(viz_samples, 0)  # Non-negative (physical constraint)

    fig, ax = plt.subplots(figsize=(10, 2.5))
    n, bins, patches = ax.hist(viz_samples, bins=50, color='skyblue', alpha=0.7, density=True)
    
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
    ax.set_title('Probability Distribution (CDF-based)', fontsize=10, pad=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_backtest_tab(orchestrator: PredictionOrchestrator):
    """Render compact backtesting tab."""
    st.markdown("### 📊 Model Backtest")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        all_players = players.get_players()
        player_names = sorted([p['full_name'] for p in all_players if p. get('is_active', True)])
        bt_player = st.selectbox("Player", player_names, index=None, placeholder="Search...", key="bt_player")
    with col2:
        bt_market = st.selectbox("Market", ["PTS", "REB", "AST", "PRA", "RA"], key="bt_market")
    with col3:
        bt_days = st.number_input("Games", 10, 50, 30, key="bt_days")
    
    with st.expander("⚙️ Advanced", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            bt_lookback = st.slider("Lookback", 5, 20, 15, key="bt_lookback")
        with c2:
            bt_offset = st.number_input("Line Offset", -10.0, 10.0, 0.0, 0.5, key="bt_offset")
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
                ["PTS", "REB", "AST", "PRA", "RA"], 
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
                                # Use same lookback as Analyze tab (15 games default)
                                lookback = 15
                                recent = df.tail(lookback)
                                
                                # Calculate EMA same way as Analyze tab
                                # Uses span = lookback window size, not hardcoded
                                ema = recent[item['market']].ewm(span=len(recent), adjust=False).mean().iloc[-1]
                                
                                # Calculate L5 hit rate (same as Analyze tab)
                                # Use tail(5) for MOST RECENT 5 games
                                l5_games = df.tail(5)
                                l5_hit_rate = (l5_games[item['market']] > current_line).mean()
                                
                                # Hot/Cold based on L5 hit rate (matches Analyze tab)
                                if l5_hit_rate >= 0.80:
                                    form_icon, form_color = "🔥", "green"
                                elif l5_hit_rate >= 0.60:
                                    form_icon, form_color = "✅", "green"
                                elif l5_hit_rate >= 0.40:
                                    form_icon, form_color = "😐", "orange"
                                else:
                                    form_icon, form_color = "❄️", "red"
                                
                                # Offset shows projection vs line (for OVER/UNDER signal)
                                offset = ema - current_line
                                offset_color = "green" if offset > 0 else "red"
                                offset_arrow = "↑" if offset > 0 else "↓"
                                
                                # Display BOTH: EMA projection AND L5 consistency
                                st.markdown(f"**EMA:** {ema:.1f} :{offset_color}[{offset:+.1f}{offset_arrow}]")
                                st.markdown(f"**L5:** :{form_color}[{l5_hit_rate:.0%}] {form_icon}")
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

def render_parlay_tab(parlay_tracker: ParlayTracker, bankroll: float, bankroll_enabled: bool = True):
    """Render compact parlay tab."""
    st.markdown("### 🎲 Parlay Builder")
    
    if not bankroll_enabled:
        st.info("📊 **Data Collection Mode** — Parlay stakes disabled. Toggle Bankroll Mode in sidebar to enable.")
    
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
            if bankroll_enabled and bankroll > 0:
                parlay_stake = st.number_input("Stake", 10.0, float(bankroll), 50.0, 10.0, key="parlay_stake")
            else:
                parlay_stake = 0.0
                st.caption("📊 Stake tracking disabled")
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
                    
                    # --- Quality Units: Offset Rule (V16) ---
                    # Rule 1 (Offset) alone gives +2.8% ROI - this is the key filter
                    if 'projected_value' in csv_df.columns:
                        # For OVER: projection > line. For UNDER: projection < line
                        def check_offset(row):
                            if row['predicted_side'] == 'OVER':
                                return row['projected_value'] > row['line']
                            else:
                                return row['projected_value'] < row['line']
                        csv_df['passes_offset'] = csv_df.apply(check_offset, axis=1)
                        quality_df = csv_df[csv_df['passes_offset']]
                    else:
                        quality_df = csv_df[csv_df['predicted_prob'] > 0.55] if 'predicted_prob' in csv_df.columns else pd.DataFrame()
                    
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
                    st.markdown("**🎯 Quality Units (Offset Rule - V16)**")
                    st.caption("Offset Rule: Only bet when projection agrees with side (53.8% win rate, +2.8% ROI)")
                    q_c1, q_c2, q_c3, q_c4 = st.columns(4)
                    q_c1.metric("Filtered Bets", f"{quality_total:,}")
                    q_c2.metric("Win Rate", f"{quality_win_rate:.1%}")
                    if quality_net_units >= 0:
                        q_c3.metric("Quality Units", f"+{quality_net_units:.1f}u", delta="Offset rule applied")
                    else:
                        q_c3.metric("Quality Units", f"{quality_net_units:.1f}u", delta="Offset rule applied", delta_color="inverse")
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
                # Get decimal odds - check both 'odds_decimal' (new) and 'odds' (legacy)
                odds = bet.get('odds_decimal', bet.get('odds', 1.91))
                # Handle edge case where odds might be in American format (negative or >= 100)
                if odds < 0 or odds >= 100:
                    odds = american_to_decimal(odds)
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
                # Get decimal odds - check both 'odds_decimal' (new) and 'odds' (legacy)
                odds = bet.get('odds_decimal', bet.get('odds', 1.91))
                # Handle American format if needed
                if odds < 0 or odds >= 100:
                    odds = american_to_decimal(odds)
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
            
            # Schema health diagnostics (compare live df to canonical ML_EXPORT_SCHEMA)
            canonical_schema = ML_EXPORT_SCHEMA
            present_schema_cols = [c for c in canonical_schema if c in training_df.columns]
            missing_features = list(set(canonical_schema) - set(training_df.columns))
            schema_completeness = (len(present_schema_cols) / len(canonical_schema)) * 100 if canonical_schema else 0

            # Show schema completeness as a top-level metric
            sq2.metric("Schema Completeness", f"{schema_completeness:.0f}% ({len(present_schema_cols)}/{len(canonical_schema)})")

            # Date range
            if 'timestamp' in training_df.columns:
                try:
                    dates = pd.to_datetime(training_df['timestamp'])
                    sq3.metric("Date Range", f"{dates.min().strftime('%m/%d')} - {dates.max().strftime('%m/%d')}")
                except:
                    sq3.metric("Date Range", "N/A")
            else:
                sq3.metric("Date Range", "N/A")

            # If there are missing canonical columns, surface a clear warning & explanation
            if missing_features:
                # Prefer listing only canonical feature names to avoid overwhelming the user
                missing_list = ', '.join(sorted(missing_features))
                st.warning(f"Missing {len(missing_features)} canonical columns: {missing_list}")
                st.caption("These features are present in the CSV generator but missing from historical tracked bets.")

            # Avg margin if available
            if 'margin' in training_df.columns:
                avg_margin = training_df['margin'].mean()
                # If schema is incomplete, display avg margin but add a small note
                if missing_features:
                    sq3.metric("Avg Margin", f"{avg_margin:+.1f}", delta="(Schema incomplete)")
                else:
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
            canonical_feat_cols = [c for c in ML_EXPORT_SCHEMA if c.startswith('feat_')]
            present_feat_cols = [c for c in training_df.columns if c.startswith('feat_')]
            missing_feat_cols = [c for c in canonical_feat_cols if c not in present_feat_cols]

            # Prepare a display copy so we don't mutate the original DataFrame
            display_df = training_df.copy()
            if missing_feat_cols:
                # Fill missing canonical columns with sentinel values for visibility (display-only)
                for col in missing_feat_cols:
                    try:
                        display_df[col] = tracker._ml_column_sentinel(col)
                    except Exception:
                        # Fallback to NaN if sentinel helper unavailable
                        display_df[col] = ''

            # Build ordered columns: identifiers first, then canonical features
            id_cols = ['player', 'market', 'result']
            show_cols = [c for c in id_cols if c in display_df.columns] + [c for c in canonical_feat_cols if c in display_df.columns]

            st.caption(f"Showing {len(present_feat_cols)} of {len(canonical_feat_cols)} canonical feature columns")

            if missing_feat_cols:
                st.warning(f"Missing {len(missing_feat_cols)} feature(s): {', '.join(sorted(missing_feat_cols))}")
                st.caption("These features are present in the CSV generator but missing from historical tracked bets. Shown above with sentinel values for clarity.")

            if show_cols:
                st.dataframe(display_df[show_cols].tail(20), hide_index=True, width="stretch")
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


def validate_training_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive data quality validation for ML training data.
    Returns a dict with validation results and issues found.
    """
    issues = []
    warnings = []
    stats = {}
    
    # 1. Basic stats
    stats['total_rows'] = len(df)
    stats['unique_players'] = df['player'].nunique() if 'player' in df.columns else 0
    stats['unique_markets'] = df['market'].nunique() if 'market' in df.columns else 0
    stats['date_range'] = f"{df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else "N/A"
    
    # 2. Check for missing required columns
    required_cols = ['hit', 'actual_value', 'line', 'predicted_side', 'player', 'market', 'date']
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        issues.append(f"Missing required columns: {missing_required}")
    
    # 3. Check feature columns exist
    feature_cols = [c for c in df.columns if c.startswith('feat_')]
    stats['feature_count'] = len(feature_cols)
    if len(feature_cols) < 30:
        warnings.append(f"Only {len(feature_cols)} features found (expected 36+)")
    
    # 4. Check for missing values in features
    if feature_cols:
        null_counts = df[feature_cols].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if len(cols_with_nulls) > 0:
            null_pct = (cols_with_nulls / len(df) * 100).to_dict()
            warnings.append(f"Features with nulls: {null_pct}")
        stats['null_feature_cols'] = len(cols_with_nulls)
    
    # 5. Validate hit label accuracy (critical!)
    if all(c in df.columns for c in ['hit', 'actual_value', 'line', 'predicted_side']):
        df_check = df.copy()
        df_check['computed_hit'] = df_check.apply(
            lambda r: 1 if (
                (r['predicted_side'] == 'OVER' and r['actual_value'] > r['line']) or
                (r['predicted_side'] == 'UNDER' and r['actual_value'] <= r['line'])
            ) else 0, axis=1
        )
        mismatches = (df_check['hit'] != df_check['computed_hit']).sum()
        stats['label_mismatches'] = mismatches
        if mismatches > 0:
            issues.append(f"⚠️ {mismatches} rows ({mismatches/len(df)*100:.1f}%) have incorrect 'hit' labels!")
    
    # 6. Check for reasonable feature ranges (V20.3 STRICT - exact feature names with 27 features)
    range_checks = {
        'feat_avg_minutes': (10, 42),           # Player average minutes
        'feat_opponent_drtg_season': (100, 125), # Team defensive rating (V20 exact name)
        'feat_line': (0, 100),                  # Prop bet line
        'feat_ema': (0, 100),                   # Player averages
        'feat_std': (0, 30),                    # Standard deviation
        'feat_is_home': (0, 1),                 # Binary
        'feat_spread': (-25, 25),               # Point spread
        'feat_game_total': (190, 270),          # Game total
        'feat_games_played': (1, 300),          # Games in sample (V20.2: expanded to 300)
        'feat_days_rest': (0, 14),              # Days since last game
        'feat_is_b2b': (0, 1),                  # Binary
        'feat_market_scoring': (0, 1),          # Binary
        'feat_market_counting': (0, 1),         # Binary
        'feat_market_combo': (0, 1),            # Binary
        'feat_market_rare': (0, 1),             # Binary
        # V20.2: Pace and trend features
        'feat_opponent_pace': (90, 115),        # Possessions per game
        'feat_team_pace': (90, 115),            # Possessions per game
        'feat_trend_5g': (-5, 5),               # Linear slope (can be negative)
        'feat_home_avg': (0, 100),              # Home game average
        'feat_away_avg': (0, 100),              # Away game average
        # V20.3 NEW: True-Shot features
        'feat_ts_pct': (-1, 1),                 # TS% fraction (-1 = unknown)
        'feat_ts_pct_delta': (-1, 1),           # Delta between rolling and season TS%
        # V20.3 NEW: Absence-aware features (-1 sentinel = unknown)
        'feat_team_out_ppg': (-1, 100),         # PPG of teammates out (-1 = unknown)
        'feat_team_out_count': (-1, 15),        # Count of teammates out (-1 = unknown)
        'feat_opp_out_ppg': (-1, 100),          # PPG of opponents out (-1 = unknown)
        'feat_opp_out_count': (-1, 15),         # Count of opponents out (-1 = unknown)
        # V20.3 NEW: Behavior & Risk ranges (allow -1 sentinel for unknown)
        'feat_min_volatility': (-1, 20),         # Rolling std of minutes
        'feat_foul_rate': (-1, 6),               # Rolling mean of PF per game
        'feat_cv': (-1, 5),                      # Coef. of var (std/ema)
    }
    
    out_of_range = []
    for col, (min_val, max_val) in range_checks.items():
        if col in df.columns:
            below = (df[col] < min_val).sum()
            above = (df[col] > max_val).sum()
            if below > 0 or above > 0:
                out_of_range.append(f"{col}: {below} below {min_val}, {above} above {max_val}")
    
    if out_of_range:
        warnings.append(f"Values outside expected ranges: {out_of_range[:5]}")  # Show first 5
    
    # 7. Check class balance
    if 'hit' in df.columns:
        hit_rate = df['hit'].mean()
        stats['overall_hit_rate'] = hit_rate
        if hit_rate < 0.35 or hit_rate > 0.65:
            warnings.append(f"Class imbalance: {hit_rate:.1%} hit rate (expected 40-60%)")
    
    # 8. Check for duplicate rows
    if all(c in df.columns for c in ['date', 'player', 'market', 'line']):
        dupes = df.duplicated(subset=['date', 'player', 'market', 'line']).sum()
        stats['duplicates'] = dupes
        if dupes > 0:
            warnings.append(f"{dupes} duplicate rows found")
    
    # 9. Market distribution
    if 'market' in df.columns:
        market_dist = df['market'].value_counts().to_dict()
        stats['market_distribution'] = market_dist
        min_market = min(market_dist.values())
        if min_market < 100:
            warnings.append(f"Some markets have few samples: {min_market}")
    
    # 10. Temporal distribution (check for data freshness)
    if 'date' in df.columns:
        df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
        recent = (df['date_parsed'] >= pd.Timestamp.now() - pd.Timedelta(days=30)).sum()
        stats['recent_30d_samples'] = recent
        if recent < 50:
            warnings.append(f"Only {recent} samples from last 30 days")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'stats': stats
    }


def render_train_model_tab():
    """
    V20.2 STRICT: Training UI with NO legacy compatibility.
    
    RULES (per Option A directive):
    - No data inference
    - No legacy migration  
    - No fabricated labels
    - No column remapping
    - Training fails loudly on invalid data
    """
    st.markdown("### Train V20.3 Model")
    st.info("Trains mean, variance, and calibration models. **Requires a V20-generated dataset with the canonical feature set.**")
    
    # Check if data exists
    data_path = DATA_DIR / "ml_training_data.csv"
    
    if not data_path.exists():
        st.error("❌ Training data not found.")
        st.warning("**Action Required:** Go to 'Generate Dataset' tab and create a new dataset.")
        return

    # =========================================================================
    # V20.3 STRICT SCHEMA - These canonical columns MUST exist exactly as named
    # NO FALLBACKS. NO DEFAULTS. NO LEGACY MIGRATION.
    # =========================================================================
    V20_REQUIRED_FEATURES = TRAINING_FEATURE_COLUMNS.copy()  # Use canonical list (V20 features)
    
    V20_REQUIRED_TARGET = 'actual_value'  # Regression target (postgame stat)
    
    V20_REQUIRED_METADATA = ['player', 'market', 'line', 'date']

    # Data Validation Section
    st.markdown("#### 🔍 V20.3 Schema Validation")
    st.caption(f"Required features: {len(V20_REQUIRED_FEATURES)} (V20.3 empirical)")
    
    if st.button("🔬 Validate Dataset", type="secondary"):
        with st.spinner("Validating against V20.3 schema..."):
            df = pd.read_csv(data_path)
            
            # === STRICT VALIDATION ===
            errors = []
            
            # 1. Check required features (NO inference, NO remapping)
            missing_features = [f for f in V20_REQUIRED_FEATURES if f not in df.columns]
            if missing_features:
                errors.append(f"❌ Missing V20.3 features ({len(missing_features)}): {', '.join(missing_features)}")
            
            # 2. Check regression target exists (NO fallback to classification)
            if V20_REQUIRED_TARGET not in df.columns:
                errors.append(f"❌ Missing regression target: '{V20_REQUIRED_TARGET}'")
            
            # 3. Check metadata columns
            missing_meta = [m for m in V20_REQUIRED_METADATA if m not in df.columns]
            if missing_meta:
                errors.append(f"⚠️ Missing metadata: {', '.join(missing_meta)}")
            
            # 4. Detect legacy/heuristic columns (should NOT exist in V20 data)
            legacy_columns = [
                'feat_usage_mult', 'feat_rest_factor', 'feat_matchup_mult',
                'feat_blowout_prob', 'feat_fatigue_mult', 'feat_position_mult',
                'feat_opp_drtg_l5', 'feat_hit_rate_l5', 'feat_hit_rate_l10'
            ]
            found_legacy = [c for c in legacy_columns if c in df.columns]
            if found_legacy:
                errors.append(f"⚠️ Legacy columns detected (invalid for V20): {', '.join(found_legacy)}")
            
            # === DISPLAY RESULTS ===
            if errors:
                st.error("❌ **Dataset is INVALID for V20.2 training**")
                for err in errors:
                    st.write(f"  • {err}")
                st.warning("**Action Required:** Regenerate dataset using V20.2 pipeline.")
                st.caption("Go to 'Generate Dataset' tab → Run generation → Return here to train.")
            else:
                st.success("✅ **Dataset passes V20.3 schema validation**")
                st.write(f"  • Rows: {len(df):,}")
                st.write(f"  • Features: {len(V20_REQUIRED_FEATURES)} (all present)")
                st.write(f"  • Target: {V20_REQUIRED_TARGET}")
                st.write(f"  • Players: {df['player'].nunique()}")
                st.write(f"  • Markets: {df['market'].nunique()}")
    
    st.markdown("---")
    
    if st.button("🏋️ Start Training", type="primary"):
        status = st.status("Training in progress...", expanded=True)
        try:
            # 1. Load Data
            status.write("Loading dataset...")
            df = pd.read_csv(data_path)
            
            # =========================================================
            # V20.2 STRICT: NO LEGACY COMPATIBILITY
            # If data is invalid, training FAILS. No migration. No inference.
            # =========================================================
            
            # Check required features - ALL 24 MUST EXIST
            missing_features = [f for f in V20_REQUIRED_FEATURES if f not in df.columns]
            if missing_features:
                status.update(label="❌ Schema Validation Failed", state="error")
                st.error(f"**FATAL:** Missing V20.3 features ({len(missing_features)}):")
                for f in missing_features:
                    st.write(f"  • `{f}`")
                st.warning("Dataset is incompatible with V20.3. Regenerate using 'Generate Dataset' tab.")
                st.info("V20.3 requires 27 features. Your dataset may have an older schema.")
                return
            
            # Check regression target (NO fallback to classification)
            if V20_REQUIRED_TARGET not in df.columns:
                status.update(label="❌ Schema Validation Failed", state="error")
                st.error(f"**FATAL:** Missing regression target '{V20_REQUIRED_TARGET}'")
                st.warning("V20 requires actual stat values for regression. Regenerate dataset.")
                return
            
            status.write(f"✅ Schema validation passed ({len(V20_REQUIRED_FEATURES)} features)")
            
            # 2. Prepare Data (strict - no fillna inference)
            status.write(f"Processing {len(df)} samples with {len(V20_REQUIRED_FEATURES)} features...")
            
            # Check for NaN in features (fail, don't fill)
            X = df[V20_REQUIRED_FEATURES]
            nan_counts = X.isna().sum()
            if nan_counts.sum() > 0:
                bad_cols = nan_counts[nan_counts > 0].to_dict()
                status.update(label="❌ Data Quality Failed", state="error")
                st.error(f"**FATAL:** NaN values in features: {bad_cols}")
                st.warning("Dataset contains missing values. Regenerate with complete data.")
                return
            
            X = X.apply(pd.to_numeric, errors='coerce')
            y = pd.to_numeric(df[V20_REQUIRED_TARGET], errors='coerce')
            
            # Check for NaN in target
            if y.isna().sum() > 0:
                status.update(label="❌ Data Quality Failed", state="error")
                st.error(f"**FATAL:** {y.isna().sum()} NaN values in target '{V20_REQUIRED_TARGET}'")
                st.warning("Dataset has missing actual values. Regenerate with complete data.")
                return
            
            status.write("✅ Data quality passed (no NaN)")
            
            # 3. Train Model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            status.write("Training V20.2 Regression model...")
            model = XGBRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=4,
                subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror'
            )
            model.fit(X_train, y_train)
            
            # 4. Evaluate
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            
            # 5. Save
            status.write("💾 Saving model...")
            joblib.dump(model, DATA_DIR / "nba_model_v20.pkl")
            joblib.dump(V20_REQUIRED_FEATURES, DATA_DIR / "nba_features_v20.pkl")
            
            status.update(label="✅ Training Complete!", state="complete")
            
            st.success("V20.2 Regression Model Trained!")
            c1, c2 = st.columns(2)
            c1.metric("Mean Absolute Error", f"{mae:.2f}")
            c2.metric("R² Score", f"{r2:.3f}")
            
            st.balloons()
            
        except Exception as e:
            status.update(label="❌ Training Failed", state="error")
            st.error(f"Error: {str(e)}")

def render_guide_tab():
    """Renders the comprehensive user guide - V20 Empirical."""
    st.markdown("### 📘 User Guide")
    st.caption(f"V20 Empirical ML | {CURRENT_VERSION}")

    # --- SECTION 1: CORE WORKFLOW ---
    with st.expander("🚀 Quick Start", expanded=True):
        st.markdown("""
        1. **Analyze:** Pick a player/market, hit "Run Analysis".
        2. **Decide:** Look for **EV+ bets** with probability in the 55-75% zone.
        3. **Track:** Click **"💾 Track"** to save the bet.
        4. **Resolve:** Next day, mark Win/Loss and enter actual score in **Bets Tab**.
        5. **Train:** With 500+ samples, retrain in **ML Tab** for better predictions.
        """)

    # --- SECTION 2: V20 ML ARCHITECTURE ---
    with st.expander("🤖 V20 ML Architecture", expanded=True):
        st.markdown("""
        **Pure Empirical ML Pipeline:**
        
        V20 uses ML as the **sole prediction engine**. No heuristics, no hand-tuned multipliers.
        
        * **Mean Model:** XGBRegressor predicts expected stat value E[stat]
        * **Variance Model:** Separate XGBRegressor predicts uncertainty σ[stat]
        * **CDF Probability:** P(Over) = 1 - Φ((line - μ) / σ)
        * **Calibration:** Isotonic regression corrects probability biases
        
        **15 Raw Observable Features:**
        - Statistical: avg_minutes, ema, std
        - Opponent: drtg_season
        - Context: line, spread, game_total
        - Rest: days_rest, is_home, is_b2b
        - Sample: games_played
        - Market: scoring/counting/combo/rare (one-hot)
        """)

    # --- SECTION 3: BET DECISION RULES ---
    with st.expander("📊 Bet Decision Rules"):
        st.markdown("""
        **The 4 Rules (all must pass):**
        
        1. **Offset:** Projection must align with recommended side
           - OVER: Model projection > Vegas line
           - UNDER: Model projection < Vegas line
        
        2. **Probability Zone:** 55% - 80%
           - Below 55% = thin edge, high variance
           - Above 80% = suspiciously confident, likely mispriced
        
        3. **Positive EV:** Expected Value > 0%
           - EV = (Prob × Win) - ((1-Prob) × Loss)
        
        4. **Stability:** CV ≤ 30%
           - CV = std / mean
           - High variance players are harder to predict
        """)

    # --- SECTION 4: INTERPRETING THE UI ---
    with st.expander("🎯 Understanding the Output"):
        st.markdown("""
        **Ticket Card:**
        - **Win Prob:** CDF-based probability for recommended side
        - **Predicted:** ML regression prediction (expected stat value)
        - **Std Dev:** Uncertainty from variance model
        - **Stake:** Kelly criterion suggestion (% of bankroll)
        
        **ML Decoder:**
        - **Model Group:** Which specialized model (scoring/counting/combo/rare)
        - **Raw ML:** Uncalibrated probability from CDF
        - **Calibrated:** After isotonic correction
        - **Calibrator:** ✅ Active means calibration curve is applied
        
        **Key Factors:**
        - All raw observables that fed into the model
        - Use these to sanity-check the prediction
        """)

    # --- SECTION 5: ML TRAINING ---
    with st.expander("🧠 Training Your Model"):
        st.markdown("""
        **The Training Loop:**
        
        1. **Track Bets:** Log every bet with actual scores
        2. **Generate Dataset:** ML Tab → Generate Dataset (bulk backtests)
        3. **Train Models:** ML Tab → Train Brain
        4. **Models Created:**
           - `nba_model_{group}.pkl` - Mean regression
           - `nba_variance_{group}.pkl` - Uncertainty model
           - `nba_calibrator_{group}.pkl` - Probability calibration
        
        **Minimum Data:** 500+ samples per market group recommended
        """)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    st.title("🏀 NBA Prop Analyzer")
    
    # Initialize session state
    if 'version' not in st. session_state or st.session_state['version'] != CURRENT_VERSION: 
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state['version'] = CURRENT_VERSION
    
    if 'analysis_result' not in st. session_state: 
        st.session_state. analysis_result = None
    if 'parlay_builder' not in st. session_state: 
        st.session_state.parlay_builder = []
    
    # Initialize components
    orchestrator = PredictionOrchestrator()
    tracker = Tracker()
    parlay_tracker = ParlayTracker()
    
    # Compact Sidebar
    with st. sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Bankroll Mode Toggle
        bankroll_enabled = st.toggle("💰 Bankroll Mode", value=True, help="Turn off when just gathering data for ML")
        st.session_state.bankroll_enabled = bankroll_enabled
        
        if bankroll_enabled:
            bankroll = st.number_input("💰 Bankroll (₱)", 100, 1000000, 600)
            
            # Unit Calculator (1 unit = 1%)
            unit_size = bankroll * 0.01
            st.caption(f"1u = ₱{unit_size:,.0f} | 5u = ₱{unit_size*5:,.0f}")
            st.caption("Stakes: A=5u | B=3u | C=1u")
        else:
            bankroll = 0  # No bankroll tracking
            unit_size = 0
            st.info("📊 Data Collection Mode")
            st.caption("Stakes hidden • P/L tracking disabled")
        
        # Quick stats with P/L in units (only show when bankroll enabled)
        if bankroll_enabled:
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
        
        with st.form(key='analysis_form'):
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

            # Show injury reports for BOTH teams
            # Logic moved inside form so it updates on submit
            if opp_in:
                # Get player's team abbrev (we'll determine this from the analysis context)
                player_team_abbrev = None
                if player_in:
                    try:
                        p_list = players.find_players_by_full_name(player_in)
                        if p_list:
                            p_id = p_list[0]['id']
                            from nba_api.stats.endpoints import commonplayerinfo
                            import time
                            time.sleep(0.3)
                            player_info = commonplayerinfo.CommonPlayerInfo(player_id=p_id).get_data_frames()[0]
                            if 'TEAM_ABBREVIATION' in player_info.columns:
                                player_team_abbrev = player_info['TEAM_ABBREVIATION'].iloc[0]
                    except:
                        pass

                inj_col1, inj_col2 = st.columns(2)

                # Opponent injuries (raw observational data)
                with inj_col1:
                    with st.expander(f"🏥 {opp_in} Injuries (Opponent)", expanded=False):
                        try:
                            injuries = orchestrator.injury_manager.get_team_injury_report(opp_in)
                            if injuries:
                                for inj in injuries:
                                    status = inj.get('status', 'Unknown')
                                    emoji = {'OUT': '🔴', 'DOUBTFUL': '🟠', 'QUESTIONABLE': '🟡', 'DAY-TO-DAY': '🟡', 'PROBABLE': '🟢'}.get(status, '⚪')
                                    pos = inj.get('position', '')
                                    pos_str = f" [{pos}]" if pos else ""
                                    injury_desc = inj.get('injury', '')
                                    st.markdown(f"{emoji} **{inj.get('name', 'Unknown')}**{pos_str} — {status}" + (f" ({injury_desc})" if injury_desc else ""))
                                st.caption(f"Source: {injuries[0].get('source', 'ESPN')}")
                            else:
                                st.info("No injuries reported")
                        except Exception as e:
                            st.caption(f"Could not fetch injuries: {e}")
                
                # Player's team injuries (raw observational data)
                with inj_col2:
                    if player_team_abbrev and player_team_abbrev != opp_in:
                        with st.expander(f"🏥 {player_team_abbrev} Injuries (Player's Team)", expanded=False):
                            try:
                                team_injuries = orchestrator.injury_manager.get_team_injury_report(player_team_abbrev)
                                if team_injuries:
                                    for inj in team_injuries:
                                        status = inj.get('status', 'Unknown')
                                        emoji = {'OUT': '🔴', 'DOUBTFUL': '🟠', 'QUESTIONABLE': '🟡', 'DAY-TO-DAY': '🟡', 'PROBABLE': '🟢'}.get(status, '⚪')
                                        pos = inj.get('position', '')
                                        pos_str = f" [{pos}]" if pos else ""
                                        injury_desc = inj.get('injury', '')
                                        st.markdown(f"{emoji} **{inj.get('name', 'Unknown')}**{pos_str} — {status}" + (f" ({injury_desc})" if injury_desc else ""))
                                    st.caption(f"Source: {team_injuries[0].get('source', 'ESPN')}")
                                else:
                                    st.info("No injuries reported")
                            except Exception as e:
                                st.caption(f"Could not fetch injuries: {e}")

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

            # V20.3 FIX: Cannot disable submit button based on form inputs (deadlock)
            # Validation must happen inside the submitted block
            submitted = st.form_submit_button("Run Analysis", type="primary")

        if submitted:
            if not player_in:
                st.error("Please select a player.")
            elif not is_valid:
                st.error(f"Invalid inputs: {error_msg}")
            else:
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
            
            # V20.2: Always show result - EV is displayed in card, user decides
            # Main recommendation - Ticket Style Card
            render_ticket_card(result, bankroll, bankroll_enabled)
        
            # Action buttons
            col_track, col_parlay = st.columns(2)
            # Tag selection for bet source tracking
            tag_options = ["Sniper", "Robot_Top_Pick", "Gut_Feel", "Live_Bet"]
            # V20: Auto-tag based on EV only
            default_tag = "Sniper" if result.decision.expected_value > 0 else "Gut_Feel"
            selected_tag = st.selectbox("Bet Source Tag:", tag_options, 
                index=tag_options.index(default_tag), key="bet_tag_select",
                help="Tag for filtering training data - 'Sniper' = EV+ picks")
            
            with col_track: 
                if st.button("💾 Track", use_container_width=True):
                    try:
                        features = result.features
                        # Build payload with static metadata
                        bet_payload = {
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
                            "predicted_mean": result.decision.predicted_mean,
                            "predicted_std": result.decision.predicted_std,
                            "tag": selected_tag,
                        }
                        # Dynamically merge features based on canonical schema
                        bet_payload.update(extract_features_dynamically(features, market=result.market))
                        tracker.log_bet(bet_payload)
                        st.toast(f"Bet tracked! [{selected_tag}]", icon="💾")
                    except Exception as e:
                        st.error(f"Failed to track bet: {e}")
                        logger.error(f"Bet tracking failed: {e}")
            
            with col_parlay: 
                # V19: Show parlay button if EV is positive
                if result.decision.expected_value > 0:
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
                            'position': 'N/A',  # V19: player_position removed from FeatureVector
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
            
            # =========================================================================
            # V18: CLV ANALYSIS - The Best Predictor of Long-Term Edge
            # =========================================================================
            if stats.get('clv_count', 0) >= 5:
                with st.expander("📊 CLV Analysis (Closing Line Value)", expanded=False):
                    st.caption("**CLV = Best predictor of sharp betting.** +CLV means you beat the closing line.")
                    
                    clv1, clv2, clv3, clv4 = st.columns(4)
                    
                    avg_clv = stats.get('avg_clv', 0)
                    clv_color = "normal" if avg_clv >= 0 else "inverse"
                    clv1.metric("Avg CLV", f"{avg_clv:+.2f} pts", 
                               delta=f"{stats.get('clv_positive_rate', 0):.0%} positive",
                               delta_color=clv_color,
                               help="Average line movement in your favor")
                    
                    clv2.metric("+CLV Win Rate", f"{stats.get('clv_positive_win_rate', 0):.0%}",
                               delta=f"{stats.get('clv_positive_decided', 0)} bets",
                               help="Win rate when you beat the closing line")
                    
                    clv3.metric("-CLV Win Rate", f"{stats.get('clv_negative_win_rate', 0):.0%}",
                               delta=f"{stats.get('clv_negative_decided', 0)} bets",
                               delta_color="inverse",
                               help="Win rate when line moved against you")
                    
                    clv_edge = stats.get('clv_edge', 0)
                    edge_emoji = "🎯" if clv_edge > 0.05 else ("⚠️" if clv_edge < 0 else "➖")
                    clv4.metric(f"{edge_emoji} CLV Edge", f"{clv_edge:+.1%}",
                               help="+CLV win rate minus -CLV win rate. Positive = you're sharp.")
                    
                    if clv_edge > 0.05:
                        st.success("✅ **You're beating closing lines!** This is the #1 indicator of long-term profitability.")
                    elif clv_edge < -0.05:
                        st.warning("⚠️ **Negative CLV edge.** Consider waiting for better line movement or getting sharper entries.")
            
            # =========================================================================
            # V18: GRADE PERFORMANCE ANALYSIS - Data-Driven Grade Evaluation
            # =========================================================================
            grade_stats = stats.get('grade_stats', {})
            if grade_stats and len(grade_stats) >= 2:
                with st.expander("📋 Grade Performance Analysis", expanded=False):
                    st.caption("**Performance by bet grade.** Use this to calibrate grade thresholds.")
                    
                    # Create a DataFrame for display
                    grade_rows = []
                    for grade in ['A', 'B', 'C', 'D', 'F']:
                        if grade in grade_stats:
                            gs = grade_stats[grade]
                            grade_rows.append({
                                'Grade': grade,
                                'Bets': gs['count'],
                                'Wins': gs['wins'],
                                'Win Rate': f"{gs['win_rate']:.1%}",
                                'ROI': f"{gs['roi']:+.1%}",
                                'Status': '✅ Profitable' if gs['roi'] > 0 else '❌ Losing'
                            })
                    
                    if grade_rows:
                        grade_df = pd.DataFrame(grade_rows)
                        st.dataframe(grade_df, hide_index=True, use_container_width=True)
                        
                        # Recommendations based on data
                        a_stats = grade_stats.get('A', {})
                        b_stats = grade_stats.get('B', {})
                        c_stats = grade_stats.get('C', {})
                        
                        if a_stats.get('roi', 0) < 0 and a_stats.get('count', 0) >= 10:
                            st.warning("⚠️ **Grade A bets are losing.** Consider tightening A-grade EV threshold.")
                        if c_stats.get('roi', 0) > 0.05 and c_stats.get('count', 0) >= 10:
                            st.info("💡 **Grade C bets are profitable.** You might be too conservative with grading.")
                        if b_stats.get('win_rate', 0) > a_stats.get('win_rate', 0) and a_stats.get('count', 0) >= 10:
                            st.info("💡 **Grade B has higher win rate than A.** Check if high-EV bets are mispriced.")
                    
                    # V18: Threshold Optimization Button
                    st.markdown("---")
                    if st.button("🔧 Optimize Grade Thresholds", help="Analyze your bet history to find optimal EV thresholds"):
                        with st.spinner("Analyzing bet history..."):
                            opt_result = tracker.optimize_grade_thresholds()
                        
                        if opt_result.get('status') == 'insufficient_data':
                            st.warning(opt_result.get('message'))
                        else:
                            st.markdown("**📊 Threshold Optimization Analysis**")
                            
                            # Show current vs optimal
                            curr = opt_result.get('current_thresholds', {})
                            optim = opt_result.get('optimal_thresholds', {})
                            
                            thresh_data = []
                            for grade in ['A', 'B', 'C']:
                                change = ""
                                if optim[grade] != curr[grade]:
                                    diff = (optim[grade] - curr[grade]) * 100
                                    change = f"{'⬆️' if diff > 0 else '⬇️'} {abs(diff):.1f}%"
                                thresh_data.append({
                                    'Grade': grade,
                                    'Current EV': f"{curr[grade]*100:.1f}%",
                                    'Optimal EV': f"{optim[grade]*100:.1f}%",
                                    'Change': change or '✓ OK'
                                })
                            
                            st.dataframe(pd.DataFrame(thresh_data), hide_index=True)
                            
                            # Show recommendation
                            st.info(f"💡 {opt_result.get('recommendation', 'No changes needed.')}")
                            
                            # Show performance at optimal thresholds
                            opt_perf = opt_result.get('grade_performance', {})
                            if opt_perf:
                                st.caption("**Expected Performance at Optimal Thresholds:**")
                                for grade, perf in opt_perf.items():
                                    emoji = "🟢" if perf.get('roi', 0) > 0 else "🔴"
                                    st.write(f"{emoji} Grade {grade}: {perf.get('count', 0)} bets, {perf.get('win_rate', 0):.1%} win rate, {perf.get('roi', 0):+.1f}% ROI")
            
            # Score Box Summary (if we have margin data)
            if stats.get('margin_count', 0) > 0:
                with st.expander("📦 Score Box Analysis", expanded=False):
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
        render_parlay_tab(parlay_tracker, bankroll, bankroll_enabled)
    
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
    # Required on Windows to allow spawned child processes to import this module
    try:
        multiprocessing.freeze_support()
    except Exception:
        pass
    main()
                
