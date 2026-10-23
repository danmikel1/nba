import streamlit as st
from groq import Groq
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
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from scipy import stats  # V19: For CDF-based probability calculation
import requests
from bs4 import BeautifulSoup
from data_engine import DataEngine
from trading_engine import TradingEngine
from depth_chart_manager import DepthChartManager
from persistence_utils import atomic_json_save, safe_json_load
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
    def get_data_engine(data_dir: Path, season: str, allow_fetch: bool = True) -> DataEngine:
        """Singleton provider for DataEngine using Streamlit cache.

        `allow_fetch` mirrors `CONFIG.API_ALLOW_REFRESH` and is passed to the
        underlying DataEngine so that the cached instance honours the setting.
        """
        return DataEngine(data_dir=data_dir, season=season, allow_fetch=allow_fetch)

logging.basicConfig(level=logging.INFO if _IN_MAIN_PROCESS else logging.WARNING)
logger = logging.getLogger(__name__)

CURRENT_VERSION = 'v21.5'


def _get_dynamic_seasons():
    """Return dynamic season strings based on today's date.

    - If month >= Oct, season starts in current year (e.g., Oct 2025 -> '2025-26').
    - If month < Oct, season starts in previous year (e.g., Jan 2026 -> '2025-26').
    Returns: (current_season, prev_season, training_seasons_tuple)
    """
    now = datetime.now()
    start_year = now.year if now.month >= 10 else now.year - 1

    def fmt(y):
        return f"{y}-{str(y+1)[-2:]}"

    curr = fmt(start_year)
    prev = fmt(start_year - 1)
    train = (curr, prev, fmt(start_year - 2), fmt(start_year - 3))
    return curr, prev, train


_DYNAMIC_CURRENT_SEASON, _DYNAMIC_PREV_SEASON, _DYNAMIC_TRAINING_SEASONS = _get_dynamic_seasons()


@dataclass(frozen=True)
class Config:
    """
    V20 EMPIRICAL CONFIG: No beliefs, only observables.
    
    All heuristic multipliers, penalties, thresholds REMOVED.
    Model learns relationships from data, not hardcoded assumptions.
    """
    # === IDENTIFIERS ===
    CURRENT_SEASON: str = _DYNAMIC_CURRENT_SEASON
    PREV_SEASON: str = _DYNAMIC_PREV_SEASON
    
    # === MULTI-SEASON TRAINING (V20.1) ===
    # More data = better generalization
    TRAINING_SEASONS: tuple = _DYNAMIC_TRAINING_SEASONS
    
    # === API / CACHE ===
    API_DELAY: float = 1.0
    API_MAX_RETRIES: int = 3
    CACHE_TTL_TEAM_STATS: int = 3600    # 1 hour
    CACHE_TTL_PLAYER_IDS: int = 3600    # 1 hour (was 86400 / 24h)
    CACHE_TTL_GAME_LOGS: int = 600      # 10 minutes (was 1800 / 30m)
    
    # === API CONTROLS ===
    # Allows the app to run completely offline using on-disk caches.
    # When False, any attempt to refresh data will use stale files or fall
    # back to defaults instead of hitting stats.nba.com (useful when the
    # service is down or you're just browsing previous bets).
    API_ALLOW_REFRESH: bool = True

    # === ML COMMITMENT ===
    # ML is the SOLE prediction engine. No fallbacks.
    ML_REQUIRE_MODEL: bool = True
    ML_CALIBRATION_ENABLED: bool = True

    # === LIVE CALIBRATION ===
    # Temperature scaling corrects overconfident probabilities against sharp
    # sportsbook lines.  T > 1 = overconfident.  Fitted from bet_tracker.json
    # outcomes via scripts/fit_live_calibrator.py.  Set to 1.0 to disable.
    # V26: Reduced from 4.75 → 2.40 after line correction (realistic half-point
    # training lines via OLS mapping from OddsBlaze DK data).
    LIVE_CALIBRATION_TEMPERATURE: float = 2.40
    
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
PREDICTION_LOG_FILE = DATA_DIR / "prediction_log.json"  # All predictions for analysis

# --- Module-level caches for morning_refresh.py data ---
_FATIGUE_CACHE: dict = {}  # Lazy-loaded from data/fatigue_index.json
_ALTITUDE_FT_LOOKUP = {"DEN": 5280, "UTA": 4327, "OKC": 1200}

# Arena coordinates for historical travel distance computation (lat, lon)
_ARENA_COORDS = {
    "ATL": (33.7573, -84.3963), "BOS": (42.3662, -71.0621),
    "BKN": (40.6826, -73.9754), "CHA": (35.2251, -80.8392),
    "CHI": (41.8807, -87.6742), "CLE": (41.4965, -81.6882),
    "DAL": (32.7905, -96.8103), "DEN": (39.7487, -105.0077),
    "DET": (42.3410, -83.0552), "GSW": (37.7680, -122.3877),
    "HOU": (29.7508, -95.3621), "IND": (39.7639, -86.1555),
    "LAC": (33.9425, -118.1072), "LAL": (34.0430, -118.2673),
    "MEM": (35.1382, -90.0505), "MIA": (25.7814, -80.1870),
    "MIL": (43.0450, -87.9170), "MIN": (44.9795, -93.2760),
    "NOP": (29.9490, -90.0823), "NYK": (40.7505, -73.9934),
    "OKC": (35.4634, -97.5151), "ORL": (28.5392, -81.3839),
    "PHI": (39.9012, -75.1720), "PHX": (33.4457, -112.0712),
    "POR": (45.5316, -122.6668), "SAC": (38.5802, -121.4997),
    "SAS": (29.4270, -98.4375), "TOR": (43.6435, -79.3791),
    "UTA": (40.7683, -111.9011), "WAS": (38.8981, -77.0209),
}

def _haversine_miles(coord1: tuple, coord2: tuple) -> float:
    """Great-circle distance in miles between two (lat, lon) points."""
    from math import radians, sin, cos, sqrt, atan2
    R = 3958.8
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

        
# =============================================================================
# DATA STRUCTURES
# =============================================================================

# V20 EMPIRICAL: No grades, no risk categories - just data
# ResultQuality kept for audit trail only (post-hoc analysis)
@st.cache_data(ttl=86400) # Cache for 24 hours
def get_all_nba_players():
    return players.get_players()

@st.cache_resource
def get_groq_client():
    """Initialize Groq client securely."""
    if "GROQ_API_KEY" in st.secrets:
        return Groq(api_key=st.secrets["GROQ_API_KEY"])
    return None
    
def get_ai_second_opinion(prop_analysis, backtest_json, history_context=None):
    """
    V21.4: RAW FEATURE AI (No Hard-Coded Heuristics)
    
    Philosophy: Feed the LLM the same features the ML model uses.
    Let the LLM interpret them, not Python hard-coded rules.
    
    The ML model learned relationships from data.
    The LLM provides human-readable interpretation of the same features.
    """
    
    # ==========================================================================
    # 1. EXTRACT FEATURES (Same as ML Model Input)
    # ==========================================================================
    player = prop_analysis.get('player', 'Unknown')
    position = prop_analysis.get('position', '')
    opponent = prop_analysis.get('opponent', backtest_json.get('opponent', ''))
    market = prop_analysis.get('market', 'PTS')
    
    # --- Model Outputs ---
    line = float(prop_analysis.get('line', backtest_json.get('line', 0.0)) or 0.0)
    proj = float(prop_analysis.get('projected', prop_analysis.get('projected_value', 0.0)) or 0.0)
    prob = float(prop_analysis.get('prob', 0.0) or 0.0)
    ev = float(prop_analysis.get('ev', 0.0) or 0.0)
    signal = prop_analysis.get('predicted_side', prop_analysis.get('signal', 'OVER'))
    spread = float(prop_analysis.get('spread', backtest_json.get('spread', 0.0)) or 0.0)
    
    # --- Raw Features (No Interpretation Yet) ---
    ema = float(prop_analysis.get('feat_ema', 0) or 0)
    season_avg = float(prop_analysis.get('feat_season_avg', 0) or 0)
    l10_avg = float(prop_analysis.get('feat_l10_avg', 0) or 0)
    trend_5g = float(prop_analysis.get('feat_trend_5g', 0) or 0)
    std = float(prop_analysis.get('feat_std', 0) or 0)
    cv = float(prop_analysis.get('feat_cv', 0) or 0)
    avg_minutes = float(prop_analysis.get('feat_avg_minutes', 0) or 0)
    min_volatility = float(prop_analysis.get('feat_min_volatility', 0) or 0)
    days_rest = int(prop_analysis.get('feat_days_rest', 1) or 1)
    is_b2b = bool(prop_analysis.get('feat_is_b2b', False))
    is_home = bool(prop_analysis.get('feat_is_home', True))
    depth_rank = int(prop_analysis.get('feat_depth_rank', 4) or 4)
    hit_rate_l10 = float(prop_analysis.get('feat_hit_rate_l10', 0) or 0)
    hit_rate_l20 = float(prop_analysis.get('feat_hit_rate_l20', 0) or 0)
    team_out_ppg = float(prop_analysis.get('feat_team_out_ppg', 0) or 0)
    opp_drtg = float(prop_analysis.get('feat_opponent_drtg_season', 110) or 110)
    opp_rank_vs_pos = int(prop_analysis.get('feat_opp_rank_vs_pos', 15) or 15)
    home_avg = float(prop_analysis.get('feat_home_avg', ema) or ema)
    away_avg = float(prop_analysis.get('feat_away_avg', ema) or ema)
    games_played = int(prop_analysis.get('feat_games_played', 0) or 0)
    
    # --- Derived (Math, Not Heuristics) ---
    edge = (proj - line) if signal == 'OVER' else (line - proj)
    depth_labels = {1: 'Starter', 2: 'Rotation', 3: 'Bench', 4: 'Unknown'}
    depth_str = depth_labels.get(depth_rank, 'Unknown')
    location = "HOME" if is_home else "AWAY"
    location_avg = home_avg if is_home else away_avg

# ==========================================================================
    # 2. BUILD RAW FEATURE PROMPT
    # ==========================================================================
    # Compute signal coherence for trend analysis
    trend_aligns = (trend_5g > 0 and signal == 'OVER') or (trend_5g < 0 and signal == 'UNDER') or abs(trend_5g) < 0.3
    trend_status = "aligned" if trend_aligns else "contradictory"

    # --- NEW: MATH SAFETY CHECK ---
    # We calculate the severity here so the LLM doesn't have to guess
    trend_flag = ""
    if not trend_aligns and abs(trend_5g) > 1.5:
        trend_flag = f" [⚠️ CRITICAL: Exceeds ±1.5 Threshold!]"
    
    prompt = f"""
You validate ML sports betting predictions. Your default answer is TAKE.

┌───────────────────────────────────────────────────────────────────────────────┐
│ BET: {player} ({position}) {market} {signal} {line} vs {opponent}             │
│ Model Projection: {proj:.1f} | Win Prob: {prob*100:.0f}% | EV: {ev*100:+.1f}% │
└───────────────────────────────────────────────────────────────────────────────┘

FEATURES:
• Recent: Season={season_avg:.1f}, L10={l10_avg:.1f}, EMA={ema:.1f}
• Trend: {trend_5g:+.2f}/game ({trend_status} with {signal}){trend_flag}
• Volatility: StdDev={std:.1f}, CV={cv*100:.0f}%, MinutesVol={min_volatility:.1f}
• Sample: {games_played} games, Hit Rate L10={hit_rate_l10*100:.0f}%
• Context: {location}, Days Rest={days_rest}, B2B={"Yes" if is_b2b else "No"}
• Matchup: OppDRTG={opp_drtg:.1f}, Rank vs Pos=#{opp_rank_vs_pos}/30
• Injuries: Team missing {team_out_ppg:.1f}ppg, Opp missing {prop_analysis.get('feat_opp_out_ppg', 0):.1f}ppg

════════════════════════════════════════════════════════════════════════════════
RULES (FADE requires ALL of these to be true for the specific criterion)
════════════════════════════════════════════════════════════════════════════════
FADE ONLY if one of these EXACT conditions is met:
1. games_played < 15
2. CV > 35%
3. Trend magnitude > 1.5/game AND trend direction opposes the signal
4. MinutesVol > 8 AND signal is OVER

If NONE of these conditions are met, output TAKE. Do not invent other reasons.

OUTPUT (exactly this format, nothing else):
VERDICT: [TAKE/FADE]
REASON: [One sentence citing the specific feature values that support your verdict]
CONFIDENCE: [1-10]
"""
    
    try:
        from llm_engine import query_llm
        system_msg = (
            "You validate ML betting predictions against 4 specific numeric thresholds. "
            "If none are triggered, output TAKE. Do not add subjective commentary or invent concerns not in the rules."
        )
        res = query_llm(prompt=prompt, system_prompt=system_msg, temperature=0.2)
        if res.get("success"):
            return res.get("response", "").strip()

        # Fallback to direct Groq client
        client = get_groq_client()
        if client:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=120
            )
            return completion.choices[0].message.content.strip()

        return res.get("response", "No LLM response available.")
    except Exception as e:
        return f"AI Generation Error: {str(e)}"
    
    
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

    # === V20.4 NEW: ADVANCED CONTEXT ===
    feat_usage_rate: float = -1.0   # Rolling usage intensity
    feat_h2h_avg: float = -1.0      # Avg vs this opponent
    feat_season_avg: float = 0.0  # True baseline
    feat_l10_avg: float = 0.0
    feat_eff_per_min: float = 0.0
    feat_fatigue_load: float = 0.0  # (Minutes / Rest) interaction
    feat_form_gap: float = 0.0      # (Recent - Season) delta

    # === V21.6 NEW: TRAVEL / ALTITUDE FEATURES (from morning_refresh.py) ===
    feat_miles_96h: float = 0.0     # Miles traveled by player's team in last 96 h
    feat_altitude_next: int = 0     # Altitude in feet of the game's arena

    # === V20.5 NEW: HIT RATE FEATURES ===
    feat_hit_rate_l10: float = 0.0  # % of last 10 games over line
    feat_hit_rate_l20: float = 0.0  # % of last 20 games over line

    # === V20.5 NEW: DEFENSE VS POSITION ===
    feat_opp_rank_vs_pos: int = 15   # 1=Best defense, 30=Worst defense

    # === V21.1 NEW: DEPTH CHART RANK ===
    # 1=Starter, 2=Backup, 3=Reserve, 4=Deep Bench (Not Found)
    # Dynamically adjusted when starters are injured ("Next Man Up")
    feat_depth_rank: int = 4

    # === MARKET IDENTITY (one-hot encoded) ===
    market_scoring: int = 0  # PTS
    market_counting: int = 0 # REB, AST  
    market_combo: int = 0    # PRA, PR, PA, RA

    # === BEHAVIOR & RISK FEATURES ===
    # Rolling std dev of minutes across recent games (pre-game observable)
    feat_min_volatility: float = -1.0
    # Rolling mean of personal fouls per game over lookback window (pre-game observable)
    feat_foul_rate: float = -1.0
    # Coefficient of variation for target stat (std / ema)
    feat_cv: float = -1.0
    
    # === DATA QUALITY (audit only, not used in ML) ===
    data_quality: DataQuality = field(default_factory=DataQuality)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for ML pipeline."""
        return asdict(self)

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Return canonical training feature column names.

        Ordering strategy:
        1. Keep a stable legacy observable ordering first (prefixed with 'feat_')
        2. Followed by all remaining dataclass fields (in declaration order) converted to 'feat_' names
        This ensures fields like 'team_out_count' (non-feat name) are included automatically.
        """
        from dataclasses import fields
        identifiers = {'player_id', 'player_name', 'opponent_abbrev', 'market', 'data_quality'}

        # Canonical legacy observable order to preserve historical column layout
        legacy = [
            'avg_minutes', 'ema', 'std', 'opponent_drtg_season', 'line', 'spread',
            'game_total', 'days_rest', 'is_home', 'is_b2b', 'games_played',
            'opponent_pace', 'team_pace', 'trend_5g', 'home_avg', 'away_avg'
        ]

        # All dataclass field names in declaration order (excluding identifiers)
        all_attrs = [f.name for f in fields(cls) if f.name not in identifiers]

        ordered: List[str] = []
        for name in legacy:
            if name in all_attrs and name not in ordered:
                ordered.append(name)

        # Append remaining attributes preserving declaration order
        for name in all_attrs:
            if name not in ordered:
                ordered.append(name)

        # Map to canonical 'feat_' column names (leave existing 'feat_' names intact)
        feat_columns: List[str] = []
        for name in ordered:
            feat_columns.append(name if name.startswith('feat_') else f'feat_{name}')

        # Copilot: Remove 'feat_hit_rate_l10' and 'feat_hit_rate_l20' from training feature list.
        # Keep the fields in the FeatureVector dataclass for live/inference usage — only exclude from training.
        feat_columns = [c for c in feat_columns if c not in ('feat_hit_rate_l10', 'feat_hit_rate_l20')]

        return feat_columns

    @property
    def feat_fields(self) -> List[str]:
        """Instance-level canonical ML column names (with 'feat_' prefix)."""
        return self.__class__.get_feature_names()

    def to_ml_array(self) -> np.ndarray:
        """Convert feature values to a numeric numpy array using canonical feature list.

        Ordering is exactly `get_feature_names()` to guarantee alignment with CSV/training.
        """
        feature_names = self.__class__.get_feature_names()
        arr: List[float] = []

        for col in feature_names:
            # Derive attribute name (strip 'feat_' prefix for legacy mapped fields)
            attr = col[len('feat_'):]
            raw = getattr(self, attr, None) if hasattr(self, attr) else getattr(self, col, None)

            # Determine sentinel
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


@dataclass
class Projection:
    """
    V19 REFACTOR: Model projection output.
    
    Outputs only empirically meaningful values:
    - predicted_mean: E[stat] from regression model
    - predicted_std: [stat] uncertainty
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
    V21.2: Added quality_score for risk-adjusted ranking (EV / sigma)
    """
    recommended_side: str  # "OVER" or "UNDER"
    probability: float     # P(winning the recommended side)
    expected_value: float  # EV at given odds
    predicted_mean: float  # E[stat] from model
    predicted_std: float   # [stat] uncertainty
    kelly_stake: float     # Suggested stake (EV-based)
    kelly_fraction: float  # Stake as fraction of bankroll
    quality_score: float = 0.0  # V21.2: Sharpe-like risk-adjusted EV (EV / sigma)


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
    
    # V21.2: Player position for AI analyst
    player_position: Optional[str] = None
    
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


def safe_clean_for_arrow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robustly clean DataFrame to prevent PyArrow crashes in Streamlit.
    Forces strict types for boolean and integer columns.
    """
    if df is None or df.empty:
        return df
        
    df_clean = df.copy()
    
    # 1. Fix Boolean Columns (handles 1, 0, 1.0, "True", True -> bool)
    # Includes both 'feat_' versions and standard versions
    bool_targets = [
        'feat_is_home', 'feat_is_b2b', 
        'is_home', 'is_b2b', 'hit', 'passes_offset'
    ]
    for col in bool_targets:
        if col in df_clean.columns:
            # Force numeric (True->1, False->0), fill NaNs with 0, then cast to bool
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int).astype(bool)

    # 2. Fix Integer Columns (handles 1.0 -> 1, "5" -> 5)
    int_targets = [
        'market_scoring', 'market_counting', 'market_combo',
        'feat_market_scoring', 'feat_market_counting', 'feat_market_combo',
        'feat_team_out_count', 'feat_opp_out_count', 'feat_games_played'
    ]
    for col in int_targets:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(-1).astype(int)
            
    return df_clean

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

        # attempt to warm position cache from disk before any API calls
        self._preload_positions_from_json()

        # Integrate DataEngine
        if data_engine:
            self.data_engine = data_engine
        elif _IN_MAIN_PROCESS:
            # Use singleton if available
            # pass API_ALLOW_REFRESH so that the singleton respects offline mode
            self.data_engine = get_data_engine(
                DATA_DIR,
                config.CURRENT_SEASON,
                allow_fetch=getattr(config, 'API_ALLOW_REFRESH', True)
            )
        else:
            # Create fresh instance for worker process
            # Note: Workers will share disk cache (parquet) so this is efficient
            self.data_engine = DataEngine(
                DATA_DIR,
                config.CURRENT_SEASON,
                allow_fetch=getattr(config, 'API_ALLOW_REFRESH', True)
            )
        # inject any preloaded positions into the newly created data_engine
        if self._position_cache and hasattr(self.data_engine, '_position_cache'):
            count = len(self._position_cache)
            try:
                self.data_engine._position_cache.update(self._position_cache)
                logger.debug(f"Injected {count} positions into data_engine cache")
            except Exception:
                logger.debug("Failed to inject position cache into data_engine")
    
    def _preload_positions_from_json(self):
        """Load player positions from disk if available and cache them.

        Searches DATA_DIR and its parent for player_positions.json which is the
        same file written by DataLoader.fetch_player_positions(). The file may
        contain a top-level dict with a "positions" key or be the positions map
        directly. Keys are string player IDs and values are position codes.
        Loaded entries are converted to int keys and stored in
        ``self._position_cache``. If a ``data_engine`` instance already exists
        and it exposes a ``_position_cache`` attr we also update that so the
        downstream engine can use the warmed cache immediately.
        """
        paths = [
            DATA_DIR / 'player_positions.json',
            DATA_DIR.parent / 'player_positions.json'
        ]
        positions = {}
        for p in paths:
            if p.exists():
                try:
                    with open(p, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            positions_dict = data.get('positions', data)
                            for k, v in positions_dict.items():
                                try:
                                    positions[int(k)] = v
                                except Exception:
                                    continue
                except Exception as e:
                    logger.warning(f"Failed to load positions from {p}: {e}")
                break
        if not positions:
            logger.debug("No player_positions.json found for preloading")
            return
        self._position_cache.update(positions)
        count = len(positions)
        logger.info(f"DataLoader: pre-loaded {count} player positions from {p}")
        if hasattr(self, 'data_engine') and getattr(self.data_engine, '_position_cache', None) is not None:
            try:
                self.data_engine._position_cache.update(positions)
                logger.debug(f"Injected {count} positions into data_engine cache")
            except Exception:
                logger.debug("Failed to inject positions into data_engine at preload time")

    def _api_call_with_retry(self, func, description: str = "API call"):
        """Execute API call with retry logic. Exponential backoff for rate limits.

        Respects CONFIG.API_ALLOW_REFRESH; if refresh is disabled we immediately
        raise a DataLoaderError to prevent long sleeps and cooldown loops.
        """
        import multiprocessing
        import random

        # honor offline mode
        if not getattr(self.config, 'API_ALLOW_REFRESH', True):
            logger.warning(f"API calls disabled; skipping {description}")
            raise DataLoaderError(f"API disabled, cannot perform {description}")

        last_exception = None
        cooldown_count = 0
        MAX_COOLDOWNS = 5  # More patient - 5 cooldowns before failing
        
        current_process = multiprocessing.current_process()
        if current_process.name != 'MainProcess':
            error_msg = (
                f" API CALL BLOCKED IN WORKER PROCESS! \n"
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
                        logger.error(f" {MAX_COOLDOWNS} consecutive timeouts. API is blocked. Failing fast.")
                        raise DataLoaderError(f"{description} failed: API rate limited after {cooldown_count} timeouts")
                    # Exponential backoff: 10s, 20s, 30s, 40s (reduced from 30/60/90/120)
                    wait_time = 10 * cooldown_count
                    logger.warning(f" API Cooldown {cooldown_count}/{MAX_COOLDOWNS}. Waiting {wait_time}s...")
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
        """Fetch player's primary position using DataEngine."""
        return _self.data_engine._get_real_position(player_id)
    
    @_conditional_cache(ttl=CONFIG.CACHE_TTL_GAME_LOGS)
    def fetch_game_logs(_self, player_id: int, season: str = None) -> pd.DataFrame:
        """
        Fetch player game logs for a season.
        Uses DataEngine for bulk fetching and caching.
        """
        if season is None:
            season = _self.config.CURRENT_SEASON
        
        try: 
            # Use DataEngine to get player logs from cached master file
            df = _self.data_engine.get_player_logs(player_id, season)
            
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
    
    def fetch_multi_season_logs(_self, player_id: int) -> pd.DataFrame:
        """Fetch logs from current and previous season."""
        try:
            # DataEngine makes this fast - hits local parquet for both seasons
            df_current = _self.fetch_game_logs(player_id, _self.config.CURRENT_SEASON)
            df_prev = _self.fetch_game_logs(player_id, _self.config.PREV_SEASON)

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
        """
        Fetch league-wide team stats for a specific season.

        V21.6 DISK CACHE: Writes to data/cache/team_stats_{season}.pkl on first fetch.
        On subsequent app restarts, reads from disk (TTL: 6h current season, 7 days historical).
        Falls back to stale cache if API is unavailable (rate limited, timeout, etc.).
        """
        import joblib

        if season is None:
            season = _self.config.CURRENT_SEASON

        # --- Disk cache path & TTL ---
        cache_dir = DATA_DIR / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"team_stats_{season}.pkl"
        ttl_hours = 6.0 if season == _self.config.CURRENT_SEASON else 168.0  # 7 days historical

        # --- Cache hit ---
        if cache_path.exists():
            age_hours = (time.time() - cache_path.stat().st_mtime) / 3600.0
            if age_hours < ttl_hours:
                try:
                    result = joblib.load(cache_path)
                    logger.debug(f"Team stats for {season} loaded from disk cache ({age_hours:.1f}h old)")
                    return result
                except Exception as e:
                    logger.warning(f"Failed to read team stats cache for {season}: {e}")

        # --- Cache miss / stale — fetch from API ---
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

            result = (stats.set_index('TEAM_ID'), stats['PACE'].mean(), stats['DEF_RATING'].mean())

            # Save to disk
            try:
                joblib.dump(result, cache_path)
                logger.info(f"Team stats for {season} cached to disk")
            except Exception as e:
                logger.warning(f"Failed to save team stats cache for {season}: {e}")

            return result

        except Exception as e:
            logger.error(f"Failed to fetch team stats for {season}: {e}")

            # --- Fallback: return stale cache rather than defaults ---
            if cache_path.exists():
                try:
                    result = joblib.load(cache_path)
                    age_hours = (time.time() - cache_path.stat().st_mtime) / 3600.0
                    logger.warning(f"Using stale team stats cache for {season} ({age_hours:.1f}h old) due to API failure")
                    return result
                except Exception as ce:
                    logger.warning(f"Stale cache also unreadable for {season}: {ce}")

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
            logger.info(f" Fetching team stats for {season}...")
            time.sleep(self.config.API_DELAY)  # Rate limit protection
            stats_df, avg_pace, avg_def = self.fetch_team_stats(season)
            all_stats[season] = (stats_df, avg_pace, avg_def)
            logger.info(f"   {season}: {len(stats_df)} teams, pace={avg_pace:.1f}, drtg={avg_def:.1f}")
        
        return all_stats

    def fetch_player_positions(self, seasons: List[str] = None, player_ids: List[int] = None, force_refresh: bool = False) -> Dict[int, str]:
        """
        Bulk-fetch player positions using a combination of static metadata and
        the `leaguedashplayerstats` endpoint per season. Optionally, will run
        per-player lookups (parallel threads) for only the provided `player_ids`.
        Results are cached to disk at DATA_DIR/player_positions.json to avoid
        repeated API calls.

        Returns:
            Dict[player_id:int -> position:str]  # position in {PG, SG, SF, PF, C}
        """
        cache_path = DATA_DIR / 'player_positions.json'

        # Load from cache unless force_refresh
        if not force_refresh and cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as fh:
                    raw = json.load(fh)
                # keys are strings in JSON - convert back to ints
                cached = {int(k): v for k, v in raw.get('positions', {}).items()}
                logger.info(f"Loaded player positions cache ({len(cached)} entries)")
            except Exception as e:
                logger.warning(f"Failed to read player positions cache: {e}")
                cached = {}
        else:
            cached = {}

        positions: Dict[int, str] = dict(cached)  # start with cache

        # 1) Static player metadata (instant, no API)
        static_count = 0
        try:
            for p in players.get_players():
                pid = p.get('id')
                pos = (p.get('position') or '').strip()
                # Normalize common short codes
                if pos == 'G':
                    norm = 'PG'
                elif pos == 'F':
                    norm = 'PF'
                elif pos in ('PG', 'SG', 'SF', 'PF', 'C'):
                    norm = pos
                else:
                    norm = ''
                if pid and norm and pid not in positions:
                    positions[pid] = norm
                    static_count += 1
        except Exception as e:
            logger.warning(f"Failed to build static positions map: {e}")

        # 2) Augment with per-season league player stats (one call per season)
        if seasons is None:
            seasons = list(CONFIG.TRAINING_SEASONS)

        # If we already have good cache coverage for the requested players,
        # skip the expensive per-season API calls entirely.
        _need_season_fetch = True
        if player_ids and positions:
            _hit = sum(1 for pid in player_ids if pid in positions)
            _coverage = _hit / len(player_ids) if player_ids else 0
            if _coverage >= 0.90:
                logger.info(f"Position cache has {_coverage:.0%} coverage ({_hit}/{len(player_ids)}) — skipping season API calls")
                _need_season_fetch = False

        per_season_count = 0
        if _need_season_fetch:
            for i, season in enumerate(seasons):
                try:
                    if i > 0:
                        time.sleep(3)  # Rate-limit between season calls
                    logger.info(f"Fetching league player stats for positions (season={season})...")
                    df = leaguedashplayerstats.LeagueDashPlayerStats(season=season, per_mode_detailed='PerGame', timeout=30).get_data_frames()[0]

                    # Accept a variety of possible column names for position
                    pos_cols = [c for c in df.columns if c.upper() in ('PLAYER_POSITION','POSITION','PLAYER_POS','POSITION_SHORT','POS')]
                    if 'PLAYER_ID' in df.columns and pos_cols:
                        col = pos_cols[0]
                        for _, row in df[['PLAYER_ID', col]].iterrows():
                            pid = int(row['PLAYER_ID'])
                            raw_pos = (row.get(col) or '').strip()
                            if raw_pos == 'G':
                                norm = 'PG'
                            elif raw_pos == 'F':
                                norm = 'PF'
                            elif raw_pos in ('PG', 'SG', 'SF', 'PF', 'C'):
                                norm = raw_pos
                            else:
                                norm = ''
                            if norm:
                                positions[pid] = norm
                                per_season_count += 1
                    else:
                        logger.warning(f"Unexpected columns in player stats for {season}: {df.columns.tolist()[:10]}")
                except Exception as e:
                    logger.warning(f"Failed to fetch league player stats for {season}: {e}")
                    # continue to next season

        # 3) For any requested player_ids still missing, do limited parallel per-player lookups
        if player_ids:
            missing = [pid for pid in player_ids if pid not in positions]
            if missing:
                logger.info(f"Fetching individual positions for {len(missing)} players (parallel)...")
                from concurrent.futures import ThreadPoolExecutor, as_completed

                def _get_pos(pid):
                    try:
                        return pid, self.get_player_position(pid)
                    except Exception as e:
                        logger.warning(f"Failed to get position for player {pid}: {e}")
                        return pid, 'SF'

                max_workers = min(3, max(1, len(missing)))
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = {ex.submit(_get_pos, pid): pid for pid in missing}
                    for fut in as_completed(futures):
                        pid, pos = fut.result()
                        positions[pid] = pos

        # Persist to disk (only if we found anything meaningful)
        try:
            if positions:
                out = {'time': datetime.now().isoformat(), 'positions': {str(k): v for k, v in positions.items()}}
                with open(cache_path, 'w', encoding='utf-8') as fh:
                    json.dump(out, fh, indent=2)
                logger.info(f"Saved player positions cache ({len(positions)} entries) -> {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to write player positions cache: {e}")

        logger.info(f"Positions: static={static_count}, per_season={per_season_count}, total_cached={len(positions)}")
        return positions

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

    def _sanitize_injury_player_name(self, raw_name: str) -> str:
        """Clean malformed ESPN injury names (e.g., 'Pistons. Immanuel Quickley')."""
        if not raw_name:
            return ""

        import re

        name = str(raw_name).strip()
        name = re.sub(r"\s+", " ", name)

        # ESPN text blobs can leak a team prefix into the player capture.
        team_tokens = [
            'hawks', 'celtics', 'nets', 'hornets', 'bulls', 'cavaliers',
            'mavericks', 'nuggets', 'pistons', 'warriors', 'rockets', 'pacers',
            'clippers', 'lakers', 'grizzlies', 'heat', 'bucks', 'timberwolves',
            'pelicans', 'knicks', 'thunder', 'magic', '76ers', 'suns',
            'trail blazers', 'blazers', 'kings', 'spurs', 'raptors', 'jazz', 'wizards'
        ]
        if '. ' in name:
            left, right = name.split('. ', 1)
            left_norm = left.strip().lower()
            if left_norm in team_tokens or left_norm.endswith('ers') or left_norm.endswith('s'):
                name = right.strip()

        # Remove stray punctuation fragments that are not part of names.
        name = re.sub(r"^[^A-Za-z]+", "", name)
        name = re.sub(r"[^A-Za-z\s'\.-]", "", name).strip()

        # Keep only plausible person names (at least first + last token).
        if len(name.split()) < 2:
            return ""
        return name

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
                cleaned = []
                for inj in cached.get('data', []):
                    if not isinstance(inj, dict):
                        continue
                    name = self._sanitize_injury_player_name(inj.get('name', ''))
                    if not name:
                        continue
                    c = dict(inj)
                    c['name'] = name
                    cleaned.append(c)
                return cleaned
        
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
        
        url_abbrev = team_abbrev.lower()
        if team_abbrev == 'UTA':
            url_abbrev = 'utah'
        elif team_abbrev == 'NOP':
            url_abbrev = 'no' # ESPN sometimes uses 'no' for New Orleans
        elif team_abbrev == 'NYK':
            url_abbrev = 'ny' # ESPN sometimes uses 'ny' for Knicks
        elif team_abbrev == 'GSW':
            url_abbrev = 'gs' # ESPN sometimes uses 'gs' for Warriors
        elif team_abbrev == 'SAS':
            url_abbrev = 'sa' # ESPN sometimes uses 'sa' for Spurs
            
        slug = espn_team_slugs.get(team_abbrev)
        if not slug:
            return []
        
        try:
            import re
            # Use the corrected url_abbrev
            url = f"https://www.espn.com/nba/team/injuries/_/name/{url_abbrev}/{slug}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(' ', strip=True)
            
            # Use regex to find injury patterns: "PlayerName Position Status StatusType"
            # Pattern matches: "Bruce Brown G Status Day-to-day", "VJ Edgecombe G Status Out"
            # Handles all-caps initials (VJ, TJ, KJ) and dotted initials (T.J., D.J.)
            pattern = r"([A-Z][a-zA-Z\'\.\-]+(?:\s+[A-Z][a-zA-Z\'\.\-]+)+)\s+([GFC])\s+Status\s+(Out|Day-to-day|Doubtful|Questionable|Probable)"
            matches = re.findall(pattern, text)
            
            seen_names = set()
            for match in matches:
                player_name = self._sanitize_injury_player_name(match[0])
                if not player_name:
                    continue
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
            all_players = get_all_nba_players()
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
        # V21.1: DepthChartManager for "Next Man Up" scenarios
        self.depth_manager = DepthChartManager()
    
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
        
        season_avg = 0.0
        if len(df) > 0:
            try:
                season_avg = df[stat_col].mean()
            except Exception:
                season_avg = 0.0

        l10_avg = 0.0
        l10_subset = df.tail(10)
        if len(l10_subset) > 0:
            try:
                l10_avg = l10_subset[stat_col].mean()
            except Exception:
                l10_avg = 0.0
        
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

        # NOTE: usage_rate is now calculated in build_feature_vector() where
        # actual team_pace is available for proper USG% normalization.

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
            'season_avg': float(season_avg),
            'l10_avg': float(l10_avg)
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
    
    def _calculate_usage_rate(self, df: pd.DataFrame, lookback: int, team_pace: float) -> float:
        """
        Calculate proper NBA Usage Rate using actual team pace.
        
        Standard formula: USG% = (FGA + 0.44*FTA + TOV) * (TeamMIN/5) / (PlayerMIN * TeamPoss)
        
        TeamMIN = 5 players * 48 min = 240, so TeamMIN/5 = 48.
        TeamPoss ≈ team_pace (possessions per 48 min).
        
            USG% = (FGA + 0.44*FTA + TOV) * 48 / (PlayerMIN * team_pace)
        
        Expected output: ~0.15-0.20 for role players, ~0.25-0.35 for stars, max ~0.42.
        """
        recent = df.tail(lookback)
        if len(recent) == 0:
            return 0.0
        try:
            required_cols = ['FGA', 'FTA', 'TOV', 'MIN_FLOAT']
            if not all(c in recent.columns for c in required_cols):
                return 0.0
            
            pace = max(team_pace, 80.0)  # Floor at 80 to avoid division issues
            possessions_used = recent['FGA'] + 0.44 * recent['FTA'] + recent['TOV']
            # USG% = possessions_used * 48 / (MIN * pace)
            usg_series = (possessions_used * 48.0) / (recent['MIN_FLOAT'] * pace)
            # Clean infinites/NaNs
            usg_series = usg_series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            # Clamp to realistic range [0, 0.50]
            usg_series = usg_series.clip(0.0, 0.50)
            return float(usg_series.mean())
        except Exception:
            return 0.0

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
        player_team_id: int = None,
        player_team_abbrev: str = None,
        data_loader: 'DataLoader' = None,
        player_position: str = None,
        game_season: str = None,  # V21.1: Season for historical DvP lookup
        is_backtest: bool = False  # V21.4: True = use ground truth, False = use depth chart + injuries
    ) -> FeatureVector: 
        """
        Build pure empirical feature vector for ML models.
        
        V21.1: Added game_season parameter to fix time-travel bug.
        When building features for a 2023 game, DvP rank now uses
        2023-24 defensive stats instead of current season.
        """
        
        # --- Lazy-load fatigue_index.json (one-time per process) ---
        if not _FATIGUE_CACHE:
            _fatigue_path = DATA_DIR / "fatigue_index.json"
            if _fatigue_path.exists():
                try:
                    import json as _json
                    _raw = _json.loads(_fatigue_path.read_text(encoding="utf-8"))
                    _FATIGUE_CACHE.update(_raw.get("teams", {}))
                    logger.debug(f"Loaded fatigue_index.json ({len(_FATIGUE_CACHE)} teams)")
                except Exception as _fe:
                    logger.warning(f"Could not load fatigue_index.json: {_fe}")

        # Initialize data quality tracker
        data_quality = DataQuality()
        
        # === CORE STATISTICAL FEATURES (raw from game logs) ===
        stats = self.calculate_statistical_features(df, stat_col, lookback, data_quality)
        
        # === OPPONENT DEFENSIVE RATING ===
        opponent_drtg_season = avg_def
        opponent_pace = 100.0
        team_pace = 100.0
        
        if len(team_stats) > 0:
            if opponent_id in team_stats.index:
                opp = team_stats.loc[opponent_id]
                opponent_drtg_season = opp.get('DEF_RATING', avg_def)
                opp_pace_val = opp.get('PACE', None)
                if opp_pace_val is not None:
                    opponent_pace = opp_pace_val
                else:
                    data_quality.used_default_pace = True
                    data_quality.add_warning("Using default opponent pace")
            else:
                data_quality.used_default_def_rating = True
                data_quality.used_default_pace = True
                data_quality.add_warning(f"Opponent {opponent_id} not in team_stats")
            
            if player_team_id and player_team_id in team_stats.index:
                player_team = team_stats.loc[player_team_id]
                team_pace_val = player_team.get('PACE', None)
                if team_pace_val is not None:
                    team_pace = team_pace_val
                else:
                    data_quality.used_default_pace = True
                    data_quality.add_warning("Using default team pace")
            elif player_team_id:
                data_quality.used_default_pace = True
                data_quality.add_warning(f"Player team {player_team_id} not in team_stats")
        
        if len(team_stats) == 0:
            data_quality.missing_team_stats = True
            data_quality.used_default_def_rating = True
            data_quality.used_default_pace = True
            data_quality.add_warning("Team stats unavailable")
        
        # === DAYS REST ===
        actual_days_rest = days_rest if days_rest >= 0 else (0 if is_b2b else 1)
        
        # === ABSENCE FEATURES ===
        team_out_ppg = -1.0
        team_out_count = -1
        opp_out_ppg = -1.0
        opp_out_count = -1
        
        # V21.7: Skip live injury scrape during backtest — absence features
        # are computed from bulk game logs and injected by the backtest loop.
        if player_team_abbrev and opponent_abbrev and not is_backtest:
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
        
        # === TRUE-SHOOTING% FEATURES ===
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
            feat_ts_pct = feat_ts_pct
            feat_ts_pct_delta = feat_ts_pct_delta
        
        # === HEAD-TO-HEAD HISTORY ===
        h2h_avg = -1.0
        try:
            if opponent_abbrev and 'MATCHUP' in df.columns:
                h2h_games = df[df['MATCHUP'].str.contains(opponent_abbrev, case=False, na=False)]
                if len(h2h_games) > 0:
                    h2h_avg = h2h_games[stat_col].mean()
                else:
                    h2h_avg = stats['ema']
        except Exception:
            h2h_avg = stats['ema']

        # === BUILD FEATURE VECTOR ===
        # Compute efficiency per minute (avoid divide-by-zero; require meaningful minutes)
        try:
            avg_minutes_local = float(stats.get('avg_minutes', 0.0) or 0.0)
            ema_local = float(stats.get('ema', 0.0) or 0.0)
            eff_per_min = ema_local / avg_minutes_local if avg_minutes_local > 5.0 else 0.0
        except Exception:
            eff_per_min = 0.0

        # Derived features (new formulas per spec)
        try:
            # Fatigue load: smoother inverse of rest days
            fatigue_load = avg_minutes_local / (actual_days_rest + 0.5)
        except Exception:
            fatigue_load = 0.0

        try:
            season_avg = float(stats.get('season_avg', 0.0) or 0.0)
            l10_avg = float(stats.get('l10_avg', 0.0) or 0.0)
            # Form gap: relative gap normalized by season average (avoid division by zero)
            form_gap = (l10_avg - season_avg) / max(1.0, season_avg) if season_avg is not None else 0.0
        except Exception:
            form_gap = 0.0

        # === NEW FEATURES: Hit Rates and DvP ===
        # Get additional features from DataEngine
        # V21.1 FIX: Pass game_season for historical DvP lookup (fixes time-travel bug)
        player_data = {}
        if data_loader and hasattr(data_loader, 'data_engine'):
            try:
                # Use game_season if provided, otherwise fall back to current season
                lookup_season = game_season if game_season else self.config.CURRENT_SEASON
                player_data = data_loader.data_engine.fetch_player_data(
                    player_id=player_id,
                    opponent_id=opponent_id,
                    market=market,
                    line=line,
                    season=lookup_season,
                    player_position=player_position
                )
            except Exception as e:
                logger.warning(f"Failed to fetch additional player data: {e}")
                player_data = {}

        # Calculate hit rates from historical data
        try:
            if df.empty:
                feat_hit_rate_l10 = 0.0
                feat_hit_rate_l20 = 0.0
            else:
                # Sort by date descending
                temp_df = df.sort_values('GAME_DATE', ascending=False)
                
                # Calculate ACTUAL_VALUE
                if market == 'PRA':
                    temp_df['ACTUAL_VALUE'] = temp_df['PTS'] + temp_df['REB'] + temp_df['AST']
                elif market == 'PR':
                    temp_df['ACTUAL_VALUE'] = temp_df['PTS'] + temp_df['REB']
                elif market == 'PA':
                    temp_df['ACTUAL_VALUE'] = temp_df['PTS'] + temp_df['AST']
                elif market == 'RA':
                    temp_df['ACTUAL_VALUE'] = temp_df['REB'] + temp_df['AST']
                else:
                    # Single stat markets
                    temp_df['ACTUAL_VALUE'] = temp_df[market]
                
                # Hit rates
                last_10 = temp_df.head(10)
                hits_10 = sum(
                    1 for _, game in last_10.iterrows()
                    if 'line' in game and pd.notna(game['line']) and game['line'] > 0
                    and game['ACTUAL_VALUE'] > game['line']
                )
                feat_hit_rate_l10 = hits_10 / float(len(last_10)) if len(last_10) > 0 else -1.0
                
                last_20 = temp_df.head(20)
                hits_20 = sum(
                    1 for _, game in last_20.iterrows()
                    if 'line' in game and pd.notna(game['line']) and game['line'] > 0
                    and game['ACTUAL_VALUE'] > game['line']
                )
                feat_hit_rate_l20 = hits_20 / float(len(last_20)) if len(last_20) > 0 else -1.0
        except Exception as e:
            logger.warning(f"Failed to calculate hit rates: {e}")
            feat_hit_rate_l10 = 0.0
            feat_hit_rate_l20 = 0.0

        feat_opp_rank_vs_pos = player_data.get('feat_opp_rank_vs_pos', 15)

        # === V21.3 GROUND TRUTH DEPTH LOGIC ===
        # Uses actual starter data from boxscores (no heuristics!)
        #
        # Mode A: Historical (Training) - uses starter cache (ground truth from boxscore)
        # Mode B: Inference (Live) - uses depth chart + injury report
        
        feat_depth_rank = 4  # Default: Deep Bench / Not Found
        game_id = None       # Will be set if historical data available
        
        logger.debug(f"[DEPTH] Starting depth logic for {player_name}: df_empty={df.empty}, team={player_team_abbrev}, is_backtest={is_backtest}")
        
        try:
            # --- Step A: Extract GAME_ID and raw minutes from DataFrame row ---
            game_min = None
            
            if not df.empty:
                # V21.6 FIX: Use tail(1) for MOST RECENT historical game
                # (df is sorted ascending by date, so tail(1) = latest game)
                recent_row = df.tail(1)
                
                # Extract Game ID for starter cache lookup (backtest only)
                for col in ['GAME_ID', 'Game_ID', 'gameId']:
                    if col in recent_row.columns:
                        game_id = str(recent_row[col].iloc[0])
                        break
                
                # Extract minutes from most recent game
                if 'MIN' in recent_row.columns:
                    min_val = recent_row['MIN'].iloc[0]
                    if pd.notna(min_val):
                        if isinstance(min_val, str) and ':' in min_val:
                            parts = min_val.split(':')
                            game_min = float(parts[0]) + float(parts[1]) / 60.0
                        else:
                            game_min = float(min_val)
                elif 'min' in recent_row.columns:
                    min_val = recent_row['min'].iloc[0]
                    if pd.notna(min_val):
                        if isinstance(min_val, str) and ':' in min_val:
                            parts = min_val.split(':')
                            game_min = float(parts[0]) + float(parts[1]) / 60.0
                        else:
                            game_min = float(min_val)
                elif 'MIN_FLOAT' in recent_row.columns:
                    min_val = recent_row['MIN_FLOAT'].iloc[0]
                    if pd.notna(min_val):
                        game_min = float(min_val)
            
            # Get the calculated avg_minutes from stats
            feat_avg_minutes = float(stats.get('avg_minutes', 0.0) or 0.0)
            
            # --- Step B: DNP Check (0 minutes = didn't play) ---
            if game_min is not None and game_min == 0:
                feat_depth_rank = 3  # DNP / Deep Bench
                logger.debug(f"DNP detected for {player_name}: MIN=0, setting depth_rank=3")
            
            # --- Step C: Backtest Mode - Use Starter Cache (GROUND TRUTH) ---
            # Only use this for backtesting where we want actual starter status
            elif is_backtest and game_id is not None:
                from starter_cache import get_starter_cache
                starter_cache = get_starter_cache()
                
                # Check if we have starter data from offline cache
                started = starter_cache.get_starter_status(game_id, player_id)
                logger.debug(f"[DEPTH] Step C (Backtest): game={game_id}, player={player_id}, started={started}")
                
                if started is not None:
                    # Starter cache has this game (built offline from bulk data)
                    feat_depth_rank = 1 if started else 2
                    logger.debug(f"Starter cache hit for {player_name}: game={game_id}, started={started} -> rank={feat_depth_rank}")
                else:
                    # Cache miss - game not in bulk data or cache not built
                    # Fall back to minutes-based inference from the row data
                    if game_min is not None:
                        feat_depth_rank = 1 if (game_min >= 26 or feat_avg_minutes >= 30) else 2
                        logger.debug(f"Starter cache miss for {game_id}, using minutes fallback: {game_min:.1f} min -> rank {feat_depth_rank}")
                    else:
                        feat_depth_rank = 2 if feat_avg_minutes >= 20 else 3
                        logger.debug(f"No minutes data for {player_name} in {game_id}, using avg_min fallback")
            
            # --- Step D: Inference Mode (Live Betting) ---
            # No backtest mode - use depth chart + injury report for upcoming game
            elif player_team_abbrev:
                # Get base depth rank from ESPN depth chart
                base_depth_rank = self.depth_manager.get_player_depth_rank(player_name, player_team_abbrev)
                
                # V21.2 FIX: If depth chart lookup failed (rank 4) but player averages 28+ min,
                # they're clearly a starter - ESPN sometimes has name mismatches
                if base_depth_rank == 4 and feat_avg_minutes >= 28.0:
                    base_depth_rank = 1  # Override to Starter
                    logger.info(f"Depth chart fallback: {player_name} avg {feat_avg_minutes:.1f} min -> Starter")
                elif base_depth_rank == 4 and feat_avg_minutes >= 20.0:
                    base_depth_rank = 2  # Override to Backup
                    logger.info(f"Depth chart fallback: {player_name} avg {feat_avg_minutes:.1f} min -> Backup")
                
                # Dynamic Promotion: If player is backup (Rank 2+), check if starters are OUT
                if base_depth_rank > 1:
                    # Get list of injured players marked OUT on the team
                    team_injuries = self.injury_manager.get_team_injury_report(player_team_abbrev)
                    out_players = [
                        inj['name'] for inj in team_injuries 
                        if inj.get('status') in ('OUT', 'DOUBTFUL')
                    ]
                    
                    if out_players:
                        # Use effective rank considering injuries
                        feat_depth_rank = self.depth_manager.get_effective_depth_rank(
                            player_name=player_name,
                            team_abbrev=player_team_abbrev,
                            injured_players=out_players
                        )
                        if feat_depth_rank != base_depth_rank:
                            logger.info(f"Dynamic Promotion: {player_name} ({player_team_abbrev}) "
                                       f"promoted from Rank {base_depth_rank} to Rank {feat_depth_rank} "
                                       f"(starters OUT: {out_players})")
                    else:
                        feat_depth_rank = base_depth_rank
                else:
                    feat_depth_rank = base_depth_rank
                    
        except Exception as e:
            logger.warning(f"Failed to compute depth rank for {player_name}: {e}")
            feat_depth_rank = 4

        return FeatureVector(
            # Identifiers
            player_id=player_id,
            player_name=player_name,
            opponent_abbrev=opponent_abbrev,
            market=market,

            # Raw observables
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

            # Context
            opponent_pace=opponent_pace,
            team_pace=team_pace,
            trend_5g=stats['trend_5g'],
            home_avg=stats['home_avg'],
            away_avg=stats['away_avg'],

            # New Features
            feat_ts_pct=feat_ts_pct,
            feat_ts_pct_delta=feat_ts_pct_delta,
            team_out_ppg=team_out_ppg,
            team_out_count=team_out_count,
            opp_out_ppg=opp_out_ppg,
            opp_out_count=opp_out_count,
            feat_min_volatility=stats.get('min_volatility', 0.0),
            feat_foul_rate=stats.get('foul_rate', 0.0),
            feat_cv=stats.get('cv', 0.0),

            # Market Identity
            market_scoring=1 if market == 'PTS' else 0,
            market_counting=1 if market in ['REB', 'AST'] else 0,
            market_combo=1 if market in ['PRA', 'PR', 'PA', 'RA'] else 0,

            data_quality=data_quality,

            # V20.4 NEW FEATURES
            feat_usage_rate=self._calculate_usage_rate(df, lookback, team_pace),
            feat_h2h_avg=h2h_avg,

            #  ADDED NEW FEATURES HERE 
            feat_season_avg=season_avg,
            feat_l10_avg=l10_avg,
            feat_fatigue_load=fatigue_load,
            feat_form_gap=form_gap,

            # V21.6 NEW: Travel / altitude (from morning_refresh.py)
            feat_miles_96h=float(_FATIGUE_CACHE.get(player_team_abbrev, {}).get("miles_96h", 0.0)),
            # Altitude = venue city: if player is home → player's city, if away → opponent's city
            feat_altitude_next=_ALTITUDE_FT_LOOKUP.get(player_team_abbrev or "", 0) if is_home else _ALTITUDE_FT_LOOKUP.get(opponent_abbrev, 0),

            #  NEW: Efficiency per minute
            feat_eff_per_min=eff_per_min,

            #  NEW: Hit Rate Features
            feat_hit_rate_l10=feat_hit_rate_l10,
            feat_hit_rate_l20=feat_hit_rate_l20,

            #  NEW: Defense vs. Position
            feat_opp_rank_vs_pos=feat_opp_rank_vs_pos,

            # V21.2 NEW: Depth Chart Rank (Composite Logic)
            feat_depth_rank=feat_depth_rank,
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
    - Variance model predicts [stat] (uncertainty)
    - Probability via Quantile Interpolation (no Gaussian assumption)
    - Calibration curves correct probability biases
    - Ablation mode logs feature importance
    
    Model Files (in data/ directory):
    - nba_quantile_{group}.pkl: Dict with {'q05': model, 'q50': model, 'q95': model}
    - nba_calibrator_{group}.pkl: Isotonic calibration curve
    
    V21.5 QUANTILE UPGRADE (Calibration Fix):
    - Widened quantiles from q10/q90 (80%) to q05/q95 (90% interval)
    - Empirical data showed 78% coverage on 80% interval = underconfident
    - P(Over) calculated via linear interpolation between quantiles
    - Removes Gaussian assumption - handles skewed player distributions
    - Synthetic std = (q95 - q05) / 3.29 for UI compatibility
    - Backward compatible: loads old q10/q90 models if q05/q95 not found
    """
    def __init__(self, config: Config = CONFIG, tracker: 'Tracker' = None):
        self.config = config
        self.tracker = tracker
        # Per-market quantile models: {market: ({'q10': model, 'q50': model, 'q90': model}, feature_names)}
        # Lookup order: exact market (PTS) -> group (scoring) -> universal
        self.ml_models: Dict[str, Tuple[Dict[str, Any], List[str]]] = {}
        # Calibration curves: {market_or_group: IsotonicRegression}
        self.calibrators: Dict[str, Any] = {}
        self.ml_model = None  # Legacy fallback (single model)
        self.model_features: List[str] = []
        self._load_ml_models()
        self._load_calibrators()

    def _load_ml_models(self):
        """
        Load quantile regression models with priority:
        1. Per-market models: nba_quantile_{MARKET}.pkl (e.g., PTS, REB, AST)
        2. Group models: nba_quantile_{GROUP}.pkl (scoring, counting, combo, rare)
        3. Universal model: nba_quantile_universal.pkl
        4. Legacy models: nba_model_{group}.pkl (wrapped as pseudo-quantile)
        
        V21.1 PER-MARKET: Each market can have its own specialized model.
        """
        # All possible markets
        individual_markets = ['PTS', 'REB', 'AST', 'PRA', 'PR', 'PA', 'RA', '3PM', 'FG3M', 'STL', 'BLK']
        market_groups = ['scoring', 'counting', 'combo', 'rare', 'universal']
        
        loaded_count = 0
        per_market_count = 0
        group_count = 0
        
        def _extract_features_from_model(model) -> List[str]:
            """Extract feature names from a trained model."""
            if model and hasattr(model, 'feature_names_in_'):
                return [str(f) for f in model.feature_names_in_]
            elif model and hasattr(model, 'get_booster'):
                booster_names = model.get_booster().feature_names
                return [str(f) for f in booster_names] if booster_names else []
            return []
        
        def _load_quantile_model(key: str, path: Path) -> bool:
            """Attempt to load a quantile model dict from path."""
            nonlocal loaded_count, self
            if not path.exists():
                return False
            try:
                quantile_dict = joblib.load(path)
                
                # Validate it's a proper quantile dict
                if isinstance(quantile_dict, dict) and 'q50' in quantile_dict:
                    # Detect quantile width: new models have q05/q95, old have q10/q90
                    is_wide_quantile = 'q05' in quantile_dict or 'q95' in quantile_dict
                    quantile_dict['_wide_quantile'] = is_wide_quantile
                    
                    feature_names = _extract_features_from_model(quantile_dict.get('q50'))
                    self.ml_models[key] = (quantile_dict, feature_names)
                    loaded_count += 1
                    
                    if not self.model_features and feature_names:
                        self.model_features = feature_names
                    if is_wide_quantile:
                        logger.info(f" V21.5: Wide quantile model (q05/q95) for {key}")
                    return True
                else:
                    logger.warning(f"Invalid quantile model format for {key}")
            except Exception as e:
                logger.warning(f"Failed to load {key} quantile model: {e}")
            return False
        
        def _load_legacy_model(key: str, path: Path) -> bool:
            """Attempt to load a legacy mean-only model and wrap as pseudo-quantile."""
            nonlocal loaded_count, self
            if not path.exists():
                return False
            try:
                model = joblib.load(path)
                feature_names = _extract_features_from_model(model)
                
                # Wrap legacy model as pseudo-quantile (q50 only)
                legacy_quantile = {'q50': model, 'q10': None, 'q90': None, '_legacy': True}
                self.ml_models[key] = (legacy_quantile, feature_names)
                loaded_count += 1
                logger.info(f" Loaded legacy model for {key} (wrapped as quantile)")
                
                if not self.model_features and feature_names:
                    self.model_features = feature_names
                return True
            except Exception as e:
                logger.warning(f"Failed to load {key} legacy model: {e}")
            return False
        
        # 1. Load per-market models first (highest priority)
        for market in individual_markets:
            quantile_path = DATA_DIR / f"nba_quantile_{market}.pkl"
            if _load_quantile_model(market, quantile_path):
                per_market_count += 1
                logger.info(f" Loaded per-market quantile model: {market}")
        
        # 2. Load group models (fallback for markets without specific model)
        for group in market_groups:
            quantile_path = DATA_DIR / f"nba_quantile_{group}.pkl"
            legacy_path = DATA_DIR / f"nba_model_{group}.pkl"
            
            if _load_quantile_model(group, quantile_path):
                group_count += 1
                logger.info(f" Loaded group quantile model: {group}")
            elif _load_legacy_model(group, legacy_path):
                group_count += 1
        
        # Log summary
        if loaded_count > 0:
            logger.info(f" ML MODELS: {per_market_count} per-market, {group_count} group models")
            if self.model_features:
                logger.info(f"   Models expect {len(self.model_features)} features")
        else:
            logger.error(" NO ML MODELS FOUND - Train models first!")
            if self.config.ML_REQUIRE_MODEL:
                logger.error("   Predictions will FAIL until models are trained.")
        
        # 3. Legacy fallback (only if no models found at all)
        if not self.ml_models:
            model_path = DATA_DIR / "nba_model.pkl"
            if model_path.exists():
                try:
                    self.ml_model = joblib.load(model_path)
                    if hasattr(self.ml_model, 'feature_names_in_'):
                        self.model_features = list(self.ml_model.feature_names_in_)
                    logger.info(" Loaded legacy universal ML model")
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
            logger.info(f" Loaded {loaded_count} calibration curves")
        else:
            logger.info(" No calibration curves found (using raw probabilities)")

    def _get_model_for_market(self, market: str) -> Tuple[Dict[str, Any], List[str]]:
        """
        Get the appropriate quantile models and feature names for a market.
        
        V21.1 PER-MARKET: Lookup priority:
        1. Exact market match (e.g., 'PTS' -> nba_quantile_PTS.pkl)
        2. Market group (e.g., 'PTS' -> 'scoring' -> nba_quantile_scoring.pkl)
        3. Universal fallback (nba_quantile_universal.pkl)
        4. Legacy single model (wrapped as pseudo-quantile)
        
        Returns: (quantile_dict, feature_names) where quantile_dict has 'q10', 'q50', 'q90'.
        """
        # 1. Check for per-market model first
        if market in self.ml_models:
            return self.ml_models[market]
        
        # 2. Check for group model
        group = ML_MARKET_GROUPS.get(market, 'universal')
        if group in self.ml_models:
            return self.ml_models[group]
        
        # 3. Universal fallback
        if 'universal' in self.ml_models:
            return self.ml_models['universal']
        
        # 4. Legacy fallback: wrap single model
        if self.ml_model is not None:
            return ({'q50': self.ml_model, 'q10': None, 'q90': None, '_legacy': True}, self.model_features)
        
        return ({}, self.model_features)

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
            # V20.3 NEW: Behavior & Risk features
            'feat_min_volatility': float(features.feat_min_volatility),
            'feat_foul_rate': float(features.feat_foul_rate),
            'feat_cv': float(features.feat_cv),
            
            #  V20.4 NEW: ADD THESE LINES TO FIX THE WARNINGS
            'feat_usage_rate': float(getattr(features, 'feat_usage_rate', 0.0)),
            'feat_h2h_avg': float(getattr(features, 'feat_h2h_avg', 0.0)),
            'feat_season_avg': float(getattr(features, 'feat_season_avg', 0.0)),
            'feat_l10_avg': float(getattr(features, 'feat_l10_avg', 0.0)),
            # V20.5 NEW: Hit-rate & DvP features (added to silence model warnings)
            'feat_hit_rate_l10': float(getattr(features, 'feat_hit_rate_l10', 0.0)),
            'feat_hit_rate_l20': float(getattr(features, 'feat_hit_rate_l20', 0.0)),
            'feat_opp_rank_vs_pos': float(getattr(features, 'feat_opp_rank_vs_pos', 15.0)),
            # V21.1 NEW: Depth chart rank
            'feat_depth_rank': float(getattr(features, 'feat_depth_rank', 4.0)),
            # V21.7: Travel features
            'feat_miles_96h': float(getattr(features, 'feat_miles_96h', 0.0)),
            'feat_altitude_next': float(getattr(features, 'feat_altitude_next', 0)),
            # Backcompat/missing feature fallbacks (silence older model warnings)
            'feat_eff_per_min': float(getattr(features, 'feat_eff_per_min', 0.0)),
            'feat_fatigue_load': float(getattr(features, 'feat_fatigue_load', 0.0)),
            'feat_form_gap': float(getattr(features, 'feat_form_gap', 0.0)),
            'feat_market_rare': float(getattr(features, 'market_rare', 0.0)),
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
        V21 QUANTILE: Get P(Over) probability via quantile interpolation.
        
        Uses the full quantile regression output and returns P(Over).
        This is a convenience wrapper around get_ml_regression_output().
        
        Returns: P(Over) probability (for compatibility with existing code)
        """
        regression = self.get_ml_regression_output(features)
        
        if not regression.get('has_model'):
            return None
        
        p_over = regression.get('p_over')
        return p_over if p_over is not None else None
    
    def get_ml_regression_output(self, features: FeatureVector) -> Dict[str, Any]:
        """
        V21.5 QUANTILE REGRESSION: Get predictions from q05, q50, q95 models.
        
        Returns dict with:
        - predicted_value: Median (q50) prediction
        - predicted_std: Synthetic std = (q95 - q05) / 3.29
        - p_over: P(stat > line) via quantile interpolation
        - p_under: P(stat <= line)
        - confidence_interval: (q05, q95) - true 90% interval
        - quantiles: {'q_low': val, 'q50': val, 'q_high': val}
        
        Backward compatible with old q10/q90 models.
        """
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
            
            # Check for old classifier model (backwards compat)
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
            
            # === V21 QUANTILE PREDICTIONS ===
            pred_50 = float(q50_model.predict(X)[0])
            result['predicted_value'] = pred_50
            result['quantiles']['q50'] = pred_50
            
            # Get quantile predictions (q05/q95 for new models, q10/q90 for old)
            is_wide = quantile_dict.get('_wide_quantile', False)
            is_legacy = quantile_dict.get('_legacy', False)
            
            if is_wide:
                # V21.5: Wide quantiles (90% interval)
                q_low_model = quantile_dict.get('q05')
                q_high_model = quantile_dict.get('q95')
                z_divisor = 3.29  # z-span for 90% of normal distribution
                fallback_z = 1.645  # ~5th/95th percentile
            else:
                # Old models: Narrow quantiles (80% interval)
                q_low_model = quantile_dict.get('q10')
                q_high_model = quantile_dict.get('q90')
                z_divisor = 2.56  # z-span for 80% of normal distribution
                fallback_z = 1.28  # ~10th/90th percentile
            
            if q_low_model is not None and q_high_model is not None and not is_legacy:
                pred_low = float(q_low_model.predict(X)[0])
                pred_high = float(q_high_model.predict(X)[0])
            else:
                # Legacy model fallback: estimate quantiles from empirical std
                emp_std = features.std if (features.std is not None and features.std > 0) else 1.0
                pred_low = pred_50 - fallback_z * emp_std
                pred_high = pred_50 + fallback_z * emp_std
            
            # V22 FIX: When quantile models cross, the model is confused.
            # Instead of silently clamping (which hides uncertainty), widen the
            # interval using empirical std as a floor to signal low confidence.
            if pred_low > pred_50 or pred_high < pred_50:
                logger.debug(f"Quantile crossing detected: q_low={pred_low:.1f}, q50={pred_50:.1f}, q_high={pred_high:.1f} — widening interval")
                emp_std = features.std if (features.std is not None and features.std > 0) else 3.0
                fallback_z_cross = z_divisor / 2  # half the z-span to get per-side z
                pred_low = pred_50 - fallback_z_cross * emp_std
                pred_high = pred_50 + fallback_z_cross * emp_std
            pred_low = max(0, pred_low)
            pred_high = max(pred_low + 0.1, pred_high)  # Ensure minimum spread
            
            result['quantiles']['q_low'] = pred_low
            result['quantiles']['q_high'] = pred_high
            result['is_wide_quantile'] = is_wide
            
            # Synthetic std for UI compatibility
            spread = pred_high - pred_low
            predicted_std = spread / z_divisor if spread > 0 else 1.0
            result['predicted_std'] = predicted_std
            
            # Confidence interval (90% for wide, 80% for old)
            result['confidence_interval'] = (max(0, pred_low), pred_high)
            
            # === V21.5 PROBABILITY VIA QUANTILE INTERPOLATION ===
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
        """
        V22 CALIBRATION FIX: Split-normal CDF replaces linear interpolation.
        
        The old linear interpolation inflated probabilities near the median
        because it assumed a uniform distribution between quantile points.
        Real NBA stat distributions are peaked near the median.
        
        Fix: Fit a split-normal distribution to the three quantile points
        (q_low, q50, q_high) and compute P(Over) using the proper CDF.
        
        Split-normal handles asymmetric distributions:
        - sigma_lower = (q50 - q_low) / z_value  (left half)
        - sigma_upper = (q_high - q50) / z_value  (right half)
        - When line < q50: use sigma_lower for CDF
        - When line > q50: use sigma_upper for CDF
        """
        # z-values corresponding to the quantile levels
        if is_wide:
            z_val = 1.645  # z for 5th/95th percentile
        else:
            z_val = 1.28   # z for 10th/90th percentile
        
        # Handle degenerate cases
        if q_low >= q_high or q_high - q_low < 0.1:
            # No meaningful spread — return 0.50 (no edge)
            return 0.50
        
        # Compute split-normal sigmas
        sigma_lower = max((q50 - q_low) / z_val, 0.1)
        sigma_upper = max((q_high - q50) / z_val, 0.1)
        
        # Use the appropriate sigma for each side of the median
        if line <= q50:
            z_score = (line - q50) / sigma_lower
        else:
            z_score = (line - q50) / sigma_upper
        
        # P(Over) = 1 - Phi(z) where Phi is the standard normal CDF
        p_over = 1.0 - stats.norm.cdf(z_score)
        
        # Technical bounds only (no artificial caps)
        return float(max(0.02, min(0.98, p_over)))
    
    def get_ml_prediction_details(self, features: FeatureVector) -> Dict[str, Any]:
        """
        V21 QUANTILE: Get detailed ML prediction info for UI display.
        Returns quantile regression outputs and model metadata.
        """
        market = getattr(features, 'market', None) or 'PTS'
        group = ML_MARKET_GROUPS.get(market, 'universal')
        quantile_dict, expected_features = self._get_model_for_market(market)
        
        # Check if calibrator exists for this market group
        has_calibrator = group in self.calibrators if hasattr(self, 'calibrators') else False
        
        # Check if we have full quantile models or legacy
        has_full_quantile = (
            quantile_dict and 
            (quantile_dict.get('q05') is not None or quantile_dict.get('q10') is not None) and 
            (quantile_dict.get('q95') is not None or quantile_dict.get('q90') is not None) and
            not quantile_dict.get('_legacy', False)
        )
        
        # Get regression output
        regression = self.get_ml_regression_output(features)
        
        details = {
            'market': market,
            'model_group': group,
            'has_model': regression.get('has_model', False),
            'has_quantile_models': has_full_quantile,
            'has_calibrator': has_calibrator,
            'is_regression': regression.get('is_regression', False),
            'is_quantile': regression.get('is_quantile', False),
            'num_features': len(expected_features) if expected_features else 0,
            # V21 Quantile outputs
            'predicted_value': regression.get('predicted_value'),
            'predicted_std': regression.get('predicted_std'),
            'p_over': regression.get('p_over'),
            'p_under': regression.get('p_under'),
            'confidence_interval': regression.get('confidence_interval'),
            'quantiles': regression.get('quantiles', {}),
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
        Probability calculations use CDF: P(Over) = 1 - ((line - ) / )
        
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
        # Defensive predicted std: fall back to features.std or 1.0 if missing
        ml_std_safe = ml_std if (ml_std is not None and ml_std > 0) else (features.std if (features.std is not None and features.std > 0) else 1.0)
        ci = (
            ci_low if ci_low is not None else max(0, adjusted_proj - ml_std_safe * 1.645),
            ci_high if ci_high is not None else adjusted_proj + ml_std_safe * 1.645
        )
        
        return ProjectionResult(
            base_projection=float(ml_proj),
            final_projection=float(adjusted_proj),
            confidence_interval=ci,
            adjustments={
                'predicted_std': float(ml_std_safe),
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
            # IsotonicRegression.predict() takes 1D array
            calibrated = float(calibrator.predict([raw_prob])[0])
            return max(0.01, min(0.99, calibrated))
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
    - Pure CDF calculation: P(Over) = 1 - ((line - ) / )
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
        
        Formula: P(Over) = 1 - ((line - ) / )
        
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
    V21.6: ML-Driven Decision Engine — zero heuristics.
    
    All decisions derive from:
    - Kelly Criterion (mathematical optimal sizing)
    - ML model outputs (probability, projection, quantile-derived std)
    
    NO hardcoded multipliers, tiers, caps, or magic numbers.
    """
    
    # Fractional Kelly to control bankroll volatility.
    # 0.25 = quarter-Kelly, a standard conservative choice in the literature.
    # This is a configurable risk preference, not a heuristic.
    KELLY_FRACTION: float = 0.25
    
    def __init__(self, config: Config = CONFIG):
        self.config = config
    
    def calculate_ev(self, prob: float, odds: float) -> float:
        """Calculate expected value.  EV = p(o-1) - (1-p). Pure math."""
        return (prob * (odds - 1)) - (1 - prob)
    
    def calculate_kelly_stake(
        self, 
        prob: float,
        odds: float,
        bankroll: float
    ) -> Tuple[float, float]:
        """
        True Kelly Criterion stake sizing.
        
        f* = (bp - q) / b   where b = odds-1, p = win_prob, q = 1-p
        Stake = bankroll * f* * KELLY_FRACTION
        
        No hardcoded caps or multipliers — the math self-limits:
        - Negative f* → no bet (mathematically correct)
        - Small edge → small f* → small stake (naturally proportional)
        - High prob + high odds → large f* (deserved aggression)
        """
        if prob <= 0 or prob >= 1 or odds <= 1 or bankroll <= 0:
            return 0.0, 0.0
        
        b = odds - 1
        q = 1.0 - prob
        f_star = (b * prob - q) / b
        
        if f_star <= 0:
            return 0.0, 0.0
        
        fraction = f_star * self.KELLY_FRACTION
        stake = bankroll * fraction
        
        return stake, fraction
    
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
        V21.6: ML-DRIVEN DECISION ENGINE.
        
        The ML model (quantile regression) is the PRIMARY decision maker:
        - Side selection: ML's p_over determines OVER/UNDER directly
        - Probability: ML's quantile-interpolated probability (distribution-free)
        - Fallback: Simulation CDF only when ML probability is unavailable
        
        This eliminates the prior inconsistency where the simulation engine's
        Gaussian CDF picked the side but the ML's probability was used for EV,
        potentially creating side/probability disagreements.
        """
        # 1. ML MODEL DECIDES (primary path)
        ml_prob_over = projection.ml_prob if projection.ml_prob is not None else None
        
        if ml_prob_over is not None:
            # ML model directly determines side and probability
            if ml_prob_over > 0.5:
                side = "OVER"
                prob = ml_prob_over
            else:
                side = "UNDER"
                prob = 1.0 - ml_prob_over
        else:
            # 2. FALLBACK: Simulation CDF (only when ML model unavailable)
            prob_over_cdf = simulation.over_prob
            prob_under_cdf = simulation.under_prob
            
            if prob_over_cdf > prob_under_cdf:
                side = "OVER"
                prob = prob_over_cdf
            else:
                side = "UNDER"
                prob = prob_under_cdf

        # V25.2 TEMPERATURE SCALING: Correct overconfident probabilities.
        # Model is well-calibrated on training data but overconfident against
        # sharp sportsbook lines (ECE=0.106).  Temperature scaling (T=4.75,
        # fitted from 489 live outcomes) compresses probabilities toward 0.50,
        # reducing ECE to 0.022.  Replaces crude linear shrinkage + hard cap.
        T = self.config.LIVE_CALIBRATION_TEMPERATURE
        if T > 1.0:
            from scipy.special import logit, expit
            prob = float(expit(logit(np.clip(prob, 0.02, 0.98)) / T))

        # 3. Calculate EV (pure math: EV = p(o-1) - q)
        ev = self.calculate_ev(prob, odds)

        # 4. Kelly Criterion stake (f* = (bp - q) / b)
        stake, fraction = self.calculate_kelly_stake(prob, odds, bankroll)
        
        # 5. Get predicted mean and std from ML model
        predicted_mean = projection.final_projection
        predicted_std = projection.adjustments.get('predicted_std', features.std)
        
        # 6. Quality score: risk-adjusted EV (EV / sigma)
        # Standard Sharpe-like metric. No floors — if std is tiny, the model
        # is very confident, so quality *should* be high.
        quality_score = ev / predicted_std if (ev > 0 and predicted_std > 0) else 0.0

        return BetDecision(
            recommended_side=side,
            probability=prob,
            expected_value=ev,
            predicted_mean=predicted_mean,
            predicted_std=predicted_std,
            kelly_stake=stake,
            kelly_fraction=fraction,
            quality_score=quality_score
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
        NBA season runs Oct-Jun. Oct 2024  "2024-25", Jan 2025  "2024-25"
        
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

        The snapshot includes both legacy non-prefixed attribute names (for backward
        compatibility) and canonical `feat_`-prefixed names used for ML training.
        """
        snap: Dict[str, Any] = {
            'snapshot_date': prediction_date.strftime('%Y-%m-%d'),
            'snapshot_season': self._get_season_for_date(prediction_date),
        }

        # Populate canonical ML features dynamically
        for col in TRAINING_FEATURE_COLUMNS:
            # derive attribute candidate names
            attr = col[len('feat_'):]
            candidates = [attr, col]
            raw_val = None
            for cand in candidates:
                if hasattr(features, cand):
                    raw_val = getattr(features, cand)
                    break

            # Coerce with sensible sentinels
            if raw_val is None:
                if col.endswith('_count'):
                    val = -1
                elif col in ('feat_is_home', 'feat_is_b2b'):
                    val = 0
                else:
                    val = -1.0
            else:
                try:
                    if col.endswith('_count'):
                        val = int(raw_val)
                    elif col in ('feat_is_home', 'feat_is_b2b'):
                        val = 1 if bool(raw_val) else 0
                    else:
                        val = float(raw_val)
                except Exception:
                    if col.endswith('_count'):
                        val = -1
                    elif col in ('feat_is_home', 'feat_is_b2b'):
                        val = 0
                    else:
                        val = -1.0
                    logger.debug(f"Failed to coerce feature '{col}' in snapshot; using sentinel {val}")

            # Store both canonical (feat_) and legacy (attr) names for backward compatibility
            snap[col] = val
            snap[attr] = val

        # Audit field
        snap['had_warnings'] = len(features.data_quality.warnings) > 0 if features.data_quality else False
        return snap
    
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
        # EV of +5%  expect ~0.3 pts CLV, EV of -5%  expect ~-0.3 pts CLV
        ev_component = ev * 6.0  # Scale EV to reasonable CLV range
        
        # Probability component (high confidence  market likely agrees)
        prob_component = (prob - 0.5) * 1.0  # 60% prob  +0.1, 40%  -0.1
        
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
        line_offset: float = 0.0,  # Test at actual average  offset
        fixed_spread: float = 0.0,
        progress_callback=None,
        preloaded_df: pd.DataFrame = None,  # Optional: use pre-loaded game logs
        bulk_loader: 'BulkGameLogLoader' = None,  # V20.3: For absence feature computation
        preloaded_team_stats_by_season: Dict[str, Tuple[pd.DataFrame, float, float]] = None,  # V20.3: Per-season team stats
        preloaded_position: str = None,  # V20.3: Pre-fetched player position (avoids API call in workers)
        historical_fatigue_map: Dict = None,  # V21.7: (team_abbrev, date_str) -> miles_96h
        historical_odds_cache: Dict = None  # V23: (norm_name, market, date_str) -> {line, over_odds, under_odds}
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
            
            # Determine line (synthetic prop line with noise — V22)
            # Real player prop lines require a higher API tier; prop lines
            # remain synthetic.  Real *game-level* spread/total from the
            # Odds API are applied further below when computing features.
            rolling_avg = historical_df[market].tail(lookback).mean()
            hist_std = historical_df[market].tail(lookback).std()
            line_noise_std = max(0.5, hist_std * 0.25) if hist_std > 0 else 1.0
            line_noise = self._rng.normal(0, line_noise_std)
            raw_line = rolling_avg + line_offset + line_noise
            # Snap to realistic sportsbook half-point lines (e.g. 19.5, 20.5) to avoid continuous noise lines
            line = max(0.5, round(float(raw_line) * 2) / 2)
            if line % 1 == 0:
                line += 0.5
            
            # Extract game context
            matchup = target_game.get('MATCHUP', '')
            is_home = 'vs.' in matchup
            
            # Determine player's team AND opponent from matchup string
            if 'vs.' in matchup:
                player_team_abbrev = matchup.split('vs.')[0].strip().split()[-1]
                opp_abbrev = matchup.split('vs.')[-1].strip()
            elif '@' in matchup:
                player_team_abbrev = matchup.split('@')[0].strip().split()[-1]
                opp_abbrev = matchup.split('@')[-1].strip()
            else:
                player_team_abbrev = None
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
                    
                    # Calculate Synthetic Spread  (V24 — corrected formula)
                    # Net rating per-100-poss × (game_pace / 100) → expected
                    # margin in actual game points.  Home-court advantage is
                    # added directly in point-space (the old /2.5 divisor
                    # compressed spreads ~2.5× and cut HCA to 1.2 pts).
                    net_rtg_diff = player_net_rtg - opp_net_rtg
                    home_adj = HOME_ADVANTAGE if is_home else -HOME_ADVANTAGE
                    pace_factor = (player_pace + opp_pace) / 200.0   # ≈ 1.0
                    synthetic_spread = -(net_rtg_diff * pace_factor + home_adj)
                    
                    # Calculate Synthetic Total
                    # Total  (PlayerTeam_Pace + OppTeam_Pace) * 1.15
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

            # V23: Prefer real Vegas spread/total from Odds API cache when
            # available.  This significantly improves feat_spread and
            # feat_game_total accuracy vs the Net Rating synthetic calculation.
            if historical_odds_cache:
                date_str = prediction_date.strftime('%Y-%m-%d')
                if player_team_abbrev and opp_abbrev:
                    real_spread, real_total = None, None
                    # Try both home/away orderings
                    for h, a in [(player_team_abbrev, opp_abbrev), (opp_abbrev, player_team_abbrev)]:
                        game_data = historical_odds_cache.get((h, a, date_str))
                        if game_data:
                            raw_spread = game_data.get('spread')
                            real_total = game_data.get('total')
                            if raw_spread is not None:
                                home_abbrev = game_data.get('home_abbrev', h)
                                # Adjust spread to player's team perspective
                                if player_team_abbrev == home_abbrev:
                                    real_spread = float(raw_spread)
                                else:
                                    real_spread = -float(raw_spread)
                            break
                    if real_spread is not None:
                        backtest_spread = real_spread
                    if real_total is not None:
                        backtest_game_total = float(real_total)
            
            # =========================================================
            # V21.6 TEMPORAL FIX: COMPUTE ABSENCE FEATURES FROM PREVIOUS GAME
            # Uses most recent game before prediction date as proxy for
            # who is available (not the target game's actual box score).
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
                # V21.1: Pass game_season for correct historical DvP lookup
                # V21.4: is_backtest=True uses ground truth from starter cache
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
                    # V21.7: Pass team abbrev for altitude + fatigue features
                    player_team_abbrev=player_team_abbrev,  
                    data_loader=self.data_loader,
                    player_position=player_position,
                    game_season=game_season,  # V21.1: Historical DvP fix
                    is_backtest=True  # V21.4: Use ground truth starter data
                )

                # V20.3: Inject absence features into feature vector
                features.team_out_ppg = absence_features['team_out_ppg']
                features.team_out_count = absence_features['team_out_count']
                features.opp_out_ppg = absence_features['opp_out_ppg']
                features.opp_out_count = absence_features['opp_out_count']

                # V21.7: Override travel features with historical values
                # build_feature_vector uses _FATIGUE_CACHE (today's snapshot only);
                # for training we need the game-date-specific fatigue.
                if historical_fatigue_map and player_team_abbrev:
                    date_str = prediction_date.strftime('%Y-%m-%d')
                    features.feat_miles_96h = float(
                        historical_fatigue_map.get((player_team_abbrev, date_str), 0.0)
                    )

                # =========================================================
                # TEMPORAL INTEGRITY: FREEZE FEATURE SNAPSHOT
                # Capture EXACTLY what was known at prediction time
                # This is our audit trail for walk-forward integrity
                # =========================================================
                frozen_snapshot = self._freeze_feature_snapshot(features, prediction_date)

                # Get actual result (needed for both model and no-model paths)
                actual_value = target_game[market]

                # Generate projection (may fail if no models trained yet)
                try:
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

                    predicted_side = decision.recommended_side
                    predicted_prob = decision.probability
                    predicted_ev = decision.expected_value

                except MLModelRequiredError:
                    # =========================================================
                    # V21.6: NO-MODEL TRAINING DATA GENERATION
                    # When no models exist yet, we still need to record
                    # features + actual_value for training CSV generation.
                    # Use EMA-based fallback for prediction fields.
                    # =========================================================
                    ema_proj = float(getattr(features, 'ema', line))
                    predicted_side = "OVER" if ema_proj > line else "UNDER"
                    predicted_prob = 0.5
                    predicted_ev = 0.0

                hit = (predicted_side == "OVER" and actual_value > line) or \
                      (predicted_side == "UNDER" and actual_value <= line)

                # =========================================================
                # V19 TEMPORAL INTEGRITY: Use FROZEN snapshot for features
                # NOT the potentially-mutated features object
                # V20: Raw observables only (no usage_mult, rest_factor)
                # =========================================================
                feature_dict = frozen_snapshot.copy()

                # Ensure internal audit keys match expected schema (prefixed with _)
                feature_dict['_snapshot_date'] = frozen_snapshot.get('snapshot_date')
                feature_dict['_snapshot_season'] = frozen_snapshot.get('snapshot_season')
                feature_dict['_had_warnings'] = frozen_snapshot.get('had_warnings', False)

                results.append(BacktestResult(
                    date=target_game['GAME_DATE'].strftime('%Y-%m-%d'),
                    player_name=player_name,
                    market=market,
                    line=line,
                    predicted_side=predicted_side,
                    predicted_prob=predicted_prob,
                    predicted_ev=predicted_ev,
                    actual_value=actual_value,
                    hit=hit,
                    grade='EV+' if predicted_ev > 0 else 'EV-',
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
}

# Canonical feature list used for model training (exact order - canonical numeric features)
# V20.3: Base V20 features plus absence-awareness and TS%
# NOTE: Generated dynamically from the FeatureVector dataclass to keep a single
# source of truth for feature additions/removals.
# Copilot: Remove 'feat_hit_rate_l10' and 'feat_hit_rate_l20' from TRAINING_FEATURE_COLUMNS (training only).
TRAINING_FEATURE_COLUMNS = [c for c in FeatureVector.get_feature_names() if c not in ('feat_hit_rate_l10', 'feat_hit_rate_l20')]

# Canonical export schema used for writing ML training CSVs (single source of truth)
ML_EXPORT_METADATA = [
    'date', 'player', 'opponent', 'market', 'position',
    'line', 'predicted_side', 'predicted_prob', 'predicted_ev', 'projected_value',
    'result', 'hit', 'actual_value', 'margin', 'margin_pct', 'result_quality',
    'tag', 'grade',
]

# The full ordered schema for CSV export: metadata followed by canonical training features
# Copilot: Remove 'feat_hit_rate_l10' and 'feat_hit_rate_l20' from this schema list.
# These columns should no longer be written to the training CSV.
ML_EXPORT_SCHEMA = ML_EXPORT_METADATA + [c for c in TRAINING_FEATURE_COLUMNS.copy() if c not in ('feat_hit_rate_l10', 'feat_hit_rate_l20')]

def get_market_group_features(market: str) -> Dict[str, int]:
    """
    Convert market to one-hot encoded group features.
    Returns dict with feat_market_* keys (0 or 1).
    """
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
        
        logger.info(f"   Fetched {len(df)} players from NBA API in 1 request")
        
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
        logger.info(f"   {division}: {division_counts[division]} players")
    
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
        
        logger.info(f" Loading {total_seasons} seasons of data for ML training...")
        
        for i, season in enumerate(seasons):
            try:
                logger.info(f" Fetching ALL game logs for {season} (API call {i+1}/{total_seasons})...")
                time.sleep(self.config.API_DELAY * 2)
                
                logs = leaguegamelog.LeagueGameLog(season=season, player_or_team_abbreviation='P', timeout=120)
                df = logs.get_data_frames()[0]
                
                if df is not None and len(df) > 0:
                    df['SEASON'] = season
                    all_logs.append(df)
                    self._seasons_loaded.append(season)
                    logger.info(f"   {season}: {len(df):,} game log entries")
                else:
                    logger.warning(f"   {season}: No data returned")
                    
                if progress_callback: progress_callback((i + 1) / total_seasons * 0.3)
                    
            except Exception as e:
                logger.error(f"   Failed to fetch {season} logs: {e}")
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

        logger.info(f"   Organized data for {len(self._cache):,} unique players (with computed stats)")

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
    
    def get_team_roster(self, team_id: int, cutoff_date: pd.Timestamp = None) -> Dict[int, Dict]:
        """
        Get all players for a team using the optimized index + CACHING.
        
        V21.6 TEMPORAL FIX: When cutoff_date is provided, only count games
        BEFORE that date for PPG/max_date. This prevents future data from
        leaking into absence features during walk-forward backtesting.
        
        Cache results to prevent filtering 7.5 million times.
        NOTE: When cutoff_date is provided, caching uses a date-qualified key.
        """
        if not self._loaded: return {}
        
        if not self._indexes_built: self._build_optimization_indexes()
        
        # 1. CHECK CACHE (Fastest) — date-qualified key when cutoff provided
        cache_key = (team_id, cutoff_date.strftime('%Y-%m-%d') if cutoff_date is not None else None)
        if cache_key in self._roster_cache:
            return self._roster_cache[cache_key]
            
        # 2. COMPUTE
        player_ids = self._roster_index.get(team_id, set())
        
        roster = {}
        for pid in player_ids:
            df = self._cache.get(pid)
            # Filter for this team's games
            team_games = df[df['TEAM_ID'] == team_id]
            
            # V21.6 TEMPORAL FIX: Only use games BEFORE cutoff_date
            if cutoff_date is not None:
                team_games = team_games[team_games['GAME_DATE'] < cutoff_date]
            
            if len(team_games) > 0:
                ppg = team_games['PTS'].mean() if 'PTS' in team_games.columns else 0.0
                name = team_games['PLAYER_NAME'].iloc[0] if 'PLAYER_NAME' in team_games.columns else 'Unknown'
                min_date = team_games['GAME_DATE'].min()
                max_date = team_games['GAME_DATE'].max()
                
                roster[pid] = {
                    'name': name, 
                    'ppg': ppg, 
                    'games': len(team_games),
                    'min_date': min_date,
                    'max_date': max_date
                }
        
        # 3. SAVE TO CACHE
        self._roster_cache[cache_key] = roster
        return roster
    
    def get_players_who_played_on_date(self, game_date: pd.Timestamp, team_id: int) -> set:
        """Get set of player_ids who played on date using optimized index."""
        if not self._loaded: return set()
        if not self._indexes_built: self._build_optimization_indexes()
        
        # INSTANT LOOKUP (O(1)) instead of scanning 844 dataframes
        key = f"{game_date.strftime('%Y-%m-%d')}_{team_id}"
        return self._daily_index.get(key, set())
    
    def _get_most_recent_participants(self, before_date: pd.Timestamp, team_id: int) -> set:
        """
        V21.6 TEMPORAL FIX: Get players who participated in the team's most
        recent game BEFORE before_date.
        
        This is the pre-game observable proxy for "who is available today."
        Uses the _daily_index for fast lookups, scanning backwards from
        before_date until we find a game.
        """
        if not self._loaded: return set()
        if not self._indexes_built: self._build_optimization_indexes()
        
        # Scan backwards up to 14 days to find the most recent game
        for days_back in range(1, 15):
            check_date = before_date - pd.Timedelta(days=days_back)
            key = f"{check_date.strftime('%Y-%m-%d')}_{team_id}"
            participants = self._daily_index.get(key, set())
            if participants:
                return participants
        
        # No recent game found — return empty (will treat everyone as unknown)
        return set()
    
    def compute_absence_features(self, game_date: pd.Timestamp, player_team_id: int, opponent_team_id: int, player_id: int) -> Dict[str, float]:
        """
        V21.6 TEMPORAL FIX: Absence calculation without future data leakage.
        
        Previously this used get_players_who_played_on_date(game_date) which
        looked up the TARGET game's actual box score — a critical data leak.
        
        FIX: Use the PREVIOUS game's participation as a proxy. If a player
        played in the most recent game before today, assume they're available.
        If they didn't play recently, assume they're out.
        This is the best pre-game information available.
        """
        if not self._loaded: return {'team_out_ppg': -1.0, 'team_out_count': -1, 'opp_out_ppg': -1.0, 'opp_out_count': -1}
        
        # Ensure indexes are built (only happens once per worker)
        if not self._indexes_built: self._build_optimization_indexes()
        
        # 1. Get rosters with TEMPORAL cutoff (PPG only from games before prediction date)
        team_roster = self.get_team_roster(player_team_id, cutoff_date=game_date)
        opp_roster = self.get_team_roster(opponent_team_id, cutoff_date=game_date)
        
        # 2. V21.6 FIX: Use PREVIOUS game's participation (not target game's box score)
        # Find the most recent game date BEFORE the target for each team
        team_prev_played = self._get_most_recent_participants(game_date, player_team_id)
        opp_prev_played = self._get_most_recent_participants(game_date, opponent_team_id)
        
        # 3. Calculate missing production
        team_out_ppg = 0.0
        team_out_count = 0
        
        # Define ghost cutoff: players who haven't played in the last 60 days are treated as gone
        ghost_cutoff = game_date - pd.Timedelta(days=60)
        
        # Iterate roster (~15 items) - Super fast now
        for pid, info in team_roster.items():
            if pid == player_id: continue
            # Check if they did NOT play in the most recent previous game
            if pid not in team_prev_played:
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
            if pid not in opp_prev_played:
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
_worker_fatigue_map = None
_worker_odds_cache = None

def _init_worker(bulk_cache_dict, team_stats_dict, config_obj, fatigue_map=None, odds_cache=None):
    """
    Initialize worker process with shared data ONE TIME.
    This eliminates overhead of sending data for every task.
    """
    global _worker_bulk_loader, _worker_team_stats, _worker_config, _worker_fatigue_map, _worker_odds_cache
    
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

    # 5. Store historical fatigue map for travel feature computation
    _worker_fatigue_map = fatigue_map if fatigue_map else {}

    # 6. Store historical odds cache for real line lookups (V23)
    _worker_odds_cache = odds_cache if odds_cache else {}


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
    global _worker_bulk_loader, _worker_team_stats, _worker_config, _worker_odds_cache
    
    worker_id = os.getpid()
    
    # [LOG 1] STARTING TASK
    logger.info(f" [Worker {worker_id}] STARTING Task {task_idx}/{total_tasks}: {player_name} ({market})")
    
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
            preloaded_position=player_position,
            historical_fatigue_map=_worker_fatigue_map,  # V21.7: Historical travel fatigue
            historical_odds_cache=_worker_odds_cache  # V23: Real historical odds lines
        )
        
        # [LOG 3] COMPLETION
        if summary is not None and len(summary.results_df) > 0:
            result_df = summary.results_df.copy()
            result_df['player_id'] = player_id
            logger.info(f" [Worker {worker_id}] FINISHED Task {task_idx}/{total_tasks}: Generated {len(result_df)} samples")
            return result_df
        
        logger.info(f" [Worker {worker_id}] FINISHED Task {task_idx}/{total_tasks}: No results generated")
        return None
        
    except Exception as e:
        logger.error(f" [Worker {worker_id}] CRASHED Task {task_idx}/{total_tasks} ({player_name}): {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def _build_historical_fatigue_map(training_seasons: list, bulk_loader: 'BulkGameLogLoader' = None) -> dict:
    """
    Build a lookup of (team_abbrev, 'YYYY-MM-DD') -> miles_96h for all games
    across multiple training seasons.

    If a bulk_loader is provided with cached game logs, builds the map
    entirely offline (zero API calls).  Otherwise falls back to
    LeagueGameLog API calls (1 per season).

    Returns dict keyed by (team_abbrev, date_str).
    """
    from collections import defaultdict

    # ── Try to build from bulk_loader cache (fully offline) ──────────────
    if bulk_loader is not None and bulk_loader._loaded:
        logger.info("Building fatigue map from bulk loader cache (offline)...")
        # Collect unique (team, date, arena) tuples from all player game logs
        team_game_set: dict = defaultdict(set)   # abbrev -> set of (date, arena)
        for _pid, pdf in bulk_loader._cache.items():
            if pdf.empty:
                continue
            for _, row in pdf.iterrows():
                abbrev = row.get('TEAM_ABBREVIATION', '')
                if not abbrev or abbrev not in _ARENA_COORDS:
                    continue
                matchup = row.get('MATCHUP', '')
                game_date = pd.to_datetime(row['GAME_DATE'])
                if ' vs. ' in matchup:
                    arena = abbrev
                elif ' @ ' in matchup:
                    arena = matchup.split('@ ')[-1].strip()
                else:
                    arena = abbrev
                team_game_set[abbrev].add((game_date, arena))

        # Convert sets to sorted lists
        team_games: dict = {}
        for abbrev, game_set in team_game_set.items():
            team_games[abbrev] = sorted(game_set, key=lambda x: x[0])

        logger.info(f"  Extracted games for {len(team_games)} teams from bulk cache")

    else:
        # ── Fallback: fetch via API ─────────────────────────────────────
        logger.info("Building fatigue map via API (bulk loader not available)...")
        all_games = []
        for season in training_seasons:
            try:
                time.sleep(0.6)
                glg = leaguegamelog.LeagueGameLog(
                    season=season,
                    season_type_all_star='Regular Season',
                    league_id='00',
                )
                gdf = glg.get_data_frames()[0]
                gdf['GAME_DATE'] = pd.to_datetime(gdf['GAME_DATE'])
                all_games.append(gdf)
                logger.info(f"  Fatigue map: fetched {len(gdf)} game-team rows for {season}")
            except Exception as e:
                logger.warning(f"Failed to fetch game log for season {season}: {e}")

        if not all_games:
            return {}

        full_df = pd.concat(all_games, ignore_index=True)

        team_games: dict = defaultdict(list)
        for _, row in full_df.iterrows():
            abbrev = row.get('TEAM_ABBREVIATION', '')
            if not abbrev or abbrev not in _ARENA_COORDS:
                continue
            game_date = row['GAME_DATE']
            matchup = row.get('MATCHUP', '')
            if ' vs. ' in matchup:
                arena = abbrev
            elif ' @ ' in matchup:
                arena = matchup.split('@ ')[-1].strip()
            else:
                arena = abbrev
            team_games[abbrev].append((game_date, arena))

        for abbrev in team_games:
            team_games[abbrev].sort(key=lambda x: x[0])

    # ── Compute cumulative miles_96h for each (team, game_date) ─────────
    fatigue_map: dict = {}
    for abbrev, game_list in team_games.items():
        for idx, (game_date, game_arena) in enumerate(game_list):
            cutoff = game_date - pd.Timedelta(hours=96)
            arenas_in_window = []
            for j in range(idx, -1, -1):
                prev_date, prev_arena = game_list[j]
                if prev_date < cutoff:
                    break
                arenas_in_window.append(prev_arena)
            arenas_in_window.reverse()

            total_miles = 0.0
            for k in range(1, len(arenas_in_window)):
                if arenas_in_window[k - 1] != arenas_in_window[k]:
                    c1 = _ARENA_COORDS.get(arenas_in_window[k - 1])
                    c2 = _ARENA_COORDS.get(arenas_in_window[k])
                    if c1 and c2:
                        total_miles += _haversine_miles(c1, c2)

            fatigue_map[(abbrev, game_date.strftime('%Y-%m-%d'))] = round(total_miles, 1)

    logger.info(f"Historical fatigue map: {len(fatigue_map)} (team, date) pairs across {len(training_seasons)} seasons")
    return fatigue_map


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

    # FIX 3: ensure we always write a fresh training CSV — remove any existing file
    output_path = DATA_DIR / output_file
    try:
        if output_path.exists():
            output_path.unlink()
            logger.info(f"Cleared existing training data: {output_path}")
    except Exception as e:
        logger.warning(f"Failed to remove existing training data {output_path}: {e}")

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
        logger.error(" Failed to bulk load game logs. Aborting.")
        return pd.DataFrame()
    
    logger.info(f" Bulk data loaded: {bulk_loader.player_count:,} players available offline")
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
        logger.error(" No players found in bulk data with sufficient games")
        return pd.DataFrame()

    logger.info(f"Pre-fetching team stats for {len(config.TRAINING_SEASONS)} seasons...")
    all_season_team_stats = data_loader.fetch_all_seasons_team_stats(list(config.TRAINING_SEASONS))
    
    # Serialize team stats for init
    team_stats_serialized = {}
    for season, (stats_df, avg_pace, avg_def) in all_season_team_stats.items():
        records = stats_df.reset_index().to_dict('records') if len(stats_df) > 0 else []
        team_stats_serialized[season] = (records, avg_pace, avg_def)

    # Serialize bulk logs for init
    logger.info(" Serializing bulk data for workers (One-time cost)...")
    bulk_cache_serialized = {}
    for pid, pdf in bulk_loader._cache.items():
        bulk_cache_serialized[pid] = pdf.to_dict('records')

    # =========================================================================
    # STEP 2b: Build historical fatigue map for travel features (OFFLINE)
    # =========================================================================
    logger.info("Building historical fatigue map from bulk cache (zero API calls)...")
    historical_fatigue = _build_historical_fatigue_map(list(config.TRAINING_SEASONS), bulk_loader=bulk_loader)
    if progress_callback: progress_callback(0.38)

    # =========================================================================
    # STEP 2c: Load historical odds cache (V23 — real spreads/totals from The Odds API)
    # =========================================================================
    historical_odds_cache = {}
    try:
        from odds_engine import OddsEngine
        oe = OddsEngine()
        if oe.cache_stats["total_games"] > 0:
            historical_odds_cache = dict(oe._index)  # (home_abbrev, away_abbrev, date_str) -> game_data
            logger.info(f"  Loaded {len(historical_odds_cache)} game odds for real spread/total features")
        else:
            logger.info("  No game odds cached — will use synthetic spread/total")
    except Exception as e:
        logger.warning(f"  Could not load odds cache: {e} — using synthetic spread/total")
    if progress_callback: progress_callback(0.39)

    # === Prefetch player positions using bulk season calls + cache ===
    logger.info("Prefetching player positions for selected players (bulk + cache)...")
    preloaded_positions = {}

    # Attempt to load a consolidated positions map (static + per-season stats)
    try:
        top_ids = [p['id'] for p in top_players]
        pos_map = data_loader.fetch_player_positions(list(config.TRAINING_SEASONS), player_ids=top_ids)
    except Exception as e:
        logger.warning(f"Failed to fetch consolidated player positions: {e}")
        pos_map = {}

    found = 0
    for p in top_players:
        pid = p['id']
        pos = pos_map.get(pid)
        if not pos:
            logger.debug(f"Position missing for player {pid}, defaulting to 'SF'")
            pos = 'SF'
        else:
            found += 1
        preloaded_positions[pid] = pos

    logger.info(f"Prefetched positions for {found}/{len(top_players)} players from bulk cache/map")

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
        
        # Use preloaded position (avoid API in workers)
        position = preloaded_positions.get(player_id, 'SF')
        
        for market in markets:
            task_idx = len(tasks) + 1
            # Note: We do NOT pass bulk_cache_serialized here anymore!
            tasks.append((
                player_id, player_name, market, lookback, test_days,
                player_df_dict, position, task_idx, 0 # total updated below
            ))
            
    total_tasks = len(tasks)
    tasks = [(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], total_tasks) for t in tasks]
    
    logger.info(f" Prepared {total_tasks} tasks. Starting execution...")
    
    if total_tasks == 0:
        return pd.DataFrame()

    # =========================================================================
    # STEP 4: Execute with Initializer (The Speed Fix)
    # =========================================================================
    max_workers = max(1, os.cpu_count() - 1)
    logger.info(f" Launching {max_workers} workers with shared memory init...")
    
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
            logger.info(f" Launching {max_workers} workers with shared memory init...")
            # Pass the heavy data ONCE via initializer
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_worker,
                initargs=(bulk_cache_serialized, team_stats_serialized, config, historical_fatigue, historical_odds_cache)
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
            # Initialize globals for sequential mode
            _init_worker(bulk_cache_serialized, team_stats_serialized, config, historical_fatigue, historical_odds_cache)
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
        # Initialize globals for sequential mode
        _init_worker(bulk_cache_serialized, team_stats_serialized, config, historical_fatigue, historical_odds_cache)
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

    # NOTE: We removed merging with any pre-existing CSV earlier to guarantee
    # that generation produces a fresh, self-contained training file.
    # Use the freshly composed `combined_df` as the final output.
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

    logger.info(f" ML training data saved to {output_path}")
    if progress_callback: progress_callback(1.0)

    return final_df

def generate_ml_data_streamlit():
    """Streamlit UI wrapper for ML data generation (No Form = No Redirect)."""
    st.markdown("###  Generate ML Training Data")
    
    st.info(f"""
    **V20.3 OPTIMIZED**: Only **8 API calls** total, regardless of player count!
    - Seasons: {', '.join(CONFIG.TRAINING_SEASONS)}
    - API Calls: **8 fixed** (4 game logs + 4 team stats per season)
    - Player list extracted from bulk data (0 extra calls)
    - Step 1: Bulk download (~60-90 seconds)
    - Step 2: Parallel backtesting (100% offline, uses all CPU cores)
    - **Temporal Fix**: Each game uses its season's actual DRTG/Pace
    """)
    
    #  REMOVED st.form wrapper to prevent tab reset
    
    col1, col2 = st.columns(2)
    with col1:
        # Added unique keys to prevent conflict
        num_players = st.slider("Number of Players", 10, 600, 50, 10, key="gen_num")
        test_days = st.slider("Games per Player", 30, 300, 150, 10,
                                help="4 seasons = ~328 games max per player", key="gen_days")
    
    with col2:
        markets = st.multiselect(
            "Markets",
            ['PTS', 'REB', 'AST', 'PRA', 'RA', 'PA', 'PR', '3PM', 'STL', 'BLK'],
            default=['PTS', 'REB', 'AST', 'PRA', 'RA', 'PA', 'PR'],
            key="gen_markets"
        )
        output_file = st.text_input("Output Filename", "ml_training_data.csv", key="gen_file")
    
    #  STANDARD BUTTON (No Form Submit event)
    if st.button(" Generate Dataset", type="primary"):
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
                #  CLEAN IMMEDIATELY
                try:
                    bool_cols = ['feat_is_home', 'feat_is_b2b']
                    for col in bool_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(bool)

                    int_cols = ['market_scoring', 'market_counting', 'market_combo', 'hit']
                    for col in int_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                            
                except Exception as e:
                    st.warning(f"Auto-cleaning warning: {e}")

                # Save to session state
                st.session_state['last_ml_training_data'] = df
                st.session_state['last_ml_output_file'] = output_file
                
                #  NO RERUN call here. Just let it flow to the display block below.
                
            else:
                st.error("Failed to generate training data. Check logs for errors.")

    #  RESULTS DISPLAY
    # This runs immediately after the button click logic finishes
    if 'last_ml_training_data' in st.session_state:
        df = st.session_state['last_ml_training_data'].copy()
        filename = st.session_state.get('last_ml_output_file', 'ml_training_data.csv')
        
        st.success(f" Generated {len(df)} training samples!")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Samples", len(df))
        if 'hit' in df.columns:
            col2.metric("Win Rate", f"{df['hit'].mean():.1%}")
        if 'player' in df.columns:
            col3.metric("Unique Players", df['player'].nunique())
        
        st.dataframe(df.head(20), width="stretch")
        
        csv = df.to_csv(index=False)
        st.download_button(
            " Download CSV",
            csv,
            filename,
            "text/csv",
            key='download-ml-data'
        )


# Shared singleton accessor for BulkGameLogLoader
@st.cache_resource(show_spinner="Initializing Data Engine (One-time)...")
def get_shared_bulk_loader():
    """Singleton accessor for the heavy BulkGameLogLoader.

    This loads bulk logs for all configured training seasons once and reuses the
    loader across Streamlit sessions and UI tabs.
    """
    loader = BulkGameLogLoader(CONFIG)
    # Load all seasons defined in config to ensure full history
    loader.load_all_game_logs()
    return loader


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
                # print(f" Recency fetch failed ({e}), falling back to full season data.")
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
            # Pre-fetch player position to avoid per-worker API calls
            try:
                player_position = self.data_loader.get_player_position(p_obj['id'])
            except Exception:
                player_position = 'SF'

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
                data_loader=self.data_loader,  # V20.3: For PPG lookups
                player_position=player_position
            )
            
            # Step 6: Generate projection
            projection = self.model_engine. generate_projection(features)
            
            # Step 7: V21.6 CDF fallback (only used if ML prob unavailable)
            # Use model's predicted_std when available, fall back to empirical std
            sim_std = projection.adjustments.get('predicted_std', features.std) or features.std
            simulation = self.simulation_engine.compute_cdf_probability(
                mean=projection.final_projection,
                std=sim_std,
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
                player_position=player_position,  # V21.2: For AI analyst
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
            
            # FIX: Retrieve shared loader for absence calculations
            bulk_loader = get_shared_bulk_loader()

            return self.backtester.run_backtest(
                player_id=p_obj['id'],
                player_name=p_obj['full_name'],
                market=market,
                lookback=lookback,
                test_days=test_days,
                line_offset=line_offset,
                fixed_spread=fixed_spread,
                progress_callback=progress_callback,
                bulk_loader=bulk_loader
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
        self._load_failed = False
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

    def _ml_column_sentinel(self, col: str):
        """Return a sensible sentinel value for a given ML column name."""
        if col.endswith('_count'):
            return -1
        if col in ('feat_is_home', 'feat_is_b2b'):
            return 0
        # Default sentinel for floats
        return -1.0

    def add_bet(self, bet_data: Dict):
        """Alias for save_bet / log_bet to support both player props and game-level tracking."""
        self.save_bet(bet_data)

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
        
        # 4. Add model_version for tracking (v21.2+)
        if 'model_version' not in bet_data:
            bet_data['model_version'] = CURRENT_VERSION
            
        # 4. Ensure all canonical ML feature columns exist on saved bets to prevent schema drift.
        # This stores explicit sentinel values for any missing V20.3 features so older records
        # remain compatible with downstream diagnostics and export code.
        for col in TRAINING_FEATURE_COLUMNS:
            if col not in bet_data:
                bet_data[col] = self._ml_column_sentinel(col)

        bets.append(bet_data)
        self._save(bets)

    def get_bets(self) -> list: 
        """Loads bets and auto-fixes any legacy data safely."""
        try:
            if self.file.exists():
                data = safe_json_load(self.file, default=[])
                if not isinstance(data, list):
                    data = []
                
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
                        
                self._load_failed = False
                return data
            return []
        except Exception as exc:
            self._load_failed = True
            import traceback, shutil, time
            logger.exception("Failed to load bets from tracker JSON — creating a backup and halting write operations to prevent data loss")
            try:
                corrupt_name = self.file.with_name(f"{self.file.name}.corrupt.{int(time.time())}")
                shutil.copy2(self.file, corrupt_name)
                logger.warning(f"Backed up corrupt tracker to {corrupt_name}")
            except Exception:
                logger.exception("Failed to back up corrupt tracker file")
            raise RuntimeError(
                f"Tracker file at {self.file} failed to load cleanly: {exc}. "
                "Halting to prevent accidental overwrite of existing bets."
            ) from exc

    def _save(self, bets: list):
        if getattr(self, '_load_failed', False):
            raise RuntimeError(
                "Refusing to save bets: tracker failed to load cleanly. Overwrite aborted to protect bet history."
            )
        try:
            atomic_json_save(self.file, bets, indent=2)
        except Exception as e:
            logging.error(f"Failed to save bets atomically: {e}")
            raise

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

    def auto_settle_pending_bets(self) -> dict:
        """
        Auto-settles all 'Pending' bets in the tracker against official ESPN box scores and game scores.
        Handles both:
          1. Game lines (SPREAD, TOTAL)
          2. Player props (PTS, REB, AST, 3PT, PRA, PR, PA, RA)
        """
        import requests
        from datetime import datetime, timezone

        pending_bets = [b for b in self.get_bets() if b.get('result') == 'Pending']
        if not pending_bets:
            return {"settled_count": 0, "wins": 0, "losses": 0, "pushes": 0, "details": []}

        espn_scoreboards_by_date = {}
        espn_boxscores_by_date = {}

        settled_count = 0
        wins = 0
        losses = 0
        pushes = 0
        details = []

        for bet in pending_bets:
            b_id = bet.get('id')
            bet_date = bet.get('date', '')
            if not bet_date:
                continue

            market = str(bet.get('market', '')).upper()
            bet_type = str(bet.get('bet_type', '')).lower()
            side = str(bet.get('side', '')).upper()
            line = float(bet.get('line', 0.0))
            is_game = market in ['SPREAD', 'TOTAL', 'spread', 'total'] or bet_type in ['spread', 'total']

            if is_game:
                if bet_date not in espn_scoreboards_by_date:
                    try:
                        resp = requests.get(
                            "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
                            params={"dates": bet_date.replace("-", "")},
                            timeout=10
                        )
                        espn_scoreboards_by_date[bet_date] = resp.json().get("events", []) if resp.status_code == 200 else []
                    except Exception:
                        espn_scoreboards_by_date[bet_date] = []

                events = espn_scoreboards_by_date[bet_date]
                matched_event = None
                p_text = (str(bet.get('player', '')) + " " + str(bet.get('opponent', ''))).upper()

                for ev in events:
                    comp = ev.get("competitions", [{}])[0]
                    competitors = comp.get("competitors", [])
                    abbrevs = [c.get("team", {}).get("abbreviation", "").upper() for c in competitors]
                    names = [c.get("team", {}).get("name", "").upper() for c in competitors]
                    if any(a in p_text for a in abbrevs if a) or any(n in p_text for n in names if n):
                        status = comp.get("status", {}).get("type", {})
                        if status.get("state") == "post" or "FINAL" in status.get("description", "").upper():
                            matched_event = comp
                            break

                if not matched_event:
                    continue

                home_c = next((c for c in matched_event.get("competitors", []) if c.get("homeAway") == "home"), {})
                away_c = next((c for c in matched_event.get("competitors", []) if c.get("homeAway") == "away"), {})

                try:
                    h_score = float(home_c.get("score", 0))
                    a_score = float(away_c.get("score", 0))
                except (ValueError, TypeError):
                    continue

                if market == 'TOTAL' or bet_type == 'total':
                    actual_total = h_score + a_score
                    is_over = 'OVER' in side
                    if actual_total > line:
                        res = "Win" if is_over else "Loss"
                    elif actual_total < line:
                        res = "Loss" if is_over else "Win"
                    else:
                        res = "Push"

                    self.update_result(b_id, res, closing_line=line, actual_value=actual_total)
                    settled_count += 1
                    if res == "Win": wins += 1
                    elif res == "Loss": losses += 1
                    else: pushes += 1
                    details.append({"id": b_id, "type": "game_total", "result": res, "actual": actual_total, "line": line})

                elif market == 'SPREAD' or bet_type == 'spread':
                    h_abbr = home_c.get("team", {}).get("abbreviation", "").upper()
                    picked_home = h_abbr in side or "HOME" in side
                    margin = (h_score - a_score) if picked_home else (a_score - h_score)
                    ats_margin = margin + line
                    if ats_margin > 0.05:
                        res = "Win"
                    elif ats_margin < -0.05:
                        res = "Loss"
                    else:
                        res = "Push"

                    self.update_result(b_id, res, closing_line=line, actual_value=margin)
                    settled_count += 1
                    if res == "Win": wins += 1
                    elif res == "Loss": losses += 1
                    else: pushes += 1
                    details.append({"id": b_id, "type": "game_spread", "result": res, "actual": margin, "line": line})

            else:
                if bet_date not in espn_boxscores_by_date:
                    try:
                        from scripts.harvest_player_props import fetch_espn_player_boxscores
                        espn_boxscores_by_date[bet_date] = fetch_espn_player_boxscores(bet_date)
                    except Exception:
                        espn_boxscores_by_date[bet_date] = {}

                boxscores = espn_boxscores_by_date[bet_date]
                if not boxscores:
                    continue

                player_name = str(bet.get('player', '')).strip().lower()
                matched_stats = None
                for b_name, b_stats in boxscores.items():
                    if b_name in player_name or player_name in b_name:
                        matched_stats = b_stats
                        break

                if not matched_stats:
                    continue

                actual_stat = matched_stats.get(market)
                if actual_stat is None:
                    continue

                is_over = 'OVER' in side or side == 'OVER'
                if actual_stat > line:
                    res = "Win" if is_over else "Loss"
                elif actual_stat < line:
                    res = "Loss" if is_over else "Win"
                else:
                    res = "Push"

                self.update_result(b_id, res, closing_line=line, actual_value=actual_stat)
                settled_count += 1
                if res == "Win": wins += 1
                elif res == "Loss": losses += 1
                else: pushes += 1
                details.append({"id": b_id, "type": f"prop_{market}", "result": res, "actual": actual_stat, "line": line})

        return {
            "settled_count": settled_count,
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "details": details,
        }

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

        # --- BACKWARDS-COMPAT: expose win_rate_all but compute primary win_rate over staked bets ---
        total_decided = wins + losses
        win_rate_all = wins / total_decided if total_decided > 0 else 0

        # Primary metric: only count decided bets that had a positive stake
        decided_staked = [b for b in bets if b.get('result') in ['Win', 'Loss'] and float(b.get('stake', 0) or 0) > 0]
        wins_staked = len([b for b in decided_staked if b.get('result') == 'Win'])
        total_decided_staked = len(decided_staked)
        win_rate = wins_staked / total_decided_staked if total_decided_staked > 0 else 0

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
            'total_decided': total_decided,
            # Primary win_rate (staked-only) for reporting/business logic
            'win_rate': win_rate,
            # Back-compat / audit: win rate across ALL decided bets (including 0-stake)
            'win_rate_all': win_rate_all,
            # Staked-specific counters
            'total_decided_staked': total_decided_staked, 'wins_staked': wins_staked,
            'total_profit': total_profit,
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

    def get_equity_curve(self, in_units: bool = False, unit_size: float | None = None) -> list:
        """Return a cumulative P/L series (only for bets with a positive stake).

        - Filters to decided bets (Win/Loss) where stake > 0
        - Sorts by `date` then `id` for deterministic ordering
        - Computes stake-weighted P/L: win -> stake*(odds-1), loss -> -stake
        - If `in_units` is True and `unit_size` provided, values are divided by `unit_size`.

        Returns: list[dict] where each dict is {'Bet #': int, 'P/L': float}
        """
        bets = self.get_bets()
        decided_staked = [
            b for b in bets
            if b.get('result') in ['Win', 'Loss'] and float(b.get('stake', 0) or 0) > 0
        ]
        # deterministic ordering
        decided_staked.sort(key=lambda x: (x.get('date', ''), x.get('id', 0)))

        cumulative = []
        running_total = 0.0
        for b in decided_staked:
            stake = float(b.get('stake', 0) or 0)
            odds = b.get('odds_decimal', b.get('odds', 1.91))
            # Normalize American odds if present
            if isinstance(odds, (int, float)) and (odds < 0 or odds >= 100):
                odds = american_to_decimal(odds)

            if b.get('result') == 'Win':
                running_total += stake * (odds - 1)
            else:
                running_total -= stake

            val = running_total
            if in_units and unit_size and unit_size > 0:
                val = running_total / float(unit_size)

            cumulative.append({'Bet #': len(cumulative) + 1, 'P/L': val})

        return cumulative

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

        # Build the canonical row (Base Metadata)
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
        }

        # Populate canonical ML features dynamically from TRAINING_FEATURE_COLUMNS
        for col in TRAINING_FEATURE_COLUMNS:
            # Prefer explicit 'feat_' keys in bet, but fall back to legacy attr name
            attr = col[len('feat_'):]
            raw = None
            for key in (col, attr):
                if key in bet:
                    raw = bet.get(key)
                    break

            #  LAZY BACKFILL for feat_eff_per_min
            # If the feature is missing or sentinel (-1.0), try to calculate it on-the-fly
            if col == 'feat_eff_per_min' and (raw is None or raw == -1.0):
                try:
                    # Grab raw ingredients from the old record
                    ema = float(bet.get('ema', bet.get('feat_ema', 0.0)) or 0.0)
                    mins = float(bet.get('avg_minutes', bet.get('feat_avg_minutes', 0.0)) or 0.0)
                    
                    if mins > 5.0:
                        raw = ema / mins  # Calculate Efficiency
                    else:
                        raw = 0.0
                except Exception:
                    raw = 0.0

            #  LAZY BACKFILL for feat_fatigue_load
            # Compute as avg_minutes / (days_rest + 0.5) when missing
            if col == 'feat_fatigue_load' and (raw is None or raw == -1.0):
                try:
                    mins = float(bet.get('avg_minutes', bet.get('feat_avg_minutes', 0.0)) or 0.0)
                    days = float(bet.get('days_rest', bet.get('feat_days_rest', 1)) or 1.0)
                    raw = mins / (days + 0.5)
                except Exception:
                    raw = 0.0

            #  LAZY BACKFILL for feat_form_gap
            # Compute as (l10_avg - season_avg) / max(1.0, season_avg)
            if col == 'feat_form_gap' and (raw is None or raw == -1.0):
                try:
                    l10 = bet.get('feat_l10_avg', bet.get('l10_avg', None))
                    season = bet.get('feat_season_avg', bet.get('season_avg', None))
                    if l10 is not None and season is not None:
                        season_val = float(season) if season is not None else 0.0
                        raw = (float(l10) - season_val) / max(1.0, season_val)
                    else:
                        # Fallback: use ema relative gap when l10 missing
                        ema = bet.get('ema', bet.get('feat_ema', None))
                        if ema is not None and season is not None:
                            season_val = float(season) if season is not None else 0.0
                            raw = (float(ema) - season_val) / max(1.0, season_val)
                        else:
                            raw = 0.0
                except Exception:
                    raw = 0.0

            # If still missing, use sentinel
            if raw is None:
                row[col] = self._ml_column_sentinel(col)
                continue

            # Coerce types safely
            try:
                if col.endswith('_count'):
                    val = int(raw or -1)
                elif col in ('feat_is_home', 'feat_is_b2b'):
                    val = 1 if bool(raw) else 0
                elif col.startswith('feat_market_'):
                    val = int(raw)
                else:
                    val = float(raw)
            except Exception:
                val = self._ml_column_sentinel(col)
                # logger.debug(f"Failed to coerce feature '{col}' from bet; using sentinel {val}")

            row[col] = val

        # Market identity features (ensure canonical one-hot is consistent with declared market)
        row.update(get_market_group_features(bet.get('market', 'PTS')))

        return row

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
                # Only export STAKED bets (stake > 0) - unstaked predictions go to prediction_log.json
                stake = float(bet.get('stake', 0) or 0)
                if not (bet.get('result') in ['Win', 'Loss'] and bet.get('actual_value') is not None and not bet.get('exported_to_csv', False) and stake > 0):
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
        """Count bets that can be exported to training CSV (staked, decided, not yet exported)."""
        bets = self.get_bets()
        return len([
            b for b in bets 
            if b.get('result') in ['Win', 'Loss'] 
            and b.get('actual_value') is not None 
            and not b.get('exported_to_csv', False)
            and float(b.get('stake', 0) or 0) > 0  # Only staked bets
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

    def update_stake(self, bet_id: int, new_stake: float):
        """Update the stake for a specific bet."""
        current_bets = self.get_bets()
        for bet in current_bets:
            if bet.get('id') == bet_id:
                bet['stake'] = new_stake
                break
        self._save(current_bets)

    def get_retrain_recommendation(self) -> dict:
        """
        Check if model retraining is recommended based on recent performance.
        
        Criteria:
        - Minimum 50 decided staked bets for statistical significance
        - Win rate below 52% triggers retrain recommendation
        - Also checks days since last retrain (weekly recommendation)
        
        Returns dict with:
        - should_retrain: bool
        - reason: str (explanation)
        - win_rate: float (current win rate)
        - sample_size: int (number of decided staked bets)
        - days_since_retrain: int (if retrain_date stored)
        """
        bets = self.get_bets()
        
        # Get decided staked bets
        decided_staked = [
            b for b in bets 
            if b.get('result') in ['Win', 'Loss'] 
            and float(b.get('stake', 0) or 0) > 0
        ]
        
        sample_size = len(decided_staked)
        wins = len([b for b in decided_staked if b.get('result') == 'Win'])
        win_rate = wins / sample_size if sample_size > 0 else 0.0
        
        # Check last retrain date from config
        config_path = DATA_DIR / 'retrain_config.json'
        days_since_retrain = None
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    last_retrain = config.get('last_retrain_date')
                    if last_retrain:
                        from datetime import datetime
                        last_dt = datetime.fromisoformat(last_retrain)
                        days_since_retrain = (datetime.now() - last_dt).days
            except:
                pass
        
        # Determine recommendation
        reasons = []
        should_retrain = False
        
        if sample_size < 50:
            reasons.append(f"Need 50+ decided bets for analysis (have {sample_size})")
        else:
            if win_rate < 0.52:
                should_retrain = True
                reasons.append(f"Win rate {win_rate*100:.1f}% below 52% threshold")
            
            if days_since_retrain is not None and days_since_retrain >= 7:
                should_retrain = True
                reasons.append(f"Weekly retrain due ({days_since_retrain} days since last)")
            elif days_since_retrain is None:
                reasons.append("No retrain history found - consider establishing baseline")
        
        return {
            'should_retrain': should_retrain,
            'reason': '; '.join(reasons) if reasons else 'Model performing well',
            'win_rate': win_rate,
            'sample_size': sample_size,
            'days_since_retrain': days_since_retrain
        }

    def mark_retrain_complete(self):
        """Mark that a retrain was completed today."""
        from datetime import datetime
        config_path = DATA_DIR / 'retrain_config.json'
        config = {}
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
            except:
                pass
        config['last_retrain_date'] = datetime.now().isoformat()
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)


class PredictionLogger:
    """
    Logs ALL predictions for analysis (staked and non-staked).
    
    This is separate from bet_tracker.json which only tracks placed bets.
    Use this for:
    - Analyzing rejected bet patterns
    - Comparing staked vs non-staked outcomes
    - Model calibration studies
    """
    
    def __init__(self, file_path: Path = PREDICTION_LOG_FILE):
        self.file = file_path
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        if not self.file.exists():
            self._save([])
    
    def _save(self, predictions: list):
        try:
            atomic_json_save(self.file, predictions, indent=2)
        except Exception as e:
            logger.error(f"Failed to save prediction log atomically: {e}")
            raise
    
    def get_predictions(self) -> list:
        try:
            data = safe_json_load(self.file, default=[])
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"Failed to load prediction log: {e}")
            return []
    
    def log_prediction(self, prediction: dict):
        """
        Log a single prediction with all features.
        
        Expected fields:
        - player, opponent, market, line, side
        - predicted_mean, predicted_std, prob, ev
        - was_staked: bool (did it pass risk engine?)
        - stake: float (0 if not staked)
        - All feat_* fields
        """
        import time
        from datetime import datetime
        
        predictions = self.get_predictions()
        
        if 'id' not in prediction:
            prediction['id'] = int(time.time() * 1000)
        if 'timestamp' not in prediction:
            prediction['timestamp'] = datetime.now().isoformat()
        if 'date' not in prediction:
            prediction['date'] = datetime.now().strftime("%Y-%m-%d")
        
        predictions.append(prediction)
        self._save(predictions)
    
    def log_batch(self, predictions: list):
        """Log multiple predictions at once (e.g., after risk engine runs)."""
        import time
        from datetime import datetime
        
        existing = self.get_predictions()
        
        for pred in predictions:
            if 'id' not in pred:
                pred['id'] = int(time.time() * 1000)
                time.sleep(0.001)  # Ensure unique IDs
            if 'timestamp' not in pred:
                pred['timestamp'] = datetime.now().isoformat()
            if 'date' not in pred:
                pred['date'] = datetime.now().strftime("%Y-%m-%d")
            existing.append(pred)
        
        self._save(existing)
    
    def get_stats(self) -> dict:
        """Get summary statistics of logged predictions."""
        predictions = self.get_predictions()
        
        total = len(predictions)
        staked = len([p for p in predictions if p.get('was_staked', False)])
        not_staked = total - staked
        
        # Get date range
        dates = [p.get('date') for p in predictions if p.get('date')]
        min_date = min(dates) if dates else None
        max_date = max(dates) if dates else None
        
        return {
            'total_predictions': total,
            'staked': staked,
            'not_staked': not_staked,
            'staked_pct': staked / total * 100 if total > 0 else 0,
            'date_range': f"{min_date} to {max_date}" if min_date and max_date else "N/A"
        }
    
    def clear_old_predictions(self, days: int = 30):
        """Remove predictions older than N days to manage file size."""
        from datetime import datetime, timedelta
        
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        predictions = self.get_predictions()
        
        kept = [p for p in predictions if p.get('date', '9999-99-99') >= cutoff]
        removed = len(predictions) - len(kept)
        
        self._save(kept)
        return removed


class ParlayTracker: 
    """Tracks parlay bets."""
    
    def __init__(self, file_path: Path = PARLAY_FILE):
        self.file = file_path
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        if not self.file.exists():
            self._save([])

    def _normalize_parlay_record(self, raw: Any, fallback_idx: int = 0) -> Optional[Dict[str, Any]]:
        """Normalize legacy/external parlay payloads into app schema."""
        if not isinstance(raw, dict):
            return None

        # IDs can come in many formats (int, numeric string, slip_id).
        raw_id = raw.get('id', raw.get('slip_id'))
        try:
            pid = int(raw_id)
        except Exception:
            pid = int(time.time() * 1000) + fallback_idx

        raw_legs = raw.get('legs', [])
        if not isinstance(raw_legs, list):
            raw_legs = []

        legs: List[Dict[str, Any]] = []
        for leg in raw_legs:
            if not isinstance(leg, dict):
                continue
            legs.append({
                'player': leg.get('player', 'Unknown'),
                'opponent': leg.get('opponent', ''),
                'market': leg.get('market', ''),
                'line': float(leg.get('line', 0.0) or 0.0),
                'side': leg.get('side', ''),
                'odds': float(leg.get('odds', 1.0) or 1.0),
                'result': leg.get('result', 'Pending'),
            })

        # External exporters may use "status"; internal code uses "result".
        raw_result = str(raw.get('result', raw.get('status', 'Pending')) or 'Pending').strip().lower()
        if raw_result in ('win', 'won'):
            result = 'Win'
        elif raw_result in ('loss', 'lost'):
            result = 'Loss'
        else:
            result = 'Pending'

        stake = float(raw.get('stake', 0.0) or 0.0)
        combined_odds = float(raw.get('combined_odds', 1.0) or 1.0)
        potential_payout = float(
            raw.get('potential_payout', raw.get('potential_winnings', stake * combined_odds))
            or (stake * combined_odds)
        )

        parlay = {
            'id': pid,
            'name': raw.get('name') or raw.get('type') or f"Parlay #{fallback_idx + 1}",
            'date': raw.get('date') or raw.get('date_placed') or datetime.now().strftime("%Y-%m-%d %H:%M"),
            'legs': legs,
            'num_legs': int(raw.get('num_legs', len(legs)) or len(legs)),
            'stake': stake,
            'combined_odds': combined_odds,
            'combined_prob': float(raw.get('combined_prob', raw.get('hit_probability_naive', 0.0)) or 0.0),
            'potential_payout': potential_payout,
            'potential_profit': float(raw.get('potential_profit', potential_payout - stake) or (potential_payout - stake)),
            'result': result,
            'legs_hit': int(raw.get('legs_hit', 0) or 0),
            'legs_decided': int(raw.get('legs_decided', 0) or 0),
            'actual_payout': float(raw.get('actual_payout', 0.0) or 0.0),
        }

        # Keep status consistent with leg-level outcomes whenever leg statuses exist.
        self._recalculate_parlay_status(parlay)
        return parlay

    def get_parlays(self) -> list:
        try:
            with open(self.file, 'r') as f:
                content = f.read()
                payload = json.loads(content) if content else []

                # Backward compatibility:
                # - legacy/internal format: [ ...parlays ]
                # - external wrapped format: {"meta": {...}, "parlays": [ ... ]}
                if isinstance(payload, dict):
                    raw_parlays = payload.get('parlays', [])
                elif isinstance(payload, list):
                    raw_parlays = payload
                else:
                    raw_parlays = []

                normalized = []
                for idx, raw in enumerate(raw_parlays):
                    item = self._normalize_parlay_record(raw, fallback_idx=idx)
                    if item is not None:
                        normalized.append(item)
                return normalized
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
            if parlay.get('id') == parlay_id:
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
        parlays = [p for p in parlays if isinstance(p, dict) and p.get('id') != parlay_id]
        self._save(parlays)

    def get_stats(self) -> dict:
        parlays = [p for p in self.get_parlays() if isinstance(p, dict)]
        total_parlays = len(parlays)
        wins = len([p for p in parlays if p.get('result') == 'Win'])
        losses = len([p for p in parlays if p.get('result') == 'Loss'])
        pending = len([p for p in parlays if p.get('result', 'Pending') == 'Pending'])
        decided_parlays = [p for p in parlays if p.get('result', 'Pending') != 'Pending']
        total_staked = sum(float(p.get('stake', 0.0) or 0.0) for p in decided_parlays)
        total_returned = sum(float(p.get('actual_payout', 0.0) or 0.0) for p in decided_parlays)
        total_profit = total_returned - total_staked
        roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
        avg_legs = (sum(int(p.get('num_legs', 0) or 0) for p in parlays) / total_parlays) if total_parlays > 0 else 0
        total_legs = sum(int(p.get('legs_decided', 0) or 0) for p in parlays)
        total_legs_hit = sum(int(p.get('legs_hit', 0) or 0) for p in parlays)
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
            # Preserve wrapped schema if the existing file uses {meta, parlays}.
            wrapped_mode = False
            try:
                if self.file.exists():
                    existing = safe_json_load(self.file, default=None)
                    wrapped_mode = isinstance(existing, dict) and 'parlays' in existing
            except Exception:
                wrapped_mode = False

            if wrapped_mode:
                total_stake = sum(float(p.get('stake', 0.0) or 0.0) for p in data if isinstance(p, dict))
                total_potential = sum(float(p.get('potential_payout', 0.0) or 0.0) for p in data if isinstance(p, dict))
                payload = {
                    'meta': {
                        'last_updated': datetime.now().isoformat(),
                        'total_parlays': len(data),
                        'total_stake_php': round(total_stake, 2),
                        'total_potential_winnings_php': round(total_potential, 2),
                    },
                    'parlays': data,
                }
                atomic_json_save(self.file, payload, indent=4)
            else:
                atomic_json_save(self.file, data, indent=4)
        except Exception as e:
            logger.error(f"Failed to save parlay tracker atomically: {e}")
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
    with st.expander(f" Data Quality: :{color}[{dq.grade}] ({dq.score:.0f}/100)", expanded=False):
        # Data Sources Status
        st.markdown("** Data Sources:**")
        col1, col2 = st.columns(2)
        
        with col1:
            # Team Stats
            if dq.missing_team_stats:
                st.markdown(" **Team Stats:** Unavailable")
            elif dq.team_stats_age_hours > 24:
                st.markdown(f" **Team Stats:** {dq.team_stats_age_hours:.0f}h old")
            else:
                st.markdown(" **Team Stats:** Fresh (<24h)")
        
        with col2:
            # Sample Size
            if features.games_played >= 15:
                st.markdown(f" **Sample Size:** {features.games_played} games")
            elif features.games_played >= 10:
                st.markdown(f" **Sample Size:** {features.games_played} games")
            else:
                st.markdown(f" **Sample Size:** {features.games_played} games (small)")
        
        # Show warnings if any
        if dq.has_issues:
            st.markdown("---")
            st.caption("** Fallbacks Used:**")
            
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
                st.markdown(f" {warning}")


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
    side_icon = '' if is_over else ''
    
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
<div style="font-size: 18px; font-weight: 700; color: #aaa;">{predicted_std:.1f}</div>
<div style="font-size: 10px; color: #888;">Std Dev</div>
</div>
<div style="background: rgba(255,255,255,0.05); border-radius: 8px; padding: 10px;">
<div style="font-size: 18px; font-weight: 700; color: white;">{"" + f"{decision.kelly_stake:.0f}" if bankroll_enabled else ""}</div>
<div style="font-size: 10px; color: #888;">{"Stake" if bankroll_enabled else "N/A"}</div>
</div>
</div>
<div style="text-align: center; font-size: 11px; color: #666; border-top: 1px solid #333; padding-top: 10px;">
90% CI: [{ci_low:.1f}, {ci_high:.1f}] | Games: {features.games_played}
</div>
</div>"""

    st.markdown(ticket_html, unsafe_allow_html=True)
    if projection.ml_prob is not None:
        with st.expander(" ML Decoder", expanded=False):
            render_ml_decoder(result)


def render_ml_decoder(result: AnalysisResult):
    """Render the ML signal decoder panel with V18 calibration info."""
    features = result.features
    projection = result.projection
    # Safe getter for ml_details (backwards compatibility with cached results)
    ml_details = getattr(result, 'ml_details', None)
    
    # ROW 0: ML MODEL INFO (V18 - Calibration Details)
    if ml_details and ml_details.get('has_model'):
        st.caption(" **ML Model Info**")
        m1, m2, m3, m4 = st.columns(4)
        
        model_group = ml_details.get('model_group', 'universal')
        group_emoji = {'scoring': '', 'counting': '', 'combo': '', 'rare': '', 'universal': ''}
        m1.metric("Model", f"{group_emoji.get(model_group, '')} {model_group.title()}")
        
        raw_prob = ml_details.get('raw_prob')
        if raw_prob is not None:
            m2.metric("Raw ML", f"{raw_prob:.1%}")
        
        calibrated_prob = ml_details.get('calibrated_prob')
        if calibrated_prob is not None:
            delta = ml_details.get('calibration_delta', 0)
            delta_str = f"{delta:+.1%}" if delta else None
            m3.metric("Calibrated", f"{calibrated_prob:.1%}", delta_str)
            
        has_calib = ml_details.get('has_calibrator', False)
        m4.metric("Calibrator", " Active" if has_calib else " None")
        
        st.divider()
    
    # ROW 1: KEY STATS
    st.caption(" **Key Stats**")
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
    st.caption(" **Key Factors**")
    e1, e2, e3, e4 = st.columns(4)
    
    # Display metrics without heuristic warnings (trust the model)
    e1.metric("Avg Min", f"{features.avg_minutes:.1f}")
    ema_val = getattr(features, "ema", -1.0) or -1.0
    ema_display = "N/A" if ema_val < 0 else f"{ema_val:.1f}"
    e1.markdown(
        f"<div style='font-size:11px;color:#888;margin-top:6px'>EMA</div>"
        f"<div style='font-size:16px;color:#aaa;font-weight:700'>{ema_display}</div>",
        unsafe_allow_html=True,
    )
    
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
    st.markdown(f"### {decision.recommended_side} {result.line}  :{ev_color}[EV: {ev:+.1%}]")
    
    # --- 2. KEY METRICS ---
    
    if projection.ml_prob is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Win Prob", f"{decision.probability:.0%}")
        c2.metric("EV", f"{decision.expected_value:+.1%}")
        c3.metric("Stake", f"{decision.kelly_stake:.0f}", f"{decision.kelly_fraction:.1%}")
        
        # V20 EMPIRICAL: Display raw probability - no categorical labels
        c4.metric("ML P(Over)", f"{projection.ml_prob:.0%}")
        
    else:
        # Fallback if no ML model loaded
        c1, c2, c3 = st.columns(3)
        c1.metric("Win Prob", f"{decision.probability:.0%}")
        c2.metric("EV", f"{decision.expected_value:+.1%}")
        c3.metric("Stake", f"{decision.kelly_stake:.0f}", f"{decision.kelly_fraction:.1%}")
    
    # --- 3. V19 SIGNAL DECODER (STREAMLINED) ---
    
    with st.expander(" ML Signal Decoder", expanded=False):
        # V20.2 EMPIRICAL: Display raw observables only - NO categorical labels
        # The ML model interprets these values; humans should not add labels
        
        st.caption(" **Raw Observables (Model Inputs)**")
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
        st.caption(" **Feature Values**")
        d1, d2, d3 = st.columns(3)
        
        with d1:
            st.metric("Avg Minutes", f"{features.avg_minutes:.1f}")
            
        with d2:
            st.metric("Opp DRTG", f"{features.opponent_drtg_season:.1f}")
            
        with d3:
            st.metric("Game Total", f"{features.game_total:.0f}")
        
        # V20.2 NEW: Pace and trend features
        st.divider()
        st.caption(" **V20.2 Features**")
        p1, p2, p3 = st.columns(3)
        
        with p1:
            st.metric("Opp Pace", f"{features.opponent_pace:.1f}")
            
        with p2:
            st.metric("Team Pace", f"{features.team_pace:.1f}")
            
        with p3:
            trend_color = "" if features.trend_5g > 0 else "" if features.trend_5g < 0 else ""
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


def render_backtest_tab(orchestrator):
    """Render compact backtesting tab with Form Optimization & Bug Fixes."""
    st.markdown("###  Model Backtest")
    
    #  START FORM
    with st.form(key='backtest_form'):
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            # Use the cached player loader we added earlier
            all_players = get_all_nba_players() 
            player_names = sorted([p['full_name'] for p in all_players if p.get('is_active', True)])
            bt_player = st.selectbox("Player", player_names, index=None, placeholder="Search...", key="bt_player")
        with col2:
            bt_market = st.selectbox("Market", ["PTS", "REB", "AST", "PRA", "RA"], key="bt_market")
        with col3:
            bt_days = st.number_input("Games", 10, 50, 30, key="bt_days")
        
        with st.expander(" Advanced", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                bt_lookback = st.slider("Lookback", 5, 20, 15, key="bt_lookback")
            with c2:
                bt_offset = st.number_input("Line Offset", -10.0, 10.0, 0.0, 0.5, key="bt_offset")
            with c3:
                bt_spread = st.number_input("Spread", -20.0, 20.0, 0.0, 0.5, key="bt_spread")
        
        submitted = st.form_submit_button(" Run Backtest", type="primary")

    #  LOGIC EXECUTION
    if submitted:
        if not bt_player:
            st.error("Please select a player to backtest.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(pct):
                progress_bar.progress(pct)
                status_text.text(f"Processing... {pct:.0%}")
            
            with st.spinner(f"Running backtest for {bt_player}..."):
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
            else:
                # Save to session state so it survives refreshes
                st.session_state['last_backtest_summary'] = summary

    #  RESULTS DISPLAY (Persists via session state)
    if 'last_backtest_summary' in st.session_state:
        summary = st.session_state['last_backtest_summary']
        
        # Safety Check: Ensure the dataframe isn't empty before accessing it
        if summary.results_df is None or summary.results_df.empty:
            st.warning("No results to display.")
            return

        #  FIX 1: Get player name safely from the DATAFRAME, not the widget
        try:
            safe_player_name = summary.results_df['player'].iloc[0]
        except:
            safe_player_name = "Backtest_Result"
            
        st.success(f" {summary.total_predictions} predictions analyzed for {safe_player_name}")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Win Rate", f"{summary.win_rate:.0%}")
        c2.metric("ROI", f"{summary.roi:+.1%}")
        c3.metric("Brier", f"{summary.brier_score:.3f}")
        c4.metric("Record", f"{summary.wins}-{summary.losses}")
        
        with st.expander(" Detailed Results", expanded=True):
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
                #  FIX 2: Replace use_container_width with width="stretch"
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
                #  FIX 2: Replace use_container_width with width="stretch"
                st.dataframe(pd.DataFrame(grade_data), hide_index=True, width="stretch")
            
            #  FIX 2: Replace use_container_width with width="stretch"
            st.dataframe(summary.results_df, hide_index=True, width="stretch")
            
            csv = summary.results_df.to_csv(index=False)
            #  FIX 1: Use the safe_player_name variable we created above
            st.download_button(
                " Download CSV", 
                csv, 
                f"backtest_{safe_player_name.replace(' ', '_')}.csv", 
                "text/csv"
            )

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

@st.cache_data(ttl=600, show_spinner=False)
def get_cached_watchlist_stats(player_id: int, market: str, line: float, _loader) -> dict:
    try:
        time.sleep(0.6)
        df = _loader.fetch_game_logs(player_id)
        if df.empty or market not in df.columns: 
            return None
        
        recent = df.tail(15)
        ema = recent[market].ewm(span=len(recent), adjust=False).mean().iloc[-1]
        l5_hit_rate = (df.tail(5)[market] > line).mean()
        
        return {"ema": ema, "l5_hit_rate": l5_hit_rate, "valid": True}
    except:
        return None

def render_watchlist_tab(orchestrator):
    """
    Optimized Watchlist: 
    1. Uses st.form to batch inputs (No more refreshing while typing).
    2. Uses cached helper to calculate stats (No more freezing).
    """
    st.markdown("###  Watchlist")
    
    # --- LOAD DATA ---
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = load_watchlist()

    # --- SECTION 1: ADD PLAYER (OPTIMIZED) ---
    with st.container(border=True):
        st.caption("Add to Watchlist")
        
        #  THE FIX: Wrapping inputs in a form stops the "Triple Reload"
        with st.form(key="add_watchlist_form", clear_on_submit=True):
            c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
            
            with c1:
                all_players = get_all_nba_players()
                active_players = [p['full_name'] for p in all_players if p.get('is_active', True)]
                # Note: We load the list once. Streamlit handles the search UI efficiently.
                target_player = st.selectbox("Player", active_players, index=None, placeholder="Search Name...")
            
            with c2:
                target_market = st.selectbox("Stat", ["PTS", "REB", "AST", "PRA", "RA"])
                
            with c3:
                target_line = st.number_input("Line", 0.5, 100.0, 15.5, 0.5)

            with c4:
                st.write(" ")  # Spacer for alignment (non-empty for accessibility)
                # The form only submits when this is clicked
                submit_add = st.form_submit_button("Add ", use_container_width="stretch")

        # Logic runs ONLY on submit
        if submit_add:
            if target_player:
                # Check for duplicates
                if not any(x['player'] == target_player and x['market'] == target_market for x in st.session_state.watchlist):
                    import uuid
                    st.session_state.watchlist.append({
                        'id': str(uuid.uuid4())[:8],
                        'player': target_player, 
                        'market': target_market, 
                        'line': target_line,
                        'added_date': datetime.now().strftime("%Y-%m-%d")
                    })
                    save_watchlist(st.session_state.watchlist)
                    st.rerun()
            else:
                st.warning(" Please select a player.")

    # --- SECTION 2: VIEW LIST ---
    if not st.session_state.watchlist:
        st.info("Watchlist empty.")
    else:
        st.write("---")
        h1, h2, h3, h4 = st.columns([2, 1, 2, 0.5])
        h1.caption("Player")
        h2.caption("Target Line")
        h3.caption("Trends (Cached )")
        
        # Create a copy to iterate safely
        current_list = st.session_state.watchlist.copy()
        
        for i, item in enumerate(current_list):
            item_id = item.get('id', str(i))
            
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([2, 1, 2, 0.5])
                
                with c1:
                    st.subheader(item['player'])
                    st.caption(f"**{item['market']}**")
                
                with c2:
                    # Note: Changing this DOES trigger a reload, which is good! 
                    # We want to see the new offset immediately.
                    # Because stats are cached, the reload will be instant.
                    new_line = st.number_input(
                        "Line", 
                        value=float(item.get('line', 10.5)), 
                        step=0.5, 
                        key=f"wl_line_{item_id}",
                        label_visibility="collapsed"
                    )
                    
                    # Save if changed
                    if new_line != item.get('line'):
                        st.session_state.watchlist[i]['line'] = new_line
                        save_watchlist(st.session_state.watchlist)

                with c3:
                    #  CACHED LOOKUP (Fast)
                    p_obj = next((p for p in all_players if p['full_name'] == item['player']), None)
                    
                    if p_obj:
                        stats = get_cached_watchlist_stats(
                            player_id=p_obj['id'], 
                            market=item['market'], 
                            line=new_line,
                            _loader=orchestrator.data_loader # Pass loader for data access
                        )
                        
                        if stats and stats['valid']:
                            # Visuals
                            ema = stats['ema']
                            offset = ema - new_line
                            off_color = "green" if offset > 0 else "red"
                            off_sym = "" if offset > 0 else ""
                            
                            l5 = stats['l5_hit_rate']
                            if l5 >= 0.8: icon, color = "", "green"
                            elif l5 >= 0.6: icon, color = "", "green"
                            elif l5 >= 0.4: icon, color = "", "orange"
                            else: icon, color = "", "red"

                            st.markdown(f"**EMA:** {ema:.1f} :{off_color}[{offset:+.1f}{off_sym}]")
                            st.markdown(f"**L5:** :{color}[{l5:.0%}] {icon}")
                        else:
                            st.caption("Need more data")
                    else:
                        st.caption("Player ID not found")

                with c4:
                    # Accessibility: provide a hidden label instead of empty string
                    if st.button("Remove watchlist item", key=f"del_{item_id}"):
                        st.session_state.watchlist.pop(i)
                        save_watchlist(st.session_state.watchlist)
                        st.rerun()

def render_parlay_tab(parlay_tracker: ParlayTracker, bankroll: float, bankroll_enabled: bool = True):
    """Render compact parlay tab."""
    st.markdown("###  Parlay Builder")
    
    if not bankroll_enabled:
        st.info(" **Data Collection Mode**  Parlay stakes disabled. Toggle Bankroll Mode in sidebar to enable.")
    
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
                if c2.button("", key=f"remove_leg_{i}"):
                    st.session_state.parlay_builder.pop(i)
                    st.rerun()
        
        # Multivariate Correlated SGP Monte Carlo simulation
        from simulation_engine import SimulationEngine
        sim_engine = SimulationEngine()
        sim_res = sim_engine.simulate_correlated_parlay(builder)
        joint_prob = sim_res.get("joint_prob", combined_prob)
        ind_prob = sim_res.get("independent_prob", combined_prob)
        corr_edge = sim_res.get("correlation_edge_pct", 0.0)
        fair_odds = sim_res.get("fair_decimal_odds", 1.0 / max(joint_prob, 1e-4))
        parlay_ev = (joint_prob * combined_odds - 1.0) * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Book Odds", f"{combined_odds:.2f}")
        c2.metric("SGP Joint Prob", f"{joint_prob:.1%}", help="Cholesky Multivariate Monte Carlo Joint Win Probability")
        c3.metric("Fair Odds", f"{fair_odds:.2f}")
        c4.metric("Parlay EV", f"{parlay_ev:+.1f}%", delta=f"{corr_edge:+.1f}% corr")

        if len(builder) >= 2:
            if corr_edge > 2.0:
                st.success(f"🔥 **Positive Synergy!** Correlated legs boost true win prob by **{corr_edge:+.1f}%** over bookmaker assumption.")
            elif corr_edge < -2.0:
                st.warning(f"⚠️ **Cannibalization Drag!** Correlated legs reduce joint prob by **{corr_edge:.1f}%** below naive multiplication.")
        
        c1, c2 = st.columns(2)
        with c1:
            if bankroll_enabled and bankroll > 0:
                parlay_stake = st.number_input("Stake", 10.0, float(bankroll), 50.0, 10.0, key="parlay_stake")
            else:
                parlay_stake = 0.0
                st.caption(" Stake tracking disabled")
        with c2:
            parlay_name = st.text_input("Name", placeholder="Optional...", key="parlay_name")
        
        c1, c2 = st.columns(2)
        if c1.button(" Create", type="primary", use_container_width="stretch"):
            parlay_tracker.create_parlay(legs=builder.copy(), stake=parlay_stake, name=parlay_name)
            st.session_state.parlay_builder = []
            st.toast("Parlay created!")
            st.rerun()
        if c2.button(" Clear", use_container_width="stretch"):
            st.session_state.parlay_builder = []
            st.rerun()
    else:
        st.info("Add legs from Analyze tab.")
    
    # History
    with st.expander(" History", expanded=False):
        stats = parlay_tracker.get_stats()
        if stats['total_parlays'] > 0:
            c1, c2, c3 = st.columns(3)
            c1.metric("Win%", f"{stats['win_rate']:.0%}")
            c2.metric("P/L", f"{stats['total_profit']:+,.0f}")
            c3.metric("Legs", f"{stats['total_legs_hit']}/{stats['total_legs']}")
        
        parlays = parlay_tracker.get_parlays()
        for parlay in sorted(parlays, key=lambda x: x['id'], reverse=True):
            result = parlay['result']
            icon = "" if result == 'Win' else "" if result == 'Loss' else ""
            
            with st.container(border=True):
                col1, col2, col3 = st.columns([3, 1, 0.5])
                with col1:
                    parlay_name = parlay.get('name') or f"#{parlay['id']}"
                    st.markdown(f"{icon} **{parlay_name}** ({parlay['num_legs']})")
                    st.caption(f"{parlay['stake']:.0f} @ {parlay['combined_odds']:.2f}")
                with col2:
                    if result == 'Win': 
                        st.markdown(f"**+{parlay['actual_payout'] - parlay['stake']:,.0f}**")
                    elif result == 'Loss':
                        st.markdown(f"**-{parlay['stake']:,.0f}**")
                    else:
                        st.markdown(f"*{parlay['potential_profit']:,.0f}*")
                with col3:
                    # Accessibility: give button a descriptive (hidden) label
                    if st.button("Delete parlay", key=f"del_parlay_{parlay['id']}"):
                        parlay_tracker.delete_parlay(parlay['id'])
                        st.rerun()
                
                with st.expander(f"Legs ({parlay['legs_hit']}/{parlay['legs_decided']})"):
                    for i, leg in enumerate(parlay['legs']):
                        leg_result = leg.get('result', 'Pending')
                        leg_icon = "" if leg_result == 'Win' else "" if leg_result == 'Loss' else "" if leg_result == 'Push' else ""
                        
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
        
        if st.button(" Clear All", key="clear_parlays"):
            parlay_tracker.clear_history()
            st.rerun()


def render_ml_data_tab(tracker: Tracker):
    """Render enhanced ML data tab with better preview."""
    st.markdown("###  ML Training Data")
    
    from persistence_utils import load_ml_training_data
    parquet_path = DATA_DIR / "ml_training_data.parquet"
    csv_path = DATA_DIR / "ml_training_data.csv"
    has_data = parquet_path.exists() or csv_path.exists()
    
    # =========================================================================
    # SECTION 1: THEORETICAL PERFORMANCE (The Math - from Parquet/CSV)
    # =========================================================================
    with st.expander(" Theoretical Performance (The Math)", expanded=False):
        st.caption("Based on simulation data from ml_training_data.parquet / .csv")
        
        if has_data:
            try:
                csv_df = load_ml_training_data(DATA_DIR)
                skipped_warning = False
                
                # Quick check for corruption
                try:
                    with open(csv_path, 'r', encoding='utf-8') as f:
                        raw_line_count = sum(1 for _ in f)
                    if raw_line_count > len(csv_df) + 10:
                        skipped_warning = True
                        st.warning(f" CSV has {raw_line_count - len(csv_df) - 1} corrupted rows. Click 'Repair CSV' below.")
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
                    st.markdown("** Theoretical Win Rate (All Sim Bets)**")
                    th_c1, th_c2, th_c3, th_c4 = st.columns(4)
                    th_c1.metric("Total Samples", f"{total_sim_bets:,}")
                    th_c2.metric("Win Rate", f"{sim_win_rate:.1%}")
                    if sim_net_units >= 0:
                        th_c3.metric("Net Units", f"+{sim_net_units:.1f}u")
                    else:
                        th_c3.metric("Net Units", f"{sim_net_units:.1f}u")
                    th_c4.metric("ROI", f"{sim_roi:+.1f}%")
                    
                    st.markdown("---")
                    st.markdown("** Quality Units (Offset Rule - V16)**")
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
                    if st.button(" Repair CSV", type="primary"):
                        clean_df = pd.read_csv(csv_path, on_bad_lines='skip')
                        backup_path = DATA_DIR / "ml_training_data_backup.csv"
                        import shutil
                        shutil.copy(csv_path, backup_path)
                        clean_df.to_csv(csv_path, index=False)
                        st.success(f" Repaired! Kept {len(clean_df):,} valid rows. Backup saved to {backup_path.name}")
                        st.rerun()
                    
            except Exception as e:
                st.warning(f"Could not read CSV file: {e}")
        else:
            st.info(" No training data file found. Generate data using backtest or track bets to create one.")
    
    # =========================================================================
    # SECTION 2: REAL PERFORMANCE (The Human - from tracker.history ONLY)
    # =========================================================================
    with st.expander(" Real Performance (Your Actual Bets)", expanded=True):
        st.caption("Based on YOUR tracked bets in bet_tracker.json - this is what actually happened")
        
        tracker_bets = tracker.get_bets()
        # Only consider bets where a positive stake was placed for "Real Performance"
        decided_bets = [b for b in tracker_bets if b.get('result') in ['Win', 'Loss'] and float(b.get('stake', 0) or 0) > 0]

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
                st.caption(f" {len(pending_bets)} pending bets awaiting results")
            
            # Real performance chart (cumulative units over time)
            st.markdown("** Cumulative Units Over Time**")
            
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
            st.info(" No decided bets yet. Track bets and update their results to see your real performance.")
    
    # =========================================================================
    # SECTION 2.5: RETRAIN RECOMMENDATION
    # =========================================================================
    with st.expander(" Retrain Recommendation", expanded=True):
        retrain_rec = tracker.get_retrain_recommendation()
        
        r1, r2, r3 = st.columns(3)
        r1.metric("Sample Size", f"{retrain_rec['sample_size']} bets")
        r2.metric("Win Rate", f"{retrain_rec['win_rate']*100:.1f}%")
        
        if retrain_rec['days_since_retrain'] is not None:
            r3.metric("Days Since Retrain", retrain_rec['days_since_retrain'])
        else:
            r3.metric("Days Since Retrain", "N/A")
        
        if retrain_rec['should_retrain']:
            st.warning(f"⚠️ **RETRAIN RECOMMENDED:** {retrain_rec['reason']}")
            if st.button(" Mark Retrain Complete"):
                tracker.mark_retrain_complete()
                st.success("Retrain marked complete!")
                st.rerun()
        else:
            st.success(f"✅ **Model OK:** {retrain_rec['reason']}")
    
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
        with st.expander(" Data Quality Summary", expanded=False):
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
                loss_cols[0].write(f" Bad Beat: {quality_counts.get('bad_beat', 0)}")
                loss_cols[1].write(f" Close Loss: {quality_counts.get('close_loss', 0)}")
                loss_cols[2].write(f" Clear Loss: {quality_counts.get('clear_loss', 0)}")
                loss_cols[3].write(f" Bad Read: {quality_counts.get('bad_read', 0)}")
                
                # Wins
                win_cols = st.columns(4)
                win_cols[0].write(f" Sweat Win: {quality_counts.get('sweat_win', 0)}")
                win_cols[1].write(f" Close Win: {quality_counts.get('close_win', 0)}")
                win_cols[2].write(f" Solid Win: {quality_counts.get('solid_win', 0)}")
                win_cols[3].write(f" Blowout Win: {quality_counts.get('blowout_win', 0)}")
        
        # Tabbed data views
        view_tab1, view_tab2, view_tab3 = st.tabs([" Summary", " Features", " Raw Data"])
        
        with view_tab1:
            # Key columns for quick overview
            summary_cols = ['player', 'market', 'line', 'side', 'result', 'margin', 'result_quality', 'feat_prob', 'feat_ev']
            display_cols = [c for c in summary_cols if c in training_df.columns]
            if display_cols:
                st.dataframe(safe_clean_for_arrow(training_df[display_cols].tail(20)), hide_index=True, width="stretch")
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
                st.dataframe(safe_clean_for_arrow(display_df[show_cols].tail(20)), hide_index=True, width="stretch")
            else:
                st.info("No feature columns found")
        
        with view_tab3:
            # Full raw data with column count
            st.caption(f"All {len(training_df.columns)} columns  {len(training_df)} rows")
            st.dataframe(safe_clean_for_arrow(training_df.tail(30)), hide_index=True, width="stretch")
        
        # Download buttons
        st.markdown("---")
        dc1, dc2, dc3 = st.columns(3)
        csv_data = training_df.to_csv(index=False)
        dc1.download_button(" Download CSV", csv_data, "ml_training_data.csv", "text/csv", use_container_width="stretch")
        json_data = training_df.to_json(orient='records', indent=2)
        dc2.download_button(" Download JSON", json_data, "ml_training_data.json", "application/json", use_container_width="stretch")
        
        # Show column list
        with dc3:
            if st.button(" Show Columns", use_container_width="stretch"):
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
        
        st.caption(f" Need 100-200+ samples for XGBoost. Currently: {feature_stats['decided_with_features']}")
    else: 
        c4.metric("Win Rate", "N/A")
        st.info("Track bets to collect training data. Log bets with the  Track button after analysis.")


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
            issues.append(f" {mismatches} rows ({mismatches/len(df)*100:.1f}%) have incorrect 'hit' labels!")
    
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
    from xgboost import XGBRegressor
    from sklearn.isotonic import IsotonicRegression
    """
    V21.5 QUANTILE REGRESSION: Training UI.
    
    Trains 3 XGBRegressor models for quantiles 0.05, 0.50, 0.95.
    - Removes Gaussian assumption (handles skewed player distributions)
    - Wider 90% intervals fix overconfidence (old 80% only hit 78%)
    - P(Over) calculated via interpolation, not CDF
    """
    st.markdown("### Train V21.5 Quantile Model")
    st.info("""
    **V21.5 CALIBRATED QUANTILE REGRESSION**: Wider 90% intervals fix overconfidence.
    
    **Calibration Fix:**
    -  Old q10/q90 (80% target) → only 78% actual coverage = overconfident
    -  **New q05/q95 (90% target) → properly calibrated probabilities**
    -  No more betting big on "90% certainty" that only hits 78%
    
    **Benefits:**
    -  Per-market models: PTS, REB, AST each get specialized quantile predictions
    -  No Gaussian assumption - handles skewed player distributions
    -  Backward compatible with old models (auto-detected)
    """)
    
    # Check if data exists (Parquet or CSV)
    from persistence_utils import load_ml_training_data
    parquet_path = DATA_DIR / "ml_training_data.parquet"
    csv_path = DATA_DIR / "ml_training_data.csv"
    
    if not parquet_path.exists() and not csv_path.exists():
        st.error(" Training data not found.")
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
    st.markdown("####  V20.3 Schema Validation")
    st.caption(f"Required features: {len(V20_REQUIRED_FEATURES)} (V20.3 empirical)")
    
    if st.button(" Validate Dataset", type="secondary"):
        with st.spinner("Validating against V20.3 schema..."):
            df = load_ml_training_data(DATA_DIR)
            
            # === STRICT VALIDATION ===
            errors = []
            
            # 1. Check required features (NO inference, NO remapping)
            missing_features = [f for f in V20_REQUIRED_FEATURES if f not in df.columns]
            if missing_features:
                errors.append(f" Missing V20.3 features ({len(missing_features)}): {', '.join(missing_features)}")
            
            # 2. Check regression target exists (NO fallback to classification)
            if V20_REQUIRED_TARGET not in df.columns:
                errors.append(f" Missing regression target: '{V20_REQUIRED_TARGET}'")
            
            # 3. Check metadata columns
            missing_meta = [m for m in V20_REQUIRED_METADATA if m not in df.columns]
            if missing_meta:
                errors.append(f" Missing metadata: {', '.join(missing_meta)}")
            
            # 4. Detect legacy/heuristic columns (should NOT exist in V20 data)
            legacy_columns = [
                'feat_usage_mult', 'feat_rest_factor', 'feat_matchup_mult',
                'feat_blowout_prob', 'feat_fatigue_mult', 'feat_position_mult',
                'feat_opp_drtg_l5', 'feat_hit_rate_l5', 'feat_hit_rate_l10'
            ]
            found_legacy = [c for c in legacy_columns if c in df.columns]
            if found_legacy:
                errors.append(f" Legacy columns detected (invalid for V20): {', '.join(found_legacy)}")
            
            # === DISPLAY RESULTS ===
            if errors:
                st.error(" **Dataset is INVALID for V20.2 training**")
                for err in errors:
                    st.write(f"   {err}")
                st.warning("**Action Required:** Regenerate dataset using V20.2 pipeline.")
                st.caption("Go to 'Generate Dataset' tab  Run generation  Return here to train.")
            else:
                st.success(" **Dataset passes V20.3 schema validation**")
                st.write(f"   Rows: {len(df):,}")
                st.write(f"   Features: {len(V20_REQUIRED_FEATURES)} (all present)")
                st.write(f"   Target: {V20_REQUIRED_TARGET}")
                st.write(f"   Players: {df['player'].nunique()}")
                st.write(f"   Markets: {df['market'].nunique()}")
    
    st.markdown("---")
    
    # =========================================================================
    # V21.1 PER-MARKET TRAINING OPTIONS
    # =========================================================================
    st.markdown("####  Training Mode")
    
    training_mode = st.radio(
        "Select training mode:",
        options=["per_market", "universal"],
        format_func=lambda x: " Per-Market Models (Recommended)" if x == "per_market" else " Universal Model (All markets)",
        horizontal=True,
        help="Per-market trains separate models for PTS, REB, AST, etc. Universal trains one model for all markets."
    )
    
    # Market selection for per-market mode
    all_markets = ['PTS', 'REB', 'AST', 'PRA', 'PR', 'PA', 'RA', '3PM', 'STL', 'BLK']
    selected_markets = all_markets  # Default: all
    
    if training_mode == "per_market":
        st.markdown("##### Select markets to train:")
        col1, col2 = st.columns(2)
        with col1:
            select_all = st.checkbox("Select All Markets", value=True)
        with col2:
            min_samples = st.number_input("Min samples per market", min_value=50, value=100, step=50,
                                          help="Markets with fewer samples will use universal fallback")
        
        if not select_all:
            selected_markets = st.multiselect(
                "Choose markets:",
                options=all_markets,
                default=['PTS', 'REB', 'AST', 'PRA'],
                help="Hold Ctrl/Cmd to select multiple"
            )
    
    st.markdown("---")
    
    if st.button(" Start Training", type="primary"):
        status = st.status("Training in progress...", expanded=True)
        try:
            # 1. Load Data (High-performance Parquet preferred)
            status.write("Loading dataset (preferring Parquet)...")
            df = load_ml_training_data(DATA_DIR)
            
            # =========================================================
            # V21.1 QUANTILE: Train q10, q50, q90 models
            # NO LEGACY COMPATIBILITY - strict schema enforcement
            # =========================================================
            
            # Check required features - ALL MUST EXIST
            missing_features = [f for f in V20_REQUIRED_FEATURES if f not in df.columns]
            if missing_features:
                status.update(label=" Schema Validation Failed", state="error")
                st.error(f"**FATAL:** Missing V20.3 features ({len(missing_features)}):")
                for f in missing_features:
                    st.write(f"   `{f}`")
                st.warning("Dataset is incompatible with V20.3. Regenerate using 'Generate Dataset' tab.")
                st.info("V20.3 requires 27 features. Your dataset may have an older schema.")
                return
            
            # Check regression target (NO fallback to classification)
            if V20_REQUIRED_TARGET not in df.columns:
                status.update(label=" Schema Validation Failed", state="error")
                st.error(f"**FATAL:** Missing regression target '{V20_REQUIRED_TARGET}'")
                st.warning("V21 requires actual stat values for regression. Regenerate dataset.")
                return
            
            status.write(f" Schema validation passed ({len(V20_REQUIRED_FEATURES)} features)")
            
            # XGBoost base params
            # V21.6 REGULARISATION: Explicit complexity penalties + early stopping
            # min_child_weight=5: prevents tiny leaves memorising rare players
            # reg_alpha/lambda: L1+L2 penalty on leaf weights
            # gamma: requires minimum loss reduction to split (structural regulariser)
            # n_estimators is an upper bound — early stopping will halt before this
            from hardware_utils import get_xgboost_hardware_params
            # Load tuned GPU hyperparameters if available
            market_gpu_params = {}
            hp_path = Path("config/model_hyperparams.json")
            if hp_path.exists():
                try:
                    with open(hp_path, "r", encoding="utf-8") as f:
                        hp_cfg = json.load(f)
                    if model_name in hp_cfg and isinstance(hp_cfg[model_name], dict):
                        market_gpu_params = hp_cfg[model_name]
                        st.caption(f"⚡ Loaded GPU-tuned hyperparameters for {model_name}")
                except Exception:
                    pass

            base_params = get_xgboost_hardware_params({
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
                **market_gpu_params,
            })
            
            # V21.5 CALIBRATION FIX: Wider quantiles (90% interval)
            # Old: [0.10, 0.50, 0.90] gave 78% coverage (underconfident)
            # New: [0.05, 0.50, 0.95] targets 90% coverage (properly calibrated)
            quantiles = [0.05, 0.50, 0.95]
            quantile_names = ['q05', 'q50', 'q95']
            
            def _interpolate_prob(line, q_low, q50, q_high):
                """V22: Split-normal CDF (matches ModelEngine fix)."""
                z_val = 1.645  # wide quantiles (q05/q95)
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

            def train_quantile_models(X_train, y_train, X_test, y_test, model_name: str, lines_train=None, lines_test=None):
                """Train q05/q50/q95 models + optional isotonic calibrator.

                V21.6 UPGRADES:
                - Early stopping: halts training when val RMSE stops improving (25 rounds)
                - evals_result: captures train vs val RMSE per boosting round for learning curves
                - Calibrator fitted on chronological val slice from training set (no test leakage)
                """
                from sklearn.model_selection import TimeSeriesSplit

                quantile_models = {}
                all_evals = {}  # {q_name: evals_result dict}
                best_iters = {}  # record early-stop iteration per quantile

                for q, name in zip(quantiles, quantile_names):
                    evals_result = {}
                    use_early_stop = (name == 'q50')
                    params = {
                        **base_params,
                        'objective': 'reg:quantileerror',
                        'quantile_alpha': q,
                        'eval_metric': 'mae',
                    }
                    if use_early_stop:
                        params['early_stopping_rounds'] = 25

                    model = XGBRegressor(**params)
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_train, y_train), (X_test, y_test)],
                        verbose=False,
                    )
                    # Capture evals via booster
                    try:
                        evals_result = model.evals_result()
                    except Exception:
                        evals_result = {}
                    quantile_models[name] = model
                    all_evals[name] = evals_result
                    best_iter = getattr(model, 'best_iteration', base_params['n_estimators'])
                    best_iters[name] = best_iter
                    logger.info(f"  {model_name}/{name}: {'early stop @' if use_early_stop else 'ran'} iter {best_iter}")

                # Mark as wide quantile model for backward compat detection
                quantile_models['_wide_quantile'] = True
                # return full history so caller can plot/inspect
                # Store learning curves for UI rendering
                quantile_models['_evals_result'] = all_evals

                # Evaluate on held-out TEST set
                pred_05 = quantile_models['q05'].predict(X_test)
                pred_50 = quantile_models['q50'].predict(X_test)
                pred_95 = quantile_models['q95'].predict(X_test)

                mae = mean_absolute_error(y_test, pred_50)
                r2 = r2_score(y_test, pred_50)
                in_interval = ((y_test >= pred_05) & (y_test <= pred_95)).mean()
                coverage = in_interval * 100
                avg_width = (pred_95 - pred_05).mean()

                # === TIME-SERIES CROSS-VALIDATION (5 folds) ===
                # Runs on median quantile (q50) only for speed.
                # Fold MAEs tell us if the model generalises across seasons
                # or just memorises the most recent one.
                cv_mae_folds = []
                try:
                    tscv = TimeSeriesSplit(n_splits=5)
                    X_cv = pd.concat([X_train, X_test])
                    y_cv = pd.concat([y_train, y_test])
                    for fold_train_idx, fold_val_idx in tscv.split(X_cv):
                        fold_model = XGBRegressor(
                            **base_params,
                            objective='reg:quantileerror',
                            quantile_alpha=0.50,
                            eval_metric='mae',
                            early_stopping_rounds=25,
                        )
                        Xf_tr, Xf_val = X_cv.iloc[fold_train_idx], X_cv.iloc[fold_val_idx]
                        yf_tr, yf_val = y_cv.iloc[fold_train_idx], y_cv.iloc[fold_val_idx]
                        fold_model.fit(
                            Xf_tr, yf_tr,
                            eval_set=[(Xf_val, yf_val)],
                            verbose=False,
                        )
                        fold_pred = fold_model.predict(Xf_val)
                        cv_mae_folds.append(mean_absolute_error(yf_val, fold_pred))
                except Exception as e:
                    logger.warning(f"CV failed for {model_name}: {e}")

                cv_mae_mean = float(np.mean(cv_mae_folds)) if cv_mae_folds else None
                cv_mae_std  = float(np.std(cv_mae_folds))  if cv_mae_folds else None

                # === CALIBRATION (Isotonic Regression) ===
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
                        from model_engine import SigmoidCalibrator
                        cal = SigmoidCalibrator(clip_min=0.01, clip_max=0.99)
                        cal.fit(raw_probs_val, actual_over_val)
                        calibrator = cal

                        if raw_probs_test is not None and actual_over_test is not None:
                            cal_probs = cal.predict(raw_probs_test)
                            cal_delta = abs(cal_probs.mean() - actual_over_test.mean()) - abs(raw_probs_test.mean() - actual_over_test.mean())
                    except Exception as e:
                        logger.warning(f"Calibration fitting failed for {model_name}: {e}")

                return quantile_models, calibrator, {
                    'name': model_name,
                    'samples': len(y_train) + len(y_test),
                    'mae': mae,
                    'r2': r2,
                    'coverage': coverage,
                    'avg_width': avg_width,
                    'calibrated': calibrator is not None,
                    'cal_delta': round(cal_delta, 4),
                    'cv_mae_mean': round(cv_mae_mean, 4) if cv_mae_mean is not None else None,
                    'cv_mae_std':  round(cv_mae_std,  4) if cv_mae_std  is not None else None,
                    'evals_result': all_evals,
                }, best_iters
            
            all_metrics = []
            
            if training_mode == "per_market":
                # =========================================================
                # PER-MARKET TRAINING
                # =========================================================
                status.write(f" Training per-market models for {len(selected_markets)} markets...")
                
                # Check market column exists
                if 'market' not in df.columns:
                    status.update(label=" Schema Validation Failed", state="error")
                    st.error("**FATAL:** 'market' column not found in dataset")
                    return
                
                trained_markets = []
                skipped_markets = []
                
                for market in selected_markets:
                    market_df = df[df['market'] == market]
                    
                    if len(market_df) < min_samples:
                        status.write(f"   {market}: Skipping (only {len(market_df)} samples, need {min_samples})")
                        skipped_markets.append((market, len(market_df)))
                        continue
                    
                    status.write(f"   {market}: Training on {len(market_df)} samples...")
                    
                    # Prepare data
                    X = market_df[V20_REQUIRED_FEATURES].apply(pd.to_numeric, errors='coerce')
                    y = pd.to_numeric(market_df[V20_REQUIRED_TARGET], errors='coerce')
                    
                    # Skip if too many NaN
                    if X.isna().sum().sum() > len(X) * 0.1:
                        status.write(f"   {market}: Skipping (too many NaN values)")
                        skipped_markets.append((market, 'NaN'))
                        continue
                    
                    # Fill remaining NaN with median for training stability
                    X = X.fillna(X.median())
                    
                    # Drop rows where target is NaN
                    valid_mask = ~y.isna()
                    X = X[valid_mask]
                    y = y[valid_mask]
                    
                    if len(y) < min_samples:
                        skipped_markets.append((market, len(y)))
                        continue
                    
                    # Get lines for calibration (from feat_line feature or metadata 'line' column)
                    if 'feat_line' in market_df.columns:
                        market_lines = pd.to_numeric(market_df['feat_line'], errors='coerce').fillna(0)
                    elif 'line' in market_df.columns:
                        market_lines = pd.to_numeric(market_df['line'], errors='coerce').fillna(0)
                    else:
                        market_lines = pd.Series(0, index=market_df.index)
                    market_lines = market_lines[valid_mask]
                    
                    # Time-aware split: sort chronologically and perform a non-shuffled holdout
                    # This prevents future->past leakage and gives an out-of-time estimate.
                    if 'date' in market_df.columns:
                        ordered_idx = market_df.loc[valid_mask].sort_values('date').index
                        X = X.loc[ordered_idx]
                        y = y.loc[ordered_idx]
                        market_lines = market_lines.loc[ordered_idx]

                    X_train, X_test, y_train, y_test, lines_train, lines_test = train_test_split(
                        X, y, market_lines, test_size=0.2, shuffle=False
                    )
                    # temporal cross-validation on training set
                    tscv = TimeSeriesSplit(n_splits=5)
                    fold_scores = []
                    for tr_idx, val_idx in tscv.split(X_train):
                        m_cv = XGBRegressor(**base_params)
                        m_cv.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
                        pred_cv = m_cv.predict(X_train.iloc[val_idx])
                        prob_cv = (pred_cv >= lines_train.iloc[val_idx]).astype(float)
                        actual_cv = (y_train.iloc[val_idx] > lines_train.iloc[val_idx]).astype(float)
                        fold_scores.append(accuracy_score(actual_cv, prob_cv))
                    status.write(f"    CV win rates: {fold_scores}, std={np.std(fold_scores):.3f}")
                    
                    # Train quantile models + calibrator (pass both train/test lines for validation split)
                    ret = train_quantile_models(
                        X_train, y_train, X_test, y_test, market, lines_train=lines_train.values, lines_test=lines_test.values
                    )
                    # function now returns (models, calibrator, metrics, best_iters)
                    quantile_models, calibrator, metrics, best_iters = ret
                    # extract evals history from metrics dict
                    evals_hist = metrics.get('evals_result', {})
                    all_metrics.append(metrics)
                    # learning curves and early stop info
                    status.write(f"    {market}: best iters {best_iters}")
                    # plot q50 curve if available
                    if 'q50' in evals_hist:
                        curve = pd.DataFrame({
                            'train_rmse': evals_hist['q50']['validation_0']['rmse'],
                            'val_rmse': evals_hist['q50']['validation_1']['rmse']
                        })
                        status.line_chart(curve)
                    
                    # Save per-market model
                    model_path = DATA_DIR / f"nba_quantile_{market}.pkl"
                    joblib.dump(quantile_models, model_path)
                    
                    # Save calibrator
                    if calibrator is not None:
                        group = ML_MARKET_GROUPS.get(market, 'universal')
                        cal_path = DATA_DIR / f"nba_calibrator_{group}.pkl"
                        joblib.dump(calibrator, cal_path)
                        status.write(f"    {market}: MAE={metrics['mae']:.2f}, Coverage={metrics['coverage']:.1f}%, Calibrator saved ({group})")
                    else:
                        status.write(f"    {market}: MAE={metrics['mae']:.2f}, Coverage={metrics['coverage']:.1f}%, No calibrator")
                    trained_markets.append(market)
                
                # Also train universal fallback for skipped markets
                if skipped_markets:
                    status.write(" Training universal fallback for markets with insufficient data...")
                    
                    X_all = df[V20_REQUIRED_FEATURES].apply(pd.to_numeric, errors='coerce').fillna(df[V20_REQUIRED_FEATURES].median())
                    y_all = pd.to_numeric(df[V20_REQUIRED_TARGET], errors='coerce')
                    valid_mask = ~y_all.isna()
                    X_all = X_all[valid_mask]
                    y_all = y_all[valid_mask]
                    
                    # Get lines for calibration
                    if 'feat_line' in df.columns:
                        all_lines = pd.to_numeric(df['feat_line'], errors='coerce').fillna(0)[valid_mask]
                    elif 'line' in df.columns:
                        all_lines = pd.to_numeric(df['line'], errors='coerce').fillna(0)[valid_mask]
                    else:
                        all_lines = pd.Series(0, index=X_all.index)
                    
                    # Chronological holdout for universal fallback
                    if 'date' in df.columns:
                        ordered_idx = df[valid_mask].sort_values('date').index
                        X_all = X_all.loc[ordered_idx]
                        y_all = y_all.loc[ordered_idx]
                        all_lines = all_lines.loc[ordered_idx]

                    X_train, X_test, y_train, y_test, lines_train, lines_test = train_test_split(
                        X_all, y_all, all_lines, test_size=0.2, shuffle=False
                    )
                    # CV on universal training set
                    tscv = TimeSeriesSplit(n_splits=5)
                    fold_scores = []
                    for tr_idx, val_idx in tscv.split(X_train):
                        m_cv = XGBRegressor(**base_params)
                        m_cv.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
                        pred_cv = m_cv.predict(X_train.iloc[val_idx])
                        prob_cv = (pred_cv >= lines_train.iloc[val_idx]).astype(float)
                        actual_cv = (y_train.iloc[val_idx] > lines_train.iloc[val_idx]).astype(float)
                        fold_scores.append(accuracy_score(actual_cv, prob_cv))
                    status.write(f"    Universal CV win rates: {fold_scores}, std={np.std(fold_scores):.3f}")
                    ret = train_quantile_models(
                        X_train, y_train, X_test, y_test, "universal", lines_train=lines_train.values, lines_test=lines_test.values
                    )
                    quantile_models, calibrator, metrics, best_iters_uni = ret
                    evals_hist_uni = metrics.get('evals_result', {})
                    all_metrics.append(metrics)
                    status.write(f"    Universal: best iters {best_iters_uni}")
                    if 'q50' in evals_hist_uni:
                        curve_u = pd.DataFrame({
                            'train_rmse': evals_hist_uni['q50']['validation_0']['rmse'],
                            'val_rmse': evals_hist_uni['q50']['validation_1']['rmse']
                        })
                        status.line_chart(curve_u)
                    
                    universal_path = DATA_DIR / "nba_quantile_universal.pkl"
                    joblib.dump(quantile_models, universal_path)
                    if calibrator is not None:
                        joblib.dump(calibrator, DATA_DIR / "nba_calibrator_universal.pkl")
                    status.write(f"    Universal: MAE={metrics['mae']:.2f}, Coverage={metrics['coverage']:.1f}%")
                
                status.write(f" Trained {len(trained_markets)} per-market models")
                if skipped_markets:
                    status.write(f"   Skipped {len(skipped_markets)} markets (using universal fallback)")
                
            else:
                # =========================================================
                # UNIVERSAL TRAINING (all markets combined)
                # =========================================================
                status.write(" Training universal model on all markets...")
                
                X = df[V20_REQUIRED_FEATURES].apply(pd.to_numeric, errors='coerce')
                y = pd.to_numeric(df[V20_REQUIRED_TARGET], errors='coerce')
                
                # Check for NaN
                nan_counts = X.isna().sum()
                if nan_counts.sum() > 0:
                    bad_cols = nan_counts[nan_counts > 0].to_dict()
                    status.write(f"   Warning: NaN values in {len(bad_cols)} features, filling with median")
                    X = X.fillna(X.median())
                
                valid_mask = ~y.isna()
                X = X[valid_mask]
                y = y[valid_mask]
                
                status.write(f"   Processing {len(y)} samples with {len(V20_REQUIRED_FEATURES)} features...")
                
                # Get lines for calibration
                if 'feat_line' in df.columns:
                    uni_lines = pd.to_numeric(df['feat_line'], errors='coerce').fillna(0)[valid_mask]
                elif 'line' in df.columns:
                    uni_lines = pd.to_numeric(df['line'], errors='coerce').fillna(0)[valid_mask]
                else:
                    uni_lines = pd.Series(0, index=X.index)
                
                # Time-aware train/test split: sort by date then hold out the latest 20%
                if 'date' in df.columns:
                    ordered_idx = df[valid_mask].sort_values('date').index
                    X = X.loc[ordered_idx]
                    y = y.loc[ordered_idx]
                    uni_lines = uni_lines.loc[ordered_idx]

                X_train, X_test, y_train, y_test, lines_train, lines_test = train_test_split(
                    X, y, uni_lines, test_size=0.2, shuffle=False
                )
                # CV on universal training set
                tscv = TimeSeriesSplit(n_splits=5)
                fold_scores = []
                for tr_idx, val_idx in tscv.split(X_train):
                    m_cv = XGBRegressor(**base_params)
                    m_cv.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
                    pred_cv = m_cv.predict(X_train.iloc[val_idx])
                    prob_cv = (pred_cv >= lines_train.iloc[val_idx]).astype(float)
                    actual_cv = (y_train.iloc[val_idx] > lines_train.iloc[val_idx]).astype(float)
                    fold_scores.append(accuracy_score(actual_cv, prob_cv))
                status.write(f"    Universal CV win rates: {fold_scores}, std={np.std(fold_scores):.3f}")
                
                # Train quantile models + calibrator (use validation split taken from training set)
                ret = train_quantile_models(
                    X_train, y_train, X_test, y_test, "universal", lines_train=lines_train.values, lines_test=lines_test.values
                )
                quantile_models, calibrator, metrics, best_iters_uni = ret
                evals_hist_uni = metrics.get('evals_result', {})
                all_metrics.append(metrics)
                status.write(f"    Universal: best iters {best_iters_uni}")
                if 'q50' in evals_hist_uni:
                    curve_u = pd.DataFrame({
                        'train_rmse': evals_hist_uni['q50']['validation_0']['rmse'],
                        'val_rmse': evals_hist_uni['q50']['validation_1']['rmse']
                    })
                    status.line_chart(curve_u)
                
                # Save universal model
                universal_path = DATA_DIR / "nba_quantile_universal.pkl"
                joblib.dump(quantile_models, universal_path)
                if calibrator is not None:
                    joblib.dump(calibrator, DATA_DIR / "nba_calibrator_universal.pkl")
                    status.write("   Calibration curve saved")
            
            # Save feature list
            joblib.dump(V20_REQUIRED_FEATURES, DATA_DIR / "nba_features_v21.pkl")
            
            status.update(label=" Training Complete!", state="complete")
            
            st.success(f" V21.1 Quantile Regression Models Trained!")
            
            # Display metrics in table
            st.markdown("#### Model Performance Summary")
            
            # ── Strip non-tabular fields before building the display df ──
            display_rows = []
            for m in all_metrics:
                display_rows.append({k: v for k, v in m.items()
                                     if k not in ('evals_result',)})
            metrics_df = pd.DataFrame(display_rows)
            metrics_df = metrics_df.rename(columns={
                'name': 'Model',
                'samples': 'Samples',
                'mae': 'MAE',
                'r2': 'R²',
                'coverage': 'Coverage %',
                'avg_width': 'Avg Width',
                'calibrated': 'Calibrated',
                'cal_delta': 'Cal Δ',
                'cv_mae_mean': 'CV MAE',
                'cv_mae_std': 'CV Std',
            })

            # Format numeric columns
            for col in ['MAE', 'R²', 'Avg Width', 'Cal Δ', 'CV MAE', 'CV Std']:
                if col in metrics_df.columns:
                    metrics_df[col] = metrics_df[col].apply(
                        lambda x: f"{x:.3f}" if x is not None else "—")
            if 'Coverage %' in metrics_df.columns:
                metrics_df['Coverage %'] = metrics_df['Coverage %'].apply(
                    lambda x: f"{x:.1f}%" if x is not None else "—")

            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            # ── CV Std flag ──
            if 'CV Std' in metrics_df.columns:
                try:
                    max_std = max(
                        float(v) for v in metrics_df['CV Std'] if v not in (None, "—")
                    )
                    if max_std > 1.5:
                        st.warning(f"⚠️ High CV Std ({max_std:.3f}) — model generalisation varies across seasons. "
                                   "Consider adding more historical data or tightening regularisation.")
                    else:
                        st.success(f"✅ CV Std stable ({max_std:.3f}) — model generalises consistently across folds.")
                except Exception:
                    pass

            # ── Learning Curves (q50 only, train vs val RMSE) ──
            st.markdown("#### 📈 Learning Curves — Train vs Validation RMSE (q50)")
            st.caption("Gap between lines = overfitting. Early stopping iteration marked where val RMSE bottomed out.")

            curve_data_found = False
            for m in all_metrics:
                er = m.get('evals_result', {})
                q50_er = er.get('q50', {})
                # evals_result keys are 'validation_0' (train) and 'validation_1' (test)
                train_rmse = q50_er.get('validation_0', {}).get('rmse')
                val_rmse   = q50_er.get('validation_1', {}).get('rmse')
                if train_rmse and val_rmse:
                    curve_data_found = True
                    n = min(len(train_rmse), len(val_rmse))
                    curve_df = pd.DataFrame({
                        'Iteration': list(range(1, n + 1)),
                        'Train RMSE': train_rmse[:n],
                        'Val RMSE':   val_rmse[:n],
                    })
                    best_iter = int(np.argmin(val_rmse[:n])) + 1
                    with st.expander(f"{m['name']} — stopped @ iter {best_iter}/{base_params['n_estimators']}", expanded=False):
                        st.line_chart(curve_df.set_index('Iteration')[['Train RMSE', 'Val RMSE']])
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Best Val RMSE", f"{min(val_rmse[:n]):.3f}", f"@ iter {best_iter}")
                        col_b.metric("Final Train RMSE", f"{train_rmse[best_iter-1]:.3f}")
                        gap = train_rmse[best_iter-1] - min(val_rmse[:n])
                        col_c.metric("Train-Val Gap", f"{gap:+.3f}",
                                     delta_color="inverse" if gap < -0.1 else "normal")

            if not curve_data_found:
                st.info("Learning curve data not available for this run.")
            
            # Overall interpretation
            avg_coverage = np.mean([m['coverage'] for m in all_metrics])
            cal_count = sum(1 for m in all_metrics if m.get('calibrated', False))
            if avg_coverage < 75:
                st.warning(" Low average coverage: Models may be overconfident.")
            elif avg_coverage > 85:
                st.info(" High average coverage: Models are conservative.")
            else:
                st.success(f" Good calibration: Average coverage {avg_coverage:.1f}% is near target 90%.")
            
            if cal_count > 0:
                st.success(f" {cal_count} isotonic calibration curves trained and saved.")
            else:
                st.warning(" No calibration curves were fitted (missing line data in training set).")
                st.caption("Ensure the training CSV has a 'line' or 'feat_line' column.")
            
            if training_mode == "per_market":
                st.info(f"""
                **Per-Market Models Saved:**
                - {len([m for m in all_metrics if m['name'] != 'universal'])} market-specific models
                - Fallback universal model for other markets
                
                **Model lookup priority:** PTS → scoring group → universal
                """)
            
            st.balloons()
            
        except Exception as e:
            status.update(label=" Training Failed", state="error")
            st.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

def render_analytics_tab(tracker: 'Tracker'):
    """
    V21.4: Analytics Dashboard - Performance tracking center.
    Replaces the old Guide tab with actionable performance data.
    
    SMART DATE PARSING: Bets placed between 12:00 AM and 1:00 PM PH Time (UTC+8)
    are attributed to the PREVIOUS day's game slate (since NBA games in the US
    finish late night / early morning PH time).
    """
    st.markdown("### Analytics Dashboard")
    st.caption(f"{CURRENT_VERSION} | Performance Tracking Center")

    bets = tracker.get_bets()

    # =========================================================================
    # HELPER: Smart Date Parsing (PH Timezone Shift Logic)
    # =========================================================================
    def _normalize_calendar_date(value: Any) -> str:
        """Normalize stored dates to YYYY-MM-DD for calendar grouping."""
        if not value:
            return ''

        if isinstance(value, datetime):
            return value.strftime('%Y-%m-%d')

        text = str(value).strip()
        if not text:
            return ''

        try:
            parsed = datetime.fromisoformat(text)
            return parsed.strftime('%Y-%m-%d')
        except ValueError:
            pass

        try:
            return pd.to_datetime(text).strftime('%Y-%m-%d')
        except Exception:
            return text[:10]

    def _get_adjusted_game_date(bet: Dict) -> str:
        """
        Returns the 'Adjusted Game Date' for a bet based on PH timezone logic.
        
        Logic:
        - The bet 'id' field is a Unix timestamp in milliseconds.
        - Convert to PH timezone (UTC+8).
        - If the local PH hour is between 0 (12:00 AM) and 13 (1:00 PM inclusive),
          the bet is for the PREVIOUS day's game slate (US games end late).
        - Otherwise, keep the same date.
        
        Examples (PH Time):
        - Bet at Feb 4 12:07 AM -> Adjusted to Feb 3 (previous day's games)
        - Bet at Feb 4 5:47 PM  -> Stays Feb 4 (current day's games)
        """
        from datetime import timezone
        
        bet_id = bet.get('id')
        fallback_date = bet.get('date', '')
        
        if not bet_id:
            # No timestamp available, use the stored date as-is
            return fallback_date
        
        try:
            # Convert milliseconds timestamp to datetime (UTC)
            ts_seconds = bet_id / 1000.0
            utc_dt = datetime.fromtimestamp(ts_seconds, tz=timezone.utc)
            
            # Convert to PH timezone (UTC+8)
            ph_offset = timedelta(hours=8)
            ph_dt = utc_dt + ph_offset
            
            # Check if hour is between 0 (midnight) and 13 (1 PM) inclusive
            # These bets are for the PREVIOUS day's game slate
            if 0 <= ph_dt.hour <= 13:
                adjusted_dt = ph_dt - timedelta(days=1)
            else:
                adjusted_dt = ph_dt
            
            return adjusted_dt.strftime('%Y-%m-%d')
            
        except (ValueError, TypeError, OSError):
            # If timestamp parsing fails, fall back to stored date
            return _normalize_calendar_date(fallback_date)

    # =========================================================================
    # HELPER: Calculate daily P/L from resolved bets with adjusted dates
    # =========================================================================
    def _build_daily_pl(bets_list: list, parlays_list: list = None) -> Dict[str, Dict]:
        """Build per-date P/L map from resolved bets AND parlays using Adjusted Game Date."""
        daily = {}
        
        # Process individual bets
        for b in bets_list:
            if not isinstance(b, dict):
                continue
            result = b.get('result', 'Pending')
            if result not in ('Win', 'Loss'):
                continue
            
            # Use ADJUSTED game date instead of raw date
            date_str = _get_adjusted_game_date(b)
            if not date_str:
                continue
            
            stake = float(b.get('stake', 0) or 0)
            odds = b.get('odds_decimal', b.get('odds', 1.91))
            if isinstance(odds, (int, float)) and (odds < 0 or odds >= 100):
                odds = american_to_decimal(odds)

            if date_str not in daily:
                daily[date_str] = {'profit': 0.0, 'wins': 0, 'losses': 0, 'bets': 0, 'parlays': 0}

            daily[date_str]['bets'] += 1
            if result == 'Win':
                daily[date_str]['profit'] += stake * (odds - 1)
                daily[date_str]['wins'] += 1
            else:
                daily[date_str]['profit'] -= stake
                daily[date_str]['losses'] += 1
        
        # Process parlays (use the date field directly from parlay)
        if parlays_list:
            for p in parlays_list:
                if not isinstance(p, dict):
                    continue
                result = p.get('result', 'Pending')
                if result not in ('Win', 'Loss'):
                    continue
                
                # Parlays have 'date' field; use it directly (no adjusted date needed for parlay creation date)
                date_str = _normalize_calendar_date(p.get('date', ''))
                if not date_str:
                    continue
                
                stake = float(p.get('stake', 0) or 0)
                actual_payout = float(p.get('actual_payout', 0) or 0)
                
                if date_str not in daily:
                    daily[date_str] = {'profit': 0.0, 'wins': 0, 'losses': 0, 'bets': 0, 'parlays': 0}
                
                daily[date_str]['parlays'] += 1
                if result == 'Win':
                    daily[date_str]['profit'] += actual_payout - stake  # Profit = payout - stake
                    daily[date_str]['wins'] += 1
                else:
                    daily[date_str]['profit'] -= stake  # Loss = lose the stake
                    daily[date_str]['losses'] += 1
        
        return daily

    # =========================================================================
    # SECTION 1: Monthly P/L Calendar
    # =========================================================================
    st.markdown("#### Monthly P/L Calendar")
    st.caption("Dates shifted for PH timezone (bets 12AM-1PM attributed to previous day)")
    
    # Load parlay data
    parlays = []
    try:
        # Reuse ParlayTracker parser so both list and {meta, parlays} formats work.
        parlays = ParlayTracker(PARLAY_FILE).get_parlays()
    except Exception as e:
        logger.warning(f"Failed to load parlays: {e}")
    
    # Build daily P/L including both bets and parlays
    daily_pl = _build_daily_pl(bets, parlays)

    if not daily_pl:
        st.info("No resolved bets yet. Resolve some bets to see your P/L calendar.")
    else:
        import calendar as cal_mod

        # Month selector
        all_dates = sorted(daily_pl.keys())
        earliest = datetime.strptime(_normalize_calendar_date(all_dates[0]), '%Y-%m-%d')
        latest = datetime.strptime(_normalize_calendar_date(all_dates[-1]), '%Y-%m-%d')

        available_months = []
        cursor = earliest.replace(day=1)
        while cursor <= latest:
            available_months.append(cursor.strftime('%Y-%m'))
            if cursor.month == 12:
                cursor = cursor.replace(year=cursor.year + 1, month=1)
            else:
                cursor = cursor.replace(month=cursor.month + 1)

        col_month, col_summary = st.columns([1, 2])
        with col_month:
            selected_month = st.selectbox(
                "Month",
                available_months,
                index=len(available_months) - 1,
                key='analytics_month'
            )

        sel_year, sel_month_num = int(selected_month[:4]), int(selected_month[5:])
        month_name = cal_mod.month_name[sel_month_num]
        cal_matrix = cal_mod.monthcalendar(sel_year, sel_month_num)

        # Monthly totals
        month_profit = 0.0
        month_wins = 0
        month_losses = 0
        month_green_days = 0
        month_red_days = 0
        for ds, info in daily_pl.items():
            if ds.startswith(selected_month):
                month_profit += info['profit']
                month_wins += info['wins']
                month_losses += info['losses']
                if info['profit'] > 0:
                    month_green_days += 1
                elif info['profit'] < 0:
                    month_red_days += 1

        with col_summary:
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Month P/L", f"₱{month_profit:+,.2f}")
            s2.metric("Record", f"{month_wins}W - {month_losses}L")
            s3.metric("Green Days", f"{month_green_days}")
            s4.metric("Red Days", f"{month_red_days}")

        # Build calendar HTML
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        header_cells = ''.join(
            f'<th style="padding:6px 8px;text-align:center;color:#888;font-size:11px;'
            f'font-weight:600;border-bottom:1px solid #333;">{d}</th>'
            for d in day_names
        )

        rows_html = ''
        for week in cal_matrix:
            cells = ''
            for day in week:
                if day == 0:
                    cells += '<td style="padding:8px;background:rgba(30,30,30,0.3);border:1px solid #222;"></td>'
                else:
                    ds = f"{sel_year}-{sel_month_num:02d}-{day:02d}"
                    info = daily_pl.get(ds)
                    if info is None:
                        # No bets this day
                        cells += (
                            f'<td style="padding:8px;text-align:center;background:rgba(50,50,50,0.3);'
                            f'border:1px solid #333;border-radius:4px;vertical-align:top;">'
                            f'<div style="font-size:10px;color:#555;">{day}</div></td>'
                        )
                    else:
                        p = info['profit']
                        w, l = info['wins'], info['losses']
                        bets_count = info.get('bets', 0)
                        parlays_count = info.get('parlays', 0)
                        if p > 0:
                            bg = 'rgba(0,200,83,0.15)'
                            border_clr = '#00c853'
                            clr = '#00c853'
                        elif p < 0:
                            bg = 'rgba(244,67,54,0.15)'
                            border_clr = '#f44336'
                            clr = '#f44336'
                        else:
                            bg = 'rgba(100,100,100,0.15)'
                            border_clr = '#666'
                            clr = '#888'
                        tooltip = f"Wins: {w} | Losses: {l} | Bets: {bets_count} | Parlays: {parlays_count} | Net: ₱{p:+,.2f}"
                        cells += (
                            f'<td title="{tooltip}" style="padding:6px 4px;text-align:center;'
                            f'background:{bg};border:1px solid {border_clr};border-radius:6px;'
                            f'vertical-align:top;cursor:default;">'
                            f'<div style="font-size:10px;color:#aaa;">{day}</div>'
                            f'<div style="font-size:13px;font-weight:700;color:{clr};margin-top:2px;">'
                            f'₱{p:+,.0f}</div>'
                            f'<div style="font-size:9px;color:#666;margin-top:1px;">{w}W {l}L</div>'
                            f'<div style="font-size:8px;color:#555;margin-top:0px;">{bets_count}B {parlays_count}P</div>'
                            f'</td>'
                        )
            rows_html += f'<tr>{cells}</tr>'

        cal_html = (
            f'<table style="width:100%;border-collapse:separate;border-spacing:3px;'
            f'font-family:sans-serif;margin:8px 0;">'
            f'<thead><tr>{header_cells}</tr></thead>'
            f'<tbody>{rows_html}</tbody></table>'
        )
        st.markdown(cal_html, unsafe_allow_html=True)

    st.markdown("---")

    # =========================================================================
    # SECTION 2: Alpha Shield - Risk Aversion Savings
    # =========================================================================
    st.markdown("#### Alpha Shield")

    # Accepted bets: stake > 0 and decided
    accepted_decided = [
        b for b in bets
        if b.get('result') in ('Win', 'Loss') and float(b.get('stake', 0) or 0) > 0
    ]
    avg_stake = (
        sum(float(b.get('stake', 0) or 0) for b in accepted_decided) / len(accepted_decided)
        if accepted_decided else 0.0
    )

    # Rejected bets: stake == 0 and decided
    rejected_decided = [
        b for b in bets
        if b.get('result') in ('Win', 'Loss') and float(b.get('stake', 0) or 0) == 0
    ]

    if not rejected_decided or avg_stake == 0:
        st.info("Not enough resolved rejected bets or accepted bets to compute Risk Aversion Savings.")
    else:
        hypo_pl = 0.0
        hypo_wins = 0
        hypo_losses = 0
        for b in rejected_decided:
            odds = b.get('odds_decimal', b.get('odds', 1.91))
            if isinstance(odds, (int, float)) and (odds < 0 or odds >= 100):
                odds = american_to_decimal(odds)
            if b.get('result') == 'Win':
                hypo_pl += avg_stake * (odds - 1)
                hypo_wins += 1
            else:
                hypo_pl -= avg_stake
                hypo_losses += 1

        # Capital preserved = negative of hypothetical P/L
        # If hypo_pl < 0 (rejected bets would have lost money), capital_preserved > 0 (good)
        capital_preserved = -hypo_pl
        hypo_win_rate = hypo_wins / len(rejected_decided) if rejected_decided else 0
        accepted_win_rate = (
            len([b for b in accepted_decided if b.get('result') == 'Win']) / len(accepted_decided)
            if accepted_decided else 0
        )

        if capital_preserved >= 0:
            shield_color = '#00c853'
            shield_bg = 'rgba(0,200,83,0.12)'
            shield_border = '#00c853'
            shield_icon = '&#x1F6E1;'
            shield_label = 'Risk Engine Savings'
        else:
            shield_color = '#f44336'
            shield_bg = 'rgba(244,67,54,0.12)'
            shield_border = '#f44336'
            shield_icon = '&#x26A0;'
            shield_label = 'Opportunity Cost of AI Filtering'

        shield_html = f"""
<div style="background:{shield_bg};border:2px solid {shield_border};border-radius:12px;
padding:20px;margin:10px 0;text-align:center;">
<div style="font-size:14px;color:#aaa;margin-bottom:8px;">{shield_icon} {shield_label}</div>
<div style="font-size:36px;font-weight:800;color:{shield_color};">₱{abs(capital_preserved):,.2f}</div>
<div style="font-size:12px;color:#666;margin-top:10px;">
Hypothetical P/L of {len(rejected_decided)} bets rejected by the Risk Engine
(using ₱{avg_stake:.2f} avg accepted stake)
</div>
</div>"""
        st.markdown(shield_html, unsafe_allow_html=True)

        # Breakdown metrics
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Rejected Bets", f"{len(rejected_decided)}")
        a2.metric("Rejected Win Rate", f"{hypo_win_rate:.0%}")
        a3.metric("Accepted Win Rate", f"{accepted_win_rate:.0%}")
        a4.metric("Filter Edge", f"{(accepted_win_rate - hypo_win_rate):+.1%}")

    st.markdown("---")

    # =========================================================================
    # SECTION 3: Bankroll Trend Chart (Cumulative P/L from Bets + Parlays)
    # =========================================================================
    st.markdown("#### Bankroll Trend")

    equity_curve = tracker.get_equity_curve()
    if not equity_curve and not parlays:
        st.info("No resolved staked bets or parlays yet. Place and resolve bets to see your bankroll trend.")
    else:
        # Build a date-based cumulative P/L series using ADJUSTED dates for bets, regular dates for parlays
        decided_staked = [
            b for b in bets 
            if b.get('result') in ('Win', 'Loss') and float(b.get('stake', 0) or 0) > 0
        ]
        
        # Create combined list of (date, profit) tuples from both bets and parlays
        transactions = []
        
        # Add individual bets with adjusted dates
        for b in decided_staked:
            stake = float(b.get('stake', 0) or 0)
            odds = b.get('odds_decimal', b.get('odds', 1.91))
            if isinstance(odds, (int, float)) and (odds < 0 or odds >= 100):
                odds = american_to_decimal(odds)
            if b.get('result') == 'Win':
                profit = stake * (odds - 1)
            else:
                profit = -stake
            transactions.append((_get_adjusted_game_date(b), b.get('id', 0), profit))
        
        # Add parlays with their dates
        for p in parlays:
            if p.get('result') not in ('Win', 'Loss'):
                continue
            stake = float(p.get('stake', 0) or 0)
            actual_payout = float(p.get('actual_payout', 0) or 0)
            if p.get('result') == 'Win':
                profit = actual_payout - stake
            else:
                profit = -stake
            # Use parlay date, add a pseudo-id so parlays come after bets on same day
            parlay_id = float(p.get('id', 0)) / 1000.0  # Smaller sort order so bets process first
            transactions.append((p.get('date', ''), parlay_id, profit))
        
        # Sort by date, then by id for same-day ordering
        transactions = sorted(transactions, key=lambda x: (x[0], x[1]))
        
        dates_for_curve = []
        running = 0.0
        daily_cumulative = []
        for date_str, _, profit in transactions:
            running += profit
            dates_for_curve.append(date_str)
            daily_cumulative.append(running)

        if dates_for_curve:
            chart_df = pd.DataFrame({'date': dates_for_curve, 'Cumulative P/L': daily_cumulative})
            # Aggregate to last value per date for cleaner chart
            chart_df_daily = chart_df.groupby('date', sort=True).last().reset_index()
            
            line_color = '#00c853' if daily_cumulative[-1] >= 0 else '#f44336'
            st.line_chart(chart_df_daily.set_index('date')['Cumulative P/L'], height=300, color=line_color)

            # Quick stats row under the chart
            total_pl = daily_cumulative[-1] if daily_cumulative else 0
            total_staked_sum = sum(float(b.get('stake', 0) or 0) for b in decided_staked) + sum(float(p.get('stake', 0) or 0) for p in parlays if p.get('result') in ('Win', 'Loss'))
            roi = (total_pl / total_staked_sum * 100) if total_staked_sum > 0 else 0
            max_drawdown = 0.0
            peak = 0.0
            for val in daily_cumulative:
                if val > peak:
                    peak = val
                dd = peak - val
                if dd > max_drawdown:
                    max_drawdown = dd

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total P/L", f"₱{total_pl:+,.2f}")
            c2.metric("ROI", f"{roi:+.1f}%")
            c3.metric("Total Staked", f"₱{total_staked_sum:,.2f}")
            c4.metric("Max Drawdown", f"₱{max_drawdown:,.2f}")
        else:
            st.info("No transaction history available.")

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    st.title(" NBA Prop Analyzer")
    
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
        st.markdown("###  Settings")
        
        # Emergency: Reset caches (clears cached data in Streamlit)
        if st.button(" Reset Cache", use_container_width="stretch"):
            try:
                st.cache_data.clear()
            except Exception:
                pass
            try:
                if hasattr(st, 'cache_resource'):
                    st.cache_resource.clear()
            except Exception:
                pass
            try:
                st.toast("Cache cleared! Reloading...")
            except Exception:
                st.info("Cache cleared! Reloading...")
            time.sleep(1)
            st.rerun()
        
        # Bankroll Mode Toggle
        bankroll_enabled = st.toggle(" Bankroll Mode", value=True, help="Turn off when just gathering data for ML")
        st.session_state.bankroll_enabled = bankroll_enabled
        
        if bankroll_enabled:
            bankroll = st.number_input(" Bankroll ()", 100.0, 1000000.0, 500.0)
            
            # Unit Calculator (1 unit = 1%)
            unit_size = bankroll * 0.01
            st.caption(f"1u = {unit_size:,.0f} | 5u = {unit_size*5:,.0f}")
            st.caption("Stakes: A=5u | B=3u | C=1u")
        else:
            bankroll = 0  # No bankroll tracking
            unit_size = 0
            st.info(" Data Collection Mode")
            st.caption("Stakes hidden  P/L tracking disabled")
        
        # Quick stats with P/L in units (only show when bankroll enabled)
        if bankroll_enabled:
            stats = tracker.get_stats()
            if stats['total_decided'] > 0:
                st.markdown("---")
                profit_color = "green" if stats['total_profit'] >= 0 else "red"
                profit_units = stats['total_profit'] / unit_size if unit_size > 0 else 0
                st.markdown(f"**Record:** {stats['wins']}-{stats['losses']} ({stats['win_rate']:.0%} (staked))")
                st.markdown(f"**P/L:** :{profit_color}[{stats['total_profit']:+,.0f}] ({profit_units:+.1f}u)")
        
        if st.session_state.parlay_builder:
            st.markdown(f" Parlay: {len(st.session_state.parlay_builder)} legs")
        
        # Hardware & AI Engine Status
        try:
            from hardware_utils import check_cuda_availability
            from llm_engine import get_active_llm_provider
            has_cuda, gpu_name = check_cuda_availability()
            active_llm = get_active_llm_provider()
            st.markdown("---")
            st.markdown("##### ⚡ Hardware & AI")
            if has_cuda:
                st.caption(f"🚀 **GPU:** {gpu_name}")
            else:
                st.caption("🖥️ **Compute:** CPU Mode")
            st.caption(f"🧠 **AI:** {active_llm['name']}")
        except Exception:
            pass

        st.caption(f"{CURRENT_VERSION}")
    
    # Main tabs - simplified names for mobile

    tab_games, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "🏀 Games", "Analyze", "Backtest", "Watchlist", "H2H", "Splits", "Bets", "Parlays", "ML", "Analytics"
    ])

    # Tab 0: Game Predictions & Lines
    with tab_games:
        try:
            from game_ui import render_games_tab
            render_games_tab(tracker=tracker, bankroll_current=float(bankroll if bankroll > 0 else 500.0))
        except Exception as _game_tab_err:
            st.error(f"Error loading Game Predictions: {_game_tab_err}")
            logger.exception("Error in game predictions tab")

    # Tab 1: Analyzer
    with tab1:
        # Mobile-friendly stacked layout with expander for advanced options
        all_players = get_all_nba_players()
        player_names = sorted([p['full_name'] for p in all_players if p.get('is_active', True)])
        
        # --- Auto-fetch today's odds once per session ---
        if "odds_fetched" not in st.session_state:
            st.session_state.odds_fetched = False
            st.session_state.odds_today = {}  # (home, away, date) -> game data
        if not st.session_state.odds_fetched:
            try:
                from odds_engine import get_odds_engine
                _oe = get_odds_engine()
                _oe.fetch_today()
                # Build a quick team->game lookup for today
                from datetime import date as _dt_date
                _today_str = _dt_date.today().strftime("%Y-%m-%d")
                _today_games = {}
                for k, v in _oe._index.items():
                    if k[2] == _today_str:
                        _today_games[k] = v
                st.session_state.odds_today = _today_games
                st.session_state.odds_today_date = _today_str
                st.session_state.odds_fetched = True
                if _today_games:
                    st.caption(f"Live odds loaded for {len(_today_games)} games today")
            except Exception as _oe_err:
                logger.warning(f"Could not auto-fetch today's odds: {_oe_err}")
                st.session_state.odds_fetched = True  # Don't retry on every rerun

        #  START FORM: Prevents reloading until "Run" is clicked
        with st.form(key='analysis_form'):
            
            # Primary inputs
            col1, col2 = st.columns(2)
            with col1:
                player_in = st.selectbox(" Player", player_names, index=None, placeholder="Search...")
            with col2:
                nba_teams = teams.get_teams()
                team_opts = sorted([t['abbreviation'] for t in nba_teams])
                try:
                    def_idx = team_opts.index('LAL')
                except ValueError:
                    def_idx = 0
                opp_in = st.selectbox(" vs", team_opts, index=def_idx)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                market = st.selectbox(" Market", ["PTS", "REB", "AST", "PRA", "3PM", "PA", "PR", "RA", "STL", "BLK"])
            with col2:
                line_in = st.number_input(" Line", 0.5, 100.0, 25.5, 0.5)
            with col3:
                odds_in = st.number_input(" Odds", 1.01, 10.00, 1.85, 0.01)
            
            # Show injury reports (Updates on Submit)
            player_team_abbrev = None  # Initialize for auto-odds lookup later
            if opp_in:
                # Get player's team abbrev logic
                if player_in:
                    try:
                        p_list = players.find_players_by_full_name(player_in)
                        if p_list:
                            p_id = p_list[0]['id']
                            # Note: We can't use st.cache here easily inside form, 
                            # but this runs only on submit now, so it's fine.
                            from nba_api.stats.endpoints import commonplayerinfo
                            # time.sleep(0.3) # Removed sleep for speed, add back if hitting rate limits
                            player_info = commonplayerinfo.CommonPlayerInfo(player_id=p_id).get_data_frames()[0]
                            if 'TEAM_ABBREVIATION' in player_info.columns:
                                player_team_abbrev = player_info['TEAM_ABBREVIATION'].iloc[0]
                    except:
                        pass
                
                inj_col1, inj_col2 = st.columns(2)
                
                # Opponent injuries
                with inj_col1:
                    with st.expander(f" {opp_in} Injuries (Opponent)", expanded=False):
                        try:
                            injuries = orchestrator.injury_manager.get_team_injury_report(opp_in)
                            if injuries:
                                for inj in injuries:
                                    status = inj.get('status', 'Unknown')
                                    emoji = {'OUT': '', 'DOUBTFUL': '', 'QUESTIONABLE': '', 'DAY-TO-DAY': '', 'PROBABLE': ''}.get(status, '')
                                    pos = inj.get('position', '')
                                    pos_str = f" [{pos}]" if pos else ""
                                    injury_desc = inj.get('injury', '')
                                    st.markdown(f"{emoji} **{inj.get('name', 'Unknown')}**{pos_str}  {status}" + (f" ({injury_desc})" if injury_desc else ""))
                                st.caption(f"Source: {injuries[0].get('source', 'ESPN')}")
                            else:
                                st.info("No injuries reported")
                        except Exception as e:
                            st.caption(f"Could not fetch injuries: {e}")
                
                # Player Team injuries
                with inj_col2:
                    if player_team_abbrev and player_team_abbrev != opp_in:
                        with st.expander(f" {player_team_abbrev} Injuries (Player's Team)", expanded=False):
                            try:
                                team_injuries = orchestrator.injury_manager.get_team_injury_report(player_team_abbrev)
                                if team_injuries:
                                    for inj in team_injuries:
                                        status = inj.get('status', 'Unknown')
                                        emoji = {'OUT': '', 'DOUBTFUL': '', 'QUESTIONABLE': '', 'DAY-TO-DAY': '', 'PROBABLE': ''}.get(status, '')
                                        pos = inj.get('position', '')
                                        pos_str = f" [{pos}]" if pos else ""
                                        injury_desc = inj.get('injury', '')
                                        st.markdown(f"{emoji} **{inj.get('name', 'Unknown')}**{pos_str}  {status}" + (f" ({injury_desc})" if injury_desc else ""))
                                    st.caption(f"Source: {team_injuries[0].get('source', 'ESPN')}")
                                else:
                                    st.info("No injuries reported")
                            except Exception as e:
                                st.caption(f"Could not fetch injuries: {e}")

            # Advanced options
            with st.expander("Advanced Options", expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    is_home = st.checkbox("Home", True)
                with c2:
                    days_rest = st.selectbox("Rest", [0, 1, 2, 3], index=1,
                                           format_func=lambda x: "B2B" if x == 0 else f"{x}d")
                with c3:
                    spread = st.number_input("Spread (auto)", -25.0, 25.0, 0.0, 0.5,
                                            help="Leave at 0 to auto-fetch from Odds API")
                with c4:
                    game_total = st.number_input("Total (auto)", 180.0, 280.0, 225.0, 0.5,
                                                help="Leave at 225 to auto-fetch from Odds API")
                lookback = st.slider("Lookback Games", 5, 30, 15)
            
            is_b2b = days_rest == 0
            is_valid, error_msg = validate_inputs(line_in, odds_in)
            
            if not is_valid: 
                st.error(error_msg)
            
            #  SUBMIT BUTTON
            # This replaces the old st.button. It triggers the form submission.
            submitted = st.form_submit_button(" Run Analysis", type="primary", disabled=not is_valid)

        #  LOGIC EXECUTION (Only runs when button is clicked)
        if submitted:
            if not player_in:
                st.error("Please select a player.")
            else:
                # --- Auto-fetch spread/total from Odds API if left at defaults ---
                _auto_spread = spread
                _auto_total = game_total
                _odds_source = None
                if spread == 0.0 or game_total == 225.0:
                    try:
                        from odds_engine import get_odds_engine
                        _oe = get_odds_engine()
                        _today_str = st.session_state.get("odds_today_date", "")
                        if not _today_str:
                            from datetime import date as _dt_date
                            _today_str = _dt_date.today().strftime("%Y-%m-%d")
                        # Resolve player team abbreviation
                        _pt_abbrev = player_team_abbrev  # Set in injury section above
                        if not _pt_abbrev:
                            # Fallback: try to resolve from player name
                            try:
                                _p_list = players.find_players_by_full_name(player_in)
                                if _p_list:
                                    from nba_api.stats.endpoints import commonplayerinfo
                                    _pi = commonplayerinfo.CommonPlayerInfo(player_id=_p_list[0]['id']).get_data_frames()[0]
                                    if 'TEAM_ABBREVIATION' in _pi.columns:
                                        _pt_abbrev = _pi['TEAM_ABBREVIATION'].iloc[0]
                            except Exception:
                                pass
                        if _pt_abbrev:
                            _s, _t = _oe.get_spread_and_total(
                                _pt_abbrev, opp_in, _today_str, is_home=is_home
                            )
                            if _s is not None and spread == 0.0:
                                _auto_spread = _s
                            if _t is not None and game_total == 225.0:
                                _auto_total = _t
                            if _s is not None or _t is not None:
                                _odds_source = "Odds API"
                    except Exception as _oe_err:
                        logger.warning(f"Auto-fetch odds failed: {_oe_err}")

                if _odds_source:
                    st.info(f"Auto-fetched: Spread **{_auto_spread:+.1f}** | Total **{_auto_total:.1f}** ({_odds_source})")

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
                        spread=_auto_spread,
                        bankroll=bankroll,
                        days_rest=days_rest,
                        game_total=_auto_total
                    )
                    
                    if not result.success:
                        st.error(result.error)
                    else: 
                        st.session_state.analysis_result = result
        
        # Display results (Persists after reload because it checks session_state)
        result = st.session_state.analysis_result
        if result and result.success:
            st.markdown("---")
            
            # Main recommendation - Ticket Style Card
            render_ticket_card(result, bankroll, bankroll_enabled)

            st.divider()
    
            with st.container(border=True):
                c_ai_1, c_ai_2 = st.columns([3, 1])
                with c_ai_1:
                    st.subheader("AI Analyst")
                    st.caption("Get a second opinion. compares the math (EV) vs. the tape (Game Logs).")
                with c_ai_2:
                    # Unique key ensures no conflict
                    ask_ai = st.button("Consult groq", key="tab1_ask_ai", use_container_width="stretch")

                if ask_ai:
                        with st.spinner(f"Analyzing {result.player_name}..."):
                            # 1. PREPARE CONTEXT
                            team_str = "N/A"
                            if result.game_logs is not None and not result.game_logs.empty:
                                try:
                                    team_str = result.game_logs.iloc[-1]['MATCHUP'].split()[0]
                                except: pass

                            feats = result.features
                            is_home = feats.is_home
                            loc_avg = feats.home_avg if is_home else feats.away_avg
                            loc_type = "Home" if is_home else "Away"
                            
                            # ---------------------------------------------------------
                            #  FIX: FORCE AI TO SEE THE "SMART" NUMBER (23.2)
                            # ---------------------------------------------------------
                            final_proj = None
                            
                            # 1. Check the PROJECTION Object (Correct Location)
                            # This is where the card gets its "23.2"
                            if hasattr(result, 'projection') and hasattr(result.projection, 'final_projection'):
                                final_proj = result.projection.final_projection
                            
                            # 2. Check Decision/Result (Backups)
                            if final_proj is None:
                                if hasattr(result.decision, 'final_projection'):
                                    final_proj = result.decision.final_projection
                                elif hasattr(result, 'projected_value'):
                                    final_proj = result.projected_value
                                
                            # 3. Fallback to EMA (Only if everything else fails)
                            if final_proj is None:
                                final_proj = feats.ema

                            # Calculate the Gap (Cushion)
                            gap = float(result.line) - float(final_proj)
                            if result.decision.recommended_side == 'OVER':
                                gap = -gap 

                            # ---------------------------------------------------------
                            # PAYLOAD WITH FULL FEATURE CONTEXT FOR AI ANALYST
                            # V21.2: Pass ALL features so AI can detect red flags
                            # ---------------------------------------------------------
                            prop_payload = {
                                # --- Core Identifiers ---
                                "player": result.player_name,
                                "team": team_str,
                                "position": getattr(result, 'player_position', '') or '',  # V21.2: Safe fallback for cached results
                                "opponent": result.opponent_name,
                                "market": result.market,
                                "line": result.line,
                                
                                # --- Model Outputs ---
                                "predicted_side": result.decision.recommended_side,
                                "signal": result.decision.recommended_side,  # Alias
                                "ev": result.decision.expected_value,
                                "prob": result.decision.probability,
                                "projected": round(final_proj, 1),
                                "projected_value": round(final_proj, 1),  # Alias
                                "gap": round(gap, 1),
                                "spread": feats.spread,
                                
                                # --- Performance Features ---
                                # NOTE: FeatureVector uses 'feat_*' prefix for these
                                "feat_season_avg": getattr(feats, 'feat_season_avg', 0.0),
                                "feat_l10_avg": getattr(feats, 'feat_l10_avg', 0.0),
                                "feat_ema": round(feats.ema, 1),
                                "ema": round(feats.ema, 1),  # Alias
                                "feat_std": feats.std,
                                "std": feats.std,  # Alias
                                "feat_h2h_avg": getattr(feats, 'feat_h2h_avg', -1.0),
                                
                                # --- Trend & Form ---
                                "feat_trend_5g": feats.trend_5g,
                                "feat_form_gap": getattr(feats, 'feat_form_gap', 0.0),
                                "feat_cv": getattr(feats, 'feat_cv', 0.0),
                                
                                # --- Minutes & Role ---
                                "feat_avg_minutes": feats.avg_minutes,
                                "avg_minutes": feats.avg_minutes,  # Alias
                                "feat_min_volatility": getattr(feats, 'feat_min_volatility', 0.0),
                                "feat_depth_rank": getattr(feats, 'feat_depth_rank', 4),
                                
                                # --- Fatigue & Rest ---
                                "feat_days_rest": feats.days_rest,
                                "days_rest": feats.days_rest,  # Alias
                                "feat_is_b2b": feats.is_b2b,
                                "is_b2b": feats.is_b2b,  # Alias
                                "feat_is_home": feats.is_home,
                                "is_home": feats.is_home,  # Alias
                                
                                # --- Hit Rates ---
                                "feat_hit_rate_l10": getattr(feats, 'feat_hit_rate_l10', 0.0),
                                "feat_hit_rate_l20": getattr(feats, 'feat_hit_rate_l20', 0.0),
                                
                                # --- Matchup Quality ---
                                "feat_opponent_drtg_season": feats.opponent_drtg_season,
                                "opponent_drtg_season": feats.opponent_drtg_season,  # Alias
                                "feat_opp_rank_vs_pos": getattr(feats, 'feat_opp_rank_vs_pos', 15),
                                
                                # --- Location Splits ---
                                "feat_home_avg": feats.home_avg,
                                "feat_away_avg": feats.away_avg,
                                "loc_type": loc_type,
                                "loc_avg": round(loc_avg, 1),
                                
                                # --- Absence Context ---
                                "feat_team_out_count": feats.team_out_count,
                                "team_out_count": feats.team_out_count,  # Alias
                                "feat_team_out_ppg": getattr(feats, 'team_out_ppg', 0.0), 
                                "team_out_ppg": getattr(feats, 'team_out_ppg', 0.0),
                                "feat_opp_out_count": feats.opp_out_count,
                                "opp_out_count": feats.opp_out_count,  # Alias
                                
                                # --- Games Played (CRITICAL FOR AI) ---
                                "feat_games_played": feats.games_played,
                                "games_played": feats.games_played,  # Alias
                                
                                # --- Additional Context ---
                                "feat_usage_rate": getattr(feats, 'feat_usage_rate', 0.0),
                                "total": feats.game_total,
                            }

                            # 2. RUN LIVE BACKTEST (Robust Settings)
                            status_text = st.empty()
                            status_text.text("Running Simulation (Matching CSV + Buckets)...")
                            
                            history_context = None
                            backtest_payload = {}
                            
                            try:
                                # Canonicalize Name
                                p_list = players.find_players_by_full_name(result.player_name)
                                canonical_name = p_list[0]['full_name'] if p_list else result.player_name
                                p_id = p_list[0]['id'] if p_list else None
                                
                                # A. Scientific Backtest
                                bt_summary = orchestrator.run_backtest(
                                    player_name=canonical_name,
                                    market=result.market,
                                    lookback=25,    
                                    test_days=45    
                                )
                                
                                res_df = getattr(bt_summary, 'results', None) if bt_summary else None

                                # B. Fallback
                                if (res_df is None or res_df.empty) and p_id:
                                    status_text.text("Scientific Backtest incomplete. Running Manual Fallback...")
                                    raw_logs = orchestrator.data_loader.fetch_game_logs(p_id)
                                    if not raw_logs.empty:
                                        mkt = result.market
                                        if mkt not in raw_logs.columns:
                                            if mkt == 'PRA': raw_logs['PRA'] = raw_logs.get('PTS',0) + raw_logs.get('REB',0) + raw_logs.get('AST',0)
                                            elif mkt == 'PA': raw_logs['PA'] = raw_logs.get('PTS',0) + raw_logs.get('AST',0)
                                            elif mkt == 'PR': raw_logs['PR'] = raw_logs.get('PTS',0) + raw_logs.get('REB',0)
                                            elif mkt == 'RA': raw_logs['RA'] = raw_logs.get('REB',0) + raw_logs.get('AST',0)
                                        
                                        if mkt in raw_logs.columns:
                                            raw_logs = raw_logs.sort_values('GAME_DATE', ascending=False).head(30)
                                            raw_logs['hit'] = raw_logs.apply(
                                                lambda x: 1 if (result.decision.recommended_side == 'OVER' and x[mkt] > result.line) or 
                                                             (result.decision.recommended_side == 'UNDER' and x[mkt] <= result.line) else 0, axis=1
                                            )
                                            raw_logs['margin'] = raw_logs[mkt] - result.line
                                            if result.decision.recommended_side == 'UNDER': raw_logs['margin'] = -raw_logs['margin']
                                            
                                            res_df = pd.DataFrame({
                                                'date': raw_logs['GAME_DATE'], 'market': mkt, 'line': result.line,
                                                'pred': result.decision.recommended_side, 'actual': raw_logs[mkt],
                                                'hit': raw_logs['hit'], 'margin': raw_logs['margin']
                                            })

                                # C. Calibration (Buckets)
                                bucket_name = "N/A"
                                bucket_wr = "N/A"
                                bucket_count = 0
                                if bt_summary and hasattr(bt_summary, 'calibration_by_bucket'):
                                    prob_pct = result.decision.probability * 100
                                    found_bucket = None
                                    if 50 <= prob_pct < 55: found_bucket = "50-55%"
                                    elif 55 <= prob_pct < 60: found_bucket = "55-60%"
                                    elif 60 <= prob_pct < 65: found_bucket = "60-65%"
                                    elif 65 <= prob_pct < 70: found_bucket = "65-70%"
                                    elif 70 <= prob_pct < 75: found_bucket = "70-75%"
                                    elif prob_pct >= 75: found_bucket = "75-100%"
                                    
                                    if found_bucket and found_bucket in bt_summary.calibration_by_bucket:
                                        b_data = bt_summary.calibration_by_bucket[found_bucket]
                                        bucket_name = found_bucket
                                        try:
                                            bucket_wr = round(b_data['actual'] * 100, 1)
                                            bucket_count = b_data['count']
                                        except: pass

                                # D. Format Output
                                if res_df is not None and not res_df.empty:
                                    res_df = res_df.sort_values('date', ascending=False)
                                    wins = res_df[res_df['hit'] == 1].shape[0]
                                    total = len(res_df)
                                    win_rate = (wins / total) * 100
                                    
                                    last_5_str = []
                                    for _, row in res_df.head(5).iterrows():
                                        icon = "" if row['hit'] == 1 else ""
                                        m_val = row.get('margin', 0)
                                        last_5_str.append(f"{icon}({m_val:+.1f})")

                                    history_context = {
                                        'record': f"{wins}-{total-wins}",
                                        'win_rate': round(win_rate, 1),
                                        'count': total,
                                        'last_5': ", ".join(last_5_str)
                                    }
                                    
                                    csv_cols = [c for c in ['date', 'market', 'line', 'pred', 'actual', 'margin'] if c in res_df.columns]
                                    logs_csv = res_df[csv_cols].head(10).to_csv(index=False)
                                    
                                    backtest_payload = {
                                        "l30_rate": int(win_rate),
                                        "avg_diff": res_df['margin'].mean(),
                                        "logs_csv": logs_csv,
                                        "bucket_name": bucket_name,
                                        "bucket_wr": bucket_wr,
                                        "bucket_count": bucket_count
                                    }
                                    status_text.empty()
                                else:
                                    status_text.warning(f"No game data found for {result.player_name}")

                            except Exception as e:
                                st.error(f"Analysis Error: {e}")
                                
                            # 3. GET OPINION
                            from llm_engine import get_active_llm_provider
                            active_llm = get_active_llm_provider()
                            llm_badge = f"🟢 {active_llm['name']}" if active_llm.get('is_local') else f"☁️ {active_llm['name']}"
                            insight = get_ai_second_opinion(prop_payload, backtest_payload, history_context=history_context)
                            st.markdown(f"### 🤖 AI Analyst Verdict ({llm_badge})")
                            st.markdown(insight)
            
            col_track, col_parlay = st.columns(2)
            # Tag selection
            tag_options = ["Sniper", "Robot_Top_Pick", "Gut_Feel", "Live_Bet"]
            default_tag = "Sniper" if result.decision.expected_value > 0 else "Gut_Feel"
            selected_tag = st.selectbox("Bet Source Tag:", tag_options, 
                index=tag_options.index(default_tag), key="bet_tag_select")
            
            with col_track: 
                if st.button(" Track", use_container_width="stretch"):
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
                            "quality_score": result.decision.quality_score,  # V21.2: Sharpe-like EV/sigma
                            "tag": selected_tag,
                        }
                        # Dynamically merge features based on canonical schema
                        bet_payload.update(extract_features_dynamically(features, market=result.market))
                        tracker.log_bet(bet_payload)
                        st.toast(f"Bet tracked! [{selected_tag}]")
                    except Exception as e:
                        st.error(f"Failed to track bet: {e}")
                        logger.error(f"Bet tracking failed: {e}")
            
            with col_parlay: 
                if result.decision.expected_value > 0:
                    if st.button("Parlay", use_container_width="stretch"):
                        leg = {
                            'player': result.player_name,
                            'opponent': result.opponent_name,
                            'market': result.market,
                            'line': result.line,
                            'side': result.decision.recommended_side,
                            'odds': result.odds,
                            'prob': result.decision.probability,
                            'proj': result.projection.final_projection,
                            'mean': result.projection.final_projection,
                            'std': getattr(result.projection, 'std_dev', 4.0),
                            'team': getattr(result, 'team_name', getattr(result.features, 'team', '')),
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
                            st.toast("Added!")
                            st.rerun()
                        pass 
                else: 
                     st.button("Parlay", disabled=True, use_container_width="stretch")

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
            # Modern Top Bar with Auto-Settle
            stats = tracker.get_stats()
            all_bets = tracker.get_bets()

            top_head_c1, top_head_c2 = st.columns([3, 1.2])
            with top_head_c1:
                st.markdown("### 📋 Bet Tracker")
            with top_head_c2:
                pending_count = stats.get('pending', 0)
                if pending_count > 0:
                    if st.button(f"⚡ Auto-Settle ({pending_count})", type="primary", use_container_width=True, help="Auto-settles pending game and player prop bets against official ESPN box scores"):
                        with st.spinner("Grading against official final scores & box scores..."):
                            settle_res = tracker.auto_settle_pending_bets()
                            if settle_res.get('settled_count', 0) > 0:
                                st.success(f"Settled {settle_res['settled_count']} bets! ({settle_res['wins']}W - {settle_res['losses']}L - {settle_res['pushes']}P)")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.info("No matching final scores found yet (games may still be in progress).")

            # Section 1: The Dashboard
            if stats['total_decided'] > 0:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Win% (staked)", f"{stats['win_rate']:.0%}")
                c2.metric("Record", f"{stats['wins']}-{stats['losses']}")
                c3.metric("P/L", f"{stats['total_profit']:+,.0f}")
                c4.metric("Pending", stats['pending'])

                # Sub-metrics: Game Lines vs Player Props
                decided_bets = [b for b in all_bets if b.get('result') in ['Win', 'Loss', 'Push']]
                game_decided = [b for b in decided_bets if b.get('market') in ['SPREAD', 'TOTAL', 'spread', 'total'] or b.get('bet_type') in ['spread', 'total']]
                prop_decided = [b for b in decided_bets if b not in game_decided]

                if game_decided or prop_decided:
                    sub_c1, sub_c2 = st.columns(2)
                    with sub_c1:
                        if game_decided:
                            gw = sum(1 for b in game_decided if b.get('result') == 'Win')
                            gl = sum(1 for b in game_decided if b.get('result') == 'Loss')
                            gpnl = sum((b.get('stake', 0) * (float(b.get('odds_decimal') or 1.91) - 1.0)) if b.get('result') == 'Win' else -float(b.get('stake', 0)) for b in game_decided if b.get('result') in ['Win', 'Loss'])
                            gcolor = "green" if gpnl >= 0 else "red"
                            st.caption(f"🏀 **Game Lines:** {gw}-{gl} ({gw/max(1, gw+gl):.0%}) | P/L: :{gcolor}[${gpnl:+,.0f}]")
                    with sub_c2:
                        if prop_decided:
                            pw = sum(1 for b in prop_decided if b.get('result') == 'Win')
                            pl = sum(1 for b in prop_decided if b.get('result') == 'Loss')
                            ppnl = sum((b.get('stake', 0) * (float(b.get('odds_decimal') or 1.91) - 1.0)) if b.get('result') == 'Win' else -float(b.get('stake', 0)) for b in prop_decided if b.get('result') in ['Win', 'Loss'])
                            pcolor = "green" if ppnl >= 0 else "red"
                            st.caption(f"👤 **Player Props:** {pw}-{pl} ({pw/max(1, pw+pl):.0%}) | P/L: :{pcolor}[${ppnl:+,.0f}]")
                
                with st.expander(" Equity Curve", expanded=True):
                    # Use stake-filtered, stake-weighted equity (ignore zero-stake bets)
                    cumulative = tracker.get_equity_curve()
                    if len(cumulative) >= 2:
                        eq_df = pd.DataFrame(cumulative)
                        st.line_chart(eq_df.set_index('Bet #')['P/L'], height=250, color="#00c853")
            
            # Section 2: The Risk Engine
            if bankroll_enabled:
                st.markdown("---")
                st.markdown("###  Quant Risk Engine")
                
                # Bankroll Sync Logic (CRITICAL)
                if 'bankroll' in locals():
                    real_bankroll = bankroll
                elif 'bankroll_current' in st.session_state:
                    real_bankroll = st.session_state['bankroll_current']
                else:
                    real_bankroll = 1500.0
                
                real_bankroll = float(real_bankroll)
                
                # Display the locked bankroll amount
                st.info(f" Using Bankroll: {real_bankroll:,.0f} (synced from sidebar)")
                
                # Optional override (hidden by default)
                with st.expander("Override Bankroll (Advanced)", expanded=False):
                    override_bankroll = st.number_input("Custom Bankroll", value=real_bankroll, min_value=0.0, step=100.0, key="override_bankroll")
                    if st.button("Apply Override", key="apply_override"):
                        real_bankroll = override_bankroll
                        st.success(f"Bankroll overridden to {real_bankroll:,.0f}")
                
                if st.button(" Run Risk Engine", key="run_risk_engine"):
                    pending_bets = [b for b in all_bets if b.get('result') == 'Pending']
                    
                    if not pending_bets:
                        st.warning("No pending bets to analyze.")
                    else:
                        df = pd.DataFrame(pending_bets)
                        required_cols = ['predicted_mean', 'predicted_std', 'line', 'side', 'odds']
                        if not all(col in df.columns for col in required_cols):
                            st.error("Pending bets missing required columns for risk analysis.")
                        else:
                            engine = TradingEngine()
                            processed_df = engine.process_bets(
                                df,
                                real_bankroll,
                                return_all=True,
                                auto_tune_from_tracker=True,
                                tracker_file='data/bet_tracker.json',
                            )
                            
                            st.session_state.processed_bets = processed_df
                            st.session_state.risk_run = True
                            st.rerun()
                
                if 'processed_bets' in st.session_state and st.session_state.get('risk_run'):
                    processed_df = st.session_state.processed_bets
                    
                    approved = processed_df[processed_df['is_bet'] == True]
                    rejected = processed_df[processed_df['is_bet'] == False]
                    
                    st.markdown("####  Approved Bets")
                    if not approved.empty:
                        for _, bet in approved.iterrows():
                            with st.container(border=True):
                                proj_val = bet.get('predicted_mean', bet.get('proj', 0))
                                st.markdown(f"**{bet['player']}** {bet['market']} - {bet['side']} {bet['line']} @ {bet['odds']}")
                                st.caption(f"Proj: {proj_val:.1f} | EV: {bet['predicted_ev']:.3%} | Stake: {bet['rec_stake']:.0f}")
                    else:
                        st.info("No bets approved.")
                    
                    st.markdown("####  Rejected Bets")
                    if not rejected.empty:
                        for _, bet in rejected.iterrows():
                            with st.container(border=True):
                                proj_val = bet.get('predicted_mean', bet.get('proj', 0))
                                st.markdown(f"**{bet['player']}** {bet['market']} - {bet['side']} {bet['line']} @ {bet['odds']}")
                                st.caption(f"Proj: {proj_val:.1f} | EV: {bet['predicted_ev']:.3%} | Stake: {bet['rec_stake']:.0f}", help="Rejected due to low EV or stake")
                    else:
                        st.info("No bets rejected.")
                    
                    if st.button(" Optimize & Commit", key="commit_risk"):
                        # Initialize prediction logger
                        pred_logger = PredictionLogger()
                        predictions_to_log = []

                        # Collect IDs that survived deduplication / processing
                        processed_ids = {int(i) for i in processed_df['id'].dropna()}

                        # Zero-out any pending bet that was DROPPED during process_bets
                        # (e.g. duplicate player entries removed by drop_duplicates).
                        # Without this, those bets keep stale stakes from previous runs.
                        all_pending_ids = {int(b['id']) for b in all_bets if b.get('result') == 'Pending' and b.get('id') is not None}
                        dropped_ids = all_pending_ids - processed_ids
                        for dropped_id in dropped_ids:
                            tracker.update_stake(dropped_id, 0.0)
                        
                        for _, bet in processed_df.iterrows():
                            b_id = bet['id']
                            # Prefer the Risk Engine's recommended stake (rec_stake) over a forced flat bet
                            stake_val = None
                            if 'rec_stake' in bet and pd.notna(bet['rec_stake']):
                                stake_val = float(bet['rec_stake'])
                            elif 'recommended_stake' in bet and pd.notna(bet['recommended_stake']):
                                stake_val = float(bet['recommended_stake'])
                            else:
                                stake_val = 0.0

                            new_stake = float(stake_val) if bet.get('is_bet') else 0.0
                            tracker.update_stake(b_id, new_stake)
                            
                            # Log ALL predictions (staked and non-staked) for analysis
                            pred_entry = bet.to_dict()
                            pred_entry['was_staked'] = new_stake > 0
                            pred_entry['final_stake'] = new_stake
                            predictions_to_log.append(pred_entry)
                        
                        # Batch log all predictions
                        if predictions_to_log:
                            pred_logger.log_batch(predictions_to_log)

                        # Show what we committed for user verification
                        preview = processed_df.copy()
                        if 'rec_stake' in preview.columns:
                            preview['committed_stake'] = preview['rec_stake'].where(preview['is_bet'], 0.0)
                        else:
                            preview['committed_stake'] = 0.0

                        st.dataframe(preview[['player', 'market', 'line', 'predicted_std', 'rec_stake']].fillna('N/A'))
                        st.success("Bets optimized and committed!")
                        del st.session_state.processed_bets
                        del st.session_state.risk_run
                        st.rerun()
            
            # Section 3: The Manager
            st.markdown("---")
            st.markdown("###  Bet Manager")
            
            if all_bets:
                f_seg, f_col1, f_col2, f_col3, f_col4 = st.columns([1.6, 1.4, 1.4, 0.8, 1])
                with f_seg:
                    bet_type_seg = st.selectbox("Category:", ["All Bets", "🏀 Game Lines", "👤 Player Props"], index=0)
                with f_col1:
                    filter_opts = ["Pending", "Win", "Loss", "Push"]
                    selected_filters = st.multiselect("Filter Result:", filter_opts, default=["Pending", "Win", "Loss"])
                with f_col2:
                    all_tags = list(set(b.get('tag', 'legacy') for b in all_bets))
                    selected_tags = st.multiselect("Filter Tag:", all_tags, default=all_tags)
                with f_col3:
                    view_limit = st.selectbox("Show Last:", [10, 20, 50, "All"], index=1)
                with f_col4:
                    sort_option = st.selectbox("Sort by:", ["Result/ID", "Stake Asc", "Stake Desc", "Team"], index=0)
                
                filtered_bets = [b for b in all_bets 
                                if b.get('result', 'Pending') in selected_filters 
                                and b.get('tag', 'legacy') in selected_tags]

                if bet_type_seg == "🏀 Game Lines":
                    filtered_bets = [b for b in filtered_bets if b.get('market') in ['SPREAD', 'TOTAL', 'spread', 'total'] or b.get('bet_type') in ['spread', 'total']]
                elif bet_type_seg == "👤 Player Props":
                    filtered_bets = [b for b in filtered_bets if not (b.get('market') in ['SPREAD', 'TOTAL', 'spread', 'total'] or b.get('bet_type') in ['spread', 'total'])]
                
                if sort_option == "Result/ID":
                    sort_map = {'Pending': 0, 'Win': 1, 'Loss': 2, 'Push': 3}
                    filtered_bets.sort(key=lambda x: (sort_map.get(x.get('result', 'Pending'), 99), -x.get('id', 0)))
                elif sort_option == "Stake Asc":
                    filtered_bets.sort(key=lambda x: (x.get('stake', 0), -x.get('id', 0)))
                elif sort_option == "Stake Desc":
                    filtered_bets.sort(key=lambda x: (-x.get('stake', 0), -x.get('id', 0)))
                elif sort_option == "Team":
                    filtered_bets.sort(key=lambda x: (x.get('opponent', ''), -x.get('id', 0)))
                
                total_count = len(filtered_bets)
                if view_limit != "All":
                    filtered_bets = filtered_bets[:int(view_limit)]
                
                if not filtered_bets:
                    st.info("No bets found.")
                else:
                    st.caption(f"Showing {len(filtered_bets)} of {total_count} bets")
                    
                    with st.form(key="batch_bet_editor"):
                        c1, c2, c3, c4, c5 = st.columns([3, 1.2, 1, 1, 0.5])
                        c1.caption("Bet Details")
                        c2.caption("Result")
                        c3.caption("Actual")
                        c4.caption("Closing")
                        c5.caption("Del")
                        
                        for bet in filtered_bets:
                            b_id = bet['id']
                            bet_res = bet.get('result', 'Pending')
                            
                            with st.container(border=True):
                                c1, c2, c3, c4, c5 = st.columns([3, 1.2, 1, 1, 0.5])
                                
                                with c1:
                                    color = {"Win": "green", "Loss": "red", "Push": "blue"}.get(bet_res, "gray")
                                    icon = {"Win": "✅", "Loss": "❌", "Push": "⚪"}.get(bet_res, "⏳")
                                    proj_val = bet.get('proj', bet.get('predicted_mean', 0))
                                    odds_val = bet.get('odds_decimal', bet.get('odds'))
                                    odds_text = f"{float(odds_val):.2f}" if odds_val is not None else "N/A"
                                    is_game_bet = bet.get('market') in ['SPREAD', 'TOTAL', 'spread', 'total'] or bet.get('bet_type') in ['spread', 'total']
                                    type_icon = "🏀" if is_game_bet else "👤"

                                    clv_badge = ""
                                    if bet.get('clv') is not None and abs(float(bet.get('clv', 0))) > 0.05:
                                        clv_val = float(bet['clv'])
                                        clv_color = "green" if clv_val > 0 else "red"
                                        clv_badge = f" | CLV: :{clv_color}[{clv_val:+.1f} pts]"

                                    tag_badge = f" `[{bet.get('tag')}]`" if bet.get('tag') else ""
                                    opponent_text = f" vs {bet.get('opponent')}" if bet.get('opponent') else ""
                                    st.markdown(f"**{type_icon} {bet['player']}** `{bet['market']}`{tag_badge}")
                                    st.caption(f"{icon} :{color}[{bet['side']} {bet['line']}]{opponent_text} | Proj: {proj_val:.1f} | EV: {bet.get('ev',0):.1%} | Stake: ${bet.get('stake', 0):.0f} | Odds: {odds_text}{clv_badge}")
                                
                                with c2:
                                    opts = ["Pending", "Win", "Loss", "Push"]
                                    idx = opts.index(bet_res) if bet_res in opts else 0
                                    st.selectbox("Res", opts, index=idx, key=f"s_{b_id}", label_visibility="collapsed")
                                
                                with c3:
                                    st.number_input("Act", value=float(bet.get('actual_value', 0)), step=1.0, key=f"av_{b_id}", label_visibility="collapsed")
                                
                                with c4:
                                    st.number_input("Clv", value=float(bet.get('closing_line', bet['line'])), step=0.5, key=f"cl_{b_id}", label_visibility="collapsed")
                                
                                with c5:
                                    # Accessibility: provide a non-empty label but hide it visually
                                    st.checkbox("Delete bet", key=f"del_{b_id}", label_visibility="collapsed")
                                
                                if bet.get('margin') is not None:
                                    m = bet['margin']
                                    mc = 'green' if m > 0 else 'red'
                                    st.caption(f":{mc}[Margin: {m:+.1f}] | {bet.get('result_quality','').replace('_',' ')}")
                        
                        submitted = st.form_submit_button(" Save Changes", type="primary", use_container_width="stretch")
                    
                    if submitted:
                        changes = 0
                        for bet in filtered_bets:
                            b_id = bet['id']
                            
                            if st.session_state.get(f"del_{b_id}"):
                                tracker.delete_bet(b_id)
                                changes += 1
                                continue
                            
                            new_res = st.session_state.get(f"s_{b_id}")
                            new_act = st.session_state.get(f"av_{b_id}")
                            new_cl = st.session_state.get(f"cl_{b_id}")
                            
                            old_res = bet.get('result', 'Pending')
                            old_act = float(bet.get('actual_value', 0) or 0)
                            old_cl = float(bet.get('closing_line', 0) or 0)
                            
                            if (new_res != old_res) or (abs(new_act - old_act) > 0.01) or (abs(new_cl - old_cl) > 0.01):
                                tracker.update_result(b_id, new_res, new_cl, new_act)
                                changes += 1
                        
                        if changes > 0:
                            st.success(f"Updated {changes} bets!")
                            st.rerun()
                        else:
                            st.info("No changes detected.")
            else:
                # Diagnostic helper when tracker returns no bets
                tracker_path = tracker.file
                if tracker_path.exists():
                    size = tracker_path.stat().st_size
                    mtime = datetime.fromtimestamp(tracker_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                    st.warning(f"Tracker file present but no bets loaded — {tracker_path} ({size} bytes, modified {mtime})")

                    c1, c2 = st.columns([1, 1])
                    with c1:
                        if st.button("Reload tracker from disk"):
                            # Force a hard reload of the Streamlit app so Tracker re-reads the file
                            st.experimental_rerun()
                    with c2:
                        if st.button("Backup & reset tracker (safe)"):
                            import shutil, time
                            bak = tracker_path.with_name(f"{tracker_path.name}.bak.{int(time.time())}")
                            shutil.copy2(tracker_path, bak)
                            tracker._save([])
                            st.success(f"Backed up to {bak.name} and reset tracker (empty).")
                            st.experimental_rerun()

                    st.caption("If you recently ran a re-analysis, click 'Reload tracker from disk'. If the file is corrupted, 'Backup & reset' will preserve the original and create an empty tracker.")
                else:
                    st.info("Tracker is empty. No tracker file found on disk.")
    
    
            # Section 4: Real Harvested Closing Odds Feed
            st.markdown("---")
            with st.expander("📊 Real Harvested Closing Odds Feed (League-Wide Database)", expanded=False):
                st.caption("Farmed daily closing lines and settled game outcomes from official sportsbooks (DraftKings, FanDuel, BetMGM).")
                h_tab1, h_tab2 = st.tabs(["🏀 Settled Games (Spreads & Totals)", "👤 Settled Player Props (Closing Lines)"])
                with h_tab1:
                    games_csv = Path("data/harvested_odds/settled_games.csv")
                    if games_csv.exists():
                        g_df = pd.read_csv(games_csv)
                        if not g_df.empty:
                            st.dataframe(g_df.tail(20), use_container_width=True)
                        else:
                            st.info("No settled games harvested yet.")
                    else:
                        st.info("Harvested games database will appear here after first game settlement.")
                with h_tab2:
                    props_pqt = Path("data/harvested_odds/settled_player_props.parquet")
                    if props_pqt.exists():
                        p_df = pd.read_parquet(props_pqt)
                        if not p_df.empty:
                            m_filter = st.selectbox("Filter Market:", ["ALL"] + sorted(p_df['market'].unique()), key="harvest_market_filter")
                            if m_filter != "ALL":
                                p_df = p_df[p_df['market'] == m_filter]
                            disp_cols = [c for c in ['date', 'player', 'market', 'closing_line', 'line_movement', 'actual_value', 'margin', 'result', 'side', 'minutes_played'] if c in p_df.columns]
                            st.dataframe(p_df[disp_cols].tail(50), use_container_width=True)
                        else:
                            st.info("No settled player props harvested yet.")
                    else:
                        st.info("Harvested player props database will appear here after first prop settlement.")

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
        render_analytics_tab(tracker)


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
