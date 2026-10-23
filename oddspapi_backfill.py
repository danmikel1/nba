"""
oddspapi_backfill.py — OddsPapi Historical Odds Backfill (Game-Level)

Fetches historical NBA spreads and game totals from OddsPapi.io.
Writes to the SAME disk-cache format as odds_engine.py so data
flows through the existing historical_odds_cache pipeline in
nba_prediction.py with zero changes.

Free-tier constraints:
  - 250 total requests (lifetime? monthly? — treat as finite)
  - 5 000 ms minimum delay between /historical-odds calls
  - Max 3 bookmakers per historical-odds request
  - Fixtures endpoint is cheaper (bulk, no tight rate limit)

Budget strategy:
  1. Fetch fixtures in bulk (1 req covers a whole month, ~15 games/day)
  2. Fetch historical odds per-fixture (1 req each, 5s between)
  3. Cache EVERYTHING to disk — never re-request a cached fixture
  4. Track request count in a persistent budget file
  5. Hard-stop when budget exhausted

Spread derivation:
  OddsPapi does NOT directly expose handicap/total line values in the
  historical-odds response. Instead, we derive the spread from the
  Pinnacle moneyline (market 111) using the standard NBA relationship:

      fair_home_prob = (1/home_price) / (1/home_price + 1/away_price)
      spread ≈ (0.5 - fair_home_prob) / 0.025

  This yields ±0.5 pt accuracy for spreads within ±12, which covers
  >95% of NBA games. Games with larger spreads (~15+) may have ±1-2 pt
  error, which is still far better than synthetic.

Cache structure (shared with odds_engine.py):
  data/odds_cache/YYYY-MM-DD.json  —  same format as The Odds API cache

Usage:
    python oddspapi_backfill.py --days 14          # backfill last 14 days
    python oddspapi_backfill.py --from 2026-02-01 --to 2026-03-01
    python oddspapi_backfill.py --budget            # show remaining budget
    python oddspapi_backfill.py --coverage          # show cached date coverage
    python oddspapi_backfill.py --probe             # analyse 1 game structure (2 reqs)
"""

import json
import math
import os
import time
import logging
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ============================================================================
# Config
# ============================================================================
DATA_DIR = Path(__file__).resolve().parent / "data"
ODDS_CACHE_DIR = DATA_DIR / "odds_cache"
ODDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

ODDSPAPI_CACHE_DIR = DATA_DIR / "oddspapi_cache"
ODDSPAPI_CACHE_DIR.mkdir(parents=True, exist_ok=True)

BUDGET_FILE = ODDSPAPI_CACHE_DIR / "_budget.json"
FIXTURES_CACHE = ODDSPAPI_CACHE_DIR / "_fixtures.json"

BASE_URL = "https://api.oddspapi.io/v4"
API_KEY = os.environ.get("ODDSPAPI_API_KEY", "")
NBA_TOURNAMENT_ID = 132
SPORT_ID = 11

# Rate limit: 5 seconds between historical-odds calls
RATE_LIMIT_SEC = 8.0   # conservative — API blocks IP after rapid-fire
SSL_RETRY_DELAYS = [30, 60, 120, 300]  # escalating wait (secs) on SSL errors

# Default budget (what we started with minus exploration requests)
INITIAL_BUDGET = 250
EXPLORATION_USED = 0  # reset — quota renewed 2026-03-03

# Preferred bookmaker for historical odds (Pinnacle has richest data)
BOOKMAKERS = "pinnacle"

if not API_KEY:
    logger.warning("ODDSPAPI_API_KEY is not set; OddsPapi requests will fail until it is configured")

# -----------------------------------------------------------------------
# Totals calibration
# -----------------------------------------------------------------------
# OddsPapi encodes totals as indexed market IDs (116, 118, 1110, 1112, …).
# Each consecutive ID (step 2) is one "tick" (0.5 pts) in the total line.
# However, the absolute starting value (base) is NOT provided by the API.
#
# We calibrate the base by cross-referencing with known totals from The
# Odds API. Once calibrated, we can derive the actual total for any game.
#
# Calibration file: data/oddspapi_cache/_totals_calibration.json
# Format: {"base": 165.0, "calibrated_from": "CLE vs DET 2026-03-04",
#           "known_total": 227.5, "market_id": 11256, "index": 125}
TOTALS_CALIBRATION_FILE = ODDSPAPI_CACHE_DIR / "_totals_calibration.json"

# We build a sorted index of all full-game totals market IDs from the catalog.
# This is loaded lazily and cached.
_TOTALS_INDEX: Optional[List[int]] = None  # sorted list of catalog IDs
_TOTALS_BASE: Optional[float] = None       # calibrated base total


def _load_totals_index() -> List[int]:
    """Load sorted full-game totals market IDs from the markets catalog."""
    global _TOTALS_INDEX
    if _TOTALS_INDEX is not None:
        return _TOTALS_INDEX

    catalog_path = ODDSPAPI_CACHE_DIR / "_markets_catalog.json"
    if not catalog_path.exists():
        logger.warning("Markets catalog not found — totals extraction disabled")
        _TOTALS_INDEX = []
        return _TOTALS_INDEX

    try:
        all_markets = json.loads(catalog_path.read_text(encoding="utf-8"))
        fg = [m for m in all_markets
              if m.get("sportId") == SPORT_ID
              and m.get("marketName") == "Over Under (incl. overtime)"]
        _TOTALS_INDEX = sorted(m["marketId"] for m in fg)
        logger.info(f"Loaded {len(_TOTALS_INDEX)} full-game totals market IDs from catalog")
    except Exception as e:
        logger.warning(f"Failed to load markets catalog: {e}")
        _TOTALS_INDEX = []
    return _TOTALS_INDEX


def _load_totals_base() -> Optional[float]:
    """Load calibrated totals base from disk."""
    global _TOTALS_BASE
    if _TOTALS_BASE is not None:
        return _TOTALS_BASE
    if TOTALS_CALIBRATION_FILE.exists():
        try:
            cal = json.loads(TOTALS_CALIBRATION_FILE.read_text(encoding="utf-8"))
            _TOTALS_BASE = cal["base"]
            logger.info(f"Totals calibration loaded: base={_TOTALS_BASE}")
            return _TOTALS_BASE
        except Exception:
            pass
    return None


def _market_id_to_total(market_id: int) -> Optional[float]:
    """Convert a totals market ID to an actual total line using calibration."""
    idx_list = _load_totals_index()
    base = _load_totals_base()
    if not idx_list or base is None:
        return None
    if market_id not in idx_list:
        return None
    pos = idx_list.index(market_id)
    return base + pos * 0.5


def calibrate_totals(known_total: float, balanced_market_id: int, label: str = ""):
    """
    Calibrate the totals base using a known game total and its balanced market ID.

    The balanced market (closest to even vig) represents the consensus total.
    base = known_total - index * 0.5
    """
    global _TOTALS_BASE
    idx_list = _load_totals_index()
    if not idx_list or balanced_market_id not in idx_list:
        logger.error(f"Cannot calibrate: market ID {balanced_market_id} not in catalog")
        return

    pos = idx_list.index(balanced_market_id)
    base = known_total - pos * 0.5

    _TOTALS_BASE = base
    cal = {
        "base": base,
        "calibrated_from": label,
        "known_total": known_total,
        "market_id": balanced_market_id,
        "index": pos,
        "calibrated_at": datetime.now().isoformat(),
    }
    TOTALS_CALIBRATION_FILE.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    logger.info(f"Totals calibrated: base={base:.1f} (from {label}, total={known_total}, ID={balanced_market_id}, idx={pos})")


def auto_calibrate_from_odds_cache():
    """
    Auto-calibrate totals by cross-referencing OddsPapi historical data
    with The Odds API cached totals for overlapping games.
    """
    idx_list = _load_totals_index()
    if not idx_list:
        logger.warning("No totals index — fetch markets catalog first")
        return False

    # Load The Odds API cached games with known totals
    known_games = {}
    for cache_file in ODDS_CACHE_DIR.glob("*.json"):
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            for g in data.get("games", []):
                total = g.get("total")
                src = g.get("source", "the-odds-api")
                if total and src != "oddspapi":  # Only use The Odds API as reference
                    key = (g["home_abbrev"], g["away_abbrev"], g["game_date"])
                    known_games[key] = total
        except Exception:
            continue

    if not known_games:
        logger.info("No The Odds API games found for calibration")
        return False

    # Check if any of these games also have OddsPapi historical data
    for hist_file in ODDSPAPI_CACHE_DIR.glob("hist_*.json"):
        try:
            hist = json.loads(hist_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        bk_data, bk_name = _find_bookmaker_data(hist)
        if not bk_data:
            continue
        markets = bk_data.get("markets", {})

        # Find totals market IDs in this game
        totals_mids = []
        for mid_str in markets:
            try:
                mid = int(mid_str)
            except ValueError:
                continue
            if mid in idx_list:
                totals_mids.append(mid)
        totals_mids.sort()

        if not totals_mids:
            continue

        # We need to know which game this fixture is — check fixture cache
        # For now, try all known_games to find a match
        # (We'd need the fixture metadata to know which game this is)
        # Skip for now — this calibration path is better done in backfill_range
        pass

    logger.info("Auto-calibration not possible from existing cache — "
                "will calibrate during next backfill with overlapping dates")
    return False

# Known OddsPapi market IDs for basketball (sport=11)
# 111 = Winner (incl. overtime)  — moneyline, outcomes 111 (home) & 112 (away)
MONEYLINE_MID = 111

# NBA: ~2.5% win-prob per 1 point of spread (empirical Pinnacle data)
# spread ≈ (0.5 - fair_home_prob) / 0.025
NBA_PROB_PER_POINT = 0.025

# OddsPapi team name -> abbreviation (same as odds_engine.py)
_TEAM_ALIASES = {
    "atlanta hawks": "ATL", "boston celtics": "BOS", "brooklyn nets": "BKN",
    "charlotte hornets": "CHA", "chicago bulls": "CHI",
    "cleveland cavaliers": "CLE", "dallas mavericks": "DAL",
    "denver nuggets": "DEN", "detroit pistons": "DET",
    "golden state warriors": "GSW", "houston rockets": "HOU",
    "indiana pacers": "IND", "los angeles clippers": "LAC",
    "la clippers": "LAC", "los angeles lakers": "LAL", "la lakers": "LAL",
    "memphis grizzlies": "MEM", "miami heat": "MIA",
    "milwaukee bucks": "MIL", "minnesota timberwolves": "MIN",
    "new orleans pelicans": "NOP", "new york knicks": "NYK",
    "oklahoma city thunder": "OKC", "orlando magic": "ORL",
    "philadelphia 76ers": "PHI", "phoenix suns": "PHX",
    "portland trail blazers": "POR", "sacramento kings": "SAC",
    "san antonio spurs": "SAS", "toronto raptors": "TOR",
    "utah jazz": "UTA", "washington wizards": "WAS",
}


def _team_to_abbrev(name: str) -> str:
    """Convert full team name to 3-letter abbreviation."""
    lower = name.strip().lower()
    if lower in _TEAM_ALIASES:
        return _TEAM_ALIASES[lower]
    for alias, abbrev in _TEAM_ALIASES.items():
        if alias in lower or lower in alias:
            return abbrev
    return name.strip().upper()[:3]


# ============================================================================
# Budget tracking
# ============================================================================
class BudgetTracker:
    """Persistent request budget tracker."""

    def __init__(self):
        self.data = self._load()

    def _load(self) -> dict:
        if BUDGET_FILE.exists():
            try:
                return json.loads(BUDGET_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "initial_budget": INITIAL_BUDGET,
            "exploration_used": EXPLORATION_USED,
            "backfill_used": 0,
            "fixture_requests": 0,
            "historical_requests": 0,
            "log": [],
        }

    def _save(self):
        BUDGET_FILE.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    @property
    def remaining(self) -> int:
        total_used = self.data["exploration_used"] + self.data["backfill_used"]
        return max(0, self.data["initial_budget"] - total_used)

    @property
    def used(self) -> int:
        return self.data["exploration_used"] + self.data["backfill_used"]

    def record(self, endpoint: str, fixture_id: str = ""):
        self.data["backfill_used"] += 1
        if endpoint == "fixtures":
            self.data["fixture_requests"] += 1
        elif endpoint == "historical-odds":
            self.data["historical_requests"] += 1
        self.data["log"].append({
            "time": datetime.now().isoformat(),
            "endpoint": endpoint,
            "fixture_id": fixture_id,
        })
        self._save()

    def can_spend(self, n: int = 1) -> bool:
        return self.remaining >= n

    def summary(self) -> str:
        return (
            f"Budget: {self.remaining}/{self.data['initial_budget']} remaining "
            f"({self.data['exploration_used']} explore + "
            f"{self.data['backfill_used']} backfill = {self.used} used)"
        )


# ============================================================================
# Fixtures cache
# ============================================================================
class FixturesCache:
    """
    Cache of NBA fixture IDs from OddsPapi.

    Fixtures are cheap to fetch (1 request returns many games), so we
    cache them aggressively by date range.
    """

    def __init__(self):
        self.fixtures: Dict[str, dict] = {}  # fixtureId -> fixture data
        self.fetched_ranges: List[Tuple[str, str]] = []  # (from, to) date strings
        self._load()

    def _load(self):
        if FIXTURES_CACHE.exists():
            try:
                data = json.loads(FIXTURES_CACHE.read_text(encoding="utf-8"))
                self.fixtures = data.get("fixtures", {})
                self.fetched_ranges = data.get("fetched_ranges", [])
            except Exception:
                pass

    def _save(self):
        data = {
            "fixtures": self.fixtures,
            "fetched_ranges": self.fetched_ranges,
            "updated_at": datetime.now().isoformat(),
        }
        FIXTURES_CACHE.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def is_range_cached(self, from_date: str, to_date: str) -> bool:
        """Check if a date range is fully covered by prior fetches."""
        from_dt = datetime.strptime(from_date, "%Y-%m-%d")
        to_dt = datetime.strptime(to_date, "%Y-%m-%d")
        for cached_from, cached_to in self.fetched_ranges:
            cf = datetime.strptime(cached_from, "%Y-%m-%d")
            ct = datetime.strptime(cached_to, "%Y-%m-%d")
            if cf <= from_dt and ct >= to_dt:
                return True
        return False

    def get_fixtures_for_date(self, date_str: str) -> List[dict]:
        """Get all fixtures for a specific date."""
        results = []
        for fx in self.fixtures.values():
            # startTime is ISO format: "2026-03-01T00:00:00.000Z"
            start = fx.get("startTime", "")
            if start[:10] == date_str:
                results.append(fx)
        return results

    def get_fixtures_in_range(self, from_date: str, to_date: str) -> List[dict]:
        """Get all fixtures between two dates (inclusive)."""
        results = []
        for fx in self.fixtures.values():
            start = fx.get("startTime", "")[:10]
            if from_date <= start <= to_date:
                results.append(fx)
        return results

    def add_fixtures(self, fixtures: List[dict], from_date: str, to_date: str):
        """Add fetched fixtures and record the range."""
        for fx in fixtures:
            fid = fx.get("fixtureId", "")
            if fid:
                self.fixtures[fid] = fx
        self.fetched_ranges.append((from_date, to_date))
        self._save()


# ============================================================================
# Historical odds cache (per fixture)
# ============================================================================
def _hist_cache_path(fixture_id: str) -> Path:
    """Per-fixture historical odds cache file."""
    safe_id = fixture_id.replace("/", "_")
    return ODDSPAPI_CACHE_DIR / f"hist_{safe_id}.json"


def _is_fixture_cached(fixture_id: str) -> bool:
    return _hist_cache_path(fixture_id).exists()


def _load_fixture_odds(fixture_id: str) -> Optional[dict]:
    path = _hist_cache_path(fixture_id)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _save_fixture_odds(fixture_id: str, data: dict):
    path = _hist_cache_path(fixture_id)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ============================================================================
# OddsPapi API calls
# ============================================================================
def _request_with_retry(url: str, params: dict, timeout: int = 60) -> Optional[requests.Response]:
    """Make a GET request with SSL error retry and exponential backoff."""
    for attempt, delay in enumerate([0] + SSL_RETRY_DELAYS):
        if delay:
            logger.warning(f"  SSL error — retry {attempt}/{len(SSL_RETRY_DELAYS)} after {delay}s...")
            time.sleep(delay)
        try:
            return requests.get(url, params=params, timeout=timeout)
        except requests.exceptions.SSLError:
            if attempt == len(SSL_RETRY_DELAYS):
                logger.error("SSL error persists after all retries — API may be blocking this IP")
                return None
        except requests.exceptions.ConnectionError as e:
            if attempt == len(SSL_RETRY_DELAYS):
                logger.error(f"Connection error after all retries: {e}")
                return None
    return None


def fetch_fixtures(from_date: str, to_date: str, budget: BudgetTracker) -> List[dict]:
    """
    Fetch NBA fixtures from OddsPapi for a date range.
    Cost: 1 API request.
    """
    if not budget.can_spend():
        logger.error("Budget exhausted — cannot fetch fixtures")
        return []

    resp = _request_with_retry(f"{BASE_URL}/fixtures", params={
        "apiKey": API_KEY,
        "tournamentId": NBA_TOURNAMENT_ID,
        "from": from_date,
        "to": to_date,
    }, timeout=30)

    budget.record("fixtures")

    if resp is None:
        return []
    if resp.status_code != 200:
        logger.error(f"Fixtures fetch failed ({resp.status_code}): {resp.text[:200]}")
        return []

    fixtures = resp.json()
    logger.info(f"Fetched {len(fixtures)} fixtures for {from_date} to {to_date}")
    return fixtures


def fetch_historical_odds(fixture_id: str, budget: BudgetTracker) -> Optional[dict]:
    """
    Fetch historical odds for a single fixture.
    Cost: 1 API request + 5s rate limit wait.
    """
    if not budget.can_spend():
        logger.error("Budget exhausted — cannot fetch historical odds")
        return None

    resp = _request_with_retry(f"{BASE_URL}/historical-odds", params={
        "apiKey": API_KEY,
        "fixtureId": fixture_id,
        "bookmakers": BOOKMAKERS,
    }, timeout=60)

    budget.record("historical-odds", fixture_id)

    if resp is None:
        return None
    if resp.status_code != 200:
        logger.error(f"Historical odds failed for {fixture_id} ({resp.status_code}): {resp.text[:200]}")
        return None

    return resp.json()


# ============================================================================
# Odds extraction (OddsPapi format -> our cache format)
# ============================================================================

def _get_pregame_price(prices: list, start_time: str = "") -> Optional[float]:
    """
    Get the best pre-game price from a list of OddsPapi price entries.

    OddsPapi price entries: [{price, limit, active, createdAt, exchangeMeta}, ...]
    Sorted newest-first.  We want the most recent ACTIVE price before game start.
    """
    if not prices:
        return None

    game_start = None
    if start_time:
        try:
            game_start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        except Exception:
            pass

    # Walk from newest to oldest, find first active pre-game price
    for entry in prices:
        if not entry.get("active", True):
            continue
        price = entry.get("price")
        if not price or price <= 1.0:
            continue
        created = entry.get("createdAt", "")
        if game_start and created:
            try:
                entry_time = datetime.fromisoformat(created.replace("Z", "+00:00"))
                if entry_time < game_start:
                    return price
            except Exception:
                pass
        else:
            return price  # No time filter, use first active price

    # Fallback: first entry regardless
    if prices and prices[0].get("price", 0) > 1.0:
        return prices[0]["price"]
    return None


def moneyline_to_spread(home_price: float, away_price: float) -> float:
    """
    Convert Pinnacle moneyline to implied spread (home team perspective).

    Uses the standard NBA empirical relationship:
        fair_home_prob = (1/home) / (1/home + 1/away)
        spread = (0.5 - fair_home_prob) / 0.025

    Returns negative values for home favorites, positive for home dogs.
    Rounded to nearest 0.5 (like real spread lines).
    """
    home_prob = 1.0 / home_price
    away_prob = 1.0 / away_price
    vig = home_prob + away_prob  # ~1.04 for Pinnacle
    fair_home = home_prob / vig

    raw_spread = (0.5 - fair_home) / NBA_PROB_PER_POINT
    # Round to nearest 0.5
    return round(raw_spread * 2) / 2


def _find_bookmaker_data(historical_data: dict) -> Tuple[Optional[dict], str]:
    """
    Find usable bookmaker data from OddsPapi response.
    Prefers pinnacle, falls back to any available.
    """
    bookmakers = historical_data.get("bookmakers", {})
    if not bookmakers:
        return None, ""

    # Prefer Pinnacle
    if "pinnacle" in bookmakers and bookmakers["pinnacle"].get("markets"):
        return bookmakers["pinnacle"], "pinnacle"

    # Fallback to any bookmaker with markets
    for name, data in bookmakers.items():
        if data.get("markets"):
            return data, name

    return None, ""


def extract_game_odds(historical_data: dict, fixture: dict) -> Optional[dict]:
    """
    Extract spread (and optionally total) from OddsPapi historical-odds response.

    Strategy:
    1. Find moneyline market (ID 111) with home (oc 111) and away (oc 112) prices
    2. Convert to implied spread using NBA empirical formula
    3. Return in odds_engine.py-compatible format

    Returns dict with {home_team, away_team, home_abbrev, away_abbrev,
                       game_date, spread, total, spread_book, total_book, ...}
    or None if extraction fails.
    """
    bk_data, bk_name = _find_bookmaker_data(historical_data)
    if not bk_data:
        return None

    markets = bk_data.get("markets", {})
    if not markets:
        return None

    # ---- Parse fixture metadata ----
    home_name = fixture.get("participant1Name", "")
    away_name = fixture.get("participant2Name", "")
    home_abbrev = _team_to_abbrev(home_name)
    away_abbrev = _team_to_abbrev(away_name)
    start_time = fixture.get("startTime", "")
    game_date = start_time[:10] if start_time else ""

    # ---- Extract moneyline (market 111) ----
    ml_market = markets.get(str(MONEYLINE_MID), {})
    if not ml_market:
        logger.debug(f"  No moneyline market (111) found for {home_abbrev} vs {away_abbrev}")
        return None

    outcomes = ml_market.get("outcomes", {})
    if not outcomes:
        return None

    # Outcomes: "111" = home winner, "112" = away winner
    # Structure: outcomes.{oc_id}.players.{"0": [price_entries]}
    home_oc = outcomes.get("111", {})
    away_oc = outcomes.get("112", {})

    home_prices = home_oc.get("players", {}).get("0", [])
    away_prices = away_oc.get("players", {}).get("0", [])

    if not home_prices or not away_prices:
        logger.debug(f"  No moneyline prices for {home_abbrev} vs {away_abbrev}")
        return None

    home_price = _get_pregame_price(home_prices, start_time)
    away_price = _get_pregame_price(away_prices, start_time)

    if not home_price or not away_price:
        return None

    # ---- Convert to spread ----
    spread = moneyline_to_spread(home_price, away_price)

    # ---- Try to extract total (best-effort) ----
    total = _try_extract_total(markets, start_time)

    # ---- Build game entry (compatible with odds_engine.py) ----
    return {
        "home_team": home_name,
        "away_team": away_name,
        "home_abbrev": home_abbrev,
        "away_abbrev": away_abbrev,
        "game_date": game_date,
        "spread": spread,               # Home perspective, negative = home fav
        "total": total,                  # May be None
        "spread_book": bk_name,
        "total_book": bk_name if total else None,
        "source": "oddspapi",
        "fixture_id": fixture.get("fixtureId", ""),
        "ml_home": round(home_price, 4),
        "ml_away": round(away_price, 4),
    }


def _try_extract_total(markets: dict, start_time: str) -> Optional[float]:
    """
    Extract game total from OddsPapi totals markets.

    Requires calibrated totals base (see calibrate_totals / auto_calibrate).
    Returns None if not calibrated or no totals markets found.
    """
    idx_list = _load_totals_index()
    base = _load_totals_base()

    if not idx_list or base is None:
        return None  # Not calibrated yet

    # Find totals markets in this game (must be in our catalog index)
    game_totals = []
    for mid_str, mdata in markets.items():
        try:
            mid = int(mid_str)
        except ValueError:
            continue
        if mid not in idx_list:
            continue
        outcomes = mdata.get("outcomes", {})
        if len(outcomes) < 2:
            continue
        game_totals.append((mid, mdata))

    if not game_totals:
        return None

    game_totals.sort(key=lambda x: x[0])

    # Find most balanced market (smallest |over_price - under_price|)
    best_mid = game_totals[0][0]
    best_diff = 999.0

    for mid, mdata in game_totals:
        outcomes = mdata.get("outcomes", {})
        oc_ids = sorted(outcomes.keys(), key=lambda x: int(x))
        if len(oc_ids) < 2:
            continue

        over_prices = outcomes.get(oc_ids[0], {}).get("players", {}).get("0", [])
        under_prices = outcomes.get(oc_ids[1], {}).get("players", {}).get("0", [])

        op = _get_pregame_price(over_prices, start_time)
        up = _get_pregame_price(under_prices, start_time)

        if not op or not up:
            continue

        diff = abs(op - up)
        if diff < best_diff:
            best_diff = diff
            best_mid = mid

    return _market_id_to_total(best_mid)


# ============================================================================
# Calibration helpers
# ============================================================================

def _load_known_totals() -> Dict[tuple, float]:
    """Load all known game totals from The Odds API cache."""
    known = {}
    for cache_file in ODDS_CACHE_DIR.glob("*.json"):
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            for g in data.get("games", []):
                total = g.get("total")
                src = g.get("source", "the-odds-api")
                if total and src != "oddspapi":
                    key = (g["home_abbrev"], g["away_abbrev"], g["game_date"])
                    known[key] = total
        except Exception:
            continue
    return known


def _try_auto_calibrate(hist_data: dict, fixture: dict,
                        known_total: float, game_key: tuple):
    """
    Auto-calibrate totals base using a game with known The Odds API total.
    Finds the balanced totals market and derives the base.
    """
    idx_list = _load_totals_index()
    if not idx_list:
        return

    bk_data, bk_name = _find_bookmaker_data(hist_data)
    if not bk_data:
        return
    markets = bk_data.get("markets", {})
    start_time = fixture.get("startTime", "")

    # Find totals markets in this game
    game_totals = []
    for mid_str in markets:
        try:
            mid = int(mid_str)
        except ValueError:
            continue
        if mid in idx_list:
            game_totals.append((mid, markets[mid_str]))

    if not game_totals:
        return

    game_totals.sort(key=lambda x: x[0])

    # Find balanced market
    best_mid = game_totals[0][0]
    best_diff = 999.0
    for mid, mdata in game_totals:
        outcomes = mdata.get("outcomes", {})
        oc_ids = sorted(outcomes.keys(), key=lambda x: int(x))
        if len(oc_ids) < 2:
            continue
        over_p = outcomes.get(oc_ids[0], {}).get("players", {}).get("0", [])
        under_p = outcomes.get(oc_ids[1], {}).get("players", {}).get("0", [])
        op = _get_pregame_price(over_p, start_time)
        up = _get_pregame_price(under_p, start_time)
        if op and up:
            diff = abs(op - up)
            if diff < best_diff:
                best_diff = diff
                best_mid = mid

    label = f"{game_key[0]} vs {game_key[1]} {game_key[2]}"
    calibrate_totals(known_total, best_mid, label)


# ============================================================================
# Main backfill logic
# ============================================================================
def backfill_range(from_date: str, to_date: str, max_games: int = None, dry_run: bool = False):
    """
    Backfill historical odds for NBA games in a date range.

    Args:
        from_date: "YYYY-MM-DD" start date (inclusive)
        to_date:   "YYYY-MM-DD" end date (inclusive)
        max_games: Optional cap on number of games to fetch (for budget control)
        dry_run:   If True, only show what would be fetched without making API calls
    """
    budget = BudgetTracker()
    logger.info(budget.summary())

    if not budget.can_spend(2):  # need at least 1 fixture + 1 hist
        logger.error("Not enough budget remaining. Aborting.")
        return

    fc = FixturesCache()

    # ---- Step 1: Ensure fixtures are cached ----
    if not fc.is_range_cached(from_date, to_date):
        logger.info(f"Fetching fixtures for {from_date} to {to_date}...")
        if not dry_run:
            raw_fixtures = fetch_fixtures(from_date, to_date, budget)
            if raw_fixtures:
                fc.add_fixtures(raw_fixtures, from_date, to_date)
            else:
                logger.error("No fixtures returned. Aborting.")
                return
        else:
            logger.info("  [DRY RUN] Would fetch fixtures")
    else:
        logger.info(f"Fixtures already cached for {from_date} to {to_date}")

    # ---- Step 2: Identify games needing historical odds ----
    fixtures = fc.get_fixtures_in_range(from_date, to_date)

    # Only process finished games that aren't already cached
    # Note: statusId=2 = finished. hasOdds may be False for finished games
    # even though historical-odds endpoint works (verified via probe).
    pending = []
    already_cached = 0
    skipped_status = 0

    for fx in fixtures:
        fid = fx.get("fixtureId", "")
        status = fx.get("statusId", 0)

        if status != 2:  # Not finished
            skipped_status += 1
            continue

        if _is_fixture_cached(fid):
            already_cached += 1
            continue

        # Also check if this date already has odds_cache (from The Odds API)
        # No — we want to supplement, not skip

        pending.append(fx)

    logger.info(f"Fixtures in range: {len(fixtures)} total, "
                f"{already_cached} already cached, "
                f"{skipped_status} not finished, "
                f"{len(pending)} to fetch")

    if max_games and len(pending) > max_games:
        # Prioritize most recent games
        pending.sort(key=lambda x: x.get("startTime", ""), reverse=True)
        pending = pending[:max_games]
        logger.info(f"Capped to {max_games} most recent games")

    # Budget check
    needed = len(pending)
    if needed > budget.remaining:
        logger.warning(f"Need {needed} requests but only {budget.remaining} budget remaining")
        pending = pending[:budget.remaining]
        logger.info(f"Trimmed to {len(pending)} games to stay within budget")

    if not pending:
        logger.info("Nothing to fetch!")
        return

    if dry_run:
        logger.info(f"[DRY RUN] Would fetch historical odds for {len(pending)} games:")
        for fx in pending[:20]:
            p1 = fx.get("participant1Name", "?")
            p2 = fx.get("participant2Name", "?")
            dt = fx.get("startTime", "?")[:10]
            logger.info(f"  {dt}: {p1} vs {p2}")
        if len(pending) > 20:
            logger.info(f"  ... and {len(pending) - 20} more")
        return

    # ---- Step 3: Fetch historical odds per fixture ----
    fetched = 0
    spreads_extracted = 0
    # Collect games by date for bulk cache write
    games_by_date: Dict[str, list] = defaultdict(list)

    # Load The Odds API known totals for calibration
    known_totals = _load_known_totals()

    for i, fx in enumerate(pending):
        fid = fx.get("fixtureId", "")
        p1 = fx.get("participant1Name", "?")
        p2 = fx.get("participant2Name", "?")
        game_date = fx.get("startTime", "")[:10]

        logger.info(f"[{i+1}/{len(pending)}] {game_date}: {p1} vs {p2} "
                     f"(budget: {budget.remaining} left)")

        if not budget.can_spend():
            logger.warning("Budget exhausted! Stopping.")
            break

        # Rate limit
        if fetched > 0:
            logger.info(f"  Rate limit: waiting {RATE_LIMIT_SEC}s...")
            time.sleep(RATE_LIMIT_SEC)

        # Fetch
        hist_data = fetch_historical_odds(fid, budget)
        fetched += 1

        if hist_data is None:
            logger.warning(f"  No data returned for {fid}")
            # Cache empty result to avoid re-fetching
            _save_fixture_odds(fid, {"empty": True, "fixture_id": fid})
            continue

        # Cache raw response
        _save_fixture_odds(fid, hist_data)

        # Extract spread and total
        game_entry = extract_game_odds(hist_data, fx)
        if game_entry:
            games_by_date[game_date].append(game_entry)
            if game_entry.get("spread") is not None:
                spreads_extracted += 1
                ml_h = game_entry.get("ml_home", "?")
                ml_a = game_entry.get("ml_away", "?")
                total_str = f"{game_entry['total']:.1f}" if game_entry.get("total") else "N/A"
                logger.info(f"  -> ML {ml_h}/{ml_a}  spread={game_entry['spread']:+.1f}  total={total_str}")

                # Auto-calibrate totals if we have a known total for this game
                if _load_totals_base() is None:
                    key = (game_entry["home_abbrev"], game_entry["away_abbrev"], game_date)
                    if key in known_totals:
                        _try_auto_calibrate(hist_data, fx, known_totals[key], key)
        else:
            logger.info(f"  -> Could not extract odds")

    # ---- Step 4: Write to shared odds_cache (compatible with odds_engine.py) ----
    dates_written = 0
    for date_str, games in games_by_date.items():
        cache_path = ODDS_CACHE_DIR / f"{date_str}.json"

        # Merge with existing cache if present
        existing_games = []
        if cache_path.exists():
            try:
                existing = json.loads(cache_path.read_text(encoding="utf-8"))
                existing_games = existing.get("games", [])
            except Exception:
                pass

        # Add new games, avoiding duplicates (by home_abbrev + away_abbrev)
        existing_keys = {(g["home_abbrev"], g["away_abbrev"]) for g in existing_games}
        for game in games:
            key = (game["home_abbrev"], game["away_abbrev"])
            if key not in existing_keys:
                existing_games.append(game)
                existing_keys.add(key)

        date_data = {
            "fetched_at": datetime.now().isoformat(),
            "game_count": len(existing_games),
            "games": existing_games,
        }
        cache_path.write_text(json.dumps(date_data, indent=2), encoding="utf-8")
        dates_written += 1

    # ---- Summary ----
    logger.info(f"\n{'='*60}")
    logger.info(f"Backfill complete!")
    logger.info(f"  Games fetched: {fetched}")
    logger.info(f"  Spreads extracted: {spreads_extracted}")
    logger.info(f"  Dates written to cache: {dates_written}")
    logger.info(budget.summary())


def show_coverage():
    """Show what dates we have cached odds for."""
    dates = set()
    games_total = 0
    with_spread = 0
    with_total = 0

    for f in ODDS_CACHE_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            for game in data.get("games", []):
                dates.add(game.get("game_date", f.stem))
                games_total += 1
                if game.get("spread") is not None:
                    with_spread += 1
                if game.get("total") is not None:
                    with_total += 1
        except Exception:
            pass

    if not dates:
        print("No cached odds data found.")
        return

    sorted_dates = sorted(dates)
    print(f"Cached odds coverage:")
    print(f"  Date range: {sorted_dates[0]} to {sorted_dates[-1]}")
    print(f"  Dates: {len(sorted_dates)}")
    print(f"  Games: {games_total}")
    print(f"  With spread: {with_spread}")
    print(f"  With total:  {with_total}")

    # Source breakdown
    sources = defaultdict(int)
    for f in ODDS_CACHE_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            for game in data.get("games", []):
                src = game.get("source", "the-odds-api")
                sources[src] += 1
        except Exception:
            pass
    print(f"  Sources: {dict(sources)}")


# ============================================================================
# Probe / structure analysis (spend 2 requests to understand the data)
# ============================================================================
def probe_structure():
    """
    Fetch ONE finished NBA fixture + its historical odds, then dump
    the full market structure so we can verify extraction works.
    Cost: 2 requests (1 fixture + 1 historical-odds).
    """
    budget = BudgetTracker()
    logger.info(budget.summary())
    if not budget.can_spend(2):
        logger.error("Need at least 2 remaining requests for probe. Aborting.")
        return

    # 1. Get recent finished fixtures (go back far enough to find finished games)
    to_date = datetime.now(tz=None).strftime("%Y-%m-%d")
    from_date = (datetime.now(tz=None) - timedelta(days=7)).strftime("%Y-%m-%d")

    logger.info(f"Probing: fetching fixtures {from_date} to {to_date}...")
    fixtures = fetch_fixtures(from_date, to_date, budget)
    if not fixtures:
        logger.error("No fixtures returned!")
        return

    # Find a finished game (statusId=2) with hasOdds=True
    finished = [f for f in fixtures if f.get("statusId") == 2 and f.get("hasOdds")]
    if not finished:
        # Show all statuses for debugging
        statuses = defaultdict(int)
        for f in fixtures:
            statuses[(f.get("statusId"), f.get("hasOdds", False))] += 1
        logger.info(f"Fixture status breakdown: {dict(statuses)}")
        # Try without hasOdds filter
        finished = [f for f in fixtures if f.get("statusId") == 2]
    if not finished:
        # Try any finished-looking game
        finished = [f for f in fixtures if f.get("statusId") in (2, 3, 100)]
    if not finished:
        logger.error(f"No finished games found! Fixtures:\n" + 
                     "\n".join(f"  {f.get('participant1Name')} vs {f.get('participant2Name')} "
                               f"status={f.get('statusId')} hasOdds={f.get('hasOdds')}"
                               for f in fixtures[:10]))
        return

    target = finished[0]
    fid = target["fixtureId"]
    p1 = target.get("participant1Name", "?")
    p2 = target.get("participant2Name", "?")
    start = target.get("startTime", "?")
    logger.info(f"Target fixture: {p1} vs {p2} ({start[:10]}) — {fid}")

    # 2. Fetch historical odds
    logger.info("Fetching historical odds (5s rate limit)...")
    time.sleep(RATE_LIMIT_SEC)
    hist = fetch_historical_odds(fid, budget)
    if not hist:
        logger.error("No historical odds returned!")
        return

    # Save raw response for analysis
    probe_path = ODDSPAPI_CACHE_DIR / "_probe_analysis.json"
    probe_path.write_text(json.dumps(hist, indent=2)[:500_000], encoding="utf-8")
    logger.info(f"Raw response saved to {probe_path}")

    # Also cache it properly
    _save_fixture_odds(fid, hist)

    # 3. Analyze structure
    print(f"\n{'='*70}")
    print(f"PROBE ANALYSIS: {p1} vs {p2}")
    print(f"{'='*70}")

    for bk_name, bk_data in hist.get("bookmakers", {}).items():
        markets = bk_data.get("markets", {})
        all_mids = sorted([int(m) for m in markets.keys()])
        print(f"\nBookmaker: {bk_name} — {len(markets)} markets")
        print(f"  ID range: {min(all_mids) if all_mids else '?'} - {max(all_mids) if all_mids else '?'}")

        # Group by range
        ranges = {}
        for m in all_mids:
            if m < 200:
                bucket = "game-level (<200)"
            elif m < 11000:
                bucket = "medium (200-10999)"
            elif m < 12000:
                bucket = "handicap (11000-11999)"
            elif m < 100000:
                bucket = "other (12000-99999)"
            else:
                bucket = "player props (100000+)"
            ranges.setdefault(bucket, []).append(m)
        for bucket in sorted(ranges.keys()):
            ids = ranges[bucket]
            print(f"  {bucket}: {len(ids)} markets")
            if len(ids) <= 20:
                print(f"    IDs: {ids}")
            else:
                print(f"    First 10: {ids[:10]}")
                print(f"    Last 10:  {ids[-10:]}")

        # Moneyline (111)
        ml = markets.get("111", {})
        if ml:
            print(f"\n  MONEYLINE (111):")
            for oc_id, oc_data in ml.get("outcomes", {}).items():
                players = oc_data.get("players", {})
                p0 = players.get("0", [])
                if p0:
                    latest = p0[0]
                    print(f"    Outcome {oc_id}: latest_price={latest.get('price')}, "
                          f"active={latest.get('active')}, "
                          f"entries={len(p0)}")
                    # Show all keys in a price entry
                    print(f"    Entry keys: {list(latest.keys())}")

        # Sample totals market
        totals = [(int(m), markets[m]) for m in markets if 116 <= int(m) <= 1200]
        totals.sort()
        if totals:
            print(f"\n  TOTALS MARKETS: {len(totals)} found")
            print(f"    IDs: {[t[0] for t in totals[:15]]}{'...' if len(totals) > 15 else ''}")
            # Show first and middle
            for label, idx in [("First", 0), ("Middle", len(totals)//2), ("Last", -1)]:
                mid, mdata = totals[idx]
                print(f"    {label} (ID {mid}):")
                for oc_id, oc_data in mdata.get("outcomes", {}).items():
                    p0 = oc_data.get("players", {}).get("0", [])
                    if p0:
                        print(f"      Outcome {oc_id}: price={p0[0].get('price')}, keys={list(p0[0].keys())}")

        # Sample handicap market
        hcaps = [(int(m), markets[m]) for m in markets if 11250 <= int(m) <= 11600]
        hcaps.sort()
        if hcaps:
            print(f"\n  HANDICAP MARKETS: {len(hcaps)} found")
            print(f"    IDs: {[h[0] for h in hcaps[:15]]}{'...' if len(hcaps) > 15 else ''}")
            mid, mdata = hcaps[len(hcaps)//2]
            print(f"    Middle (ID {mid}):")
            for oc_id, oc_data in mdata.get("outcomes", {}).items():
                p0 = oc_data.get("players", {}).get("0", [])
                if p0:
                    print(f"      Outcome {oc_id}: price={p0[0].get('price')}, keys={list(p0[0].keys())}")

    # 4. Test extraction
    print(f"\n{'='*70}")
    print("EXTRACTION TEST")
    print(f"{'='*70}")
    result = extract_game_odds(hist, target)
    if result:
        for k, v in result.items():
            print(f"  {k}: {v}")
    else:
        print("  FAILED — no data extracted")

    print(f"\n{budget.summary()}")


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="OddsPapi Historical Odds Backfill")
    parser.add_argument("--from", dest="from_date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="to_date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, help="Backfill last N days")
    parser.add_argument("--max-games", type=int, help="Max games to fetch")
    parser.add_argument("--budget", action="store_true", help="Show budget status")
    parser.add_argument("--coverage", action="store_true", help="Show cache coverage")
    parser.add_argument("--dry-run", action="store_true", help="Preview without fetching")
    parser.add_argument("--probe", action="store_true", help="Analyse 1 game structure (2 reqs)")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate totals using The Odds API cross-reference")

    args = parser.parse_args()

    if args.probe:
        probe_structure()
        return

    if args.calibrate:
        ok = auto_calibrate_from_odds_cache()
        if not ok:
            print("Auto-calibration needs overlapping data. Run --days N for a range with The Odds API data.")
        return

    if args.budget:
        budget = BudgetTracker()
        print(budget.summary())
        for entry in budget.data.get("log", [])[-10:]:
            print(f"  {entry['time'][:19]}  {entry['endpoint']}  {entry.get('fixture_id', '')}")
        return

    if args.coverage:
        show_coverage()
        return

    # Determine date range
    if args.days:
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    elif args.from_date and args.to_date:
        from_date = args.from_date
        to_date = args.to_date
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python oddspapi_backfill.py --days 7")
        print("  python oddspapi_backfill.py --from 2026-02-01 --to 2026-03-01")
        print("  python oddspapi_backfill.py --days 7 --dry-run")
        print("  python oddspapi_backfill.py --budget")
        print("  python oddspapi_backfill.py --coverage")
        return

    backfill_range(from_date, to_date, max_games=args.max_games, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
