"""
Fetch 2025-26 season spreads/totals from OddsPapi and update nba_2008-2025.csv.

Phase 1 (no API calls): Extract odds from 86 already-cached OddsPapi historical
odds files, auto-calibrate totals using The Odds API cross-reference.

Phase 2 (optional, uses API): Fetch historical odds for games still missing data.

Run:
  python scripts/fetch_oddspapi_season_odds.py                   # Phase 1 only (offline)
  python scripts/fetch_oddspapi_season_odds.py --fetch-missing 10 # Also fetch up to 10 new
  python scripts/fetch_oddspapi_season_odds.py --fixtures-only    # Just fetch fixtures
  python scripts/fetch_oddspapi_season_odds.py --reset-budget 14  # Reset budget counter
"""

import json
import sys
import os
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict

import requests
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
ODDSPAPI_CACHE_DIR = DATA_DIR / "oddspapi_cache"
ODDSPAPI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
ODDS_CACHE_DIR = DATA_DIR / "odds_cache"
CSV_PATH = DATA_DIR / "nba_2008-2025.csv"

BASE_URL = "https://api.oddspapi.io/v4"
API_KEY = os.environ.get("ODDSPAPI_API_KEY", "")
NBA_TOURNAMENT_ID = 132
SPORT_ID = 11
BOOKMAKERS = "pinnacle"
RATE_LIMIT_SEC = 8.0

BUDGET_FILE = ODDSPAPI_CACHE_DIR / "_budget.json"
FIXTURES_CACHE = ODDSPAPI_CACHE_DIR / "_fixtures.json"
TOTALS_CALIBRATION_FILE = ODDSPAPI_CACHE_DIR / "_totals_calibration.json"
MARKETS_CATALOG_FILE = ODDSPAPI_CACHE_DIR / "_markets_catalog.json"

NBA_PROB_PER_POINT = 0.025

if not API_KEY:
    logger.warning("ODDSPAPI_API_KEY is not set; OddsPapi requests will fail until it is configured")

# ========== Team mappings ==========
STD_TO_CSV = {
    "ATL": "atl", "BOS": "bos", "BKN": "bkn", "CHA": "cha", "CHI": "chi",
    "CLE": "cle", "DAL": "dal", "DEN": "den", "DET": "det", "GSW": "gs",
    "HOU": "hou", "IND": "ind", "LAC": "lac", "LAL": "lal", "MEM": "mem",
    "MIA": "mia", "MIL": "mil", "MIN": "min", "NOP": "no",  "NYK": "ny",
    "OKC": "okc", "ORL": "orl", "PHI": "phi", "PHX": "phx", "POR": "por",
    "SAC": "sac", "SAS": "sa",  "TOR": "tor", "UTA": "utah", "WAS": "wsh",
}

TEAM_ALIASES = {
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

CSV_TO_STD = {v: k for k, v in STD_TO_CSV.items()}


def _team_to_abbrev(name: str) -> str:
    lower = name.strip().lower()
    if lower in TEAM_ALIASES:
        return TEAM_ALIASES[lower]
    for alias, abbrev in TEAM_ALIASES.items():
        if alias in lower or lower in alias:
            return abbrev
    return name.strip().upper()[:3]


def _utc_to_local_date(utc_str: str) -> str:
    """Convert UTC startTime to US Eastern local date (approximate).
    
    Most NBA games start 7-10pm ET, which is 00:00-03:00 UTC next day.
    We subtract 5 hours (EST offset) to get the correct local date.
    """
    try:
        utc_dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
    except Exception:
        utc_dt = datetime.strptime(utc_str[:19], "%Y-%m-%dT%H:%M:%S")
    # EST = UTC - 5h
    local_dt = utc_dt - timedelta(hours=5)
    return local_dt.strftime("%Y-%m-%d")


# ========== Budget ==========
class Budget:
    def __init__(self):
        self.data = self._load()

    def _load(self):
        if BUDGET_FILE.exists():
            try:
                return json.loads(BUDGET_FILE.read_text())
            except Exception:
                pass
        return {"initial_budget": 250, "exploration_used": 0, "backfill_used": 0,
                "fixture_requests": 0, "historical_requests": 0, "log": []}

    def save(self):
        BUDGET_FILE.write_text(json.dumps(self.data, indent=2))

    @property
    def remaining(self):
        return max(0, self.data["initial_budget"] - self.data["exploration_used"] - self.data["backfill_used"])

    def record(self, endpoint, fixture_id=""):
        self.data["backfill_used"] += 1
        if "fixture" in endpoint:
            self.data["fixture_requests"] += 1
        else:
            self.data["historical_requests"] += 1
        self.data["log"].append({"time": datetime.now().isoformat(), "endpoint": endpoint, "fixture_id": fixture_id})
        self.save()

    def reset_to(self, used_so_far):
        self.data["exploration_used"] = 0
        self.data["backfill_used"] = used_so_far
        self.data["log"] = []
        self.save()
        logger.info(f"Budget reset: {self.remaining} remaining (used={used_so_far})")


# ========== Totals calibration ==========
_TOTALS_INDEX: Optional[List[int]] = None
_TOTALS_BASE: Optional[float] = None


def _load_totals_index() -> List[int]:
    global _TOTALS_INDEX
    if _TOTALS_INDEX is not None:
        return _TOTALS_INDEX
    if not MARKETS_CATALOG_FILE.exists():
        logger.warning("Markets catalog not found --- totals disabled")
        _TOTALS_INDEX = []
        return _TOTALS_INDEX
    try:
        all_markets = json.loads(MARKETS_CATALOG_FILE.read_text(encoding="utf-8"))
        fg = [m for m in all_markets
              if m.get("sportId") == SPORT_ID
              and m.get("marketName") == "Over Under (incl. overtime)"]
        _TOTALS_INDEX = sorted(m["marketId"] for m in fg)
        logger.info(f"Loaded {len(_TOTALS_INDEX)} totals market IDs from catalog")
    except Exception as e:
        logger.warning(f"Failed to load markets catalog: {e}")
        _TOTALS_INDEX = []
    return _TOTALS_INDEX


def _load_totals_base() -> Optional[float]:
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
    idx_list = _load_totals_index()
    base = _load_totals_base()
    if not idx_list or base is None:
        return None
    if market_id not in idx_list:
        return None
    pos = idx_list.index(market_id)
    return base + pos * 0.5


def _get_pregame_price(prices, start_time=""):
    if not prices:
        return None
    game_start = None
    if start_time:
        try:
            game_start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        except Exception:
            pass

    # Try to find last pre-game price
    best = None
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
                    best = price  # keep last one before game start
            except Exception:
                if best is None:
                    best = price
        else:
            if best is None:
                best = price

    # Fallback to first valid price
    if best is None and prices:
        for entry in prices:
            p = entry.get("price", 0)
            if p > 1.0:
                return p
    return best


def _try_extract_total(markets: dict, start_time: str) -> Optional[float]:
    """Extract game total from OddsPapi totals markets using calibration."""
    idx_list = _load_totals_index()
    base = _load_totals_base()
    if not idx_list or base is None:
        return None

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


def auto_calibrate_totals():
    """Cross-reference OddsPapi cache with The Odds API cache to calibrate totals."""
    global _TOTALS_BASE
    idx_list = _load_totals_index()
    if not idx_list:
        logger.warning("No totals index loaded --- skipping calibration")
        return False

    # Load known totals from The Odds API cache
    known_totals: Dict[tuple, float] = {}  # (date, home_abbrev, away_abbrev) -> total
    for cache_file in ODDS_CACHE_DIR.glob("*.json"):
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            for g in data.get("games", []):
                total = g.get("total") or g.get("game_total")
                if total and float(total) > 100:
                    home = g.get("home_abbrev", "").upper()
                    away = g.get("away_abbrev", "").upper()
                    gd = g.get("game_date", cache_file.stem)
                    known_totals[(gd, home, away)] = float(total)
        except Exception:
            continue

    logger.info(f"Loaded {len(known_totals)} known totals from odds_cache")

    # Load fixtures for date-mapping
    fixtures = {}
    if FIXTURES_CACHE.exists():
        try:
            fixtures = json.loads(FIXTURES_CACHE.read_text())["fixtures"]
        except Exception:
            pass

    # Find overlapping games
    calibrations = []
    for hist_file in ODDSPAPI_CACHE_DIR.glob("hist_*.json"):
        fid = hist_file.stem.replace("hist_", "")
        fx = fixtures.get(fid)
        if not fx:
            continue

        home = _team_to_abbrev(fx.get("participant1Name", ""))
        away = _team_to_abbrev(fx.get("participant2Name", ""))
        utc_date = fx.get("startTime", "")[:10]
        local_date = _utc_to_local_date(fx.get("startTime", ""))
        start_time = fx.get("startTime", "")

        # Try both UTC date and local date for matching
        known = known_totals.get((local_date, home, away)) or known_totals.get((utc_date, home, away))
        if not known:
            continue

        # Load historical odds
        try:
            hist = json.loads(hist_file.read_text())
        except Exception:
            continue

        bookmakers = hist.get("bookmakers", {})
        bk_data = bookmakers.get("pinnacle") or next(iter(bookmakers.values()), {})
        markets = bk_data.get("markets", {})

        # Find the balanced totals market
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
            continue

        game_totals.sort(key=lambda x: x[0])
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

        pos = idx_list.index(best_mid)
        base = known - pos * 0.5
        label = f"{away} @ {home} {local_date}"
        calibrations.append((base, label, known, best_mid, pos))

    if not calibrations:
        logger.warning("No overlapping games found for calibration")
        return False

    # Use median base for robustness
    bases = [c[0] for c in calibrations]
    median_base = sorted(bases)[len(bases) // 2]

    # Find the calibration closest to median
    best = min(calibrations, key=lambda c: abs(c[0] - median_base))
    _TOTALS_BASE = best[0]

    cal = {
        "base": best[0],
        "calibrated_from": best[1],
        "known_total": best[2],
        "market_id": best[3],
        "index": best[4],
        "calibrated_at": datetime.now().isoformat(),
        "num_cross_references": len(calibrations),
        "base_range": [min(bases), max(bases)],
        "base_median": median_base,
    }
    TOTALS_CALIBRATION_FILE.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    logger.info(f"Totals auto-calibrated: base={best[0]:.1f} from {best[1]} "
                f"({len(calibrations)} cross-refs, range={min(bases):.1f}-{max(bases):.1f})")
    return True


# ========== Moneyline to spread ==========
def moneyline_to_spread(home_price, away_price):
    home_prob = 1.0 / home_price
    away_prob = 1.0 / away_price
    vig = home_prob + away_prob
    fair_home = home_prob / vig
    raw_spread = (0.5 - fair_home) / NBA_PROB_PER_POINT
    return round(raw_spread * 2) / 2


# ========== Fixtures ==========
def load_fixtures_cache():
    if FIXTURES_CACHE.exists():
        try:
            data = json.loads(FIXTURES_CACHE.read_text())
            return data.get("fixtures", {}), data.get("fetched_ranges", [])
        except Exception:
            pass
    return {}, []


def save_fixtures_cache(fixtures, ranges):
    data = {"fixtures": fixtures, "fetched_ranges": ranges, "updated_at": datetime.now().isoformat()}
    FIXTURES_CACHE.write_text(json.dumps(data, indent=2))


def fetch_fixtures_range(from_date, to_date, budget):
    if budget.remaining < 1:
        logger.error("No budget remaining!")
        return []
    logger.info(f"Fetching fixtures {from_date} to {to_date}...")
    try:
        resp = requests.get(f"{BASE_URL}/fixtures", params={
            "apiKey": API_KEY,
            "tournamentId": NBA_TOURNAMENT_ID,
            "from": from_date,
            "to": to_date,
        }, timeout=30)
        budget.record("fixtures")
        if resp.status_code != 200:
            logger.error(f"  Failed ({resp.status_code}): {resp.text[:200]}")
            return []
        fixtures = resp.json()
        logger.info(f"  Got {len(fixtures)} fixtures")
        return fixtures
    except Exception as e:
        logger.error(f"  Request failed: {e}")
        return []


def is_range_covered(ranges, from_d, to_d):
    from_dt = datetime.strptime(from_d, "%Y-%m-%d")
    to_dt = datetime.strptime(to_d, "%Y-%m-%d")
    for cf, ct in ranges:
        if datetime.strptime(cf, "%Y-%m-%d") <= from_dt and datetime.strptime(ct, "%Y-%m-%d") >= to_dt:
            return True
    return False


def fetch_all_season_fixtures(budget):
    fixtures, ranges = load_fixtures_cache()
    start = datetime(2025, 10, 21)
    end = datetime(2026, 3, 5)
    chunk_days = 14
    current = start
    new_count = 0
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        from_d = current.strftime("%Y-%m-%d")
        to_d = chunk_end.strftime("%Y-%m-%d")
        if is_range_covered(ranges, from_d, to_d):
            logger.info(f"  {from_d} to {to_d} --- cached, skip")
        else:
            if budget.remaining < 1:
                break
            result = fetch_fixtures_range(from_d, to_d, budget)
            if result:
                for fx in result:
                    fid = fx.get("fixtureId", "")
                    if fid:
                        fixtures[fid] = fx
                        new_count += 1
                ranges.append((from_d, to_d))
            time.sleep(2)
        current = chunk_end
    save_fixtures_cache(fixtures, ranges)
    logger.info(f"Fixtures: {len(fixtures)} total ({new_count} new)")
    return fixtures


# ========== Historical odds ==========
def hist_cache_path(fixture_id):
    return ODDSPAPI_CACHE_DIR / f"hist_{fixture_id.replace('/', '_')}.json"


def fetch_historical_odds(fixture_id, budget):
    cache = hist_cache_path(fixture_id)
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass
    if budget.remaining < 1:
        return None
    logger.info(f"  Fetching odds for {fixture_id}...")
    try:
        resp = requests.get(f"{BASE_URL}/historical-odds", params={
            "apiKey": API_KEY,
            "fixtureId": fixture_id,
            "bookmakers": BOOKMAKERS,
        }, timeout=60)
        budget.record("historical-odds", fixture_id)
        if resp.status_code != 200:
            logger.warning(f"    {resp.status_code}: {resp.text[:100]}")
            return None
        data = resp.json()
        cache.write_text(json.dumps(data, indent=2))
        return data
    except Exception as e:
        logger.error(f"    Request failed: {e}")
        return None


def extract_odds_from_historical(hist_data, fixture):
    """Extract spread + total from cached OddsPapi historical odds."""
    bookmakers = hist_data.get("bookmakers", {})
    if not bookmakers:
        return None

    bk_data = bookmakers.get("pinnacle", {})
    if not bk_data.get("markets"):
        for name, data in bookmakers.items():
            if data.get("markets"):
                bk_data = data
                break

    markets = bk_data.get("markets", {})
    if not markets:
        return None

    home_name = fixture.get("participant1Name", "")
    away_name = fixture.get("participant2Name", "")
    start_time = fixture.get("startTime", "")

    # Moneyline -> spread (market 111)
    ml_market = markets.get("111", {})
    outcomes = ml_market.get("outcomes", {})
    home_oc = outcomes.get("111", {})
    away_oc = outcomes.get("112", {})

    home_prices = home_oc.get("players", {}).get("0", [])
    away_prices = away_oc.get("players", {}).get("0", [])

    home_price = _get_pregame_price(home_prices, start_time)
    away_price = _get_pregame_price(away_prices, start_time)

    spread = None
    if home_price and away_price:
        spread = moneyline_to_spread(home_price, away_price)

    # Total from calibrated totals markets
    total = _try_extract_total(markets, start_time)

    return {
        "home_abbrev": _team_to_abbrev(home_name),
        "away_abbrev": _team_to_abbrev(away_name),
        "game_date_utc": start_time[:10],
        "game_date_local": _utc_to_local_date(start_time),
        "spread": spread,
        "total": total,
        "ml_home": home_price,
        "ml_away": away_price,
        "source": "oddspapi",
    }


# ========== Matching ==========
def match_fixture_to_game(fixture, csv_games):
    """Match OddsPapi fixture to CSV row, handling UTC->local date offset."""
    home_abbrev = _team_to_abbrev(fixture.get("participant1Name", ""))
    away_abbrev = _team_to_abbrev(fixture.get("participant2Name", ""))
    utc_date = fixture.get("startTime", "")[:10]
    local_date = _utc_to_local_date(fixture.get("startTime", ""))

    home_csv = STD_TO_CSV.get(home_abbrev, home_abbrev.lower())
    away_csv = STD_TO_CSV.get(away_abbrev, away_abbrev.lower())

    # Try local date first (most common), then UTC date
    for game_date in [local_date, utc_date]:
        match = csv_games[
            (csv_games["date"] == game_date) &
            (csv_games["home"] == home_csv) &
            (csv_games["away"] == away_csv)
        ]
        if not match.empty:
            return match.index[0]
    return None


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset-budget", type=int, default=None,
                        help="Reset budget tracker (e.g., --reset-budget 14)")
    parser.add_argument("--fixtures-only", action="store_true",
                        help="Only fetch fixtures, skip everything else")
    parser.add_argument("--fetch-missing", type=int, default=0,
                        help="Fetch up to N new historical-odds (costs API budget)")
    parser.add_argument("--skip-fixtures", action="store_true",
                        help="Skip fixture fetching (use cached)")
    args = parser.parse_args()

    budget = Budget()
    if args.reset_budget is not None:
        budget.reset_to(args.reset_budget)

    # Step 1: Load/fetch fixtures
    if args.skip_fixtures:
        fixtures, _ = load_fixtures_cache()
        logger.info(f"Using cached fixtures: {len(fixtures)}")
    else:
        logger.info(f"\n=== Step 1: Fetch season fixtures (budget: {budget.remaining}) ===")
        fixtures = fetch_all_season_fixtures(budget)
        logger.info(f"Budget after fixtures: {budget.remaining}")

    if args.fixtures_only:
        return

    # Step 2: Auto-calibrate totals
    logger.info("\n=== Step 2: Auto-calibrate totals ===")
    if TOTALS_CALIBRATION_FILE.exists():
        _load_totals_base()
        logger.info("Using existing calibration")
    else:
        auto_calibrate_totals()

    # Step 3: Match fixtures to CSV
    logger.info("\n=== Step 3: Match fixtures to CSV games ===")
    df = pd.read_csv(CSV_PATH)
    s26 = df[df["season"] == 2026].copy()
    logger.info(f"Season 2026 in CSV: {len(s26)} games")

    fixture_matches = []
    for fid, fx in fixtures.items():
        csv_idx = match_fixture_to_game(fx, s26)
        if csv_idx is not None:
            has_spread = pd.notna(s26.loc[csv_idx, "spread"])
            fixture_matches.append({
                "fixture_id": fid,
                "csv_idx": csv_idx,
                "has_spread": has_spread,
                "cached": hist_cache_path(fid).exists(),
            })

    logger.info(f"Matched {len(fixture_matches)} / {len(fixtures)} fixtures to CSV")
    cached = sum(1 for m in fixture_matches if m["cached"])
    missing_spread = sum(1 for m in fixture_matches if not m["has_spread"])
    logger.info(f"  With cached odds: {cached}")
    logger.info(f"  Missing spread: {missing_spread}")

    # Step 4: Optionally fetch more historical odds
    if args.fetch_missing > 0:
        logger.info(f"\n=== Step 4: Fetch up to {args.fetch_missing} historical odds ===")
        # Prioritize: missing spread, then uncached
        fetch_queue = []
        seen = set()
        for m in fixture_matches:
            if not m["has_spread"] and not m["cached"] and m["fixture_id"] not in seen:
                fetch_queue.append(m)
                seen.add(m["fixture_id"])
        for m in fixture_matches:
            if not m["cached"] and m["fixture_id"] not in seen:
                fetch_queue.append(m)
                seen.add(m["fixture_id"])

        fetch_queue = fetch_queue[:args.fetch_missing]
        fetched = 0
        for m in fetch_queue:
            if budget.remaining < 1:
                break
            hist = fetch_historical_odds(m["fixture_id"], budget)
            if hist:
                fetched += 1
                m["cached"] = True
            time.sleep(RATE_LIMIT_SEC)
        logger.info(f"Fetched {fetched}/{len(fetch_queue)} (budget: {budget.remaining})")

    # Step 5: Extract odds from all cached hist files -> update CSV
    logger.info("\n=== Step 5: Extract odds and update CSV ===")
    updated_spread = 0
    updated_total = 0
    filled_spread = 0

    for m in fixture_matches:
        if not m["cached"]:
            continue

        fid = m["fixture_id"]
        fx = fixtures[fid]
        cache = hist_cache_path(fid)

        try:
            hist = json.loads(cache.read_text())
        except Exception:
            continue

        odds = extract_odds_from_historical(hist, fx)
        if not odds:
            continue

        csv_idx = m["csv_idx"]
        old_spread = df.loc[csv_idx, "spread"]

        # Update spread
        if odds["spread"] is not None:
            signed_spread = odds["spread"]
            abs_spread = abs(signed_spread)
            whos_favored = "home" if signed_spread <= 0 else "away"

            was_missing = pd.isna(old_spread)
            df.loc[csv_idx, "spread"] = abs_spread
            df.loc[csv_idx, "whos_favored"] = whos_favored

            if was_missing:
                filled_spread += 1
                home_name = fx.get("participant1Name", "")
                away_name = fx.get("participant2Name", "")
                logger.info(f"  FILLED: {odds['game_date_local']} {away_name} @ {home_name} "
                           f"spread={abs_spread} ({whos_favored})")
            else:
                updated_spread += 1

            # Update moneylines
            if odds.get("ml_home") and odds.get("ml_away"):
                ml_h, ml_a = odds["ml_home"], odds["ml_away"]
                am_h = round((ml_h - 1) * 100) if ml_h >= 2.0 else round(-100 / (ml_h - 1))
                am_a = round((ml_a - 1) * 100) if ml_a >= 2.0 else round(-100 / (ml_a - 1))
                df.loc[csv_idx, "moneyline_home"] = am_h
                df.loc[csv_idx, "moneyline_away"] = am_a

        # Update total (only if in sane NBA range)
        if odds.get("total") is not None and 180 <= odds["total"] <= 280:
            df.loc[csv_idx, "total"] = odds["total"]
            updated_total += 1
        elif odds.get("total") is not None:
            logger.debug(f"  Skipped unreasonable total: {odds['total']}")

    logger.info(f"\nSpread: {filled_spread} filled, {updated_spread} updated from Pinnacle")
    logger.info(f"Total: {updated_total} updated from calibrated markets")

    # Save
    df.to_csv(CSV_PATH, index=False)
    logger.info(f"Saved {CSV_PATH}")

    # Summary
    s26_final = df[df["season"] == 2026]
    with_spread = s26_final["spread"].notna().sum()
    with_total = s26_final["total"].notna().sum()
    logger.info(f"\n{'='*50}")
    logger.info(f"FINAL: {len(s26_final)} games")
    logger.info(f"  With spread: {with_spread} ({100*with_spread/len(s26_final):.1f}%)")
    logger.info(f"  With total:  {with_total} ({100*with_total/len(s26_final):.1f}%)")
    logger.info(f"  Budget remaining: {budget.remaining}")


if __name__ == "__main__":
    main()
