"""
scripts/harvest_closing_odds.py — Passive Harvester for NBA Closing Odds & Settled Outcomes

Farms daily closing game odds (spreads, totals, moneylines) for free:
  1. Primary: The Odds API (multi-bookmaker consensus: DraftKings, FanDuel, BetMGM)
  2. Fallback & Score Engine: ESPN Public API (100% free, unauthenticated DraftKings lines & final scores)
  3. Settler: Matches closing lines against final scores, records ATS and O/U winners,
     and evaluates game prediction model performance.

Usage:
    # Run automatic harvest + settlement for today and recent games
    python scripts/harvest_closing_odds.py

    # Harvest closing lines for a specific date (YYYY-MM-DD)
    python scripts/harvest_closing_odds.py --action harvest --date 2026-10-22

    # Settle finished games and compute model PnL / cover stats
    python scripts/harvest_closing_odds.py --action settle --date 2026-10-22
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

# Load environment variables (.env)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from persistence_utils import atomic_json_save, safe_json_load

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("odds_harvester")

HARVEST_DIR = PROJECT_ROOT / "data" / "harvested_odds"
HARVEST_DIR.mkdir(parents=True, exist_ok=True)

SETTLED_CSV = HARVEST_DIR / "settled_games.csv"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"

# Standard NBA team abbreviations
TEAM_MAP = {
    "atlanta hawks": "ATL", "hawks": "ATL", "atl": "ATL",
    "boston celtics": "BOS", "celtics": "BOS", "bos": "BOS",
    "brooklyn nets": "BKN", "nets": "BKN", "bkn": "BKN",
    "charlotte hornets": "CHA", "hornets": "CHA", "cha": "CHA",
    "chicago bulls": "CHI", "bulls": "CHI", "chi": "CHI",
    "cleveland cavaliers": "CLE", "cavaliers": "CLE", "cavs": "CLE", "cle": "CLE",
    "dallas mavericks": "DAL", "mavericks": "DAL", "mavs": "DAL", "dal": "DAL",
    "denver nuggets": "DEN", "nuggets": "DEN", "den": "DEN",
    "detroit pistons": "DET", "pistons": "DET", "det": "DET",
    "golden state warriors": "GSW", "warriors": "GSW", "gsw": "GSW", "gs": "GSW",
    "houston rockets": "HOU", "rockets": "HOU", "hou": "HOU",
    "indiana pacers": "IND", "pacers": "IND", "ind": "IND",
    "los angeles clippers": "LAC", "la clippers": "LAC", "clippers": "LAC", "lac": "LAC",
    "los angeles lakers": "LAL", "la lakers": "LAL", "lakers": "LAL", "lal": "LAL",
    "memphis grizzlies": "MEM", "grizzlies": "MEM", "mem": "MEM",
    "miami heat": "MIA", "heat": "MIA", "mia": "MIA",
    "milwaukee bucks": "MIL", "bucks": "MIL", "mil": "MIL",
    "minnesota timberwolves": "MIN", "timberwolves": "MIN", "wolves": "MIN", "min": "MIN",
    "new orleans pelicans": "NOP", "pelicans": "NOP", "no": "NOP", "nop": "NOP",
    "new york knicks": "NYK", "knicks": "NYK", "ny": "NYK", "nyk": "NYK",
    "oklahoma city thunder": "OKC", "thunder": "OKC", "okc": "OKC",
    "orlando magic": "ORL", "magic": "ORL", "orl": "ORL",
    "philadelphia 76ers": "PHI", "76ers": "PHI", "sixers": "PHI", "phi": "PHI",
    "phoenix suns": "PHX", "suns": "PHX", "phx": "PHX",
    "portland trail blazers": "POR", "trail blazers": "POR", "blazers": "POR", "por": "POR",
    "sacramento kings": "SAC", "kings": "SAC", "sac": "SAC",
    "san antonio spurs": "SAS", "spurs": "SAS", "sa": "SAS", "sas": "SAS",
    "toronto raptors": "TOR", "raptors": "TOR", "tor": "TOR",
    "utah jazz": "UTA", "jazz": "UTA", "utah": "UTA", "uta": "UTA",
    "washington wizards": "WAS", "wizards": "WAS", "wsh": "WAS", "was": "WAS",
}


def normalize_team(team_str: str) -> str:
    """Normalize team name or abbreviation to standard 3-letter uppercase."""
    if not team_str:
        return ""
    clean = team_str.strip().lower()
    if clean in TEAM_MAP:
        return TEAM_MAP[clean]
    for k, v in TEAM_MAP.items():
        if k in clean or clean in k:
            return v
    return team_str.strip().upper()[:3]


# ===========================================================================
# 1. THE ODDS API INGESTION
# ===========================================================================

def harvest_from_the_odds_api(api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch game-level odds from The Odds API across all bookmakers.
    Consumes 1 request from monthly quota.
    """
    key = api_key or os.getenv("ODDS_API_KEY")
    if not key:
        logger.info("The Odds API key not found in environment; skipping Odds API source.")
        return []

    params = {
        "apiKey": key,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "decimal",
    }

    try:
        resp = requests.get(ODDS_API_URL, params=params, timeout=15)
        rem = resp.headers.get("x-requests-remaining", "?")
        used = resp.headers.get("x-requests-used", "?")
        logger.info(f"The Odds API: status={resp.status_code}, quota used={used}, remaining={rem}")

        if resp.status_code != 200:
            logger.warning(f"The Odds API returned status {resp.status_code}: {resp.text[:200]}")
            return []

        events = resp.json()
        games = []
        preferred_books = ["draftkings", "fanduel", "betmgm", "bovada", "caesars"]

        for ev in events:
            home = normalize_team(ev.get("home_team", ""))
            away = normalize_team(ev.get("away_team", ""))
            commence = ev.get("commence_time", "")
            if not home or not away:
                continue

            game_date = commence[:10] if commence else datetime.now(timezone.utc).strftime("%Y-%m-%d")

            # Extract best / consensus odds
            spread_home = None
            spread_away = None
            total_line = None
            ml_home = None
            ml_away = None
            selected_book = "unknown"

            bookmakers = ev.get("bookmakers", [])
            # Sort by bookmaker preference
            bookmakers.sort(key=lambda b: preferred_books.index(b.get("key")) if b.get("key") in preferred_books else 99)

            for b in bookmakers:
                bk_name = b.get("key")
                for m in b.get("markets", []):
                    m_key = m.get("key")
                    outcomes = m.get("outcomes", [])
                    if m_key == "spreads" and spread_home is None:
                        for out in outcomes:
                            team_norm = normalize_team(out.get("name", ""))
                            if team_norm == home:
                                spread_home = out.get("point")
                            elif team_norm == away:
                                spread_away = out.get("point")
                        if spread_home is not None:
                            selected_book = bk_name

                    elif m_key == "totals" and total_line is None:
                        for out in outcomes:
                            if out.get("name", "").upper() == "OVER":
                                total_line = out.get("point")

                    elif m_key == "h2h" and ml_home is None:
                        for out in outcomes:
                            team_norm = normalize_team(out.get("name", ""))
                            if team_norm == home:
                                ml_home = out.get("price")
                            elif team_norm == away:
                                ml_away = out.get("price")

            games.append({
                "source": "the_odds_api",
                "event_id": ev.get("id"),
                "date": game_date,
                "commence_time": commence,
                "home_team": home,
                "away_team": away,
                "matchup": f"{away} @ {home}",
                "spread_home": spread_home,
                "spread_away": spread_away,
                "total": total_line,
                "ml_home_decimal": ml_home,
                "ml_away_decimal": ml_away,
                "bookmaker": selected_book,
                "harvested_at": datetime.now(timezone.utc).isoformat(),
            })

        logger.info(f"Harvested {len(games)} games from The Odds API")
        return games
    except Exception as e:
        logger.error(f"Error fetching from The Odds API: {e}")
        return []


# ===========================================================================
# 2. ESPN PUBLIC API INGESTION (FREE FALLBACK + SCORES)
# ===========================================================================

def harvest_from_espn(date_str: str) -> List[Dict[str, Any]]:
    """
    Harvest upcoming or in-progress game lines from ESPN scoreboard + pickcenter.
    100% Free, no authentication required.
    """
    espn_date = date_str.replace("-", "")
    try:
        resp = requests.get(ESPN_SCOREBOARD, params={"dates": espn_date}, timeout=15)
        if resp.status_code != 200:
            logger.warning(f"ESPN scoreboard error {resp.status_code} for date {date_str}")
            return []
        data = resp.json()
    except Exception as e:
        logger.error(f"ESPN scoreboard request failed: {e}")
        return []

    events = data.get("events", [])
    games = []

    for ev in events:
        event_id = ev.get("id")
        comp = ev.get("competitions", [{}])[0]
        status_info = comp.get("status", {}).get("type", {})
        status_state = status_info.get("state", "pre")  # "pre", "in", "post"
        status_desc = status_info.get("description", "Scheduled")

        competitors = comp.get("competitors", [])
        home_team = None
        away_team = None
        home_score = None
        away_score = None

        for c in competitors:
            abbrev = normalize_team(c.get("team", {}).get("abbreviation", ""))
            score_val = c.get("score")
            score_int = int(score_val) if score_val is not None and str(score_val).isdigit() else None
            if c.get("homeAway") == "home":
                home_team = abbrev
                home_score = score_int
            else:
                away_team = abbrev
                away_score = score_int

        if not home_team or not away_team:
            continue

        # Look up odds in summary pickcenter
        spread_home = None
        total_line = None
        provider = "DraftKings"

        try:
            r_sum = requests.get(ESPN_SUMMARY, params={"event": event_id}, timeout=10)
            if r_sum.status_code == 200:
                sum_data = r_sum.json()
                pickcenter = sum_data.get("pickcenter", [])
                if pickcenter:
                    pc = pickcenter[0]
                    provider = pc.get("provider", {}).get("name", "DraftKings")
                    total_val = pc.get("overUnder")
                    spread_val = pc.get("spread")
                    if total_val is not None:
                        total_line = float(total_val)
                    if spread_val is not None:
                        spread_home = float(spread_val)
        except Exception as ex:
            logger.debug(f"Could not load ESPN summary for event {event_id}: {ex}")

        games.append({
            "source": "espn",
            "event_id": event_id,
            "date": date_str,
            "commence_time": ev.get("date", ""),
            "home_team": home_team,
            "away_team": away_team,
            "matchup": f"{away_team} @ {home_team}",
            "spread_home": spread_home,
            "spread_away": -spread_home if spread_home is not None else None,
            "total": total_line,
            "home_score": home_score,
            "away_score": away_score,
            "status": status_desc,
            "state": status_state,
            "bookmaker": provider,
            "harvested_at": datetime.now(timezone.utc).isoformat(),
        })

    logger.info(f"Harvested {len(games)} games from ESPN for {date_str}")
    return games


# ===========================================================================
# 3. HARVESTING MERGE & PERSISTENCE
# ===========================================================================

def harvest_daily_odds(target_date: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Harvest odds for target_date using all available sources (Odds API + ESPN).
    Merges data and atomically writes to data/harvested_odds/odds_{date}.json.
    """
    if not target_date:
        target_date = datetime.now().strftime("%Y-%m-%d")

    logger.info(f"--- Starting Odds Harvest for {target_date} ---")

    # Fetch from sources
    odds_api_games = harvest_from_the_odds_api()
    espn_games = harvest_from_espn(target_date)

    # Index by (home_team, away_team)
    merged_games: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # 1. Fill from ESPN (gives schedule, status, and DraftKings lines)
    for g in espn_games:
        key = (g["home_team"], g["away_team"])
        merged_games[key] = g

    # 2. Enrich/overlay with The Odds API (multi-bookmaker consensus lines)
    for g in odds_api_games:
        if g.get("date") != target_date:
            continue
        key = (g["home_team"], g["away_team"])
        if key in merged_games:
            existing = merged_games[key]
            # If existing lacks lines or Odds API provides consensus, update lines
            if g.get("spread_home") is not None:
                existing["spread_home"] = g["spread_home"]
                existing["spread_away"] = g["spread_away"]
                existing["spread_book"] = g["bookmaker"]
            if g.get("total") is not None:
                existing["total"] = g["total"]
                existing["total_book"] = g["bookmaker"]
            if g.get("ml_home_decimal") is not None:
                existing["ml_home"] = g["ml_home_decimal"]
                existing["ml_away"] = g["ml_away_decimal"]
        else:
            merged_games[key] = g

    result_list = list(merged_games.values())

    # Atomically save to target date cache
    out_file = HARVEST_DIR / f"odds_{target_date}.json"
    data_payload = {
        "date": target_date,
        "harvested_at": datetime.now(timezone.utc).isoformat(),
        "game_count": len(result_list),
        "games": result_list,
    }
    atomic_json_save(out_file, data_payload)
    logger.info(f"Saved {len(result_list)} games to {out_file}")

    return result_list


# ===========================================================================
# 4. GAME SETTLEMENT & HISTORICAL TRACKING
# ===========================================================================

def settle_games_for_date(date_str: str) -> List[Dict[str, Any]]:
    """
    Checks finished games on date_str, pairs them with saved closing lines,
    evaluates ATS and O/U results, and appends to data/harvested_odds/settled_games.csv.
    """
    logger.info(f"--- Settling Finished Games for {date_str} ---")

    # Load closing lines file if present
    odds_file = HARVEST_DIR / f"odds_{date_str}.json"
    saved_odds_lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if odds_file.exists():
        odds_data = safe_json_load(odds_file, default={})
        for g in odds_data.get("games", []):
            k = (g.get("home_team"), g.get("away_team"))
            saved_odds_lookup[k] = g

    # Fetch live final scores from ESPN
    espn_games = harvest_from_espn(date_str)
    settled_records = []

    # Optional: load model to compute prediction on settled games
    model_pred_func = None
    try:
        from game_prediction import predict_game
        model_pred_func = predict_game
    except Exception as e:
        logger.debug(f"game_prediction module not loaded for settlement: {e}")

    for g in espn_games:
        if g.get("state") != "post" or g.get("home_score") is None or g.get("away_score") is None:
            continue

        home = g["home_team"]
        away = g["away_team"]
        h_score = g["home_score"]
        a_score = g["away_score"]
        actual_margin = h_score - a_score  # positive = home win
        actual_total = h_score + a_score

        # Get closing spread & total
        saved = saved_odds_lookup.get((home, away), {})
        spread = saved.get("spread_home") if saved.get("spread_home") is not None else g.get("spread_home")
        total = saved.get("total") if saved.get("total") is not None else g.get("total")

        if spread is None or total is None:
            logger.info(f"Missing spread/total for finished game {away} @ {home}; skipping settlement.")
            continue

        # ATS Result:
        # spread is from home perspective (e.g. -3.5 means home must win by > 3.5)
        # Margin > -spread means home covered (e.g. margin=4 > -(-3.5) is false, wait:)
        # In sports betting, Home Spread = -3.5. Home covers if margin > 3.5 (i.e. margin + spread > 0).
        diff_spread = actual_margin + spread
        if diff_spread > 0:
            ats_winner = "HOME"
        elif diff_spread < 0:
            ats_winner = "AWAY"
        else:
            ats_winner = "PUSH"

        # O/U Result
        diff_total = actual_total - total
        if diff_total > 0:
            ou_winner = "OVER"
        elif diff_total < 0:
            ou_winner = "UNDER"
        else:
            ou_winner = "PUSH"

        record = {
            "date": date_str,
            "matchup": f"{away} @ {home}",
            "home_team": home,
            "away_team": away,
            "home_score": h_score,
            "away_score": a_score,
            "actual_margin": actual_margin,
            "actual_total": actual_total,
            "closing_spread": spread,
            "closing_total": total,
            "ats_winner": ats_winner,
            "ou_winner": ou_winner,
            "settled_at": datetime.now(timezone.utc).isoformat(),
        }

        # If model available, run prediction and check accuracy
        if model_pred_func:
            try:
                pred = model_pred_func(
                    home=home,
                    away=away,
                    vegas_spread=-spread if spread < 0 else spread,  # match model conventions
                    vegas_total=total,
                )
                if pred:
                    record["pred_margin"] = pred.pred_margin
                    record["pred_total"] = pred.pred_total
                    record["spread_edge"] = pred.spread_edge
                    record["total_edge"] = pred.total_edge
                    record["model_ats_pick"] = pred.ats_pick
                    record["model_ou_pick"] = pred.ou_pick
                    record["model_ats_correct"] = int((pred.ats_pick.startswith("HOME") and ats_winner == "HOME") or
                                                      (pred.ats_pick.startswith("AWAY") and ats_winner == "AWAY"))
                    record["model_ou_correct"] = int((pred.ou_pick == "OVER" and ou_winner == "OVER") or
                                                     (pred.ou_pick == "UNDER" and ou_winner == "UNDER"))
            except Exception as e:
                logger.debug(f"Model prediction failed for settlement: {e}")

        settled_records.append(record)

    if not settled_records:
        logger.info(f"No newly settled games found for {date_str}.")
        return []

    # Append to settled_games.csv atomically
    df_new = pd.DataFrame(settled_records)
    if SETTLED_CSV.exists():
        try:
            df_existing = pd.read_csv(SETTLED_CSV)
            # Deduplicate by (date, matchup)
            combined = pd.concat([df_existing, df_new], ignore_index=True)
            combined.drop_duplicates(subset=["date", "matchup"], keep="last", inplace=True)
        except Exception:
            combined = df_new
    else:
        combined = df_new

    combined.to_csv(SETTLED_CSV, index=False)
    logger.info(f"Appended {len(settled_records)} settled games to {SETTLED_CSV} (Total: {len(combined)} records)")
    return settled_records


# ===========================================================================
# 5. CLI & AUTOMATION ENTRY POINT
# ===========================================================================

def run_auto_harvest():
    """
    Standard automated daily routine:
    1. Harvests closing odds for today and tomorrow.
    2. Settles games for today and yesterday.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info(f"=== Starting Auto Odds Harvest ({today}) ===")

    # Harvest today and tomorrow
    harvest_daily_odds(today)
    harvest_daily_odds(tomorrow)

    # Settle yesterday and today
    settle_games_for_date(yesterday)
    settle_games_for_date(today)

    logger.info("=== Auto Harvest Routine Completed Successfully ===")


def main():
    parser = argparse.ArgumentParser(description="NBA Passive Odds Harvester & Settler")
    parser.add_argument("--action", choices=["auto", "harvest", "settle"], default="auto",
                        help="Action to perform: 'auto' (harvest & settle), 'harvest', or 'settle'")
    parser.add_argument("--date", type=str, default=None,
                        help="Target date in YYYY-MM-DD format (defaults to current date)")
    args = parser.parse_args()

    target_date = args.date or datetime.now().strftime("%Y-%m-%d")

    if args.action == "auto":
        run_auto_harvest()
    elif args.action == "harvest":
        harvest_daily_odds(target_date)
    elif args.action == "settle":
        settle_games_for_date(target_date)


if __name__ == "__main__":
    main()
