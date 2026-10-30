"""
scripts/harvest_player_props.py — Automated Real Player Prop Harvester & Settlement Engine

Passively farms real sportsbook closing player props (DraftKings, FanDuel, BetMGM, Caesars)
for all core NBA markets:
  - PTS  (Points)
  - REB  (Rebounds)
  - AST  (Assists)
  - 3PT  (3-Pointers Made)
  - PRA  (Points + Rebounds + Assists)
  - PR   (Points + Rebounds)
  - PA   (Points + Assists)
  - RA   (Rebounds + Assists)

CLV & Line Movement Tracking:
  - When lines are first captured (morning / opening): records opening_line.
  - When re-run closer to tip-off: updates closing_line and tracks line_movement (steam).
  - Once a game tips off: automatically locks closing_line (prevents live in-game drift).
  - Settles completed games against official ESPN box scores.

Usage:
    # Full automatic cycle: harvest today's lines + settle yesterday's games
    python scripts/harvest_player_props.py

    # Harvest props for a specific date (YYYY-MM-DD)
    python scripts/harvest_player_props.py --action harvest --date 2026-10-22

    # Settle player props against final box scores
    python scripts/harvest_player_props.py --action settle --date 2026-10-21
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from persistence_utils import atomic_json_save, safe_json_load

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("props_harvester")

HARVEST_DIR = PROJECT_ROOT / "data" / "harvested_odds"
HARVEST_DIR.mkdir(parents=True, exist_ok=True)

SETTLED_PROPS_PARQUET = HARVEST_DIR / "settled_player_props.parquet"
SETTLED_PROPS_CSV = HARVEST_DIR / "settled_player_props.csv"

# Core market mapping to Action Network customPickTypes
PROP_MARKET_MAP = {
    "PTS": "core_bet_type_27_points",
    "REB": "core_bet_type_23_rebounds",
    "AST": "core_bet_type_26_assists",
    "3PT": "core_bet_type_21_3fgm",
    "PRA": "core_bet_type_85_points_rebounds_assists",
    "PR": "core_bet_type_86_points_rebounds",
    "PA": "core_bet_type_87_points_assists",
    "RA": "core_bet_type_88_rebounds_assists",
}

# Action Network Sportsbook IDs in raw_markets:
#   "30" = Official Opening Line ("Open")
#   "68" = DraftKings
#   "69" = FanDuel
#   "15" = Market Consensus
#   "49" = Caesars
#   "71" = BetRivers
#   "75" = BetMGM (often contains alt/ladder lines when main lines exist elsewhere)
BOOK_ID_NAMES = {
    "30": "Open",
    "68": "DraftKings",
    "69": "FanDuel",
    "15": "Consensus",
    "49": "Caesars",
    "71": "BetRivers",
    "75": "BetMGM",
}
CLOSING_BOOK_PRIORITY = ["68", "69", "15", "49", "71", "30", "75"]

API_BASE = "https://api.actionnetwork.com/web/v2/scoreboard/nba/markets"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Origin": "https://www.actionnetwork.com",
    "Referer": "https://www.actionnetwork.com/nba/props",
}


def american_to_decimal(american_odds: int) -> float:
    """Convert American moneyline odds to European decimal odds."""
    if american_odds > 0:
        return round(1.0 + (american_odds / 100.0), 3)
    elif american_odds < 0:
        return round(1.0 + (100.0 / abs(american_odds)), 3)
    return 1.909


def normalize_player_name(raw_name: str) -> str:
    """Clean and normalize player names (stripping accents, apostrophes, periods, hyphens, and suffixes)."""
    if not raw_name:
        return ""
    import unicodedata
    s = unicodedata.normalize("NFKD", str(raw_name)).encode("ASCII", "ignore").decode("utf-8")
    clean = s.strip()
    for suffix in [" Jr.", " Sr.", " Jr", " Sr", " III", " II", " IV", " jr", " sr", " iii", " ii", " iv"]:
        if clean.endswith(suffix):
            clean = clean[:-len(suffix)]
    for ch in ("'", "’", "`", "."):
        clean = clean.replace(ch, "")
    clean = clean.replace("-", " ")
    return " ".join(clean.split())


def is_game_started(start_time_str: str) -> bool:
    """Check if the game has already tipped off in UTC."""
    if not start_time_str:
        return False
    try:
        clean_time = start_time_str.replace("Z", "+00:00")
        game_dt = datetime.fromisoformat(clean_time)
        return datetime.now(timezone.utc) >= game_dt
    except Exception:
        return False


# ===========================================================================
# 1. HARVESTING REAL CLOSING PLAYER PROPS
# ===========================================================================

def fetch_market_lines_http(pick_type: str, date_compact: str) -> Optional[Dict[str, Any]]:
    """Fetch raw market payload via HTTP requests."""
    url = f"{API_BASE}?customPickTypes={pick_type}&date={date_compact}"
    try:
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=12)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.debug(f"HTTP request failed for {pick_type}: {e}")
    return None


def fetch_market_lines_playwright(pick_type: str, date_compact: str) -> Optional[Dict[str, Any]]:
    """Headless Playwright fallback if HTTP request is blocked."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(user_agent=DEFAULT_HEADERS["User-Agent"], ignore_https_errors=True)
            page = context.new_page()
            url = f"{API_BASE}?customPickTypes={pick_type}&date={date_compact}"
            resp = page.goto(url, timeout=15000)
            if resp and resp.status == 200:
                content = page.inner_text("body")
                data = json.loads(content)
                browser.close()
                return data
            browser.close()
    except Exception as e:
        logger.error(f"Playwright fallback failed for {pick_type}: {e}")
    return None


def harvest_player_props_for_date(date_str: str) -> List[Dict[str, Any]]:
    """
    Harvest all 8 player prop markets for date_str (YYYY-MM-DD).
    Tracks opening lines, line movement, and locks closing lines once game starts.
    Saves snapshot to data/harvested_odds/player_props_YYYY-MM-DD.json.
    """
    date_compact = date_str.replace("-", "")
    logger.info(f"--- Harvesting Player Props for {date_str} ---")

    out_file = HARVEST_DIR / f"player_props_{date_str}.json"
    existing_lookup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    if out_file.exists():
        old_data = safe_json_load(out_file, default={})
        for p in old_data.get("props", []):
            k = (p.get("player"), p.get("market"), p.get("side"))
            existing_lookup[k] = p

    all_props_dict: Dict[Tuple[str, str, str], Dict[str, Any]] = existing_lookup.copy()
    now_iso = datetime.now(timezone.utc).isoformat()

    for market_code, pick_type in PROP_MARKET_MAP.items():
        data = fetch_market_lines_http(pick_type, date_compact)
        if not data:
            logger.info(f"  Attempting Playwright fallback for {market_code}...")
            data = fetch_market_lines_playwright(pick_type, date_compact)

        if not data:
            logger.warning(f"  Could not load lines for market {market_code}")
            continue

        raw_markets = data.get("markets", {})
        raw_players = data.get("players", [])
        raw_games = data.get("games", [])

        # Build player lookup
        player_lookup = {}
        for p in raw_players:
            pid = p.get("id")
            if pid is not None:
                player_lookup[pid] = {
                    "full_name": p.get("full_name", ""),
                    "abbr": p.get("abbr", ""),
                    "position": p.get("position", p.get("primary_position", "")),
                    "team_id": p.get("team_id"),
                }

        # Build game lookup
        game_lookup = {}
        for g in raw_games:
            gid = g.get("id")
            if gid is not None:
                home = g.get("home_team", {}).get("abbr", "")
                away = g.get("away_team", {}).get("abbr", "")
                game_lookup[gid] = {
                    "home_team": home,
                    "away_team": away,
                    "matchup": f"{away} @ {home}",
                    "start_time": g.get("start_time", ""),
                }

        market_count = 0
        # raw_markets is keyed by book_id ("30"=Open, "68"=DraftKings, "69"=FanDuel, "15"=Consensus, etc.)
        # Group all book quotes by (player_id, event_id, side) so we never overwrite Open with Alt books
        grouped_quotes: Dict[Tuple[Any, Any, str], Dict[str, Dict[str, Any]]] = {}
        for book_id_str, book_market_dict in raw_markets.items():
            if not isinstance(book_market_dict, dict):
                continue
            event_items = book_market_dict.get("event", {}).get(pick_type, [])
            for line_item in event_items:
                if line_item.get("is_alt_market"):
                    continue
                pid = line_item.get("player_id")
                event_id = line_item.get("event_id")
                side = str(line_item.get("side", "over")).upper()
                if pid is None or event_id is None:
                    continue
                grouped_quotes.setdefault((pid, event_id, side), {})[str(book_id_str)] = line_item

        for (pid, event_id, side), books_map in grouped_quotes.items():
            p_info = player_lookup.get(pid, {})
            g_info = game_lookup.get(event_id, {})
            player_name = p_info.get("full_name") or p_info.get("abbr", f"Player_{pid}")

            # 1. Select best Current / Closing quote according to sharp book priority
            curr_item = None
            selected_book_id_str = None
            for b_id in CLOSING_BOOK_PRIORITY:
                if b_id in books_map:
                    curr_item = books_map[b_id]
                    selected_book_id_str = b_id
                    break
            if not curr_item:
                selected_book_id_str, curr_item = next(iter(books_map.items()))

            line_val = float(curr_item.get("value", 0))
            odds_american = int(curr_item.get("odds", -110))
            odds_decimal = american_to_decimal(odds_american)
            book_id_int = int(curr_item.get("book_id") or selected_book_id_str)
            book_name = BOOK_ID_NAMES.get(str(selected_book_id_str), f"Book_{selected_book_id_str}")

            # 2. Extract official Opening quote from Book "30" ("Open") when available
            open_item = books_map.get("30")
            api_open_line = float(open_item.get("value", line_val)) if open_item else None
            api_open_odds_am = int(open_item.get("odds", odds_american)) if open_item else None
            api_open_odds_dec = american_to_decimal(api_open_odds_am) if api_open_odds_am is not None else None

            key = (player_name, market_code, side)
            start_time_str = g_info.get("start_time", "")
            game_started = is_game_started(start_time_str)

            if key in existing_lookup:
                existing = existing_lookup[key]
                # Determine true opening line (prefer official Book 30 "Open" if existing had 0 movement or no Book 30 backfill)
                open_line = api_open_line if api_open_line is not None else existing.get("opening_line", line_val)
                open_odds_am = api_open_odds_am if api_open_odds_am is not None else existing.get("opening_odds_american", odds_american)
                open_odds_dec = api_open_odds_dec if api_open_odds_dec is not None else existing.get("opening_odds_decimal", odds_decimal)

                # If game already started and was previously locked, preserve locked closing line (unless existing came from Book 75 alt)
                if existing.get("is_closed") and str(existing.get("book_id")) != "75":
                    locked_close = float(existing.get("closing_line", existing.get("line", line_val)))
                    locked_odds_am = int(existing.get("odds_american", odds_american))
                    locked_odds_dec = float(existing.get("odds_decimal", odds_decimal))
                    existing["opening_line"] = open_line
                    existing["opening_odds_american"] = open_odds_am
                    existing["opening_odds_decimal"] = open_odds_dec
                    existing["line_movement"] = round(locked_close - open_line, 2)
                    existing["odds_movement"] = round(locked_odds_dec - open_odds_dec, 3)
                    all_props_dict[key] = existing
                    market_count += 1
                    continue

                line_mov = round(line_val - open_line, 2)
                odds_mov = round(odds_decimal - open_odds_dec, 3)

                prop_record = {
                    **existing,
                    "line": line_val,
                    "closing_line": line_val,
                    "line_movement": line_mov,
                    "opening_line": open_line,
                    "odds_american": odds_american,
                    "odds_decimal": odds_decimal,
                    "opening_odds_american": open_odds_am,
                    "opening_odds_decimal": open_odds_dec,
                    "odds_movement": odds_mov,
                    "book_id": book_id_int,
                    "book_name": book_name,
                    "is_closed": game_started,
                    "last_updated_at": now_iso,
                }
                if game_started and not existing.get("closed_at"):
                    prop_record["closed_at"] = now_iso
            else:
                # Initial line capture — use Book 30 ("Open") as opening_line if available
                open_line = api_open_line if api_open_line is not None else line_val
                open_odds_am = api_open_odds_am if api_open_odds_am is not None else odds_american
                open_odds_dec = api_open_odds_dec if api_open_odds_dec is not None else odds_decimal

                line_mov = round(line_val - open_line, 2)
                odds_mov = round(odds_decimal - open_odds_dec, 3)

                prop_record = {
                    "date": date_str,
                    "event_id": event_id,
                    "player_id": pid,
                    "player": player_name,
                    "position": p_info.get("position", ""),
                    "home_team": g_info.get("home_team", ""),
                    "away_team": g_info.get("away_team", ""),
                    "matchup": g_info.get("matchup", ""),
                    "start_time": start_time_str,
                    "market": market_code,
                    "line": line_val,
                    "opening_line": open_line,
                    "closing_line": line_val,
                    "line_movement": line_mov,
                    "side": side,
                    "odds_american": odds_american,
                    "odds_decimal": odds_decimal,
                    "opening_odds_american": open_odds_am,
                    "opening_odds_decimal": open_odds_dec,
                    "odds_movement": odds_mov,
                    "book_id": book_id_int,
                    "book_name": book_name,
                    "first_seen_at": now_iso,
                    "last_updated_at": now_iso,
                    "is_closed": game_started,
                }

            all_props_dict[key] = prop_record
            market_count += 1

        logger.info(f"  {market_code:5s}: processed {market_count} lines")

    all_props = list(all_props_dict.values())
    closed_count = sum(1 for p in all_props if p.get("is_closed"))
    movement_count = sum(1 for p in all_props if abs(p.get("line_movement", 0.0)) > 0.01)

    atomic_json_save(out_file, {"date": date_str, "count": len(all_props), "props": all_props})
    logger.info(
        f"✅ Saved {len(all_props):,} props to {out_file} "
        f"({closed_count} closed/locked, {movement_count} lines moved since open)"
    )
    return all_props


# ===========================================================================
# 2. SETTLEMENT ENGINE: REAL BOXSCORES & OUTCOME EVALUATION
# ===========================================================================

def fetch_espn_player_boxscores(date_str: str) -> Dict[str, Dict[str, float]]:
    """
    Fetch all player box scores for date_str from ESPN.
    Returns: {normalized_player_name: {'PTS': x, 'REB': y, 'AST': z, '3PT': k, 'PRA': w, ...}}
    """
    espn_date = date_str.replace("-", "")
    player_stats: Dict[str, Dict[str, float]] = {}

    try:
        resp = requests.get(ESPN_SCOREBOARD, params={"dates": espn_date}, timeout=15)
        if resp.status_code != 200:
            return {}
        events = resp.json().get("events", [])
    except Exception as e:
        logger.error(f"Failed to fetch ESPN scoreboard for settlement: {e}")
        return {}

    for ev in events:
        eid = ev.get("id")
        if not eid:
            continue

        # Strictly enforce that games must be FINAL/completed before taking boxscores
        status = ev.get("status", {}).get("type", {})
        if not status:
            comp_status = ev.get("competitions", [{}])[0].get("status", {}).get("type", {})
            status = comp_status or {}
        state = str(status.get("state", "")).lower()
        completed = status.get("completed", False) is True
        desc = str(status.get("description", "")).upper()
        is_final = (state == "post" and completed) or ("FINAL" in desc and state == "post")
        if not is_final:
            continue

        try:
            r_sum = requests.get(ESPN_SUMMARY, params={"event": eid}, timeout=10)
            if r_sum.status_code != 200:
                continue
            box = r_sum.json().get("boxscore", {})
            team_players = box.get("players", [])

            for team_box in team_players:
                stats_group = team_box.get("statistics", [{}])[0]
                labels = stats_group.get("labels", [])
                athletes = stats_group.get("athletes", [])

                # Map label indices
                label_idx = {lbl: i for i, lbl in enumerate(labels)}
                pts_idx = label_idx.get("PTS")
                reb_idx = label_idx.get("REB")
                ast_idx = label_idx.get("AST")
                t3_idx = label_idx.get("3PT")
                min_idx = label_idx.get("MIN")

                for a in athletes:
                    name = a.get("athlete", {}).get("displayName", "")
                    raw_stats = a.get("stats", [])
                    if not name or not raw_stats:
                        continue

                    def get_stat(idx: Optional[int]) -> float:
                        if idx is not None and idx < len(raw_stats):
                            val_str = str(raw_stats[idx]).strip()
                            if "-" in val_str:  # e.g. "2-5" (makes-attempts)
                                return float(val_str.split("-")[0])
                            try:
                                return float(val_str)
                            except ValueError:
                                return 0.0
                        return 0.0

                    pts = get_stat(pts_idx)
                    reb = get_stat(reb_idx)
                    ast = get_stat(ast_idx)
                    t3 = get_stat(t3_idx)
                    minutes = get_stat(min_idx)

                    norm_name = normalize_player_name(name).lower()
                    player_stats[norm_name] = {
                        "PTS": pts,
                        "REB": reb,
                        "AST": ast,
                        "3PT": t3,
                        "PRA": pts + reb + ast,
                        "PR": pts + reb,
                        "PA": pts + ast,
                        "RA": reb + ast,
                        "MIN": minutes,
                    }
        except Exception as e:
            logger.debug(f"Error parsing event {eid} boxscore: {e}")

    logger.info(f"Loaded ESPN box scores for {len(player_stats)} players on {date_str}")
    return player_stats


def settle_player_props_for_date(date_str: str) -> Optional[pd.DataFrame]:
    """
    Settle saved player props against real ESPN box scores for date_str.
    Uses closing_line as the final evaluated line and records line_movement.
    Appends outcomes to data/harvested_odds/settled_player_props.parquet and .csv.
    """
    props_file = HARVEST_DIR / f"player_props_{date_str}.json"
    if not props_file.exists():
        logger.warning(f"No harvested player props found at {props_file}. Harvest first!")
        return None

    raw_payload = safe_json_load(props_file, default={})
    prop_list = raw_payload.get("props", [])
    if not prop_list:
        logger.warning(f"Empty props payload in {props_file}")
        return None

    boxscores = fetch_espn_player_boxscores(date_str)
    if not boxscores:
        logger.warning(f"No ESPN box scores available yet for {date_str} (games still in progress or unscheduled)")
        return None

    settled_rows: List[Dict[str, Any]] = []

    for prop in prop_list:
        player_name = prop.get("player", "")
        norm_name = normalize_player_name(player_name).lower()

        # Find in boxscores (exact or partial match)
        stats = boxscores.get(norm_name)
        if not stats:
            for b_name, b_stats in boxscores.items():
                if b_name in norm_name or norm_name in b_name:
                    stats = b_stats
                    break

        if not stats:
            continue

        market = prop.get("market", "PTS")
        closing_line = float(prop.get("closing_line", prop.get("line", 0)))
        opening_line = float(prop.get("opening_line", closing_line))
        line_movement = float(prop.get("line_movement", 0.0))
        side = str(prop.get("side", "OVER")).upper()
        actual_val = stats.get(market)

        if actual_val is None:
            continue

        # Evaluate outcome against the true closing line
        margin = round(actual_val - closing_line, 2)
        if actual_val > closing_line:
            over_won = True
            result = "WIN" if side == "OVER" else "LOSS"
        elif actual_val < closing_line:
            over_won = False
            result = "WIN" if side == "UNDER" else "LOSS"
        else:
            over_won = None
            result = "PUSH"

        settled_record = {
            **prop,
            "line": closing_line,
            "closing_line": closing_line,
            "opening_line": opening_line,
            "line_movement": line_movement,
            "actual_value": actual_val,
            "margin": margin,
            "result": result,
            "over_hit": over_won,
            "minutes_played": stats.get("MIN", 0),
            "settled_at": datetime.now(timezone.utc).isoformat(),
        }
        settled_rows.append(settled_record)

    if not settled_rows:
        logger.warning("No props could be matched to box scores.")
        return None

    new_df = pd.DataFrame(settled_rows)
    logger.info(f"Successfully settled {len(new_df):,} props for {date_str}")

    # Append to Master Parquet and CSV
    if SETTLED_PROPS_PARQUET.exists():
        try:
            existing_df = pd.read_parquet(SETTLED_PROPS_PARQUET)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.drop_duplicates(
                subset=["date", "player", "market", "side"],
                keep="last",
                inplace=True,
            )
        except Exception:
            combined_df = new_df
    else:
        combined_df = new_df

    combined_df.to_parquet(SETTLED_PROPS_PARQUET, index=False)
    combined_df.to_csv(SETTLED_PROPS_CSV, index=False)
    logger.info(f"✅ Master settled props updated: {len(combined_df):,} total records ({SETTLED_PROPS_PARQUET})")

    return new_df


# ===========================================================================
# 3. CLI ENTRY POINT
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Real NBA Player Prop Harvester & Settlement Engine")
    parser.add_argument("--action", choices=["auto", "harvest", "settle"], default="auto")
    parser.add_argument("--date", type=str, default=None, help="Target date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    today_str = datetime.now().strftime("%Y-%m-%d")
    yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    target_date = args.date or today_str

    if args.action == "harvest":
        harvest_player_props_for_date(target_date)
    elif args.action == "settle":
        settle_player_props_for_date(target_date)
    else:
        # Full automatic cycle: harvest today's lines + settle yesterday's finished games
        logger.info("=== RUNNING AUTOMATIC PROPS HARVEST & SETTLEMENT CYCLE ===")
        harvest_player_props_for_date(today_str)
        settle_player_props_for_date(yesterday_str)
        logger.info("=== AUTOMATIC CYCLE COMPLETED ===")


if __name__ == "__main__":
    main()
