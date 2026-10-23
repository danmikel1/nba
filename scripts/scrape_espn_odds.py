"""
Scrape NBA game odds from ESPN's public API for the 2025-26 season gap.

The Kaggle dataset covers through June 2025. This fills Oct 2025 - Feb 2026.
ESPN JSON API endpoints (public, no auth):
  - Scoreboard: site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=YYYYMMDD
  - Game Summary: site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event=GAMEID

The summary endpoint has a 'pickcenter' field with DraftKings spreads/totals.
"""

import json
import time
import requests
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ODDS_CACHE_DIR = Path("data/odds_cache")
ODDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"

# ESPN team abbreviation → our standard abbreviation
ESPN_TEAM_MAP = {
    "ATL": "ATL", "BKN": "BKN", "BOS": "BOS", "CHA": "CHA", "CHI": "CHI",
    "CLE": "CLE", "DAL": "DAL", "DEN": "DEN", "DET": "DET", "GS": "GSW",
    "GSW": "GSW", "HOU": "HOU", "IND": "IND", "LAC": "LAC", "LAL": "LAL",
    "MEM": "MEM", "MIA": "MIA", "MIL": "MIL", "MIN": "MIN", "NO": "NOP",
    "NOP": "NOP", "NY": "NYK", "NYK": "NYK", "OKC": "OKC", "ORL": "ORL",
    "PHI": "PHI", "PHX": "PHX", "POR": "POR", "SA": "SAS", "SAS": "SAS",
    "SAC": "SAC", "TOR": "TOR", "UTAH": "UTA", "UTA": "UTA", "WSH": "WAS",
    "WAS": "WAS",
}

TEAM_NAMES = {
    "ATL": "Atlanta Hawks", "BKN": "Brooklyn Nets", "BOS": "Boston Celtics",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}


def get_games_for_date(date_str: str) -> list[dict]:
    """Get all finished NBA games from ESPN scoreboard for YYYY-MM-DD."""
    espn_date = date_str.replace("-", "")
    try:
        resp = requests.get(ESPN_SCOREBOARD, params={"dates": espn_date}, timeout=15)
        if resp.status_code != 200:
            return []
        data = resp.json()
    except Exception as e:
        logger.warning(f"Scoreboard fetch failed for {date_str}: {e}")
        return []

    games = []
    for event in data.get("events", []):
        comp = event.get("competitions", [{}])[0]
        status = comp.get("status", {}).get("type", {}).get("name", "")
        if status != "STATUS_FINAL":
            continue

        competitors = comp.get("competitors", [])
        home_team = away_team = None
        for c in competitors:
            abbrev = c.get("team", {}).get("abbreviation", "")
            if c.get("homeAway") == "home":
                home_team = abbrev
            else:
                away_team = abbrev

        games.append({
            "event_id": event["id"],
            "home_espn": home_team,
            "away_espn": away_team,
            "name": event.get("shortName", ""),
        })

    return games


def get_odds_for_game(event_id: str) -> dict | None:
    """Get spread/total from ESPN pickcenter for a specific game."""
    try:
        resp = requests.get(ESPN_SUMMARY, params={"event": event_id}, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception as e:
        logger.warning(f"Summary fetch failed for {event_id}: {e}")
        return None

    pickcenter = data.get("pickcenter", [])
    if not pickcenter:
        return None

    pc = pickcenter[0]
    spread = pc.get("spread")
    total = pc.get("overUnder")
    details = pc.get("details", "")
    provider = pc.get("provider", {}).get("name", "espn")

    # Parse details to get spread direction: "HOU -14.5" means HOU favored
    # spread is always positive in ESPN pickcenter; details tells us who
    home_spread = None
    if spread is not None:
        # Get home/away from awayTeamOdds/homeTeamOdds
        home_odds = pc.get("homeTeamOdds", {})
        away_odds = pc.get("awayTeamOdds", {})
        home_fav = home_odds.get("favorite", False)
        away_fav = away_odds.get("favorite", False)

        if home_fav:
            home_spread = -abs(spread)
        elif away_fav:
            home_spread = abs(spread)
        else:
            home_spread = 0.0

    return {
        "spread": home_spread,
        "total": total,
        "provider": provider,
        "details": details,
    }


def scrape_espn_odds(from_date: str, to_date: str, dry_run: bool = False):
    """
    Scrape NBA odds from ESPN for a date range.

    Args:
        from_date: Start date YYYY-MM-DD
        to_date: End date YYYY-MM-DD (inclusive)
        dry_run: Show stats only
    """
    start = datetime.strptime(from_date, "%Y-%m-%d")
    end = datetime.strptime(to_date, "%Y-%m-%d")

    total_days = (end - start).days + 1
    logger.info(f"Scraping ESPN odds from {from_date} to {to_date} ({total_days} days)")

    games_by_date: dict[str, list] = defaultdict(list)
    total_games = 0
    total_odds = 0
    no_odds = 0

    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")

        # Skip if this date already has data in odds_cache
        cache_path = ODDS_CACHE_DIR / f"{date_str}.json"
        if cache_path.exists():
            try:
                existing = json.loads(cache_path.read_text(encoding="utf-8"))
                existing_count = len(existing.get("games", []))
                if existing_count > 0:
                    current += timedelta(days=1)
                    continue
            except Exception:
                pass

        games = get_games_for_date(date_str)
        if not games:
            current += timedelta(days=1)
            continue

        logger.info(f"{date_str}: {len(games)} finished games")
        total_games += len(games)

        for game in games:
            home_abbr = ESPN_TEAM_MAP.get(game["home_espn"], game["home_espn"])
            away_abbr = ESPN_TEAM_MAP.get(game["away_espn"], game["away_espn"])

            if not dry_run:
                time.sleep(0.3)  # gentle rate limit
                odds = get_odds_for_game(game["event_id"])
            else:
                odds = None

            if odds and odds["spread"] is not None:
                total_odds += 1
                entry = {
                    "home_team": TEAM_NAMES.get(home_abbr, home_abbr),
                    "away_team": TEAM_NAMES.get(away_abbr, away_abbr),
                    "home_abbrev": home_abbr,
                    "away_abbrev": away_abbr,
                    "game_date": date_str,
                    "spread": odds["spread"],
                    "total": odds["total"],
                    "spread_book": odds["provider"],
                    "total_book": odds["provider"],
                    "source": "espn",
                    "espn_event_id": game["event_id"],
                }
                games_by_date[date_str].append(entry)
            else:
                no_odds += 1

        # Brief pause between dates
        time.sleep(0.5)
        current += timedelta(days=1)

    logger.info(f"\nScraped: {total_games} games, {total_odds} with odds, {no_odds} without")

    if dry_run:
        logger.info("[DRY RUN] Not writing files.")
        return total_odds

    # Write to odds_cache
    dates_written = 0
    for date_str, games in sorted(games_by_date.items()):
        cache_path = ODDS_CACHE_DIR / f"{date_str}.json"

        existing_games = []
        if cache_path.exists():
            try:
                existing = json.loads(cache_path.read_text(encoding="utf-8"))
                existing_games = existing.get("games", [])
            except Exception:
                pass

        existing_keys = {(g["home_abbrev"], g["away_abbrev"]) for g in existing_games}
        for game in games:
            key = (game["home_abbrev"], game["away_abbrev"])
            if key not in existing_keys:
                existing_games.append(game)
                existing_keys.add(key)

        cache_data = {
            "fetched_at": datetime.now().isoformat(),
            "game_count": len(existing_games),
            "games": existing_games,
        }
        cache_path.write_text(json.dumps(cache_data, indent=2), encoding="utf-8")
        dates_written += 1

    logger.info(f"Wrote {dates_written} date files to odds_cache")
    return total_odds


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scrape ESPN NBA odds into odds_cache")
    parser.add_argument("--from", dest="from_date", default="2025-10-21")
    parser.add_argument("--to", dest="to_date", default="2026-02-19")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    scrape_espn_odds(args.from_date, args.to_date, dry_run=args.dry_run)
