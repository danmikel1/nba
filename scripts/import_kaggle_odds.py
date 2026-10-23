"""
Import Kaggle NBA betting data into our odds_cache format.

Source: https://www.kaggle.com/datasets/cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024
File:   data/nba_2008-2025.csv
Format: One row per game, with spread, total, moneylines, scores.

Writes to data/odds_cache/YYYY-MM-DD.json — same format as odds_engine.py
and oddspapi_backfill.py. Existing OddsPapi/The Odds API entries are preserved
(they take priority since they have richer metadata).
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

KAGGLE_CSV = Path("data/nba_2008-2025.csv")
ODDS_CACHE_DIR = Path("data/odds_cache")
ODDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Kaggle abbreviation → our standard abbreviation
TEAM_MAP = {
    "atl": "ATL", "bkn": "BKN", "bos": "BOS", "cha": "CHA", "chi": "CHI",
    "cle": "CLE", "dal": "DAL", "den": "DEN", "det": "DET", "gs": "GSW",
    "hou": "HOU", "ind": "IND", "lac": "LAC", "lal": "LAL", "mem": "MEM",
    "mia": "MIA", "mil": "MIL", "min": "MIN", "no": "NOP", "ny": "NYK",
    "okc": "OKC", "orl": "ORL", "phi": "PHI", "phx": "PHX", "por": "POR",
    "sa": "SAS", "sac": "SAC", "tor": "TOR", "utah": "UTA", "was": "WAS",
    "wsh": "WAS",  # Washington uses 'wsh' in some rows
}

# Full team names for the cache entries
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


def import_kaggle_odds(min_season: int = 2023, dry_run: bool = False):
    """
    Import Kaggle NBA betting data into odds_cache.

    Args:
        min_season: Minimum season year (end-year). 2023 = 2022-23 season.
        dry_run: If True, show stats but don't write files.
    """
    df = pd.read_csv(KAGGLE_CSV)
    df = df[df["season"] >= min_season].copy()
    print(f"Kaggle data: {len(df)} games from season {min_season} onward")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Build game entries grouped by date
    games_by_date: dict[str, list] = defaultdict(list)
    skipped = 0
    converted = 0

    for _, row in df.iterrows():
        date_str = str(row["date"])
        home_kaggle = str(row["home"]).strip().lower()
        away_kaggle = str(row["away"]).strip().lower()

        home_abbrev = TEAM_MAP.get(home_kaggle)
        away_abbrev = TEAM_MAP.get(away_kaggle)

        if not home_abbrev or not away_abbrev:
            print(f"  WARN: unmapped team: home={home_kaggle} away={away_kaggle}")
            skipped += 1
            continue

        # Spread: Kaggle stores as positive number, whos_favored gives direction
        raw_spread = float(row["spread"]) if pd.notna(row["spread"]) else None
        favored = str(row.get("whos_favored", "")).strip().lower()

        if raw_spread is not None and raw_spread > 0:
            # Convert to home-team-perspective spread
            # Negative = home favored, Positive = away favored
            if favored == "home":
                home_spread = -raw_spread
            elif favored == "away":
                home_spread = raw_spread
            else:
                home_spread = 0.0  # push / pick'em
        elif raw_spread == 0:
            home_spread = 0.0
        else:
            home_spread = None

        game_total = float(row["total"]) if pd.notna(row["total"]) else None

        game_entry = {
            "home_team": TEAM_NAMES.get(home_abbrev, home_abbrev),
            "away_team": TEAM_NAMES.get(away_abbrev, away_abbrev),
            "home_abbrev": home_abbrev,
            "away_abbrev": away_abbrev,
            "game_date": date_str,
            "spread": home_spread,
            "total": game_total,
            "spread_book": "kaggle-sbr-espn",
            "total_book": "kaggle-sbr-espn",
            "source": "kaggle",
        }
        games_by_date[date_str].append(game_entry)
        converted += 1

    print(f"Converted: {converted} games across {len(games_by_date)} dates")
    print(f"Skipped: {skipped}")

    if dry_run:
        print("[DRY RUN] Not writing files.")
        return converted

    # Write to odds_cache, merging with existing data
    dates_written = 0
    dates_merged = 0
    games_added = 0
    games_skipped_existing = 0

    for date_str, new_games in sorted(games_by_date.items()):
        cache_path = ODDS_CACHE_DIR / f"{date_str}.json"

        existing_games = []
        if cache_path.exists():
            try:
                existing = json.loads(cache_path.read_text(encoding="utf-8"))
                existing_games = existing.get("games", [])
                dates_merged += 1
            except Exception:
                pass

        # Build set of existing game keys (home_abbrev, away_abbrev)
        existing_keys = {(g["home_abbrev"], g["away_abbrev"]) for g in existing_games}

        added_this_date = 0
        for game in new_games:
            key = (game["home_abbrev"], game["away_abbrev"])
            if key in existing_keys:
                games_skipped_existing += 1
                continue
            existing_games.append(game)
            existing_keys.add(key)
            added_this_date += 1
            games_added += 1

        cache_data = {
            "fetched_at": datetime.now().isoformat(),
            "game_count": len(existing_games),
            "games": existing_games,
        }
        cache_path.write_text(json.dumps(cache_data, indent=2), encoding="utf-8")
        dates_written += 1

    print(f"\nImport complete!")
    print(f"  Dates written: {dates_written}")
    print(f"  Dates merged (had existing data): {dates_merged}")
    print(f"  Games added: {games_added}")
    print(f"  Games skipped (existing OddsPapi/Odds API data): {games_skipped_existing}")
    return games_added


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Import Kaggle NBA odds into odds_cache")
    parser.add_argument("--min-season", type=int, default=2023,
                        help="Minimum season end-year (default: 2023 = 2022-23)")
    parser.add_argument("--dry-run", action="store_true", help="Show stats only")
    args = parser.parse_args()
    import_kaggle_odds(min_season=args.min_season, dry_run=args.dry_run)
