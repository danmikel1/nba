"""
Backfill feat_miles_96h and feat_altitude_next in ml_training_data.csv.

These features were added after historical bets were logged, so all existing
rows have 0s. This script reconstructs the correct values from the full
2025-26 game log.

Uses:
  - LeagueDashPlayerStats  → player name → team abbreviation
  - LeagueGameLog           → full season schedule + matchup strings
  - ARENA_COORDS/ALTITUDE_FT from morning_refresh.py
"""
import sys
import time
import json
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
from collections import defaultdict

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from nba_api.stats.endpoints import leaguedashplayerstats, leaguegamelog

# ── Shared constants (mirror morning_refresh.py) ─────────────────────────────
ARENA_COORDS = {
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

ALTITUDE_FT = {"DEN": 5280, "UTA": 4327, "OKC": 1200}

CSV_PATH  = Path(__file__).parent.parent / "data" / "ml_training_data.csv"


def haversine_miles(c1: tuple, c2: tuple) -> float:
    R = 3958.8
    lat1, lon1 = radians(c1[0]), radians(c1[1])
    lat2, lon2 = radians(c2[0]), radians(c2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def build_player_team_map() -> dict:
    """Return {player_name_lower: team_abbrev} for active 2025-26 players."""
    print("Fetching player → team mapping …")
    time.sleep(0.6)
    dash = leaguedashplayerstats.LeagueDashPlayerStats(
        season="2025-26",
        per_mode_detailed="PerGame",
        season_type_all_star="Regular Season",
    )
    df = dash.get_data_frames()[0]
    mapping = {}
    for _, row in df.iterrows():
        mapping[row["PLAYER_NAME"].lower()] = row["TEAM_ABBREVIATION"]
    print(f"  {len(mapping)} players mapped")
    return mapping


def build_historical_fatigue(season: str = "2025-26") -> dict:
    """
    Fetch the full-season game log once and compute miles_96h for
    every (team_abbrev, YYYY-MM-DD) pair.

    Returns:
        {(team_abbrev, date_str): miles_traveled_last_96h}
    """
    print("Fetching full-season game log …")
    time.sleep(0.6)
    log = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
        league_id="00",
    )
    games_df = log.get_data_frames()[0]
    games_df["GAME_DATE"] = pd.to_datetime(games_df["GAME_DATE"])
    print(f"  {len(games_df)} game-team rows loaded")

    # Build per-team game list sorted by date
    # matchup: "LAL vs. GSW" (home) or "LAL @ GSW" (away)
    team_games: dict = defaultdict(list)
    for _, row in games_df.iterrows():
        abbrev = row["TEAM_ABBREVIATION"]
        date   = row["GAME_DATE"]
        matchup = row["MATCHUP"]
        # Arena = where the game was played
        if " vs. " in matchup:
            arena = abbrev                           # home game
        else:
            arena = matchup.split("@ ")[-1].strip() # away game
        team_games[abbrev].append((date, arena))

    # Sort each team's games by date
    for abbrev in team_games:
        team_games[abbrev].sort(key=lambda x: x[0])

    # Compute miles_96h for each (team, game_date)
    fatigue: dict = {}
    for abbrev, game_list in team_games.items():
        for idx, (game_date, game_arena) in enumerate(game_list):
            cutoff = game_date - pd.Timedelta(hours=96)
            miles = 0.0
            prev_arena = abbrev  # start at home
            for prev_date, prev_arena_candidate in reversed(game_list[:idx]):
                if prev_date < cutoff:
                    break
                prev_arena = prev_arena_candidate
            # Travel: from prev_arena to game_arena
            if prev_arena != game_arena:
                c1 = ARENA_COORDS.get(prev_arena)
                c2 = ARENA_COORDS.get(game_arena)
                if c1 and c2:
                    miles = haversine_miles(c1, c2)
            fatigue[(abbrev, game_date.strftime("%Y-%m-%d"))] = round(miles, 1)

    print(f"  Historical fatigue computed for {len(fatigue)} (team, date) pairs")
    return fatigue


def main():
    # 1. Load training CSV
    df = pd.read_csv(CSV_PATH)
    print(f"\nLoaded training CSV: {len(df)} rows")
    rows_before = (df["feat_miles_96h"] != 0).sum()
    print(f"  feat_miles_96h non-zero before: {rows_before}")

    # 2. Fetch support data
    player_team = build_player_team_map()
    fatigue     = build_historical_fatigue()

    # 3. Compute features row-by-row
    miles_filled   = 0
    alt_filled     = 0
    team_not_found = 0

    for i, row in df.iterrows():
        player_name = str(row.get("player", "")).lower()
        date_str    = str(row.get("date", ""))[:10]
        opponent    = str(row.get("opponent", ""))
        is_home     = int(row.get("feat_is_home", 0))

        # Get player's team
        team = player_team.get(player_name)
        if not team:
            team_not_found += 1
            # Still fill altitude for away games (we have opponent)
        
        # feat_altitude_next
        if is_home:
            alt = ALTITUDE_FT.get(team or "", 0)
        else:
            alt = ALTITUDE_FT.get(opponent, 0)
        
        if alt != int(row.get("feat_altitude_next", 0)):
            df.at[i, "feat_altitude_next"] = alt
            alt_filled += 1

        # feat_miles_96h
        if team:
            miles = fatigue.get((team, date_str), 0.0)
            if miles != float(row.get("feat_miles_96h", 0.0)):
                df.at[i, "feat_miles_96h"] = miles
                miles_filled += 1

    # 4. Save
    df.to_csv(CSV_PATH, index=False)
    rows_after = (df["feat_miles_96h"] != 0).sum()

    print(f"\nResults:")
    print(f"  feat_miles_96h  updated: {miles_filled} rows  (non-zero now: {rows_after})")
    print(f"  feat_altitude_next updated: {alt_filled} rows")
    print(f"  Players not found in team map: {team_not_found}")
    print(f"\n✅ {CSV_PATH.name} saved.")


if __name__ == "__main__":
    main()
