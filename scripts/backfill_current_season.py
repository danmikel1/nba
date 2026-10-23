"""
Fetch 2025-26 season game data and append to nba_2008-2025.csv.
Uses: NBA API (free) for scores + odds_cache for spreads/totals.
"""
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime
from nba_api.stats.endpoints import leaguegamelog

DATA_DIR = Path("data")
ODDS_CACHE_DIR = DATA_DIR / "odds_cache"
CSV_PATH = DATA_DIR / "nba_2008-2025.csv"

# Team ID -> CSV abbreviation mapping
TEAM_ID_TO_CSV = {
    1610612737: "atl", 1610612738: "bos", 1610612751: "bkn", 1610612766: "cha",
    1610612741: "chi", 1610612739: "cle", 1610612742: "dal", 1610612743: "den",
    1610612765: "det", 1610612744: "gs",  1610612745: "hou", 1610612754: "ind",
    1610612746: "lac", 1610612747: "lal", 1610612763: "mem", 1610612748: "mia",
    1610612749: "mil", 1610612750: "min", 1610612740: "no",  1610612752: "ny",
    1610612760: "okc", 1610612753: "orl", 1610612755: "phi", 1610612756: "phx",
    1610612757: "por", 1610612758: "sac", 1610612759: "sa",  1610612761: "tor",
    1610612762: "utah",1610612764: "wsh",
}

# Standard abbrev -> CSV abbrev
STD_TO_CSV = {
    "ATL": "atl", "BOS": "bos", "BKN": "bkn", "CHA": "cha", "CHI": "chi",
    "CLE": "cle", "DAL": "dal", "DEN": "den", "DET": "det", "GSW": "gs",
    "HOU": "hou", "IND": "ind", "LAC": "lac", "LAL": "lal", "MEM": "mem",
    "MIA": "mia", "MIL": "mil", "MIN": "min", "NOP": "no",  "NYK": "ny",
    "OKC": "okc", "ORL": "orl", "PHI": "phi", "PHX": "phx", "POR": "por",
    "SAC": "sac", "SAS": "sa",  "TOR": "tor", "UTA": "utah", "WAS": "wsh",
}

def fetch_team_games():
    """Fetch team-level game log for 2025-26 season."""
    print("Fetching 2025-26 team game logs from NBA API...")
    time.sleep(1)
    log = leaguegamelog.LeagueGameLog(
        season="2025-26",
        player_or_team_abbreviation="T",
        season_type_all_star="Regular Season"
    )
    df = log.get_data_frames()[0]
    print(f"  Got {len(df)} team-game rows")
    return df

def parse_matchup(matchup_str):
    """Parse 'LAL vs. BOS' or 'LAL @ BOS' into (team, opponent, is_home)."""
    if " vs. " in matchup_str:
        parts = matchup_str.split(" vs. ")
        return parts[0].strip(), parts[1].strip(), True
    elif " @ " in matchup_str:
        parts = matchup_str.split(" @ ")
        return parts[0].strip(), parts[1].strip(), False
    return None, None, None

def load_odds_cache():
    """Load all odds cache files into a lookup dict."""
    odds = {}
    for f in ODDS_CACHE_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            for game in data.get("games", []):
                home = game.get("home_abbrev", "")
                away = game.get("away_abbrev", "")
                gdate = game.get("game_date", f.stem)
                if home and away:
                    key = (STD_TO_CSV.get(home, home.lower()),
                           STD_TO_CSV.get(away, away.lower()),
                           gdate)
                    odds[key] = {
                        "spread": game.get("spread"),
                        "total": game.get("total"),
                    }
        except Exception:
            pass
    print(f"  Loaded {len(odds)} games from odds cache")
    return odds

def load_odds_from_training_csv():
    """Fallback: extract per-game odds from the training data CSV."""
    print("  Loading odds from training CSV as fallback...")
    cols = ["date", "opponent", "feat_spread", "feat_game_total", "feat_is_home"]
    df = pd.read_csv(DATA_DIR / "ml_training_data.csv", usecols=cols)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # Group by date + opponent + is_home to get unique games
    grouped = df.groupby(["date", "opponent", "feat_is_home"]).agg(
        spread=("feat_spread", "median"),
        total=("feat_game_total", "median"),
    ).reset_index()

    odds = {}
    for _, r in grouped.iterrows():
        # We know the opponent and is_home, but not the player's team from this groupby
        # Store by (opponent, date, is_home) for fuzzy matching later
        odds[(r["opponent"], r["date"], int(r["feat_is_home"]))] = {
            "spread": r["spread"],
            "total": r["total"],
        }
    print(f"  Loaded {len(odds)} game-odds from training CSV")
    return odds

def build_game_rows(team_df, odds_cache, training_odds):
    """
    Convert team-level game log to game-level rows matching nba_2008-2025.csv format.
    Each game appears twice in team_df (one per team); we merge into one row.
    """
    # Parse matchups and identify home/away
    team_df = team_df.copy()
    team_df["GAME_DATE"] = pd.to_datetime(team_df["GAME_DATE"])
    team_df["date_str"] = team_df["GAME_DATE"].dt.strftime("%Y-%m-%d")

    parsed = team_df.apply(lambda r: parse_matchup(r["MATCHUP"]), axis=1, result_type="expand")
    team_df["team_abbr"] = parsed[0]
    team_df["opp_abbr"] = parsed[1]
    team_df["is_home"] = parsed[2]

    team_df["csv_team"] = team_df["TEAM_ID"].map(TEAM_ID_TO_CSV)

    # Split into home/away
    home_df = team_df[team_df["is_home"] == True].copy()
    away_df = team_df[team_df["is_home"] == False].copy()

    print(f"  Home rows: {len(home_df)}, Away rows: {len(away_df)}")

    # Merge on GAME_ID
    merged = home_df.merge(
        away_df,
        on="GAME_ID",
        suffixes=("_home", "_away"),
    )
    print(f"  Merged games: {len(merged)}")

    rows = []
    matched_odds = 0
    matched_training = 0

    for _, g in merged.iterrows():
        date_str = g["date_str_home"]
        home_csv = g["csv_team_home"]
        away_csv = g["csv_team_away"]
        score_home = int(g["PTS_home"])
        score_away = int(g["PTS_away"])

        # Try to get odds from cache
        spread = None
        total = None

        # Key format: (home_csv, away_csv, date)
        odds_key = (home_csv, away_csv, date_str)
        if odds_key in odds_cache:
            od = odds_cache[odds_key]
            spread = od.get("spread")
            total = od.get("total")
            if spread is not None:
                matched_odds += 1

        # Fallback to training CSV odds
        if spread is None or total is None:
            # Training data stores opponent from player perspective
            # Try home team perspective: opponent = away team, is_home = 1
            home_abbr = g["team_abbr_home"]
            away_abbr = g["team_abbr_away"]
            tkey = (away_abbr, date_str, 1)
            if tkey in training_odds:
                tod = training_odds[tkey]
                if spread is None and tod["spread"] is not None:
                    spread = tod["spread"]
                    matched_training += 1
                if total is None and tod["total"] is not None:
                    total = tod["total"]

            # Also try away perspective
            if spread is None or total is None:
                tkey2 = (home_abbr, date_str, 0)
                if tkey2 in training_odds:
                    tod = training_odds[tkey2]
                    if spread is None and tod["spread"] is not None:
                        # Flip sign (training stores from player's perspective)
                        spread = -tod["spread"]
                        matched_training += 1
                    if total is None and tod["total"] is not None:
                        total = tod["total"]

        # Determine who's favored
        if spread is not None:
            abs_spread = abs(spread)
            if spread < 0:
                whos_favored = "home"
            elif spread > 0:
                whos_favored = "away"
            else:
                whos_favored = "home"
        else:
            abs_spread = np.nan
            whos_favored = ""

        # Quarter scores not available from LeagueGameLog team endpoint
        # We'll set them to 0 — can be backfilled later
        row = {
            "season": 2026,
            "date": date_str,
            "regular": True,
            "playoffs": False,
            "away": away_csv,
            "home": home_csv,
            "score_away": score_away,
            "score_home": score_home,
            "q1_away": 0, "q2_away": 0, "q3_away": 0, "q4_away": 0, "ot_away": 0,
            "q1_home": 0, "q2_home": 0, "q3_home": 0, "q4_home": 0, "ot_home": 0,
            "whos_favored": whos_favored,
            "spread": abs_spread if pd.notna(abs_spread) else np.nan,
            "total": total,
            "moneyline_away": np.nan,
            "moneyline_home": np.nan,
            "h2_spread": np.nan,
            "h2_total": np.nan,
            "id_spread": np.nan,
            "id_total": np.nan,
        }
        rows.append(row)

    print(f"  Odds matched from cache: {matched_odds}")
    print(f"  Odds matched from training CSV: {matched_training}")
    print(f"  Total games without odds: {len(rows) - matched_odds - matched_training}")

    return pd.DataFrame(rows)

def main():
    # 1. Fetch team game logs
    team_df = fetch_team_games()

    # 2. Load odds
    print("\nLoading odds data...")
    odds_cache = load_odds_cache()
    training_odds = load_odds_from_training_csv()

    # 3. Build game rows
    print("\nBuilding game-level rows...")
    new_games = build_game_rows(team_df, odds_cache, training_odds)
    new_games.sort_values("date", inplace=True)
    print(f"\n2025-26 season: {len(new_games)} games")
    print(f"Date range: {new_games['date'].min()} to {new_games['date'].max()}")
    print(f"Games with spread: {new_games['spread'].notna().sum()}")
    print(f"Games with total: {new_games['total'].notna().sum()}")

    # 4. Load existing CSV and check for overlap
    existing = pd.read_csv(CSV_PATH)
    existing_dates = set(existing["date"].unique())
    overlap = new_games[new_games["date"].isin(existing_dates)]
    if len(overlap) > 0:
        print(f"\nWARNING: {len(overlap)} games overlap with existing data — skipping those")
        new_games = new_games[~new_games["date"].isin(existing_dates)]

    if new_games.empty:
        print("No new games to add!")
        return

    # 5. Append to CSV
    print(f"\nAppending {len(new_games)} new games to {CSV_PATH}...")
    combined = pd.concat([existing, new_games], ignore_index=True)
    combined.to_csv(CSV_PATH, index=False)
    print(f"Done! CSV now has {len(combined)} total games")
    print(f"Seasons: {sorted(combined['season'].unique())}")

if __name__ == "__main__":
    main()
