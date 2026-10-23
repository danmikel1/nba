"""
morning_refresh.py — Daily Pre-Market Data Ingestion

Run once per morning before analysis. Pulls advanced signals from NBA API
and writes JSON files that nba_prediction.py reads at analysis time:
  - data/fatigue_index.json   (travel miles, B2B, altitude)

Note: foul_rates.json and shot_quality.json were previously generated here
but are no longer consumed by the prediction pipeline.  feat_foul_rate is
computed directly from game logs; shot quality has no model feature.

Usage:
    python morning_refresh.py
"""

import json
import time
import logging
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

from nba_api.stats.endpoints import (
    leaguegamelog,
    leaguedashplayerptshot,
    leaguedashplayerstats,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ─────────────────────  Arena Coordinates (lat, lon)  ─────────────────────
ARENA_COORDS = {
    "ATL": (33.7573, -84.3963),
    "BOS": (42.3662, -71.0621),
    "BKN": (40.6826, -73.9754),
    "CHA": (35.2251, -80.8392),
    "CHI": (41.8807, -87.6742),
    "CLE": (41.4965, -81.6882),
    "DAL": (32.7905, -96.8103),
    "DEN": (39.7487, -105.0077),
    "DET": (42.3410, -83.0552),
    "GSW": (37.7680, -122.3877),
    "HOU": (29.7508, -95.3621),
    "IND": (39.7639, -86.1555),
    "LAC": (33.9425, -118.1072),
    "LAL": (34.0430, -118.2673),
    "MEM": (35.1382, -90.0505),
    "MIA": (25.7814, -80.1870),
    "MIL": (43.0450, -87.9170),
    "MIN": (44.9795, -93.2760),
    "NOP": (29.9490, -90.0823),
    "NYK": (40.7505, -73.9934),
    "OKC": (35.4634, -97.5151),
    "ORL": (28.5392, -81.3839),
    "PHI": (39.9012, -75.1720),
    "PHX": (33.4457, -112.0712),
    "POR": (45.5316, -122.6668),
    "SAC": (38.5802, -121.4997),
    "SAS": (29.4270, -98.4375),
    "TOR": (43.6435, -79.3791),
    "UTA": (40.7683, -111.9011),
    "WAS": (38.8981, -77.0209),
}

ALTITUDE_FT = {"DEN": 5280, "UTA": 4327, "OKC": 1200}


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def haversine_miles(coord1: tuple, coord2: tuple) -> float:
    """Great-circle distance in miles between two (lat, lon) points."""
    R = 3958.8  # Earth radius in miles
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def _arena_for_matchup(matchup: str, team_abbrev: str) -> str:
    """Return the arena (team abbrev) where the game was played.

    Matchup formats from the NBA API:
        Home game:  "LAL vs. GSW"
        Away game:  "LAL @ GSW"
    """
    if " vs. " in matchup:
        # Home game — arena is the team's own arena
        return matchup.split(" vs. ")[0].strip()
    elif " @ " in matchup:
        # Away game — arena is the opponent's
        return matchup.split(" @ ")[1].strip()
    # Fallback
    return team_abbrev


# ═══════════════════════════════════════════════════════════════════════════
#  1) FATIGUE INDEX
# ═══════════════════════════════════════════════════════════════════════════

def build_fatigue_index() -> None:
    """
    For each team, compute miles traveled and games played in the last 96 h.

    Writes  data/fatigue_index.json
    """
    logger.info("Building fatigue index …")
    try:
        time.sleep(0.6)
        glog = leaguegamelog.LeagueGameLog(
            season="2025-26",
            player_or_team_abbreviation="T",
            season_type_all_star="Regular Season",
        )
        df = glog.get_data_frames()[0]
    except Exception as e:
        logger.error(f"Failed to fetch team game log: {e}")
        return

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE", ascending=True)

    now = datetime.utcnow()
    cutoff = now - timedelta(hours=96)
    teams: dict = {}

    for abbrev in ARENA_COORDS:
        try:
            team_df = df[df["TEAM_ABBREVIATION"] == abbrev].copy()
            recent = team_df[team_df["GAME_DATE"] >= cutoff]

            if recent.empty:
                teams[abbrev] = {"miles_96h": 0.0, "games_96h": 0, "altitude_ft": ALTITUDE_FT.get(abbrev, 0)}
                continue

            # Build an ordered list of arena locations visited
            arenas_visited = []
            for _, row in recent.iterrows():
                arena_abbrev = _arena_for_matchup(row["MATCHUP"], abbrev)
                coord = ARENA_COORDS.get(arena_abbrev)
                if coord:
                    arenas_visited.append(coord)

            # Sum consecutive haversine distances
            total_miles = 0.0
            for i in range(1, len(arenas_visited)):
                total_miles += haversine_miles(arenas_visited[i - 1], arenas_visited[i])

            teams[abbrev] = {
                "miles_96h": round(total_miles, 1),
                "games_96h": len(recent),
                "altitude_ft": ALTITUDE_FT.get(abbrev, 0),
            }
        except Exception as e:
            logger.warning(f"Fatigue calc failed for {abbrev}: {e}")
            teams[abbrev] = {"miles_96h": 0.0, "games_96h": 0, "altitude_ft": 0}

    out = {"generated": datetime.utcnow().isoformat(), "teams": teams}
    (DATA_DIR / "fatigue_index.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info(f"  ✓ fatigue_index.json  ({len(teams)} teams)")


# ═══════════════════════════════════════════════════════════════════════════
#  2) FOUL RATES
# ═══════════════════════════════════════════════════════════════════════════

def build_foul_rates() -> None:
    """
    Per-player PF/36 and per-team fouls-drawn-per-100-possessions.

    Writes  data/foul_rates.json
    """
    logger.info("Building foul rates …")
    players_out: dict = {}
    teams_out: dict = {}

    # --- Player-level PF per 36 min ---
    try:
        time.sleep(0.6)
        player_stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season="2025-26",
            per_mode_detailed="Per36",
            season_type_all_star="Regular Season",
        )
        pdf = player_stats.get_data_frames()[0]
        for _, row in pdf.iterrows():
            pid = str(int(row["PLAYER_ID"]))
            players_out[pid] = {
                "name": row["PLAYER_NAME"],
                "pf_per36": round(float(row.get("PF", 0.0)), 2),
            }
        logger.info(f"  Player foul rates: {len(players_out)} players")
    except Exception as e:
        logger.error(f"Failed to fetch player foul stats: {e}")

    # --- Team-level fouls drawn per 100 possessions ---
    try:
        time.sleep(0.6)
        # Use league-wide player stats aggregated by team
        team_stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season="2025-26",
            per_mode_detailed="Per100Possessions",
            season_type_all_star="Regular Season",
        )
        tdf = team_stats.get_data_frames()[0]
        # Group by team and sum fouls drawn (FTM proxy: free throws made)
        if "TEAM_ID" in tdf.columns:
            team_agg = tdf.groupby("TEAM_ID").agg({"FTM": "sum", "GP": "max"}).reset_index()
            for _, row in team_agg.iterrows():
                tid = str(int(row["TEAM_ID"]))
                gp = max(int(row["GP"]), 1)
                teams_out[tid] = {
                    "fouls_drawn_per100": round(float(row["FTM"]) / gp, 2),
                }
        logger.info(f"  Team foul-drawing rates: {len(teams_out)} teams")
    except Exception as e:
        logger.error(f"Failed to fetch team foul-drawing stats: {e}")

    out = {"generated": datetime.utcnow().isoformat(), "players": players_out, "teams": teams_out}
    (DATA_DIR / "foul_rates.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("  ✓ foul_rates.json")


# ═══════════════════════════════════════════════════════════════════════════
#  3) SHOT QUALITY
# ═══════════════════════════════════════════════════════════════════════════

def build_shot_quality() -> None:
    """
    Per-player wide-open and tight-defense FG% & frequency.

    Uses bulk LeagueDashPlayerPtShot endpoint (3 API calls total)
    instead of per-player calls (~250 calls / ~3 min).

    Writes  data/shot_quality.json
    """
    logger.info("Building shot quality (bulk mode — 3 API calls) …")

    try:
        # --- 1. Wide Open (6+ feet) — all players in one call ---
        time.sleep(0.6)
        wide_resp = leaguedashplayerptshot.LeagueDashPlayerPtShot(
            season="2025-26",
            season_type_all_star="Regular Season",
            per_mode_simple="PerGame",
            close_def_dist_range_nullable="6+ Feet - Wide Open",
        )
        wide_df = wide_resp.get_data_frames()[0]
        logger.info(f"  Wide-open data: {len(wide_df)} players")

        # --- 2. Very Tight (0-2 feet) ---
        time.sleep(0.6)
        vtight_resp = leaguedashplayerptshot.LeagueDashPlayerPtShot(
            season="2025-26",
            season_type_all_star="Regular Season",
            per_mode_simple="PerGame",
            close_def_dist_range_nullable="0-2 Feet - Very Tight",
        )
        vtight_df = vtight_resp.get_data_frames()[0]
        logger.info(f"  Very-tight data: {len(vtight_df)} players")

        # --- 3. Tight (2-4 feet) ---
        time.sleep(0.6)
        tight_resp = leaguedashplayerptshot.LeagueDashPlayerPtShot(
            season="2025-26",
            season_type_all_star="Regular Season",
            per_mode_simple="PerGame",
            close_def_dist_range_nullable="2-4 Feet - Tight",
        )
        tight_df = tight_resp.get_data_frames()[0]
        logger.info(f"  Tight data: {len(tight_df)} players")
    except Exception as e:
        logger.error(f"Failed to fetch shot quality data: {e}")
        return

    # --- 4. Merge tight categories (average FG%, sum freq) ---
    tight_combined = pd.concat([vtight_df, tight_df], ignore_index=True)
    tight_agg = tight_combined.groupby(["PLAYER_ID", "PLAYER_NAME"]).agg(
        tight_fg_pct=("FG_PCT", "mean"),
        tight_freq=("FGA_FREQUENCY", "sum"),
    ).reset_index()

    # --- 5. Build results keyed by PLAYER_ID ---
    wide_lookup = wide_df.set_index("PLAYER_ID")
    tight_lookup = tight_agg.set_index("PLAYER_ID")

    all_pids = set(wide_df["PLAYER_ID"].tolist()) | set(tight_agg["PLAYER_ID"].tolist())

    # GP filter (>= 10) from any frame
    gp_lookup: dict = {}
    for df in [wide_df, vtight_df, tight_df]:
        for _, row in df.iterrows():
            pid = row["PLAYER_ID"]
            gp_lookup[pid] = max(gp_lookup.get(pid, 0), int(row.get("GP", 0)))

    results: dict = {}
    for pid in all_pids:
        if gp_lookup.get(pid, 0) < 10:
            continue

        entry: dict = {}
        if pid in wide_lookup.index:
            entry["name"] = str(wide_lookup.loc[pid, "PLAYER_NAME"])
            entry["wide_open_fg_pct"] = round(float(wide_lookup.loc[pid, "FG_PCT"]), 4)
            entry["wide_open_freq"] = round(float(wide_lookup.loc[pid, "FGA_FREQUENCY"]), 4)
        else:
            entry["wide_open_fg_pct"] = None
            entry["wide_open_freq"] = None

        if pid in tight_lookup.index:
            if "name" not in entry:
                entry["name"] = ""
            entry["tight_fg_pct"] = round(float(tight_lookup.loc[pid, "tight_fg_pct"]), 4)
            entry["tight_freq"] = round(float(tight_lookup.loc[pid, "tight_freq"]), 4)
        else:
            entry["tight_fg_pct"] = None
            entry["tight_freq"] = None

        if "name" not in entry:
            entry["name"] = ""

        results[str(pid)] = entry

    out = {"generated": datetime.utcnow().isoformat(), "players": results}
    (DATA_DIR / "shot_quality.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info(f"  ✓ shot_quality.json  ({len(results)} players)")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    overall_start = time.time()
    print(f"\n=== Morning Refresh Started === {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. Fatigue Index
    try:
        t0 = time.time()
        build_fatigue_index()
        logger.info(f"  Fatigue index completed in {time.time() - t0:.1f}s")
    except Exception as e:
        logger.error(f"Fatigue index FAILED: {e}")

    # 2. Fetch today's odds from The Odds API (1 request)
    try:
        from odds_engine import OddsEngine
        oe = OddsEngine()
        if oe.is_configured:
            t0 = time.time()
            count = oe.fetch_today()
            logger.info(f"  Odds fetch completed in {time.time() - t0:.1f}s — {count} lines")
        else:
            logger.info("  Odds API key not configured — skipping odds fetch")
    except Exception as e:
        logger.error(f"Odds fetch FAILED: {e}")

    # NOTE: build_foul_rates() and build_shot_quality() are still defined
    # above but no longer called — their outputs are unused by the pipeline.
    # feat_foul_rate comes from game logs; shot_quality has no model feature.

    elapsed = time.time() - overall_start
    print(f"\n=== Morning Refresh Complete === Total: {elapsed:.1f}s\n")
