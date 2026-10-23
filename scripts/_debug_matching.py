"""Check existing cached historical odds coverage"""
import json
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

cache_dir = Path("data/oddspapi_cache")
hist_files = list(cache_dir.glob("hist_*.json"))
print(f"Cached hist files: {len(hist_files)}")

# Load fixtures
fixtures = json.loads(Path("data/oddspapi_cache/_fixtures.json").read_text())["fixtures"]

# Match cached hist files to fixtures
matched_fixtures = []
for hf in hist_files:
    fid = hf.stem.replace("hist_", "")
    if fid in fixtures:
        fx = fixtures[fid]
        matched_fixtures.append(fx)

print(f"Hist files matching a fixture: {len(matched_fixtures)}")

# Show date range
if matched_fixtures:
    dates = [f["startTime"][:10] for f in matched_fixtures]
    print(f"Date range of cached odds: {min(dates)} to {max(dates)}")

# Check what hasOdds looks like for these
has_odds_count = sum(1 for f in matched_fixtures if f.get("hasOdds"))
print(f"hasOdds=True among cached: {has_odds_count}/{len(matched_fixtures)}")

# Now check the matching problem: show date offset
# UTC startTime vs what our CSV has
df = pd.read_csv("data/nba_2008-2025.csv")
s26 = df[df["season"] == 2026]

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

STD_TO_CSV = {
    "ATL": "atl", "BOS": "bos", "BKN": "bkn", "CHA": "cha", "CHI": "chi",
    "CLE": "cle", "DAL": "dal", "DEN": "den", "DET": "det", "GSW": "gs",
    "HOU": "hou", "IND": "ind", "LAC": "lac", "LAL": "lal", "MEM": "mem",
    "MIA": "mia", "MIL": "mil", "MIN": "min", "NOP": "no",  "NYK": "ny",
    "OKC": "okc", "ORL": "orl", "PHI": "phi", "PHX": "phx", "POR": "por",
    "SAC": "sac", "SAS": "sa",  "TOR": "tor", "UTA": "utah", "WAS": "wsh",
}

def team_to_abbrev(name):
    lower = name.strip().lower()
    if lower in TEAM_ALIASES:
        return TEAM_ALIASES[lower]
    for alias, abbrev in TEAM_ALIASES.items():
        if alias in lower or lower in alias:
            return abbrev
    return name.strip().upper()[:3]

# Try matching with date offset
print("\nMatching with UTC-to-local date adjustment:")
matched_utc = 0
matched_adj = 0
for fid, fx in fixtures.items():
    home = team_to_abbrev(fx["participant1Name"])
    away = team_to_abbrev(fx["participant2Name"])
    home_csv = STD_TO_CSV.get(home, home.lower())
    away_csv = STD_TO_CSV.get(away, away.lower())
    
    utc_date = fx["startTime"][:10]
    # Try UTC date
    m1 = s26[(s26["date"] == utc_date) & (s26["home"] == home_csv) & (s26["away"] == away_csv)]
    if not m1.empty:
        matched_utc += 1
        continue
    
    # Try date - 1 day
    utc_dt = datetime.strptime(utc_date, "%Y-%m-%d")
    adj_date = (utc_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    m2 = s26[(s26["date"] == adj_date) & (s26["home"] == home_csv) & (s26["away"] == away_csv)]
    if not m2.empty:
        matched_adj += 1
    
print(f"Matched on UTC date: {matched_utc}")
print(f"Matched on UTC-1 day: {matched_adj}")
print(f"Total matched: {matched_utc + matched_adj}")

# Check home/away mapping
# In OddsPapi, is participant1 always home?
# Let's check with a known game
print("\nChecking home/away: first 3 fixtures with adjusted dates")
count = 0
for fid, fx in fixtures.items():
    home = team_to_abbrev(fx["participant1Name"])
    away = team_to_abbrev(fx["participant2Name"])
    home_csv = STD_TO_CSV.get(home, home.lower())
    away_csv = STD_TO_CSV.get(away, away.lower())
    
    utc_date = fx["startTime"][:10]
    utc_dt = datetime.strptime(utc_date, "%Y-%m-%d")
    adj_date = (utc_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    
    m = s26[(s26["date"].isin([utc_date, adj_date])) & (s26["home"] == home_csv) & (s26["away"] == away_csv)]
    if not m.empty:
        print(f"  Fixture: {fx['participant1Name']} (p1) vs {fx['participant2Name']} (p2)")
        row = m.iloc[0]
        print(f"  CSV:     home={row['home']}, away={row['away']}")
        count += 1
        if count >= 3:
            break
    
    # Also check swapped (p1=away, p2=home)
    m2 = s26[(s26["date"].isin([utc_date, adj_date])) & (s26["home"] == away_csv) & (s26["away"] == home_csv)]
    if not m2.empty and m.empty:
        print(f"  SWAPPED: {fx['participant1Name']} (p1) vs {fx['participant2Name']} (p2)")
        row = m2.iloc[0]
        print(f"  CSV:     home={row['home']}, away={row['away']}")
        count += 1
        if count >= 3:
            break
