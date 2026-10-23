"""Debug fixture matching and hasOdds"""
import json
from pathlib import Path
from collections import Counter
import pandas as pd

data = json.loads(Path("data/oddspapi_cache/_fixtures.json").read_text())
fixtures = data["fixtures"]

has_odds = sum(1 for f in fixtures.values() if f.get("hasOdds"))
no_odds = sum(1 for f in fixtures.values() if not f.get("hasOdds"))
print(f"hasOdds=True: {has_odds}, hasOdds=False: {no_odds}")

# Check start times (UTC)
hours = Counter()
for f in fixtures.values():
    st = f.get("startTime", "")
    if "T" in st:
        hours[st.split("T")[1][:5]] += 1
print("\nStart time hours (UTC):")
for h, c in sorted(hours.items(), key=lambda x: -x[1])[:10]:
    print(f"  {h}: {c}")

# Dates: UTC vs local
dates_utc = [f.get("startTime", "")[:10] for f in fixtures.values()]
print(f"\nFixture date range: {min(dates_utc)} to {max(dates_utc)}")

# Show games that cross midnight UTC (early US dates)
crossing = []
for f in fixtures.values():
    st = f.get("startTime", "")
    hour = int(st.split("T")[1][:2])
    if hour < 5:  # UTC midnight-5am = US evening
        crossing.append(st[:10])

print(f"\nGames with UTC date != US date (midnight-5am UTC): {len(crossing)}")

# Load CSV and check matching
df = pd.read_csv("data/nba_2008-2025.csv")
s26 = df[df["season"] == 2026]
print(f"\nCSV 2026 dates: {s26['date'].min()} to {s26['date'].max()}")
print(f"CSV 2026 games: {len(s26)}")

# Sample fixtures to understand home/away
print("\nSample fixtures:")
for f in list(fixtures.values())[:5]:
    p1 = f.get("participant1Name", "")
    p2 = f.get("participant2Name", "")
    st = f.get("startTime", "")
    ho = f.get("hasOdds", False)
    print(f"  {st[:10]} {p1} vs {p2} (hasOdds={ho})")

# Check how existing oddspapi_backfill.py successfully matched
# Look at cached historical odds files
cache_dir = Path("data/oddspapi_cache")
hist_files = list(cache_dir.glob("hist_*.json"))
print(f"\nExisting historical odds files: {len(hist_files)}")
if hist_files:
    sample_hist = json.loads(hist_files[0].read_text())
    print(f"Sample hist keys: {list(sample_hist.keys())[:10]}")
    bks = sample_hist.get("bookmakers", {})
    for bk_name, bk_data in list(bks.items())[:2]:
        markets = bk_data.get("markets", {})
        print(f"  Bookmaker: {bk_name}, markets: {list(markets.keys())[:10]}")
