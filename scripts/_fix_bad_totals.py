"""Fix unreasonable totals in the 2026 season data."""
import json
import pandas as pd
from pathlib import Path

df = pd.read_csv("data/nba_2008-2025.csv")
s26 = df[df["season"] == 2026]

# Find games with unreasonable totals (outside 180-280)
bad = s26[(s26["total"] > 280) | (s26["total"] < 180)]
print(f"Games with unreasonable totals: {len(bad)}")

# Load totals from odds_cache
odds_cache = Path("data/odds_cache")
known_totals = {}
for f in odds_cache.glob("*.json"):
    try:
        data = json.loads(f.read_text())
        for g in data.get("games", []):
            total = g.get("total") or g.get("game_total")
            if total and float(total) > 100:
                home = g.get("home_abbrev", "").lower()
                away = g.get("away_abbrev", "").lower()
                gd = g.get("game_date", f.stem)
                known_totals[(gd, home, away)] = float(total)
    except Exception:
        continue

print(f"Known totals from odds_cache: {len(known_totals)}")

# Load totals from training CSV as secondary fallback
ml_csv = pd.read_csv("data/ml_training_data.csv")
training_totals = {}
for _, row in ml_csv.iterrows():
    gt = row.get("feat_game_total")
    if pd.notna(gt) and gt > 100:
        gd = str(row.get("game_date", ""))[:10]
        home = str(row.get("team", "")).lower()
        away = str(row.get("opponent", "")).lower()
        key = (gd, home, away)
        if key not in training_totals:
            training_totals[key] = gt

print(f"Known totals from training CSV: {len(training_totals)}")

# Fix each bad row
fixed = 0
for idx, row in bad.iterrows():
    key1 = (row["date"], row["home"], row["away"])
    total = known_totals.get(key1)
    if not total:
        key2 = (row["date"], row["home"], row["away"])
        total = training_totals.get(key2)
    
    if total and 180 <= total <= 280:
        old = df.loc[idx, "total"]
        df.loc[idx, "total"] = total
        print(f"  Fixed: {row['date']} {row['away']} @ {row['home']}: {old} -> {total}")
        fixed += 1
    else:
        # Use season average as last resort
        good_totals = s26[(s26["total"] >= 180) & (s26["total"] <= 280)]["total"]
        avg = good_totals.median()
        old = df.loc[idx, "total"]
        df.loc[idx, "total"] = round(avg * 2) / 2
        print(f"  Avg fallback: {row['date']} {row['away']} @ {row['home']}: {old} -> {round(avg * 2) / 2}")
        fixed += 1

print(f"\nFixed {fixed} games")
df.to_csv("data/nba_2008-2025.csv", index=False)
print("Saved CSV")

# Verify
s26 = df[df["season"] == 2026]
print(f"\nTotal range now: {s26['total'].min()} - {s26['total'].max()}")
print(f"Total mean: {s26['total'].mean():.1f}")
