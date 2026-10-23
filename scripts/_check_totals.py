"""Spot check calibrated totals"""
import pandas as pd

df = pd.read_csv("data/nba_2008-2025.csv")
s26 = df[df["season"] == 2026]
recent = s26[s26["date"] >= "2026-02-19"]
print(f"Recent games (Feb 19+): {len(recent)}")
print(f"Total range: {recent['total'].min()} - {recent['total'].max()}")
print(f"Total mean: {recent['total'].mean():.1f}")
print("\nSample:")
for _, r in recent.head(15).iterrows():
    date = r["date"]
    away = r["away"]
    home = r["home"]
    spread = r["spread"]
    total = r["total"]
    actual = r["score_home"] + r["score_away"]
    print(f"  {date} {away:4s} @ {home:4s}  spread={spread:5.1f}  total={total:5.1f}  actual={actual}")

print(f"\nMissing spread games:")
missing = s26[s26["spread"].isna()]
for _, r in missing.iterrows():
    print(f"  {r['date']} {r['away']} @ {r['home']}")
