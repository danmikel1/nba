import pandas as pd
import numpy as np

df = pd.read_csv('data/ml_training_data.csv', usecols=['date','player','market','line','feat_line','actual_value','predicted_prob','hit'])
print(f"Total rows: {len(df):,}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print()

print("line == feat_line?", (df['line'] == df['feat_line']).mean())
print()

print("=== Line stats ===")
print(df['line'].describe())
print()

lines = df['line'].dropna()
is_half = (lines % 0.5 == 0) & (lines % 1.0 != 0)
is_whole = lines % 1.0 == 0
print(f"Half-point lines (.5): {is_half.mean():.1%}")
print(f"Whole-point lines (.0): {is_whole.mean():.1%}")
print(f"Other: {(~is_half & ~is_whole).mean():.1%}")
print()

print("=== Line vs Actual ===")
print(f"Avg line:   {df['line'].mean():.2f}")
print(f"Avg actual: {df['actual_value'].mean():.2f}")
print(f"Avg gap:    {(df['actual_value'] - df['line']).mean():.2f}")
print()

df2 = df.dropna(subset=['predicted_prob','hit'])
print(f"Rows with predicted_prob: {len(df2):,}")
if len(df2) > 0:
    print(f"Avg predicted_prob: {df2['predicted_prob'].mean():.4f}")
    print(f"Avg hit rate:       {df2['hit'].mean():.4f}")
print()

# Check where lines come from
print("=== Line source analysis ===")
# Real sportsbook lines end in .5 (to avoid pushes)
# Season-average based lines won't have that pattern
for mkt in ['PTS','REB','AST','PRA','PA','3PM']:
    sub = df[df['market'] == mkt]['line'].dropna()
    if len(sub) < 100:
        continue
    pct_half = ((sub % 0.5 == 0) & (sub % 1.0 != 0)).mean()
    pct_round = (sub % 1.0 == 0).mean()
    pct_other = 1 - pct_half - pct_round
    print(f"  {mkt}: n={len(sub):,}  .5={pct_half:.0%}  .0={pct_round:.0%}  other={pct_other:.0%}  avg={sub.mean():.1f}")
print()

# Sample lines to see if they look like season averages
print("=== Sample lines (last 20) ===")
print(df[['date','player','market','line','actual_value']].tail(20).to_string())
print()

# Check the odds_cache for historical odds
import os, json
odds_dir = 'data/odds_cache'
if os.path.exists(odds_dir):
    files = os.listdir(odds_dir)
    print(f"Odds cache files: {len(files)}")
    if files:
        print(f"Sample: {files[:5]}")
else:
    print("No odds_cache directory")

oddspapi_dir = 'data/oddspapi_cache'
if os.path.exists(oddspapi_dir):
    files = os.listdir(oddspapi_dir)
    print(f"OddsAPI cache files: {len(files)}")
    if files:
        print(f"Sample: {files[:5]}")
        # Check a sample file
        sample = os.path.join(oddspapi_dir, files[0])
        with open(sample, encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            print(f"Sample entry keys: {list(data[0].keys()) if isinstance(data[0], dict) else type(data[0])}")
        elif isinstance(data, dict):
            print(f"Sample keys: {list(data.keys())[:10]}")
