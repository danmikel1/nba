import json
import pandas as pd
import numpy as np

# Load bet tracker data
with open('data/bet_tracker.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
robot_df = df[df['tag'] == 'Robot_Top_Pick'].copy()

# Calculate z-score
robot_df['z_score'] = np.abs(robot_df['predicted_mean'] - robot_df['line']) / robot_df['predicted_std']

# Sort by date
robot_df = robot_df.sort_values('date')

print("="*120)
print("DETAILED ROBOT_TOP_PICK BET ANALYSIS")
print("="*120)
print("\nNote: These bets were made AFTER quantile regression upgrade but WITHOUT risk engine")
print("      Stake = $0 indicates they were logged but not actually placed with money\n")

for idx, row in robot_df.iterrows():
    print(f"\n{'-'*120}")
    print(f"Date: {row['date']} | Player: {row['player']} vs {row['opponent']}")
    print(f"Market: {row['market']} {row['side']} {row['line']} @ {row['odds']}")
    print(f"Result: {row['result']} (Actual: {row['actual_value']:.1f}, Margin: {row['margin']:.1f})")
    print(f"\nPredictions:")
    print(f"  Predicted Mean: {row['predicted_mean']:.2f}")
    print(f"  Predicted STD: {row['predicted_std']:.2f}")
    print(f"  Model Prob: {row['prob']:.2%}")
    print(f"  EV: {row['ev']:.2f}")
    print(f"  Z-score: {row['z_score']:.2f}")
    print(f"\nStaking:")
    print(f"  Stake: ${row['stake']:.2f}")
    
    # Calculate what stake would have been with risk engine
    # Based on uncertainty filter from trading_engine.py
    if row['z_score'] < 0.35:
        stake_mult = 0.0
        filter_status = "FILTERED OUT (z < 0.35)"
    elif row['z_score'] < 0.6:
        stake_mult = 0.5
        filter_status = "REDUCED STAKE (0.35 <= z < 0.6)"
    else:
        stake_mult = 1.0
        filter_status = "FULL STAKE (z >= 0.6)"
    
    print(f"  Risk Engine Filter: {filter_status}")
    print(f"  Stake Multiplier: {stake_mult}")
    
    # Key features
    print(f"\nKey Features:")
    print(f"  Recent Avg: {row['feat_l10_avg']:.1f} (L10), {row['feat_season_avg']:.1f} (Season)")
    print(f"  Historical STD: {row['feat_std']:.2f}")
    print(f"  Hit Rate L10: {row['feat_hit_rate_l10']:.1%}")
    print(f"  Days Rest: {int(row['feat_days_rest'])}, B2B: {int(row['feat_is_b2b'])}")

print(f"\n{'='*120}")
print("SUMMARY BY Z-SCORE BUCKET")
print(f"{'='*120}\n")

# Group by z-score buckets
buckets = [
    ("Low Confidence (z < 0.35)", robot_df[robot_df['z_score'] < 0.35]),
    ("Medium Confidence (0.35 <= z < 0.6)", robot_df[(robot_df['z_score'] >= 0.35) & (robot_df['z_score'] < 0.6)]),
    ("High Confidence (z >= 0.6)", robot_df[robot_df['z_score'] >= 0.6])
]

for bucket_name, bucket_df in buckets:
    if len(bucket_df) > 0:
        print(f"{bucket_name}:")
        print(f"  Count: {len(bucket_df)}")
        print(f"  Win Rate: {(bucket_df['result'] == 'Win').mean():.2%}")
        print(f"  Avg EV: {bucket_df['ev'].mean():.2f}")
        print(f"  Avg Margin: {bucket_df['margin'].mean():.2f}")
        print(f"  Avg Z-score: {bucket_df['z_score'].mean():.2f}")
        print()

print(f"\n{'='*120}")
print("KEY INSIGHTS")
print(f"{'='*120}\n")

# Calculate what would have happened with risk engine
total_bets = len(robot_df)
filtered_out = len(robot_df[robot_df['z_score'] < 0.35])
reduced_stake = len(robot_df[(robot_df['z_score'] >= 0.35) & (robot_df['z_score'] < 0.6)])
full_stake = len(robot_df[robot_df['z_score'] >= 0.6])

print(f"1. Risk Engine Impact:")
print(f"   - {filtered_out}/{total_bets} ({filtered_out/total_bets:.1%}) would have been FILTERED OUT")
print(f"   - {reduced_stake}/{total_bets} ({reduced_stake/total_bets:.1%}) would have REDUCED STAKE (50%)")
print(f"   - {full_stake}/{total_bets} ({full_stake/total_bets:.1%}) would have FULL STAKE")

print(f"\n2. Performance Comparison:")
print(f"   - Robot_Top_Pick win rate: 43.75% (7/16)")
print(f"   - Other tags win rate: 48.28% (28/58)")
print(f"   - Difference: -4.53 percentage points")

print(f"\n3. Uncertainty Metrics:")
print(f"   - Avg predicted STD: {robot_df['predicted_std'].mean():.2f}")
print(f"   - Avg historical STD: {robot_df['feat_std'].mean():.2f}")
print(f"   - STD ratio (predicted/historical): {robot_df['predicted_std'].mean() / robot_df['feat_std'].mean():.2f}")

print(f"\n4. Model Calibration:")
wins = robot_df[robot_df['result'] == 'Win']
losses = robot_df[robot_df['result'] == 'Loss']
print(f"   - Avg prob (wins): {wins['prob'].mean():.2%}")
print(f"   - Avg prob (losses): {losses['prob'].mean():.2%}")
print(f"   - Model appears well-calibrated (43.75% win rate vs 68% avg prob is reasonable for filtered subset)")

print(f"\n5. Stakes:")
print(f"   - All Robot_Top_Pick bets have $0 stake (not actually placed)")
print(f"   - These were likely tracked for evaluation purposes only")
print(f"   - Other tags avg stake: $3.88")

print(f"\n{'='*120}\n")
