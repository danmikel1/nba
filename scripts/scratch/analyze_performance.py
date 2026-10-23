import json
import pandas as pd
import numpy as np

# Load bet tracker data
with open('data/bet_tracker.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

print("="*80)
print("OVERALL PERFORMANCE SUMMARY")
print("="*80)
print(f"Total bets: {len(df)}")
print(f"\nBets by tag:")
print(df['tag'].value_counts())

# Overall metrics
df['win'] = (df['result'] == 'Win').astype(int)
df['push'] = (df['result'] == 'Push').astype(int)
df['loss'] = (df['result'] == 'Loss').astype(int)

print(f"\n\nOVERALL METRICS:")
print(f"Win rate: {df['win'].sum()}/{len(df)} = {df['win'].mean():.2%}")
print(f"Push rate: {df['push'].sum()}/{len(df)} = {df['push'].mean():.2%}")
print(f"Loss rate: {df['loss'].sum()}/{len(df)} = {df['loss'].mean():.2%}")
print(f"Total EV: {df['ev'].sum():.2f}")
print(f"Avg stake: ${df['stake'].mean():.2f}")
print(f"Avg EV: {df['ev'].mean():.2f}")
print(f"Avg odds: {df['odds'].mean():.2f}")

# Calculate ROI
total_staked = df['stake'].sum()
total_return = sum(row['stake'] * row['odds'] if row['result'] == 'Win' 
                   else (row['stake'] if row['result'] == 'Push' else 0) 
                   for _, row in df.iterrows())
roi = ((total_return - total_staked) / total_staked) * 100

print(f"\nTotal staked: ${total_staked:.2f}")
print(f"Total return: ${total_return:.2f}")
print(f"Net profit: ${total_return - total_staked:.2f}")
print(f"ROI: {roi:.2f}%")

print("\n" + "="*80)
print("ROBOT_TOP_PICK ANALYSIS (Post Quantile Regression - No Risk Engine)")
print("="*80)

robot_df = df[df['tag'] == 'Robot_Top_Pick'].copy()

if len(robot_df) > 0:
    print(f"\nTotal Robot_Top_Pick bets: {len(robot_df)}")
    print(f"\nResults breakdown:")
    print(robot_df['result'].value_counts())
    
    print(f"\n\nROBOT_TOP_PICK METRICS:")
    print(f"Win rate: {robot_df['win'].sum()}/{len(robot_df)} = {robot_df['win'].mean():.2%}")
    print(f"Push rate: {robot_df['push'].sum()}/{len(robot_df)} = {robot_df['push'].mean():.2%}")
    print(f"Loss rate: {robot_df['loss'].sum()}/{len(robot_df)} = {robot_df['loss'].mean():.2%}")
    print(f"Total EV: {robot_df['ev'].sum():.2f}")
    print(f"Avg stake: ${robot_df['stake'].mean():.2f}")
    print(f"Avg EV: {robot_df['ev'].mean():.2f}")
    print(f"Avg odds: {robot_df['odds'].mean():.2f}")
    print(f"Avg prob: {robot_df['prob'].mean():.2%}")
    
    # ROI for Robot_Top_Pick
    robot_staked = robot_df['stake'].sum()
    robot_return = sum(row['stake'] * row['odds'] if row['result'] == 'Win' 
                       else (row['stake'] if row['result'] == 'Push' else 0) 
                       for _, row in robot_df.iterrows())
    robot_roi = ((robot_return - robot_staked) / robot_staked) * 100
    
    print(f"\nTotal staked: ${robot_staked:.2f}")
    print(f"Total return: ${robot_return:.2f}")
    print(f"Net profit: ${robot_return - robot_staked:.2f}")
    print(f"ROI: {robot_roi:.2f}%")
    
    # Margin analysis
    print(f"\n\nMARGIN ANALYSIS:")
    print(f"Avg margin: {robot_df['margin'].mean():.2f}")
    print(f"Avg margin (wins only): {robot_df[robot_df['win']==1]['margin'].mean():.2f}")
    print(f"Avg margin (losses only): {robot_df[robot_df['loss']==1]['margin'].mean():.2f}")
    
    # Market breakdown
    print(f"\n\nMARKET BREAKDOWN:")
    for market in robot_df['market'].unique():
        market_df = robot_df[robot_df['market'] == market]
        print(f"\n{market}:")
        print(f"  Bets: {len(market_df)}")
        print(f"  Win rate: {market_df['win'].mean():.2%}")
        print(f"  Avg EV: {market_df['ev'].mean():.2f}")
    
    # Side breakdown
    print(f"\n\nSIDE BREAKDOWN:")
    for side in robot_df['side'].unique():
        side_df = robot_df[robot_df['side'] == side]
        print(f"\n{side}:")
        print(f"  Bets: {len(side_df)}")
        print(f"  Win rate: {side_df['win'].mean():.2%}")
        print(f"  Avg EV: {side_df['ev'].mean():.2f}")

print("\n" + "="*80)
print("COMPARISON: ROBOT_TOP_PICK vs OTHER TAGS")
print("="*80)

# Compare with other tags
other_df = df[df['tag'] != 'Robot_Top_Pick'].copy()

if len(robot_df) > 0 and len(other_df) > 0:
    print(f"\n{'Metric':<25} {'Robot_Top_Pick':<20} {'Other Tags':<20}")
    print("-"*65)
    print(f"{'Count':<25} {len(robot_df):<20} {len(other_df):<20}")
    print(f"{'Win Rate':<25} {robot_df['win'].mean():<20.2%} {other_df['win'].mean():<20.2%}")
    print(f"{'Avg EV':<25} {robot_df['ev'].mean():<20.2f} {other_df['ev'].mean():<20.2f}")
    print(f"{'Avg Stake':<25} ${robot_df['stake'].mean():<19.2f} ${other_df['stake'].mean():<19.2f}")
    print(f"{'Avg Odds':<25} {robot_df['odds'].mean():<20.2f} {other_df['odds'].mean():<20.2f}")
    print(f"{'Avg Prob':<25} {robot_df['prob'].mean():<20.2%} {other_df['prob'].mean():<20.2%}")
    
    # ROI comparison
    other_staked = other_df['stake'].sum()
    other_return = sum(row['stake'] * row['odds'] if row['result'] == 'Win' 
                       else (row['stake'] if row['result'] == 'Push' else 0) 
                       for _, row in other_df.iterrows())
    other_roi = ((other_return - other_staked) / other_staked) * 100
    
    print(f"{'ROI':<25} {robot_roi:<20.2f}% {other_roi:<20.2f}%")

# Check for predicted_std and sigma values
print("\n" + "="*80)
print("UNCERTAINTY METRICS (Robot_Top_Pick)")
print("="*80)

if len(robot_df) > 0:
    if 'predicted_std' in robot_df.columns:
        print(f"\nPredicted STD statistics:")
        print(f"  Mean: {robot_df['predicted_std'].mean():.2f}")
        print(f"  Median: {robot_df['predicted_std'].median():.2f}")
        print(f"  Min: {robot_df['predicted_std'].min():.2f}")
        print(f"  Max: {robot_df['predicted_std'].max():.2f}")
    
    if 'feat_std' in robot_df.columns:
        print(f"\nFeature STD statistics:")
        print(f"  Mean: {robot_df['feat_std'].mean():.2f}")
        print(f"  Median: {robot_df['feat_std'].median():.2f}")
    
    # Check if z-score can be calculated
    if 'predicted_mean' in robot_df.columns and 'line' in robot_df.columns and 'predicted_std' in robot_df.columns:
        robot_df['z_score'] = np.abs(robot_df['predicted_mean'] - robot_df['line']) / robot_df['predicted_std']
        print(f"\nZ-score statistics (abs):")
        print(f"  Mean: {robot_df['z_score'].mean():.2f}")
        print(f"  Median: {robot_df['z_score'].median():.2f}")
        print(f"  Min: {robot_df['z_score'].min():.2f}")
        print(f"  Max: {robot_df['z_score'].max():.2f}")
        
        # Breakdown by z-score ranges
        print(f"\nPerformance by z-score range:")
        print(f"  z < 0.35: {len(robot_df[robot_df['z_score'] < 0.35])} bets")
        print(f"  0.35 <= z < 0.6: {len(robot_df[(robot_df['z_score'] >= 0.35) & (robot_df['z_score'] < 0.6)])} bets")
        print(f"  z >= 0.6: {len(robot_df[robot_df['z_score'] >= 0.6])} bets")

# ============================================================================
# CLV ANALYSIS (if closing lines populated)
# ============================================================================
print("\n" + "="*80)
print("CLOSING LINE VALUE (CLV) ANALYSIS")
print("="*80)
print("""
CLV (Closing Line Value) measures if the market moved in your favor after you bet.
CLV > 0 = market agreed with your bet (sharp money aligned)
CLV < 0 = market disagreed with your bet
CLV = 0 = either not settled yet or line didn't move

This separates LUCKY from SHARP: you can win with bad EV if CLV > 0.
""")

# Convert CLV to numeric
df['clv'] = pd.to_numeric(df.get('clv', 0), errors='coerce').fillna(0)
df['closing_line'] = pd.to_numeric(df.get('closing_line', 0), errors='coerce').fillna(0)

# Count settled bets (have non-zero closing_line)
settled = df[df['closing_line'] != 0].copy()

if len(settled) > 0:
    print(f"\nSettled bets with closing lines: {len(settled)}/{len(df)} ({len(settled)/len(df)*100:.1f}%)")
    print(f"\nCLV Statistics:")
    print(f"  Mean CLV: {settled['clv'].mean():+.2f} pts")
    print(f"  Median CLV: {settled['clv'].median():+.2f} pts")
    print(f"  Std Dev: {settled['clv'].std():.2f} pts")
    print(f"  % with +CLV: {(settled['clv'] > 0).mean()*100:.1f}%")
    print(f"  % with -CLV: {(settled['clv'] < 0).mean()*100:.1f}%")
    print(f"  % with 0 CLV: {(settled['clv'] == 0).mean()*100:.1f}%")
    
    # CLV correlation with outcomes
    print(f"\nCLV vs Outcomes:")
    pos_clv = settled[settled['clv'] > 0]
    neg_clv = settled[settled['clv'] < 0]
    zero_clv = settled[settled['clv'] == 0]
    
    if len(pos_clv) > 0:
        print(f"  +CLV bets ({len(pos_clv)}): {pos_clv['win'].mean()*100:.1f}% win rate")
    if len(neg_clv) > 0:
        print(f"  -CLV bets ({len(neg_clv)}): {neg_clv['win'].mean()*100:.1f}% win rate")
    if len(zero_clv) > 0:
        print(f"  0 CLV bets ({len(zero_clv)}): {zero_clv['win'].mean()*100:.1f}% win rate")
    
    # Correlation between EV and CLV
    if 'ev' in df.columns:
        corr = settled[['ev', 'clv']].corr().iloc[0, 1]
        print(f"\nEV vs CLV correlation: {corr:.3f}")
        if abs(corr) > 0.1:
            print(f"  ✓ Model's EV estimates align with market movement (sharpness check)")
        else:
            print(f"  ⚠️  Weak correlation—model may be overconfident or market has different info")
    
    # By tag
    print(f"\nCLV by tag:")
    for tag in df['tag'].unique():
        tag_settled = settled[settled['tag'] == tag]
        if len(tag_settled) > 0:
            print(f"  {tag}: {len(tag_settled)} settled, "
                  f"avg CLV {tag_settled['clv'].mean():+.2f}, "
                  f"{(tag_settled['clv'] > 0).mean()*100:.0f}% positive")
else:
    print("\n❌ No closing lines populated yet.")
    print("   Use closing_line_cli.py or closing_line_helper.py to populate 'closing_line' field.")
    print("   Once done, CLV analysis will appear here.")
