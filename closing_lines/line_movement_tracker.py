"""
line_movement_tracker.py — Track and understand market line changes

Props move because:
1. Sharp money: If you got a good price, sharps may agree (line moves your way) or disagree (line moves against you)
2. Public action: Casual bettors push line in one direction
3. Injury/news: Information surprises change expectations (can move 2-3+ pts)
4. Game momentum: As game develops, expectation changes

This tool helps you:
1. Record line movement patterns (build intuition about what's "normal")
2. Compare your closing lines to market data (if available)
3. Detect stale/inaccurate closing lines
4. Track CLV correlation (sharp money agreement)

Usage:
    python line_movement_tracker.py --summary         # Overall stats
    python line_movement_tracker.py --by-market       # By market type
    python line_movement_tracker.py --timeseries      # Movement over time
    python line_movement_tracker.py --correlation     # CLV vs result correlation
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List


TRACKER_PATH = Path(__file__).resolve().parent.parent / "data" / "bet_tracker.json"


def load_tracker() -> pd.DataFrame:
    """Load bet_tracker.json as DataFrame."""
    with open(TRACKER_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    # Convert to numeric
    for col in ['line', 'closing_line', 'clv', 'ev', 'prob', 'odds', 'result']:
        if col in df.columns:
            if col == 'result':
                continue  # Keep as string
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['line_move'] = df['closing_line'] - df['line']
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    return df


def movement_summary(df: pd.DataFrame):
    """Overall line movement summary."""
    print("\n" + "="*100)
    print("LINE MOVEMENT SUMMARY")
    print("="*100)
    
    closed = df[df['closing_line'] != 0].copy()
    if len(closed) == 0:
        print("No closing lines yet.")
        return
    
    # Distribution
    moves = closed['line_move'].values
    
    print(f"\nBets with closing lines: {len(closed)}/{len(df)}")
    print(f"\nMovement distribution:")
    print(f"  Mean:   {moves.mean():+.3f} pts (market moved this much on avg)")
    print(f"  Median: {np.median(moves):+.3f} pts")
    print(f"  Std:    {moves.std():.3f} pts")
    print(f"  Min:    {moves.min():+.1f} pts (worst movement)")
    print(f"  Max:    {moves.max():+.1f} pts (best movement)")
    print(f"  Skew:   {pd.Series(moves).skew():.2f} (>0 = more positive movement)")
    
    # Buckets
    buckets = {
        'Extreme negative (-3 to -2)': (moves >= -3) & (moves < -2),
        'Very negative (-2 to -1)': (moves >= -2) & (moves < -1),
        'Negative (-1 to -0.5)': (moves >= -1) & (moves < -0.5),
        'Slight negative (-0.5 to 0)': (moves >= -0.5) & (moves < 0),
        'Slight positive (0 to +0.5)': (moves > 0) & (moves <= 0.5),
        'Positive (0.5 to +1)': (moves > 0.5) & (moves <= 1),
        'Very positive (1 to +2)': (moves > 1) & (moves <= 2),
        'Extreme positive (>+2)': (moves > 2),
        'ZERO movement': (moves == 0),
    }
    
    print(f"\nMovement by bucket (% of bets):")
    for label, mask in buckets.items():
        count = mask.sum()
        pct = count / len(closed) * 100
        if count > 0:
            print(f"  {label:35} {count:4} bets ({pct:5.1f}%)")
    
    # Win rate by movement direction
    print(f"\nWin rate by movement direction:")
    pos_move = closed[closed['line_move'] > 0]
    neg_move = closed[closed['line_move'] < 0]
    zero_move = closed[closed['line_move'] == 0]
    
    if len(pos_move) > 0:
        wr = (pos_move['result'] == 'Win').mean()
        print(f"  +CLV (market moved your way): {wr*100:.1f}% win rate ({len(pos_move)} bets)")
    if len(neg_move) > 0:
        wr = (neg_move['result'] == 'Win').mean()
        print(f"  -CLV (market moved against you): {wr*100:.1f}% win rate ({len(neg_move)} bets)")
    if len(zero_move) > 0:
        wr = (zero_move['result'] == 'Win').mean()
        print(f"  0 CLV (no movement): {wr*100:.1f}% win rate ({len(zero_move)} bets)")


def movement_by_market(df: pd.DataFrame):
    """Line movement patterns by market type."""
    print("\n" + "="*100)
    print("LINE MOVEMENT BY MARKET")
    print("="*100)
    
    closed = df[df['closing_line'] != 0].copy()
    if len(closed) == 0:
        print("No closing lines yet.")
        return
    
    print("\nMarket volatility (how much lines typically move):")
    print(f"{'Market':<8} {'Bets':<6} {'Avg Move':<12} {'Std Dev':<10} {'% Too Big':<12}")
    print("-" * 50)
    
    # Known reasonable limits per market
    limits = {'PTS': 3, 'REB': 1.5, 'AST': 1.5, 'PA': 3, 'PRA': 3}
    
    for market in sorted(df['market'].unique()):
        m_closed = closed[closed['market'] == market]
        if len(m_closed) > 0:
            avg_move = m_closed['line_move'].mean()
            std_move = m_closed['line_move'].std()
            limit = limits.get(market, 2.5)
            extreme = (m_closed['line_move'].abs() > limit).mean() * 100
            
            emoji = "✓" if std_move < 1.0 else "⚠️" if std_move < 2.0 else "🚨"
            print(f"{market:<8} {len(m_closed):<6} {avg_move:+.3f} pts   {std_move:<10.3f} {extreme:>10.1f}%  {emoji}")


def movement_by_side(df: pd.DataFrame):
    """Do OVER vs UNDER bets move differently?"""
    print("\n" + "="*100)
    print("LINE MOVEMENT BY SIDE (OVER vs UNDER)")
    print("="*100)
    
    closed = df[df['closing_line'] != 0].copy()
    if len(closed) == 0:
        print("No closing lines yet.")
        return
    
    print("\nDoes market line tend to move toward OVER or UNDER?")
    print(f"{'Side':<8} {'Bets':<6} {'Avg Move':<12} {'% Negative':<12} {'Interpretation'}")
    print("-" * 70)
    
    for side in ['OVER', 'UNDER']:
        s_closed = closed[closed['side'] == side]
        if len(s_closed) > 0:
            avg_move = s_closed['line_move'].mean()
            pct_neg = (s_closed['line_move'] < 0).mean() * 100
            
            if avg_move > 0.2:
                interp = "Market moved toward OVER (props go up)"
            elif avg_move < -0.2:
                interp = "Market moved toward UNDER (props go down)"
            else:
                interp = "No systematic bias"
            
            print(f"{side:<8} {len(s_closed):<6} {avg_move:+.3f} pts   {pct_neg:>10.1f}%   {interp}")


def movement_timeseries(df: pd.DataFrame):
    """How has line movement evolved over time?"""
    print("\n" + "="*100)
    print("LINE MOVEMENT OVER TIME")
    print("="*100)
    
    closed = df[df['closing_line'] != 0].copy()
    if len(closed) == 0:
        print("No closing lines yet.")
        return
    
    closed = closed[closed['date'].notna()].copy()
    closed = closed.sort_values('date')
    
    # Rolling window (7-day)
    closed['date_only'] = closed['date'].dt.date
    daily = closed.groupby('date_only').agg({
        'line_move': ['mean', 'std', 'count']
    }).reset_index()
    daily.columns = ['date', 'avg_move', 'std_move', 'n_bets']
    
    print("\nDaily line movement (7-day rolling window):")
    print(f"{'Date':<12} {'Avg Move':<12} {'Std Dev':<10} {'Bets':<6}")
    print("-" * 45)
    
    for _, row in daily.iterrows():
        move = row['avg_move']
        emoji = "📈" if move > 0.2 else "📉" if move < -0.2 else "→"
        print(f"{row['date']:<12} {move:+.3f} pts   {row['std_move']:<10.3f} {int(row['n_bets']):<6} {emoji}")
    
    # Trend
    if len(daily) > 1:
        first_avg = daily['avg_move'].iloc[:len(daily)//2].mean()
        last_avg = daily['avg_move'].iloc[len(daily)//2:].mean()
        delta = last_avg - first_avg
        print(f"\nTrend: Early period avg={first_avg:+.3f}, Recent avg={last_avg:+.3f}")
        if abs(delta) > 0.1:
            trend = "IMPROVING (more positive movement)" if delta > 0 else "DEGRADING (more negative)"
            print(f"  → {trend}")


def clv_outcome_correlation(df: pd.DataFrame):
    """Do CLV and outcomes correlate? (validator for CLV calculation)"""
    print("\n" + "="*100)
    print("CLV vs OUTCOME CORRELATION (Sharpness Check)")
    print("="*100)
    
    closed = df[df['closing_line'] != 0].copy()
    if len(closed) == 0:
        print("No closing lines yet.")
        return
    
    # Binary win/loss
    closed['is_win'] = (closed['result'] == 'Win').astype(int)
    
    # Can we correlate CLV with winning?
    closed['clv'] = pd.to_numeric(closed.get('clv', 0), errors='coerce').fillna(0)
    
    if closed['clv'].std() < 0.01:
        print("⚠️  All CLV values are identical (likely not calculated correctly)")
        print("   CLV should show variation unless all lines are truly static")
        return
    
    correlation = closed[['clv', 'is_win']].corr().iloc[0, 1]
    
    print(f"\nCLV-to-Win correlation: {correlation:.3f}")
    
    if abs(correlation) < 0.1:
        print(f"  🔴 Weak/no correlation (correlation near 0)")
        print(f"     → Your CLV data may be inaccurate or closing lines are stale")
    elif correlation > 0.15:
        print(f"  🟢 Strong correlation (correlation > 0.15)")
        print(f"     → Your models show REAL edge (sharp money agreement with picks)")
    else:
        print(f"  🟡 Moderate correlation ({correlation:.3f})")
        print(f"     → Reasonable; edge is present but noisy")
    
    # By EV level
    closed['ev'] = pd.to_numeric(closed.get('ev', 0), errors='coerce').fillna(0)
    
    print(f"\nWin rate by CLV + EV level:")
    closed['clv_sign'] = np.where(closed['clv'] > 0, '+CLV', 
                                   np.where(closed['clv'] < 0, '-CLV', '0CLV'))
    closed['ev_sign'] = np.where(closed['ev'] > 0.1, '+EV', 
                                 np.where(closed['ev'] < -0.05, '-EV', '~EV'))
    
    for clv_s in ['+CLV', '0CLV', '-CLV']:
        for ev_s in ['+EV', '~EV', '-EV']:
            subset = closed[(closed['clv_sign'] == clv_s) & (closed['ev_sign'] == ev_s)]
            if len(subset) > 5:
                wr = subset['is_win'].mean() * 100
                print(f"  {clv_s:5} + {ev_s:3}: {wr:5.1f}% ({len(subset):3} bets)")


def main():
    parser = argparse.ArgumentParser(description="Track and validate line movement")
    parser.add_argument('--summary', action='store_true', help='Overall movement summary')
    parser.add_argument('--by-market', action='store_true', help='Movement by market')
    parser.add_argument('--by-side', action='store_true', help='Movement by OVER/UNDER')
    parser.add_argument('--timeseries', action='store_true', help='Daily movement trends')
    parser.add_argument('--correlation', action='store_true', help='CLV vs win rate correlation')
    args = parser.parse_args()
    
    df = load_tracker()
    
    # If no args, run all
    run_all = not any(vars(args).values())
    
    if args.summary or run_all:
        movement_summary(df)
    if args.by_market or run_all:
        movement_by_market(df)
    if args.by_side or run_all:
        movement_by_side(df)
    if args.timeseries or run_all:
        movement_timeseries(df)
    if args.correlation or run_all:
        clv_outcome_correlation(df)


if __name__ == "__main__":
    main()
