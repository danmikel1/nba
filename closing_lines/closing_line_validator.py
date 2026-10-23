"""
closing_line_validator.py — Validate closing lines for errors & unrealistic movement

Since lines are "constantly moving", this tool helps ensure you're:
1. Capturing closing lines at the right time (not stale)
2. Recording realistic values (not typos)
3. Detecting suspicious patterns (duplicates, zero movement when expected)

Usage:
    python closing_line_validator.py                    # Full analysis
    python closing_line_validator.py --check-duplicates  # Find duplicates
    python closing_line_validator.py --market PTS       # Focus on one market
    python closing_line_validator.py --fix-suspicious   # Mark questionable bets
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta


TRACKER_PATH = Path(__file__).resolve().parent.parent / "data" / "bet_tracker.json"


def load_tracker() -> pd.DataFrame:
    """Load bet_tracker.json as DataFrame."""
    with open(TRACKER_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)


def save_tracker(df: pd.DataFrame):
    """Save DataFrame back to bet_tracker.json."""
    with open(TRACKER_PATH, 'w', encoding='utf-8') as f:
        json.dump(df.to_dict('records'), f, indent=2, ensure_ascii=False)


def analyze_closing_lines(df: pd.DataFrame):
    """Comprehensive analysis of closing line quality."""
    print("\n" + "="*100)
    print("CLOSING LINE VALIDATION REPORT")
    print("="*100)
    
    # Convert to numeric
    df['line'] = pd.to_numeric(df.get('line', 0), errors='coerce').fillna(0)
    df['closing_line'] = pd.to_numeric(df.get('closing_line', 0), errors='coerce').fillna(0)
    df['line_move'] = df['closing_line'] - df['line']
    
    # Coverage
    with_closing = (df['closing_line'] != 0).sum()
    print(f"\nCoverage: {with_closing}/{len(df)} bets ({with_closing/len(df)*100:.1f}%)")
    
    if with_closing == 0:
        print("❌ NO CLOSING LINES RECORDED YET")
        print("   → You need to populate closing_line field first")
        print("   → Use closing_line_cli.py to record real closing lines")
        return df
    
    # Movement analysis
    closed = df[df['closing_line'] != 0].copy()
    
    print(f"\n📊 LINE MOVEMENT (closed bets only):")
    print(f"  Mean movement: {closed['line_move'].mean():+.3f} pts")
    print(f"  Median movement: {closed['line_move'].median():+.3f} pts")
    print(f"  Std dev: {closed['line_move'].std():.3f} pts")
    print(f"  Min: {closed['line_move'].min():.1f} pts")
    print(f"  Max: {closed['line_move'].max():.1f} pts")
    print(f"  % with zero movement: {(closed['line_move'] == 0).mean()*100:.1f}%")
    
    # Red flags for closing lines
    print(f"\n🚩 SUSPICIOUS PATTERNS:")
    
    # All identical movement
    if closed['line_move'].std() < 0.01 and abs(closed['line_move'].mean()) < 0.01:
        print(f"  ⚠️  ALL lines have ZERO movement (0.0 pts)")
        print(f"      → Lines NEVER move? This is unrealistic for props.")
        print(f"      → Either all lines are synthetic/not real, or copied from opening lines.")
        print(f"      → Action: Verify closing_line values are ACTUAL market closing lines, not defaults.")
        
    # Clustered movement
    elif closed['line_move'].std() < 0.1:
        print(f"  ⚠️  Very low movement variance ({closed['line_move'].std():.3f} pts)")
        print(f"      → Lines cluster around {closed['line_move'].mean():.2f} pts movement")
        print(f"      → May indicate you're recording at similar times/conditions")
    
    else:
        print(f"  ✓ Lines show realistic variation ({closed['line_move'].std():.3f} pts std dev)")
    
    # Extreme movers
    extreme = closed[closed['line_move'].abs() > 2.0]
    if len(extreme) > 0:
        print(f"\n  ⚠️  {len(extreme)} bets with >2.0 pt movement:")
        for _, row in extreme.head(5).iterrows():
            print(f"      {row['player']} {row['market']} {row['side']}: "
                  f"{row['line']} → {row['closing_line']} ({row['line_move']:+.1f} pts)")
        if len(extreme) > 5:
            print(f"      ... and {len(extreme)-5} more")
    
    # Suspicious duplicates (identical opening + closing for same player/market)
    print(f"\n🔍 DUPLICATE DETECTION:")
    dups = closed[closed['line'] == closed['closing_line']]
    if len(dups) > 0:
        dup_pct = len(dups) / len(closed) * 100
        print(f"  {len(dups)} bets ({dup_pct:.1f}%) with identical open/close lines")
        
        if dup_pct > 80:
            print(f"  ⚠️  CRITICAL: {dup_pct:.0f}% of closed lines show ZERO movement!")
            print(f"      This suggests lines are not REAL closing lines from the market.")
            print(f"      → Are you copying opening lines instead of actual closing lines?")
            print(f"      → Closing_line should be the line WHEN THE BET SETTLED, not when placed.")
        
        # Show examples
        print(f"\n  Examples of zero-movement bets (may indicate copy errors):")
        for _, row in dups.head(10).iterrows():
            print(f"    {row['player']} {row['market']} {row['side']}: "
                  f"O/C {row['line']:.1f} (set {row['date'][:10]})")
    
    # Market breakdown
    print(f"\n📈 MOVEMENT BY MARKET:")
    for market in sorted(df['market'].unique()):
        m_closed = closed[closed['market'] == market]
        if len(m_closed) > 0:
            m_avg = m_closed['line_move'].mean()
            m_std = m_closed['line_move'].std()
            m_zeros = (m_closed['line_move'] == 0).sum()
            print(f"  {market:5} ({len(m_closed):3} bets): "
                  f"avg move={m_avg:+.2f}, std={m_std:.2f}, "
                  f"zeros={m_zeros} ({m_zeros/len(m_closed)*100:.0f}%)")
    
    # By side
    print(f"\n👁 MOVEMENT BY SIDE (OVER vs UNDER):")
    for side in ['OVER', 'UNDER']:
        s_closed = closed[closed['side'] == side]
        if len(s_closed) > 0:
            s_avg = s_closed['line_move'].mean()
            s_std = s_closed['line_move'].std()
            print(f"  {side:5}: avg move={s_avg:+.3f}, std={s_std:.3f}")
    
    # Time-based analysis
    print(f"\n⏰ RECORDING TIMING:")
    closed['date'] = pd.to_datetime(closed['date'], errors='coerce')
    if closed['date'].notna().any():
        date_range = (closed['date'].max() - closed['date'].min()).days
        print(f"  Date range: {closed['date'].min().date()} to {closed['date'].max().date()} ({date_range} days)")
        
        # Check for clustering (all recorded at once?)
        daily = closed.groupby(closed['date'].dt.date).size()
        if daily.std() > 10:
            print(f"  ⚠️  Uneven recording pattern (std dev: {daily.std():.0f} bets/day)")
            print(f"      Recorded on {len(daily)} days: {daily.min()}-{daily.max()} bets/day")
        else:
            print(f"  ✓ Spread across {len(daily)} days, fairly consistent")
    
    return df, closed


def check_for_typos(df: pd.DataFrame) -> List[Dict]:
    """Detect probable typos in closing lines."""
    print("\n" + "="*100)
    print("TYPO DETECTION")
    print("="*100)
    
    df['line'] = pd.to_numeric(df.get('line', 0), errors='coerce').fillna(0)
    df['closing_line'] = pd.to_numeric(df.get('closing_line', 0), errors='coerce').fillna(0)
    df['line_move'] = df['closing_line'] - df['line']
    
    closed = df[df['closing_line'] != 0].copy()
    if len(closed) == 0:
        print("No closing lines to check.")
        return []
    
    suspected = []
    
    # Rule 1: Closing line far from opening (>3.0 pts typically rare)
    far = closed[closed['line_move'].abs() > 3.0]
    if len(far) > 0:
        print(f"\n🚨 Lines moved >3.0 pts ({len(far)} bets):")
        print(f"   (Usually only happens for very volatile props or data entry errors)")
        for _, row in far.iterrows():
            print(f"   {row['player']} {row['market']}: {row['line']} → {row['closing_line']} "
                  f"({row['line_move']:+.1f})")
            suspected.append({
                'type': 'extreme_movement',
                'player': row['player'],
                'market': row['market'],
                'issue': f"Line moved {row['line_move']:.1f} pts (unusual)"
            })
    
    # Rule 2: Closing line == 0 (probably blank/missing)
    nulled = closed[closed['closing_line'] == 0]
    if len(nulled) > 0:
        print(f"\n❌ {len(nulled)} bets marked as closed but closer = 0 (missing/blank)")
        suspected.append({
            'type': 'missing_closing_line',
            'count': len(nulled),
            'issue': "Closing line not actually filled in"
        })
    
    # Rule 3: Closing line far outside reasonable range for typical markets
    # PTS typically move ±1-2 pts, REB ±0.5-1.5, etc.
    market_limits = {
        'PTS': 3.0,   # Points props rarely move >3
        'REB': 1.5,   # Rebounds rarely move >1.5
        'AST': 1.5,   # Assists rarely move >1.5
        'PA': 3.0,    # Points+Assists similar to PTS
        'PRA': 3.0,   # Points+Rebounds+Assists
    }
    
    for market, limit in market_limits.items():
        m_far = closed[(closed['market'] == market) & (closed['line_move'].abs() > limit)]
        if len(m_far) > 0:
            print(f"\n⚠️  {market} bets exceed typical move limit ({limit:.1f} pts): {len(m_far)} instances")
            for _, row in m_far.head(3).iterrows():
                print(f"   {row['player']}: {row['line']} → {row['closing_line']} ({row['line_move']:+.1f})")
    
    # Rule 4: Closing line = round number (might indicate placeholder or guess)
    rounded = closed[
        (closed['closing_line'] % 1 == 0) |
        (closed['closing_line'] % 0.5 == 0)
    ]
    print(f"\n⚠️  {len(rounded)} bets with closing line at .5 or .0 (may be rounded/estimated):")
    print(f"   (Real closing lines often have odd decimals like 28.7, not just 28.5 or 29.0)")
    
    if len(suspected) > 0:
        print(f"\n🚩 Total suspected issues: {len(suspected)}")
    
    return suspected


def recommend_validation_windows():
    """Suggest when to capture closing lines to avoid staleness."""
    print("\n" + "="*100)
    print("RECOMMENDED CLOSING LINE CAPTURE WINDOWS")
    print("="*100)
    
    windows = {
        "Perfect": "5-10 mins after game start (most accurate, minimal line movement)",
        "Good": "30 mins after game start (standard window, still fresh)",
        "Acceptable": "1-2 hours after game start (some line drift)",
        "Stale": "3+ hours after game start (significant line movement risk)",
        "Dead": "Next morning (line may have moved 2-3+ pts)",
    }
    
    print("\nCapture timing recommendations:")
    for quality, desc in windows.items():
        print(f"  {quality:12} ← {desc}")
    
    print("""
Strategy:
  1. If you bet RIGHT at game start → capture closing line 5-10 min later
  2. If you bet DURING game → capture 5-10 min after your actual placement
  3. If betting multiple times → record closing line for EACH bet separately
  
Why it matters:
  - If you wait too long, market may reprrice 2-3 pts away
  - Big moves indicate sharp money coming in (useful info for future models)  
  - Small/zero moves suggest your bet was "square" (public got same price)

Current tracker data suggests ALL bets have zero movement.
This is only realistic if:
  ✓ All bets were synthetic (for backtesting)
  ✓ You're filling in fake/placeholder lines
  
Fix: Record ACTUAL closing lines from your sportsbook, not opening line copies.
""")


def main():
    parser = argparse.ArgumentParser(description="Validate closing lines")
    parser.add_argument('--check-duplicates', action='store_true', help='Flag duplicate patterns')
    parser.add_argument('--check-typos', action='store_true', help='Detect probable typos')
    parser.add_argument('--market', help='Focus on one market (e.g., PTS)')
    parser.add_argument('--windows', action='store_true', help='Show recommended capture windows')
    args = parser.parse_args()
    
    df = load_tracker()
    
    if args.windows:
        recommend_validation_windows()
        return
    
    result = analyze_closing_lines(df)
    if isinstance(result, tuple):
        df, closed = result
    else:
        df = result
        closed = None
    
    if args.check_typos:
        check_for_typos(df)
    
    if args.market and closed is not None:
        print(f"\n{args.market} Market Analysis:")
        market_df = closed[closed['market'] == args.market]
        if len(market_df) == 0:
            print(f"No bets in {args.market} market with closing lines")
        else:
            print(market_df[['player', 'line', 'closing_line', 'side', 'result']].to_string(index=False))


if __name__ == "__main__":
    main()
