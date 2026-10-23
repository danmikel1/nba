"""
closing_line_data_quality.py — Quick assessment of closing line data quality

Run this to get a YES/NO answer: Is my closing line data REAL or SYNTHETIC?

Usage:
    python closing_line_data_quality.py         # Quick score (0-100)
    python closing_line_data_quality.py --detailed  # Full report
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple


TRACKER_PATH = Path(__file__).resolve().parent.parent / "data" / "bet_tracker.json"


def assess_data_quality() -> Tuple[int, Dict]:
    """
    Score closing line data quality (0-100).
    100 = definitely real data
    0 = definitely synthetic
    """
    with open(TRACKER_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df['line'] = pd.to_numeric(df.get('line', 0), errors='coerce').fillna(0)
    df['closing_line'] = pd.to_numeric(df.get('closing_line', 0), errors='coerce').fillna(0)
    
    closed = df[df['closing_line'] != 0].copy()
    
    if len(closed) == 0:
        return 0, {'reason': 'No closing lines at all'}
    
    # Initialize scoring
    score = 0
    details = {}
    
    # ========== SIGNAL 1: Movement Variance (40 points) ==========
    moves = closed['closing_line'] - closed['line']
    variance = moves.std()
    
    if variance > 1.0:
        score += 40
        details['movement_variance'] = f"✓ {variance:.3f} pts (real data has variance)"
    elif variance > 0.3:
        score += 25
        details['movement_variance'] = f"~ {variance:.3f} pts (some variance, may be real)"
    elif variance > 0.01:
        score += 10
        details['movement_variance'] = f"⚠️  {variance:.6f} (very low variance, likely synthetic)"
    else:
        score += 0
        details['movement_variance'] = f"❌ {variance:.10f} (zero variance, definitely synthetic)"
    
    # ========== SIGNAL 2: Movement Distribution (25 points) ==========
    zero_pct = (moves == 0).mean() * 100
    
    realistic_zero_pct = 20  # Expect 20-30% to genuinely have no movement
    
    if 15 <= zero_pct <= 35:
        score += 25
        details['zero_movement_rate'] = f"✓ {zero_pct:.1f}% static (realistic)"
    elif 5 <= zero_pct < 15 or 35 < zero_pct <= 50:
        score += 15
        details['zero_movement_rate'] = f"~ {zero_pct:.1f}% static (acceptable)"
    elif zero_pct > 80:
        score += 0
        details['zero_movement_rate'] = f"❌ {zero_pct:.1f}% static (almost all copied)"
    else:
        score += 5
        details['zero_movement_rate'] = f"⚠️  {zero_pct:.1f}% static"
    
    # ========== SIGNAL 3: Movement Range (15 points) ==========
    move_range = moves.max() - moves.min()
    
    if move_range > 3.0:
        score += 15
        details['movement_range'] = f"✓ {move_range:.1f} pts (full range)"
    elif move_range > 1.5:
        score += 10
        details['movement_range'] = f"~ {move_range:.1f} pts (moderate range)"
    elif move_range > 0.5:
        score += 5
        details['movement_range'] = f"⚠️  {move_range:.1f} pts (narrow range)"
    else:
        score += 0
        details['movement_range'] = f"❌ {move_range:.1f} pts (no real variation)"
    
    # ========== SIGNAL 4: Movement by Market Type (10 points) ==========
    market_consistency = True
    for market in closed['market'].unique():
        m_moves = closed[closed['market'] == market]['closing_line'] - closed[closed['market'] == market]['line']
        if m_moves.std() < 0.1:
            market_consistency = False
    
    if market_consistency:
        score += 10
        details['market_variation'] = "✓ Different markets show different movement patterns"
    else:
        score += 0
        details['market_variation'] = "❌ All markets show identical patterns (likely synthetic)"
    
    # ========== SIGNAL 5: Decimal Patterns (10 points) ==========
    closing_decimals = (closed['closing_line'] % 1).value_counts()
    decimal_diversity = len(closing_decimals)
    
    # Real closing lines have varied decimals (.3, .7, .2, etc.)
    # Synthetic often have .0 and .5 only
    if decimal_diversity > 5:
        score += 10
        details['decimal_patterns'] = f"✓ {decimal_diversity} different decimal positions (realistic)"
    elif decimal_diversity > 2:
        score += 5
        details['decimal_patterns'] = f"~ {decimal_diversity} decimal patterns (somewhat realistic)"
    else:
        score += 0
        details['decimal_patterns'] = f"❌ {decimal_diversity} decimals (likely .0 and .5 only)"
    
    return score, details


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Check closing line data quality")
    parser.add_argument('--detailed', action='store_true', help='Show full breakdown')
    args = parser.parse_args()
    
    score, details = assess_data_quality()
    
    print("\n" + "="*80)
    print("CLOSING LINE DATA QUALITY ASSESSMENT")
    print("="*80)
    
    # Grade
    if score >= 80:
        grade = "✅ REAL DATA"
        interpretation = "Closing lines appear genuine (from actual market)"
    elif score >= 60:
        grade = "🟡 MIXED/QUESTIONABLE"
        interpretation = "Some real patterns, but may have synthetic/fake entries"
    elif score >= 40:
        grade = "⚠️  MOSTLY SYNTHETIC"
        interpretation = "Most or all closing lines appear to be defaults/copies"
    else:
        grade = "❌ DEFINITELY SYNTHETIC"
        interpretation = "Closing lines are NOT real market data"
    
    print(f"\nDATA QUALITY SCORE: {score}/100")
    print(f"VERDICT: {grade}")
    print(f"INTERPRETATION: {interpretation}")
    
    if args.detailed:
        print("\nDetailed Breakdown:")
        for key, value in details.items():
            print(f"  {key:25} {value}")
    
    # Recommendation
    print("\n" + "-"*80)
    if score >= 80:
        print("✅ GOOD: Your data quality is acceptable for analysis")
        print("   Next: Run python analyze_performance.py to use CLV insights")
    elif score >= 60:
        print("⚠️  CAUTION: Data has quality issues")
        print("   Action: Use closing_line_validator.py to find/fix problematic entries")
    else:
        print("❌ CRITICAL: Data appears synthetic or poorly populated")
        print("   Action: Use closing_line_cli.py to record REAL closing lines")
        print("   Then re-run this tool to verify improvement")


if __name__ == "__main__":
    main()
