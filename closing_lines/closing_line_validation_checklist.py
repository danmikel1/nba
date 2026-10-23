"""
closing_line_validation_checklist.py — Interactive validation as you record closing lines

Use this WHILE recording closing lines (or right after) to validate accuracy.
Prevents common errors:
  • Swapped OVER/UNDER
  • Typos (28.5 typed as 285)
  • Stale lines (recorded too late)
  • Copy-paste from opening line
  • Unrealistic movement

Usage:
    python closing_line_validation_checklist.py [--player "Name"] [--market PTS]
    
Example:
    python closing_line_validation_checklist.py --player "Luka Doncic" --market PTS
    
    or just:
    python closing_line_validation_checklist.py  # then type info when prompted
"""

import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import sys


TRACKER_PATH = Path(__file__).resolve().parent.parent / "data" / "bet_tracker.json"


def load_recent_bets(player: Optional[str] = None, market: Optional[str] = None, n: int = 10) -> pd.DataFrame:
    """Load recent bets for validation context."""
    with open(TRACKER_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    # Filter
    if player:
        df = df[df['player'].str.contains(player, case=False, na=False)]
    if market:
        df = df[df['market'].str.upper() == market.upper()]
    
    # Recent first
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date', ascending=False).head(n)
    
    return df


def validate_closing_line(player: str, market: str, opening_line: float, side: str, 
                          closing_line: float, time_recorded: Optional[str] = None) -> Dict:
    """
    Validate a closing line entry. Returns dict with:
    {
        'valid': bool,
        'confidence': 'high' | 'medium' | 'low',
        'issues': [list of warnings],
        'suggestions': [list of fixes]
    }
    """
    result = {
        'valid': True,
        'confidence': 'high',
        'issues': [],
        'suggestions': []
    }
    
    side = side.upper()
    
    print("\n" + "="*80)
    print(f"VALIDATING CLOSING LINE: {player} {market} {side}")
    print("="*80)
    
    # ========== CHECK 1: Range sanity ==========
    movement = closing_line - opening_line
    
    # Market-specific limits
    limits = {
        'PTS': {'normal': 1.5, 'max': 3.0},
        'REB': {'normal': 1.0, 'max': 2.0},
        'AST': {'normal': 1.0, 'max': 2.0},
        'PA': {'normal': 1.5, 'max': 3.0},
        'PRA': {'normal': 1.5, 'max': 3.5},
    }
    
    market_upper = market.upper()
    limit_info = limits.get(market_upper, {'normal': 1.5, 'max': 3.0})
    
    if abs(movement) > limit_info['max']:
        result['issues'].append(
            f"Movement {movement:+.1f} pts exceeds typical max ({limit_info['max']:.1f} pts)"
        )
        result['confidence'] = 'low'
        result['suggestions'].append(
            f"Double-check: Did line really move {movement:+.1f} pts? "
            f"({opening_line:.1f} → {closing_line:.1f})"
        )
    elif abs(movement) > limit_info['normal']:
        result['issues'].append(
            f"Large movement {movement:+.1f} pts (typical: {limit_info['normal']:.1f} pts)"
        )
        result['confidence'] = 'medium'
    
    # ========== CHECK 2: Typo detection ==========
    # Off by decimal point (e.g., 28.5 → 285)
    if abs(closing_line / opening_line - 1) > 5:
        result['issues'].append(
            f"Closing line {closing_line} is 5x+ different from opening {opening_line}"
        )
        result['suggestions'].append(
            f"Is this a decimal point typo? Did you mean {closing_line/10:.1f}?"
        )
        result['valid'] = False
    
    # Zero closing line
    if closing_line == 0:
        result['issues'].append("Closing line is 0 (missing/blank)")
        result['valid'] = False
        result['suggestions'].append("Enter the actual closing line from your sportsbook")
    
    # ========== CHECK 3: Exact match to opening (likely copy error) ==========
    if movement == 0:
        result['issues'].append("No movement: closing_line == opening_line")
        result['suggestions'].append(
            "⚠️  Did you accidentally copy the opening line? "
            "Check your sportsbook for the ACTUAL closing line when bet settled."
        )
        result['confidence'] = 'low'
    
    # ========== CHECK 4: Same side realism ==========
    # For OVER bets, if market agreed (line moved up), CLV > 0
    # For UNDER bets, if market agreed (line moved down), CLV < 0
    
    if side == 'OVER':
        if movement < -1.0:
            result['issues'].append(
                f"You bet OVER but line moved DOWN {movement:+.1f} pts "
                f"(market disagreed with you)"
            )
            result['confidence'] = 'low' if result['confidence'] == 'low' else 'medium'
    elif side == 'UNDER':
        if movement > 1.0:
            result['issues'].append(
                f"You bet UNDER but line moved UP {movement:+.1f} pts "
                f"(market disagreed with you)"
            )
            result['confidence'] = 'low' if result['confidence'] == 'low' else 'medium'
    
    # ========== CHECK 5: Recording time sanity ==========
    if time_recorded:
        # Rough heuristic: if bet was placed early morning (6 AM) and recorded next day = very stale
        print(f"\n⏰ Recorded at: {time_recorded}")
        print("   → Ideal: 5-30 mins after game start")
        print("   ⚠️  If >2 hours late: line may have drifted significantly")
    
    # ========== CHECK 6: Context from recent bets ==========
    recent = load_recent_bets(player=player, market=market, n=5)
    if len(recent) > 0:
        print(f"\n📚 Recent {player} {market} bets:")
        for i, (_, bet) in enumerate(recent.head(3).iterrows()):
            open_line = bet.get('line')
            close_line = bet.get('closing_line', 0)
            move = close_line - open_line if close_line != 0 else 0
            marker = "← CURRENT" if i == 0 else ""
            print(f"   {bet['side']:5} {open_line:5.1f} → {close_line:5.1f} ({move:+.1f}) {marker}")
        
        # Check if this closing line is realistic for this player/market
        avg_move = (recent['closing_line'] - recent['line']).mean()
        if abs(movement - avg_move) > 1.5:
            result['issues'].append(
                f"Movement {movement:+.1f} differs significantly from this player's average {avg_move:+.1f}"
            )
            result['suggestions'].append(
                f"Not necessarily wrong, but unusual. Double-check the line entered."
            )
    
    # ========== FINAL REPORT ==========
    print("\n" + "-"*80)
    print(f"VALIDATION RESULT: {result['confidence'].upper()}")
    print("-"*80)
    
    if result['issues']:
        print("\n⚠️  ISSUES DETECTED:")
        for i, issue in enumerate(result['issues'], 1):
            print(f"  {i}. {issue}")
    
    if result['suggestions']:
        print("\n💡 SUGGESTIONS:")
        for i, sugg in enumerate(result['suggestions'], 1):
            print(f"  {i}. {sugg}")
    
    if result['valid'] and result['confidence'] == 'high':
        print("\n✅ VALID: Closing line looks good!")
    elif result['valid'] and result['confidence'] == 'medium':
        print("\n✓ ACCEPTABLE: Closing line is plausible, but verify if possible")
    else:
        print("\n❌ INVALID or SUSPICIOUS: Please review before saving")
    
    return result


def interactive_checklist():
    """Interactive mode: walk through validation step by step."""
    print("\n" + "="*80)
    print("CLOSING LINE VALIDATION CHECKLIST")
    print("="*80)
    
    # Gather info
    print("\nEnter closing line details:")
    
    player = input("Player name: ").strip()
    if not player:
        print("Cancelled.")
        return
    
    market = input("Market (PTS/REB/AST/PA/PRA): ").strip().upper()
    if market not in ['PTS', 'REB', 'AST', 'PA', 'PRA']:
        print(f"⚠️  Unknown market: {market} (continuing anyway)")
    
    side = input("Side (OVER/UNDER): ").strip().upper()
    if side not in ['OVER', 'UNDER']:
        print(f"Invalid side: {side}")
        return
    
    try:
        opening = float(input("Opening line (e.g., 28.5): "))
        closing = float(input("Closing line (e.g., 29.0): "))
    except ValueError:
        print("Invalid number entered")
        return
    
    time = input("Time recorded (optional): ").strip() or None
    
    # Validate
    result = validate_closing_line(player, market, opening, side, closing, time)
    
    # Offer to save
    while True:
        if result['valid'] and result['confidence'] == 'high':
            save = input("\nSave? (y/n): ").strip().lower()
        elif result['valid']:
            save = input("\nSave anyway? (y/n): ").strip().lower()
        else:
            save = input("\nFix and try again? (y/n/save): ").strip().lower()
        
        if save == 'y':
            print("✅ Would save (integration with closing_line_cli.py needed)")
            break
        elif save == 'n':
            print("✓ Not saved.")
            break
        else:
            print("Enter y or n")


def main():
    parser = argparse.ArgumentParser(description="Validate closing lines")
    parser.add_argument('--player', help='Player name')
    parser.add_argument('--market', help='Market (PTS, REB, AST, PA, PRA)')
    parser.add_argument('--opening', type=float, help='Opening line')
    parser.add_argument('--closing', type=float, help='Closing line')
    parser.add_argument('--side', help='OVER or UNDER')
    args = parser.parse_args()
    
    if all([args.player, args.market, args.opening, args.closing, args.side]):
        validate_closing_line(args.player, args.market, args.opening, 
                              args.side, args.closing)
    else:
        interactive_checklist()


if __name__ == "__main__":
    main()
