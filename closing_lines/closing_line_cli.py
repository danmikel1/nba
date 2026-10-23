"""
closing_line_cli.py — Fast CLI for Closing Line Entry

No UI overhead—just terminal input. Perfect for quick batch entry.

Usage:
    python closing_line_cli.py              # Interactive mode
    python closing_line_cli.py --csv file.csv  # Batch from CSV
    python closing_line_cli.py --bets 5     # Show top 5 open bets, prompt

Example CSV format:
    player,market,line,side,closing_line
    Luka Doncic,PTS,28.5,OVER,28.5
    Jayson Tatum,REB,8.5,OVER,9.0
"""

import json
import argparse
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


TRACKER_PATH = Path(__file__).resolve().parent.parent / "data" / "bet_tracker.json"


def load_tracker() -> List[Dict]:
    """Load bet_tracker.json."""
    if not TRACKER_PATH.exists():
        print(f"❌ Tracker not found: {TRACKER_PATH}")
        sys.exit(1)
    with open(TRACKER_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_tracker(data: List[Dict]) -> bool:
    """Save bet_tracker.json."""
    try:
        with open(TRACKER_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"❌ Save failed: {e}")
        return False


def calculate_clv(opening_line: float, closing_line: float, side: str) -> float:
    """Calculate Closing Line Value (CLV) in points."""
    if pd.isna(opening_line) or pd.isna(closing_line):
        return 0.0
    delta = closing_line - opening_line
    if side.upper() in ['UNDER', 'U']:
        delta = -delta
    return round(float(delta), 2)


def get_open_bets(data: List[Dict]) -> List[Dict]:
    """Get list of open bets (closing_line missing or zero)."""
    open_bets = [
        b for b in data
        if not b.get('closing_line') or pd.to_numeric(b.get('closing_line', 0), errors='coerce') == 0
    ]
    # Sort by date (newest first)
    open_bets = sorted(open_bets, key=lambda x: x.get('date', ''), reverse=True)
    return open_bets


def display_bet(bet: Dict, index: int = None) -> str:
    """Format bet for display."""
    prefix = f"[{index}]" if index is not None else ""
    return (
        f"{prefix} {bet.get('date', 'N/A')[:10]} | "
        f"{bet.get('player', 'N/A'):15} | "
        f"{bet.get('market', 'N/A'):3} {bet.get('side', 'N/A'):5} | "
        f"Line: {bet.get('line', 'N/A'):6} | "
        f"Odds: {bet.get('odds', 'N/A'):.2f}" if isinstance(bet.get('odds'), (int, float)) else f"Odds: {bet.get('odds', 'N/A')}"
    )


def interactive_mode(tracker_data: List[Dict]):
    """Interactive mode: show open bets, prompt for closing lines."""
    open_bets = get_open_bets(tracker_data)
    
    if not open_bets:
        print("✅ All bets have closing lines!")
        return
    
    print(f"\n📋 Open Bets ({len(open_bets)} total):")
    print("-" * 120)
    
    for i, bet in enumerate(open_bets[:20]):  # Show top 20
        print(display_bet(bet, i))
    
    if len(open_bets) > 20:
        print(f"... and {len(open_bets) - 20} more")
    
    print("\n" + "=" * 120)
    print("Quick Entry: type '[index] [closing_line]' (e.g., '0 28.5')")
    print("Type 'q' to quit, 'skip' to skip current, or 'all [line]' to set all to same line.")
    print("=" * 120)
    
    updated = 0
    for i, bet in enumerate(open_bets):
        opening = bet.get('line', 'N/A')
        
        while True:
            user_input = input(f"\n[{i}] {bet.get('player', 'N/A')} {bet.get('market', '')}/{bet.get('side', '')} "
                              f"(Open→{opening}): ").strip()
            
            if user_input.lower() == 'q':
                print(f"✅ Updated {updated} bets. Saving...")
                if save_tracker(tracker_data):
                    print("✅ Saved!")
                return
            
            if user_input.lower() == 'skip':
                break
            
            try:
                closing = float(user_input)
                # Find and update the bet
                for j, b in enumerate(tracker_data):
                    if (b.get('player') == bet.get('player') and
                        b.get('line') == bet.get('line') and
                        b.get('date') == bet.get('date') and
                        b.get('market') == bet.get('market')):
                        
                        tracker_data[j]['closing_line'] = closing
                        clv_val = calculate_clv(opening, closing, bet.get('side', 'OVER'))
                        tracker_data[j]['clv'] = clv_val
                        
                        result = "+" if clv_val > 0 else "-" if clv_val < 0 else "="
                        print(f"  ✓ Closing: {closing:.1f} | CLV: {clv_val:+.2f} {result}")
                        updated += 1
                        break
                break
            except ValueError:
                print("  ❌ Invalid input. Enter a number (e.g., 28.5)")
    
    print(f"\n✅ Updated {updated} bets. Saving...")
    if save_tracker(tracker_data):
        print("✅ Saved!")


def csv_mode(csv_file: str, tracker_data: List[Dict]):
    """Batch mode: load CSV and apply."""
    try:
        csv_df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"❌ CSV load failed: {e}")
        sys.exit(1)
    
    required_cols = ['player', 'market', 'line', 'side', 'closing_line']
    missing = [c for c in required_cols if c not in csv_df.columns]
    if missing:
        print(f"❌ Missing columns: {missing}")
        sys.exit(1)
    
    print(f"\n📥 Loaded {len(csv_df)} rows from {csv_file}")
    print(csv_df.to_string(index=False))
    
    confirm = input("\n⚠️  Apply all? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Aborted.")
        return
    
    updated = 0
    for _, row in csv_df.iterrows():
        player = row['player'].strip()
        market = row['market'].strip().upper()
        line = pd.to_numeric(row['line'], errors='coerce')
        closing = pd.to_numeric(row['closing_line'], errors='coerce')
        side = row['side'].strip().upper()
        
        if pd.isna(closing) or pd.isna(line):
            continue
        
        # Match in tracker
        for i, bet in enumerate(tracker_data):
            if (bet.get('player', '').strip().lower() == player.lower() and
                bet.get('market', '').strip().upper() == market and
                pd.to_numeric(bet.get('line', np.nan), errors='coerce') == line):
                
                tracker_data[i]['closing_line'] = closing
                clv_val = calculate_clv(line, closing, side)
                tracker_data[i]['clv'] = clv_val
                updated += 1
                print(f"  ✓ {player} {market} {side}: {closing:.1f} (CLV {clv_val:+.2f})")
                break
    
    print(f"\n✅ Matched {updated}/{len(csv_df)} rows.")
    if save_tracker(tracker_data):
        print("✅ Saved!")


def main():
    parser = argparse.ArgumentParser(description="Closing Line Entry CLI")
    parser.add_argument('--csv', help='Load closing lines from CSV')
    parser.add_argument('--bets', type=int, help='Show top N open bets')
    args = parser.parse_args()
    
    tracker_data = load_tracker()
    
    if args.csv:
        csv_mode(args.csv, tracker_data)
    elif args.bets:
        open_bets = get_open_bets(tracker_data)[:args.bets]
        print(f"\n📋 Top {min(args.bets, len(open_bets))} Open Bets:")
        print("-" * 120)
        for i, bet in enumerate(open_bets):
            print(display_bet(bet, i))
    else:
        interactive_mode(tracker_data)


if __name__ == "__main__":
    main()
