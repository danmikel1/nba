"""
closing_line_recorder.py — Lightweight integration for bet settlement

This module provides a simple function to call right after a bet settles.
No form, no UI—just a quick stdin prompt + auto-update.

Usage in your main pipeline:
    from closing_line_recorder import record_closing_line_for_bet
    
    # After you know a bet has settled:
    record_closing_line_for_bet(
        player="Luka Doncic",
        market="PTS",
        opening_line=28.5,
        side="OVER"
    )
    # Prompts: "Closing line for Luka Doncic PTS OVER (open 28.5)? "
    # You type: 28.5
    # Auto-saves to bet_tracker.json with CLV calculated
"""

import json
from pathlib import Path
from typing import Optional, Dict
import pandas as pd


TRACKER_PATH = Path(__file__).resolve().parent.parent / "data" / "bet_tracker.json"


def calculate_clv(opening_line: float, closing_line: float, side: str) -> float:
    """Calculate Closing Line Value (CLV) in points."""
    if pd.isna(opening_line) or pd.isna(closing_line):
        return 0.0
    delta = closing_line - opening_line
    if side.upper() in ['UNDER', 'U', '-']:
        delta = -delta
    return round(float(delta), 2)


def record_closing_line_for_bet(
    player: str,
    market: str,
    opening_line: float,
    side: str,
    date: Optional[str] = None,
) -> bool:
    """
    Record a closing line for a specific bet (interactive prompt).
    
    Args:
        player: Player name (e.g., "Luka Doncic")
        market: Market code (e.g., "PTS", "REB", "AST")
        opening_line: Original bet line
        side: "OVER" or "UNDER"
        date: Optional date filter (YYYY-MM-DD) if multiple bets same player/market
    
    Returns:
        True if saved successfully, False otherwise.
    
    Example:
        from closing_line_recorder import record_closing_line_for_bet
        record_closing_line_for_bet("Luka Doncic", "PTS", 28.5, "OVER")
    """
    
    if not TRACKER_PATH.exists():
        print(f"❌ Tracker not found: {TRACKER_PATH}")
        return False
    
    try:
        with open(TRACKER_PATH, 'r', encoding='utf-8') as f:
            tracker_data = json.load(f)
    except Exception as e:
        print(f"❌ Load failed: {e}")
        return False
    
    # Prompt for closing line
    market_upper = market.upper()
    side_upper = side.upper()
    prompt = (
        f"\n⚡ Closing line for {player} {market_upper} {side_upper} "
        f"(open {opening_line})? "
    )
    
    try:
        user_input = input(prompt).strip()
        if not user_input or user_input.lower() == 'skip':
            print("⏭️  Skipped.")
            return False
        
        closing_line = float(user_input)
    except ValueError:
        print(f"❌ Invalid input (not a number)")
        return False
    except KeyboardInterrupt:
        print("\n⏭️  Cancelled.")
        return False
    
    # Find matching bet in tracker
    matched = False
    for i, bet in enumerate(tracker_data):
        if (bet.get('player', '').lower() == player.lower() and
            bet.get('market', '').upper() == market_upper and
            abs(float(bet.get('line', 0)) - opening_line) < 0.01 and
            (date is None or bet.get('date', '').startswith(date))):
            
            # Update
            tracker_data[i]['closing_line'] = closing_line
            clv = calculate_clv(opening_line, closing_line, side)
            tracker_data[i]['clv'] = clv
            
            # Save
            try:
                with open(TRACKER_PATH, 'w', encoding='utf-8') as f:
                    json.dump(tracker_data, f, indent=2, ensure_ascii=False)
                
                result = "+" if clv > 0 else "-" if clv < 0 else "="
                print(f"  ✓ Saved! Closing: {closing_line:.1f} | CLV: {clv:+.2f} {result}")
                matched = True
                break
            except Exception as e:
                print(f"❌ Save failed: {e}")
                return False
    
    if not matched:
        print(f"⚠️  No matching bet found for {player} {market} {side}")
        print(f"   (Check player name / market / line in bet_tracker.json)")
        return False
    
    return True


def batch_record_from_dict(bets_dict: Dict) -> int:
    """
    Record multiple closing lines from a dict.
    
    Example:
        bets = {
            'Luka Doncic|PTS|OVER': 28.5,
            'Jayson Tatum|REB|OVER': 9.0,
        }
        batch_record_from_dict(bets)
    """
    updated = 0
    for key, closing_line in bets_dict.items():
        parts = key.split('|')
        if len(parts) != 3:
            print(f"⚠️  Skipping invalid key: {key}")
            continue
        
        player, market, side = parts
        # Need opening line — awkward, so skip this version
        print(f"⚠️  use CLI/CSV instead for bulk")
        break
    
    return updated


if __name__ == "__main__":
    # Quick test
    record_closing_line_for_bet("Test Player", "PTS", 25.5, "OVER")
