#!/usr/bin/env python3
"""
Reanalyze Pending Bets with V21.4 AI Analyst

Goes through all pending bets in bet_tracker.json and re-evaluates them
with the new V21.4 AI analyst. FADEs are removed from the tracker.

Usage:
    python scripts/reanalyze_pending_bets.py              # Dry run (show what would happen)
    python scripts/reanalyze_pending_bets.py --execute    # Actually delete FADEs
"""

import sys
import json
import time
import argparse
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nba_prediction import get_ai_second_opinion

BET_TRACKER_PATH = Path(__file__).parent.parent / "data" / "bet_tracker.json"


def load_pending_bets():
    """Load all pending bets from tracker."""
    with open(BET_TRACKER_PATH, 'r') as f:
        bets = json.load(f)
    
    pending = [b for b in bets if b.get('result', 'Pending') == 'Pending']
    return bets, pending


def build_prop_analysis(bet: dict) -> dict:
    """Convert bet tracker entry to prop_analysis format for AI analyst."""
    return {
        'player': bet.get('player', 'Unknown'),
        'position': '',  # Not stored in tracker, but AI can work without
        'opponent': bet.get('opponent', ''),
        'market': bet.get('market', 'PTS'),
        'line': bet.get('line', 0),
        'projected': bet.get('proj', bet.get('predicted_mean', 0)),
        'projected_value': bet.get('proj', bet.get('predicted_mean', 0)),
        'prob': bet.get('prob', 0),
        'ev': bet.get('ev', 0),
        'predicted_side': bet.get('side', 'OVER'),
        'signal': bet.get('side', 'OVER'),
        'spread': bet.get('feat_spread', 0),
        
        # Features (pass through all feat_* fields)
        **{k: v for k, v in bet.items() if k.startswith('feat_')}
    }


def parse_verdict(ai_response: str) -> tuple[str, str, int]:
    """
    Parse AI analyst response.
    Returns: (verdict, key_feature, confidence)
    """
    verdict = "UNKNOWN"
    key_feature = ""
    confidence = 5
    
    for line in ai_response.split('\n'):
        line = line.strip()
        if line.startswith('VERDICT:'):
            v = line.replace('VERDICT:', '').strip().upper()
            if 'TAKE' in v:
                verdict = 'TAKE'
            elif 'FADE' in v:
                verdict = 'FADE'
        elif line.startswith('KEY FEATURE:'):
            key_feature = line.replace('KEY FEATURE:', '').strip()
        elif line.startswith('CONFIDENCE:'):
            try:
                confidence = int(''.join(c for c in line if c.isdigit())[:2])
            except:
                confidence = 5
    
    return verdict, key_feature, confidence


def main():
    parser = argparse.ArgumentParser(description='Reanalyze pending bets with V21.4 AI Analyst')
    parser.add_argument('--execute', action='store_true', help='Actually delete FADEs (default: dry run)')
    parser.add_argument('--min-confidence', type=int, default=6, help='Minimum confidence to act on FADE (default: 6)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("V21.4 AI ANALYST - PENDING BET REANALYSIS")
    print("=" * 80)
    print(f"Mode: {'🔴 EXECUTE (will delete FADEs)' if args.execute else '🟡 DRY RUN (preview only)'}")
    print(f"Min Confidence for FADE: {args.min_confidence}")
    print()
    
    all_bets, pending = load_pending_bets()
    print(f"📊 Total Bets: {len(all_bets)}")
    print(f"⏳ Pending Bets: {len(pending)}")
    print()
    
    if not pending:
        print("✅ No pending bets to analyze.")
        return
    
    results = {'TAKE': [], 'FADE': [], 'UNKNOWN': []}
    
    for i, bet in enumerate(pending, 1):
        player = bet.get('player', 'Unknown')
        market = bet.get('market', 'PTS')
        line = bet.get('line', 0)
        side = bet.get('side', 'OVER')
        prob = bet.get('prob', 0)
        ev = bet.get('ev', 0)
        bet_id = bet.get('id', 0)
        
        print(f"[{i}/{len(pending)}] Analyzing: {player} {market} {side} {line}...")
        
        # Build prop analysis and get AI opinion
        prop_analysis = build_prop_analysis(bet)
        backtest_json = {'opponent': bet.get('opponent', ''), 'spread': bet.get('feat_spread', 0)}
        
        try:
            ai_response = get_ai_second_opinion(prop_analysis, backtest_json)
            verdict, key_feature, confidence = parse_verdict(ai_response)
        except Exception as e:
            print(f"   ⚠️ Error: {e}")
            verdict = "UNKNOWN"
            key_feature = "Error"
            confidence = 0
        
        # Store result
        result_entry = {
            'id': bet_id,
            'player': player,
            'market': market,
            'side': side,
            'line': line,
            'prob': prob,
            'ev': ev,
            'verdict': verdict,
            'key_feature': key_feature,
            'confidence': confidence
        }
        results[verdict].append(result_entry)
        
        # Print result
        emoji = '✅' if verdict == 'TAKE' else '❌' if verdict == 'FADE' else '❓'
        print(f"   {emoji} {verdict} (Confidence: {confidence}) - {key_feature}")
        
        # Rate limit for Groq API (1.5s to avoid 429s)
        time.sleep(1.5)
    
    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ TAKE: {len(results['TAKE'])} bets")
    print(f"❌ FADE: {len(results['FADE'])} bets")
    print(f"❓ UNKNOWN: {len(results['UNKNOWN'])} bets")
    print()
    
    # Filter FADEs by confidence
    high_confidence_fades = [f for f in results['FADE'] if f['confidence'] >= args.min_confidence]
    low_confidence_fades = [f for f in results['FADE'] if f['confidence'] < args.min_confidence]
    
    if high_confidence_fades:
        print(f"🎯 HIGH-CONFIDENCE FADEs (>= {args.min_confidence}): {len(high_confidence_fades)}")
        for f in high_confidence_fades:
            print(f"   - {f['player']} {f['market']} {f['side']} {f['line']} (Conf: {f['confidence']}, Prob: {f['prob']*100:.1f}%)")
        print()
    
    if low_confidence_fades:
        print(f"⚠️ LOW-CONFIDENCE FADEs (< {args.min_confidence}): {len(low_confidence_fades)} - KEEPING")
        for f in low_confidence_fades:
            print(f"   - {f['player']} {f['market']} {f['side']} {f['line']} (Conf: {f['confidence']})")
        print()
    
    # Execute deletions
    if args.execute and high_confidence_fades:
        fade_ids = {f['id'] for f in high_confidence_fades}
        original_count = len(all_bets)
        
        # Filter out FADEs
        all_bets = [b for b in all_bets if b.get('id') not in fade_ids]
        
        # Save
        with open(BET_TRACKER_PATH, 'w') as f:
            json.dump(all_bets, f, indent=2)
        
        print(f"🗑️ DELETED {original_count - len(all_bets)} high-confidence FADEs from bet_tracker.json")
    elif high_confidence_fades and not args.execute:
        print(f"💡 Run with --execute to delete {len(high_confidence_fades)} high-confidence FADEs")
    
    # Save detailed report
    report_path = Path(__file__).parent.parent / "data" / "pending_reanalysis_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 Detailed report saved: {report_path}")


if __name__ == "__main__":
    main()
