# -*- coding: utf-8 -*-
"""Re-analyze pending bets with calibrated models and batch through Risk Engine."""
import json
import sys
from pathlib import Path
from datetime import datetime
import os
import traceback

# Force UTF-8 output
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

sys.path.insert(0, str(Path(__file__).parent))

import warnings
warnings.filterwarnings('ignore')

from nba_prediction import PredictionOrchestrator, CONFIG, TradingEngine
import pandas as pd
import numpy as np


def main():
    # ── Load pending bets ────────────────────────────────────────────────
    with open('data/bet_tracker.json', encoding='utf-8') as f:
        all_bets = json.load(f)

    pending = [b for b in all_bets if b.get('result') == 'Pending']

    if not pending:
        print("No pending bets found.")
        return

    print("=" * 100)
    print("  BATCH RE-ANALYSIS  (Calibrated Models + Risk Engine)")
    print("=" * 100)
    print(f"\n  Pending bets: {len(pending)}")
    print(f"  Timestamp:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    orchestrator = PredictionOrchestrator(CONFIG)
    trading = TradingEngine()

    # ── Phase 1: Re-run each bet through the calibrated model ────────────
    batch_rows = []       # rows for TradingEngine DataFrame
    comparison = []       # for before/after comparison output
    failed = []

    for i, bet in enumerate(pending, 1):
        player   = bet.get('player', 'Unknown')
        market   = bet.get('market', 'PTS')
        line     = float(bet.get('line', 0))
        odds     = float(bet.get('odds', 1.91))
        opponent = bet.get('opponent', 'Unknown')
        orig_side  = bet.get('side', 'OVER')
        orig_ev    = float(bet.get('ev', 0))
        orig_prob  = float(bet.get('prob', 0.5))
        orig_stake = float(bet.get('stake', 0))
        bet_id     = bet['id']

        print(f"  [{i}/{len(pending)}] {player} {market} {orig_side} {line} vs {opponent} ... ", end="", flush=True)

        try:
            result_obj = orchestrator.run_analysis(
                player_name=player,
                opponent_name=opponent,
                market=market,
                line=line,
                odds=odds,
                is_home=bool(bet.get('feat_is_home', bet.get('is_home', True))),
                is_b2b=bool(bet.get('feat_is_b2b', bet.get('is_b2b', False))),
                lookback=15,
                spread=float(bet.get('feat_spread', bet.get('spread', 0))),
                bankroll=1500,
                days_rest=int(bet.get('feat_days_rest', bet.get('days_rest', 1))),
                game_total=float(bet.get('feat_game_total', bet.get('game_total', 225)))
            )

            if not result_obj or not result_obj.success:
                print("FAILED (no result)")
                failed.append({'id': bet_id, 'player': player, 'reason': 'analysis returned None'})
                continue

            decision = result_obj.decision
            features = result_obj.features

            new_side = decision.recommended_side
            new_ev   = decision.expected_value
            new_prob = decision.probability
            new_pred = decision.predicted_mean
            new_std  = features.std if hasattr(features, 'std') else 3.0
            feat_std = features.feat_std if hasattr(features, 'feat_std') else new_std

            # Build row for batch processing
            batch_rows.append({
                'id':             bet_id,
                'player':         player,
                'market':         market,
                'line':           line,
                'odds':           odds,
                'predicted_mean': new_pred,
                'predicted_std':  new_std,
                'feat_std':       feat_std,
                'prob':           new_prob,
                'side':           new_side,
                'opponent':       opponent,
            })

            comparison.append({
                'id':          bet_id,
                'player':      player,
                'market':      market,
                'line':        line,
                'orig_side':   orig_side,
                'new_side':    new_side,
                'side_flip':   orig_side != new_side,
                'orig_prob':   orig_prob,
                'new_prob':    new_prob,
                'orig_ev':     orig_ev,
                'new_ev':      new_ev,
                'orig_stake':  orig_stake,
                'new_pred':    new_pred,
            })

            tag = "FLIP!" if orig_side != new_side else "ok"
            print(f"done  Prob {orig_prob:.1%}->{new_prob:.1%}  EV {orig_ev:+.1%}->{new_ev:+.1%}  [{tag}]")

        except Exception as e:
            print(f"ERROR: {e}")
            failed.append({'id': bet_id, 'player': player, 'reason': str(e)})

    if not batch_rows:
        print("\nAll analyses failed. Nothing to process.")
        return

    # ── Phase 2: Batch through Risk Engine ──────────────────────────────
    print(f"\n{'='*100}")
    print("  RISK ENGINE  (Batch Processing)")
    print(f"{'='*100}\n")

    df = pd.DataFrame(batch_rows)
    bankroll = 1500.0
    processed = trading.process_bets(df, bankroll, return_all=True)

    approved = processed[processed['is_bet'] == True].copy()
    rejected = processed[processed['is_bet'] == False].copy()

    print(f"  Bankroll:  ${bankroll:,.0f}")
    print(f"  Analyzed:  {len(processed)}")
    print(f"  Approved:  {len(approved)}")
    print(f"  Rejected:  {len(rejected)}")

    # ── Display approved bets ──
    if not approved.empty:
        print(f"\n  {'─'*96}")
        print(f"  APPROVED BETS")
        print(f"  {'─'*96}")
        print(f"  {'Player':<22} {'Market':<6} {'Side':<6} {'Line':>6} {'Odds':>6} {'Prob':>7} {'EV':>8} {'QS':>7} {'Stake':>8}")
        print(f"  {'─'*96}")
        for _, row in approved.iterrows():
            print(f"  {row['player']:<22} {row['market']:<6} {row['side']:<6} {row['line']:>6.1f} {row['odds']:>6.2f} {row['predicted_prob']:>6.1%} {row['predicted_ev']:>+7.1%} {row['quality_score']:>7.3f} ${row['rec_stake']:>6.0f}")
        total_stake = approved['rec_stake'].sum()
        print(f"  {'─'*96}")
        print(f"  {'TOTAL':>62} ${total_stake:>6.0f}")

    # ── Display rejected bets ──
    if not rejected.empty:
        print(f"\n  {'─'*96}")
        print(f"  REJECTED BETS")
        print(f"  {'─'*96}")
        print(f"  {'Player':<22} {'Market':<6} {'Side':<6} {'Line':>6} {'Odds':>6} {'Prob':>7} {'EV':>8} {'QS':>7} {'Reason'}")
        print(f"  {'─'*96}")
        for _, row in rejected.iterrows():
            reason = "Neg EV" if row['predicted_ev'] <= 0 else "Low QS" if row['quality_score'] < 0.015 else "Cap (Top 5)"
            print(f"  {row['player']:<22} {row['market']:<6} {row['side']:<6} {row['line']:>6.1f} {row['odds']:>6.2f} {row['predicted_prob']:>6.1%} {row['predicted_ev']:>+7.1%} {row['quality_score']:>7.3f} {reason}")

    # ── Phase 3: Before / After comparison ──────────────────────────────
    print(f"\n{'='*100}")
    print("  BEFORE vs AFTER COMPARISON")
    print(f"{'='*100}")

    comp_df = pd.DataFrame(comparison)
    # Merge in risk engine results
    risk_cols = processed[['id', 'predicted_ev', 'predicted_prob', 'quality_score', 'rec_stake', 'is_bet']].copy()
    comp_df = comp_df.merge(risk_cols, on='id', how='left')

    flips = comp_df[comp_df['side_flip'] == True]
    if not flips.empty:
        print(f"\n  SIDE FLIPS ({len(flips)}):")
        for _, r in flips.iterrows():
            print(f"    {r['player']}: {r['orig_side']} -> {r['new_side']}  (prob {r['orig_prob']:.1%} -> {r['new_prob']:.1%})")

    ev_changes = comp_df[abs(comp_df['new_ev'] - comp_df['orig_ev']) > 0.02]
    if not ev_changes.empty:
        print(f"\n  SIGNIFICANT EV CHANGES ({len(ev_changes)}):")
        for _, r in ev_changes.iterrows():
            delta = r['new_ev'] - r['orig_ev']
            arrow = "UP" if delta > 0 else "DN"
            print(f"    {r['player']}: EV {r['orig_ev']:+.1%} -> {r['new_ev']:+.1%} ({arrow} {abs(delta):.1%})")

    # ── Phase 4: Update bet_tracker.json ────────────────────────────────
    print(f"\n{'='*100}")
    print("  UPDATING BET TRACKER")
    print(f"{'='*100}\n")

    # Build lookup from risk engine results
    update_lookup = {}
    for _, row in processed.iterrows():
        update_lookup[row['id']] = {
            'predicted_mean': float(row['predicted_mean']),
            'predicted_std':  float(row['predicted_std']),
            'prob':           float(row['predicted_prob']),
            'ev':             float(row['predicted_ev']),
            'side':           row['side'],
            'stake':          float(row['rec_stake']),
            'quality_score':  float(row['quality_score']),
            'recalibrated':   True,
            'recalibrated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

    updated_count = 0
    for bet in all_bets:
        if bet['id'] in update_lookup:
            upd = update_lookup[bet['id']]
            old_side  = bet.get('side', '?')
            old_stake = float(bet.get('stake', 0))

            bet['predicted_mean'] = upd['predicted_mean']
            bet['predicted_std']  = upd['predicted_std']
            bet['prob']           = upd['prob']
            bet['ev']             = upd['ev']
            bet['side']           = upd['side']
            bet['stake']          = upd['stake']
            bet['quality_score']  = upd['quality_score']
            bet['recalibrated']   = True
            bet['recalibrated_at'] = upd['recalibrated_at']

            status = ""
            if old_side != upd['side']:
                status += f" SIDE:{old_side}->{upd['side']}"
            if abs(old_stake - upd['stake']) > 0.5:
                status += f" STAKE:${old_stake:.0f}->${upd['stake']:.0f}"
            if not status:
                status = " (values updated)"

            print(f"  Updated #{bet['id']} {bet['player']}:{status}")
            updated_count += 1

    # Save
    with open('data/bet_tracker.json', 'w', encoding='utf-8') as f:
        json.dump(all_bets, f, indent=2, ensure_ascii=False)

    print(f"\n  {updated_count} bets updated in bet_tracker.json")

    # ── Failed bets ──
    if failed:
        print(f"\n  FAILED ({len(failed)}):")
        for f_bet in failed:
            print(f"    #{f_bet['id']} {f_bet['player']}: {f_bet['reason']}")

    print(f"\n{'='*100}")
    print("  DONE - Bets are ready for the Risk Engine in the app")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
