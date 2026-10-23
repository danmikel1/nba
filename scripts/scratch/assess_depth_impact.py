"""
Assess impact of depth_rank bug fixes on pending UNDER bets.
For players now identified as starters (depth=1), recalculate projections
to see how much worse the UNDER bets have become.
"""

import json
import sys
import pickle
import numpy as np
from pathlib import Path

def load_model():
    """Load the variance model"""
    model_dir = Path('data/variance_sweep_models')
    model_file = model_dir / 'variance_model_feat_cv_feat_min_volatility_feat_ema_feat_avg_minutes_feat_usage_rate_feat_opponent_pace_feat_team_pace.feats.csv'
    
    # Find the pkl file
    pkl_files = list(model_dir.glob('*.pkl'))
    if not pkl_files:
        raise FileNotFoundError("No model .pkl file found")
    
    with open(pkl_files[0], 'rb') as f:
        model = pickle.load(f)
    
    return model

def recalculate_projection(bet, new_depth_rank, model):
    """Recalculate projection with corrected depth_rank"""
    
    # Build feature vector (variance model uses specific features)
    features = np.array([[
        bet['feat_cv'],
        bet['feat_min_volatility'],
        bet['feat_ema'],
        bet['feat_avg_minutes'],
        bet['feat_usage_rate'],
        bet['feat_opponent_pace'],
        bet['feat_team_pace']
    ]])
    
    # Predict new variance
    new_variance = model.predict(features)[0]
    new_std = np.sqrt(new_variance)
    
    # For mean prediction, depth_rank affects it
    # Rough estimate: starters (depth=1) perform ~10-15% better than rotation (depth=2)
    depth_multiplier = 1.12 if new_depth_rank == 1 else 1.0
    old_multiplier = 1.12 if bet['feat_depth_rank'] == 1 else 1.0
    
    # Adjust mean based on depth change
    adjustment_ratio = depth_multiplier / old_multiplier
    new_mean = bet['predicted_mean'] * adjustment_ratio
    
    return new_mean, new_std

def calculate_ev(mean, std, line, side, odds):
    """Calculate EV for a bet"""
    from scipy.stats import norm
    
    if side == 'UNDER':
        prob_win = norm.cdf(line, loc=mean, scale=std)
    else:  # OVER
        prob_win = 1 - norm.cdf(line, loc=mean, scale=std)
    
    ev = prob_win * (odds - 1) - (1 - prob_win)
    return ev, prob_win

def main():
    # Load bet tracker
    with open('data/bet_tracker.json', 'r') as f:
        bets = json.load(f)
    
    # Load model
    try:
        model = load_model()
        use_model = True
    except Exception as e:
        print(f"Warning: Could not load model ({e}). Using simple adjustment.")
        use_model = False
    
    # Players affected by depth_rank bug
    affected_players = {
        'De\'Anthony Melton': {'old_depth': 2, 'new_depth': 1, 'reason': 'Curry OUT - promoted to starting PG'},
        'Moses Moody': {'old_depth': 2, 'new_depth': 1, 'reason': 'Actually starting SF'},
        'Al Horford': {'old_depth': 2, 'new_depth': 1, 'reason': 'Porzingis OUT - promoted to starting C'}
    }
    
    print("=" * 80)
    print("DEPTH_RANK BUG IMPACT ASSESSMENT")
    print("=" * 80)
    print()
    
    total_ev_loss = 0
    affected_bets = []
    
    for bet in bets:
        if (bet.get('result') == 'Pending' and 
            bet.get('stake', 0) > 0 and 
            bet.get('player') in affected_players):
            
            player_info = affected_players[bet['player']]
            
            # Simple adjustment: starters perform ~12% better than rotation
            # This affects the mean projection
            old_depth = bet['feat_depth_rank']
            new_depth = player_info['new_depth']
            
            if old_depth == new_depth:
                continue  # No change
            
            # Estimate new projection
            depth_boost = 1.12 if new_depth == 1 else (1/1.12)
            new_mean = bet['predicted_mean'] * depth_boost
            std = bet['predicted_std']
            
            # Recalculate EV with new projection
            from scipy.stats import norm
            line = bet['line']
            side = bet['side']
            odds = bet['odds']
            
            if side == 'UNDER':
                old_prob = norm.cdf(line, loc=bet['predicted_mean'], scale=std)
                new_prob = norm.cdf(line, loc=new_mean, scale=std)
            else:
                old_prob = 1 - norm.cdf(line, loc=bet['predicted_mean'], scale=std)
                new_prob = 1 - norm.cdf(line, loc=new_mean, scale=std)
            
            old_ev = old_prob * (odds - 1) - (1 - old_prob)
            new_ev = new_prob * (odds - 1) - (1 - new_prob)
            
            ev_change = new_ev - old_ev
            ev_change_pct = (ev_change / old_ev * 100) if old_ev != 0 else 0
            
            total_ev_loss += ev_change
            
            print(f"\n{'='*80}")
            print(f"Player: {bet['player']}")
            print(f"Reason: {player_info['reason']}")
            print(f"{'='*80}")
            print(f"Market: {bet['market']} {side} {line} @ {odds}")
            print(f"Current Stake: ${bet['stake']:.2f}")
            print()
            print(f"DEPTH RANK: {old_depth} → {new_depth}")
            print()
            print(f"OLD PROJECTION (with depth={old_depth}):")
            print(f"  Mean: {bet['predicted_mean']:.2f}")
            print(f"  Std:  {std:.2f}")
            print(f"  Win Prob: {old_prob:.1%}")
            print(f"  EV: {old_ev:+.3f} ({old_ev*100:+.1f}%)")
            print()
            print(f"NEW PROJECTION (with depth={new_depth}):")
            print(f"  Mean: {new_mean:.2f} ({(new_mean - bet['predicted_mean']):+.2f})")
            print(f"  Std:  {std:.2f}")
            print(f"  Win Prob: {new_prob:.1%} ({(new_prob - old_prob)*100:+.1f}%)")
            print(f"  EV: {new_ev:+.3f} ({new_ev*100:+.1f}%)")
            print()
            print(f"IMPACT:")
            print(f"  EV Change: {ev_change:+.3f} ({ev_change_pct:+.1f}%)")
            
            if side == 'UNDER':
                print(f"  Player now projects {new_mean:.1f} (was {bet['predicted_mean']:.1f})")
                if new_mean > bet['predicted_mean']:
                    print(f"  ⚠️  UNDER bet is now WORSE (player will score more)")
            else:
                print(f"  Player now projects {new_mean:.1f} (was {bet['predicted_mean']:.1f})")
                if new_mean > bet['predicted_mean']:
                    print(f"  ✓ OVER bet is now BETTER (player will score more)")
            
            # Recommendation
            print()
            print("RECOMMENDATION:")
            if new_ev < -0.05:
                print(f"  🚫 CLOSE THIS BET - Now -EV ({new_ev*100:.1f}%)")
            elif new_ev < 0:
                print(f"  ⚠️  Consider closing - Slightly -EV ({new_ev*100:.1f}%)")
            elif ev_change < -0.1:
                print(f"  ⚠️  Significantly degraded - Lost {abs(ev_change)*100:.1f}% EV")
            else:
                print(f"  ✓ Still positive EV but monitor closely")
            
            affected_bets.append({
                'player': bet['player'],
                'side': side,
                'line': line,
                'stake': bet['stake'],
                'old_ev': old_ev,
                'new_ev': new_ev,
                'ev_change': ev_change
            })
    
    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total affected bets: {len(affected_bets)}")
    print(f"Total EV loss: {total_ev_loss:+.3f} ({total_ev_loss*100:+.1f}%)")
    print()
    
    # Check for hedge opportunities
    print("=" * 80)
    print("HEDGE ANALYSIS")
    print("=" * 80)
    print()
    print("To hedge these UNDER bets, you would need to bet OVER on the same lines.")
    print("However, this requires:")
    print("  1. OVER lines available at favorable odds")
    print("  2. Sufficient margin to profit on both sides")
    print()
    print("Current situation:")
    for bet_info in affected_bets:
        if bet_info['side'] == 'UNDER':
            print(f"  {bet_info['player']}: UNDER {bet_info['line']} @ risk ${bet_info['stake']:.2f}")
            print(f"    → Need OVER {bet_info['line']} to hedge")
            print(f"    → New EV with depth=1: {bet_info['new_ev']*100:+.1f}%")
    
    print()
    print("🔍 Check your sportsbook for OVER lines on these players to hedge.")

if __name__ == '__main__':
    try:
        from scipy.stats import norm
        main()
    except ImportError:
        print("Error: scipy not installed. Run: pip install scipy")
        sys.exit(1)
