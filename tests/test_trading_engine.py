import pandas as pd
import numpy as np
from trading_engine import TradingEngine


def test_process_bets_handles_none_std():
    """Verify process_bets handles missing predicted_std gracefully."""
    te = TradingEngine()
    df = pd.DataFrame({
        'predicted_mean': [20.0, 15.0],
        'predicted_std': [None, 3.0],
        'line': [22.5, 13.0],
        'side': ['OVER', 'UNDER'],
        'odds': [1.91, 1.91],
        'prob': [0.65, 0.70],
    })

    out = te.process_bets(df, bankroll=1000, return_all=True)
    # ensure no NaNs in sigma or predicted_prob
    assert np.all(np.isfinite(out['sigma']))
    assert np.all(out['predicted_prob'].between(0.0, 1.0))
    # sigma should be at least the hard floor (1.0)
    assert (out['sigma'] >= 1.0).all()


def test_z_score_threshold_filters():
    """Bets with tiny edge should be filtered out by QS threshold."""
    te = TradingEngine()
    df = pd.DataFrame({
        'predicted_mean': [10.0, 50.0],
        'predicted_std': [5.0, 5.0],
        'line': [11.0, 49.0],
        'side': ['OVER', 'UNDER'],
        'odds': [1.91, 1.91],
        'prob': [0.52, 0.53],  # tiny edge → low EV → low QS
    })

    out = te.process_bets(df, bankroll=1000, return_all=True)
    # Both have very low prob → low/negative EV → below MIN_EV gate → no bets funded
    assert out['is_bet'].sum() == 0


def test_kelly_scales_stake_with_edge():
    """Higher edge gets proportionally larger stake via Kelly."""
    te = TradingEngine()
    bankroll = 10000.0  # larger bankroll so stakes don't both hit the 6% cap
    df = pd.DataFrame({
        'player': ['Star', 'Marginal'],
        'predicted_mean': [30.0, 22.0],
        'predicted_std': [4.0, 4.0],
        'line': [25.0, 20.0],
        'side': ['OVER', 'OVER'],
        'odds': [1.91, 1.91],
        'prob': [0.56, 0.55],  # Star has bigger edge; more likely to earn within the QS cap
    })

    out = te.process_bets(df, bankroll=bankroll, return_all=True)

    star = out[out['player'] == 'Star'].iloc[0]
    marginal = out[out['player'] == 'Marginal'].iloc[0]
    # Both should be funded, Star with more money
    assert star['is_bet']
    assert marginal['is_bet']
    assert star['rec_stake'] > marginal['rec_stake']


def test_max_stake_pct_enforced():
    """Every funded bet must be <= MAX_STAKE_PCT * bankroll."""
    te = TradingEngine()
    bankroll = 10000.0
    df = pd.DataFrame({
        'predicted_mean': [80.0, 80.0, 80.0],
        'predicted_std': [1.0, 1.0, 1.0],
        'line': [35.0, 35.0, 35.0],
        'side': ['OVER', 'OVER', 'OVER'],
        'odds': [2.0, 2.0, 2.0],
        'prob': [0.99, 0.99, 0.99],
    })
    out = te.process_bets(df, bankroll=bankroll, return_all=True)
    assert (out['rec_stake'] <= 0.03 * bankroll + 1e-6).all()


def test_opponent_exposure_limit():
    """Max 2 bets against the same opponent."""
    te = TradingEngine()
    df = pd.DataFrame({
        'player': ['A', 'B', 'C'],
        'opponent': ['BOS', 'BOS', 'BOS'],
        'predicted_mean': [30.0, 28.0, 26.0],
        'predicted_std': [3.0, 3.0, 3.0],
        'line': [24.0, 22.0, 20.0],
        'side': ['OVER', 'OVER', 'OVER'],
        'odds': [1.91, 1.91, 1.91],
        'prob': [0.72, 0.71, 0.70],  # all high-edge
    })

    out = te.process_bets(df, bankroll=1000, return_all=True)
    funded = out[out['is_bet'] == True]
    # Should be at most 2 bets against BOS
    assert len(funded) <= 2


def test_min_ev_gate():
    """Bets below MIN_EV_THRESHOLD should be rejected even if QS > threshold."""
    te = TradingEngine()
    df = pd.DataFrame({
        'predicted_mean': [20.0],
        'predicted_std': [0.5],   # Extremely low sigma → inflated QS
        'line': [19.5],
        'side': ['OVER'],
        'odds': [1.50],           # Low odds → low EV despite decent prob
        'prob': [0.55],           # EV = 0.55*0.50 - 0.45 = -0.175 → negative
    })

    out = te.process_bets(df, bankroll=1000, return_all=True)
    # Negative EV → should not be funded
    assert out.iloc[0]['is_bet'] == False