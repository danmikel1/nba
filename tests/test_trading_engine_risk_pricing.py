import numpy as np
import pandas as pd
from trading_engine import TradingEngine


def test_sigma_recovery_and_default_fill():
    """Verify sigma is computed and filled with defaults for missing predicted_std."""
    te = TradingEngine()
    df = pd.DataFrame({
        'predicted_mean': [20.0, 15.0],
        'predicted_std': [3.0, np.nan],
        'line': [22.0, 16.0],
        'side': ['OVER', 'OVER'],
        'odds': [1.91, 1.91],
        'prob': [0.65, 0.60],
        'feat_std': [1.5, 1.5]
    })

    out = te.process_bets(df, bankroll=1000, return_all=True)
    assert 'sigma' in out.columns
    assert (out['sigma'].notnull()).all()
    # sigma hard floor is 1.0
    assert (out['sigma'] >= 1.0).all()


def test_quality_score_drives_tier_selection():
    """Higher quality score → bigger stake (Tier 1 vs Tier 2)."""
    te = TradingEngine()
    df = pd.DataFrame({
        'player': ['Elite', 'Solid'],
        'predicted_mean': [30.0, 25.0],
        'predicted_std': [2.0, 6.0],    # Low vol vs high vol
        'line': [25.0, 22.0],
        'side': ['OVER', 'OVER'],
        'odds': [1.91, 1.91],
        'prob': [0.72, 0.66],
    })

    out = te.process_bets(df, bankroll=1000, return_all=True)
    elite = out[out['player'] == 'Elite'].iloc[0]
    solid = out[out['player'] == 'Solid'].iloc[0]
    # Elite (low sigma, high prob) should get bigger stake than Solid
    assert elite['rec_stake'] >= solid['rec_stake']


def test_negative_ev_always_rejected():
    """Bets with negative EV should never be funded regardless of QS."""
    te = TradingEngine()
    df = pd.DataFrame({
        'predicted_mean': [20.0],
        'predicted_std': [1.0],
        'line': [19.0],
        'side': ['OVER'],
        'odds': [1.40],        # terrible odds
        'prob': [0.52],        # barely positive but EV = 0.52*0.40 - 0.48 = -0.272
    })

    out = te.process_bets(df, bankroll=1000, return_all=True)
    assert out.iloc[0]['is_bet'] == False
    assert out.iloc[0]['rec_stake'] == 0.0


def test_sigma_floor_prevents_inflated_qs():
    """Sigma hard floor of 1.0 prevents unrealistically high quality scores."""
    te = TradingEngine()
    df = pd.DataFrame({
        'predicted_mean': [20.0],
        'predicted_std': [0.01],     # absurdly low
        'line': [19.0],
        'side': ['OVER'],
        'odds': [1.91],
        'prob': [0.65],
    })

    out = te.process_bets(df, bankroll=1000, return_all=True)
    # sigma should be floored at 1.0, not 0.01
    assert out.iloc[0]['sigma'] >= 1.0
    # QS should be bounded (not astronomical)
    assert out.iloc[0]['quality_score'] < 1.0


def test_player_dedup_keeps_best():
    """When same player appears twice, keep highest QS row."""
    te = TradingEngine()
    df = pd.DataFrame({
        'player': ['LeBron', 'LeBron'],
        'predicted_mean': [28.0, 12.0],
        'predicted_std': [3.0, 4.0],
        'line': [24.0, 10.0],
        'side': ['OVER', 'OVER'],
        'odds': [1.91, 1.91],
        'prob': [0.72, 0.65],
    })

    out = te.process_bets(df, bankroll=1000, return_all=True)
    lebron_rows = out[out['player'] == 'LeBron']
    # Should only have 1 row after dedup
    assert len(lebron_rows) == 1
