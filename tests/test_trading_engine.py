import pandas as pd
import numpy as np
from trading_engine import TradingEngine


def test_process_bets_handles_none_std():
    te = TradingEngine()
    def logvar(s):
        return np.log1p(s**2)

    df = pd.DataFrame({
        'predicted_mean': [20.0, 15.0],
        'predicted_std': [None, logvar(2.0)],
        'line': [22.5, 13.0],
        'side': ['OVER', 'UNDER'],
        'odds': [1.91, 1.91]
    })

    out = te.process_bets(df, bankroll=1000)
    # ensure no NaNs in z_score or win_prob
    assert np.all(np.isfinite(out['z_score']))
    assert np.all(out['win_prob'].between(0.0, 1.0))


def test_z_score_threshold_filters():
    te = TradingEngine()
    def logvar(s):
        return np.log1p(s**2)

    df = pd.DataFrame({
        'predicted_mean': [10.0, 50.0],
        'predicted_std': [logvar(5.0), logvar(5.0)],
        'line': [11.0, 49.0],
        'side': ['OVER', 'UNDER'],
        'odds': [1.91, 1.91]
    })

    # require abs z >= 1.0
    out = te.process_bets(df, bankroll=1000, require_abs_z=True, z_score_threshold=1.0)
    # first has z_score = (11-10)/5 = 0.2 -> filtered out
    # second has z_score = (49-50)/5 = -0.2 abs 0.2 -> filtered out
    assert out['is_bet'].sum() == 0


def test_volatility_adjusted_ranking_orders():
    te = TradingEngine()
    def logvar(s):
        return np.log1p(s**2)

    df = pd.DataFrame({
        'predicted_mean': [30.0, 30.0],
        'predicted_std': [logvar(2.0), logvar(6.0)],
        'line': [25.0, 25.0],
        'side': ['OVER', 'OVER'],
        'odds': [1.91, 1.91]
    })

    out = te.process_bets(df, bankroll=1000)
    # the lower volatility (std=2) should have higher vol_adj_rank
    ranks = out['vol_adj_rank'].values
    assert ranks[0] >= ranks[1]