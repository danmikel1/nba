import numpy as np
import pandas as pd
from trading_engine import TradingEngine


def make_logvar_sigma(sigma):
    # returns log(1 + sigma^2)
    return np.log1p(sigma ** 2)


def test_sigma_recovery_and_default_fill():
    te = TradingEngine()
    df = pd.DataFrame({
        'predicted_mean': [20.0, 15.0],
        'predicted_std': [make_logvar_sigma(2.0), np.nan],  # first has sigma=2, second is missing
        'line': [22.0, 16.0],
        'side': ['OVER', 'OVER'],
        'odds': [1.91, 1.91],
        'feat_std': [1.5, 1.5]
    })

    out = te.process_bets(df, bankroll=1000)
    # Both rows should either be filtered or present depending on z
    # Ensure sigma column exists and is finite
    assert 'sigma' in out.columns
    assert (out['sigma'].notnull()).all()


def test_uncertainty_filter_multipliers():
    te = TradingEngine()
    # Case 1: high confidence -> z_score > 0.6 -> multiplier 1.0
    df1 = pd.DataFrame({
        'predicted_mean': [30.0],
        'predicted_std': [make_logvar_sigma(2.0)],  # sigma=2
        'line': [25.0],
        'side': ['OVER'],
        'odds': [1.91]
    })
    out1 = te.process_bets(df1, bankroll=1000, sigma_multiplier=1.0)
    assert len(out1) == 1
    assert out1.iloc[0]['stake_multiplier'] == 1.0

    # Case 2: mid confidence -> z in [0.35,0.6) -> multiplier 0.5
    # Use sigma large so z small
    # construct a mid-confidence case with z ~ 0.4: diff ~ 0.4 * sigma (sigma=10) => diff=4
    df2 = pd.DataFrame({
        'predicted_mean': [29.7],  # 29.7 - 25.7 = 4.0 -> z=0.4 when sigma=10
        'predicted_std': [make_logvar_sigma(10.0)],  # sigma=10
        'line': [25.7],
        'side': ['OVER'],
        'odds': [1.91]
    })
    out2 = te.process_bets(df2, bankroll=1000, sigma_multiplier=1.0)
    assert len(out2) == 1
    assert out2.iloc[0]['stake_multiplier'] == 0.5

    # Case 3: low confidence -> z < 0.35 -> filtered out (stake_multiplier==0)
    df3 = pd.DataFrame({
        'predicted_mean': [25.1],
        'predicted_std': [make_logvar_sigma(10.0)],
        'line': [25.0],
        'side': ['OVER'],
        'odds': [1.91]
    })
    out3 = te.process_bets(df3, bankroll=1000, sigma_multiplier=1.0)
    # should be filtered to 0 rows since stake_multiplier == 0
    assert len(out3) == 0


def test_pricing_score_sorting():
    te = TradingEngine()
    df = pd.DataFrame({
        'predicted_mean': [30.0, 28.0],
        'predicted_std': [make_logvar_sigma(2.0), make_logvar_sigma(2.0)],
        'line': [25.0, 25.0],
        'side': ['OVER', 'OVER'],
        'odds': [1.91, 1.91]
    })
    out = te.process_bets(df, bankroll=1000)
    # pricing_score should be descending
    scores = out['pricing_score'].values
    assert scores[0] >= scores[1]


def test_sigma_multiplier_reduces_confidence_and_stake():
    te = TradingEngine()
    # design a case where multiplier moves z from high-confidence bucket -> mid-confidence
    # base sigma = 2.0, diff = 1.6 -> z_base = 0.8 (stake_multiplier=1.0)
    # with multiplier=1.6 -> z_cal = 0.5 (stake_multiplier=0.5)
    df = pd.DataFrame({
        'predicted_mean': [26.6],
        'predicted_std': [make_logvar_sigma(2.0)],
        'line': [25.0],
        'side': ['OVER'],
        'odds': [1.91]
    })

    out_base = te.process_bets(df, bankroll=1000, sigma_multiplier=1.0)
    out_cal = te.process_bets(df, bankroll=1000, sigma_multiplier=1.6)

    assert len(out_base) == 1
    assert len(out_cal) == 1
    assert out_base.iloc[0]['stake_multiplier'] == 1.0
    assert out_cal.iloc[0]['stake_multiplier'] == 0.5
    # calibrated recommended stake should be lower
    assert out_cal.iloc[0]['rec_stake'] < out_base.iloc[0]['rec_stake']


def test_default_sigma_multiplier_is_1_6():
    te = TradingEngine()
    df = pd.DataFrame({
        'predicted_mean': [26.6],
        'predicted_std': [make_logvar_sigma(2.0)],
        'line': [25.0],
        'side': ['OVER'],
        'odds': [1.91]
    })

    out_default = te.process_bets(df, bankroll=1000)
    out_explicit = te.process_bets(df, bankroll=1000, sigma_multiplier=1.6)

    assert len(out_default) == 1
    assert len(out_explicit) == 1
    assert out_default.iloc[0]['stake_multiplier'] == out_explicit.iloc[0]['stake_multiplier']
    assert out_default.iloc[0]['rec_stake'] == out_explicit.iloc[0]['rec_stake']
