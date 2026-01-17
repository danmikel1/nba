import os
import sys
import pytest
import types

# Ensure project root is on sys.path so tests can import the application module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import nba_prediction as npred


class DummyFeatures:
    pass


def test_extract_all_present():
    f = DummyFeatures()
    # Set most attributes expected by TRAINING_FEATURE_COLUMNS
    f.avg_minutes = 28.5
    f.ema = 15.2
    f.std = 3.4
    f.opponent_drtg_season = 110.5
    f.line = 24.0
    f.spread = -5.5
    f.game_total = 220.0
    f.days_rest = 2
    f.is_home = True
    f.is_b2b = False
    f.games_played = 42
    f.opponent_pace = 98.0
    f.team_pace = 101.5
    f.trend_5g = 0.8
    f.home_avg = 20.0
    f.away_avg = 18.5
    f.feat_ts_pct = 0.57
    f.feat_ts_pct_delta = 0.02
    f.team_out_ppg = 6.5
    f.team_out_count = 1
    f.opp_out_ppg = 4.0
    f.opp_out_count = 0
    # Market identity should be provided by function via market argument
    f.feat_min_volatility = 1.2
    f.feat_foul_rate = 2.3
    f.feat_cv = 0.18

    out = npred.extract_features_dynamically(f, market='PTS')

    # Check a few representative fields
    assert out['feat_avg_minutes'] == float(f.avg_minutes)
    assert out['feat_ema'] == float(f.ema)
    assert out['feat_is_home'] == 1
    assert out['feat_is_b2b'] == 0
    assert out['feat_team_out_count'] == int(f.team_out_count)
    assert out['feat_ts_pct'] == float(f.feat_ts_pct)

    # Market identity
    assert out['feat_market_scoring'] == 1
    assert out['feat_market_counting'] == 0


def test_missing_features_fill_sentinels():
    f = DummyFeatures()
    # Only set a couple of attributes
    f.avg_minutes = 15.0
    f.is_home = False

    out = npred.extract_features_dynamically(f, market='REB')

    # Present attribute should be coerced
    assert out['feat_avg_minutes'] == float(15.0)
    # Missing float feature -> -1.0 sentinel
    assert out['feat_ema'] == -1.0
    # Missing count -> -1
    assert out['feat_team_out_count'] == -1
    # Boolean mapping -> 0
    assert out['feat_is_home'] == 0
    # Market identity for REB (counting)
    assert out['feat_market_counting'] == 1
    assert out['feat_market_scoring'] == 0


def test_coercion_fallback_on_bad_types():
    f = DummyFeatures()
    # Provide a bad type for a count
    f.team_out_count = "notanint"
    f.is_home = "also-not-bool"

    out = npred.extract_features_dynamically(f, market='PTS')

    # team_out_count should fall back to sentinel -1
    assert out['feat_team_out_count'] == -1
    # is_home should be interpreted as bool -> 1 for non-empty string
    # but our coercion expects bool casting; ensure we handle unexpected types consistently
    # According to implementation, is_home gets cast to bool then 1/0; non-empty string -> True -> 1
    assert out['feat_is_home'] == 1


if __name__ == '__main__':
    pytest.main([__file__])
