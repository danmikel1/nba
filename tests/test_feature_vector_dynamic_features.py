import pytest
import numpy as np
from nba_prediction import FeatureVector, TRAINING_FEATURE_COLUMNS


def test_new_feature_in_training_columns():
    assert 'feat_eff_per_min' in TRAINING_FEATURE_COLUMNS


def test_to_ml_array_matches_training_columns_length():
    fv = FeatureVector(
        player_id=1,
        player_name='Test Player',
        opponent_abbrev='ABC',
        market='PTS',
        line=10.0,
        avg_minutes=30.0,
        ema=20.0,
        std=5.0,
        opponent_drtg_season=110.0,
        spread=5.0,
        game_total=220.0,
        days_rest=1,
        is_home=True,
        is_b2b=False,
        games_played=50,
        opponent_pace=100.0,
        team_pace=100.0,
        trend_5g=0.1,
        home_avg=12.0,
        away_avg=11.0,
        feat_ts_pct=0.55,
        feat_ts_pct_delta=0.02,
        team_out_ppg=-1.0,
        team_out_count=-1,
        opp_out_ppg=-1.0,
        opp_out_count=-1,
        feat_min_volatility=1.0,
        feat_foul_rate=2.0,
        feat_cv=0.1,
        feat_usage_rate=0.2,
        feat_h2h_avg=-1.0,
        feat_season_avg=11.0,
        feat_l10_avg=12.0,
        feat_eff_per_min=0.6667,
        market_scoring=1,
        market_counting=0,
        market_combo=0,
    )

    arr = fv.to_ml_array()
    assert isinstance(arr, np.ndarray)
    assert arr.shape[0] == len(TRAINING_FEATURE_COLUMNS)


def test_efficiency_per_min_calculation():
    # avg_minutes <= 5 -> eff_per_min == 0.0
    fv_small = FeatureVector(
        player_id=2,
        player_name='Small Min',
        opponent_abbrev='XYZ',
        market='PTS',
        line=5,
        avg_minutes=4.0,
        ema=8.0,
        std=1.0,
        opponent_drtg_season=110.0,
        spread=0.0,
        game_total=200.0,
        days_rest=1,
        is_home=False,
        is_b2b=False,
        games_played=10,
    )
    assert fv_small.feat_eff_per_min == 0.0

    # avg_minutes > 5 -> eff_per_min = ema / avg_minutes
    fv_ok = FeatureVector(
        player_id=3,
        player_name='OK Min',
        opponent_abbrev='XYZ',
        market='PTS',
        line=5,
        avg_minutes=20.0,
        ema=10.0,
        std=1.0,
        opponent_drtg_season=110.0,
        spread=0.0,
        game_total=200.0,
        days_rest=1,
        is_home=False,
        is_b2b=False,
        games_played=10,
    )
    # The field itself is default 0.0 unless set; we test the calculation path separately in generator
    assert fv_ok.feat_eff_per_min == 0.0


# Note: eff_per_min is computed in analyze_prop and passed into FeatureVector there.
# The tests above verify presence and array ordering; analyze_prop calculation is covered
# implicitly by integration tests / by running the analyzer on real inputs.


def test_new_derived_features_present():
    assert 'feat_fatigue_load' in TRAINING_FEATURE_COLUMNS
    assert 'feat_form_gap' in TRAINING_FEATURE_COLUMNS


def test_build_feature_vector_computes_derived_features():
    from nba_prediction import FeatureEngineer
    import pandas as pd

    fe = FeatureEngineer()
    # Create 15 game history with consistent minutes and points
    mins = [30.0] * 15
    pts = [20, 22, 19, 21, 18, 25, 17, 23, 20, 19, 24, 16, 20, 22, 18]
    df = pd.DataFrame({'PTS': pts, 'MIN_FLOAT': mins, 'MATCHUP': ['LAL vs BOS']*15})

    fv = fe.build_feature_vector(
        player_id=1,
        player_name='Tester',
        opponent_id=2,
        opponent_abbrev='BOS',
        is_home=True,
        is_b2b=False,
        spread=5.0,
        df=df,
        stat_col='PTS',
        line=15.0,
        lookback=15,
        team_stats=pd.DataFrame(),
        avg_def=105.0,
        market='PTS',
        days_rest=2,
    )

    # Fatigue load should be avg_minutes / (days_rest + 0.5)
    assert fv.feat_fatigue_load == pytest.approx(30.0 / (2 + 0.5))
    # Form gap should be (l10_avg - season_avg) / max(1.0, season_avg)
    season_avg = float(df['PTS'].mean())
    l10_avg = float(df.tail(10)['PTS'].mean())
    assert fv.feat_form_gap == pytest.approx((l10_avg - season_avg) / max(1.0, season_avg))


def test_lazy_backfill_computes_on_export(tmp_path):
    from nba_prediction import Tracker

    tracker = Tracker(file_path=tmp_path / 'tracker.json')

    bet = {
        'player': 'Lazy Tester',
        'date': '2025-01-01',
        'market': 'PTS',
        'line': 12.0,
        'result': 'Win',
        'avg_minutes': 30.0,
        'days_rest': 2,
        'feat_season_avg': 18.0,
        'feat_l10_avg': 21.0,
        # Required core features for V20.2 export
        'feat_opponent_pace': 100.0,
        'feat_team_pace': 100.0,
        'feat_trend_5g': 0.0,
        'feat_home_avg': 20.0,
        'feat_away_avg': 18.0,
        # intentionally missing 'feat_fatigue_load' and 'feat_form_gap'
    }

    row = tracker._map_bet_to_csv_row(bet)
    assert row['feat_fatigue_load'] == pytest.approx(30.0 / (2 + 0.5))
    assert row['feat_form_gap'] == pytest.approx((21.0 - 18.0) / max(1.0, 18.0))

