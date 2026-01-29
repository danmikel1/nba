import math
from nba_prediction import FeatureVector, ModelEngine


def make_feature_vector(market='PTS'):
    # Minimal but valid FeatureVector for testing variance prediction
    return FeatureVector(
        player_id=123,
        player_name='Test Player',
        opponent_abbrev='BOS',
        market=market,
        line=12.5,
        avg_minutes=25.0,
        ema=14.0,
        std=5.0,
        opponent_drtg_season=110.0,
        spread=0.0,
        game_total=220.0,
        days_rest=2,
        is_home=True,
        is_b2b=False,
        games_played=30
    )


def test_variance_model_feature_subset_respected():
    me = ModelEngine()
    fv = make_feature_vector('PTS')

    out = me.get_ml_regression_output(fv)
    assert out['is_regression'] is True
    assert out['has_model'] is True
    assert out['predicted_value'] is not None
    # predicted_std must be numeric and finite
    assert out['predicted_std'] is not None
    assert math.isfinite(out['predicted_std']) and out['predicted_std'] > 0


def test_variance_feature_selection_for_combo():
    me = ModelEngine()
    fv = make_feature_vector('PRA')
    out = me.get_ml_regression_output(fv)
    assert out['is_regression']
    assert math.isfinite(out['predicted_std']) and out['predicted_std'] > 0
