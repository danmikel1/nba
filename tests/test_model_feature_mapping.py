import logging
from nba_prediction import ModelEngine, FeatureVector


def test_model_expected_features_mapped_no_warnings(caplog):
    """Ensure _extract_features_for_model doesn't log unknown-feature warnings for model-expected features."""
    caplog.set_level(logging.WARNING)

    me = ModelEngine()
    # Minimal valid FeatureVector (only required positional args)
    fv = FeatureVector(
        player_id=0, player_name='TBD', opponent_abbrev='OPP', market='PTS',
        line=10.0, avg_minutes=20.0, ema=5.0, std=2.0, opponent_drtg_season=110.0,
        spread=0.0, game_total=210.0, days_rest=1, is_home=False, is_b2b=False, games_played=10
    )

    # Get the expected features from the model for a common market
    _, _, expected_features = me._get_model_for_market('PTS')

    # Capture any warnings emitted during feature extraction
    caplog.clear()
    _ = me._extract_features_for_model(fv, expected_features)

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any('Unknown feature requested by model' in w for w in warnings), (
        "Model expected features produced unknown-feature warnings: " + str(warnings)
    )
