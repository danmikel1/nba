import re
from nba_prediction import Config


def test_cache_ttls_and_season_format():
    # TTLs
    assert Config.CACHE_TTL_PLAYER_IDS == 3600, "CACHE_TTL_PLAYER_IDS should be 3600 (1 hour)"
    assert Config.CACHE_TTL_GAME_LOGS == 600, "CACHE_TTL_GAME_LOGS should be 600 (10 mins)"
    # Season format
    pattern = re.compile(r'^\d{4}-\d{2}$')
    assert pattern.match(Config.CURRENT_SEASON), f"CURRENT_SEASON has unexpected format: {Config.CURRENT_SEASON}"
    assert pattern.match(Config.PREV_SEASON), f"PREV_SEASON has unexpected format: {Config.PREV_SEASON}"
    assert isinstance(Config.TRAINING_SEASONS, tuple) and len(Config.TRAINING_SEASONS) >= 3
