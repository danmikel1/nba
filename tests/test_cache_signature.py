import inspect
from nba_prediction import DataLoader


def test_cached_methods_use_underscore_self():
    """Ensure cached instance methods avoid hashing `self` by naming the first arg `_self`.
    This prevents Streamlit from attempting to hash the whole instance (which can be unhashable).
    """
    sig_fetch = inspect.signature(DataLoader.fetch_game_logs)
    params_fetch = list(sig_fetch.parameters.keys())
    assert params_fetch, "fetch_game_logs missing parameters in signature"
    assert params_fetch[0].startswith('_'), "fetch_game_logs must use an underscore-prefixed first argument (e.g., `_self`) to avoid Streamlit hashing `self`"

    sig_team = inspect.signature(DataLoader.fetch_team_stats)
    params_team = list(sig_team.parameters.keys())
    assert params_team, "fetch_team_stats missing parameters in signature"
    assert params_team[0].startswith('_'), "fetch_team_stats must use an underscore-prefixed first argument (e.g., `_self`) to avoid Streamlit hashing `self`"
