import pandas as pd
import pytest
from nba_prediction import BulkGameLogLoader


def make_player_df(player_id, team_id, dates_pts_name):
    """Helper: build a DataFrame with GAME_DATE (pd.Timestamp), TEAM_ID, PTS, PLAYER_NAME"""
    rows = []
    for d, pts, name in dates_pts_name:
        rows.append({'GAME_DATE': pd.Timestamp(d), 'TEAM_ID': team_id, 'PTS': pts, 'PLAYER_NAME': name})
    return pd.DataFrame(rows)


def test_ghost_player_cutoff_filters_old_players():
    loader = BulkGameLogLoader()
    loader._loaded = True
    team_id = 1610612737

    # Dates
    game_date = pd.Timestamp('2025-12-01')
    recent_date = (game_date - pd.Timedelta(days=10)).strftime('%Y-%m-%d')
    old_date = (game_date - pd.Timedelta(days=100)).strftime('%Y-%m-%d')
    played_today = game_date.strftime('%Y-%m-%d')

    # Player 1: active recently (should be counted as out if not playing today)
    loader._cache[1] = make_player_df(1, team_id, [ (recent_date, 20, 'Active Player') for _ in range(6)])
    # Player 2: last played long ago (should be treated as ghost and NOT counted)
    loader._cache[2] = make_player_df(2, team_id, [ (old_date, 15, 'Old Player') for _ in range(10)])
    # Player 3: played today (should NOT be counted)
    loader._cache[3] = make_player_df(3, team_id, [ (played_today, 12, 'Starter') ])
    # Player 4: low sample size (games < 5) should be skipped
    loader._cache[4] = make_player_df(4, team_id, [ (recent_date, 8, 'Low Games') for _ in range(2)])

    # Build roster index manually and set indexes_built to True to avoid full index build
    loader._roster_index[team_id] = {1, 2, 3, 4}
    loader._indexes_built = True

    # Populate the daily index: only player 3 played on the game_date
    key = f"{game_date.strftime('%Y-%m-%d')}_{team_id}"
    loader._daily_index[key] = {3}

    # Call compute_absence_features
    res = loader.compute_absence_features(game_date, player_team_id=team_id, opponent_team_id=999, player_id=3)

    # Player 1 should be counted (ppg ~20), player 2 excluded (old), player 4 excluded (games<5)
    assert res['team_out_count'] == 1
    assert pytest.approx(res['team_out_ppg'], rel=1e-3) == 20.0
    # Opponent had no roster -> counts should be 0 or sentinel
    assert res['opp_out_count'] == 0
    assert res['opp_out_ppg'] == 0.0
