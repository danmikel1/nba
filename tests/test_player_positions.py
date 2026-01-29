import pandas as pd
import pytest
from nba_prediction import DataLoader, CONFIG
import nba_prediction as npred


def test_fetch_player_positions_uses_per_season(monkeypatch, tmp_path):
    # Mock players.get_players to be empty (so we only rely on per-season)
    monkeypatch.setattr(npred.players, 'get_players', lambda: [])

    # Mock LeagueDashPlayerStats to return a df with PLAYER_ID and PLAYER_POSITION
    class Dummy:
        def __init__(self, *args, **kwargs):
            pass
        def get_data_frames(self):
            df = pd.DataFrame([{'PLAYER_ID': 1, 'PLAYER_POSITION': 'G'}])
            return [df]

    monkeypatch.setattr(npred.leaguedashplayerstats, 'LeagueDashPlayerStats', Dummy)

    loader = DataLoader(CONFIG)
    positions = loader.fetch_player_positions(seasons=['2025-26'], force_refresh=True)

    assert positions.get(1) in ('PG', 'SG', 'G', 'PG'), "Position should be normalized/available"


def test_fetch_player_positions_fallback_to_per_player(monkeypatch):
    # Static map and per-season return no useful position column
    monkeypatch.setattr(npred.players, 'get_players', lambda: [])

    class DummyNoPos:
        def __init__(self, *args, **kwargs):
            pass
        def get_data_frames(self):
            return [pd.DataFrame([{'PLAYER_ID': 2, 'PLAYER_NAME': 'X'}])]

    monkeypatch.setattr(npred.leaguedashplayerstats, 'LeagueDashPlayerStats', DummyNoPos)

    # Monkeypatch DataLoader.get_player_position to return 'C' for player 2
    def fake_get_pos(self, pid):
        return 'C' if pid == 2 else 'SF'

    monkeypatch.setattr(DataLoader, 'get_player_position', fake_get_pos)

    loader = DataLoader(CONFIG)
    positions = loader.fetch_player_positions(seasons=['2025-26'], player_ids=[2], force_refresh=True)

    assert positions.get(2) == 'C'
