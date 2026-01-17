import pytest
from types import SimpleNamespace
import nba_prediction


def test_run_backtest_passes_shared_loader(monkeypatch):
    # Prepare orchestrator
    orchestrator = nba_prediction.PredictionOrchestrator()

    # Monkeypatch players.get_players to return a predictable player
    monkeypatch.setattr('nba_prediction.players.get_players', lambda: [{'full_name': 'Test Player', 'id': 123}])

    # Sentinel loader object
    sentinel_loader = object()
    monkeypatch.setattr('nba_prediction.get_shared_bulk_loader', lambda: sentinel_loader)

    captured = {}
    def fake_run_backtest(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(success=True)

    monkeypatch.setattr(orchestrator.backtester, 'run_backtest', fake_run_backtest)

    res = orchestrator.run_backtest(player_name='Test Player', market='PTS')

    assert captured.get('bulk_loader') is sentinel_loader, "Shared bulk_loader was not passed to Backtester"
    assert res.success
