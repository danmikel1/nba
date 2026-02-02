import pytest
from pathlib import Path
from nba_prediction import Tracker


def test_winrate_counts_only_staked(tmp_path):
    tr = Tracker(file_path=tmp_path / 'tracker.json')

    # Three decided bets: one win with stake, one loss with stake, one win with zero stake
    bets = [
        {"id": 1, "player": "A", "market": "PTS", "result": "Win", "stake": 10.0, "odds": 1.91},
        {"id": 2, "player": "B", "market": "PTS", "result": "Loss", "stake": 5.0, "odds": 1.91},
        {"id": 3, "player": "C", "market": "PTS", "result": "Win", "stake": 0.0, "odds": 1.91},
    ]

    # Save initial bets
    tr._save(bets)

    stats = tr.get_stats()

    # win_rate_all should reflect all decided bets (2 wins / 3 decided)
    assert pytest.approx(stats.get('win_rate_all', 0.0), rel=1e-3) == 2/3

    # win_rate (primary) should reflect only staked decided bets (1 win / 2 staked decided)
    assert pytest.approx(stats['win_rate'], rel=1e-3) == 1/2

    # total_profit should account for stakes (win: +0.91*10, loss: -5)
    assert pytest.approx(stats['total_profit'], rel=1e-3) == pytest.approx(10.0 * (1.91 - 1) - 5.0)

    # Equity curve should only include staked decided bets and be stake-weighted
    eq = tr.get_equity_curve()
    assert len(eq) == 2
    assert pytest.approx(eq[0]['P/L'], rel=1e-3) == pytest.approx(10.0 * (1.91 - 1))
    assert pytest.approx(eq[1]['P/L'], rel=1e-3) == pytest.approx(10.0 * (1.91 - 1) - 5.0)
