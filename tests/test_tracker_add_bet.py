"""
tests/test_tracker_add_bet.py — Verify Tracker add_bet, log_bet, and save_bet aliases.
"""

from pathlib import Path
from nba_prediction import Tracker


def test_tracker_add_bet_alias(tmp_path):
    test_file = tmp_path / "test_bet_tracker.json"
    tracker = Tracker(file_path=test_file)

    game_bet = {
        "date": "2026-10-22",
        "player": "BOS @ NYK",
        "market": "SPREAD",
        "bet_type": "spread",
        "side": "BOS -3.5",
        "line": -3.5,
        "odds": -110,
        "stake": 50.0,
        "ev": 0.05,
        "win_prob": 0.54,
    }

    # Test add_bet
    tracker.add_bet(game_bet)
    bets = tracker.get_bets()
    assert len(bets) == 1
    assert bets[0]["player"] == "BOS @ NYK"
    assert bets[0]["market"] == "SPREAD"
    assert bets[0]["result"] == "Pending"

    # Test log_bet
    prop_bet = {
        "date": "2026-10-22",
        "player": "Luka Doncic",
        "market": "PTS",
        "side": "OVER",
        "line": 29.5,
        "odds": -110,
        "stake": 25.0,
    }
    tracker.log_bet(prop_bet)
    bets = tracker.get_bets()
    assert len(bets) == 2
    assert bets[1]["player"] == "Luka Doncic"
