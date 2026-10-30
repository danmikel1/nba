"""
tests/test_tracker_deduplication.py — Verify Tracker Duplicate Prevention for Games & Props
"""

import pytest
from nba_prediction import Tracker


def test_tracker_deduplication():
    tracker = Tracker()

    test_game_bet = {
        "date": "2026-10-24",
        "player": "TEST @ MOCK",
        "market": "SPREAD",
        "side": "TEST +3.5",
        "stake": 20.0,
        "result": "Pending",
    }

    # First save should succeed
    saved_first = tracker.save_bet(test_game_bet)
    assert saved_first is True, "First save of unique bet should succeed"
    bet_id = test_game_bet.get("id")

    try:
        # Check duplicate detection
        existing = tracker.is_duplicate_bet(test_game_bet)
        assert existing is not None, "Should identify identical bet as duplicate"
        assert existing["id"] == bet_id

        # Second save of identical bet must be blocked
        saved_second = tracker.save_bet(test_game_bet)
        assert saved_second is False, "Second save of identical bet must be rejected"

    finally:
        # Cleanup
        if bet_id:
            tracker.delete_bet(bet_id)
