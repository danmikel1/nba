"""
tests/test_harvest_closing_odds.py — Unit Tests for Odds Harvester & Game UI Components
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from scripts.harvest_closing_odds import normalize_team, harvest_daily_odds, settle_games_for_date
from game_ui import compute_win_prob_and_kelly
from persistence_utils import safe_json_load


def test_normalize_team():
    """Verify team normalization handles various formats."""
    assert normalize_team("Los Angeles Lakers") == "LAL"
    assert normalize_team("lakers") == "LAL"
    assert normalize_team("LAL") == "LAL"
    assert normalize_team("Boston Celtics") == "BOS"
    assert normalize_team("celtics") == "BOS"
    assert normalize_team("GSW") == "GSW"
    assert normalize_team("Golden State Warriors") == "GSW"
    assert normalize_team("warriors") == "GSW"


def test_compute_win_prob_and_kelly():
    """Verify probability and Kelly staking behavior."""
    # Positive edge -> positive stake
    prob, ev, stake = compute_win_prob_and_kelly(
        edge_pts=3.0,
        std_dev=11.5,
        bankroll=1000.0,
        kelly_fraction=0.25,
    )
    assert prob > 0.50
    assert ev > 0.0
    assert stake > 0.0
    assert stake <= 1000.0 * 0.05  # capped at 5% max

    # Zero or negative edge -> zero stake
    prob_neg, ev_neg, stake_neg = compute_win_prob_and_kelly(
        edge_pts=-2.0,
        std_dev=11.5,
        bankroll=1000.0,
        kelly_fraction=0.25,
    )
    assert prob_neg < 0.50
    assert ev_neg < 0.0
    assert stake_neg == 0.0


def test_settle_games_logic(tmp_path):
    """Test matching finished games against closing lines with mock ESPN response."""
    test_date = "2026-10-20"
    mock_espn_games = [
        {
            "source": "espn",
            "event_id": "12345",
            "date": test_date,
            "home_team": "LAL",
            "away_team": "BOS",
            "matchup": "BOS @ LAL",
            "spread_home": -4.5,
            "total": 224.5,
            "home_score": 115,
            "away_score": 105,  # actual_margin = +10, actual_total = 220
            "status": "Final",
            "state": "post",
        }
    ]

    with patch("scripts.harvest_closing_odds.harvest_from_espn", return_value=mock_espn_games):
        with patch("scripts.harvest_closing_odds.HARVEST_DIR", tmp_path):
            with patch("scripts.harvest_closing_odds.SETTLED_CSV", tmp_path / "settled_games.csv"):
                settled = settle_games_for_date(test_date)
                assert len(settled) == 1
                rec = settled[0]
                assert rec["home_team"] == "LAL"
                assert rec["actual_margin"] == 10
                assert rec["actual_total"] == 220
                # Margin +10 vs spread -4.5 -> Home won by 10, covered -4.5 -> HOME
                assert rec["ats_winner"] == "HOME"
                # Total 220 vs line 224.5 -> UNDER
                assert rec["ou_winner"] == "UNDER"
                assert (tmp_path / "settled_games.csv").exists()
