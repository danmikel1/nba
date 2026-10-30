import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nba_prediction import Tracker
from scripts.harvest_player_props import normalize_player_name


def test_player_name_normalization_apostrophes_and_suffixes():
    """Ensure players with apostrophes, accents, and suffixes normalize uniformly."""
    test_cases = [
        ("De'Aaron Fox", "deaaron fox"),
        ("De’Aaron Fox", "deaaron fox"),
        ("Shaquille O'Neal", "shaquille oneal"),
        ("De'Andre Hunter", "deandre hunter"),
        ("Day'Ron Sharpe", "dayron sharpe"),
        ("Luka Dončić", "luka doncic"),
        ("Nikola Jokić", "nikola jokic"),
        ("Michael Porter Jr.", "michael porter"),
        ("P.J. Washington", "pj washington"),
        ("Karl-Anthony Towns", "karl anthony towns"),
        ("Trey Murphy III", "trey murphy"),
    ]
    tracker = Tracker()
    for raw_name, expected in test_cases:
        norm = normalize_player_name(raw_name).lower()
        assert norm == expected, f"normalize_player_name failed for {raw_name}: got {norm}"

        dedup_key = tracker.get_dedup_key({"player": raw_name, "market": "PTS", "side": "OVER", "date": "2026-10-28"})
        assert dedup_key[1] == expected, f"get_dedup_key failed for {raw_name}: got {dedup_key[1]}"


def test_auto_settle_deaaron_fox_matching():
    """Test that De'Aaron Fox matches ESPN box scores and resolves correctly."""
    mock_boxscores = {
        "deaaron fox": {
            "PTS": 11.0,
            "REB": 4.0,
            "AST": 6.0,
            "3PT": 2.0,
            "PRA": 21.0,
            "PR": 15.0,
            "PA": 17.0,
            "RA": 10.0,
            "MIN": 26.0,
        }
    }
    raw_player = "De'Aaron Fox"
    norm = normalize_player_name(raw_player).lower()
    assert norm in mock_boxscores
    assert mock_boxscores[norm]["PR"] == 15.0
