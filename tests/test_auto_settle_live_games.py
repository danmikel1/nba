import pytest
from unittest.mock import patch, MagicMock
from nba_prediction import Tracker
from scripts.harvest_player_props import fetch_espn_player_boxscores


def test_auto_settle_skips_in_progress_game(tmp_path):
    test_tracker_file = tmp_path / "test_tracker.json"
    test_tracker_file.write_text("[]", encoding="utf-8")

    tracker = Tracker(test_tracker_file)
    tracker.add_bet({
        "id": 999901,
        "date": "2026-10-28",
        "player": "LAL @ LAC",
        "opponent": "LAC",
        "market": "SPREAD",
        "bet_type": "spread",
        "side": "LAL -5.5",
        "line": -5.5,
        "result": "Pending",
    })

    # Mock ESPN response for 2026-10-28 where LAL @ LAC is in progress (state: in, completed: False)
    mock_events_oct28 = [
        {
            "id": "401909892",
            "name": "Los Angeles Lakers at LA Clippers",
            "competitions": [
                {
                    "status": {
                        "type": {
                            "id": "2",
                            "state": "in",
                            "completed": False,
                            "description": "In Progress",
                            "detail": "6:28 - 3rd Quarter",
                        }
                    },
                    "competitors": [
                        {
                            "homeAway": "home",
                            "score": "51",
                            "team": {"abbreviation": "LAC", "name": "Clippers"},
                        },
                        {
                            "homeAway": "away",
                            "score": "63",
                            "team": {"abbreviation": "LAL", "name": "Lakers"},
                        },
                    ],
                }
            ],
        }
    ]

    # Mock ESPN response for 2026-10-27 where POR @ LAL was final (Lakers lost by 7)
    mock_events_oct27 = [
        {
            "id": "401909880",
            "name": "Portland Trail Blazers at Los Angeles Lakers",
            "competitions": [
                {
                    "status": {
                        "type": {
                            "id": "3",
                            "state": "post",
                            "completed": True,
                            "description": "Final",
                        }
                    },
                    "competitors": [
                        {
                            "homeAway": "home",
                            "score": "129",
                            "team": {"abbreviation": "LAL", "name": "Lakers"},
                        },
                        {
                            "homeAway": "away",
                            "score": "136",
                            "team": {"abbreviation": "POR", "name": "Trail Blazers"},
                        },
                    ],
                }
            ],
        }
    ]

    def mock_get(url, params=None, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        dates = (params or {}).get("dates", "")
        if "20261028" in dates:
            resp.json.return_value = {"events": mock_events_oct28}
        elif "20261027" in dates:
            resp.json.return_value = {"events": mock_events_oct27}
        else:
            resp.json.return_value = {"events": []}
        return resp

    with patch("requests.get", side_effect=mock_get):
        result = tracker.auto_settle_pending_bets()

    # Must NOT settle the in-progress bet, and must NOT match yesterday's POR @ LAL game
    assert result["settled_count"] == 0
    bets = tracker.get_bets()
    assert bets[0]["result"] == "Pending"
    assert bets[0].get("actual_value") is None


def test_fetch_player_boxscores_ignores_in_progress():
    mock_events = [
        {
            "id": "12345",
            "name": "Los Angeles Lakers at LA Clippers",
            "status": {
                "type": {
                    "state": "in",
                    "completed": False,
                    "description": "In Progress",
                }
            },
            "competitions": [
                {
                    "status": {
                        "type": {
                            "state": "in",
                            "completed": False,
                            "description": "In Progress",
                        }
                    }
                }
            ],
        }
    ]

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"events": mock_events}

    with patch("requests.get", return_value=mock_resp):
        stats = fetch_espn_player_boxscores("2026-10-28")

    assert stats == {}
