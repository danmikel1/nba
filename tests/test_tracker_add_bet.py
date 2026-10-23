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


def test_tracker_auto_settle_pending_bets(tmp_path, monkeypatch):
    test_file = tmp_path / "test_bet_tracker.json"
    tracker = Tracker(file_path=test_file)

    # 1. Empty pending bets
    res_empty = tracker.auto_settle_pending_bets()
    assert res_empty["settled_count"] == 0

    # 2. Add pending game total and spread
    game_total_bet = {
        "id": "bet-total-1",
        "date": "2026-10-22",
        "player": "DEN @ OKC",
        "market": "TOTAL",
        "bet_type": "total",
        "side": "OVER",
        "line": 224.5,
        "odds": -110,
        "stake": 100.0,
        "result": "Pending"
    }
    game_spread_bet = {
        "id": "bet-spread-1",
        "date": "2026-10-22",
        "player": "DEN @ OKC",
        "market": "SPREAD",
        "bet_type": "spread",
        "side": "OKC -4.5",
        "line": -4.5,
        "odds": -110,
        "stake": 100.0,
        "result": "Pending"
    }
    prop_bet = {
        "id": "bet-prop-1",
        "date": "2026-10-22",
        "player": "Chet Holmgren",
        "market": "PTS",
        "bet_type": "prop",
        "side": "OVER",
        "line": 18.5,
        "odds": -110,
        "stake": 50.0,
        "result": "Pending"
    }

    tracker.add_bet(game_total_bet)
    tracker.add_bet(game_spread_bet)
    tracker.add_bet(prop_bet)

    # Mock requests.get for ESPN scoreboard
    class MockScoreboardResponse:
        status_code = 200
        def json(self):
            return {
                "events": [
                    {
                        "competitions": [
                            {
                                "status": {
                                    "type": {"state": "post", "description": "Final"}
                                },
                                "competitors": [
                                    {
                                        "homeAway": "home",
                                        "team": {"abbreviation": "OKC", "name": "Thunder"},
                                        "score": "120"
                                    },
                                    {
                                        "homeAway": "away",
                                        "team": {"abbreviation": "DEN", "name": "Nuggets"},
                                        "score": "110"
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }

    import requests
    monkeypatch.setattr(requests, "get", lambda url, params=None, timeout=None: MockScoreboardResponse())

    # Mock fetch_espn_player_boxscores
    import scripts.harvest_player_props as harvest_mod
    monkeypatch.setattr(harvest_mod, "fetch_espn_player_boxscores", lambda date_str: {
        "chet holmgren": {
            "PTS": 24,
            "REB": 9,
            "AST": 3
        }
    })

    res = tracker.auto_settle_pending_bets()
    assert res["settled_count"] == 3
    assert res["wins"] == 3
    assert res["losses"] == 0

    settled_bets = tracker.get_bets()
    total_bet_res = next(b for b in settled_bets if b["id"] == "bet-total-1")
    assert total_bet_res["result"] == "Win"
    assert total_bet_res["actual_value"] == 230.0

    spread_bet_res = next(b for b in settled_bets if b["id"] == "bet-spread-1")
    assert spread_bet_res["result"] == "Win"
    assert spread_bet_res["actual_value"] == 10.0  # 120 - 110 = 10, margin + (-4.5) > 0

    prop_bet_res = next(b for b in settled_bets if b["id"] == "bet-prop-1")
    assert prop_bet_res["result"] == "Win"
    assert prop_bet_res["actual_value"] == 24.0
