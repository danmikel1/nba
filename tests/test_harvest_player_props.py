"""
tests/test_harvest_player_props.py — Unit tests for real player prop harvester and settlement engine
"""

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from scripts.harvest_player_props import (
    american_to_decimal,
    harvest_player_props_for_date,
    normalize_player_name,
    PROP_MARKET_MAP,
    settle_player_props_for_date,
)


def test_american_to_decimal():
    assert american_to_decimal(-110) == 1.909
    assert american_to_decimal(150) == 2.5
    assert american_to_decimal(-200) == 1.5
    assert american_to_decimal(100) == 2.0


def test_normalize_player_name():
    assert normalize_player_name("Luka Doncic Jr.") == "Luka Doncic"
    assert normalize_player_name("De'Aaron Fox") == "DeAaron Fox"
    assert normalize_player_name("Peyton Watson") == "Peyton Watson"


def test_market_map_keys():
    assert "PTS" in PROP_MARKET_MAP
    assert "REB" in PROP_MARKET_MAP
    assert "AST" in PROP_MARKET_MAP
    assert "3PT" in PROP_MARKET_MAP
    assert "PRA" in PROP_MARKET_MAP


def test_settle_player_props_logic(tmp_path):
    test_date = "2026-10-22"
    dummy_props = {
        "date": test_date,
        "count": 2,
        "props": [
            {
                "date": test_date,
                "player": "Joel Embiid",
                "market": "PTS",
                "line": 25.5,
                "side": "OVER",
                "odds_american": -110,
                "odds_decimal": 1.909,
            },
            {
                "date": test_date,
                "player": "Donovan Mitchell",
                "market": "AST",
                "line": 5.5,
                "side": "UNDER",
                "odds_american": -120,
                "odds_decimal": 1.833,
            },
        ],
    }

    dummy_boxscores = {
        "joel embiid": {"PTS": 30.0, "REB": 11.0, "AST": 4.0, "3PT": 1.0, "PRA": 45.0, "MIN": 34.0},
        "donovan mitchell": {"PTS": 24.0, "REB": 3.0, "AST": 4.0, "3PT": 3.0, "PRA": 31.0, "MIN": 36.0},
    }

    with patch("scripts.harvest_player_props.HARVEST_DIR", tmp_path), \
         patch("scripts.harvest_player_props.fetch_espn_player_boxscores", return_value=dummy_boxscores), \
         patch("scripts.harvest_player_props.SETTLED_PROPS_PARQUET", tmp_path / "settled_player_props.parquet"), \
         patch("scripts.harvest_player_props.SETTLED_PROPS_CSV", tmp_path / "settled_player_props.csv"):

        props_file = tmp_path / f"player_props_{test_date}.json"
        props_file.write_text(json.dumps(dummy_props), encoding="utf-8")

        df = settle_player_props_for_date(test_date)
        assert df is not None
        assert len(df) == 2

        # Embiid OVER 25.5 with 30 pts -> WIN
        embiid_row = df[df["player"] == "Joel Embiid"].iloc[0]
        assert embiid_row["actual_value"] == 30.0
        assert embiid_row["margin"] == 4.5
        assert embiid_row["result"] == "WIN"
        assert bool(embiid_row["over_hit"]) is True

        # Mitchell UNDER 5.5 with 4 ast -> WIN
        mitchell_row = df[df["player"] == "Donovan Mitchell"].iloc[0]
        assert mitchell_row["actual_value"] == 4.0
        assert mitchell_row["margin"] == -1.5
        assert mitchell_row["result"] == "WIN"
        assert bool(mitchell_row["over_hit"]) is False


def test_harvest_extracts_open_vs_closing_line_movement(tmp_path):
    """
    Verify that harvest_player_props_for_date extracts Book '30' (Open) as opening_line,
    selects sharp closing book '68' (DraftKings) over '75' (BetMGM alt), and computes
    nonzero line_movement and odds_movement.
    """
    mock_api_payload = {
        "players": [
            {"id": 999, "full_name": "Nikola Jokic", "abbr": "N.Jokic", "position": "C", "team_id": 1}
        ],
        "games": [
            {
                "id": 295142,
                "start_time": "2099-10-23T01:30:00.000Z",
                "home_team": {"abbr": "OKC"},
                "away_team": {"abbr": "DEN"},
            }
        ],
        "markets": {
            "30": {  # Official Open Book
                "event": {
                    "core_bet_type_27_points": [
                        {"player_id": 999, "event_id": 295142, "side": "over", "value": 27.5, "odds": -104, "book_id": 30}
                    ]
                }
            },
            "68": {  # DraftKings Current/Closing
                "event": {
                    "core_bet_type_27_points": [
                        {"player_id": 999, "event_id": 295142, "side": "over", "value": 24.5, "odds": -110, "book_id": 68}
                    ]
                }
            },
            "75": {  # BetMGM Ladder/Alt (should not override DraftKings)
                "event": {
                    "core_bet_type_27_points": [
                        {"player_id": 999, "event_id": 295142, "side": "over", "value": 18.5, "odds": -350, "book_id": 75}
                    ]
                }
            },
        },
    }

    def fake_fetch(pick_type, date_compact):
        if pick_type == "core_bet_type_27_points":
            return mock_api_payload
        return {"players": [], "games": [], "markets": {}}

    with patch("scripts.harvest_player_props.HARVEST_DIR", tmp_path), \
         patch("scripts.harvest_player_props.fetch_market_lines_http", side_effect=fake_fetch):
        props = harvest_player_props_for_date("2026-10-22")
        assert len(props) == 1
        p = props[0]
        assert p["player"] == "Nikola Jokic"
        assert p["opening_line"] == 27.5
        assert p["closing_line"] == 24.5
        assert p["line"] == 24.5
        assert p["line_movement"] == -3.0
        assert p["book_name"] == "DraftKings"
        assert p["opening_odds_american"] == -104
        assert p["odds_american"] == -110

