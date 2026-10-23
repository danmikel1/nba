"""
tests/test_game_prediction.py — Unit Tests for NBA Game-Level Prediction Engine
"""

import pytest
import pandas as pd
from pathlib import Path

from game_prediction import (
    load_game_data,
    build_team_histories,
    predict_game,
    GamePrediction,
    GAME_FEATURE_COLUMNS,
    STD_TO_CSV,
    TEAM_ABBREV_MAP,
)


def test_team_abbrev_mappings():
    """Verify bidirectional team abbreviation maps."""
    assert len(TEAM_ABBREV_MAP) >= 30
    assert TEAM_ABBREV_MAP["lal"] == "LAL"
    assert TEAM_ABBREV_MAP["bos"] == "BOS"
    assert STD_TO_CSV["LAL"] == "lal"
    assert STD_TO_CSV["BOS"] == "bos"


def test_load_game_data_columns():
    """Verify that load_game_data creates all necessary targets and signs."""
    df = load_game_data()
    assert not df.empty
    assert "actual_margin" in df.columns
    assert "actual_total" in df.columns
    assert "signed_spread" in df.columns

    sample = df.dropna(subset=["spread", "whos_favored"]).iloc[0]
    if sample["whos_favored"] == "home":
        assert sample["signed_spread"] == sample["spread"]
    else:
        assert sample["signed_spread"] == -sample["spread"]


def test_build_team_histories():
    """Verify team rolling histories compute expected rolling features."""
    df = load_game_data().head(100)
    histories = build_team_histories(df)
    assert len(histories) > 0

    first_team = next(iter(histories.values()))
    assert "margin" in first_team.columns
    assert "ppg_l10" in first_team.columns
    assert "pag_l10" in first_team.columns
    assert "win_streak" in first_team.columns
    assert "ats_streak" in first_team.columns


def test_predict_game_returns_valid_prediction():
    """Verify predict_game returns a well-formed GamePrediction with valid quantiles."""
    pred = predict_game(
        home="LAL",
        away="BOS",
        vegas_spread=-3.5,
        vegas_total=220.5,
        moneyline_home=-150,
        moneyline_away=130,
    )
    assert isinstance(pred, GamePrediction)
    assert pred.home == "LAL"
    assert pred.away == "BOS"
    assert pred.vegas_spread == -3.5
    assert pred.vegas_total == 220.5

    # Quantile ordering checks
    assert pred.pred_margin_q10 <= pred.pred_margin
    assert pred.pred_margin <= pred.pred_margin_q90
    assert pred.pred_total_q10 <= pred.pred_total
    assert pred.pred_total <= pred.pred_total_q90

    # Picks and confidence checks
    assert pred.ats_pick in ["HOME ATS", "AWAY ATS"]
    assert pred.ou_pick in ["OVER", "UNDER"]
    assert pred.confidence_spread in ["Low", "Medium", "High"]
    assert pred.confidence_total in ["Low", "Medium", "High"]
