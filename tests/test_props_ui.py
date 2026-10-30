"""
tests/test_props_ui.py — Unit tests for the Real Sportsbook Best Prop Edges Scanner (`props_ui.py`)
"""

from nba_prediction import CONFIG, ModelEngine
from props_ui import (
    _build_bet_payload_from_row,
    load_harvested_props_for_date,
    score_slate_props,
)


def test_score_slate_props_filters_sweet_spot_and_line_movement():
    """Verify score_slate_props scores real harvested props, applies Sniper EV band, and preserves line movement."""
    me = ModelEngine(CONFIG)
    raw = load_harvested_props_for_date("2026-10-23", force_refresh=False)
    assert len(raw) > 0, "Expected harvested props for 2026-10-23"

    picks, full = score_slate_props(
        raw_props=raw,
        model_engine=me,
        bankroll=1000.0,
        kelly_fraction=0.25,
        min_ev_pct=4.0,
        max_ev_pct=14.0,
        market_filter="SNIPER_COMBOS",
        sniper_gates=True,
    )

    assert not full.empty, "Expected full scored DataFrame to be non-empty"
    assert not picks.empty, "Expected Sniper picks DataFrame to be non-empty"

    # Verify EV band filter (4.0% to 14.0%)
    assert (picks["ev_pct"] >= 4.0).all()
    assert (picks["ev_pct"] <= 14.0).all()

    # Verify PA OVER is blocked by Sniper gates
    pa_over = picks[(picks["market"] == "PA") & (picks["side"] == "OVER")]
    assert len(pa_over) == 0, "Expected PA OVER to be blocked by Sniper gates"

    # Verify payload builder produces all required Tracker keys
    sample_payload = _build_bet_payload_from_row(picks.iloc[0])
    for key in [
        "date",
        "player",
        "opponent",
        "market",
        "line",
        "opening_line",
        "closing_line",
        "side",
        "odds",
        "ev",
        "proj",
        "stake",
        "prob",
        "quality_score",
        "feat_ema",
    ]:
        assert key in sample_payload, f"Missing key {key} in bet payload"


def test_score_slate_props_unders_only():
    """Verify that setting unders_only=True filters picks strictly to UNDER."""
    me = ModelEngine(CONFIG)
    raw = load_harvested_props_for_date("2026-10-23", force_refresh=False)
    assert len(raw) > 0

    picks, _ = score_slate_props(
        raw_props=raw,
        model_engine=me,
        unders_only=True,
    )
    assert not picks.empty
    assert (picks["side"] == "UNDER").all(), "Expected all picks to be UNDER when unders_only=True"
