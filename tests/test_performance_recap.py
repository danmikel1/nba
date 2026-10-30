import pytest
from bet_slip_share import render_performance_recap_png

def test_render_performance_recap_empty():
    png_bytes = render_performance_recap_png([])
    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 1000
    assert png_bytes[:4] == b"\x89PNG"

def test_render_performance_recap_with_bets():
    sample_bets = [
        {"id": "b1", "result": "Win", "odds": -110, "market": "POINTS", "stake": 10.0},
        {"id": "b2", "result": "Loss", "odds": -110, "market": "REBOUNDS", "stake": 10.0},
        {"id": "b3", "result": "Win", "odds": 120, "bet_type": "spread", "market": "SPREAD", "stake": 10.0},
        {"id": "b4", "result": "Push", "odds": -110, "bet_type": "total", "market": "TOTAL", "stake": 10.0},
        {"id": "b5", "result": "Pending", "odds": -110, "market": "ASSISTS", "stake": 10.0},
    ]
    png_bytes = render_performance_recap_png(
        sample_bets,
        period_title="OCTOBER 2026 RECAP",
        period_subtitle="Test Subtitle",
        currency="PHP",
        currency_symbol="₱",
        hypothetical_stake=1000.0,
    )
    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 5000
    assert png_bytes[:4] == b"\x89PNG"

def test_render_performance_recap_currencies():
    for curr, sym in [("USD", "$"), ("EUR", "€"), ("PHP", "₱")]:
        png_bytes = render_performance_recap_png(
            [{"result": "Win", "odds": 1.91}],
            currency=curr,
            currency_symbol=sym,
            hypothetical_stake=50.0,
        )
        assert png_bytes[:4] == b"\x89PNG"
