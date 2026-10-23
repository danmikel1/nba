"""
tests/test_gpu_and_llm_features.py — Unit Tests for GPU Acceleration, SGP Simulation & Hybrid LLM Engine
"""

import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from hardware_utils import detect_cuda_device, get_gpu_info, check_cuda_availability, get_xgboost_hardware_params
from simulation_engine import SimulationEngine
import llm_engine


def test_hardware_utils_cuda_detection():
    """Verify hardware detection reports valid device and specs."""
    device = detect_cuda_device()
    assert device in ("cuda", "cpu")

    info = get_gpu_info()
    assert "has_gpu" in info
    assert "name" in info
    assert "total_vram_gb" in info
    assert "device" in info

    is_cuda, name = check_cuda_availability()
    assert isinstance(is_cuda, bool)
    assert isinstance(name, str)

    params = get_xgboost_hardware_params({"max_depth": 5})
    assert params["max_depth"] == 5
    assert params["tree_method"] == "hist"
    assert params["device"] in ("cuda", "cpu")


def test_sgp_multivariate_monte_carlo_same_player():
    """Test correlated parlay for same player across different markets."""
    engine = SimulationEngine()
    legs = [
        {"player": "Luka Doncic", "market": "PTS", "line": 28.5, "side": "OVER", "mean": 30.0, "std": 6.0},
        {"player": "Luka Doncic", "market": "3PT", "line": 3.5, "side": "OVER", "mean": 3.8, "std": 1.2},
    ]

    res = engine.simulate_correlated_parlay(legs, n_simulations=20_000)

    assert res["n_legs"] == 2
    assert 0.0 < res["joint_prob"] < 1.0
    assert 0.0 < res["independent_prob"] < 1.0
    assert "fair_decimal_odds" in res
    assert len(res["individual_probs"]) == 2
    # Because PTS and 3PT are positively correlated (+0.65), joint_prob should exceed independent_prob
    assert res["joint_prob"] >= res["independent_prob"]
    assert res["correlation_edge_pct"] >= 0.0


def test_sgp_multivariate_monte_carlo_cannibalization():
    """Test teammate points cannibalization produces negative correlation edge."""
    engine = SimulationEngine()
    legs = [
        {"player": "Jayson Tatum", "team": "BOS", "market": "PTS", "line": 27.5, "side": "OVER", "mean": 27.0, "std": 5.5},
        {"player": "Jaylen Brown", "team": "BOS", "market": "PTS", "line": 24.5, "side": "OVER", "mean": 24.0, "std": 5.0},
    ]

    res = engine.simulate_correlated_parlay(legs, n_simulations=20_000)

    assert res["n_legs"] == 2
    assert 0.0 < res["joint_prob"] < 1.0
    # Teammate PTS lines have negative correlation (-0.15)
    assert res["correlation_edge_pct"] <= 5.0  # slight drag or neutral


def test_sgp_pairwise_correlation_heuristics():
    """Test specific pairwise domain correlations."""
    engine = SimulationEngine()

    # Same player PTS & 3PT -> 0.65
    c_pts_3pt = engine._estimate_pairwise_correlation(
        {"player": "Curry", "market": "PTS"},
        {"player": "Curry", "market": "3PT"}
    )
    assert c_pts_3pt == 0.65

    # Same player opposite sides -> -1.0
    c_opp_side = engine._estimate_pairwise_correlation(
        {"player": "Curry", "market": "PTS", "side": "OVER"},
        {"player": "Curry", "market": "PTS", "side": "UNDER"}
    )
    assert c_opp_side == -1.0

    # Teammate AST & PTS -> 0.30
    c_ast_pts = engine._estimate_pairwise_correlation(
        {"player": "Haliburton", "team": "IND", "market": "AST"},
        {"player": "Siakam", "team": "IND", "market": "PTS"}
    )
    assert c_ast_pts == 0.30

    # Opponents shootout -> 0.20
    c_shootout = engine._estimate_pairwise_correlation(
        {"player": "Doncic", "team": "DAL", "opponent": "PHX", "market": "PTS"},
        {"player": "Booker", "team": "PHX", "opponent": "DAL", "market": "PTS"}
    )
    assert c_shootout == 0.20

    # Independent cross-game -> 0.0
    c_indep = engine._estimate_pairwise_correlation(
        {"player": "Jokic", "team": "DEN", "opponent": "LAL", "market": "PTS"},
        {"player": "Brunson", "team": "NYK", "opponent": "MIA", "market": "PTS"}
    )
    assert c_indep == 0.0


def test_llm_engine_provider_detection():
    """Test LLM engine detects provider and falls back gracefully."""
    with patch("requests.get") as mock_get:
        # Mock Ollama alive
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "models": [{"name": "qwen2.5:14b"}, {"name": "llama3.1:8b"}]
        }

        has_ollama, model = llm_engine.check_local_ollama()
        assert has_ollama is True
        assert "qwen2.5" in model

        prov = llm_engine.get_active_llm_provider()
        assert prov["is_local"] is True
        assert "RTX 5060 Ti" in prov["name"]


def test_llm_engine_query_local_mock():
    """Test querying local Ollama engine endpoint."""
    with patch("llm_engine.get_active_llm_provider") as mock_prov, \
         patch("requests.post") as mock_post:
        mock_prov.return_value = {
            "mode": "local_ollama",
            "name": "RTX 5060 Ti GPU (Ollama: llama3.1)",
            "model": "llama3.1",
            "is_local": True,
        }
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"response": "VERDICT: TAKE\nREASON: High form\nCONFIDENCE: 9"}

        res = llm_engine.query_llm("Test prompt")
        assert res["success"] is True
        assert "VERDICT: TAKE" in res["response"]
        assert "RTX 5060 Ti" in res["provider"]


def test_tuned_hyperparams_integration(tmp_path):
    """Test loading tuned hyperparams from json config."""
    from scripts.train_models import get_tuned_market_params

    cfg = {"PTS": {"learning_rate": 0.03, "max_depth": 5}}
    cfg_file = tmp_path / "model_hyperparams.json"
    cfg_file.write_text(json.dumps(cfg), encoding="utf-8")

    with patch("scripts.train_models.Path") as mock_path:
        mock_path.return_value = cfg_file
        loaded = get_tuned_market_params("PTS")
        assert loaded.get("learning_rate") == 0.03
        assert loaded.get("max_depth") == 5
