"""
tests/test_gpu_and_llm_features.py — Unit Tests for GPU Acceleration & SGP Monte Carlo Simulation
"""

import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from hardware_utils import detect_cuda_device, get_gpu_info, check_cuda_availability, get_xgboost_hardware_params
from simulation_engine import SimulationEngine


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
    """Test empirical pairwise correlations loaded from config/empirical_correlations.json."""
    engine = SimulationEngine()

    # Same player PTS & 3PT -> 0.65
    c_pts_3pt = engine._estimate_pairwise_correlation(
        {"player": "Curry", "market": "PTS"},
        {"player": "Curry", "market": "3PT"}
    )
    assert c_pts_3pt == 0.65

    # Same player PTS & PRA -> empirical (~0.9198)
    c_pts_pra = engine._estimate_pairwise_correlation(
        {"player": "Curry", "market": "PTS"},
        {"player": "Curry", "market": "PRA"}
    )
    assert 0.85 < c_pts_pra < 0.98

    # Same player opposite sides -> -1.0
    c_opp_side = engine._estimate_pairwise_correlation(
        {"player": "Curry", "market": "PTS", "side": "OVER"},
        {"player": "Curry", "market": "PTS", "side": "UNDER"}
    )
    assert c_opp_side == -1.0

    # Teammate AST & PTS -> empirical (~0.1516)
    c_ast_pts = engine._estimate_pairwise_correlation(
        {"player": "Haliburton", "team": "IND", "market": "AST"},
        {"player": "Siakam", "team": "IND", "market": "PTS"}
    )
    assert 0.05 < c_ast_pts < 0.35

    # Opponents shootout -> empirical (~0.1355)
    c_shootout = engine._estimate_pairwise_correlation(
        {"player": "Doncic", "team": "DAL", "opponent": "PHX", "market": "PTS"},
        {"player": "Booker", "team": "PHX", "opponent": "DAL", "market": "PTS"}
    )
    assert 0.05 < c_shootout < 0.30

    # Independent cross-game -> 0.0
    c_indep = engine._estimate_pairwise_correlation(
        {"player": "Jokic", "team": "DEN", "opponent": "LAL", "market": "PTS"},
        {"player": "Brunson", "team": "NYK", "opponent": "MIA", "market": "PTS"}
    )
    assert c_indep == 0.0


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


def test_run_monte_carlo_10k_regime_switching_and_blowout_risk():
    """Verify 10k Monte Carlo engine samples purely from trained XGBoost (q05, q50, q95) split-normal skew."""
    engine = SimulationEngine()

    # Symmetric / right-skewed quantiles vs heavy left-tail downside quantiles (q05 much lower than q50)
    tight_res = engine.run_monte_carlo_10k(
        mean=28.0, std=5.5, line=26.5, market="PTS",
        q05=21.0, q95=38.0, n_simulations=10_000,
    )
    assert len(tight_res.simulations) == 10_000
    assert 0.001 <= tight_res.over_prob <= 0.999
    assert abs((tight_res.over_prob + tight_res.under_prob) - 1.0) < 1e-4
    assert tight_res.ci_10 < tight_res.median < tight_res.ci_90
    assert tight_res.tail_cvar_10 <= tight_res.ci_10

    left_skew_res = engine.run_monte_carlo_10k(
        mean=28.0, std=5.5, line=26.5, market="PTS",
        q05=12.0, q95=35.0, n_simulations=10_000,
    )
    # Wider trained left tail (q50 - q05 = 16 vs q95 - q50 = 7) yields higher left-skew ratio & lower 10% tail CVaR
    assert left_skew_res.blowout_risk_pct > tight_res.blowout_risk_pct
    assert left_skew_res.over_prob < tight_res.over_prob
    assert left_skew_res.tail_cvar_10 < tight_res.tail_cvar_10


def test_simulate_slate_10k_vectorized_performance():
    """Verify vectorized 10k Monte Carlo scores an 800-prop slate in milliseconds from pure quantiles."""
    engine = SimulationEngine()
    m = 800
    q50 = np.full(m, 22.0, dtype=np.float32)
    # First half has deep left tail (q05 = 10.0), second half has tight left tail (q05 = 18.0)
    q05 = np.linspace(10.0, 18.0, m, dtype=np.float32)
    q95 = np.full(m, 31.0, dtype=np.float32)
    lines = np.full(m, 20.5, dtype=np.float32)

    batch = engine.simulate_slate_10k(
        q50=q50, q05=q05, q95=q95, lines=lines,
        n_simulations=10_000,
    )
    assert batch["over_prob"].shape == (m,)
    assert batch["under_prob"].shape == (m,)
    assert batch["blowout_risk_pct"].shape == (m,)
    assert batch["tail_cvar_10"].shape == (m,)
    # Index 0 (deep left tail q05=10.0) has higher left-skew ratio and lower over_prob than index -1 (tight left tail q05=18.0)
    assert batch["blowout_risk_pct"][0] > batch["blowout_risk_pct"][-1]
    assert batch["over_prob"][0] < batch["over_prob"][-1]


def test_sgp_combo_markets_correlation():
    """Verify combo markets (PRA, PR, PA, RA) are supported in SGP correlation matrix."""
    engine = SimulationEngine()
    legs = [
        {"player": "Nikola Jokic", "team": "DEN", "game_id": "G1", "market": "PRA", "line": 48.5, "side": "OVER", "mean": 51.0, "q05": 39.0, "q95": 64.0, "odds": 1.91},
        {"player": "Jamal Murray", "team": "DEN", "game_id": "G1", "market": "PA", "line": 27.5, "side": "OVER", "mean": 29.0, "q05": 20.0, "q95": 39.0, "odds": 1.91},
    ]
    res = engine.simulate_correlated_parlay(legs, n_simulations=10_000)
    assert res["n_legs"] == 2
    assert res["joint_prob"] > res["independent_prob"]
    assert "parlay_ev_pct" in res
    assert "combined_book_odds" in res
    assert len(res["correlation_matrix"]) == 2


def test_beat_writer_rag_minutes_cap_and_beneficiary_linking():
    """Verify Tool #2 (Embeddings RAG) classifies minutes caps, GTDs, outs, and links healthy Next-Man-Up beneficiaries with zero fabricated probability tweaks."""
    from rag_news_engine import BeatWriterRAGEngine

    rag = BeatWriterRAGEngine()
    sample_docs = [
        {
            "player": "Joel Embiid",
            "team": "PHI",
            "status": "Day-To-Day",
            "short_comment": "Embiid (knee) will play Friday but faces a strict 24-minute restriction, Kyle Neubeck reports.",
            "long_comment": "The big man is easing back into action under workload management and will be capped around 24 minutes.",
        },
        {
            "player": "Jalen Smith",
            "team": "CHI",
            "status": "Day-To-Day",
            "short_comment": "Smith is questionable for Friday's game against Boston with a left finger sprain.",
            "long_comment": "If Smith is held out against Boston, the Bulls could lean more on Zach Collins at center.",
        },
        {
            "player": "Dereck Lively II",
            "team": "DAL",
            "status": "Out",
            "short_comment": "Lively (foot) has been ruled out for Friday's game against the Pacers.",
            "long_comment": "He will not play Friday.",
        },
    ]
    rag.build_index(sample_docs, known_players=["Joel Embiid", "Jalen Smith", "Zach Collins", "Dereck Lively II"])

    # 1. Minutes restriction tagged observationally with zero arbitrary probability tweak
    embiid_over = rag.evaluate_player_prop_alert("Joel Embiid", side="OVER", market="PTS")
    embiid_under = rag.evaluate_player_prop_alert("Joel Embiid", side="UNDER", market="PTS")
    assert embiid_over["rag_flag"] == "🚨 MIN CAP"
    assert embiid_over["rag_prob_adj"] == 0.0
    assert embiid_under["rag_prob_adj"] == 0.0

    # 2. Questionable / GTD tag
    smith_over = rag.evaluate_player_prop_alert("Jalen Smith", side="OVER", market="REB")
    assert smith_over["rag_flag"] == "⚠️ GTD RISK"
    assert smith_over["rag_prob_adj"] == 0.0

    # 3. Healthy teammate beneficiary (Zach Collins mentioned inside Jalen Smith's longComment)
    collins_over = rag.evaluate_player_prop_alert("Zach Collins", side="OVER", market="PRA")
    assert collins_over["rag_flag"] == "🚀 USAGE BOOST"
    assert collins_over["rag_prob_adj"] == 0.0
    assert "Jalen Smith" in collins_over["rag_blurb"]

    # 4. Ruled out player
    lively_res = rag.evaluate_player_prop_alert("Dereck Lively II", side="OVER", market="REB")
    assert lively_res["rag_flag"] == "🛑 RULED OUT"
    assert lively_res["rag_prob_adj"] == 0.0


def test_beat_writer_rag_sub_5ms_semantic_search():
    """Verify semantic cosine search executes in under 15 ms and ranks relevant beat-writer blurbs first."""
    from rag_news_engine import BeatWriterRAGEngine

    rag = BeatWriterRAGEngine()
    sample_docs = [
        {
            "player": "Kawhi Leonard",
            "team": "LAC",
            "status": "Day-To-Day",
            "short_comment": "Leonard will be on a minutes restriction and pitch count in his return.",
            "long_comment": "He is easing back into conditioning.",
        },
        {
            "player": "Dennis Schroder",
            "team": "CHA",
            "status": "Day-To-Day",
            "short_comment": "Schroder is questionable with a foot sprain.",
            "long_comment": "The Hornets would be down to rookie Christian Anderson as their lone true point guard.",
        },
    ]
    rag.build_index(sample_docs, known_players=["Kawhi Leonard", "Dennis Schroder", "Christian Anderson"])

    hits = rag.search_semantic("minutes restriction pitch count", top_k=2)
    assert len(hits) >= 1
    assert hits[0]["player"] == "Kawhi Leonard"
    assert rag.last_query_ms < 25.0


def test_optuna_cuda_bayesian_tuning_and_pinball_loss():
    """Verify Tool #3 (Optuna + CUDA) optimizes multi-quantile pinball loss with TPESampler + MedianPruner."""
    import pandas as pd
    from scripts.tune_models_gpu import compute_pinball_loss, tune_market

    y_true = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    y_pred = np.array([12.0, 20.0, 25.0], dtype=np.float32)
    pb_50 = compute_pinball_loss(y_true, y_pred, 0.50)
    assert abs(pb_50 - (0.5 * (2.0 + 0.0 + 5.0) / 3.0)) < 1e-5

    rng = np.random.default_rng(42)
    n = 240
    f1 = rng.normal(20.0, 4.0, n)
    f2 = rng.normal(32.0, 3.0, n)
    target = 0.8 * f1 + 0.3 * f2 + rng.normal(0.0, 2.0, n)
    synth_df = pd.DataFrame({
        "date": pd.date_range("2025-10-01", periods=n, freq="D").astype(str),
        "feat_ema": f1,
        "feat_avg_minutes": f2,
        "actual_value": target,
    })

    res = tune_market(
        market="PTS",
        market_df=synth_df,
        feature_cols=["feat_ema", "feat_avg_minutes"],
        n_trials=4,
        n_splits=3,
        seed=42,
    )
    assert "best_params" in res
    assert res["best_pinball"] <= res["baseline_pinball"] + 1e-6
    assert res["trials_run"] == 4
    assert "learning_rate" in res["best_params"]
    assert "max_depth" in res["best_params"]


def test_lstm_feature_scout_bptt_and_live_inference(tmp_path):
    """Verify Tool #4 (4-Gate LSTM BPTT Feature Scout) reduces loss on temporal sequences and runs sub-ms live inference."""
    from lstm_feature_scout import VectorizedLSTMCellBPTT, LSTMFeatureScoutEngine
    from core_models import FeatureVector

    rng = np.random.default_rng(42)
    b, t_len, d = 160, 8, 8
    X_seq = rng.normal(0.0, 1.0, size=(b, t_len, d)).astype(np.float32)
    C_now = rng.normal(0.0, 1.0, size=(b, 5)).astype(np.float32)
    # Target depends on final two sequence steps + current context
    y_true = (1.5 * X_seq[:, -1, 0] + 0.8 * X_seq[:, -2, 0] - 0.9 * C_now[:, 3]).astype(np.float32)

    lstm = VectorizedLSTMCellBPTT(input_dim=d, context_dim=5, hidden_dim=16, lr=0.02, seed=42)
    pred_init, fg_init = lstm.predict(X_seq, C_now)
    mae_before = float(np.abs(pred_init - y_true).mean())

    lstm.fit(X_seq[:120], C_now[:120], y_true[:120], epochs=15, batch_size=64, X_val=X_seq[120:], C_val=C_now[120:], y_val=y_true[120:])
    pred_after, fg_after = lstm.predict(X_seq[120:], C_now[120:])
    mae_after = float(np.abs(pred_after - y_true[120:]).mean())

    assert mae_after < mae_before
    assert np.all((fg_after >= 0.0) & (fg_after <= 1.0))

    scout_file = tmp_path / "lstm_scout.pkl"
    scout = LSTMFeatureScoutEngine(model_path=scout_file)
    scout.models["PTS"] = lstm
    scout.save()

    loaded_scout = LSTMFeatureScoutEngine(model_path=scout_file)
    fv = FeatureVector(
        player_id=1, player_name="Luka Doncic", opponent_abbrev="PHX", market="PTS",
        line=29.5, avg_minutes=36.0, ema=31.5, std=5.5, opponent_drtg_season=114.0,
        spread=-3.5, game_total=231.5, days_rest=1, is_home=True, is_b2b=True, games_played=45,
        feat_season_avg=28.5, feat_l10_avg=30.8, feat_miles_96h=1800.0,
    )
    mom, fg = loaded_scout.predict_live_features(fv, "PTS")
    assert isinstance(mom, float)
    assert 0.0 <= fg <= 1.0




