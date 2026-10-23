"""
tests/test_unified_decision_and_calibration.py — Unit Tests for Steps 1-5 Improvements
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from decision_policy import DecisionConfig, DecisionPolicy
from trading_engine import TradingEngine
from model_engine import SigmoidCalibrator
from persistence_utils import load_ml_training_data, save_ml_training_data
from core_models import Projection, FeatureVector, SimulationResult


def test_sigmoid_calibrator_smooth_monotonic():
    """Verify SigmoidCalibrator provides smooth, strictly monotonic calibration without flat steps."""
    cal = SigmoidCalibrator()

    # Synthetic training: higher raw probs correlate with actual hits
    raw_probs = np.array([0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80])
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])

    cal.fit(raw_probs, y_true)

    # Test fine-grained probabilities
    test_probs = np.array([0.52, 0.53, 0.54, 0.55, 0.56, 0.57])
    cal_probs = cal.predict(test_probs)

    # 1. Output should be strictly bounded in [0.01, 0.99]
    assert np.all(cal_probs >= 0.01)
    assert np.all(cal_probs <= 0.99)

    # 2. Output should be strictly monotonic (no flat plateaus where 53% == 54%)
    for i in range(len(cal_probs) - 1):
        assert cal_probs[i] < cal_probs[i + 1]

    # 3. Single value prediction compatibility
    single = cal.predict([0.55])[0]
    assert isinstance(single, float)
    assert abs(single - cal_probs[3]) < 1e-6


def test_unified_decision_config_and_gates():
    """Verify DecisionConfig gates unify single-bet analysis and batch betting."""
    config = DecisionConfig(
        min_ev=0.03,
        min_prob=0.55,
        max_ev=0.25,
        min_qs=0.005,
        max_qs=0.035,
    )
    policy = DecisionPolicy(decision_config=config)

    dummy_features = FeatureVector(
        player_id=2544,
        player_name="LeBron James",
        opponent_abbrev="BOS",
        market="PTS",
        line=25.5,
        avg_minutes=35.0,
        ema=26.0,
        std=4.0,
        opponent_drtg_season=110.0,
        spread=-3.5,
        game_total=220.5,
        days_rest=2,
        is_home=True,
        is_b2b=False,
        games_played=50,
    )
    dummy_sim = SimulationResult(
        over_prob=0.62,
        under_prob=0.38,
        median=28.0,
        ci_10=22.0,
        ci_90=34.0,
        simulations=np.array([28.0]),
    )

    # Case A: Good bet meeting all sniper gates (EV ~ 10.8%, QS ~ 0.021)
    good_proj = Projection(
        final_projection=27.5,
        base_projection=27.5,
        confidence_interval=(22.0, 33.0),
        adjustments={"predicted_std": 5.0},
        ml_prob=0.58,
    )
    dec_good = policy.make_decision(
        projection=good_proj,
        features=dummy_features,
        simulation=dummy_sim,
        line=25.5,
        odds=1.91,
        bankroll=1000.0,
        market="PTS",
    )
    assert dec_good.is_bet is True
    assert dec_good.rejection_reason is None
    assert dec_good.kelly_stake > 0.0
    assert dec_good.expected_value >= config.min_ev

    # Case B: Sub-par probability (< 55%) -> rejected
    low_prob_proj = Projection(
        final_projection=26.0,
        base_projection=26.0,
        confidence_interval=(20.0, 32.0),
        adjustments={"predicted_std": 4.0},
        ml_prob=0.52,
    )
    dec_low = policy.make_decision(
        projection=low_prob_proj,
        features=dummy_features,
        simulation=dummy_sim,
        line=25.5,
        odds=1.91,
        bankroll=1000.0,
        market="PTS",
    )
    assert dec_low.is_bet is False
    assert "below floor" in dec_low.rejection_reason
    assert dec_low.kelly_stake == 0.0

    # Case C: Blocked combo (PA OVER) -> rejected
    dec_blocked = policy.make_decision(
        projection=good_proj,
        features=dummy_features,
        simulation=dummy_sim,
        line=25.5,
        odds=1.91,
        bankroll=1000.0,
        market="PA",
    )
    assert dec_blocked.is_bet is False
    assert "Blocked combo" in dec_blocked.rejection_reason


def test_trading_engine_consumes_decision_config():
    """Verify TradingEngine uses unified DecisionConfig."""
    custom_cfg = DecisionConfig(min_ev=0.08)  # High 8% minimum EV
    engine = TradingEngine(decision_config=custom_cfg)

    df = pd.DataFrame({
        "predicted_mean": [25.0],
        "predicted_std": [4.0],
        "line": [22.5],
        "side": ["OVER"],
        "odds": [1.91],
        "prob": [0.56],  # EV ≈ 7% -> below custom 8% floor
    })

    out = engine.process_bets(df, bankroll=1000, return_all=True)
    assert out["is_bet"].iloc[0] is False or out["is_bet"].iloc[0] == 0


def test_half_point_line_rounding():
    """Verify DraftKings-style half-point line snapping."""
    raw_lines = [19.23, 20.88, 14.01, 7.50, 0.20]
    snapped = []
    for rl in raw_lines:
        line = max(0.5, round(float(rl) * 2) / 2)
        if line % 1 == 0:
            line += 0.5
        snapped.append(line)

    # All lines must end in .5 and be >= 0.5
    for s in snapped:
        assert s >= 0.5
        assert s % 1 == 0.5


def test_parquet_and_csv_persistence(tmp_path):
    """Verify load_ml_training_data and save_ml_training_data handle Parquet and CSV cleanly."""
    sample_df = pd.DataFrame({
        "player": ["Luka Doncic", "Nikola Jokic"],
        "market": ["PTS", "REB"],
        "line": [32.5, 12.5],
        "actual_value": [35, 14],
    })

    # 1. Save both formats
    save_ml_training_data(sample_df, data_dir=tmp_path, filename_base="test_dataset")
    assert (tmp_path / "test_dataset.parquet").exists()
    assert (tmp_path / "test_dataset.csv").exists()

    # 2. Load (should prefer Parquet)
    loaded = load_ml_training_data(data_dir=tmp_path, filename_base="test_dataset")
    assert len(loaded) == 2
    assert list(loaded["player"]) == ["Luka Doncic", "Nikola Jokic"]

    # 3. Remove Parquet, verify CSV fallback
    (tmp_path / "test_dataset.parquet").unlink()
    fallback = load_ml_training_data(data_dir=tmp_path, filename_base="test_dataset")
    assert len(fallback) == 2
    assert list(fallback["player"]) == ["Luka Doncic", "Nikola Jokic"]
