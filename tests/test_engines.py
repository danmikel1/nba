"""
tests/test_engines.py — Unit tests for extracted engines.
"""
from pathlib import Path
import numpy as np
import pytest

from core_models import FeatureVector, Projection, SimulationResult, BetDecision
from decision_policy import DecisionPolicy
from simulation_engine import SimulationEngine
from model_engine import ModelEngine


def test_decision_policy_ev():
    policy = DecisionPolicy()
    # At odds 2.0 (even money), 60% probability has EV = 0.60 * 1 - 0.40 = +0.20 (+20%)
    ev = policy.calculate_ev(0.60, 2.0)
    assert abs(ev - 0.20) < 1e-5


def test_decision_policy_kelly():
    policy = DecisionPolicy()
    # At odds 2.0 (b=1), p=0.60, q=0.40 -> f* = (1*0.6 - 0.4)/1 = 0.20
    # Quarter Kelly -> fraction = 0.05 (5%)
    # Bankroll = 1000 -> stake = 50.0
    stake, frac = policy.calculate_kelly_stake(0.60, 2.0, 1000.0)
    assert abs(frac - 0.05) < 1e-4
    assert abs(stake - 50.0) < 1e-4


def test_simulation_engine_cdf():
    engine = SimulationEngine()
    # Mean 20.0, line 20.0 -> P(Over) = 0.50
    res = engine.compute_cdf_probability(mean=20.0, std=4.0, line=20.0)
    assert abs(res.over_prob - 0.50) < 1e-3
    assert abs(res.under_prob - 0.50) < 1e-3
    assert res.ci_10 < 20.0 < res.ci_90


def test_model_engine_initialization():
    engine = ModelEngine()
    assert len(engine.ml_models) > 0
    assert len(engine.calibrators) > 0
