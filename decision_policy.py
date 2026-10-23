"""
decision_policy.py — Unified Decision Policy & Risk Configuration

Calculates Expected Value (EV), fractional Kelly stake sizing, temperature scaling,
and sniper decision gates shared across single-bet analysis and batch risk engine.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Set, Tuple

import numpy as np
from scipy.special import expit, logit

from core_models import BetDecision, FeatureVector, Projection, SimulationResult

logger = logging.getLogger(__name__)

DECISION_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "decision_config.json"


@dataclass
class DecisionConfig:
    """Unified decision and risk management configuration."""
    min_ev: float = 0.02           # 2% minimum edge
    max_ev: float = 0.30           # 30% ceiling (live ROI collapses above 30%)
    min_prob: float = 0.55         # 55% win probability floor
    prob_cap: float = 0.68         # 68% overconfidence cap
    prob_shrink_factor: float = 0.85
    min_qs: float = 0.005          # Quality score floor
    max_qs: float = 0.035          # Quality score ceiling
    kelly_fraction: float = 0.25   # Quarter Kelly
    max_stake_pct: float = 0.03    # 3% bankroll ceiling per bet
    min_stake_pct: float = 0.005   # 0.5% dust filter
    blocked_combos: Set[Tuple[str, str]] = field(default_factory=lambda: {('PA', 'OVER')})
    sniper_mode: bool = True
    sniper_max_bets: int = 25
    sniper_max_per_opponent: int = 2

    @classmethod
    def load_from_file(cls, path: Path = DECISION_CONFIG_PATH) -> "DecisionConfig":
        """Load configuration from JSON file with fallback to defaults."""
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                # Convert blocked_combos from list of lists to set of tuples
                raw_blocked = data.get("blocked_combos", [["PA", "OVER"]])
                blocked = {tuple(x) for x in raw_blocked}
                return cls(
                    min_ev=float(data.get("min_ev", 0.02)),
                    max_ev=float(data.get("max_ev", 0.30)),
                    min_prob=float(data.get("min_prob", 0.55)),
                    prob_cap=float(data.get("prob_cap", 0.68)),
                    prob_shrink_factor=float(data.get("prob_shrink_factor", 0.85)),
                    min_qs=float(data.get("min_qs", 0.005)),
                    max_qs=float(data.get("max_qs", 0.035)),
                    kelly_fraction=float(data.get("kelly_fraction", 0.25)),
                    max_stake_pct=float(data.get("max_stake_pct", 0.03)),
                    min_stake_pct=float(data.get("min_stake_pct", 0.005)),
                    blocked_combos=blocked,
                    sniper_mode=bool(data.get("sniper_mode", True)),
                    sniper_max_bets=int(data.get("sniper_max_bets", 25)),
                    sniper_max_per_opponent=int(data.get("sniper_max_per_opponent", 2)),
                )
            except Exception as e:
                logger.warning(f"Failed to load decision config from {path}: {e}")
        return cls()


class DecisionPolicy:
    """
    ML-Driven Decision Engine.
    
    All decisions derive from:
    - Kelly Criterion (mathematical optimal sizing)
    - ML model outputs (probability, projection, quantile-derived std)
    - Live temperature scaling & probability shrinkage
    - Unified sniper decision gates
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        decision_config: Optional[DecisionConfig] = None,
    ):
        self.config = config
        self.decision_config = decision_config or DecisionConfig.load_from_file()
        # Honor config.KELLY_FRACTION override if provided by legacy Config object
        self.KELLY_FRACTION: float = (
            getattr(config, "KELLY_FRACTION", self.decision_config.kelly_fraction)
            if config else self.decision_config.kelly_fraction
        )

    def calculate_ev(self, prob: float, odds: float) -> float:
        """Calculate expected value. EV = p(o-1) - (1-p)."""
        return (prob * (odds - 1)) - (1 - prob)

    def calculate_kelly_stake(
        self,
        prob: float,
        odds: float,
        bankroll: float,
    ) -> Tuple[float, float]:
        """
        True Kelly Criterion stake sizing.
        
        f* = (bp - q) / b   where b = odds-1, p = win_prob, q = 1-p
        Stake = bankroll * f* * KELLY_FRACTION
        """
        if prob <= 0 or prob >= 1 or odds <= 1 or bankroll <= 0:
            return 0.0, 0.0

        b = odds - 1
        q = 1.0 - prob
        f_star = (b * prob - q) / b

        if f_star <= 0:
            return 0.0, 0.0

        fraction = f_star * self.KELLY_FRACTION
        stake = bankroll * fraction

        return stake, fraction

    def make_decision(
        self,
        projection: Projection,
        features: FeatureVector,
        simulation: SimulationResult,
        line: float,
        odds: float = 1.91,
        bankroll: float = 10000.0,
        market: Optional[str] = None,
    ) -> BetDecision:
        """
        Primary decision maker using ML quantile regression predictions and sniper gates.
        """
        ml_prob_over = projection.ml_prob if projection.ml_prob is not None else None

        if ml_prob_over is not None:
            if ml_prob_over > 0.5:
                side = "OVER"
                prob = ml_prob_over
            else:
                side = "UNDER"
                prob = 1.0 - ml_prob_over
        else:
            # Fallback to simulation CDF only when ML model probability is unavailable
            prob_over_cdf = simulation.over_prob
            prob_under_cdf = simulation.under_prob
            if prob_over_cdf > prob_under_cdf:
                side = "OVER"
                prob = prob_over_cdf
            else:
                side = "UNDER"
                prob = prob_under_cdf

        # Temperature scaling for sportsbook calibration
        T = getattr(self.config, "LIVE_CALIBRATION_TEMPERATURE", 1.0) if self.config else 1.0
        if T > 1.0:
            prob = float(expit(logit(np.clip(prob, 0.02, 0.98)) / T))

        # Expected Value & Sizing
        ev = self.calculate_ev(prob, odds)
        stake, fraction = self.calculate_kelly_stake(prob, odds, bankroll)

        predicted_mean = projection.final_projection
        predicted_std = (
            projection.adjustments.get("predicted_std", features.std)
            if projection.adjustments else features.std
        )
        quality_score = ev / predicted_std if (ev > 0 and predicted_std > 0) else 0.0

        # Evaluate Unified Sniper Decision Gates
        is_bet = True
        rejection_reason = None
        mkt = (market or getattr(features, "market", "")).upper()

        if mkt and (mkt, side) in self.decision_config.blocked_combos:
            is_bet = False
            rejection_reason = f"Blocked combo ({mkt} {side})"
        elif prob < self.decision_config.min_prob:
            is_bet = False
            rejection_reason = f"Probability {prob:.1%} below floor {self.decision_config.min_prob:.1%}"
        elif ev < self.decision_config.min_ev:
            is_bet = False
            rejection_reason = f"EV {ev:.1%} below floor {self.decision_config.min_ev:.1%}"
        elif ev > self.decision_config.max_ev:
            is_bet = False
            rejection_reason = f"EV {ev:.1%} above ceiling {self.decision_config.max_ev:.1%}"
        elif quality_score < self.decision_config.min_qs:
            is_bet = False
            rejection_reason = f"Quality score {quality_score:.4f} below floor {self.decision_config.min_qs}"
        elif quality_score > self.decision_config.max_qs:
            is_bet = False
            rejection_reason = f"Quality score {quality_score:.4f} above ceiling {self.decision_config.max_qs}"

        if not is_bet:
            stake = 0.0
            fraction = 0.0
        else:
            # Apply maximum stake ceiling and minimum dust threshold
            fraction = min(fraction, self.decision_config.max_stake_pct)
            if fraction < self.decision_config.min_stake_pct:
                stake = 0.0
                fraction = 0.0
                is_bet = False
                rejection_reason = "Dust stake below minimum stake percent"
            else:
                stake = bankroll * fraction

        return BetDecision(
            recommended_side=side,
            probability=prob,
            expected_value=ev,
            predicted_mean=predicted_mean,
            predicted_std=predicted_std,
            kelly_stake=stake,
            kelly_fraction=fraction,
            quality_score=quality_score,
            is_bet=is_bet,
            rejection_reason=rejection_reason,
        )
