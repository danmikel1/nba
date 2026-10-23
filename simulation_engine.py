"""
simulation_engine.py — CDF-based and Monte Carlo Simulation Engine

V20 EMPIRICAL: Pure CDF probability calculations and standard Gaussian
sampling for histogram/distribution rendering.
"""

from typing import Optional, Any
import numpy as np
from scipy import stats

from core_models import SimulationResult, FeatureVector


class SimulationEngine:
    """
    Simulation & CDF Probability Calculation Engine.
    
    Formula: P(Over) = 1 - Phi((line - mean) / std)
    """
    
    def __init__(self, config: Optional[Any] = None):
        self.config = config
    
    def compute_cdf_probability(
        self,
        mean: float,
        std: float,
        line: float,
        market: Optional[str] = None,
        features: Optional[FeatureVector] = None
    ) -> SimulationResult:
        """Pure CDF probability calculation."""
        if std <= 0:
            std = 1.0
        
        z_score = (line - mean) / std
        p_under = stats.norm.cdf(z_score)
        p_over = 1.0 - p_under
        
        ci_10 = mean - 1.28 * std
        ci_90 = mean + 1.28 * std
        sims = np.array([mean])
        
        return SimulationResult(
            over_prob=float(max(0.001, min(0.999, p_over))),
            under_prob=float(max(0.001, min(0.999, p_under))),
            median=mean,
            ci_10=max(0, ci_10),
            ci_90=ci_90,
            simulations=sims
        )
    
    def run_simulation(
        self, 
        mean: float, 
        std: float, 
        line: float, 
        market: Optional[str] = None,
        features: Optional[FeatureVector] = None,
        simulations: int = 1000,
        use_mixture: bool = False
    ) -> SimulationResult: 
        """Simple normal-distribution Monte Carlo for histogram generation."""
        if std <= 0:
            std = 1.0
        
        sims = np.random.normal(mean, std, simulations)
        sims = np.maximum(sims, 0)
        
        over_rate = float((sims > line).mean())
        under_rate = float((sims <= line).mean())
        
        return SimulationResult(
            over_prob=max(0.001, min(0.999, over_rate)),
            under_prob=max(0.001, min(0.999, under_rate)),
            median=float(np.median(sims)),
            ci_10=float(np.percentile(sims, 10)),
            ci_90=float(np.percentile(sims, 90)),
            simulations=sims
        )
    
    def run_cdf_simulation(
        self, 
        predicted_mean: float, 
        predicted_std: float, 
        line: float, 
        market: Optional[str] = None
    ) -> SimulationResult:
        """Alias for compute_cdf_probability."""
        return self.compute_cdf_probability(
            mean=predicted_mean,
            std=predicted_std,
            line=line,
            market=market
        )

    def _estimate_pairwise_correlation(self, leg_a: dict, leg_b: dict) -> float:
        """
        Estimate pairwise Pearson correlation coefficient between two prop legs.
        Accounts for:
          - Same player across different markets (e.g. PTS & 3PT, PTS & AST)
          - Teammates on the same squad (assist synergy vs usage cannibalization)
          - Opponents in the same game (shootout pace correlation)
          - Independent cross-game legs (zero correlation)
        """
        player_a = str(leg_a.get("player", "")).strip().lower()
        player_b = str(leg_b.get("player", "")).strip().lower()
        m_a = str(leg_a.get("market", "")).upper()
        m_b = str(leg_b.get("market", "")).upper()
        team_a = str(leg_a.get("team", "")).strip().upper()
        team_b = str(leg_b.get("team", "")).strip().upper()
        opp_a = str(leg_a.get("opponent", "")).strip().upper()
        opp_b = str(leg_b.get("opponent", "")).strip().upper()
        side_a = str(leg_a.get("side", "OVER")).upper()
        side_b = str(leg_b.get("side", "OVER")).upper()

        # 1. Same Player
        if player_a and player_b and player_a == player_b:
            if m_a == m_b:
                return 1.0 if side_a == side_b else -1.0
            pair = {m_a, m_b}
            if pair == {"PTS", "3PT"}:
                return 0.65
            if pair == {"PTS", "PRA"}:
                return 0.75
            if pair == {"REB", "PRA"}:
                return 0.50
            if pair == {"AST", "PRA"}:
                return 0.45
            if pair == {"PTS", "AST"}:
                return 0.25
            if pair == {"PTS", "REB"}:
                return 0.15
            if pair == {"REB", "AST"}:
                return 0.10
            return 0.20

        # Check team / game alignment
        is_same_team = (team_a and team_b and team_a == team_b) or (opp_a and opp_b and opp_a == opp_b)
        is_opponents = (team_a and opp_b and team_a == opp_b) or (opp_a and team_b and opp_a == team_b)

        # 2. Teammates
        if is_same_team:
            if (m_a == "AST" and m_b == "PTS") or (m_a == "PTS" and m_b == "AST"):
                return 0.30  # Direct assist-to-bucket synergy
            if m_a == "PTS" and m_b == "PTS":
                return -0.15  # Shot/usage cannibalization
            if m_a == "REB" and m_b == "REB":
                return -0.15  # Rebound competition
            if m_a == "AST" and m_b == "AST":
                return -0.10  # Playmaking distribution
            return 0.05

        # 3. Opponents in the same game
        if is_opponents:
            if (m_a in {"PTS", "3PT"}) and (m_b in {"PTS", "3PT"}):
                return 0.20  # Fast-paced shootout benefits both teams
            return 0.0

        # 4. Independent cross-game props
        return 0.0

    def simulate_correlated_parlay(
        self,
        legs: list,
        correlation_matrix: Optional[np.ndarray] = None,
        n_simulations: int = 100_000,
    ) -> dict:
        """
        Multivariate Monte Carlo simulation for correlated Same-Game Parlays (SGP).
        Models the true joint probability distribution across multiple player props.
        """
        n_legs = len(legs)
        if n_legs == 0:
            return {"joint_prob": 0.0, "independent_prob": 0.0, "correlation_edge_pct": 0.0}

        means = np.array([float(l.get("mean", l.get("proj", l.get("line", 0)))) for l in legs])
        sigmas = np.array([max(0.5, float(l.get("std", 4.0))) for l in legs])
        lines = np.array([float(l["line"]) for l in legs])
        sides = [str(l.get("side", "OVER")).upper() for l in legs]

        # Build correlation matrix if not supplied
        if correlation_matrix is None:
            R = np.eye(n_legs)
            for i in range(n_legs):
                for j in range(i + 1, n_legs):
                    corr = self._estimate_pairwise_correlation(legs[i], legs[j])
                    R[i, j] = R[j, i] = corr
        else:
            R = np.asarray(correlation_matrix, dtype=float)

        # Cholesky decomposition of correlation matrix with positive-semidefinite protection
        try:
            R = (R + R.T) / 2.0
            np.fill_diagonal(R, 1.0)
            eigvals, eigvecs = np.linalg.eigh(R)
            eigvals = np.maximum(eigvals, 1e-4)
            R_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
            d = np.sqrt(np.diag(R_psd))
            R_norm = R_psd / np.outer(d, d)
            L = np.linalg.cholesky(R_norm)
        except Exception:
            L = np.eye(n_legs)

        # Generate correlated multivariate normal draws: (n_simulations, n_legs)
        Z = np.random.standard_normal((n_simulations, n_legs))
        correlated_Z = Z @ L.T
        draws = means + correlated_Z * sigmas

        # Evaluate leg hits
        hits_matrix = np.zeros((n_simulations, n_legs), dtype=bool)
        for i in range(n_legs):
            if sides[i] == "OVER":
                hits_matrix[:, i] = draws[:, i] > lines[i]
            else:
                hits_matrix[:, i] = draws[:, i] < lines[i]

        all_hit = np.all(hits_matrix, axis=1)
        joint_prob = float(np.mean(all_hit))

        # Independent probability (what bookmakers assume without correlation)
        ind_probs = [float(np.mean(hits_matrix[:, i])) for i in range(n_legs)]
        independent_prob = float(np.prod(ind_probs))

        corr_edge = joint_prob - independent_prob
        fair_odds = 1.0 / max(joint_prob, 1e-5)

        return {
            "n_legs": n_legs,
            "joint_prob": round(joint_prob, 4),
            "independent_prob": round(independent_prob, 4),
            "correlation_edge_pct": round((corr_edge / max(independent_prob, 1e-5)) * 100, 2),
            "fair_decimal_odds": round(fair_odds, 2),
            "individual_probs": [round(p, 3) for p in ind_probs],
        }
