import pandas as pd
import numpy as np
from scipy.stats import norm

class TradingEngine:
    def calculate_kelly_stake(self, win_prob, odds, bankroll, fractional=0.10):
        """
        Calculate the optimal stake using the Kelly Criterion.

        Args:
            win_prob (float): Win probability between 0 and 1.
            odds (float): Decimal odds.
            bankroll (float): Total bankroll.
            fractional (float): Fractional Kelly multiplier, default 0.10 (conservative).

        Returns:
            float: Recommended stake amount.
        """
        if win_prob <= 0 or win_prob >= 1 or odds <= 1 or bankroll <= 0:
            return 0.0
        
        b = odds - 1
        p = win_prob
        q = 1 - p
        f_star = (b * p - q) / b
        stake = bankroll * (f_star * fractional)
        max_stake = 0.05 * bankroll
        return min(stake, max_stake)

    def process_bets(self, df, bankroll, ev_threshold=0.025, fractional_kelly=0.20, default_sigma: float = 3.0, sigma_multiplier: float = 1.6, z_score_threshold: float = 0.0, require_abs_z: bool = False, max_bets=8, return_all: bool = False):
        """
        Risk-Pricing: Convert model log-variance to sigma and price bets with uncertainty filtering.

        Notes:
        - Defaults tuned from backtest sweep (fractional_kelly=0.10, default_sigma=3.0).
        - `sigma_multiplier` is a runtime calibration multiplier applied to recovered sigma. Use this
          as a short-term mitigation when the variance model underestimates uncertainty (e.g. use
          multiplier ~= mean(MAE/sigma) observed in calibration diagnostics).

        Steps (per Risk Pricing spec):
        1) Recover sigma from log-variance: sigma = sqrt(expm1(df['predicted_std']))
           - Fill NaNs with a default sigma value (default_sigma)
           - Floor sigma: sigma = max(sigma, 0.7 * df['feat_std']) if 'feat_std' exists
           - Apply `sigma_multiplier` to produce a calibrated sigma used downstream
        2) True prob: use Normal CDF with z_metric = (line - predicted_mean) / sigma
        3) Confidence z-score: z_score = abs(predicted_mean - line) / sigma
        4) Uncertainty filter: stake_multiplier = {0 if z<0.35, 0.5 if 0.35<=z<0.6, 1.0 if z>=0.6}
        5) Pricing score: pricing_score = adjusted_ev * (z_score / (1 + sigma))
           - adjusted_ev = true_ev (no additional penalty here; keep simple)
        6) Filter: keep only rows where stake_multiplier > 0
        7) Sort by pricing_score descending and return
        """
        df = df.copy()

        # --- 1) Recover sigma (model stores log-variance: log(1 + var)) ---
        # Treat NaNs by substituting log(1 + default_sigma^2)
        safe_logvar = df['predicted_std'].copy()
        default_logvar = np.log1p(default_sigma ** 2)
        safe_logvar = safe_logvar.fillna(default_logvar)

        # Invert log1p -> var, then take sqrt
        with np.errstate(over='ignore', invalid='ignore'):
            sigma = np.sqrt(np.expm1(safe_logvar.astype(float)))

        # Floor sigma to 0.7 * empirical feature std if available, else ensure a sane lower bound
        if 'feat_std' in df.columns:
            floor = 0.7 * df['feat_std'].fillna(1.0)
        else:
            floor = 0.7
        sigma = np.maximum(sigma, floor)

        # Apply runtime calibration multiplier (short-term mitigation for underestimation)
        sigma = sigma * float(sigma_multiplier)

        # Safety: replace any remaining NaN/inf with default_sigma
        sigma = np.where(np.isfinite(sigma), sigma, default_sigma)

        # --- 2) True probability via Normal CDF ---
        z_metric = (df['line'] - df['predicted_mean']) / sigma
        prob = np.where(df['side'] == 'UNDER', norm.cdf(z_metric), 1 - norm.cdf(z_metric))

        # --- 3) Confidence z-score (distance in sigmas)
        z_score = np.abs(df['predicted_mean'] - df['line']) / sigma

        # --- 4) Uncertainty filter -> stake multiplier
        stake_multiplier = np.zeros_like(z_score, dtype=float)
        stake_multiplier[z_score >= 0.6] = 1.0
        mid_mask = (z_score >= 0.35) & (z_score < 0.6)
        stake_multiplier[mid_mask] = 0.5

        # --- 5) Compute EV and pricing score (Hunger Games flow) ---
        # Compute nominal EV from model-derived prob (keep existing prob if available)
        true_ev = prob * (df['odds'] - 1) - (1 - prob)

        # Apply stake multipliers determined earlier
        # Calculate Kelly stakes (conservative fractional_kelly applied)
        raw_kelly = np.array([self.calculate_kelly_stake(p, o, bankroll, fractional_kelly) for p, o in zip(df.get('predicted_prob', prob), df['odds'])])
        raw_kelly = np.maximum(raw_kelly, 0.0)
        rec_stake = raw_kelly * stake_multiplier

        # Quality ranking: combine profitability and signal strength
        # Prefer columns expected by downstream systems (predicted_ev, projected_value)
        quality_score = (df.get('predicted_ev', true_ev).astype(float)) * (z_score)

        df['quality_score'] = quality_score
        df['rec_stake'] = rec_stake
        df['predicted_ev'] = df.get('predicted_ev', true_ev)
        df['predicted_prob'] = df.get('predicted_prob', prob)

        # --- Compatibility columns (restore original outputs expected elsewhere) ---
        adjusted_ev = true_ev  # legacy name; keep equal to true_ev for now
        pricing_score = adjusted_ev * (z_score / (1.0 + sigma))

        df['sigma'] = sigma
        df['z_metric'] = z_metric
        df['prob'] = prob
        df['win_prob'] = prob
        df['z_score'] = z_score
        df['stake_multiplier'] = stake_multiplier
        df['true_ev'] = true_ev
        df['adjusted_ev'] = adjusted_ev
        df['pricing_score'] = pricing_score
        df['rec_stake'] = rec_stake
        df['is_bet'] = df['rec_stake'] > 0
        df['vol_adj_rank'] = np.where(sigma == 0, 0.0, (df['adjusted_ev'].astype(float) / sigma))

        # Sort by Quality, keep only top N (the Hunger Games)
        df = df.sort_values(by='quality_score', ascending=False).copy()
        if len(df) > max_bets:
            # zero out stakes for everyone below the top `max_bets`
            df.iloc[max_bets:, df.columns.get_loc('rec_stake')] = 0.0
            df.iloc[max_bets:, df.columns.get_loc('is_bet')] = False

        # Ensure rows failing preliminary filters (stake_multiplier, require_abs_z) have rec_stake == 0
        if 'stake_multiplier' in df.columns:
            df.loc[df['stake_multiplier'] <= 0, 'rec_stake'] = 0.0
            df.loc[df['stake_multiplier'] <= 0, 'is_bet'] = False

        if require_abs_z and z_score_threshold > 0.0:
            df.loc[df['z_score'] < z_score_threshold, 'rec_stake'] = 0.0
            df.loc[df['z_score'] < z_score_threshold, 'is_bet'] = False

        # Default behavior: filter out zero-stake rows (backwards-compatible for tests and execution).
        if not return_all:
            out = df[df['rec_stake'] > 0].copy()
            # optional absolute z requirement (legacy behavior)
            if require_abs_z and z_score_threshold > 0.0:
                out = out[out['z_score'] >= z_score_threshold]
            return out

        # return_all=True -> return full table including rejected rows (for UI/display)
        return df
