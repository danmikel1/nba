import pandas as pd
import numpy as np
from scipy.stats import norm as scipy_norm  # renamed — select_safety_parlay had a local `norm` shadowing this
import logging
from datetime import datetime
import os
import json
from typing import Optional, Any

logger = logging.getLogger(__name__)

from decision_policy import DecisionConfig, DecisionPolicy

# Load unified decision configuration
DEFAULT_DECISION_CONFIG = DecisionConfig.load_from_file()

# --- SNIPER SETTINGS ---
SNIPER_MODE = DEFAULT_DECISION_CONFIG.sniper_mode
SNIPER_MAX_BETS = DEFAULT_DECISION_CONFIG.sniper_max_bets
SNIPER_MAX_PER_OPPONENT = DEFAULT_DECISION_CONFIG.sniper_max_per_opponent

# Entry gates
MIN_EV_THRESHOLD = DEFAULT_DECISION_CONFIG.min_ev
MAX_EV_THRESHOLD = DEFAULT_DECISION_CONFIG.max_ev
MIN_QS_THRESHOLD = DEFAULT_DECISION_CONFIG.min_qs
MAX_QS_THRESHOLD = DEFAULT_DECISION_CONFIG.max_qs
MIN_PROB = DEFAULT_DECISION_CONFIG.min_prob

# Probability calibration
PROB_CAP = DEFAULT_DECISION_CONFIG.prob_cap
PROB_SHRINK_RANGE = (0.60, PROB_CAP)
PROB_SHRINK_FACTOR = DEFAULT_DECISION_CONFIG.prob_shrink_factor
BLOCKED_COMBOS = DEFAULT_DECISION_CONFIG.blocked_combos

# Staking
KELLY_FRACTION = DEFAULT_DECISION_CONFIG.kelly_fraction
MAX_STAKE_PCT = DEFAULT_DECISION_CONFIG.max_stake_pct
MIN_STAKE_PCT = DEFAULT_DECISION_CONFIG.min_stake_pct
DEFAULT_RESULTS_EXPORT = os.path.join(os.path.dirname(__file__), 'data', '2026-03-30T06-19_export.csv')

# Adaptive filter controls (off by default; opt-in per call)
AUTO_TUNE_MIN_SETTLED = 300
AUTO_TUNE_MAX_RELAX = 0.80
AUTO_TUNE_MAX_BET_BOOST = 2.5


class TradingEngine:

    def __init__(self, decision_config: Optional[DecisionConfig] = None):
        self.decision_config = decision_config or DecisionConfig.load_from_file()
        self.decision_policy = DecisionPolicy(decision_config=self.decision_config)
        # Optional market-level multipliers learned from model result exports.
        # Keys are market labels (e.g., PTS/REB/AST), values are stake multipliers.
        self.market_reliability = {}

    def configure_from_results(self, export_csv: str) -> dict:
        """
        Build market reliability multipliers from a model export CSV.

        Expected columns include: Model, R², Coverage %, CV Std.
        Returns a dict like {'PTS': 1.01, 'REB': 0.92, 'UNIVERSAL': 1.08}.
        """
        if not export_csv or not os.path.exists(export_csv):
            logger.warning(f"Results export not found: {export_csv}")
            self.market_reliability = {}
            return self.market_reliability

        try:
            perf = pd.read_csv(export_csv)
        except Exception as exc:
            logger.warning(f"Failed to read results export {export_csv}: {exc}")
            self.market_reliability = {}
            return self.market_reliability

        required_cols = {'Model', 'R²', 'Coverage %', 'CV Std'}
        if not required_cols.issubset(set(perf.columns)):
            logger.warning(
                f"Results export missing required columns. Found: {list(perf.columns)}"
            )
            self.market_reliability = {}
            return self.market_reliability

        multipliers = {}
        for _, row in perf.iterrows():
            model = str(row.get('Model', '')).strip().upper()
            if not model:
                continue

            r2 = float(row.get('R²', 0.0))
            cv_std = float(row.get('CV Std', 0.0))

            coverage_raw = row.get('Coverage %', 0.0)
            if isinstance(coverage_raw, str):
                coverage = float(coverage_raw.replace('%', '').strip()) / 100.0
            else:
                coverage = float(coverage_raw)
                if coverage > 1.0:
                    coverage = coverage / 100.0

            # Normalize to [0, 1] around observed NBA model ranges.
            r2_score = np.clip((r2 - 0.40) / 0.30, 0.0, 1.0)
            stability_score = np.clip(1.0 - (cv_std / 0.20), 0.0, 1.0)
            coverage_score = np.clip(1.0 - abs(coverage - 0.89) / 0.10, 0.0, 1.0)

            reliability = 0.50 * r2_score + 0.30 * stability_score + 0.20 * coverage_score

            # Keep adjustments moderate: do not overreact to one export snapshot.
            multiplier = float(np.clip(0.75 + 0.35 * reliability, 0.75, 1.10))
            multipliers[model] = multiplier

        self.market_reliability = multipliers
        if multipliers:
            logger.info(
                "Loaded market reliability multipliers: "
                + ", ".join([f"{k}={v:.3f}" for k, v in sorted(multipliers.items())])
            )
        return self.market_reliability

    def _get_market_multiplier(self, market_value: str) -> float:
        if not self.market_reliability:
            return 1.0
        if market_value is None:
            return self.market_reliability.get('UNIVERSAL', 1.0)
        key = str(market_value).strip().upper()
        return self.market_reliability.get(key, self.market_reliability.get('UNIVERSAL', 1.0))

    def _compute_adaptive_gates(
        self,
        tracker_file: str,
        min_settled: int,
        current_max_bets: int,
    ) -> dict:
        """
        Adapt floor thresholds if historical filtering is inverted.

        If rejected bets win more often than accepted bets and rejections dominate volume,
        relax EV/QS floors and increase max_bets for this run.
        """
        gate_state = {
            'min_ev': MIN_EV_THRESHOLD,
            'min_qs': MIN_QS_THRESHOLD,
            'max_bets': current_max_bets,
            'filter_edge': 0.0,
            'accepted_count': 0,
            'rejected_count': 0,
            'adapted': False,
        }

        if not tracker_file or not os.path.exists(tracker_file):
            return gate_state

        try:
            with open(tracker_file, 'r', encoding='utf-8') as f:
                bets = json.load(f)
        except Exception as exc:
            logger.warning(f"Adaptive gates: failed to read tracker file {tracker_file}: {exc}")
            return gate_state

        if not isinstance(bets, list):
            return gate_state

        decided = [
            b for b in bets
            if isinstance(b, dict) and b.get('result') in ('Win', 'Loss')
        ]
        if len(decided) < int(min_settled):
            return gate_state

        accepted = [b for b in decided if float(b.get('stake', 0) or 0) > 0]
        rejected = [b for b in decided if float(b.get('stake', 0) or 0) == 0]

        if not accepted or not rejected:
            return gate_state

        accepted_wr = sum(1 for b in accepted if b.get('result') == 'Win') / len(accepted)
        rejected_wr = sum(1 for b in rejected if b.get('result') == 'Win') / len(rejected)
        filter_edge = accepted_wr - rejected_wr
        reject_ratio = len(rejected) / max(len(accepted), 1)

        gate_state['filter_edge'] = float(filter_edge)
        gate_state['accepted_count'] = int(len(accepted))
        gate_state['rejected_count'] = int(len(rejected))

        # Trigger adaptation only when both quality inversion and over-rejection are present.
        if filter_edge >= 0 or reject_ratio <= 1.20:
            return gate_state

        edge_pressure = min(1.0, abs(filter_edge) / 0.05)
        volume_pressure = min(1.0, (reject_ratio - 1.20) / 2.00)
        pressure = 0.60 * edge_pressure + 0.40 * volume_pressure

        relax = AUTO_TUNE_MAX_RELAX * pressure
        min_ev = max(0.0, MIN_EV_THRESHOLD * (1.0 - relax))
        min_qs = max(0.0, MIN_QS_THRESHOLD * (1.0 - relax))
        max_bets_boost = 1.0 + (AUTO_TUNE_MAX_BET_BOOST - 1.0) * pressure
        tuned_max_bets = max(current_max_bets, int(round(current_max_bets * max_bets_boost)))

        gate_state['min_ev'] = float(min_ev)
        gate_state['min_qs'] = float(min_qs)
        gate_state['max_bets'] = int(tuned_max_bets)
        gate_state['adapted'] = True

        logger.warning(
            "Adaptive gates triggered: "
            f"edge={filter_edge:+.2%}, reject_ratio={reject_ratio:.2f}, "
            f"min_ev {MIN_EV_THRESHOLD:.3%}->{min_ev:.3%}, "
            f"min_qs {MIN_QS_THRESHOLD:.4f}->{min_qs:.4f}, "
            f"max_bets {current_max_bets}->{tuned_max_bets}"
        )
        return gate_state

    def calculate_kelly_stake(
        self,
        win_prob: float,
        odds: float,
        bankroll: float,
        fractional: float = KELLY_FRACTION,
        max_stake_pct: float = MAX_STAKE_PCT,
    ) -> float:
        """
        Fractional Kelly stake.

        Args:
            win_prob:      Win probability [0, 1].
            odds:          Decimal odds (must be > 1).
            bankroll:      Current bankroll.
            fractional:    Kelly fraction (0.25 = quarter Kelly).
            max_stake_pct: Hard cap as fraction of bankroll.

        Returns:
            Recommended stake in currency units.
        """
        if win_prob <= 0 or win_prob >= 1 or odds <= 1 or bankroll <= 0:
            return 0.0

        b = odds - 1
        p = win_prob
        q = 1 - p
        f_star = (b * p - q) / b

        if f_star <= 0:
            return 0.0

        stake = bankroll * f_star * fractional
        return min(stake, max_stake_pct * bankroll)

    def save_performance_metrics(
        self,
        df: pd.DataFrame,
        bankroll: float,
        performance_file: str = 'data/performance.csv',
        batch_date: str = None,
    ) -> dict:
        """Append batch-level system metrics to performance.csv."""
        if batch_date is None:
            batch_date = datetime.utcnow().isoformat()

        total_stake = float(df['rec_stake'].sum()) if 'rec_stake' in df.columns else 0.0

        if 'result' in df.columns:
            pnl = df.apply(
                lambda r: r['rec_stake'] * (r['odds'] - 1)
                if r['result'] == 'Win'
                else (-r['rec_stake'] if r['result'] == 'Loss' else 0.0),
                axis=1,
            ).sum()
            hit_rate = float((df['result'] == 'Win').mean()) if len(df) else 0.0
        else:
            pnl = 0.0
            hit_rate = 0.0

        avg_ev  = float(df['ev'].mean())  if 'ev'   in df.columns else 0.0
        avg_prob = float(df['prob'].mean()) if 'prob' in df.columns else 0.0
        roi = (pnl / total_stake * 100.0) if total_stake > 0 else 0.0

        if os.path.exists(performance_file):
            perf_df = pd.read_csv(performance_file)
            cume_pnl = float(perf_df['cume_pnl'].iloc[-1] + pnl) if len(perf_df) > 0 else pnl
        else:
            perf_df = pd.DataFrame()
            cume_pnl = pnl

        drawdown = 0.0
        if not perf_df.empty and 'cume_pnl' in perf_df.columns:
            peak = perf_df['cume_pnl'].cummax().iloc[-1]
            drawdown = max(0.0, peak - cume_pnl)

        row = {
            'date':         batch_date,
            'bankroll':     bankroll,
            'total_stake':  total_stake,
            'pnl':          pnl,
            'cume_pnl':     cume_pnl,
            'avg_ev':       avg_ev,
            'avg_prob':     avg_prob,
            'hit_rate':     hit_rate,
            'roi_pct':      roi,
            'drawdown':     drawdown,
            'mean_odds':    float(df['odds'].mean())   if 'odds' in df.columns else 0.0,
            'median_odds':  float(df['odds'].median()) if 'odds' in df.columns else 0.0,
            'total_bets':   int((df['rec_stake'] > 0).sum()) if 'rec_stake' in df.columns else 0,
            'total_pnl':    pnl,
        }

        perf_df = pd.concat([perf_df, pd.DataFrame([row])], ignore_index=True)
        perf_df.to_csv(performance_file, index=False)

        return row

    def _calibrate_prob(self, prob_raw: np.ndarray, prob_shrinkage: float, prob_cap: float) -> np.ndarray:
        """
        Three-stage probability calibration — applied BEFORE EV/QS are computed
        so that downstream ranking reflects corrected confidence, not inflated raw probs.

        Stage 1 — global shrinkage toward 0.5 (pass-through when prob_shrinkage=1.0).
        Stage 2 — hard cap at prob_cap (default 0.68).
        Stage 3 — multiplicative shrinkage in the 60-68% overconfidence band.
                   Replaces the old additive penalty (-0.12), which was fragile:
                   a multiplicative pull toward 0.5 scales as the model drifts and
                   doesn't risk pushing probs below the floor with a fixed constant.
        """
        prob = 0.5 + (prob_raw - 0.5) * float(prob_shrinkage)
        prob = np.clip(prob, 1.0 - float(prob_cap), float(prob_cap))

        # Multiplicative shrinkage in the overconfidence band
        lo, hi = PROB_SHRINK_RANGE
        band_mask = (prob >= lo) & (prob <= hi)
        prob[band_mask] = 0.5 + (prob[band_mask] - 0.5) * PROB_SHRINK_FACTOR

        # Safety floor after shrinkage
        prob = np.maximum(prob, 0.50)
        return prob

    def process_bets(
        self,
        df: pd.DataFrame,
        bankroll: float,
        ev_threshold: float = MIN_EV_THRESHOLD,    # kept for backward compat; MIN_EV_THRESHOLD used internally
        fractional_kelly: float = KELLY_FRACTION,
        max_stake_pct: float = MAX_STAKE_PCT,
        default_sigma: float = 3.0,
        sigma_multiplier: float = 1.0,             # reserved; not removed to preserve call signature
        z_score_threshold: float = 0.0,            # reserved (non-sniper path only)
        require_abs_z: bool = False,               # reserved (non-sniper path only)
        max_bets: int = SNIPER_MAX_BETS,
        return_all: bool = False,
        quantile_model: bool = True,
        min_stake: float = 6.0,
        prob_shrinkage: float = 1.0,
        prob_cap: float = PROB_CAP,
        log_performance: bool = False,
        performance_file: str = 'data/performance.csv',
        results_export_file: str = None,
        auto_tune_from_tracker: bool = False,
        tracker_file: str = 'data/bet_tracker.json',
        min_settled_for_tuning: int = AUTO_TUNE_MIN_SETTLED,
    ) -> pd.DataFrame:
        """
        Risk-Pricing: Capital Allocation for AI-Approved Bets.

        V26.0 — Calibration + Ranking Fix (Mar 2026):

        KEY CHANGE: probability calibration now runs before EV and QS are
        computed.  In V25.x, prob_cap and the heat penalty were applied after
        quality_score was calculated, so QS ranking still used inflated raw
        probs — defeating the calibration entirely.  Fixed order:
            raw_prob → calibrate → EV → QS → gates → staking → selection.

        Other changes vs V25.2:
        - Hard additive heat penalty replaced with multiplicative shrinkage
          (PROB_SHRINK_FACTOR=0.85 in 60-68% band).  More stable as model drifts.
        - MAX_EV_THRESHOLD=0.20 added: live data (1,304 bets) shows ROI of
          -12% at EV 15-20% and -18% at EV 25%+.  The model overclaims edge
          in this range and Sniper was selecting these bets preferentially.
        - MAX_QS_THRESHOLD=0.020 added: mirrors EV ceiling — QS above 0.02
          corresponds to the overconfident tail (-8% ROI live).
        - Calibration logic extracted to _calibrate_prob() for testability.
        - scipy.stats.norm renamed scipy_norm to avoid shadowing by
          select_safety_parlay's local norm() helper.
        - total_bets and total_pnl added to save_performance_metrics output
          (were missing from performance.csv, causing NaN in downstream charts).
        """
        df = df.copy()

        if results_export_file:
            self.configure_from_results(results_export_file)
        elif not self.market_reliability and os.path.exists(DEFAULT_RESULTS_EXPORT):
            self.configure_from_results(DEFAULT_RESULTS_EXPORT)

        gate_state = {
            'min_ev': self.decision_config.min_ev,
            'min_qs': self.decision_config.min_qs,
            'max_bets': max_bets,
            'adapted': False,
        }
        if auto_tune_from_tracker:
            gate_state = self._compute_adaptive_gates(
                tracker_file=tracker_file,
                min_settled=min_settled_for_tuning,
                current_max_bets=max_bets,
            )

        min_ev_threshold = float(gate_state['min_ev'])
        min_qs_threshold = float(gate_state['min_qs'])
        effective_max_bets = int(gate_state['max_bets'])

        # ------------------------------------------------------------------
        # 1. Sigma (volatility)
        # ------------------------------------------------------------------
        if quantile_model:
            sigma = df['predicted_std'].copy().fillna(default_sigma).astype(float)
        else:
            safe_logvar = df['predicted_std'].copy()
            default_logvar = np.log1p(default_sigma ** 2)
            safe_logvar = safe_logvar.fillna(default_logvar)
            with np.errstate(over='ignore', invalid='ignore'):
                sigma = np.sqrt(np.expm1(safe_logvar.astype(float)))

        if 'feat_std' in df.columns:
            floor = 0.7 * df['feat_std'].fillna(1.0)
            sigma = np.maximum(sigma, floor)

        sigma = np.maximum(sigma, 1.0)  # hard floor prevents QS inflation on tight lines

        # ------------------------------------------------------------------
        # 2. Raw probability
        # ------------------------------------------------------------------
        if quantile_model and 'prob' in df.columns:
            prob_raw = df['prob'].copy().fillna(0.5).astype(float)
        else:
            z_metric = (df['line'] - df['predicted_mean']) / sigma
            prob_raw = np.where(
                df['side'] == 'UNDER',
                scipy_norm.cdf(z_metric),
                1 - scipy_norm.cdf(z_metric),
            )
        prob_raw = np.asarray(prob_raw, dtype=float)

        # ------------------------------------------------------------------
        # 3. Calibrate probability  ← MUST happen before EV and QS
        # ------------------------------------------------------------------
        prob = self._calibrate_prob(prob_raw, prob_shrinkage, prob_cap)

        # ------------------------------------------------------------------
        # 4. EV and Quality Score — computed from calibrated prob
        # ------------------------------------------------------------------
        true_ev = prob * (df['odds'].values - 1) - (1 - prob)
        quality_score = true_ev / sigma

        df['quality_score'] = quality_score
        df['predicted_ev']  = true_ev
        df['predicted_prob'] = prob
        df['sigma']          = sigma

        # ------------------------------------------------------------------
        # 5. Staking
        # ------------------------------------------------------------------
        if SNIPER_MODE:
            rec_stake = np.array([
                self.calculate_kelly_stake(p, o, bankroll, fractional_kelly, max_stake_pct)
                for p, o in zip(prob, df['odds'])
            ])

            # Market-aware stake scaling from exported model quality results.
            if 'market' in df.columns and self.market_reliability:
                market_mult = np.array([
                    self._get_market_multiplier(market)
                    for market in df['market']
                ])
                rec_stake = rec_stake * market_mult

            # Gate 1: quality score band (risk-adjusted edge — both floor and ceiling)
            rec_stake = np.where(
                (quality_score >= min_qs_threshold) & (quality_score <= self.decision_config.max_qs),
                rec_stake, 0.0,
            )

            # Gate 2: EV band (absolute edge — both floor and ceiling)
            rec_stake = np.where(
                (true_ev >= min_ev_threshold) & (true_ev <= self.decision_config.max_ev),
                rec_stake, 0.0,
            )

            # Gate 3: minimum probability
            rec_stake = np.where(prob >= self.decision_config.min_prob, rec_stake, 0.0)

            # Gate 4: blocked market/side combos
            if 'market' in df.columns and 'side' in df.columns:
                combo_mask = pd.Series(False, index=df.index)
                for market, side in self.decision_config.blocked_combos:
                    combo_mask |= (
                        df['market'].astype(str).str.upper().eq(market)
                        & df['side'].astype(str).str.upper().eq(side)
                    )
                rec_stake = np.where(combo_mask.values, 0.0, rec_stake)

            # Gate 5: minimum stake (dust filter)
            min_floor = max(min_stake, bankroll * self.decision_config.min_stake_pct)
            rec_stake = np.where(rec_stake >= min_floor, rec_stake, 0.0)

            funded = int((rec_stake > 0).sum())
            if funded > 0:
                stakes = rec_stake[rec_stake > 0]
                logger.info(
                    f"Kelly Staking: {funded} bets funded | "
                    f"avg {stakes.mean():.0f} ({stakes.mean() / bankroll * 100:.1f}%) | "
                    f"range [{stakes.min():.0f}–{stakes.max():.0f}]"
                )

        else:
            # Non-sniper fallback: raw Kelly scaled by z-score conviction
            raw_kelly = np.array([
                self.calculate_kelly_stake(p, o, bankroll, fractional_kelly, max_stake_pct)
                for p, o in zip(prob, df['odds'])
            ])
            z_score = np.abs(df['predicted_mean'] - df['line']) / sigma
            stake_multiplier = np.where(
                z_score >= 0.6, 1.0,
                np.where(z_score >= 0.35, 0.5, 0.0),
            )
            rec_stake = raw_kelly * stake_multiplier

        df['rec_stake'] = rec_stake
        df['is_bet'] = df['rec_stake'] > 0

        # ------------------------------------------------------------------
        # 6. Selection and exposure control  (Sniper only)
        # ------------------------------------------------------------------
        if SNIPER_MODE:
            df = df.sort_values(by='quality_score', ascending=False).copy()

            # One bet per player — keep highest QS
            if 'player' in df.columns:
                df = df.drop_duplicates(subset=['player'], keep='first')

            # Opponent exposure cap
            if 'opponent' in df.columns:
                opp_counts: dict = {}
                for idx in df[df['rec_stake'] > 0].index:
                    opp = df.loc[idx, 'opponent']
                    opp_counts[opp] = opp_counts.get(opp, 0) + 1
                    if opp_counts[opp] > SNIPER_MAX_PER_OPPONENT:
                        df.loc[idx, 'rec_stake'] = 0.0
                        df.loc[idx, 'is_bet'] = False
                        logger.info(f"Exposure cap: dropped extra bet vs {opp}")

            # Top-N cut
            funded_count = int((df['rec_stake'] > 0).sum())
            if funded_count > effective_max_bets:
                logger.info(f"RISK ENGINE: {funded_count} approved, capping at {effective_max_bets}")
                funded_idx = df[df['rec_stake'] > 0].index
                df.loc[funded_idx[effective_max_bets:], 'rec_stake'] = 0.0
                df.loc[funded_idx[effective_max_bets:], 'is_bet'] = False

        # ------------------------------------------------------------------
        # 7. (Optional) persist batch metrics
        # ------------------------------------------------------------------
        if log_performance:
            self.save_performance_metrics(df[df['rec_stake'] > 0], bankroll, performance_file=performance_file)

        if not return_all:
            return df[df['rec_stake'] > 0].copy()

        return df

    def select_safety_parlay(self, df: pd.DataFrame, legs: int = 3) -> pd.DataFrame:
        """Select a safer multi-leg parlay from unique matchups based on safety score."""
        if df.empty:
            return df

        work = df.copy()
        work['prob']          = pd.to_numeric(work['prob'],          errors='coerce').fillna(0.5)
        work['quality_score'] = pd.to_numeric(work['quality_score'], errors='coerce').fillna(0)
        work['odds']          = pd.to_numeric(work['odds'],          errors='coerce').fillna(1)

        pmin, pmax = work['prob'].min(),          work['prob'].max()
        qmin, qmax = work['quality_score'].min(), work['quality_score'].max()
        inv        = (1.0 / work['odds']).replace([np.inf, -np.inf], np.nan).fillna(0)
        imin, imax = inv.min(), inv.max()

        # Local helper — named _norm to avoid any future collision with module-level imports
        def _norm(x, a, b):
            return 0.5 if b == a else (x - a) / (b - a)

        work['safety'] = work.apply(
            lambda r: round(
                0.65 * _norm(r['prob'],          pmin, pmax)
                + 0.2 * _norm(r['quality_score'], qmin, qmax)
                + 0.15 * _norm(1.0 / r['odds'] if r['odds'] > 0 else 0, imin, imax)
                + (0.03 if str(r.get('side', '')).upper() == 'UNDER' else 0.0),
                6,
            ),
            axis=1,
        )

        per_match = work.sort_values('safety', ascending=False).groupby('matchup', as_index=False).first()
        return per_match.sort_values('safety', ascending=False).head(legs)
