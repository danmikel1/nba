"""
scripts/backtest_monte_carlo_10k.py — Head-to-Head A/B Backtest of Tool #1 (10k Monte Carlo)

Benchmarks the new 10,000-Trial Regime-Switching & Cholesky SGP Monte Carlo Engine against
the legacy Analytical Split-Normal CDF across 3 real sportsbook datasets:
  1. Point-in-Time Full-Slate DraftKings Props (4,540 props, 2026-02-19 to 2026-03-02)
  2. Verified Real Tracked Sportsbook Props (1,294 decided props, 2026-02-19 to 2026-03-22)
  3. Opening Week Settled Props w/ Box-Score Minutes Audit (4,393 props, 2026-10-21 to 2026-10-28)
  4. Same-Game Parlay (SGP) Cholesky Correlation Backtest (Positively Correlated vs Cannibalized SGPs)
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nba_prediction import CONFIG, ModelEngine
from simulation_engine import SimulationEngine
from trading_engine import TradingEngine

logging.basicConfig(level=logging.ERROR)

DATA_DIR = PROJECT_ROOT / "data"
ODDSBLAZE_DIR = DATA_DIR / "oddsblaze_cache"
SETTLED_PARQUET = DATA_DIR / "harvested_odds" / "settled_player_props.parquet"
TRAIN_PARQUET = DATA_DIR / "ml_training_data.parquet"
HIST_TRACKER_JSON = DATA_DIR / "bet_tracker_fixed_opponent.json"

MARKET_MAP = {
    "Player Points": "PTS",
    "Player Rebounds": "REB",
    "Player Assists": "AST",
    "Player Points + Rebounds + Assists": "PRA",
    "Player Points + Rebounds": "PR",
    "Player Points + Assists": "PA",
    "Player Rebounds + Assists": "RA",
}


def norm_p(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(".", "")
        .replace("'", "")
        .replace("-", " ")
    )


def score_ab_dataframe(
    merged: pd.DataFrame,
    model_engine: ModelEngine,
    trading_engine: TradingEngine,
    sim_engine: SimulationEngine,
    line_col: str = "dk_line",
) -> pd.DataFrame:
    """Scores every row with BOTH (A) Pure Analytical CDF and (B) 10k Regime-Switching Monte Carlo."""
    results = []
    for mkt, mdf in merged.groupby("market"):
        q_dict, exp_feats = model_engine._get_model_for_market(mkt)
        if not q_dict or not q_dict.get("q50"):
            continue
        for f_col in exp_feats:
            if f_col not in mdf.columns:
                mdf[f_col] = 0.0

        X_arr = mdf[exp_feats].values
        q50 = q_dict["q50"].predict(X_arr)
        q05 = q_dict["q05"].predict(X_arr)
        q95 = q_dict["q95"].predict(X_arr)

        sigma_lo = np.maximum((q50 - q05) / 1.645, 0.5)
        sigma_hi = np.maximum((q95 - q50) / 1.645, 0.5)
        pred_std = np.maximum((q95 - q05) / 3.29, 1.0)
        if "feat_std" in mdf.columns:
            pred_std = np.maximum(pred_std, 0.7 * mdf["feat_std"].fillna(1.0).values)

        lines_arr = mdf[line_col].values
        diff = lines_arr - q50
        p_under_cdf = np.where(diff <= 0, norm.cdf(diff / sigma_lo), norm.cdf(diff / sigma_hi))
        p_over_cdf = 1.0 - p_under_cdf

        spreads_arr = mdf["feat_spread"].fillna(0.0).values if "feat_spread" in mdf.columns else np.zeros(len(mdf))
        foul_arr = mdf["feat_foul_rate"].fillna(2.2).values if "feat_foul_rate" in mdf.columns else np.full(len(mdf), 2.2)
        mins_arr = mdf["feat_avg_minutes"].fillna(28.0).values if "feat_avg_minutes" in mdf.columns else np.full(len(mdf), 28.0)

        mc_batch = sim_engine.simulate_slate_10k(
            q50=q50,
            q05=q05,
            q95=q95,
            lines=lines_arr,
            sides=mdf["side"].values,
            spreads=spreads_arr,
            avg_minutes=mins_arr,
            foul_rates=foul_arr,
            n_simulations=10_000,
        )

        side_upper = mdf["side"].astype(str).str.upper().values
        raw_cdf = np.where(side_upper == "OVER", p_over_cdf, p_under_cdf)
        raw_mc = mc_batch["mc_side_prob"]
        raw_blend = 0.65 * raw_cdf + 0.35 * raw_mc

        cal_cdf = trading_engine._calibrate_prob(raw_cdf, prob_shrinkage=0.75, prob_cap=0.68)
        cal_mc = trading_engine._calibrate_prob(raw_mc, prob_shrinkage=0.75, prob_cap=0.68)
        cal_blend = trading_engine._calibrate_prob(raw_blend, prob_shrinkage=0.75, prob_cap=0.68)

        mdf = mdf.copy()
        mdf["side"] = side_upper
        mdf["q50"] = q50
        mdf["q05"] = q05
        mdf["q95"] = q95
        mdf["predicted_std"] = pred_std
        mdf["edge_pts"] = np.where(side_upper == "OVER", q50 - lines_arr, lines_arr - q50)

        mdf["prob_cdf"] = cal_cdf
        mdf["prob_mc"] = cal_mc
        mdf["prob_blend"] = cal_blend

        mdf["ev_cdf"] = cal_cdf * (mdf["odds_dec"] - 1.0) - (1.0 - cal_cdf)
        mdf["ev_mc"] = cal_mc * (mdf["odds_dec"] - 1.0) - (1.0 - cal_mc)
        mdf["ev_blend"] = cal_blend * (mdf["odds_dec"] - 1.0) - (1.0 - cal_blend)

        mdf["blowout_risk_pct"] = mc_batch["blowout_risk_pct"]
        mdf["foul_risk_pct"] = mc_batch["foul_risk_pct"]
        mdf["tail_cvar_10"] = mc_batch["tail_cvar_10"]

        mdf["won"] = np.where(
            side_upper == "OVER",
            mdf["actual_value"] > lines_arr,
            mdf["actual_value"] < lines_arr,
        )
        mdf["push"] = mdf["actual_value"] == lines_arr
        mdf["pnl"] = np.where(mdf["won"], mdf["odds_dec"] - 1.0, np.where(mdf["push"], 0.0, -1.0))
        results.append(mdf)

    if not results:
        return pd.DataFrame()
    res = pd.concat(results, ignore_index=True)
    res = res[~res["push"]]
    best = (
        res[res["edge_pts"] > 0]
        .sort_values("ev_blend", ascending=False)
        .drop_duplicates(["date", "name_norm", "market"])
        .copy()
    )
    return best


def print_row(label: str, sub: pd.DataFrame):
    if len(sub) == 0:
        return
    w = int(sub["won"].sum())
    tot = len(sub)
    l = tot - w
    wr = w / tot
    pnl = float(sub["pnl"].sum())
    roi = (pnl / tot) * 100.0
    print(f"{label:<46} {tot:<7,d} {w:<6,d} {l:<6,d} {wr:<10.2%} {pnl:<+10.2f}u {roi:<+9.2f}%")


def main():
    model_engine = ModelEngine(CONFIG)
    trading_engine = TradingEngine()
    sim_engine = SimulationEngine()

    df_train = pd.read_parquet(TRAIN_PARQUET)
    df_train["name_norm"] = df_train["player"].apply(norm_p)

    # =========================================================================
    # 1. POINT-IN-TIME FULL-SLATE DRAFTKINGS BACKTEST (4,540 Props)
    # =========================================================================
    real_rows = []
    for f in sorted(ODDSBLAZE_DIR.glob("*_draftkings.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        if data.get("_empty"):
            continue
        dt = f.stem.split("_")[0]
        for ev in data.get("events", []):
            ev_id = ev.get("id", f"{dt}_game")
            for o in ev.get("odds", []):
                if "player" not in o or not o.get("main", True):
                    continue
                m = MARKET_MAP.get(o.get("market", ""))
                if not m:
                    continue
                sel = o.get("selection", {})
                side = str(sel.get("side", "")).upper()
                if side not in ("OVER", "UNDER"):
                    continue
                line = sel.get("line")
                if line is None:
                    continue
                price_str = str(o.get("price", "-110"))
                try:
                    am = int(price_str.replace("+", ""))
                except ValueError:
                    am = -110
                dec = 1.0 + (am / 100.0) if am > 0 else 1.0 + (100.0 / abs(am))
                p_team = (o.get("player") or {}).get("team", {}).get("abbreviation", "")
                real_rows.append({
                    "date": dt,
                    "event_id": ev_id,
                    "name_norm": norm_p(o["player"]["name"]),
                    "player_team": p_team,
                    "market": m,
                    "side": side,
                    "dk_line": float(line),
                    "odds_am": am,
                    "odds_dec": dec,
                })

    if real_rows:
        df_dk = pd.DataFrame(real_rows).groupby(["date", "name_norm", "market", "side"], as_index=False).last()
        merged_mid = df_train.merge(df_dk, on=["date", "name_norm", "market"], how="inner")
        best_mid = score_ab_dataframe(merged_mid, model_engine, trading_engine, sim_engine, line_col="dk_line")

        brier_cdf = float(np.mean((best_mid["prob_cdf"] - best_mid["won"].astype(float)) ** 2))
        brier_mc = float(np.mean((best_mid["prob_mc"] - best_mid["won"].astype(float)) ** 2))
        brier_blend = float(np.mean((best_mid["prob_blend"] - best_mid["won"].astype(float)) ** 2))

        print("\n" + "=" * 96)
        print("1. POINT-IN-TIME FULL-SLATE DRAFTKINGS A/B BACKTEST (2026-02-19 to 2026-03-02 | 4,540 Props)")
        print("=" * 96)
        print(
            f"Brier Calibration Error (Lower is Better): "
            f"Analytical CDF = {brier_cdf:.5f} | 10k MC Pure = {brier_mc:.5f} | 10k MC Blended = {brier_blend:.5f}"
        )
        print("-" * 96)
        print(f"{'Strategy / Engine Variant':<46} {'Bets':<7} {'Wins':<6} {'Loss':<6} {'Win Rate':<10} {'Net Units':<11} {'ROI':<9}")
        print("-" * 96)

        print_row("Unfiltered Board (All 6,223 Props)", best_mid)
        print_row("A1) Legacy CDF (4% <= EV <= 14%, All Mkts)", best_mid[(best_mid["ev_cdf"] >= 0.04) & (best_mid["ev_cdf"] <= 0.14)])
        print_row("B1) 10k Monte Carlo (4% <= EV <= 14%, All Mkts)", best_mid[(best_mid["ev_mc"] >= 0.04) & (best_mid["ev_mc"] <= 0.14)])

        def apply_full_sniper_gates(df_in: pd.DataFrame, prob_col: str, ev_col: str, block_blowout_overs: bool = False) -> pd.DataFrame:
            p = df_in[df_in["market"].isin(["PRA", "PR", "PA", "RA", "AST"])].copy()
            p = p[(p[ev_col] >= 0.04) & (p[ev_col] <= 0.14)]
            for blk_mkt, blk_side in trading_engine.decision_config.blocked_combos:
                p = p[~((p["market"] == blk_mkt) & (p["side"] == blk_side))]
            min_line_mask = np.where(
                p["market"].isin(["PRA", "PR", "PA", "PTS"]),
                p["dk_line"] >= 10.5,
                np.where(p["market"] == "RA", p["dk_line"] >= 5.5, p["dk_line"] >= 3.5),
            )
            min_edge_mask = np.where(
                p["market"].isin(["PRA", "PR", "PA", "PTS"]),
                p["edge_pts"] >= 1.5,
                p["edge_pts"] >= 0.8,
            )
            odds_mask = (p["odds_am"] >= -135) & (p["odds_am"] <= 135)
            p = p[min_line_mask & min_edge_mask & odds_mask]
            p = p[p[prob_col] >= 0.54]
            if block_blowout_overs:
                p = p[~((p["side"] == "OVER") & (p["blowout_risk_pct"] >= 18.0))]
            norm_sigma = p["predicted_std"] / np.sqrt(np.maximum(p["dk_line"].values, 4.0))
            p["qs"] = p[ev_col] / np.maximum(norm_sigma, 0.25)
            p = p.sort_values(["qs", ev_col], ascending=[False, False])
            p = p.drop_duplicates(subset=["date", "name_norm"], keep="first")
            if "event_id" in p.columns:
                p = p.groupby(["date", "event_id"], group_keys=False).head(3)
            return p

        print("-" * 96)
        print_row("A2) Legacy CDF + Full 7 Sniper Gates", apply_full_sniper_gates(best_mid, "prob_cdf", "ev_cdf", False))
        print_row("B2) 10k MC Pure + Full 7 Sniper Gates", apply_full_sniper_gates(best_mid, "prob_mc", "ev_mc", False))
        print_row("C2) 10k MC Blended + Full 7 Sniper Gates", apply_full_sniper_gates(best_mid, "prob_blend", "ev_blend", False))
        print_row("D2) 10k MC + Sniper Gates + Blowout Filter", apply_full_sniper_gates(best_mid, "prob_mc", "ev_mc", True))
        print_row(
            "E2) 10k MC AST + RA Combos (EV >= 2%)",
            best_mid[(best_mid["market"].isin(["AST", "RA"])) & (best_mid["ev_mc"] >= 0.02)],
        )

        # =====================================================================
        # 2. SAME-GAME PARLAY (SGP) CORRELATION BACKTEST ON REAL DK SLATES
        # =====================================================================
        print("\n" + "=" * 96)
        print("2. 10K MONTE CARLO SGP CORRELATION BACKTEST (2-Leg Same-Game Parlays from EV>=2% Edges)")
        print("=" * 96)
        edges_mid = best_mid[best_mid["ev_blend"] >= 0.02].copy()
        pos_sgps, neg_sgps = [], []
        for (dt, eid), grp in edges_mid.groupby(["date", "event_id"]):
            if len(grp) < 2:
                continue
            top_rows = grp.sort_values("ev_blend", ascending=False).head(4)
            rows_list = [r for _, r in top_rows.iterrows()]
            for i in range(len(rows_list)):
                for j in range(i + 1, len(rows_list)):
                    r1, r2 = rows_list[i], rows_list[j]
                    if r1["name_norm"] == r2["name_norm"] and r1["market"] == r2["market"]:
                        continue
                    legs = [
                        {
                            "player": r1["player"],
                            "team": str(r1.get("player_team") or ""),
                            "opponent": str(r1.get("opponent_team") or ""),
                            "market": r1["market"],
                            "line": float(r1["dk_line"]),
                            "side": r1["side"],
                            "mean": float(r1["q50"]),
                            "std": float(r1["predicted_std"]),
                            "odds": float(r1["odds_dec"]),
                        },
                        {
                            "player": r2["player"],
                            "team": str(r2.get("player_team") or ""),
                            "opponent": str(r2.get("opponent_team") or ""),
                            "market": r2["market"],
                            "line": float(r2["dk_line"]),
                            "side": r2["side"],
                            "mean": float(r2["q50"]),
                            "std": float(r2["predicted_std"]),
                            "odds": float(r2["odds_dec"]),
                        },
                    ]
                    sim_res = sim_engine.simulate_correlated_parlay(legs, n_simulations=10_000, seed=42)
                    both_hit = bool(r1["won"] and r2["won"])
                    comb_odds = float(r1["odds_dec"] * r2["odds_dec"])
                    pnl = (comb_odds - 1.0) if both_hit else -1.0
                    rec = {
                        "won": both_hit,
                        "pnl": pnl,
                        "corr_edge": sim_res["correlation_edge_pct"],
                        "parlay_ev": sim_res["parlay_ev_pct"],
                    }
                    if sim_res["correlation_edge_pct"] > 0.5:
                        pos_sgps.append(rec)
                    elif sim_res["correlation_edge_pct"] < -0.5:
                        neg_sgps.append(rec)

        for lbl, lst in [
            ("Positively Correlated SGPs (Corr Edge > +0.5%)", pos_sgps),
            ("Negatively Correlated / Cannibalized SGPs (<-0.5%)", neg_sgps),
        ]:
            if lst:
                sdf = pd.DataFrame(lst)
                print_row(lbl, sdf)

    # =========================================================================
    # 3. OPENING WEEK ACTUAL MINUTES & BLOWOUT BENCHING VERIFICATION
    # =========================================================================
    if SETTLED_PARQUET.exists():
        df_oct = pd.read_parquet(SETTLED_PARQUET)
        df_oct["res_up"] = df_oct["result"].astype(str).str.upper()
        df_oct = df_oct[df_oct["res_up"].isin(["WIN", "LOSS"]) & df_oct["actual_value"].notna()].copy()
        df_oct["name_norm"] = df_oct["player"].apply(norm_p)
        feat_cols = [c for c in df_train.columns if c.startswith("feat_")]
        latest_f = df_train.sort_values("date").groupby(["name_norm", "market"], as_index=False)[["name_norm", "market"] + feat_cols].last()
        m_oct = df_oct.merge(latest_f, on=["name_norm", "market"], how="inner")
        m_oct["dk_line"] = pd.to_numeric(m_oct["closing_line"], errors="coerce")
        m_oct["odds_dec"] = pd.to_numeric(m_oct["odds_decimal"], errors="coerce").fillna(1.909)
        scored_oct = score_ab_dataframe(m_oct, model_engine, trading_engine, sim_engine, line_col="dk_line")

        print("\n" + "=" * 96)
        print("3. REAL BOX-SCORE MINUTES & BLOWOUT REGIME VERIFICATION (2026-10-21 to 2026-10-28)")
        print("=" * 96)
        if "minutes_played" in scored_oct.columns and "feat_avg_minutes" in scored_oct.columns:
            starters = scored_oct[scored_oct["feat_avg_minutes"] >= 28.0].copy()
            starters["min_delta"] = starters["minutes_played"] - starters["feat_avg_minutes"]
            low_bo = starters[starters["blowout_risk_pct"] < 10.0]
            high_bo = starters[starters["blowout_risk_pct"] >= 18.0]
            print(
                f"  Low Blowout Risk (<10%) Starters  : Avg Min Diff = {low_bo['min_delta'].mean():+.2f} min | "
                f"4th-Qtr Bench Rate (>5m cut) = {(low_bo['min_delta'] <= -5.0).mean():.1%}"
            )
            print(
                f"  High Blowout Risk (>=18%) Starters: Avg Min Diff = {high_bo['min_delta'].mean():+.2f} min | "
                f"4th-Qtr Bench Rate (>5m cut) = {(high_bo['min_delta'] <= -5.0).mean():.1%}"
            )
            high_bo_over = high_bo[high_bo["side"] == "OVER"]
            high_bo_under = high_bo[high_bo["side"] == "UNDER"]
            print(
                f"  High Blowout Risk (>=18%) OVER Hit Rate : {high_bo_over['won'].mean():.2%} ({len(high_bo_over)} props)"
            )
            print(
                f"  High Blowout Risk (>=18%) UNDER Hit Rate: {high_bo_under['won'].mean():.2%} ({len(high_bo_under)} props)"
            )


if __name__ == "__main__":
    main()
