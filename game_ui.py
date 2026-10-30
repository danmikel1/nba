"""
game_ui.py — Streamlit UI Component for Game-Level NBA Predictions & Betting

Integrates game-level spread and total models directly into the Streamlit dashboard:
  - Live & upcoming game odds ingestion (The Odds API + ESPN fallback)
  - XGBoost Quantile Regression predictions (Spread & Over/Under)
  - Sweet-Spot Edge Filtering (1.0 to 2.5 pts) to avoid >3.0 pt unpriced star-injury traps
  - Mathematical Edge and Fractional Kelly stake sizing
  - 1-click "🚀 Track Top Game Picks" batch button + per-card "➕ Track" buttons
  - Custom / hypothetical game predictor
  - Settled game outcomes and historical model performance tracking
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import math
import time

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

logger = logging.getLogger(__name__)

from game_prediction import predict_game, GamePrediction, STD_TO_CSV
from scripts.harvest_closing_odds import harvest_daily_odds, settle_games_for_date, HARVEST_DIR, SETTLED_CSV
from persistence_utils import safe_json_load

# NBA 30 Teams List (Abbreviation -> Full Name)
NBA_TEAMS = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "LA Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}

# Empirical fallback standard deviations (used only when quantile bounds q10/q90 are unavailable)
STD_SPREAD = 11.5
STD_TOTAL = 14.2
STANDARD_DECIMAL_ODDS = 1.909  # -110 American odds
Z_80_QUANTILE = 1.2815515655446004  # stats.norm.ppf(0.90) for [q10, q90] split-normal


def compute_win_prob_and_kelly(
    edge_pts: float,
    std_dev: float,
    bankroll: float,
    kelly_fraction: float = 0.25,
    decimal_odds: float = STANDARD_DECIMAL_ODDS,
    max_stake_pct: float = 0.05,
    q10: Optional[float] = None,
    q50: Optional[float] = None,
    q90: Optional[float] = None,
    line: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Computes pure data-driven win probability from trained (q10, q50, q90) split-normal
    quantiles when available, and fractional Kelly stake.
    Returns: (win_prob, ev_pct, recommended_stake)
    """
    if q10 is not None and q50 is not None and q90 is not None and line is not None and q90 > q10:
        sigma_left = max((q50 - q10) / Z_80_QUANTILE, 1.0)
        sigma_right = max((q90 - q50) / Z_80_QUANTILE, 1.0)
        if line <= q50:
            # Picking OVER / Home Cover: line is in the left tail below q50
            z = (q50 - line) / sigma_left
        else:
            # Picking UNDER / Away Cover: line is in the right tail above q50
            z = (line - q50) / sigma_right
        win_prob = float(np.clip(stats.norm.cdf(z), 0.5001, 0.99))
    else:
        z = float(edge_pts) / max(std_dev, 1.0)
        win_prob = float(stats.norm.cdf(z))

    # Expected value: (p * decimal_odds) - 1
    ev_pct = (win_prob * decimal_odds) - 1.0

    # Kelly criterion: f* = (p * b - 1) / (b - 1)
    b = decimal_odds - 1.0
    kelly_f = (win_prob * decimal_odds - 1.0) / b if b > 0 else 0.0

    if kelly_f > 0 and ev_pct > 0:
        raw_stake = bankroll * kelly_f * kelly_fraction
        # Cap at max stake percentage of bankroll
        rec_stake = min(raw_stake, bankroll * max_stake_pct)
        rec_stake = round(max(rec_stake, 0.0), 2)
    else:
        rec_stake = 0.0

    return win_prob, ev_pct, rec_stake


def load_harvested_games(target_date: str, force_refresh: bool = False) -> List[Dict[str, Any]]:
    """Load cached odds file or trigger fresh harvest if missing or lines are empty."""
    file_path = HARVEST_DIR / f"odds_{target_date}.json"
    if not force_refresh and file_path.exists():
        data = safe_json_load(file_path, default={})
        games = data if isinstance(data, list) else data.get("games", [])
        if any(g.get("spread_home") is not None and g.get("total") is not None for g in games):
            return games

    # If missing, empty, or missing lines, perform on-demand harvest
    try:
        return harvest_daily_odds(target_date)
    except Exception as e:
        logger.warning(f"Could not load live odds for {target_date}: {e}")
        return []


def score_slate_games(
    valid_games: List[Dict[str, Any]],
    bankroll: float = 1000.0,
    kelly_fraction: float = 0.25,
    min_edge: float = 1.0,
    max_edge: float = 20.0,
    road_ats_priority: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Scores all games on the slate and extracts actionable "Best Game Picks"
    within the [min_edge, max_edge] sweet spot.

    Returns:
        (evaluated_games, best_bet_payloads)
    """
    evaluated_games: List[Dict[str, Any]] = []
    best_bet_payloads: List[Dict[str, Any]] = []

    for g in valid_games:
        home = g.get("home_team", "")
        away = g.get("away_team", "")
        # Sportsbook spread_home is negative when home is favored (e.g. -4.5).
        # game_prediction.py expects positive vegas_spread when home is favored (+4.5).
        book_spread_home = float(g.get("spread_home", 0.0))
        model_vegas_spread = -book_spread_home
        total = float(g.get("total", 220.0))
        ml_home = g.get("ml_home")
        ml_away = g.get("ml_away")

        pred = predict_game(
            home=home,
            away=away,
            vegas_spread=model_vegas_spread,
            vegas_total=total,
            moneyline_home=ml_home,
            moneyline_away=ml_away,
        )
        if not pred:
            continue

        # SPREAD evaluation (pure quantile split-normal from q10, q50, q90)
        spread_edge = pred.spread_edge  # pred_margin - model_vegas_spread
        abs_spread_edge = abs(spread_edge)
        spread_pick_team = home if spread_edge > 0 else away
        # Sportsbook line for the picked team:
        spread_pick_line = book_spread_home if spread_edge > 0 else -book_spread_home
        spread_side_str = f"{spread_pick_team} {spread_pick_line:+.1f}"
        spread_q_std = max((pred.pred_margin_q90 - pred.pred_margin_q10) / (2.0 * Z_80_QUANTILE), 1.0)
        spread_prob, spread_ev, spread_stake = compute_win_prob_and_kelly(
            edge_pts=abs_spread_edge,
            std_dev=spread_q_std,
            bankroll=bankroll,
            kelly_fraction=kelly_fraction,
            q10=pred.pred_margin_q10,
            q50=pred.pred_margin,
            q90=pred.pred_margin_q90,
            line=pred.vegas_spread,
        )

        is_spread_sweet = (min_edge <= abs_spread_edge <= max_edge)
        if road_ats_priority and spread_pick_team != away:
            is_spread_sweet = False

        if is_spread_sweet:
            best_bet_payloads.append({
                "date": g.get("date", datetime.now().strftime("%Y-%m-%d")),
                "player": f"{away} @ {home}",
                "market": "SPREAD",
                "bet_type": "spread",
                "side": spread_side_str,
                "line": float(spread_pick_line),
                "opening_line": float(spread_pick_line),
                "closing_line": float(spread_pick_line),
                "odds": -110,
                "odds_decimal": STANDARD_DECIMAL_ODDS,
                "predicted_mean": float(pred.pred_margin),
                "predicted_std": float(round(spread_q_std, 2)),
                "edge_pts": float(round(abs_spread_edge, 1)),
                "ev": float(spread_ev),
                "win_prob": float(spread_prob),
                "stake": float(spread_stake if spread_stake > 0 else round(bankroll * 0.01, 2)),
                "opponent": away if spread_pick_team == home else home,
                "book": g.get("bookmaker", "DraftKings"),
                "tag": "game_sniper" if spread_pick_team == away else "game_model",
                "model_version": "game_xgb_v2_optuna_lstm",
                "result": "Pending",
                "tracked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

        # TOTAL evaluation (pure quantile split-normal from q10, q50, q90)
        total_edge = pred.total_edge
        abs_total_edge = abs(total_edge)
        total_pick = "OVER" if total_edge > 0 else "UNDER"
        total_side_str = f"{total_pick} {pred.vegas_total:.1f}"
        total_q_std = max((pred.pred_total_q90 - pred.pred_total_q10) / (2.0 * Z_80_QUANTILE), 1.0)
        total_prob, total_ev, total_stake = compute_win_prob_and_kelly(
            edge_pts=abs_total_edge,
            std_dev=total_q_std,
            bankroll=bankroll,
            kelly_fraction=kelly_fraction,
            q10=pred.pred_total_q10,
            q50=pred.pred_total,
            q90=pred.pred_total_q90,
            line=pred.vegas_total,
        )

        is_total_sweet = (min_edge <= abs_total_edge <= max_edge)
        if is_total_sweet:
            best_bet_payloads.append({
                "date": g.get("date", datetime.now().strftime("%Y-%m-%d")),
                "player": f"{away} @ {home}",
                "market": "TOTAL",
                "bet_type": "total",
                "side": total_side_str,
                "line": float(pred.vegas_total),
                "opening_line": float(pred.vegas_total),
                "closing_line": float(pred.vegas_total),
                "odds": -110,
                "odds_decimal": STANDARD_DECIMAL_ODDS,
                "predicted_mean": float(pred.pred_total),
                "predicted_std": float(round(total_q_std, 2)),
                "edge_pts": float(round(abs_total_edge, 1)),
                "ev": float(total_ev),
                "win_prob": float(total_prob),
                "stake": float(total_stake if total_stake > 0 else round(bankroll * 0.01, 2)),
                "opponent": f"{away} vs {home}",
                "book": g.get("bookmaker", "DraftKings"),
                "tag": "game_model",
                "model_version": "game_xgb_v2_optuna_lstm",
                "result": "Pending",
                "tracked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

        evaluated_games.append({
            "game": g,
            "pred": pred,
            "book_spread_home": book_spread_home,
            "is_spread_sweet": is_spread_sweet,
            "is_total_sweet": is_total_sweet,
        })

    # Sort best picks by edge_pts descending
    best_bet_payloads.sort(key=lambda x: x.get("edge_pts", 0.0), reverse=True)
    return evaluated_games, best_bet_payloads


def render_game_card(
    game: Dict[str, Any],
    pred: GamePrediction,
    tracker: Any,
    bankroll: float,
    kelly_fraction: float,
    min_edge: float,
    card_index: int,
    max_edge: float = 20.0,
):
    """Renders a styled prediction card for a single matchup with Spread and Total markets."""
    home = pred.home
    away = pred.away
    home_name = NBA_TEAMS.get(home, home)
    away_name = NBA_TEAMS.get(away, away)
    book = game.get("bookmaker", "Consensus")
    commence = game.get("commence_time", "")

    # Sportsbook home spread (negative when home is favored)
    book_spread_home = float(game.get("spread_home", -pred.vegas_spread))

    # SPREAD:
    spread_edge = pred.spread_edge
    abs_s_edge = abs(spread_edge)
    spread_pick_team = home if spread_edge > 0 else away
    spread_line = book_spread_home if spread_edge > 0 else -book_spread_home
    spread_q_std = max((pred.pred_margin_q90 - pred.pred_margin_q10) / (2.0 * Z_80_QUANTILE), 1.0)
    spread_prob, spread_ev, spread_stake = compute_win_prob_and_kelly(
        edge_pts=abs_s_edge,
        std_dev=spread_q_std,
        bankroll=bankroll,
        kelly_fraction=kelly_fraction,
        q10=pred.pred_margin_q10,
        q50=pred.pred_margin,
        q90=pred.pred_margin_q90,
        line=pred.vegas_spread,
    )

    # TOTAL:
    total_edge = pred.total_edge
    abs_t_edge = abs(total_edge)
    total_pick = "OVER" if total_edge > 0 else "UNDER"
    total_q_std = max((pred.pred_total_q90 - pred.pred_total_q10) / (2.0 * Z_80_QUANTILE), 1.0)
    total_prob, total_ev, total_stake = compute_win_prob_and_kelly(
        edge_pts=abs_t_edge,
        std_dev=total_q_std,
        bankroll=bankroll,
        kelly_fraction=kelly_fraction,
        q10=pred.pred_total_q10,
        q50=pred.pred_total,
        q90=pred.pred_total_q90,
        line=pred.vegas_total,
    )

    with st.container(border=True):
        # Card Header
        h_col1, h_col2 = st.columns([3, 1])
        with h_col1:
            st.markdown(f"#### 🏀 {away_name} @ {home_name}")
            st.caption(f"Matchup: `{away} @ {home}` | Lines from: `{book}`")
        with h_col2:
            if commence:
                try:
                    dt_utc = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                    dt_pht = dt_utc + timedelta(hours=8)
                    dt_et = dt_utc - timedelta(hours=4)
                    st.caption(f"⏰ **Tipoff:** {dt_pht.strftime('%a %b %d • %I:%M %p')} PHT  \n*(US ET: {dt_et.strftime('%a %I:%M %p')})*")
                except Exception:
                    st.caption(f"Tipoff: {commence}")

        # Split into Spread and Total columns
        col_spread, col_total = st.columns(2)

        # ---------------- SPREAD COLUMN ----------------
        with col_spread:
            is_s_sweet = (min_edge <= abs_s_edge <= max_edge)
            is_s_trap = (abs_s_edge > max_edge)
            s_badge = (
                " 🔥 **HIGH CONVICTION**"
                if (is_s_sweet and spread_pick_team == away)
                else (" ✅ **ACTIONABLE**" if is_s_sweet else "")
            )
            st.markdown(f"##### 📏 Spread Market{s_badge}")
            st.write(f"**Vegas Spread:** `{home} {book_spread_home:+.1f}` (`{away} {-book_spread_home:+.1f}`)")
            pred_home_spread = -pred.pred_margin
            st.write(
                f"**Model Fair Spread:** `{home} {pred_home_spread:+.1f}` "
                f"(Proj Margin: `{pred.pred_margin:+.1f}`, 80% CI: `{pred.pred_margin_q10:+.1f}` to `{pred.pred_margin_q90:+.1f}`)"
            )

            edge_color = "green" if is_s_sweet else ("orange" if is_s_trap else "gray")
            st.markdown(
                f"**Edge:** :{edge_color}[`{spread_edge:+.1f} pts`] → **{spread_pick_team} {spread_line:+.1f}** ({pred.confidence_spread})"
            )
            st.caption(f"Win Prob: `{spread_prob:.1%}` | EV: `{spread_ev:+.1%}` | Rec Stake: `${spread_stake:.2f}`")

            # One-click Track Spread Bet button
            spread_btn_key = f"track_spread_{home}_{away}_{card_index}"
            bet_side = f"{spread_pick_team} {spread_line:+.1f}"
            spread_bet_data = {
                "date": game.get("date", datetime.now().strftime("%Y-%m-%d")),
                "player": f"{away} @ {home}",
                "market": "SPREAD",
                "bet_type": "spread",
                "side": bet_side,
                "line": float(spread_line),
                "opening_line": float(spread_line),
                "closing_line": float(spread_line),
                "odds": -110,
                "odds_decimal": STANDARD_DECIMAL_ODDS,
                "predicted_mean": float(pred.pred_margin),
                "predicted_std": float(round(spread_q_std, 2)),
                "ev": float(spread_ev),
                "win_prob": float(spread_prob),
                "stake": float(spread_stake),
                "opponent": away if spread_pick_team == home else home,
                "tag": "game_model",
                "model_version": "game_xgb_v2_optuna_lstm",
                "result": "Pending",
                "tracked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            existing_spread = tracker.is_duplicate_bet(spread_bet_data) if hasattr(tracker, "is_duplicate_bet") else None
            if existing_spread:
                st.button(f"✓ Tracked: {bet_side}", key=spread_btn_key, disabled=True, use_container_width=True)
            else:
                if st.button(f"➕ Track {bet_side}", key=spread_btn_key, use_container_width=True):
                    try:
                        if hasattr(tracker, 'save_bet'):
                            tracker.save_bet(spread_bet_data)
                        elif hasattr(tracker, 'add_bet'):
                            tracker.add_bet(spread_bet_data)
                        else:
                            tracker.log_bet(spread_bet_data)
                        st.toast(f"✅ Tracked Spread Bet: {bet_side} (${spread_stake:.0f})")
                        time.sleep(0.3)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to track bet: {e}")

        # ---------------- TOTAL COLUMN ----------------
        with col_total:
            is_t_sweet = (min_edge <= abs_t_edge <= max_edge)
            is_t_trap = (abs_t_edge > max_edge)
            t_badge = " ✅ **ACTIONABLE**" if is_t_sweet else ""
            st.markdown(f"##### 🎯 Game Total (O/U){t_badge}")
            st.write(f"**Vegas Line:** `{pred.vegas_total:.1f}`")
            st.write(f"**Model Projection:** `{pred.pred_total:.1f}` (80% CI: `{pred.pred_total_q10:.1f}` to `{pred.pred_total_q90:.1f}`)")

            t_edge_color = "green" if is_t_sweet else ("orange" if is_t_trap else "gray")
            st.markdown(f"**Edge:** :{t_edge_color}[`{total_edge:+.1f} pts`] → **{pred.ou_pick} {pred.vegas_total:.1f}** ({pred.confidence_total})")
            st.caption(f"Win Prob: `{total_prob:.1%}` | EV: `{total_ev:+.1%}` | Rec Stake: `${total_stake:.2f}`")

            # One-click Track Total Bet button
            total_btn_key = f"track_total_{home}_{away}_{card_index}"
            bet_side_total = f"{total_pick} {pred.vegas_total:.1f}"
            total_bet_data = {
                "date": game.get("date", datetime.now().strftime("%Y-%m-%d")),
                "player": f"{away} @ {home}",
                "market": "TOTAL",
                "bet_type": "total",
                "side": bet_side_total,
                "line": float(pred.vegas_total),
                "opening_line": float(pred.vegas_total),
                "closing_line": float(pred.vegas_total),
                "odds": -110,
                "odds_decimal": STANDARD_DECIMAL_ODDS,
                "predicted_mean": float(pred.pred_total),
                "predicted_std": float(round(total_q_std, 2)),
                "ev": float(total_ev),
                "win_prob": float(total_prob),
                "stake": float(total_stake),
                "opponent": f"{away} vs {home}",
                "tag": "game_model",
                "model_version": "game_xgb_v2_optuna_lstm",
                "result": "Pending",
                "tracked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            existing_total = tracker.is_duplicate_bet(total_bet_data) if hasattr(tracker, "is_duplicate_bet") else None
            if existing_total:
                st.button(f"✓ Tracked: {bet_side_total}", key=total_btn_key, disabled=True, use_container_width=True)
            else:
                if st.button(f"➕ Track {pred.ou_pick} {pred.vegas_total:.1f}", key=total_btn_key, use_container_width=True):
                    try:
                        if hasattr(tracker, 'save_bet'):
                            tracker.save_bet(total_bet_data)
                        elif hasattr(tracker, 'add_bet'):
                            tracker.add_bet(total_bet_data)
                        else:
                            tracker.log_bet(total_bet_data)
                        st.toast(f"✅ Tracked Total Bet: {bet_side_total} (${total_stake:.0f})")
                        time.sleep(0.3)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to track bet: {e}")


def render_game_quant_arsenal_telemetry():
    """Renders Optuna + CUDA and 4-Gate LSTM telemetry for the Game Model (Spread & Total)."""
    hp_path = Path("config/game_model_hyperparams.json")
    lstm_path = Path("config/game_lstm_scout_metrics.json")
    if not hp_path.exists() and not lstm_path.exists():
        return

    with st.expander("⚡ Game Model Quant Arsenal Telemetry (Optuna + CUDA & 4-Gate Team LSTM)", expanded=False):
        if hp_path.exists():
            try:
                hp = safe_json_load(hp_path, default={})
                rows = []
                for mkt in ["spread", "total"]:
                    if mkt not in hp:
                        continue
                    val = hp[mkt]
                    meta = hp.get(f"_{mkt}_meta", {})
                    rows.append({
                        "Game Model": mkt.upper(),
                        "Games": f"{int(meta.get('samples', 0)):,}",
                        "GPU Trials": f"{meta.get('trials_run', 0)} ({meta.get('trials_pruned', 0)} pruned)",
                        "Pinball Loss": f"{meta.get('baseline_pinball', 0.0):.4f} → {meta.get('best_pinball', 0.0):.4f}",
                        "Pinball Δ": f"{meta.get('pinball_improvement_pct', 0.0):+.2f}%",
                        "q50 MAE": f"{meta.get('baseline_mae', 0.0):.3f} → {meta.get('best_mae', 0.0):.3f}",
                        "Depth / LR": f"d={val.get('max_depth', 4)} / lr={val.get('learning_rate', 0.04):.4f}",
                    })
                if rows:
                    st.markdown("##### ⚡ Optuna + CUDA Bayesian Hyperparameter Telemetry (`q10, q50, q90`)")
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            except Exception:
                pass

        if lstm_path.exists():
            try:
                lm = safe_json_load(lstm_path, default={})
                l_rows = []
                for k, meta in lm.items():
                    if isinstance(meta, dict):
                        l_rows.append({
                            "LSTM Head": k.upper(),
                            "Team-Games": f"{int(meta.get('samples', 0)):,}",
                            "Baseline Resid MAE": f"{meta.get('baseline_mae', 0.0):.4f}",
                            "LSTM OOF Resid MAE": f"{meta.get('lstm_oof_mae', 0.0):.4f}",
                            "OOF Corr (r)": f"{meta.get('oof_corr', 0.0):+.4f}",
                        })
                if l_rows:
                    st.markdown("##### 🧠 4-Gate Team Sequential LSTM Scout (`home/away_lstm_margin_mom` & `total_mom`)")
                    st.dataframe(pd.DataFrame(l_rows), hide_index=True, use_container_width=True)
            except Exception:
                pass


def render_games_tab(tracker: Any, bankroll_current: float = 1000.0):
    """Main entry point for the 🏀 Games tab."""
    st.markdown("### 🏀 NBA Game Predictions & Line Value")
    st.caption(
        "Pure Data-Driven Game XGBoost Quantile Models (`q10, q50, q90`) + 4-Gate Team LSTM & Optuna CUDA tuning. "
        "Settled 2026-27 track record: **79W–57L (58.1% overall, +14.8u)** and **64W–45L (58.7%, +13.2u) at Edge >= 2.0 pts**."
    )

    # Top Control Bar
    # Default date to latest harvested odds file with valid lines if available, else today
    default_date = datetime.now().date()
    existing_files = sorted(HARVEST_DIR.glob("odds_*.json"))
    if existing_files:
        for f in reversed(existing_files):
            try:
                data = safe_json_load(f, default={})
                g_list = data if isinstance(data, list) else data.get("games", [])
                if any(g.get("spread_home") is not None and g.get("total") is not None for g in g_list):
                    latest_dt_str = f.stem.replace("odds_", "")
                    default_date = datetime.strptime(latest_dt_str, "%Y-%m-%d").date()
                    break
            except Exception:
                pass

    ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([1.2, 1.2, 1.5, 1.1, 1.1])

    with ctrl1:
        selected_date = st.date_input("Slate Date", value=default_date, key="game_slate_date")
        date_str = selected_date.strftime("%Y-%m-%d")

    with ctrl2:
        kelly_frac = st.selectbox(
            "Kelly Sizing",
            [0.10, 0.25, 0.50, 1.0],
            index=1,
            format_func=lambda x: f"{x}x {'(1/4 Kelly)' if x == 0.25 else ''}",
        )

    with ctrl3:
        edge_band = st.slider(
            "Pure Model Edge Filter (pts)",
            min_value=0.5,
            max_value=20.0,
            value=(2.0, 20.0),
            step=0.5,
            help="Pure Game Model (0 artificial caps): |Edge| >= 2.0 pts hits 64W-45L (58.7% Win Rate, +13.18u, +12.1% ROI) across settled games.",
        )
        min_edge, max_edge = float(edge_band[0]), float(edge_band[1])

    with ctrl4:
        road_ats_only = st.checkbox(
            "🎯 Road ATS Sniper Only",
            value=False,
            help="Filter Best Picks strictly to Away ATS picks (197W-135L, 59.3% Win Rate, +13.3% ROI in 2024-25 backtest).",
        )

    with ctrl5:
        refresh_btn = st.button("🔄 Refresh Live Lines", use_container_width=True)

    render_game_quant_arsenal_telemetry()

    if refresh_btn:
        with st.spinner(f"Harvesting live DraftKings game lines for {date_str}..."):
            load_harvested_games(date_str, force_refresh=True)
            st.toast(f"Harvested live odds for {date_str}!")
            st.rerun()

    # Load slate games (automatically triggers on-demand harvest if lines missing)
    games = load_harvested_games(date_str, force_refresh=False)

    # Filter games that have valid spread and total
    valid_games = [g for g in games if g.get("spread_home") is not None and g.get("total") is not None]

    if not valid_games:
        st.info(f"No game lines found for **{date_str}**. Click **🔄 Refresh Live Lines** above or select an active slate date.")
    else:
        evaluated_games, best_game_picks = score_slate_games(
            valid_games=valid_games,
            bankroll=bankroll_current,
            kelly_fraction=kelly_frac,
            min_edge=min_edge,
            max_edge=max_edge,
            road_ats_priority=road_ats_only,
        )

        spread_picks_n = sum(1 for p in best_game_picks if p["market"] == "SPREAD")
        total_picks_n = sum(1 for p in best_game_picks if p["market"] == "TOTAL")
        total_exp = sum(float(p.get("stake", 0.0)) for p in best_game_picks)

        # Top Metrics & 1-Click Batch Track Bar
        gm1, gm2, gm3, gm4 = st.columns(4)
        gm1.metric("Matchups Posted", f"{len(evaluated_games)}", f"Slate: {date_str}")
        gm2.metric("Sweet-Spot Spreads", f"{spread_picks_n}", f"{min_edge:.1f}–{max_edge:.1f} pt edge")
        gm3.metric("Sweet-Spot Totals", f"{total_picks_n}", f"{min_edge:.1f}–{max_edge:.1f} pt edge")
        gm4.metric("Recommended Exposure", f"${total_exp:,.0f}", f"Bankroll: ${bankroll_current:,.0f}")

        if best_game_picks:
            with st.container(border=True):
                b_col1, b_col2 = st.columns([3, 1.3])
                with b_col1:
                    st.markdown(f"#### 🔥 Best Game Edges Worth Betting ({len(best_game_picks)} Qualifying Picks)")
                    picks_summary = " • ".join(
                        [f"**{p['side']}** (`{p['player']}`, Edge `+{p['edge_pts']:.1f} pts`)" for p in best_game_picks[:6]]
                    )
                    st.markdown(picks_summary)
                with b_col2:
                    existing_bets = tracker.get_bets() if hasattr(tracker, "get_bets") else []
                    untracked_game_picks = [
                        bp for bp in best_game_picks
                        if not (hasattr(tracker, "is_duplicate_bet") and tracker.is_duplicate_bet(bp, existing_bets))
                    ]

                    if not untracked_game_picks:
                        st.button(f"✓ All {len(best_game_picks)} Picks Tracked", disabled=True, use_container_width=True, key="all_games_already_tracked")
                    else:
                        btn_txt = (
                            f"🚀 Track {len(untracked_game_picks)} Best Game Picks"
                            if len(untracked_game_picks) < len(best_game_picks)
                            else f"🚀 Track All {len(best_game_picks)} Best Game Picks"
                        )
                        if st.button(
                            btn_txt,
                            type="primary",
                            use_container_width=True,
                            key="track_batch_best_game_picks",
                        ):
                            logged_n = 0
                            for bp in untracked_game_picks:
                                if hasattr(tracker, "save_bet"):
                                    ok = tracker.save_bet(bp)
                                elif hasattr(tracker, "add_bet"):
                                    ok = tracker.add_bet(bp)
                                else:
                                    ok = tracker.log_bet(bp)
                                if ok or ok is None:
                                    logged_n += 1
                            if len(untracked_game_picks) < len(best_game_picks):
                                st.success(f"✅ Tracked {logged_n} new Best Game Picks ({len(best_game_picks) - logged_n} were already tracked)!")
                            else:
                                st.success(f"✅ Tracked {logged_n} Best Game Picks into Bet Tracker!")
                            time.sleep(0.4)
                            st.rerun()

        st.markdown(f"#### 📅 Today's Matchups ({len(evaluated_games)} games with posted lines)")
        for idx, item in enumerate(evaluated_games):
            render_game_card(
                game=item["game"],
                pred=item["pred"],
                tracker=tracker,
                bankroll=bankroll_current,
                kelly_fraction=kelly_frac,
                min_edge=min_edge,
                card_index=idx,
                max_edge=max_edge,
            )

    # =========================================================================
    # SECTION 2: CUSTOM / HYPOTHETICAL MATCHUP PREDICTOR
    # =========================================================================
    with st.expander("🔮 Custom / Hypothetical Game Predictor", expanded=False):
        st.markdown("Predict any matchup or test hypothetical spreads and totals.")
        team_options = sorted(list(NBA_TEAMS.keys()))

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            custom_home = st.selectbox("Home Team", team_options, index=team_options.index("BOS"))
        with c2:
            custom_away = st.selectbox("Away Team", team_options, index=team_options.index("LAL"))
        with c3:
            custom_spread = st.number_input("Vegas Spread (Home, e.g. -4.5 = Home Fav)", value=-4.5, step=0.5, format="%.1f")
        with c4:
            custom_total = st.number_input("Vegas Game Total", value=224.5, step=0.5, format="%.1f")

        if st.button("Generate Custom Prediction", key="btn_custom_pred"):
            custom_pred = predict_game(
                home=custom_home,
                away=custom_away,
                vegas_spread=-float(custom_spread),
                vegas_total=custom_total,
            )
            if custom_pred:
                dummy_game = {
                    "home_team": custom_home,
                    "away_team": custom_away,
                    "spread_home": float(custom_spread),
                    "total": float(custom_total),
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "bookmaker": "Custom",
                }
                render_game_card(
                    game=dummy_game,
                    pred=custom_pred,
                    tracker=tracker,
                    bankroll=bankroll_current,
                    kelly_fraction=kelly_frac,
                    min_edge=min_edge,
                    card_index=999,
                    max_edge=max_edge,
                )

    # =========================================================================
    # SECTION 3: SETTLED GAMES & MODEL TRACK RECORD
    # =========================================================================
    with st.expander("📊 Settled Closing Lines & Harvest Performance", expanded=True):
        col_hdr, col_act = st.columns([3, 1.2])
        with col_hdr:
            st.markdown("##### Historical Settled Games & Model Evaluation")
            st.caption("Closing odds from DraftKings automatically graded against official NBA final scores.")
        with col_act:
            if st.button("🔄 Check & Settle Finished Games", use_container_width=True):
                with st.spinner("Settling games from recent slates..."):
                    settled_any = 0
                    for d_offset in range(4, -1, -1):
                        chk_date = (datetime.now() - timedelta(days=d_offset)).strftime("%Y-%m-%d")
                        try:
                            res = settle_games_for_date(chk_date)
                            settled_any += len(res)
                        except Exception as _set_err:
                            logger.debug(f"Settlement check for {chk_date}: {_set_err}")
                    st.success(f"Settled {settled_any} newly finished games!")
                    st.rerun()

        if SETTLED_CSV.exists():
            try:
                df_settled = pd.read_csv(SETTLED_CSV)
                if not df_settled.empty:
                    # Auto-backfill model predictions if any rows are missing them (e.g. settled on TrueNAS server)
                    needs_backfill = (
                        "model_ats_correct" not in df_settled.columns
                        or df_settled["model_ats_correct"].isna().any()
                        or "model_ats_pick" not in df_settled.columns
                        or df_settled["model_ats_pick"].isna().any()
                    )
                    if needs_backfill:
                        try:
                            from game_prediction import predict_game
                            for col in ["model_ats_pick", "model_ou_pick"]:
                                if col in df_settled.columns:
                                    df_settled[col] = df_settled[col].astype("object")
                                else:
                                    df_settled[col] = pd.Series(dtype="object")
                            needs_save = False
                            for idx, r in df_settled.iterrows():
                                if (
                                    pd.isna(r.get("model_ats_pick"))
                                    and pd.notna(r.get("closing_spread"))
                                    and pd.notna(r.get("closing_total"))
                                ):
                                    pred = predict_game(
                                        home=r["home_team"],
                                        away=r["away_team"],
                                        vegas_spread=-float(r["closing_spread"]),
                                        vegas_total=float(r["closing_total"]),
                                    )
                                    if pred:
                                        df_settled.at[idx, "pred_margin"] = pred.pred_margin
                                        df_settled.at[idx, "pred_total"] = pred.pred_total
                                        df_settled.at[idx, "spread_edge"] = pred.spread_edge
                                        df_settled.at[idx, "total_edge"] = pred.total_edge
                                        df_settled.at[idx, "model_ats_pick"] = pred.ats_pick
                                        df_settled.at[idx, "model_ou_pick"] = pred.ou_pick
                                        df_settled.at[idx, "model_ats_correct"] = int(
                                            (pred.ats_pick.startswith("HOME") and r["ats_winner"] == "HOME")
                                            or (pred.ats_pick.startswith("AWAY") and r["ats_winner"] == "AWAY")
                                        )
                                        df_settled.at[idx, "model_ou_correct"] = int(
                                            (pred.ou_pick == "OVER" and r["ou_winner"] == "OVER")
                                            or (pred.ou_pick == "UNDER" and r["ou_winner"] == "UNDER")
                                        )
                                        needs_save = True
                            if needs_save:
                                df_settled.to_csv(SETTLED_CSV, index=False)
                        except Exception as _bf_err:
                            logger.debug(f"Auto-backfill model predictions error: {_bf_err}")

                    # Calculate verified stats
                    valid_ats = df_settled[df_settled["model_ats_correct"].notna()]
                    valid_ou = df_settled[df_settled["model_ou_correct"].notna()]

                    ats_w = int(valid_ats["model_ats_correct"].sum()) if not valid_ats.empty else 0
                    ats_n = len(valid_ats)
                    ats_l = ats_n - ats_w
                    ats_pct = ats_w / ats_n if ats_n > 0 else 0.0

                    ou_w = int(valid_ou["model_ou_correct"].sum()) if not valid_ou.empty else 0
                    ou_n = len(valid_ou)
                    ou_l = ou_n - ou_w
                    ou_pct = ou_w / ou_n if ou_n > 0 else 0.0

                    comb_w = ats_w + ou_w
                    comb_n = ats_n + ou_n
                    comb_pct = comb_w / comb_n if comb_n > 0 else 0.0

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Settled Games", f"{len(df_settled)}")
                    m2.metric("Spread (ATS) Record", f"{ats_w}W - {ats_l}L ({ats_pct:.1%})")
                    m3.metric("Total (O/U) Record", f"{ou_w}W - {ou_l}L ({ou_pct:.1%})")
                    m4.metric("Overall Accuracy", f"{comb_pct:.1%} ({comb_w}/{comb_n})")

                    # Formatted Display Table
                    df_disp = df_settled.copy().sort_values("date", ascending=False)
                    df_disp["Final Score"] = df_disp.apply(
                        lambda r: f"{int(r['away_score'])} @ {int(r['home_score'])}"
                        if pd.notna(r.get("home_score")) and pd.notna(r.get("away_score"))
                        else "N/A",
                        axis=1,
                    )
                    df_disp["Closing Spread"] = df_disp.apply(
                        lambda r: f"{r['home_team']} {float(r['closing_spread']):+.1f}"
                        if pd.notna(r.get("closing_spread"))
                        else "N/A",
                        axis=1,
                    )
                    df_disp["ATS Result"] = df_disp.apply(
                        lambda r: "✅ WIN" if r.get("model_ats_correct") == 1 else ("❌ LOSS" if r.get("model_ats_correct") == 0 else "—"),
                        axis=1,
                    )
                    df_disp["Closing Total"] = df_disp.apply(
                        lambda r: f"O/U {float(r['closing_total']):.1f} (Act: {int(r['actual_total'])})"
                        if pd.notna(r.get("closing_total")) and pd.notna(r.get("actual_total"))
                        else "N/A",
                        axis=1,
                    )
                    df_disp["O/U Result"] = df_disp.apply(
                        lambda r: "✅ WIN" if r.get("model_ou_correct") == 1 else ("❌ LOSS" if r.get("model_ou_correct") == 0 else "—"),
                        axis=1,
                    )

                    disp_cols = [
                        "date", "matchup", "Final Score", "Closing Spread", "ats_winner",
                        "model_ats_pick", "ATS Result", "Closing Total", "ou_winner", "model_ou_pick", "O/U Result"
                    ]
                    col_rename = {
                        "date": "Date",
                        "matchup": "Matchup",
                        "ats_winner": "ATS Winner",
                        "model_ats_pick": "Model Pick",
                        "ou_winner": "O/U Winner",
                        "model_ou_pick": "Model O/U",
                    }
                    avail = [c for c in disp_cols if c in df_disp.columns]
                    st.dataframe(df_disp[avail].rename(columns=col_rename), use_container_width=True, height=280)
                else:
                    st.info("No settled games in dataset yet.")
            except Exception as e:
                st.warning(f"Could not load settled games: {e}")
        else:
            st.info("No settled games file found. Click 'Check & Settle Finished Games' to begin farming settled outcomes.")
