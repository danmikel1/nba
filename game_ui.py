"""
game_ui.py — Streamlit UI Component for Game-Level NBA Predictions & Betting

Integrates game-level spread and total models directly into the Streamlit dashboard:
  - Live & upcoming game odds ingestion (The Odds API + ESPN fallback)
  - XGBoost Quantile Regression predictions (Spread & Over/Under)
  - Mathematical Edge and Fractional Kelly stake sizing
  - One-click bet tracking directly into `Tracker` (bet_tracker.json)
  - Custom / hypothetical game predictor
  - Settled game outcomes and historical model performance tracking
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

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

# Empirical standard deviations for NBA game outcomes
STD_SPREAD = 11.5  # standard deviation of NBA game margins around spread
STD_TOTAL = 14.2   # standard deviation of NBA game totals around O/U line
STANDARD_DECIMAL_ODDS = 1.909  # -110 American odds


def compute_win_prob_and_kelly(
    edge_pts: float,
    std_dev: float,
    bankroll: float,
    kelly_fraction: float = 0.25,
    decimal_odds: float = STANDARD_DECIMAL_ODDS,
    max_stake_pct: float = 0.05,
) -> Tuple[float, float, float]:
    """
    Computes win probability and fractional Kelly stake.
    Returns: (win_prob, ev_pct, recommended_stake)
    """
    # z-score of the edge
    z = edge_pts / max(std_dev, 1.0)
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


def load_harvested_games(target_date: str) -> List[Dict[str, Any]]:
    """Load cached odds file or trigger fresh harvest if missing."""
    file_path = HARVEST_DIR / f"odds_{target_date}.json"
    if file_path.exists():
        data = safe_json_load(file_path, default={})
        games = data.get("games", [])
        if games:
            return games

    # If missing or empty, perform on-demand harvest
    try:
        return harvest_daily_odds(target_date)
    except Exception as e:
        st.warning(f"Could not load live odds for {target_date}: {e}")
        return []


def render_game_card(
    game: Dict[str, Any],
    pred: GamePrediction,
    tracker: Any,
    bankroll: float,
    kelly_fraction: float,
    min_edge: float,
    card_index: int,
):
    """Renders a styled prediction card for a single matchup with Spread and Total markets."""
    home = pred.home
    away = pred.away
    home_name = NBA_TEAMS.get(home, home)
    away_name = NBA_TEAMS.get(away, away)
    book = game.get("bookmaker", "Consensus")
    commence = game.get("commence_time", "")

    # Calculate probabilities and Kelly stakes
    # SPREAD:
    spread_edge = pred.spread_edge
    spread_pick_team = home if spread_edge > 0 else away
    spread_line = pred.vegas_spread if spread_edge > 0 else -pred.vegas_spread
    spread_prob, spread_ev, spread_stake = compute_win_prob_and_kelly(
        edge_pts=abs(spread_edge),
        std_dev=STD_SPREAD,
        bankroll=bankroll,
        kelly_fraction=kelly_fraction,
    )

    # TOTAL:
    total_edge = pred.total_edge
    total_pick = "OVER" if total_edge > 0 else "UNDER"
    total_prob, total_ev, total_stake = compute_win_prob_and_kelly(
        edge_pts=abs(total_edge),
        std_dev=STD_TOTAL,
        bankroll=bankroll,
        kelly_fraction=kelly_fraction,
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
                    dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                    st.caption(f"Tipoff: {dt.strftime('%I:%M %p UTC')}")
                except Exception:
                    pass

        # Split into Spread and Total columns
        col_spread, col_total = st.columns(2)

        # ---------------- SPREAD COLUMN ----------------
        with col_spread:
            st.markdown("##### 📏 Spread Market")
            st.write(f"**Vegas Line:** `{home} {pred.vegas_spread:+.1f}`")
            st.write(f"**Model Projection:** `{home} {pred.pred_margin:+.1f}` (80% CI: `{pred.pred_margin_q10:+.1f}` to `{pred.pred_margin_q90:+.1f}`)")

            edge_color = "green" if abs(spread_edge) >= min_edge else "gray"
            st.markdown(f"**Edge:** :{edge_color}[`{spread_edge:+.1f} pts`] → **{pred.ats_pick}** ({pred.confidence_spread})")
            st.caption(f"Win Prob: `{spread_prob:.1%}` | EV: `{spread_ev:+.1%}` | Rec Stake: `${spread_stake:.2f}`")

            # One-click Track Spread Bet button
            spread_btn_key = f"track_spread_{home}_{away}_{card_index}"
            if st.button(f"➕ Track {pred.ats_pick}", key=spread_btn_key, use_container_width=True):
                bet_side = f"{spread_pick_team} {spread_line:+.1f}"
                bet_data = {
                    "date": game.get("date", datetime.now().strftime("%Y-%m-%d")),
                    "player": f"{away} @ {home}",
                    "market": "SPREAD",
                    "bet_type": "spread",
                    "side": bet_side,
                    "line": float(spread_line),
                    "odds": -110,
                    "odds_decimal": STANDARD_DECIMAL_ODDS,
                    "predicted_mean": float(pred.pred_margin),
                    "predicted_std": float(STD_SPREAD),
                    "ev": float(spread_ev),
                    "win_prob": float(spread_prob),
                    "stake": float(spread_stake),
                    "opponent": away if spread_pick_team == home else home,
                    "tag": "game_model",
                    "model_version": "game_xgb_v1",
                    "result": "Pending",
                }
                try:
                    if hasattr(tracker, 'add_bet'):
                        tracker.add_bet(bet_data)
                    elif hasattr(tracker, 'log_bet'):
                        tracker.log_bet(bet_data)
                    else:
                        tracker.save_bet(bet_data)
                    st.success(f"Tracked Spread Bet: {bet_side} (${spread_stake:.0f})")
                except Exception as e:
                    st.error(f"Failed to track bet: {e}")

        # ---------------- TOTAL COLUMN ----------------
        with col_total:
            st.markdown("##### 🎯 Game Total (O/U)")
            st.write(f"**Vegas Line:** `{pred.vegas_total:.1f}`")
            st.write(f"**Model Projection:** `{pred.pred_total:.1f}` (80% CI: `{pred.pred_total_q10:.1f}` to `{pred.pred_total_q90:.1f}`)")

            t_edge_color = "green" if abs(total_edge) >= min_edge else "gray"
            st.markdown(f"**Edge:** :{t_edge_color}[`{total_edge:+.1f} pts`] → **{pred.ou_pick}** ({pred.confidence_total})")
            st.caption(f"Win Prob: `{total_prob:.1%}` | EV: `{total_ev:+.1%}` | Rec Stake: `${total_stake:.2f}`")

            # One-click Track Total Bet button
            total_btn_key = f"track_total_{home}_{away}_{card_index}"
            if st.button(f"➕ Track {pred.ou_pick} {pred.vegas_total:.1f}", key=total_btn_key, use_container_width=True):
                bet_side = f"{total_pick} {pred.vegas_total:.1f}"
                bet_data = {
                    "date": game.get("date", datetime.now().strftime("%Y-%m-%d")),
                    "player": f"{away} @ {home}",
                    "market": "TOTAL",
                    "bet_type": "total",
                    "side": bet_side,
                    "line": float(pred.vegas_total),
                    "odds": -110,
                    "odds_decimal": STANDARD_DECIMAL_ODDS,
                    "predicted_mean": float(pred.pred_total),
                    "predicted_std": float(STD_TOTAL),
                    "ev": float(total_ev),
                    "win_prob": float(total_prob),
                    "stake": float(total_stake),
                    "opponent": f"{away} vs {home}",
                    "tag": "game_model",
                    "model_version": "game_xgb_v1",
                    "result": "Pending",
                }
                try:
                    if hasattr(tracker, 'add_bet'):
                        tracker.add_bet(bet_data)
                    elif hasattr(tracker, 'log_bet'):
                        tracker.log_bet(bet_data)
                    else:
                        tracker.save_bet(bet_data)
                    st.success(f"Tracked Total Bet: {bet_side} (${total_stake:.0f})")
                except Exception as e:
                    st.error(f"Failed to track bet: {e}")


def render_games_tab(tracker: Any, bankroll_current: float = 1000.0):
    """Main entry point for the 🏀 Games tab."""
    st.markdown("### 🏀 NBA Game Predictions & Line Value")
    st.caption("Game-level XGBoost Quantile Regression models trained on 17 seasons of historical data (2008–2025).")

    # Top Control Bar
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.5, 1.2, 1.2, 1.2])

    with ctrl1:
        selected_date = st.date_input("Slate Date", value=datetime.now().date())
        date_str = selected_date.strftime("%Y-%m-%d")

    with ctrl2:
        kelly_frac = st.selectbox(
            "Kelly Sizing",
            [0.10, 0.25, 0.50, 1.0],
            index=1,
            format_func=lambda x: f"{x}x {'(Quarter Kelly)' if x == 0.25 else ''}",
        )

    with ctrl3:
        min_edge = st.slider("Min Edge (pts)", min_value=0.5, max_value=5.0, value=1.5, step=0.5)

    with ctrl4:
        refresh_btn = st.button("🔄 Refresh Live Lines", use_container_width=True)

    if refresh_btn:
        with st.spinner("Harvesting live odds from The Odds API & ESPN..."):
            harvest_daily_odds(date_str)
            st.success("Odds refreshed!")
            st.rerun()

    # Load slate games
    games = load_harvested_games(date_str)

    # Filter games that have valid spread and total
    valid_games = [g for g in games if g.get("spread_home") is not None and g.get("total") is not None]

    st.markdown(f"#### 📅 Today's Matchups ({len(valid_games)} games with posted lines)")

    if not valid_games:
        st.info(f"No posted game lines found for {date_str}. Use the button above to refresh, or try the Custom Matchup Predictor below.")
    else:
        for idx, g in enumerate(valid_games):
            home = g.get("home_team", "")
            away = g.get("away_team", "")
            spread = float(g.get("spread_home", 0.0))
            total = float(g.get("total", 220.0))
            ml_home = g.get("ml_home")
            ml_away = g.get("ml_away")

            pred = predict_game(
                home=home,
                away=away,
                vegas_spread=spread,
                vegas_total=total,
                moneyline_home=ml_home,
                moneyline_away=ml_away,
            )

            if pred:
                render_game_card(
                    game=g,
                    pred=pred,
                    tracker=tracker,
                    bankroll=bankroll_current,
                    kelly_fraction=kelly_frac,
                    min_edge=min_edge,
                    card_index=idx,
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
            custom_spread = st.number_input("Vegas Spread (Home)", value=-4.5, step=0.5, format="%.1f")
        with c4:
            custom_total = st.number_input("Vegas Game Total", value=224.5, step=0.5, format="%.1f")

        if st.button("Generate Custom Prediction", key="btn_custom_pred"):
            custom_pred = predict_game(
                home=custom_home,
                away=custom_away,
                vegas_spread=custom_spread,
                vegas_total=custom_total,
            )
            if custom_pred:
                dummy_game = {
                    "home_team": custom_home,
                    "away_team": custom_away,
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
                )

    # =========================================================================
    # SECTION 3: SETTLED GAMES & MODEL TRACK RECORD
    # =========================================================================
    with st.expander("📊 Settled Closing Lines & Harvest Performance", expanded=True):
        col_hdr, col_act = st.columns([3, 1])
        with col_hdr:
            st.markdown("##### Historical Settled Games")
            st.caption("Harvested closing lines automatically matched against official final scores.")
        with col_act:
            if st.button("🔄 Check & Settle Finished Games", use_container_width=True):
                with st.spinner("Settling games from yesterday and today..."):
                    today = datetime.now().strftime("%Y-%m-%d")
                    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
                    settle_games_for_date(yesterday)
                    settle_games_for_date(today)
                    st.success("Games settled!")
                    st.rerun()

        if SETTLED_CSV.exists():
            try:
                df_settled = pd.read_csv(SETTLED_CSV)
                if not df_settled.empty:
                    # Metrics row
                    ats_acc = df_settled["model_ats_correct"].mean() if "model_ats_correct" in df_settled else 0.0
                    ou_acc = df_settled["model_ou_correct"].mean() if "model_ou_correct" in df_settled else 0.0
                    total_settled = len(df_settled)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Settled Games", f"{total_settled}")
                    m2.metric("Spread (ATS) Win Rate", f"{ats_acc:.1%}" if pd.notna(ats_acc) else "N/A")
                    m3.metric("Total (O/U) Win Rate", f"{ou_acc:.1%}" if pd.notna(ou_acc) else "N/A")
                    m4.metric("Dataset File", "settled_games.csv")

                    # Display dataframe
                    display_cols = [
                        "date", "matchup", "actual_margin", "actual_total",
                        "closing_spread", "closing_total", "ats_winner", "ou_winner",
                        "model_ats_pick", "model_ou_pick", "model_ats_correct", "model_ou_correct"
                    ]
                    available_cols = [c for c in display_cols if c in df_settled.columns]
                    st.dataframe(df_settled[available_cols].sort_values("date", ascending=False), height=250)
                else:
                    st.info("No settled games in dataset yet.")
            except Exception as e:
                st.warning(f"Could not load settled games: {e}")
        else:
            st.info("No settled games file found. Click 'Check & Settle Finished Games' to begin farming settled outcomes.")
