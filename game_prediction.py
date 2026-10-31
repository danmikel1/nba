"""

game_prediction.py — Game-level spread & total prediction models.



Separate from nba_prediction.py (player props).

Uses nba_2008-2025.csv historical game data to engineer team-level

features and train XGBoost regression models for:

  1. Spread model  → predicts actual margin (home − away)

  2. Total model   → predicts actual total  (home + away)



Usage:

    # Train models

    python game_prediction.py train [--seasons 2015-2025] [--test-seasons 2024-2025]



    # Predict today's games

    python game_prediction.py predict --home LAL --away BOS --spread -3.5 --total 220.5



    # Backtest

    python game_prediction.py backtest --seasons 2024-2025

"""



import argparse

import json

import logging

import sys

from dataclasses import dataclass, field

from pathlib import Path

from typing import Optional, Tuple



import joblib

import numpy as np

import pandas as pd

from scipy import stats

from sklearn.isotonic import IsotonicRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.model_selection import TimeSeriesSplit, train_test_split

from xgboost import XGBRegressor



# ---------------------------------------------------------------------------

# Config

# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

logger = logging.getLogger(__name__)



BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

MODELS_DIR = BASE_DIR / "models" / "game"

MODELS_DIR.mkdir(parents=True, exist_ok=True)



GAME_CSV = DATA_DIR / "nba_2008-2025.csv"



# Team abbreviation mapping (CSV uses lowercase short names)

TEAM_ABBREV_MAP = {

    "atl": "ATL", "bkn": "BKN", "bos": "BOS", "cha": "CHA", "chi": "CHI",

    "cle": "CLE", "dal": "DAL", "den": "DEN", "det": "DET", "gs": "GSW",

    "hou": "HOU", "ind": "IND", "lac": "LAC", "lal": "LAL", "mem": "MEM",

    "mia": "MIA", "mil": "MIL", "min": "MIN", "no": "NOP", "ny": "NYK",

    "okc": "OKC", "orl": "ORL", "phi": "PHI", "phx": "PHX", "por": "POR",

    "sa": "SAS", "sac": "SAC", "tor": "TOR", "utah": "UTA", "wsh": "WAS",

}



# Reverse map (standard → csv key) for convenience

STD_TO_CSV = {v: k for k, v in TEAM_ABBREV_MAP.items()}



# ---------------------------------------------------------------------------

# Feature list (game-level)

# ---------------------------------------------------------------------------

GAME_FEATURE_COLUMNS = [

    # Vegas lines (strongest signals)

    "vegas_spread",            # posted spread (positive = home favored by X)

    "vegas_total",             # posted O/U

    "implied_home_prob",       # implied win prob from moneylines



    # Team rolling offense/defense (home)

    "home_ppg_l10",            # home team points per game last 10

    "home_pag_l10",            # home team points allowed per game last 10

    "home_ppg_l20",

    "home_pag_l20",

    "home_margin_l10",         # avg margin last 10

    "home_margin_l20",



    # Team rolling offense/defense (away)

    "away_ppg_l10",

    "away_pag_l10",

    "away_ppg_l20",

    "away_pag_l20",

    "away_margin_l10",

    "away_margin_l20",



    # Pace proxies

    "home_pace_l10",           # avg game total in home team's last 10

    "away_pace_l10",



    # Rest & scheduling

    "home_rest_days",          # days since last game

    "away_rest_days",

    "home_b2b",                # back-to-back flag

    "away_b2b",



    # Streaks & momentum

    "home_win_streak",         # positive = win streak, negative = losing streak

    "away_win_streak",

    "home_ats_streak",         # against the spread streak

    "away_ats_streak",

    "home_ou_streak",          # over/under streak

    "away_ou_streak",



    # Scoring trends

    "home_1h_pct_l10",        # % of points scored in 1st half (last 10)

    "away_1h_pct_l10",

    "home_q4_ppg_l10",        # 4th quarter points (clutch proxy)

    "away_q4_ppg_l10",



    # Head-to-head

    "h2h_home_margin_l5",      # home team margin vs this opponent (last 5 meetings)



    # Season context

    "season_game_number",      # team's nth game this season

    "is_playoffs",             # playoff flag



    # Derived edge-finding features (deviation from Vegas expectations)

    "rest_diff",               # home_rest - away_rest (rest advantage)

    "pace_interaction",        # combined pace signal

    "form_diff",               # home recent margin - away recent margin

    "offense_vs_defense_home", # home ppg vs away defensive rating

    "offense_vs_defense_away", # away ppg vs home defensive rating

    "implied_total_from_form", # sum of team ppg averages vs vegas total

    "implied_margin_from_form",# margin from team strength vs vegas spread

    "b2b_mismatch",            # one team on b2b, other rested



    # Tool #4: 4-Gate LSTM Sequential Team Momentum Features

    "home_lstm_margin_mom",

    "away_lstm_margin_mom",

    "lstm_margin_diff",

    "home_lstm_total_mom",

    "away_lstm_total_mom",

    "lstm_total_sum",

]





# ---------------------------------------------------------------------------

# Feature engineering

# ---------------------------------------------------------------------------



def load_game_data(path: Path = GAME_CSV) -> pd.DataFrame:

    """Load and clean the historical game CSV."""

    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])

    df.sort_values("date", inplace=True)

    df.reset_index(drop=True, inplace=True)



    # Derived targets

    df["actual_margin"] = df["score_home"] - df["score_away"]

    df["actual_total"] = df["score_home"] + df["score_away"]



    # Signed spread: positive means home favored

    # The CSV "spread" column is always positive (magnitude).

    # "whos_favored" tells us direction.

    df["signed_spread"] = df.apply(

        lambda r: r["spread"] if r["whos_favored"] == "home" else -r["spread"]

        if pd.notna(r["spread"]) else 0.0, axis=1

    )



    return df





def _moneyline_to_prob(ml: float) -> float:

    """Convert American moneyline to implied probability."""

    if pd.isna(ml) or ml == 0:

        return 0.5

    if ml > 0:

        return 100.0 / (ml + 100.0)

    else:

        return abs(ml) / (abs(ml) + 100.0)





def _safe_rolling(series: pd.Series, window: int, min_periods: int = 5) -> pd.Series:

    """Shifted rolling mean (excludes current game)."""

    return series.shift(1).rolling(window=window, min_periods=min_periods).mean()





def build_team_histories(df: pd.DataFrame) -> dict:

    """

    Build per-team game history DataFrames for rolling feature computation.

    Returns dict[team_csv_key] → DataFrame of that team's games in order.

    """

    records = []

    for _, row in df.iterrows():

        date = row["date"]

        season = row["season"]

        playoffs = row["playoffs"]



        # Home perspective

        records.append({

            "team": row["home"],

            "opponent": row["away"],

            "date": date,

            "season": season,

            "is_home": 1,

            "pts_for": row["score_home"],

            "pts_against": row["score_away"],

            "margin": row["score_home"] - row["score_away"],

            "game_total": row["score_home"] + row["score_away"],

            "q1": row["q1_home"],

            "q2": row["q2_home"],

            "q3": row["q3_home"],

            "q4": row["q4_home"],

            "ot": row["ot_home"],

            "spread": row.get("signed_spread", 0),

            "total_line": row["total"],

            "won": int(row["score_home"] > row["score_away"]),

            "covered": int((row["score_home"] - row["score_away"]) > row.get("signed_spread", 0))

                        if pd.notna(row.get("signed_spread")) else 0,

            "went_over": int((row["score_home"] + row["score_away"]) > row["total"])

                         if pd.notna(row.get("total")) else 0,

            "playoffs": playoffs,

        })



        # Away perspective

        records.append({

            "team": row["away"],

            "opponent": row["home"],

            "date": date,

            "season": season,

            "is_home": 0,

            "pts_for": row["score_away"],

            "pts_against": row["score_home"],

            "margin": row["score_away"] - row["score_home"],

            "game_total": row["score_home"] + row["score_away"],

            "q1": row["q1_away"],

            "q2": row["q2_away"],

            "q3": row["q3_away"],

            "q4": row["q4_away"],

            "ot": row["ot_away"],

            "spread": -row.get("signed_spread", 0),

            "total_line": row["total"],

            "won": int(row["score_away"] > row["score_home"]),

            "covered": int((row["score_away"] - row["score_home"]) > -row.get("signed_spread", 0))

                        if pd.notna(row.get("signed_spread")) else 0,

            "went_over": int((row["score_home"] + row["score_away"]) > row["total"])

                         if pd.notna(row.get("total")) else 0,

            "playoffs": playoffs,

        })



    team_df = pd.DataFrame(records)

    team_df.sort_values(["team", "date"], inplace=True)



    result = {}

    for team, group in team_df.groupby("team"):

        g = group.copy()

        g.reset_index(drop=True, inplace=True)



        # Rolling features (shifted to avoid leakage)

        g["ppg_l10"] = _safe_rolling(g["pts_for"], 10)

        g["pag_l10"] = _safe_rolling(g["pts_against"], 10)

        g["ppg_l20"] = _safe_rolling(g["pts_for"], 20, min_periods=10)

        g["pag_l20"] = _safe_rolling(g["pts_against"], 20, min_periods=10)

        g["margin_l10"] = _safe_rolling(g["margin"], 10)

        g["margin_l20"] = _safe_rolling(g["margin"], 20, min_periods=10)

        g["pace_l10"] = _safe_rolling(g["game_total"], 10)



        # Quarter scoring

        g["_1h_pts"] = g["q1"] + g["q2"]

        g["_total_pts"] = g["pts_for"]

        g["_1h_pct"] = g["_1h_pts"] / g["_total_pts"].replace(0, 1)

        g["_1h_pct_l10"] = _safe_rolling(g["_1h_pct"], 10)

        g["q4_ppg_l10"] = _safe_rolling(g["q4"], 10)



        # Streaks (computed iteratively)

        win_streak = []

        ats_streak = []

        ou_streak = []

        ws = 0

        ats_s = 0

        ou_s = 0

        for i, r in g.iterrows():

            win_streak.append(ws)

            ats_streak.append(ats_s)

            ou_streak.append(ou_s)



            # Update streak (after recording pre-game value)

            if r["won"] == 1:

                ws = ws + 1 if ws > 0 else 1

            else:

                ws = ws - 1 if ws < 0 else -1



            if r["covered"] == 1:

                ats_s = ats_s + 1 if ats_s > 0 else 1

            else:

                ats_s = ats_s - 1 if ats_s < 0 else -1



            if r["went_over"] == 1:

                ou_s = ou_s + 1 if ou_s > 0 else 1

            else:

                ou_s = ou_s - 1 if ou_s < 0 else -1



        g["win_streak"] = win_streak

        g["ats_streak"] = ats_streak

        g["ou_streak"] = ou_streak



        # Rest days

        g["rest_days"] = g["date"].diff().dt.days.fillna(7).clip(0, 14)

        g["b2b"] = (g["rest_days"] <= 1).astype(int)



        # Season game number

        g["season_game_num"] = g.groupby("season").cumcount() + 1



        # Tool #4: 4-Gate LSTM sequential features (8-game lookback window)

        X_seq, C_now = _build_team_seq_tensors(g, include_last=False)

        lstm_scout = _get_game_lstm_scout()

        if lstm_scout and "margin_lstm" in lstm_scout and "total_lstm" in lstm_scout:

            m_pred, _ = lstm_scout["margin_lstm"].predict(X_seq, C_now)

            t_pred, _ = lstm_scout["total_lstm"].predict(X_seq, C_now)

            g["lstm_margin_mom"] = m_pred

            g["lstm_total_mom"] = t_pred

        else:

            g["lstm_margin_mom"] = np.zeros(len(g), dtype=np.float32)

            g["lstm_total_mom"] = np.zeros(len(g), dtype=np.float32)



        result[team] = g



    return result





_cached_game_lstm_scout = None





def _get_game_lstm_scout():

    global _cached_game_lstm_scout

    if _cached_game_lstm_scout is not None:

        return _cached_game_lstm_scout

    scout_path = MODELS_DIR / "game_lstm_scout.pkl"

    if scout_path.exists():

        try:

            _cached_game_lstm_scout = joblib.load(scout_path)

            return _cached_game_lstm_scout

        except Exception as e:

            logger.warning(f"Failed to load game LSTM scout: {e}")

    return None





def _build_team_seq_tensors(g: pd.DataFrame, include_last: bool = False, seq_len: int = 8):

    """Build (N, 8, 6) sequential tensor and (N, 4) context matrix for a team DataFrame."""

    n = len(g)

    margin_res = np.clip((g["margin"].fillna(0).to_numpy(dtype=np.float32) - g["spread"].fillna(0).to_numpy(dtype=np.float32)) / 12.0, -3.5, 3.5)

    total_res = np.clip((g["game_total"].fillna(220).to_numpy(dtype=np.float32) - g["total_line"].fillna(220).to_numpy(dtype=np.float32)) / 15.0, -3.5, 3.5)

    pts_for_z = (g["pts_for"].fillna(110).to_numpy(dtype=np.float32) - 112.0) / 12.0

    pts_against_z = (g["pts_against"].fillna(110).to_numpy(dtype=np.float32) - 112.0) / 12.0

    rest_z = g["rest_days"].fillna(2).to_numpy(dtype=np.float32) / 7.0

    b2b_f = g["b2b"].fillna(0).to_numpy(dtype=np.float32)



    step_mat = np.stack([margin_res, total_res, pts_for_z, pts_against_z, rest_z, b2b_f], axis=1).astype(np.float32)



    if include_last:

        X_live = np.zeros((1, seq_len, 6), dtype=np.float32)

        tail_steps = step_mat[-seq_len:]

        X_live[0, seq_len - len(tail_steps):, :] = tail_steps

        m10 = float(g["margin"].tail(10).mean()) if n > 0 else 0.0

        p10 = float(g["game_total"].tail(10).mean()) if n > 0 else 220.0

        C_live = np.array([[m10 / 10.0, (p10 - 220.0) / 15.0, rest_z[-1] if n > 0 else 0.3, b2b_f[-1] if n > 0 else 0.0]], dtype=np.float32)

        return X_live, C_live



    X_seq = np.zeros((n, seq_len, 6), dtype=np.float32)

    for k in range(1, seq_len + 1):

        t_pos = seq_len - k

        if n > k:

            X_seq[k:, t_pos, :] = step_mat[:-k]



    m10_arr = g["margin_l10"].fillna(0).to_numpy(dtype=np.float32) / 10.0

    p10_arr = (g["pace_l10"].fillna(220).to_numpy(dtype=np.float32) - 220.0) / 15.0

    C_now = np.stack([m10_arr, p10_arr, rest_z, b2b_f], axis=1).astype(np.float32)

    return X_seq, C_now





def _get_h2h_avg_margin(team_hist: pd.DataFrame, opponent: str, n: int = 5) -> float:

    """Get average margin in last n meetings vs a specific opponent (pre-game)."""

    matchups = team_hist[team_hist["opponent"] == opponent]

    if len(matchups) < 2:

        return 0.0

    # Exclude the most recent (current game row comes from the join logic)

    recent = matchups.tail(n)

    return recent["margin"].mean()





def build_game_features(df: pd.DataFrame, team_histories: dict) -> pd.DataFrame:

    """

    Build the feature matrix for every game in df.

    Each row = one game. Features are looked up from team_histories

    using the most recent pre-game values (no leakage).

    """

    features_list = []



    for idx, row in df.iterrows():

        home = row["home"]

        away = row["away"]

        date = row["date"]



        home_hist = team_histories.get(home)

        away_hist = team_histories.get(away)



        if home_hist is None or away_hist is None:

            continue



        # Find the row in team history that matches this game (home perspective)

        hh = home_hist[(home_hist["date"] == date) & (home_hist["opponent"] == away) & (home_hist["is_home"] == 1)]

        ah = away_hist[(away_hist["date"] == date) & (away_hist["opponent"] == home) & (away_hist["is_home"] == 0)]



        if hh.empty or ah.empty:

            continue



        hh = hh.iloc[0]

        ah = ah.iloc[0]



        # Vegas

        impl_home_prob = _moneyline_to_prob(row.get("moneyline_home", np.nan))



        # H2H: use pre-game history only (exclude current game)

        h2h_mask = (home_hist["opponent"] == away) & (home_hist["date"] < date)

        h2h_games = home_hist[h2h_mask].tail(5)

        h2h_margin = h2h_games["margin"].mean() if len(h2h_games) >= 2 else 0.0



        feat = {

            # Vegas

            "vegas_spread": row.get("signed_spread", 0.0),

            "vegas_total": row["total"],

            "implied_home_prob": impl_home_prob,



            # Home rolling

            "home_ppg_l10": hh.get("ppg_l10", np.nan),

            "home_pag_l10": hh.get("pag_l10", np.nan),

            "home_ppg_l20": hh.get("ppg_l20", np.nan),

            "home_pag_l20": hh.get("pag_l20", np.nan),

            "home_margin_l10": hh.get("margin_l10", np.nan),

            "home_margin_l20": hh.get("margin_l20", np.nan),



            # Away rolling

            "away_ppg_l10": ah.get("ppg_l10", np.nan),

            "away_pag_l10": ah.get("pag_l10", np.nan),

            "away_ppg_l20": ah.get("ppg_l20", np.nan),

            "away_pag_l20": ah.get("pag_l20", np.nan),

            "away_margin_l10": ah.get("margin_l10", np.nan),

            "away_margin_l20": ah.get("margin_l20", np.nan),



            # Pace

            "home_pace_l10": hh.get("pace_l10", np.nan),

            "away_pace_l10": ah.get("pace_l10", np.nan),



            # Rest

            "home_rest_days": hh.get("rest_days", 3),

            "away_rest_days": ah.get("rest_days", 3),

            "home_b2b": hh.get("b2b", 0),

            "away_b2b": ah.get("b2b", 0),



            # Streaks

            "home_win_streak": hh.get("win_streak", 0),

            "away_win_streak": ah.get("win_streak", 0),

            "home_ats_streak": hh.get("ats_streak", 0),

            "away_ats_streak": ah.get("ats_streak", 0),

            "home_ou_streak": hh.get("ou_streak", 0),

            "away_ou_streak": ah.get("ou_streak", 0),



            # Scoring trends

            "home_1h_pct_l10": hh.get("_1h_pct_l10", 0.5),

            "away_1h_pct_l10": ah.get("_1h_pct_l10", 0.5),

            "home_q4_ppg_l10": hh.get("q4_ppg_l10", np.nan),

            "away_q4_ppg_l10": ah.get("q4_ppg_l10", np.nan),



            # H2H

            "h2h_home_margin_l5": h2h_margin,



            # Context

            "season_game_number": hh.get("season_game_num", 40),

            "is_playoffs": int(row.get("playoffs", False)),



            # Derived edge-finding features

            "rest_diff": hh.get("rest_days", 3) - ah.get("rest_days", 3),

            "pace_interaction": (hh.get("pace_l10", 220) + ah.get("pace_l10", 220)) / 2.0,

            "form_diff": (hh.get("margin_l10", 0) or 0) - (ah.get("margin_l10", 0) or 0),

            "offense_vs_defense_home": (hh.get("ppg_l10", 110) or 110) - (ah.get("pag_l10", 110) or 110),

            "offense_vs_defense_away": (ah.get("ppg_l10", 110) or 110) - (hh.get("pag_l10", 110) or 110),

            "implied_total_from_form": ((hh.get("ppg_l10", 110) or 110) + (ah.get("ppg_l10", 110) or 110))

                                       - row["total"],

            "implied_margin_from_form": ((hh.get("margin_l10", 0) or 0) - (ah.get("margin_l10", 0) or 0))

                                        - row.get("signed_spread", 0),

            "b2b_mismatch": hh.get("b2b", 0) - ah.get("b2b", 0),

            "home_lstm_margin_mom": float(hh.get("lstm_margin_mom", 0.0) or 0.0),

            "away_lstm_margin_mom": float(ah.get("lstm_margin_mom", 0.0) or 0.0),

            "lstm_margin_diff": float(hh.get("lstm_margin_mom", 0.0) or 0.0) - float(ah.get("lstm_margin_mom", 0.0) or 0.0),

            "home_lstm_total_mom": float(hh.get("lstm_total_mom", 0.0) or 0.0),

            "away_lstm_total_mom": float(ah.get("lstm_total_mom", 0.0) or 0.0),

            "lstm_total_sum": float(hh.get("lstm_total_mom", 0.0) or 0.0) + float(ah.get("lstm_total_mom", 0.0) or 0.0),



            # Targets

            "_actual_margin": row["actual_margin"],

            "_actual_total": row["actual_total"],

            "_date": date,

            "_season": row["season"],

            "_home": home,

            "_away": away,

        }



        features_list.append(feat)



    result = pd.DataFrame(features_list)

    logger.info(f"Built {len(result)} game feature rows")

    return result





# ---------------------------------------------------------------------------

# Model training

# ---------------------------------------------------------------------------



from hardware_utils import get_xgboost_hardware_params



XGBOOST_PARAMS = get_xgboost_hardware_params({

    "n_estimators": 600,

    "learning_rate": 0.04,

    "max_depth": 5,

    "min_child_weight": 8,

    "subsample": 0.8,

    "colsample_bytree": 0.8,

    "reg_alpha": 1.0,

    "reg_lambda": 2.0,

    "gamma": 0.15,

    "random_state": 42,

    "verbosity": 0,

})



QUANTILES = [0.10, 0.50, 0.90]

QUANTILE_NAMES = ["q10", "q50", "q90"]





def get_tuned_game_params(model_name: str) -> dict:

    hp_path = BASE_DIR / "config" / "game_model_hyperparams.json"

    if hp_path.exists():

        try:

            hp = json.loads(hp_path.read_text(encoding="utf-8"))

            if model_name in hp and isinstance(hp[model_name], dict):

                logger.info(f"Loaded Optuna CUDA hyperparameters for game/{model_name}: {hp[model_name]}")

                return hp[model_name]

        except Exception as e:

            logger.warning(f"Could not load game hyperparams: {e}")

    return {}





def train_game_model(

    X_train: pd.DataFrame, y_train: pd.Series,

    X_test: pd.DataFrame, y_test: pd.Series,

    model_name: str,

    vegas_col: str = None,

    vegas_test: np.ndarray = None,

):

    """

    Train quantile regression models (q10, q50, q90) for a game-level target.



    Returns: (models_dict, metrics_dict)

    """

    models = {}

    best_iters = {}



    tuned_hp = get_tuned_game_params(model_name)

    merged_xgb_params = {**XGBOOST_PARAMS, **tuned_hp}

    for q, name in zip(QUANTILES, QUANTILE_NAMES):

        use_early_stop = True

        params = {

            **merged_xgb_params,

            "objective": "reg:quantileerror",

            "quantile_alpha": q,

            "eval_metric": "quantile",

            "early_stopping_rounds": 30,

        }



        model = XGBRegressor(**params)

        model.fit(

            X_train, y_train,

            eval_set=[(X_test, y_test)],

            verbose=False,

        )

        models[name] = model

        bi = getattr(model, "best_iteration", XGBOOST_PARAMS["n_estimators"])

        best_iters[name] = bi

        logger.info(f"  {model_name}/{name}: {'early stop @' if use_early_stop else 'ran'} iter {bi}")



    # Evaluate

    pred_10 = models["q10"].predict(X_test)

    pred_50 = models["q50"].predict(X_test)

    pred_90 = models["q90"].predict(X_test)



    mae = mean_absolute_error(y_test, pred_50)

    rmse = np.sqrt(mean_squared_error(y_test, pred_50))

    r2 = r2_score(y_test, pred_50)

    coverage = ((y_test.values >= pred_10) & (y_test.values <= pred_90)).mean() * 100

    avg_width = (pred_90 - pred_10).mean()



    # Compare vs Vegas baseline

    vegas_mae = None

    if vegas_test is not None:

        vegas_mae = mean_absolute_error(y_test, vegas_test)



    # Time-series CV

    cv_mae_folds = []

    try:

        tscv = TimeSeriesSplit(n_splits=5)

        X_cv = pd.concat([X_train, X_test])

        y_cv = pd.concat([y_train, y_test])

        for fold_tr, fold_val in tscv.split(X_cv):

            fold_model = XGBRegressor(

                **XGBOOST_PARAMS,

                objective="reg:quantileerror",

                quantile_alpha=0.50,

                eval_metric="rmse",

                early_stopping_rounds=25,

            )

            fold_model.fit(

                X_cv.iloc[fold_tr], y_cv.iloc[fold_tr],

                eval_set=[(X_cv.iloc[fold_val], y_cv.iloc[fold_val])],

                verbose=False,

            )

            cv_mae_folds.append(mean_absolute_error(y_cv.iloc[fold_val], fold_model.predict(X_cv.iloc[fold_val])))

    except Exception as e:

        logger.warning(f"CV failed for {model_name}: {e}")



    cv_mae = float(np.mean(cv_mae_folds)) if cv_mae_folds else None

    cv_std = float(np.std(cv_mae_folds)) if cv_mae_folds else None



    metrics = {

        "name": model_name,

        "samples_train": len(y_train),

        "samples_test": len(y_test),

        "mae": round(mae, 3),

        "rmse": round(rmse, 3),

        "r2": round(r2, 4),

        "coverage_80pct": round(coverage, 1),

        "avg_width": round(avg_width, 2),

        "vegas_mae": round(vegas_mae, 3) if vegas_mae is not None else None,

        "improvement_vs_vegas": round(vegas_mae - mae, 3) if vegas_mae is not None else None,

        "cv_mae": round(cv_mae, 3) if cv_mae else None,

        "cv_std": round(cv_std, 3) if cv_std else None,

    }



    return models, metrics





def get_feature_importance(models: dict, feature_names: list) -> pd.DataFrame:

    """Extract feature importance from the q50 model."""

    q50 = models.get("q50")

    if q50 is None:

        return pd.DataFrame()



    imp = q50.feature_importances_

    fi = pd.DataFrame({"feature": feature_names, "importance": imp})

    fi.sort_values("importance", ascending=False, inplace=True)

    fi.reset_index(drop=True, inplace=True)

    return fi





# ---------------------------------------------------------------------------

# Full pipeline

# ---------------------------------------------------------------------------



def run_training(

    min_season: int = 2010,

    max_season: int = 2025,

    test_seasons: tuple = (2024, 2025),

):

    """

    End-to-end: load data → engineer features → train spread & total models.

    """

    logger.info("=" * 60)

    logger.info("Game Prediction Model — Training Pipeline")

    logger.info("=" * 60)



    # 1. Load data

    logger.info("Loading game data...")

    df = load_game_data()

    df = df[(df["season"] >= min_season) & (df["season"] <= max_season)]

    df = df[df["spread"].notna()]  # drop rows without spread

    logger.info(f"  {len(df):,} games from {min_season}–{max_season}")



    # 2. Build team histories

    logger.info("Building team rolling histories...")

    team_histories = build_team_histories(df)

    logger.info(f"  {len(team_histories)} teams tracked")



    # 3. Build feature matrix

    logger.info("Engineering game features...")

    feat_df = build_game_features(df, team_histories)



    # Drop rows with missing features (early-season cold start)

    feat_X = feat_df[GAME_FEATURE_COLUMNS].copy()

    valid_mask = feat_X.notna().all(axis=1)

    feat_df = feat_df[valid_mask].copy()

    feat_X = feat_X[valid_mask].copy()

    logger.info(f"  {len(feat_df):,} games with complete features")



    # 4. Train/test split by season

    test_mask = feat_df["_season"].isin(test_seasons)

    train_mask = ~test_mask



    X_train = feat_X[train_mask]

    X_test = feat_X[test_mask]



    logger.info(f"  Train: {len(X_train):,} games | Test: {len(X_test):,} games")



    # ---- SPREAD MODEL ----

    logger.info("\n--- Training SPREAD model ---")

    y_train_spread = feat_df.loc[train_mask, "_actual_margin"]

    y_test_spread = feat_df.loc[test_mask, "_actual_margin"]

    vegas_spread_test = feat_df.loc[test_mask, "vegas_spread"].values  # baseline: predicted margin = -spread



    # Vegas predicts margin as -spread (if home is favored by 5, vegas predicts home wins by 5)

    spread_models, spread_metrics = train_game_model(

        X_train, y_train_spread, X_test, y_test_spread,

        model_name="spread",

        vegas_test=vegas_spread_test,

    )



    # ---- TOTAL MODEL ----

    logger.info("\n--- Training TOTAL model ---")

    y_train_total = feat_df.loc[train_mask, "_actual_total"]

    y_test_total = feat_df.loc[test_mask, "_actual_total"]

    vegas_total_test = feat_df.loc[test_mask, "vegas_total"].values  # baseline



    total_models, total_metrics = train_game_model(

        X_train, y_train_total, X_test, y_test_total,

        model_name="total",

        vegas_test=vegas_total_test,

    )



    # 5. Save models

    joblib.dump(spread_models, MODELS_DIR / "game_spread_model.pkl")

    joblib.dump(total_models, MODELS_DIR / "game_total_model.pkl")

    joblib.dump(GAME_FEATURE_COLUMNS, MODELS_DIR / "game_features.pkl")



    # Save feature importances

    spread_fi = get_feature_importance(spread_models, GAME_FEATURE_COLUMNS)

    total_fi = get_feature_importance(total_models, GAME_FEATURE_COLUMNS)

    spread_fi.to_csv(MODELS_DIR / "spread_feature_importance.csv", index=False)

    total_fi.to_csv(MODELS_DIR / "total_feature_importance.csv", index=False)



    # 6. Print results

    logger.info("\n" + "=" * 60)

    logger.info("TRAINING RESULTS")

    logger.info("=" * 60)



    for name, metrics in [("SPREAD", spread_metrics), ("TOTAL", total_metrics)]:

        logger.info(f"\n  {name} Model:")

        logger.info(f"    Train games:   {metrics['samples_train']:,}")

        logger.info(f"    Test games:    {metrics['samples_test']:,}")

        logger.info(f"    MAE:           {metrics['mae']:.3f}")

        logger.info(f"    RMSE:          {metrics['rmse']:.3f}")

        logger.info(f"    R²:            {metrics['r2']:.4f}")

        logger.info(f"    80% Coverage:  {metrics['coverage_80pct']:.1f}%")

        logger.info(f"    Interval Width:{metrics['avg_width']:.2f}")

        if metrics["vegas_mae"] is not None:

            logger.info(f"    Vegas MAE:     {metrics['vegas_mae']:.3f}")

            imp = metrics["improvement_vs_vegas"]

            direction = "BETTER" if imp > 0 else "WORSE"

            logger.info(f"    vs Vegas:      {abs(imp):.3f} pts {direction}")

        if metrics["cv_mae"]:

            logger.info(f"    CV MAE:        {metrics['cv_mae']:.3f} ± {metrics['cv_std']:.3f}")



    logger.info(f"\n  Top 10 Spread Features:")

    for _, r in spread_fi.head(10).iterrows():

        logger.info(f"    {r['feature']:<30s} {r['importance']:.4f}")



    logger.info(f"\n  Top 10 Total Features:")

    for _, r in total_fi.head(10).iterrows():

        logger.info(f"    {r['feature']:<30s} {r['importance']:.4f}")



    # Save metrics to JSON for later reference

    results = {

        "spread": spread_metrics,

        "total": total_metrics,

        "training_config": {

            "min_season": min_season,

            "max_season": max_season,

            "test_seasons": list(test_seasons),

            "n_features": len(GAME_FEATURE_COLUMNS),

            "xgb_params": XGBOOST_PARAMS,

        }

    }

    with open(MODELS_DIR / "game_model_metrics.json", "w") as f:

        json.dump(results, f, indent=2, default=str)



    logger.info(f"\nModels saved to {MODELS_DIR}/")

    return spread_models, total_models, results





# ---------------------------------------------------------------------------

# Backtest

# ---------------------------------------------------------------------------



def run_backtest(seasons: tuple = (2024, 2025)):

    """

    Run backtest on held-out seasons. Evaluates ATS and O/U betting performance.

    """

    logger.info("=" * 60)

    logger.info(f"Game Prediction Backtest — Seasons {seasons}")

    logger.info("=" * 60)



    # Load models

    spread_models = joblib.load(MODELS_DIR / "game_spread_model.pkl")

    total_models = joblib.load(MODELS_DIR / "game_total_model.pkl")



    # Load data & features

    df = load_game_data()

    df = df[df["season"].isin(seasons)]



    # Need full data to build team histories

    full_df = load_game_data()

    team_histories = build_team_histories(full_df)

    feat_df = build_game_features(df, team_histories)



    feat_X = feat_df[GAME_FEATURE_COLUMNS].copy()

    valid = feat_X.notna().all(axis=1)

    feat_df = feat_df[valid].copy()

    feat_X = feat_X[valid].copy()



    if feat_df.empty:

        logger.error("No valid games for backtest")

        return



    # Predictions

    spread_pred = spread_models["q50"].predict(feat_X)

    spread_q10 = spread_models["q10"].predict(feat_X)

    spread_q90 = spread_models["q90"].predict(feat_X)



    total_pred = total_models["q50"].predict(feat_X)

    total_q10 = total_models["q10"].predict(feat_X)

    total_q90 = total_models["q90"].predict(feat_X)



    feat_df = feat_df.copy()

    feat_df["pred_margin"] = spread_pred

    feat_df["pred_margin_q10"] = spread_q10

    feat_df["pred_margin_q90"] = spread_q90

    feat_df["pred_total"] = total_pred

    feat_df["pred_total_q10"] = total_q10

    feat_df["pred_total_q90"] = total_q90



    # ATS analysis

    feat_df["model_edge_spread"] = feat_df["pred_margin"] - feat_df["vegas_spread"]

    feat_df["bet_home_ats"] = feat_df["model_edge_spread"] > 0  # model says home covers

    feat_df["home_covered"] = feat_df["_actual_margin"] > feat_df["vegas_spread"]



    # O/U analysis

    feat_df["model_edge_total"] = feat_df["pred_total"] - feat_df["vegas_total"]

    feat_df["bet_over"] = feat_df["model_edge_total"] > 0  # model says over

    feat_df["went_over"] = feat_df["_actual_total"] > feat_df["vegas_total"]



    # Overall ATS

    ats_correct = (feat_df["bet_home_ats"] == feat_df["home_covered"]).mean()

    ou_correct = (feat_df["bet_over"] == feat_df["went_over"]).mean()



    logger.info(f"\n  Overall ATS accuracy: {ats_correct:.1%} ({len(feat_df)} games)")

    logger.info(f"  Overall O/U accuracy: {ou_correct:.1%} ({len(feat_df)} games)")



    # Edge-filtered (only bet when model disagrees with Vegas by > threshold)

    for threshold in [1.0, 2.0, 3.0, 4.0, 5.0]:

        # ATS

        ats_mask = feat_df["model_edge_spread"].abs() > threshold

        if ats_mask.sum() >= 10:

            ats_filt = feat_df[ats_mask]

            ats_acc = (ats_filt["bet_home_ats"] == ats_filt["home_covered"]).mean()

            logger.info(f"  ATS edge>{threshold:.0f}: {ats_acc:.1%} ({ats_mask.sum()} games)")



        # O/U

        ou_mask = feat_df["model_edge_total"].abs() > threshold

        if ou_mask.sum() >= 10:

            ou_filt = feat_df[ou_mask]

            ou_acc = (ou_filt["bet_over"] == ou_filt["went_over"]).mean()

            logger.info(f"  O/U edge>{threshold:.0f}: {ou_acc:.1%} ({ou_mask.sum()} games)")



    # Save detailed backtest results

    backtest_path = MODELS_DIR / "game_backtest_results.csv"

    feat_df.to_csv(backtest_path, index=False)

    logger.info(f"\n  Detailed results saved to {backtest_path}")



    return feat_df





# ---------------------------------------------------------------------------

# Predict a single game (for live use)

# ---------------------------------------------------------------------------



@dataclass

class GamePrediction:

    """Prediction output for a single game."""

    home: str

    away: str

    pred_margin: float

    pred_margin_q10: float

    pred_margin_q90: float

    pred_total: float

    pred_total_q10: float

    pred_total_q90: float

    vegas_spread: float

    vegas_total: float

    spread_edge: float      # pred_margin - vegas_spread (positive = home cover value)

    total_edge: float        # pred_total - vegas_total  (positive = over value)

    ats_pick: str            # "HOME ATS" or "AWAY ATS"

    ou_pick: str             # "OVER" or "UNDER"

    confidence_spread: str   # Low / Medium / High

    confidence_total: str





_cached_team_histories: Optional[dict] = None

_cached_spread_models: Optional[dict] = None

_cached_total_models: Optional[dict] = None





def get_game_models() -> Tuple[dict, dict]:

    """Load and cache spread and total XGBoost models."""

    global _cached_spread_models, _cached_total_models

    if _cached_spread_models is None or _cached_total_models is None:

        _cached_spread_models = joblib.load(MODELS_DIR / "game_spread_model.pkl")

        _cached_total_models = joblib.load(MODELS_DIR / "game_total_model.pkl")
        for _d in (_cached_spread_models, _cached_total_models):
            if isinstance(_d, dict):
                for _m in _d.values():
                    if hasattr(_m, "set_params"):
                        try:
                            _m.set_params(device="cpu")
                            _m.get_booster().set_param({"device": "cpu"})
                        except Exception:
                            pass

    return _cached_spread_models, _cached_total_models





def get_cached_team_histories() -> dict:

    """Load and cache rolling team histories for instant inference."""

    global _cached_team_histories

    if _cached_team_histories is None:

        full_df = load_game_data()

        _cached_team_histories = build_team_histories(full_df)

    return _cached_team_histories





def predict_game(

    home: str, away: str,

    vegas_spread: float, vegas_total: float,

    moneyline_home: float = None,

    moneyline_away: float = None,

) -> Optional[GamePrediction]:

    """

    Predict spread and total for a single upcoming game.

    Requires trained models and historical data context.

    """

    # Load models

    try:

        spread_models, total_models = get_game_models()

    except FileNotFoundError:

        logger.error("Models not found. Run training first: python game_prediction.py train")

        return None



    # Normalize team names

    home_csv = STD_TO_CSV.get(home.upper(), home.lower())

    away_csv = STD_TO_CSV.get(away.upper(), away.lower())



    # Load data and build rolling context (cached in memory)

    team_histories = get_cached_team_histories()



    home_hist = team_histories.get(home_csv)

    away_hist = team_histories.get(away_csv)



    if home_hist is None or away_hist is None:

        logger.error(f"Team history not found: home={home_csv}, away={away_csv}")

        return None



    # Compute unshifted post-game rolling windows including the team's most recent completed game

    hh = home_hist.iloc[-1]

    ah = away_hist.iloc[-1]



    h_ppg_10 = float(home_hist["pts_for"].tail(10).mean())

    h_pag_10 = float(home_hist["pts_against"].tail(10).mean())

    h_ppg_20 = float(home_hist["pts_for"].tail(20).mean())

    h_pag_20 = float(home_hist["pts_against"].tail(20).mean())

    h_mar_10 = float(home_hist["margin"].tail(10).mean())

    h_mar_20 = float(home_hist["margin"].tail(20).mean())

    h_pace_10 = float(home_hist["game_total"].tail(10).mean())

    h_1h_10 = float(home_hist["_1h_pct"].tail(10).mean())

    h_q4_10 = float(home_hist["q4"].tail(10).mean())



    a_ppg_10 = float(away_hist["pts_for"].tail(10).mean())

    a_pag_10 = float(away_hist["pts_against"].tail(10).mean())

    a_ppg_20 = float(away_hist["pts_for"].tail(20).mean())

    a_pag_20 = float(away_hist["pts_against"].tail(20).mean())

    a_mar_10 = float(away_hist["margin"].tail(10).mean())

    a_mar_20 = float(away_hist["margin"].tail(20).mean())

    a_pace_10 = float(away_hist["game_total"].tail(10).mean())

    a_1h_10 = float(away_hist["_1h_pct"].tail(10).mean())

    a_q4_10 = float(away_hist["q4"].tail(10).mean())



    # Live 4-Gate LSTM sequential momentum features

    h_lstm_m, h_lstm_t = 0.0, 0.0

    a_lstm_m, a_lstm_t = 0.0, 0.0

    lstm_scout = _get_game_lstm_scout()

    if lstm_scout and "margin_lstm" in lstm_scout and "total_lstm" in lstm_scout:

        try:

            Xh, Ch = _build_team_seq_tensors(home_hist, include_last=True)

            Xa, Ca = _build_team_seq_tensors(away_hist, include_last=True)

            h_lstm_m = float(lstm_scout["margin_lstm"].predict(Xh, Ch)[0][0])

            h_lstm_t = float(lstm_scout["total_lstm"].predict(Xh, Ch)[0][0])

            a_lstm_m = float(lstm_scout["margin_lstm"].predict(Xa, Ca)[0][0])

            a_lstm_t = float(lstm_scout["total_lstm"].predict(Xa, Ca)[0][0])

        except Exception:

            pass



    impl_home_prob = _moneyline_to_prob(moneyline_home) if moneyline_home else 0.5



    # H2H

    h2h_games = home_hist[home_hist["opponent"] == away_csv].tail(5)

    h2h_margin = h2h_games["margin"].mean() if len(h2h_games) >= 2 else 0.0



    feat = pd.DataFrame([{

        "vegas_spread": vegas_spread,

        "vegas_total": vegas_total,

        "implied_home_prob": impl_home_prob,

        "home_ppg_l10": h_ppg_10,

        "home_pag_l10": h_pag_10,

        "home_ppg_l20": h_ppg_20,

        "home_pag_l20": h_pag_20,

        "home_margin_l10": h_mar_10,

        "home_margin_l20": h_mar_20,

        "away_ppg_l10": a_ppg_10,

        "away_pag_l10": a_pag_10,

        "away_ppg_l20": a_ppg_20,

        "away_pag_l20": a_pag_20,

        "away_margin_l10": a_mar_10,

        "away_margin_l20": a_mar_20,

        "home_pace_l10": h_pace_10,

        "away_pace_l10": a_pace_10,

        "home_rest_days": hh.get("rest_days", 2),

        "away_rest_days": ah.get("rest_days", 2),

        "home_b2b": hh.get("b2b", 0),

        "away_b2b": ah.get("b2b", 0),

        "home_win_streak": hh.get("win_streak", 0),

        "away_win_streak": ah.get("win_streak", 0),

        "home_ats_streak": hh.get("ats_streak", 0),

        "away_ats_streak": ah.get("ats_streak", 0),

        "home_ou_streak": hh.get("ou_streak", 0),

        "away_ou_streak": ah.get("ou_streak", 0),

        "home_1h_pct_l10": h_1h_10,

        "away_1h_pct_l10": a_1h_10,

        "home_q4_ppg_l10": h_q4_10,

        "away_q4_ppg_l10": a_q4_10,

        "h2h_home_margin_l5": h2h_margin,

        "season_game_number": max(hh.get("season_game_num", 40), ah.get("season_game_num", 40)),

        "is_playoffs": 0,

        # Derived

        "rest_diff": (hh.get("rest_days", 2) or 2) - (ah.get("rest_days", 2) or 2),

        "pace_interaction": (h_pace_10 + a_pace_10) / 2.0,

        "form_diff": h_mar_10 - a_mar_10,

        "offense_vs_defense_home": h_ppg_10 - a_pag_10,

        "offense_vs_defense_away": a_ppg_10 - h_pag_10,

        "implied_total_from_form": (h_ppg_10 + a_ppg_10) - vegas_total,

        "implied_margin_from_form": (h_mar_10 - a_mar_10) - vegas_spread,

        "b2b_mismatch": (hh.get("b2b", 0) or 0) - (ah.get("b2b", 0) or 0),

        "home_lstm_margin_mom": h_lstm_m,

        "away_lstm_margin_mom": a_lstm_m,

        "lstm_margin_diff": h_lstm_m - a_lstm_m,

        "home_lstm_total_mom": h_lstm_t,

        "away_lstm_total_mom": a_lstm_t,

        "lstm_total_sum": h_lstm_t + a_lstm_t,

    }])



    expected_cols = list(getattr(spread_models["q50"], "feature_names_in_", GAME_FEATURE_COLUMNS))

    for col in expected_cols:

        if col not in feat.columns:

            feat[col] = 0.0

    feat = feat[expected_cols]



    # Predict

    pred_margin = float(spread_models["q50"].predict(feat)[0])

    pred_margin_q10 = float(spread_models["q10"].predict(feat)[0])

    pred_margin_q90 = float(spread_models["q90"].predict(feat)[0])

    pred_total = float(total_models["q50"].predict(feat)[0])

    pred_total_q10 = float(total_models["q10"].predict(feat)[0])

    pred_total_q90 = float(total_models["q90"].predict(feat)[0])



    spread_edge = pred_margin - vegas_spread

    total_edge = pred_total - vegas_total



    # Confidence levels based on edge magnitude

    def _conf(edge, width):

        ratio = abs(edge) / max(width, 1)

        if ratio > 0.25:

            return "High"

        elif ratio > 0.10:

            return "Medium"

        return "Low"



    spread_width = pred_margin_q90 - pred_margin_q10

    total_width = pred_total_q90 - pred_total_q10



    return GamePrediction(

        home=home.upper(),

        away=away.upper(),

        pred_margin=round(pred_margin, 1),

        pred_margin_q10=round(pred_margin_q10, 1),

        pred_margin_q90=round(pred_margin_q90, 1),

        pred_total=round(pred_total, 1),

        pred_total_q10=round(pred_total_q10, 1),

        pred_total_q90=round(pred_total_q90, 1),

        vegas_spread=vegas_spread,

        vegas_total=vegas_total,

        spread_edge=round(spread_edge, 1),

        total_edge=round(total_edge, 1),

        ats_pick="HOME ATS" if spread_edge > 0 else "AWAY ATS",

        ou_pick="OVER" if total_edge > 0 else "UNDER",

        confidence_spread=_conf(spread_edge, spread_width),

        confidence_total=_conf(total_edge, total_width),

    )





def print_prediction(pred: GamePrediction):

    """Pretty-print a game prediction."""

    print(f"\n{'=' * 55}")

    print(f"  {pred.away} @ {pred.home}")

    print(f"{'=' * 55}")

    print(f"  {'SPREAD':30s}")

    print(f"    Vegas Spread:       {pred.vegas_spread:+.1f}")

    print(f"    Model Margin:       {pred.pred_margin:+.1f}  ({pred.pred_margin_q10:+.1f} to {pred.pred_margin_q90:+.1f})")

    print(f"    Edge:               {pred.spread_edge:+.1f} pts -> {pred.ats_pick}  [{pred.confidence_spread}]")

    print(f"  {'TOTAL':30s}")

    print(f"    Vegas Total:        {pred.vegas_total:.1f}")

    print(f"    Model Total:        {pred.pred_total:.1f}  ({pred.pred_total_q10:.1f} to {pred.pred_total_q90:.1f})")

    print(f"    Edge:               {pred.total_edge:+.1f} pts -> {pred.ou_pick}  [{pred.confidence_total}]")

    print(f"{'=' * 55}")





# ---------------------------------------------------------------------------

# CLI

# ---------------------------------------------------------------------------



def main():

    parser = argparse.ArgumentParser(description="NBA Game Spread & Total Prediction")

    sub = parser.add_subparsers(dest="command")



    # Train

    train_p = sub.add_parser("train", help="Train spread & total models")

    train_p.add_argument("--min-season", type=int, default=2010)

    train_p.add_argument("--max-season", type=int, default=2025)

    train_p.add_argument("--test-seasons", type=int, nargs="+", default=[2024, 2025])



    # Backtest

    bt_p = sub.add_parser("backtest", help="Backtest on held-out seasons")

    bt_p.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025])



    # Predict

    pred_p = sub.add_parser("predict", help="Predict a single game")

    pred_p.add_argument("--home", required=True, help="Home team (e.g. LAL)")

    pred_p.add_argument("--away", required=True, help="Away team (e.g. BOS)")

    pred_p.add_argument("--spread", type=float, required=True, help="Vegas spread (positive = home favored)")

    pred_p.add_argument("--total", type=float, required=True, help="Vegas O/U total")

    pred_p.add_argument("--ml-home", type=float, default=None, help="Home moneyline (e.g. -150)")

    pred_p.add_argument("--ml-away", type=float, default=None, help="Away moneyline (e.g. +130)")



    args = parser.parse_args()



    if args.command == "train":

        run_training(

            min_season=args.min_season,

            max_season=args.max_season,

            test_seasons=tuple(args.test_seasons),

        )



    elif args.command == "backtest":

        run_backtest(seasons=tuple(args.seasons))



    elif args.command == "predict":

        pred = predict_game(

            home=args.home,

            away=args.away,

            vegas_spread=args.spread,

            vegas_total=args.total,

            moneyline_home=args.ml_home,

            moneyline_away=args.ml_away,

        )

        if pred:

            print_prediction(pred)



    else:

        parser.print_help()





if __name__ == "__main__":

    main()

