#!/usr/bin/env python3
"""
Game Model Quant Arsenal Training Pipeline (Tool #3 Optuna + CUDA & Tool #4 4-Gate LSTM)
========================================================================================
1. Trains the 4-Gate LSTM Sequential Feature Scout on 8-game team rolling histories
   for Spread Margin Residuals and Total Residuals (Chronological OOF + Full Fit).
2. Tunes Game Spread and Game Total Multi-Quantile (q10, q50, q90) XGBoost models
   using Optuna Bayesian TPE + MedianPruner on NVIDIA CUDA (`QuantileDMatrix`).
3. Retrains the production Game Spread and Game Total models with the 43 features
   (including the 6 LSTM sequential features) and Optuna-tuned hyperparameters.
"""
import json
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import game_prediction as gp
from hardware_utils import get_gpu_info, get_xgboost_hardware_params
from lstm_feature_scout import VectorizedLSTMCellBPTT


def train_game_lstm_scout(df: pd.DataFrame) -> dict:
    """
    Train Tool #4: 4-Gate LSTM Sequential Feature Scout for NBA Teams.
    Predicts sequential momentum residuals for Margin (ATS) and Game Total (O/U).
    """
    print("=" * 72)
    print("🧠 STEP 1: TRAINING 4-GATE LSTM SEQUENTIAL SCOUT FOR GAME MODEL")
    print("=" * 72)
    t0 = time.time()

    # Build raw per-team histories (before LSTM is loaded)
    gp._cached_game_lstm_scout = {}
    team_histories = gp.build_team_histories(df)

    seq_list = []
    ctx_list = []
    y_margin_list = []
    y_total_list = []
    dates_list = []

    for team, g in team_histories.items():
        X_seq, C_now = gp._build_team_seq_tensors(g, include_last=False)
        # Target 1: Margin residual vs Vegas spread (actual margin - expected margin)
        # Note: in build_team_histories, g["spread"] is signed (+ means team favored by X)
        y_m = (
            pd.to_numeric(g["margin"], errors="coerce").fillna(0.0).values
            - pd.to_numeric(g["spread"], errors="coerce").fillna(0.0).values
        ) / 12.0
        # Target 2: Total residual vs Vegas total line
        y_t = (
            pd.to_numeric(g["game_total"], errors="coerce").fillna(220.0).values
            - pd.to_numeric(g["total_line"], errors="coerce").fillna(220.0).values
        ) / 15.0

        # Skip cold-start first 8 games of each team history
        if len(g) > 8:
            seq_list.append(X_seq[8:])
            ctx_list.append(C_now[8:])
            y_margin_list.append(y_m[8:])
            y_total_list.append(y_t[8:])
            dates_list.append(pd.to_datetime(g["date"]).values[8:])

    X_all = np.concatenate(seq_list, axis=0).astype(np.float32)
    C_all = np.concatenate(ctx_list, axis=0).astype(np.float32)
    y_m_all = np.clip(np.concatenate(y_margin_list, axis=0).astype(np.float32), -3.0, 3.0)
    y_t_all = np.clip(np.concatenate(y_total_list, axis=0).astype(np.float32), -3.0, 3.0)
    dates_all = np.concatenate(dates_list, axis=0)

    # Sort strictly chronologically across all team-games
    order = np.argsort(dates_all)
    X_all = X_all[order]
    C_all = C_all[order]
    y_m_all = y_m_all[order]
    y_t_all = y_t_all[order]

    print(f"Built {len(X_all):,} chronological team-game sequences (seq_len=8, input_dim=6, ctx_dim=4)")

    tscv = TimeSeriesSplit(n_splits=4)
    oof_m = np.zeros(len(X_all), dtype=np.float32)
    oof_t = np.zeros(len(X_all), dtype=np.float32)
    oof_mask = np.zeros(len(X_all), dtype=bool)

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_all)):
        m_cell = VectorizedLSTMCellBPTT(input_dim=6, context_dim=4, hidden_dim=20, lr=0.008, seed=42 + fold)
        m_cell.fit(
            X_all[tr_idx], C_all[tr_idx], y_m_all[tr_idx],
            X_val=X_all[val_idx], C_val=C_all[val_idx], y_val=y_m_all[val_idx],
            epochs=25, batch_size=512,
        )
        p_m, _ = m_cell.predict(X_all[val_idx], C_all[val_idx])
        oof_m[val_idx] = p_m

        t_cell = VectorizedLSTMCellBPTT(input_dim=6, context_dim=4, hidden_dim=20, lr=0.008, seed=142 + fold)
        t_cell.fit(
            X_all[tr_idx], C_all[tr_idx], y_t_all[tr_idx],
            X_val=X_all[val_idx], C_val=C_all[val_idx], y_val=y_t_all[val_idx],
            epochs=25, batch_size=512,
        )
        p_t, _ = t_cell.predict(X_all[val_idx], C_all[val_idx])
        oof_t[val_idx] = p_t
        oof_mask[val_idx] = True

    # Fit production LSTM cells on full dataset (90/10 split for early stopping)
    split_90 = int(len(X_all) * 0.90)
    prod_m_cell = VectorizedLSTMCellBPTT(input_dim=6, context_dim=4, hidden_dim=20, lr=0.008, seed=777)
    prod_m_cell.fit(
        X_all[:split_90], C_all[:split_90], y_m_all[:split_90],
        X_val=X_all[split_90:], C_val=C_all[split_90:], y_val=y_m_all[split_90:],
        epochs=30, batch_size=512,
    )
    prod_t_cell = VectorizedLSTMCellBPTT(input_dim=6, context_dim=4, hidden_dim=20, lr=0.008, seed=888)
    prod_t_cell.fit(
        X_all[:split_90], C_all[:split_90], y_t_all[:split_90],
        X_val=X_all[split_90:], C_val=C_all[split_90:], y_val=y_t_all[split_90:],
        epochs=30, batch_size=512,
    )

    m_corr = float(np.corrcoef(oof_m[oof_mask], y_m_all[oof_mask])[0, 1])
    t_corr = float(np.corrcoef(oof_t[oof_mask], y_t_all[oof_mask])[0, 1])
    m_mae = float(np.mean(np.abs(oof_m[oof_mask] - y_m_all[oof_mask])) * 12.0)
    t_mae = float(np.mean(np.abs(oof_t[oof_mask] - y_t_all[oof_mask])) * 15.0)

    payload = {
        "margin_lstm": prod_m_cell,
        "total_lstm": prod_t_cell,
    }
    gp.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    scout_path = gp.MODELS_DIR / "game_lstm_scout.pkl"
    joblib.dump(payload, scout_path)

    metrics = {
        "architecture": "4-Gate LSTM with Exact BPTT (Team 8-Game Lookback)",
        "samples": int(len(X_all)),
        "seq_len": 8,
        "hidden_dim": 20,
        "margin_oof_corr": round(m_corr, 4),
        "margin_oof_mae_pts": round(m_mae, 3),
        "total_oof_corr": round(t_corr, 4),
        "total_oof_mae_pts": round(t_mae, 3),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    metrics_path = ROOT / "config" / "game_lstm_scout_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Reload into game_prediction singleton
    gp._cached_game_lstm_scout = None
    gp._get_game_lstm_scout()

    print(f"  [MARGIN LSTM] OOF r = {m_corr:+.4f} | Residual MAE = {m_mae:.3f} pts")
    print(f"  [TOTAL  LSTM] OOF r = {t_corr:+.4f} | Residual MAE = {t_mae:.3f} pts")
    print(f"  Saved Game LSTM Scout -> {scout_path} ({metrics['elapsed_sec']:.1f}s)\n")
    return metrics


def multi_quantile_pinball_loss(y_true: np.ndarray, preds_2d: np.ndarray, alphas=(0.10, 0.50, 0.90)) -> float:
    total = 0.0
    for idx, alpha in enumerate(alphas):
        diff = y_true - preds_2d[:, idx]
        loss = np.where(diff >= 0.0, alpha * diff, (alpha - 1.0) * diff)
        total += float(np.mean(loss))
    return total / len(alphas)


def tune_game_models_optuna_cuda(df: pd.DataFrame, n_trials: int = 25) -> dict:
    """
    Train Tool #3: Optuna Bayesian TPE + CUDA Multi-Quantile Hyperparameter Tuning
    for Game Spread and Game Total models.
    """
    gpu_info = get_gpu_info()
    hw_str = f"{gpu_info.get('name', 'CUDA')} ({gpu_info.get('device', 'cuda')})"
    print("=" * 72)
    print(f"⚡ STEP 2: OPTUNA BAYESIAN TPE + CUDA TUNING ({hw_str})")
    print("=" * 72)

    team_histories = gp.build_team_histories(df)
    feat_df = gp.build_game_features(df, team_histories)
    feat_X = feat_df[gp.GAME_FEATURE_COLUMNS].copy()
    valid_mask = feat_X.notna().all(axis=1)
    feat_df = feat_df[valid_mask].copy().reset_index(drop=True)
    feat_X = feat_X[valid_mask].copy().reset_index(drop=True)

    # Use training seasons (< 2024) for CV hyperparameter optimization so test seasons (2024, 2025) stay strictly held out
    train_df = feat_df[~feat_df["_season"].isin((2024, 2025))].reset_index(drop=True)
    X_tr_all = feat_X.loc[~feat_df["_season"].isin((2024, 2025))].reset_index(drop=True).to_numpy(dtype=np.float32)

    hw_params = get_xgboost_hardware_params({})
    quantiles = [0.10, 0.50, 0.90]
    tscv = TimeSeriesSplit(n_splits=3)

    results = {"_metadata": {
        "hardware": hw_str,
        "sampler": "TPESampler(multivariate=True, seed=42)",
        "pruner": "MedianPruner",
        "quantiles": quantiles,
        "n_features": len(gp.GAME_FEATURE_COLUMNS),
    }}

    for target_name, target_col in [("spread", "_actual_margin"), ("total", "_actual_total")]:
        t0 = time.time()
        y_all = train_df[target_col].to_numpy(dtype=np.float32)

        folds = []
        for tr_idx, val_idx in tscv.split(X_tr_all):
            dtrain = xgb.QuantileDMatrix(X_tr_all[tr_idx], y_all[tr_idx])
            dval = xgb.QuantileDMatrix(X_tr_all[val_idx], y_all[val_idx], ref=dtrain)
            folds.append((dtrain, dval, y_all[val_idx]))

        # Baseline evaluation
        base_params = {
            "objective": "reg:quantileerror",
            "quantile_alpha": quantiles,
            "eval_metric": "quantile",
            "learning_rate": 0.04,
            "max_depth": 5,
            "min_child_weight": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 1.0,
            "reg_lambda": 2.0,
            "gamma": 0.15,
            "seed": 42,
            "verbosity": 0,
            **hw_params,
        }
        base_losses = []
        for dtrain, dval, y_val in folds:
            bst = xgb.train(base_params, dtrain, num_boost_round=600, evals=[(dval, "val")], early_stopping_rounds=30, verbose_eval=False)
            preds = bst.predict(dval)
            if preds.ndim == 1:
                preds = np.column_stack([preds, preds, preds])
            base_losses.append(multi_quantile_pinball_loss(y_val, preds))
        baseline_loss = float(np.mean(base_losses))

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "reg:quantileerror",
                "quantile_alpha": quantiles,
                "eval_metric": "quantile",
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "min_child_weight": trial.suggest_int("min_child_weight", 5, 45),
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.07, log=True),
                "subsample": trial.suggest_float("subsample", 0.60, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 0.95),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 12.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 20.0, log=True),
                "gamma": trial.suggest_float("gamma", 0.01, 2.0, log=True),
                "seed": 42,
                "verbosity": 0,
                **hw_params,
            }
            n_est = trial.suggest_int("n_estimators", 350, 900, step=50)
            fold_losses = []
            for step_idx, (dtrain, dval, y_val) in enumerate(folds):
                bst = xgb.train(
                    params, dtrain, num_boost_round=n_est,
                    evals=[(dval, "val")], early_stopping_rounds=30, verbose_eval=False,
                )
                preds = bst.predict(dval)
                if preds.ndim == 1:
                    preds = np.column_stack([preds, preds, preds])
                fl = multi_quantile_pinball_loss(y_val, preds)
                fold_losses.append(fl)
                trial.report(float(np.mean(fold_losses)), step=step_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return float(np.mean(fold_losses))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_loss = float(study.best_value)
        imp_pct = (baseline_loss - best_loss) / baseline_loss * 100.0
        best_p = study.best_params
        best_clean = {
            "n_estimators": int(best_p["n_estimators"]),
            "max_depth": int(best_p["max_depth"]),
            "min_child_weight": int(best_p["min_child_weight"]),
            "learning_rate": round(float(best_p["learning_rate"]), 5),
            "subsample": round(float(best_p["subsample"]), 4),
            "colsample_bytree": round(float(best_p["colsample_bytree"]), 4),
            "reg_alpha": round(float(best_p["reg_alpha"]), 4),
            "reg_lambda": round(float(best_p["reg_lambda"]), 4),
            "gamma": round(float(best_p["gamma"]), 4),
        }
        results[target_name] = best_clean
        results["_metadata"][target_name] = {
            "baseline_pinball_loss": round(baseline_loss, 5),
            "best_pinball_loss": round(best_loss, 5),
            "improvement_pct": round(imp_pct, 2),
            "elapsed_sec": round(time.time() - t0, 2),
        }
        print(
            f"  [{target_name.upper():6s}] Pinball Loss: {baseline_loss:.4f} -> {best_loss:.4f} "
            f"({imp_pct:+.2f}%) | depth={best_clean['max_depth']}, lr={best_clean['learning_rate']:.4f}, "
            f"mcw={best_clean['min_child_weight']} ({time.time() - t0:.1f}s)"
        )

    hp_path = ROOT / "config" / "game_model_hyperparams.json"
    hp_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  Saved Game Optuna Hyperparameters -> {hp_path}\n")
    return results


def main():
    df = gp.load_game_data()
    df = df[(df["season"] >= 2010) & (df["season"] <= 2025)]
    df = df[df["spread"].notna()].reset_index(drop=True)

    # 1. Train Game 4-Gate LSTM Scout
    train_game_lstm_scout(df)

    # 2. Tune Game Spread & Total Models with Optuna + CUDA
    tune_game_models_optuna_cuda(df, n_trials=25)

    # 3. Run full end-to-end training & out-of-sample evaluation (2024-2025)
    print("=" * 72)
    print("🏀 STEP 3: RETRAINING PRODUCTION GAME QUANTILE MODELS (43 FEATURES)")
    print("=" * 72)
    _, _, all_metrics = gp.run_training()
    print("\n--- SPREAD METRICS (2024-2025 Held-Out) ---")
    print(json.dumps(all_metrics["spread"], indent=2))
    print("\n--- TOTAL METRICS (2024-2025 Held-Out) ---")
    print(json.dumps(all_metrics["total"], indent=2))


if __name__ == "__main__":
    main()
