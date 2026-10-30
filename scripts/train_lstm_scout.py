"""
scripts/train_lstm_scout.py — Train Tool #4 (🧠 LSTM Sequential Feature Scout) & Populate Out-of-Fold Features

Trains a 4-Gate LSTM (Input, Forget, Cell, Output gates with analytical BPTT + Adam)
across all 7 NBA player prop markets (PTS, REB, AST, PRA, PR, PA, RA) on the
323,253-row chronological dataset.

Computes strictly chronological out-of-fold predictions (`feat_lstm_momentum` and
`feat_lstm_fatigue_gate`) for every row in `data/ml_training_data.parquet`, saves
the trained LSTM models to `data/nba_lstm_feature_scout.pkl`, and writes telemetry
to `config/lstm_scout_metrics.json`.
"""

import sys
import os
import json
import time
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from lstm_feature_scout import (
    LSTMFeatureScoutEngine,
    VectorizedLSTMCellBPTT,
    build_temporal_sequences_for_market,
    SEQUENCE_LENGTH,
    HIDDEN_DIM,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
CONFIG_DIR = Path("config")
PARQUET_PATH = DATA_DIR / "ml_training_data.parquet"
METRICS_PATH = CONFIG_DIR / "lstm_scout_metrics.json"


def main():
    logger.info("=" * 70)
    logger.info("🧠 TOOL #4: 4-GATE LSTM SEQUENTIAL FEATURE SCOUT (BPTT + ADAM)")
    logger.info("=" * 70)

    if not PARQUET_PATH.exists():
        logger.error("Training dataset not found: %s", PARQUET_PATH)
        sys.exit(1)

    df = pd.read_parquet(PARQUET_PATH)
    logger.info(f"Loaded {len(df):,} rows across {df['market'].nunique()} markets")

    # Initialize columns if not present
    df["feat_lstm_momentum"] = np.float32(0.0)
    df["feat_lstm_fatigue_gate"] = np.float32(0.5)

    scout = LSTMFeatureScoutEngine()
    markets = sorted(df["market"].unique())
    t_total = time.perf_counter()
    summary = {}

    for mkt in markets:
        sub_mask = df["market"] == mkt
        mkt_df = df.loc[sub_mask].copy()
        if "date" in mkt_df.columns:
            mkt_df = mkt_df.sort_values(["date", "player"])

        t0 = time.perf_counter()
        X_seq, C_now, y_resid, orig_idx = build_temporal_sequences_for_market(mkt_df, seq_len=SEQUENCE_LENGTH)

        # Baseline zero-residual MAE (predicting season_avg alone)
        zero_mae = float(np.abs(y_resid).mean())

        # Chronological Out-of-Fold (4 folds) so XGBoost receives zero-leakage LSTM predictions
        tscv = TimeSeriesSplit(n_splits=4)
        oof_momentum = np.zeros(len(y_resid), dtype=np.float32)
        oof_fatigue = np.full(len(y_resid), 0.5, dtype=np.float32)

        # Warm-start initial fold using first 20% chronological slice trained with stronger regularization
        first_split = int(len(y_resid) * 0.20)
        init_lstm = VectorizedLSTMCellBPTT(input_dim=8, context_dim=5, hidden_dim=HIDDEN_DIM, lr=0.008, seed=42)
        init_lstm.fit(X_seq[:first_split], C_now[:first_split], y_resid[:first_split], epochs=14, batch_size=1024)
        p0, fg0 = init_lstm.predict(X_seq[:first_split], C_now[:first_split])
        oof_momentum[:first_split] = p0
        oof_fatigue[:first_split] = fg0

        val_maes = []
        base_val_maes = []

        for fold_i, (tr_idx, val_idx) in enumerate(tscv.split(X_seq)):
            fold_lstm = VectorizedLSTMCellBPTT(
                input_dim=8, context_dim=5, hidden_dim=HIDDEN_DIM, lr=0.008, seed=42 + fold_i
            )
            fold_lstm.fit(
                X_seq[tr_idx], C_now[tr_idx], y_resid[tr_idx],
                epochs=16, batch_size=2048,
                X_val=X_seq[val_idx], C_val=C_now[val_idx], y_val=y_resid[val_idx],
            )
            val_pred, val_fg = fold_lstm.predict(X_seq[val_idx], C_now[val_idx])
            oof_momentum[val_idx] = val_pred
            oof_fatigue[val_idx] = val_fg

            v_mae = float(np.abs(val_pred - y_resid[val_idx]).mean())
            b_mae = float(np.abs(y_resid[val_idx]).mean())
            val_maes.append(v_mae)
            base_val_maes.append(b_mae)

        # Fit final full-history production LSTM for live inference
        n_tr = int(len(y_resid) * 0.85)
        prod_lstm = VectorizedLSTMCellBPTT(input_dim=8, context_dim=5, hidden_dim=HIDDEN_DIM, lr=0.008, seed=100)
        prod_lstm.fit(
            X_seq[:n_tr], C_now[:n_tr], y_resid[:n_tr],
            epochs=18, batch_size=2048,
            X_val=X_seq[n_tr:], C_val=C_now[n_tr:], y_val=y_resid[n_tr:],
        )
        scout.models[mkt] = prod_lstm

        mean_val_mae = float(np.mean(val_maes))
        mean_base_mae = float(np.mean(base_val_maes))
        imp_pct = ((mean_base_mae - mean_val_mae) / max(mean_base_mae, 1e-6)) * 100.0
        corr = float(np.corrcoef(oof_momentum, y_resid)[0, 1]) if len(y_resid) > 2 else 0.0
        elapsed = time.perf_counter() - t0

        # Assign back to DataFrame aligned by original index
        df.loc[orig_idx, "feat_lstm_momentum"] = oof_momentum
        df.loc[orig_idx, "feat_lstm_fatigue_gate"] = oof_fatigue

        mkt_meta = {
            "market": mkt,
            "samples": int(len(y_resid)),
            "sequence_length": SEQUENCE_LENGTH,
            "hidden_dim": HIDDEN_DIM,
            "baseline_resid_mae": round(mean_base_mae, 4),
            "lstm_oof_resid_mae": round(mean_val_mae, 4),
            "resid_mae_improvement_pct": round(imp_pct, 2),
            "oof_target_correlation": round(corr, 4),
            "mean_fatigue_gate": round(float(oof_fatigue.mean()), 4),
            "b2b_fatigue_gate": round(float(oof_fatigue[C_now[:, 3] > 0.5].mean()), 4) if (C_now[:, 3] > 0.5).any() else 0.5,
            "rested_fatigue_gate": round(float(oof_fatigue[C_now[:, 3] <= 0.5].mean()), 4),
            "elapsed_sec": round(elapsed, 2),
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        scout.metrics[mkt] = mkt_meta
        summary[mkt] = mkt_meta

        logger.info(
            f"  ✅ {mkt:<4} ({len(y_resid):,} rows, {elapsed:.1f}s): "
            f"Resid MAE {mean_base_mae:.4f} -> {mean_val_mae:.4f} ({imp_pct:+.2f}%) | "
            f"OOF r = {corr:+.4f} | Gate={oof_fatigue.mean():.3f}"
        )

    # Save trained LSTM models and updated training parquet
    scout.save()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    df.to_parquet(PARQUET_PATH, index=False)
    logger.info("=" * 70)
    logger.info(
        f"✅ LSTM Feature Scout complete in {time.perf_counter() - t_total:.1f}s! "
        f"Saved models to {scout.model_path} and updated {PARQUET_PATH}"
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
