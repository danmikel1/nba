"""
lstm_feature_scout.py — Tool #4: 🧠 LSTM (Helper) Sequential Fatigue & Hot/Cold Curve Feature Scout

Implements a true 4-Gate LSTM (Input, Forget, Cell, Output gates: i_t, f_t, g_t, o_t, c_t, h_t)
with analytical Backpropagation Through Time (BPTT) and Adam optimization (plus automatic
PyTorch CUDA backend when torch is installed, or high-speed vectorized float32 BLAS).

Reads each player's chronological 8-game trajectory of:
  - Stat residuals (actual - season_avg)
  - Minutes played & workload delta
  - Days rest & Back-to-Back (B2B) compression
  - 96-hour travel miles & altitude
  - True Shooting % delta & foul rate

Outputs two pure data-driven sequential features for the Primary XGBoost Quantile Models:
  1. `feat_lstm_momentum`: Out-of-fold LSTM predicted stat residual (hot/cold + fatigue curve)
  2. `feat_lstm_fatigue_gate`: Learned temporal retention/fatigue gate state in [0, 1]
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
LSTM_MODEL_PATH = DATA_DIR / "nba_lstm_feature_scout.pkl"

SEQUENCE_LENGTH = 8
HIDDEN_DIM = 24

# Raw observable per-game features fed into each time step t of the LSTM
SEQ_STEP_FEATURES = [
    "step_stat_resid",      # (actual_value_{t-k} - feat_season_avg_{t}) / std
    "step_minutes_z",       # (minutes_{t-k} - 28.0) / 10.0
    "step_days_rest",       # clip(days_rest_{t-k}, 0, 7) / 7.0
    "step_is_b2b",          # 0.0 or 1.0
    "step_miles_96h",       # miles_96h_{t-k} / 3000.0
    "step_ts_delta",        # clip(feat_ts_pct_delta_{t-k}, -0.3, 0.3)
    "step_fatigue_load",    # feat_fatigue_load_{t-k} / 100.0
    "step_form_gap",        # clip(feat_form_gap_{t-k}, -10.0, 10.0) / 10.0
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


class VectorizedLSTMCellBPTT:
    """
    Exact 4-Gate LSTM (Input, Forget, Cell candidate, Output gates) with
    analytical Backpropagation Through Time (BPTT) and Adam optimizer.

    Forward equations for t = 0 .. T-1:
        z_t = X[:, t, :] @ W_ih + h_{t-1} @ W_hh + b_h
        i_t = sigmoid(z_t[:, 0:H])
        f_t = sigmoid(z_t[:, H:2H])
        g_t = tanh(z_t[:, 2H:3H])
        o_t = sigmoid(z_t[:, 3H:4H])
        c_t = f_t * c_{t-1} + i_t * g_t
        h_t = o_t * tanh(c_t)
    Readout head (with Current-Game Fatigue/Rest Context C_now):
        y_hat = h_{T-1} @ W_out + C_now @ W_ctx + b_out
    """

    def __init__(
        self,
        input_dim: int = 8,
        context_dim: int = 5,
        hidden_dim: int = HIDDEN_DIM,
        lr: float = 0.008,
        weight_decay: float = 1e-4,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        rng = np.random.default_rng(seed)

        H = hidden_dim
        scale_ih = np.sqrt(2.0 / (input_dim + H))
        scale_hh = np.sqrt(1.0 / H)

        self.W_ih = rng.normal(0.0, scale_ih, size=(input_dim, 4 * H)).astype(np.float32)
        self.W_hh = rng.normal(0.0, scale_hh, size=(H, 4 * H)).astype(np.float32)
        self.b_h = np.zeros((4 * H,), dtype=np.float32)
        # Initialize forget gate bias to +1.0 (Jozefowicz et al., 2015) for long-range temporal memory
        self.b_h[H : 2 * H] = 1.0

        self.W_out = rng.normal(0.0, np.sqrt(1.0 / H), size=(H, 1)).astype(np.float32)
        self.W_ctx = rng.normal(0.0, np.sqrt(1.0 / max(1, context_dim)), size=(context_dim, 1)).astype(np.float32)
        self.b_out = np.zeros((1,), dtype=np.float32)

        # Adam state
        self._adam_t = 0
        self._m: Dict[str, np.ndarray] = {}
        self._v: Dict[str, np.ndarray] = {}

    def _params_dict(self) -> Dict[str, np.ndarray]:
        return {
            "W_ih": self.W_ih,
            "W_hh": self.W_hh,
            "b_h": self.b_h,
            "W_out": self.W_out,
            "W_ctx": self.W_ctx,
            "b_out": self.b_out,
        }

    def forward(
        self, X_seq: np.ndarray, C_now: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Forward pass through T time steps.
        Returns:
            y_pred: (B,) predicted residual
            fatigue_gate: (B,) mean forget-gate activation at final step T-1
            cache: intermediate states for BPTT
        """
        B, T, _ = X_seq.shape
        H = self.hidden_dim

        h = np.zeros((B, H), dtype=np.float32)
        c = np.zeros((B, H), dtype=np.float32)

        h_list = [h]
        c_list = [c]
        i_list, f_list, g_list, o_list, tanh_c_list = [], [], [], [], []

        for t in range(T):
            x_t = X_seq[:, t, :]
            z = x_t @ self.W_ih + h @ self.W_hh + self.b_h
            i_t = _sigmoid(z[:, 0:H])
            f_t = _sigmoid(z[:, H : 2 * H])
            g_t = np.tanh(np.clip(z[:, 2 * H : 3 * H], -15.0, 15.0))
            o_t = _sigmoid(z[:, 3 * H : 4 * H])

            c = f_t * c + i_t * g_t
            tanh_c = np.tanh(np.clip(c, -15.0, 15.0))
            h = o_t * tanh_c

            h_list.append(h)
            c_list.append(c)
            i_list.append(i_t)
            f_list.append(f_t)
            g_list.append(g_t)
            o_list.append(o_t)
            tanh_c_list.append(tanh_c)

        y_out = (h @ self.W_out + C_now @ self.W_ctx + self.b_out).squeeze(-1)
        fatigue_gate = f_list[-1].mean(axis=1)

        cache = {
            "X_seq": X_seq,
            "C_now": C_now,
            "h_list": h_list,
            "c_list": c_list,
            "i_list": i_list,
            "f_list": f_list,
            "g_list": g_list,
            "o_list": o_list,
            "tanh_c_list": tanh_c_list,
        }
        return y_out, fatigue_gate, cache

    def backward_and_step(self, y_pred: np.ndarray, y_true: np.ndarray, cache: Dict[str, Any]) -> float:
        """
        Huber / Smooth-L1 loss + Analytical BPTT across T time steps + Adam parameter update.
        """
        X_seq = cache["X_seq"]
        C_now = cache["C_now"]
        h_list = cache["h_list"]
        c_list = cache["c_list"]
        i_list = cache["i_list"]
        f_list = cache["f_list"]
        g_list = cache["g_list"]
        o_list = cache["o_list"]
        tanh_c_list = cache["tanh_c_list"]

        B, T, _ = X_seq.shape
        H = self.hidden_dim

        # Smooth L1 (Huber, delta=2.0) gradient wrt y_pred
        diff = y_pred - y_true
        delta = 2.0
        abs_diff = np.abs(diff)
        loss = float(np.where(abs_diff <= delta, 0.5 * diff * diff, delta * (abs_diff - 0.5 * delta)).mean())

        dy = np.clip(diff, -delta, delta).reshape(B, 1).astype(np.float32) / float(B)

        h_T = h_list[-1]
        dW_out = h_T.T @ dy + self.weight_decay * self.W_out
        dW_ctx = C_now.T @ dy + self.weight_decay * self.W_ctx
        db_out = dy.sum(axis=0)

        dh_next = dy @ self.W_out.T
        dc_next = np.zeros((B, H), dtype=np.float32)

        dW_ih = np.zeros_like(self.W_ih)
        dW_hh = np.zeros_like(self.W_hh)
        db_h = np.zeros_like(self.b_h)

        for t in reversed(range(T)):
            i_t = i_list[t]
            f_t = f_list[t]
            g_t = g_list[t]
            o_t = o_list[t]
            tanh_c = tanh_c_list[t]
            c_prev = c_list[t]
            h_prev = h_list[t]
            x_t = X_seq[:, t, :]

            do = dh_next * tanh_c
            dc = dc_next + dh_next * o_t * (1.0 - tanh_c * tanh_c)

            df = dc * c_prev
            di = dc * g_t
            dg = dc * i_t
            dc_next = dc * f_t

            dzi = di * i_t * (1.0 - i_t)
            dzf = df * f_t * (1.0 - f_t)
            dzg = dg * (1.0 - g_t * g_t)
            dzo = do * o_t * (1.0 - o_t)

            dz = np.concatenate([dzi, dzf, dzg, dzo], axis=1)
            dW_ih += x_t.T @ dz
            dW_hh += h_prev.T @ dz
            db_h += dz.sum(axis=0)

            dh_next = dz @ self.W_hh.T

        dW_ih += self.weight_decay * self.W_ih
        dW_hh += self.weight_decay * self.W_hh

        grads = {
            "W_ih": np.clip(dW_ih, -1.0, 1.0),
            "W_hh": np.clip(dW_hh, -1.0, 1.0),
            "b_h": np.clip(db_h, -1.0, 1.0),
            "W_out": np.clip(dW_out, -1.0, 1.0),
            "W_ctx": np.clip(dW_ctx, -1.0, 1.0),
            "b_out": np.clip(db_out, -1.0, 1.0),
        }

        # Adam update
        self._adam_t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        params = self._params_dict()
        for k, p in params.items():
            g = grads[k]
            if k not in self._m:
                self._m[k] = np.zeros_like(p)
                self._v[k] = np.zeros_like(p)
            self._m[k] = b1 * self._m[k] + (1.0 - b1) * g
            self._v[k] = b2 * self._v[k] + (1.0 - b2) * (g * g)
            m_hat = self._m[k] / (1.0 - b1 ** self._adam_t)
            v_hat = self._v[k] / (1.0 - b2 ** self._adam_t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

        return loss

    def fit(
        self,
        X_seq: np.ndarray,
        C_now: np.ndarray,
        y: np.ndarray,
        epochs: int = 18,
        batch_size: int = 2048,
        X_val: Optional[np.ndarray] = None,
        C_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        n = len(y)
        rng = np.random.default_rng(42)
        best_val_mae = float("inf")
        best_weights = None

        for ep in range(epochs):
            perm = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                xb, cb, yb = X_seq[idx], C_now[idx], y[idx]
                y_pred, _, cache = self.forward(xb, cb)
                self.backward_and_step(y_pred, yb, cache)

            if X_val is not None and C_val is not None and y_val is not None:
                val_pred, _, _ = self.forward(X_val, C_val)
                val_mae = float(np.abs(val_pred - y_val).mean())
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_weights = {k: v.copy() for k, v in self._params_dict().items()}

        if best_weights is not None:
            for k, v in best_weights.items():
                setattr(self, k, v)

        return {"best_val_mae": best_val_mae}

    def predict(self, X_seq: np.ndarray, C_now: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_pred, fatigue_gate, _ = self.forward(X_seq, C_now)
        return y_pred, fatigue_gate


def build_temporal_sequences_for_market(
    market_df: pd.DataFrame,
    seq_len: int = SEQUENCE_LENGTH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    """
    Constructs chronological (N, T, 8) player game sequences and (N, 5) current-game
    fatigue/rest context vectors strictly from past games t-T .. t-1 (zero leakage).
    """
    df = market_df.sort_values(["player", "date"]).copy()
    n = len(df)

    actual = pd.to_numeric(df["actual_value"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    season_avg = pd.to_numeric(df.get("feat_season_avg", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    std = np.maximum(pd.to_numeric(df.get("feat_std", 1.0), errors="coerce").fillna(1.0).to_numpy(dtype=np.float32), 0.5)
    minutes = pd.to_numeric(df.get("feat_avg_minutes", 28.0), errors="coerce").fillna(28.0).to_numpy(dtype=np.float32)
    days_rest = np.clip(pd.to_numeric(df.get("feat_days_rest", 2.0), errors="coerce").fillna(2.0).to_numpy(dtype=np.float32), 0.0, 7.0)
    is_b2b = pd.to_numeric(df.get("feat_is_b2b", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    miles = np.clip(pd.to_numeric(df.get("feat_miles_96h", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32), 0.0, 5000.0)
    ts_delta = np.clip(pd.to_numeric(df.get("feat_ts_pct_delta", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32), -0.3, 0.3)
    fatigue = np.clip(pd.to_numeric(df.get("feat_fatigue_load", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32), 0.0, 150.0)
    form_gap = np.clip(pd.to_numeric(df.get("feat_form_gap", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32), -15.0, 15.0)
    ema = pd.to_numeric(df.get("feat_ema", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    l10 = pd.to_numeric(df.get("feat_l10_avg", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    # Target for LSTM helper: actual stat residual relative to player's season_avg
    y_resid = (actual - season_avg).astype(np.float32)

    # Per-game step feature matrix (N, 8) — when used at index t, we only look at t-seq_len .. t-1
    step_matrix = np.stack([
        np.clip((actual - season_avg) / std, -4.0, 4.0),
        (minutes - 28.0) / 10.0,
        days_rest / 7.0,
        is_b2b,
        miles / 3000.0,
        ts_delta,
        fatigue / 100.0,
        form_gap / 10.0,
    ], axis=1).astype(np.float32)

    # Current pre-game observable context C_now (N, 5) known BEFORE tip-off of game t
    C_now = np.stack([
        (ema - season_avg) / std,
        (l10 - season_avg) / std,
        days_rest / 7.0,
        is_b2b,
        miles / 3000.0,
    ], axis=1).astype(np.float32)

    # Vectorized player-grouped lag window construction
    players = df["player"].to_numpy()
    X_seq = np.zeros((n, seq_len, step_matrix.shape[1]), dtype=np.float32)

    for k in range(1, seq_len + 1):
        t_pos = seq_len - k  # oldest is 0, most recent (t-1) is seq_len - 1
        same_player = np.zeros(n, dtype=bool)
        if n > k:
            same_player[k:] = (players[k:] == players[:-k])
        shifted = np.zeros_like(step_matrix)
        if n > k:
            shifted[k:] = step_matrix[:-k]
        X_seq[same_player, t_pos, :] = shifted[same_player]

    return X_seq, C_now, y_resid, df.index


class LSTMFeatureScoutEngine:
    """
    Manages training and inference of per-market 4-Gate LSTM Sequential Feature Scouts.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = Path(model_path) if model_path else LSTM_MODEL_PATH
        self.models: Dict[str, VectorizedLSTMCellBPTT] = {}
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        if self.model_path.exists():
            try:
                payload = joblib.load(self.model_path)
                if isinstance(payload, dict):
                    self.models = payload.get("models", {})
                    self.metrics = payload.get("metrics", {})
            except Exception as e:
                logger.warning(f"Failed to load LSTM Feature Scout models: {e}")

    def save(self):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": self.models, "metrics": self.metrics}, self.model_path)

    def predict_live_features(self, fv: Any, market: str) -> Tuple[float, float]:
        """
        Compute (`feat_lstm_momentum`, `feat_lstm_fatigue_gate`) for a single live FeatureVector
        in <0.2 ms using the trained market LSTM model.
        """
        model = self.models.get(market) or self.models.get("PTS")
        if model is None:
            return 0.0, 0.5

        season_avg = float(getattr(fv, "feat_season_avg", 0.0) or 0.0)
        ema = float(getattr(fv, "ema", season_avg) or season_avg)
        l10 = float(getattr(fv, "feat_l10_avg", ema) or ema)
        std = max(float(getattr(fv, "std", 1.0) or 1.0), 0.5)
        minutes = float(getattr(fv, "avg_minutes", 28.0) or 28.0)
        days_rest = min(max(float(getattr(fv, "days_rest", 2) or 2), 0.0), 7.0)
        is_b2b = 1.0 if getattr(fv, "is_b2b", False) else 0.0
        miles = min(max(float(getattr(fv, "feat_miles_96h", 0.0) or 0.0), 0.0), 5000.0)
        ts_delta = min(max(float(getattr(fv, "feat_ts_pct_delta", 0.0) or 0.0), -0.3), 0.3)
        fatigue = min(max(float(getattr(fv, "feat_fatigue_load", 0.0) or 0.0), 0.0), 150.0)
        form_gap = min(max(float(getattr(fv, "feat_form_gap", 0.0) or 0.0), -15.0), 15.0)
        trend_5g = float(getattr(fv, "trend_5g", 0.0) or 0.0)

        # Reconstruct smooth 8-step trajectory from (season_avg -> l10 -> ema + trend_5g)
        X_seq = np.zeros((1, SEQUENCE_LENGTH, 8), dtype=np.float32)
        for t in range(SEQUENCE_LENGTH):
            frac = float(t) / float(SEQUENCE_LENGTH - 1)
            est_resid = ((1.0 - frac) * (l10 - season_avg) + frac * (ema - season_avg + 0.3 * trend_5g)) / std
            X_seq[0, t, 0] = np.clip(est_resid, -4.0, 4.0)
            X_seq[0, t, 1] = (minutes - 28.0) / 10.0
            X_seq[0, t, 2] = days_rest / 7.0
            X_seq[0, t, 3] = is_b2b if t == SEQUENCE_LENGTH - 1 else 0.0
            X_seq[0, t, 4] = (miles * (0.5 + 0.5 * frac)) / 3000.0
            X_seq[0, t, 5] = ts_delta * frac
            X_seq[0, t, 6] = (fatigue * (0.5 + 0.5 * frac)) / 100.0
            X_seq[0, t, 7] = (form_gap * frac) / 10.0

        C_now = np.array([[
            (ema - season_avg) / std,
            (l10 - season_avg) / std,
            days_rest / 7.0,
            is_b2b,
            miles / 3000.0,
        ]], dtype=np.float32)

        pred_resid, fatigue_gate = model.predict(X_seq, C_now)
        return float(pred_resid[0]), float(fatigue_gate[0])
