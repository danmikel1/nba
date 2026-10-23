"""
Compare old vs new model calibration, and find optimal temperature for new models.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy import stats as scipy_stats
from scipy.special import expit, logit

DATA_DIR = Path("data")


def _interpolate_prob(line, q_low, q50, q_high):
    z_val = 1.645
    if q_low >= q_high or q_high - q_low < 0.1:
        return 0.50
    sigma_lower = max((q50 - q_low) / z_val, 0.1)
    sigma_upper = max((q_high - q50) / z_val, 0.1)
    z = (line - q50) / (sigma_lower if line <= q50 else sigma_upper)
    p_over = 1.0 - scipy_stats.norm.cdf(z)
    return float(max(0.02, min(0.98, p_over)))


def ece(probs, outcomes, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        total += mask.sum() * abs(probs[mask].mean() - outcomes[mask].mean())
    return total / len(probs)


def calibrate(probs, T):
    """Apply temperature scaling."""
    clipped = np.clip(probs, 0.02, 0.98)
    return expit(logit(clipped) / T)


def main():
    # Load bet tracker for real outcomes
    bt = json.loads((DATA_DIR / "bet_tracker.json").read_text(encoding="utf-8"))
    if isinstance(bt, dict):
        bt = bt.get("bets", [])
    bets_with_result = [
        b for b in bt
        if b.get("result") in ("Won", "Win", "Loss", "Lost") and b.get("prob")
    ]
    print(f"Bets with results: {len(bets_with_result)}")

    if not bets_with_result:
        print("No bets with results found")
        return

    probs = np.array([b["prob"] for b in bets_with_result])
    outcomes = np.array([1.0 if b["result"] in ("Won", "Win") else 0.0 for b in bets_with_result])

    # These are the LIVE probabilities from the old model (already temperature-scaled at T=4.75)
    old_T = 4.75
    print(f"\n=== Live bet tracker (old model, already T={old_T} scaled) ===")
    print(f"  n={len(probs)}  mean_prob={probs.mean():.3f}  actual_wr={outcomes.mean():.3f}")
    print(f"  ECE={ece(probs, outcomes):.4f}")
    print(f"  Accuracy at p>0.5: {((probs > 0.5) == outcomes).mean():.1%}")

    # Now test new model: load and predict on overlap dates
    from nba_prediction import TRAINING_FEATURE_COLUMNS

    MARKET_MAP = {
        "Player Points": "PTS", "Player Rebounds": "REB", "Player Assists": "AST",
        "Player Points + Rebounds + Assists": "PRA", "Player Points + Rebounds": "PR",
        "Player Points + Assists": "PA", "Player Rebounds + Assists": "RA",
    }

    cache_dir = DATA_DIR / "oddsblaze_cache"
    real = []
    for f in sorted(cache_dir.glob("*_draftkings.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        if data.get("_empty"):
            continue
        date_part = f.stem.split("_")[0]
        for ev in data.get("events", []):
            for o in ev.get("odds", []):
                if "player" not in o:
                    continue
                m = MARKET_MAP.get(o.get("market", ""))
                if not m:
                    continue
                sel = o.get("selection", {})
                if sel.get("side") != "Over":
                    continue
                name = o["player"]["name"].strip().lower().replace(".", "").replace("'", "").replace("-", " ")
                real.append({"date": date_part, "name_norm": name, "market": m, "dk_line": sel.get("line")})
    df_real = pd.DataFrame(real).groupby(["date", "name_norm", "market"], as_index=False).last()

    df = pd.read_csv(DATA_DIR / "ml_training_data.csv", low_memory=False)
    overlap_dates = sorted(df_real["date"].unique())
    df_over = df[df["date"].isin(overlap_dates)].copy()
    df_over["name_norm"] = df_over["player"].str.strip().str.lower().str.replace(".", "", regex=False).str.replace("'", "", regex=False).str.replace("-", " ", regex=False)
    merged = df_over.merge(df_real, on=["date", "name_norm", "market"], how="inner")

    # Predict with new models using REAL DK lines
    all_raw_probs = []
    all_actuals = []
    for mkt in sorted(merged["market"].unique()):
        model_path = DATA_DIR / f"nba_quantile_{mkt}.pkl"
        if not model_path.exists():
            continue
        qm = joblib.load(model_path)
        s = merged[merged["market"] == mkt]
        X = s[TRAINING_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0)
        pred_05 = qm["q05"].predict(X)
        pred_50 = qm["q50"].predict(X)
        pred_95 = qm["q95"].predict(X)
        raw_p = np.array([
            _interpolate_prob(l, q05, q50, q95)
            for l, q05, q50, q95 in zip(s["dk_line"], pred_05, pred_50, pred_95)
        ])
        actual = (s["actual_value"].values > s["dk_line"].values).astype(float)
        all_raw_probs.extend(raw_p)
        all_actuals.extend(actual)

    all_raw_probs = np.array(all_raw_probs)
    all_actuals = np.array(all_actuals)

    print(f"\n=== New models vs REAL DK lines (overlap dates, n={len(all_raw_probs)}) ===")
    print(f"  Raw (no T scaling):")
    print(f"    ECE={ece(all_raw_probs, all_actuals):.4f}  acc={((all_raw_probs > 0.5) == all_actuals).mean():.1%}")
    print(f"    mean_prob={all_raw_probs.mean():.3f}  actual_over={all_actuals.mean():.3f}")

    # Grid search for optimal temperature
    print(f"\n  Temperature search:")
    best_t, best_ece = 1.0, 999
    for t in np.arange(0.5, 6.0, 0.05):
        cal_p = calibrate(all_raw_probs, t)
        e = ece(cal_p, all_actuals)
        acc = ((cal_p > 0.5) == all_actuals).mean()
        if e < best_ece:
            best_ece = e
            best_t = t
        if abs(t - 1.0) < 0.01 or abs(t - 2.0) < 0.01 or abs(t - 3.0) < 0.01 or abs(t - 4.75) < 0.01:
            print(f"    T={t:.2f}: ECE={e:.4f}  acc={acc:.1%}")

    print(f"\n  Optimal T={best_t:.2f}: ECE={best_ece:.4f}")
    cal_optimal = calibrate(all_raw_probs, best_t)
    acc_opt = ((cal_optimal > 0.5) == all_actuals).mean()
    print(f"    acc={acc_opt:.1%}  mean_prob={cal_optimal.mean():.3f}")

    # Compare: T=1.0 (no scaling) vs old T=4.75 vs optimal
    for t_val in [1.0, best_t, 4.75]:
        cal_p = calibrate(all_raw_probs, t_val)
        e = ece(cal_p, all_actuals)
        acc = ((cal_p > 0.5) == all_actuals).mean()
        strong = cal_p > 0.60
        strong_wr = all_actuals[strong].mean() if strong.sum() > 0 else 0
        print(f"    T={t_val:.2f}: ECE={e:.4f}  acc={acc:.1%}  strong(p>.60): {strong.sum()} picks, {strong_wr:.1%} WR")


if __name__ == "__main__":
    main()
