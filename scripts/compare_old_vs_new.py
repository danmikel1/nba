"""Compare old model (bet_tracker probabilities) vs new model on same bets."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy import stats as scipy_stats
from scipy.special import expit, logit

DATA_DIR = Path("data")

from nba_prediction import TRAINING_FEATURE_COLUMNS


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
    clipped = np.clip(probs, 0.02, 0.98)
    return expit(logit(clipped) / T)


def main():
    # Load bet tracker
    bt = json.loads((DATA_DIR / "bet_tracker.json").read_text(encoding="utf-8"))
    if isinstance(bt, dict):
        bt = bt.get("bets", [])

    bets = [b for b in bt if b.get("result") in ("Won", "Win", "Loss", "Lost")]
    print(f"=== Replaying {len(bets)} historical bets through OLD vs NEW model ===\n")

    # Extract features and predict with new model
    feat_cols = TRAINING_FEATURE_COLUMNS
    rows = []
    for b in bets:
        feat_vals = {c: b.get(c, 0) for c in feat_cols}
        rows.append({
            **feat_vals,
            "market": b["market"],
            "line": b["line"],
            "side": b["side"],
            "old_prob": b["prob"],
            "result": 1.0 if b["result"] in ("Won", "Win") else 0.0,
            "actual_value": b.get("actual_value"),
            "odds": b.get("odds", -110),
            "stake": b.get("stake", 0),
            "ev": b.get("ev", 0),
            "tag": b.get("tag", ""),
        })

    df = pd.DataFrame(rows)
    print(f"Markets: {df['market'].value_counts().to_dict()}")
    print(f"Tags: {df['tag'].value_counts().to_dict()}\n")

    # Predict with new models
    new_raw_probs = []
    for idx, row in df.iterrows():
        mkt = row["market"]
        model_path = DATA_DIR / f"nba_quantile_{mkt}.pkl"
        if not model_path.exists():
            model_path = DATA_DIR / "nba_quantile_universal.pkl"
        qm = joblib.load(model_path)

        X = pd.DataFrame([{c: row[c] for c in feat_cols}])
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        q05 = qm["q05"].predict(X)[0]
        q50 = qm["q50"].predict(X)[0]
        q95 = qm["q95"].predict(X)[0]

        p = _interpolate_prob(row["line"], q05, q50, q95)
        # Flip for Under bets
        if row["side"] == "Under":
            p = 1.0 - p
        new_raw_probs.append(p)

    df["new_raw_prob"] = new_raw_probs
    df["new_prob_T240"] = calibrate(df["new_raw_prob"].values, 2.40)
    # Old probs were already T=4.75 scaled in production
    old_probs = df["old_prob"].values
    outcomes = df["result"].values

    # =============================================
    # OVERALL COMPARISON
    # =============================================
    print("=" * 65)
    print("OVERALL COMPARISON")
    print("=" * 65)

    for label, probs in [
        ("Old model (T=4.75, production)", old_probs),
        ("New model (T=2.40)", df["new_prob_T240"].values),
        ("New model (raw, no T)", df["new_raw_prob"].values),
    ]:
        acc = ((probs > 0.5) == outcomes).mean()
        e = ece(probs, outcomes)
        mean_p = probs.mean()
        wr = outcomes.mean()
        print(f"\n  {label}:")
        print(f"    Accuracy: {acc:.1%}  ({int(acc*len(outcomes))}/{len(outcomes)})")
        print(f"    ECE: {e:.4f}")
        print(f"    Mean prob: {mean_p:.3f}  Actual WR: {wr:.3f}  Gap: {mean_p-wr:+.3f}")

    # =============================================
    # BY MARKET
    # =============================================
    print(f"\n{'=' * 65}")
    print("BY MARKET")
    print("=" * 65)
    for mkt in sorted(df["market"].unique()):
        s = df[df["market"] == mkt]
        if len(s) < 5:
            continue
        old_acc = ((s["old_prob"].values > 0.5) == s["result"].values).mean()
        new_acc = ((s["new_prob_T240"].values > 0.5) == s["result"].values).mean()
        old_ece = ece(s["old_prob"].values, s["result"].values)
        new_ece = ece(s["new_prob_T240"].values, s["result"].values)
        delta_acc = new_acc - old_acc
        print(f"  {mkt:4s} (n={len(s):3d}): old_acc={old_acc:.1%}  new_acc={new_acc:.1%}  ({delta_acc:+.1%})  old_ECE={old_ece:.3f}  new_ECE={new_ece:.3f}")

    # =============================================
    # CONFIDENCE BUCKETS
    # =============================================
    print(f"\n{'=' * 65}")
    print("CONFIDENCE BUCKETS (new model T=2.40)")
    print("=" * 65)
    for lo, hi, label in [(0.50, 0.55, "Low  0.50-0.55"), (0.55, 0.60, "Med  0.55-0.60"),
                           (0.60, 0.70, "High 0.60-0.70"), (0.70, 1.0, "Best 0.70+")]:
        mask = (df["new_prob_T240"] >= lo) & (df["new_prob_T240"] < hi)
        if mask.sum() == 0:
            print(f"  {label}: 0 bets")
            continue
        wr = outcomes[mask].mean()
        n = mask.sum()
        print(f"  {label}: {n:3d} bets  WR={wr:.1%}  avg_prob={df.loc[mask,'new_prob_T240'].mean():.3f}")

    # =============================================
    # SIMULATED P/L
    # =============================================
    print(f"\n{'=' * 65}")
    print("SIMULATED P/L (flat $10 stakes)")
    print("=" * 65)

    for label, prob_col, threshold in [
        ("Old model all", "old_prob", 0.50),
        ("New model all", "new_prob_T240", 0.50),
        ("New model p>0.55", "new_prob_T240", 0.55),
        ("New model p>0.60", "new_prob_T240", 0.60),
    ]:
        mask = df[prob_col] >= threshold
        sub = df[mask]
        if len(sub) == 0:
            continue
        wins = sub["result"].sum()
        losses = len(sub) - wins
        # Assume -110 odds for simplification
        profit = wins * (10 * 100/110) - losses * 10
        roi = profit / (len(sub) * 10) * 100
        print(f"  {label:25s}: {len(sub):3d} bets  W/L={int(wins)}/{int(losses)}  P/L=${profit:+.0f}  ROI={roi:+.1f}%")

    # =============================================
    # WOULD-HAVE-CHANGED PICKS
    # =============================================
    print(f"\n{'=' * 65}")
    print("PICKS THAT WOULD CHANGE (old took, new wouldn't / vice versa)")
    print("=" * 65)
    old_take = old_probs > 0.55
    new_take = df["new_prob_T240"].values > 0.55

    both = old_take & new_take
    old_only = old_take & ~new_take
    new_only = ~old_take & new_take

    print(f"  Both take (p>0.55):       {both.sum():3d} bets  WR={outcomes[both].mean():.1%}" if both.sum() > 0 else "  Both: 0")
    print(f"  Old takes, New drops:     {old_only.sum():3d} bets  WR={outcomes[old_only].mean():.1%}" if old_only.sum() > 0 else "  Old only: 0")
    print(f"  New takes, Old drops:     {new_only.sum():3d} bets  WR={outcomes[new_only].mean():.1%}" if new_only.sum() > 0 else "  New only: 0")


if __name__ == "__main__":
    main()
