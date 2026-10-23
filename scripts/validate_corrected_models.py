"""
Validate retrained models by checking calibration on the 12 overlap dates
where we have real DK lines + actual outcomes.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy import stats as scipy_stats

DATA_DIR = Path("data")

MARKET_MAP = {
    "Player Points": "PTS", "Player Rebounds": "REB", "Player Assists": "AST",
    "Player Points + Rebounds + Assists": "PRA", "Player Points + Rebounds": "PR",
    "Player Points + Assists": "PA", "Player Rebounds + Assists": "RA",
}


def _interpolate_prob(line, q_low, q50, q_high):
    z_val = 1.645
    if q_low >= q_high or q_high - q_low < 0.1:
        return 0.50
    sigma_lower = max((q50 - q_low) / z_val, 0.1)
    sigma_upper = max((q_high - q50) / z_val, 0.1)
    z = (line - q50) / (sigma_lower if line <= q50 else sigma_upper)
    p_over = 1.0 - scipy_stats.norm.cdf(z)
    return float(max(0.02, min(0.98, p_over)))


def main():
    from nba_prediction import TRAINING_FEATURE_COLUMNS

    # Load corrected training data (overlap dates only)
    df = pd.read_csv(DATA_DIR / "ml_training_data.csv", low_memory=False)

    # Load real DK lines
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

    overlap_dates = sorted(df_real["date"].unique())
    df_over = df[df["date"].isin(overlap_dates)].copy()
    df_over["name_norm"] = df_over["player"].str.strip().str.lower().str.replace(".", "", regex=False).str.replace("'", "", regex=False).str.replace("-", " ", regex=False)

    merged = df_over.merge(df_real, on=["date", "name_norm", "market"], how="inner")
    print(f"Overlap rows with real DK lines + actual outcomes: {len(merged):,}")

    # For each market, load model and predict
    results = []
    for mkt in sorted(merged["market"].unique()):
        model_path = DATA_DIR / f"nba_quantile_{mkt}.pkl"
        if not model_path.exists():
            print(f"  {mkt}: model not found, skipping")
            continue

        qm = joblib.load(model_path)
        s = merged[merged["market"] == mkt].copy()

        X = s[TRAINING_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0)

        pred_05 = qm["q05"].predict(X)
        pred_50 = qm["q50"].predict(X)
        pred_95 = qm["q95"].predict(X)

        # Probability using REAL DK lines
        raw_probs_dk = np.array([
            _interpolate_prob(l, q05, q50, q95)
            for l, q05, q50, q95 in zip(s["dk_line"], pred_05, pred_50, pred_95)
        ])
        actual_over_dk = (s["actual_value"].values > s["dk_line"].values).astype(float)

        # Probability using corrected training lines (feat_line)
        raw_probs_corr = np.array([
            _interpolate_prob(l, q05, q50, q95)
            for l, q05, q50, q95 in zip(s["feat_line"], pred_05, pred_50, pred_95)
        ])
        actual_over_corr = (s["actual_value"].values > s["feat_line"].values).astype(float)

        # Accuracy at p>0.5 threshold
        acc_dk = ((raw_probs_dk > 0.5) == actual_over_dk).mean()
        acc_corr = ((raw_probs_corr > 0.5) == actual_over_corr).mean()

        # Calibration (ECE)
        def ece(probs, outcomes, n_bins=10):
            bins = np.linspace(0, 1, n_bins + 1)
            total = 0
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (probs >= lo) & (probs < hi)
                if mask.sum() == 0:
                    continue
                avg_pred = probs[mask].mean()
                avg_true = outcomes[mask].mean()
                total += mask.sum() * abs(avg_pred - avg_true)
            return total / len(probs)

        ece_dk = ece(raw_probs_dk, actual_over_dk)
        ece_corr = ece(raw_probs_corr, actual_over_corr)

        # MAE of stat prediction
        mae_stat = np.abs(pred_50 - s["actual_value"].values).mean()

        print(f"\n  {mkt} (n={len(s):,}):")
        print(f"    Stat prediction MAE: {mae_stat:.2f}")
        print(f"    vs REAL DK lines: acc={acc_dk:.1%}  ECE={ece_dk:.4f}  mean_prob={raw_probs_dk.mean():.3f}  actual_over={actual_over_dk.mean():.3f}")
        print(f"    vs corrected lines: acc={acc_corr:.1%}  ECE={ece_corr:.4f}  mean_prob={raw_probs_corr.mean():.3f}  actual_over={actual_over_corr.mean():.3f}")

        # Strong picks (prob > 0.60)
        strong_mask_dk = raw_probs_dk > 0.60
        if strong_mask_dk.sum() > 0:
            strong_acc = (actual_over_dk[strong_mask_dk]).mean()
            print(f"    Strong Over picks (p>0.60 vs DK): {strong_mask_dk.sum()} picks, {strong_acc:.1%} hit rate")

        results.append({
            "market": mkt, "n": len(s), "mae_stat": mae_stat,
            "acc_dk": acc_dk, "ece_dk": ece_dk,
            "acc_corr": acc_corr, "ece_corr": ece_corr,
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = sum(r["n"] for r in results)
    weighted_acc_dk = sum(r["acc_dk"] * r["n"] for r in results) / total
    weighted_ece_dk = sum(r["ece_dk"] * r["n"] for r in results) / total
    weighted_acc_corr = sum(r["acc_corr"] * r["n"] for r in results) / total
    weighted_ece_corr = sum(r["ece_corr"] * r["n"] for r in results) / total
    print(f"Total rows: {total:,}")
    print(f"Weighted accuracy vs REAL DK lines: {weighted_acc_dk:.1%}")
    print(f"Weighted ECE vs REAL DK lines:      {weighted_ece_dk:.4f}")
    print(f"Weighted accuracy vs corrected:     {weighted_acc_corr:.1%}")
    print(f"Weighted ECE vs corrected:          {weighted_ece_corr:.4f}")


if __name__ == "__main__":
    main()
