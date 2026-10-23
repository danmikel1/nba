"""
Correct synthetic training lines → realistic half-point sportsbook lines.

Uses per-market OLS regressions fitted on 12 overlap dates where we have both
synthetic training data and real DraftKings lines from OddsBlaze.

Transforms:  line, feat_line  →  OLS(feat_ema, feat_season_avg, feat_l10_avg, feat_std)
             rounded to nearest 0.5

Usage:
    python scripts/correct_training_lines.py            # fit + transform
    python scripts/correct_training_lines.py --dry-run  # show stats only
"""

import sys, os, json, argparse, shutil, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "oddsblaze_cache"
COEFF_PATH = DATA_DIR / "line_correction_coefficients.json"

MARKET_MAP = {
    "Player Points": "PTS",
    "Player Rebounds": "REB",
    "Player Assists": "AST",
    "Player Points + Rebounds + Assists": "PRA",
    "Player Points + Rebounds": "PR",
    "Player Points + Assists": "PA",
    "Player Rebounds + Assists": "RA",
}

FEATURES = ["feat_ema", "feat_season_avg", "feat_l10_avg", "feat_std"]


def load_real_lines():
    """Load real DraftKings lines from OddsBlaze cache."""
    real = []
    for f in sorted(CACHE_DIR.glob("*_draftkings.json")):
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
                name = (
                    o["player"]["name"]
                    .strip().lower()
                    .replace(".", "")
                    .replace("'", "")
                    .replace("-", " ")
                )
                real.append({
                    "date": date_part,
                    "name_norm": name,
                    "market": m,
                    "dk_line": sel.get("line"),
                })
    df = pd.DataFrame(real)
    if df.empty:
        return df
    return df.groupby(["date", "name_norm", "market"], as_index=False).last()


def fit_coefficients(df_train, df_real):
    """Fit per-market OLS from training features → real DK lines."""
    overlap_dates = sorted(df_real["date"].unique())
    df_sub = df_train[df_train["date"].isin(overlap_dates)].copy()
    df_sub["name_norm"] = (
        df_sub["player"]
        .str.strip().str.lower()
        .str.replace(".", "", regex=False)
        .str.replace("'", "", regex=False)
        .str.replace("-", " ", regex=False)
    )

    merged = df_sub.merge(df_real, on=["date", "name_norm", "market"], how="inner")
    logger.info(f"Overlap matched rows: {len(merged):,}")

    coefficients = {}
    for mkt in sorted(merged["market"].unique()):
        s = merged[merged["market"] == mkt].dropna(subset=FEATURES + ["dk_line"])
        if len(s) < 30:
            logger.warning(f"  {mkt}: only {len(s)} rows, skipping")
            continue

        X = s[FEATURES].values
        y = s["dk_line"].values
        lr = LinearRegression().fit(X, y)

        pred = lr.predict(X)
        pred_r = np.round(pred - 0.5) + 0.5
        mae = np.abs(pred_r - y).mean()
        exact = (np.abs(pred_r - y) == 0).mean()
        r2 = lr.score(X, y)

        coefficients[mkt] = {
            "coef": lr.coef_.tolist(),
            "intercept": float(lr.intercept_),
            "features": FEATURES,
            "n_samples": len(s),
            "r2": round(r2, 4),
            "mae_rounded": round(mae, 3),
            "exact_match": round(exact, 3),
        }
        logger.info(
            f"  {mkt}: n={len(s):,}  R²={r2:.4f}  MAE={mae:.3f}  exact={exact:.1%}"
        )

    return coefficients


def apply_correction(df, coefficients):
    """Transform line and feat_line using OLS coefficients, round to 0.5."""
    df = df.copy()
    corrected = 0
    uncorrected = 0

    for mkt, coeff in coefficients.items():
        mask = df["market"] == mkt
        subset = df.loc[mask]
        if subset.empty:
            continue

        X = subset[FEATURES].fillna(0).values
        w = np.array(coeff["coef"])
        b = coeff["intercept"]

        predicted = X @ w + b
        # Round to nearest X.5 (real sportsbook lines are always half-points)
        corrected_lines = np.round(predicted - 0.5) + 0.5
        # Enforce minimum 0.5 (no negative or zero lines)
        corrected_lines = np.maximum(corrected_lines, 0.5)

        df.loc[mask, "line"] = corrected_lines
        df.loc[mask, "feat_line"] = corrected_lines
        corrected += mask.sum()

    uncorrected = len(df) - corrected
    if uncorrected > 0:
        logger.warning(f"{uncorrected} rows had no market coefficients — left unchanged")

    return df, corrected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Show stats only")
    args = parser.parse_args()

    csv_path = DATA_DIR / "ml_training_data.csv"
    if not csv_path.exists():
        logger.error("Training data not found")
        sys.exit(1)

    # Step 1: Load real lines
    logger.info("Loading real DraftKings lines from OddsBlaze cache...")
    df_real = load_real_lines()
    if df_real.empty:
        logger.error("No real lines found in OddsBlaze cache")
        sys.exit(1)
    logger.info(f"  {len(df_real):,} real lines across {df_real['date'].nunique()} dates")

    # Step 2: Load training data
    logger.info("Loading training data...")
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"  {len(df):,} rows")

    # Step 3: Fit coefficients
    logger.info("Fitting per-market OLS coefficients...")
    coefficients = fit_coefficients(df, df_real)
    if not coefficients:
        logger.error("No coefficients fitted")
        sys.exit(1)

    # Save coefficients
    with open(COEFF_PATH, "w") as f:
        json.dump(coefficients, f, indent=2)
    logger.info(f"Saved coefficients to {COEFF_PATH}")

    # Step 4: Show before stats
    logger.info("\n=== BEFORE correction ===")
    for mkt in sorted(coefficients.keys()):
        s = df[df["market"] == mkt]
        lines = s["feat_line"]
        pct_half = (lines % 1).isin([0.5]).mean()
        logger.info(f"  {mkt}: mean={lines.mean():.2f}  std={lines.std():.2f}  pct_halfpoint={pct_half:.1%}")

    if args.dry_run:
        # Show what correction would look like on a sample
        df_sample, n = apply_correction(df.head(1000), coefficients)
        logger.info(f"\n[DRY RUN] Would correct {len(df):,} rows. Sample stats on first 1000:")
        for mkt in sorted(coefficients.keys()):
            s = df_sample[df_sample["market"] == mkt]
            if s.empty:
                continue
            lines = s["feat_line"]
            pct_half = (lines % 1).isin([0.5]).mean()
            logger.info(f"  {mkt}: mean={lines.mean():.2f}  pct_halfpoint={pct_half:.1%}")
        return

    # Step 5: Apply correction
    logger.info("\nApplying line correction...")
    df_corrected, n_corrected = apply_correction(df, coefficients)
    logger.info(f"  Corrected {n_corrected:,} rows")

    # Step 6: Show after stats
    logger.info("\n=== AFTER correction ===")
    for mkt in sorted(coefficients.keys()):
        s = df_corrected[df_corrected["market"] == mkt]
        lines = s["feat_line"]
        pct_half = (lines % 1).isin([0.5]).mean()
        logger.info(f"  {mkt}: mean={lines.mean():.2f}  std={lines.std():.2f}  pct_halfpoint={pct_half:.1%}")

    # Step 7: Backup original and save
    backup_path = DATA_DIR / "ml_training_data.csv.backup_pre_line_correction"
    if not backup_path.exists():
        shutil.copy2(csv_path, backup_path)
        logger.info(f"Backed up original to {backup_path.name}")
    else:
        logger.info(f"Backup already exists: {backup_path.name}")

    df_corrected.to_csv(csv_path, index=False)
    logger.info(f"Saved corrected data to {csv_path}")

    # Step 8: Verify
    verify = pd.read_csv(csv_path, usecols=["market", "line", "feat_line"], nrows=10000)
    all_half = (verify["feat_line"] % 1 == 0.5).all()
    logger.info(f"\nVerification: all lines at half-points = {all_half}")
    logger.info("Done! Run `python scripts/train_models.py` to retrain.")


if __name__ == "__main__":
    main()
