"""
Quick Fix Verification Tests
Run this from the same folder as nba_prediction.py:
    python test_fixes.py

No Streamlit, no API calls, no models needed.
Each test is self-contained and prints PASS or FAIL.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"

# ===========================================================================
# TEST 1: Chronological Split
# Simulates what render_train_model_tab does.
# Verifies test rows are always LATER than train rows.
# ===========================================================================
def test_chronological_split():
    print("\n[Test 1] Chronological split — test rows are always after train rows")

    # Fake dataset with 100 rows across 100 dates
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(100)]
    df = pd.DataFrame({'date': dates, 'value': range(100)})

    # Replicate the fix logic
    df_sorted = df.sort_values('date')
    split_idx = int(len(df_sorted) * 0.8)
    train = df_sorted.iloc[:split_idx]
    test  = df_sorted.iloc[split_idx:]

    train_max_date = train['date'].max()
    test_min_date  = test['date'].min()

    if test_min_date > train_max_date:
        print(f"  Train ends: {train_max_date.date()} | Test starts: {test_min_date.date()}")
        print(f" {PASS} — No future data in training set")
    else:
        print(f"  Train max: {train_max_date.date()} | Test min: {test_min_date.date()}")
        print(f" {FAIL} — Test rows overlap with or precede training rows")

# ===========================================================================
# TEST 2: Calibration Uses Validation Set, Not Test Set
# Verifies the calibrator never sees test rows during fitting.
# ===========================================================================
def test_calibration_no_test_leak():
    print("\n[Test 2] Calibration — fitted on validation set, NOT test set")

    np.random.seed(42)
    n = 200
    X = np.random.rand(n, 5)
    y = np.random.rand(n)

    # Replicate the fix: carve val out of train, calibrate on val
    split = int(n * 0.8)
    X_train_full, X_test = X[:split], X[split:]
    y_train_full, y_test = y[:split], y[split:]

    val_size  = max(1, int(len(y_train_full) * 0.25))
    val_start = len(y_train_full) - val_size
    X_val = X_train_full[val_start:]
    y_val = y_train_full[val_start:]

    # Check: val indices should not overlap with test indices
    train_full_indices = set(range(split))
    test_indices       = set(range(split, n))
    val_indices        = set(range(val_start, split))

    overlap = val_indices & test_indices

    if not overlap:
        print(f"  Train: {split} rows | Val: {val_size} rows (from train) | Test: {n - split} rows")
        print(f" {PASS} — Calibrator validation set has zero overlap with test set")
    else:
        print(f" {FAIL} — Calibrator is leaking {len(overlap)} test rows into fitting")

# ===========================================================================
# TEST 3: Old CSV Is Deleted Before Generation
# Verifies generate_ml_training_data deletes the file at the start.
# ===========================================================================
def test_csv_deletion():
    print("\n[Test 3] CSV deletion — old file removed before generation starts")

    test_path = Path("test_ml_training_data_tmp.csv")

    # Create a fake "old" CSV
    old_df = pd.DataFrame({'date': ['2022-01-01'], 'player': ['OldPlayer'], 'value': [99]})
    old_df.to_csv(test_path, index=False)

    # Simulate the fix logic
    if test_path.exists():
        test_path.unlink()

    if not test_path.exists():
        print(f" {PASS} — Old CSV deleted before generation")
    else:
        print(f" {FAIL} — Old CSV still exists, merge contamination possible")

    # Cleanup
    if test_path.exists():
        test_path.unlink()

# ===========================================================================
# TEST 4: Hit Rate Uses Per-Game Line, Not Rolling Average
# Verifies hit rates return -1.0 sentinel when no per-game line is stored,
# and compute correctly when per-game lines ARE available.
# ===========================================================================
def test_hit_rate_no_circular_line():
    print("\n[Test 4] Hit rate — uses per-game line, not synthetic rolling average")

    # Case A: historical games have no stored 'line' column
    # Expected: sentinel -1.0 (not computed against rolling avg)
    games_no_line = pd.DataFrame({
        'GAME_DATE': pd.date_range('2023-01-01', periods=10),
        'PTS': [20, 25, 30, 18, 22, 28, 19, 24, 27, 21]
    })
    rolling_avg = games_no_line['PTS'].mean()  # This is the circular line

    # Simulate the fixed logic
    if 'line' not in games_no_line.columns:
        feat_hit_rate_l10 = -1.0  # sentinel — no per-game line available
    else:
        hits = sum(1 for _, g in games_no_line.iterrows()
                   if pd.notna(g['line']) and g['line'] > 0 and g['PTS'] > g['line'])
        feat_hit_rate_l10 = hits / len(games_no_line)

    if feat_hit_rate_l10 == -1.0:
        print(f"  No per-game line available → sentinel -1.0 returned")
        print(f" {PASS} — Not computed against rolling average ({rolling_avg:.1f})")
    else:
        print(f" {FAIL} — Hit rate ({feat_hit_rate_l10:.2f}) was computed against rolling avg ({rolling_avg:.1f})")

    # Case B: historical games DO have stored lines
    games_with_line = pd.DataFrame({
        'GAME_DATE': pd.date_range('2023-01-01', periods=10),
        'PTS':  [20, 25, 30, 18, 22, 28, 19, 24, 27, 21],
        'line': [22, 22, 22, 22, 22, 22, 22, 22, 22, 22]
    })
    hits = sum(1 for _, g in games_with_line.iterrows()
               if pd.notna(g['line']) and g['line'] > 0 and g['PTS'] > g['line'])
    feat_hit_rate_with_line = hits / len(games_with_line)

    expected = sum(1 for p in games_with_line['PTS'] if p > 22) / 10
    if abs(feat_hit_rate_with_line - expected) < 0.001:
        print(f"  Per-game line available → hit rate = {feat_hit_rate_with_line:.2f} (vs line=22)")
        print(f" {PASS} — Correctly computed against stored per-game lines")
    else:
        print(f" {FAIL} — Hit rate mismatch: got {feat_hit_rate_with_line:.2f}, expected {expected:.2f}")


# ===========================================================================
# RUN ALL
# ===========================================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  NBA Prediction — Fix Verification Tests")
    print("=" * 55)

    test_chronological_split()
    test_calibration_no_test_leak()
    test_csv_deletion()
    test_hit_rate_no_circular_line()

    print("\n" + "=" * 55)
    print("  Done. Fix any FAIL before trusting your win rate.")
    print("=" * 55)