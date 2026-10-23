"""
Fit live calibration temperature from bet_tracker.json outcomes.

Temperature scaling corrects model overconfidence against sharp sportsbook
lines.  T > 1 means the model is overconfident.  The fitted T should be
copied into Config.LIVE_CALIBRATION_TEMPERATURE in nba_prediction.py.

Usage:
    python scripts/fit_live_calibrator.py          # fit from all settled bets
    python scripts/fit_live_calibrator.py --min 200 # require at least 200 samples
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.optimize import minimize_scalar

DATA_PATH = os.path.join("data", "bet_tracker.json")


def fit_temperature(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Fit temperature T that minimizes negative log-likelihood."""
    probs = np.clip(probs, 0.02, 0.98)

    def nll(T):
        cal = expit(logit(probs) / T)
        cal = np.clip(cal, 1e-7, 1 - 1e-7)
        return -np.mean(outcomes * np.log(cal) + (1 - outcomes) * np.log(1 - cal))

    result = minimize_scalar(nll, bounds=(0.5, 10.0), method="bounded")
    return result.x


def main():
    parser = argparse.ArgumentParser(description="Fit live calibration temperature")
    parser.add_argument("--min", type=int, default=100,
                        help="Minimum settled bets required (default: 100)")
    args = parser.parse_args()

    with open(DATA_PATH, encoding="utf-8") as f:
        bets = json.load(f)

    df = pd.DataFrame(bets)
    settled = df[df["result"].isin(["Win", "Loss"])].copy()

    if len(settled) < args.min:
        print(f"Only {len(settled)} settled bets (need {args.min}). Skipping.")
        sys.exit(0)

    # prob column stores the probability for the chosen side (already > 0.5)
    # outcome = 1 if that side won
    probs = settled["prob"].values.astype(float)
    outcomes = (settled["result"] == "Win").values.astype(float)

    T = fit_temperature(probs, outcomes)

    # Reliability stats
    cal_probs = expit(logit(np.clip(probs, 0.02, 0.98)) / T)
    ece_before = _ece(probs, outcomes)
    ece_after = _ece(cal_probs, outcomes)

    print(f"Settled bets:  {len(settled)}")
    print(f"Win rate:      {outcomes.mean():.1%}")
    print(f"Avg predicted: {probs.mean():.4f} (before) -> {cal_probs.mean():.4f} (after)")
    print(f"Temperature:   T = {T:.3f}")
    print(f"ECE:           {ece_before:.4f} -> {ece_after:.4f}")
    print()
    print(f"Update Config.LIVE_CALIBRATION_TEMPERATURE = {T:.2f}")


def _ece(probs, outcomes, n_bins=10):
    """Expected Calibration Error."""
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    total = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / total * abs(probs[mask].mean() - outcomes[mask].mean())
    return ece


if __name__ == "__main__":
    main()
