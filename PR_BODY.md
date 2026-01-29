## Summary

Short-term: deploy runtime `sigma_multiplier = 1.6` to correct an under‑estimated σ in the PTS variance model (temporary mitigation).

Medium-term: add extended variance-model retrain/sweep tooling and a pilot; started a long-running hyperparameter + feature-subset sweep.

Safety: conservative Kelly defaults preserved; added CI calibration gate and unit tests to prevent regressions.

---

## Why

- The scoring variance model underestimates uncertainty (Mean MAE/σ ≈ 1.6). A multiplicative runtime calibration (≈1.6) restores conservative sizing while we finish retraining and calibration.

---

## Changes

- trading_engine.py: runtime `sigma_multiplier` default set to **1.6** (short-term, reversible).
- scripts/: pilot & extended sweep tooling (`variance_retrain_experiment.py`, `variance_retrain_full_sweep.py`, `experiment_sigma_multiplier.py`, `validate_market_calibration.py`).
- tests/: CI calibration gate (`test_calibration_gate.py`), trading risk-pricing tests expanded.
- CHANGELOG.md: documented short-term mitigation and ongoing sweep.

---

## Artifacts (local)

- Models: `data/nba_variance_*.pkl`, `data/nba_model_*.pkl`, `data/nba_calibrator_*.pkl`
- Feature list: `data/nba_features_v20.pkl`
- Diagnostics: `data/calibration_variance.png`, `data/calibration_pit.png`
- Backtests: `data/multi_player_backtest_risk_pricing_sweep.csv`
- Experiments: `data/sigma_multiplier_experiment.csv`, `data/variance_retrain_pilot_summary.csv`

---

## How to validate locally

1. Run unit tests: `python -m pytest tests/test_trading_engine_risk_pricing.py -q` (or full suite `python -m pytest -q`).
2. Calibration forensic: `python -m scripts.validate_market_calibration` (set `RUN_CALIBRATION_GATE=1` to force local gate).
3. Sanity backtest: `python -m scripts.multi_player_backtest_risk_pricing_sweep --top-n 25 --test-days 30`.

---

## Acceptance criteria

- CI passes (including calibration gate when run on full dataset).
- No material regression in mean-model MAE (±5%) and risk-pricing P&L vs baseline in short sweeps.
- Reviewer sign-off: ML lead, Trading/Risk owner, Backtest owner.

---

## Rollback plan

- Revert `sigma_multiplier` to **1.0** (single-line change) and redeploy.

---

## Suggested reviewers

- @ml-lead  (model owner)
- @risk-owner (trading/risk)
- @backtest-owner (backtest owner)

---

## Notes

- This PR is intentionally conservative: multiplier is temporary. The extended sweep will produce candidate variance models to replace the mitigation.
- CI gate prevents accidental promotion while calibration is above threshold.