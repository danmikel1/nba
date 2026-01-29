## 2026-01-29 — Variance calibration & risk-pricing updates

- Deployment: set runtime `sigma_multiplier = 1.6` (temporary mitigation for under‑estimated model σ in `PTS` scoring market).
- Improvement: started extended variance-model retrain/tuning pipeline (long-running sweep saved under `data/`).
- Tests: added CI calibration gate (`tests/test_calibration_gate.py`) to prevent MAE/σ regressions.

Notes:
- Short-term mitigation is reversible; full retrain + calibrator improvements are in progress (see `scripts/variance_retrain_full_sweep.py`).
