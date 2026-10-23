## 2026-03-06 — Realistic line correction & model retrain (V26)

### Problem
Training data used 100% synthetic lines (rolling avg + noise → values like `9.420449`).
Real sportsbook lines are always half-points (e.g., `9.5`). This mismatch caused:
- Model overconfidence (ECE 0.106 raw)  
- Required heavy temperature scaling T=4.75 to compensate
- Trading engine saw distorted probability signals

### Solution
1. **OddsBlaze Rewind backfill**: Fetched real DraftKings player prop lines for 12 recent dates
2. **Per-market OLS correction**: Fitted `feat_ema + season_avg + l10_avg + std → dk_line` regressions (R² 0.87-0.88)
3. **Transformed all 497,945 training rows**: `line` and `feat_line` now realistic X.5 half-points
4. **Retrained all 7 per-market XGBoost quantile models** + isotonic calibrators
5. **Reduced temperature**: T=4.75 → T=2.40 (line correction fixed most overconfidence)

### Accuracy (on 12 overlap dates, 6,398 rows vs real DK lines)
- **ECE: 0.029** (was 0.106 raw / 0.022 after heavy T scaling)
- Single-stat markets: AST 61% exact match, REB 54%, RA 38%
- All lines now proper half-point format matching real sportsbooks

### Files changed
- `data/ml_training_data.csv` — corrected lines (backup: `.backup_pre_line_correction`)
- `data/line_correction_coefficients.json` — per-market OLS coefficients
- `data/nba_quantile_*.pkl` — retrained models
- `data/nba_calibrator_*.pkl` — retrained calibrators
- `nba_prediction.py` — LIVE_CALIBRATION_TEMPERATURE: 4.75 → 2.40
- `scripts/correct_training_lines.py` — line correction pipeline (new)
- `scripts/validate_corrected_models.py` — validation script (new)

---

## 2026-01-29 — Variance calibration & risk-pricing updates

- Deployment: set runtime `sigma_multiplier = 1.6` (temporary mitigation for under‑estimated model σ in `PTS` scoring market).
- Improvement: started extended variance-model retrain/tuning pipeline (long-running sweep saved under `data/`).
- Tests: added CI calibration gate (`tests/test_calibration_gate.py`) to prevent MAE/σ regressions.

Notes:
- Short-term mitigation is reversible; full retrain + calibrator improvements are in progress (see `scripts/variance_retrain_full_sweep.py`).
