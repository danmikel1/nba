import os
import pytest


def test_calibration_gate():
    """CI gate: run only in CI — fail if Mean MAE/σ across bins > 1.2 for scoring markets.

    Rationale: the gate should run in CI (where full data/artifacts exist). Local devs can
    run the validator manually. The test is skipped unless `CI=1` or `RUN_CALIBRATION_GATE` is set.
    """
    # Run gate only in CI environments to avoid failing local dev runs
    if not (os.environ.get('CI') or os.environ.get('RUN_CALIBRATION_GATE')):
        pytest.skip('Calibration gate runs only in CI; set RUN_CALIBRATION_GATE=1 to run locally')

    data_csv = os.path.join(os.path.dirname(__file__), '..', 'data', 'ml_training_data.csv')
    if not os.path.exists(data_csv):
        pytest.skip('training CSV not present; skipping calibration gate')

    try:
        from scripts.validate_market_calibration import validate_calibration
    except ImportError:
        pytest.skip('scripts.validate_market_calibration not found')

    ratio = validate_calibration()
    assert ratio <= 1.2, f'Mean MAE/σ too high: {ratio:.3f} (>1.2)'
