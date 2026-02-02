import pytest

# Ensure flattened `feat_` columns are converted to a FeatureVector without loading heavy models
# Use a small, importable utility that mirrors the repo's flattened->feature mapping
from nba_ml_utils import fv_from_flat_snapshot as _fv_from_snapshot


def test_fv_from_flattened_snapshot_sets_core_fields():
    snap = {
        'player_id': 999,
        'player_name': 'Unit Test',
        'line': 12.5,
        'feat_std': 1.5,
        'feat_days_rest': 2,
        'feat_usage_rate': 0.321,
        'feat_avg_minutes': 28.0,
    }

    fv = _fv_from_snapshot(snap)

    # core numeric mappings (defensive checks)
    assert hasattr(fv, 'std')
    assert pytest.approx(1.5, rel=1e-3) == fv.std
    assert hasattr(fv, 'days_rest')
    assert fv.days_rest == 2
    assert hasattr(fv, 'usage_rate')
    assert pytest.approx(0.321, rel=1e-3) == fv.usage_rate
    assert hasattr(fv, 'avg_minutes')
    assert pytest.approx(28.0, rel=1e-3) == fv.avg_minutes


def test_population_loop_works_with_flattened_row():
    # Emulate the minimal population loop used by backtest runners
    row = {
        'player': 'Unit Test',
        'feat_std': 0.9,
        'feat_days_rest': 1,
        'feat_usage_rate': 0.12,
    }

    fv = _fv_from_snapshot(row)
    # basic sanity
    assert fv.std == pytest.approx(0.9)
    assert fv.days_rest == 1
    assert fv.usage_rate == pytest.approx(0.12)
