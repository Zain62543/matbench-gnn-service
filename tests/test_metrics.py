import numpy as np
import pytest

from matbench_gnn.metrics import aggregate_folds, regression_metrics


def test_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    m = regression_metrics(y, y)
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["r2"] == pytest.approx(1.0)


def test_known_values():
    m = regression_metrics([0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 1.0, -1.0])
    assert m["mae"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(1.0)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        regression_metrics([1, 2, 3], [1, 2])


def test_empty_raises():
    with pytest.raises(ValueError):
        regression_metrics([], [])


def test_constant_target_r2_is_nan():
    assert np.isnan(regression_metrics([2.0, 2.0], [1.0, 3.0])["r2"])


def test_aggregate_folds():
    agg = aggregate_folds([{"mae": 1.0}, {"mae": 3.0}])
    assert agg["mae"]["mean"] == pytest.approx(2.0)
    assert agg["mae"]["std"] == pytest.approx(np.sqrt(2.0))


def test_aggregate_single_fold_has_zero_std():
    assert aggregate_folds([{"mae": 1.0}])["mae"]["std"] == 0.0
