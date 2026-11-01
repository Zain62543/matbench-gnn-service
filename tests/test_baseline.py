import numpy as np
import pytest

from matbench_gnn.baseline import build_model


@pytest.mark.parametrize("name", ["rf", "hgb"])
def test_model_fits_and_handles_nan(name):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 5))
    y = x[:, 0] * 2.0 + rng.normal(scale=0.01, size=60)
    x[3, 2] = np.nan  # simulates a failed featurisation

    model = build_model(name, seed=0).fit(x, y)
    pred = model.predict(x)

    assert pred.shape == (60,)
    assert np.isfinite(pred).all()


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model("nope")


def test_seed_is_reproducible():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 4))
    y = x.sum(axis=1)
    p1 = build_model("rf", seed=7).fit(x, y).predict(x)
    p2 = build_model("rf", seed=7).fit(x, y).predict(x)
    np.testing.assert_allclose(p1, p2)
