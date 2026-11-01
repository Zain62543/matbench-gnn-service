import numpy as np

from matbench_gnn.plots import parity_plot


def test_parity_plot_writes_file(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.normal(size=50)
    out = tmp_path / "nested" / "parity.png"

    parity_plot(y, y + rng.normal(scale=0.1, size=50), out, title="test")

    assert out.exists()
    assert out.stat().st_size > 0
