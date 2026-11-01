"""Plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parity_plot(
    y_true, y_pred, path: str | Path, title: str = "", unit: str = "log10(GPa)"
) -> None:
    """Save a predicted-vs-true parity plot."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))

    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=200)
    ax.scatter(y_true, y_pred, s=6, alpha=0.4, linewidths=0)
    ax.plot([lo, hi], [lo, hi], color="black", lw=1)
    ax.set_xlabel(f"DFT ({unit})")
    ax.set_ylabel(f"Predicted ({unit})")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
