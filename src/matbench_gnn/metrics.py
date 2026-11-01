"""Regression metrics and fold aggregation.

Matbench reports MAE per fold and summarises as mean +/- std over the five
official folds. We mirror that so numbers are comparable with the leaderboard.
"""

from __future__ import annotations

import numpy as np


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return MAE, RMSE and R^2 for one set of predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("empty input")

    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float("nan") if ss_tot == 0 else 1.0 - float(np.sum(err**2)) / ss_tot
    return {"mae": mae, "rmse": rmse, "r2": r2}


def aggregate_folds(fold_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean and sample standard deviation of each metric across folds."""
    if not fold_metrics:
        raise ValueError("no folds to aggregate")
    out: dict[str, dict[str, float]] = {}
    for key in fold_metrics[0]:
        values = np.array([m[key] for m in fold_metrics], dtype=float)
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        out[key] = {"mean": float(values.mean()), "std": std}
    return out
