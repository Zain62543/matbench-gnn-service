"""Magpie + tree-ensemble baseline on a Matbench task.

Usage:
    python -m matbench_gnn.baseline --task matbench_log_kvrh --model rf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from matbench_gnn.features import magpie_features
from matbench_gnn.metrics import aggregate_folds, regression_metrics
from matbench_gnn.plots import parity_plot

TASKS = ("matbench_log_kvrh", "matbench_log_gvrh")
MODELS = ("rf", "hgb")


def build_model(name: str = "rf", seed: int = 0) -> Pipeline:
    """Imputer + regressor. Imputation guards against NaNs from failed featurisation."""
    if name == "rf":
        reg = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=seed)
    elif name == "hgb":
        reg = HistGradientBoostingRegressor(random_state=seed)
    else:
        raise ValueError(f"unknown model {name!r}; choose from {MODELS}")
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", reg)])


def run_task(task_name: str, model_name: str, seed: int, out_dir: Path) -> dict:
    """Run all five official folds, save metrics, predictions and a parity plot."""
    from matbench.bench import MatbenchBenchmark

    mb = MatbenchBenchmark(autoload=False, subset=[task_name])
    task = next(iter(mb.tasks))
    task.load()

    out_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics, all_true, all_pred = [], [], []

    for fold in task.folds:
        train_x, train_y = task.get_train_and_val_data(fold)
        test_x, test_y = task.get_test_data(fold, include_target=True)

        model = build_model(model_name, seed)
        model.fit(magpie_features(train_x), train_y.to_numpy())
        pred = model.predict(magpie_features(test_x))

        task.record(fold, pred, params={"model": model_name, "seed": seed})
        m = regression_metrics(test_y.to_numpy(), pred)
        fold_metrics.append(m)
        all_true.append(test_y.to_numpy())
        all_pred.append(pred)
        print(f"fold {fold}: MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} R2={m['r2']:.3f}")

    summary = {
        "task": task_name,
        "model": model_name,
        "seed": seed,
        "folds": fold_metrics,
        "aggregate": aggregate_folds(fold_metrics),
    }
    stem = f"{task_name}_{model_name}"
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(
        {"y_true": np.concatenate(all_true), "y_pred": np.concatenate(all_pred)}
    ).to_csv(out_dir / f"{stem}_predictions.csv", index=False)
    parity_plot(
        np.concatenate(all_true),
        np.concatenate(all_pred),
        out_dir / f"{stem}_parity.png",
        title=f"{task_name} ({model_name})",
    )
    mb.to_file(str(out_dir / f"{stem}_matbench.json.gz"))

    agg = summary["aggregate"]["mae"]
    print(f"\n{task_name} [{model_name}] MAE = {agg['mean']:.4f} +/- {agg['std']:.4f}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, default="matbench_log_kvrh")
    parser.add_argument("--model", choices=MODELS, default="rf")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    run_task(args.task, args.model, args.seed, args.out)


if __name__ == "__main__":
    main()
