# matbench-gnn-service

How well can structure-only models predict elastic moduli, and where do they fail?

This project predicts DFT-computed bulk modulus (`matbench_log_kvrh`) and shear modulus
(`matbench_log_gvrh`) from crystal structure, using the fixed splits of the
[Matbench](https://matbench.materialsproject.org) benchmark so results are directly
comparable with published numbers.

**Status:** Initial project setup and feature extraction tests completed. 
Baseline model evaluation is in progress.

## Results

Metric: MAE in log10(GPa), reported as mean ± standard deviation across
the 5 official Matbench folds.

| Model | Representation | `log_kvrh` MAE | `log_gvrh` MAE |
|---|---|---|---|
| Random forest | Magpie (composition) | _run to fill_ | _run to fill_ |
| Gradient boosting | Magpie (composition) | _run to fill_ | _run to fill_ |
| CGCNN-style GNN | Crystal graph | _planned_ | _planned_ |

Fill the table only with numbers produced by the commands below. Do not copy leaderboard values.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest -q
python -m matbench_gnn.baseline --task matbench_log_kvrh --model rf
```

Outputs go to `results/`: per-fold metrics (JSON), pooled predictions (CSV), a parity plot,
and a Matbench-format file you can validate or submit.

## Project layout

```
src/matbench_gnn/
  features.py   Magpie composition features (matminer)
  baseline.py   Imputer + RF / gradient boosting, 5-fold Matbench runner, CLI
  metrics.py    MAE / RMSE / R2 and fold aggregation
  plots.py      Parity plots
tests/          Unit tests (metrics, model, featuriser)
.github/        CI: ruff + pytest on Python 3.10 and 3.11
```

## Roadmap

- [x] Matbench loader, Magpie baseline, metrics, tests, CI
- [ ] Record baseline results for both tasks
- [ ] CGCNN-style GNN in PyTorch Geometric (`pip install -e ".[gnn]"`)
- [ ] Error analysis: worst-predicted chemistries and why
- [ ] SHAP feature importance for the baseline
- [ ] FastAPI `/predict` endpoint (CIF in, prediction out) + Dockerfile (`pip install -e ".[serve]"`)

## Notes

- The baseline uses composition only, so it ignores geometry. That is intentional: it is the
  reference a structure-aware model has to beat.
- On small datasets (~10k structures) a tuned tree ensemble can match a simple GNN. Whatever
  the outcome, it will be reported as measured.

## License

MIT
