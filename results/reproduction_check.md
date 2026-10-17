# Fresh clone reproduction check

Procedure (all commands run, output seen): `git clone` of this repository into a new directory, a new
virtual environment, `pip install -r requirements-dev.txt` and `pip install -e .`, then
`finanomaly fetch`, `build-dataset`, `quality`, `ruff check`, `ruff format --check`, `mypy src api`,
`pytest --cov --cov-fail-under=80`, `baselines`, `experiment --family all`, `register`, `promote`.

Outcome:

* Data: 7057 rows, 1999-01-04 to 2026-10-16, quality checks passed.
* Lint, format and type checks clean; test coverage 85.04 percent (threshold 80).
* The 28 rows of `runs_summary.csv` match the original run exactly: the maximum absolute difference
  in every numeric column (event recall, false alarm rate, AP, ROC AUC, separation, thresholds,
  parameter counts) is 0.0.
* Same configurations registered (isolation forest 4, autoencoder 5, LightGBM 2) and the same
  champion promoted.
* Not repeated in the clone: `evaluate`, `explain`, `robustness`, Docker and API steps.
