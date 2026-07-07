# Results Artifacts

This directory stores the durable outputs used for the final report and
presentation.

- `experiments.csv`: append-only experiment ledger keyed by config hash.
- `figures/`: generated plots for the report and slides.
- `tables/`: report-ready metric tables and confidence intervals.
- `logs/`: command logs for reproducibility.
- `artifacts/`: serialized intermediate outputs such as predictions,
  calibration curves, and risk-coverage data.

Raw datasets and large checkpoints should remain outside version control.
