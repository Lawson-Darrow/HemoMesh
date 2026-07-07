# Artifact Manifest

Use this manifest to keep experiment outputs report-ready as the project grows.

## Required For Each Run

- Saved config file or command arguments.
- Row appended to `results/experiments.csv`.
- Dataset identifier, split name, and checksum when applicable.
- Random seed and package version.
- Headline metrics with confidence intervals when available.
- Any generated predictions needed to reproduce calibration or deferral plots.

## Output Destinations

- Figures: `results/figures/`
- Tables: `results/tables/`
- Logs: `results/logs/`
- Serialized arrays or curves: `results/artifacts/`
- Notebook analyses: `notebooks/`
- Report source and drafts: `report/`
- Slide source and drafts: `slides/`

## Naming Convention

Prefer names that begin with the milestone and config hash:

```text
m1_<config_hash>_baseline_metrics.csv
m3_<config_hash>_calibration_curve.csv
m4_<config_hash>_risk_coverage.csv
m5_<config_hash>_shift_summary.csv
```

This keeps report figures traceable to the exact run that produced them.
