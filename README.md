# HemoMesh

HemoMesh studies calibrated uncertainty for geometric deep-learning surrogates of
coronary hemodynamics. The core task is per-node wall shear stress (WSS) vector
prediction on coronary artery surface meshes, with pressure as a secondary
target. The research contribution is not only to predict the field, but to
quantify when and where the surrogate should defer to a full simulation.

## Research Question

Can a mesh-based surrogate for coronary WSS produce spatially resolved
uncertainty estimates that remain calibrated in distribution, degrade
transparently under distribution shift, and support selective deferral to
simulation on high-risk regions or cases?

## Scope

The project uses the Suk et al. coronary mesh dataset from
`sukjulian/coronary-mesh-convolution`, which contains synthetic steady-flow
coronary artery simulations with per-node WSS, pressure, mesh coordinates,
faces, and inlet indices. No new CFD data generation is part of the core scope.

Core components:

- Data loading and validation for Suk-style HDF5 coronary mesh datasets.
- Modern-stack MLP and mesh-GNN training for WSS and pressure fields.
- Field metrics for WSS and pressure regression.
- Calibration, uncertainty, and selective-deferral metrics.
- Baseline logging and reproducible experiment manifests.
- A report-ready artifact structure for figures, tables, logs, and notebooks.

Out of scope for the core project:

- Clinical vFFR or diagnostic-accuracy claims.
- New SimVascular CFD generation.
- Pulsatile-flow experiments.
- Patient-specific real-geometry benchmarking as a quantitative claim.

## Repository Layout

```text
hemomesh/
  data/            Dataset inspection, checksums, and HDF5 loading
  models/          Baseline model definitions and integration points
  uq/              Conformal, ensemble, and MC-dropout uncertainty utilities
  metrics/         Field, calibration, and selective-deferral metrics
  shift/           Topology and severity split helpers
  baselines/       Baseline reproduction and logging tools
tests/             Unit and smoke tests
notebooks/         Report-oriented analysis notebooks
results/           Experiment log, figures, tables, and run artifacts
report/            Final written report materials
slides/            Presentation materials
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,analysis]"
```

Install the optional training stack when running the model experiments:

```bash
python -m pip install -e ".[torch]"
```

## Dataset

Expected Suk dataset layout:

```text
vessel-datasets/
  stead/
    single/
      raw/database.hdf5
    bifurcating/
      raw/database.hdf5
```

Inspect a downloaded dataset:

```bash
hemomesh-inspect vessel-datasets/stead/single/raw/database.hdf5 --subset single
hemomesh-inspect vessel-datasets/stead/bifurcating/raw/database.hdf5 --subset bifurcating
```

The inspection command reports case counts, tensor shapes, and MD5 checksums so
the report can cite exact data provenance.

## Primary Training Workflow

The main backbone workflow trains a dense per-node MLP and a message-passing
MeshGNN on the Suk meshes:

```bash
hemomesh-train --config configs/m2_mlp.yaml
hemomesh-train --config configs/m2_mesh_gnn.yaml
```

In Colab, use `notebooks/01_backbones.ipynb` for the same flow plus report-ready
table and figure generation.

## Uncertainty Workflow

The M3 workflow calibrates uncertainty on the MeshGNN backbone with
split-conformal WSS residual radii and MC-dropout risk-coverage analysis:

```bash
hemomesh-m3-uq --config configs/m3_uq_mesh_gnn.yaml
```

In Colab, use `notebooks/02_uq_calibration.ipynb`.

## Optional Baseline Logging

Once baseline predictions or pretrained outputs are available, log the WSS
approximation error through the experiment harness:

```bash
hemomesh-log-baseline \
  --name gem_gcn_pretrained \
  --subset single \
  --truth path/to/wss_truth.npy \
  --prediction path/to/wss_prediction.npy
```

Each run appends a config-hash-keyed row to `results/experiments.csv`.

## Testing

```bash
pytest
```

The smoke tests build a tiny synthetic HDF5 sample and verify that loading,
field metrics, and baseline logging work without the full dataset.

## Reporting Artifacts

The project is organized so the final report and presentation can be assembled
from saved outputs:

- `results/experiments.csv` stores run metadata and headline metrics.
- `results/figures/` stores generated figures.
- `results/tables/` stores report-ready tables.
- `results/logs/` stores run logs.
- `results/artifacts/` stores serialized intermediate outputs.
- `notebooks/` stores analysis notebooks used to build report figures.

## Citation And Data License

The Suk et al. code repository is MIT licensed and the associated dataset is
released under CC BY 4.0. Cite the original dataset and paper when reporting
results. This repository contains project code and experiment scaffolding; the
raw dataset and model checkpoints should be downloaded separately.
