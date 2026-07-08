# Colab Workflows

Colab is the default runtime for HemoMesh model experiments. The local workspace
remains the source repository for code, report artifacts, and GitHub history;
Colab provides the Linux GPU environment.

## Runtime

In Colab, choose:

- Runtime type: Python
- Hardware accelerator: GPU

## Primary M2 Backbone Workflow

Use `notebooks/01_backbones.ipynb` for the main project path. It clones the
public repository, installs the modern training stack, verifies the Suk dataset,
trains the dense MLP and MeshGNN backbones, and writes:

```text
results/artifacts/m2_mlp_summary.json
results/artifacts/m2_mesh_gnn_summary.json
results/tables/m2_mlp_case_metrics.csv
results/tables/m2_mesh_gnn_case_metrics.csv
results/tables/m2_backbone_comparison.csv
results/figures/m2_backbone_comparison.svg
```

Equivalent command-line runs:

```bash
hemomesh-train --config configs/m2_mlp.yaml
hemomesh-train --config configs/m2_mesh_gnn.yaml
```

## Primary M3 UQ Workflow

Use `notebooks/02_uq_calibration.ipynb` after the M2 backbone workflow. It runs
split-conformal WSS residual calibration and MC-dropout risk-coverage analysis,
then writes:

```text
results/artifacts/m3_uq_summary.json
results/artifacts/m3_uq_summary.md
results/tables/m3_uq_summary.csv
results/tables/m3_risk_coverage.csv
results/figures/m3_risk_coverage.svg
results/figures/m3_uncertainty_error_scatter.svg
```

Equivalent command-line run:

```bash
hemomesh-m3-uq --config configs/m3_uq_mesh_gnn.yaml
```

## Optional M1 GEM-GCN Reproduction Inputs

The legacy pretrained GEM-GCN reproduction is now optional supporting material.
It needs:

- Public HemoMesh repository: `https://github.com/Lawson-Darrow/HemoMesh`
- Suk dataset under `vessel-datasets/stead/`
- Suk pretrained weights under `model-weights/`
- Upstream baseline implementation: `sukjulian/coronary-mesh-convolution`

Raw HDF5 files and checkpoint files should not be committed to GitHub. Download
them inside Colab or copy them from Drive.

## Optional M1 GEM-GCN Flow

1. Open `notebooks/00_colab_gem_gcn_baseline.ipynb` in Colab.
1. Run the setup cells to clone HemoMesh and the upstream implementation.
1. Download the Suk dataset and pretrained weights inside Colab.
1. Install the baseline dependencies.
1. Run the baseline cell, which calls:

```bash
bash scripts/run_suk_gem_gcn_baseline.sh
```

1. Download or copy back:

```text
results/logs/m1_suk_gem_gcn_single.log
results/logs/m1_suk_gem_gcn_bifurcating.log
```

1. Place those logs in the local workspace under `results/logs/`.

## Optional GEM-GCN Expected Output

Each log should include the upstream metric table with:

- AE
- NMAE
- D_max
- D_mean
- L_max
- L_median
- CS_mean

After the logs are available locally, parse them into `results/experiments.csv`
and cite the values in an optional M1 baseline reproduction section.

## Notes

- Colab can clone the HemoMesh repository directly because it is public.
- Raw HDF5 files, checkpoint files, upstream scratch clones, and Colab run
  outputs remain ignored by Git.
- If Colab's default Python/PyTorch image cannot install the upstream stack, use
  a Python 3.9 Linux runtime that follows
  `external/coronary-mesh-convolution/environment.yml`.
