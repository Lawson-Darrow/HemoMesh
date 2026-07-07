# Colab Baseline Workflow

Colab is the default runtime for reproducing the pretrained Suk et al. GEM-GCN
baseline. The local workspace remains the source repository for code, report
artifacts, and GitHub history; Colab provides the Linux GPU environment.

## Runtime

In Colab, choose:

- Runtime type: Python
- Hardware accelerator: GPU

## Inputs

The baseline needs:

- Public HemoMesh repository: `https://github.com/Lawson-Darrow/HemoMesh`
- Suk dataset under `vessel-datasets/stead/`
- Suk pretrained weights under `model-weights/`
- Upstream baseline implementation: `sukjulian/coronary-mesh-convolution`

Raw HDF5 files and checkpoint files should not be committed to GitHub. Download
them inside Colab or copy them from Drive.

## Recommended Flow

1. Open `notebooks/00_colab_gem_gcn_baseline.ipynb` in Colab.
2. Run the setup cells to clone HemoMesh and the upstream implementation.
3. Download the Suk dataset and pretrained weights inside Colab.
4. Install the baseline dependencies.
5. Run the baseline cell, which calls:

```bash
bash scripts/run_suk_gem_gcn_baseline.sh
```

6. Download or copy back:

```text
results/logs/m1_suk_gem_gcn_single.log
results/logs/m1_suk_gem_gcn_bifurcating.log
```

7. Place those logs in the local workspace under `results/logs/`.

## Expected Output

Each log should include the upstream metric table with:

- AE
- NMAE
- D_max
- D_mean
- L_max
- L_median
- CS_mean

After the logs are available locally, parse them into `results/experiments.csv`
and cite the values in the M1 baseline reproduction section of the report.

## Notes

- Colab can clone the HemoMesh repository directly because it is public.
- Raw HDF5 files, checkpoint files, upstream scratch clones, and Colab run
  outputs remain ignored by Git.
- If Colab's default Python/PyTorch image cannot install the upstream stack, use
  a Python 3.9 Linux runtime that follows
  `external/coronary-mesh-convolution/environment.yml`.
