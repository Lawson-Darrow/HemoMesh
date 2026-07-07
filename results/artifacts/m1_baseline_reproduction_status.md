# M1 GEM-GCN Baseline Reproduction Status

## Ready

- Suk dataset is present under `vessel-datasets/stead/`.
- Pretrained model weights are present locally under `model-weights/`.
- Upstream implementation is available locally under `external/coronary-mesh-convolution/`.
- The reusable runner is `scripts/run_suk_gem_gcn_baseline.sh`.

## Weight Manifest

See `results/artifacts/m1_model_weights_manifest.json` for checkpoint sizes and MD5 sums.

Expected active checkpoints:

- `model-weights/stead_single.pt`
- `model-weights/stead_bifurcating.pt`

## Local Blocker

The lightweight project environment does not include the compiled dependencies needed by
the upstream GEM-GCN implementation:

- PyTorch
- PyTorch Geometric and compiled extensions
- GEM-CNN
- VTK
- OpenMesh-related geometry tooling

This is expected. The baseline should be run in an environment matching
`external/coronary-mesh-convolution/environment.yml`.

## Reproduction Command

After activating the matching environment:

```bash
bash scripts/run_suk_gem_gcn_baseline.sh
```

Expected log outputs:

- `results/logs/m1_suk_gem_gcn_single.log`
- `results/logs/m1_suk_gem_gcn_bifurcating.log`

The logs should contain the upstream metric table with approximation error, NMAE,
absolute error summaries, target scale summaries, and cosine similarity.
