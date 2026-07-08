#!/usr/bin/env bash
# Run Suk et al. GEM-GCN pretrained baselines and save logs for M1 reproduction.
#
# Prerequisites:
#   1. Suk dataset under vessel-datasets/stead/{single,bifurcating}/raw/database.hdf5
#   2. Pretrained weights under model-weights/stead_{single,bifurcating}.pt
#   3. Upstream implementation under external/coronary-mesh-convolution
#   4. A Python environment matching external/coronary-mesh-convolution/environment.yml

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UPSTREAM="$ROOT/external/coronary-mesh-convolution"
RESULTS="$ROOT/results/logs"
CLEAR_PROCESSED="${CLEAR_PROCESSED:-1}"

mkdir -p "$RESULTS"

if [ ! -d "$UPSTREAM" ]; then
  echo "Missing upstream implementation: $UPSTREAM"
  echo "Clone with: git clone https://github.com/sukjulian/coronary-mesh-convolution $UPSTREAM"
  exit 1
fi

for subset in single bifurcating; do
  data_file="$ROOT/vessel-datasets/stead/$subset/raw/database.hdf5"
  weight_file="$ROOT/model-weights/stead_$subset.pt"
  if [ ! -f "$data_file" ]; then
    echo "Missing dataset file: $data_file"
    exit 1
  fi
  if [ ! -f "$weight_file" ]; then
    echo "Missing pretrained weight file: $weight_file"
    exit 1
  fi
  if [ "$CLEAR_PROCESSED" = "1" ]; then
    rm -rf "$ROOT/vessel-datasets/stead/$subset/processed"
  fi
done

ln -sfn "$ROOT/vessel-datasets" "$UPSTREAM/vessel-datasets"
ln -sfn "$ROOT/model-weights" "$UPSTREAM/model-weights"

python - <<'PY'
from pathlib import Path

path = Path("external/coronary-mesh-convolution/datasets.py")
text = path.read_text(encoding="utf-8")
old = "self.data, self.slices = torch.load(self.processed_paths[0])"
new = "self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)"
if old in text:
    path.write_text(text.replace(old, new), encoding="utf-8")
    print("Patched upstream dataset loading for PyTorch 2.6+ compatibility.")
elif new in text:
    print("Upstream dataset loading patch already present.")
else:
    raise SystemExit("Could not locate expected torch.load call in upstream datasets.py")
PY

(
  cd "$UPSTREAM"
  python - <<'PY'
import torch
import torch_geometric

print(f"Torch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"PyG version: {torch_geometric.__version__}")
PY
  python main.py --model gem_gcn --artery_type single --num_epochs 0 \
    2>&1 | tee "$RESULTS/m1_suk_gem_gcn_single.log"
  python main.py --model gem_gcn --artery_type bifurcating --num_epochs 0 \
    2>&1 | tee "$RESULTS/m1_suk_gem_gcn_bifurcating.log"
)
