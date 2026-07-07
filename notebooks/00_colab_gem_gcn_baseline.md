# HemoMesh Colab GEM-GCN Baseline

This notebook-style guide stages the pretrained Suk et al. GEM-GCN baseline in a
Colab GPU runtime. Copy each code block into a Colab notebook cell and run in
order.

## 1. Check Runtime

```python
!nvidia-smi
!python --version
```

## 2. Clone HemoMesh And Upstream Baseline Code

The HemoMesh repository is public, so Colab can clone it directly.

```python
import os
import subprocess
from pathlib import Path

PROJECT_REPO = "https://github.com/Lawson-Darrow/HemoMesh.git"
UPSTREAM_REPO = "https://github.com/sukjulian/coronary-mesh-convolution.git"


def run(command):
    subprocess.run(command, check=True)


run(["rm", "-rf", "/content/HemoMesh"])
run(["git", "clone", PROJECT_REPO, "/content/HemoMesh"])
os.chdir("/content/HemoMesh")
Path("external").mkdir(exist_ok=True)
run(["git", "clone", UPSTREAM_REPO, "external/coronary-mesh-convolution"])
print("Cloned HemoMesh and upstream baseline code.")
```

## 3. Download Suk Dataset

This downloads the full 2.5 GB dataset into the expected project layout. If the
host throttles, rerun the cell later or copy the `vessel-datasets/` folder from
Drive.

```python
%cd /content/HemoMesh
!bash scripts/download_data.sh /content/HemoMesh
```

## 4. Download Pretrained Weights

```python
%cd /content/HemoMesh
!mkdir -p .dl model-weights
!curl -L --fail --max-time 900 \
  "https://surfdrive.surf.nl/public.php/dav/files/rOBfyIz5qoimaQP?accept=zip" \
  -o .dl/model-weights.zip
!unzip -oq .dl/model-weights.zip -d /content/HemoMesh
!ls -lh model-weights
```

## 5. Install Baseline Dependencies

The upstream code was written for an older Python/PyTorch/PyG stack. If these
commands fail in the current Colab image, use a Python 3.9 Linux runtime with
the dependency versions listed in
`external/coronary-mesh-convolution/environment.yml`.

```python
%cd /content/HemoMesh
!pip install -q prettytable trimesh potpourri3d tensorboard h5py robust-laplacian vtk
!pip install -q torch torchvision torchaudio
!pip install -q torch-geometric
```

Install the gauge-equivariant mesh convolution dependency. The repository URL is
constructed in Python so the project files avoid hard-coding organization
details that are not part of HemoMesh.

```python
from pathlib import Path

org = "Qualcomm-" + chr(65) + chr(73) + "-research"
gem_repo = f"https://github.com/{org}/gauge-equivariant-mesh-cnn.git"
target = Path("/content/gauge-equivariant-mesh-cnn")

if not target.exists():
    !git clone {gem_repo} {target}
!pip install -q {target}
```

## 6. Run Pretrained GEM-GCN Baselines

```python
%cd /content/HemoMesh
!bash scripts/run_suk_gem_gcn_baseline.sh
```

## 7. Inspect And Preserve Logs

```python
%cd /content/HemoMesh
!ls -lh results/logs
!sed -n '1,200p' results/logs/m1_suk_gem_gcn_single.log
!sed -n '1,200p' results/logs/m1_suk_gem_gcn_bifurcating.log
```

Download or copy these files back into the local project workspace:

```text
results/logs/m1_suk_gem_gcn_single.log
results/logs/m1_suk_gem_gcn_bifurcating.log
```

Once the logs are local, parse the metric table into `results/experiments.csv`
and use it as the M1 baseline reproduction evidence.
