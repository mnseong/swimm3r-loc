#!/usr/bin/env bash
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python -c "import torch, torchvision; assert torch.cuda.is_available(), 'torch without CUDA'; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'torchvision', torchvision.__version__)"
python -c "import numpy, torch; torch.from_numpy(numpy.zeros(1))" 2>/dev/null || python -m pip install "numpy<2"
NP="$(python -c "import numpy; print(numpy.__version__)")"
python -m pip install "numpy==$NP" scipy opencv-python-headless pillow tqdm matplotlib einops roma trimesh loralib huggingface_hub safetensors pyyaml torchmetrics ninja packaging setuptools wheel
command -v nvcc >/dev/null || export PATH="/usr/local/cuda/bin:$PATH"
command -v nvcc >/dev/null || { echo "no nvcc on PATH or in /usr/local/cuda/bin"; exit 1; }
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
export MAX_JOBS="${MAX_JOBS:-6}"
cd "$D/third_party/diff_gauss_pose"
rm -rf build *.egg-info
python -m pip install --no-deps --no-build-isolation .
cd "$D"
python -c "
import diff_gauss_pose
from diff_gauss_pose import GaussianRasterizationSettings, GaussianRasterizer
print('setup ok', diff_gauss_pose.__file__)"
