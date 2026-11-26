#!/usr/bin/env bash
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M="${1:-all}"
case "$M" in v9c|v9c_best) M=v8 ;; esac
case "$M" in v3|v8|all) ;; *) echo "usage: setup.sh [v3|v8|v9c|all]"; exit 1 ;; esac
python -c "import torch, torchvision; assert torch.cuda.is_available(), 'torch without CUDA'; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'torchvision', torchvision.__version__)"
python -c "import numpy, torch; torch.from_numpy(numpy.zeros(1))" 2>/dev/null || python -m pip install "numpy<2"
NP="$(python -c "import numpy; print(numpy.__version__)")"
python -m pip install "numpy==$NP" scipy opencv-python-headless pillow tqdm matplotlib einops roma trimesh loralib huggingface_hub safetensors pyyaml torchmetrics rich ninja typing_extensions packaging setuptools wheel
command -v nvcc >/dev/null || export PATH="/usr/local/cuda/bin:$PATH"
command -v nvcc >/dev/null || { echo "no nvcc on PATH or in /usr/local/cuda/bin"; exit 1; }
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
export MAX_JOBS="${MAX_JOBS:-6}"
if [ "$M" != v8 ]; then
  cd "$D/third_party/diff_gauss_pose"
  rm -rf build *.egg-info
  python -m pip install --no-deps --no-build-isolation --force-reinstall .
  cd "$D"
  python -c "
import diff_gauss_pose
from diff_gauss_pose import GaussianRasterizationSettings, GaussianRasterizer
print('setup ok v3 (Gaussian rasteriser)', diff_gauss_pose.__file__)"
fi
if [ "$M" != v3 ]; then
  cd "$D/ubs/submodules"
  rm -rf build gsplat/csrc*.so
  python setup.py build_ext --inplace
  cd "$D"
  PYTHONPATH="$D/ubs/submodules" python -c "
import gsplat
from gsplat.cuda._backend import _C
assert _C is not None, 'the Stage 2 rasteriser did not load'
print('setup ok v8 and v9c (Stage 2 rasteriser)', gsplat.__version__, _C.__file__)"
fi
