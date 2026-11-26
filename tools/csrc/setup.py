import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CppExtension, CUDAExtension

here = os.path.dirname(os.path.abspath(__file__))
cuda = os.environ.get("WGN_CPU_ONLY", "0") != "1" and torch.cuda.is_available() and CUDA_HOME is not None
if cuda and not os.environ.get("TORCH_CUDA_ARCH_LIST"):
    major, minor = torch.cuda.get_device_capability(0)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "%d.%d" % (major, minor)
src = [os.path.join(here, "water_gn.cpp")] + ([os.path.join(here, "water_gn_cuda.cu")] if cuda else [])
if cuda:
    ext = CUDAExtension("water_gn_ext", src, define_macros=[("WITH_CUDA", None)], extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]})
else:
    ext = CppExtension("water_gn_ext", src, extra_compile_args=["-O3"])
print("[water_gn] building with CUDA" if cuda else "[water_gn] building for the CPU only")
setup(name="water_gn_ext", ext_modules=[ext], cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)})
