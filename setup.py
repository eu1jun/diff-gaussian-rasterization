from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

# Get absolute path to the glm directory in the local-libs
glm_local_path = os.path.expanduser('~/glm')

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "ext.cpp"],
            extra_compile_args={
                "cxx": ["-std=c++17"],
                "nvcc": [
                    "-Xcompiler", "-fno-gnu-unique",
                    "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"),
                    "-I" + glm_local_path,  # Use expanded home directory
                    "-gencode", "arch=compute_75,code=sm_75",
                    "-lineinfo"
                ]
            })
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
