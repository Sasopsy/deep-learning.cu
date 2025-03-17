import os
import subprocess
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def get_compute_capability():
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'])
        compute_capability = output.decode('utf-8').strip().replace('.', '')
        return f'sm_{compute_capability}'
    except Exception as e:
        print(f"Error getting compute capability: {e}")
        return 'sm_80'  # Default to sm_80 if unable to get compute capability

compute_capability = get_compute_capability()

# Get PyTorch lib path for RPATH
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')

setup(
    name='dlcu',
    version='0.1.0',
    description='CUDA implementations of deep learning operations with PyTorch bindings',
    author='Your Name',
    author_email='your.email@example.com',
    ext_modules=[
        CUDAExtension(
            name='dlcu',
            sources=['torch_binding.cu'],
            include_dirs=['./csrc/kernels'],
            extra_compile_args={
                'cxx': ['-std=c++17', '-O3', '-fopenmp'],
                'nvcc': [
                    '-std=c++17',
                    '-O3',
                    f'-arch={compute_capability}',
                    '-Xcompiler', '-fopenmp',
                    '-I./csrc/kernels',
                    '--use_fast_math'
                ]
            },
            extra_link_args=[f'-Wl,-rpath,{torch_lib_path}']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires='>=3.7',
    install_requires=[
        'torch>=2.0.0',
    ],
)
