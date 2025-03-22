# Deep Learning CUDA Extensions

This repository contains CUDA implementations of various deep learning operations with Python bindings for PyTorch.

## Implemented Kernels (Tested)

- [LayerNorm](csrc/kernels/layernorm)
- [Linear](csrc/kernels/linear) ($\mathbf{XW^{T}}$)
- [Softmax](csrc/kernels/softmax)
- [Residual](csrc/kernels/residual)
- [Flash Attention](csrc/kernels/attention/flash_attention)

Check out the [docs](docs/kernels) for more details.

## Installation

### Requirements
- CUDA Toolkit (11.0+)
- PyTorch (2.0+)
- C++17 compatible compiler

### Installing from Source

```bash
# Clone the repository
git clone https://github.com/Sasopsy/deep-learning.cu.git
cd deep-learning.cu

# Install the package
python setup.py install
```

## Usage

```python
import torch
import dlcu

# LayerNorm example
input_tensor = torch.randn(32, 512, 768, device='cuda')
weight = torch.randn(768, device='cuda')
bias = torch.randn(768, device='cuda')
output = dlcu.layernorm_forward(input_tensor, weight, bias)

# Flash Attention example
q = torch.randn(1, 8, 1024, 64, device='cuda')
k = torch.randn(1, 8, 1024, 64, device='cuda')
v = torch.randn(1, 8, 1024, 64, device='cuda')
output = dlcu.flash_attention_forward(q, k, v)
```

## Testing

See [PyTorch tests README](tests/torch/README.md) and [C++ tests README](tests/cpp/README.md) for testing instructions.


