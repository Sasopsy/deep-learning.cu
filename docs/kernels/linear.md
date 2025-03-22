# `cuda_kernel::linear`

Performs the following the linear operation analogous to `torch.nn.functional.linear`:
$$
\mathbf{XW^{T} + b}  
$$
where $\mathbf{X} \in \mathbb{R}^{BT \times C}$ is the input, $\mathbf{W} \in \mathbb{R}^{OC \times C}$ is the weight and $\mathbf{b} \in \mathbb{R}^{1 \times C}$ is the bias.

```cpp
#include "kernels/kernels.hpp"

cuda_kernel::linear::launch<float>(
    input, weight, bias, output, BT, C, OC, kernel_choice, stream);
```

### Parameters

- `input`: Pointer to the input matrix.
- `weight`: Pointer to the weight matrix.
- `bias`: Pointer to the bias vector (optional).
- `output`: Pointer to the output matrix.
- `BT`: Total number of elements in the batch.
- `C`: Number of input channels.
- `OC`: Number of output channels.
- `kernel_choice`: Index of the kernel to use (0: naive, 1: shared_mem, 2: blocktiling_2d).
- `stream`: CUDA stream to use for the kernel launch.

## Implementations

### Naive Implementation (0)

You can find the different implementations in the [linear kernels](../../csrc/kernels/linear) folder.

The naive implementation assigns each thread to compute one element of the output matrix. This is straightforward but may not be the most efficient for larger matrices.

### Shared Memory Implementation (1)

This implementation uses shared memory to cache input and weight tiles, reducing the number of global memory accesses. Each thread block computes a sub-matrix of the output.

### Block Tiling 2D Implementation (2)

This implementation divides the computation into 2D tiles, with each thread block computing a tile of the output matrix. It uses shared memory to cache input and weight tiles, optimizing memory access patterns.

## Torch Usage

To use the linear kernel in a PyTorch extension, call the `linear_forward` function defined in [`torch_binding.cuh`](../../csrc/kernels/linear/torch_binding.cuh) in the . The function signature is:

```cpp
torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int kernel_choice = LINEAR_DEFAULT_KERNEL);
```

### Example

```python
import torch
import dlcu

input = torch.randn(32, 128, 64).cuda()
weight = torch.randn(256, 64).cuda()
bias = torch.randn(256).cuda()

output = dlcu.linear_forward(input, weight, bias, kernel_choice=2)
```

## Macros

- `LINEAR_NUM_KERNELS`: Total number of kernel implementations available.
- `LINEAR_DEFAULT_KERNEL`: Default kernel index to use if an invalid index is provided.
- `LINEAR_DEFAULT_BC`: Default block size for input channels.
- `LINEAR_DEFAULT_BOC`: Default block size for output channels.
- `LINEAR_DEFAULT_BBT`: Default block size for batch elements.
- `LINEAR_DEFAULT_TBT`: Default tile size for batch elements.
- `LINEAR_DEFAULT_TOC`: Default tile size for output channels.
- `LINEAR_DEFAULT_BLOCK_SIZE`: Default block size for shared memory implementation.

