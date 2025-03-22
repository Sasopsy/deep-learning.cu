# `cuda_kernel::softmax`

Calculates softmax scores of inputs with shape $(...,C)$ across the `-1` dimension:
\[
\mathbf{Y}_{ij} = \frac{e^{\mathbf{X}_{ij}}}{\sum_{k} e^{\mathbf{X}_{ik}}}
\]
where $\mathbf{X} \in \mathbb{R}^{N \times C}$ is the input and $\mathbf{Y} \in \mathbb{R}^{N \times C}$ is the output.

```cpp
#include "kernels/kernels.hpp"

cuda_kernel::softmax::launch<float>(
    output, input, N, C, kernel_choice, stream);
```

### Parameters

- `output`: Pointer to the output matrix.
- `input`: Pointer to the input matrix.
- `N`: Number of rows in the input matrix.
- `C`: Number of columns in the input matrix.
- `kernel_choice`: Index of the kernel to use (0: naive, 1: shared_mem, 2: intra_warp).
- `stream`: CUDA stream to use for the kernel launch.

## Implementations

You can find the different implementations in the [softmax kernels](../../csrc/kernels/softmax) folder.

### Naive Implementation (0)

The naive implementation assigns each thread to compute one element of the output matrix. This is straightforward but may not be the most efficient for larger matrices.

### Shared Memory Implementation (1)

This implementation uses shared memory to cache input tiles, reducing the number of global memory accesses. Each thread block computes a sub-matrix of the output.

### Intra-Warp Implementation (2)

This implementation performs intra-warp reductions to compute the softmax function efficiently. It is optimized for scenarios where the number of columns is large.

## Torch Usage

To use the softmax kernel in a PyTorch extension, call the `softmax_forward` function defined in [`torch_binding.cuh`](../../csrc/kernels/softmax/torch_binding.cuh). The function signature is:

```cpp
torch::Tensor softmax_forward(torch::Tensor input, int kernel_choice = SOFTMAX_DEFAULT_KERNEL);
```

### Python Example

```python
import torch
import dlcu

input = torch.randn(32, 128).cuda()
output = dlcu.softmax_forward(input, kernel_choice=2)
```

## Macros

- `SOFTMAX_NUM_KERNELS`: Total number of kernel implementations available.
- `SOFTMAX_DEFAULT_KERNEL`: Default kernel index to use if an invalid index is provided.
- `SOFTMAX_DEFAULT_IWARP_SIZE`: Default warp size for intra-warp implementation.
- `SOFTMAX_DEFAULT_SMEM_SIZE`: Default shared memory size for shared memory implementation.
