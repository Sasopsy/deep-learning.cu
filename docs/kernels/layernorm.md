# `cuda_kernel::layernorm`

Performs layer normalization analogous with inputs of shape $(...,C)$ across the `-1` dimension analogous to `torch.nn.functional.layer_norm`:

$$\mathbf{Y} = \frac{\mathbf{X} - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

where $\mathbf{X} \in \mathbb{R}^{N \times C}$ is the input, $\mu \in \mathbb{R}^{N}$ is the mean, $\sigma^2 \in \mathbb{R}^{N}$ is the variance, $\epsilon$ is a small constant for numerical stability, $\gamma \in \mathbb{R}^{C}$ is the weight, and $\beta \in \mathbb{R}^{C}$ is the bias.

```cpp
#include "kernels/kernels.hpp"

cuda_kernel::layernorm::launch<float>(
    output, input, weight, bias, N, C, kernel_choice, stream);
```

### Parameters

- `output`: Pointer to the output matrix.
- `input`: Pointer to the input matrix.
- `weight`: Pointer to the weight vector.
- `bias`: Pointer to the bias vector.
- `N`: Number of rows in the input matrix.
- `C`: Number of columns in the input matrix.
- `kernel_choice`: Index of the kernel to use (0: naive, 1: shared_mem, 2: intra_warp, 3: variance_estimate).
- `stream`: CUDA stream to use for the kernel launch.

## Implementations

You can find the different implementations in the [layernorm kernels](../../csrc/kernels/layernorm) folder.

### Naive Implementation (0)

The naive implementation assigns each thread to compute one element of the output matrix. This is straightforward but may not be the most efficient for larger matrices.

### Shared Memory Implementation (1)

This implementation uses shared memory to cache input tiles, reducing the number of global memory accesses. Each thread block computes a sub-matrix of the output.

### Intra-Warp Implementation (2)

This implementation performs intra-warp reductions to compute the layer normalization function efficiently. It is optimized for scenarios where the number of columns is large.

### Variance Estimate Implementation (3)

This implementation calculates the variance using the formula $\text{Var}(x) = \mathbb{E}[x^2] - \mathbb{E}[x]^2$, requiring only a single pass over the data.

## Torch Usage

To use the layernorm kernel in a PyTorch extension, call the `layernorm_forward` function defined in [`torch_binding.cuh`](../../csrc/kernels/layernorm/torch_binding.cuh). The function signature is:

```cpp
torch::Tensor layernorm_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int kernel_choice = LAYERNORM_DEFAULT_KERNEL);
```

### Example

```python
import torch
import dlcu

input = torch.randn(32, 128, 64).cuda()
weight = torch.randn(64).cuda()
bias = torch.randn(64).cuda()

output = dlcu.layernorm_forward(input, weight, bias, kernel_choice=2)
```

## Macros

- `LAYERNORM_NUM_KERNELS`: Total number of kernel implementations available.
- `LAYERNORM_DEFAULT_KERNEL`: Default kernel index to use if an invalid index is provided.
- `LAYERNORM_DEFAULT_IWARP_SIZE`: Default warp size for intra-warp implementation.
- `LAYERNORM_DEFAULT_SMEM_SIZE`: Default shared memory size for shared memory implementation.
- `LAYERNORM_DEFAULT_IWARP_SIZE_VARIANCE_ESTIMATE`: Default warp size for variance estimate implementation.
