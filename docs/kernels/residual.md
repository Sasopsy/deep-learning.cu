# `cuda_kernel::residual`

Performs the residual addition operation analogous to `torch.add`:
$$\mathbf{Y} = \mathbf{X} + \mathbf{R}$$
where $\mathbf{X} \in \mathbb{R}^{N \times C}$ is the input and $\mathbf{R} \in \mathbb{R}^{N \times C}$ is the residual.

```cpp
#include "kernels/kernels.hpp"

cuda_kernel::residual::launch<float>(
    output, input, residual, size, kernel_choice, stream);
```

### Parameters

- `output`: Pointer to the output matrix.
- `input`: Pointer to the input matrix.
- `residual`: Pointer to the residual matrix.
- `size`: Total number of elements.
- `kernel_choice`: Index of the kernel to use (0: naive, 1: vectorised).
- `stream`: CUDA stream to use for the kernel launch.

## Implementations

You can find the different implementations in the [residual kernels](../../csrc/kernels/residual) folder.

### Naive Implementation (0)

The naive implementation assigns each thread to compute one element of the output matrix. This is straightforward but may not be the most efficient for larger matrices.

### Vectorised Implementation (1)

This implementation uses vectorized operations to perform the residual addition efficiently. It is optimized for scenarios where the number of elements is large.

## Torch Usage

To use the residual kernel in a PyTorch extension, call the `residual_forward` function defined in [`torch_binding.cuh`](../../csrc/kernels/residual/torch_binding.cuh). The function signature is:

```cpp
torch::Tensor residual_forward(torch::Tensor input, torch::Tensor residual, int kernel_choice = RESIDUAL_DEFAULT_KERNEL);
```

### Example

```python
import torch
import dlcu

input = torch.randn(32, 128, 64).cuda()
residual = torch.randn(32, 128, 64).cuda()

output = dlcu.residual_forward(input, residual, kernel_choice=1)
```

## Macros

- `RESIDUAL_NUM_KERNELS`: Total number of kernel implementations available.
- `RESIDUAL_DEFAULT_KERNEL`: Default kernel index to use if an invalid index is provided.
