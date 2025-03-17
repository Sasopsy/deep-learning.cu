#include <torch/extension.h>
#include "csrc/kernels/torch_binding.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layernorm_forward", &layernorm_forward, R"doc(
        LayerNorm forward (CUDA)

        Args:
            input (torch.Tensor): Input tensor.
            weight (torch.Tensor): Weight tensor.
            bias (torch.Tensor): Bias tensor.
            kernel_choice (int, optional): Kernel choice index (default: LAYERNORM_DEFAULT_KERNEL).
                0: naive, 1: shared_mem, 2: intra_warp, 3: variance_estimate.

        Returns:
            torch.Tensor: Output tensor.
    )doc",
    py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("kernel_choice") = LAYERNORM_DEFAULT_KERNEL);

    m.def("linear_forward", &linear_forward, R"doc(
        Linear forward (CUDA)

        Args:
            input (torch.Tensor): Input tensor.
            weight (torch.Tensor): Weight tensor.
            bias (torch.Tensor): Bias tensor.
            kernel_choice (int, optional): Kernel choice index (default: LINEAR_DEFAULT_KERNEL).
                0: naive, 1: shared_mem, 2: blocktiling_2d.

        Returns:
            torch.Tensor: Output tensor.
    )doc",
    py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("kernel_choice") = LINEAR_DEFAULT_KERNEL);

    m.def("softmax_forward", &softmax_forward, R"doc(
        Softmax forward (CUDA)

        Args:
            input (torch.Tensor): Input tensor.
            kernel_choice (int, optional): Kernel choice index (default: SOFTMAX_DEFAULT_KERNEL).
                0: naive, 1: shared_mem, 2: intra_warp.

        Returns:
            torch.Tensor: Output tensor.
    )doc",
    py::arg("input"), py::arg("kernel_choice") = SOFTMAX_DEFAULT_KERNEL);

    m.def("residual_forward", &residual_forward, R"doc(
        Residual forward (CUDA)

        Args:
            input (torch.Tensor): Input tensor.
            residual (torch.Tensor): Residual tensor.
            kernel_choice (int, optional): Kernel choice index (default: RESIDUAL_DEFAULT_KERNEL).
                0: naive, 1: vectorised.

        Returns:
            torch.Tensor: Output tensor.
    )doc",
    py::arg("input"), py::arg("residual"), py::arg("kernel_choice") = RESIDUAL_DEFAULT_KERNEL);
    
    m.def("flash_attention_forward", &flash_attention_forward, R"doc(
        Flash Attention forward (CUDA)

        Args:
            Q (torch.Tensor): Query tensor of shape [B, H, seq_len_q, d].
            K (torch.Tensor): Key tensor of shape [B, H, seq_len_kv, d].
            V (torch.Tensor): Value tensor of shape [B, H, seq_len_kv, d].
            softmax_scale (0.0, optional): Scaling factor for softmax. If 0.0, uses 1/sqrt(d).
            kernel_choice (int, optional): Kernel choice index (default: FLASH_ATTENTION_DEFAULT_KERNEL).
                0: fa_2.

        Returns:
            torch.Tensor: Output tensor of shape [B, H, seq_len_q, d].
    )doc",
    py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("softmax_scale") = 0.0f, py::arg("kernel_choice") = FLASH_ATTENTION_DEFAULT_KERNEL);
}
