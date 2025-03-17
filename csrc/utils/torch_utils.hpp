#pragma once
#include <torch/extension.h>

void check_tensor(torch::Tensor tensor) {
    if (tensor.scalar_type() != torch::kFloat32) {
        throw std::invalid_argument("Tensor must be of type float32!");
    }
    if (!tensor.is_cuda()) {
        throw std::invalid_argument("Tensor must be on CUDA!");
    }
}