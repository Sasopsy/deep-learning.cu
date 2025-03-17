#include <torch/extension.h>
#include "../../utils/torch_utils.hpp"
#include "../../common.hpp"
#include "launch.cuh"


torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int kernel_choice = LINEAR_DEFAULT_KERNEL) {
    check_tensor(input);
    check_tensor(weight);
    check_tensor(bias);

    // Get the last dimension size (input features)
    int C = input.size(-1);
    int OC = weight.size(0);

    if (weight.size(1) != C) {
        throw std::invalid_argument("Input channels and weight channels must match!");
    }

    // Calculate batch size (total elements / feature size)
    int BT = input.numel() / input.size(-1);

    // Create output shape by replacing the last dimension
    auto output_sizes = input.sizes().vec();
    output_sizes.back() = OC;
    
    // Create output tensor with proper dimensions
    auto output = torch::empty(output_sizes, input.options());

    cuda_kernel::linear::launch<float>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        BT, C, OC, kernel_choice);

    return output;
}