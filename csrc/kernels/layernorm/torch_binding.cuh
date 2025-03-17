#include <torch/extension.h>
#include "../../utils/torch_utils.hpp"
#include "../../common.hpp"
#include "launch.cuh"

torch::Tensor layernorm_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int kernel_choice = LAYERNORM_DEFAULT_KERNEL) {
    check_tensor(input);  
    check_tensor(weight);
    check_tensor(bias);

    int N = input.numel() / input.size(-1);
    int C = input.size(-1);
    auto output = torch::empty_like(input);

    cuda_kernel::layernorm::launch<float>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        N, C, kernel_choice);

    return output;
}

