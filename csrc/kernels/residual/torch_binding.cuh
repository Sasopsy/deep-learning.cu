#include <torch/extension.h>
#include "../../utils/torch_utils.hpp"
#include "../../common.hpp"
#include "launch.cuh"

torch::Tensor residual_forward(torch::Tensor input, torch::Tensor residual, int kernel_choice = RESIDUAL_DEFAULT_KERNEL) {
    check_tensor(input);
    check_tensor(residual);

    int size = input.numel();
    auto output = torch::empty_like(input);

    cuda_kernel::residual::launch<float>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        residual.data_ptr<float>(),
        size, kernel_choice);

    return output;
}