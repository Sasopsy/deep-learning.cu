#include <torch/extension.h>
#include "../../utils/torch_utils.hpp"
#include "../../common.hpp"
#include "launch.cuh"


torch::Tensor softmax_forward(torch::Tensor input, int kernel_choice = SOFTMAX_DEFAULT_KERNEL) {
    check_tensor(input);

    int N = input.numel() / input.size(-1);
    int C = input.size(-1);
    auto output = torch::empty_like(input);

    cuda_kernel::softmax::launch<float>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        N, C, kernel_choice);

    return output;
}
