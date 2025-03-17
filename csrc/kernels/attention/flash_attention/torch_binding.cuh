#include <torch/extension.h>
#include "../../../utils/torch_utils.hpp"
#include "../../../common.hpp"
#include "launch.cuh"


torch::Tensor flash_attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, float softmax_scale, int kernel_choice = FLASH_ATTENTION_DEFAULT_KERNEL) {
    // Check tensor type and device
    check_tensor(Q);
    check_tensor(K);
    check_tensor(V);

    // Check whether B, H, and d are the same for Q, K, and V
    // And display shapes if they don't match
    if (Q.size(0) != K.size(0) || Q.size(0) != V.size(0) || Q.size(1) != K.size(1) || Q.size(1) != V.size(1) || Q.size(3) != K.size(3) || Q.size(3) != V.size(3)) {
        std::string error_message = "Batch size, number of heads, or embedding dimension do not match for Q, K, and V! ";
        error_message += "Q: " + std::to_string(Q.size(0)) + "x" + std::to_string(Q.size(1)) + "x" + std::to_string(Q.size(3)) + ", ";
        error_message += "K: " + std::to_string(K.size(0)) + "x" + std::to_string(K.size(1)) + "x" + std::to_string(K.size(3)) + ", ";
        error_message += "V: " + std::to_string(V.size(0)) + "x" + std::to_string(V.size(1)) + "x" + std::to_string(V.size(3));
        throw std::invalid_argument(error_message);
    }

    // Check tensor shapes
    const int B = Q.size(0);  // batch size
    const int H = Q.size(1);  // number of heads
    const int seq_len_q = Q.size(2);  // sequence length of query
    const int seq_len_kv = K.size(2);  // sequence length of key/value
    const int d = Q.size(3);  // embedding dimension
    
    // Create output tensor with same shape as Q
    auto O = torch::zeros_like(Q);
    
    // Calculate softmax scale if softmax_scale is not provided
    if (softmax_scale == 0.0f) {
        softmax_scale = 1.0f / sqrt(d);
    }
    
    // Launch float32 kernel only
    cuda_kernel::attention::flash_attention::launch<float>(
        Q.data_ptr<float>(), 
        K.data_ptr<float>(), 
        V.data_ptr<float>(),
        O.data_ptr<float>(), 
        softmax_scale, 
        B, H, seq_len_q, seq_len_kv, d
    );
    
    return O;
}
