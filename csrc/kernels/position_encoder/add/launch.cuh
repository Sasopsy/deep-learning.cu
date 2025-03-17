#include "../../../utils/utils.hpp"
#include "../../../common.hpp"
#include "naive.cuh"

#define POSENC_ADD_NUM_KERNELS 1

#ifndef POSENC_ADD_DEFAULT_KERNEL
    #define POSENC_ADD_DEFAULT_KERNEL 0  // 0=naive
#endif

namespace cuda_kernel::position_encoder::add {

void cpu(float* pos_enc, float* input, float* output, int B, int T, int d, int start){
    for (int b = 0; b < B; b++){
        for (int i = start; i < start + T; i++){
            for (int j = 0; j < d; j++){
                output[b * T * d + i * d + j] = input[b * T * d + i * d + j] + pos_enc[i * d + j];
            }
        }
    }
}

template <typename floatX>
void launch(floatX* pos_enc, floatX* input, floatX* output, int B, int T, int d, int start = 0, int kernel_choice = POSENC_ADD_DEFAULT_KERNEL, cudaStream_t stream = 0) {
    if (kernel_choice < 0 || kernel_choice >= POSENC_ADD_NUM_KERNELS) {
        kernel_choice = POSENC_ADD_DEFAULT_KERNEL;
    }

    switch (kernel_choice) {
        case 0:
            launch_naive<floatX>(pos_enc, input, output, B, T, d, start, stream);
            break;
    }
}
} // namespace cuda_kernel::position_encoder::add