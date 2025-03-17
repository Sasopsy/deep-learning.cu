#include "../../../utils/utils.hpp"
#include "../../../common.hpp"
#include "naive.cuh"
#include "warp_divergence.cuh"

#define POSENC_FUSED_NUM_KERNELS 2

#ifndef POSENC_FUSED_DEFAULT_KERNEL
    #define POSENC_FUSED_DEFAULT_KERNEL 1  // 0=naive 1=warp_divergence
#endif

namespace cuda_kernel::position_encoder::fused {

void cpu(float* input, float* output, int B, int T, int d, int start){
    for (int b = 0; b < B; b++){
        for (int i = 0; i < T; i++){
            for (int j = 0; j < d; j++){
                if (j % 2 == 0){
                    output[b * T * d + i * d + j] = input[b * T * d + i * d + j] + sin(i / pow(10000, j / d));
                } else {
                    output[b * T * d + i * d + j] = input[b * T * d + i * d + j] +  cos(i / pow(10000, j / d));
                }
            }
        }
    }
}

template <typename floatX>
void launch(floatX* input, floatX* output, int B, int T, int d, int start = 0, int kernel_choice = POSENC_FUSED_DEFAULT_KERNEL, cudaStream_t stream = 0) {
    if (kernel_choice < 0 || kernel_choice >= POSENC_FUSED_NUM_KERNELS) {
        kernel_choice = POSENC_FUSED_DEFAULT_KERNEL;
    }

    switch (kernel_choice) {
        case 0:
            launch_naive<floatX>(input, output, B, T, d, start, stream);
            break;
        case 1:
            launch_warp_divergence<floatX>(input, output, B, T, d, start, stream);
            break;
    }
}

}