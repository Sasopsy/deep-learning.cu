#include "../../../utils/utils.hpp"
#include "../../../common.hpp"
#include "naive.cuh"
#include "warp_divergence.cuh"

#define POSENC_PREFILL_NUM_KERNELS 2

#ifndef POSENC_PREFILL_DEFAULT_KERNEL
    #define POSENC_PREFILL_DEFAULT_KERNEL 1  // 0=naive 1=warp_divergence
#endif

namespace cuda_kernel::position_encoder::prefill{

void cpu(float* pos_enc, int B, int T, int d, int start){
    for (int b = 0; b < B; b++) {
        for (int t = start; t < start + T; t++){
            for (int i = 0; i < d; i++){
                if (i % 2 == 0){
                    pos_enc[b * T * d + (t - start) * d + i] = sin(t / pow(10000, i / d));
                } else {
                    pos_enc[b * T * d + (t - start) * d + i] = cos(t / pow(10000, i / d));
                }
            }
        }
    }
}

template <typename floatX>
void launch(floatX* pos_enc, int B, int T, int d, int start = 0, int kernel_choice = POSENC_PREFILL_DEFAULT_KERNEL, cudaStream_t stream = 0) {
    if (kernel_choice < 0 || kernel_choice >= POSENC_PREFILL_NUM_KERNELS) {
        kernel_choice = POSENC_PREFILL_DEFAULT_KERNEL;
    }

    switch (kernel_choice) {
        case 0:
            launch_naive<floatX>(pos_enc, B, T, d, start, stream);
            break;
        case 1:
            launch_warp_divergence<floatX>(pos_enc, B, T, d, start, stream);
            break;
    }
}

}