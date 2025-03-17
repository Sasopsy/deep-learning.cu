#pragma once
#include <cuda_runtime.h>
#include "naive.cuh"
#include "shared_mem.cuh"
#include "intra_warp.cuh"
#include "../../utils/utils.hpp"
#include "../../common.hpp"

#define SOFTMAX_NUM_KERNELS 3

#ifndef SOFTMAX_DEFAULT_KERNEL
    #define SOFTMAX_DEFAULT_KERNEL 2  // 0=naive, 1=shared_mem, 2=intra_warp
#endif

namespace cuda_kernel::softmax {

void cpu(float* output, const float* input, int N, int C) {
    // input is (N, C)
    // output is (N, C), each row of input will get softmaxed
    for (int i = 0; i < N; i++) {
        const float* input_row = input + i * C;
        float* output_row = output + i * C;

        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) {
            if (input_row[j] > maxval) {
                maxval = input_row[j];
            }
        }
        // Note: since we want to ensure that the CUDA-kernels are accurate,
        // we do this accumulation in higher precision, so we can be assured
        // that our ground-truth is of high quality.
        double sum = 0.0;
        for (int j = 0; j < C; j++) {
            output_row[j] = expf(input_row[j] - maxval);
            sum += output_row[j];
        }
        float norm = 1.f / (float)sum;
        for (int j = 0; j < C; j++) {
            output_row[j] *= norm;
        }
    }
}

template <typename floatX>
void launch(floatX* output, const floatX* input, int N, int C,
            int kernel_choice = SOFTMAX_DEFAULT_KERNEL, cudaStream_t stream = 0) {
    
    // Validate kernel choice with fallback logic
    if(kernel_choice < 0 || kernel_choice >= SOFTMAX_NUM_KERNELS) {
        kernel_choice = SOFTMAX_DEFAULT_KERNEL;
        if(kernel_choice < 0 || kernel_choice >= SOFTMAX_NUM_KERNELS) {
            kernel_choice = 2; // Final fallback to intra_warp
        }
    }

    switch(kernel_choice) {
        case 0:
            launch_naive<floatX>(output, input, N, C, stream);
            break;
            
        case 1:
            launch_shared_mem<floatX>(output, input, N, C, stream);
            break;
            
        case 2:
            launch_intra_warp<floatX>(output, input, N, C, stream);
            break;
    }
}

} // namespace cuda_kernel::softmax
