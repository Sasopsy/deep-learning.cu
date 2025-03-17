#pragma once
#include <cuda_runtime.h>
#include "naive.cuh"
#include "shared_mem.cuh"
#include "intra_warp.cuh"
#include "variance_estimate.cuh"
#include "../../utils/utils.hpp"
#include "../../common.hpp"

#define LAYERNORM_NUM_KERNELS 4

#ifndef LAYERNORM_DEFAULT_KERNEL
    #define LAYERNORM_DEFAULT_KERNEL 3  // 0=naive, 1=shared_mem, 2=intra_warp, 3=variance_estimate
#endif

namespace cuda_kernel::layernorm {

void cpu(float* output, const float* input, const float* weight, const float* bias,
    int N, int C) {
// input is (B, T, C)
// output is (B, T, C)
// mean is (B, T)
// rstd is (B, T)
// weight is (C)
// bias is (C)

float eps = CUDA_EPSILON;

    for (int n = 0; n < N; n++) {
        // seek to the inputut position input[n,:]
        const float* x = input + n * C;

        // calculate the mean
        float m = 0.0f;
        for (int i = 0; i < C; i++) {
            m += x[i];
        }
        m = m/C;

        // calculate the variance (withoutput any bias correction)
        float v = 0.0f;
        for (int i = 0; i < C; i++) {
            float xshift = x[i] - m;
            v += xshift * xshift;
        }
        v = v/C;

        // calculate the rstd
        float s = 1.0f / sqrtf(v + eps);

        // seek to the outputput position in output[n,:]
        float* output_bt = output + n * C;

        for (int i = 0; i < C; i++) {
            float norm = (s * (x[i] - m)); // normalized outputput
            float o = norm * weight[i] + bias[i]; // scale and shift it
            output_bt[i] = o; // write
        }
    }
}

template <typename floatX>
void launch(floatX* output, const floatX* input, const floatX* weight, const floatX* bias,
            int N, int C, int kernel_choice = LAYERNORM_DEFAULT_KERNEL, cudaStream_t stream = 0) {

    // Validate kernel choice with fallback logic
    if(kernel_choice < 0 || kernel_choice >= LAYERNORM_NUM_KERNELS) {
        kernel_choice = LAYERNORM_DEFAULT_KERNEL;
        if(kernel_choice < 0 || kernel_choice >= LAYERNORM_NUM_KERNELS) {
            kernel_choice = 3; // Fallback to variance estimate
        }
    }

    switch(kernel_choice) {
        case 0:
            launch_naive<floatX>(output, input, weight, bias, N, C, stream);
            break;
            
        case 1:
            launch_shared_mem<floatX>(output, input, weight, bias, N, C, stream);
            break;
            
        case 2:
            launch_intra_warp<floatX>(output, input, weight, bias, N, C, stream);
            break;
            
        case 3:
            launch_variance_estimate<floatX>(output, input, weight, bias, N, C, stream);
            break;
    }
}

} // namespace cuda_kernel::layernorm

