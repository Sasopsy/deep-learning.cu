#pragma once
#include <cuda_runtime.h>
#include "naive.cuh"
#include "shared_mem.cuh"
#include "blocktiling_2d.cuh"
#include "../../utils/utils.hpp"
#include "../../common.hpp"

#define LINEAR_NUM_KERNELS 3

#ifndef LINEAR_DEFAULT_KERNEL
    #define LINEAR_DEFAULT_KERNEL 2  // 0=naive, 1=shared_mem, 2=blocktiling_2d
#endif

namespace cuda_kernel::linear {

void cpu(const float* input,
        const float* weight,
        const float* bias,
        float* output, 
        int BT, int C, int OC) {;
    // out is (B,T,OC), inp is (B,T,C), weight is (OC, C)
    for (uint bt = 0; bt < BT; bt++) {
        float* out_bt = output + bt * OC;
        const float* input_bt = input + bt * C;
        for (uint o = 0; o < OC; o++) {
            float val = (bias != nullptr) ? bias[o] : 0.0f;
            const float* wrow = weight + o * C;
            for (uint i = 0; i < C; i++) {
                val += input_bt[i] * wrow[i];
            }
            out_bt[o] = val;
        }
    }
}

template <typename floatX>
void launch(const floatX* input, const floatX* weight, const floatX* bias,
           floatX* output, int BT, int C, int OC,
           int kernel_choice = LINEAR_DEFAULT_KERNEL, cudaStream_t stream = 0) {
    
    // Validate kernel choice and fallback to default if invalid
    if(kernel_choice < 0 || kernel_choice >= LINEAR_NUM_KERNELS) {
        kernel_choice = LINEAR_DEFAULT_KERNEL;
        // Final fallback if LINEAR_DEFAULT_KERNEL was invalid
        if(kernel_choice < 0 || kernel_choice >= LINEAR_NUM_KERNELS) {
            kernel_choice = 2;
        }
        std::cout << "Invalid kernel choice. Falling back to default kernel: " << kernel_choice << std::endl;
    }

    switch(kernel_choice) {
        case 0:
            launch_naive<floatX>(input, weight, bias, output, BT, C, OC, stream);
            break;
            
        case 1:
            launch_shared_mem<floatX>(input, weight, bias, output, BT, C, OC, stream);
            break;
            
        case 2:
            launch_blocktiling_2d<floatX>(input, weight, bias, output, BT, C, OC, stream);
            break;
    }
}

} // namespace cuda_kernel::linear
