#pragma once
#include "naive.cuh"
#include "vectorised.cuh"
#include "../../types/pack128.hpp"
#include "../../utils/utils.hpp"
#include "../../common.hpp"

#define RESIDUAL_NUM_KERNELS 2

#ifndef RESIDUAL_DEFAULT_KERNEL
    #define RESIDUAL_DEFAULT_KERNEL 1  // 0: naive, 1: vectorised
#endif

namespace cuda_kernel::residual{

void cpu(float* out, const float* in, const float* residual, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = in[i] + residual[i];
    }
}

template <typename floatX>
void launch(floatX* out, const floatX* in, const floatX* residual, int size, int kernel_choice = RESIDUAL_DEFAULT_KERNEL, cudaStream_t stream = 0) {
    // Fallback to kernel_choice 0 if passed choice is invalid.
    if(kernel_choice < 0 || kernel_choice >= RESIDUAL_NUM_KERNELS) {
         kernel_choice = RESIDUAL_DEFAULT_KERNEL;
    }   
    switch(kernel_choice) {
         case 0:
             launch_naive<floatX>(out, in, residual, size, stream);
             break;
         case 1:
             launch_vectorised<floatX>(out, in, residual, size, stream);
             break;
    }
}

} // namespace cuda_kernel::residual
