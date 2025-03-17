#pragma once
#include <iostream>
#include <cstring>
#include <cuda_runtime.h>
#include <cassert>
#include <omp.h>
#include "../functions/cuda_functions.cuh"
#include "../../utils/utils.hpp"
#include "../../common.hpp"

namespace cuda_kernel::layernorm {

template <typename floatX>
__global__ void naive(floatX* output, const floatX* input, const floatX* weight, const floatX* bias, 
                        int N, int C) {

    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    floatX eps = CUDA_EPSILON;

    if (idx < N) {
        // seek to the inputut position input[idx,:]
        const floatX* x = input + idx * C;
        // calculate the mean
        floatX m = 0.0f;
        for (uint i = 0; i < C; i++) {
            m += x[i];
        }
        m = m / C;
        // calculate the variance (withoutput any bias correction)
        floatX v = 0.0f;
        for (uint i = 0; i < C; i++) {
            floatX xshift = x[i] - m;
            v += xshift * xshift;
        }
        v = v / C;
        // calculate the rstd
        floatX s = 1.0f / sqrtf(v + eps);
        // seek to the outputput position in output[idx,:]
        floatX* output_idx = output + idx * C;
        for (uint i = 0; i < C; i++) {
            floatX n = (s * (x[i] - m)); // normalized outputput
            floatX o = n * weight[i] + bias[i]; // scale and shift it
            output_idx[i] = o; // write
        }
    }
}

template <typename floatX>
void launch_naive(floatX* output, const floatX* input, const floatX* weight, const floatX* bias,
                  int N, int C, cudaStream_t stream = 0) {
    const int block_size = 256;
    dim3 gridDim(ceil_div<int>(N, block_size));
    naive<floatX><<<gridDim, block_size, 0, stream>>>(output, input, weight, bias, N, C);
    cudaCheckError(cudaGetLastError());
}

}  // namespace cuda_kernel::layernorm