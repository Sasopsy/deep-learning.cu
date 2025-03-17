#pragma once
#include <iostream>
#include <cstring>
#include <cuda_runtime.h>
#include <cassert>
#include <omp.h>
#include "../functions/cuda_functions.cuh"
#include "../../utils/utils.hpp"
#include "../../common.hpp"

#ifndef LAYERNORM_DEFAULT_SMEM_SIZE
    #define LAYERNORM_DEFAULT_SMEM_SIZE 512  // Must be power of two
#endif

namespace cuda_kernel::layernorm {

namespace cf = cuda_functions;

template <typename floatX, int mem_size>
__global__ void shared_mem(floatX* output, const floatX* input, const floatX* weight, const floatX* bias, 
                            int N, int C){
    // input is (N, C)
    // in each row of C elements, firs calculates mean, then rstd and returns normalized outputput

    __shared__ float shared_mean_rstd[mem_size];
    uint idx = blockIdx.x; // ranges [0, N)
    uint tid = threadIdx.x; // ranges [0, block_size)
    uint block_size = blockDim.x;
    const floatX* x = input + idx * C;

    // thread coarsening for shared_mean_rstd
    float sum = 0.0f;
    for (uint i = tid; i < C; i += block_size) {
        sum += floatX_to_float<floatX>(x[i]);
    }
    shared_mean_rstd[tid] = sum;

    // reductions
    for (uint stride = block_size / 2; stride > 0; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared_mean_rstd[tid] += shared_mean_rstd[tid + stride];
        }
    }
    __syncthreads();

    float m = shared_mean_rstd[0] / C;

    // thread coarsening for rstd
    float diffSquaredSum = 0.0f;
    for (uint i = tid; i < C; i += block_size) {
        float diff = floatX_to_float<floatX>(x[i]) - m;
        diffSquaredSum += diff * diff;
    }
    shared_mean_rstd[tid] = diffSquaredSum;

    //reductions 
    for (uint stride = block_size / 2; stride > 0; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared_mean_rstd[tid] += shared_mean_rstd[tid + stride];
        }
    }
    __syncthreads();

    float s = rsqrtf(shared_mean_rstd[0] / C + CUDA_EPSILON);

    // Calculate the normalized output
    floatX* output_row = output + idx * C;
    for (uint i = tid; i < C; i += block_size) {
        float output = (s * (floatX_to_float<floatX>(x[i]) - m));
        output_row[i] = output * weight[i] + bias[i];
    }
}

template <typename floatX, int mem_size = LAYERNORM_DEFAULT_SMEM_SIZE>
void launch_shared_mem(floatX* output, const floatX* input, const floatX* weight, const floatX* bias,
                        int N, int C, cudaStream_t stream = 0) {
    static_assert(mem_size >= WARP_SIZE && (mem_size & (mem_size - 1)) == 0,
                    "mem_size must be power of two and >=WARP_SIZE");
    dim3 gridDim(N);
    shared_mem<floatX, mem_size><<<gridDim, mem_size, 0, stream>>>(output, input, weight, bias, N, C);
    cudaCheckError(cudaGetLastError());
}

}  // namespace cuda_kernel::layernorm