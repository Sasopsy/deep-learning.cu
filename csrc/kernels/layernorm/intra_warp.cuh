#pragma once
#include <iostream>
#include <cstring>
#include <cuda_runtime.h>
#include <cassert>
#include <omp.h>
#include "../functions/cuda_functions.cuh"
#include "../../utils/utils.hpp"
#include "../../common.hpp"

#ifndef LAYERNORM_DEFAULT_IWARP_SIZE
    #define LAYERNORM_DEFAULT_IWARP_SIZE 256  // Must be multiple of WARP_SIZE
#endif

namespace cuda_kernel::layernorm {

namespace cf = cuda_functions;

template <typename floatX, int mem_size>
__global__ void intra_warp(floatX* output, const floatX* input, const floatX* weight, const floatX* bias, 
                            int N, int C) {
    // output is (N, C) just like input.
    // mem_size must be multiple of WARP_SIZE
    // each row of C elements is handled by block_size threads
    // furthermore, each block_size threads get executed in warps of WARP_SIZE threads
    // special reduction operations warp_reduce_max/warp_reduce_sum are used for intra-warp reductions
    // shared memory is used for inter-warp reduction

    static_assert(mem_size % WARP_SIZE == 0, "mem_size must be a multiple of WARP_SIZE");

    __shared__ float shared_mean_rstd[mem_size/WARP_SIZE];
    uint block_size = blockDim.x;
    uint idx = blockIdx.x;
    uint tid = threadIdx.x;
    uint warpId = threadIdx.x / WARP_SIZE; // warp index within a block
    uint laneId = threadIdx.x % WARP_SIZE; // thread index within a warp

    uint warpsPerBlock = blockDim.x / WARP_SIZE;

    // Get one row of inputut
    const floatX* x = input + idx * C;

    // First mean calculation with thread coarsening
    float sum = 0.0f;
    for (uint i = tid; i < C; i += block_size) {
        sum += floatX_to_float<floatX>(x[i]);
    }
    // now intra warp reductions
    sum = cf::warp_reduce_sum<float>(sum);

    if (laneId == 0) {
        shared_mean_rstd[warpId] = sum;
    }
    __syncthreads();

    // The O'th thread of the block reduces the max values in shared memory, i.e. across warps
    if (tid == 0) {
        float m = 0.0f;
        for (uint i = 0; i < warpsPerBlock; i++) {
            m += shared_mean_rstd[i];
        }
        m = m / C;
        shared_mean_rstd[0] = m;
    }
    __syncthreads();
    
    // Broadcast mean to all threads
    float m = shared_mean_rstd[0];

    // First rstd calculation with thread coarsening
    float diffSquaredSum = 0.0f;
    for (uint i = tid; i < C; i += block_size) {
        float diff = floatX_to_float<floatX>(x[i]) - m;
        diffSquaredSum += diff * diff;
    }

    // Now intra warp reductions
    diffSquaredSum = cf::warp_reduce_sum<float>(diffSquaredSum);

    if (laneId == 0) {
        shared_mean_rstd[warpId] = diffSquaredSum;
    }
    __syncthreads();
    
    // The O'th thread of the block reduces the max values in shared memory, i.e. across warps
    if (tid == 0) {
        float s = 0.0f;
        for (uint i = 0; i < warpsPerBlock; i++) {
            s += shared_mean_rstd[i];
        }
        s = rsqrtf(s / C + CUDA_EPSILON);
        shared_mean_rstd[0] = s;
    }
    __syncthreads();

    float s = floatX_to_float<floatX>(shared_mean_rstd[0]);

    // Calculate normalized outputputs
    floatX* output_row = output + idx * C;
    for (uint i = tid; i < C; i += block_size) {
        float output = (s * (floatX_to_float<floatX>(x[i]) - m));
        output_row[i] = output * weight[i] + bias[i];
    }

}

template <typename floatX, int mem_size = LAYERNORM_DEFAULT_IWARP_SIZE>
void launch_intra_warp(floatX* output, const floatX* input, const floatX* weight, const floatX* bias,
                        int N, int C, cudaStream_t stream = 0) {
    static_assert(mem_size % WARP_SIZE == 0, "mem_size must be multiple of WARP_SIZE");
    dim3 gridDim(N);
    intra_warp<floatX, mem_size><<<gridDim, mem_size, 0, stream>>>(output, input, weight, bias, N, C);
    cudaCheckError(cudaGetLastError());
}

}  // namespace cuda_kernel::layernorm