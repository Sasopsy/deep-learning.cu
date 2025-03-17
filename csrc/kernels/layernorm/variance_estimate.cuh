#pragma once
#include <iostream>
#include <cstring>
#include <cuda_runtime.h>
#include <cassert>
#include <omp.h>
#include "../functions/cuda_functions.cuh"
#include "../../utils/utils.hpp"
#include "../../common.hpp"

#ifndef LAYERNORM_DEFAULT_IWARP_SIZE_VARIANCE_ESTIMATE
    #define LAYERNORM_DEFAULT_IWARP_SIZE_VARIANCE_ESTIMATE 256  // Must be multiple of WARP_SIZE
#endif

namespace cuda_kernel::layernorm {

namespace cf = cuda_functions;

template <typename floatX, int mem_size>
__global__ void variance_estimate(floatX* output, const floatX* input, const floatX* weight, const floatX* bias, 
                                  int N, int C){
    // Just like intra warp but variance is calculated using 
    // Var(x) = E(x**2) - E(x)**2
    // Only single pass over the data is needed, which is great!

    static_assert(mem_size % WARP_SIZE == 0, "block_size must be a multiple of WARP_SIZE");

    __shared__ float shared_mean[mem_size/WARP_SIZE];
    __shared__ float shared_mean_squared[mem_size/WARP_SIZE];
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
    float sum_squared = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        float val = floatX_to_float<floatX>(x[i]);
        sum += val;
        sum_squared += val * val;

    }
    // now intra warp reductions
    sum = cf::warp_reduce_sum<float>(sum);
    sum_squared = cf::warp_reduce_sum<float>(sum_squared);

    if (laneId == 0) {
        shared_mean[warpId] = sum;
        shared_mean_squared[warpId] = sum_squared;
    }
    __syncthreads();

    // The O'th thread of the block reduces the max values in shared memory, i.e. across warps
    if (tid == 0) {
        float m = 0.0f;
        float m2 = 0.f;
        for (uint i = 0; i < warpsPerBlock; i++) {
            m += shared_mean[i];
            m2 += shared_mean_squared[i];
        }
        m = m / C;
        m2 = m2 / C; 
        shared_mean[0] = m;
        shared_mean_squared[0] = m2;
    }
    __syncthreads();
    
    // Broadcast mean to all threads
    float m = shared_mean[0];
    float m2 = shared_mean_squared[0];
    float s = rsqrtf(m2 - m * m + CUDA_EPSILON);

    // Calculate normalized outputputs
    floatX* output_row = output + idx * C;
    for (uint i = tid; i < C; i += block_size) {
        float output = (s * (floatX_to_float<floatX>(x[i]) - m));
        output_row[i] = output * weight[i] + bias[i];
    }
}

template <typename floatX, int mem_size = LAYERNORM_DEFAULT_IWARP_SIZE_VARIANCE_ESTIMATE>
void launch_variance_estimate(floatX* output, const floatX* input, const floatX* weight, const floatX* bias,
                                int N, int C, cudaStream_t stream = 0) {
    static_assert(mem_size % WARP_SIZE == 0, "mem_size must be multiple of WARP_SIZE");
    dim3 gridDim(N);
    variance_estimate<floatX, mem_size><<<gridDim, mem_size, 0, stream>>>(output, input, weight, bias, N, C);
    cudaCheckError(cudaGetLastError());
}
}