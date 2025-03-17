#pragma once
#include <iostream>
#include <cstring>
#include <cuda_runtime.h>
#include <cassert>
#include <omp.h>
#include "../functions/reduction.cuh"
#include "../../utils/utils.hpp"
#include "../../common.hpp"

#ifndef SOFTMAX_DEFAULT_IWARP_SIZE
    #define SOFTMAX_DEFAULT_IWARP_SIZE 256  // Must be multiple of WARP_SIZE
#endif

namespace cuda_kernel::softmax {

namespace cf = cuda_functions;

template <typename floatX, int mem_size>
__global__ void intra_warp(floatX* output, const floatX* input, int N, int C) {

    static_assert(mem_size % WARP_SIZE == 0, "mem_size must be a multiple of WARP_SIZE");

    __shared__ float shared[mem_size/WARP_SIZE];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / WARP_SIZE; // warp index within a block
    int laneId = threadIdx.x % WARP_SIZE; // thread index within a warp

    // the number of warps per block. recall that blockDim.x is block_size
    int warpsPerBlock = blockDim.x / WARP_SIZE;

    // shared[] must be allocated to have warpsPerBlock elements
    // those will be used for max and sum values
    float* max_or_sum_storage = shared;

    // one row of input, i.e. input[idx, :] of shape (C,)
    const floatX* x = input + idx * C;

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x) {
        maxval = fmaxf(maxval, floatX_to_float<floatX>(x[i]));
    }
    // now within-warp reductions for maxval
    maxval = cf::warp_reduce_max<float>(maxval);

    // the 0th thread of each warp writes the maxval of that warp to shared memory
    if (laneId == 0) max_or_sum_storage[warpId] = maxval;
    __syncthreads();

    // now the 0th thread of the block reduces the max values in shared memory, i.e. across warps
    if (tid == 0) {
        float val = max_or_sum_storage[tid];
        for (int i = 1; i < warpsPerBlock; i++) {
            val = fmaxf(val, max_or_sum_storage[i]);
        }
        // store the final max in the first position
        max_or_sum_storage[0] = val;
    }
    __syncthreads();
    // broadcast the max to all threads
    float offset = max_or_sum_storage[0];

    // compute expf and write the result to global memory
    for (int i = tid; i < C; i += blockDim.x) {
        float input_val = floatX_to_float<floatX>(x[i]);
        float val = __expf(input_val - offset);
        output[idx * C + i] = float_to_floatX<floatX>(val);
    }

    // okay now we calculated exp(x - max(x))
    // step 2: sum all the values and divide by the sum

    // thread coarsening for sum
    x = output + idx * C;
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        sumval += floatX_to_float<floatX>(x[i]);
    }
    // within-warp reduction for sumval
    sumval = cf::warp_reduce_sum<float>(sumval);

    // write sumval to shared memory
    if (laneId == 0) max_or_sum_storage[warpId] = sumval;
    __syncthreads();

    // inter-thread reduction of sum
    if (tid == 0) {
        float val = max_or_sum_storage[tid];
        for (int i = 1; i < warpsPerBlock; ++i) {
            val += max_or_sum_storage[i];
        }
        max_or_sum_storage[0] = val;
    }
    __syncthreads();
    // broadcast the sum to all threads
    float sum = max_or_sum_storage[0];

    // divide the whole row by the sum
    for (int i = tid; i < C; i += blockDim.x) {
        float out_val = floatX_to_float<floatX>(x[i]) / sum;
        output[idx * C + i] = float_to_floatX<floatX>(out_val);
    }
}

template <typename floatX, int mem_size = SOFTMAX_DEFAULT_IWARP_SIZE>
void launch_intra_warp(floatX* output, const floatX* input, int N, int C, cudaStream_t stream = 0) {
    static_assert(mem_size % WARP_SIZE == 0, "mem_size must be multiple of WARP_SIZE");
    dim3 gridDim(N);
    intra_warp<floatX, mem_size><<<gridDim, mem_size, 0, stream>>>(output, input, N, C);
    cudaCheckError(cudaGetLastError());
}

}  // namespace cuda_kernel::softmax