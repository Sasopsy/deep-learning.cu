#pragma once
#include <iostream>
#include <cstring>
#include <cuda_runtime.h>
#include <cassert>
#include <omp.h>
#include "../functions/reduction.cuh"
#include "../../utils/utils.hpp"
#include "../../common.hpp"

// Configuration macros for shared memory kernels
#ifndef SOFTMAX_DEFAULT_SMEM_SIZE
    #define SOFTMAX_DEFAULT_SMEM_SIZE 512  // Must be power of two and >=WARP_SIZE
#endif

namespace cuda_kernel::softmax {

namespace cf = cuda_functions;

template <typename floatX, int mem_size>
__global__ void shared_mem(floatX* output, const floatX* input, int N, int C) {
    // input is (N, C)
    // in each row of C elements, first calculates maxval, then returns expf(val - maxval)

    __shared__ float shared[mem_size];
    uint idx = blockIdx.x; // ranges [0, N)
    uint tid = threadIdx.x; // ranges [0, block_size)
    uint block_size = blockDim.x;
    const floatX* x = input + idx * C; // idx-th row of input

    // thread coarsening
    float maxval = -INFINITY;
    for (uint i = tid; i < C; i += block_size) {
        maxval = fmaxf(maxval, floatX_to_float<floatX>(x[i]));
    }
    shared[tid] = maxval;

    // reductions
    for (uint stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
    }
    __syncthreads();

    float offset = shared[0];
    // compute expf and write the result to global memory
    for (uint i = tid; i < C; i += block_size) {
        output[idx * C + i] = __expf(floatX_to_float<floatX>(x[i]) - offset);
    }
    __syncthreads();

    // thread coarsening again, for the sum
    x = output + idx * C; // idx-th row of output
    float sumval = 0.0f;
    for (uint i = tid; i < C; i += block_size) {
        sumval += floatX_to_float<floatX>(x[i]);
    }
    shared[tid] = sumval;

    // reductions
    for (uint stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    // broadcast the sum to all threads in the block
    __syncthreads();

    float sum = shared[0];
    // divide the input values by the sum
    for (uint i = tid; i < C; i += block_size) {
        float out_val = floatX_to_float<floatX>(x[i]) / sum;
        output[idx * C + i] = float_to_floatX<floatX>(out_val);
    }
}

template <typename floatX, int mem_size = SOFTMAX_DEFAULT_SMEM_SIZE>
void launch_shared_mem(floatX* output, const floatX* input, int N, int C, cudaStream_t stream = 0) {
    static_assert(mem_size >= WARP_SIZE && (mem_size & (mem_size - 1)) == 0, 
                  "mem_size must be power of two and >=WARP_SIZE");
    dim3 gridDim(N);
    shared_mem<floatX, mem_size><<<gridDim, mem_size, 0, stream>>>(output, input, N, C);
    cudaCheckError(cudaGetLastError());
}

}  // namespace cuda_kernel::softmax