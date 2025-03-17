#pragma once
#include <iostream>
#include <cstring>
#include <cuda_runtime.h>
#include <cassert>
#include <omp.h>
#include "../../utils/utils.hpp"
#include "../../common.hpp"

#ifndef LINEAR_DEFAULT_BLOCK_SIZE
    #define LINEAR_DEFAULT_BLOCK_SIZE 32
#endif

namespace cuda_kernel::linear {

template <typename floatX,const int block_size>
__global__ void shared_mem(const floatX* input,
                          const floatX* weight, 
                          const floatX* bias,
                          floatX* output, 
                          int BT, int C,int OC){
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // in the naive kernel, every thread handles one element of output
    // in this kernel, every thread handles one element of output, but we use shared memory to cache input and weight
    // Essentially calculating input * weight.T
    assert(block_size*block_size == blockDim.x);

    // o_row - Row block of input
    // o_col - Row block of weight
    const uint o_row = blockIdx.x;
    const uint o_col = blockIdx.y;

    // the inner row & col that we're accessing in this thread
    const uint thread_col = threadIdx.x % block_size;
    const uint thread_row = threadIdx.x / block_size;

    // Allocate shared memory
    __shared__ floatX shared_input[block_size * block_size];
    __shared__ floatX shared_weight[block_size * block_size];
    __shared__ floatX shared_bias[block_size];

    // Advance pointers to starting positions
    input += o_row * block_size * C;
    weight += o_col * block_size * C;

    // Initialize output with bias if present
    floatX tmp = get_zero<floatX>();
    const int global_col = o_col * block_size + thread_col;
    const int global_row = o_row * block_size + thread_row;

    // Load bias into shared memory
    if (bias && global_col < OC) {
        shared_bias[thread_col] = bias[global_col];
    }
    __syncthreads();

    for (int tile = 0; tile < C; tile += block_size){

        // Load data into shared memory withing bounds for input.
        if ((tile + thread_col) < C && (o_row * block_size + thread_row) < BT ){
            shared_input[thread_row * block_size + thread_col] = input[thread_row * C + thread_col];
        } else {
            shared_input[thread_row * block_size + thread_col] = get_zero<floatX>();
        }

        // Load data into shared memory within bounds for weight.
        if ((tile + thread_col) < C && (o_col * block_size + thread_row) < OC ){
            shared_weight[thread_row * block_size + thread_col] = weight[thread_row * C + thread_col];
        } else {
            shared_weight[thread_row * block_size + thread_col] = get_zero<floatX>();
        }

        __syncthreads();

        input += block_size;
        weight += block_size;

        for (int c = 0; c < block_size; c++){
            tmp += shared_input[thread_row * block_size + c] * 
                   shared_weight[thread_col * block_size + c];
        }
        __syncthreads();
        }

    // Add bias if present
    if (bias && global_col < OC) {
        tmp += shared_bias[thread_col];
    }
    
    if (global_row < BT && global_col < OC){
        output[global_row * OC + global_col] = tmp;
    }
}


template <typename floatX, int block_size = LINEAR_DEFAULT_BLOCK_SIZE>
void launch_shared_mem(const floatX* input, const floatX* weight, const floatX* bias,
                      floatX* output, int BT, int C, int OC, cudaStream_t stream = 0) {
    dim3 blockDim(block_size * block_size);
    dim3 gridDim(ceil_div<int>(BT, block_size), ceil_div<int>(OC, block_size));
    
    shared_mem<floatX, block_size><<<gridDim, blockDim, 0, stream>>>(input, weight, bias, output, BT, C, OC);
    cudaCheckError(cudaGetLastError());
}

} // namespace cuda_kernel::linear