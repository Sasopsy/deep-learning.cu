#pragma once
#include <iostream>
#include <cstring>
#include <cuda_runtime.h>
#include <cassert>
#include <omp.h>
#include <float.h>
#include "../../../utils/utils.hpp"
#include "../../../common.hpp"

namespace cuda_kernel::position_encoder::add {

template <typename floatX>
__global__ void naive(floatX* pos_enc, floatX* input, floatX* output, int B, int T, int d, int start) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * T * d){
        int b = idx / (T * d);
        int i = (idx % (T * d)) / d + start;
        int j = idx % d;
        output[idx] = input[idx] + pos_enc[i * d + j];
    }
}

template <typename floatX>
void launch_naive(floatX* pos_enc, floatX* input, floatX* output, int B, int T, int d, int start = 0, cudaStream_t stream = 0) {
    int block_size = 256;
    int num_blocks = ceil_div<int>(B * T * d, block_size);
    naive<floatX><<<num_blocks, block_size, 0, stream>>>(pos_enc, input, output, B, T, d, start);
    cudaCheckError(cudaGetLastError());
}

}