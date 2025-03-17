#pragma once
#include <iostream>
#include <cstring>
#include <cuda_runtime.h>
#include <cassert>
#include <omp.h>
#include <float.h>
#include "../../../utils/utils.hpp"
#include "../../../common.hpp"

namespace cuda_kernel::position_encoder::fused {

template <typename floatX>
__global__ void warp_divergence(floatX* input, floatX* output, int B, int T, int d, int start){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * T * d){
        int b = idx / (T * d);
        int i = (idx % (T * d)) / d + start;
        int j = idx % d;
        float pos_enc_val;
        pos_enc_val = (j % 2) * cos(i / pow(10000, j / d)) + (1 - j % 2) * sin(i / pow(10000, j / d));
        output[idx] = input[idx] + float_to_floatX<floatX>(pos_enc_val);
    }
}

template <typename floatX>
void launch_warp_divergence(floatX* input, floatX* output, int B, int T, int d, int start = 0, cudaStream_t stream = 0) {
    int block_size = 256;
    int num_blocks = ceil_div<int>(B * T * d, block_size);
    warp_divergence<floatX><<<num_blocks, block_size, 0, stream>>>(input, output, B, T, d, start);
    cudaCheckError(cudaGetLastError());
}

} // namespace cuda_kernel::position_encoder::fused

