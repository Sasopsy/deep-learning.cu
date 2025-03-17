#pragma once
#include <iostream>
#include <cstring>
#include <cuda_runtime.h>
#include <cassert>
#include <omp.h>
#include "../functions/reduction.cuh"
#include "../../utils/utils.hpp"
#include "../../common.hpp"

namespace cuda_kernel::softmax {

template <typename floatX>
__global__ void naive(floatX* output, const floatX* input, int N, int C) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const floatX* input_row = input + i * C;
        floatX* output_row = output + i * C;

        // Find max value using float for stability
        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) {
            float val = floatX_to_float<floatX>(input_row[j]);
            if (val > maxval) {
                maxval = val;
            }
        }

        // Compute exponentials and sum using float precision
        float sum = 0.0f;
        
        for (int j = 0; j < C; j++) {
            float input_val = floatX_to_float<floatX>(input_row[j]);
            float val = __expf(input_val - maxval);
            output_row[j] = float_to_floatX<floatX>(val);
            sum += val;
        }

        // Normalize and write results
        for (int j = 0; j < C; j++) {
            float out_val = floatX_to_float<floatX>(output_row[j]) / sum;
            output_row[j] = float_to_floatX<floatX>(out_val);
        }
    }
}

template <typename floatX>
void launch_naive(floatX* output, const floatX* input, int N, int C, cudaStream_t stream = 0) {
    const int block_size = 256;
    dim3 gridDim(ceil_div<int>(N, block_size));
    naive<floatX><<<gridDim, block_size, 0, stream>>>(output, input, N, C);
    cudaCheckError(cudaGetLastError());
}

}  // namespace cuda_kernel::softmax