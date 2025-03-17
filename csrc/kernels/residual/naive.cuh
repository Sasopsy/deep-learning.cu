#pragma once
#include <iostream>
#include <cstring>
#include <float.h>
#include "../../utils/utils.hpp"
#include "../../common.hpp"

namespace cuda_kernel::residual{


template <typename floatX>
__global__ void naive(floatX* out, const floatX* in, const floatX* residual, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = in[idx] + residual[idx];
    }
}
    
template <typename floatX>
void launch_naive(floatX* out, const floatX* in, const floatX* residual, int size, cudaStream_t stream = 0) {
    const int block_size = 256;
    int grid_size = ceil_div<int>(size, block_size);
    naive<floatX><<<grid_size, block_size, 0, stream>>>(out, in, residual, size);
    cudaCheckError(cudaGetLastError());
}

} // namespace cuda_kernel::residual