#pragma once
#include <iostream>
#include <cstring>
#include <float.h>
#include "../../types/pack128.hpp"
#include "../../utils/utils.hpp"
#include "../../common.hpp"

namespace cuda_kernel::residual{

template <typename floatX>
__global__ void vectorised(floatX* out, const floatX* in, const floatX* residual, int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * Packed128<floatX>::size;
    if (idx < size) {
        Packed128<floatX> packed_out;
        Packed128<floatX> packed_in = Packed128<floatX>::load128cs(in + idx);
        Packed128<floatX> packed_residual = Packed128<floatX>::load128cs(residual + idx);
        for (int i=0; i<Packed128<floatX>::size; i++) {
            packed_out[i] = (packed_in[i] + packed_residual[i]);
        }
        packed_out.store128cs(out + idx);
    }
}

template <typename floatX>
void launch_vectorised(floatX* out, const floatX* in, const floatX* residual, int size, cudaStream_t stream = 0) {
    const int block_size = 256;
    int effectiveSize = ceil_div<int>(size, Packed128<floatX>::size); // Ceil-divide by Packed128<floatX>::size
    int grid_size = ceil_div<int>(effectiveSize, block_size);
    vectorised<floatX><<<grid_size, block_size, 0, stream>>>(out, in, residual, size);
    cudaCheckError(cudaGetLastError());
}

} // namespace cuda_kernel::residual