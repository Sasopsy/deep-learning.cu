#pragma once
#include <iostream>
#include <cstring>
#include <cuda_runtime.h>
#include <cassert>
#include <omp.h>
#include "../../utils/utils.hpp"
#include "../../common.hpp"

namespace cuda_kernel::linear {

template <typename floatX>
__global__ void naive(const floatX* input, 
                      const floatX* weight, 
                      const floatX* bias, 
                      floatX* output, 
                      int BT, int C,int OC){
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // in the naive kernel, every thread handles one element of output

    int oc = blockIdx.x * blockDim.x + threadIdx.x;  // x
    int bt = blockIdx.y * blockDim.y + threadIdx.y;  // y

    if (oc < OC && bt < BT) {
        floatX val = (bias != NULL) ? bias[oc] : get_zero<floatX>();
        floatX const *wrow = weight + oc * C;
        floatX const *inp = input + bt * C;
        for (int i = 0; i < C; i++) {
            val += wrow[i] * inp[i];
        }
        output[bt * OC + oc] = val;
    
    }
}

template <typename floatX>
void launch_naive(const floatX* input, const floatX* weight, const floatX* bias,
                  floatX* output, int BT, int C, int OC, cudaStream_t stream = 0) {
    dim3 blockDim(256, 1);  // x dimension for OC
    dim3 gridDim(ceil_div<int>(OC, blockDim.x), ceil_div<int>(BT, blockDim.y));
    
    naive<floatX><<<gridDim, blockDim, 0, stream>>>(input, weight, bias, output, BT, C, OC);
    cudaCheckError(cudaGetLastError());
}

} // namespace cuda_kernel::linear

