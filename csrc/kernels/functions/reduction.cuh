#pragma once
#include <cstring>
#include <cuda_runtime.h>
#include <float.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "../../utils/utils.hpp"
#include "../../common.hpp"

namespace cuda_functions {

// requires all 32 threads in the warp to be active, but should work for any block size
// uses non-dynamic shared memory so every call increases shared memory requirements by 128 bytes
// the fact it's unique shared memory allows us to avoid an extra __syncthreads() call at the end
// but if called inside a loop, the shared memory will be implicitly reused, so set final_sync to 1

#ifndef FLOAT_TYPE
    #define FLOAT_TYPE float
#endif

// Source: https://github.com/karpathy/llm.c/blob/llama3/dev/cuda/common.h#L29
using reduction_func_t = FLOAT_TYPE (*) (FLOAT_TYPE);

template<typename floatX,reduction_func_t warp_reduction>
__device__ inline floatX block_reduce(floatX val, bool final_sync, floatX out_of_bounds) {
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    __shared__ floatX shared_val[WARP_SIZE];
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    floatX warp_val = warp_reduction(val);
    if (lane_id == 0) { shared_val[warp_id] = warp_val; }
    __syncthreads();
    warp_val = (lane_id < num_warps) ? shared_val[lane_id] : out_of_bounds;
    floatX block_val = warp_reduction(warp_val);

    if (final_sync) {
        __syncthreads(); // only needed in loops when effectively reusing shared memory etc.
    }
    return block_val;
}

// Helper function to call block_reduce with default arguments
template<typename floatX, reduction_func_t warp_reduction>
__device__ inline floatX block_reduce(floatX val) {
    return block_reduce<floatX, warp_reduction>(val, false, 0.0f);
}

template<typename floatX>
__device__ floatX warp_reduce_sum(floatX val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

template<typename floatX>
__device__ float warp_reduce_max(floatX val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = max<floatX>(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

} // namespace cuda_functions