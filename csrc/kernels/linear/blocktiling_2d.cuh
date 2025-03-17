#pragma once
#include <iostream>
#include <cstring>
#include <cuda_runtime.h>
#include <cassert>
#include <omp.h>
#include "../../utils/utils.hpp"
#include "../../common.hpp"

#ifndef LINEAR_DEFAULT_BC
    #define LINEAR_DEFAULT_BC 32
#endif

#ifndef LINEAR_DEFAULT_BOC
    #define LINEAR_DEFAULT_BOC 64
#endif

#ifndef LINEAR_DEFAULT_BBT
    #define LINEAR_DEFAULT_BBT 64
#endif

#ifndef LINEAR_DEFAULT_TBT
    #define LINEAR_DEFAULT_TBT 8
#endif

#ifndef LINEAR_DEFAULT_TOC
    #define LINEAR_DEFAULT_TOC 8
#endif

namespace cuda_kernel::linear {

template <typename floatX, const int Bbt, const int Bc, const int Boc, const int Tbt, const int Toc>
__global__ void __launch_bounds__((Bbt * Boc) / (Tbt * Toc), 1) 
                blocktiling_2d(const floatX* input, 
                              const floatX* weight, 
                              const floatX* bias, 
                              floatX* output, 
                              int BT, int C,int OC){

    static_assert(Boc % Toc == 0, "Boc must be divisible by Toc");
    static_assert(Bbt % Tbt == 0, "Bbt must be divisible by Tbt");

    const uint o_row = blockIdx.x;
    const uint o_col = blockIdx.y;

    // Total number of outputs computed per block.
    const uint tot_res_blocktile = Bbt * Boc;
    // Each thread computes a sub-tile of size (Tbt x Toc) so the number of threads is:
    const uint num_threads_blocktile = tot_res_blocktile / (Tbt * Toc);
    assert(num_threads_blocktile == blockDim.x);

    // Map threads in a 2D grid over the output tile.
    const int thread_col = threadIdx.x % (Boc / Toc);
    const int thread_row = threadIdx.x / (Boc / Toc);

    // Allocate shared memory for block tiles.
    __shared__ floatX shared_input[Bbt * (Bc + 1)];   // input tile of shape (Bbt x Bc)
    __shared__ floatX shared_weight[Boc * (Bc + 1)];    // weight tile of shape (Boc x Bc)
    __shared__ floatX shared_bias[Boc];           // bias tile of length Boc

    // Adjust global pointers to the start of the corresponding block tile.
    input  += (o_row * Bbt * C);
    weight += (o_col * Boc * C);
    output += (o_row * Bbt * OC + o_col * Boc);

    // Load bias into shared memory.
    const uint stride_wgt = num_threads_blocktile;
    for (uint load_offset = threadIdx.x; load_offset < Boc; load_offset += stride_wgt) {
        uint global_bias_idx = o_col * Boc + load_offset;
        if (global_bias_idx < OC) {
            shared_bias[load_offset] = (bias != nullptr) ? bias[global_bias_idx] : get_zero<floatX>();
        }
    }
    __syncthreads();

    // Each thread loads its portion of bias into registers.
    floatX reg_bias[Toc] = { get_zero<floatX>() };
    for (uint idx = 0; idx < Toc; idx++) {
        uint biasIdx = thread_col * Toc + idx;
        if (biasIdx < Boc)
            reg_bias[idx] = shared_bias[biasIdx];
    }

    // Declare registers for accumulation and tile data.
    floatX thread_results[Tbt * Toc] = { get_zero<floatX>() };
    floatX reg_bt[Tbt] = { get_zero<floatX>() };
    floatX reg_oc[Toc] = { get_zero<floatX>() };

    // Loop over the input channel dimension in tiles of size Bc.
    for (uint tile = 0; tile < C; tile += Bc) {
        const uint stride_inp = num_threads_blocktile;

        // Load input tile: shape (Bbt x Bc) from global memory.
        for (uint load_offset = threadIdx.x; load_offset < (Bbt * Bc); load_offset += stride_inp) {
            uint local_row = load_offset / Bc;      // within the block-tile row
            uint local_col = load_offset % Bc;        // within the tile columns
            // global row index is (o_row*Bbt + local_row); columns come directly from the current tile.
            if ((tile + local_col) < C && (o_row * Bbt + local_row) < BT)
                shared_input[local_row * Bc + local_col] = input[local_row * C + local_col];
            else
                shared_input[local_row * Bc + local_col] = get_zero<floatX>();
        }

        // Load weight tile: shape (Boc x Bc).
        for (uint load_offset = threadIdx.x; load_offset < (Boc * Bc); load_offset += stride_wgt) {
            uint local_row = load_offset / Bc;      // within the block tile for weight
            uint local_col = load_offset % Bc;
            // global weight row index is (o_col*Boc + local_row); use current tile segment.
            if ((tile + local_col) < C && (o_col * Boc + local_row) < OC)
                shared_weight[local_row * Bc + local_col] = weight[local_row * C + local_col];
            else
                shared_weight[local_row * Bc + local_col] = get_zero<floatX>();
        }
        __syncthreads();

        // Each thread accumulates its output sub-tile using the tile loaded in shared memory.
        for (int dot_idx = 0; dot_idx < Bc; dot_idx++) {
            // Load Tbt elements from the input tile.
            for (uint i = 0; i < Tbt; i++) {
                uint in_row = thread_row * Tbt + i;
                reg_bt[i] = shared_input[in_row * Bc + dot_idx];
            }
            // Load Toc elements from the weight tile.
            for (uint i = 0; i < Toc; i++) {
                uint w_row = thread_col * Toc + i;
                reg_oc[i] = shared_weight[w_row * Bc + dot_idx];
            }
            // Multiply and accumulate the dot product.
            for (uint res_idx_bt = 0; res_idx_bt < Tbt; res_idx_bt++) {
                for (uint res_idx_oc = 0; res_idx_oc < Toc; res_idx_oc++) {
                    thread_results[res_idx_bt * Toc + res_idx_oc] += reg_bt[res_idx_bt] * reg_oc[res_idx_oc];
                }
            }
        }
        __syncthreads();

        // Advance input/weight pointers for the next tile.
        input  += Bc;
        weight += Bc;
    } // end for each tile along C

    // Add bias to the accumulated results.
    for (uint res_idx_bt = 0; res_idx_bt < Tbt; res_idx_bt++) {
        for (uint res_idx_oc = 0; res_idx_oc < Toc; res_idx_oc++) {
            thread_results[res_idx_bt * Toc + res_idx_oc] += reg_bias[res_idx_oc];
        }
    }

    // Write back the results to global memory.
    for (uint res_idx_bt = 0; res_idx_bt < Tbt; res_idx_bt++) {
        for (uint res_idx_oc = 0; res_idx_oc < Toc; res_idx_oc++) {
            const uint out_row_block = thread_row * Tbt + res_idx_bt;
            const uint out_col_block = thread_col * Toc + res_idx_oc;
            if (o_row * Bbt + out_row_block < BT && o_col * Boc + out_col_block < OC)
                output[out_row_block * OC + out_col_block] = thread_results[res_idx_bt * Toc + res_idx_oc];
        }
    }
}


template <typename floatX, int Bbt = LINEAR_DEFAULT_BBT, int Bc = LINEAR_DEFAULT_BC, int Boc = LINEAR_DEFAULT_BOC, int Tbt = LINEAR_DEFAULT_TBT, int Toc = LINEAR_DEFAULT_TOC>
void launch_blocktiling_2d(const floatX* input, const floatX* weight, const floatX* bias,
                          floatX* output, int BT, int C, int OC, cudaStream_t stream = 0) {
    dim3 gridDim(ceil_div<int>(BT, Bbt), ceil_div<int>(OC, Boc) );
    const int num_threads = (Bbt * Boc) / (Tbt * Toc);
    
    blocktiling_2d<floatX, Bbt, Bc, Boc, Tbt, Toc>
        <<<gridDim, num_threads, 0, stream>>>(input, weight, bias, output, BT, C, OC);
    
    cudaCheckError(cudaGetLastError());
}

} // namespace cuda_kernel::linear