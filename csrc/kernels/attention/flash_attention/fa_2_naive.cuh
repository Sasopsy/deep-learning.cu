#pragma once
#include <iostream>
#include <cstring>
#include <cuda_runtime.h>
#include <cassert>
#include <omp.h>
#include <float.h>
#include "../../functions/reduction.cuh"
#include "../../../utils/utils.hpp"
#include "../../../common.hpp"


#ifndef FLASH_ATTENTION_2_DEFAULT_WARPS_PER_BLOCK
    #define FLASH_ATTENTION_2_DEFAULT_WARPS_PER_BLOCK 16
#endif 


namespace cuda_kernel::attention::flash_attention {

namespace cf = cuda_functions;

template <typename floatX>
__global__ void fa_2_naive(const floatX* Q, const floatX* K, const floatX* V, floatX* O, floatX scale, 
                    const int B, const int H, const int seq_len_q, const int seq_len_kv, const int d,
                    const int Br, const int Bc) {
    // Calculate number of blocks
    const int Tr = ceil_div<int>(seq_len_q, Br);   // Number of row blocks
    const int Tc = ceil_div<int>(seq_len_kv, Bc);  // Number of column blocks

    // Shared memory for Q, K, V, O tiles and auxiliary arrays
    extern __shared__ char shared_mem[];
    floatX* shared_q = reinterpret_cast<floatX*>(shared_mem);
    floatX* shared_k = reinterpret_cast<floatX*>(shared_q + Br * d);
    floatX* shared_v = reinterpret_cast<floatX*>(shared_k + Bc * d);
    floatX* shared_o = reinterpret_cast<floatX*>(shared_v + Bc * d);
    float* S = reinterpret_cast<float*>(shared_o + Br * d);
    float* l_prev_shared = reinterpret_cast<float*>(S + Br * Bc);
    float* max_prev_shared = reinterpret_cast<float*>(l_prev_shared + Br);

    // Calculate base pointers for this block
    int batch_head = blockIdx.x;
    int row_block = blockIdx.y;

    // Calculate starting positions in global memory
    const floatX* Q_start = Q + (batch_head * seq_len_q + row_block * Br) * d;
    const floatX* K_base = K + (batch_head * seq_len_kv) * d;
    const floatX* V_base = V + (batch_head * seq_len_kv) * d;
    floatX* O_start = O + (batch_head * seq_len_q + row_block * Br) * d;

    // Warp related variables
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;

    // Initialize running statistics for online softmax with shared arrays
    // float m_i = -INFINITY;
    // float l_i = 0.0f;

    // Initialize l_prev and max_prev
    for (int i = threadIdx.x; i < Br; i += blockDim.x) {
        if (i < Br){
            l_prev_shared[i] = 0.0f;
            max_prev_shared[i] = -INFINITY;
        }
    }
    __syncthreads();

    // Before processing tiles, initialize shared_o to zeros
    for (int i = threadIdx.x; i < Br * d; i += blockDim.x) {
        shared_o[i] = float_to_floatX<floatX>(0.0f);
    }
    __syncthreads();
    
    // Load Q tile
    for (int i = threadIdx.x; i < Br * d; i += blockDim.x) {
        int row = i / d;
        int col = i % d;
        if (row_block * Br + row < seq_len_q) {
            shared_q[i] = Q_start[row * d + col];
        }
    }
    __syncthreads();

    // Process input in tiles
    for (int tile_col = 0; tile_col < seq_len_kv; tile_col += Bc) {
        const int active_cols = min(Bc, seq_len_kv - tile_col);

        // Set pointers to K and V tiles
        const floatX* K_tile = K_base + tile_col * d;
        const floatX* V_tile = V_base + tile_col * d;

        // Load K and V tiles
        for (int i = threadIdx.x; i < Bc * d; i += blockDim.x) {
            int row = i / d;
            int col = i % d;
            if (row < active_cols) {
                shared_k[i] = K_tile[row * d + col];
                shared_v[i] = V_tile[row * d + col];
            } 
        }
        __syncthreads();

        // Compute S
        for (int i = threadIdx.x; i < Br * active_cols; i += blockDim.x) {
            int row = i / active_cols;
            int col = i % active_cols;
            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < d; j++) {
                sum += floatX_to_float<floatX>(shared_q[row * d + j]) * floatX_to_float<floatX>(shared_k[col * d + j]);
            }
            S[i] = sum * floatX_to_float<floatX>(scale);
        }
        __syncthreads();

        // Row wise output computation. Each warp handles a single row.
        for (int i = 0; i < Br; i += warps_per_block) {
            const int row = warp_id + i;
            if (row >= Br) continue;

            // Find max value in row
            float m_prev = max_prev_shared[row];
            float l_prev = l_prev_shared[row];

            // Find max in current tile with full warp reduction
            // rowmax(S_i)
            float max_val = m_prev;
            for (int col = lane_id; col < active_cols; col += WARP_SIZE) {
                max_val = fmaxf(max_val, S[row * active_cols + col]);
            }
            max_val = cf::warp_reduce_max<float>(max_val);
            max_val = __shfl_sync(0xFFFFFFFF, max_val, 0);   // Broadcast max_val to all threads

            // Compute exponentials and sum with full warp sync
            // P_i = exp(S_i - max(S_i)) 
            // rowsum(P_i)
            float row_sum = 0.0f;
            for (int col = lane_id; col < active_cols; col += WARP_SIZE) {
                const float s = __expf(S[row * active_cols + col] - max_val);
                S[row * active_cols + col] = s;
                row_sum += s;
            }
            row_sum = cf::warp_reduce_sum<float>(row_sum);
            row_sum = __shfl_sync(0xFFFFFFFF, row_sum, 0);  // Broadcast to all lanes

            // Calculate new normalization factors
            // l_i = e^(m_i-1 - m_i) * l_i-1 + rowsum(P_i)
            const float m_new = fmaxf(m_prev, max_val);
            const float l_new = __expf(m_prev - m_new) * l_prev + row_sum;
            
            // Update output values using consistent values across warp
            // O_i = e^(m_i-1 - m_i) ** (-1) * O_i-1 + P_i * V
            for (int j = lane_id; j < d; j += WARP_SIZE) {
                float pv = 0.0f;
                #pragma unroll
                for (int col = 0; col < active_cols; ++col) {
                    pv += S[row * active_cols + col] * floatX_to_float<floatX>(shared_v[col * d + j]);
                }
                const float acc = __expf(m_prev - m_new) * 
                                floatX_to_float<floatX>(shared_o[row * d + j]) + pv;
                shared_o[row * d + j] = float_to_floatX<floatX>(acc);
            }

            // Update max_prev and l_prev
            if (lane_id == 0) {
                max_prev_shared[row] = m_new;
                l_prev_shared[row] = l_new;
            }   
                
        }
        __syncthreads();
    }

    // Write out results
    // O_i = O_i_tc / l_i
    #pragma unroll
    for (int i = 0; i < Br; i += warps_per_block) {
        int row = warp_id + i;
        if (row >= Br || row_block * Br + row >= seq_len_q) continue;
        const float norm = l_prev_shared[row] + CUDA_EPSILON;
        for (int j = lane_id; j < d; j += WARP_SIZE) {
            float out = floatX_to_float<floatX>(shared_o[row * d + j]) / norm;
            O_start[row * d + j] = float_to_floatX<floatX>(out);
        }
    }
}


constexpr float SHARED_MEM_FACTOR = 0.95f;
constexpr int BR_ALIGN = 8;
constexpr int BC_ALIGN = 32;  
constexpr int MIN_BR = 8;     
constexpr int MIN_BC = 32;    
constexpr int MAX_BR = 128;    
constexpr int MAX_BC = 128; 

template <typename floatX>
void calculate_block_sizes(const int d, int& Br, int& Bc) {
    const int max_mem = MAX_SHARED_MEM_SIZE * SHARED_MEM_FACTOR;
    
    // Start with fixed Bc = 32 (aligned to warp size)
    Bc = BC_ALIGN;
    
    // Binary search for optimal Br
    int low = MIN_BR;
    int high = MAX_BR;
    int best_br = MIN_BR;
    
    while (low <= high) {
        int mid = low + (high - low) / 2;
        // Align to BR_ALIGN
        int aligned_mid = (mid / BR_ALIGN) * BR_ALIGN;
        if (aligned_mid < MIN_BR) aligned_mid = MIN_BR;
        
        // Calculate memory requirements
        size_t mem_required = (aligned_mid * d + Bc * d + Bc * d + aligned_mid * d) * sizeof(floatX) + 
                             (aligned_mid * Bc) * sizeof(float) + 
                             (2 * aligned_mid) * sizeof(float);
        
        if (mem_required <= max_mem) {
            // This size works, try bigger
            best_br = aligned_mid;
            low = mid + 1;
        } else {
            // Too big, try smaller
            high = mid - 1;
        }
    }
    
    Br = best_br;

    size_t mem_required = (Br * d + Bc * d + Bc * d + Br * d) * sizeof(floatX) + 
                         (Br * Bc) * sizeof(float) + 
                         (2 * Br) * sizeof(float);
    
    #ifdef DEBUG
    std::cout << "Max shared memory: " << max_mem << " bytes" << std::endl;
    std::cout << "Required shared memory: " << mem_required << " bytes" << std::endl;
    std::cout << "Using block sizes: Br=" << Br << ", Bc=" << Bc << std::endl;
    #endif

    if (mem_required > max_mem) {
        std::cerr << "Memory requirements exceed maximum shared memory size!" << std::endl;
        std::cerr << "Required: " << mem_required << " bytes, Max: " << max_mem << " bytes" << std::endl;
        assert(false);
    }

}



template <typename floatX, int warps_per_block = FLASH_ATTENTION_2_DEFAULT_WARPS_PER_BLOCK>
void launch_fa2(const floatX* Q, const floatX* K, const floatX* V, floatX* O, floatX scale, 
                const int B, const int H, const int seq_len_q, const int seq_len_kv, const int d,
                cudaStream_t stream = 0) {
    int Br;
    int Bc;
    calculate_block_sizes<floatX>(d, Br, Bc);

    dim3 grid(B * H, ceil_div<int>(seq_len_q, Br));
    dim3 block(warps_per_block * WARP_SIZE);
    
    // Calculate shared memory size including l_prev and max_prev arrays
    size_t shared_mem_size = (Br * d + Bc * d + Bc * d + Br * d) * sizeof(floatX) + // Q, K, V, O
                            (Br * Bc) * sizeof(float) +  // S matrix
                            (2 * Br) * sizeof(float);    // l_prev and max_prev arrays
    
    fa_2_naive<floatX><<<grid, block, shared_mem_size, stream>>>(
        Q, K, V, O, scale, B, H, seq_len_q, seq_len_kv, d, Br, Bc
    );
    cudaCheckError(cudaGetLastError());
}

} // namespace cuda_kernel::attention::flash_attention