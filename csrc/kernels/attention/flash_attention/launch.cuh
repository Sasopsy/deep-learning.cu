#pragma once
#include <cuda_runtime.h>
#include "../../functions/reduction.cuh"
#include "../../matmul/launch.cuh"
#include "../../softmax/launch.cuh"
#include "../../linear/launch.cuh"
#include "../../../utils/utils.hpp"
#include "../../../common.hpp"
#include "fa_2_naive.cuh"

#define FLASH_ATTENTION_NUM_KERNELS 1

#ifndef FLASH_ATTENTION_DEFAULT_KERNEL
    #define FLASH_ATTENTION_DEFAULT_KERNEL 0  // 0=fa_2_naive
#endif

namespace cuda_kernel::attention::flash_attention {

void cpu(const float* Q, const float* K, const float* V, float* O, float scale,
    const int B, const int H, const int seq_len_q, const int seq_len_kv, const int d) {

    // Temporary storage for intermediate results
    float* QK = new float[B * H * seq_len_q * seq_len_kv];  // For Q * K^T
    float* softmax_out = new float[B * H * seq_len_q * seq_len_kv];  // For softmax(QK)

    // Process each batch and head
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            // Calculate base indices for this batch and head
            const int batch_head_offset = (b * H + h);
            const float* q_batch = Q + batch_head_offset * seq_len_q * d;
            const float* k_batch = K + batch_head_offset * seq_len_kv * d;
            const float* v_batch = V + batch_head_offset * seq_len_kv * d;
            float* o_batch = O + batch_head_offset * seq_len_q * d;
            float* qk_batch = QK + batch_head_offset * seq_len_q * seq_len_kv;
            float* softmax_batch = softmax_out + batch_head_offset * seq_len_q * seq_len_kv;
            
            // Step 1: Compute Q * K^T using linear kernel
            cuda_kernel::linear::cpu(
                q_batch,      // input - all query vectors
                k_batch,      // weight - all key vectors
                nullptr,      // no bias
                qk_batch,     // output - attention scores for all queries against all keys
                seq_len_q,    // BT = seq_len_q (number of query vectors)
                d,            // C = d (embedding dimension)
                seq_len_kv    // OC = seq_len_kv (number of key vectors)
            );
            
            // Apply scaling factor
            for (int i = 0; i < seq_len_q * seq_len_kv; i++) {
                qk_batch[i] *= scale;
            }
            
            // Step 2: Apply softmax to scaled QK along seq_len_kv dimension
            cuda_kernel::softmax::cpu(
                softmax_batch, qk_batch,
                seq_len_q, seq_len_kv
            );
            
            // Step 3: Compute (softmax(QK)) * V using matmul
            cuda_kernel::matmul::cpu(
                softmax_batch, v_batch, o_batch,
                seq_len_q, d, seq_len_kv, 1
            );
        }
    }

    // Clean up temporary storage
    delete[] QK;
    delete[] softmax_out;
}

template <typename floatX>
void launch(const floatX* Q, const floatX* K, const floatX* V, floatX* O, floatX scale, 
           const int B, const int H, const int seq_len_q, const int seq_len_kv, const int d, 
           int kernel_choice = 0, cudaStream_t stream = 0) {
    if (kernel_choice < 0 || kernel_choice >= FLASH_ATTENTION_NUM_KERNELS) {
        kernel_choice = 1;
    }

    switch (kernel_choice) {
        case 0:
            launch_fa2<floatX>(Q, K, V, O, scale, B, H, seq_len_q, seq_len_kv, d, stream);
            break;
    }
}

}  // namespace cuda_kernel::attention::flash_attention