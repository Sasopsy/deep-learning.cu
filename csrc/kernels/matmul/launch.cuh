#pragma once
#include <cuda_runtime.h>
#include "../../utils/utils.hpp"
#include "../../common.hpp"

namespace cuda_kernel::matmul {

void cpu(const float* A, const float* B, float* C, 
         const int M, const int N, const int K, const int batch) {
   #pragma omp parallel for collapse(1)
   for (int b = 0; b < batch; b++) {
       // Calculate batch offsets
       const float* a_batch = A + b * M * K;
       const float* b_batch = B + b * K * N;
       float* c_batch = C + b * M * N;

       // Perform matrix multiplication for this batch
       for (int m = 0; m < M; m++) {
           for (int n = 0; n < N; n++) {
               float sum = 0.0f;
               #pragma unroll 4
               for (int k = 0; k < K; k++) {
                   sum += a_batch[m * K + k] * b_batch[k * N + n];
               }
               c_batch[m * N + n] = sum;
           }
       }   
   }
}


}