#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include "precision.hpp"

// CUDA error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Ceiling division helper
template<class T>
__host__ __device__ T ceil_div(T dividend, T divisor) {
    return (dividend + divisor-1) / divisor;
}

// Stream synchronization helper
inline void sync_stream(cudaStream_t stream = 0) {
    cudaCheckError(cudaStreamSynchronize(stream));
}

// Device memory allocation helper
template<typename floatX>
inline floatX* device_malloc(size_t size) {
    floatX* ptr;
    cudaCheckError(cudaMalloc(&ptr, size * sizeof(floatX)));
    return ptr;
}

// Device memory free helper
template<typename floatX>
inline void device_free(floatX* ptr) {
    if (ptr != nullptr) {
        cudaCheckError(cudaFree(ptr));
    }
}









