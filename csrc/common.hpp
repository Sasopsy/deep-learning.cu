#pragma once
#include <cstring>
#include <cuda_runtime.h>
#include <float.h>
#include <cuda.h>

// Define and initialize the device properties
struct DevicePropInitializer {
    cudaDeviceProp prop;
    
    DevicePropInitializer() {
        int device;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&prop, device);
    }
};

static DevicePropInitializer devicePropInit;
#define deviceProp devicePropInit.prop

// Device constants
#define MAX_SHARED_MEM_SIZE deviceProp.sharedMemPerBlock
#define MAX_THREADS_PER_BLOCK 1024
#define WARP_SIZE 32U

// Mathematical constants
#define CUDA_PI 3.14159265358979323846
#define CUDA_EPSILON 1e-6f



