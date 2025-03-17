#pragma once

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>
#include <cassert>

// Utility for timing
struct Timer {
    cudaEvent_t start, stop;
    
    Timer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    
    ~Timer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void begin() {
        cudaEventRecord(start);
    }

    float end() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        return milliseconds;
    }
};

// CPU-GPU memory management helpers
template<typename T>
struct HostDevice {
    T* host = nullptr;
    T* device = nullptr;
    size_t size;

    HostDevice(size_t n) : size(n) {
        host = new T[n];
        cudaMalloc(&device, n * sizeof(T));
    }

    ~HostDevice() {
        if (host) delete[] host;
        if (device) cudaFree(device);
    }

    void copyToDevice() {
        cudaMemcpy(device, host, size * sizeof(T), cudaMemcpyHostToDevice);
    }

    void copyToHost() {
        cudaMemcpy(host, device, size * sizeof(T), cudaMemcpyDeviceToHost);
    }
};

// Result comparison utilities
template<typename T>
bool compareResults(const T* a, const T* b, size_t size, T tolerance = 1e-5, bool verbose = false) {
    bool passed = true;
    size_t numErrors = 0;
    const size_t maxErrors = 10;  // Maximum number of errors to print

    for (size_t i = 0; i < size; i++) {
        T diff = std::abs(a[i] - b[i]);
        if (diff > tolerance) {
            passed = false;
            numErrors++;
            if (verbose && numErrors <= maxErrors) {
                std::cout << "Mismatch at index " << i 
                         << ": " << std::setprecision(8) << a[i] 
                         << " vs " << std::setprecision(8) << b[i]
                         << " (diff: " << diff << ")" << std::endl;
            }
        }
    }

    if (verbose) {
        if (passed) {
            std::cout << "All values match within tolerance " << tolerance << std::endl;
        } else {
            std::cout << "Total mismatches: " << numErrors << " / " << size << std::endl;
        }
    }

    return passed;
}

// Performance measurement helper
template<typename Func>
float measureKernelPerformance(Func kernel, int iterations = 5) {
    // Warmup
    for (int i = 0; i < 5; i++) {
        kernel();
        cudaDeviceSynchronize();
    }

    Timer timer;
    timer.begin();
    
    for (int i = 0; i < iterations; i++) {
        kernel();
    }
    cudaDeviceSynchronize();
    
    float ms = timer.end();
    return ms / iterations;
}

// Random data initialization
template<typename T>
void fillRandom(T* data, size_t size, T min = -1.0, T max = 1.0) {
    for (size_t i = 0; i < size; i++) {
        float r = static_cast<float>(rand()) / RAND_MAX;
        data[i] = static_cast<T>(min + r * (max - min));
    }
}

// Print array helper (for debugging)
template<typename T>
void printArray(const T* arr, size_t size, const char* name = "array", int precision = 4) {
    std::cout << name << " [" << size << "]: ";
    for (size_t i = 0; i < std::min(size, size_t(10)); i++) {
        std::cout << std::fixed << std::setprecision(precision) << arr[i] << " ";
    }
    if (size > 10) std::cout << "...";
    std::cout << std::endl;
}
