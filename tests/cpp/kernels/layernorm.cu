#include <iostream>
#include <cstring>
#include <cuda_runtime.h>
#include <cassert>
#include <map>
#include <vector>
#include "csrc/utils/utils.hpp"
#include "csrc/kernels/layernorm/launch.cuh"
#include "../test_utils.hpp"

namespace ck = cuda_kernel;

void checkResults(const float* cpu_result, const float* gpu_result, int N, int C) {
    bool passed = compareResults(cpu_result, gpu_result, N * C, 1e-5f, true);
    if (passed) {
        std::cout << "\033[32mResults match within tolerance!\033[0m" << std::endl;
    } else {
        std::cout << "\033[31mResults don't match within tolerance!\033[0m" << std::endl;
    }
}

void measureKernelTime(float* d_output, const float* d_input, const float* d_weight, const float* d_bias, 
                      int N, int C, int kernel_choice) {
    auto kernel = [&]() {
        ck::layernorm::launch<float>(d_output, d_input, d_weight, d_bias, N, C, kernel_choice);
    };
    
    float ms = measureKernelPerformance(kernel);
    std::cout << "Kernel " << kernel_choice << " average execution time: " << ms << " ms" << std::endl;
}

void displayKernelChoices() {
    std::cout << "Kernel Choices:" << std::endl;
    std::cout << "0: Naive Kernel" << std::endl;
    std::cout << "1: Shared Memory Optimized Kernel" << std::endl;
    std::cout << "2: Intra Warp Optimized Kernel" << std::endl;
    std::cout << "3: Variance Estimate Kernel" << std::endl;
}

int getUserKernelChoice() {
    int choice;
    std::cout << "Enter the kernel choice: ";
    std::cin >> choice;
    return choice;
}

void printParams(int N, int C, int kernel_choice) {
    std::cout << "Parameters used:" << std::endl;
    std::cout << "N: " << N << std::endl;
    std::cout << "C: " << C << std::endl;
    std::cout << "Kernel Choice: " << kernel_choice << std::endl;
    std::cout << "Macros: " << std::endl;
    std::cout << "LAYERNORM_DEFAULT_SMEM_SIZE: " << LAYERNORM_DEFAULT_SMEM_SIZE << std::endl;
    std::cout << "LAYERNORM_DEFAULT_IWARP_SIZE: " << LAYERNORM_DEFAULT_IWARP_SIZE << std::endl;
}

void handleError(const std::string& error_message) {
    std::cerr << "Error: " << error_message << std::endl;
    exit(EXIT_FAILURE);
}

int getUserChoice() {
    int choice;
    std::cout << "Enter 0 to test all kernels, or 1 to choose a specific kernel: ";
    std::cin >> choice;
    return choice;
}

void testAllKernels(const std::vector<std::map<std::string, int>>& configurations) {
    for (const auto& config : configurations) {
        int N = config.at("N");
        int C = config.at("C");

        std::cout << "Testing configuration: N=" << N << ", C=" << C << std::endl;

        // Allocate memory and initialize data
        HostDevice<float> input(N * C);
        HostDevice<float> weight(C);
        HostDevice<float> bias(C);
        HostDevice<float> output(N * C);
        HostDevice<float> cpu_output(N * C);

        fillRandom(input.host, N * C);
        fillRandom(weight.host, C);
        fillRandom(bias.host, C);

        input.copyToDevice();
        weight.copyToDevice();
        bias.copyToDevice();

        for (int kernel_choice = 0; kernel_choice < 4; ++kernel_choice) {
            std::cout << "Testing kernel choice: " << kernel_choice << std::endl;

            // Launch the selected kernel
            ck::layernorm::launch<float>(output.device, input.device, weight.device, bias.device, N, C, kernel_choice);

            // Copy results back to host
            output.copyToHost();

            // Compute reference result on CPU
            ck::layernorm::cpu(cpu_output.host, input.host, weight.host, bias.host, N, C);

            // Check results
            checkResults(cpu_output.host, output.host, N, C);

            // Measure kernel performance
            measureKernelTime(output.device, input.device, weight.device, bias.device, N, C, kernel_choice);

            // Print parameters used
            printParams(N, C, kernel_choice);

            std::cout << " " << std::endl;
            
            // Clear device memory
            cudaDeviceSynchronize();
            cudaDeviceReset();
            
            // Reallocate and copy data back if we're not on the last iteration
            if (kernel_choice < 3) {
                input = HostDevice<float>(N * C);
                weight = HostDevice<float>(C);
                bias = HostDevice<float>(C);
                output = HostDevice<float>(N * C);
                
                fillRandom(input.host, N * C);
                fillRandom(weight.host, C);
                fillRandom(bias.host, C);
                
                input.copyToDevice();
                weight.copyToDevice();
                bias.copyToDevice();
            }
        }
    }
}

void testSpecificKernel(const std::vector<std::map<std::string, int>>& configurations, int kernel_choice) {
    for (const auto& config : configurations) {
        int N = config.at("N");
        int C = config.at("C");

        std::cout << "Testing configuration: N=" << N << ", C=" << C << std::endl;

        // Allocate memory and initialize data
        HostDevice<float> input(N * C);
        HostDevice<float> weight(C);
        HostDevice<float> bias(C);
        HostDevice<float> output(N * C);
        HostDevice<float> cpu_output(N * C);

        fillRandom(input.host, N * C);
        fillRandom(weight.host, C);
        fillRandom(bias.host, C);

        input.copyToDevice();
        weight.copyToDevice();
        bias.copyToDevice();

        std::cout << "Testing kernel choice: " << kernel_choice << std::endl;

        // Launch the selected kernel
        ck::layernorm::launch<float>(output.device, input.device, weight.device, bias.device, N, C, kernel_choice);

        // Copy results back to host
        output.copyToHost();

        // Compute reference result on CPU
        ck::layernorm::cpu(cpu_output.host, input.host, weight.host, bias.host, N, C);

        // Check results
        checkResults(cpu_output.host, output.host, N, C);

        // Measure kernel performance
        measureKernelTime(output.device, input.device, weight.device, bias.device, N, C, kernel_choice);

        // Print parameters used
        printParams(N, C, kernel_choice);

        std::cout << " " << std::endl;
        
        // Clear device memory after each configuration test
        cudaDeviceSynchronize();
        cudaDeviceReset();
    }
}

int main() {
    // List of dictionaries with different choices for N and C
    std::vector<std::map<std::string, int>> configurations = {
        // {{"N", 128}, {"C", 64}},
        // {{"N", 256}, {"C", 128}},
        {{"N", 8*128}, {"C", 768}}
    };

    // Get user choice
    int user_choice = getUserChoice();

    if (user_choice == 0) {
        // Test all kernels with different configurations
        testAllKernels(configurations);
    } else {
        // Display kernel choices and get user selection
        displayKernelChoices();
        int kernel_choice = getUserKernelChoice();
        testSpecificKernel(configurations, kernel_choice);
    }

    return 0;
}
