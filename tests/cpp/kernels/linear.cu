#include <iostream>
#include <cstring>
#include <cuda_runtime.h>
#include <cassert>
#include <map>
#include <vector>
#include "csrc/utils/utils.hpp"
#include "csrc/kernels/linear/launch.cuh"
#include "../test_utils.hpp"

namespace ck = cuda_kernel;

void checkResults(const float* cpu_result, const float* gpu_result, int N, int C) {
    bool passed = compareResults(cpu_result, gpu_result, N * C, 1e-4f, true);
    if (passed) {
        std::cout << "\033[32mResults match within tolerance!\033[0m" << std::endl;
    } else {
        std::cout << "\033[31mResults don't match within tolerance!\033[0m" << std::endl;
    }
}

void measureKernelTime(float* d_output, const float* d_input, const float* d_weight, const float* d_bias, 
                      int N, int C, int OC, int kernel_choice) {
    auto kernel = [&]() {
        ck::linear::launch<float>(d_input, d_weight, d_bias, d_output, N, C, OC, kernel_choice);
    };
    
    float ms = measureKernelPerformance(kernel);
    std::cout << "Kernel " << kernel_choice << " average execution time: " << ms << " ms" << std::endl;
}

void displayKernelChoices() {
    std::cout << "Kernel Choices:" << std::endl;
    std::cout << "0: Naive Kernel" << std::endl;
    std::cout << "1: Shared Memory Optimized Kernel" << std::endl;
    std::cout << "2: Block Tiling Optimized Kernel" << std::endl;
}

int getUserKernelChoice() {
    int choice;
    std::cout << "Enter the kernel choice: ";
    std::cin >> choice;
    return choice;
}

void printParams(int BT, int C, int OC, int kernel_choice) {
    std::cout << "Parameters used:" << std::endl;
    std::cout << "BT: " << BT << std::endl;
    std::cout << "C: " << C << std::endl;
    std::cout << "OC: " << OC << std::endl;
    std::cout << "Kernel Choice: " << kernel_choice << std::endl;
    std::cout << "Macros: " << std::endl;
    std::cout << "LINEAR_DEFAULT_BLOCK_SIZE: " << LINEAR_DEFAULT_BLOCK_SIZE << std::endl;
    std::cout << "LINEAR_DEFAULT_BC: " << LINEAR_DEFAULT_BC << std::endl;
    std::cout << "LINEAR_DEFAULT_BOC: " << LINEAR_DEFAULT_BOC << std::endl;
    std::cout << "LINEAR_DEFAULT_BBT: " << LINEAR_DEFAULT_BBT << std::endl;
    std::cout << "LINEAR_DEFAULT_TBT: " << LINEAR_DEFAULT_TBT << std::endl;
    std::cout << "LINEAR_DEFAULT_TOC: " << LINEAR_DEFAULT_TOC << std::endl;
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
        int BT = config.at("BT");
        int C = config.at("C");
        int OC = config.at("OC");

        std::cout << "Testing configuration: BT=" << BT << ", C=" << C << ", OC=" << OC << std::endl;

        // Allocate memory and initialize data
        HostDevice<float> input(BT * C);
        HostDevice<float> weight(OC * C);
        HostDevice<float> bias(OC);
        HostDevice<float> output(BT * OC);
        HostDevice<float> cpu_output(BT * OC);

        fillRandom(input.host, BT * C);
        fillRandom(weight.host, OC * C);
        fillRandom(bias.host, OC);

        input.copyToDevice();
        weight.copyToDevice();
        bias.copyToDevice();

        for (int kernel_choice = 0; kernel_choice < 3; ++kernel_choice) {
            std::cout << "Testing kernel choice: " << kernel_choice << std::endl;

            // Launch the selected kernel
            ck::linear::launch<float>(input.device, weight.device, bias.device, output.device, BT, C, OC, kernel_choice);

            // Copy results back to host
            output.copyToHost();

            // Compute reference result on CPU
            ck::linear::cpu(input.host, weight.host, bias.host, cpu_output.host, BT, C, OC);

            // Check results
            checkResults(cpu_output.host, output.host, BT, OC);

            // Measure kernel performance
            measureKernelTime(output.device, input.device, weight.device, bias.device, BT, C, OC, kernel_choice);

            // Print parameters used
            printParams(BT, C, OC, kernel_choice);

            std::cout << " " << std::endl;
            
            // Clear device memory
            cudaDeviceSynchronize();
            cudaDeviceReset();
            
            // Reallocate and copy data back if we're not on the last iteration
            if (kernel_choice < 2) {
                input = HostDevice<float>(BT * C);
                weight = HostDevice<float>(OC * C);
                bias = HostDevice<float>(OC);
                output = HostDevice<float>(BT * OC);
                cpu_output = HostDevice<float>(BT * OC);

                fillRandom(input.host, BT * C);
                fillRandom(weight.host, OC * C);
                fillRandom(bias.host, OC);

                input.copyToDevice();
                weight.copyToDevice();
                bias.copyToDevice();
            }
        }
    }
}

void testSpecificKernel(const std::vector<std::map<std::string, int>>& configurations, int kernel_choice) {
    for (const auto& config : configurations) {
        int BT = config.at("BT");
        int C = config.at("C");
        int OC = config.at("OC");

        std::cout << "Testing configuration: BT=" << BT << ", C=" << C << ", OC=" << OC << std::endl;

        // Allocate memory and initialize data
        HostDevice<float> input(BT * C);
        HostDevice<float> weight(OC * C);
        HostDevice<float> bias(OC);
        HostDevice<float> output(BT * OC);
        HostDevice<float> cpu_output(BT * OC);

        fillRandom(input.host, BT * C);
        fillRandom(weight.host, OC * C);
        fillRandom(bias.host, OC);

        input.copyToDevice();
        weight.copyToDevice();
        bias.copyToDevice();

        std::cout << "Testing kernel choice: " << kernel_choice << std::endl;

        // Launch the selected kernel
        ck::linear::launch<float>(input.device, weight.device, bias.device, output.device, BT, C, OC, kernel_choice);

        // Copy results back to host
        output.copyToHost();

        // Compute reference result on CPU
        ck::linear::cpu(input.host, weight.host, bias.host, cpu_output.host, BT, C, OC);

        // Check results
        checkResults(cpu_output.host, output.host, BT, OC);

        // Measure kernel performance
        measureKernelTime(output.device, input.device, weight.device, bias.device, BT, C, OC, kernel_choice);

        // Print parameters used
        printParams(BT, C, OC, kernel_choice);

        std::cout << " " << std::endl;
    }
}

int main() {
    // List of dictionaries with different choices for BT, C, and OC
    std::vector<std::map<std::string, int>> configurations = {
        // {{"BT", 128}, {"C", 64}, {"OC", 256}},
        // {{"BT", 256}, {"C", 128}, {"OC", 512}},
        {{"BT", 8*128}, {"C", 768}, {"OC", 768}}
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

