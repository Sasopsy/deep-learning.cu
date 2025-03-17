#include <iostream>
#include <cuda_runtime.h>
#include <cassert>
#include <map>
#include <vector>
#include "../test_utils.hpp"
#include "csrc/kernels/residual/launch.cuh"

namespace ck = cuda_kernel;

void checkResults(const float* cpu_result, const float* gpu_result, int size) {
    bool passed = compareResults(cpu_result, gpu_result, size, 1e-5f, true);
    if (passed) {
        std::cout << "Results match within tolerance!" << std::endl;
    }
}

void measureKernelTime(float* d_output, const float* d_input, const float* d_residual, int size, int kernel_choice) {
    auto kernel = [&]() {
        ck::residual::launch<float>(d_output, d_input, d_residual, size, kernel_choice);
    };
    
    float ms = measureKernelPerformance(kernel);
    std::cout << "Kernel " << kernel_choice << " average execution time: " << ms << " ms" << std::endl;
}

void displayKernelChoices() {
    std::cout << "Kernel Choices:" << std::endl;
    std::cout << "0: Naive Kernel" << std::endl;
    std::cout << "1: Vectorised Kernel" << std::endl;
}

int getUserKernelChoice() {
    int choice;
    std::cout << "Enter the kernel choice: ";
    std::cin >> choice;
    return choice;
}

void printParams(int size, int kernel_choice) {
    std::cout << "Parameters used:" << std::endl;
    std::cout << "Size: " << size << std::endl;
    std::cout << "Kernel Choice: " << kernel_choice << std::endl;
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

void testAllKernels(const std::vector<int>& sizes) {
    for (const auto& size : sizes) {
        std::cout << "Testing size: " << size << std::endl;

        // Allocate memory and initialize data
        HostDevice<float> input(size);
        HostDevice<float> residual(size);
        HostDevice<float> output(size);
        HostDevice<float> cpu_output(size);

        fillRandom(input.host, size);
        fillRandom(residual.host, size);

        input.copyToDevice();
        residual.copyToDevice();

        for (int kernel_choice = 0; kernel_choice < 2; ++kernel_choice) {
            std::cout << "Testing kernel choice: " << kernel_choice << std::endl;

            // Launch the selected kernel
            ck::residual::launch<float>(output.device, input.device, residual.device, size, kernel_choice);

            // Copy results back to host
            output.copyToHost();

            // Compute reference result on CPU
            ck::residual::cpu(cpu_output.host, input.host, residual.host, size);

            // Check results
            checkResults(cpu_output.host, output.host, size);

            // Measure kernel performance
            measureKernelTime(output.device, input.device, residual.device, size, kernel_choice);

            // Print parameters used
            printParams(size, kernel_choice);

            std::cout << " " << std::endl;
        }
    }
}

void testSpecificKernel(const std::vector<int>& sizes, int kernel_choice) {
    for (const auto& size : sizes) {
        std::cout << "Testing size: " << size << std::endl;

        // Allocate memory and initialize data
        HostDevice<float> input(size);
        HostDevice<float> residual(size);
        HostDevice<float> output(size);
        HostDevice<float> cpu_output(size);

        fillRandom(input.host, size);
        fillRandom(residual.host, size);

        input.copyToDevice();
        residual.copyToDevice();

        std::cout << "Testing kernel choice: " << kernel_choice << std::endl;

        // Launch the selected kernel
        ck::residual::launch<float>(output.device, input.device, residual.device, size, kernel_choice);

        // Copy results back to host
        output.copyToHost();

        // Compute reference result on CPU
        ck::residual::cpu(cpu_output.host, input.host, residual.host, size);

        // Check results
        checkResults(cpu_output.host, output.host, size);

        // Measure kernel performance
        measureKernelTime(output.device, input.device, residual.device, size, kernel_choice);

        // Print parameters used
        printParams(size, kernel_choice);

        std::cout << " " << std::endl;
    }
}

int main() {
    // List of sizes to test
    std::vector<int> sizes = {8*128*768};

    // Get user choice
    int user_choice = getUserChoice();

    if (user_choice == 0) {
        // Test all kernels with different sizes
        testAllKernels(sizes);
    } else {
        // Display kernel choices and get user selection
        displayKernelChoices();
        int kernel_choice = getUserKernelChoice();
        testSpecificKernel(sizes, kernel_choice);
    }

    return 0;
}
