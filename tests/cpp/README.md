
# CPP Tests for Deep Learning Kernels

This document provides instructions on how to build and run the CPP tests for the deep learning kernels.

## Prerequisites

- CMake (version 3.10 or higher)
- CUDA Toolkit
- A C++ compiler with OpenMP support

## Building the Tests

1. Navigate to the `tests/cpp` directory:
    ```sh
    cd /root/deep-learning.cu/tests/cpp/kernels
    ```

2. Create a build directory and navigate into it:
    ```sh
    mkdir build
    cd build
    ```

3. Run CMake to configure the project:
    ```sh
    cmake ..
    ```

4. Build the project:
    ```sh
    make
    ```

## Running the Tests

To run a specific test, use the following command:
```sh
make run_<test_name>
```
For example, to run the `softmax` test:
```sh
make run_softmax
```

### Test Options

Each test executable may prompt for specific options. Below is an example for the softmax test.

#### Softmax Test

- **Kernel Choices:**
  - `0`: Naive Kernel
  - `1`: Shared Memory Optimized Kernel
  - `2`: Intra Warp Optimized Kernel

- **User Prompts:**
  - Enter `0` to test all kernels, or `1` to choose a specific kernel.
  - If choosing a specific kernel, enter the kernel choice.

## Additional Information

- The tests will display the parameters used and the average execution time for each kernel.
- Results will be compared against CPU reference implementations to ensure correctness.
