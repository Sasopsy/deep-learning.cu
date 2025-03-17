cmake_minimum_required(VERSION 3.10)

# Set the project name
project(kernel_test LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED False)

# Find OpenMP package
find_package(OpenMP)
if(OpenMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Set CUDA flags and architectures
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES "all") # Adjust based on your GPU architecture

# Add CUDA optimization flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} \
    --use_fast_math \
    -O3 \
    --ptxas-options=-v \
    -lineinfo")

# Add release mode optimizations for both C++ and CUDA
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -ftree-vectorize")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

# Include both the project root and the parent directory
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/../..) # Add parent of tests/cpp

# Print the key directories for debugging
message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message(STATUS "Parent directory: ${PROJECT_SOURCE_DIR}/../..")

# Find CUDA package
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Create a list of all test source files (*.cu files in current directory)
file(GLOB TEST_SOURCES "${PROJECT_SOURCE_DIR}/*.cu")
message(STATUS "Found test files: ${TEST_SOURCES}")

# Create executables for each test file
foreach(TEST_SOURCE ${TEST_SOURCES})
    # Extract the filename without extension
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    
    # Add an executable for this test
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    
    # Link CUDA libraries
    target_link_libraries(${TEST_NAME} ${CUDA_LIBRARIES})
    
    # Link OpenMP if found
    if(OpenMP_FOUND)
        target_link_libraries(${TEST_NAME} OpenMP::OpenMP_CXX)
    endif()
    
    # Set the proper include directories for this target
    target_include_directories(${TEST_NAME} PRIVATE 
        ${PROJECT_SOURCE_DIR}         # Current directory
        ${PROJECT_SOURCE_DIR}/../../..   # Parent directory to access csrc
    )
    
    # Set CUDA-specific flags for this target
    set_target_properties(${TEST_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    # Add a custom target to run the test
    add_custom_target(run_${TEST_NAME}
        COMMAND ${TEST_NAME}
        DEPENDS ${TEST_NAME}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running ${TEST_NAME}"
    )
endforeach()

# Optional: Create a target to run all tests
add_custom_target(run_all_tests
    COMMENT "Running all tests"
)

foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_dependencies(run_all_tests run_${TEST_NAME})
endforeach()

# Print a message when configuration is complete
message(STATUS "CMake configuration complete. Available targets:")
message(STATUS "  - Build specific test: make <test_name> (e.g., make flash_attention)")
message(STATUS "  - Run specific test: make run_<test_name> (e.g., make run_flash_attention)")
message(STATUS "  - Run all tests: make run_all_tests")
message(STATUS "  - To build with maximum optimizations: cmake -DCMAKE_BUILD_TYPE=Release ..")