#pragma once
#include <iostream>
#include <cstring>
#include <cuda_runtime.h>
#include <float.h>
#include "../utils/utils.hpp"
#include "../common.hpp"
#include <cuda.h>
// Source: https://github.com/karpathy/llm.c/blob/master/dev/cuda/utils.h#L108
// ----------------------------------------------------------------------------
// Packed128 data structure, which forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.
template<class ElementType>
class alignas(16) Packed128 {
public:
    // e.g. sizeof(int4) is 16 (4 X 4 bytes), sizeof(bfloat16) = 2, so size = 8
    // so in the case where ElementType = bfloat16, we store 8 elements in one Packed128
    static constexpr const int size = sizeof(int4) / sizeof(ElementType);

    // Note: = default implicitly generates a __device__ function, but explicitly
    // adding __device__ causes a lot of warnings.
    Packed128() = default;

    __device__ explicit Packed128(int4 bits) {
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&payload, &bits, sizeof(bits));
    }

    __device__ static Packed128 constant(ElementType value) {
        Packed128 result;
        for(int k = 0; k < size; ++k) {
            result.payload[k] = value;
        }
        return result;
    }

    __device__ static Packed128 zeros() {
        return constant(0);
    }

    __device__ static Packed128 ones() {
        return constant(1);
    }

    __device__ ElementType& operator[](int index) {
        return payload[index];
    }

    __device__ const ElementType& operator[](int index) const {
        return payload[index];
    }

    __device__ int4 get_bits() const {
        int4 bits;
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&bits, &payload, sizeof(bits));
        return bits;
    }

    // Helper functions

    // load a Packed128 from an aligned memory address
    __device__ static Packed128 load128(const ElementType* address) {
        return Packed128{*reinterpret_cast<const int4*>(address)};
    }

    // load a Packed128 from an aligned memory address with streaming cache hint
    __device__ static Packed128 load128cs(const ElementType* address) {
        return Packed128{__ldcs(reinterpret_cast<const int4*>(address))};
    }

    // store a Packed128 to an aligned memory address
    __device__ void store128(ElementType* target) const {
        *reinterpret_cast<int4*>(target) = get_bits();
    }

    // store a Packed128 to an aligned memory address with streaming cache hint
    __device__ void store128cs(ElementType* target) const {
        __stcs(reinterpret_cast<int4*>(target), get_bits());
    }

    // store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
    __device__ void store128cg(ElementType* target) const {
        __stcg(reinterpret_cast<int4*>(target), get_bits());
    }

private:
    ElementType payload[size];
};