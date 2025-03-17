#pragma once
#include <cstring>
#include <cuda_runtime.h>
#include <float.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda.h>

// Intrinsic functions for float and floatX conversions 
template <typename floatX>
__device__ __forceinline__ float floatX_to_float(floatX val);
template <typename floatX>
__device__ __forceinline__ floatX float_to_floatX(float val);

// Template specializations for different data types
template <>
__device__ __forceinline__ float floatX_to_float<__half>(__half val) { return __half2float(val); }
template <>
__device__ __forceinline__ __half float_to_floatX<__half>(float val) { return __float2half(val); }

// template <>
// __device__ __forceinline__ float floatX_to_float<__nv_bfloat16>(__nv_bfloat16 val) { return __bfloat162float(val); }
// template <>
// __device__ __forceinline__ __nv_bfloat16 float_to_floatX<__nv_bfloat16>(float val) { return __float2bfloat16(val); }

template <>
__device__ __forceinline__ float floatX_to_float<float>(float val) { return val; }
template <>
__device__ __forceinline__ float float_to_floatX<float>(float val) { return val; }

// ----------------------------------------------------------------------------
// Intrinsic for getting floatX zero
template <typename floatX>
__device__ inline floatX get_zero() {};

template <>
__device__ inline float get_zero<float>() {
    return 0.0f;
}

template <>
__device__ inline __half get_zero<__half>() {
    return __ushort_as_half((unsigned short)0x0000U);
}

// template <>
// __device__ inline __nv_bfloat16 get_zero<__nv_bfloat16>() {
//     return __ushort_as_bfloat16((unsigned short)0x0000U);
// }

// ----------------------------------------------------------------------------
// Intrinsic for get floatX negative infinity 
template <typename floatX>
__device__ __forceinline__ constexpr floatX get_neg_inf() {
    // Primary template handles unsupported types
    static_assert(sizeof(floatX) == 0, "Unsupported type for get_neg_inf");
}

// Specialization for float (fp32)
template <>
__device__ __forceinline__ constexpr float get_neg_inf<float>() {
    return -INFINITY;
}

// Specialization for __half precision (fp16)
template <>
__device__ __forceinline__ __half get_neg_inf<__half>() {
    return __ushort_as_half(0xFC00); // CUDA intrinsic for bit pattern conversion
}


// // Specialization for bfloat16
// template <>
// __device__ __forceinline__ __nv_bfloat16 get_neg_inf<__nv_bfloat16>() {
//     return __ushort_as_bfloat16(0xFF80); // CUDA intrinsic for bfloat16
// }

// ----------------------------------------------------------------------------
// Max intrinsic for different precisions
template <typename floatX>
__device__ __forceinline__ floatX max(floatX a, floatX b);

template <>
__device__ __forceinline__ float max<float>(float a, float b) {
    // Fast fmaxf intrinsic with IEEE-754 compliance
    return fmaxf(a, b);
}

template <>
__device__ __forceinline__ __half max<__half>(__half a, __half b) {
    // Direct hardware-accelerated __half-precision max
    return __hmax(a, b);
}

// template <>
// __device__ __forceinline__ __nv_bfloat16 max<__nv_bfloat16>(__nv_bfloat16 a, __nv_bfloat16 b) {
//     // BF16 needs conversion to float for accurate comparison
//     return __float2bfloat16(fmaxf(__bfloat162float(a), __bfloat162float(b)));
// }

// ----------------------------------------------------------------------------
// Exp intrinsic for different precisions
template <typename floatX>
__device__ __forceinline__ floatX exp(floatX x);

template <>
__device__ __forceinline__ float exp<float>(float x) {
    // Fast approximate exp (2 ULP error, ~33B ops/sec)
    return __expf(x); 
}

template <>
__device__ __forceinline__ __half exp<__half>(__half x) {
    // Convert to float for computation, maintain precision
    return __float2half(expf(__half2float(x)));
}

// template <>
// __device__ __forceinline__ __nv_bfloat16 exp<__nv_bfloat16>(__nv_bfloat16 x) {
//     // BF16 requires float conversion path
//     return __float2bfloat16(expf(__bfloat162float(x)));
// }