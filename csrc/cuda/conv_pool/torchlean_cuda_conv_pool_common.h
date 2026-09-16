#pragma once

#include "torchlean_cuda_buffer.h"
#include "torchlean_size_common.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

// Shape math shared by the CUDA conv/pool code and the CPU stub.
// TorchLean shapes use Nat subtraction, so negative intermediate results clamp at zero.

enum { TORCHLEAN_CUDA_CONV_POOL_MAX_RANK = 8 };

static inline void checkBufSize(torchlean_cuda_buffer* b, size_t elems, const char* msg) {
  if (b->size != elems) {
    lean_internal_panic(msg);
  }
}

static inline uint32_t outDim(uint32_t in, uint32_t k, uint32_t stride, uint32_t padding) {
  if (stride == 0) {
    lean_internal_panic("torchlean_cuda_conv_pool: stride must be > 0");
  }
  if (k == 0) {
    return 0;
  }
  // Invalid geometry has no windows; do not turn saturated subtraction into a phantom window.
  uint64_t inPad = (uint64_t)in + 2ull * (uint64_t)padding;
  if (inPad < (uint64_t)k) {
    return 0;
  }
  uint64_t numer = inPad - (uint64_t)k;
  uint64_t out = numer / (uint64_t)stride + 1ull;
  if (out > (uint64_t)UINT32_MAX) {
#if defined(__CUDACC__)
    lean_internal_panic("torchlean_cuda_conv_pool: outDim overflow");
#else
    lean_internal_panic("torchlean_cuda_conv_pool_stub: outDim overflow");
#endif
  }
  return (uint32_t)out;
}

// N-D pooling follows `Spec.poolOutSpatialPad`: empty inputs and padding beyond half the kernel
// are invalid axes, even when the generic sliding-window formula would produce a positive length.
static inline uint32_t poolOutDim(uint32_t in, uint32_t k, uint32_t stride, uint32_t padding) {
  if (in == 0 || k == 0 || padding > k / 2u) {
    return 0;
  }
  return outDim(in, k, stride, padding);
}

static inline uint32_t outDimTranspose(uint32_t in, uint32_t k, uint32_t stride, uint32_t padding) {
  if (stride == 0) {
    lean_internal_panic("torchlean_cuda_conv_pool: stride must be > 0");
  }
  if (in == 0 || k == 0) {
    return 0;
  }
  // TorchLean spec uses Nat subtraction (truncated at 0):
  //   out = ((in - 1) * stride + k) - 2 * padding
  // Addition must precede the saturated subtraction.
  uint64_t t = (uint64_t)(in - 1) * (uint64_t)stride + (uint64_t)k;
  uint64_t sub = 2ull * (uint64_t)padding;
  if (t >= sub) {
    t -= sub;
  } else {
    t = 0ull;
  }
  if (t > (uint64_t)UINT32_MAX) {
#if defined(__CUDACC__)
    lean_internal_panic("torchlean_cuda_conv_pool: outDimTranspose overflow");
#else
    lean_internal_panic("torchlean_cuda_conv_pool_stub: outDimTranspose overflow");
#endif
  }
  return (uint32_t)t;
}

// Validate after the ABI's binary64-to-binary32 conversion: a finite nonzero Lean `Float` may
// overflow to infinity or underflow to zero when passed to a float32 kernel.
static inline float checked_smoothmax_beta(double beta, const char* msg) {
  const float betaF = (float)beta;
  if (!isfinite(betaF) || betaF == 0.0f) {
    lean_internal_panic(msg);
  }
  return betaF;
}

static inline void read_u32_array(b_lean_obj_arg arrObj, uint32_t* out, int n, const char* msg) {
  if (!lean_is_array((lean_object*)arrObj)) {
    lean_internal_panic(msg);
  }
  if ((int)lean_array_size(arrObj) != n) {
    lean_internal_panic(msg);
  }
  for (int i = 0; i < n; ++i) {
    uint32_t d = 0;
    if (!nat_to_u32_checked(lean_array_get_core(arrObj, (size_t)i), &d)) {
      lean_internal_panic(msg);
    }
    out[i] = d;
  }
}

static inline int read_rank_checked(b_lean_obj_arg arrObj, const char* msg) {
  if (!lean_is_array((lean_object*)arrObj)) {
    lean_internal_panic(msg);
  }
  const size_t n = lean_array_size(arrObj);
  if (n > (size_t)TORCHLEAN_CUDA_CONV_POOL_MAX_RANK) {
    lean_internal_panic(msg);
  }
  return (int)n;
}

static inline size_t prod_u32(const uint32_t* dims, int n) {
  // Match Nat product: any zero axis makes the product zero, regardless of axis order.
  for (int i = 0; i < n; ++i) {
    if (dims[i] == 0u) {
      return 0;
    }
  }
  size_t p = 1;
  for (int i = 0; i < n; ++i) {
    p = checked_mul_size(p, (size_t)dims[i], "torchlean_cuda_conv_pool: dimension product overflow");
  }
  return p;
}

static inline size_t checked_channel_spatial_size(uint32_t channels, size_t spatialSize,
                                                  const char* msg) {
  return checked_mul_size((size_t)channels, spatialSize, msg);
}

static inline size_t checked_conv_kernel_size(uint32_t outerC, uint32_t innerC,
                                              size_t spatialSize, const char* msg) {
  return checked_mul_size(
      checked_mul_size((size_t)outerC, (size_t)innerC, msg), spatialSize, msg);
}

#if defined(__CUDACC__)
#define TORCHLEAN_CUDA_CONV_POOL_HD __host__ __device__
#else
#define TORCHLEAN_CUDA_CONV_POOL_HD
#endif

// Keep padded coordinates wide until the bounds check. A UInt32 product plus a UInt32
// kernel coordinate fits UInt64, but can overflow signed arithmetic or a narrowed index.
TORCHLEAN_CUDA_CONV_POOL_HD static inline int window_input_coord(
    uint32_t out, uint32_t kernel, uint32_t stride, uint32_t padding,
    uint32_t inputSize, uint32_t* input) {
  const uint64_t padded = (uint64_t)out * (uint64_t)stride + (uint64_t)kernel;
  if (padded < padding || padded - padding >= inputSize) {
    return 0;
  }
  *input = (uint32_t)(padded - padding);
  return 1;
}

// Invert input = output * stride + kernel - padding, requiring exact divisibility.
TORCHLEAN_CUDA_CONV_POOL_HD static inline int window_output_coord(
    uint32_t input, uint32_t kernel, uint32_t stride, uint32_t padding,
    uint32_t outputSize, uint32_t* output) {
  const uint64_t padded = (uint64_t)input + (uint64_t)padding;
  if (stride == 0 || padded < kernel) {
    return 0;
  }
  const uint64_t offset = padded - kernel;
  const uint64_t index = offset / stride;
  if (offset % stride != 0 || index >= outputSize) {
    return 0;
  }
  *output = (uint32_t)index;
  return 1;
}

TORCHLEAN_CUDA_CONV_POOL_HD static inline int64_t floor_div_i64(int64_t a, int64_t b) {
  // b must be > 0. Integer division truncates toward 0; use floor for negatives.
  int64_t q = a / b;
  int64_t r = a % b;
  if (r != 0 && a < 0) {
    q -= 1;
  }
  return q;
}

TORCHLEAN_CUDA_CONV_POOL_HD static inline int64_t ceil_div_i64(int64_t a, int64_t b) {
  // b must be > 0.
  return -floor_div_i64(-a, b);
}

#undef TORCHLEAN_CUDA_CONV_POOL_HD
