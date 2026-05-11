#pragma once

#include "torchlean_cuda_buffer.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

// Shared helpers for convolution and pooling FFI implementations.
//
// Both CUDA and CPU-stub files include this header so their shape arithmetic stays synchronized.
// The formulas intentionally mirror TorchLean's Nat-style specs: subtractions truncate at zero
// instead of becoming negative, and host wrappers reject invalid kernel/stride metadata before work
// reaches the device.

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
  // TorchLean spec uses Nat subtraction (truncated at 0).
  uint64_t inPad = (uint64_t)in + 2ull * (uint64_t)padding;
  uint64_t numer = (inPad >= (uint64_t)k) ? (inPad - (uint64_t)k) : 0ull;
  uint64_t out = numer / (uint64_t)stride + 1ull;
  return (uint32_t)out;
}

static inline uint32_t outDimTranspose(uint32_t in, uint32_t k, uint32_t stride, uint32_t padding) {
  // TorchLean spec uses Nat subtraction (truncated at 0):
  //   out = (in - 1) * stride - 2 * padding + k
  uint64_t t = (in > 0) ? ((uint64_t)(in - 1) * (uint64_t)stride) : 0ull;
  uint64_t sub = 2ull * (uint64_t)padding;
  if (t >= sub) {
    t -= sub;
  } else {
    t = 0ull;
  }
  t += (uint64_t)k;
  if (t > (uint64_t)UINT32_MAX) {
#if defined(__CUDACC__)
    lean_internal_panic("torchlean_cuda_conv_pool: outDimTranspose overflow");
#else
    lean_internal_panic("torchlean_cuda_conv_pool_stub: outDimTranspose overflow");
#endif
  }
  return (uint32_t)t;
}

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
    uint32_t d = nat_to_u32_or_oob(lean_array_get_core(arrObj, (size_t)i));
    if (d == UINT32_MAX) {
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
  size_t p = 1;
  for (int i = 0; i < n; ++i) {
    p *= (size_t)dims[i];
  }
  return p;
}

#if defined(__CUDACC__)
#define TORCHLEAN_CUDA_CONV_POOL_HD __host__ __device__
#else
#define TORCHLEAN_CUDA_CONV_POOL_HD
#endif

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
