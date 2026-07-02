#pragma once

#include <lean/lean.h>

#include <limits.h>
#include <stdint.h>
#include <stddef.h>

// Size arithmetic used by both CUDA code and CPU stubs.
// Keep it shared so impossible buffers fail the same way in both builds.

static inline size_t checked_mul_size(size_t a, size_t b, const char* msg) {
  if (a != 0 && b > SIZE_MAX / a) {
    lean_internal_panic(msg);
  }
  return a * b;
}

static inline size_t checked_mul3_size(size_t a, size_t b, size_t c, const char* msg) {
  return checked_mul_size(checked_mul_size(a, b, msg), c, msg);
}

static inline size_t checked_mul4_size(size_t a, size_t b, size_t c, size_t d, const char* msg) {
  return checked_mul_size(checked_mul3_size(a, b, c, msg), d, msg);
}

static inline size_t checked_mul_acc_size(size_t acc, size_t next, const char* msg) {
  return checked_mul_size(acc, next, msg);
}

static inline size_t checked_add_size(size_t a, size_t b, const char* msg) {
  if (b > SIZE_MAX - a) {
    lean_internal_panic(msg);
  }
  return a + b;
}

static inline size_t checked_bytes_size(size_t count, size_t elemSize, const char* msg) {
  return checked_mul_size(count, elemSize, msg);
}

static inline uint64_t torchlean_float_bytes_for(size_t count) {
  const uint64_t max = UINT64_MAX / (uint64_t)sizeof(float);
  if ((uint64_t)count > max) {
    return UINT64_MAX;
  }
  return (uint64_t)count * (uint64_t)sizeof(float);
}
