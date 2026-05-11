#pragma once

#include <lean/lean.h>

#include <cuda_runtime.h>

// Shared CUDA helpers (header-only).
//
// Native CUDA failures are treated as boundary failures, not as ordinary recoverable Lean values.
// This keeps the FFI surface simple and makes backend bugs fail loudly during tests. Host-side
// wrappers should perform deterministic validation (shape, rank, index arrays) before launching
// kernels; `checkCuda` is only for CUDA runtime/driver errors.

static inline void checkCuda(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    lean_internal_panic(msg);
  }
}

static inline void torchlean_cuda_free_checked(void** ptr, const char* msg) {
  if (ptr && *ptr) {
    checkCuda(cudaFree(*ptr), msg);
    *ptr = nullptr;
  }
}
