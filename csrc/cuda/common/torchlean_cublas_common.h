#pragma once

#include <lean/lean.h>

#include "torchlean_cuda_common.h"

#include <cublas_v2.h>

// Shared cuBLAS helpers (header-only).
//
// TorchLean stores tensors in Lean-facing row-major buffers, while cuBLAS speaks column-major by
// default. The call sites document each row-major/column-major trick locally; this header only owns
// the reusable handle/error policy.
//
// FFI contract:
// - cuBLAS failures are not recoverable Lean exceptions; they are runtime-boundary panics.
// - one handle is cached per host thread and per active CUDA device;
// - callers must still validate dimensions before passing them to cuBLAS' `int`-shaped API.

static inline void checkCublas(cublasStatus_t s, const char* msg) {
  if (s != CUBLAS_STATUS_SUCCESS) {
    lean_internal_panic(msg);
  }
}

// cuBLAS handle creation is relatively expensive; cache one per host thread.
// We also track the current CUDA device to avoid reusing a handle across devices.
static inline cublasHandle_t getCublasHandle() {
  static thread_local cublasHandle_t handle = nullptr;
  static thread_local int handleDevice = -1;

  int dev = -1;
  checkCuda(cudaGetDevice(&dev), "cudaGetDevice failed");

  if (handle == nullptr || handleDevice != dev) {
    if (handle != nullptr) {
      // Best-effort cleanup on device switches.
      checkCublas(cublasDestroy(handle), "cublasDestroy failed");
      handle = nullptr;
      handleDevice = -1;
    }
    checkCublas(cublasCreate(&handle), "cublasCreate failed");
    handleDevice = dev;
  }

  return handle;
}
