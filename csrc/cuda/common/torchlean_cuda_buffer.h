#pragma once

#include <lean/lean.h>

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Lean runtime helpers (shared by CUDA and CPU stubs).
//
// This header is the native side of `NN.Runtime.Autograd.Engine.Cuda.Buffer`.
// The exported functions deliberately keep a tiny ABI:
// - Lean owns an external object that points at `torchlean_cuda_buffer`;
// - `size` is always the number of float32 elements, not bytes;
// - `data` points to device memory in the CUDA build and host memory in the CPU-stub build;
// - all native callers must validate shape/size metadata before touching `data`.
//
// This is a trusted boundary. The Lean layer can prove shape-level contracts around these calls, but
// it cannot inspect C pointer lifetimes or CUDA runtime behavior.

// Convert a Lean `Nat` to `uint32_t`, treating non-scalars / large values as out-of-bounds.
//
// In Lean's C runtime, small naturals are represented as tagged scalars; non-scalars are treated
// as out-of-bounds.
static inline uint32_t nat_to_u32_or_oob(b_lean_obj_arg o) {
  if (!lean_is_scalar(o)) {
    return UINT32_MAX;
  }
  const size_t v = lean_unbox(o);
  if (v > (size_t)UINT32_MAX) {
    return UINT32_MAX;
  }
  return (uint32_t)v;
}

static inline uint32_t nat_to_u32_or_panic(b_lean_obj_arg o, const char* msg) {
  uint32_t v = nat_to_u32_or_oob(o);
  if (v == UINT32_MAX) {
    lean_internal_panic(msg);
  }
  return v;
}

typedef struct {
  size_t size;  // number of float32 elements
  float* data;  // device/host pointer (depending on build)
} torchlean_cuda_buffer;

// Helpers implemented by `torchlean_cuda_tensor.cu` / `torchlean_cuda_tensor_stub.c`.
torchlean_cuda_buffer* torchlean_cuda_buffer_unbox(b_lean_obj_arg obj);
lean_obj_res torchlean_cuda_buffer_box(torchlean_cuda_buffer* b);
torchlean_cuda_buffer* torchlean_cuda_buffer_alloc(size_t n);

// Deterministic reductions toggle.
//
// Some kernels use `atomicAdd`, which is fast but can be non-deterministic. When enabled, TorchLean
// uses fixed-order reductions for reproducibility (slower).
LEAN_EXPORT void torchlean_cuda_set_deterministic_reductions(uint32_t on);
LEAN_EXPORT uint32_t torchlean_cuda_get_deterministic_reductions();
LEAN_EXPORT uint32_t torchlean_cuda_get_deterministic_reductions_u(uint32_t u);

// Wrapper used by the Lean binding: sets the flag and returns the observed value.
LEAN_EXPORT uint32_t torchlean_cuda_set_deterministic_reductions_checked(uint32_t on);

#ifdef __cplusplus
}  // extern "C"
#endif
