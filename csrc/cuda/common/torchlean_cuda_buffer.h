#pragma once

#include <lean/lean.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef TORCHLEAN_LIBTORCH
#include <ATen/ATen.h>
#include <memory>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Lean runtime helpers (shared by CUDA and CPU stubs).
//
// This header is the native side of `NN.Runtime.Autograd.Engine.Cuda.Buffer`.
// The exported functions deliberately keep a tiny ABI:
// - Lean owns an external object that points at `torchlean_cuda_buffer`;
// - `size` is always the number of float32 elements, not bytes;
// - the CUDA build owns an ATen tensor; the CPU stub owns a host `data` allocation;
// - all native callers must validate shape/size metadata before accessing storage.
//
// This is a trusted boundary. The Lean layer can prove shape-level contracts around these calls, but
// it cannot inspect C pointer lifetimes or CUDA runtime behavior.

// Convert a Lean `Nat` to `uint32_t`. In Lean's C runtime, values in this range are represented as
// tagged scalars; boxed or larger naturals fail the conversion.
static inline bool nat_to_u32_checked(b_lean_obj_arg o, uint32_t* out) {
  if (!lean_is_scalar(o)) {
    return false;
  }
  const size_t v = lean_unbox(o);
  if (v > (size_t)UINT32_MAX) {
    return false;
  }
  *out = (uint32_t)v;
  return true;
}

static inline uint32_t nat_to_u32_or_panic(b_lean_obj_arg o, const char* msg) {
  uint32_t v = 0;
  if (!nat_to_u32_checked(o, &v)) {
    lean_internal_panic(msg);
  }
  return v;
}

typedef struct {
  size_t size;  // number of float32 elements
#ifdef TORCHLEAN_LIBTORCH
  at::Tensor tensor;
  // Operator-specific forward state, retained until the owning tape node is released.
  std::shared_ptr<void> context;
#else
  float* data;  // CPU stub storage
  // CPU attention stubs retain their forward state with the same buffer lifetime.
  void* context;
  void (*delete_context)(void*);
#endif
} torchlean_cuda_buffer;

// Helpers implemented by LibTorch runtime.cpp / torchlean_cuda_tensor_stub.c.
torchlean_cuda_buffer* torchlean_cuda_buffer_unbox(b_lean_obj_arg obj);
lean_obj_res torchlean_cuda_buffer_box(torchlean_cuda_buffer* b);
torchlean_cuda_buffer* torchlean_cuda_buffer_alloc(size_t n);
void torchlean_cuda_buffer_drop_unboxed(torchlean_cuda_buffer* b);

static inline lean_object* torchlean_cuda_box_buffer_pair(
    torchlean_cuda_buffer* a,
    torchlean_cuda_buffer* b) {
  lean_object* pair = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(pair, 0, torchlean_cuda_buffer_box(a));
  lean_ctor_set(pair, 1, torchlean_cuda_buffer_box(b));
  return pair;
}

static inline lean_object* torchlean_cuda_box_four_buffers(
    torchlean_cuda_buffer* first,
    torchlean_cuda_buffer* second,
    torchlean_cuda_buffer* third,
    torchlean_cuda_buffer* fourth) {
  lean_object* tail2 = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(tail2, 0, torchlean_cuda_buffer_box(third));
  lean_ctor_set(tail2, 1, torchlean_cuda_buffer_box(fourth));
  lean_object* tail1 = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(tail1, 0, torchlean_cuda_buffer_box(second));
  lean_ctor_set(tail1, 1, tail2);
  lean_object* out = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(out, 0, torchlean_cuda_buffer_box(first));
  lean_ctor_set(out, 1, tail1);
  return out;
}

static inline void torchlean_cuda_require_same_size2(
    const torchlean_cuda_buffer* a,
    const torchlean_cuda_buffer* b,
    const char* fn) {
  if (a->size != b->size) {
    char msg[192];
    snprintf(msg, sizeof(msg), "%s: size mismatch (%zu vs %zu)", fn, a->size, b->size);
    lean_internal_panic(msg);
  }
}

static inline void torchlean_cuda_require_same_size3(
    const torchlean_cuda_buffer* a,
    const torchlean_cuda_buffer* b,
    const torchlean_cuda_buffer* c,
    const char* fn) {
  if (a->size != b->size || a->size != c->size) {
    char msg[224];
    snprintf(msg, sizeof(msg), "%s: size mismatch (%zu vs %zu vs %zu)", fn, a->size, b->size,
             c->size);
    lean_internal_panic(msg);
  }
}

// A broadcast inserts output axes and expands singleton input axes. Every input
// axis must occur exactly once: missing axes can hide an empty input dimension,
// and repeated axes disagree between forward indexing and deterministic VJPs.
// Keep this check shared by CUDA and CPU stubs, before any buffer is accessed.
static inline void torchlean_cuda_require_broadcast_map(
    const uint32_t* axis_map,
    size_t input_rank,
    size_t output_rank,
    const char* fn) {
  size_t mapped_axes = 0;
  for (size_t axis = 0; axis < output_rank; ++axis) {
    const uint32_t mapped = axis_map[axis];
    if (mapped == 0) {
      continue;
    }
    if ((size_t)mapped > input_rank) {
      char msg[224];
      snprintf(msg, sizeof(msg), "%s: axisMap out of range", fn);
      lean_internal_panic(msg);
    }
    for (size_t previous = 0; previous < axis; ++previous) {
      if (axis_map[previous] == mapped) {
        char msg[224];
        snprintf(msg, sizeof(msg), "%s: axisMap repeats input axis", fn);
        lean_internal_panic(msg);
      }
    }
    ++mapped_axes;
  }
  if (mapped_axes != input_rank) {
    char msg[224];
    snprintf(msg, sizeof(msg), "%s: axisMap omits input axis", fn);
    lean_internal_panic(msg);
  }
}

#ifndef TORCHLEAN_LIBTORCH
// Internal CPU policy, captured from the environment on first use and immutable thereafter.
// This helper does not expose a runtime setting through Lean or emulate LibTorch controls.
uint32_t torchlean_cpu_deterministic_reductions(void);
#endif

// Allocator telemetry. These counters are diagnostic only. The allocator counters track payloads
// owned by TorchLean buffers; the wrapper counters track Lean external objects from boxing through
// finalization. LibTorch allocated/reserved storage is reported separately.
LEAN_EXPORT uint64_t torchlean_cuda_allocator_live_bytes(uint32_t u);
LEAN_EXPORT uint64_t torchlean_cuda_allocator_peak_bytes(uint32_t u);
LEAN_EXPORT uint64_t torchlean_cuda_allocator_alloc_count(uint32_t u);
LEAN_EXPORT uint64_t torchlean_cuda_allocator_free_count(uint32_t u);
LEAN_EXPORT uint64_t torchlean_cuda_wrapper_live_count(uint32_t u);
LEAN_EXPORT uint64_t torchlean_cuda_wrapper_peak_count(uint32_t u);
LEAN_EXPORT uint64_t torchlean_cuda_wrapper_alloc_count(uint32_t u);
LEAN_EXPORT uint64_t torchlean_cuda_wrapper_finalize_count(uint32_t u);
LEAN_EXPORT uint64_t torchlean_cuda_allocator_device_free_bytes(uint32_t u);
LEAN_EXPORT uint64_t torchlean_cuda_allocator_device_total_bytes(uint32_t u);

// Lean `Buffer × Buffer × Buffer` as nested pairs.
static inline lean_object* torchlean_cuda_box_three_buffers(
    torchlean_cuda_buffer* first, torchlean_cuda_buffer* second,
    torchlean_cuda_buffer* third) {
  lean_object* tail = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(tail, 0, torchlean_cuda_buffer_box(second));
  lean_ctor_set(tail, 1, torchlean_cuda_buffer_box(third));
  lean_object* out = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(out, 0, torchlean_cuda_buffer_box(first));
  lean_ctor_set(out, 1, tail);
  return out;
}

#ifdef __cplusplus
}  // extern "C"
#endif
