#pragma once

#include <lean/lean.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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
  size_t size;       // number of float32 elements
  float* data;       // device/host pointer (depending on build)
  void* arena_reg;   // non-NULL while tracked by an open `withCudaArena` scope (see torchlean_cuda_arena.h)
  size_t arena_freed_depth;  // debug only: 0 = live; else (reclaiming arena depth + 1). See the UAF detector below.
} torchlean_cuda_buffer;

// Helpers implemented by `torchlean_cuda_tensor.cu` / `torchlean_cuda_tensor_stub.c`.
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

// Arena use-after-free detector (debug only; opt in with `TORCHLEAN_ARENA_DEBUG=1`).
//
// A device buffer reclaimed by `arena_exit` keeps its struct (Lean still owns the external object) but
// has `size == 0`/`data == NULL`, and in debug mode `arena_freed_depth` records the epoch that freed
// it (depth + 1). Touching it afterwards is the same hazard as using a `release`d buffer. The size
// helpers below are the common choke point for every binary/ternary op, so asserting liveness here
// turns a stale operand into a panic naming the epoch that freed it, instead of a silent corrupt
// kernel launch — and it catches the case the bare size check cannot (both operands freed ⇒ `0 == 0`).
// When the flag is off, the whole detector is one predicted branch on a cached int.
static inline int torchlean_arena_debug_enabled(void) {
  static int cached = -1;  // -1 = env not yet read
  if (cached < 0) {
    const char* v = getenv("TORCHLEAN_ARENA_DEBUG");
    cached = (v && v[0] && !(v[0] == '0' && v[1] == '\0')) ? 1 : 0;
  }
  return cached;
}

static inline void torchlean_arena_assert_live(
    const torchlean_cuda_buffer* b, const char* role, const char* fn) {
  if (b->arena_freed_depth != 0) {
    char msg[224];
    snprintf(msg, sizeof(msg),
             "use-after-arena-free: %s %s was reclaimed by arena_exit at depth %zu",
             fn, role, b->arena_freed_depth - 1);
    lean_internal_panic(msg);
  }
}

static inline void torchlean_cuda_require_same_size2(
    const torchlean_cuda_buffer* a,
    const torchlean_cuda_buffer* b,
    const char* fn) {
  if (torchlean_arena_debug_enabled()) {
    torchlean_arena_assert_live(a, "lhs", fn);
    torchlean_arena_assert_live(b, "rhs", fn);
  }
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
  if (torchlean_arena_debug_enabled()) {
    torchlean_arena_assert_live(a, "arg0", fn);
    torchlean_arena_assert_live(b, "arg1", fn);
    torchlean_arena_assert_live(c, "arg2", fn);
  }
  if (a->size != b->size || a->size != c->size) {
    char msg[224];
    snprintf(msg, sizeof(msg), "%s: size mismatch (%zu vs %zu vs %zu)", fn, a->size, b->size,
             c->size);
    lean_internal_panic(msg);
  }
}

// Deterministic reductions toggle.
//
// Some kernels use `atomicAdd`, which is fast but can be non-deterministic. When enabled, TorchLean
// uses fixed-order reductions for reproducibility (slower).
LEAN_EXPORT void torchlean_cuda_set_deterministic_reductions(uint32_t on);
LEAN_EXPORT uint32_t torchlean_cuda_get_deterministic_reductions();
LEAN_EXPORT uint32_t torchlean_cuda_get_deterministic_reductions_u(uint32_t u);

// Wrapper used by the Lean binding: sets the flag and returns the observed value.
LEAN_EXPORT uint32_t torchlean_cuda_set_deterministic_reductions_checked(uint32_t on);

// Allocator telemetry. These counters are diagnostic only. The allocator counters track payloads
// created through `torchlean_cuda_buffer_alloc`; the wrapper counters track Lean external objects
// from boxing through finalization.
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

// Scoped device-memory arena (`withCudaArena`).  `enter` opens an allocation epoch; `exit` reclaims
// every device buffer allocated in it except the `keep` array, which is promoted to the parent scope.
// See `torchlean_cuda_arena.h` for the model.  Both are IO actions on the Lean side.
LEAN_EXPORT lean_obj_res torchlean_cuda_arena_enter(lean_obj_arg world);
LEAN_EXPORT lean_obj_res torchlean_cuda_arena_exit(b_lean_obj_arg keep, lean_obj_arg world);

#ifdef __cplusplus
}  // extern "C"
#endif
