#pragma once

#ifndef TORCHLEAN_LIBTORCH
#error "Compile the LibTorch backend with TORCHLEAN_LIBTORCH defined."
#endif

#include <lean/lean.h>

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/core/DeviceGuard.h>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <utility>
#include <vector>

// Native side of `NN.Runtime.Autograd.Engine.Cuda.Buffer`. Lean owns an external object that points
// at a `torchlean_cuda_buffer`; `size` counts float32 elements, not bytes. Callers validate shape
// metadata before touching storage. This is a trusted boundary: Lean proves shape contracts around
// these calls but cannot see tensor lifetimes or CUDA behavior.
struct torchlean_cuda_buffer {
  size_t size;
  at::Tensor tensor;
  // Operator-specific forward state, retained until the owning tape node is released.
  std::shared_ptr<void> context;
};

extern "C" {
torchlean_cuda_buffer* torchlean_cuda_buffer_unbox(b_lean_obj_arg obj);
lean_obj_res torchlean_cuda_buffer_box(torchlean_cuda_buffer* b);
torchlean_cuda_buffer* torchlean_cuda_buffer_alloc(size_t n);
void torchlean_cuda_buffer_drop_unboxed(torchlean_cuda_buffer* b);
LEAN_EXPORT uint32_t torchlean_cuda_buffer_release(b_lean_obj_arg object);

// Diagnostic counters. The allocator counters track payloads owned by TorchLean buffers; the
// wrapper counters track Lean external objects from boxing through finalization.
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
}

// Lean `Nat` to `uint32_t`. Values in range are tagged scalars; boxed naturals fail.
inline bool nat_to_u32_checked(b_lean_obj_arg o, uint32_t* out) {
  if (!lean_is_scalar(o)) return false;
  const size_t v = lean_unbox(o);
  if (v > UINT32_MAX) return false;
  *out = static_cast<uint32_t>(v);
  return true;
}

inline uint32_t nat_to_u32_or_panic(b_lean_obj_arg o, const char* msg) {
  uint32_t v = 0;
  if (!nat_to_u32_checked(o, &v)) lean_internal_panic(msg);
  return v;
}

inline size_t checked_mul_size(size_t a, size_t b, const char* msg) {
  if (a != 0 && b > SIZE_MAX / a) lean_internal_panic(msg);
  return a * b;
}

inline size_t checked_bytes_size(size_t count, size_t elemSize, const char* msg) {
  return checked_mul_size(count, elemSize, msg);
}

namespace torchlean {

void initialize();
c10::Device device();

// The Lean tape owns differentiation. Native calls must not record a second graph.
template <typename F>
auto invoke(F&& body) -> decltype(body()) {
  try {
    initialize();
    at::NoGradGuard no_grad;
    c10::DeviceGuard guard(device());
    return std::forward<F>(body)();
  } catch (const std::exception& error) {
    lean_internal_panic(error.what());
    std::abort();
  }
}

const at::Tensor& tensor(b_lean_obj_arg object);
at::TensorOptions options();
torchlean_cuda_buffer* owned(at::Tensor value);
lean_obj_res box(at::Tensor value);

inline at::Tensor shaped(b_lean_obj_arg object, at::IntArrayRef shape) {
  return tensor(object).reshape(shape);
}

inline void require(bool condition, const char* message) {
  if (!condition) {
    lean_internal_panic(message);
    std::abort();
  }
}

inline std::vector<int64_t> dimensions(b_lean_obj_arg values, const char* message) {
  const size_t n = lean_array_size(values);
  std::vector<int64_t> result;
  result.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    result.push_back(nat_to_u32_or_panic(lean_array_uget(values, i), message));
  }
  return result;
}

// Lean `A × B` is a constructor with two object fields; longer tuples nest to the right.
inline lean_obj_res cons(lean_obj_res first, lean_obj_res second) {
  lean_object* out = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(out, 0, first);
  lean_ctor_set(out, 1, second);
  return out;
}

inline lean_obj_res pair(at::Tensor first, at::Tensor second) {
  const auto drop = torchlean_cuda_buffer_drop_unboxed;
  std::unique_ptr<torchlean_cuda_buffer, decltype(drop)> a(owned(std::move(first)), drop);
  std::unique_ptr<torchlean_cuda_buffer, decltype(drop)> b(owned(std::move(second)), drop);
  return cons(torchlean_cuda_buffer_box(a.release()), torchlean_cuda_buffer_box(b.release()));
}

inline lean_obj_res triple(at::Tensor first, at::Tensor second, at::Tensor third) {
  const auto drop = torchlean_cuda_buffer_drop_unboxed;
  std::unique_ptr<torchlean_cuda_buffer, decltype(drop)> a(owned(std::move(first)), drop);
  std::unique_ptr<torchlean_cuda_buffer, decltype(drop)> b(owned(std::move(second)), drop);
  std::unique_ptr<torchlean_cuda_buffer, decltype(drop)> c(owned(std::move(third)), drop);
  return cons(torchlean_cuda_buffer_box(a.release()),
              cons(torchlean_cuda_buffer_box(b.release()), torchlean_cuda_buffer_box(c.release())));
}

inline lean_obj_res quadruple(
    at::Tensor first, at::Tensor second, at::Tensor third, at::Tensor fourth) {
  const auto drop = torchlean_cuda_buffer_drop_unboxed;
  std::unique_ptr<torchlean_cuda_buffer, decltype(drop)> a(owned(std::move(first)), drop);
  std::unique_ptr<torchlean_cuda_buffer, decltype(drop)> b(owned(std::move(second)), drop);
  std::unique_ptr<torchlean_cuda_buffer, decltype(drop)> c(owned(std::move(third)), drop);
  std::unique_ptr<torchlean_cuda_buffer, decltype(drop)> d(owned(std::move(fourth)), drop);
  return cons(torchlean_cuda_buffer_box(a.release()),
              cons(torchlean_cuda_buffer_box(b.release()),
                   cons(torchlean_cuda_buffer_box(c.release()),
                        torchlean_cuda_buffer_box(d.release()))));
}

}  // namespace torchlean
