#pragma once

#ifndef TORCHLEAN_LIBTORCH
#error "Compile the LibTorch backend with TORCHLEAN_LIBTORCH defined."
#endif

#include <lean/lean.h>

#include "stable.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// Native side of `TorchLean.CUDA.LibTorch.Buffer`. Lean owns an external object that points
// at a `torchlean_cuda_buffer`; `size` counts elements, not bytes. Callers validate shape
// metadata before touching storage. This is a trusted boundary: Lean proves shape contracts around
// these calls but cannot see tensor lifetimes or CUDA behavior.
struct __attribute__((visibility("hidden"))) torchlean_cuda_buffer {
  size_t size;
  // Empty once the buffer has been released.
  std::optional<torchlean::Tensor> tensor;
  // Encoded arithmetic stores complete scalar words in a byte tensor.
  lean_object* format = nullptr;
  size_t width = 1;
};

extern "C" {
torchlean_cuda_buffer* torchlean_cuda_buffer_unbox(b_lean_obj_arg obj);
lean_obj_res torchlean_cuda_buffer_box(torchlean_cuda_buffer* b);
void torchlean_cuda_buffer_drop_unboxed(torchlean_cuda_buffer* b);
LEAN_EXPORT uint32_t torchlean_cuda_buffer_release(b_lean_obj_arg object);

// Diagnostic counters, read as `IO UInt64` actions. The allocator counters track payloads owned
// by TorchLean buffers, and the wrapper counters track Lean external objects from boxing through
// finalization.
LEAN_EXPORT lean_obj_res torchlean_cuda_allocator_live_bytes(void);
LEAN_EXPORT lean_obj_res torchlean_cuda_allocator_peak_bytes(void);
LEAN_EXPORT lean_obj_res torchlean_cuda_allocator_alloc_count(void);
LEAN_EXPORT lean_obj_res torchlean_cuda_allocator_free_count(void);
LEAN_EXPORT lean_obj_res torchlean_cuda_wrapper_live_count(void);
LEAN_EXPORT lean_obj_res torchlean_cuda_wrapper_peak_count(void);
LEAN_EXPORT lean_obj_res torchlean_cuda_wrapper_alloc_count(void);
LEAN_EXPORT lean_obj_res torchlean_cuda_wrapper_finalize_count(void);
LEAN_EXPORT lean_obj_res torchlean_cuda_allocator_device_free_bytes(void);
LEAN_EXPORT lean_obj_res torchlean_cuda_allocator_device_total_bytes(void);

// For native tests built against ATen. `of_handle` boxes a tensor as a buffer, taking ownership
// of the handle, with the checks of results; the caller checks that the tensor does not require
// gradients, which the Stable ABI cannot read. `handle` borrows the tensor of a live buffer.
LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_of_handle(AtenTensorHandle tensor);
LEAN_EXPORT AtenTensorHandle torchlean_cuda_buffer_handle(b_lean_obj_arg object);
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

namespace torchlean __attribute__((visibility("hidden"))) {

// Exported for the native tests, whose copies of `invoke` call them.
__attribute__((visibility("default"))) void initialize();
__attribute__((visibility("default"))) std::int32_t device_index();

inline Device device() { return Device(kCUDA, device_index()); }

// The Lean tape owns differentiation. Native calls must not record a second graph.
template <typename F>
auto invoke(F&& body) -> decltype(body()) {
  try {
    initialize();
    const no_grad grad_mode;
    const cuda_guard guard(device_index());
    return std::forward<F>(body)();
  } catch (const std::exception& error) {
    lean_internal_panic(error.what());
    std::abort();
  }
}

// The IO form of invoke: an exception becomes a Lean IO error, and an allocation failure the error
// `resourceExhausted`.
template <typename F>
lean_obj_res io(F&& body) {
  try {
    initialize();
    const no_grad grad_mode;
    const cuda_guard guard(device_index());
    return lean_io_result_mk_ok(std::forward<F>(body)());
  } catch (const out_of_memory& error) {
    return lean_io_result_mk_error(
        lean_mk_io_error_resource_exhausted(0, lean_mk_string(error.what())));
  } catch (const std::bad_alloc& error) {
    return lean_io_result_mk_error(
        lean_mk_io_error_resource_exhausted(0, lean_mk_string(error.what())));
  } catch (const std::exception& error) {
    return lean_io_result_mk_error(
        lean_mk_io_error_other_error(0, lean_mk_string(error.what())));
  }
}

const Tensor& tensor(b_lean_obj_arg object);
tensor_options options();
torchlean_cuda_buffer* owned(Tensor value);
lean_obj_res box(Tensor value);

/// `x.to(kCPU, dtype).contiguous()`: a contiguous host copy of `x`, in `dtype` if one is given.
inline Tensor host_copy(const Tensor& x, std::optional<ScalarType> dtype = std::nullopt) {
  return aten<"contiguous">(aten<"to.dtype_layout">(x, tensor_options{dtype, Device(kCPU)}));
}

/// `host.to(options().dtype(dtype), false, true)`: a copy of `host` on the selected device in
/// `dtype`. The copy blocks, so the host storage may change once it returns.
inline Tensor device_copy(const Tensor& host, ScalarType dtype) {
  return aten<"to.dtype_layout">(host, options().dtype(dtype), false, true);
}

// One launch of a generated kernel: its number of threads and its arguments, each a 64-bit
// device address or size.
struct launch {
  uint64_t count;
  std::vector<uint64_t> arguments;
};

// A generated kernel compiled with the module cache and floating-point policy of custom
// operations. Its module stays loaded while the handle is held, also after the cache has
// evicted it.
using rows_kernel = std::shared_ptr<const void>;

// Compile generated source into a `rows_kernel`.
rows_kernel compile_rows(const std::string& source);

// Run the launches of a compiled kernel in order on the current stream, 256 threads to a
// block, skipping launches without threads. Returns when the launches have finished.
void launch_rows(const rows_kernel& kernel, std::vector<launch>& launches);

// Compile generated source with `compile_rows`, then run its launches as `launch_rows` of a
// compiled kernel does.
void launch_rows(const std::string& source, std::vector<launch>& launches);

// The output of the generated kernel `source`: `count` scalars of `stride` entries of `scalar`
// each, one thread per scalar, reading `inputs`, contiguous tensors of `scalar` on the selected
// device. A read the kernel reports out of range throws.
Tensor run_kernel(b_lean_obj_arg source, const std::vector<Tensor>& inputs, ScalarType scalar,
                  uint64_t count, uint64_t stride = 1);

// The number of visible CUDA devices, 0 when the driver cannot be initialized.
int device_count();

// The selected device's free memory, or its total memory if `total`, as the driver reports it.
uint64_t memory_info(bool total);

// Returns when the selected device has finished all its work.
void synchronize();

inline Tensor shaped(b_lean_obj_arg object, ints shape) {
  return aten<"reshape">(tensor(object), shape);
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

// The double `value` rounded to nearest in the buffer dtype `dtype`, binary32 or binary64.
inline double rounded(double value, ScalarType dtype) {
  return dtype == kFloat ? static_cast<double>(static_cast<float>(value)) : value;
}

// The pure ABI's dimension helpers panic. IO operations must reject malformed metadata by
// throwing instead, so `io` can return an ordinary Lean error without terminating the process.
inline std::vector<int64_t> io_dimensions(b_lean_obj_arg object) {
  TORCHLEAN_CHECK(lean_is_array(object), "LibTorch: expected dimension array");
  std::vector<int64_t> result;
  result.reserve(lean_array_size(object));
  for (size_t i = 0; i < lean_array_size(object); ++i) {
    const auto dim = lean_array_uget(object, i);
    TORCHLEAN_CHECK(lean_is_scalar(dim) && lean_unbox(dim) <= UINT32_MAX,
                    "LibTorch: dimension exceeds UInt32 ABI");
    result.push_back(static_cast<int64_t>(lean_unbox(dim)));
  }
  return result;
}

inline int64_t io_volume(const std::vector<int64_t>& shape) {
  if (std::find(shape.begin(), shape.end(), 0) != shape.end()) return 0;
  int64_t result = 1;
  for (const auto dim : shape) {
    TORCHLEAN_CHECK(dim > 0 && result <= INT64_MAX / dim, "LibTorch: dimension product overflow");
    result *= dim;
  }
  return result;
}

inline const Tensor& io_tensor(b_lean_obj_arg object, const std::vector<int64_t>& shape) {
  const auto& value = tensor(object);
  TORCHLEAN_CHECK(value.numel() == io_volume(shape), "LibTorch: buffer size mismatch");
  return value;
}

// Lean `A × B` is a constructor with two object fields; longer tuples nest to the right.
inline lean_obj_res cons(lean_obj_res first, lean_obj_res second) {
  lean_object* out = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(out, 0, first);
  lean_ctor_set(out, 1, second);
  return out;
}

inline lean_obj_res pair(Tensor first, Tensor second) {
  const auto drop = torchlean_cuda_buffer_drop_unboxed;
  std::unique_ptr<torchlean_cuda_buffer, decltype(drop)> a(owned(std::move(first)), drop);
  std::unique_ptr<torchlean_cuda_buffer, decltype(drop)> b(owned(std::move(second)), drop);
  return cons(torchlean_cuda_buffer_box(a.release()), torchlean_cuda_buffer_box(b.release()));
}

inline lean_obj_res triple(Tensor first, Tensor second, Tensor third) {
  const auto drop = torchlean_cuda_buffer_drop_unboxed;
  std::unique_ptr<torchlean_cuda_buffer, decltype(drop)> a(owned(std::move(first)), drop);
  std::unique_ptr<torchlean_cuda_buffer, decltype(drop)> b(owned(std::move(second)), drop);
  std::unique_ptr<torchlean_cuda_buffer, decltype(drop)> c(owned(std::move(third)), drop);
  return cons(torchlean_cuda_buffer_box(a.release()),
              cons(torchlean_cuda_buffer_box(b.release()), torchlean_cuda_buffer_box(c.release())));
}

inline lean_obj_res quadruple(Tensor first, Tensor second, Tensor third, Tensor fourth) {
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
