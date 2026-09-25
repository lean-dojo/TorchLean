#pragma once

#ifndef TORCHLEAN_LIBTORCH
#error "Compile the LibTorch backend with TORCHLEAN_LIBTORCH defined."
#endif

#include "../cuda/common/torchlean_cuda_buffer.h"
#include "../cuda/common/torchlean_size_common.h"

#include <ATen/core/grad_mode.h>
#include <c10/core/DeviceGuard.h>
#include <cstdlib>
#include <exception>
#include <utility>
#include <vector>

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

inline lean_obj_res pair(at::Tensor first, at::Tensor second) {
  const auto drop = torchlean_cuda_buffer_drop_unboxed;
  std::unique_ptr<torchlean_cuda_buffer, decltype(drop)> a(owned(std::move(first)), drop);
  std::unique_ptr<torchlean_cuda_buffer, decltype(drop)> b(owned(std::move(second)), drop);
  return torchlean_cuda_box_buffer_pair(a.release(), b.release());
}

inline lean_obj_res triple(at::Tensor first, at::Tensor second, at::Tensor third) {
  const auto drop = torchlean_cuda_buffer_drop_unboxed;
  std::unique_ptr<torchlean_cuda_buffer, decltype(drop)> a(owned(std::move(first)), drop);
  std::unique_ptr<torchlean_cuda_buffer, decltype(drop)> b(owned(std::move(second)), drop);
  std::unique_ptr<torchlean_cuda_buffer, decltype(drop)> c(owned(std::move(third)), drop);
  return torchlean_cuda_box_three_buffers(a.release(), b.release(), c.release());
}

inline lean_obj_res quadruple(
    at::Tensor first, at::Tensor second, at::Tensor third, at::Tensor fourth) {
  const auto drop = torchlean_cuda_buffer_drop_unboxed;
  std::unique_ptr<torchlean_cuda_buffer, decltype(drop)> a(owned(std::move(first)), drop);
  std::unique_ptr<torchlean_cuda_buffer, decltype(drop)> b(owned(std::move(second)), drop);
  std::unique_ptr<torchlean_cuda_buffer, decltype(drop)> c(owned(std::move(third)), drop);
  std::unique_ptr<torchlean_cuda_buffer, decltype(drop)> d(owned(std::move(fourth)), drop);
  return torchlean_cuda_box_four_buffers(a.release(), b.release(), c.release(), d.release());
}

}  // namespace torchlean
