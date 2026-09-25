#include "torchlean_libtorch.h"

#include <cstring>
#include <limits>

// FloatArray stores binary64; this adapter keeps its dtype when using ATen matmul.
extern "C" LEAN_EXPORT lean_obj_res torchlean_dgemm_cuda(
    b_lean_obj_arg a_object, b_lean_obj_arg b_object, uint32_t m, uint32_t n, uint32_t p) {
  return torchlean::invoke([&]() {
    const size_t a_size = checked_mul_size(m, n, "dgemm: A size overflow");
    const size_t b_size = checked_mul_size(n, p, "dgemm: B size overflow");
    const size_t c_size = checked_mul_size(m, p, "dgemm: output size overflow");
    TORCH_CHECK(lean_sarray_size(a_object) == a_size, "dgemm: A size mismatch");
    TORCH_CHECK(lean_sarray_size(b_object) == b_size, "dgemm: B size mismatch");
    TORCH_CHECK(c_size <= INT64_MAX, "dgemm: output exceeds ATen element count");
    const auto cpu = at::TensorOptions().dtype(at::kDouble).device(at::kCPU);
    at::Tensor result;
    if (c_size == 0 || n == 0) {
      result = at::zeros({m, p}, cpu);
    } else {
      const auto gpu = cpu.device(torchlean::device());
      const auto a = at::from_blob(lean_float_array_cptr(a_object), {m, n}, cpu).to(gpu);
      const auto b = at::from_blob(lean_float_array_cptr(b_object), {n, p}, cpu).to(gpu);
      result = at::matmul(a, b).to(cpu).contiguous();
    }
    auto* out = lean_mk_empty_float_array(lean_box(c_size));
    lean_sarray_set_size(out, c_size);
    if (c_size != 0)
      std::memcpy(lean_float_array_cptr(out), result.const_data_ptr<double>(),
                  checked_bytes_size(c_size, sizeof(double), "dgemm: output byte size overflow"));
    return out;
  });
}
