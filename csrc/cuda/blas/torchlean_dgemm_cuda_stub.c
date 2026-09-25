#include <lean/lean.h>

#include "torchlean_size_common.h"

#include <stddef.h>
#include <stdint.h>

// CPU fallback for `torchlean_dgemm_cuda`.
//
// This file exports the same symbol as the LibTorch implementation so ordinary `lake build`
// works without LibTorch or a CUDA toolkit. Keep size checks and row-major semantics aligned with
// `csrc/libtorch/blas.cpp`; accumulation order can differ between ATen and this CPU loop.
LEAN_EXPORT lean_obj_res torchlean_dgemm_cuda(b_lean_obj_arg AObj, b_lean_obj_arg BObj,
                                             uint32_t m, uint32_t n, uint32_t p) {
  size_t M = (size_t)m, N = (size_t)n, P = (size_t)p;
  size_t aSz = checked_mul_size(M, N, "torchlean_dgemm_cuda_stub: A size overflow");
  size_t bSz = checked_mul_size(N, P, "torchlean_dgemm_cuda_stub: B size overflow");
  size_t cSz = checked_mul_size(M, P, "torchlean_dgemm_cuda_stub: C size overflow");

  lean_object* A = (lean_object*)AObj;
  lean_object* B = (lean_object*)BObj;

  if (lean_sarray_size(A) != aSz) {
    lean_internal_panic("torchlean_dgemm_cuda_stub: A.size mismatch");
  }
  if (lean_sarray_size(B) != bSz) {
    lean_internal_panic("torchlean_dgemm_cuda_stub: B.size mismatch");
  }

  const double* a = lean_float_array_cptr(A);
  const double* b = lean_float_array_cptr(B);

  lean_object* out = lean_mk_empty_float_array(lean_box(cSz));
  double* c = lean_float_array_cptr(out);

  for (size_t i = 0; i < M; ++i) {
    for (size_t k = 0; k < P; ++k) {
      double acc = 0.0;
      for (size_t j = 0; j < N; ++j) {
        acc += a[i * N + j] * b[j * P + k];
      }
      c[i * P + k] = acc;
    }
  }
  lean_sarray_set_size(out, cSz);
  return out;
}
