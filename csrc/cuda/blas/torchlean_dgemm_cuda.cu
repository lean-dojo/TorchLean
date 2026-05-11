#include <lean/lean.h>

#include "torchlean_cublas_common.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>

static inline size_t checked_mul_size(size_t a, size_t b, const char* msg) {
  if (a != 0 && b > SIZE_MAX / a) {
    lean_internal_panic(msg);
  }
  return a * b;
}

template <typename T>
static inline void torchlean_cuda_free_checked(T** ptr, const char* msg) {
  if (*ptr) {
    checkCuda(cudaFree(*ptr), msg);
    *ptr = nullptr;
  }
}

// Trusted native implementation for Lean `FloatArray` DGEMM.
//
// This file is separate from the float32 `Cuda.Buffer` kernels because Lean `FloatArray` stores
// binary64 values and the implementation uses cuBLAS DGEMM. It is still part of the CUDA trusted
// boundary: Lean checks the sizes it passes in, while this native code owns allocation, host/device
// copies, and cuBLAS launch semantics.
//
// A: m x n (row-major in Lean)
// B: n x p (row-major in Lean)
// C: m x p (row-major returned to Lean)
//
// cuBLAS expects column-major by default. We compute C = A*B (row-major) by using the identity:
// (A*B)^T = B^T * A^T, and treating row-major as transposed column-major.
extern "C" LEAN_EXPORT lean_obj_res torchlean_dgemm_cuda(b_lean_obj_arg AObj, b_lean_obj_arg BObj,
                                                        uint32_t m, uint32_t n, uint32_t p) {
  // Copy inputs from Lean FloatArray.
  lean_object* A = (lean_object*)AObj;
  lean_object* B = (lean_object*)BObj;

  size_t aSz = checked_mul_size((size_t)m, (size_t)n, "torchlean_dgemm_cuda: A size overflow");
  size_t bSz = checked_mul_size((size_t)n, (size_t)p, "torchlean_dgemm_cuda: B size overflow");
  size_t cSz = checked_mul_size((size_t)m, (size_t)p, "torchlean_dgemm_cuda: C size overflow");

  if (lean_sarray_size(A) != aSz) {
    lean_internal_panic("torchlean_dgemm_cuda: A.size mismatch");
  }
  if (lean_sarray_size(B) != bSz) {
    lean_internal_panic("torchlean_dgemm_cuda: B.size mismatch");
  }
  if (m > (uint32_t)INT_MAX || n > (uint32_t)INT_MAX || p > (uint32_t)INT_MAX) {
    lean_internal_panic("torchlean_dgemm_cuda: dimensions exceed cuBLAS int range");
  }

  const double* hA = lean_float_array_cptr(A);
  const double* hB = lean_float_array_cptr(B);

  if (cSz == 0) {
    return lean_mk_empty_float_array(lean_box(0));
  }

  if (n == 0) {
    // Matrix product over an empty inner dimension yields a zero matrix.
    lean_object* out = lean_mk_empty_float_array(lean_box(cSz));
    double* hC = lean_float_array_cptr(out);
    for (size_t i = 0; i < cSz; ++i) {
      hC[i] = 0.0;
    }
    lean_sarray_set_size(out, cSz);
    return out;
  }

  double *dA = nullptr, *dB = nullptr, *dC = nullptr;
  cudaError_t err = cudaMalloc((void**)&dA, aSz * sizeof(double));
  if (err != cudaSuccess) {
    checkCuda(err, "cudaMalloc dA failed");
  }
  err = cudaMalloc((void**)&dB, bSz * sizeof(double));
  if (err != cudaSuccess) {
    torchlean_cuda_free_checked(&dA, "cudaFree dA after dB alloc failure failed");
    checkCuda(err, "cudaMalloc dB failed");
  }
  err = cudaMalloc((void**)&dC, cSz * sizeof(double));
  if (err != cudaSuccess) {
    torchlean_cuda_free_checked(&dB, "cudaFree dB after dC alloc failure failed");
    torchlean_cuda_free_checked(&dA, "cudaFree dA after dC alloc failure failed");
    checkCuda(err, "cudaMalloc dC failed");
  }

  err = cudaMemcpy(dA, hA, aSz * sizeof(double), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    torchlean_cuda_free_checked(&dC, "cudaFree dC after A copy failure failed");
    torchlean_cuda_free_checked(&dB, "cudaFree dB after A copy failure failed");
    torchlean_cuda_free_checked(&dA, "cudaFree dA after A copy failure failed");
    checkCuda(err, "cudaMemcpy A failed");
  }
  err = cudaMemcpy(dB, hB, bSz * sizeof(double), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    torchlean_cuda_free_checked(&dC, "cudaFree dC after B copy failure failed");
    torchlean_cuda_free_checked(&dB, "cudaFree dB after B copy failure failed");
    torchlean_cuda_free_checked(&dA, "cudaFree dA after B copy failure failed");
    checkCuda(err, "cudaMemcpy B failed");
  }

  cublasHandle_t handle = getCublasHandle();

  const double alpha = 1.0;
  const double beta  = 0.0;

  // Compute (A*B)^T as a column-major GEMM:
  // C^T (p x m) = B^T (p x n) * A^T (n x m)
  // In column-major, B is (n x p) row-major => treat as (p x n) column-major without transpose.
  // Likewise A is (m x n) row-major => treat as (n x m) column-major without transpose.
  cublasStatus_t stat =
      cublasDgemm(handle,
                  CUBLAS_OP_N, CUBLAS_OP_N,
                  (int)p, (int)m, (int)n,
                  &alpha,
                  dB, (int)p,
                  dA, (int)n,
                  &beta,
                  dC, (int)p);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    torchlean_cuda_free_checked(&dC, "cudaFree dC after cublasDgemm failure failed");
    torchlean_cuda_free_checked(&dB, "cudaFree dB after cublasDgemm failure failed");
    torchlean_cuda_free_checked(&dA, "cudaFree dA after cublasDgemm failure failed");
    checkCublas(stat, "cublasDgemm failed");
  }

  // Copy back. dC holds C^T in column-major, which is exactly C in row-major memory order.
  lean_object* out = lean_mk_empty_float_array(lean_box(cSz));
  double* hC = lean_float_array_cptr(out);
  err = cudaMemcpy(hC, dC, cSz * sizeof(double), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    torchlean_cuda_free_checked(&dC, "cudaFree dC after C copy failure failed");
    torchlean_cuda_free_checked(&dB, "cudaFree dB after C copy failure failed");
    torchlean_cuda_free_checked(&dA, "cudaFree dA after C copy failure failed");
    checkCuda(err, "cudaMemcpy C failed");
  }
  lean_sarray_set_size(out, cSz);

  torchlean_cuda_free_checked(&dA, "cudaFree dA failed");
  torchlean_cuda_free_checked(&dB, "cudaFree dB failed");
  torchlean_cuda_free_checked(&dC, "cudaFree dC failed");

  return out;
}
