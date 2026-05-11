#include <lean/lean.h>
#include <lean/mimalloc.h>

#include "torchlean_cuda_buffer.h"
#include "torchlean_cuda_common.h"
#include "torchlean_cuda_deterministic_reductions_env.h"
#include "torchlean_cuda_rng_common.h"

#include <cuda_runtime.h>

#include <assert.h>
#include <atomic>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

// CUDA implementation of TorchLean's float32 buffer runtime.
//
// This file owns the lowest-level executable tensor backend used by Lean's `Cuda.Buffer` wrapper:
// allocation/finalization, host/device copies, scalar elementwise kernels, whole-buffer reductions,
// seeded random buffers, and manual release hooks used by long training loops.
//
// Important contract:
// - every exported function receives already-shaped Lean metadata and must still validate sizes;
// - returned buffers are boxed as Lean external objects with a CUDA finalizer;
// - reductions may use a fast non-deterministic path unless deterministic reductions are enabled;
// - the CPU fallback in `torchlean_cuda_tensor_stub.c` must expose the same exported symbols.

// Global (process-wide) toggle used to trade performance for bit-stable reproducibility in
// atomic-accumulation kernels.
static std::atomic<uint32_t> g_torchlean_deterministic_reductions{0u};
// 0 = uninitialized, 1 = initialized, 2 = another thread is initializing.
static std::atomic<uint32_t> g_torchlean_deterministic_reductions_inited{0u};

static void torchlean_cuda_free_best_effort(void* ptr, const char* what) {
  if (!ptr) {
    return;
  }
  cudaError_t err = cudaFree(ptr);
  if (err != cudaSuccess) {
    fprintf(stderr, "TorchLean CUDA warning: %s: %s\n", what, cudaGetErrorString(err));
  }
}

static void torchlean_cuda_buffer_finalize(void* ptr) {
  torchlean_cuda_buffer* b = (torchlean_cuda_buffer*)ptr;
  if (!b) {
    return;
  }
  if (b->data) {
    // cudaFree is synchronous w.r.t. work using this allocation, so freeing is safe even if GC
    // runs before a kernel finishes.
    torchlean_cuda_free_best_effort(b->data, "cudaFree buffer finalizer failed");
  }
  free(b);
}

// `torchlean_cuda_buffer` holds no Lean references.
static void torchlean_cuda_buffer_foreach(void* _ptr, b_lean_obj_arg _fn) {
  (void)_ptr;
  (void)_fn;
}

static lean_external_class* torchlean_cuda_buffer_class = NULL;

static lean_external_class* torchlean_cuda_buffer_get_class(void) {
  if (!torchlean_cuda_buffer_class) {
    torchlean_cuda_buffer_class =
        lean_register_external_class(torchlean_cuda_buffer_finalize, torchlean_cuda_buffer_foreach);
  }
  return torchlean_cuda_buffer_class;
}

extern "C" torchlean_cuda_buffer* torchlean_cuda_buffer_unbox(b_lean_obj_arg obj) {
  lean_object* o = (lean_object*)obj;
  if (!lean_is_external(o)) {
    lean_internal_panic("torchlean_cuda_buffer: expected external object");
  }
  return (torchlean_cuda_buffer*)lean_get_external_data(o);
}

extern "C" lean_obj_res torchlean_cuda_buffer_box(torchlean_cuda_buffer* b) {
  return lean_alloc_external(torchlean_cuda_buffer_get_class(), b);
}

extern "C" torchlean_cuda_buffer* torchlean_cuda_buffer_alloc(size_t n) {
  torchlean_cuda_buffer* b = (torchlean_cuda_buffer*)malloc(sizeof(torchlean_cuda_buffer));
  if (!b) {
    lean_internal_panic_out_of_memory();
  }
  b->size = n;
  b->data = NULL;
  if (n > 0) {
    checkCuda(cudaMalloc((void**)&b->data, n * sizeof(float)), "cudaMalloc buffer failed");
  }
  return b;
}

// --- Kernels -----------------------------------------------------------------

static constexpr int kBlockSize = 256;
static_assert(kBlockSize > 0 && (kBlockSize & (kBlockSize - 1)) == 0,
              "kBlockSize must remain a power of two for shared-memory reductions");

#define TORCHLEAN_GRID_STRIDE_LOOP(I, N)                                                    \
  for (size_t I = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;           \
       I < (N);                                                                             \
       I += (size_t)gridDim.x * (size_t)blockDim.x)

__global__ void torchlean_fill_f32(float* out, size_t n, float v) {
  TORCHLEAN_GRID_STRIDE_LOOP(i, n) {
    out[i] = v;
  }
}

__global__ void torchlean_abs_f32(const float* in, float* out, size_t n) {
  TORCHLEAN_GRID_STRIDE_LOOP(i, n) {
    out[i] = fabsf(in[i]);
  }
}

__global__ void torchlean_sqrt_f32(const float* in, float* out, size_t n) {
  TORCHLEAN_GRID_STRIDE_LOOP(i, n) {
    out[i] = sqrtf(in[i]);
  }
}

__global__ void torchlean_exp_f32(const float* in, float* out, size_t n) {
  TORCHLEAN_GRID_STRIDE_LOOP(i, n) {
    out[i] = expf(in[i]);
  }
}

__global__ void torchlean_log_f32(const float* in, float* out, size_t n) {
  TORCHLEAN_GRID_STRIDE_LOOP(i, n) {
    out[i] = logf(in[i]);
  }
}

__global__ void torchlean_inv_f32(const float* in, float* out, size_t n) {
  TORCHLEAN_GRID_STRIDE_LOOP(i, n) {
    out[i] = 1.0f / in[i];
  }
}

__global__ void torchlean_clamp_f32(const float* in, float* out, size_t n, float lo, float hi) {
  TORCHLEAN_GRID_STRIDE_LOOP(i, n) {
    float x = in[i];
    x = fmaxf(x, lo);
    x = fminf(x, hi);
    out[i] = x;
  }
}

__global__ void torchlean_max_f32(const float* a, const float* b, float* out, size_t n) {
  TORCHLEAN_GRID_STRIDE_LOOP(i, n) {
    out[i] = fmaxf(a[i], b[i]);
  }
}

__global__ void torchlean_min_f32(const float* a, const float* b, float* out, size_t n) {
  TORCHLEAN_GRID_STRIDE_LOOP(i, n) {
    out[i] = fminf(a[i], b[i]);
  }
}

__global__ void torchlean_div_f32(const float* a, const float* b, float* out, size_t n) {
  TORCHLEAN_GRID_STRIDE_LOOP(i, n) {
    out[i] = a[i] / b[i];
  }
}

__global__ void torchlean_add_f32(const float* a, const float* b, float* out, size_t n) {
  TORCHLEAN_GRID_STRIDE_LOOP(i, n) {
    out[i] = a[i] + b[i];
  }
}

__global__ void torchlean_sub_f32(const float* a, const float* b, float* out, size_t n) {
  TORCHLEAN_GRID_STRIDE_LOOP(i, n) {
    out[i] = a[i] - b[i];
  }
}

__global__ void torchlean_mul_f32(const float* a, const float* b, float* out, size_t n) {
  TORCHLEAN_GRID_STRIDE_LOOP(i, n) {
    out[i] = a[i] * b[i];
  }
}

__global__ void torchlean_scale_f32(const float* in, float* out, size_t n, float c) {
  TORCHLEAN_GRID_STRIDE_LOOP(i, n) {
    out[i] = in[i] * c;
  }
}

__global__ void torchlean_axpy_f32(const float* a, const float* b, float* out, size_t n,
                                  float c) {
  TORCHLEAN_GRID_STRIDE_LOOP(i, n) {
    out[i] = a[i] + c * b[i];
  }
}

__global__ void torchlean_abs_bwd_f32(const float* x, const float* dLdy, float* dLdx, size_t n) {
  TORCHLEAN_GRID_STRIDE_LOOP(i, n) {
    float v = x[i];
    float s = (v > 0.0f) ? 1.0f : ((v < 0.0f) ? -1.0f : 0.0f);
    dLdx[i] = s * dLdy[i];
  }
}

__global__ void torchlean_sqrt_bwd_f32(const float* x, const float* dLdy, float* dLdx, size_t n) {
  TORCHLEAN_GRID_STRIDE_LOOP(i, n) {
    float v = x[i];
    if (v > 0.0f) {
      dLdx[i] = dLdy[i] * (1.0f / (2.0f * sqrtf(v)));
    } else {
      dLdx[i] = 0.0f;
    }
  }
}

__global__ void torchlean_clamp_bwd_f32(const float* x, const float* dLdy, float* dLdx, size_t n,
                                       float lo, float hi) {
  TORCHLEAN_GRID_STRIDE_LOOP(i, n) {
    float v = x[i];
    dLdx[i] = (v > lo && v < hi) ? dLdy[i] : 0.0f;
  }
}

__global__ void torchlean_relu_f32(const float* x, float* y, size_t n) {
  TORCHLEAN_GRID_STRIDE_LOOP(i, n) {
    float v = x[i];
    y[i] = (v > 0.0f) ? v : 0.0f;
  }
}

__global__ void torchlean_relu_bwd_f32(const float* x, const float* dLdy, float* dLdx, size_t n) {
  TORCHLEAN_GRID_STRIDE_LOOP(i, n) {
    float v = x[i];
    dLdx[i] = (v > 0.0f) ? dLdy[i] : 0.0f;
  }
}

__global__ void torchlean_max_bwd_f32(const float* a, const float* b, const float* dLdy, float* dA,
                                     float* dB, size_t n) {
  TORCHLEAN_GRID_STRIDE_LOOP(i, n) {
    float av = a[i];
    float bv = b[i];
    float g = dLdy[i];
    if (av > bv) {
      dA[i] = g;
      dB[i] = 0.0f;
    } else if (bv > av) {
      dA[i] = 0.0f;
      dB[i] = g;
    } else {
      dA[i] = 0.5f * g;
      dB[i] = 0.5f * g;
    }
  }
}

__global__ void torchlean_min_bwd_f32(const float* a, const float* b, const float* dLdy, float* dA,
                                     float* dB, size_t n) {
  TORCHLEAN_GRID_STRIDE_LOOP(i, n) {
    float av = a[i];
    float bv = b[i];
    float g = dLdy[i];
    if (bv > av) {
      // a < b
      dA[i] = g;
      dB[i] = 0.0f;
    } else if (av > bv) {
      // b < a
      dA[i] = 0.0f;
      dB[i] = g;
    } else {
      dA[i] = 0.5f * g;
      dB[i] = 0.5f * g;
    }
  }
}

__global__ void torchlean_reduce_sum_f32(const float* in, float* out, size_t n) {
  __shared__ float sdata[kBlockSize];
  const size_t tid = (size_t)threadIdx.x;
  const size_t base = (size_t)blockIdx.x * (size_t)blockDim.x + tid;
  const size_t stride = (size_t)gridDim.x * (size_t)blockDim.x;

  float sum = 0.0f;
  for (size_t i = base; i < n; i += stride) {
    sum += in[i];
  }
  sdata[tid] = sum;
  __syncthreads();

  // Tree reduction in shared memory.
  for (unsigned int s = (unsigned int)blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < (size_t)s) {
      sdata[tid] += sdata[tid + (size_t)s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(out, sdata[0]);
  }
}

// Deterministic reduction: each block writes a partial sum to `partial[blockIdx.x]`.
// No atomics; final scalar is computed via iterative reduction over the partials.
__global__ void torchlean_reduce_sum_partials_f32(const float* in, float* partial, size_t n) {
  __shared__ float sdata[kBlockSize];
  const size_t tid = (size_t)threadIdx.x;

  const size_t base = (size_t)blockIdx.x * (size_t)blockDim.x + tid;
  const size_t stride = (size_t)gridDim.x * (size_t)blockDim.x;

  float sum = 0.0f;
  for (size_t i = base; i < n; i += stride) {
    sum += in[i];
  }
  sdata[tid] = sum;
  __syncthreads();

  for (unsigned int s = (unsigned int)blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < (size_t)s) {
      sdata[tid] += sdata[tid + (size_t)s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial[(size_t)blockIdx.x] = sdata[0];
  }
}

__global__ void torchlean_scale1_f32(float* out, float scale) { out[0] *= scale; }

__global__ void torchlean_rand_uniform_f32(float* out, size_t n, uint64_t key) {
  TORCHLEAN_GRID_STRIDE_LOOP(i, n) {
    // Match the pure Lean RNG helper: reduce SplitMix64 output modulo 2^32 via the low 32 bits.
    uint64_t z = torchlean_splitmix64(key + (uint64_t)i);
    uint32_t u = (uint32_t)z;
    const double denom = 4294967296.0;
    out[i] = (float)(((double)u) / denom);
  }
}

__global__ void torchlean_bernoulli_mask_f32(float* out, size_t n, float keepProb, uint64_t key) {
  TORCHLEAN_GRID_STRIDE_LOOP(i, n) {
    uint64_t z = torchlean_splitmix64(key + (uint64_t)i);
    uint32_t u = (uint32_t)z;
    const double denom = 4294967296.0;
    float u01 = (float)(((double)u) / denom);
    out[i] = (keepProb > u01) ? 1.0f : 0.0f;
  }
}

static inline dim3 torchlean_blocks_for(size_t n) {
  size_t blocks = (n + (size_t)kBlockSize - 1) / (size_t)kBlockSize;
  if (blocks == 0) {
    blocks = 1;
  }
  // CUDA grid dimension is `uint32_t`-like; clamp to a safe max. Kernels use grid-stride loops,
  // so correctness is preserved even when the theoretical grid size would exceed this cap.
  if (blocks > 2147483647ULL) {
    blocks = 2147483647ULL;
  }
  return dim3((unsigned int)blocks);
}

static inline unsigned int torchlean_det_reduce_blocks_for(size_t n) {
  // For deterministic reductions we intentionally cap the number of blocks.
  // This keeps scratch allocations bounded while still covering all elements via grid-stride loops.
  size_t blocks = (n + (size_t)kBlockSize - 1) / (size_t)kBlockSize;
  if (blocks == 0) {
    blocks = 1;
  }
  if (blocks > 65535ULL) {
    blocks = 65535ULL;
  }
  return (unsigned int)blocks;
}

static void torchlean_reduce_sum_deterministic(const float* in, size_t n, float* outScalar) {
  if (n == 0) {
    float zero = 0.0f;
    checkCuda(cudaMemcpy(outScalar, &zero, sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy deterministic reduceSum init failed");
    return;
  }

  // Stage 1: partials over the original input.
  unsigned int blocks = torchlean_det_reduce_blocks_for(n);
  float* partial = nullptr;
  checkCuda(cudaMalloc((void**)&partial, (size_t)blocks * sizeof(float)),
            "cudaMalloc deterministic reduceSum partials failed");
  torchlean_reduce_sum_partials_f32<<<dim3(blocks), dim3(kBlockSize)>>>(in, partial, n);
  checkCuda(cudaGetLastError(), "cuda deterministic reduceSum partial kernel launch failed");

  // Iteratively reduce partials until we have a single scalar.
  size_t curSize = (size_t)blocks;
  while (curSize > 1) {
    unsigned int nextBlocks = torchlean_det_reduce_blocks_for(curSize);
    float* next = nullptr;
    checkCuda(cudaMalloc((void**)&next, (size_t)nextBlocks * sizeof(float)),
              "cudaMalloc deterministic reduceSum next partials failed");
    torchlean_reduce_sum_partials_f32<<<dim3(nextBlocks), dim3(kBlockSize)>>>(partial, next, curSize);
    checkCuda(cudaGetLastError(), "cuda deterministic reduceSum next partial kernel launch failed");
    checkCuda(cudaFree(partial), "cudaFree deterministic reduceSum partials failed");
    partial = next;
    curSize = (size_t)nextBlocks;
  }

  checkCuda(cudaMemcpy(outScalar, partial, sizeof(float), cudaMemcpyDeviceToDevice),
            "cudaMemcpy deterministic reduceSum final copy failed");
  checkCuda(cudaFree(partial), "cudaFree deterministic reduceSum final partial failed");
}

// --- Exports -----------------------------------------------------------------

extern "C" LEAN_EXPORT void torchlean_cuda_set_deterministic_reductions(uint32_t on) {
  // Treat any non-zero as "on".
  g_torchlean_deterministic_reductions.store(on ? 1u : 0u, std::memory_order_relaxed);
  g_torchlean_deterministic_reductions_inited.store(1u, std::memory_order_release);
}

extern "C" LEAN_EXPORT uint32_t torchlean_cuda_get_deterministic_reductions() {
  uint32_t state = g_torchlean_deterministic_reductions_inited.load(std::memory_order_acquire);
  if (state == 0u) {
    uint32_t expected = 0u;
    if (g_torchlean_deterministic_reductions_inited.compare_exchange_strong(
          expected, 2u, std::memory_order_acq_rel, std::memory_order_acquire)) {
      g_torchlean_deterministic_reductions.store(torchlean_read_deterministic_reductions_env(),
                                                std::memory_order_relaxed);
      g_torchlean_deterministic_reductions_inited.store(1u, std::memory_order_release);
    } else {
      while (g_torchlean_deterministic_reductions_inited.load(std::memory_order_acquire) == 2u) {
      }
    }
  } else if (state == 2u) {
    while (g_torchlean_deterministic_reductions_inited.load(std::memory_order_acquire) == 2u) {
    }
  }
  return g_torchlean_deterministic_reductions.load(std::memory_order_acquire);
}

extern "C" LEAN_EXPORT uint32_t torchlean_cuda_get_deterministic_reductions_u(uint32_t u) {
  (void)u;
  return torchlean_cuda_get_deterministic_reductions();
}

extern "C" LEAN_EXPORT uint32_t torchlean_cuda_set_deterministic_reductions_checked(uint32_t on) {
  torchlean_cuda_set_deterministic_reductions(on);
  return torchlean_cuda_get_deterministic_reductions();
}

extern "C" LEAN_EXPORT uint32_t torchlean_cuda_buffer_size(b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  if (b->size > 0xFFFFFFFFULL) {
    lean_internal_panic("torchlean_cuda_buffer_size: buffer too large for UInt32");
  }
  return (uint32_t)b->size;
}

extern "C" LEAN_EXPORT uint32_t torchlean_cuda_buffer_release(b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  if (!b || !b->data) {
    return 0;
  }
  checkCuda(cudaFree(b->data), "cudaFree buffer release failed");
  // Explicit release is an eager-runtime lifetime hint. We mark the handle as empty so accidental
  // reuse fails by size checks instead of touching freed device memory.
  b->data = NULL;
  b->size = 0;
  return 1;
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_release_then(
    b_lean_obj_arg scratchObj, b_lean_obj_arg keepObj) {
  (void)torchlean_cuda_buffer_release(scratchObj);
  lean_inc((lean_object*)keepObj);
  return (lean_object*)keepObj;
}

extern "C" LEAN_EXPORT uint32_t torchlean_runtime_collect_allocator(uint32_t force) {
  mi_collect(force != 0);
  return 1;
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_zeros(uint32_t n) {
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc((size_t)n);
  if (n == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  checkCuda(cudaMemset(out->data, 0, (size_t)n * sizeof(float)), "cudaMemset zeros failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_full(uint32_t n, double v) {
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc((size_t)n);
  if (n == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  dim3 blocks = torchlean_blocks_for((size_t)n);
  dim3 threads = dim3(kBlockSize);
  torchlean_fill_f32<<<blocks, threads>>>(out->data, (size_t)n, (float)v);
  checkCuda(cudaGetLastError(), "cuda full kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_rand_uniform(uint32_t n, uint64_t key) {
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc((size_t)n);
  if (n == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  dim3 blocks = torchlean_blocks_for((size_t)n);
  dim3 threads = dim3(kBlockSize);
  torchlean_rand_uniform_f32<<<blocks, threads>>>(out->data, (size_t)n, key);
  checkCuda(cudaGetLastError(), "cuda rand_uniform kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_bernoulli_mask(uint32_t n, double keepProb,
                                                                        uint64_t key) {
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc((size_t)n);
  if (n == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  dim3 blocks = torchlean_blocks_for((size_t)n);
  dim3 threads = dim3(kBlockSize);
  torchlean_bernoulli_mask_f32<<<blocks, threads>>>(out->data, (size_t)n, (float)keepProb, key);
  checkCuda(cudaGetLastError(), "cuda bernoulli_mask kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_of_float_array(b_lean_obj_arg AObj) {
  lean_object* A = (lean_object*)AObj;
  size_t n = lean_sarray_size(A);
  const double* src = lean_float_array_cptr(A);

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(n);
  if (n == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  float* tmp = (float*)malloc(n * sizeof(float));
  if (!tmp) {
    lean_internal_panic_out_of_memory();
  }
  for (size_t i = 0; i < n; ++i) {
    tmp[i] = (float)src[i];
  }
  checkCuda(cudaMemcpy(out->data, tmp, n * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy H2D failed");
  free(tmp);
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_to_float_array(b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  size_t n = b->size;

  lean_object* out = lean_mk_empty_float_array(lean_box(n));
  lean_sarray_set_size(out, n);
  double* dst = lean_float_array_cptr(out);

  if (n == 0) {
    return out;
  }

  float* tmp = (float*)malloc(n * sizeof(float));
  if (!tmp) {
    lean_internal_panic_out_of_memory();
  }
  checkCuda(cudaMemcpy(tmp, b->data, n * sizeof(float), cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H failed");
  for (size_t i = 0; i < n; ++i) {
    dst[i] = (double)tmp[i];
  }
  free(tmp);
  return out;
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_abs(b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(b->size);
  if (b->size == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  dim3 blocks = torchlean_blocks_for(b->size);
  dim3 threads = dim3(kBlockSize);
  torchlean_abs_f32<<<blocks, threads>>>(b->data, out->data, b->size);
  checkCuda(cudaGetLastError(), "cuda abs kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_abs_bwd(b_lean_obj_arg XObj,
                                                                 b_lean_obj_arg GObj) {
  torchlean_cuda_buffer* x = torchlean_cuda_buffer_unbox(XObj);
  torchlean_cuda_buffer* g = torchlean_cuda_buffer_unbox(GObj);
  if (x->size != g->size) {
    lean_internal_panic("torchlean_cuda_buffer_abs_bwd: size mismatch");
  }
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(x->size);
  if (x->size == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  dim3 blocks = torchlean_blocks_for(x->size);
  dim3 threads = dim3(kBlockSize);
  torchlean_abs_bwd_f32<<<blocks, threads>>>(x->data, g->data, out->data, x->size);
  checkCuda(cudaGetLastError(), "cuda abs_bwd kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_sqrt(b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(b->size);
  if (b->size == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  dim3 blocks = torchlean_blocks_for(b->size);
  dim3 threads = dim3(kBlockSize);
  torchlean_sqrt_f32<<<blocks, threads>>>(b->data, out->data, b->size);
  checkCuda(cudaGetLastError(), "cuda sqrt kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_sqrt_bwd(b_lean_obj_arg XObj,
                                                                  b_lean_obj_arg GObj) {
  torchlean_cuda_buffer* x = torchlean_cuda_buffer_unbox(XObj);
  torchlean_cuda_buffer* g = torchlean_cuda_buffer_unbox(GObj);
  if (x->size != g->size) {
    lean_internal_panic("torchlean_cuda_buffer_sqrt_bwd: size mismatch");
  }
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(x->size);
  if (x->size == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  dim3 blocks = torchlean_blocks_for(x->size);
  dim3 threads = dim3(kBlockSize);
  torchlean_sqrt_bwd_f32<<<blocks, threads>>>(x->data, g->data, out->data, x->size);
  checkCuda(cudaGetLastError(), "cuda sqrt_bwd kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_exp(b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(b->size);
  if (b->size == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  dim3 blocks = torchlean_blocks_for(b->size);
  dim3 threads = dim3(kBlockSize);
  torchlean_exp_f32<<<blocks, threads>>>(b->data, out->data, b->size);
  checkCuda(cudaGetLastError(), "cuda exp kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_log(b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(b->size);
  if (b->size == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  dim3 blocks = torchlean_blocks_for(b->size);
  dim3 threads = dim3(kBlockSize);
  torchlean_log_f32<<<blocks, threads>>>(b->data, out->data, b->size);
  checkCuda(cudaGetLastError(), "cuda log kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_inv(b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(b->size);
  if (b->size == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  dim3 blocks = torchlean_blocks_for(b->size);
  dim3 threads = dim3(kBlockSize);
  torchlean_inv_f32<<<blocks, threads>>>(b->data, out->data, b->size);
  checkCuda(cudaGetLastError(), "cuda inv kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_clamp(b_lean_obj_arg BObj, double lo,
                                                                double hi) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(b->size);
  if (b->size == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  dim3 blocks = torchlean_blocks_for(b->size);
  dim3 threads = dim3(kBlockSize);
  torchlean_clamp_f32<<<blocks, threads>>>(b->data, out->data, b->size, (float)lo, (float)hi);
  checkCuda(cudaGetLastError(), "cuda clamp kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_clamp_bwd(b_lean_obj_arg XObj,
                                                                   b_lean_obj_arg GObj, double lo,
                                                                   double hi) {
  torchlean_cuda_buffer* x = torchlean_cuda_buffer_unbox(XObj);
  torchlean_cuda_buffer* g = torchlean_cuda_buffer_unbox(GObj);
  if (x->size != g->size) {
    lean_internal_panic("torchlean_cuda_buffer_clamp_bwd: size mismatch");
  }
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(x->size);
  if (x->size == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  dim3 blocks = torchlean_blocks_for(x->size);
  dim3 threads = dim3(kBlockSize);
  torchlean_clamp_bwd_f32<<<blocks, threads>>>(x->data, g->data, out->data, x->size, (float)lo,
                                              (float)hi);
  checkCuda(cudaGetLastError(), "cuda clamp_bwd kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_max(b_lean_obj_arg AObj,
                                                             b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* a = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  if (a->size != b->size) {
    lean_internal_panic("torchlean_cuda_buffer_max: size mismatch");
  }
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(a->size);
  if (a->size == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  dim3 blocks = torchlean_blocks_for(a->size);
  dim3 threads = dim3(kBlockSize);
  torchlean_max_f32<<<blocks, threads>>>(a->data, b->data, out->data, a->size);
  checkCuda(cudaGetLastError(), "cuda max kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_max_bwd(b_lean_obj_arg AObj,
                                                                 b_lean_obj_arg BObj,
                                                                 b_lean_obj_arg GObj) {
  torchlean_cuda_buffer* a = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* g = torchlean_cuda_buffer_unbox(GObj);
  if (a->size != b->size || a->size != g->size) {
    lean_internal_panic("torchlean_cuda_buffer_max_bwd: size mismatch");
  }

  torchlean_cuda_buffer* dA = torchlean_cuda_buffer_alloc(a->size);
  torchlean_cuda_buffer* dB = torchlean_cuda_buffer_alloc(a->size);
  if (a->size == 0) {
    lean_object* pair = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(pair, 0, torchlean_cuda_buffer_box(dA));
    lean_ctor_set(pair, 1, torchlean_cuda_buffer_box(dB));
    return pair;
  }

  dim3 blocks = torchlean_blocks_for(a->size);
  dim3 threads = dim3(kBlockSize);
  torchlean_max_bwd_f32<<<blocks, threads>>>(a->data, b->data, g->data, dA->data, dB->data, a->size);
  checkCuda(cudaGetLastError(), "cuda max_bwd kernel launch failed");

  lean_object* pair = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(pair, 0, torchlean_cuda_buffer_box(dA));
  lean_ctor_set(pair, 1, torchlean_cuda_buffer_box(dB));
  return pair;
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_min(b_lean_obj_arg AObj,
                                                             b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* a = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  if (a->size != b->size) {
    lean_internal_panic("torchlean_cuda_buffer_min: size mismatch");
  }
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(a->size);
  if (a->size == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  dim3 blocks = torchlean_blocks_for(a->size);
  dim3 threads = dim3(kBlockSize);
  torchlean_min_f32<<<blocks, threads>>>(a->data, b->data, out->data, a->size);
  checkCuda(cudaGetLastError(), "cuda min kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_min_bwd(b_lean_obj_arg AObj,
                                                                 b_lean_obj_arg BObj,
                                                                 b_lean_obj_arg GObj) {
  torchlean_cuda_buffer* a = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* g = torchlean_cuda_buffer_unbox(GObj);
  if (a->size != b->size || a->size != g->size) {
    lean_internal_panic("torchlean_cuda_buffer_min_bwd: size mismatch");
  }

  torchlean_cuda_buffer* dA = torchlean_cuda_buffer_alloc(a->size);
  torchlean_cuda_buffer* dB = torchlean_cuda_buffer_alloc(a->size);
  if (a->size == 0) {
    lean_object* pair = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(pair, 0, torchlean_cuda_buffer_box(dA));
    lean_ctor_set(pair, 1, torchlean_cuda_buffer_box(dB));
    return pair;
  }

  dim3 blocks = torchlean_blocks_for(a->size);
  dim3 threads = dim3(kBlockSize);
  torchlean_min_bwd_f32<<<blocks, threads>>>(a->data, b->data, g->data, dA->data, dB->data, a->size);
  checkCuda(cudaGetLastError(), "cuda min_bwd kernel launch failed");

  lean_object* pair = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(pair, 0, torchlean_cuda_buffer_box(dA));
  lean_ctor_set(pair, 1, torchlean_cuda_buffer_box(dB));
  return pair;
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_div(b_lean_obj_arg AObj,
                                                             b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* a = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  if (a->size != b->size) {
    lean_internal_panic("torchlean_cuda_buffer_div: size mismatch");
  }
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(a->size);
  if (a->size == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  dim3 blocks = torchlean_blocks_for(a->size);
  dim3 threads = dim3(kBlockSize);
  torchlean_div_f32<<<blocks, threads>>>(a->data, b->data, out->data, a->size);
  checkCuda(cudaGetLastError(), "cuda div kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_relu(b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(b->size);
  if (b->size == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  dim3 blocks = torchlean_blocks_for(b->size);
  dim3 threads = dim3(kBlockSize);
  torchlean_relu_f32<<<blocks, threads>>>(b->data, out->data, b->size);
  checkCuda(cudaGetLastError(), "cuda relu kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_relu_bwd(b_lean_obj_arg XObj,
                                                                  b_lean_obj_arg GObj) {
  torchlean_cuda_buffer* x = torchlean_cuda_buffer_unbox(XObj);
  torchlean_cuda_buffer* g = torchlean_cuda_buffer_unbox(GObj);
  if (x->size != g->size) {
    lean_internal_panic("torchlean_cuda_buffer_relu_bwd: size mismatch");
  }
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(x->size);
  if (x->size == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  dim3 blocks = torchlean_blocks_for(x->size);
  dim3 threads = dim3(kBlockSize);
  torchlean_relu_bwd_f32<<<blocks, threads>>>(x->data, g->data, out->data, x->size);
  checkCuda(cudaGetLastError(), "cuda relu_bwd kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_add(b_lean_obj_arg AObj,
                                                             b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* a = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  if (a->size != b->size) {
    char msg[160];
    snprintf(msg, sizeof(msg), "torchlean_cuda_buffer_add: size mismatch (%zu vs %zu)", a->size,
             b->size);
    lean_internal_panic(msg);
  }
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(a->size);
  if (a->size == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  dim3 blocks = torchlean_blocks_for(a->size);
  dim3 threads = dim3(kBlockSize);
  torchlean_add_f32<<<blocks, threads>>>(a->data, b->data, out->data, a->size);
  checkCuda(cudaGetLastError(), "cuda add kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_sub(b_lean_obj_arg AObj,
                                                             b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* a = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  if (a->size != b->size) {
    lean_internal_panic("torchlean_cuda_buffer_sub: size mismatch");
  }
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(a->size);
  if (a->size == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  dim3 blocks = torchlean_blocks_for(a->size);
  dim3 threads = dim3(kBlockSize);
  torchlean_sub_f32<<<blocks, threads>>>(a->data, b->data, out->data, a->size);
  checkCuda(cudaGetLastError(), "cuda sub kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_mul(b_lean_obj_arg AObj,
                                                             b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* a = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  if (a->size != b->size) {
    lean_internal_panic("torchlean_cuda_buffer_mul: size mismatch");
  }
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(a->size);
  if (a->size == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  dim3 blocks = torchlean_blocks_for(a->size);
  dim3 threads = dim3(kBlockSize);
  torchlean_mul_f32<<<blocks, threads>>>(a->data, b->data, out->data, a->size);
  checkCuda(cudaGetLastError(), "cuda mul kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_scale(b_lean_obj_arg BObj, double c) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(b->size);
  if (b->size == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  dim3 blocks = torchlean_blocks_for(b->size);
  dim3 threads = dim3(kBlockSize);
  torchlean_scale_f32<<<blocks, threads>>>(b->data, out->data, b->size, (float)c);
  checkCuda(cudaGetLastError(), "cuda scale kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_axpy(b_lean_obj_arg AObj,
                                                              b_lean_obj_arg BObj, double c) {
  torchlean_cuda_buffer* a = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  if (a->size != b->size) {
    lean_internal_panic("torchlean_cuda_buffer_axpy: size mismatch");
  }
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(a->size);
  if (a->size == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  dim3 blocks = torchlean_blocks_for(a->size);
  dim3 threads = dim3(kBlockSize);
  torchlean_axpy_f32<<<blocks, threads>>>(a->data, b->data, out->data, a->size, (float)c);
  checkCuda(cudaGetLastError(), "cuda axpy kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_sum(b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(1);

  if (torchlean_cuda_get_deterministic_reductions()) {
    // Deterministic (fixed-order) reduction: no atomics; bit-stable across runs.
    torchlean_reduce_sum_deterministic(b->data, b->size, out->data);
  } else {
    // Fast reduction: atomics are correct but the interleaving order is non-deterministic.
    float zero = 0.0f;
    checkCuda(cudaMemcpy(out->data, &zero, sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy reduceSum init failed");
    if (b->size != 0) {
      dim3 blocks = torchlean_blocks_for(b->size);
      dim3 threads = dim3(kBlockSize);
      torchlean_reduce_sum_f32<<<blocks, threads>>>(b->data, out->data, b->size);
      checkCuda(cudaGetLastError(), "cuda reduceSum kernel launch failed");
    }
  }
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_mean(b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(1);

  if (b->size == 0) {
    float nanv = NAN;
    checkCuda(cudaMemcpy(out->data, &nanv, sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy reduceMean init failed");
    return torchlean_cuda_buffer_box(out);
  }

  if (torchlean_cuda_get_deterministic_reductions()) {
    // Deterministic: compute sum without atomics, then scale.
    torchlean_reduce_sum_deterministic(b->data, b->size, out->data);
  } else {
    float zero = 0.0f;
    checkCuda(cudaMemcpy(out->data, &zero, sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy reduceMean init failed");
    dim3 blocks = torchlean_blocks_for(b->size);
    dim3 threads = dim3(kBlockSize);
    torchlean_reduce_sum_f32<<<blocks, threads>>>(b->data, out->data, b->size);
    checkCuda(cudaGetLastError(), "cuda reduceMean reduce kernel launch failed");
  }

  float scale = 1.0f / (float)b->size;
  torchlean_scale1_f32<<<dim3(1), dim3(1)>>>(out->data, scale);
  checkCuda(cudaGetLastError(), "cuda reduceMean scale kernel launch failed");

  return torchlean_cuda_buffer_box(out);
}
