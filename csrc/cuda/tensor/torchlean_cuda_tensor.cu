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
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Float32 `Cuda.Buffer` runtime: allocation, copies, elementwise kernels, reductions, RNG, and
// explicit release hooks.
// We still check sizes at the FFI boundary, even when Lean has already tracked the shape.

// Process-wide switch for deterministic reductions.
static std::atomic<uint32_t> g_torchlean_deterministic_reductions{0u};
// 0 = uninitialized, 1 = initialized, 2 = another thread is initializing.
static std::atomic<uint32_t> g_torchlean_deterministic_reductions_inited{0u};

static std::atomic<uint64_t> g_torchlean_cuda_live_bytes{0u};
static std::atomic<uint64_t> g_torchlean_cuda_peak_bytes{0u};
static std::atomic<uint64_t> g_torchlean_cuda_alloc_count{0u};
static std::atomic<uint64_t> g_torchlean_cuda_free_count{0u};
static std::atomic<uint64_t> g_torchlean_cuda_wrapper_live_count{0u};
static std::atomic<uint64_t> g_torchlean_cuda_wrapper_peak_count{0u};
static std::atomic<uint64_t> g_torchlean_cuda_wrapper_alloc_count{0u};
static std::atomic<uint64_t> g_torchlean_cuda_wrapper_finalize_count{0u};

static void torchlean_cuda_note_wrapper_alloc(void) {
  g_torchlean_cuda_wrapper_alloc_count.fetch_add(1u, std::memory_order_relaxed);
  const uint64_t live =
      g_torchlean_cuda_wrapper_live_count.fetch_add(1u, std::memory_order_relaxed) + 1u;
  uint64_t peak = g_torchlean_cuda_wrapper_peak_count.load(std::memory_order_relaxed);
  while (live > peak &&
         !g_torchlean_cuda_wrapper_peak_count.compare_exchange_weak(
             peak, live, std::memory_order_relaxed, std::memory_order_relaxed)) {
  }
}

static void torchlean_cuda_note_wrapper_finalize(void) {
  g_torchlean_cuda_wrapper_finalize_count.fetch_add(1u, std::memory_order_relaxed);
  g_torchlean_cuda_wrapper_live_count.fetch_sub(1u, std::memory_order_relaxed);
}

// 0 = CPU stub, 1 = native CUDA with a visible device, 2 = native CUDA without one.
extern "C" LEAN_EXPORT uint32_t torchlean_cuda_runtime_status(uint32_t token) {
  (void)token;
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess || count <= 0) {
    // Clear the runtime error so a diagnostic probe does not poison the next CUDA call.
    (void)cudaGetLastError();
    return 2u;
  }
  return 1u;
}

extern "C" void torchlean_cuda_kernels_flush_scratch_cache(void);
extern "C" void torchlean_cuda_conv_pool_flush_scratch_cache(void);
extern "C" void torchlean_cuda_blas_flush_scratch_cache(void);

struct torchlean_cuda_cached_block {
  size_t size;
  float* data;
  cudaEvent_t ready;
};

static void torchlean_cuda_free_best_effort(void* ptr, const char* what);
static void torchlean_cuda_destroy_event_best_effort(cudaEvent_t event, const char* what);
static void torchlean_cuda_synchronize_event_best_effort(cudaEvent_t event, const char* what);

static pthread_mutex_t g_torchlean_cuda_cache_mutex = PTHREAD_MUTEX_INITIALIZER;
static torchlean_cuda_cached_block* g_torchlean_cuda_cache = nullptr;
static size_t g_torchlean_cuda_cache_count = 0;
static size_t g_torchlean_cuda_cache_slot_capacity = 0;
// Total device bytes currently held in the reuse cache (sum of the cached blocks' byte sizes),
// guarded by `g_torchlean_cuda_cache_mutex`. The cache holds buffers Lean has already dropped, so
// these bytes are NOT counted in `live_bytes`; left unbounded the cache can grow without limit.
static size_t g_torchlean_cuda_cache_bytes = 0;

// Optional byte cap on the reuse cache, read once from the environment.
// `TORCHLEAN_CUDA_CACHE_CAP_BYTES` is the maximum total bytes the cache may hold; 0 (the default)
// leaves it unbounded, preserving the prior behaviour exactly. When set, a returned block that would
// grow the cache past the cap is freed immediately instead of cached. The value must be a plain
// decimal byte count; a malformed or overflowing value is rejected with a warning and leaves the
// cache unbounded, rather than being silently misread as some other cap.
//
// The environment is read exactly once, under `pthread_once`, so concurrent first callers cannot
// race on the parse: the initializer runs on a single thread while the others block, and the value
// is published before any caller observes it. (A plain function-local static assigned after its
// declaration would be a data race, and would depend on `-fthreadsafe-statics` being enabled.)
static size_t g_torchlean_cuda_cache_cap_value = 0;
static pthread_once_t g_torchlean_cuda_cache_cap_once = PTHREAD_ONCE_INIT;

// Strict decimal parser for the cap: accepts exactly a non-empty digit string whose value fits
// in `size_t` — no sign, no whitespace, no base prefix — and reports malformed or overflowing
// input instead of guessing. A hand-rolled loop rather than `strtoull` also keeps this file free
// of the glibc >= 2.38 `__isoc23_strtoull` symbol redirection, so the object stays self-contained.
static bool torchlean_cuda_parse_cache_cap(const char* s, size_t* out) {
  size_t value = 0;
  for (const char* p = s; *p; ++p) {
    if (*p < '0' || *p > '9') {
      return false;
    }
    const size_t digit = (size_t)(*p - '0');
    if (value > (SIZE_MAX - digit) / 10) {
      return false;
    }
    value = value * 10 + digit;
  }
  *out = value;
  return true;
}

static void torchlean_cuda_cache_byte_cap_init(void) {
  const char* v = getenv("TORCHLEAN_CUDA_CACHE_CAP_BYTES");
  if (!v || !v[0]) {
    return;  // unset or empty: the cache stays unbounded (cap 0)
  }
  size_t parsed = 0;
  if (torchlean_cuda_parse_cache_cap(v, &parsed)) {
    g_torchlean_cuda_cache_cap_value = parsed;
  } else {
    fprintf(stderr,
            "TorchLean CUDA warning: ignoring TORCHLEAN_CUDA_CACHE_CAP_BYTES='%s': "
            "expected an unsigned decimal byte count; the cache stays unbounded\n",
            v);
  }
}

static size_t torchlean_cuda_cache_byte_cap(void) {
  pthread_once(&g_torchlean_cuda_cache_cap_once, torchlean_cuda_cache_byte_cap_init);
  return g_torchlean_cuda_cache_cap_value;
}

// Reuse exact-size buffers only after CUDA has recorded that all earlier work using the block has
// completed. This lowers allocator pressure during long training loops without forcing a global
// device synchronization on every free.
static void torchlean_cuda_cache_push(torchlean_cuda_cached_block block) {
  if (g_torchlean_cuda_cache_count == g_torchlean_cuda_cache_slot_capacity) {
    size_t new_capacity =
        g_torchlean_cuda_cache_slot_capacity == 0 ? 16 : g_torchlean_cuda_cache_slot_capacity * 2;
    void* next =
        realloc(g_torchlean_cuda_cache, new_capacity * sizeof(torchlean_cuda_cached_block));
    if (!next) {
      lean_internal_panic_out_of_memory();
    }
    g_torchlean_cuda_cache = (torchlean_cuda_cached_block*)next;
    g_torchlean_cuda_cache_slot_capacity = new_capacity;
  }
  g_torchlean_cuda_cache[g_torchlean_cuda_cache_count++] = block;
}

static float* torchlean_cuda_take_cached_block(size_t n) {
  if (n == 0) {
    return NULL;
  }
  torchlean_cuda_lock(&g_torchlean_cuda_cache_mutex, "pthread_mutex_lock buffer cache failed");
  for (size_t i = 0; i < g_torchlean_cuda_cache_count; ++i) {
    if (g_torchlean_cuda_cache[i].size != n) {
      continue;
    }
    cudaError_t ready = cudaEventQuery(g_torchlean_cuda_cache[i].ready);
    if (ready == cudaSuccess) {
      float* data = g_torchlean_cuda_cache[i].data;
      torchlean_cuda_destroy_event_best_effort(g_torchlean_cuda_cache[i].ready,
                                               "cudaEventDestroy cached buffer reuse failed");
      g_torchlean_cuda_cache[i] = g_torchlean_cuda_cache[g_torchlean_cuda_cache_count - 1];
      g_torchlean_cuda_cache_count--;
      g_torchlean_cuda_cache_bytes -= (size_t)torchlean_float_bytes_for(n);
      torchlean_cuda_unlock(&g_torchlean_cuda_cache_mutex,
                            "pthread_mutex_unlock buffer cache failed");
      return data;
    }
    if (ready != cudaErrorNotReady) {
      fprintf(stderr, "TorchLean CUDA warning: cudaEventQuery cached buffer failed: %s\n",
              cudaGetErrorString(ready));
    }
  }
  torchlean_cuda_unlock(&g_torchlean_cuda_cache_mutex, "pthread_mutex_unlock buffer cache failed");
  return NULL;
}

static void torchlean_cuda_return_cached_block(size_t n, float* data) {
  if (!data || n == 0) {
    return;
  }
  cudaEvent_t ready = nullptr;
  cudaError_t err = cudaEventCreateWithFlags(&ready, cudaEventDisableTiming);
  if (err != cudaSuccess) {
    fprintf(stderr, "TorchLean CUDA warning: cudaEventCreate cached buffer failed: %s\n",
            cudaGetErrorString(err));
    torchlean_cuda_free_best_effort(data, "cudaFree uncached buffer after event-create failure failed");
    return;
  }
  err = cudaEventRecord(ready, 0);
  if (err != cudaSuccess) {
    fprintf(stderr, "TorchLean CUDA warning: cudaEventRecord cached buffer failed: %s\n",
            cudaGetErrorString(err));
    torchlean_cuda_destroy_event_best_effort(ready,
                                             "cudaEventDestroy buffer after record failure failed");
    torchlean_cuda_free_best_effort(data, "cudaFree uncached buffer after event-record failure failed");
    return;
  }
  const size_t incoming = (size_t)torchlean_float_bytes_for(n);
  const size_t cap = torchlean_cuda_cache_byte_cap();
  bool over_cap = false;
  torchlean_cuda_lock(&g_torchlean_cuda_cache_mutex, "pthread_mutex_lock buffer return failed");
  // Overflow-safe form of `g_torchlean_cuda_cache_bytes + incoming > cap`: the sum is never formed,
  // so it cannot wrap `size_t`. A single incoming block larger than the cap trips it directly;
  // otherwise `cap - incoming` is a well-defined non-negative headroom that the current total must
  // not exceed.
  if (cap != 0 &&
      (incoming > cap || g_torchlean_cuda_cache_bytes > cap - incoming)) {
    over_cap = true;
  } else {
    torchlean_cuda_cache_push({n, data, ready});
    g_torchlean_cuda_cache_bytes += incoming;
  }
  torchlean_cuda_unlock(&g_torchlean_cuda_cache_mutex, "pthread_mutex_unlock buffer return failed");
  if (over_cap) {
    // Caching this block would grow the process-global cache past the byte cap, so free it now
    // instead. The just-recorded event signals when pending work on the block completes; wait on it
    // before releasing the device memory, exactly as the flush path does, so an in-flight kernel
    // never reads freed memory.
    torchlean_cuda_synchronize_event_best_effort(ready, "cudaEventSynchronize over-cap buffer failed");
    torchlean_cuda_destroy_event_best_effort(ready, "cudaEventDestroy over-cap buffer failed");
    torchlean_cuda_free_best_effort(data, "cudaFree over-cap buffer failed");
  }
}

static void torchlean_cuda_flush_cached_blocks(void) {
  torchlean_cuda_lock(&g_torchlean_cuda_cache_mutex, "pthread_mutex_lock buffer flush failed");
  torchlean_cuda_cached_block* blocks = g_torchlean_cuda_cache;
  size_t count = g_torchlean_cuda_cache_count;
  g_torchlean_cuda_cache = nullptr;
  g_torchlean_cuda_cache_count = 0;
  g_torchlean_cuda_cache_slot_capacity = 0;
  g_torchlean_cuda_cache_bytes = 0;
  torchlean_cuda_unlock(&g_torchlean_cuda_cache_mutex, "pthread_mutex_unlock buffer flush failed");

  for (size_t i = 0; i < count; ++i) {
    torchlean_cuda_cached_block block = blocks[i];
    torchlean_cuda_synchronize_event_best_effort(block.ready,
                                                 "cudaEventSynchronize cached buffer failed");
    torchlean_cuda_destroy_event_best_effort(block.ready,
                                             "cudaEventDestroy cached buffer failed");
    torchlean_cuda_free_best_effort(block.data, "cudaFree cached buffer failed");
  }
  free(blocks);
}

static void torchlean_cuda_note_alloc(size_t n) {
  const uint64_t bytes = torchlean_float_bytes_for(n);
  g_torchlean_cuda_alloc_count.fetch_add(1u, std::memory_order_relaxed);
  const uint64_t live =
      g_torchlean_cuda_live_bytes.fetch_add(bytes, std::memory_order_relaxed) + bytes;
  uint64_t peak = g_torchlean_cuda_peak_bytes.load(std::memory_order_relaxed);
  while (live > peak &&
         !g_torchlean_cuda_peak_bytes.compare_exchange_weak(
             peak, live, std::memory_order_relaxed, std::memory_order_relaxed)) {
  }
}

static void torchlean_cuda_note_free(size_t n) {
  const uint64_t bytes = torchlean_float_bytes_for(n);
  g_torchlean_cuda_free_count.fetch_add(1u, std::memory_order_relaxed);
  uint64_t live = g_torchlean_cuda_live_bytes.load(std::memory_order_relaxed);
  while (true) {
    const uint64_t next = live > bytes ? live - bytes : 0u;
    if (g_torchlean_cuda_live_bytes.compare_exchange_weak(
            live, next, std::memory_order_relaxed, std::memory_order_relaxed)) {
      return;
    }
  }
}

static void torchlean_cuda_panic_malloc_failed(size_t n, cudaError_t err) {
  size_t freeBytes = 0;
  size_t totalBytes = 0;
  (void)cudaMemGetInfo(&freeBytes, &totalBytes);
  char msg[512];
  snprintf(msg, sizeof(msg),
           "cudaMalloc buffer failed: requested=%llu bytes live=%llu peak=%llu "
           "allocs=%llu frees=%llu cuda_free=%llu cuda_total=%llu error=%s",
           (unsigned long long)torchlean_float_bytes_for(n),
           (unsigned long long)g_torchlean_cuda_live_bytes.load(std::memory_order_relaxed),
           (unsigned long long)g_torchlean_cuda_peak_bytes.load(std::memory_order_relaxed),
           (unsigned long long)g_torchlean_cuda_alloc_count.load(std::memory_order_relaxed),
           (unsigned long long)g_torchlean_cuda_free_count.load(std::memory_order_relaxed),
           (unsigned long long)freeBytes, (unsigned long long)totalBytes,
           cudaGetErrorString(err));
  lean_internal_panic(msg);
}

static void torchlean_cuda_free_best_effort(void* ptr, const char* what) {
  if (!ptr) {
    return;
  }
  cudaError_t err = cudaFree(ptr);
  if (err != cudaSuccess) {
    fprintf(stderr, "TorchLean CUDA warning: %s: %s\n", what, cudaGetErrorString(err));
  }
}

static void torchlean_cuda_destroy_event_best_effort(cudaEvent_t event, const char* what) {
  if (!event) {
    return;
  }
  cudaError_t err = cudaEventDestroy(event);
  if (err != cudaSuccess) {
    fprintf(stderr, "TorchLean CUDA warning: %s: %s\n", what, cudaGetErrorString(err));
  }
}

static void torchlean_cuda_synchronize_event_best_effort(cudaEvent_t event, const char* what) {
  if (!event) {
    return;
  }
  cudaError_t err = cudaEventSynchronize(event);
  if (err != cudaSuccess) {
    fprintf(stderr, "TorchLean CUDA warning: %s: %s\n", what, cudaGetErrorString(err));
  }
}

static bool torchlean_cuda_buffer_release_data(torchlean_cuda_buffer* b) {
  if (!b || !b->data) {
    return false;
  }
  torchlean_cuda_note_free(b->size);
  torchlean_cuda_return_cached_block(b->size, b->data);
  b->data = NULL;
  b->size = 0;
  return true;
}

static void torchlean_cuda_buffer_finalize(void* ptr) {
  torchlean_cuda_buffer* b = (torchlean_cuda_buffer*)ptr;
  if (!b) {
    return;
  }
  (void)torchlean_cuda_buffer_release_data(b);
  free(b);
  torchlean_cuda_note_wrapper_finalize();
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
  lean_obj_res boxed = lean_alloc_external(torchlean_cuda_buffer_get_class(), b);
  torchlean_cuda_note_wrapper_alloc();
  return boxed;
}

extern "C" void torchlean_cuda_buffer_drop_unboxed(torchlean_cuda_buffer* b) {
  if (!b) {
    return;
  }
  (void)torchlean_cuda_buffer_release_data(b);
  free(b);
}

extern "C" torchlean_cuda_buffer* torchlean_cuda_buffer_alloc(size_t n) {
  torchlean_cuda_buffer* b = (torchlean_cuda_buffer*)malloc(sizeof(torchlean_cuda_buffer));
  if (!b) {
    lean_internal_panic_out_of_memory();
  }
  b->size = n;
  b->data = NULL;
  if (n > 0) {
    b->data = torchlean_cuda_take_cached_block(n);
  }
  if (n > 0 && !b->data) {
    const size_t bytes =
        checked_bytes_size(n, sizeof(float), "torchlean_cuda_buffer_alloc: byte size overflow");
    cudaError_t err = cudaMalloc((void**)&b->data, bytes);
    if (err != cudaSuccess) {
      if (err == cudaErrorMemoryAllocation) {
        // `cudaMalloc` records the allocation failure in CUDA's per-thread last-error slot as
        // well as returning it directly. Clear that copy before retrying; otherwise the next
        // successful kernel launch can be blamed for this already-recovered OOM.
        (void)cudaGetLastError();
        torchlean_cuda_flush_cached_blocks();
        err = cudaMalloc((void**)&b->data, bytes);
      }
      if (err != cudaSuccess) {
        free(b);
        torchlean_cuda_panic_malloc_failed(n, err);
      }
    }
  }
  if (n > 0) {
    torchlean_cuda_note_alloc(n);
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

// Scaled product exponential `exp((c*a)*b)` fused into one kernel (one launch, one result
// buffer) instead of the four elementwise ops of the composed form. Left-associated and in fp32
// throughout, so it is bit-identical to `exp(mul(mul(full(c), a), b))`.
__global__ void torchlean_scaled_prod_exp_f32(const float* a, const float* b, float* out, size_t n,
                                              float c) {
  TORCHLEAN_GRID_STRIDE_LOOP(i, n) {
    out[i] = expf((c * a[i]) * b[i]);
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

__global__ void torchlean_adam_step_f32(
    const float* parameters,
    const float* gradient,
    const float* first_moment,
    const float* second_moment,
    float* updated_parameters,
    float* updated_first_moment,
    float* updated_second_moment,
    size_t n,
    float beta1,
    float one_minus_beta1,
    float beta2,
    float one_minus_beta2,
    float first_moment_correction,
    float second_moment_correction,
    float epsilon,
    float decay,
    float update_scale) {
  TORCHLEAN_GRID_STRIDE_LOOP(i, n) {
    // Keep the same rounded stages as the former composition of scale, mul, axpy, sqrt, add,
    // and div kernels. Explicit round-to-nearest intrinsics prevent the compiler from contracting
    // operations across those semantic stage boundaries.
    const float m_scaled = __fmul_rn(first_moment[i], beta1);
    const float m = __fmaf_rn(one_minus_beta1, gradient[i], m_scaled);
    const float g2 = __fmul_rn(gradient[i], gradient[i]);
    const float v_scaled = __fmul_rn(second_moment[i], beta2);
    const float v = __fmaf_rn(one_minus_beta2, g2, v_scaled);
    const float m_hat = __fmul_rn(m, first_moment_correction);
    const float v_hat = __fmul_rn(v, second_moment_correction);
    const float denominator = __fadd_rn(__fsqrt_rn(v_hat), epsilon);
    const float normalized_update = __fdiv_rn(m_hat, denominator);
    const float decayed_parameter = __fmaf_rn(decay, parameters[i], parameters[i]);

    updated_first_moment[i] = m;
    updated_second_moment[i] = v;
    updated_parameters[i] =
        __fmaf_rn(update_scale, normalized_update, decayed_parameter);
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

// Tanh-approximate GELU from Activation.Math.geluSpec. Explicit round-to-nearest operations keep
// the polynomial stages visible instead of allowing contraction into a different floating program.
__device__ __forceinline__ float torchlean_gelu_f32_value(float x) {
  constexpr float kCoeff = 0.044715f;
  constexpr float kSqrtTwoOverPi = 0.79788456080286535588f;
  const float cubic0 = __fmul_rn(kCoeff, x);
  const float cubic1 = __fmul_rn(cubic0, x);
  const float cubic = __fmul_rn(cubic1, x);
  const float inner = __fadd_rn(x, cubic);
  const float tanhTerm = tanhf(__fmul_rn(kSqrtTwoOverPi, inner));
  const float scaledInput = __fmul_rn(x, __fadd_rn(1.0f, tanhTerm));
  return __fdiv_rn(scaledInput, 2.0f);
}

__device__ __forceinline__ float torchlean_gelu_f32_deriv(float x) {
  constexpr float kCoeff = 0.044715f;
  constexpr float kSqrtTwoOverPi = 0.79788456080286535588f;
  const float cubic0 = __fmul_rn(kCoeff, x);
  const float cubic1 = __fmul_rn(cubic0, x);
  const float cubic = __fmul_rn(cubic1, x);
  const float inner = __fadd_rn(x, cubic);
  const float tanhTerm = tanhf(__fmul_rn(kSqrtTwoOverPi, inner));
  const float sechTerm = __fsub_rn(1.0f, __fmul_rn(tanhTerm, tanhTerm));
  const float scaledCoeff = __fmul_rn(3.0f, kCoeff);
  const float quadratic0 = __fmul_rn(scaledCoeff, x);
  const float quadratic = __fmul_rn(quadratic0, x);
  const float innerDeriv =
      __fmul_rn(kSqrtTwoOverPi, __fadd_rn(1.0f, quadratic));
  const float derivativeTerm =
      __fmul_rn(__fmul_rn(x, sechTerm), innerDeriv);
  const float numerator =
      __fadd_rn(__fadd_rn(1.0f, tanhTerm), derivativeTerm);
  return __fdiv_rn(numerator, 2.0f);
}

__global__ void torchlean_gelu_f32(const float* x, float* y, size_t n) {
  TORCHLEAN_GRID_STRIDE_LOOP(i, n) {
    y[i] = torchlean_gelu_f32_value(x[i]);
  }
}

__global__ void torchlean_gelu_bwd_f32(
    const float* x, const float* dLdy, float* dLdx, size_t n) {
  TORCHLEAN_GRID_STRIDE_LOOP(i, n) {
    dLdx[i] = __fmul_rn(torchlean_gelu_f32_deriv(x[i]), dLdy[i]);
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

__global__ void torchlean_rand_normal_f32(float* out, size_t n, float mean, float std,
                                          uint64_t key) {
  TORCHLEAN_GRID_STRIDE_LOOP(i, n) {
    uint32_t r1 = (uint32_t)torchlean_splitmix64(key + (uint64_t)(2 * i));
    uint32_t r2 = (uint32_t)torchlean_splitmix64(key + (uint64_t)(2 * i + 1));
    float u1 = ((float)r1 + 1.0f) / 4294967297.0f;
    float u2 = (float)(((double)r2) / 4294967296.0);
    float z = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853071795864769f * u2);
    out[i] = mean + std * z;
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
  // Deterministic reductions cap the number of blocks.
  // Grid-stride loops still cover every element with bounded scratch allocation.
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
  partial = (float*)torchlean_cuda_scratch_alloc_bytes(
      checked_bytes_size((size_t)blocks, sizeof(float),
                         "cudaMalloc deterministic reduceSum partials failed"),
      "cudaMalloc deterministic reduceSum partials failed");
  torchlean_reduce_sum_partials_f32<<<dim3(blocks), dim3(kBlockSize)>>>(in, partial, n);
  checkCuda(cudaGetLastError(), "cuda deterministic reduceSum partial kernel launch failed");

  // Iteratively reduce partials until we have a single scalar.
  size_t curSize = (size_t)blocks;
  while (curSize > 1) {
    unsigned int nextBlocks = torchlean_det_reduce_blocks_for(curSize);
    float* next = nullptr;
    next = (float*)torchlean_cuda_scratch_alloc_bytes(
        checked_bytes_size((size_t)nextBlocks, sizeof(float),
                           "cudaMalloc deterministic reduceSum next partials failed"),
        "cudaMalloc deterministic reduceSum next partials failed");
    torchlean_reduce_sum_partials_f32<<<dim3(nextBlocks), dim3(kBlockSize)>>>(partial, next, curSize);
    checkCuda(cudaGetLastError(), "cuda deterministic reduceSum next partial kernel launch failed");
    torchlean_cuda_scratch_free_bytes(
        (void**)&partial,
        checked_bytes_size(curSize, sizeof(float),
                           "cudaFree deterministic reduceSum partials failed"),
        "cudaFree deterministic reduceSum partials failed");
    partial = next;
    curSize = (size_t)nextBlocks;
  }

  checkCuda(cudaMemcpy(outScalar, partial, sizeof(float), cudaMemcpyDeviceToDevice),
            "cudaMemcpy deterministic reduceSum final copy failed");
  torchlean_cuda_scratch_free_bytes(
      (void**)&partial,
      checked_bytes_size(curSize, sizeof(float),
                         "cudaFree deterministic reduceSum final partial failed"),
      "cudaFree deterministic reduceSum final partial failed");
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

extern "C" LEAN_EXPORT uint64_t torchlean_cuda_allocator_live_bytes(uint32_t u) {
  (void)u;
  return g_torchlean_cuda_live_bytes.load(std::memory_order_relaxed);
}

extern "C" LEAN_EXPORT uint64_t torchlean_cuda_allocator_peak_bytes(uint32_t u) {
  (void)u;
  return g_torchlean_cuda_peak_bytes.load(std::memory_order_relaxed);
}

extern "C" LEAN_EXPORT uint64_t torchlean_cuda_allocator_alloc_count(uint32_t u) {
  (void)u;
  return g_torchlean_cuda_alloc_count.load(std::memory_order_relaxed);
}

extern "C" LEAN_EXPORT uint64_t torchlean_cuda_allocator_free_count(uint32_t u) {
  (void)u;
  return g_torchlean_cuda_free_count.load(std::memory_order_relaxed);
}

extern "C" LEAN_EXPORT uint64_t torchlean_cuda_wrapper_live_count(uint32_t u) {
  (void)u;
  return g_torchlean_cuda_wrapper_live_count.load(std::memory_order_relaxed);
}

extern "C" LEAN_EXPORT uint64_t torchlean_cuda_wrapper_peak_count(uint32_t u) {
  (void)u;
  return g_torchlean_cuda_wrapper_peak_count.load(std::memory_order_relaxed);
}

extern "C" LEAN_EXPORT uint64_t torchlean_cuda_wrapper_alloc_count(uint32_t u) {
  (void)u;
  return g_torchlean_cuda_wrapper_alloc_count.load(std::memory_order_relaxed);
}

extern "C" LEAN_EXPORT uint64_t torchlean_cuda_wrapper_finalize_count(uint32_t u) {
  (void)u;
  return g_torchlean_cuda_wrapper_finalize_count.load(std::memory_order_relaxed);
}

extern "C" LEAN_EXPORT uint64_t torchlean_cuda_allocator_device_free_bytes(uint32_t u) {
  (void)u;
  size_t freeBytes = 0;
  size_t totalBytes = 0;
  cudaError_t err = cudaMemGetInfo(&freeBytes, &totalBytes);
  return err == cudaSuccess ? (uint64_t)freeBytes : 0u;
}

extern "C" LEAN_EXPORT uint64_t torchlean_cuda_allocator_device_total_bytes(uint32_t u) {
  (void)u;
  size_t freeBytes = 0;
  size_t totalBytes = 0;
  cudaError_t err = cudaMemGetInfo(&freeBytes, &totalBytes);
  return err == cudaSuccess ? (uint64_t)totalBytes : 0u;
}

extern "C" LEAN_EXPORT uint64_t torchlean_cuda_allocator_cache_bytes(uint32_t u) {
  (void)u;
  torchlean_cuda_lock(&g_torchlean_cuda_cache_mutex, "pthread_mutex_lock cache-bytes query failed");
  uint64_t bytes = (uint64_t)g_torchlean_cuda_cache_bytes;
  torchlean_cuda_unlock(&g_torchlean_cuda_cache_mutex, "pthread_mutex_unlock cache-bytes query failed");
  return bytes;
}

extern "C" LEAN_EXPORT uint64_t torchlean_cuda_allocator_cache_cap_bytes(uint32_t u) {
  (void)u;
  return (uint64_t)torchlean_cuda_cache_byte_cap();
}

extern "C" LEAN_EXPORT uint32_t torchlean_cuda_buffer_size(b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  if (b->size > 0xFFFFFFFFULL) {
    lean_internal_panic("torchlean_cuda_buffer_size: buffer too large for UInt32");
  }
  return (uint32_t)b->size;
}

extern "C" LEAN_EXPORT uint32_t torchlean_cuda_buffer_size_with_token(
    b_lean_obj_arg BObj, uint32_t token) {
  (void)token;
  return torchlean_cuda_buffer_size(BObj);
}

extern "C" LEAN_EXPORT uint32_t torchlean_cuda_buffer_release(b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  // Explicit release is an eager-runtime lifetime hint. We mark the handle as empty so accidental
  // reuse fails by size checks instead of touching freed device memory.
  return torchlean_cuda_buffer_release_data(b) ? 1 : 0;
}

extern "C" LEAN_EXPORT uint32_t torchlean_cuda_buffer_release_with_token(
    b_lean_obj_arg BObj, uint32_t token) {
  (void)token;
  return torchlean_cuda_buffer_release(BObj);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_release_then(
    b_lean_obj_arg scratchObj, b_lean_obj_arg keepObj) {
  (void)torchlean_cuda_buffer_release(scratchObj);
  lean_inc((lean_object*)keepObj);
  return (lean_object*)keepObj;
}

extern "C" LEAN_EXPORT uint32_t torchlean_runtime_collect_allocator(uint32_t force) {
  const bool force_collect = force != 0;
  if (force_collect) {
    torchlean_cuda_flush_cached_blocks();
    torchlean_cuda_scratch_flush();
    torchlean_cuda_kernels_flush_scratch_cache();
    torchlean_cuda_conv_pool_flush_scratch_cache();
    torchlean_cuda_blas_flush_scratch_cache();
  }
  mi_collect(force_collect);
  mi_heap_collect(mi_heap_get_default(), force_collect);
  mi_collect_reduce(0);
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

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_full_with_token(
    uint32_t n, double v, uint32_t token) {
  (void)token;
  return torchlean_cuda_buffer_full(n, v);
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

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_rand_normal(
    uint32_t n, double meanHost, double stdHost, uint64_t key) {
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc((size_t)n);
  if (n == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  dim3 blocks = torchlean_blocks_for((size_t)n);
  dim3 threads = dim3(kBlockSize);
  torchlean_rand_normal_f32<<<blocks, threads>>>(out->data, (size_t)n, (float)meanHost,
                                                 (float)stdHost, key);
  checkCuda(cudaGetLastError(), "cuda rand_normal kernel launch failed");
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

  const size_t bytes =
      checked_bytes_size(n, sizeof(float), "torchlean_cuda_buffer_of_float_array: tmp size overflow");
  float* tmp = (float*)malloc(bytes);
  if (!tmp) {
    lean_internal_panic_out_of_memory();
  }
  for (size_t i = 0; i < n; ++i) {
    tmp[i] = (float)src[i];
  }
  cudaError_t err = cudaMemcpy(out->data, tmp, bytes, cudaMemcpyHostToDevice);
  free(tmp);
  checkCuda(err, "cudaMemcpy H2D failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res
torchlean_cuda_buffer_of_float_array_with_token(b_lean_obj_arg AObj, uint32_t token) {
  (void)token;
  return torchlean_cuda_buffer_of_float_array(AObj);
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

  const size_t bytes =
      checked_bytes_size(n, sizeof(float), "torchlean_cuda_buffer_to_float_array: tmp size overflow");
  float* tmp = (float*)malloc(bytes);
  if (!tmp) {
    lean_internal_panic_out_of_memory();
  }
  cudaError_t err = cudaMemcpy(tmp, b->data, bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    free(tmp);
    checkCuda(err, "cudaMemcpy D2H failed");
  }
  for (size_t i = 0; i < n; ++i) {
    dst[i] = (double)tmp[i];
  }
  free(tmp);
  return out;
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_to_float_array_io(
    b_lean_obj_arg BObj) {
  return lean_io_result_mk_ok(torchlean_cuda_buffer_to_float_array(BObj));
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_to_float32_bytes_io(
    b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  size_t bytes =
      checked_bytes_size(b->size, sizeof(float), "cuda checkpoint payload size overflow");
  lean_object* out = lean_alloc_sarray(1, bytes, bytes);
  if (bytes != 0) {
    float* values = (float*)malloc(bytes);
    if (!values) {
      lean_internal_panic_out_of_memory();
    }
    checkCuda(cudaMemcpy(values, b->data, bytes, cudaMemcpyDeviceToHost),
              "cuda checkpoint D2H copy failed");
    uint8_t* dst = (uint8_t*)lean_sarray_cptr(out);
    for (size_t i = 0; i < b->size; ++i) {
      uint32_t bits;
      memcpy(&bits, &values[i], sizeof(bits));
      dst[4 * i] = (uint8_t)(bits & 0xffu);
      dst[4 * i + 1] = (uint8_t)((bits >> 8) & 0xffu);
      dst[4 * i + 2] = (uint8_t)((bits >> 16) & 0xffu);
      dst[4 * i + 3] = (uint8_t)((bits >> 24) & 0xffu);
    }
    free(values);
  }
  return lean_io_result_mk_ok(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_of_float32_bytes_io(
    b_lean_obj_arg BytesObj) {
  lean_object* bytesObj = (lean_object*)BytesObj;
  size_t bytes = lean_sarray_size(bytesObj);
  if (bytes % sizeof(float) != 0) {
    lean_internal_panic("torchlean: float32 checkpoint payload is not a multiple of four bytes");
  }
  size_t n = bytes / sizeof(float);
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(n);
  if (bytes != 0) {
    float* values = (float*)malloc(bytes);
    if (!values) {
      lean_internal_panic_out_of_memory();
    }
    const uint8_t* src = (const uint8_t*)lean_sarray_cptr(bytesObj);
    for (size_t i = 0; i < n; ++i) {
      uint32_t bits = (uint32_t)src[4 * i] |
                      ((uint32_t)src[4 * i + 1] << 8) |
                      ((uint32_t)src[4 * i + 2] << 16) |
                      ((uint32_t)src[4 * i + 3] << 24);
      memcpy(&values[i], &bits, sizeof(bits));
    }
    checkCuda(cudaMemcpy(out->data, values, bytes, cudaMemcpyHostToDevice),
              "cuda checkpoint H2D copy failed");
    free(values);
  }
  return lean_io_result_mk_ok(torchlean_cuda_buffer_box(out));
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_float_array_to_float32_bytes(
    b_lean_obj_arg AObj) {
  lean_object* valuesObj = (lean_object*)AObj;
  size_t n = lean_sarray_size(valuesObj);
  size_t bytes = checked_bytes_size(n, sizeof(float), "float32 checkpoint payload size overflow");
  lean_object* out = lean_alloc_sarray(1, bytes, bytes);
  const double* src = lean_float_array_cptr(valuesObj);
  uint8_t* dst = (uint8_t*)lean_sarray_cptr(out);
  for (size_t i = 0; i < n; ++i) {
    float value = (float)src[i];
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    dst[4 * i] = (uint8_t)(bits & 0xffu);
    dst[4 * i + 1] = (uint8_t)((bits >> 8) & 0xffu);
    dst[4 * i + 2] = (uint8_t)((bits >> 16) & 0xffu);
    dst[4 * i + 3] = (uint8_t)((bits >> 24) & 0xffu);
  }
  return out;
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_float32_bytes_to_float_array(
    b_lean_obj_arg BytesObj) {
  lean_object* bytesObj = (lean_object*)BytesObj;
  size_t bytes = lean_sarray_size(bytesObj);
  if (bytes % sizeof(float) != 0) {
    lean_internal_panic("torchlean: float32 checkpoint payload is not a multiple of four bytes");
  }
  size_t n = bytes / sizeof(float);
  lean_object* out = lean_mk_empty_float_array(lean_box(n));
  lean_sarray_set_size(out, n);
  const uint8_t* src = (const uint8_t*)lean_sarray_cptr(bytesObj);
  double* dst = lean_float_array_cptr(out);
  for (size_t i = 0; i < n; ++i) {
    uint32_t bits = (uint32_t)src[4 * i] |
                    ((uint32_t)src[4 * i + 1] << 8) |
                    ((uint32_t)src[4 * i + 2] << 16) |
                    ((uint32_t)src[4 * i + 3] << 24);
    float value;
    memcpy(&value, &bits, sizeof(bits));
    dst[i] = (double)value;
  }
  return out;
}

#define TORCHLEAN_DEFINE_UNARY_BUFFER_EXPORT(EXPORT_NAME, KERNEL, LABEL)                         \
  extern "C" LEAN_EXPORT lean_obj_res EXPORT_NAME(b_lean_obj_arg BObj) {                        \
    torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);                                \
    torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(b->size);                           \
    if (b->size == 0) {                                                                          \
      return torchlean_cuda_buffer_box(out);                                                     \
    }                                                                                            \
    dim3 blocks = torchlean_blocks_for(b->size);                                                 \
    dim3 threads = dim3(kBlockSize);                                                             \
    KERNEL<<<blocks, threads>>>(b->data, out->data, b->size);                                    \
    checkCuda(cudaGetLastError(), "cuda " LABEL " kernel launch failed");                      \
    return torchlean_cuda_buffer_box(out);                                                       \
  }

#define TORCHLEAN_DEFINE_BINARY_BUFFER_EXPORT(EXPORT_NAME, KERNEL, LABEL)                        \
  extern "C" LEAN_EXPORT lean_obj_res EXPORT_NAME(b_lean_obj_arg AObj, b_lean_obj_arg BObj) {   \
    torchlean_cuda_buffer* a = torchlean_cuda_buffer_unbox(AObj);                                \
    torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);                                \
    torchlean_cuda_require_same_size2(a, b, #EXPORT_NAME);                                      \
    torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(a->size);                           \
    if (a->size == 0) {                                                                          \
      return torchlean_cuda_buffer_box(out);                                                     \
    }                                                                                            \
    dim3 blocks = torchlean_blocks_for(a->size);                                                 \
    dim3 threads = dim3(kBlockSize);                                                             \
    KERNEL<<<blocks, threads>>>(a->data, b->data, out->data, a->size);                           \
    checkCuda(cudaGetLastError(), "cuda " LABEL " kernel launch failed");                      \
    return torchlean_cuda_buffer_box(out);                                                       \
  }

#define TORCHLEAN_DEFINE_UNARY_SCALAR_BUFFER_EXPORT(EXPORT_NAME, KERNEL, LABEL)                  \
  extern "C" LEAN_EXPORT lean_obj_res EXPORT_NAME(b_lean_obj_arg BObj, double c) {               \
    torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);                                \
    torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(b->size);                           \
    if (b->size == 0) {                                                                          \
      return torchlean_cuda_buffer_box(out);                                                     \
    }                                                                                            \
    dim3 blocks = torchlean_blocks_for(b->size);                                                 \
    dim3 threads = dim3(kBlockSize);                                                             \
    /* Attribute the launch check below to this kernel, not to a stale error from an earlier op. */\
    torchlean_cuda_clear_pending_error();                                                        \
    KERNEL<<<blocks, threads>>>(b->data, out->data, b->size, (float)c);                          \
    torchlean_cuda_check_launch("cuda " LABEL " kernel launch failed");                          \
    return torchlean_cuda_buffer_box(out);                                                       \
  }

#define TORCHLEAN_DEFINE_BINARY_SCALAR_BUFFER_EXPORT(EXPORT_NAME, KERNEL, LABEL)                 \
  extern "C" LEAN_EXPORT lean_obj_res EXPORT_NAME(b_lean_obj_arg AObj, b_lean_obj_arg BObj,     \
                                                  double c) {                                    \
    torchlean_cuda_buffer* a = torchlean_cuda_buffer_unbox(AObj);                                \
    torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);                                \
    torchlean_cuda_require_same_size2(a, b, #EXPORT_NAME);                                      \
    torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(a->size);                           \
    if (a->size == 0) {                                                                          \
      return torchlean_cuda_buffer_box(out);                                                     \
    }                                                                                            \
    dim3 blocks = torchlean_blocks_for(a->size);                                                 \
    dim3 threads = dim3(kBlockSize);                                                             \
    KERNEL<<<blocks, threads>>>(a->data, b->data, out->data, a->size, (float)c);                 \
    checkCuda(cudaGetLastError(), "cuda " LABEL " kernel launch failed");                      \
    return torchlean_cuda_buffer_box(out);                                                       \
  }

TORCHLEAN_DEFINE_UNARY_BUFFER_EXPORT(torchlean_cuda_buffer_abs, torchlean_abs_f32, "abs")

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_abs_bwd(b_lean_obj_arg XObj,
                                                                 b_lean_obj_arg GObj) {
  torchlean_cuda_buffer* x = torchlean_cuda_buffer_unbox(XObj);
  torchlean_cuda_buffer* g = torchlean_cuda_buffer_unbox(GObj);
  torchlean_cuda_require_same_size2(x, g, "torchlean_cuda_buffer_abs_bwd");
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

TORCHLEAN_DEFINE_UNARY_BUFFER_EXPORT(torchlean_cuda_buffer_sqrt, torchlean_sqrt_f32, "sqrt")

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_sqrt_bwd(b_lean_obj_arg XObj,
                                                                  b_lean_obj_arg GObj) {
  torchlean_cuda_buffer* x = torchlean_cuda_buffer_unbox(XObj);
  torchlean_cuda_buffer* g = torchlean_cuda_buffer_unbox(GObj);
  torchlean_cuda_require_same_size2(x, g, "torchlean_cuda_buffer_sqrt_bwd");
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

TORCHLEAN_DEFINE_UNARY_BUFFER_EXPORT(torchlean_cuda_buffer_exp, torchlean_exp_f32, "exp")

TORCHLEAN_DEFINE_UNARY_BUFFER_EXPORT(torchlean_cuda_buffer_log, torchlean_log_f32, "log")

TORCHLEAN_DEFINE_UNARY_BUFFER_EXPORT(torchlean_cuda_buffer_inv, torchlean_inv_f32, "inv")

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
  torchlean_cuda_require_same_size2(x, g, "torchlean_cuda_buffer_clamp_bwd");
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

TORCHLEAN_DEFINE_BINARY_BUFFER_EXPORT(torchlean_cuda_buffer_max, torchlean_max_f32, "max")

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_max_bwd(b_lean_obj_arg AObj,
                                                                 b_lean_obj_arg BObj,
                                                                 b_lean_obj_arg GObj) {
  torchlean_cuda_buffer* a = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* g = torchlean_cuda_buffer_unbox(GObj);
  torchlean_cuda_require_same_size3(a, b, g, "torchlean_cuda_buffer_max_bwd");

  torchlean_cuda_buffer* dA = torchlean_cuda_buffer_alloc(a->size);
  torchlean_cuda_buffer* dB = torchlean_cuda_buffer_alloc(a->size);
  if (a->size == 0) {
    return torchlean_cuda_box_buffer_pair(dA, dB);
  }

  dim3 blocks = torchlean_blocks_for(a->size);
  dim3 threads = dim3(kBlockSize);
  torchlean_max_bwd_f32<<<blocks, threads>>>(a->data, b->data, g->data, dA->data, dB->data, a->size);
  checkCuda(cudaGetLastError(), "cuda max_bwd kernel launch failed");

  return torchlean_cuda_box_buffer_pair(dA, dB);
}

TORCHLEAN_DEFINE_BINARY_BUFFER_EXPORT(torchlean_cuda_buffer_min, torchlean_min_f32, "min")

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_min_bwd(b_lean_obj_arg AObj,
                                                                 b_lean_obj_arg BObj,
                                                                 b_lean_obj_arg GObj) {
  torchlean_cuda_buffer* a = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* g = torchlean_cuda_buffer_unbox(GObj);
  torchlean_cuda_require_same_size3(a, b, g, "torchlean_cuda_buffer_min_bwd");

  torchlean_cuda_buffer* dA = torchlean_cuda_buffer_alloc(a->size);
  torchlean_cuda_buffer* dB = torchlean_cuda_buffer_alloc(a->size);
  if (a->size == 0) {
    return torchlean_cuda_box_buffer_pair(dA, dB);
  }

  dim3 blocks = torchlean_blocks_for(a->size);
  dim3 threads = dim3(kBlockSize);
  torchlean_min_bwd_f32<<<blocks, threads>>>(a->data, b->data, g->data, dA->data, dB->data, a->size);
  checkCuda(cudaGetLastError(), "cuda min_bwd kernel launch failed");

  return torchlean_cuda_box_buffer_pair(dA, dB);
}

TORCHLEAN_DEFINE_BINARY_BUFFER_EXPORT(torchlean_cuda_buffer_div, torchlean_div_f32, "div")

TORCHLEAN_DEFINE_UNARY_BUFFER_EXPORT(torchlean_cuda_buffer_relu, torchlean_relu_f32, "relu")

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_relu_bwd(b_lean_obj_arg XObj,
                                                                  b_lean_obj_arg GObj) {
  torchlean_cuda_buffer* x = torchlean_cuda_buffer_unbox(XObj);
  torchlean_cuda_buffer* g = torchlean_cuda_buffer_unbox(GObj);
  torchlean_cuda_require_same_size2(x, g, "torchlean_cuda_buffer_relu_bwd");
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

TORCHLEAN_DEFINE_UNARY_BUFFER_EXPORT(torchlean_cuda_buffer_gelu, torchlean_gelu_f32, "gelu")

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_gelu_bwd(b_lean_obj_arg XObj,
                                                                  b_lean_obj_arg GObj) {
  torchlean_cuda_buffer* x = torchlean_cuda_buffer_unbox(XObj);
  torchlean_cuda_buffer* g = torchlean_cuda_buffer_unbox(GObj);
  torchlean_cuda_require_same_size2(x, g, "torchlean_cuda_buffer_gelu_bwd");
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(x->size);
  if (x->size == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  dim3 blocks = torchlean_blocks_for(x->size);
  dim3 threads = dim3(kBlockSize);
  torchlean_gelu_bwd_f32<<<blocks, threads>>>(x->data, g->data, out->data, x->size);
  checkCuda(cudaGetLastError(), "cuda gelu_bwd kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

TORCHLEAN_DEFINE_BINARY_BUFFER_EXPORT(torchlean_cuda_buffer_add, torchlean_add_f32, "add")

TORCHLEAN_DEFINE_BINARY_BUFFER_EXPORT(torchlean_cuda_buffer_sub, torchlean_sub_f32, "sub")

TORCHLEAN_DEFINE_BINARY_BUFFER_EXPORT(torchlean_cuda_buffer_mul, torchlean_mul_f32, "mul")

TORCHLEAN_DEFINE_UNARY_SCALAR_BUFFER_EXPORT(torchlean_cuda_buffer_scale, torchlean_scale_f32,
                                            "scale")

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_copy_and_release(
    b_lean_obj_arg BObj) {
  lean_obj_res out = torchlean_cuda_buffer_scale(BObj, 1.0);
  (void)torchlean_cuda_buffer_release_data(torchlean_cuda_buffer_unbox(BObj));
  return out;
}

TORCHLEAN_DEFINE_BINARY_SCALAR_BUFFER_EXPORT(torchlean_cuda_buffer_axpy, torchlean_axpy_f32,
                                             "axpy")

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_adam_step(
    b_lean_obj_arg ParametersObj,
    b_lean_obj_arg GradientObj,
    b_lean_obj_arg FirstMomentObj,
    b_lean_obj_arg SecondMomentObj,
    double beta1,
    double oneMinusBeta1,
    double beta2,
    double oneMinusBeta2,
    double firstMomentCorrection,
    double secondMomentCorrection,
    double epsilon,
    double decay,
    double updateScale) {
  torchlean_cuda_buffer* parameters = torchlean_cuda_buffer_unbox(ParametersObj);
  torchlean_cuda_buffer* gradient = torchlean_cuda_buffer_unbox(GradientObj);
  torchlean_cuda_buffer* first_moment = torchlean_cuda_buffer_unbox(FirstMomentObj);
  torchlean_cuda_buffer* second_moment = torchlean_cuda_buffer_unbox(SecondMomentObj);
  torchlean_cuda_require_same_size3(
      parameters, gradient, first_moment, "torchlean_cuda_buffer_adam_step");
  torchlean_cuda_require_same_size2(
      parameters, second_moment, "torchlean_cuda_buffer_adam_step");

  torchlean_cuda_buffer* updated_parameters = torchlean_cuda_buffer_alloc(parameters->size);
  torchlean_cuda_buffer* updated_first_moment = torchlean_cuda_buffer_alloc(parameters->size);
  torchlean_cuda_buffer* updated_second_moment = torchlean_cuda_buffer_alloc(parameters->size);
  if (parameters->size != 0) {
    dim3 blocks = torchlean_blocks_for(parameters->size);
    dim3 threads = dim3(kBlockSize);
    torchlean_adam_step_f32<<<blocks, threads>>>(
        parameters->data,
        gradient->data,
        first_moment->data,
        second_moment->data,
        updated_parameters->data,
        updated_first_moment->data,
        updated_second_moment->data,
        parameters->size,
        (float)beta1,
        (float)oneMinusBeta1,
        (float)beta2,
        (float)oneMinusBeta2,
        (float)firstMomentCorrection,
        (float)secondMomentCorrection,
        (float)epsilon,
        (float)decay,
        (float)updateScale);
    checkCuda(cudaGetLastError(), "cuda adam_step kernel launch failed");
  }
  return torchlean_cuda_box_three_buffers(
      updated_parameters, updated_first_moment, updated_second_moment);
}

TORCHLEAN_DEFINE_BINARY_SCALAR_BUFFER_EXPORT(torchlean_cuda_buffer_scaled_prod_exp,
                                             torchlean_scaled_prod_exp_f32, "scaled_prod_exp")

#undef TORCHLEAN_DEFINE_BINARY_SCALAR_BUFFER_EXPORT
#undef TORCHLEAN_DEFINE_UNARY_SCALAR_BUFFER_EXPORT
#undef TORCHLEAN_DEFINE_BINARY_BUFFER_EXPORT
#undef TORCHLEAN_DEFINE_UNARY_BUFFER_EXPORT

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
    torchlean_cuda_clear_pending_error();
    torchlean_reduce_sum_f32<<<blocks, threads>>>(b->data, out->data, b->size);
    torchlean_cuda_check_launch("cuda reduceMean reduce kernel launch failed");
  }

  float scale = 1.0f / (float)b->size;
  torchlean_cuda_clear_pending_error();
  torchlean_scale1_f32<<<dim3(1), dim3(1)>>>(out->data, scale);
  torchlean_cuda_check_launch("cuda reduceMean scale kernel launch failed");

  return torchlean_cuda_buffer_box(out);
}
