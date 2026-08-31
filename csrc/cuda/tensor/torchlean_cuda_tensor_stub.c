#include <lean/lean.h>
#include <lean/mimalloc.h>

#include "torchlean_cuda_buffer.h"
#include "torchlean_cuda_deterministic_reductions_env.h"
#include "torchlean_cuda_rng_common.h"
#include "torchlean_size_common.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Host-memory version of the buffer runtime symbols.
// This keeps default builds CUDA-free while preserving the same edge-case behavior.

// Keep the deterministic-reductions flag for API parity, and initialize it lazily from the
// environment.
static uint32_t g_torchlean_deterministic_reductions = 0u;
static uint32_t g_torchlean_deterministic_reductions_inited = 0u;
static uint64_t g_torchlean_cuda_live_bytes = 0u;
static uint64_t g_torchlean_cuda_peak_bytes = 0u;
static uint64_t g_torchlean_cuda_alloc_count = 0u;
static uint64_t g_torchlean_cuda_free_count = 0u;
static uint64_t g_torchlean_cuda_wrapper_live_count = 0u;
static uint64_t g_torchlean_cuda_wrapper_peak_count = 0u;
static uint64_t g_torchlean_cuda_wrapper_alloc_count = 0u;
static uint64_t g_torchlean_cuda_wrapper_finalize_count = 0u;

// 0 = CPU stub, 1 = native CUDA with a visible device, 2 = native CUDA without one.
LEAN_EXPORT uint32_t torchlean_cuda_runtime_status(uint32_t token) {
  (void)token;
  return 0u;
}

static void torchlean_cuda_note_alloc(size_t n) {
  const uint64_t bytes = torchlean_float_bytes_for(n);
  g_torchlean_cuda_alloc_count++;
  g_torchlean_cuda_live_bytes += bytes;
  if (g_torchlean_cuda_live_bytes > g_torchlean_cuda_peak_bytes) {
    g_torchlean_cuda_peak_bytes = g_torchlean_cuda_live_bytes;
  }
}

static void torchlean_cuda_note_free(size_t n) {
  const uint64_t bytes = torchlean_float_bytes_for(n);
  g_torchlean_cuda_free_count++;
  g_torchlean_cuda_live_bytes =
      g_torchlean_cuda_live_bytes > bytes ? g_torchlean_cuda_live_bytes - bytes : 0u;
}

LEAN_EXPORT void torchlean_cuda_set_deterministic_reductions(uint32_t on) {
  g_torchlean_deterministic_reductions = on ? 1u : 0u;
  g_torchlean_deterministic_reductions_inited = 1u;
}

LEAN_EXPORT uint32_t torchlean_cuda_get_deterministic_reductions() {
  if (!g_torchlean_deterministic_reductions_inited) {
    g_torchlean_deterministic_reductions = torchlean_read_deterministic_reductions_env();
    g_torchlean_deterministic_reductions_inited = 1u;
  }
  return g_torchlean_deterministic_reductions;
}

LEAN_EXPORT uint32_t torchlean_cuda_get_deterministic_reductions_u(uint32_t u) {
  (void)u;
  return torchlean_cuda_get_deterministic_reductions();
}

LEAN_EXPORT uint32_t torchlean_cuda_set_deterministic_reductions_checked(uint32_t on) {
  torchlean_cuda_set_deterministic_reductions(on);
  return torchlean_cuda_get_deterministic_reductions();
}

static bool torchlean_cuda_buffer_release_data(torchlean_cuda_buffer* b) {
  if (!b || !b->data) {
    return false;
  }
  free(b->data);
  torchlean_cuda_note_free(b->size);
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
  g_torchlean_cuda_wrapper_finalize_count++;
  if (g_torchlean_cuda_wrapper_live_count > 0u) {
    g_torchlean_cuda_wrapper_live_count--;
  }
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

torchlean_cuda_buffer* torchlean_cuda_buffer_unbox(b_lean_obj_arg obj) {
  lean_object* o = (lean_object*)obj;
  if (!lean_is_external(o)) {
    lean_internal_panic("torchlean_cuda_buffer_stub: expected external object");
  }
  return (torchlean_cuda_buffer*)lean_get_external_data(o);
}

lean_obj_res torchlean_cuda_buffer_box(torchlean_cuda_buffer* b) {
  lean_obj_res boxed = lean_alloc_external(torchlean_cuda_buffer_get_class(), b);
  g_torchlean_cuda_wrapper_alloc_count++;
  g_torchlean_cuda_wrapper_live_count++;
  if (g_torchlean_cuda_wrapper_live_count > g_torchlean_cuda_wrapper_peak_count) {
    g_torchlean_cuda_wrapper_peak_count = g_torchlean_cuda_wrapper_live_count;
  }
  return boxed;
}

void torchlean_cuda_buffer_drop_unboxed(torchlean_cuda_buffer* b) {
  if (!b) {
    return;
  }
  (void)torchlean_cuda_buffer_release_data(b);
  free(b);
}

torchlean_cuda_buffer* torchlean_cuda_buffer_alloc(size_t n) {
  torchlean_cuda_buffer* b = (torchlean_cuda_buffer*)malloc(sizeof(torchlean_cuda_buffer));
  if (!b) {
    lean_internal_panic_out_of_memory();
  }
  b->size = n;
  b->data = NULL;
  if (n > 0) {
    const size_t bytes =
        checked_bytes_size(n, sizeof(float), "torchlean_cuda_buffer_alloc_stub: byte size overflow");
    b->data = (float*)malloc(bytes);
    if (!b->data) {
      free(b);
      lean_internal_panic_out_of_memory();
    }
    torchlean_cuda_note_alloc(n);
  }
  return b;
}

LEAN_EXPORT uint64_t torchlean_cuda_allocator_live_bytes(uint32_t u) {
  (void)u;
  return g_torchlean_cuda_live_bytes;
}

LEAN_EXPORT uint64_t torchlean_cuda_allocator_peak_bytes(uint32_t u) {
  (void)u;
  return g_torchlean_cuda_peak_bytes;
}

LEAN_EXPORT uint64_t torchlean_cuda_allocator_alloc_count(uint32_t u) {
  (void)u;
  return g_torchlean_cuda_alloc_count;
}

LEAN_EXPORT uint64_t torchlean_cuda_allocator_free_count(uint32_t u) {
  (void)u;
  return g_torchlean_cuda_free_count;
}

LEAN_EXPORT uint64_t torchlean_cuda_wrapper_live_count(uint32_t u) {
  (void)u;
  return g_torchlean_cuda_wrapper_live_count;
}

LEAN_EXPORT uint64_t torchlean_cuda_wrapper_peak_count(uint32_t u) {
  (void)u;
  return g_torchlean_cuda_wrapper_peak_count;
}

LEAN_EXPORT uint64_t torchlean_cuda_wrapper_alloc_count(uint32_t u) {
  (void)u;
  return g_torchlean_cuda_wrapper_alloc_count;
}

LEAN_EXPORT uint64_t torchlean_cuda_wrapper_finalize_count(uint32_t u) {
  (void)u;
  return g_torchlean_cuda_wrapper_finalize_count;
}

LEAN_EXPORT uint64_t torchlean_cuda_allocator_device_free_bytes(uint32_t u) {
  (void)u;
  return 0u;
}

LEAN_EXPORT uint64_t torchlean_cuda_allocator_device_total_bytes(uint32_t u) {
  (void)u;
  return 0u;
}

// The CPU stub frees dropped buffers immediately and keeps no reuse cache, so there is nothing to
// cap and no cached bytes to report.
LEAN_EXPORT uint64_t torchlean_cuda_allocator_cache_bytes(uint32_t u) {
  (void)u;
  return 0u;
}

LEAN_EXPORT uint64_t torchlean_cuda_allocator_cache_cap_bytes(uint32_t u) {
  (void)u;
  return 0u;
}

LEAN_EXPORT uint32_t torchlean_cuda_buffer_size(b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  if (b->size > 0xFFFFFFFFULL) {
    lean_internal_panic("torchlean_cuda_buffer_size_stub: buffer too large for UInt32");
  }
  return (uint32_t)b->size;
}

LEAN_EXPORT uint32_t torchlean_cuda_buffer_size_with_token(
    b_lean_obj_arg BObj, uint32_t token) {
  (void)token;
  return torchlean_cuda_buffer_size(BObj);
}

LEAN_EXPORT uint32_t torchlean_cuda_buffer_release(b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  // Explicit release is an eager-runtime lifetime hint. We mark the handle as empty so accidental
  // reuse fails by size checks instead of touching freed memory.
  return torchlean_cuda_buffer_release_data(b) ? 1 : 0;
}

LEAN_EXPORT uint32_t torchlean_cuda_buffer_release_with_token(
    b_lean_obj_arg BObj, uint32_t token) {
  (void)token;
  return torchlean_cuda_buffer_release(BObj);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_release_then(
    b_lean_obj_arg scratchObj, b_lean_obj_arg keepObj) {
  (void)torchlean_cuda_buffer_release(scratchObj);
  lean_inc((lean_object*)keepObj);
  return (lean_object*)keepObj;
}

LEAN_EXPORT uint32_t torchlean_runtime_collect_allocator(uint32_t force) {
  const bool force_collect = force != 0;
  mi_collect(force_collect);
  mi_heap_collect(mi_heap_get_default(), force_collect);
  mi_collect_reduce(0);
  return 1;
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_zeros(uint32_t n) {
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc((size_t)n);
  for (size_t i = 0; i < (size_t)n; ++i) {
    out->data[i] = 0.0f;
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_full(uint32_t n, double v) {
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc((size_t)n);
  float fv = (float)v;
  for (size_t i = 0; i < (size_t)n; ++i) {
    out->data[i] = fv;
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_full_with_token(
    uint32_t n, double v, uint32_t token) {
  (void)token;
  return torchlean_cuda_buffer_full(n, v);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_rand_uniform(uint32_t n, uint64_t key) {
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc((size_t)n);
  const double denom = 4294967296.0;  // 2^32
  for (size_t i = 0; i < (size_t)n; ++i) {
    uint64_t z = torchlean_splitmix64(key + (uint64_t)i);
    // Match the CUDA runtime and pure Lean helper: reduce modulo 2^32 via the low 32 bits.
    uint32_t u = (uint32_t)z;
    out->data[i] = (float)(((double)u) / denom);
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_rand_normal(
    uint32_t n, double mean, double std, uint64_t key) {
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc((size_t)n);
  for (size_t i = 0; i < (size_t)n; ++i) {
    uint32_t r1 = (uint32_t)torchlean_splitmix64(key + (uint64_t)(2 * i));
    uint32_t r2 = (uint32_t)torchlean_splitmix64(key + (uint64_t)(2 * i + 1));
    double u1 = ((double)r1 + 1.0) / 4294967297.0;
    double u2 = ((double)r2) / 4294967296.0;
    double z = sqrt(-2.0 * log(u1)) * cos(6.2831853071795864769 * u2);
    out->data[i] = (float)(mean + std * z);
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_bernoulli_mask(uint32_t n, double keepProb, uint64_t key) {
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc((size_t)n);
  const double denom = 4294967296.0;  // 2^32
  float kp = (float)keepProb;
  for (size_t i = 0; i < (size_t)n; ++i) {
    uint64_t z = torchlean_splitmix64(key + (uint64_t)i);
    uint32_t u = (uint32_t)z;
    float u01 = (float)(((double)u) / denom);
    out->data[i] = (kp > u01) ? 1.0f : 0.0f;
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_of_float_array(b_lean_obj_arg AObj) {
  lean_object* A = (lean_object*)AObj;
  size_t n = lean_sarray_size(A);
  const double* src = lean_float_array_cptr(A);

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(n);
  for (size_t i = 0; i < n; ++i) {
    out->data[i] = (float)src[i];
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_of_float_array_with_token(
    b_lean_obj_arg AObj, uint32_t token) {
  (void)token;
  return torchlean_cuda_buffer_of_float_array(AObj);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_to_float_array(b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  size_t n = b->size;

  lean_object* out = lean_mk_empty_float_array(lean_box(n));
  lean_sarray_set_size(out, n);
  double* dst = lean_float_array_cptr(out);
  for (size_t i = 0; i < n; ++i) {
    dst[i] = (double)b->data[i];
  }
  return out;
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_to_float_array_io(b_lean_obj_arg BObj) {
  return lean_io_result_mk_ok(torchlean_cuda_buffer_to_float_array(BObj));
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_to_float32_bytes_io(b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  size_t bytes = torchlean_float_bytes_for(b->size);
  lean_object* out = lean_alloc_sarray(1, bytes, bytes);
  uint8_t* dst = (uint8_t*)lean_sarray_cptr(out);
  for (size_t i = 0; i < b->size; ++i) {
    uint32_t bits;
    memcpy(&bits, &b->data[i], sizeof(bits));
    dst[4 * i] = (uint8_t)(bits & 0xffu);
    dst[4 * i + 1] = (uint8_t)((bits >> 8) & 0xffu);
    dst[4 * i + 2] = (uint8_t)((bits >> 16) & 0xffu);
    dst[4 * i + 3] = (uint8_t)((bits >> 24) & 0xffu);
  }
  return lean_io_result_mk_ok(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_of_float32_bytes_io(b_lean_obj_arg BytesObj) {
  lean_object* bytesObj = (lean_object*)BytesObj;
  size_t bytes = lean_sarray_size(bytesObj);
  if (bytes % sizeof(float) != 0) {
    lean_internal_panic("torchlean: float32 checkpoint payload is not a multiple of four bytes");
  }
  size_t n = bytes / sizeof(float);
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(n);
  const uint8_t* src = (const uint8_t*)lean_sarray_cptr(bytesObj);
  for (size_t i = 0; i < n; ++i) {
    uint32_t bits = (uint32_t)src[4 * i] |
                    ((uint32_t)src[4 * i + 1] << 8) |
                    ((uint32_t)src[4 * i + 2] << 16) |
                    ((uint32_t)src[4 * i + 3] << 24);
    memcpy(&out->data[i], &bits, sizeof(bits));
  }
  return lean_io_result_mk_ok(torchlean_cuda_buffer_box(out));
}

LEAN_EXPORT lean_obj_res torchlean_float_array_to_float32_bytes(b_lean_obj_arg AObj) {
  lean_object* valuesObj = (lean_object*)AObj;
  size_t n = lean_sarray_size(valuesObj);
  size_t bytes = torchlean_float_bytes_for(n);
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

LEAN_EXPORT lean_obj_res torchlean_float32_bytes_to_float_array(b_lean_obj_arg BytesObj) {
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

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_abs(b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(b->size);
  for (size_t i = 0; i < b->size; ++i) {
    out->data[i] = fabsf(b->data[i]);
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_abs_bwd(b_lean_obj_arg XObj, b_lean_obj_arg GObj) {
  torchlean_cuda_buffer* x = torchlean_cuda_buffer_unbox(XObj);
  torchlean_cuda_buffer* g = torchlean_cuda_buffer_unbox(GObj);
  torchlean_cuda_require_same_size2(x, g, "torchlean_cuda_buffer_abs_bwd_stub");
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(x->size);
  for (size_t i = 0; i < x->size; ++i) {
    float v = x->data[i];
    float s = (v > 0.0f) ? 1.0f : ((v < 0.0f) ? -1.0f : 0.0f);
    out->data[i] = s * g->data[i];
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_sqrt(b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(b->size);
  for (size_t i = 0; i < b->size; ++i) {
    out->data[i] = sqrtf(b->data[i]);
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_sqrt_bwd(b_lean_obj_arg XObj, b_lean_obj_arg GObj) {
  torchlean_cuda_buffer* x = torchlean_cuda_buffer_unbox(XObj);
  torchlean_cuda_buffer* g = torchlean_cuda_buffer_unbox(GObj);
  torchlean_cuda_require_same_size2(x, g, "torchlean_cuda_buffer_sqrt_bwd_stub");
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(x->size);
  for (size_t i = 0; i < x->size; ++i) {
    float v = x->data[i];
    if (v > 0.0f) {
      out->data[i] = g->data[i] * (1.0f / (2.0f * sqrtf(v)));
    } else {
      out->data[i] = 0.0f;
    }
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_exp(b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(b->size);
  for (size_t i = 0; i < b->size; ++i) {
    out->data[i] = expf(b->data[i]);
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_log(b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(b->size);
  for (size_t i = 0; i < b->size; ++i) {
    out->data[i] = logf(b->data[i]);
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_inv(b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(b->size);
  for (size_t i = 0; i < b->size; ++i) {
    out->data[i] = 1.0f / b->data[i];
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_clamp(b_lean_obj_arg BObj, double lo, double hi) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(b->size);
  float flo = (float)lo;
  float fhi = (float)hi;
  for (size_t i = 0; i < b->size; ++i) {
    float x = b->data[i];
    if (x < flo) {
      x = flo;
    } else if (x > fhi) {
      x = fhi;
    }
    out->data[i] = x;
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_clamp_bwd(b_lean_obj_arg XObj, b_lean_obj_arg GObj,
                                                        double lo, double hi) {
  torchlean_cuda_buffer* x = torchlean_cuda_buffer_unbox(XObj);
  torchlean_cuda_buffer* g = torchlean_cuda_buffer_unbox(GObj);
  torchlean_cuda_require_same_size2(x, g, "torchlean_cuda_buffer_clamp_bwd_stub");
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(x->size);
  float flo = (float)lo;
  float fhi = (float)hi;
  for (size_t i = 0; i < x->size; ++i) {
    float v = x->data[i];
    out->data[i] = (v > flo && v < fhi) ? g->data[i] : 0.0f;
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_max(b_lean_obj_arg AObj, b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* a = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_require_same_size2(a, b, "torchlean_cuda_buffer_max_stub");
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(a->size);
  for (size_t i = 0; i < a->size; ++i) {
    out->data[i] = fmaxf(a->data[i], b->data[i]);
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_max_bwd(b_lean_obj_arg AObj, b_lean_obj_arg BObj,
                                                      b_lean_obj_arg GObj) {
  torchlean_cuda_buffer* a = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* g = torchlean_cuda_buffer_unbox(GObj);
  torchlean_cuda_require_same_size3(a, b, g, "torchlean_cuda_buffer_max_bwd_stub");
  torchlean_cuda_buffer* dA = torchlean_cuda_buffer_alloc(a->size);
  torchlean_cuda_buffer* dB = torchlean_cuda_buffer_alloc(a->size);
  for (size_t i = 0; i < a->size; ++i) {
    float av = a->data[i];
    float bv = b->data[i];
    float gg = g->data[i];
    if (av > bv) {
      dA->data[i] = gg;
      dB->data[i] = 0.0f;
    } else if (bv > av) {
      dA->data[i] = 0.0f;
      dB->data[i] = gg;
    } else {
      dA->data[i] = 0.5f * gg;
      dB->data[i] = 0.5f * gg;
    }
  }
  return torchlean_cuda_box_buffer_pair(dA, dB);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_min(b_lean_obj_arg AObj, b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* a = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_require_same_size2(a, b, "torchlean_cuda_buffer_min_stub");
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(a->size);
  for (size_t i = 0; i < a->size; ++i) {
    out->data[i] = fminf(a->data[i], b->data[i]);
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_min_bwd(b_lean_obj_arg AObj, b_lean_obj_arg BObj,
                                                      b_lean_obj_arg GObj) {
  torchlean_cuda_buffer* a = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* g = torchlean_cuda_buffer_unbox(GObj);
  torchlean_cuda_require_same_size3(a, b, g, "torchlean_cuda_buffer_min_bwd_stub");
  torchlean_cuda_buffer* dA = torchlean_cuda_buffer_alloc(a->size);
  torchlean_cuda_buffer* dB = torchlean_cuda_buffer_alloc(a->size);
  for (size_t i = 0; i < a->size; ++i) {
    float av = a->data[i];
    float bv = b->data[i];
    float gg = g->data[i];
    if (bv > av) {
      dA->data[i] = gg;
      dB->data[i] = 0.0f;
    } else if (av > bv) {
      dA->data[i] = 0.0f;
      dB->data[i] = gg;
    } else {
      dA->data[i] = 0.5f * gg;
      dB->data[i] = 0.5f * gg;
    }
  }
  return torchlean_cuda_box_buffer_pair(dA, dB);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_div(b_lean_obj_arg AObj, b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* a = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_require_same_size2(a, b, "torchlean_cuda_buffer_div_stub");
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(a->size);
  for (size_t i = 0; i < a->size; ++i) {
    out->data[i] = a->data[i] / b->data[i];
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_relu(b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(b->size);
  for (size_t i = 0; i < b->size; ++i) {
    float v = b->data[i];
    out->data[i] = (v > 0.0f) ? v : 0.0f;
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_relu_bwd(b_lean_obj_arg XObj, b_lean_obj_arg GObj) {
  torchlean_cuda_buffer* x = torchlean_cuda_buffer_unbox(XObj);
  torchlean_cuda_buffer* g = torchlean_cuda_buffer_unbox(GObj);
  torchlean_cuda_require_same_size2(x, g, "torchlean_cuda_buffer_relu_bwd_stub");
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(x->size);
  for (size_t i = 0; i < x->size; ++i) {
    float v = x->data[i];
    out->data[i] = (v > 0.0f) ? g->data[i] : 0.0f;
  }
  return torchlean_cuda_buffer_box(out);
}

static float torchlean_gelu_f32_value(float x) {
  const float coeff = 0.044715f;
  const float sqrtTwoOverPi = 0.79788456080286535588f;
  const float cubic = ((coeff * x) * x) * x;
  const float tanhTerm = tanhf(sqrtTwoOverPi * (x + cubic));
  return (x * (1.0f + tanhTerm)) / 2.0f;
}

static float torchlean_gelu_f32_deriv(float x) {
  const float coeff = 0.044715f;
  const float sqrtTwoOverPi = 0.79788456080286535588f;
  const float cubic = ((coeff * x) * x) * x;
  const float tanhTerm = tanhf(sqrtTwoOverPi * (x + cubic));
  const float sechTerm = 1.0f - tanhTerm * tanhTerm;
  const float innerDeriv = sqrtTwoOverPi * (1.0f + ((3.0f * coeff) * x) * x);
  return (1.0f + tanhTerm + (x * sechTerm) * innerDeriv) / 2.0f;
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_gelu(b_lean_obj_arg XObj) {
  torchlean_cuda_buffer* x = torchlean_cuda_buffer_unbox(XObj);
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(x->size);
  for (size_t i = 0; i < x->size; ++i) {
    out->data[i] = torchlean_gelu_f32_value(x->data[i]);
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_gelu_bwd(b_lean_obj_arg XObj,
                                                        b_lean_obj_arg GObj) {
  torchlean_cuda_buffer* x = torchlean_cuda_buffer_unbox(XObj);
  torchlean_cuda_buffer* g = torchlean_cuda_buffer_unbox(GObj);
  torchlean_cuda_require_same_size2(x, g, "torchlean_cuda_buffer_gelu_bwd_stub");
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(x->size);
  for (size_t i = 0; i < x->size; ++i) {
    out->data[i] = torchlean_gelu_f32_deriv(x->data[i]) * g->data[i];
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_add(b_lean_obj_arg AObj, b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* a = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_require_same_size2(a, b, "torchlean_cuda_buffer_add_stub");
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(a->size);
  for (size_t i = 0; i < a->size; ++i) {
    out->data[i] = a->data[i] + b->data[i];
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_sub(b_lean_obj_arg AObj, b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* a = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_require_same_size2(a, b, "torchlean_cuda_buffer_sub_stub");
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(a->size);
  for (size_t i = 0; i < a->size; ++i) {
    out->data[i] = a->data[i] - b->data[i];
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_mul(b_lean_obj_arg AObj, b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* a = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_require_same_size2(a, b, "torchlean_cuda_buffer_mul_stub");
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(a->size);
  for (size_t i = 0; i < a->size; ++i) {
    out->data[i] = a->data[i] * b->data[i];
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_scale(b_lean_obj_arg BObj, double c) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(b->size);
  float fc = (float)c;
  for (size_t i = 0; i < b->size; ++i) {
    out->data[i] = b->data[i] * fc;
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_copy_and_release(b_lean_obj_arg BObj) {
  lean_obj_res out = torchlean_cuda_buffer_scale(BObj, 1.0);
  (void)torchlean_cuda_buffer_release_data(torchlean_cuda_buffer_unbox(BObj));
  return out;
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_axpy(b_lean_obj_arg AObj, b_lean_obj_arg BObj,
                                                   double c) {
  torchlean_cuda_buffer* a = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_require_same_size2(a, b, "torchlean_cuda_buffer_axpy_stub");
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(a->size);
  float fc = (float)c;
  for (size_t i = 0; i < a->size; ++i) {
    out->data[i] = a->data[i] + fc * b->data[i];
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_adam_step(
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
      parameters, gradient, first_moment, "torchlean_cuda_buffer_adam_step_stub");
  torchlean_cuda_require_same_size2(
      parameters, second_moment, "torchlean_cuda_buffer_adam_step_stub");

  torchlean_cuda_buffer* updated_parameters = torchlean_cuda_buffer_alloc(parameters->size);
  torchlean_cuda_buffer* updated_first_moment = torchlean_cuda_buffer_alloc(parameters->size);
  torchlean_cuda_buffer* updated_second_moment = torchlean_cuda_buffer_alloc(parameters->size);
  const float beta1f = (float)beta1;
  const float one_minus_beta1f = (float)oneMinusBeta1;
  const float beta2f = (float)beta2;
  const float one_minus_beta2f = (float)oneMinusBeta2;
  const float first_correctionf = (float)firstMomentCorrection;
  const float second_correctionf = (float)secondMomentCorrection;
  const float epsilonf = (float)epsilon;
  const float decayf = (float)decay;
  const float update_scalef = (float)updateScale;

  for (size_t i = 0; i < parameters->size; ++i) {
    const float m_scaled = first_moment->data[i] * beta1f;
    const float m = fmaf(one_minus_beta1f, gradient->data[i], m_scaled);
    const float g2 = gradient->data[i] * gradient->data[i];
    const float v_scaled = second_moment->data[i] * beta2f;
    const float v = fmaf(one_minus_beta2f, g2, v_scaled);
    const float m_hat = m * first_correctionf;
    const float v_hat = v * second_correctionf;
    const float denominator = sqrtf(v_hat) + epsilonf;
    const float normalized_update = m_hat / denominator;
    const float decayed_parameter =
        fmaf(decayf, parameters->data[i], parameters->data[i]);

    updated_first_moment->data[i] = m;
    updated_second_moment->data[i] = v;
    updated_parameters->data[i] =
        fmaf(update_scalef, normalized_update, decayed_parameter);
  }
  return torchlean_cuda_box_three_buffers(
      updated_parameters, updated_first_moment, updated_second_moment);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_scaled_prod_exp(b_lean_obj_arg AObj,
                                                              b_lean_obj_arg BObj, double c) {
  torchlean_cuda_buffer* a = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_require_same_size2(a, b, "torchlean_cuda_buffer_scaled_prod_exp_stub");
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(a->size);
  float fc = (float)c;
  for (size_t i = 0; i < a->size; ++i) {
    out->data[i] = expf((fc * a->data[i]) * b->data[i]);
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_sum(b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(1);
  if (torchlean_cuda_get_deterministic_reductions()) {
    double acc = 0.0;
    for (size_t i = 0; i < b->size; ++i) {
      acc += (double)b->data[i];
    }
    out->data[0] = (float)acc;
    return torchlean_cuda_buffer_box(out);
  }
  float acc = 0.0f;
  for (size_t i = 0; i < b->size; ++i) {
    acc += b->data[i];
  }
  out->data[0] = acc;
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_mean(b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(1);
  if (b->size == 0) {
    out->data[0] = NAN;
    return torchlean_cuda_buffer_box(out);
  }
  if (torchlean_cuda_get_deterministic_reductions()) {
    double acc = 0.0;
    for (size_t i = 0; i < b->size; ++i) {
      acc += (double)b->data[i];
    }
    out->data[0] = (float)(acc / (double)b->size);
    return torchlean_cuda_buffer_box(out);
  }
  float acc = 0.0f;
  for (size_t i = 0; i < b->size; ++i) {
    acc += b->data[i];
  }
  out->data[0] = acc / (float)b->size;
  return torchlean_cuda_buffer_box(out);
}
