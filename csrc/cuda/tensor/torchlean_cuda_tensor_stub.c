#include <lean/lean.h>
#include <lean/mimalloc.h>

#include "torchlean_cuda_buffer.h"
#include "torchlean_cuda_deterministic_reductions_env.h"
#include "torchlean_cuda_rng_common.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

// CPU fallback for TorchLean's buffer runtime.
//
// The stub is deliberately not a second API: it is the same FFI surface implemented with host
// memory so `lake build` and most tests do not require CUDA. Keep behavior aligned with
// `torchlean_cuda_tensor.cu`, including edge cases such as empty buffers and deterministic-reduction
// mode. Performance is not the point here; parity and debuggability are.

// CPU stub keeps the deterministic-reductions flag for API parity; lazily init from env (C init constraints).
static uint32_t g_torchlean_deterministic_reductions = 0u;
static uint32_t g_torchlean_deterministic_reductions_inited = 0u;

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

static void torchlean_cuda_buffer_finalize(void* ptr) {
  torchlean_cuda_buffer* b = (torchlean_cuda_buffer*)ptr;
  if (!b) {
    return;
  }
  free(b->data);
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

torchlean_cuda_buffer* torchlean_cuda_buffer_unbox(b_lean_obj_arg obj) {
  lean_object* o = (lean_object*)obj;
  if (!lean_is_external(o)) {
    lean_internal_panic("torchlean_cuda_buffer_stub: expected external object");
  }
  return (torchlean_cuda_buffer*)lean_get_external_data(o);
}

lean_obj_res torchlean_cuda_buffer_box(torchlean_cuda_buffer* b) {
  return lean_alloc_external(torchlean_cuda_buffer_get_class(), b);
}

torchlean_cuda_buffer* torchlean_cuda_buffer_alloc(size_t n) {
  torchlean_cuda_buffer* b = (torchlean_cuda_buffer*)malloc(sizeof(torchlean_cuda_buffer));
  if (!b) {
    lean_internal_panic_out_of_memory();
  }
  b->size = n;
  b->data = NULL;
  if (n > 0) {
    b->data = (float*)malloc(n * sizeof(float));
    if (!b->data) {
      free(b);
      lean_internal_panic_out_of_memory();
    }
  }
  return b;
}

LEAN_EXPORT uint32_t torchlean_cuda_buffer_size(b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  if (b->size > 0xFFFFFFFFULL) {
    lean_internal_panic("torchlean_cuda_buffer_size_stub: buffer too large for UInt32");
  }
  return (uint32_t)b->size;
}

LEAN_EXPORT uint32_t torchlean_cuda_buffer_release(b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  if (!b || !b->data) {
    return 0;
  }
  free(b->data);
  // Explicit release is an eager-runtime lifetime hint. We mark the handle as empty so accidental
  // reuse fails by size checks instead of touching freed memory.
  b->data = NULL;
  b->size = 0;
  return 1;
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_release_then(
    b_lean_obj_arg scratchObj, b_lean_obj_arg keepObj) {
  (void)torchlean_cuda_buffer_release(scratchObj);
  lean_inc((lean_object*)keepObj);
  return (lean_object*)keepObj;
}

LEAN_EXPORT uint32_t torchlean_runtime_collect_allocator(uint32_t force) {
  mi_collect(force != 0);
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
  if (x->size != g->size) {
    lean_internal_panic("torchlean_cuda_buffer_abs_bwd_stub: size mismatch");
  }
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
  if (x->size != g->size) {
    lean_internal_panic("torchlean_cuda_buffer_sqrt_bwd_stub: size mismatch");
  }
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
  if (x->size != g->size) {
    lean_internal_panic("torchlean_cuda_buffer_clamp_bwd_stub: size mismatch");
  }
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
  if (a->size != b->size) {
    lean_internal_panic("torchlean_cuda_buffer_max_stub: size mismatch");
  }
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
  if (a->size != b->size || a->size != g->size) {
    lean_internal_panic("torchlean_cuda_buffer_max_bwd_stub: size mismatch");
  }
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
  lean_object* pair = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(pair, 0, torchlean_cuda_buffer_box(dA));
  lean_ctor_set(pair, 1, torchlean_cuda_buffer_box(dB));
  return pair;
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_min(b_lean_obj_arg AObj, b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* a = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  if (a->size != b->size) {
    lean_internal_panic("torchlean_cuda_buffer_min_stub: size mismatch");
  }
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
  if (a->size != b->size || a->size != g->size) {
    lean_internal_panic("torchlean_cuda_buffer_min_bwd_stub: size mismatch");
  }
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
  lean_object* pair = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(pair, 0, torchlean_cuda_buffer_box(dA));
  lean_ctor_set(pair, 1, torchlean_cuda_buffer_box(dB));
  return pair;
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_div(b_lean_obj_arg AObj, b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* a = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  if (a->size != b->size) {
    lean_internal_panic("torchlean_cuda_buffer_div_stub: size mismatch");
  }
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
  if (x->size != g->size) {
    lean_internal_panic("torchlean_cuda_buffer_relu_bwd_stub: size mismatch");
  }
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(x->size);
  for (size_t i = 0; i < x->size; ++i) {
    float v = x->data[i];
    out->data[i] = (v > 0.0f) ? g->data[i] : 0.0f;
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_add(b_lean_obj_arg AObj, b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* a = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  if (a->size != b->size) {
    char msg[160];
    snprintf(msg, sizeof(msg), "torchlean_cuda_buffer_add_stub: size mismatch (%zu vs %zu)",
             a->size, b->size);
    lean_internal_panic(msg);
  }
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(a->size);
  for (size_t i = 0; i < a->size; ++i) {
    out->data[i] = a->data[i] + b->data[i];
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_sub(b_lean_obj_arg AObj, b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* a = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  if (a->size != b->size) {
    lean_internal_panic("torchlean_cuda_buffer_sub_stub: size mismatch");
  }
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(a->size);
  for (size_t i = 0; i < a->size; ++i) {
    out->data[i] = a->data[i] - b->data[i];
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_mul(b_lean_obj_arg AObj, b_lean_obj_arg BObj) {
  torchlean_cuda_buffer* a = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  if (a->size != b->size) {
    lean_internal_panic("torchlean_cuda_buffer_mul_stub: size mismatch");
  }
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

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_axpy(b_lean_obj_arg AObj, b_lean_obj_arg BObj,
                                                   double c) {
  torchlean_cuda_buffer* a = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  if (a->size != b->size) {
    lean_internal_panic("torchlean_cuda_buffer_axpy_stub: size mismatch");
  }
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(a->size);
  float fc = (float)c;
  for (size_t i = 0; i < a->size; ++i) {
    out->data[i] = a->data[i] + fc * b->data[i];
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
