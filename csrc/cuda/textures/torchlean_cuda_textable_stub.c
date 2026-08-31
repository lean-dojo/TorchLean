// Portable CPU twin of `torchlean_cuda_textable.cu`.
//
// Linked by the default (non-CUDA) build. Every exported symbol matches the CUDA translation unit.
// Point mode (`filter_mode == 0`) is bit-identical to the CUDA kernel: both sides evaluate the
// same clamp/floor/lerp in float32 with contraction blocked (the CUDA side uses `__f*_rn`
// intrinsics; this side uses single-assignment `float` temporaries, which baseline x86-64
// `cc -O2` compiles without FMA contraction). Hardware mode (`filter_mode == 1`) emulates the
// 9-bit fixed-point lerp weight (8 fractional bits, round-to-nearest); CUDA does not specify the
// hardware's coefficient rounding, so hardware mode is compared by tolerance in both builds.

#include <lean/lean.h>

#include "torchlean_cuda_buffer.h"
#include "torchlean_cuda_textable.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

// --- External object plumbing ------------------------------------------------

static void torchlean_cuda_textable_finalize(void* ptr) {
  torchlean_cuda_textable* t = (torchlean_cuda_textable*)ptr;
  if (!t) {
    return;
  }
  free(t->host);
  free(t);
}

// `torchlean_cuda_textable` holds no Lean references.
static void torchlean_cuda_textable_foreach(void* _ptr, b_lean_obj_arg _fn) {
  (void)_ptr;
  (void)_fn;
}

static lean_external_class* torchlean_cuda_textable_class = NULL;

static lean_external_class* torchlean_cuda_textable_get_class(void) {
  if (!torchlean_cuda_textable_class) {
    torchlean_cuda_textable_class = lean_register_external_class(torchlean_cuda_textable_finalize,
                                                                torchlean_cuda_textable_foreach);
  }
  return torchlean_cuda_textable_class;
}

torchlean_cuda_textable* torchlean_cuda_textable_unbox(b_lean_obj_arg obj) {
  lean_object* o = (lean_object*)obj;
  if (!lean_is_external(o)) {
    lean_internal_panic("torchlean_cuda_textable_stub: expected external object");
  }
  return (torchlean_cuda_textable*)lean_get_external_data(o);
}

lean_obj_res torchlean_cuda_textable_box(torchlean_cuda_textable* t) {
  return lean_alloc_external(torchlean_cuda_textable_get_class(), t);
}

// --- Construction ------------------------------------------------------------

LEAN_EXPORT lean_obj_res torchlean_cuda_textable_make(b_lean_obj_arg data, b_lean_obj_arg width,
                                                      b_lean_obj_arg layers, uint8_t hw_filter) {
  const uint32_t w = nat_to_u32_or_panic(width, "torchlean_cuda_textable_make: width out of range");
  const uint32_t l = nat_to_u32_or_panic(layers, "torchlean_cuda_textable_make: layers out of range");
  if (w == 0 || l == 0) {
    lean_internal_panic("torchlean_cuda_textable_make: width and layers must be >= 1");
  }
  lean_object* A = (lean_object*)data;
  const size_t n = lean_sarray_size(A);
  if (n != (size_t)w * (size_t)l) {
    lean_internal_panic("torchlean_cuda_textable_make: data size must equal layers * width");
  }
  const double* src = lean_float_array_cptr(A);

  torchlean_cuda_textable* t =
      (torchlean_cuda_textable*)malloc(sizeof(torchlean_cuda_textable));
  if (!t) {
    lean_internal_panic_out_of_memory();
  }
  t->width = w;
  t->layers = l;
  t->filter_mode = hw_filter ? 1u : 0u;
  t->cuarray = NULL;
  t->tex = 0;
  t->host = (float*)malloc(n * sizeof(float));
  if (!t->host) {
    free(t);
    lean_internal_panic_out_of_memory();
  }
  for (size_t i = 0; i < n; ++i) {
    t->host[i] = (float)src[i];
  }
  return torchlean_cuda_textable_box(t);
}

// --- Fetch -------------------------------------------------------------------

LEAN_EXPORT lean_obj_res torchlean_cuda_textable_fetch(b_lean_obj_arg tbl, b_lean_obj_arg coords,
                                                       b_lean_obj_arg layer_idx) {
  torchlean_cuda_textable* t = torchlean_cuda_textable_unbox(tbl);
  torchlean_cuda_buffer* u = torchlean_cuda_buffer_unbox(coords);
  torchlean_cuda_buffer* l = torchlean_cuda_buffer_unbox(layer_idx);
  torchlean_cuda_require_same_size2(u, l, "torchlean_cuda_textable_fetch");

  const size_t n = u->size;
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(n);
  const float wmax = (float)(t->width - 1);
  const long lmax = (long)t->layers - 1;

  for (size_t i = 0; i < n; ++i) {
    float uc = u->data[i];
    if (uc < 0.0f) {
      uc = 0.0f;
    }
    if (uc > wmax) {
      uc = wmax;
    }
    long L = (long)(l->data[i] + 0.5f);
    if (L < 0) {
      L = 0;
    }
    if (L > lmax) {
      L = lmax;
    }
    const float j = floorf(uc);
    float f = uc - j;
    if (t->filter_mode) {
      // 9-bit fixed-point lerp weight: 8 fractional bits, round-to-nearest.
      f = floorf(f * 256.0f + 0.5f) / 256.0f;
    }
    const float* row = t->host + (size_t)L * t->width;
    const size_t j0 = (size_t)j;
    const size_t j1 = (j0 + 1 < t->width) ? j0 + 1 : t->width - 1;
    const float a = row[j0];
    const float b = row[j1];
    const float d = b - a;
    const float fd = f * d;
    out->data[i] = a + fd;
  }
  return torchlean_cuda_buffer_box(out);
}

// --- Metadata ----------------------------------------------------------------

LEAN_EXPORT lean_obj_res torchlean_cuda_textable_width(b_lean_obj_arg tbl) {
  return lean_usize_to_nat(torchlean_cuda_textable_unbox(tbl)->width);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_textable_layers(b_lean_obj_arg tbl) {
  return lean_usize_to_nat(torchlean_cuda_textable_unbox(tbl)->layers);
}

LEAN_EXPORT uint8_t torchlean_cuda_textable_filter_hw(b_lean_obj_arg tbl) {
  return torchlean_cuda_textable_unbox(tbl)->filter_mode ? 1 : 0;
}
