// CUDA implementation of layered 1-D lookup-table textures (float32).
//
// See `csrc/cuda/common/torchlean_cuda_textable.h` for the ABI and semantics contract, and
// `csrc/cuda/textures/torchlean_cuda_textable_stub.c` for the portable CPU twin. Every exported
// symbol here must exist in the stub with the same name and signature.

#include <lean/lean.h>

#include "torchlean_cuda_buffer.h"
#include "torchlean_cuda_common.h"
#include "torchlean_cuda_textable.h"

#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// --- External object plumbing ------------------------------------------------

static void torchlean_cuda_textable_finalize(void* ptr) {
  torchlean_cuda_textable* t = (torchlean_cuda_textable*)ptr;
  if (!t) {
    return;
  }
  if (t->tex != 0) {
    (void)cudaDestroyTextureObject((cudaTextureObject_t)t->tex);
  }
  if (t->cuarray) {
    (void)cudaFreeArray((cudaArray_t)t->cuarray);
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

extern "C" torchlean_cuda_textable* torchlean_cuda_textable_unbox(b_lean_obj_arg obj) {
  lean_object* o = (lean_object*)obj;
  if (!lean_is_external(o)) {
    lean_internal_panic("torchlean_cuda_textable: expected external object");
  }
  return (torchlean_cuda_textable*)lean_get_external_data(o);
}

extern "C" lean_obj_res torchlean_cuda_textable_box(torchlean_cuda_textable* t) {
  return lean_alloc_external(torchlean_cuda_textable_get_class(), t);
}

// --- Construction ------------------------------------------------------------

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_textable_make(b_lean_obj_arg data,
                                                                b_lean_obj_arg width,
                                                                b_lean_obj_arg layers,
                                                                uint8_t hw_filter) {
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

  cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
  cudaArray_t arr = NULL;
  checkCuda(cudaMalloc3DArray(&arr, &desc, make_cudaExtent(w, 0, l), cudaArrayLayered),
            "torchlean_cuda_textable_make: cudaMalloc3DArray failed");
  t->cuarray = (void*)arr;

  cudaMemcpy3DParms copy;
  memset(&copy, 0, sizeof(copy));
  copy.srcPtr = make_cudaPitchedPtr((void*)t->host, (size_t)w * sizeof(float), (size_t)w, 1);
  copy.dstArray = arr;
  copy.extent = make_cudaExtent(w, 1, l);
  copy.kind = cudaMemcpyHostToDevice;
  checkCuda(cudaMemcpy3D(&copy), "torchlean_cuda_textable_make: cudaMemcpy3D failed");

  cudaResourceDesc res;
  memset(&res, 0, sizeof(res));
  res.resType = cudaResourceTypeArray;
  res.res.array.array = arr;

  cudaTextureDesc texd;
  memset(&texd, 0, sizeof(texd));
  texd.addressMode[0] = cudaAddressModeClamp;
  texd.addressMode[1] = cudaAddressModeClamp;
  texd.filterMode = hw_filter ? cudaFilterModeLinear : cudaFilterModePoint;
  texd.readMode = cudaReadModeElementType;
  texd.normalizedCoords = 0;

  cudaTextureObject_t tex = 0;
  checkCuda(cudaCreateTextureObject(&tex, &res, &texd, NULL),
            "torchlean_cuda_textable_make: cudaCreateTextureObject failed");
  t->tex = (unsigned long long)tex;

  return torchlean_cuda_textable_box(t);
}

// --- Fetch -------------------------------------------------------------------

static constexpr int kTextableBlockSize = 256;

static inline dim3 textable_blocks_for(size_t n) {
  size_t blocks = (n + (size_t)kTextableBlockSize - 1) / (size_t)kTextableBlockSize;
  if (blocks == 0) {
    blocks = 1;
  }
  if (blocks > 2147483647ULL) {
    blocks = 2147483647ULL;
  }
  return dim3((unsigned int)blocks);
}

// One kernel for both filter modes; `hw` is a launch parameter, so the branch is warp-uniform.
// Point mode blocks FMA contraction with `__f*_rn` intrinsics: this translation unit is compiled
// with plain `-O2` (no `--fmad=false`), and the explicit lerp must stay bit-identical to the CPU
// stub's float32 arithmetic.
__global__ void torchlean_textable_fetch_f32(cudaTextureObject_t tex, const float* u,
                                             const float* layer, float* out, size_t n,
                                             uint32_t hw, float wmax) {
  size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (i >= n) {
    return;
  }
  float uc = fminf(fmaxf(u[i], 0.0f), wmax);
  int L = (int)(layer[i] + 0.5f);
  if (hw) {
    out[i] = tex1DLayered<float>(tex, uc + 0.5f, L);
  } else {
    float j = floorf(uc);
    float f = __fsub_rn(uc, j);
    float a = tex1DLayered<float>(tex, j + 0.5f, L);
    float b = tex1DLayered<float>(tex, fminf(j + 1.0f, wmax) + 0.5f, L);
    out[i] = __fadd_rn(a, __fmul_rn(f, __fsub_rn(b, a)));
  }
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_textable_fetch(b_lean_obj_arg tbl,
                                                                 b_lean_obj_arg coords,
                                                                 b_lean_obj_arg layer_idx) {
  torchlean_cuda_textable* t = torchlean_cuda_textable_unbox(tbl);
  torchlean_cuda_buffer* u = torchlean_cuda_buffer_unbox(coords);
  torchlean_cuda_buffer* l = torchlean_cuda_buffer_unbox(layer_idx);
  torchlean_cuda_require_same_size2(u, l, "torchlean_cuda_textable_fetch");

  const size_t n = u->size;
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(n);
  if (n == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  dim3 blocks = textable_blocks_for(n);
  dim3 threads = dim3(kTextableBlockSize);
  torchlean_textable_fetch_f32<<<blocks, threads>>>((cudaTextureObject_t)t->tex, u->data, l->data,
                                                    out->data, n, t->filter_mode,
                                                    (float)(t->width - 1));
  checkCuda(cudaGetLastError(), "cuda textable_fetch kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

// --- Metadata ----------------------------------------------------------------

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_textable_width(b_lean_obj_arg tbl) {
  return lean_usize_to_nat(torchlean_cuda_textable_unbox(tbl)->width);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_textable_layers(b_lean_obj_arg tbl) {
  return lean_usize_to_nat(torchlean_cuda_textable_unbox(tbl)->layers);
}

extern "C" LEAN_EXPORT uint8_t torchlean_cuda_textable_filter_hw(b_lean_obj_arg tbl) {
  return torchlean_cuda_textable_unbox(tbl)->filter_mode ? 1 : 0;
}
