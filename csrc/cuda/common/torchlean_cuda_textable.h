#pragma once

#include <lean/lean.h>

#include <stddef.h>
#include <stdint.h>

#include "torchlean_cuda_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Layered 1-D lookup-table textures (float32).
//
// This header is the native side of `NN.Runtime.Autograd.Engine.Cuda.TexTable`.
// A `torchlean_cuda_textable` holds a small, immutable, layered 1-D table of float32 samples over a
// uniform abscissa, evaluated by piecewise-linear interpolation at grid-space coordinates
// `u ∈ [0, width−1]` (clamped). Typical uses are tabulated transfer functions and calibration
// curves whose analytic form is expensive or unavailable.
//
// Two filter modes are fixed at construction:
// - `filter_mode == 0` (point): two point fetches plus an explicit float32 lerp. This mode is
//   bit-reproducible: the CUDA build and the CPU stub compute identical float32 results.
// - `filter_mode == 1` (hardware): the GPU texture unit interpolates in hardware. CUDA specifies a
//   9-bit fixed-point fraction (8 fractional bits) for the lerp weight, so results are only
//   approximate; the CPU stub emulates the quantized weight and callers must compare by tolerance.
//
// Ownership/ABI:
// - Lean owns an external object that points at `torchlean_cuda_textable`;
// - `host` is always a private float32 copy of the samples (`layers * width` elements, layer-major);
// - in the CUDA build `cuarray`/`tex` hold the `cudaArray_t` and `cudaTextureObject_t`; the CPU
//   stub leaves them NULL/0;
// - tables are immutable after construction; the finalizer releases every resource.
//
// This is a trusted boundary, like `torchlean_cuda_buffer.h`.

typedef struct {
  size_t width;        // samples per layer (>= 1)
  size_t layers;       // number of layers (>= 1)
  uint32_t filter_mode;  // 0 = point + explicit lerp (bit-reproducible), 1 = hardware linear
  float* host;         // private float32 copy, layers*width, layer-major
  void* cuarray;       // CUDA build: cudaArray_t; NULL in the CPU stub
  unsigned long long tex;  // CUDA build: cudaTextureObject_t; 0 in the CPU stub
} torchlean_cuda_textable;

// Helpers implemented by `torchlean_cuda_textable.cu` / `torchlean_cuda_textable_stub.c`.
torchlean_cuda_textable* torchlean_cuda_textable_unbox(b_lean_obj_arg obj);
lean_obj_res torchlean_cuda_textable_box(torchlean_cuda_textable* t);

// Construct a table from a Lean `FloatArray` of `layers * width` samples (layer-major, narrowed to
// float32). Panics unless `width >= 1`, `layers >= 1`, and the array has exactly `layers * width`
// entries.
LEAN_EXPORT lean_obj_res torchlean_cuda_textable_make(b_lean_obj_arg data, b_lean_obj_arg width,
                                                      b_lean_obj_arg layers, uint8_t hw_filter);

// Evaluate the table at per-element grid coordinates `coords` (clamped to `[0, width−1]`) in the
// layer selected by `layer_idx` (integral-valued float32 indices, clamped to `[0, layers−1]`).
// `coords` and `layer_idx` must have equal sizes; returns a new buffer of that size.
LEAN_EXPORT lean_obj_res torchlean_cuda_textable_fetch(b_lean_obj_arg tbl, b_lean_obj_arg coords,
                                                       b_lean_obj_arg layer_idx);

// Metadata accessors.
LEAN_EXPORT lean_obj_res torchlean_cuda_textable_width(b_lean_obj_arg tbl);
LEAN_EXPORT lean_obj_res torchlean_cuda_textable_layers(b_lean_obj_arg tbl);
LEAN_EXPORT uint8_t torchlean_cuda_textable_filter_hw(b_lean_obj_arg tbl);

#ifdef __cplusplus
}  // extern "C"
#endif
