#include <lean/lean.h>

#include "torchlean_cuda_buffer.h"
#include "torchlean_cuda_conv_pool_common.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// CPU fallback for conv/pool kernels over `Cuda.Buffer`.
//
// This stub exists so the Lean library builds and tests without a CUDA toolkit. It is also the
// easiest place to audit intended semantics: row-major flattening, Nat-style output shapes,
// ignored max-pool padding, count-include-pad average pooling, and deterministic fallback behavior.
// Keep exported symbols and edge cases aligned with `torchlean_cuda_conv_pool.cu`.

enum { K_MAX_RANK = TORCHLEAN_CUDA_CONV_POOL_MAX_RANK };

static inline void unflatten_coords(uint32_t* coords, const uint32_t* dims, int rank, size_t idx) {
  for (int ax = rank - 1; ax >= 0; --ax) {
    const uint32_t d = dims[ax];
    coords[ax] = (d == 0) ? 0 : (uint32_t)(idx % (size_t)d);
    idx = (d == 0) ? 0 : (idx / (size_t)d);
  }
}

static inline size_t flatten_coords(const uint32_t* coords, const uint32_t* dims, int rank) {
  size_t idx = 0;
  for (int ax = 0; ax < rank; ++ax) {
    idx = idx * (size_t)dims[ax] + (size_t)coords[ax];
  }
  return idx;
}

LEAN_EXPORT lean_obj_res torchlean_cuda_conv2d_fwd(
    b_lean_obj_arg inputObj, b_lean_obj_arg kernelObj, b_lean_obj_arg biasObj,
    uint32_t inC, uint32_t inH, uint32_t inW,
    uint32_t outC, uint32_t kH, uint32_t kW,
    uint32_t stride, uint32_t padding) {
  torchlean_cuda_buffer* input = torchlean_cuda_buffer_unbox(inputObj);
  torchlean_cuda_buffer* kernel = torchlean_cuda_buffer_unbox(kernelObj);
  torchlean_cuda_buffer* bias = torchlean_cuda_buffer_unbox(biasObj);

  if (kH == 0 || kW == 0) {
    lean_internal_panic("torchlean_cuda_conv2d_fwd: kH/kW must be > 0");
  }

  uint32_t outH = outDim(inH, kH, stride, padding);
  uint32_t outW = outDim(inW, kW, stride, padding);

  size_t inElems = (size_t)inC * (size_t)inH * (size_t)inW;
  size_t kElems = (size_t)outC * (size_t)inC * (size_t)kH * (size_t)kW;
  size_t bElems = (size_t)outC;
  size_t outElems = (size_t)outC * (size_t)outH * (size_t)outW;

  checkBufSize(input, inElems, "torchlean_cuda_conv2d_fwd_stub: input.size mismatch");
  checkBufSize(kernel, kElems, "torchlean_cuda_conv2d_fwd_stub: kernel.size mismatch");
  checkBufSize(bias, bElems, "torchlean_cuda_conv2d_fwd_stub: bias.size mismatch");

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outElems);
  if (outElems == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  for (uint32_t oc = 0; oc < outC; ++oc) {
    for (uint32_t oh = 0; oh < outH; ++oh) {
      for (uint32_t ow = 0; ow < outW; ++ow) {
        float acc = bias->data[oc];
        for (uint32_t ic = 0; ic < inC; ++ic) {
          for (uint32_t ky = 0; ky < kH; ++ky) {
            int ih = (int)((int64_t)oh * (int64_t)stride + (int64_t)ky - (int64_t)padding);
            if (ih < 0 || ih >= (int)inH) continue;
            for (uint32_t kx = 0; kx < kW; ++kx) {
              int iw = (int)((int64_t)ow * (int64_t)stride + (int64_t)kx - (int64_t)padding);
              if (iw < 0 || iw >= (int)inW) continue;
              size_t inIdx =
                  ((size_t)ic * (size_t)inH + (size_t)ih) * (size_t)inW + (size_t)iw;
              size_t wIdx =
                  (((size_t)oc * (size_t)inC + (size_t)ic) * (size_t)kH + (size_t)ky) *
                      (size_t)kW +
                  (size_t)kx;
              acc += input->data[inIdx] * kernel->data[wIdx];
            }
          }
        }
        size_t outIdx = ((size_t)oc * (size_t)outH + (size_t)oh) * (size_t)outW + (size_t)ow;
        out->data[outIdx] = acc;
      }
    }
  }

  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_conv2d_bwd(
    b_lean_obj_arg inputObj, b_lean_obj_arg kernelObj, b_lean_obj_arg gradObj,
    uint32_t inC, uint32_t inH, uint32_t inW,
    uint32_t outC, uint32_t kH, uint32_t kW,
    uint32_t stride, uint32_t padding) {
  torchlean_cuda_buffer* input = torchlean_cuda_buffer_unbox(inputObj);
  torchlean_cuda_buffer* kernel = torchlean_cuda_buffer_unbox(kernelObj);
  torchlean_cuda_buffer* gradOutput = torchlean_cuda_buffer_unbox(gradObj);

  if (kH == 0 || kW == 0) {
    lean_internal_panic("torchlean_cuda_conv2d_bwd: kH/kW must be > 0");
  }

  uint32_t outH = outDim(inH, kH, stride, padding);
  uint32_t outW = outDim(inW, kW, stride, padding);

  size_t inElems = (size_t)inC * (size_t)inH * (size_t)inW;
  size_t kElems = (size_t)outC * (size_t)inC * (size_t)kH * (size_t)kW;
  size_t outElems = (size_t)outC * (size_t)outH * (size_t)outW;

  checkBufSize(input, inElems, "torchlean_cuda_conv2d_bwd_stub: input.size mismatch");
  checkBufSize(kernel, kElems, "torchlean_cuda_conv2d_bwd_stub: kernel.size mismatch");
  checkBufSize(gradOutput, outElems, "torchlean_cuda_conv2d_bwd_stub: gradOutput.size mismatch");

  torchlean_cuda_buffer* dKernel = torchlean_cuda_buffer_alloc(kElems);
  torchlean_cuda_buffer* dBias = torchlean_cuda_buffer_alloc((size_t)outC);
  torchlean_cuda_buffer* dInput = torchlean_cuda_buffer_alloc(inElems);

  for (size_t i = 0; i < dKernel->size; ++i) {
    dKernel->data[i] = 0.0f;
  }
  for (size_t i = 0; i < dBias->size; ++i) {
    dBias->data[i] = 0.0f;
  }
  for (size_t i = 0; i < dInput->size; ++i) {
    dInput->data[i] = 0.0f;
  }

  // Accumulate all three gradients in one pass over gradOutput.
  for (uint32_t oc = 0; oc < outC; ++oc) {
    for (uint32_t oh = 0; oh < outH; ++oh) {
      for (uint32_t ow = 0; ow < outW; ++ow) {
        size_t goIdx = ((size_t)oc * (size_t)outH + (size_t)oh) * (size_t)outW + (size_t)ow;
        float go = gradOutput->data[goIdx];
        if (dBias->size > 0) {
          dBias->data[oc] += go;
        }

        for (uint32_t ic = 0; ic < inC; ++ic) {
          for (uint32_t ky = 0; ky < kH; ++ky) {
            int ih = (int)((int64_t)oh * (int64_t)stride + (int64_t)ky - (int64_t)padding);
            if (ih < 0 || ih >= (int)inH) continue;
            for (uint32_t kx = 0; kx < kW; ++kx) {
              int iw = (int)((int64_t)ow * (int64_t)stride + (int64_t)kx - (int64_t)padding);
              if (iw < 0 || iw >= (int)inW) continue;

              size_t inIdx =
                  ((size_t)ic * (size_t)inH + (size_t)ih) * (size_t)inW + (size_t)iw;
              size_t wIdx =
                  (((size_t)oc * (size_t)inC + (size_t)ic) * (size_t)kH + (size_t)ky) *
                      (size_t)kW +
                  (size_t)kx;

              dKernel->data[wIdx] += input->data[inIdx] * go;
              dInput->data[inIdx] += kernel->data[wIdx] * go;
            }
          }
        }
      }
    }
  }

  // Return (dKernel, dBias, dInput) as `Buffer × Buffer × Buffer`.
  lean_object* inner = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(inner, 0, torchlean_cuda_buffer_box(dBias));
  lean_ctor_set(inner, 1, torchlean_cuda_buffer_box(dInput));

  lean_object* outer = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(outer, 0, torchlean_cuda_buffer_box(dKernel));
  lean_ctor_set(outer, 1, inner);
  return outer;
}

LEAN_EXPORT lean_obj_res torchlean_cuda_convtranspose2d_fwd(
    b_lean_obj_arg inputObj, b_lean_obj_arg kernelObj, b_lean_obj_arg biasObj,
    uint32_t inC, uint32_t inH, uint32_t inW,
    uint32_t outC, uint32_t kH, uint32_t kW,
    uint32_t stride, uint32_t padding) {
  torchlean_cuda_buffer* input = torchlean_cuda_buffer_unbox(inputObj);
  torchlean_cuda_buffer* kernel = torchlean_cuda_buffer_unbox(kernelObj);
  torchlean_cuda_buffer* bias = torchlean_cuda_buffer_unbox(biasObj);

  if (kH == 0 || kW == 0) {
    lean_internal_panic("torchlean_cuda_convtranspose2d_fwd: kH/kW must be > 0");
  }

  uint32_t outH = outDimTranspose(inH, kH, stride, padding);
  uint32_t outW = outDimTranspose(inW, kW, stride, padding);

  size_t inElems = (size_t)inC * (size_t)inH * (size_t)inW;
  size_t kElems = (size_t)inC * (size_t)outC * (size_t)kH * (size_t)kW;
  size_t bElems = (size_t)outC;
  size_t outElems = (size_t)outC * (size_t)outH * (size_t)outW;

  checkBufSize(input, inElems, "torchlean_cuda_convtranspose2d_fwd_stub: input.size mismatch");
  checkBufSize(kernel, kElems, "torchlean_cuda_convtranspose2d_fwd_stub: kernel.size mismatch");
  checkBufSize(bias, bElems, "torchlean_cuda_convtranspose2d_fwd_stub: bias.size mismatch");

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outElems);
  if (outElems == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  // Output-indexed sum (matches the spec definition).
  for (uint32_t oc = 0; oc < outC; ++oc) {
    for (uint32_t oh = 0; oh < outH; ++oh) {
      for (uint32_t ow = 0; ow < outW; ++ow) {
        float acc = bias->data[oc];
        for (uint32_t ic = 0; ic < inC; ++ic) {
          for (uint32_t ky = 0; ky < kH; ++ky) {
            int ihNum = (int)((int64_t)oh + (int64_t)padding - (int64_t)ky);
            if (ihNum < 0) continue;
            if (stride == 0) continue;
            if ((ihNum % (int)stride) != 0) continue;
            int ih = ihNum / (int)stride;
            if (ih < 0 || ih >= (int)inH) continue;
            for (uint32_t kx = 0; kx < kW; ++kx) {
              int iwNum = (int)((int64_t)ow + (int64_t)padding - (int64_t)kx);
              if (iwNum < 0) continue;
              if ((iwNum % (int)stride) != 0) continue;
              int iw = iwNum / (int)stride;
              if (iw < 0 || iw >= (int)inW) continue;
              size_t inIdx =
                  ((size_t)ic * (size_t)inH + (size_t)ih) * (size_t)inW + (size_t)iw;
              size_t wIdx =
                  (((size_t)ic * (size_t)outC + (size_t)oc) * (size_t)kH + (size_t)ky) *
                      (size_t)kW +
                  (size_t)kx;
              acc += input->data[inIdx] * kernel->data[wIdx];
            }
          }
        }
        size_t outIdx = ((size_t)oc * (size_t)outH + (size_t)oh) * (size_t)outW + (size_t)ow;
        out->data[outIdx] = acc;
      }
    }
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_convtranspose2d_bwd(
    b_lean_obj_arg inputObj, b_lean_obj_arg kernelObj, b_lean_obj_arg gradObj,
    uint32_t inC, uint32_t inH, uint32_t inW,
    uint32_t outC, uint32_t kH, uint32_t kW,
    uint32_t stride, uint32_t padding) {
  torchlean_cuda_buffer* input = torchlean_cuda_buffer_unbox(inputObj);
  torchlean_cuda_buffer* kernel = torchlean_cuda_buffer_unbox(kernelObj);
  torchlean_cuda_buffer* gradOutput = torchlean_cuda_buffer_unbox(gradObj);

  if (kH == 0 || kW == 0) {
    lean_internal_panic("torchlean_cuda_convtranspose2d_bwd: kH/kW must be > 0");
  }

  uint32_t outH = outDimTranspose(inH, kH, stride, padding);
  uint32_t outW = outDimTranspose(inW, kW, stride, padding);

  size_t inElems = (size_t)inC * (size_t)inH * (size_t)inW;
  size_t kElems = (size_t)inC * (size_t)outC * (size_t)kH * (size_t)kW;
  size_t outElems = (size_t)outC * (size_t)outH * (size_t)outW;

  checkBufSize(input, inElems, "torchlean_cuda_convtranspose2d_bwd_stub: input.size mismatch");
  checkBufSize(kernel, kElems, "torchlean_cuda_convtranspose2d_bwd_stub: kernel.size mismatch");
  checkBufSize(gradOutput, outElems, "torchlean_cuda_convtranspose2d_bwd_stub: gradOutput.size mismatch");

  torchlean_cuda_buffer* dKernel = torchlean_cuda_buffer_alloc(kElems);
  torchlean_cuda_buffer* dBias = torchlean_cuda_buffer_alloc((size_t)outC);
  torchlean_cuda_buffer* dInput = torchlean_cuda_buffer_alloc(inElems);

  for (size_t i = 0; i < dKernel->size; ++i) dKernel->data[i] = 0.0f;
  for (size_t i = 0; i < dBias->size; ++i) dBias->data[i] = 0.0f;
  for (size_t i = 0; i < dInput->size; ++i) dInput->data[i] = 0.0f;

  // dBias: sum gradOutput over spatial axes.
  for (uint32_t oc = 0; oc < outC; ++oc) {
    float acc = 0.0f;
    for (uint32_t oh = 0; oh < outH; ++oh) {
      for (uint32_t ow = 0; ow < outW; ++ow) {
        size_t goIdx = ((size_t)oc * (size_t)outH + (size_t)oh) * (size_t)outW + (size_t)ow;
        acc += gradOutput->data[goIdx];
      }
    }
    if (dBias->size > 0) dBias->data[oc] = acc;
  }

  // dKernel: sum over input spatial positions.
  for (uint32_t ic = 0; ic < inC; ++ic) {
    for (uint32_t oc = 0; oc < outC; ++oc) {
      for (uint32_t ky = 0; ky < kH; ++ky) {
        for (uint32_t kx = 0; kx < kW; ++kx) {
          float acc = 0.0f;
          for (uint32_t ih = 0; ih < inH; ++ih) {
            for (uint32_t iw = 0; iw < inW; ++iw) {
              int oh = (int)((int64_t)ih * (int64_t)stride + (int64_t)ky - (int64_t)padding);
              int ow = (int)((int64_t)iw * (int64_t)stride + (int64_t)kx - (int64_t)padding);
              if (oh < 0 || oh >= (int)outH || ow < 0 || ow >= (int)outW) continue;
              size_t inIdx =
                  ((size_t)ic * (size_t)inH + (size_t)ih) * (size_t)inW + (size_t)iw;
              size_t goIdx = ((size_t)oc * (size_t)outH + (size_t)oh) * (size_t)outW + (size_t)ow;
              acc += input->data[inIdx] * gradOutput->data[goIdx];
            }
          }
          size_t wIdx =
              (((size_t)ic * (size_t)outC + (size_t)oc) * (size_t)kH + (size_t)ky) * (size_t)kW +
              (size_t)kx;
          dKernel->data[wIdx] = acc;
        }
      }
    }
  }

  // dInput: convolution of gradOutput with kernel (no bias), producing the original input shape.
  for (uint32_t ic = 0; ic < inC; ++ic) {
    for (uint32_t ih = 0; ih < inH; ++ih) {
      for (uint32_t iw = 0; iw < inW; ++iw) {
        float acc = 0.0f;
        for (uint32_t oc = 0; oc < outC; ++oc) {
          for (uint32_t ky = 0; ky < kH; ++ky) {
            int oh = (int)((int64_t)ih * (int64_t)stride + (int64_t)ky - (int64_t)padding);
            if (oh < 0 || oh >= (int)outH) continue;
            for (uint32_t kx = 0; kx < kW; ++kx) {
              int ow = (int)((int64_t)iw * (int64_t)stride + (int64_t)kx - (int64_t)padding);
              if (ow < 0 || ow >= (int)outW) continue;
              size_t goIdx = ((size_t)oc * (size_t)outH + (size_t)oh) * (size_t)outW + (size_t)ow;
              size_t wIdx =
                  (((size_t)ic * (size_t)outC + (size_t)oc) * (size_t)kH + (size_t)ky) *
                      (size_t)kW +
                  (size_t)kx;
              acc += gradOutput->data[goIdx] * kernel->data[wIdx];
            }
          }
        }
        size_t inIdx =
            ((size_t)ic * (size_t)inH + (size_t)ih) * (size_t)inW + (size_t)iw;
        dInput->data[inIdx] = acc;
      }
    }
  }

  // Return (dKernel, dBias, dInput) as `Buffer × Buffer × Buffer`.
  lean_object* inner = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(inner, 0, torchlean_cuda_buffer_box(dBias));
  lean_ctor_set(inner, 1, torchlean_cuda_buffer_box(dInput));

  lean_object* outer = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(outer, 0, torchlean_cuda_buffer_box(dKernel));
  lean_ctor_set(outer, 1, inner);
  return outer;
}

LEAN_EXPORT lean_obj_res torchlean_cuda_maxpool2d_fwd(
    b_lean_obj_arg inputObj,
    uint32_t inC, uint32_t inH, uint32_t inW,
    uint32_t kH, uint32_t kW,
    uint32_t stride, uint32_t padding) {
  torchlean_cuda_buffer* input = torchlean_cuda_buffer_unbox(inputObj);

  if (kH == 0 || kW == 0) {
    lean_internal_panic("torchlean_cuda_maxpool2d_fwd: kH/kW must be > 0");
  }

  uint32_t outH = outDim(inH, kH, stride, padding);
  uint32_t outW = outDim(inW, kW, stride, padding);

  size_t inElems = (size_t)inC * (size_t)inH * (size_t)inW;
  size_t outElems = (size_t)inC * (size_t)outH * (size_t)outW;

  checkBufSize(input, inElems, "torchlean_cuda_maxpool2d_fwd_stub: input.size mismatch");

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outElems);
  if (outElems == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  for (uint32_t c = 0; c < inC; ++c) {
    for (uint32_t oh = 0; oh < outH; ++oh) {
      for (uint32_t ow = 0; ow < outW; ++ow) {
        float best = -FLT_MAX;
        for (uint32_t ky = 0; ky < kH; ++ky) {
          int ih = (int)((int64_t)oh * (int64_t)stride + (int64_t)ky - (int64_t)padding);
          for (uint32_t kx = 0; kx < kW; ++kx) {
            int iw = (int)((int64_t)ow * (int64_t)stride + (int64_t)kx - (int64_t)padding);
            float v = -INFINITY;
            if (ih >= 0 && ih < (int)inH && iw >= 0 && iw < (int)inW) {
              size_t inIdx =
                  ((size_t)c * (size_t)inH + (size_t)ih) * (size_t)inW + (size_t)iw;
              v = input->data[inIdx];
            }
            if (v > best) {
              best = v;
            }
          }
        }
        size_t outIdx = ((size_t)c * (size_t)outH + (size_t)oh) * (size_t)outW + (size_t)ow;
        out->data[outIdx] = best;
      }
    }
  }

  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_maxpool2d_bwd(
    b_lean_obj_arg inputObj, b_lean_obj_arg gradObj,
    uint32_t inC, uint32_t inH, uint32_t inW,
    uint32_t kH, uint32_t kW,
    uint32_t stride, uint32_t padding) {
  torchlean_cuda_buffer* input = torchlean_cuda_buffer_unbox(inputObj);
  torchlean_cuda_buffer* gradOutput = torchlean_cuda_buffer_unbox(gradObj);

  if (kH == 0 || kW == 0) {
    lean_internal_panic("torchlean_cuda_maxpool2d_bwd: kH/kW must be > 0");
  }

  uint32_t outH = outDim(inH, kH, stride, padding);
  uint32_t outW = outDim(inW, kW, stride, padding);

  size_t inElems = (size_t)inC * (size_t)inH * (size_t)inW;
  size_t outElems = (size_t)inC * (size_t)outH * (size_t)outW;

  checkBufSize(input, inElems, "torchlean_cuda_maxpool2d_bwd_stub: input.size mismatch");
  checkBufSize(gradOutput, outElems, "torchlean_cuda_maxpool2d_bwd_stub: gradOutput.size mismatch");

  torchlean_cuda_buffer* dInput = torchlean_cuda_buffer_alloc(inElems);
  const int det = torchlean_cuda_get_deterministic_reductions();
  if (det) {
    // Deterministic algorithm: compute each dInput element exactly once by scanning the relevant
    // output windows and recomputing each window's argmax (ignore padded cells, strict > tie-break).
    for (uint32_t c = 0; c < inC; ++c) {
      for (uint32_t ih = 0; ih < inH; ++ih) {
        for (uint32_t iw = 0; iw < inW; ++iw) {
          float acc = 0.0f;

          int64_t ohMin = ceil_div_i64((int64_t)ih + (int64_t)padding - (int64_t)(kH - 1), (int64_t)stride);
          int64_t ohMax = floor_div_i64((int64_t)ih + (int64_t)padding, (int64_t)stride);
          int64_t owMin = ceil_div_i64((int64_t)iw + (int64_t)padding - (int64_t)(kW - 1), (int64_t)stride);
          int64_t owMax = floor_div_i64((int64_t)iw + (int64_t)padding, (int64_t)stride);

          if (ohMin < 0) ohMin = 0;
          if (owMin < 0) owMin = 0;
          if (ohMax > (int64_t)outH - 1) ohMax = (int64_t)outH - 1;
          if (owMax > (int64_t)outW - 1) owMax = (int64_t)outW - 1;

          if (ohMin <= ohMax && owMin <= owMax) {
            for (int64_t oh = ohMin; oh <= ohMax; ++oh) {
              for (int64_t ow = owMin; ow <= owMax; ++ow) {
                float best = -FLT_MAX;
                int bestIh = -1;
                int bestIw = -1;

                for (uint32_t ky = 0; ky < kH; ++ky) {
                  int candH = (int)((int64_t)oh * (int64_t)stride + (int64_t)ky - (int64_t)padding);
                  for (uint32_t kx = 0; kx < kW; ++kx) {
                    int candW = (int)((int64_t)ow * (int64_t)stride + (int64_t)kx - (int64_t)padding);
                    float v = -INFINITY;
                    if (candH >= 0 && candH < (int)inH && candW >= 0 && candW < (int)inW) {
                      size_t inIdx =
                          ((size_t)c * (size_t)inH + (size_t)candH) * (size_t)inW + (size_t)candW;
                      v = input->data[inIdx];
                    }
                    if (v > best) {
                      best = v;
                      bestIh = candH;
                      bestIw = candW;
                    }
                  }
                }

                if (bestIh == (int)ih && bestIw == (int)iw) {
                  const size_t outIdx =
                      ((size_t)c * (size_t)outH + (size_t)oh) * (size_t)outW + (size_t)ow;
                  acc += gradOutput->data[outIdx];
                }
              }
            }
          }

          const size_t inIdx = ((size_t)c * (size_t)inH + (size_t)ih) * (size_t)inW + (size_t)iw;
          dInput->data[inIdx] = acc;
        }
      }
    }
  } else {
    // Default algorithm: scatter-add from each output element into dInput (fast).
    // (On CPU this is still deterministic, but this branch matches the CUDA "atomicAdd" structure.)
    for (size_t i = 0; i < dInput->size; ++i) {
      dInput->data[i] = 0.0f;
    }

    for (uint32_t c = 0; c < inC; ++c) {
      for (uint32_t oh = 0; oh < outH; ++oh) {
        for (uint32_t ow = 0; ow < outW; ++ow) {
          size_t outIdx = ((size_t)c * (size_t)outH + (size_t)oh) * (size_t)outW + (size_t)ow;

          float best = -FLT_MAX;
          int bestIh = -1;
          int bestIw = -1;

          for (uint32_t ky = 0; ky < kH; ++ky) {
            int ih = (int)((int64_t)oh * (int64_t)stride + (int64_t)ky - (int64_t)padding);
            for (uint32_t kx = 0; kx < kW; ++kx) {
              int iw = (int)((int64_t)ow * (int64_t)stride + (int64_t)kx - (int64_t)padding);
              float v = -INFINITY;
              if (ih >= 0 && ih < (int)inH && iw >= 0 && iw < (int)inW) {
                size_t inIdx =
                    ((size_t)c * (size_t)inH + (size_t)ih) * (size_t)inW + (size_t)iw;
                v = input->data[inIdx];
              }
              // Tie-breaking: keep the first (row-major) argmax (strict >).
              if (v > best) {
                best = v;
                bestIh = ih;
                bestIw = iw;
              }
            }
          }

          if (bestIh >= 0 && bestIh < (int)inH && bestIw >= 0 && bestIw < (int)inW) {
            size_t inIdx =
                ((size_t)c * (size_t)inH + (size_t)bestIh) * (size_t)inW + (size_t)bestIw;
            dInput->data[inIdx] += gradOutput->data[outIdx];
          }
        }
      }
    }
  }

  return torchlean_cuda_buffer_box(dInput);
}

// -------------------------
// N-D Conv + Pooling entrypoints (CPU stub)
// -------------------------

LEAN_EXPORT lean_obj_res torchlean_cuda_conv_fwd(
    b_lean_obj_arg inputObj, b_lean_obj_arg kernelObj, b_lean_obj_arg biasObj,
    b_lean_obj_arg inSpatialObj, b_lean_obj_arg kernelSpatialObj,
    b_lean_obj_arg strideObj, b_lean_obj_arg paddingObj,
    uint32_t inC, uint32_t outC) {
  torchlean_cuda_buffer* input = torchlean_cuda_buffer_unbox(inputObj);
  torchlean_cuda_buffer* kernel = torchlean_cuda_buffer_unbox(kernelObj);
  torchlean_cuda_buffer* bias = torchlean_cuda_buffer_unbox(biasObj);

  const int rank = read_rank_checked(inSpatialObj, "torchlean_cuda_conv_fwd_stub: bad inSpatial");
  if (read_rank_checked(kernelSpatialObj, "torchlean_cuda_conv_fwd_stub: bad kernelSpatial") != rank ||
      read_rank_checked(strideObj, "torchlean_cuda_conv_fwd_stub: bad stride") != rank ||
      read_rank_checked(paddingObj, "torchlean_cuda_conv_fwd_stub: bad padding") != rank) {
    lean_internal_panic("torchlean_cuda_conv_fwd_stub: array rank mismatch");
  }
  if (rank <= 0) {
    lean_internal_panic("torchlean_cuda_conv_fwd_stub: rank must be > 0");
  }

  uint32_t inSpatial[K_MAX_RANK];
  uint32_t kSpatial[K_MAX_RANK];
  uint32_t stride[K_MAX_RANK];
  uint32_t padding[K_MAX_RANK];
  uint32_t outSpatial[K_MAX_RANK];

  read_u32_array(inSpatialObj, inSpatial, rank, "torchlean_cuda_conv_fwd_stub: bad inSpatial");
  read_u32_array(kernelSpatialObj, kSpatial, rank, "torchlean_cuda_conv_fwd_stub: bad kernelSpatial");
  read_u32_array(strideObj, stride, rank, "torchlean_cuda_conv_fwd_stub: bad stride");
  read_u32_array(paddingObj, padding, rank, "torchlean_cuda_conv_fwd_stub: bad padding");

  for (int ax = 0; ax < rank; ++ax) {
    if (kSpatial[ax] == 0) {
      lean_internal_panic("torchlean_cuda_conv_fwd_stub: kernel dims must be > 0");
    }
    if (stride[ax] == 0) {
      lean_internal_panic("torchlean_cuda_conv_fwd_stub: stride dims must be > 0");
    }
    outSpatial[ax] = outDim(inSpatial[ax], kSpatial[ax], stride[ax], padding[ax]);
  }

  const size_t inSpatialSize = prod_u32(inSpatial, rank);
  const size_t kSpatialSize = prod_u32(kSpatial, rank);
  const size_t outSpatialSize = prod_u32(outSpatial, rank);

  const size_t inElems = (size_t)inC * inSpatialSize;
  const size_t kElems = (size_t)outC * (size_t)inC * kSpatialSize;
  const size_t bElems = (size_t)outC;
  const size_t outElems = (size_t)outC * outSpatialSize;

  checkBufSize(input, inElems, "torchlean_cuda_conv_fwd_stub: input.size mismatch");
  checkBufSize(kernel, kElems, "torchlean_cuda_conv_fwd_stub: kernel.size mismatch");
  checkBufSize(bias, bElems, "torchlean_cuda_conv_fwd_stub: bias.size mismatch");

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outElems);
  if (outElems == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  uint32_t outCoord[K_MAX_RANK];
  uint32_t kCoord[K_MAX_RANK];

  for (uint32_t oc = 0; oc < outC; ++oc) {
    for (size_t outIdx = 0; outIdx < outSpatialSize; ++outIdx) {
      unflatten_coords(outCoord, outSpatial, rank, outIdx);
      float acc = bias->data[oc];

      for (uint32_t ic = 0; ic < inC; ++ic) {
        for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
          unflatten_coords(kCoord, kSpatial, rank, kIdx);

          int ok = 1;
          size_t inIdx = (size_t)ic;
          for (int ax = 0; ax < rank; ++ax) {
            int64_t pos = (int64_t)outCoord[ax] * (int64_t)stride[ax] + (int64_t)kCoord[ax] -
                          (int64_t)padding[ax];
            if (pos < 0 || (uint32_t)pos >= inSpatial[ax]) {
              ok = 0;
              break;
            }
            inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)(uint32_t)pos;
          }
          if (!ok) continue;

          size_t wIdx = (size_t)oc;
          wIdx = wIdx * (size_t)inC + (size_t)ic;
          for (int ax = 0; ax < rank; ++ax) {
            wIdx = wIdx * (size_t)kSpatial[ax] + (size_t)kCoord[ax];
          }

          acc += input->data[inIdx] * kernel->data[wIdx];
        }
      }

      out->data[(size_t)oc * outSpatialSize + outIdx] = acc;
    }
  }

  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_conv_bwd(
    b_lean_obj_arg inputObj, b_lean_obj_arg kernelObj, b_lean_obj_arg gradObj,
    b_lean_obj_arg inSpatialObj, b_lean_obj_arg kernelSpatialObj,
    b_lean_obj_arg strideObj, b_lean_obj_arg paddingObj,
    uint32_t inC, uint32_t outC) {
  torchlean_cuda_buffer* input = torchlean_cuda_buffer_unbox(inputObj);
  torchlean_cuda_buffer* kernel = torchlean_cuda_buffer_unbox(kernelObj);
  torchlean_cuda_buffer* gradOutput = torchlean_cuda_buffer_unbox(gradObj);

  const int rank = read_rank_checked(inSpatialObj, "torchlean_cuda_conv_bwd_stub: bad inSpatial");
  if (read_rank_checked(kernelSpatialObj, "torchlean_cuda_conv_bwd_stub: bad kernelSpatial") != rank ||
      read_rank_checked(strideObj, "torchlean_cuda_conv_bwd_stub: bad stride") != rank ||
      read_rank_checked(paddingObj, "torchlean_cuda_conv_bwd_stub: bad padding") != rank) {
    lean_internal_panic("torchlean_cuda_conv_bwd_stub: array rank mismatch");
  }
  if (rank <= 0) {
    lean_internal_panic("torchlean_cuda_conv_bwd_stub: rank must be > 0");
  }

  uint32_t inSpatial[K_MAX_RANK];
  uint32_t kSpatial[K_MAX_RANK];
  uint32_t stride[K_MAX_RANK];
  uint32_t padding[K_MAX_RANK];
  uint32_t outSpatial[K_MAX_RANK];

  read_u32_array(inSpatialObj, inSpatial, rank, "torchlean_cuda_conv_bwd_stub: bad inSpatial");
  read_u32_array(kernelSpatialObj, kSpatial, rank, "torchlean_cuda_conv_bwd_stub: bad kernelSpatial");
  read_u32_array(strideObj, stride, rank, "torchlean_cuda_conv_bwd_stub: bad stride");
  read_u32_array(paddingObj, padding, rank, "torchlean_cuda_conv_bwd_stub: bad padding");

  for (int ax = 0; ax < rank; ++ax) {
    if (kSpatial[ax] == 0) {
      lean_internal_panic("torchlean_cuda_conv_bwd_stub: kernel dims must be > 0");
    }
    if (stride[ax] == 0) {
      lean_internal_panic("torchlean_cuda_conv_bwd_stub: stride dims must be > 0");
    }
    outSpatial[ax] = outDim(inSpatial[ax], kSpatial[ax], stride[ax], padding[ax]);
  }

  const size_t inSpatialSize = prod_u32(inSpatial, rank);
  const size_t kSpatialSize = prod_u32(kSpatial, rank);
  const size_t outSpatialSize = prod_u32(outSpatial, rank);

  const size_t inElems = (size_t)inC * inSpatialSize;
  const size_t kElems = (size_t)outC * (size_t)inC * kSpatialSize;
  const size_t outElems = (size_t)outC * outSpatialSize;

  checkBufSize(input, inElems, "torchlean_cuda_conv_bwd_stub: input.size mismatch");
  checkBufSize(kernel, kElems, "torchlean_cuda_conv_bwd_stub: kernel.size mismatch");
  checkBufSize(gradOutput, outElems, "torchlean_cuda_conv_bwd_stub: gradOutput.size mismatch");

  torchlean_cuda_buffer* dKernel = torchlean_cuda_buffer_alloc(kElems);
  torchlean_cuda_buffer* dBias = torchlean_cuda_buffer_alloc((size_t)outC);
  torchlean_cuda_buffer* dInput = torchlean_cuda_buffer_alloc(inElems);

  for (size_t i = 0; i < dKernel->size; ++i) dKernel->data[i] = 0.0f;
  for (size_t i = 0; i < dBias->size; ++i) dBias->data[i] = 0.0f;
  for (size_t i = 0; i < dInput->size; ++i) dInput->data[i] = 0.0f;

  uint32_t outCoord[K_MAX_RANK];
  uint32_t kCoord[K_MAX_RANK];

  for (uint32_t oc = 0; oc < outC; ++oc) {
    for (size_t outIdx = 0; outIdx < outSpatialSize; ++outIdx) {
      const size_t goIdx = (size_t)oc * outSpatialSize + outIdx;
      const float go = gradOutput->data[goIdx];
      if (dBias->size > 0) {
        dBias->data[oc] += go;
      }

      unflatten_coords(outCoord, outSpatial, rank, outIdx);

      for (uint32_t ic = 0; ic < inC; ++ic) {
        for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
          unflatten_coords(kCoord, kSpatial, rank, kIdx);

          int ok = 1;
          size_t inIdx = (size_t)ic;
          for (int ax = 0; ax < rank; ++ax) {
            int64_t pos = (int64_t)outCoord[ax] * (int64_t)stride[ax] + (int64_t)kCoord[ax] -
                          (int64_t)padding[ax];
            if (pos < 0 || (uint32_t)pos >= inSpatial[ax]) {
              ok = 0;
              break;
            }
            inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)(uint32_t)pos;
          }
          if (!ok) continue;

          size_t wIdx = (size_t)oc;
          wIdx = wIdx * (size_t)inC + (size_t)ic;
          for (int ax = 0; ax < rank; ++ax) {
            wIdx = wIdx * (size_t)kSpatial[ax] + (size_t)kCoord[ax];
          }

          dKernel->data[wIdx] += input->data[inIdx] * go;
          dInput->data[inIdx] += kernel->data[wIdx] * go;
        }
      }
    }
  }

  lean_object* inner = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(inner, 0, torchlean_cuda_buffer_box(dBias));
  lean_ctor_set(inner, 1, torchlean_cuda_buffer_box(dInput));

  lean_object* outer = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(outer, 0, torchlean_cuda_buffer_box(dKernel));
  lean_ctor_set(outer, 1, inner);
  return outer;
}

LEAN_EXPORT lean_obj_res torchlean_cuda_convtranspose_fwd(
    b_lean_obj_arg inputObj, b_lean_obj_arg kernelObj, b_lean_obj_arg biasObj,
    b_lean_obj_arg inSpatialObj, b_lean_obj_arg kernelSpatialObj,
    b_lean_obj_arg strideObj, b_lean_obj_arg paddingObj,
    uint32_t inC, uint32_t outC) {
  torchlean_cuda_buffer* input = torchlean_cuda_buffer_unbox(inputObj);
  torchlean_cuda_buffer* kernel = torchlean_cuda_buffer_unbox(kernelObj);
  torchlean_cuda_buffer* bias = torchlean_cuda_buffer_unbox(biasObj);

  const int rank =
      read_rank_checked(inSpatialObj, "torchlean_cuda_convtranspose_fwd_stub: bad inSpatial");
  if (read_rank_checked(kernelSpatialObj, "torchlean_cuda_convtranspose_fwd_stub: bad kernelSpatial") !=
          rank ||
      read_rank_checked(strideObj, "torchlean_cuda_convtranspose_fwd_stub: bad stride") != rank ||
      read_rank_checked(paddingObj, "torchlean_cuda_convtranspose_fwd_stub: bad padding") != rank) {
    lean_internal_panic("torchlean_cuda_convtranspose_fwd_stub: array rank mismatch");
  }
  if (rank <= 0) {
    lean_internal_panic("torchlean_cuda_convtranspose_fwd_stub: rank must be > 0");
  }

  uint32_t inSpatial[K_MAX_RANK];
  uint32_t kSpatial[K_MAX_RANK];
  uint32_t stride[K_MAX_RANK];
  uint32_t padding[K_MAX_RANK];
  uint32_t outSpatial[K_MAX_RANK];

  read_u32_array(inSpatialObj, inSpatial, rank, "torchlean_cuda_convtranspose_fwd_stub: bad inSpatial");
  read_u32_array(kernelSpatialObj, kSpatial, rank,
                 "torchlean_cuda_convtranspose_fwd_stub: bad kernelSpatial");
  read_u32_array(strideObj, stride, rank, "torchlean_cuda_convtranspose_fwd_stub: bad stride");
  read_u32_array(paddingObj, padding, rank, "torchlean_cuda_convtranspose_fwd_stub: bad padding");

  for (int ax = 0; ax < rank; ++ax) {
    if (kSpatial[ax] == 0) {
      lean_internal_panic("torchlean_cuda_convtranspose_fwd_stub: kernel dims must be > 0");
    }
    if (stride[ax] == 0) {
      lean_internal_panic("torchlean_cuda_convtranspose_fwd_stub: stride dims must be > 0");
    }
    outSpatial[ax] = outDimTranspose(inSpatial[ax], kSpatial[ax], stride[ax], padding[ax]);
  }

  const size_t inSpatialSize = prod_u32(inSpatial, rank);
  const size_t kSpatialSize = prod_u32(kSpatial, rank);
  const size_t outSpatialSize = prod_u32(outSpatial, rank);

  const size_t inElems = (size_t)inC * inSpatialSize;
  const size_t kElems = (size_t)inC * (size_t)outC * kSpatialSize;
  const size_t bElems = (size_t)outC;
  const size_t outElems = (size_t)outC * outSpatialSize;

  checkBufSize(input, inElems, "torchlean_cuda_convtranspose_fwd_stub: input.size mismatch");
  checkBufSize(kernel, kElems, "torchlean_cuda_convtranspose_fwd_stub: kernel.size mismatch");
  checkBufSize(bias, bElems, "torchlean_cuda_convtranspose_fwd_stub: bias.size mismatch");

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outElems);
  if (outElems == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  uint32_t outCoord[K_MAX_RANK];
  uint32_t kCoord[K_MAX_RANK];

  // Output layout: (outC, outSpatial...); kernel layout: (inC, outC, kSpatial...).
  for (uint32_t oc = 0; oc < outC; ++oc) {
    for (size_t outIdx = 0; outIdx < outSpatialSize; ++outIdx) {
      unflatten_coords(outCoord, outSpatial, rank, outIdx);
      float acc = bias->data[oc];

      for (uint32_t ic = 0; ic < inC; ++ic) {
        for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
          unflatten_coords(kCoord, kSpatial, rank, kIdx);

          // In transpose-conv, outputCoord = inputCoord*stride + kCoord - padding.
          // Solve for inputCoord: (outputCoord + padding - kCoord) / stride, requiring divisibility.
          int ok = 1;
          size_t inIdx = (size_t)ic;
          for (int ax = 0; ax < rank; ++ax) {
            int64_t num =
                (int64_t)outCoord[ax] + (int64_t)padding[ax] - (int64_t)kCoord[ax];
            if (num < 0) {
              ok = 0;
              break;
            }
            int64_t s = (int64_t)stride[ax];
            if ((num % s) != 0) {
              ok = 0;
              break;
            }
            int64_t pos = num / s;
            if (pos < 0 || (uint32_t)pos >= inSpatial[ax]) {
              ok = 0;
              break;
            }
            inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)(uint32_t)pos;
          }
          if (!ok) continue;

          size_t wIdx = (size_t)ic;
          wIdx = wIdx * (size_t)outC + (size_t)oc;
          for (int ax = 0; ax < rank; ++ax) {
            wIdx = wIdx * (size_t)kSpatial[ax] + (size_t)kCoord[ax];
          }

          acc += input->data[inIdx] * kernel->data[wIdx];
        }
      }

      out->data[(size_t)oc * outSpatialSize + outIdx] = acc;
    }
  }

  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_convtranspose_bwd(
    b_lean_obj_arg inputObj, b_lean_obj_arg kernelObj, b_lean_obj_arg gradObj,
    b_lean_obj_arg inSpatialObj, b_lean_obj_arg kernelSpatialObj,
    b_lean_obj_arg strideObj, b_lean_obj_arg paddingObj,
    uint32_t inC, uint32_t outC) {
  torchlean_cuda_buffer* input = torchlean_cuda_buffer_unbox(inputObj);
  torchlean_cuda_buffer* kernel = torchlean_cuda_buffer_unbox(kernelObj);
  torchlean_cuda_buffer* gradOutput = torchlean_cuda_buffer_unbox(gradObj);

  const int rank =
      read_rank_checked(inSpatialObj, "torchlean_cuda_convtranspose_bwd_stub: bad inSpatial");
  if (read_rank_checked(kernelSpatialObj, "torchlean_cuda_convtranspose_bwd_stub: bad kernelSpatial") !=
          rank ||
      read_rank_checked(strideObj, "torchlean_cuda_convtranspose_bwd_stub: bad stride") != rank ||
      read_rank_checked(paddingObj, "torchlean_cuda_convtranspose_bwd_stub: bad padding") != rank) {
    lean_internal_panic("torchlean_cuda_convtranspose_bwd_stub: array rank mismatch");
  }
  if (rank <= 0) {
    lean_internal_panic("torchlean_cuda_convtranspose_bwd_stub: rank must be > 0");
  }

  uint32_t inSpatial[K_MAX_RANK];
  uint32_t kSpatial[K_MAX_RANK];
  uint32_t stride[K_MAX_RANK];
  uint32_t padding[K_MAX_RANK];
  uint32_t outSpatial[K_MAX_RANK];

  read_u32_array(inSpatialObj, inSpatial, rank, "torchlean_cuda_convtranspose_bwd_stub: bad inSpatial");
  read_u32_array(kernelSpatialObj, kSpatial, rank,
                 "torchlean_cuda_convtranspose_bwd_stub: bad kernelSpatial");
  read_u32_array(strideObj, stride, rank, "torchlean_cuda_convtranspose_bwd_stub: bad stride");
  read_u32_array(paddingObj, padding, rank, "torchlean_cuda_convtranspose_bwd_stub: bad padding");

  for (int ax = 0; ax < rank; ++ax) {
    if (kSpatial[ax] == 0) {
      lean_internal_panic("torchlean_cuda_convtranspose_bwd_stub: kernel dims must be > 0");
    }
    if (stride[ax] == 0) {
      lean_internal_panic("torchlean_cuda_convtranspose_bwd_stub: stride dims must be > 0");
    }
    outSpatial[ax] = outDimTranspose(inSpatial[ax], kSpatial[ax], stride[ax], padding[ax]);
  }

  const size_t inSpatialSize = prod_u32(inSpatial, rank);
  const size_t kSpatialSize = prod_u32(kSpatial, rank);
  const size_t outSpatialSize = prod_u32(outSpatial, rank);

  const size_t inElems = (size_t)inC * inSpatialSize;
  const size_t kElems = (size_t)inC * (size_t)outC * kSpatialSize;
  const size_t outElems = (size_t)outC * outSpatialSize;

  checkBufSize(input, inElems, "torchlean_cuda_convtranspose_bwd_stub: input.size mismatch");
  checkBufSize(kernel, kElems, "torchlean_cuda_convtranspose_bwd_stub: kernel.size mismatch");
  checkBufSize(gradOutput, outElems, "torchlean_cuda_convtranspose_bwd_stub: gradOutput.size mismatch");

  torchlean_cuda_buffer* dKernel = torchlean_cuda_buffer_alloc(kElems);
  torchlean_cuda_buffer* dBias = torchlean_cuda_buffer_alloc((size_t)outC);
  torchlean_cuda_buffer* dInput = torchlean_cuda_buffer_alloc(inElems);

  for (size_t i = 0; i < dKernel->size; ++i) dKernel->data[i] = 0.0f;
  for (size_t i = 0; i < dBias->size; ++i) dBias->data[i] = 0.0f;
  for (size_t i = 0; i < dInput->size; ++i) dInput->data[i] = 0.0f;

  uint32_t outCoord[K_MAX_RANK];
  uint32_t kCoord[K_MAX_RANK];

  // Accumulate all three gradients in one pass over gradOutput.
  for (uint32_t oc = 0; oc < outC; ++oc) {
    for (size_t outIdx = 0; outIdx < outSpatialSize; ++outIdx) {
      const size_t goIdx = (size_t)oc * outSpatialSize + outIdx;
      const float go = gradOutput->data[goIdx];
      if (dBias->size > 0) {
        dBias->data[oc] += go;
      }

      unflatten_coords(outCoord, outSpatial, rank, outIdx);

      for (uint32_t ic = 0; ic < inC; ++ic) {
        for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
          unflatten_coords(kCoord, kSpatial, rank, kIdx);

          int ok = 1;
          size_t inIdx = (size_t)ic;
          for (int ax = 0; ax < rank; ++ax) {
            int64_t num =
                (int64_t)outCoord[ax] + (int64_t)padding[ax] - (int64_t)kCoord[ax];
            if (num < 0) {
              ok = 0;
              break;
            }
            int64_t s = (int64_t)stride[ax];
            if ((num % s) != 0) {
              ok = 0;
              break;
            }
            int64_t pos = num / s;
            if (pos < 0 || (uint32_t)pos >= inSpatial[ax]) {
              ok = 0;
              break;
            }
            inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)(uint32_t)pos;
          }
          if (!ok) continue;

          size_t wIdx = (size_t)ic;
          wIdx = wIdx * (size_t)outC + (size_t)oc;
          for (int ax = 0; ax < rank; ++ax) {
            wIdx = wIdx * (size_t)kSpatial[ax] + (size_t)kCoord[ax];
          }

          dKernel->data[wIdx] += input->data[inIdx] * go;
          dInput->data[inIdx] += kernel->data[wIdx] * go;
        }
      }
    }
  }

  lean_object* inner = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(inner, 0, torchlean_cuda_buffer_box(dBias));
  lean_ctor_set(inner, 1, torchlean_cuda_buffer_box(dInput));

  lean_object* outer = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(outer, 0, torchlean_cuda_buffer_box(dKernel));
  lean_ctor_set(outer, 1, inner);
  return outer;
}

LEAN_EXPORT lean_obj_res torchlean_cuda_maxpool_fwd(
    b_lean_obj_arg inputObj,
    b_lean_obj_arg inSpatialObj, b_lean_obj_arg kernelObj,
    b_lean_obj_arg strideObj, b_lean_obj_arg paddingObj,
    uint32_t inC) {
  torchlean_cuda_buffer* input = torchlean_cuda_buffer_unbox(inputObj);

  const int rank = read_rank_checked(inSpatialObj, "torchlean_cuda_maxpool_fwd_stub: bad inSpatial");
  if (read_rank_checked(kernelObj, "torchlean_cuda_maxpool_fwd_stub: bad kernel") != rank ||
      read_rank_checked(strideObj, "torchlean_cuda_maxpool_fwd_stub: bad stride") != rank ||
      read_rank_checked(paddingObj, "torchlean_cuda_maxpool_fwd_stub: bad padding") != rank) {
    lean_internal_panic("torchlean_cuda_maxpool_fwd_stub: array rank mismatch");
  }
  if (rank <= 0) {
    lean_internal_panic("torchlean_cuda_maxpool_fwd_stub: rank must be > 0");
  }

  uint32_t inSpatial[K_MAX_RANK];
  uint32_t kSpatial[K_MAX_RANK];
  uint32_t stride[K_MAX_RANK];
  uint32_t padding[K_MAX_RANK];
  uint32_t outSpatial[K_MAX_RANK];

  read_u32_array(inSpatialObj, inSpatial, rank, "torchlean_cuda_maxpool_fwd_stub: bad inSpatial");
  read_u32_array(kernelObj, kSpatial, rank, "torchlean_cuda_maxpool_fwd_stub: bad kernel");
  read_u32_array(strideObj, stride, rank, "torchlean_cuda_maxpool_fwd_stub: bad stride");
  read_u32_array(paddingObj, padding, rank, "torchlean_cuda_maxpool_fwd_stub: bad padding");

  for (int ax = 0; ax < rank; ++ax) {
    if (kSpatial[ax] == 0) {
      lean_internal_panic("torchlean_cuda_maxpool_fwd_stub: kernel dims must be > 0");
    }
    if (stride[ax] == 0) {
      lean_internal_panic("torchlean_cuda_maxpool_fwd_stub: stride dims must be > 0");
    }
    outSpatial[ax] = outDim(inSpatial[ax], kSpatial[ax], stride[ax], padding[ax]);
  }

  const size_t inSpatialSize = prod_u32(inSpatial, rank);
  const size_t kSpatialSize = prod_u32(kSpatial, rank);
  const size_t outSpatialSize = prod_u32(outSpatial, rank);

  const size_t inElems = (size_t)inC * inSpatialSize;
  const size_t outElems = (size_t)inC * outSpatialSize;

  checkBufSize(input, inElems, "torchlean_cuda_maxpool_fwd_stub: input.size mismatch");

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outElems);
  if (outElems == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  uint32_t outCoord[K_MAX_RANK];
  uint32_t kCoord[K_MAX_RANK];

  for (uint32_t c = 0; c < inC; ++c) {
    for (size_t outIdx = 0; outIdx < outSpatialSize; ++outIdx) {
      unflatten_coords(outCoord, outSpatial, rank, outIdx);

      float best = -FLT_MAX;
      for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
        unflatten_coords(kCoord, kSpatial, rank, kIdx);

        int ok = 1;
        size_t inIdx = (size_t)c;
        for (int ax = 0; ax < rank; ++ax) {
          int64_t pos = (int64_t)outCoord[ax] * (int64_t)stride[ax] + (int64_t)kCoord[ax] -
                        (int64_t)padding[ax];
          if (pos < 0 || (uint32_t)pos >= inSpatial[ax]) {
            ok = 0;
            break;
          }
          inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)(uint32_t)pos;
        }

        float v = -INFINITY;
        if (ok) {
          v = input->data[inIdx];
        }
        if (v > best) {
          best = v;
        }
      }

      out->data[(size_t)c * outSpatialSize + outIdx] = best;
    }
  }

  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_maxpool_bwd(
    b_lean_obj_arg inputObj, b_lean_obj_arg gradObj,
    b_lean_obj_arg inSpatialObj, b_lean_obj_arg kernelObj,
    b_lean_obj_arg strideObj, b_lean_obj_arg paddingObj,
    uint32_t inC) {
  torchlean_cuda_buffer* input = torchlean_cuda_buffer_unbox(inputObj);
  torchlean_cuda_buffer* gradOutput = torchlean_cuda_buffer_unbox(gradObj);

  const int rank = read_rank_checked(inSpatialObj, "torchlean_cuda_maxpool_bwd_stub: bad inSpatial");
  if (read_rank_checked(kernelObj, "torchlean_cuda_maxpool_bwd_stub: bad kernel") != rank ||
      read_rank_checked(strideObj, "torchlean_cuda_maxpool_bwd_stub: bad stride") != rank ||
      read_rank_checked(paddingObj, "torchlean_cuda_maxpool_bwd_stub: bad padding") != rank) {
    lean_internal_panic("torchlean_cuda_maxpool_bwd_stub: array rank mismatch");
  }
  if (rank <= 0) {
    lean_internal_panic("torchlean_cuda_maxpool_bwd_stub: rank must be > 0");
  }

  uint32_t inSpatial[K_MAX_RANK];
  uint32_t kSpatial[K_MAX_RANK];
  uint32_t stride[K_MAX_RANK];
  uint32_t padding[K_MAX_RANK];
  uint32_t outSpatial[K_MAX_RANK];

  read_u32_array(inSpatialObj, inSpatial, rank, "torchlean_cuda_maxpool_bwd_stub: bad inSpatial");
  read_u32_array(kernelObj, kSpatial, rank, "torchlean_cuda_maxpool_bwd_stub: bad kernel");
  read_u32_array(strideObj, stride, rank, "torchlean_cuda_maxpool_bwd_stub: bad stride");
  read_u32_array(paddingObj, padding, rank, "torchlean_cuda_maxpool_bwd_stub: bad padding");

  for (int ax = 0; ax < rank; ++ax) {
    if (kSpatial[ax] == 0) {
      lean_internal_panic("torchlean_cuda_maxpool_bwd_stub: kernel dims must be > 0");
    }
    if (stride[ax] == 0) {
      lean_internal_panic("torchlean_cuda_maxpool_bwd_stub: stride dims must be > 0");
    }
    outSpatial[ax] = outDim(inSpatial[ax], kSpatial[ax], stride[ax], padding[ax]);
  }

  const size_t inSpatialSize = prod_u32(inSpatial, rank);
  const size_t kSpatialSize = prod_u32(kSpatial, rank);
  const size_t outSpatialSize = prod_u32(outSpatial, rank);

  const size_t inElems = (size_t)inC * inSpatialSize;
  const size_t outElems = (size_t)inC * outSpatialSize;

  checkBufSize(input, inElems, "torchlean_cuda_maxpool_bwd_stub: input.size mismatch");
  checkBufSize(gradOutput, outElems, "torchlean_cuda_maxpool_bwd_stub: gradOutput.size mismatch");

  torchlean_cuda_buffer* dInput = torchlean_cuda_buffer_alloc(inElems);
  const int det = torchlean_cuda_get_deterministic_reductions();
  if (det) {
    // Deterministic algorithm: compute each dInput element exactly once.
    uint32_t inCoord[K_MAX_RANK];
    uint32_t outCoord[K_MAX_RANK];
    uint32_t kCoord[K_MAX_RANK];
    uint32_t bestCoord[K_MAX_RANK];
    uint32_t candCoord[K_MAX_RANK];
    int64_t oMin[K_MAX_RANK];
    int64_t oMax[K_MAX_RANK];
    size_t rangeSize[K_MAX_RANK];

    for (uint32_t c = 0; c < inC; ++c) {
      for (size_t spatialIdx = 0; spatialIdx < inSpatialSize; ++spatialIdx) {
        unflatten_coords(inCoord, inSpatial, rank, spatialIdx);

        size_t totalComb = 1;
        for (int ax = 0; ax < rank; ++ax) {
          const int64_t s = (int64_t)stride[ax];
          const int64_t p = (int64_t)padding[ax];
          const int64_t kd = (int64_t)kSpatial[ax];
          int64_t lo = ceil_div_i64((int64_t)inCoord[ax] + p - (kd - 1), s);
          int64_t hi = floor_div_i64((int64_t)inCoord[ax] + p, s);
          if (lo < 0) lo = 0;
          const int64_t outD = (int64_t)outSpatial[ax];
          if (hi > outD - 1) hi = outD - 1;
          oMin[ax] = lo;
          oMax[ax] = hi;
          if (lo > hi) {
            totalComb = 0;
            break;
          }
          rangeSize[ax] = (size_t)(hi - lo + 1);
          totalComb *= rangeSize[ax];
        }

        float acc = 0.0f;
        if (totalComb > 0) {
          for (size_t t = 0; t < totalComb; ++t) {
            size_t tt = t;
            for (int ax = rank - 1; ax >= 0; --ax) {
              const size_t sz = rangeSize[ax];
              const size_t off = (sz == 0) ? 0 : (tt % sz);
              tt = (sz == 0) ? 0 : (tt / sz);
              outCoord[ax] = (uint32_t)(oMin[ax] + (int64_t)off);
            }

            // Exact inclusion check for this output coordinate.
            int includes = 1;
            for (int ax = 0; ax < rank; ++ax) {
              int64_t k = (int64_t)inCoord[ax] + (int64_t)padding[ax] -
                          (int64_t)outCoord[ax] * (int64_t)stride[ax];
              if (k < 0 || k >= (int64_t)kSpatial[ax]) {
                includes = 0;
                break;
              }
            }
            if (!includes) continue;

            const size_t outIdx = flatten_coords(outCoord, outSpatial, rank);

            float best = -FLT_MAX;
            int bestValid = 0;
            for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
              unflatten_coords(kCoord, kSpatial, rank, kIdx);

              int ok = 1;
              for (int ax = 0; ax < rank; ++ax) {
                int64_t pos = (int64_t)outCoord[ax] * (int64_t)stride[ax] + (int64_t)kCoord[ax] -
                              (int64_t)padding[ax];
                if (pos < 0 || (uint32_t)pos >= inSpatial[ax]) {
                  ok = 0;
                  break;
                }
                candCoord[ax] = (uint32_t)pos;
              }

              float v = -INFINITY;
              if (ok) {
                size_t inIdx = (size_t)c;
                for (int ax = 0; ax < rank; ++ax) {
                  inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)candCoord[ax];
                }
                v = input->data[inIdx];
              }

              if (v > best) {
                best = v;
                bestValid = ok;
                if (ok) {
                  for (int ax = 0; ax < rank; ++ax) bestCoord[ax] = candCoord[ax];
                }
              }
            }

            if (!bestValid) continue;
            int match = 1;
            for (int ax = 0; ax < rank; ++ax) {
              if (bestCoord[ax] != inCoord[ax]) {
                match = 0;
                break;
              }
            }
            if (match) {
              acc += gradOutput->data[(size_t)c * outSpatialSize + outIdx];
            }
          }
        }

        dInput->data[(size_t)c * inSpatialSize + spatialIdx] = acc;
      }
    }
  } else {
    // Default scatter-style algorithm.
    for (size_t i = 0; i < dInput->size; ++i) dInput->data[i] = 0.0f;
    if (outElems == 0) {
      return torchlean_cuda_buffer_box(dInput);
    }

    uint32_t outCoord[K_MAX_RANK];
    uint32_t kCoord[K_MAX_RANK];
    uint32_t bestCoord[K_MAX_RANK];
    uint32_t candCoord[K_MAX_RANK];

    for (uint32_t c = 0; c < inC; ++c) {
      for (size_t outIdx = 0; outIdx < outSpatialSize; ++outIdx) {
        unflatten_coords(outCoord, outSpatial, rank, outIdx);

        float best = -FLT_MAX;
        int bestValid = 0;

        for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
          unflatten_coords(kCoord, kSpatial, rank, kIdx);

          int ok = 1;
          for (int ax = 0; ax < rank; ++ax) {
            int64_t pos = (int64_t)outCoord[ax] * (int64_t)stride[ax] + (int64_t)kCoord[ax] -
                          (int64_t)padding[ax];
            if (pos < 0 || (uint32_t)pos >= inSpatial[ax]) {
              ok = 0;
              break;
            }
            candCoord[ax] = (uint32_t)pos;
          }

          float v = -INFINITY;
          if (ok) {
            size_t inIdx = (size_t)c;
            for (int ax = 0; ax < rank; ++ax) {
              inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)candCoord[ax];
            }
            v = input->data[inIdx];
          }

          if (v > best) {
            best = v;
            bestValid = ok;
            if (ok) {
              for (int ax = 0; ax < rank; ++ax) bestCoord[ax] = candCoord[ax];
            }
          }
        }

        if (bestValid) {
          size_t inIdx = (size_t)c;
          for (int ax = 0; ax < rank; ++ax) {
            inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)bestCoord[ax];
          }
          dInput->data[inIdx] += gradOutput->data[(size_t)c * outSpatialSize + outIdx];
        }
      }
    }
  }

  return torchlean_cuda_buffer_box(dInput);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_avgpool_fwd(
    b_lean_obj_arg inputObj,
    b_lean_obj_arg inSpatialObj, b_lean_obj_arg kernelObj,
    b_lean_obj_arg strideObj, b_lean_obj_arg paddingObj,
    uint32_t inC) {
  torchlean_cuda_buffer* input = torchlean_cuda_buffer_unbox(inputObj);

  const int rank = read_rank_checked(inSpatialObj, "torchlean_cuda_avgpool_fwd_stub: bad inSpatial");
  if (read_rank_checked(kernelObj, "torchlean_cuda_avgpool_fwd_stub: bad kernel") != rank ||
      read_rank_checked(strideObj, "torchlean_cuda_avgpool_fwd_stub: bad stride") != rank ||
      read_rank_checked(paddingObj, "torchlean_cuda_avgpool_fwd_stub: bad padding") != rank) {
    lean_internal_panic("torchlean_cuda_avgpool_fwd_stub: array rank mismatch");
  }
  if (rank <= 0) {
    lean_internal_panic("torchlean_cuda_avgpool_fwd_stub: rank must be > 0");
  }

  uint32_t inSpatial[K_MAX_RANK];
  uint32_t kSpatial[K_MAX_RANK];
  uint32_t stride[K_MAX_RANK];
  uint32_t padding[K_MAX_RANK];
  uint32_t outSpatial[K_MAX_RANK];

  read_u32_array(inSpatialObj, inSpatial, rank, "torchlean_cuda_avgpool_fwd_stub: bad inSpatial");
  read_u32_array(kernelObj, kSpatial, rank, "torchlean_cuda_avgpool_fwd_stub: bad kernel");
  read_u32_array(strideObj, stride, rank, "torchlean_cuda_avgpool_fwd_stub: bad stride");
  read_u32_array(paddingObj, padding, rank, "torchlean_cuda_avgpool_fwd_stub: bad padding");

  for (int ax = 0; ax < rank; ++ax) {
    if (kSpatial[ax] == 0) {
      lean_internal_panic("torchlean_cuda_avgpool_fwd_stub: kernel dims must be > 0");
    }
    if (stride[ax] == 0) {
      lean_internal_panic("torchlean_cuda_avgpool_fwd_stub: stride dims must be > 0");
    }
    outSpatial[ax] = outDim(inSpatial[ax], kSpatial[ax], stride[ax], padding[ax]);
  }

  const size_t inSpatialSize = prod_u32(inSpatial, rank);
  const size_t kSpatialSize = prod_u32(kSpatial, rank);
  const size_t outSpatialSize = prod_u32(outSpatial, rank);

  const size_t inElems = (size_t)inC * inSpatialSize;
  const size_t outElems = (size_t)inC * outSpatialSize;

  checkBufSize(input, inElems, "torchlean_cuda_avgpool_fwd_stub: input.size mismatch");

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outElems);
  if (outElems == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  uint32_t outCoord[K_MAX_RANK];
  uint32_t kCoord[K_MAX_RANK];
  const float denom = (float)kSpatialSize;

  for (uint32_t c = 0; c < inC; ++c) {
    for (size_t outIdx = 0; outIdx < outSpatialSize; ++outIdx) {
      unflatten_coords(outCoord, outSpatial, rank, outIdx);

      float acc = 0.0f;
      for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
        unflatten_coords(kCoord, kSpatial, rank, kIdx);

        int ok = 1;
        size_t inIdx = (size_t)c;
        for (int ax = 0; ax < rank; ++ax) {
          int64_t pos = (int64_t)outCoord[ax] * (int64_t)stride[ax] + (int64_t)kCoord[ax] -
                        (int64_t)padding[ax];
          if (pos < 0 || (uint32_t)pos >= inSpatial[ax]) {
            ok = 0;
            break;
          }
          inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)(uint32_t)pos;
        }

        if (ok) {
          acc += input->data[inIdx];
        }
        // else: out-of-bounds contributes 0 and is still counted in denom.
      }

      out->data[(size_t)c * outSpatialSize + outIdx] = acc / denom;
    }
  }

  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_avgpool_bwd(
    b_lean_obj_arg gradObj,
    b_lean_obj_arg inSpatialObj, b_lean_obj_arg kernelObj,
    b_lean_obj_arg strideObj, b_lean_obj_arg paddingObj,
    uint32_t inC) {
  torchlean_cuda_buffer* gradOutput = torchlean_cuda_buffer_unbox(gradObj);

  const int rank = read_rank_checked(inSpatialObj, "torchlean_cuda_avgpool_bwd_stub: bad inSpatial");
  if (read_rank_checked(kernelObj, "torchlean_cuda_avgpool_bwd_stub: bad kernel") != rank ||
      read_rank_checked(strideObj, "torchlean_cuda_avgpool_bwd_stub: bad stride") != rank ||
      read_rank_checked(paddingObj, "torchlean_cuda_avgpool_bwd_stub: bad padding") != rank) {
    lean_internal_panic("torchlean_cuda_avgpool_bwd_stub: array rank mismatch");
  }
  if (rank <= 0) {
    lean_internal_panic("torchlean_cuda_avgpool_bwd_stub: rank must be > 0");
  }

  uint32_t inSpatial[K_MAX_RANK];
  uint32_t kSpatial[K_MAX_RANK];
  uint32_t stride[K_MAX_RANK];
  uint32_t padding[K_MAX_RANK];
  uint32_t outSpatial[K_MAX_RANK];

  read_u32_array(inSpatialObj, inSpatial, rank, "torchlean_cuda_avgpool_bwd_stub: bad inSpatial");
  read_u32_array(kernelObj, kSpatial, rank, "torchlean_cuda_avgpool_bwd_stub: bad kernel");
  read_u32_array(strideObj, stride, rank, "torchlean_cuda_avgpool_bwd_stub: bad stride");
  read_u32_array(paddingObj, padding, rank, "torchlean_cuda_avgpool_bwd_stub: bad padding");

  for (int ax = 0; ax < rank; ++ax) {
    if (kSpatial[ax] == 0) {
      lean_internal_panic("torchlean_cuda_avgpool_bwd_stub: kernel dims must be > 0");
    }
    if (stride[ax] == 0) {
      lean_internal_panic("torchlean_cuda_avgpool_bwd_stub: stride dims must be > 0");
    }
    outSpatial[ax] = outDim(inSpatial[ax], kSpatial[ax], stride[ax], padding[ax]);
  }

  const size_t inSpatialSize = prod_u32(inSpatial, rank);
  const size_t kSpatialSize = prod_u32(kSpatial, rank);
  const size_t outSpatialSize = prod_u32(outSpatial, rank);

  const size_t inElems = (size_t)inC * inSpatialSize;
  const size_t outElems = (size_t)inC * outSpatialSize;

  checkBufSize(gradOutput, outElems, "torchlean_cuda_avgpool_bwd_stub: gradOutput.size mismatch");

  torchlean_cuda_buffer* dInput = torchlean_cuda_buffer_alloc(inElems);
  const float denom = (float)kSpatialSize;
  const int det = torchlean_cuda_get_deterministic_reductions();
  if (det) {
    // Deterministic algorithm: compute each dInput element exactly once.
    uint32_t inCoord[K_MAX_RANK];
    uint32_t outCoord[K_MAX_RANK];
    int64_t oMin[K_MAX_RANK];
    int64_t oMax[K_MAX_RANK];
    size_t rangeSize[K_MAX_RANK];

    for (uint32_t c = 0; c < inC; ++c) {
      for (size_t spatialIdx = 0; spatialIdx < inSpatialSize; ++spatialIdx) {
        unflatten_coords(inCoord, inSpatial, rank, spatialIdx);

        size_t totalComb = 1;
        for (int ax = 0; ax < rank; ++ax) {
          const int64_t s = (int64_t)stride[ax];
          const int64_t p = (int64_t)padding[ax];
          const int64_t kd = (int64_t)kSpatial[ax];
          int64_t lo = ceil_div_i64((int64_t)inCoord[ax] + p - (kd - 1), s);
          int64_t hi = floor_div_i64((int64_t)inCoord[ax] + p, s);
          if (lo < 0) lo = 0;
          const int64_t outD = (int64_t)outSpatial[ax];
          if (hi > outD - 1) hi = outD - 1;
          oMin[ax] = lo;
          oMax[ax] = hi;
          if (lo > hi) {
            totalComb = 0;
            break;
          }
          rangeSize[ax] = (size_t)(hi - lo + 1);
          totalComb *= rangeSize[ax];
        }

        float acc = 0.0f;
        if (totalComb > 0) {
          for (size_t t = 0; t < totalComb; ++t) {
            size_t tt = t;
            for (int ax = rank - 1; ax >= 0; --ax) {
              const size_t sz = rangeSize[ax];
              const size_t off = (sz == 0) ? 0 : (tt % sz);
              tt = (sz == 0) ? 0 : (tt / sz);
              outCoord[ax] = (uint32_t)(oMin[ax] + (int64_t)off);
            }

            int includes = 1;
            for (int ax = 0; ax < rank; ++ax) {
              int64_t k = (int64_t)inCoord[ax] + (int64_t)padding[ax] -
                          (int64_t)outCoord[ax] * (int64_t)stride[ax];
              if (k < 0 || k >= (int64_t)kSpatial[ax]) {
                includes = 0;
                break;
              }
            }
            if (!includes) continue;

            const size_t outIdx = flatten_coords(outCoord, outSpatial, rank);
            acc += gradOutput->data[(size_t)c * outSpatialSize + outIdx] / denom;
          }
        }

        dInput->data[(size_t)c * inSpatialSize + spatialIdx] = acc;
      }
    }
  } else {
    // Default scatter-style algorithm.
    for (size_t i = 0; i < dInput->size; ++i) dInput->data[i] = 0.0f;
    if (outElems == 0) {
      return torchlean_cuda_buffer_box(dInput);
    }

    uint32_t outCoord[K_MAX_RANK];
    uint32_t kCoord[K_MAX_RANK];

    for (uint32_t c = 0; c < inC; ++c) {
      for (size_t outIdx = 0; outIdx < outSpatialSize; ++outIdx) {
        const float g = gradOutput->data[(size_t)c * outSpatialSize + outIdx] / denom;
        unflatten_coords(outCoord, outSpatial, rank, outIdx);

        for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
          unflatten_coords(kCoord, kSpatial, rank, kIdx);

          int ok = 1;
          size_t inIdx = (size_t)c;
          for (int ax = 0; ax < rank; ++ax) {
            int64_t pos = (int64_t)outCoord[ax] * (int64_t)stride[ax] + (int64_t)kCoord[ax] -
                          (int64_t)padding[ax];
            if (pos < 0 || (uint32_t)pos >= inSpatial[ax]) {
              ok = 0;
              break;
            }
            inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)(uint32_t)pos;
          }
          if (ok) {
            dInput->data[inIdx] += g;
          }
        }
      }
    }
  }

  return torchlean_cuda_buffer_box(dInput);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_smooth_maxpool_fwd(
    b_lean_obj_arg inputObj, double beta,
    b_lean_obj_arg inSpatialObj, b_lean_obj_arg kernelObj,
    b_lean_obj_arg strideObj, b_lean_obj_arg paddingObj,
    uint32_t inC) {
  torchlean_cuda_buffer* input = torchlean_cuda_buffer_unbox(inputObj);

  const int rank = read_rank_checked(inSpatialObj, "torchlean_cuda_smooth_maxpool_fwd_stub: bad inSpatial");
  if (read_rank_checked(kernelObj, "torchlean_cuda_smooth_maxpool_fwd_stub: bad kernel") != rank ||
      read_rank_checked(strideObj, "torchlean_cuda_smooth_maxpool_fwd_stub: bad stride") != rank ||
      read_rank_checked(paddingObj, "torchlean_cuda_smooth_maxpool_fwd_stub: bad padding") != rank) {
    lean_internal_panic("torchlean_cuda_smooth_maxpool_fwd_stub: array rank mismatch");
  }
  if (rank <= 0) {
    lean_internal_panic("torchlean_cuda_smooth_maxpool_fwd_stub: rank must be > 0");
  }

  uint32_t inSpatial[K_MAX_RANK];
  uint32_t kSpatial[K_MAX_RANK];
  uint32_t stride[K_MAX_RANK];
  uint32_t padding[K_MAX_RANK];
  uint32_t outSpatial[K_MAX_RANK];

  read_u32_array(inSpatialObj, inSpatial, rank, "torchlean_cuda_smooth_maxpool_fwd_stub: bad inSpatial");
  read_u32_array(kernelObj, kSpatial, rank, "torchlean_cuda_smooth_maxpool_fwd_stub: bad kernel");
  read_u32_array(strideObj, stride, rank, "torchlean_cuda_smooth_maxpool_fwd_stub: bad stride");
  read_u32_array(paddingObj, padding, rank, "torchlean_cuda_smooth_maxpool_fwd_stub: bad padding");

  for (int ax = 0; ax < rank; ++ax) {
    if (kSpatial[ax] == 0) {
      lean_internal_panic("torchlean_cuda_smooth_maxpool_fwd_stub: kernel dims must be > 0");
    }
    if (stride[ax] == 0) {
      lean_internal_panic("torchlean_cuda_smooth_maxpool_fwd_stub: stride dims must be > 0");
    }
    outSpatial[ax] = outDim(inSpatial[ax], kSpatial[ax], stride[ax], padding[ax]);
  }

  const size_t inSpatialSize = prod_u32(inSpatial, rank);
  const size_t kSpatialSize = prod_u32(kSpatial, rank);
  const size_t outSpatialSize = prod_u32(outSpatial, rank);

  const size_t inElems = (size_t)inC * inSpatialSize;
  const size_t outElems = (size_t)inC * outSpatialSize;

  checkBufSize(input, inElems, "torchlean_cuda_smooth_maxpool_fwd_stub: input.size mismatch");

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outElems);
  if (outElems == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  const float betaF =
      checked_smoothmax_beta(beta, "torchlean_cuda_smooth_maxpool_fwd_stub: beta must be finite and nonzero");

  uint32_t outCoord[K_MAX_RANK];
  uint32_t kCoord[K_MAX_RANK];

  for (uint32_t c = 0; c < inC; ++c) {
    for (size_t outIdx = 0; outIdx < outSpatialSize; ++outIdx) {
      unflatten_coords(outCoord, outSpatial, rank, outIdx);

      float maxScaled = -INFINITY;
      for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
        unflatten_coords(kCoord, kSpatial, rank, kIdx);

        int ok = 1;
        size_t inIdx = (size_t)c;
        for (int ax = 0; ax < rank; ++ax) {
          int64_t pos = (int64_t)outCoord[ax] * (int64_t)stride[ax] + (int64_t)kCoord[ax] -
                        (int64_t)padding[ax];
          if (pos < 0 || (uint32_t)pos >= inSpatial[ax]) {
            ok = 0;
            break;
          }
          inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)(uint32_t)pos;
        }

        float v = 0.0f;
        if (ok) {
          v = input->data[inIdx];
        }
        maxScaled = fmaxf(maxScaled, betaF * v);
      }

      float sumExp = 0.0f;
      for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
        unflatten_coords(kCoord, kSpatial, rank, kIdx);

        int ok = 1;
        size_t inIdx = (size_t)c;
        for (int ax = 0; ax < rank; ++ax) {
          int64_t pos = (int64_t)outCoord[ax] * (int64_t)stride[ax] + (int64_t)kCoord[ax] -
                        (int64_t)padding[ax];
          if (pos < 0 || (uint32_t)pos >= inSpatial[ax]) {
            ok = 0;
            break;
          }
          inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)(uint32_t)pos;
        }

        float v = 0.0f;
        if (ok) {
          v = input->data[inIdx];
        }
        sumExp += expf(betaF * v - maxScaled);
      }

      out->data[(size_t)c * outSpatialSize + outIdx] = (maxScaled + logf(sumExp)) / betaF;
    }
  }

  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_smooth_maxpool_bwd(
    b_lean_obj_arg inputObj, b_lean_obj_arg gradObj, double beta,
    b_lean_obj_arg inSpatialObj, b_lean_obj_arg kernelObj,
    b_lean_obj_arg strideObj, b_lean_obj_arg paddingObj,
    uint32_t inC) {
  torchlean_cuda_buffer* input = torchlean_cuda_buffer_unbox(inputObj);
  torchlean_cuda_buffer* gradOutput = torchlean_cuda_buffer_unbox(gradObj);

  const int rank = read_rank_checked(inSpatialObj, "torchlean_cuda_smooth_maxpool_bwd_stub: bad inSpatial");
  if (read_rank_checked(kernelObj, "torchlean_cuda_smooth_maxpool_bwd_stub: bad kernel") != rank ||
      read_rank_checked(strideObj, "torchlean_cuda_smooth_maxpool_bwd_stub: bad stride") != rank ||
      read_rank_checked(paddingObj, "torchlean_cuda_smooth_maxpool_bwd_stub: bad padding") != rank) {
    lean_internal_panic("torchlean_cuda_smooth_maxpool_bwd_stub: array rank mismatch");
  }
  if (rank <= 0) {
    lean_internal_panic("torchlean_cuda_smooth_maxpool_bwd_stub: rank must be > 0");
  }

  uint32_t inSpatial[K_MAX_RANK];
  uint32_t kSpatial[K_MAX_RANK];
  uint32_t stride[K_MAX_RANK];
  uint32_t padding[K_MAX_RANK];
  uint32_t outSpatial[K_MAX_RANK];

  read_u32_array(inSpatialObj, inSpatial, rank, "torchlean_cuda_smooth_maxpool_bwd_stub: bad inSpatial");
  read_u32_array(kernelObj, kSpatial, rank, "torchlean_cuda_smooth_maxpool_bwd_stub: bad kernel");
  read_u32_array(strideObj, stride, rank, "torchlean_cuda_smooth_maxpool_bwd_stub: bad stride");
  read_u32_array(paddingObj, padding, rank, "torchlean_cuda_smooth_maxpool_bwd_stub: bad padding");

  for (int ax = 0; ax < rank; ++ax) {
    if (kSpatial[ax] == 0) {
      lean_internal_panic("torchlean_cuda_smooth_maxpool_bwd_stub: kernel dims must be > 0");
    }
    if (stride[ax] == 0) {
      lean_internal_panic("torchlean_cuda_smooth_maxpool_bwd_stub: stride dims must be > 0");
    }
    outSpatial[ax] = outDim(inSpatial[ax], kSpatial[ax], stride[ax], padding[ax]);
  }

  const size_t inSpatialSize = prod_u32(inSpatial, rank);
  const size_t kSpatialSize = prod_u32(kSpatial, rank);
  const size_t outSpatialSize = prod_u32(outSpatial, rank);

  const size_t inElems = (size_t)inC * inSpatialSize;
  const size_t outElems = (size_t)inC * outSpatialSize;

  checkBufSize(input, inElems, "torchlean_cuda_smooth_maxpool_bwd_stub: input.size mismatch");
  checkBufSize(gradOutput, outElems, "torchlean_cuda_smooth_maxpool_bwd_stub: gradOutput.size mismatch");

  torchlean_cuda_buffer* dInput = torchlean_cuda_buffer_alloc(inElems);
  const float betaF =
      checked_smoothmax_beta(beta, "torchlean_cuda_smooth_maxpool_bwd_stub: beta must be finite and nonzero");
  const int det = torchlean_cuda_get_deterministic_reductions();
  if (det) {
    // Deterministic algorithm: compute each dInput element exactly once.
    uint32_t inCoord[K_MAX_RANK];
    uint32_t outCoord[K_MAX_RANK];
    uint32_t kCoord[K_MAX_RANK];
    int64_t oMin[K_MAX_RANK];
    int64_t oMax[K_MAX_RANK];
    size_t rangeSize[K_MAX_RANK];

    for (uint32_t c = 0; c < inC; ++c) {
      for (size_t spatialIdx = 0; spatialIdx < inSpatialSize; ++spatialIdx) {
        unflatten_coords(inCoord, inSpatial, rank, spatialIdx);
        const size_t selfIdx = (size_t)c * inSpatialSize + spatialIdx;
        const float vSelf = input->data[selfIdx];

        size_t totalComb = 1;
        for (int ax = 0; ax < rank; ++ax) {
          const int64_t s = (int64_t)stride[ax];
          const int64_t p = (int64_t)padding[ax];
          const int64_t kd = (int64_t)kSpatial[ax];
          int64_t lo = ceil_div_i64((int64_t)inCoord[ax] + p - (kd - 1), s);
          int64_t hi = floor_div_i64((int64_t)inCoord[ax] + p, s);
          if (lo < 0) lo = 0;
          const int64_t outD = (int64_t)outSpatial[ax];
          if (hi > outD - 1) hi = outD - 1;
          oMin[ax] = lo;
          oMax[ax] = hi;
          if (lo > hi) {
            totalComb = 0;
            break;
          }
          rangeSize[ax] = (size_t)(hi - lo + 1);
          totalComb *= rangeSize[ax];
        }

        float acc = 0.0f;
        if (totalComb > 0) {
          for (size_t t = 0; t < totalComb; ++t) {
            size_t tt = t;
            for (int ax = rank - 1; ax >= 0; --ax) {
              const size_t sz = rangeSize[ax];
              const size_t off = (sz == 0) ? 0 : (tt % sz);
              tt = (sz == 0) ? 0 : (tt / sz);
              outCoord[ax] = (uint32_t)(oMin[ax] + (int64_t)off);
            }

            int includes = 1;
            for (int ax = 0; ax < rank; ++ax) {
              int64_t k = (int64_t)inCoord[ax] + (int64_t)padding[ax] -
                          (int64_t)outCoord[ax] * (int64_t)stride[ax];
              if (k < 0 || k >= (int64_t)kSpatial[ax]) {
                includes = 0;
                break;
              }
            }
            if (!includes) continue;

            const size_t outIdx = flatten_coords(outCoord, outSpatial, rank);
            const float g = gradOutput->data[(size_t)c * outSpatialSize + outIdx];

            float maxScaled = -INFINITY;
            for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
              unflatten_coords(kCoord, kSpatial, rank, kIdx);

              int ok = 1;
              size_t inIdx = (size_t)c;
              for (int ax = 0; ax < rank; ++ax) {
                int64_t pos = (int64_t)outCoord[ax] * (int64_t)stride[ax] + (int64_t)kCoord[ax] -
                              (int64_t)padding[ax];
                if (pos < 0 || (uint32_t)pos >= inSpatial[ax]) {
                  ok = 0;
                  break;
                }
                inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)(uint32_t)pos;
              }

              float v = 0.0f;
              if (ok) {
                v = input->data[inIdx];
              }
              maxScaled = fmaxf(maxScaled, betaF * v);
            }

            float sumExp = 0.0f;
            for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
              unflatten_coords(kCoord, kSpatial, rank, kIdx);

              int ok = 1;
              size_t inIdx = (size_t)c;
              for (int ax = 0; ax < rank; ++ax) {
                int64_t pos = (int64_t)outCoord[ax] * (int64_t)stride[ax] + (int64_t)kCoord[ax] -
                              (int64_t)padding[ax];
                if (pos < 0 || (uint32_t)pos >= inSpatial[ax]) {
                  ok = 0;
                  break;
                }
                inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)(uint32_t)pos;
              }

              float v = 0.0f;
              if (ok) {
                v = input->data[inIdx];
              }
              sumExp += expf(betaF * v - maxScaled);
            }

            const float w = expf(betaF * vSelf - maxScaled) / sumExp;
            acc += g * w;
          }
        }

        dInput->data[selfIdx] = acc;
      }
    }
  } else {
    // Default scatter-style algorithm.
    for (size_t i = 0; i < dInput->size; ++i) dInput->data[i] = 0.0f;
    if (outElems == 0) {
      return torchlean_cuda_buffer_box(dInput);
    }

    uint32_t outCoord[K_MAX_RANK];
    uint32_t kCoord[K_MAX_RANK];

    for (uint32_t c = 0; c < inC; ++c) {
      for (size_t outIdx = 0; outIdx < outSpatialSize; ++outIdx) {
        const float g = gradOutput->data[(size_t)c * outSpatialSize + outIdx];
        unflatten_coords(outCoord, outSpatial, rank, outIdx);

        float maxScaled = -INFINITY;
        for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
          unflatten_coords(kCoord, kSpatial, rank, kIdx);

          int ok = 1;
          size_t inIdx = (size_t)c;
          for (int ax = 0; ax < rank; ++ax) {
            int64_t pos = (int64_t)outCoord[ax] * (int64_t)stride[ax] + (int64_t)kCoord[ax] -
                          (int64_t)padding[ax];
            if (pos < 0 || (uint32_t)pos >= inSpatial[ax]) {
              ok = 0;
              break;
            }
            inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)(uint32_t)pos;
          }

          float v = 0.0f;
          if (ok) {
            v = input->data[inIdx];
          }
          maxScaled = fmaxf(maxScaled, betaF * v);
        }

        float sumExp = 0.0f;
        for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
          unflatten_coords(kCoord, kSpatial, rank, kIdx);

          int ok = 1;
          size_t inIdx = (size_t)c;
          for (int ax = 0; ax < rank; ++ax) {
            int64_t pos = (int64_t)outCoord[ax] * (int64_t)stride[ax] + (int64_t)kCoord[ax] -
                          (int64_t)padding[ax];
            if (pos < 0 || (uint32_t)pos >= inSpatial[ax]) {
              ok = 0;
              break;
            }
            inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)(uint32_t)pos;
          }

          float v = 0.0f;
          if (ok) {
            v = input->data[inIdx];
          }
          sumExp += expf(betaF * v - maxScaled);
        }

        for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
          unflatten_coords(kCoord, kSpatial, rank, kIdx);

          int ok = 1;
          size_t inIdx = (size_t)c;
          for (int ax = 0; ax < rank; ++ax) {
            int64_t pos = (int64_t)outCoord[ax] * (int64_t)stride[ax] + (int64_t)kCoord[ax] -
                          (int64_t)padding[ax];
            if (pos < 0 || (uint32_t)pos >= inSpatial[ax]) {
              ok = 0;
              break;
            }
            inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)(uint32_t)pos;
          }
          if (!ok) continue;

          float v = input->data[inIdx];
          float e = expf(betaF * v - maxScaled);
          float w = e / sumExp;
          dInput->data[inIdx] += g * w;
        }
      }
    }
  }

  return torchlean_cuda_buffer_box(dInput);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_smooth_maxpool2d_fwd(
    b_lean_obj_arg inputObj, double beta,
    uint32_t inC, uint32_t inH, uint32_t inW,
    uint32_t kH, uint32_t kW,
    uint32_t stride, uint32_t padding) {
  torchlean_cuda_buffer* input = torchlean_cuda_buffer_unbox(inputObj);

  if (kH == 0 || kW == 0) {
    lean_internal_panic("torchlean_cuda_smooth_maxpool2d_fwd: kH/kW must be > 0");
  }

  uint32_t outH = outDim(inH, kH, stride, padding);
  uint32_t outW = outDim(inW, kW, stride, padding);

  size_t inElems = (size_t)inC * (size_t)inH * (size_t)inW;
  size_t outElems = (size_t)inC * (size_t)outH * (size_t)outW;

  checkBufSize(input, inElems, "torchlean_cuda_smooth_maxpool2d_fwd_stub: input.size mismatch");

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outElems);
  if (outElems == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  float betaF =
      checked_smoothmax_beta(beta, "torchlean_cuda_smooth_maxpool2d_fwd_stub: beta must be finite and nonzero");

  for (uint32_t c = 0; c < inC; ++c) {
    for (uint32_t oh = 0; oh < outH; ++oh) {
      for (uint32_t ow = 0; ow < outW; ++ow) {
        float maxScaled = -INFINITY;
        for (uint32_t ky = 0; ky < kH; ++ky) {
          int ih = (int)((int64_t)oh * (int64_t)stride + (int64_t)ky - (int64_t)padding);
          for (uint32_t kx = 0; kx < kW; ++kx) {
            int iw = (int)((int64_t)ow * (int64_t)stride + (int64_t)kx - (int64_t)padding);
            float v = 0.0f;
            if (ih >= 0 && ih < (int)inH && iw >= 0 && iw < (int)inW) {
              size_t inIdx =
                  ((size_t)c * (size_t)inH + (size_t)ih) * (size_t)inW + (size_t)iw;
              v = input->data[inIdx];
            }
            maxScaled = fmaxf(maxScaled, betaF * v);
          }
        }

        float sumExp = 0.0f;
        for (uint32_t ky = 0; ky < kH; ++ky) {
          int ih = (int)((int64_t)oh * (int64_t)stride + (int64_t)ky - (int64_t)padding);
          for (uint32_t kx = 0; kx < kW; ++kx) {
            int iw = (int)((int64_t)ow * (int64_t)stride + (int64_t)kx - (int64_t)padding);
            float v = 0.0f;
            if (ih >= 0 && ih < (int)inH && iw >= 0 && iw < (int)inW) {
              size_t inIdx =
                  ((size_t)c * (size_t)inH + (size_t)ih) * (size_t)inW + (size_t)iw;
              v = input->data[inIdx];
            }
            sumExp += expf(betaF * v - maxScaled);
          }
        }
        size_t outIdx = ((size_t)c * (size_t)outH + (size_t)oh) * (size_t)outW + (size_t)ow;
        out->data[outIdx] = (maxScaled + logf(sumExp)) / betaF;
      }
    }
  }

  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_smooth_maxpool2d_bwd(
    b_lean_obj_arg inputObj, b_lean_obj_arg gradObj, double beta,
    uint32_t inC, uint32_t inH, uint32_t inW,
    uint32_t kH, uint32_t kW,
    uint32_t stride, uint32_t padding) {
  torchlean_cuda_buffer* input = torchlean_cuda_buffer_unbox(inputObj);
  torchlean_cuda_buffer* gradOutput = torchlean_cuda_buffer_unbox(gradObj);

  if (kH == 0 || kW == 0) {
    lean_internal_panic("torchlean_cuda_smooth_maxpool2d_bwd: kH/kW must be > 0");
  }

  uint32_t outH = outDim(inH, kH, stride, padding);
  uint32_t outW = outDim(inW, kW, stride, padding);

  size_t inElems = (size_t)inC * (size_t)inH * (size_t)inW;
  size_t outElems = (size_t)inC * (size_t)outH * (size_t)outW;

  checkBufSize(input, inElems, "torchlean_cuda_smooth_maxpool2d_bwd_stub: input.size mismatch");
  checkBufSize(gradOutput, outElems, "torchlean_cuda_smooth_maxpool2d_bwd_stub: gradOutput.size mismatch");

  torchlean_cuda_buffer* dInput = torchlean_cuda_buffer_alloc(inElems);
  const float betaF =
      checked_smoothmax_beta(beta, "torchlean_cuda_smooth_maxpool2d_bwd_stub: beta must be finite and nonzero");
  const int det = torchlean_cuda_get_deterministic_reductions();
  if (det) {
    // Deterministic algorithm: compute each dInput element exactly once.
    for (uint32_t c = 0; c < inC; ++c) {
      for (uint32_t ih = 0; ih < inH; ++ih) {
        for (uint32_t iw = 0; iw < inW; ++iw) {
          const size_t selfIdx = ((size_t)c * (size_t)inH + (size_t)ih) * (size_t)inW + (size_t)iw;
          const float vSelf = input->data[selfIdx];
          float acc = 0.0f;

          int64_t ohMin = ceil_div_i64((int64_t)ih + (int64_t)padding - (int64_t)(kH - 1), (int64_t)stride);
          int64_t ohMax = floor_div_i64((int64_t)ih + (int64_t)padding, (int64_t)stride);
          int64_t owMin = ceil_div_i64((int64_t)iw + (int64_t)padding - (int64_t)(kW - 1), (int64_t)stride);
          int64_t owMax = floor_div_i64((int64_t)iw + (int64_t)padding, (int64_t)stride);

          if (ohMin < 0) ohMin = 0;
          if (owMin < 0) owMin = 0;
          if (ohMax > (int64_t)outH - 1) ohMax = (int64_t)outH - 1;
          if (owMax > (int64_t)outW - 1) owMax = (int64_t)outW - 1;

          if (ohMin <= ohMax && owMin <= owMax) {
            for (int64_t oh = ohMin; oh <= ohMax; ++oh) {
              const int kySelf = (int)((int64_t)ih + (int64_t)padding - (int64_t)oh * (int64_t)stride);
              if (kySelf < 0 || kySelf >= (int)kH) continue;
              for (int64_t ow = owMin; ow <= owMax; ++ow) {
                const int kxSelf = (int)((int64_t)iw + (int64_t)padding - (int64_t)ow * (int64_t)stride);
                if (kxSelf < 0 || kxSelf >= (int)kW) continue;

                float maxScaled = -INFINITY;
                for (uint32_t ky = 0; ky < kH; ++ky) {
                  int candH = (int)((int64_t)oh * (int64_t)stride + (int64_t)ky - (int64_t)padding);
                  for (uint32_t kx = 0; kx < kW; ++kx) {
                    int candW = (int)((int64_t)ow * (int64_t)stride + (int64_t)kx - (int64_t)padding);
                    float v = 0.0f;
                    if (candH >= 0 && candH < (int)inH && candW >= 0 && candW < (int)inW) {
                      size_t inIdx =
                          ((size_t)c * (size_t)inH + (size_t)candH) * (size_t)inW + (size_t)candW;
                      v = input->data[inIdx];
                    }
                    maxScaled = fmaxf(maxScaled, betaF * v);
                  }
                }

                float sumExp = 0.0f;
                for (uint32_t ky = 0; ky < kH; ++ky) {
                  int candH = (int)((int64_t)oh * (int64_t)stride + (int64_t)ky - (int64_t)padding);
                  for (uint32_t kx = 0; kx < kW; ++kx) {
                    int candW = (int)((int64_t)ow * (int64_t)stride + (int64_t)kx - (int64_t)padding);
                    float v = 0.0f;
                    if (candH >= 0 && candH < (int)inH && candW >= 0 && candW < (int)inW) {
                      size_t inIdx =
                          ((size_t)c * (size_t)inH + (size_t)candH) * (size_t)inW + (size_t)candW;
                      v = input->data[inIdx];
                    }
                    sumExp += expf(betaF * v - maxScaled);
                  }
                }

                const size_t outIdx =
                    ((size_t)c * (size_t)outH + (size_t)oh) * (size_t)outW + (size_t)ow;
                const float w = expf(betaF * vSelf - maxScaled) / sumExp;
                acc += gradOutput->data[outIdx] * w;
              }
            }
          }

          dInput->data[selfIdx] = acc;
        }
      }
    }
  } else {
    // Default scatter-style algorithm.
    for (size_t i = 0; i < dInput->size; ++i) {
      dInput->data[i] = 0.0f;
    }

    for (uint32_t c = 0; c < inC; ++c) {
      for (uint32_t oh = 0; oh < outH; ++oh) {
        for (uint32_t ow = 0; ow < outW; ++ow) {
          size_t outIdx = ((size_t)c * (size_t)outH + (size_t)oh) * (size_t)outW + (size_t)ow;
          float g = gradOutput->data[outIdx];

          float maxScaled = -INFINITY;
          for (uint32_t ky = 0; ky < kH; ++ky) {
            int ih = (int)((int64_t)oh * (int64_t)stride + (int64_t)ky - (int64_t)padding);
            for (uint32_t kx = 0; kx < kW; ++kx) {
              int iw = (int)((int64_t)ow * (int64_t)stride + (int64_t)kx - (int64_t)padding);
              float v = 0.0f;
              if (ih >= 0 && ih < (int)inH && iw >= 0 && iw < (int)inW) {
                size_t inIdx =
                    ((size_t)c * (size_t)inH + (size_t)ih) * (size_t)inW + (size_t)iw;
                v = input->data[inIdx];
              }
              maxScaled = fmaxf(maxScaled, betaF * v);
            }
          }

          float sumExp = 0.0f;
          for (uint32_t ky = 0; ky < kH; ++ky) {
            int ih = (int)((int64_t)oh * (int64_t)stride + (int64_t)ky - (int64_t)padding);
            for (uint32_t kx = 0; kx < kW; ++kx) {
              int iw = (int)((int64_t)ow * (int64_t)stride + (int64_t)kx - (int64_t)padding);
              float v = 0.0f;
              if (ih >= 0 && ih < (int)inH && iw >= 0 && iw < (int)inW) {
                size_t inIdx =
                    ((size_t)c * (size_t)inH + (size_t)ih) * (size_t)inW + (size_t)iw;
                v = input->data[inIdx];
              }
              sumExp += expf(betaF * v - maxScaled);
            }
          }

          for (uint32_t ky = 0; ky < kH; ++ky) {
            int ih = (int)((int64_t)oh * (int64_t)stride + (int64_t)ky - (int64_t)padding);
            if (ih < 0 || ih >= (int)inH) continue;
            for (uint32_t kx = 0; kx < kW; ++kx) {
              int iw = (int)((int64_t)ow * (int64_t)stride + (int64_t)kx - (int64_t)padding);
              if (iw < 0 || iw >= (int)inW) continue;
              size_t inIdx =
                  ((size_t)c * (size_t)inH + (size_t)ih) * (size_t)inW + (size_t)iw;
              float v = input->data[inIdx];
              float e = expf(betaF * v - maxScaled);
              float w = e / sumExp;
              dInput->data[inIdx] += g * w;
            }
          }
        }
      }
    }
  }

  return torchlean_cuda_buffer_box(dInput);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_avgpool2d_fwd(
    b_lean_obj_arg inputObj,
    uint32_t inC, uint32_t inH, uint32_t inW,
    uint32_t kH, uint32_t kW,
    uint32_t stride, uint32_t padding) {
  torchlean_cuda_buffer* input = torchlean_cuda_buffer_unbox(inputObj);

  if (kH == 0 || kW == 0) {
    lean_internal_panic("torchlean_cuda_avgpool2d_fwd: kH/kW must be > 0");
  }

  uint32_t outH = outDim(inH, kH, stride, padding);
  uint32_t outW = outDim(inW, kW, stride, padding);

  size_t inElems = (size_t)inC * (size_t)inH * (size_t)inW;
  size_t outElems = (size_t)inC * (size_t)outH * (size_t)outW;

  checkBufSize(input, inElems, "torchlean_cuda_avgpool2d_fwd_stub: input.size mismatch");

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outElems);
  if (outElems == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  float denom = (float)((uint64_t)kH * (uint64_t)kW);
  for (uint32_t c = 0; c < inC; ++c) {
    for (uint32_t oh = 0; oh < outH; ++oh) {
      for (uint32_t ow = 0; ow < outW; ++ow) {
        float acc = 0.0f;
        for (uint32_t ky = 0; ky < kH; ++ky) {
          int ih = (int)((int64_t)oh * (int64_t)stride + (int64_t)ky - (int64_t)padding);
          for (uint32_t kx = 0; kx < kW; ++kx) {
            int iw = (int)((int64_t)ow * (int64_t)stride + (int64_t)kx - (int64_t)padding);
            if (ih >= 0 && ih < (int)inH && iw >= 0 && iw < (int)inW) {
              size_t inIdx =
                  ((size_t)c * (size_t)inH + (size_t)ih) * (size_t)inW + (size_t)iw;
              acc += input->data[inIdx];
            }
            // else: out-of-bounds contributes 0 and is still counted in denom (count_include_pad=true).
          }
        }
        size_t outIdx = ((size_t)c * (size_t)outH + (size_t)oh) * (size_t)outW + (size_t)ow;
        out->data[outIdx] = acc / denom;
      }
    }
  }

  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_avgpool2d_bwd(
    b_lean_obj_arg gradObj,
    uint32_t inC, uint32_t inH, uint32_t inW,
    uint32_t kH, uint32_t kW,
    uint32_t stride, uint32_t padding) {
  torchlean_cuda_buffer* gradOutput = torchlean_cuda_buffer_unbox(gradObj);

  if (kH == 0 || kW == 0) {
    lean_internal_panic("torchlean_cuda_avgpool2d_bwd: kH/kW must be > 0");
  }

  uint32_t outH = outDim(inH, kH, stride, padding);
  uint32_t outW = outDim(inW, kW, stride, padding);

  size_t inElems = (size_t)inC * (size_t)inH * (size_t)inW;
  size_t outElems = (size_t)inC * (size_t)outH * (size_t)outW;

  checkBufSize(gradOutput, outElems, "torchlean_cuda_avgpool2d_bwd_stub: gradOutput.size mismatch");

  torchlean_cuda_buffer* dInput = torchlean_cuda_buffer_alloc(inElems);
  const float denom = (float)((uint64_t)kH * (uint64_t)kW);
  const int det = torchlean_cuda_get_deterministic_reductions();
  if (det) {
    // Deterministic algorithm: compute each dInput element exactly once.
    for (uint32_t c = 0; c < inC; ++c) {
      for (uint32_t ih = 0; ih < inH; ++ih) {
        for (uint32_t iw = 0; iw < inW; ++iw) {
          float acc = 0.0f;

          int64_t ohMin = ceil_div_i64((int64_t)ih + (int64_t)padding - (int64_t)(kH - 1), (int64_t)stride);
          int64_t ohMax = floor_div_i64((int64_t)ih + (int64_t)padding, (int64_t)stride);
          int64_t owMin = ceil_div_i64((int64_t)iw + (int64_t)padding - (int64_t)(kW - 1), (int64_t)stride);
          int64_t owMax = floor_div_i64((int64_t)iw + (int64_t)padding, (int64_t)stride);

          if (ohMin < 0) ohMin = 0;
          if (owMin < 0) owMin = 0;
          if (ohMax > (int64_t)outH - 1) ohMax = (int64_t)outH - 1;
          if (owMax > (int64_t)outW - 1) owMax = (int64_t)outW - 1;

          if (ohMin <= ohMax && owMin <= owMax) {
            for (int64_t oh = ohMin; oh <= ohMax; ++oh) {
              const int ky = (int)((int64_t)ih + (int64_t)padding - (int64_t)oh * (int64_t)stride);
              if (ky < 0 || ky >= (int)kH) continue;
              for (int64_t ow = owMin; ow <= owMax; ++ow) {
                const int kx = (int)((int64_t)iw + (int64_t)padding - (int64_t)ow * (int64_t)stride);
                if (kx < 0 || kx >= (int)kW) continue;
                const size_t outIdx =
                    ((size_t)c * (size_t)outH + (size_t)oh) * (size_t)outW + (size_t)ow;
                acc += gradOutput->data[outIdx] / denom;
              }
            }
          }

          const size_t inIdx = ((size_t)c * (size_t)inH + (size_t)ih) * (size_t)inW + (size_t)iw;
          dInput->data[inIdx] = acc;
        }
      }
    }
  } else {
    // Default scatter-style algorithm.
    for (size_t i = 0; i < dInput->size; ++i) {
      dInput->data[i] = 0.0f;
    }

    for (uint32_t c = 0; c < inC; ++c) {
      for (uint32_t oh = 0; oh < outH; ++oh) {
        for (uint32_t ow = 0; ow < outW; ++ow) {
          size_t outIdx = ((size_t)c * (size_t)outH + (size_t)oh) * (size_t)outW + (size_t)ow;
          float g = gradOutput->data[outIdx] / denom;
          for (uint32_t ky = 0; ky < kH; ++ky) {
            int ih2 = (int)((int64_t)oh * (int64_t)stride + (int64_t)ky - (int64_t)padding);
            if (ih2 < 0 || ih2 >= (int)inH) continue;
            for (uint32_t kx = 0; kx < kW; ++kx) {
              int iw2 = (int)((int64_t)ow * (int64_t)stride + (int64_t)kx - (int64_t)padding);
              if (iw2 < 0 || iw2 >= (int)inW) continue;
              size_t inIdx =
                  ((size_t)c * (size_t)inH + (size_t)ih2) * (size_t)inW + (size_t)iw2;
              dInput->data[inIdx] += g;
            }
          }
        }
      }
    }
  }

  return torchlean_cuda_buffer_box(dInput);
}
