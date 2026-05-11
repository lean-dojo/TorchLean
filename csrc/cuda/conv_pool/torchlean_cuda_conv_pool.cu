#include <lean/lean.h>

#include "torchlean_cuda_buffer.h"
#include "torchlean_cuda_conv_pool_common.h"
#include "torchlean_cuda_common.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// CUDA implementation of TorchLean float32 conv/pool kernels over `Cuda.Buffer`.
//
// This file is intentionally explicit rather than clever: the 2D kernels serve the common fast path,
// while the N-D kernels cover the generic spec. Both paths use row-major flattened CHW / N-D layouts
// matching the Lean tensor specs.
//
// Trust-boundary contract:
// - host wrappers validate sizes, ranks, kernel sizes, stride, and padding-derived output shapes;
// - max-pool padding is ignored (`-inf` for max), matching the TorchLean/PyTorch convention;
// - atomic backward paths have deterministic alternatives controlled by the runtime flag;
// - the CPU fallback in `torchlean_cuda_conv_pool_stub.c` must stay semantically aligned.

static constexpr int kBlock = 256;
static constexpr int kMaxRank = TORCHLEAN_CUDA_CONV_POOL_MAX_RANK;
static_assert(kBlock > 0 && (kBlock & (kBlock - 1)) == 0,
              "kBlock must remain a power of two for reduction-style kernels");

static inline int grid_for(size_t n) {
  size_t blocks = (n + (size_t)kBlock - 1) / (size_t)kBlock;
  if (blocks == 0) blocks = 1;
  if (blocks > (size_t)INT_MAX) blocks = (size_t)INT_MAX;
  return (int)blocks;
}

static inline bool torchlean_cuda_deterministic_reductions_enabled() {
  return torchlean_cuda_get_deterministic_reductions() != 0;
}

// -------------------------
// Conv2d forward/backward
// -------------------------

__global__ void conv2d_fwd_kernel(const float* input, const float* kernel, const float* bias,
                                  float* output,
                                  int inC, int inH, int inW,
                                  int outC, int outH, int outW,
                                  int kH, int kW,
                                  int stride, int padding) {
  size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  size_t total = (size_t)outC * (size_t)outH * (size_t)outW;
  if (idx >= total) return;

  int ow = (int)(idx % (size_t)outW);
  size_t t = idx / (size_t)outW;
  int oh = (int)(t % (size_t)outH);
  int oc = (int)(t / (size_t)outH);

  float acc = bias ? bias[oc] : 0.0f;

  for (int ic = 0; ic < inC; ++ic) {
    for (int ky = 0; ky < kH; ++ky) {
      int ih = oh * stride + ky - padding;
      if (ih < 0 || ih >= inH) continue;
      for (int kx = 0; kx < kW; ++kx) {
        int iw = ow * stride + kx - padding;
        if (iw < 0 || iw >= inW) continue;
        size_t inIdx = ((size_t)ic * (size_t)inH + (size_t)ih) * (size_t)inW + (size_t)iw;
        size_t wIdx =
            (((size_t)oc * (size_t)inC + (size_t)ic) * (size_t)kH + (size_t)ky) * (size_t)kW +
            (size_t)kx;
        acc += input[inIdx] * kernel[wIdx];
      }
    }
  }
  output[idx] = acc;
}

__global__ void conv2d_dbias_kernel(const float* gradOutput, float* dBias,
                                   int outC, int outH, int outW) {
  int oc = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (oc >= outC) return;
  float acc = 0.0f;
  size_t base = (size_t)oc * (size_t)outH * (size_t)outW;
  size_t n = (size_t)outH * (size_t)outW;
  for (size_t i = 0; i < n; ++i) {
    acc += gradOutput[base + i];
  }
  dBias[oc] = acc;
}

__global__ void conv2d_dkernel_kernel(const float* input, const float* gradOutput,
                                     float* dKernel,
                                     int inC, int inH, int inW,
                                     int outC, int outH, int outW,
                                     int kH, int kW,
                                     int stride, int padding) {
  size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  size_t total = (size_t)outC * (size_t)inC * (size_t)kH * (size_t)kW;
  if (idx >= total) return;

  int kx = (int)(idx % (size_t)kW);
  size_t t0 = idx / (size_t)kW;
  int ky = (int)(t0 % (size_t)kH);
  size_t t1 = t0 / (size_t)kH;
  int ic = (int)(t1 % (size_t)inC);
  int oc = (int)(t1 / (size_t)inC);

  float acc = 0.0f;
  for (int oh = 0; oh < outH; ++oh) {
    int ih = oh * stride + ky - padding;
    if (ih < 0 || ih >= inH) continue;
    for (int ow = 0; ow < outW; ++ow) {
      int iw = ow * stride + kx - padding;
      if (iw < 0 || iw >= inW) continue;
      size_t inIdx = ((size_t)ic * (size_t)inH + (size_t)ih) * (size_t)inW + (size_t)iw;
      size_t goIdx = ((size_t)oc * (size_t)outH + (size_t)oh) * (size_t)outW + (size_t)ow;
      acc += input[inIdx] * gradOutput[goIdx];
    }
  }
  dKernel[idx] = acc;
}

__global__ void conv2d_dinput_kernel(const float* kernel, const float* gradOutput,
                                    float* dInput,
                                    int inC, int inH, int inW,
                                    int outC, int outH, int outW,
                                    int kH, int kW,
                                    int stride, int padding) {
  size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  size_t total = (size_t)inC * (size_t)inH * (size_t)inW;
  if (idx >= total) return;

  int iw = (int)(idx % (size_t)inW);
  size_t t0 = idx / (size_t)inW;
  int ih = (int)(t0 % (size_t)inH);
  int ic = (int)(t0 / (size_t)inH);

  int ihPad = ih + padding;
  int iwPad = iw + padding;

  float acc = 0.0f;
  for (int oc = 0; oc < outC; ++oc) {
    for (int ky = 0; ky < kH; ++ky) {
      int ohNum = ihPad - ky;
      if (ohNum < 0) continue;
      if ((ohNum % stride) != 0) continue;
      int oh = ohNum / stride;
      if (oh < 0 || oh >= outH) continue;
      for (int kx = 0; kx < kW; ++kx) {
        int owNum = iwPad - kx;
        if (owNum < 0) continue;
        if ((owNum % stride) != 0) continue;
        int ow = owNum / stride;
        if (ow < 0 || ow >= outW) continue;
        size_t goIdx = ((size_t)oc * (size_t)outH + (size_t)oh) * (size_t)outW + (size_t)ow;
        size_t wIdx =
            (((size_t)oc * (size_t)inC + (size_t)ic) * (size_t)kH + (size_t)ky) * (size_t)kW +
            (size_t)kx;
        acc += gradOutput[goIdx] * kernel[wIdx];
      }
    }
  }
  dInput[idx] = acc;
}

__global__ void add_bias_chw_kernel(float* out, const float* bias, int C, int H, int W) {
  size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  size_t total = (size_t)C * (size_t)H * (size_t)W;
  if (idx >= total) return;
  int c = (int)(idx / ((size_t)H * (size_t)W));
  out[idx] += bias[c];
}

__global__ void add_bias_cspatial_kernel(float* out, const float* bias, uint32_t C, size_t spatialSize) {
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t total = (size_t)C * spatialSize;
  if (idx >= total) return;
  const uint32_t c = (uint32_t)(idx / spatialSize);
  out[idx] += bias[c];
}

__global__ void convtranspose2d_dkernel_kernel(const float* input, const float* gradOutput,
                                               float* dKernel,
                                               int inC, int inH, int inW,
                                               int outC, int outH, int outW,
                                               int kH, int kW,
                                               int stride, int padding) {
  size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  size_t total = (size_t)inC * (size_t)outC * (size_t)kH * (size_t)kW;
  if (idx >= total) return;

  int kx = (int)(idx % (size_t)kW);
  size_t t0 = idx / (size_t)kW;
  int ky = (int)(t0 % (size_t)kH);
  size_t t1 = t0 / (size_t)kH;
  int oc = (int)(t1 % (size_t)outC);
  int ic = (int)(t1 / (size_t)outC);

  float acc = 0.0f;
  for (int ih = 0; ih < inH; ++ih) {
    for (int iw = 0; iw < inW; ++iw) {
      int oh = ih * stride + ky - padding;
      int ow = iw * stride + kx - padding;
      if (oh < 0 || oh >= outH || ow < 0 || ow >= outW) continue;
      size_t inIdx = ((size_t)ic * (size_t)inH + (size_t)ih) * (size_t)inW + (size_t)iw;
      size_t goIdx = ((size_t)oc * (size_t)outH + (size_t)oh) * (size_t)outW + (size_t)ow;
      acc += input[inIdx] * gradOutput[goIdx];
    }
  }
  dKernel[idx] = acc;
}

// -------------------------
// Pooling forward/backward
// -------------------------

__global__ void maxpool2d_fwd_kernel(const float* input, float* output,
                                     int inC, int inH, int inW,
                                     int outH, int outW,
                                     int kH, int kW,
                                     int stride, int padding) {
  size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  size_t total = (size_t)inC * (size_t)outH * (size_t)outW;
  if (idx >= total) return;

  int ow = (int)(idx % (size_t)outW);
  size_t t0 = idx / (size_t)outW;
  int oh = (int)(t0 % (size_t)outH);
  int c = (int)(t0 / (size_t)outH);

  float best = -FLT_MAX;
  for (int ky = 0; ky < kH; ++ky) {
    int ih = oh * stride + ky - padding;
    for (int kx = 0; kx < kW; ++kx) {
      int iw = ow * stride + kx - padding;
      float v = -INFINITY;
      if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
        size_t inIdx = ((size_t)c * (size_t)inH + (size_t)ih) * (size_t)inW + (size_t)iw;
        v = input[inIdx];
      }
      if (v > best) {
        best = v;
      }
    }
  }
  output[idx] = best;
}

__global__ void maxpool2d_bwd_kernel(const float* input, const float* gradOutput, float* dInput,
                                     int inC, int inH, int inW,
                                     int outH, int outW,
                                     int kH, int kW,
                                     int stride, int padding) {
  size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  size_t total = (size_t)inC * (size_t)outH * (size_t)outW;
  if (idx >= total) return;

  int ow = (int)(idx % (size_t)outW);
  size_t t0 = idx / (size_t)outW;
  int oh = (int)(t0 % (size_t)outH);
  int c = (int)(t0 / (size_t)outH);

  float best = -FLT_MAX;
  int bestIh = -1;
  int bestIw = -1;

  // Tie-breaking matches TorchLean spec: keep the first (row-major) argmax.
  for (int ky = 0; ky < kH; ++ky) {
    int ih = oh * stride + ky - padding;
    for (int kx = 0; kx < kW; ++kx) {
      int iw = ow * stride + kx - padding;
      float v = -INFINITY;
      if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
        size_t inIdx = ((size_t)c * (size_t)inH + (size_t)ih) * (size_t)inW + (size_t)iw;
        v = input[inIdx];
      }
      if (v > best) {
        best = v;
        bestIh = ih;
        bestIw = iw;
      }
    }
  }

  if (bestIh >= 0 && bestIh < inH && bestIw >= 0 && bestIw < inW) {
    size_t inIdx = ((size_t)c * (size_t)inH + (size_t)bestIh) * (size_t)inW + (size_t)bestIw;
    atomicAdd(&dInput[inIdx], gradOutput[idx]);
  }
}

// Deterministic backward for maxpool2d: compute each dInput element exactly once.
//
// Default (non-deterministic) path:
//   one thread per output element, `atomicAdd` into dInput
//   fast but floating-point accumulation order depends on GPU execution timing
//
// Deterministic path:
//   one thread per input element
//   scan the (finite) set of output windows that include this input element
//   for each such output, recompute the window argmax (ignoring padded cells, i.e. PyTorch -inf)
//   add gradOutput only if this input element is the chosen (tie-broken) argmax
//
// This is slower, but bit-stable.
__global__ void maxpool2d_bwd_det_kernel(const float* input, const float* gradOutput, float* dInput,
                                        int inC, int inH, int inW,
                                        int outH, int outW,
                                        int kH, int kW,
                                        int stride, int padding) {
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t total = (size_t)inC * (size_t)inH * (size_t)inW;
  if (idx >= total) return;

  const int iw = (int)(idx % (size_t)inW);
  size_t t0 = idx / (size_t)inW;
  const int ih = (int)(t0 % (size_t)inH);
  const int c = (int)(t0 / (size_t)inH);

  // Candidate output coordinate ranges:
  //   oh*stride <= ih+padding <= oh*stride+(kH-1)
  int64_t ohMin = ceil_div_i64((int64_t)ih + (int64_t)padding - (int64_t)(kH - 1), (int64_t)stride);
  int64_t ohMax = floor_div_i64((int64_t)ih + (int64_t)padding, (int64_t)stride);
  int64_t owMin = ceil_div_i64((int64_t)iw + (int64_t)padding - (int64_t)(kW - 1), (int64_t)stride);
  int64_t owMax = floor_div_i64((int64_t)iw + (int64_t)padding, (int64_t)stride);

  if (ohMin < 0) ohMin = 0;
  if (owMin < 0) owMin = 0;
  if (ohMax > (int64_t)outH - 1) ohMax = (int64_t)outH - 1;
  if (owMax > (int64_t)outW - 1) owMax = (int64_t)outW - 1;

  float acc = 0.0f;
  if (ohMin <= ohMax && owMin <= owMax) {
    for (int64_t oh = ohMin; oh <= ohMax; ++oh) {
      for (int64_t ow = owMin; ow <= owMax; ++ow) {
        // Recompute argmax for this output window, ignoring padded cells and matching
        // tie-breaking of the atomic kernel (strict > keeps the first row-major argmax).
        float best = -FLT_MAX;
        int bestIh = -1;
        int bestIw = -1;
        for (int ky = 0; ky < kH; ++ky) {
          int candH = (int)oh * stride + ky - padding;
          for (int kx = 0; kx < kW; ++kx) {
            int candW = (int)ow * stride + kx - padding;
            float v = -INFINITY;
            if (candH >= 0 && candH < inH && candW >= 0 && candW < inW) {
              size_t inIdx = ((size_t)c * (size_t)inH + (size_t)candH) * (size_t)inW + (size_t)candW;
              v = input[inIdx];
            }
            if (v > best) {
              best = v;
              bestIh = candH;
              bestIw = candW;
            }
          }
        }

        if (bestIh == ih && bestIw == iw) {
          const size_t outIdx = ((size_t)c * (size_t)outH + (size_t)oh) * (size_t)outW + (size_t)ow;
          acc += gradOutput[outIdx];
        }
      }
    }
  }

  dInput[idx] = acc;
}

__global__ void avgpool2d_fwd_kernel(const float* input, float* output,
                                     int inC, int inH, int inW,
                                     int outH, int outW,
                                     int kH, int kW,
                                     int stride, int padding) {
  size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  size_t total = (size_t)inC * (size_t)outH * (size_t)outW;
  if (idx >= total) return;

  int ow = (int)(idx % (size_t)outW);
  size_t t0 = idx / (size_t)outW;
  int oh = (int)(t0 % (size_t)outH);
  int c = (int)(t0 / (size_t)outH);

  float acc = 0.0f;
  for (int ky = 0; ky < kH; ++ky) {
    int ih = oh * stride + ky - padding;
    for (int kx = 0; kx < kW; ++kx) {
      int iw = ow * stride + kx - padding;
      if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
        size_t inIdx = ((size_t)c * (size_t)inH + (size_t)ih) * (size_t)inW + (size_t)iw;
        acc += input[inIdx];
      }
      // else: out-of-bounds contributes 0 and is still counted in denom (count_include_pad=true).
    }
  }
  float denom = (float)((uint64_t)kH * (uint64_t)kW);
  output[idx] = acc / denom;
}

__global__ void avgpool2d_bwd_kernel(const float* gradOutput, float* dInput,
                                     int inC, int inH, int inW,
                                     int outH, int outW,
                                     int kH, int kW,
                                     int stride, int padding) {
  size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  size_t total = (size_t)inC * (size_t)outH * (size_t)outW;
  if (idx >= total) return;

  int ow = (int)(idx % (size_t)outW);
  size_t t0 = idx / (size_t)outW;
  int oh = (int)(t0 % (size_t)outH);
  int c = (int)(t0 / (size_t)outH);

  float denom = (float)((uint64_t)kH * (uint64_t)kW);
  float g = gradOutput[idx] / denom;

  for (int ky = 0; ky < kH; ++ky) {
    int ih = oh * stride + ky - padding;
    if (ih < 0 || ih >= inH) continue;
    for (int kx = 0; kx < kW; ++kx) {
      int iw = ow * stride + kx - padding;
      if (iw < 0 || iw >= inW) continue;
      size_t inIdx = ((size_t)c * (size_t)inH + (size_t)ih) * (size_t)inW + (size_t)iw;
      atomicAdd(&dInput[inIdx], g);
    }
  }
}

// Deterministic backward for avgpool2d: compute each dInput element exactly once.
// See maxpool2d_bwd_det_kernel for the rationale and high-level shape of the algorithm.
__global__ void avgpool2d_bwd_det_kernel(const float* gradOutput, float* dInput,
                                        int inC, int inH, int inW,
                                        int outH, int outW,
                                        int kH, int kW,
                                        int stride, int padding) {
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t total = (size_t)inC * (size_t)inH * (size_t)inW;
  if (idx >= total) return;

  const int iw = (int)(idx % (size_t)inW);
  size_t t0 = idx / (size_t)inW;
  const int ih = (int)(t0 % (size_t)inH);
  const int c = (int)(t0 / (size_t)inH);

  int64_t ohMin = ceil_div_i64((int64_t)ih + (int64_t)padding - (int64_t)(kH - 1), (int64_t)stride);
  int64_t ohMax = floor_div_i64((int64_t)ih + (int64_t)padding, (int64_t)stride);
  int64_t owMin = ceil_div_i64((int64_t)iw + (int64_t)padding - (int64_t)(kW - 1), (int64_t)stride);
  int64_t owMax = floor_div_i64((int64_t)iw + (int64_t)padding, (int64_t)stride);

  if (ohMin < 0) ohMin = 0;
  if (owMin < 0) owMin = 0;
  if (ohMax > (int64_t)outH - 1) ohMax = (int64_t)outH - 1;
  if (owMax > (int64_t)outW - 1) owMax = (int64_t)outW - 1;

  const float denom = (float)((uint64_t)kH * (uint64_t)kW);
  float acc = 0.0f;
  if (ohMin <= ohMax && owMin <= owMax) {
    for (int64_t oh = ohMin; oh <= ohMax; ++oh) {
      const int ky = ih + padding - (int)oh * stride;
      if (ky < 0 || ky >= kH) continue;
      for (int64_t ow = owMin; ow <= owMax; ++ow) {
        const int kx = iw + padding - (int)ow * stride;
        if (kx < 0 || kx >= kW) continue;
        const size_t outIdx = ((size_t)c * (size_t)outH + (size_t)oh) * (size_t)outW + (size_t)ow;
        acc += gradOutput[outIdx] / denom;
      }
    }
  }

  dInput[idx] = acc;
}

__global__ void smoothmaxpool2d_fwd_kernel(const float* input, float* output,
                                          int inC, int inH, int inW,
                                          int outH, int outW,
                                          int kH, int kW,
                                          int stride, int padding,
                                          float beta) {
  size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  size_t total = (size_t)inC * (size_t)outH * (size_t)outW;
  if (idx >= total) return;

  int ow = (int)(idx % (size_t)outW);
  size_t t0 = idx / (size_t)outW;
  int oh = (int)(t0 % (size_t)outH);
  int c = (int)(t0 / (size_t)outH);

  float maxScaled = -INFINITY;
  for (int ky = 0; ky < kH; ++ky) {
    int ih = oh * stride + ky - padding;
    for (int kx = 0; kx < kW; ++kx) {
      int iw = ow * stride + kx - padding;
      float v = 0.0f;
      if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
        size_t inIdx = ((size_t)c * (size_t)inH + (size_t)ih) * (size_t)inW + (size_t)iw;
        v = input[inIdx];
      }
      maxScaled = fmaxf(maxScaled, beta * v);
    }
  }

  float sumExp = 0.0f;
  for (int ky = 0; ky < kH; ++ky) {
    int ih = oh * stride + ky - padding;
    for (int kx = 0; kx < kW; ++kx) {
      int iw = ow * stride + kx - padding;
      float v = 0.0f;
      if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
        size_t inIdx = ((size_t)c * (size_t)inH + (size_t)ih) * (size_t)inW + (size_t)iw;
        v = input[inIdx];
      }
      sumExp += expf(beta * v - maxScaled);
    }
  }

  output[idx] = (maxScaled + logf(sumExp)) / beta;
}

__global__ void smoothmaxpool2d_bwd_kernel(const float* input, const float* gradOutput, float* dInput,
                                          int inC, int inH, int inW,
                                          int outH, int outW,
                                          int kH, int kW,
                                          int stride, int padding,
                                          float beta) {
  size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  size_t total = (size_t)inC * (size_t)outH * (size_t)outW;
  if (idx >= total) return;

  int ow = (int)(idx % (size_t)outW);
  size_t t0 = idx / (size_t)outW;
  int oh = (int)(t0 % (size_t)outH);
  int c = (int)(t0 / (size_t)outH);

  float maxScaled = -INFINITY;
  for (int ky = 0; ky < kH; ++ky) {
    int ih = oh * stride + ky - padding;
    for (int kx = 0; kx < kW; ++kx) {
      int iw = ow * stride + kx - padding;
      float v = 0.0f;
      if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
        size_t inIdx = ((size_t)c * (size_t)inH + (size_t)ih) * (size_t)inW + (size_t)iw;
        v = input[inIdx];
      }
      maxScaled = fmaxf(maxScaled, beta * v);
    }
  }

  float sumExp = 0.0f;
  for (int ky = 0; ky < kH; ++ky) {
    int ih = oh * stride + ky - padding;
    for (int kx = 0; kx < kW; ++kx) {
      int iw = ow * stride + kx - padding;
      float v = 0.0f;
      if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
        size_t inIdx = ((size_t)c * (size_t)inH + (size_t)ih) * (size_t)inW + (size_t)iw;
        v = input[inIdx];
      }
      sumExp += expf(beta * v - maxScaled);
    }
  }

  float g = gradOutput[idx];

  for (int ky = 0; ky < kH; ++ky) {
    int ih = oh * stride + ky - padding;
    if (ih < 0 || ih >= inH) continue;
    for (int kx = 0; kx < kW; ++kx) {
      int iw = ow * stride + kx - padding;
      if (iw < 0 || iw >= inW) continue;
      size_t inIdx = ((size_t)c * (size_t)inH + (size_t)ih) * (size_t)inW + (size_t)iw;
      float v = input[inIdx];
      float e = expf(beta * v - maxScaled);
      float w = e / sumExp;
      atomicAdd(&dInput[inIdx], g * w);
    }
  }
}

// Deterministic backward for smoothmaxpool2d: compute each dInput element exactly once.
//
// This matches smoothmaxpool2d_bwd_kernel semantics, including:
// - padding-as-zero in the window
// - log-sum-exp stabilization via maxScaled
// - no gradient flowing to padding elements
__global__ void smoothmaxpool2d_bwd_det_kernel(const float* input, const float* gradOutput, float* dInput,
                                              int inC, int inH, int inW,
                                              int outH, int outW,
                                              int kH, int kW,
                                              int stride, int padding,
                                              float beta) {
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t total = (size_t)inC * (size_t)inH * (size_t)inW;
  if (idx >= total) return;

  const int iw = (int)(idx % (size_t)inW);
  size_t t0 = idx / (size_t)inW;
  const int ih = (int)(t0 % (size_t)inH);
  const int c = (int)(t0 / (size_t)inH);

  int64_t ohMin = ceil_div_i64((int64_t)ih + (int64_t)padding - (int64_t)(kH - 1), (int64_t)stride);
  int64_t ohMax = floor_div_i64((int64_t)ih + (int64_t)padding, (int64_t)stride);
  int64_t owMin = ceil_div_i64((int64_t)iw + (int64_t)padding - (int64_t)(kW - 1), (int64_t)stride);
  int64_t owMax = floor_div_i64((int64_t)iw + (int64_t)padding, (int64_t)stride);

  if (ohMin < 0) ohMin = 0;
  if (owMin < 0) owMin = 0;
  if (ohMax > (int64_t)outH - 1) ohMax = (int64_t)outH - 1;
  if (owMax > (int64_t)outW - 1) owMax = (int64_t)outW - 1;

  const float vSelf = input[idx];
  float acc = 0.0f;
  if (ohMin <= ohMax && owMin <= owMax) {
    for (int64_t oh = ohMin; oh <= ohMax; ++oh) {
      const int kySelf = ih + padding - (int)oh * stride;
      if (kySelf < 0 || kySelf >= kH) continue;
      for (int64_t ow = owMin; ow <= owMax; ++ow) {
        const int kxSelf = iw + padding - (int)ow * stride;
        if (kxSelf < 0 || kxSelf >= kW) continue;

        // Recompute (maxScaled, sumExp) for this output window.
        float maxScaled = -INFINITY;
        for (int ky = 0; ky < kH; ++ky) {
          int candH = (int)oh * stride + ky - padding;
          for (int kx = 0; kx < kW; ++kx) {
            int candW = (int)ow * stride + kx - padding;
            float v = 0.0f;
            if (candH >= 0 && candH < inH && candW >= 0 && candW < inW) {
              size_t inIdx = ((size_t)c * (size_t)inH + (size_t)candH) * (size_t)inW + (size_t)candW;
              v = input[inIdx];
            }
            maxScaled = fmaxf(maxScaled, beta * v);
          }
        }

        float sumExp = 0.0f;
        for (int ky = 0; ky < kH; ++ky) {
          int candH = (int)oh * stride + ky - padding;
          for (int kx = 0; kx < kW; ++kx) {
            int candW = (int)ow * stride + kx - padding;
            float v = 0.0f;
            if (candH >= 0 && candH < inH && candW >= 0 && candW < inW) {
              size_t inIdx = ((size_t)c * (size_t)inH + (size_t)candH) * (size_t)inW + (size_t)candW;
              v = input[inIdx];
            }
            sumExp += expf(beta * v - maxScaled);
          }
        }

        const float w = expf(beta * vSelf - maxScaled) / sumExp;
        const size_t outIdx = ((size_t)c * (size_t)outH + (size_t)oh) * (size_t)outW + (size_t)ow;
        acc += gradOutput[outIdx] * w;
      }
    }
  }

  dInput[idx] = acc;
}

// -------------------------
// N-D conv/pooling kernels
// -------------------------

__global__ void convnd_fwd_kernel(const float* input, const float* kernel, const float* bias,
                                  float* output,
                                  uint32_t inC, uint32_t outC,
                                  const uint32_t* inSpatial,
                                  const uint32_t* outSpatial,
                                  const uint32_t* kSpatial,
                                  const uint32_t* stride,
                                  const uint32_t* padding,
                                  int rank,
                                  size_t outSpatialSize,
                                  size_t kSpatialSize) {
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t outElems = (size_t)outC * outSpatialSize;
  if (idx >= outElems) return;

  const uint32_t oc = (uint32_t)(idx / outSpatialSize);
  const size_t spatialIdx = idx - (size_t)oc * outSpatialSize;

  uint32_t outCoord[kMaxRank];
  size_t tmp = spatialIdx;
  for (int ax = rank - 1; ax >= 0; --ax) {
    const uint32_t d = outSpatial[ax];
    outCoord[ax] = (uint32_t)(tmp % (size_t)d);
    tmp /= (size_t)d;
  }

  float acc = bias ? bias[oc] : 0.0f;

  uint32_t kCoord[kMaxRank];
  for (uint32_t ic = 0; ic < inC; ++ic) {
    for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
      size_t ktmp = kIdx;
      for (int ax = rank - 1; ax >= 0; --ax) {
        const uint32_t kd = kSpatial[ax];
        kCoord[ax] = (uint32_t)(ktmp % (size_t)kd);
        ktmp /= (size_t)kd;
      }

      bool inBounds = true;
      size_t inIdx = (size_t)ic;
      for (int ax = 0; ax < rank; ++ax) {
        const int pos = (int)outCoord[ax] * (int)stride[ax] + (int)kCoord[ax] - (int)padding[ax];
        const uint32_t dim = inSpatial[ax];
        if (pos < 0 || (uint32_t)pos >= dim) {
          inBounds = false;
          break;
        }
        inIdx = inIdx * (size_t)dim + (size_t)(uint32_t)pos;
      }
      if (!inBounds) continue;

      size_t wIdx = (size_t)oc;
      wIdx = wIdx * (size_t)inC + (size_t)ic;
      for (int ax = 0; ax < rank; ++ax) {
        const uint32_t kd = kSpatial[ax];
        wIdx = wIdx * (size_t)kd + (size_t)kCoord[ax];
      }

      acc += input[inIdx] * kernel[wIdx];
    }
  }

  output[idx] = acc;
}

__global__ void convnd_dbias_kernel(const float* gradOutput, float* dBias,
                                   uint32_t outC, size_t outSpatialSize) {
  const uint32_t oc = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (oc >= outC) return;
  float acc = 0.0f;
  const size_t base = (size_t)oc * outSpatialSize;
  for (size_t i = 0; i < outSpatialSize; ++i) {
    acc += gradOutput[base + i];
  }
  dBias[oc] = acc;
}

__global__ void convnd_dkernel_kernel(const float* input, const float* gradOutput,
                                      float* dKernel,
                                      uint32_t inC, uint32_t outC,
                                      const uint32_t* inSpatial,
                                      const uint32_t* outSpatial,
                                      const uint32_t* kSpatial,
                                      const uint32_t* stride,
                                      const uint32_t* padding,
                                      int rank,
                                      size_t outSpatialSize,
                                      size_t kSpatialSize) {
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t total = (size_t)outC * (size_t)inC * kSpatialSize;
  if (idx >= total) return;

  size_t tmp = idx;
  const size_t kIdx = tmp % kSpatialSize;
  tmp /= kSpatialSize;
  const uint32_t ic = (uint32_t)(tmp % (size_t)inC);
  const uint32_t oc = (uint32_t)(tmp / (size_t)inC);

  uint32_t kCoord[kMaxRank];
  size_t ktmp = kIdx;
  for (int ax = rank - 1; ax >= 0; --ax) {
    const uint32_t kd = kSpatial[ax];
    kCoord[ax] = (uint32_t)(ktmp % (size_t)kd);
    ktmp /= (size_t)kd;
  }

  float acc = 0.0f;

  uint32_t outCoord[kMaxRank];
  for (size_t outIdx = 0; outIdx < outSpatialSize; ++outIdx) {
    size_t otmp = outIdx;
    for (int ax = rank - 1; ax >= 0; --ax) {
      const uint32_t od = outSpatial[ax];
      outCoord[ax] = (uint32_t)(otmp % (size_t)od);
      otmp /= (size_t)od;
    }

    bool inBounds = true;
    size_t inIdx = (size_t)ic;
    for (int ax = 0; ax < rank; ++ax) {
      const int pos = (int)outCoord[ax] * (int)stride[ax] + (int)kCoord[ax] - (int)padding[ax];
      const uint32_t dim = inSpatial[ax];
      if (pos < 0 || (uint32_t)pos >= dim) {
        inBounds = false;
        break;
      }
      inIdx = inIdx * (size_t)dim + (size_t)(uint32_t)pos;
    }
    if (!inBounds) continue;

    const size_t goIdx = (size_t)oc * outSpatialSize + outIdx;
    acc += input[inIdx] * gradOutput[goIdx];
  }

  dKernel[idx] = acc;
}

__global__ void convnd_dinput_kernel(const float* kernel, const float* gradOutput,
                                     float* dInput,
                                     uint32_t inC, uint32_t outC,
                                     const uint32_t* inSpatial,
                                     const uint32_t* outSpatial,
                                     const uint32_t* kSpatial,
                                     const uint32_t* stride,
                                     const uint32_t* padding,
                                     int rank,
                                     size_t inSpatialSize,
                                     size_t outSpatialSize,
                                     size_t kSpatialSize) {
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t total = (size_t)inC * inSpatialSize;
  if (idx >= total) return;

  const uint32_t ic = (uint32_t)(idx / inSpatialSize);
  const size_t spatialIdx = idx - (size_t)ic * inSpatialSize;

  uint32_t inCoord[kMaxRank];
  size_t tmp = spatialIdx;
  for (int ax = rank - 1; ax >= 0; --ax) {
    const uint32_t id = inSpatial[ax];
    inCoord[ax] = (uint32_t)(tmp % (size_t)id);
    tmp /= (size_t)id;
  }

  uint32_t kCoord[kMaxRank];
  uint32_t outCoord[kMaxRank];

  float acc = 0.0f;
  for (uint32_t oc = 0; oc < outC; ++oc) {
    for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
      size_t ktmp = kIdx;
      for (int ax = rank - 1; ax >= 0; --ax) {
        const uint32_t kd = kSpatial[ax];
        kCoord[ax] = (uint32_t)(ktmp % (size_t)kd);
        ktmp /= (size_t)kd;
      }

      bool ok = true;
      for (int ax = 0; ax < rank; ++ax) {
        const int num = (int)inCoord[ax] + (int)padding[ax] - (int)kCoord[ax];
        const uint32_t s = stride[ax];
        if (num < 0) {
          ok = false;
          break;
        }
        if ((num % (int)s) != 0) {
          ok = false;
          break;
        }
        const uint32_t o = (uint32_t)(num / (int)s);
        if (o >= outSpatial[ax]) {
          ok = false;
          break;
        }
        outCoord[ax] = o;
      }
      if (!ok) continue;

      size_t outIdx = 0;
      for (int ax = 0; ax < rank; ++ax) {
        outIdx = outIdx * (size_t)outSpatial[ax] + (size_t)outCoord[ax];
      }
      const size_t goIdx = (size_t)oc * outSpatialSize + outIdx;

      size_t wIdx = (size_t)oc;
      wIdx = wIdx * (size_t)inC + (size_t)ic;
      for (int ax = 0; ax < rank; ++ax) {
        const uint32_t kd = kSpatial[ax];
        wIdx = wIdx * (size_t)kd + (size_t)kCoord[ax];
      }

      acc += gradOutput[goIdx] * kernel[wIdx];
    }
  }

  dInput[idx] = acc;
}

__global__ void maxpoolnd_fwd_kernel(const float* input, float* output,
                                     uint32_t inC,
                                     const uint32_t* inSpatial,
                                     const uint32_t* outSpatial,
                                     const uint32_t* kSpatial,
                                     const uint32_t* stride,
                                     const uint32_t* padding,
                                     int rank,
                                     size_t outSpatialSize,
                                     size_t kSpatialSize) {
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t outElems = (size_t)inC * outSpatialSize;
  if (idx >= outElems) return;

  const uint32_t c = (uint32_t)(idx / outSpatialSize);
  const size_t spatialIdx = idx - (size_t)c * outSpatialSize;

  uint32_t outCoord[kMaxRank];
  size_t tmp = spatialIdx;
  for (int ax = rank - 1; ax >= 0; --ax) {
    const uint32_t od = outSpatial[ax];
    outCoord[ax] = (uint32_t)(tmp % (size_t)od);
    tmp /= (size_t)od;
  }

  float best = -FLT_MAX;
  uint32_t kCoord[kMaxRank];
  for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
    size_t ktmp = kIdx;
    for (int ax = rank - 1; ax >= 0; --ax) {
      const uint32_t kd = kSpatial[ax];
      kCoord[ax] = (uint32_t)(ktmp % (size_t)kd);
      ktmp /= (size_t)kd;
    }

    bool inBounds = true;
    size_t inIdx = (size_t)c;
    for (int ax = 0; ax < rank; ++ax) {
      const int pos = (int)outCoord[ax] * (int)stride[ax] + (int)kCoord[ax] - (int)padding[ax];
      const uint32_t dim = inSpatial[ax];
      if (pos < 0 || (uint32_t)pos >= dim) {
        inBounds = false;
        break;
      }
      inIdx = inIdx * (size_t)dim + (size_t)(uint32_t)pos;
    }

    float v = -INFINITY;
    if (inBounds) {
      v = input[inIdx];
    }
    if (v > best) {
      best = v;
    }
  }

  output[idx] = best;
}

__global__ void maxpoolnd_bwd_kernel(const float* input, const float* gradOutput, float* dInput,
                                     uint32_t inC,
                                     const uint32_t* inSpatial,
                                     const uint32_t* outSpatial,
                                     const uint32_t* kSpatial,
                                     const uint32_t* stride,
                                     const uint32_t* padding,
                                     int rank,
                                     size_t outSpatialSize,
                                     size_t kSpatialSize) {
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t outElems = (size_t)inC * outSpatialSize;
  if (idx >= outElems) return;

  const uint32_t c = (uint32_t)(idx / outSpatialSize);
  const size_t spatialIdx = idx - (size_t)c * outSpatialSize;

  uint32_t outCoord[kMaxRank];
  size_t tmp = spatialIdx;
  for (int ax = rank - 1; ax >= 0; --ax) {
    const uint32_t od = outSpatial[ax];
    outCoord[ax] = (uint32_t)(tmp % (size_t)od);
    tmp /= (size_t)od;
  }

  float best = -FLT_MAX;
  bool bestValid = false;
  uint32_t bestInCoord[kMaxRank];
  uint32_t candInCoord[kMaxRank];
  uint32_t kCoord[kMaxRank];

  for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
    size_t ktmp = kIdx;
    for (int ax = rank - 1; ax >= 0; --ax) {
      const uint32_t kd = kSpatial[ax];
      kCoord[ax] = (uint32_t)(ktmp % (size_t)kd);
      ktmp /= (size_t)kd;
    }

    bool inBounds = true;
    for (int ax = 0; ax < rank; ++ax) {
      const int pos = (int)outCoord[ax] * (int)stride[ax] + (int)kCoord[ax] - (int)padding[ax];
      const uint32_t dim = inSpatial[ax];
      if (pos < 0 || (uint32_t)pos >= dim) {
        inBounds = false;
        break;
      }
      candInCoord[ax] = (uint32_t)pos;
    }

    float v = -INFINITY;
    if (inBounds) {
      size_t inIdx = (size_t)c;
      for (int ax = 0; ax < rank; ++ax) {
        inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)candInCoord[ax];
      }
      v = input[inIdx];
    }

    if (v > best) {
      best = v;
      bestValid = inBounds;
      if (inBounds) {
        for (int ax = 0; ax < rank; ++ax) {
          bestInCoord[ax] = candInCoord[ax];
        }
      }
    }
  }

  if (bestValid) {
    size_t inIdx = (size_t)c;
    for (int ax = 0; ax < rank; ++ax) {
      inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)bestInCoord[ax];
    }
    atomicAdd(&dInput[inIdx], gradOutput[idx]);
  }
}

// Deterministic backward for maxpool (N-D): compute each dInput element exactly once.
// See maxpool2d_bwd_det_kernel for the conceptual explanation.
__global__ void maxpoolnd_bwd_det_kernel(const float* input, const float* gradOutput, float* dInput,
                                        uint32_t inC,
                                        const uint32_t* inSpatial,
                                        const uint32_t* outSpatial,
                                        const uint32_t* kSpatial,
                                        const uint32_t* stride,
                                        const uint32_t* padding,
                                        int rank,
                                        size_t inSpatialSize,
                                        size_t outSpatialSize,
                                        size_t kSpatialSize) {
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t inElems = (size_t)inC * inSpatialSize;
  if (idx >= inElems) return;

  const uint32_t c = (uint32_t)(idx / inSpatialSize);
  const size_t spatialIdx = idx - (size_t)c * inSpatialSize;

  uint32_t inCoord[kMaxRank];
  size_t tmp = spatialIdx;
  for (int ax = rank - 1; ax >= 0; --ax) {
    const uint32_t d = inSpatial[ax];
    inCoord[ax] = (uint32_t)(tmp % (size_t)d);
    tmp /= (size_t)d;
  }

  // Output coordinate ranges per axis.
  int64_t oMin[kMaxRank];
  size_t rangeSize[kMaxRank];
  size_t totalComb = 1;
  for (int ax = 0; ax < rank; ++ax) {
    const int64_t s = (int64_t)stride[ax];
    const int64_t p = (int64_t)padding[ax];
    const int64_t kd = (int64_t)kSpatial[ax];
    const int64_t ocMin = ceil_div_i64((int64_t)inCoord[ax] + p - (kd - 1), s);
    const int64_t ocMax = floor_div_i64((int64_t)inCoord[ax] + p, s);
    int64_t lo = ocMin;
    int64_t hi = ocMax;
    if (lo < 0) lo = 0;
    const int64_t outD = (int64_t)outSpatial[ax];
    if (hi > outD - 1) hi = outD - 1;
    oMin[ax] = lo;
    if (lo > hi) {
      totalComb = 0;
      break;
    }
    const size_t sz = (size_t)(hi - lo + 1);
    rangeSize[ax] = sz;
    totalComb *= sz;
  }

  float acc = 0.0f;
  if (totalComb > 0) {
    uint32_t outCoord[kMaxRank];
    uint32_t kCoord[kMaxRank];
    uint32_t bestInCoord[kMaxRank];
    uint32_t candInCoord[kMaxRank];

    for (size_t t = 0; t < totalComb; ++t) {
      // Unflatten t into per-axis offsets (last axis varies fastest).
      size_t tt = t;
      for (int ax = rank - 1; ax >= 0; --ax) {
        const size_t sz = rangeSize[ax];
        const size_t off = (sz == 0) ? 0 : (tt % sz);
        tt = (sz == 0) ? 0 : (tt / sz);
        outCoord[ax] = (uint32_t)(oMin[ax] + (int64_t)off);
      }

      // Exact inclusion check.
      bool includes = true;
      for (int ax = 0; ax < rank; ++ax) {
        const int64_t k = (int64_t)inCoord[ax] + (int64_t)padding[ax] -
                          (int64_t)outCoord[ax] * (int64_t)stride[ax];
        if (k < 0 || k >= (int64_t)kSpatial[ax]) {
          includes = false;
          break;
        }
      }
      if (!includes) continue;

      // Flatten outCoord.
      size_t outIdx = 0;
      for (int ax = 0; ax < rank; ++ax) {
        outIdx = outIdx * (size_t)outSpatial[ax] + (size_t)outCoord[ax];
      }
      const size_t goIdx = (size_t)c * outSpatialSize + outIdx;

      // Recompute argmax for this output window, ignoring padded cells and matching
      // tie-breaking of the atomic kernel (strict > keeps the first row-major argmax).
      float best = -FLT_MAX;
      bool bestValid = false;
      for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
        size_t ktmp = kIdx;
        for (int ax = rank - 1; ax >= 0; --ax) {
          const uint32_t kd2 = kSpatial[ax];
          kCoord[ax] = (uint32_t)(ktmp % (size_t)kd2);
          ktmp /= (size_t)kd2;
        }

        bool inBounds = true;
        for (int ax = 0; ax < rank; ++ax) {
          const int64_t pos64 =
              (int64_t)outCoord[ax] * (int64_t)stride[ax] + (int64_t)kCoord[ax] - (int64_t)padding[ax];
          const uint32_t dim = inSpatial[ax];
          if (pos64 < 0 || (uint64_t)pos64 >= (uint64_t)dim) {
            inBounds = false;
            break;
          }
          candInCoord[ax] = (uint32_t)pos64;
        }

        float v = -INFINITY;
        if (inBounds) {
          size_t inIdx = (size_t)c;
          for (int ax = 0; ax < rank; ++ax) {
            inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)candInCoord[ax];
          }
          v = input[inIdx];
        }

        if (v > best) {
          best = v;
          bestValid = inBounds;
          if (inBounds) {
            for (int ax = 0; ax < rank; ++ax) {
              bestInCoord[ax] = candInCoord[ax];
            }
          }
        }
      }

      if (!bestValid) continue;
      bool match = true;
      for (int ax = 0; ax < rank; ++ax) {
        if (bestInCoord[ax] != inCoord[ax]) {
          match = false;
          break;
        }
      }
      if (match) {
        acc += gradOutput[goIdx];
      }
    }
  }

  dInput[idx] = acc;
}

__global__ void avgpoolnd_fwd_kernel(const float* input, float* output,
                                     uint32_t inC,
                                     const uint32_t* inSpatial,
                                     const uint32_t* outSpatial,
                                     const uint32_t* kSpatial,
                                     const uint32_t* stride,
                                     const uint32_t* padding,
                                     int rank,
                                     size_t outSpatialSize,
                                     size_t kSpatialSize) {
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t outElems = (size_t)inC * outSpatialSize;
  if (idx >= outElems) return;

  const uint32_t c = (uint32_t)(idx / outSpatialSize);
  const size_t spatialIdx = idx - (size_t)c * outSpatialSize;

  uint32_t outCoord[kMaxRank];
  size_t tmp = spatialIdx;
  for (int ax = rank - 1; ax >= 0; --ax) {
    const uint32_t od = outSpatial[ax];
    outCoord[ax] = (uint32_t)(tmp % (size_t)od);
    tmp /= (size_t)od;
  }

  float acc = 0.0f;
  uint32_t kCoord[kMaxRank];
  for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
    size_t ktmp = kIdx;
    for (int ax = rank - 1; ax >= 0; --ax) {
      const uint32_t kd = kSpatial[ax];
      kCoord[ax] = (uint32_t)(ktmp % (size_t)kd);
      ktmp /= (size_t)kd;
    }

    bool inBounds = true;
    size_t inIdx = (size_t)c;
    for (int ax = 0; ax < rank; ++ax) {
      const int pos = (int)outCoord[ax] * (int)stride[ax] + (int)kCoord[ax] - (int)padding[ax];
      const uint32_t dim = inSpatial[ax];
      if (pos < 0 || (uint32_t)pos >= dim) {
        inBounds = false;
        break;
      }
      inIdx = inIdx * (size_t)dim + (size_t)(uint32_t)pos;
    }

    if (inBounds) {
      acc += input[inIdx];
    }
    // else: out-of-bounds contributes 0 and is still counted (count_include_pad=true).
  }

  output[idx] = acc / (float)(kSpatialSize);
}

__global__ void avgpoolnd_bwd_kernel(const float* gradOutput, float* dInput,
                                     uint32_t inC,
                                     const uint32_t* inSpatial,
                                     const uint32_t* outSpatial,
                                     const uint32_t* kSpatial,
                                     const uint32_t* stride,
                                     const uint32_t* padding,
                                     int rank,
                                     size_t outSpatialSize,
                                     size_t kSpatialSize) {
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t outElems = (size_t)inC * outSpatialSize;
  if (idx >= outElems) return;

  const uint32_t c = (uint32_t)(idx / outSpatialSize);
  const size_t spatialIdx = idx - (size_t)c * outSpatialSize;

  uint32_t outCoord[kMaxRank];
  size_t tmp = spatialIdx;
  for (int ax = rank - 1; ax >= 0; --ax) {
    const uint32_t od = outSpatial[ax];
    outCoord[ax] = (uint32_t)(tmp % (size_t)od);
    tmp /= (size_t)od;
  }

  const float g = gradOutput[idx] / (float)(kSpatialSize);

  uint32_t kCoord[kMaxRank];
  for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
    size_t ktmp = kIdx;
    for (int ax = rank - 1; ax >= 0; --ax) {
      const uint32_t kd = kSpatial[ax];
      kCoord[ax] = (uint32_t)(ktmp % (size_t)kd);
      ktmp /= (size_t)kd;
    }

    bool inBounds = true;
    size_t inIdx = (size_t)c;
    for (int ax = 0; ax < rank; ++ax) {
      const int pos = (int)outCoord[ax] * (int)stride[ax] + (int)kCoord[ax] - (int)padding[ax];
      const uint32_t dim = inSpatial[ax];
      if (pos < 0 || (uint32_t)pos >= dim) {
        inBounds = false;
        break;
      }
      inIdx = inIdx * (size_t)dim + (size_t)(uint32_t)pos;
    }

    if (inBounds) {
      atomicAdd(&dInput[inIdx], g);
    }
  }
}

// Deterministic backward for avgpool (N-D): compute each dInput element exactly once.
__global__ void avgpoolnd_bwd_det_kernel(const float* gradOutput, float* dInput,
                                        uint32_t inC,
                                        const uint32_t* inSpatial,
                                        const uint32_t* outSpatial,
                                        const uint32_t* kSpatial,
                                        const uint32_t* stride,
                                        const uint32_t* padding,
                                        int rank,
                                        size_t inSpatialSize,
                                        size_t outSpatialSize,
                                        size_t kSpatialSize) {
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t inElems = (size_t)inC * inSpatialSize;
  if (idx >= inElems) return;

  const uint32_t c = (uint32_t)(idx / inSpatialSize);
  const size_t spatialIdx = idx - (size_t)c * inSpatialSize;

  uint32_t inCoord[kMaxRank];
  size_t tmp = spatialIdx;
  for (int ax = rank - 1; ax >= 0; --ax) {
    const uint32_t d = inSpatial[ax];
    inCoord[ax] = (uint32_t)(tmp % (size_t)d);
    tmp /= (size_t)d;
  }

  int64_t oMin[kMaxRank];
  size_t rangeSize[kMaxRank];
  size_t totalComb = 1;
  for (int ax = 0; ax < rank; ++ax) {
    const int64_t s = (int64_t)stride[ax];
    const int64_t p = (int64_t)padding[ax];
    const int64_t kd = (int64_t)kSpatial[ax];
    const int64_t ocMin = ceil_div_i64((int64_t)inCoord[ax] + p - (kd - 1), s);
    const int64_t ocMax = floor_div_i64((int64_t)inCoord[ax] + p, s);
    int64_t lo = ocMin;
    int64_t hi = ocMax;
    if (lo < 0) lo = 0;
    const int64_t outD = (int64_t)outSpatial[ax];
    if (hi > outD - 1) hi = outD - 1;
    oMin[ax] = lo;
    if (lo > hi) {
      totalComb = 0;
      break;
    }
    const size_t sz = (size_t)(hi - lo + 1);
    rangeSize[ax] = sz;
    totalComb *= sz;
  }

  const float invDenom = 1.0f / (float)kSpatialSize;
  float acc = 0.0f;
  if (totalComb > 0) {
    uint32_t outCoord[kMaxRank];
    for (size_t t = 0; t < totalComb; ++t) {
      size_t tt = t;
      for (int ax = rank - 1; ax >= 0; --ax) {
        const size_t sz = rangeSize[ax];
        const size_t off = (sz == 0) ? 0 : (tt % sz);
        tt = (sz == 0) ? 0 : (tt / sz);
        outCoord[ax] = (uint32_t)(oMin[ax] + (int64_t)off);
      }

      // Exact inclusion check.
      bool includes = true;
      for (int ax = 0; ax < rank; ++ax) {
        const int64_t k = (int64_t)inCoord[ax] + (int64_t)padding[ax] -
                          (int64_t)outCoord[ax] * (int64_t)stride[ax];
        if (k < 0 || k >= (int64_t)kSpatial[ax]) {
          includes = false;
          break;
        }
      }
      if (!includes) continue;

      size_t outIdx = 0;
      for (int ax = 0; ax < rank; ++ax) {
        outIdx = outIdx * (size_t)outSpatial[ax] + (size_t)outCoord[ax];
      }
      const size_t goIdx = (size_t)c * outSpatialSize + outIdx;
      acc += gradOutput[goIdx] * invDenom;
    }
  }

  dInput[idx] = acc;
}

__global__ void smoothmaxpoolnd_fwd_kernel(const float* input, float* output,
                                          uint32_t inC,
                                          const uint32_t* inSpatial,
                                          const uint32_t* outSpatial,
                                          const uint32_t* kSpatial,
                                          const uint32_t* stride,
                                          const uint32_t* padding,
                                          int rank,
                                          size_t outSpatialSize,
                                          size_t kSpatialSize,
                                          float beta) {
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t outElems = (size_t)inC * outSpatialSize;
  if (idx >= outElems) return;

  const uint32_t c = (uint32_t)(idx / outSpatialSize);
  const size_t spatialIdx = idx - (size_t)c * outSpatialSize;

  uint32_t outCoord[kMaxRank];
  size_t tmp = spatialIdx;
  for (int ax = rank - 1; ax >= 0; --ax) {
    const uint32_t od = outSpatial[ax];
    outCoord[ax] = (uint32_t)(tmp % (size_t)od);
    tmp /= (size_t)od;
  }

  float maxScaled = -INFINITY;
  uint32_t kCoord[kMaxRank];
  for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
    size_t ktmp = kIdx;
    for (int ax = rank - 1; ax >= 0; --ax) {
      const uint32_t kd = kSpatial[ax];
      kCoord[ax] = (uint32_t)(ktmp % (size_t)kd);
      ktmp /= (size_t)kd;
    }

    bool inBounds = true;
    size_t inIdx = (size_t)c;
    for (int ax = 0; ax < rank; ++ax) {
      const int pos = (int)outCoord[ax] * (int)stride[ax] + (int)kCoord[ax] - (int)padding[ax];
      const uint32_t dim = inSpatial[ax];
      if (pos < 0 || (uint32_t)pos >= dim) {
        inBounds = false;
        break;
      }
      inIdx = inIdx * (size_t)dim + (size_t)(uint32_t)pos;
    }

    float v = 0.0f;
    if (inBounds) {
      v = input[inIdx];
    }
    maxScaled = fmaxf(maxScaled, beta * v);
  }

  float sumExp = 0.0f;
  for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
    size_t ktmp = kIdx;
    for (int ax = rank - 1; ax >= 0; --ax) {
      const uint32_t kd = kSpatial[ax];
      kCoord[ax] = (uint32_t)(ktmp % (size_t)kd);
      ktmp /= (size_t)kd;
    }

    bool inBounds = true;
    size_t inIdx = (size_t)c;
    for (int ax = 0; ax < rank; ++ax) {
      const int pos = (int)outCoord[ax] * (int)stride[ax] + (int)kCoord[ax] - (int)padding[ax];
      const uint32_t dim = inSpatial[ax];
      if (pos < 0 || (uint32_t)pos >= dim) {
        inBounds = false;
        break;
      }
      inIdx = inIdx * (size_t)dim + (size_t)(uint32_t)pos;
    }

    float v = 0.0f;
    if (inBounds) {
      v = input[inIdx];
    }
    sumExp += expf(beta * v - maxScaled);
  }

  output[idx] = (maxScaled + logf(sumExp)) / beta;
}

__global__ void smoothmaxpoolnd_bwd_kernel(const float* input, const float* gradOutput, float* dInput,
                                          uint32_t inC,
                                          const uint32_t* inSpatial,
                                          const uint32_t* outSpatial,
                                          const uint32_t* kSpatial,
                                          const uint32_t* stride,
                                          const uint32_t* padding,
                                          int rank,
                                          size_t outSpatialSize,
                                          size_t kSpatialSize,
                                          float beta) {
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t outElems = (size_t)inC * outSpatialSize;
  if (idx >= outElems) return;

  const uint32_t c = (uint32_t)(idx / outSpatialSize);
  const size_t spatialIdx = idx - (size_t)c * outSpatialSize;

  uint32_t outCoord[kMaxRank];
  size_t tmp = spatialIdx;
  for (int ax = rank - 1; ax >= 0; --ax) {
    const uint32_t od = outSpatial[ax];
    outCoord[ax] = (uint32_t)(tmp % (size_t)od);
    tmp /= (size_t)od;
  }

  float maxScaled = -INFINITY;
  uint32_t kCoord[kMaxRank];
  for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
    size_t ktmp = kIdx;
    for (int ax = rank - 1; ax >= 0; --ax) {
      const uint32_t kd = kSpatial[ax];
      kCoord[ax] = (uint32_t)(ktmp % (size_t)kd);
      ktmp /= (size_t)kd;
    }

    bool inBounds = true;
    size_t inIdx = (size_t)c;
    for (int ax = 0; ax < rank; ++ax) {
      const int pos = (int)outCoord[ax] * (int)stride[ax] + (int)kCoord[ax] - (int)padding[ax];
      const uint32_t dim = inSpatial[ax];
      if (pos < 0 || (uint32_t)pos >= dim) {
        inBounds = false;
        break;
      }
      inIdx = inIdx * (size_t)dim + (size_t)(uint32_t)pos;
    }

    float v = 0.0f;
    if (inBounds) {
      v = input[inIdx];
    }
    maxScaled = fmaxf(maxScaled, beta * v);
  }

  float sumExp = 0.0f;
  for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
    size_t ktmp = kIdx;
    for (int ax = rank - 1; ax >= 0; --ax) {
      const uint32_t kd = kSpatial[ax];
      kCoord[ax] = (uint32_t)(ktmp % (size_t)kd);
      ktmp /= (size_t)kd;
    }

    bool inBounds = true;
    size_t inIdx = (size_t)c;
    for (int ax = 0; ax < rank; ++ax) {
      const int pos = (int)outCoord[ax] * (int)stride[ax] + (int)kCoord[ax] - (int)padding[ax];
      const uint32_t dim = inSpatial[ax];
      if (pos < 0 || (uint32_t)pos >= dim) {
        inBounds = false;
        break;
      }
      inIdx = inIdx * (size_t)dim + (size_t)(uint32_t)pos;
    }

    float v = 0.0f;
    if (inBounds) {
      v = input[inIdx];
    }
    sumExp += expf(beta * v - maxScaled);
  }

  const float g = gradOutput[idx];

  for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
    size_t ktmp = kIdx;
    for (int ax = rank - 1; ax >= 0; --ax) {
      const uint32_t kd = kSpatial[ax];
      kCoord[ax] = (uint32_t)(ktmp % (size_t)kd);
      ktmp /= (size_t)kd;
    }

    bool inBounds = true;
    size_t inIdx = (size_t)c;
    for (int ax = 0; ax < rank; ++ax) {
      const int pos = (int)outCoord[ax] * (int)stride[ax] + (int)kCoord[ax] - (int)padding[ax];
      const uint32_t dim = inSpatial[ax];
      if (pos < 0 || (uint32_t)pos >= dim) {
        inBounds = false;
        break;
      }
      inIdx = inIdx * (size_t)dim + (size_t)(uint32_t)pos;
    }
    if (!inBounds) continue;

    const float v = input[inIdx];
    const float e = expf(beta * v - maxScaled);
    const float w = e / sumExp;
    atomicAdd(&dInput[inIdx], g * w);
  }
}

// Deterministic backward for smoothmaxpool (N-D): compute each dInput element exactly once.
__global__ void smoothmaxpoolnd_bwd_det_kernel(const float* input, const float* gradOutput, float* dInput,
                                              uint32_t inC,
                                              const uint32_t* inSpatial,
                                              const uint32_t* outSpatial,
                                              const uint32_t* kSpatial,
                                              const uint32_t* stride,
                                              const uint32_t* padding,
                                              int rank,
                                              size_t inSpatialSize,
                                              size_t outSpatialSize,
                                              size_t kSpatialSize,
                                              float beta) {
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t inElems = (size_t)inC * inSpatialSize;
  if (idx >= inElems) return;

  const uint32_t c = (uint32_t)(idx / inSpatialSize);
  const size_t spatialIdx = idx - (size_t)c * inSpatialSize;

  uint32_t inCoord[kMaxRank];
  size_t tmp = spatialIdx;
  for (int ax = rank - 1; ax >= 0; --ax) {
    const uint32_t d = inSpatial[ax];
    inCoord[ax] = (uint32_t)(tmp % (size_t)d);
    tmp /= (size_t)d;
  }

  int64_t oMin[kMaxRank];
  size_t rangeSize[kMaxRank];
  size_t totalComb = 1;
  for (int ax = 0; ax < rank; ++ax) {
    const int64_t s = (int64_t)stride[ax];
    const int64_t p = (int64_t)padding[ax];
    const int64_t kd = (int64_t)kSpatial[ax];
    const int64_t ocMin = ceil_div_i64((int64_t)inCoord[ax] + p - (kd - 1), s);
    const int64_t ocMax = floor_div_i64((int64_t)inCoord[ax] + p, s);
    int64_t lo = ocMin;
    int64_t hi = ocMax;
    if (lo < 0) lo = 0;
    const int64_t outD = (int64_t)outSpatial[ax];
    if (hi > outD - 1) hi = outD - 1;
    oMin[ax] = lo;
    if (lo > hi) {
      totalComb = 0;
      break;
    }
    const size_t sz = (size_t)(hi - lo + 1);
    rangeSize[ax] = sz;
    totalComb *= sz;
  }

  const float vSelf = input[idx];
  float acc = 0.0f;
  if (totalComb > 0) {
    uint32_t outCoord[kMaxRank];
    uint32_t kCoord[kMaxRank];
    for (size_t t = 0; t < totalComb; ++t) {
      size_t tt = t;
      for (int ax = rank - 1; ax >= 0; --ax) {
        const size_t sz = rangeSize[ax];
        const size_t off = (sz == 0) ? 0 : (tt % sz);
        tt = (sz == 0) ? 0 : (tt / sz);
        outCoord[ax] = (uint32_t)(oMin[ax] + (int64_t)off);
      }

      // Exact inclusion check.
      bool includes = true;
      for (int ax = 0; ax < rank; ++ax) {
        const int64_t k = (int64_t)inCoord[ax] + (int64_t)padding[ax] -
                          (int64_t)outCoord[ax] * (int64_t)stride[ax];
        if (k < 0 || k >= (int64_t)kSpatial[ax]) {
          includes = false;
          break;
        }
      }
      if (!includes) continue;

      // Flatten outCoord.
      size_t outIdx = 0;
      for (int ax = 0; ax < rank; ++ax) {
        outIdx = outIdx * (size_t)outSpatial[ax] + (size_t)outCoord[ax];
      }
      const size_t goIdx = (size_t)c * outSpatialSize + outIdx;

      // Recompute (maxScaled, sumExp) for this output window.
      float maxScaled = -INFINITY;
      for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
        size_t ktmp = kIdx;
        for (int ax = rank - 1; ax >= 0; --ax) {
          const uint32_t kd2 = kSpatial[ax];
          kCoord[ax] = (uint32_t)(ktmp % (size_t)kd2);
          ktmp /= (size_t)kd2;
        }

        bool inBounds = true;
        size_t inIdx = (size_t)c;
        for (int ax = 0; ax < rank; ++ax) {
          const int64_t pos64 =
              (int64_t)outCoord[ax] * (int64_t)stride[ax] + (int64_t)kCoord[ax] - (int64_t)padding[ax];
          const uint32_t dim = inSpatial[ax];
          if (pos64 < 0 || (uint64_t)pos64 >= (uint64_t)dim) {
            inBounds = false;
            break;
          }
          inIdx = inIdx * (size_t)dim + (size_t)(uint32_t)pos64;
        }

        float v = 0.0f;
        if (inBounds) {
          v = input[inIdx];
        }
        maxScaled = fmaxf(maxScaled, beta * v);
      }

      float sumExp = 0.0f;
      for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
        size_t ktmp = kIdx;
        for (int ax = rank - 1; ax >= 0; --ax) {
          const uint32_t kd2 = kSpatial[ax];
          kCoord[ax] = (uint32_t)(ktmp % (size_t)kd2);
          ktmp /= (size_t)kd2;
        }

        bool inBounds = true;
        size_t inIdx = (size_t)c;
        for (int ax = 0; ax < rank; ++ax) {
          const int64_t pos64 =
              (int64_t)outCoord[ax] * (int64_t)stride[ax] + (int64_t)kCoord[ax] - (int64_t)padding[ax];
          const uint32_t dim = inSpatial[ax];
          if (pos64 < 0 || (uint64_t)pos64 >= (uint64_t)dim) {
            inBounds = false;
            break;
          }
          inIdx = inIdx * (size_t)dim + (size_t)(uint32_t)pos64;
        }

        float v = 0.0f;
        if (inBounds) {
          v = input[inIdx];
        }
        sumExp += expf(beta * v - maxScaled);
      }

      const float w = expf(beta * vSelf - maxScaled) / sumExp;
      acc += gradOutput[goIdx] * w;
    }
  }

  dInput[idx] = acc;
}

// -------------------------
// Lean FFI entrypoints
// -------------------------

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_conv2d_fwd(
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

  checkBufSize(input, inElems, "torchlean_cuda_conv2d_fwd: input.size mismatch");
  checkBufSize(kernel, kElems, "torchlean_cuda_conv2d_fwd: kernel.size mismatch");
  checkBufSize(bias, bElems, "torchlean_cuda_conv2d_fwd: bias.size mismatch");

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outElems);
  if (outElems == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  int grid = grid_for(outElems);
  conv2d_fwd_kernel<<<grid, kBlock>>>(input->data, kernel->data, bias->data, out->data,
                                     (int)inC, (int)inH, (int)inW,
                                     (int)outC, (int)outH, (int)outW,
                                     (int)kH, (int)kW,
                                     (int)stride, (int)padding);
  checkCuda(cudaGetLastError(), "conv2d_fwd kernel launch failed");
  checkCuda(cudaDeviceSynchronize(), "conv2d_fwd sync failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_conv2d_bwd(
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

  checkBufSize(input, inElems, "torchlean_cuda_conv2d_bwd: input.size mismatch");
  checkBufSize(kernel, kElems, "torchlean_cuda_conv2d_bwd: kernel.size mismatch");
  checkBufSize(gradOutput, outElems, "torchlean_cuda_conv2d_bwd: gradOutput.size mismatch");

  torchlean_cuda_buffer* dKernel = torchlean_cuda_buffer_alloc(kElems);
  torchlean_cuda_buffer* dBias = torchlean_cuda_buffer_alloc((size_t)outC);
  torchlean_cuda_buffer* dInput = torchlean_cuda_buffer_alloc(inElems);

  // dInput is the only output that may be nonempty even if outC==0. It is fully written by the
  // kernel; for safety with edge cases (e.g. inElems==0), memset is fine.
  if (dKernel->size > 0) {
    checkCuda(cudaMemset(dKernel->data, 0, dKernel->size * sizeof(float)), "cudaMemset dKernel failed");
  }
  if (dBias->size > 0) {
    checkCuda(cudaMemset(dBias->data, 0, dBias->size * sizeof(float)), "cudaMemset dBias failed");
  }
  if (dInput->size > 0) {
    checkCuda(cudaMemset(dInput->data, 0, dInput->size * sizeof(float)), "cudaMemset dInput failed");
  }

  if (dBias->size > 0) {
    int gridBias = (int)(((size_t)outC + (size_t)kBlock - 1) / (size_t)kBlock);
    conv2d_dbias_kernel<<<gridBias, kBlock>>>(gradOutput->data, dBias->data,
                                              (int)outC, (int)outH, (int)outW);
    checkCuda(cudaGetLastError(), "conv2d_dbias kernel launch failed");
  }

  if (dKernel->size > 0) {
    int gridK = (int)((kElems + (size_t)kBlock - 1) / (size_t)kBlock);
    conv2d_dkernel_kernel<<<gridK, kBlock>>>(input->data, gradOutput->data, dKernel->data,
                                             (int)inC, (int)inH, (int)inW,
                                             (int)outC, (int)outH, (int)outW,
                                             (int)kH, (int)kW,
                                             (int)stride, (int)padding);
    checkCuda(cudaGetLastError(), "conv2d_dkernel kernel launch failed");
  }

  if (dInput->size > 0) {
    int gridIn = (int)((inElems + (size_t)kBlock - 1) / (size_t)kBlock);
    conv2d_dinput_kernel<<<gridIn, kBlock>>>(kernel->data, gradOutput->data, dInput->data,
                                             (int)inC, (int)inH, (int)inW,
                                             (int)outC, (int)outH, (int)outW,
                                             (int)kH, (int)kW,
                                             (int)stride, (int)padding);
    checkCuda(cudaGetLastError(), "conv2d_dinput kernel launch failed");
  }

  checkCuda(cudaDeviceSynchronize(), "conv2d_bwd sync failed");

  // Return (dKernel, dBias, dInput) as `Buffer × Buffer × Buffer`.
  lean_object* inner = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(inner, 0, torchlean_cuda_buffer_box(dBias));
  lean_ctor_set(inner, 1, torchlean_cuda_buffer_box(dInput));

  lean_object* outer = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(outer, 0, torchlean_cuda_buffer_box(dKernel));
  lean_ctor_set(outer, 1, inner);
  return outer;
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_convtranspose2d_fwd(
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

  checkBufSize(input, inElems, "torchlean_cuda_convtranspose2d_fwd: input.size mismatch");
  checkBufSize(kernel, kElems, "torchlean_cuda_convtranspose2d_fwd: kernel.size mismatch");
  checkBufSize(bias, bElems, "torchlean_cuda_convtranspose2d_fwd: bias.size mismatch");

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outElems);
  if (outElems == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  // Reuse the Conv2D dInput kernel: it computes transpose-convolution when we swap channel roles.
  // See notes in `Cuda.Ops.conv_transpose2d` for the shape correspondence.
  int grid = grid_for(outElems);
  conv2d_dinput_kernel<<<grid, kBlock>>>(kernel->data, input->data, out->data,
                                         (int)outC, (int)outH, (int)outW,
                                         (int)inC, (int)inH, (int)inW,
                                         (int)kH, (int)kW,
                                         (int)stride, (int)padding);
  checkCuda(cudaGetLastError(), "convtranspose2d_fwd dinput kernel launch failed");

  add_bias_chw_kernel<<<grid, kBlock>>>(out->data, bias->data, (int)outC, (int)outH, (int)outW);
  checkCuda(cudaGetLastError(), "convtranspose2d_fwd add_bias kernel launch failed");

  checkCuda(cudaDeviceSynchronize(), "convtranspose2d_fwd sync failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_convtranspose2d_bwd(
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

  checkBufSize(input, inElems, "torchlean_cuda_convtranspose2d_bwd: input.size mismatch");
  checkBufSize(kernel, kElems, "torchlean_cuda_convtranspose2d_bwd: kernel.size mismatch");
  checkBufSize(gradOutput, outElems, "torchlean_cuda_convtranspose2d_bwd: gradOutput.size mismatch");

  torchlean_cuda_buffer* dKernel = torchlean_cuda_buffer_alloc(kElems);
  torchlean_cuda_buffer* dBias = torchlean_cuda_buffer_alloc((size_t)outC);
  torchlean_cuda_buffer* dInput = torchlean_cuda_buffer_alloc(inElems);

  if (dBias->size > 0) {
    int gridBias = (int)(((size_t)outC + (size_t)kBlock - 1) / (size_t)kBlock);
    conv2d_dbias_kernel<<<gridBias, kBlock>>>(gradOutput->data, dBias->data,
                                              (int)outC, (int)outH, (int)outW);
    checkCuda(cudaGetLastError(), "convtranspose2d_dbias kernel launch failed");
  }

  if (dKernel->size > 0) {
    int gridK = (int)((kElems + (size_t)kBlock - 1) / (size_t)kBlock);
    convtranspose2d_dkernel_kernel<<<gridK, kBlock>>>(input->data, gradOutput->data, dKernel->data,
                                                      (int)inC, (int)inH, (int)inW,
                                                      (int)outC, (int)outH, (int)outW,
                                                      (int)kH, (int)kW,
                                                      (int)stride, (int)padding);
    checkCuda(cudaGetLastError(), "convtranspose2d_dkernel kernel launch failed");
  }

  if (dInput->size > 0) {
    int gridIn = (int)((inElems + (size_t)kBlock - 1) / (size_t)kBlock);
    conv2d_fwd_kernel<<<gridIn, kBlock>>>(gradOutput->data, kernel->data, /*bias=*/nullptr, dInput->data,
                                         (int)outC, (int)outH, (int)outW,
                                         (int)inC, (int)inH, (int)inW,
                                         (int)kH, (int)kW,
                                         (int)stride, (int)padding);
    checkCuda(cudaGetLastError(), "convtranspose2d_dinput (conv2d_fwd) kernel launch failed");
  }

  checkCuda(cudaDeviceSynchronize(), "convtranspose2d_bwd sync failed");

  // Return (dKernel, dBias, dInput) as `Buffer × Buffer × Buffer`.
  lean_object* inner = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(inner, 0, torchlean_cuda_buffer_box(dBias));
  lean_ctor_set(inner, 1, torchlean_cuda_buffer_box(dInput));

  lean_object* outer = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(outer, 0, torchlean_cuda_buffer_box(dKernel));
  lean_ctor_set(outer, 1, inner);
  return outer;
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_maxpool2d_fwd(
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

  checkBufSize(input, inElems, "torchlean_cuda_maxpool2d_fwd: input.size mismatch");

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outElems);
  if (outElems == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  int grid = grid_for(outElems);
  maxpool2d_fwd_kernel<<<grid, kBlock>>>(input->data, out->data,
                                        (int)inC, (int)inH, (int)inW,
                                        (int)outH, (int)outW,
                                        (int)kH, (int)kW,
                                        (int)stride, (int)padding);
  checkCuda(cudaGetLastError(), "maxpool2d_fwd kernel launch failed");
  checkCuda(cudaDeviceSynchronize(), "maxpool2d_fwd sync failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_maxpool2d_bwd(
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

  checkBufSize(input, inElems, "torchlean_cuda_maxpool2d_bwd: input.size mismatch");
  checkBufSize(gradOutput, outElems, "torchlean_cuda_maxpool2d_bwd: gradOutput.size mismatch");

  torchlean_cuda_buffer* dInput = torchlean_cuda_buffer_alloc(inElems);
  const bool det = torchlean_cuda_deterministic_reductions_enabled();
  if (!det && dInput->size > 0) {
    // Atomic path accumulates into dInput; must clear first.
    checkCuda(cudaMemset(dInput->data, 0, dInput->size * sizeof(float)), "cudaMemset dInput failed");
  }

  if (det) {
    if (inElems > 0) {
      const int grid = grid_for(inElems);
      maxpool2d_bwd_det_kernel<<<grid, kBlock>>>(input->data, gradOutput->data, dInput->data,
                                                (int)inC, (int)inH, (int)inW,
                                                (int)outH, (int)outW,
                                                (int)kH, (int)kW,
                                                (int)stride, (int)padding);
      checkCuda(cudaGetLastError(), "maxpool2d_bwd (deterministic) kernel launch failed");
      checkCuda(cudaDeviceSynchronize(), "maxpool2d_bwd (deterministic) sync failed");
    }
  } else {
    if (outElems > 0) {
      const int grid = grid_for(outElems);
      maxpool2d_bwd_kernel<<<grid, kBlock>>>(input->data, gradOutput->data, dInput->data,
                                            (int)inC, (int)inH, (int)inW,
                                            (int)outH, (int)outW,
                                            (int)kH, (int)kW,
                                            (int)stride, (int)padding);
      checkCuda(cudaGetLastError(), "maxpool2d_bwd kernel launch failed");
      checkCuda(cudaDeviceSynchronize(), "maxpool2d_bwd sync failed");
    }
  }
  return torchlean_cuda_buffer_box(dInput);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_smooth_maxpool2d_fwd(
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

  checkBufSize(input, inElems, "torchlean_cuda_smooth_maxpool2d_fwd: input.size mismatch");

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outElems);
  if (outElems == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  int grid = grid_for(outElems);
  const float betaF =
      checked_smoothmax_beta(beta, "torchlean_cuda_smooth_maxpool2d_fwd: beta must be finite and nonzero");
  smoothmaxpool2d_fwd_kernel<<<grid, kBlock>>>(input->data, out->data,
                                              (int)inC, (int)inH, (int)inW,
                                              (int)outH, (int)outW,
                                              (int)kH, (int)kW,
                                              (int)stride, (int)padding,
                                              betaF);
  checkCuda(cudaGetLastError(), "smooth_maxpool2d_fwd kernel launch failed");
  checkCuda(cudaDeviceSynchronize(), "smooth_maxpool2d_fwd sync failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_smooth_maxpool2d_bwd(
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

  checkBufSize(input, inElems, "torchlean_cuda_smooth_maxpool2d_bwd: input.size mismatch");
  checkBufSize(gradOutput, outElems, "torchlean_cuda_smooth_maxpool2d_bwd: gradOutput.size mismatch");

  torchlean_cuda_buffer* dInput = torchlean_cuda_buffer_alloc(inElems);
  const float betaF =
      checked_smoothmax_beta(beta, "torchlean_cuda_smooth_maxpool2d_bwd: beta must be finite and nonzero");
  const bool det = torchlean_cuda_deterministic_reductions_enabled();
  if (!det && dInput->size > 0) {
    // Atomic path accumulates into dInput; must clear first.
    checkCuda(cudaMemset(dInput->data, 0, dInput->size * sizeof(float)), "cudaMemset dInput failed");
  }

  if (det) {
    if (inElems > 0) {
      const int grid = grid_for(inElems);
      smoothmaxpool2d_bwd_det_kernel<<<grid, kBlock>>>(input->data, gradOutput->data, dInput->data,
                                                      (int)inC, (int)inH, (int)inW,
                                                      (int)outH, (int)outW,
                                                      (int)kH, (int)kW,
                                                      (int)stride, (int)padding,
                                                      betaF);
      checkCuda(cudaGetLastError(), "smooth_maxpool2d_bwd (deterministic) kernel launch failed");
      checkCuda(cudaDeviceSynchronize(), "smooth_maxpool2d_bwd (deterministic) sync failed");
    }
  } else {
    if (outElems > 0) {
      const int grid = grid_for(outElems);
      smoothmaxpool2d_bwd_kernel<<<grid, kBlock>>>(input->data, gradOutput->data, dInput->data,
                                                  (int)inC, (int)inH, (int)inW,
                                                  (int)outH, (int)outW,
                                                  (int)kH, (int)kW,
                                                  (int)stride, (int)padding,
                                                  betaF);
      checkCuda(cudaGetLastError(), "smooth_maxpool2d_bwd kernel launch failed");
      checkCuda(cudaDeviceSynchronize(), "smooth_maxpool2d_bwd sync failed");
    }
  }

  return torchlean_cuda_buffer_box(dInput);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_avgpool2d_fwd(
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

  checkBufSize(input, inElems, "torchlean_cuda_avgpool2d_fwd: input.size mismatch");

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outElems);
  if (outElems == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  int grid = grid_for(outElems);
  avgpool2d_fwd_kernel<<<grid, kBlock>>>(input->data, out->data,
                                        (int)inC, (int)inH, (int)inW,
                                        (int)outH, (int)outW,
                                        (int)kH, (int)kW,
                                        (int)stride, (int)padding);
  checkCuda(cudaGetLastError(), "avgpool2d_fwd kernel launch failed");
  checkCuda(cudaDeviceSynchronize(), "avgpool2d_fwd sync failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_avgpool2d_bwd(
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

  checkBufSize(gradOutput, outElems, "torchlean_cuda_avgpool2d_bwd: gradOutput.size mismatch");

  torchlean_cuda_buffer* dInput = torchlean_cuda_buffer_alloc(inElems);
  const bool det = torchlean_cuda_deterministic_reductions_enabled();
  if (!det && dInput->size > 0) {
    // Atomic path accumulates into dInput; must clear first.
    checkCuda(cudaMemset(dInput->data, 0, dInput->size * sizeof(float)), "cudaMemset dInput failed");
  }

  if (det) {
    if (inElems > 0) {
      const int grid = grid_for(inElems);
      avgpool2d_bwd_det_kernel<<<grid, kBlock>>>(gradOutput->data, dInput->data,
                                                (int)inC, (int)inH, (int)inW,
                                                (int)outH, (int)outW,
                                                (int)kH, (int)kW,
                                                (int)stride, (int)padding);
      checkCuda(cudaGetLastError(), "avgpool2d_bwd (deterministic) kernel launch failed");
      checkCuda(cudaDeviceSynchronize(), "avgpool2d_bwd (deterministic) sync failed");
    }
  } else {
    if (outElems > 0) {
      const int grid = grid_for(outElems);
      avgpool2d_bwd_kernel<<<grid, kBlock>>>(gradOutput->data, dInput->data,
                                            (int)inC, (int)inH, (int)inW,
                                            (int)outH, (int)outW,
                                            (int)kH, (int)kW,
                                            (int)stride, (int)padding);
      checkCuda(cudaGetLastError(), "avgpool2d_bwd kernel launch failed");
      checkCuda(cudaDeviceSynchronize(), "avgpool2d_bwd sync failed");
    }
  }

  return torchlean_cuda_buffer_box(dInput);
}

// -------------------------
// N-D Conv + Pooling entrypoints
// -------------------------

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_conv_fwd(
    b_lean_obj_arg inputObj, b_lean_obj_arg kernelObj, b_lean_obj_arg biasObj,
    b_lean_obj_arg inSpatialObj, b_lean_obj_arg kernelSpatialObj,
    b_lean_obj_arg strideObj, b_lean_obj_arg paddingObj,
    uint32_t inC, uint32_t outC) {
  torchlean_cuda_buffer* input = torchlean_cuda_buffer_unbox(inputObj);
  torchlean_cuda_buffer* kernel = torchlean_cuda_buffer_unbox(kernelObj);
  torchlean_cuda_buffer* bias = torchlean_cuda_buffer_unbox(biasObj);

  const int rank = read_rank_checked(inSpatialObj, "torchlean_cuda_conv_fwd: bad inSpatial");
  if (read_rank_checked(kernelSpatialObj, "torchlean_cuda_conv_fwd: bad kernelSpatial") != rank ||
      read_rank_checked(strideObj, "torchlean_cuda_conv_fwd: bad stride") != rank ||
      read_rank_checked(paddingObj, "torchlean_cuda_conv_fwd: bad padding") != rank) {
    lean_internal_panic("torchlean_cuda_conv_fwd: array rank mismatch");
  }
  if (rank <= 0) {
    lean_internal_panic("torchlean_cuda_conv_fwd: rank must be > 0");
  }

  uint32_t hInSpatial[kMaxRank];
  uint32_t hKSpatial[kMaxRank];
  uint32_t hStride[kMaxRank];
  uint32_t hPadding[kMaxRank];
  uint32_t hOutSpatial[kMaxRank];

  read_u32_array(inSpatialObj, hInSpatial, rank, "torchlean_cuda_conv_fwd: bad inSpatial");
  read_u32_array(kernelSpatialObj, hKSpatial, rank, "torchlean_cuda_conv_fwd: bad kernelSpatial");
  read_u32_array(strideObj, hStride, rank, "torchlean_cuda_conv_fwd: bad stride");
  read_u32_array(paddingObj, hPadding, rank, "torchlean_cuda_conv_fwd: bad padding");

  for (int ax = 0; ax < rank; ++ax) {
    if (hKSpatial[ax] == 0) {
      lean_internal_panic("torchlean_cuda_conv_fwd: kernel dims must be > 0");
    }
    if (hStride[ax] == 0) {
      lean_internal_panic("torchlean_cuda_conv_fwd: stride dims must be > 0");
    }
    hOutSpatial[ax] = outDim(hInSpatial[ax], hKSpatial[ax], hStride[ax], hPadding[ax]);
  }

  const size_t inSpatialSize = prod_u32(hInSpatial, rank);
  const size_t kSpatialSize = prod_u32(hKSpatial, rank);
  const size_t outSpatialSize = prod_u32(hOutSpatial, rank);

  const size_t inElems = (size_t)inC * inSpatialSize;
  const size_t kElems = (size_t)outC * (size_t)inC * kSpatialSize;
  const size_t bElems = (size_t)outC;
  const size_t outElems = (size_t)outC * outSpatialSize;

  checkBufSize(input, inElems, "torchlean_cuda_conv_fwd: input.size mismatch");
  checkBufSize(kernel, kElems, "torchlean_cuda_conv_fwd: kernel.size mismatch");
  checkBufSize(bias, bElems, "torchlean_cuda_conv_fwd: bias.size mismatch");

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outElems);
  if (outElems == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  uint32_t* dInSpatial = nullptr;
  uint32_t* dOutSpatial = nullptr;
  uint32_t* dKSpatial = nullptr;
  uint32_t* dStride = nullptr;
  uint32_t* dPadding = nullptr;
  checkCuda(cudaMalloc((void**)&dInSpatial, (size_t)rank * sizeof(uint32_t)), "cudaMalloc inSpatial failed");
  checkCuda(cudaMalloc((void**)&dOutSpatial, (size_t)rank * sizeof(uint32_t)), "cudaMalloc outSpatial failed");
  checkCuda(cudaMalloc((void**)&dKSpatial, (size_t)rank * sizeof(uint32_t)), "cudaMalloc kSpatial failed");
  checkCuda(cudaMalloc((void**)&dStride, (size_t)rank * sizeof(uint32_t)), "cudaMalloc stride failed");
  checkCuda(cudaMalloc((void**)&dPadding, (size_t)rank * sizeof(uint32_t)), "cudaMalloc padding failed");

  checkCuda(cudaMemcpy(dInSpatial, hInSpatial, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy inSpatial failed");
  checkCuda(cudaMemcpy(dOutSpatial, hOutSpatial, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy outSpatial failed");
  checkCuda(cudaMemcpy(dKSpatial, hKSpatial, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy kSpatial failed");
  checkCuda(cudaMemcpy(dStride, hStride, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy stride failed");
  checkCuda(cudaMemcpy(dPadding, hPadding, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy padding failed");

  const int grid = grid_for(outElems);
  convnd_fwd_kernel<<<grid, kBlock>>>(input->data, kernel->data, bias->data, out->data,
                                     inC, outC,
                                     dInSpatial, dOutSpatial, dKSpatial, dStride, dPadding,
                                     rank, outSpatialSize, kSpatialSize);
  checkCuda(cudaGetLastError(), "convnd_fwd kernel launch failed");
  checkCuda(cudaDeviceSynchronize(), "convnd_fwd sync failed");

  torchlean_cuda_free_checked((void**)&dInSpatial, "cudaFree convnd inSpatial failed");
  torchlean_cuda_free_checked((void**)&dOutSpatial, "cudaFree convnd outSpatial failed");
  torchlean_cuda_free_checked((void**)&dKSpatial, "cudaFree convnd kSpatial failed");
  torchlean_cuda_free_checked((void**)&dStride, "cudaFree convnd stride failed");
  torchlean_cuda_free_checked((void**)&dPadding, "cudaFree convnd padding failed");

  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_conv_bwd(
    b_lean_obj_arg inputObj, b_lean_obj_arg kernelObj, b_lean_obj_arg gradObj,
    b_lean_obj_arg inSpatialObj, b_lean_obj_arg kernelSpatialObj,
    b_lean_obj_arg strideObj, b_lean_obj_arg paddingObj,
    uint32_t inC, uint32_t outC) {
  torchlean_cuda_buffer* input = torchlean_cuda_buffer_unbox(inputObj);
  torchlean_cuda_buffer* kernel = torchlean_cuda_buffer_unbox(kernelObj);
  torchlean_cuda_buffer* gradOutput = torchlean_cuda_buffer_unbox(gradObj);

  const int rank = read_rank_checked(inSpatialObj, "torchlean_cuda_conv_bwd: bad inSpatial");
  if (read_rank_checked(kernelSpatialObj, "torchlean_cuda_conv_bwd: bad kernelSpatial") != rank ||
      read_rank_checked(strideObj, "torchlean_cuda_conv_bwd: bad stride") != rank ||
      read_rank_checked(paddingObj, "torchlean_cuda_conv_bwd: bad padding") != rank) {
    lean_internal_panic("torchlean_cuda_conv_bwd: array rank mismatch");
  }
  if (rank <= 0) {
    lean_internal_panic("torchlean_cuda_conv_bwd: rank must be > 0");
  }

  uint32_t hInSpatial[kMaxRank];
  uint32_t hKSpatial[kMaxRank];
  uint32_t hStride[kMaxRank];
  uint32_t hPadding[kMaxRank];
  uint32_t hOutSpatial[kMaxRank];

  read_u32_array(inSpatialObj, hInSpatial, rank, "torchlean_cuda_conv_bwd: bad inSpatial");
  read_u32_array(kernelSpatialObj, hKSpatial, rank, "torchlean_cuda_conv_bwd: bad kernelSpatial");
  read_u32_array(strideObj, hStride, rank, "torchlean_cuda_conv_bwd: bad stride");
  read_u32_array(paddingObj, hPadding, rank, "torchlean_cuda_conv_bwd: bad padding");

  for (int ax = 0; ax < rank; ++ax) {
    if (hKSpatial[ax] == 0) {
      lean_internal_panic("torchlean_cuda_conv_bwd: kernel dims must be > 0");
    }
    if (hStride[ax] == 0) {
      lean_internal_panic("torchlean_cuda_conv_bwd: stride dims must be > 0");
    }
    hOutSpatial[ax] = outDim(hInSpatial[ax], hKSpatial[ax], hStride[ax], hPadding[ax]);
  }

  const size_t inSpatialSize = prod_u32(hInSpatial, rank);
  const size_t kSpatialSize = prod_u32(hKSpatial, rank);
  const size_t outSpatialSize = prod_u32(hOutSpatial, rank);

  const size_t inElems = (size_t)inC * inSpatialSize;
  const size_t kElems = (size_t)outC * (size_t)inC * kSpatialSize;
  const size_t outElems = (size_t)outC * outSpatialSize;

  checkBufSize(input, inElems, "torchlean_cuda_conv_bwd: input.size mismatch");
  checkBufSize(kernel, kElems, "torchlean_cuda_conv_bwd: kernel.size mismatch");
  checkBufSize(gradOutput, outElems, "torchlean_cuda_conv_bwd: gradOutput.size mismatch");

  torchlean_cuda_buffer* dKernel = torchlean_cuda_buffer_alloc(kElems);
  torchlean_cuda_buffer* dBias = torchlean_cuda_buffer_alloc((size_t)outC);
  torchlean_cuda_buffer* dInput = torchlean_cuda_buffer_alloc(inElems);

  if (kElems == 0 && (size_t)outC == 0 && inElems == 0) {
    // Return the empty buffers without allocating shape arrays on the device.
    lean_object* inner = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(inner, 0, torchlean_cuda_buffer_box(dBias));
    lean_ctor_set(inner, 1, torchlean_cuda_buffer_box(dInput));
    lean_object* outer = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(outer, 0, torchlean_cuda_buffer_box(dKernel));
    lean_ctor_set(outer, 1, inner);
    return outer;
  }

  uint32_t* dInSpatial = nullptr;
  uint32_t* dOutSpatial = nullptr;
  uint32_t* dKSpatial = nullptr;
  uint32_t* dStride = nullptr;
  uint32_t* dPadding = nullptr;
  checkCuda(cudaMalloc((void**)&dInSpatial, (size_t)rank * sizeof(uint32_t)), "cudaMalloc inSpatial failed");
  checkCuda(cudaMalloc((void**)&dOutSpatial, (size_t)rank * sizeof(uint32_t)), "cudaMalloc outSpatial failed");
  checkCuda(cudaMalloc((void**)&dKSpatial, (size_t)rank * sizeof(uint32_t)), "cudaMalloc kSpatial failed");
  checkCuda(cudaMalloc((void**)&dStride, (size_t)rank * sizeof(uint32_t)), "cudaMalloc stride failed");
  checkCuda(cudaMalloc((void**)&dPadding, (size_t)rank * sizeof(uint32_t)), "cudaMalloc padding failed");

  checkCuda(cudaMemcpy(dInSpatial, hInSpatial, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy inSpatial failed");
  checkCuda(cudaMemcpy(dOutSpatial, hOutSpatial, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy outSpatial failed");
  checkCuda(cudaMemcpy(dKSpatial, hKSpatial, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy kSpatial failed");
  checkCuda(cudaMemcpy(dStride, hStride, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy stride failed");
  checkCuda(cudaMemcpy(dPadding, hPadding, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy padding failed");

  if ((size_t)outC > 0) {
    const int gridBias = grid_for((size_t)outC);
    convnd_dbias_kernel<<<gridBias, kBlock>>>(gradOutput->data, dBias->data, outC, outSpatialSize);
    checkCuda(cudaGetLastError(), "convnd_dbias kernel launch failed");
  }

  if (kElems > 0) {
    const int gridK = grid_for(kElems);
    convnd_dkernel_kernel<<<gridK, kBlock>>>(input->data, gradOutput->data, dKernel->data,
                                            inC, outC,
                                            dInSpatial, dOutSpatial, dKSpatial, dStride, dPadding,
                                            rank, outSpatialSize, kSpatialSize);
    checkCuda(cudaGetLastError(), "convnd_dkernel kernel launch failed");
  }

  if (inElems > 0) {
    const int gridIn = grid_for(inElems);
    convnd_dinput_kernel<<<gridIn, kBlock>>>(kernel->data, gradOutput->data, dInput->data,
                                            inC, outC,
                                            dInSpatial, dOutSpatial, dKSpatial, dStride, dPadding,
                                            rank, inSpatialSize, outSpatialSize, kSpatialSize);
    checkCuda(cudaGetLastError(), "convnd_dinput kernel launch failed");
  }

  checkCuda(cudaDeviceSynchronize(), "convnd_bwd sync failed");

  torchlean_cuda_free_checked((void**)&dInSpatial, "cudaFree convnd_bwd inSpatial failed");
  torchlean_cuda_free_checked((void**)&dOutSpatial, "cudaFree convnd_bwd outSpatial failed");
  torchlean_cuda_free_checked((void**)&dKSpatial, "cudaFree convnd_bwd kSpatial failed");
  torchlean_cuda_free_checked((void**)&dStride, "cudaFree convnd_bwd stride failed");
  torchlean_cuda_free_checked((void**)&dPadding, "cudaFree convnd_bwd padding failed");

  lean_object* inner = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(inner, 0, torchlean_cuda_buffer_box(dBias));
  lean_ctor_set(inner, 1, torchlean_cuda_buffer_box(dInput));

  lean_object* outer = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(outer, 0, torchlean_cuda_buffer_box(dKernel));
  lean_ctor_set(outer, 1, inner);
  return outer;
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_convtranspose_fwd(
    b_lean_obj_arg inputObj, b_lean_obj_arg kernelObj, b_lean_obj_arg biasObj,
    b_lean_obj_arg inSpatialObj, b_lean_obj_arg kernelSpatialObj,
    b_lean_obj_arg strideObj, b_lean_obj_arg paddingObj,
    uint32_t inC, uint32_t outC) {
  torchlean_cuda_buffer* input = torchlean_cuda_buffer_unbox(inputObj);
  torchlean_cuda_buffer* kernel = torchlean_cuda_buffer_unbox(kernelObj);
  torchlean_cuda_buffer* bias = torchlean_cuda_buffer_unbox(biasObj);

  const int rank = read_rank_checked(inSpatialObj, "torchlean_cuda_convtranspose_fwd: bad inSpatial");
  if (read_rank_checked(kernelSpatialObj, "torchlean_cuda_convtranspose_fwd: bad kernelSpatial") != rank ||
      read_rank_checked(strideObj, "torchlean_cuda_convtranspose_fwd: bad stride") != rank ||
      read_rank_checked(paddingObj, "torchlean_cuda_convtranspose_fwd: bad padding") != rank) {
    lean_internal_panic("torchlean_cuda_convtranspose_fwd: array rank mismatch");
  }
  if (rank <= 0) {
    lean_internal_panic("torchlean_cuda_convtranspose_fwd: rank must be > 0");
  }

  uint32_t hInSpatial[kMaxRank];
  uint32_t hKSpatial[kMaxRank];
  uint32_t hStride[kMaxRank];
  uint32_t hPadding[kMaxRank];
  uint32_t hOutSpatial[kMaxRank];

  read_u32_array(inSpatialObj, hInSpatial, rank, "torchlean_cuda_convtranspose_fwd: bad inSpatial");
  read_u32_array(kernelSpatialObj, hKSpatial, rank, "torchlean_cuda_convtranspose_fwd: bad kernelSpatial");
  read_u32_array(strideObj, hStride, rank, "torchlean_cuda_convtranspose_fwd: bad stride");
  read_u32_array(paddingObj, hPadding, rank, "torchlean_cuda_convtranspose_fwd: bad padding");

  for (int ax = 0; ax < rank; ++ax) {
    if (hKSpatial[ax] == 0) {
      lean_internal_panic("torchlean_cuda_convtranspose_fwd: kernel dims must be > 0");
    }
    if (hStride[ax] == 0) {
      lean_internal_panic("torchlean_cuda_convtranspose_fwd: stride dims must be > 0");
    }
    hOutSpatial[ax] = outDimTranspose(hInSpatial[ax], hKSpatial[ax], hStride[ax], hPadding[ax]);
  }

  const size_t inSpatialSize = prod_u32(hInSpatial, rank);
  const size_t kSpatialSize = prod_u32(hKSpatial, rank);
  const size_t outSpatialSize = prod_u32(hOutSpatial, rank);

  const size_t inElems = (size_t)inC * inSpatialSize;
  const size_t kElems = (size_t)inC * (size_t)outC * kSpatialSize;
  const size_t bElems = (size_t)outC;
  const size_t outElems = (size_t)outC * outSpatialSize;

  checkBufSize(input, inElems, "torchlean_cuda_convtranspose_fwd: input.size mismatch");
  checkBufSize(kernel, kElems, "torchlean_cuda_convtranspose_fwd: kernel.size mismatch");
  checkBufSize(bias, bElems, "torchlean_cuda_convtranspose_fwd: bias.size mismatch");

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outElems);
  if (outElems == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  // Implementation detail:
  // A transposed conv forward is equivalent to the `dInput` kernel from the corresponding forward conv,
  // with the spatial/channel roles swapped:
  //   input (inC, inSpatial)   ↔ gradOutput (outC, outSpatial)
  //   output (outC, outSpatial) ↔ dInput    (inC, inSpatial)
  uint32_t* dInSpatial = nullptr;
  uint32_t* dOutSpatial = nullptr;
  uint32_t* dKSpatial = nullptr;
  uint32_t* dStride = nullptr;
  uint32_t* dPadding = nullptr;
  checkCuda(cudaMalloc((void**)&dInSpatial, (size_t)rank * sizeof(uint32_t)), "cudaMalloc inSpatial failed");
  checkCuda(cudaMalloc((void**)&dOutSpatial, (size_t)rank * sizeof(uint32_t)), "cudaMalloc outSpatial failed");
  checkCuda(cudaMalloc((void**)&dKSpatial, (size_t)rank * sizeof(uint32_t)), "cudaMalloc kSpatial failed");
  checkCuda(cudaMalloc((void**)&dStride, (size_t)rank * sizeof(uint32_t)), "cudaMalloc stride failed");
  checkCuda(cudaMalloc((void**)&dPadding, (size_t)rank * sizeof(uint32_t)), "cudaMalloc padding failed");

  // For the conv-dInput kernel, "inSpatial" refers to the transposed output shape.
  checkCuda(cudaMemcpy(dInSpatial, hOutSpatial, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy inSpatial failed");
  // ...and "outSpatial" refers to the original input shape.
  checkCuda(cudaMemcpy(dOutSpatial, hInSpatial, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy outSpatial failed");
  checkCuda(cudaMemcpy(dKSpatial, hKSpatial, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy kSpatial failed");
  checkCuda(cudaMemcpy(dStride, hStride, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy stride failed");
  checkCuda(cudaMemcpy(dPadding, hPadding, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy padding failed");

  const int grid = grid_for(outElems);
  convnd_dinput_kernel<<<grid, kBlock>>>(kernel->data, input->data, out->data,
                                        /*inC=*/outC, /*outC=*/inC,
                                        dInSpatial, dOutSpatial, dKSpatial, dStride, dPadding,
                                        rank, outSpatialSize, inSpatialSize, kSpatialSize);
  checkCuda(cudaGetLastError(), "convtranspose_fwd (convnd_dinput) kernel launch failed");

  if (outC > 0 && outSpatialSize > 0) {
    const int gridBias = grid_for(outElems);
    add_bias_cspatial_kernel<<<gridBias, kBlock>>>(out->data, bias->data, outC, outSpatialSize);
    checkCuda(cudaGetLastError(), "convtranspose_fwd add_bias kernel launch failed");
  }

  checkCuda(cudaDeviceSynchronize(), "convtranspose_fwd sync failed");

  torchlean_cuda_free_checked((void**)&dInSpatial, "cudaFree convtranspose inSpatial failed");
  torchlean_cuda_free_checked((void**)&dOutSpatial, "cudaFree convtranspose outSpatial failed");
  torchlean_cuda_free_checked((void**)&dKSpatial, "cudaFree convtranspose kSpatial failed");
  torchlean_cuda_free_checked((void**)&dStride, "cudaFree convtranspose stride failed");
  torchlean_cuda_free_checked((void**)&dPadding, "cudaFree convtranspose padding failed");

  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_convtranspose_bwd(
    b_lean_obj_arg inputObj, b_lean_obj_arg kernelObj, b_lean_obj_arg gradObj,
    b_lean_obj_arg inSpatialObj, b_lean_obj_arg kernelSpatialObj,
    b_lean_obj_arg strideObj, b_lean_obj_arg paddingObj,
    uint32_t inC, uint32_t outC) {
  torchlean_cuda_buffer* input = torchlean_cuda_buffer_unbox(inputObj);
  torchlean_cuda_buffer* kernel = torchlean_cuda_buffer_unbox(kernelObj);
  torchlean_cuda_buffer* gradOutput = torchlean_cuda_buffer_unbox(gradObj);

  const int rank = read_rank_checked(inSpatialObj, "torchlean_cuda_convtranspose_bwd: bad inSpatial");
  if (read_rank_checked(kernelSpatialObj, "torchlean_cuda_convtranspose_bwd: bad kernelSpatial") != rank ||
      read_rank_checked(strideObj, "torchlean_cuda_convtranspose_bwd: bad stride") != rank ||
      read_rank_checked(paddingObj, "torchlean_cuda_convtranspose_bwd: bad padding") != rank) {
    lean_internal_panic("torchlean_cuda_convtranspose_bwd: array rank mismatch");
  }
  if (rank <= 0) {
    lean_internal_panic("torchlean_cuda_convtranspose_bwd: rank must be > 0");
  }

  uint32_t hInSpatial[kMaxRank];
  uint32_t hKSpatial[kMaxRank];
  uint32_t hStride[kMaxRank];
  uint32_t hPadding[kMaxRank];
  uint32_t hOutSpatial[kMaxRank];

  read_u32_array(inSpatialObj, hInSpatial, rank, "torchlean_cuda_convtranspose_bwd: bad inSpatial");
  read_u32_array(kernelSpatialObj, hKSpatial, rank, "torchlean_cuda_convtranspose_bwd: bad kernelSpatial");
  read_u32_array(strideObj, hStride, rank, "torchlean_cuda_convtranspose_bwd: bad stride");
  read_u32_array(paddingObj, hPadding, rank, "torchlean_cuda_convtranspose_bwd: bad padding");

  for (int ax = 0; ax < rank; ++ax) {
    if (hKSpatial[ax] == 0) {
      lean_internal_panic("torchlean_cuda_convtranspose_bwd: kernel dims must be > 0");
    }
    if (hStride[ax] == 0) {
      lean_internal_panic("torchlean_cuda_convtranspose_bwd: stride dims must be > 0");
    }
    hOutSpatial[ax] = outDimTranspose(hInSpatial[ax], hKSpatial[ax], hStride[ax], hPadding[ax]);
  }

  const size_t inSpatialSize = prod_u32(hInSpatial, rank);
  const size_t kSpatialSize = prod_u32(hKSpatial, rank);
  const size_t outSpatialSize = prod_u32(hOutSpatial, rank);

  const size_t inElems = (size_t)inC * inSpatialSize;
  const size_t kElems = (size_t)inC * (size_t)outC * kSpatialSize;
  const size_t outElems = (size_t)outC * outSpatialSize;

  checkBufSize(input, inElems, "torchlean_cuda_convtranspose_bwd: input.size mismatch");
  checkBufSize(kernel, kElems, "torchlean_cuda_convtranspose_bwd: kernel.size mismatch");
  checkBufSize(gradOutput, outElems, "torchlean_cuda_convtranspose_bwd: gradOutput.size mismatch");

  torchlean_cuda_buffer* dKernel = torchlean_cuda_buffer_alloc(kElems);
  torchlean_cuda_buffer* dBias = torchlean_cuda_buffer_alloc((size_t)outC);
  torchlean_cuda_buffer* dInput = torchlean_cuda_buffer_alloc(inElems);

  if (kElems == 0 && (size_t)outC == 0 && inElems == 0) {
    lean_object* inner = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(inner, 0, torchlean_cuda_buffer_box(dBias));
    lean_ctor_set(inner, 1, torchlean_cuda_buffer_box(dInput));
    lean_object* outer = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(outer, 0, torchlean_cuda_buffer_box(dKernel));
    lean_ctor_set(outer, 1, inner);
    return outer;
  }

  uint32_t* dInSpatial = nullptr;
  uint32_t* dOutSpatial = nullptr;
  uint32_t* dKSpatial = nullptr;
  uint32_t* dStride = nullptr;
  uint32_t* dPadding = nullptr;
  checkCuda(cudaMalloc((void**)&dInSpatial, (size_t)rank * sizeof(uint32_t)), "cudaMalloc inSpatial failed");
  checkCuda(cudaMalloc((void**)&dOutSpatial, (size_t)rank * sizeof(uint32_t)), "cudaMalloc outSpatial failed");
  checkCuda(cudaMalloc((void**)&dKSpatial, (size_t)rank * sizeof(uint32_t)), "cudaMalloc kSpatial failed");
  checkCuda(cudaMalloc((void**)&dStride, (size_t)rank * sizeof(uint32_t)), "cudaMalloc stride failed");
  checkCuda(cudaMalloc((void**)&dPadding, (size_t)rank * sizeof(uint32_t)), "cudaMalloc padding failed");

  // In the conv kernels below, "inSpatial" refers to the transposed output shape.
  checkCuda(cudaMemcpy(dInSpatial, hOutSpatial, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy inSpatial failed");
  // ...and "outSpatial" refers to the original input shape.
  checkCuda(cudaMemcpy(dOutSpatial, hInSpatial, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy outSpatial failed");
  checkCuda(cudaMemcpy(dKSpatial, hKSpatial, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy kSpatial failed");
  checkCuda(cudaMemcpy(dStride, hStride, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy stride failed");
  checkCuda(cudaMemcpy(dPadding, hPadding, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy padding failed");

  if ((size_t)outC > 0) {
    const int gridBias = grid_for((size_t)outC);
    convnd_dbias_kernel<<<gridBias, kBlock>>>(gradOutput->data, dBias->data, outC, outSpatialSize);
    checkCuda(cudaGetLastError(), "convtranspose_bwd dbias kernel launch failed");
  }

  if (kElems > 0) {
    const int gridK = grid_for(kElems);
    convnd_dkernel_kernel<<<gridK, kBlock>>>(gradOutput->data, input->data, dKernel->data,
                                            /*inC=*/outC, /*outC=*/inC,
                                            dInSpatial, dOutSpatial, dKSpatial, dStride, dPadding,
                                            rank, inSpatialSize, kSpatialSize);
    checkCuda(cudaGetLastError(), "convtranspose_bwd dkernel kernel launch failed");
  }

  if (inElems > 0) {
    const int gridIn = grid_for(inElems);
    convnd_fwd_kernel<<<gridIn, kBlock>>>(gradOutput->data, kernel->data, /*bias=*/nullptr, dInput->data,
                                         /*inC=*/outC, /*outC=*/inC,
                                         dInSpatial, dOutSpatial, dKSpatial, dStride, dPadding,
                                         rank, inSpatialSize, kSpatialSize);
    checkCuda(cudaGetLastError(), "convtranspose_bwd dinput (convnd_fwd) kernel launch failed");
  }

  checkCuda(cudaDeviceSynchronize(), "convtranspose_bwd sync failed");

  torchlean_cuda_free_checked((void**)&dInSpatial, "cudaFree convtranspose_bwd inSpatial failed");
  torchlean_cuda_free_checked((void**)&dOutSpatial, "cudaFree convtranspose_bwd outSpatial failed");
  torchlean_cuda_free_checked((void**)&dKSpatial, "cudaFree convtranspose_bwd kSpatial failed");
  torchlean_cuda_free_checked((void**)&dStride, "cudaFree convtranspose_bwd stride failed");
  torchlean_cuda_free_checked((void**)&dPadding, "cudaFree convtranspose_bwd padding failed");

  lean_object* inner = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(inner, 0, torchlean_cuda_buffer_box(dBias));
  lean_ctor_set(inner, 1, torchlean_cuda_buffer_box(dInput));

  lean_object* outer = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(outer, 0, torchlean_cuda_buffer_box(dKernel));
  lean_ctor_set(outer, 1, inner);
  return outer;
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_maxpool_fwd(
    b_lean_obj_arg inputObj,
    b_lean_obj_arg inSpatialObj, b_lean_obj_arg kernelObj,
    b_lean_obj_arg strideObj, b_lean_obj_arg paddingObj,
    uint32_t inC) {
  torchlean_cuda_buffer* input = torchlean_cuda_buffer_unbox(inputObj);

  const int rank = read_rank_checked(inSpatialObj, "torchlean_cuda_maxpool_fwd: bad inSpatial");
  if (read_rank_checked(kernelObj, "torchlean_cuda_maxpool_fwd: bad kernel") != rank ||
      read_rank_checked(strideObj, "torchlean_cuda_maxpool_fwd: bad stride") != rank ||
      read_rank_checked(paddingObj, "torchlean_cuda_maxpool_fwd: bad padding") != rank) {
    lean_internal_panic("torchlean_cuda_maxpool_fwd: array rank mismatch");
  }
  if (rank <= 0) {
    lean_internal_panic("torchlean_cuda_maxpool_fwd: rank must be > 0");
  }

  uint32_t hInSpatial[kMaxRank];
  uint32_t hKSpatial[kMaxRank];
  uint32_t hStride[kMaxRank];
  uint32_t hPadding[kMaxRank];
  uint32_t hOutSpatial[kMaxRank];

  read_u32_array(inSpatialObj, hInSpatial, rank, "torchlean_cuda_maxpool_fwd: bad inSpatial");
  read_u32_array(kernelObj, hKSpatial, rank, "torchlean_cuda_maxpool_fwd: bad kernel");
  read_u32_array(strideObj, hStride, rank, "torchlean_cuda_maxpool_fwd: bad stride");
  read_u32_array(paddingObj, hPadding, rank, "torchlean_cuda_maxpool_fwd: bad padding");

  for (int ax = 0; ax < rank; ++ax) {
    if (hKSpatial[ax] == 0) {
      lean_internal_panic("torchlean_cuda_maxpool_fwd: kernel dims must be > 0");
    }
    if (hStride[ax] == 0) {
      lean_internal_panic("torchlean_cuda_maxpool_fwd: stride dims must be > 0");
    }
    hOutSpatial[ax] = outDim(hInSpatial[ax], hKSpatial[ax], hStride[ax], hPadding[ax]);
  }

  const size_t inSpatialSize = prod_u32(hInSpatial, rank);
  const size_t kSpatialSize = prod_u32(hKSpatial, rank);
  const size_t outSpatialSize = prod_u32(hOutSpatial, rank);

  const size_t inElems = (size_t)inC * inSpatialSize;
  const size_t outElems = (size_t)inC * outSpatialSize;

  checkBufSize(input, inElems, "torchlean_cuda_maxpool_fwd: input.size mismatch");

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outElems);
  if (outElems == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  uint32_t* dInSpatial = nullptr;
  uint32_t* dOutSpatial = nullptr;
  uint32_t* dKSpatial = nullptr;
  uint32_t* dStride = nullptr;
  uint32_t* dPadding = nullptr;
  checkCuda(cudaMalloc((void**)&dInSpatial, (size_t)rank * sizeof(uint32_t)), "cudaMalloc inSpatial failed");
  checkCuda(cudaMalloc((void**)&dOutSpatial, (size_t)rank * sizeof(uint32_t)), "cudaMalloc outSpatial failed");
  checkCuda(cudaMalloc((void**)&dKSpatial, (size_t)rank * sizeof(uint32_t)), "cudaMalloc kSpatial failed");
  checkCuda(cudaMalloc((void**)&dStride, (size_t)rank * sizeof(uint32_t)), "cudaMalloc stride failed");
  checkCuda(cudaMalloc((void**)&dPadding, (size_t)rank * sizeof(uint32_t)), "cudaMalloc padding failed");

  checkCuda(cudaMemcpy(dInSpatial, hInSpatial, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy inSpatial failed");
  checkCuda(cudaMemcpy(dOutSpatial, hOutSpatial, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy outSpatial failed");
  checkCuda(cudaMemcpy(dKSpatial, hKSpatial, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy kSpatial failed");
  checkCuda(cudaMemcpy(dStride, hStride, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy stride failed");
  checkCuda(cudaMemcpy(dPadding, hPadding, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy padding failed");

  const int grid = grid_for(outElems);
  maxpoolnd_fwd_kernel<<<grid, kBlock>>>(input->data, out->data,
                                        inC,
                                        dInSpatial, dOutSpatial, dKSpatial, dStride, dPadding,
                                        rank, outSpatialSize, kSpatialSize);
  checkCuda(cudaGetLastError(), "maxpoolnd_fwd kernel launch failed");
  checkCuda(cudaDeviceSynchronize(), "maxpoolnd_fwd sync failed");

  torchlean_cuda_free_checked((void**)&dInSpatial, "cudaFree maxpoolnd inSpatial failed");
  torchlean_cuda_free_checked((void**)&dOutSpatial, "cudaFree maxpoolnd outSpatial failed");
  torchlean_cuda_free_checked((void**)&dKSpatial, "cudaFree maxpoolnd kSpatial failed");
  torchlean_cuda_free_checked((void**)&dStride, "cudaFree maxpoolnd stride failed");
  torchlean_cuda_free_checked((void**)&dPadding, "cudaFree maxpoolnd padding failed");

  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_maxpool_bwd(
    b_lean_obj_arg inputObj, b_lean_obj_arg gradObj,
    b_lean_obj_arg inSpatialObj, b_lean_obj_arg kernelObj,
    b_lean_obj_arg strideObj, b_lean_obj_arg paddingObj,
    uint32_t inC) {
  torchlean_cuda_buffer* input = torchlean_cuda_buffer_unbox(inputObj);
  torchlean_cuda_buffer* gradOutput = torchlean_cuda_buffer_unbox(gradObj);

  const int rank = read_rank_checked(inSpatialObj, "torchlean_cuda_maxpool_bwd: bad inSpatial");
  if (read_rank_checked(kernelObj, "torchlean_cuda_maxpool_bwd: bad kernel") != rank ||
      read_rank_checked(strideObj, "torchlean_cuda_maxpool_bwd: bad stride") != rank ||
      read_rank_checked(paddingObj, "torchlean_cuda_maxpool_bwd: bad padding") != rank) {
    lean_internal_panic("torchlean_cuda_maxpool_bwd: array rank mismatch");
  }
  if (rank <= 0) {
    lean_internal_panic("torchlean_cuda_maxpool_bwd: rank must be > 0");
  }

  uint32_t hInSpatial[kMaxRank];
  uint32_t hKSpatial[kMaxRank];
  uint32_t hStride[kMaxRank];
  uint32_t hPadding[kMaxRank];
  uint32_t hOutSpatial[kMaxRank];

  read_u32_array(inSpatialObj, hInSpatial, rank, "torchlean_cuda_maxpool_bwd: bad inSpatial");
  read_u32_array(kernelObj, hKSpatial, rank, "torchlean_cuda_maxpool_bwd: bad kernel");
  read_u32_array(strideObj, hStride, rank, "torchlean_cuda_maxpool_bwd: bad stride");
  read_u32_array(paddingObj, hPadding, rank, "torchlean_cuda_maxpool_bwd: bad padding");

  for (int ax = 0; ax < rank; ++ax) {
    if (hKSpatial[ax] == 0) {
      lean_internal_panic("torchlean_cuda_maxpool_bwd: kernel dims must be > 0");
    }
    if (hStride[ax] == 0) {
      lean_internal_panic("torchlean_cuda_maxpool_bwd: stride dims must be > 0");
    }
    hOutSpatial[ax] = outDim(hInSpatial[ax], hKSpatial[ax], hStride[ax], hPadding[ax]);
  }

  const size_t inSpatialSize = prod_u32(hInSpatial, rank);
  const size_t kSpatialSize = prod_u32(hKSpatial, rank);
  const size_t outSpatialSize = prod_u32(hOutSpatial, rank);

  const size_t inElems = (size_t)inC * inSpatialSize;
  const size_t outElems = (size_t)inC * outSpatialSize;

  checkBufSize(input, inElems, "torchlean_cuda_maxpool_bwd: input.size mismatch");
  checkBufSize(gradOutput, outElems, "torchlean_cuda_maxpool_bwd: gradOutput.size mismatch");

  torchlean_cuda_buffer* dInput = torchlean_cuda_buffer_alloc(inElems);
  const bool det = torchlean_cuda_deterministic_reductions_enabled();
  if (!det && dInput->size > 0) {
    // Atomic path accumulates into dInput; must clear first.
    checkCuda(cudaMemset(dInput->data, 0, dInput->size * sizeof(float)), "cudaMemset dInput failed");
  }
  if (outElems == 0) {
    return torchlean_cuda_buffer_box(dInput);
  }

  uint32_t* dInSpatial = nullptr;
  uint32_t* dOutSpatial = nullptr;
  uint32_t* dKSpatial = nullptr;
  uint32_t* dStride = nullptr;
  uint32_t* dPadding = nullptr;
  checkCuda(cudaMalloc((void**)&dInSpatial, (size_t)rank * sizeof(uint32_t)), "cudaMalloc inSpatial failed");
  checkCuda(cudaMalloc((void**)&dOutSpatial, (size_t)rank * sizeof(uint32_t)), "cudaMalloc outSpatial failed");
  checkCuda(cudaMalloc((void**)&dKSpatial, (size_t)rank * sizeof(uint32_t)), "cudaMalloc kSpatial failed");
  checkCuda(cudaMalloc((void**)&dStride, (size_t)rank * sizeof(uint32_t)), "cudaMalloc stride failed");
  checkCuda(cudaMalloc((void**)&dPadding, (size_t)rank * sizeof(uint32_t)), "cudaMalloc padding failed");

  checkCuda(cudaMemcpy(dInSpatial, hInSpatial, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy inSpatial failed");
  checkCuda(cudaMemcpy(dOutSpatial, hOutSpatial, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy outSpatial failed");
  checkCuda(cudaMemcpy(dKSpatial, hKSpatial, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy kSpatial failed");
  checkCuda(cudaMemcpy(dStride, hStride, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy stride failed");
  checkCuda(cudaMemcpy(dPadding, hPadding, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy padding failed");

  if (det) {
    const int grid = grid_for(inElems);
    maxpoolnd_bwd_det_kernel<<<grid, kBlock>>>(input->data, gradOutput->data, dInput->data,
                                              inC,
                                              dInSpatial, dOutSpatial, dKSpatial, dStride, dPadding,
                                              rank, inSpatialSize, outSpatialSize, kSpatialSize);
    checkCuda(cudaGetLastError(), "maxpoolnd_bwd (deterministic) kernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "maxpoolnd_bwd (deterministic) sync failed");
  } else {
    const int grid = grid_for(outElems);
    maxpoolnd_bwd_kernel<<<grid, kBlock>>>(input->data, gradOutput->data, dInput->data,
                                          inC,
                                          dInSpatial, dOutSpatial, dKSpatial, dStride, dPadding,
                                          rank, outSpatialSize, kSpatialSize);
    checkCuda(cudaGetLastError(), "maxpoolnd_bwd kernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "maxpoolnd_bwd sync failed");
  }

  torchlean_cuda_free_checked((void**)&dInSpatial, "cudaFree maxpoolnd_bwd inSpatial failed");
  torchlean_cuda_free_checked((void**)&dOutSpatial, "cudaFree maxpoolnd_bwd outSpatial failed");
  torchlean_cuda_free_checked((void**)&dKSpatial, "cudaFree maxpoolnd_bwd kSpatial failed");
  torchlean_cuda_free_checked((void**)&dStride, "cudaFree maxpoolnd_bwd stride failed");
  torchlean_cuda_free_checked((void**)&dPadding, "cudaFree maxpoolnd_bwd padding failed");

  return torchlean_cuda_buffer_box(dInput);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_avgpool_fwd(
    b_lean_obj_arg inputObj,
    b_lean_obj_arg inSpatialObj, b_lean_obj_arg kernelObj,
    b_lean_obj_arg strideObj, b_lean_obj_arg paddingObj,
    uint32_t inC) {
  torchlean_cuda_buffer* input = torchlean_cuda_buffer_unbox(inputObj);

  const int rank = read_rank_checked(inSpatialObj, "torchlean_cuda_avgpool_fwd: bad inSpatial");
  if (read_rank_checked(kernelObj, "torchlean_cuda_avgpool_fwd: bad kernel") != rank ||
      read_rank_checked(strideObj, "torchlean_cuda_avgpool_fwd: bad stride") != rank ||
      read_rank_checked(paddingObj, "torchlean_cuda_avgpool_fwd: bad padding") != rank) {
    lean_internal_panic("torchlean_cuda_avgpool_fwd: array rank mismatch");
  }
  if (rank <= 0) {
    lean_internal_panic("torchlean_cuda_avgpool_fwd: rank must be > 0");
  }

  uint32_t hInSpatial[kMaxRank];
  uint32_t hKSpatial[kMaxRank];
  uint32_t hStride[kMaxRank];
  uint32_t hPadding[kMaxRank];
  uint32_t hOutSpatial[kMaxRank];

  read_u32_array(inSpatialObj, hInSpatial, rank, "torchlean_cuda_avgpool_fwd: bad inSpatial");
  read_u32_array(kernelObj, hKSpatial, rank, "torchlean_cuda_avgpool_fwd: bad kernel");
  read_u32_array(strideObj, hStride, rank, "torchlean_cuda_avgpool_fwd: bad stride");
  read_u32_array(paddingObj, hPadding, rank, "torchlean_cuda_avgpool_fwd: bad padding");

  for (int ax = 0; ax < rank; ++ax) {
    if (hKSpatial[ax] == 0) {
      lean_internal_panic("torchlean_cuda_avgpool_fwd: kernel dims must be > 0");
    }
    if (hStride[ax] == 0) {
      lean_internal_panic("torchlean_cuda_avgpool_fwd: stride dims must be > 0");
    }
    hOutSpatial[ax] = outDim(hInSpatial[ax], hKSpatial[ax], hStride[ax], hPadding[ax]);
  }

  const size_t inSpatialSize = prod_u32(hInSpatial, rank);
  const size_t kSpatialSize = prod_u32(hKSpatial, rank);
  const size_t outSpatialSize = prod_u32(hOutSpatial, rank);

  const size_t inElems = (size_t)inC * inSpatialSize;
  const size_t outElems = (size_t)inC * outSpatialSize;

  checkBufSize(input, inElems, "torchlean_cuda_avgpool_fwd: input.size mismatch");

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outElems);
  if (outElems == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  uint32_t* dInSpatial = nullptr;
  uint32_t* dOutSpatial = nullptr;
  uint32_t* dKSpatial = nullptr;
  uint32_t* dStride = nullptr;
  uint32_t* dPadding = nullptr;
  checkCuda(cudaMalloc((void**)&dInSpatial, (size_t)rank * sizeof(uint32_t)), "cudaMalloc inSpatial failed");
  checkCuda(cudaMalloc((void**)&dOutSpatial, (size_t)rank * sizeof(uint32_t)), "cudaMalloc outSpatial failed");
  checkCuda(cudaMalloc((void**)&dKSpatial, (size_t)rank * sizeof(uint32_t)), "cudaMalloc kSpatial failed");
  checkCuda(cudaMalloc((void**)&dStride, (size_t)rank * sizeof(uint32_t)), "cudaMalloc stride failed");
  checkCuda(cudaMalloc((void**)&dPadding, (size_t)rank * sizeof(uint32_t)), "cudaMalloc padding failed");

  checkCuda(cudaMemcpy(dInSpatial, hInSpatial, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy inSpatial failed");
  checkCuda(cudaMemcpy(dOutSpatial, hOutSpatial, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy outSpatial failed");
  checkCuda(cudaMemcpy(dKSpatial, hKSpatial, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy kSpatial failed");
  checkCuda(cudaMemcpy(dStride, hStride, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy stride failed");
  checkCuda(cudaMemcpy(dPadding, hPadding, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy padding failed");

  const int grid = grid_for(outElems);
  avgpoolnd_fwd_kernel<<<grid, kBlock>>>(input->data, out->data,
                                       inC,
                                       dInSpatial, dOutSpatial, dKSpatial, dStride, dPadding,
                                       rank, outSpatialSize, kSpatialSize);
  checkCuda(cudaGetLastError(), "avgpoolnd_fwd kernel launch failed");
  checkCuda(cudaDeviceSynchronize(), "avgpoolnd_fwd sync failed");

  torchlean_cuda_free_checked((void**)&dInSpatial, "cudaFree avgpoolnd inSpatial failed");
  torchlean_cuda_free_checked((void**)&dOutSpatial, "cudaFree avgpoolnd outSpatial failed");
  torchlean_cuda_free_checked((void**)&dKSpatial, "cudaFree avgpoolnd kSpatial failed");
  torchlean_cuda_free_checked((void**)&dStride, "cudaFree avgpoolnd stride failed");
  torchlean_cuda_free_checked((void**)&dPadding, "cudaFree avgpoolnd padding failed");

  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_avgpool_bwd(
    b_lean_obj_arg gradObj,
    b_lean_obj_arg inSpatialObj, b_lean_obj_arg kernelObj,
    b_lean_obj_arg strideObj, b_lean_obj_arg paddingObj,
    uint32_t inC) {
  torchlean_cuda_buffer* gradOutput = torchlean_cuda_buffer_unbox(gradObj);

  const int rank = read_rank_checked(inSpatialObj, "torchlean_cuda_avgpool_bwd: bad inSpatial");
  if (read_rank_checked(kernelObj, "torchlean_cuda_avgpool_bwd: bad kernel") != rank ||
      read_rank_checked(strideObj, "torchlean_cuda_avgpool_bwd: bad stride") != rank ||
      read_rank_checked(paddingObj, "torchlean_cuda_avgpool_bwd: bad padding") != rank) {
    lean_internal_panic("torchlean_cuda_avgpool_bwd: array rank mismatch");
  }
  if (rank <= 0) {
    lean_internal_panic("torchlean_cuda_avgpool_bwd: rank must be > 0");
  }

  uint32_t hInSpatial[kMaxRank];
  uint32_t hKSpatial[kMaxRank];
  uint32_t hStride[kMaxRank];
  uint32_t hPadding[kMaxRank];
  uint32_t hOutSpatial[kMaxRank];

  read_u32_array(inSpatialObj, hInSpatial, rank, "torchlean_cuda_avgpool_bwd: bad inSpatial");
  read_u32_array(kernelObj, hKSpatial, rank, "torchlean_cuda_avgpool_bwd: bad kernel");
  read_u32_array(strideObj, hStride, rank, "torchlean_cuda_avgpool_bwd: bad stride");
  read_u32_array(paddingObj, hPadding, rank, "torchlean_cuda_avgpool_bwd: bad padding");

  for (int ax = 0; ax < rank; ++ax) {
    if (hKSpatial[ax] == 0) {
      lean_internal_panic("torchlean_cuda_avgpool_bwd: kernel dims must be > 0");
    }
    if (hStride[ax] == 0) {
      lean_internal_panic("torchlean_cuda_avgpool_bwd: stride dims must be > 0");
    }
    hOutSpatial[ax] = outDim(hInSpatial[ax], hKSpatial[ax], hStride[ax], hPadding[ax]);
  }

  const size_t inSpatialSize = prod_u32(hInSpatial, rank);
  const size_t kSpatialSize = prod_u32(hKSpatial, rank);
  const size_t outSpatialSize = prod_u32(hOutSpatial, rank);

  const size_t inElems = (size_t)inC * inSpatialSize;
  const size_t outElems = (size_t)inC * outSpatialSize;

  checkBufSize(gradOutput, outElems, "torchlean_cuda_avgpool_bwd: gradOutput.size mismatch");

  torchlean_cuda_buffer* dInput = torchlean_cuda_buffer_alloc(inElems);
  const bool det = torchlean_cuda_deterministic_reductions_enabled();
  if (!det && dInput->size > 0) {
    // Atomic path accumulates into dInput; must clear first.
    checkCuda(cudaMemset(dInput->data, 0, dInput->size * sizeof(float)), "cudaMemset dInput failed");
  }
  if (outElems == 0) {
    return torchlean_cuda_buffer_box(dInput);
  }

  uint32_t* dInSpatial = nullptr;
  uint32_t* dOutSpatial = nullptr;
  uint32_t* dKSpatial = nullptr;
  uint32_t* dStride = nullptr;
  uint32_t* dPadding = nullptr;
  checkCuda(cudaMalloc((void**)&dInSpatial, (size_t)rank * sizeof(uint32_t)), "cudaMalloc inSpatial failed");
  checkCuda(cudaMalloc((void**)&dOutSpatial, (size_t)rank * sizeof(uint32_t)), "cudaMalloc outSpatial failed");
  checkCuda(cudaMalloc((void**)&dKSpatial, (size_t)rank * sizeof(uint32_t)), "cudaMalloc kSpatial failed");
  checkCuda(cudaMalloc((void**)&dStride, (size_t)rank * sizeof(uint32_t)), "cudaMalloc stride failed");
  checkCuda(cudaMalloc((void**)&dPadding, (size_t)rank * sizeof(uint32_t)), "cudaMalloc padding failed");

  checkCuda(cudaMemcpy(dInSpatial, hInSpatial, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy inSpatial failed");
  checkCuda(cudaMemcpy(dOutSpatial, hOutSpatial, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy outSpatial failed");
  checkCuda(cudaMemcpy(dKSpatial, hKSpatial, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy kSpatial failed");
  checkCuda(cudaMemcpy(dStride, hStride, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy stride failed");
  checkCuda(cudaMemcpy(dPadding, hPadding, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy padding failed");

  if (det) {
    const int grid = grid_for(inElems);
    avgpoolnd_bwd_det_kernel<<<grid, kBlock>>>(gradOutput->data, dInput->data,
                                              inC,
                                              dInSpatial, dOutSpatial, dKSpatial, dStride, dPadding,
                                              rank, inSpatialSize, outSpatialSize, kSpatialSize);
    checkCuda(cudaGetLastError(), "avgpoolnd_bwd (deterministic) kernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "avgpoolnd_bwd (deterministic) sync failed");
  } else {
    const int grid = grid_for(outElems);
    avgpoolnd_bwd_kernel<<<grid, kBlock>>>(gradOutput->data, dInput->data,
                                          inC,
                                          dInSpatial, dOutSpatial, dKSpatial, dStride, dPadding,
                                          rank, outSpatialSize, kSpatialSize);
    checkCuda(cudaGetLastError(), "avgpoolnd_bwd kernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "avgpoolnd_bwd sync failed");
  }

  torchlean_cuda_free_checked((void**)&dInSpatial, "cudaFree avgpoolnd_bwd inSpatial failed");
  torchlean_cuda_free_checked((void**)&dOutSpatial, "cudaFree avgpoolnd_bwd outSpatial failed");
  torchlean_cuda_free_checked((void**)&dKSpatial, "cudaFree avgpoolnd_bwd kSpatial failed");
  torchlean_cuda_free_checked((void**)&dStride, "cudaFree avgpoolnd_bwd stride failed");
  torchlean_cuda_free_checked((void**)&dPadding, "cudaFree avgpoolnd_bwd padding failed");

  return torchlean_cuda_buffer_box(dInput);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_smooth_maxpool_fwd(
    b_lean_obj_arg inputObj, double beta,
    b_lean_obj_arg inSpatialObj, b_lean_obj_arg kernelObj,
    b_lean_obj_arg strideObj, b_lean_obj_arg paddingObj,
    uint32_t inC) {
  torchlean_cuda_buffer* input = torchlean_cuda_buffer_unbox(inputObj);

  const int rank = read_rank_checked(inSpatialObj, "torchlean_cuda_smooth_maxpool_fwd: bad inSpatial");
  if (read_rank_checked(kernelObj, "torchlean_cuda_smooth_maxpool_fwd: bad kernel") != rank ||
      read_rank_checked(strideObj, "torchlean_cuda_smooth_maxpool_fwd: bad stride") != rank ||
      read_rank_checked(paddingObj, "torchlean_cuda_smooth_maxpool_fwd: bad padding") != rank) {
    lean_internal_panic("torchlean_cuda_smooth_maxpool_fwd: array rank mismatch");
  }
  if (rank <= 0) {
    lean_internal_panic("torchlean_cuda_smooth_maxpool_fwd: rank must be > 0");
  }

  uint32_t hInSpatial[kMaxRank];
  uint32_t hKSpatial[kMaxRank];
  uint32_t hStride[kMaxRank];
  uint32_t hPadding[kMaxRank];
  uint32_t hOutSpatial[kMaxRank];

  read_u32_array(inSpatialObj, hInSpatial, rank, "torchlean_cuda_smooth_maxpool_fwd: bad inSpatial");
  read_u32_array(kernelObj, hKSpatial, rank, "torchlean_cuda_smooth_maxpool_fwd: bad kernel");
  read_u32_array(strideObj, hStride, rank, "torchlean_cuda_smooth_maxpool_fwd: bad stride");
  read_u32_array(paddingObj, hPadding, rank, "torchlean_cuda_smooth_maxpool_fwd: bad padding");

  for (int ax = 0; ax < rank; ++ax) {
    if (hKSpatial[ax] == 0) {
      lean_internal_panic("torchlean_cuda_smooth_maxpool_fwd: kernel dims must be > 0");
    }
    if (hStride[ax] == 0) {
      lean_internal_panic("torchlean_cuda_smooth_maxpool_fwd: stride dims must be > 0");
    }
    hOutSpatial[ax] = outDim(hInSpatial[ax], hKSpatial[ax], hStride[ax], hPadding[ax]);
  }

  const size_t inSpatialSize = prod_u32(hInSpatial, rank);
  const size_t kSpatialSize = prod_u32(hKSpatial, rank);
  const size_t outSpatialSize = prod_u32(hOutSpatial, rank);

  const size_t inElems = (size_t)inC * inSpatialSize;
  const size_t outElems = (size_t)inC * outSpatialSize;

  checkBufSize(input, inElems, "torchlean_cuda_smooth_maxpool_fwd: input.size mismatch");

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outElems);
  if (outElems == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  uint32_t* dInSpatial = nullptr;
  uint32_t* dOutSpatial = nullptr;
  uint32_t* dKSpatial = nullptr;
  uint32_t* dStride = nullptr;
  uint32_t* dPadding = nullptr;
  checkCuda(cudaMalloc((void**)&dInSpatial, (size_t)rank * sizeof(uint32_t)), "cudaMalloc inSpatial failed");
  checkCuda(cudaMalloc((void**)&dOutSpatial, (size_t)rank * sizeof(uint32_t)), "cudaMalloc outSpatial failed");
  checkCuda(cudaMalloc((void**)&dKSpatial, (size_t)rank * sizeof(uint32_t)), "cudaMalloc kSpatial failed");
  checkCuda(cudaMalloc((void**)&dStride, (size_t)rank * sizeof(uint32_t)), "cudaMalloc stride failed");
  checkCuda(cudaMalloc((void**)&dPadding, (size_t)rank * sizeof(uint32_t)), "cudaMalloc padding failed");

  checkCuda(cudaMemcpy(dInSpatial, hInSpatial, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy inSpatial failed");
  checkCuda(cudaMemcpy(dOutSpatial, hOutSpatial, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy outSpatial failed");
  checkCuda(cudaMemcpy(dKSpatial, hKSpatial, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy kSpatial failed");
  checkCuda(cudaMemcpy(dStride, hStride, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy stride failed");
  checkCuda(cudaMemcpy(dPadding, hPadding, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy padding failed");

  const int grid = grid_for(outElems);
  const float betaF =
      checked_smoothmax_beta(beta, "torchlean_cuda_smooth_maxpool_fwd: beta must be finite and nonzero");
  smoothmaxpoolnd_fwd_kernel<<<grid, kBlock>>>(input->data, out->data,
                                              inC,
                                              dInSpatial, dOutSpatial, dKSpatial, dStride, dPadding,
                                              rank, outSpatialSize, kSpatialSize,
                                              betaF);
  checkCuda(cudaGetLastError(), "smoothmaxpoolnd_fwd kernel launch failed");
  checkCuda(cudaDeviceSynchronize(), "smoothmaxpoolnd_fwd sync failed");

  torchlean_cuda_free_checked((void**)&dInSpatial, "cudaFree smoothmaxpoolnd inSpatial failed");
  torchlean_cuda_free_checked((void**)&dOutSpatial, "cudaFree smoothmaxpoolnd outSpatial failed");
  torchlean_cuda_free_checked((void**)&dKSpatial, "cudaFree smoothmaxpoolnd kSpatial failed");
  torchlean_cuda_free_checked((void**)&dStride, "cudaFree smoothmaxpoolnd stride failed");
  torchlean_cuda_free_checked((void**)&dPadding, "cudaFree smoothmaxpoolnd padding failed");

  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_smooth_maxpool_bwd(
    b_lean_obj_arg inputObj, b_lean_obj_arg gradObj, double beta,
    b_lean_obj_arg inSpatialObj, b_lean_obj_arg kernelObj,
    b_lean_obj_arg strideObj, b_lean_obj_arg paddingObj,
    uint32_t inC) {
  torchlean_cuda_buffer* input = torchlean_cuda_buffer_unbox(inputObj);
  torchlean_cuda_buffer* gradOutput = torchlean_cuda_buffer_unbox(gradObj);

  const int rank = read_rank_checked(inSpatialObj, "torchlean_cuda_smooth_maxpool_bwd: bad inSpatial");
  if (read_rank_checked(kernelObj, "torchlean_cuda_smooth_maxpool_bwd: bad kernel") != rank ||
      read_rank_checked(strideObj, "torchlean_cuda_smooth_maxpool_bwd: bad stride") != rank ||
      read_rank_checked(paddingObj, "torchlean_cuda_smooth_maxpool_bwd: bad padding") != rank) {
    lean_internal_panic("torchlean_cuda_smooth_maxpool_bwd: array rank mismatch");
  }
  if (rank <= 0) {
    lean_internal_panic("torchlean_cuda_smooth_maxpool_bwd: rank must be > 0");
  }

  uint32_t hInSpatial[kMaxRank];
  uint32_t hKSpatial[kMaxRank];
  uint32_t hStride[kMaxRank];
  uint32_t hPadding[kMaxRank];
  uint32_t hOutSpatial[kMaxRank];

  read_u32_array(inSpatialObj, hInSpatial, rank, "torchlean_cuda_smooth_maxpool_bwd: bad inSpatial");
  read_u32_array(kernelObj, hKSpatial, rank, "torchlean_cuda_smooth_maxpool_bwd: bad kernel");
  read_u32_array(strideObj, hStride, rank, "torchlean_cuda_smooth_maxpool_bwd: bad stride");
  read_u32_array(paddingObj, hPadding, rank, "torchlean_cuda_smooth_maxpool_bwd: bad padding");

  for (int ax = 0; ax < rank; ++ax) {
    if (hKSpatial[ax] == 0) {
      lean_internal_panic("torchlean_cuda_smooth_maxpool_bwd: kernel dims must be > 0");
    }
    if (hStride[ax] == 0) {
      lean_internal_panic("torchlean_cuda_smooth_maxpool_bwd: stride dims must be > 0");
    }
    hOutSpatial[ax] = outDim(hInSpatial[ax], hKSpatial[ax], hStride[ax], hPadding[ax]);
  }

  const size_t inSpatialSize = prod_u32(hInSpatial, rank);
  const size_t kSpatialSize = prod_u32(hKSpatial, rank);
  const size_t outSpatialSize = prod_u32(hOutSpatial, rank);

  const size_t inElems = (size_t)inC * inSpatialSize;
  const size_t outElems = (size_t)inC * outSpatialSize;

  checkBufSize(input, inElems, "torchlean_cuda_smooth_maxpool_bwd: input.size mismatch");
  checkBufSize(gradOutput, outElems, "torchlean_cuda_smooth_maxpool_bwd: gradOutput.size mismatch");

  torchlean_cuda_buffer* dInput = torchlean_cuda_buffer_alloc(inElems);
  const float betaF =
      checked_smoothmax_beta(beta, "torchlean_cuda_smooth_maxpool_bwd: beta must be finite and nonzero");
  const bool det = torchlean_cuda_deterministic_reductions_enabled();
  if (!det && dInput->size > 0) {
    // Atomic path accumulates into dInput; must clear first.
    checkCuda(cudaMemset(dInput->data, 0, dInput->size * sizeof(float)), "cudaMemset dInput failed");
  }
  if (outElems == 0) {
    return torchlean_cuda_buffer_box(dInput);
  }

  uint32_t* dInSpatial = nullptr;
  uint32_t* dOutSpatial = nullptr;
  uint32_t* dKSpatial = nullptr;
  uint32_t* dStride = nullptr;
  uint32_t* dPadding = nullptr;
  checkCuda(cudaMalloc((void**)&dInSpatial, (size_t)rank * sizeof(uint32_t)), "cudaMalloc inSpatial failed");
  checkCuda(cudaMalloc((void**)&dOutSpatial, (size_t)rank * sizeof(uint32_t)), "cudaMalloc outSpatial failed");
  checkCuda(cudaMalloc((void**)&dKSpatial, (size_t)rank * sizeof(uint32_t)), "cudaMalloc kSpatial failed");
  checkCuda(cudaMalloc((void**)&dStride, (size_t)rank * sizeof(uint32_t)), "cudaMalloc stride failed");
  checkCuda(cudaMalloc((void**)&dPadding, (size_t)rank * sizeof(uint32_t)), "cudaMalloc padding failed");

  checkCuda(cudaMemcpy(dInSpatial, hInSpatial, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy inSpatial failed");
  checkCuda(cudaMemcpy(dOutSpatial, hOutSpatial, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy outSpatial failed");
  checkCuda(cudaMemcpy(dKSpatial, hKSpatial, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy kSpatial failed");
  checkCuda(cudaMemcpy(dStride, hStride, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy stride failed");
  checkCuda(cudaMemcpy(dPadding, hPadding, (size_t)rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy padding failed");

  if (det) {
    const int grid = grid_for(inElems);
    smoothmaxpoolnd_bwd_det_kernel<<<grid, kBlock>>>(input->data, gradOutput->data, dInput->data,
                                                    inC,
                                                    dInSpatial, dOutSpatial, dKSpatial, dStride, dPadding,
                                                    rank, inSpatialSize, outSpatialSize, kSpatialSize,
                                                    betaF);
    checkCuda(cudaGetLastError(), "smoothmaxpoolnd_bwd (deterministic) kernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "smoothmaxpoolnd_bwd (deterministic) sync failed");
  } else {
    const int grid = grid_for(outElems);
    smoothmaxpoolnd_bwd_kernel<<<grid, kBlock>>>(input->data, gradOutput->data, dInput->data,
                                                inC,
                                                dInSpatial, dOutSpatial, dKSpatial, dStride, dPadding,
                                                rank, outSpatialSize, kSpatialSize,
                                                betaF);
    checkCuda(cudaGetLastError(), "smoothmaxpoolnd_bwd kernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "smoothmaxpoolnd_bwd sync failed");
  }

  torchlean_cuda_free_checked((void**)&dInSpatial, "cudaFree smoothmaxpoolnd_bwd inSpatial failed");
  torchlean_cuda_free_checked((void**)&dOutSpatial, "cudaFree smoothmaxpoolnd_bwd outSpatial failed");
  torchlean_cuda_free_checked((void**)&dKSpatial, "cudaFree smoothmaxpoolnd_bwd kSpatial failed");
  torchlean_cuda_free_checked((void**)&dStride, "cudaFree smoothmaxpoolnd_bwd stride failed");
  torchlean_cuda_free_checked((void**)&dPadding, "cudaFree smoothmaxpoolnd_bwd padding failed");

  return torchlean_cuda_buffer_box(dInput);
}
