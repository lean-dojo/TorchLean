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

// Rank-polymorphic convolution and pooling kernels for `Cuda.Buffer`. Wrappers validate shape data
// before launch; the runtime flag chooses deterministic backward paths.

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

extern "C" void torchlean_cuda_conv_pool_flush_scratch_cache(void) {
  torchlean_cuda_scratch_flush();
}

struct DeviceSpatialScratch {
  uint32_t* inSpatial;
  uint32_t* outSpatial;
  uint32_t* kSpatial;
  uint32_t* stride;
  uint32_t* padding;
};

static inline DeviceSpatialScratch alloc_spatial_scratch(int rank) {
  const size_t count = (size_t)rank;
  return {
    torchlean_cuda_scratch_alloc<uint32_t>(count, "cudaMalloc inSpatial failed"),
    torchlean_cuda_scratch_alloc<uint32_t>(count, "cudaMalloc outSpatial failed"),
    torchlean_cuda_scratch_alloc<uint32_t>(count, "cudaMalloc kSpatial failed"),
    torchlean_cuda_scratch_alloc<uint32_t>(count, "cudaMalloc stride failed"),
    torchlean_cuda_scratch_alloc<uint32_t>(count, "cudaMalloc padding failed")
  };
}

static inline void free_spatial_scratch(int rank, DeviceSpatialScratch* scratch) {
  const size_t count = (size_t)rank;
  torchlean_cuda_scratch_free(&scratch->inSpatial, count, "cudaFree inSpatial failed");
  torchlean_cuda_scratch_free(&scratch->outSpatial, count, "cudaFree outSpatial failed");
  torchlean_cuda_scratch_free(&scratch->kSpatial, count, "cudaFree kSpatial failed");
  torchlean_cuda_scratch_free(&scratch->stride, count, "cudaFree stride failed");
  torchlean_cuda_scratch_free(&scratch->padding, count, "cudaFree padding failed");
}

static inline void copy_spatial_scratch(
    int rank,
    const DeviceSpatialScratch& scratch,
    const uint32_t* hInSpatial,
    const uint32_t* hOutSpatial,
    const uint32_t* hKSpatial,
    const uint32_t* hStride,
    const uint32_t* hPadding) {
  const size_t bytes = (size_t)rank * sizeof(uint32_t);
  checkCuda(cudaMemcpy(scratch.inSpatial, hInSpatial, bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy inSpatial failed");
  checkCuda(cudaMemcpy(scratch.outSpatial, hOutSpatial, bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy outSpatial failed");
  checkCuda(cudaMemcpy(scratch.kSpatial, hKSpatial, bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy kSpatial failed");
  checkCuda(cudaMemcpy(scratch.stride, hStride, bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy stride failed");
  checkCuda(cudaMemcpy(scratch.padding, hPadding, bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy padding failed");
}

__device__ inline void decode_spatial_index(
    size_t index,
    const uint32_t* dims,
    int rank,
    uint32_t* coord) {
  for (int ax = rank - 1; ax >= 0; --ax) {
    const uint32_t d = dims[ax];
    coord[ax] = (uint32_t)(index % (size_t)d);
    index /= (size_t)d;
  }
}

__device__ inline bool input_index_from_window(
    uint32_t c,
    const uint32_t* outCoord,
    const uint32_t* kCoord,
    const uint32_t* inSpatial,
    const uint32_t* stride,
    const uint32_t* padding,
    int rank,
    size_t* inIdxOut) {
  size_t inIdx = (size_t)c;
  for (int ax = 0; ax < rank; ++ax) {
    uint32_t pos;
    const uint32_t dim = inSpatial[ax];
    if (!window_input_coord(outCoord[ax], kCoord[ax], stride[ax], padding[ax],
                            dim, &pos)) {
      return false;
    }
    inIdx = inIdx * (size_t)dim + (size_t)pos;
  }
  *inIdxOut = inIdx;
  return true;
}

__global__ void add_bias_cspatial_kernel(
    float* output,
    const float* bias,
    uint32_t channels,
    size_t spatialSize) {
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t total = (size_t)channels * spatialSize;
  if (idx >= total) return;
  const uint32_t channel = (uint32_t)(idx / spatialSize);
  output[idx] += bias[channel];
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
  decode_spatial_index(spatialIdx, outSpatial, rank, outCoord);

  float acc = bias ? bias[oc] : 0.0f;

  uint32_t kCoord[kMaxRank];
  for (uint32_t ic = 0; ic < inC; ++ic) {
    for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
      decode_spatial_index(kIdx, kSpatial, rank, kCoord);

      bool inBounds = true;
      size_t inIdx = (size_t)ic;
      for (int ax = 0; ax < rank; ++ax) {
        uint32_t pos;
        const uint32_t dim = inSpatial[ax];
        if (!window_input_coord(outCoord[ax], kCoord[ax], stride[ax], padding[ax],
                                dim, &pos)) {
          inBounds = false;
          break;
        }
        inIdx = inIdx * (size_t)dim + (size_t)pos;
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
  decode_spatial_index(kIdx, kSpatial, rank, kCoord);

  float acc = 0.0f;

  uint32_t outCoord[kMaxRank];
  for (size_t outIdx = 0; outIdx < outSpatialSize; ++outIdx) {
    decode_spatial_index(outIdx, outSpatial, rank, outCoord);

    bool inBounds = true;
    size_t inIdx = (size_t)ic;
    for (int ax = 0; ax < rank; ++ax) {
      uint32_t pos;
      const uint32_t dim = inSpatial[ax];
      if (!window_input_coord(outCoord[ax], kCoord[ax], stride[ax], padding[ax],
                              dim, &pos)) {
        inBounds = false;
        break;
      }
      inIdx = inIdx * (size_t)dim + (size_t)pos;
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
  decode_spatial_index(spatialIdx, inSpatial, rank, inCoord);

  uint32_t kCoord[kMaxRank];
  uint32_t outCoord[kMaxRank];

  float acc = 0.0f;
  for (uint32_t oc = 0; oc < outC; ++oc) {
    for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
      decode_spatial_index(kIdx, kSpatial, rank, kCoord);

      bool ok = true;
      for (int ax = 0; ax < rank; ++ax) {
        if (!window_output_coord(inCoord[ax], kCoord[ax], stride[ax], padding[ax],
                                 outSpatial[ax], &outCoord[ax])) {
          ok = false;
          break;
        }
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
  decode_spatial_index(spatialIdx, outSpatial, rank, outCoord);

  float best = 0.0f;
  bool bestValid = false;
  uint32_t kCoord[kMaxRank];
  for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
    decode_spatial_index(kIdx, kSpatial, rank, kCoord);
    size_t inIdx = 0;
    const bool inBounds =
        input_index_from_window(c, outCoord, kCoord, inSpatial, stride, padding, rank, &inIdx);

    if (inBounds) {
      const float v = input[inIdx];
      if (!bestValid || v > best) {
        best = v;
        bestValid = true;
      }
    }
  }

  output[idx] = bestValid ? best : 0.0f;
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
  decode_spatial_index(spatialIdx, outSpatial, rank, outCoord);

  float best = 0.0f;
  bool bestValid = false;
  uint32_t bestInCoord[kMaxRank];
  uint32_t candInCoord[kMaxRank];
  uint32_t kCoord[kMaxRank];

  for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
    decode_spatial_index(kIdx, kSpatial, rank, kCoord);

    bool inBounds = true;
    for (int ax = 0; ax < rank; ++ax) {
      uint32_t pos;
      const uint32_t dim = inSpatial[ax];
      if (!window_input_coord(outCoord[ax], kCoord[ax], stride[ax], padding[ax],
                              dim, &pos)) {
        inBounds = false;
        break;
      }
      candInCoord[ax] = pos;
    }

    if (inBounds) {
      size_t inIdx = (size_t)c;
      for (int ax = 0; ax < rank; ++ax) {
        inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)candInCoord[ax];
      }
      const float v = input[inIdx];
      if (!bestValid || v > best) {
        best = v;
        bestValid = true;
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
  decode_spatial_index(spatialIdx, inSpatial, rank, inCoord);

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
      float best = 0.0f;
      bool bestValid = false;
      for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
        decode_spatial_index(kIdx, kSpatial, rank, kCoord);

        bool inBounds = true;
        for (int ax = 0; ax < rank; ++ax) {
          uint32_t pos;
          const uint32_t dim = inSpatial[ax];
          if (!window_input_coord(outCoord[ax], kCoord[ax], stride[ax], padding[ax],
                                  dim, &pos)) {
            inBounds = false;
            break;
          }
          candInCoord[ax] = pos;
        }

        if (inBounds) {
          size_t inIdx = (size_t)c;
          for (int ax = 0; ax < rank; ++ax) {
            inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)candInCoord[ax];
          }
          const float v = input[inIdx];
          if (!bestValid || v > best) {
            best = v;
            bestValid = true;
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
  decode_spatial_index(spatialIdx, outSpatial, rank, outCoord);

  float acc = 0.0f;
  uint32_t kCoord[kMaxRank];
  for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
    decode_spatial_index(kIdx, kSpatial, rank, kCoord);
    size_t inIdx = 0;
    const bool inBounds =
        input_index_from_window(c, outCoord, kCoord, inSpatial, stride, padding, rank, &inIdx);

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
  decode_spatial_index(spatialIdx, outSpatial, rank, outCoord);

  const float g = gradOutput[idx] / (float)(kSpatialSize);

  uint32_t kCoord[kMaxRank];
  for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
    decode_spatial_index(kIdx, kSpatial, rank, kCoord);
    size_t inIdx = 0;
    const bool inBounds =
        input_index_from_window(c, outCoord, kCoord, inSpatial, stride, padding, rank, &inIdx);

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
  decode_spatial_index(spatialIdx, inSpatial, rank, inCoord);

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
  decode_spatial_index(spatialIdx, outSpatial, rank, outCoord);

  // Choose the extremum in input space before multiplying by beta; this avoids beta*v overflow.
  float pivot = (beta > 0.0f) ? -INFINITY : INFINITY;
  uint32_t kCoord[kMaxRank];
  for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
    decode_spatial_index(kIdx, kSpatial, rank, kCoord);

    bool inBounds = true;
    size_t inIdx = (size_t)c;
    for (int ax = 0; ax < rank; ++ax) {
      uint32_t pos;
      const uint32_t dim = inSpatial[ax];
      if (!window_input_coord(outCoord[ax], kCoord[ax], stride[ax], padding[ax],
                              dim, &pos)) {
        inBounds = false;
        break;
      }
      inIdx = inIdx * (size_t)dim + (size_t)pos;
    }

    float v = 0.0f;
    if (inBounds) {
      v = input[inIdx];
    }
    pivot = (beta > 0.0f) ? fmaxf(pivot, v) : fminf(pivot, v);
  }

  float sumExp = 0.0f;
  for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
    decode_spatial_index(kIdx, kSpatial, rank, kCoord);

    bool inBounds = true;
    size_t inIdx = (size_t)c;
    for (int ax = 0; ax < rank; ++ax) {
      uint32_t pos;
      const uint32_t dim = inSpatial[ax];
      if (!window_input_coord(outCoord[ax], kCoord[ax], stride[ax], padding[ax],
                              dim, &pos)) {
        inBounds = false;
        break;
      }
      inIdx = inIdx * (size_t)dim + (size_t)pos;
    }

    float v = 0.0f;
    if (inBounds) {
      v = input[inIdx];
    }
    sumExp += expf(beta * (v - pivot));
  }

  output[idx] = pivot + logf(sumExp) / beta;
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
  decode_spatial_index(spatialIdx, outSpatial, rank, outCoord);

  // Match the N-D forward path's input-space shift when recomputing gradient weights.
  float pivot = (beta > 0.0f) ? -INFINITY : INFINITY;
  uint32_t kCoord[kMaxRank];
  for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
    decode_spatial_index(kIdx, kSpatial, rank, kCoord);

    bool inBounds = true;
    size_t inIdx = (size_t)c;
    for (int ax = 0; ax < rank; ++ax) {
      uint32_t pos;
      const uint32_t dim = inSpatial[ax];
      if (!window_input_coord(outCoord[ax], kCoord[ax], stride[ax], padding[ax],
                              dim, &pos)) {
        inBounds = false;
        break;
      }
      inIdx = inIdx * (size_t)dim + (size_t)pos;
    }

    float v = 0.0f;
    if (inBounds) {
      v = input[inIdx];
    }
    pivot = (beta > 0.0f) ? fmaxf(pivot, v) : fminf(pivot, v);
  }

  float sumExp = 0.0f;
  for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
    decode_spatial_index(kIdx, kSpatial, rank, kCoord);

    bool inBounds = true;
    size_t inIdx = (size_t)c;
    for (int ax = 0; ax < rank; ++ax) {
      uint32_t pos;
      const uint32_t dim = inSpatial[ax];
      if (!window_input_coord(outCoord[ax], kCoord[ax], stride[ax], padding[ax],
                              dim, &pos)) {
        inBounds = false;
        break;
      }
      inIdx = inIdx * (size_t)dim + (size_t)pos;
    }

    float v = 0.0f;
    if (inBounds) {
      v = input[inIdx];
    }
    sumExp += expf(beta * (v - pivot));
  }

  const float g = gradOutput[idx];

  for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
    decode_spatial_index(kIdx, kSpatial, rank, kCoord);

    bool inBounds = true;
    size_t inIdx = (size_t)c;
    for (int ax = 0; ax < rank; ++ax) {
      uint32_t pos;
      const uint32_t dim = inSpatial[ax];
      if (!window_input_coord(outCoord[ax], kCoord[ax], stride[ax], padding[ax],
                              dim, &pos)) {
        inBounds = false;
        break;
      }
      inIdx = inIdx * (size_t)dim + (size_t)pos;
    }
    if (!inBounds) continue;

    const float v = input[inIdx];
    const float e = expf(beta * (v - pivot));
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
  decode_spatial_index(spatialIdx, inSpatial, rank, inCoord);

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

      // Recompute the input-space pivot and shifted exponential sum for this output window.
      float pivot = (beta > 0.0f) ? -INFINITY : INFINITY;
      for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
        decode_spatial_index(kIdx, kSpatial, rank, kCoord);

        bool inBounds = true;
        size_t inIdx = (size_t)c;
        for (int ax = 0; ax < rank; ++ax) {
          uint32_t pos;
          const uint32_t dim = inSpatial[ax];
          if (!window_input_coord(outCoord[ax], kCoord[ax], stride[ax], padding[ax],
                                  dim, &pos)) {
            inBounds = false;
            break;
          }
          inIdx = inIdx * (size_t)dim + (size_t)pos;
        }

        float v = 0.0f;
        if (inBounds) {
          v = input[inIdx];
        }
        pivot = (beta > 0.0f) ? fmaxf(pivot, v) : fminf(pivot, v);
      }

      float sumExp = 0.0f;
      for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
        decode_spatial_index(kIdx, kSpatial, rank, kCoord);

        bool inBounds = true;
        size_t inIdx = (size_t)c;
        for (int ax = 0; ax < rank; ++ax) {
          uint32_t pos;
          const uint32_t dim = inSpatial[ax];
          if (!window_input_coord(outCoord[ax], kCoord[ax], stride[ax], padding[ax],
                                  dim, &pos)) {
            inBounds = false;
            break;
          }
          inIdx = inIdx * (size_t)dim + (size_t)pos;
        }

        float v = 0.0f;
        if (inBounds) {
          v = input[inIdx];
        }
        sumExp += expf(beta * (v - pivot));
      }

      const float w = expf(beta * (vSelf - pivot)) / sumExp;
      acc += gradOutput[goIdx] * w;
    }
  }

  dInput[idx] = acc;
}

// -------------------------
// Exported Lean FFI functions
// -------------------------

// -------------------------
// N-D conv + pooling exported functions
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

  const size_t inElems = checked_channel_spatial_size(inC, inSpatialSize, "torchlean_cuda_conv_pool: input size overflow");
  const size_t kElems = checked_conv_kernel_size(outC, inC, kSpatialSize, "torchlean_cuda_conv_pool: kernel size overflow");
  const size_t bElems = (size_t)outC;
  const size_t outElems = checked_channel_spatial_size(outC, outSpatialSize, "torchlean_cuda_conv_pool: output size overflow");

  checkBufSize(input, inElems, "torchlean_cuda_conv_fwd: input.size mismatch");
  checkBufSize(kernel, kElems, "torchlean_cuda_conv_fwd: kernel.size mismatch");
  checkBufSize(bias, bElems, "torchlean_cuda_conv_fwd: bias.size mismatch");

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outElems);
  if (outElems == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  DeviceSpatialScratch scratch = alloc_spatial_scratch(rank);
  copy_spatial_scratch(rank, scratch, hInSpatial, hOutSpatial, hKSpatial, hStride, hPadding);

  const int grid = grid_for(outElems);
  convnd_fwd_kernel<<<grid, kBlock>>>(input->data, kernel->data, bias->data, out->data,
                                     inC, outC,
                                     scratch.inSpatial, scratch.outSpatial, scratch.kSpatial,
                                     scratch.stride, scratch.padding,
                                     rank, outSpatialSize, kSpatialSize);
  checkCuda(cudaGetLastError(), "convnd_fwd kernel launch failed");
  checkCuda(cudaDeviceSynchronize(), "convnd_fwd sync failed");

  free_spatial_scratch(rank, &scratch);

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

  const size_t inElems = checked_channel_spatial_size(inC, inSpatialSize, "torchlean_cuda_conv_pool: input size overflow");
  const size_t kElems = checked_conv_kernel_size(outC, inC, kSpatialSize, "torchlean_cuda_conv_pool: kernel size overflow");
  const size_t outElems = checked_channel_spatial_size(outC, outSpatialSize, "torchlean_cuda_conv_pool: output size overflow");

  checkBufSize(input, inElems, "torchlean_cuda_conv_bwd: input.size mismatch");
  checkBufSize(kernel, kElems, "torchlean_cuda_conv_bwd: kernel.size mismatch");
  checkBufSize(gradOutput, outElems, "torchlean_cuda_conv_bwd: gradOutput.size mismatch");

  torchlean_cuda_buffer* dKernel = torchlean_cuda_buffer_alloc(kElems);
  torchlean_cuda_buffer* dBias = torchlean_cuda_buffer_alloc((size_t)outC);
  torchlean_cuda_buffer* dInput = torchlean_cuda_buffer_alloc(inElems);

  if (kElems == 0 && (size_t)outC == 0 && inElems == 0) {
    // Return the empty buffers without allocating shape arrays on the device.
    return torchlean_cuda_box_three_buffers(dKernel, dBias, dInput);
  }

  DeviceSpatialScratch scratch = alloc_spatial_scratch(rank);
  copy_spatial_scratch(rank, scratch, hInSpatial, hOutSpatial, hKSpatial, hStride, hPadding);

  if ((size_t)outC > 0) {
    const int gridBias = grid_for((size_t)outC);
    convnd_dbias_kernel<<<gridBias, kBlock>>>(gradOutput->data, dBias->data, outC, outSpatialSize);
    checkCuda(cudaGetLastError(), "convnd_dbias kernel launch failed");
  }

  if (kElems > 0) {
    const int gridK = grid_for(kElems);
    convnd_dkernel_kernel<<<gridK, kBlock>>>(input->data, gradOutput->data, dKernel->data,
                                            inC, outC,
                                            scratch.inSpatial, scratch.outSpatial, scratch.kSpatial,
                                            scratch.stride, scratch.padding,
                                            rank, outSpatialSize, kSpatialSize);
    checkCuda(cudaGetLastError(), "convnd_dkernel kernel launch failed");
  }

  if (inElems > 0) {
    const int gridIn = grid_for(inElems);
    convnd_dinput_kernel<<<gridIn, kBlock>>>(kernel->data, gradOutput->data, dInput->data,
                                            inC, outC,
                                            scratch.inSpatial, scratch.outSpatial, scratch.kSpatial,
                                            scratch.stride, scratch.padding,
                                            rank, inSpatialSize, outSpatialSize, kSpatialSize);
    checkCuda(cudaGetLastError(), "convnd_dinput kernel launch failed");
  }

  checkCuda(cudaDeviceSynchronize(), "convnd_bwd sync failed");

  free_spatial_scratch(rank, &scratch);

  return torchlean_cuda_box_three_buffers(dKernel, dBias, dInput);
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

  const size_t inElems = checked_channel_spatial_size(inC, inSpatialSize, "torchlean_cuda_conv_pool: input size overflow");
  const size_t kElems = checked_conv_kernel_size(inC, outC, kSpatialSize, "torchlean_cuda_conv_pool: kernel size overflow");
  const size_t bElems = (size_t)outC;
  const size_t outElems = checked_channel_spatial_size(outC, outSpatialSize, "torchlean_cuda_conv_pool: output size overflow");

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
  DeviceSpatialScratch scratch = alloc_spatial_scratch(rank);
  copy_spatial_scratch(rank, scratch, hOutSpatial, hInSpatial, hKSpatial, hStride, hPadding);

  const int grid = grid_for(outElems);
  convnd_dinput_kernel<<<grid, kBlock>>>(kernel->data, input->data, out->data,
                                        /*inC=*/outC, /*outC=*/inC,
                                        scratch.inSpatial, scratch.outSpatial, scratch.kSpatial,
                                        scratch.stride, scratch.padding,
                                        rank, outSpatialSize, inSpatialSize, kSpatialSize);
  checkCuda(cudaGetLastError(), "convtranspose_fwd (convnd_dinput) kernel launch failed");

  if (outC > 0 && outSpatialSize > 0) {
    const int gridBias = grid_for(outElems);
    add_bias_cspatial_kernel<<<gridBias, kBlock>>>(out->data, bias->data, outC, outSpatialSize);
    checkCuda(cudaGetLastError(), "convtranspose_fwd add_bias kernel launch failed");
  }

  checkCuda(cudaDeviceSynchronize(), "convtranspose_fwd sync failed");

  free_spatial_scratch(rank, &scratch);

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

  const size_t inElems = checked_channel_spatial_size(inC, inSpatialSize, "torchlean_cuda_conv_pool: input size overflow");
  const size_t kElems = checked_conv_kernel_size(inC, outC, kSpatialSize, "torchlean_cuda_conv_pool: kernel size overflow");
  const size_t outElems = checked_channel_spatial_size(outC, outSpatialSize, "torchlean_cuda_conv_pool: output size overflow");

  checkBufSize(input, inElems, "torchlean_cuda_convtranspose_bwd: input.size mismatch");
  checkBufSize(kernel, kElems, "torchlean_cuda_convtranspose_bwd: kernel.size mismatch");
  checkBufSize(gradOutput, outElems, "torchlean_cuda_convtranspose_bwd: gradOutput.size mismatch");

  torchlean_cuda_buffer* dKernel = torchlean_cuda_buffer_alloc(kElems);
  torchlean_cuda_buffer* dBias = torchlean_cuda_buffer_alloc((size_t)outC);
  torchlean_cuda_buffer* dInput = torchlean_cuda_buffer_alloc(inElems);

  if (kElems == 0 && (size_t)outC == 0 && inElems == 0) {
    return torchlean_cuda_box_three_buffers(dKernel, dBias, dInput);
  }

  DeviceSpatialScratch scratch = alloc_spatial_scratch(rank);
  copy_spatial_scratch(rank, scratch, hOutSpatial, hInSpatial, hKSpatial, hStride, hPadding);

  if ((size_t)outC > 0) {
    const int gridBias = grid_for((size_t)outC);
    convnd_dbias_kernel<<<gridBias, kBlock>>>(gradOutput->data, dBias->data, outC, outSpatialSize);
    checkCuda(cudaGetLastError(), "convtranspose_bwd dbias kernel launch failed");
  }

  if (kElems > 0) {
    const int gridK = grid_for(kElems);
    convnd_dkernel_kernel<<<gridK, kBlock>>>(gradOutput->data, input->data, dKernel->data,
                                            /*inC=*/outC, /*outC=*/inC,
                                            scratch.inSpatial, scratch.outSpatial, scratch.kSpatial,
                                            scratch.stride, scratch.padding,
                                            rank, inSpatialSize, kSpatialSize);
    checkCuda(cudaGetLastError(), "convtranspose_bwd dkernel kernel launch failed");
  }

  if (inElems > 0) {
    const int gridIn = grid_for(inElems);
    convnd_fwd_kernel<<<gridIn, kBlock>>>(gradOutput->data, kernel->data, /*bias=*/nullptr, dInput->data,
                                         /*inC=*/outC, /*outC=*/inC,
                                         scratch.inSpatial, scratch.outSpatial, scratch.kSpatial,
                                         scratch.stride, scratch.padding,
                                         rank, inSpatialSize, kSpatialSize);
    checkCuda(cudaGetLastError(), "convtranspose_bwd dinput (convnd_fwd) kernel launch failed");
  }

  checkCuda(cudaDeviceSynchronize(), "convtranspose_bwd sync failed");

  free_spatial_scratch(rank, &scratch);

  return torchlean_cuda_box_three_buffers(dKernel, dBias, dInput);
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
    hOutSpatial[ax] = poolOutDim(hInSpatial[ax], hKSpatial[ax], hStride[ax], hPadding[ax]);
  }

  const size_t inSpatialSize = prod_u32(hInSpatial, rank);
  const size_t kSpatialSize = prod_u32(hKSpatial, rank);
  const size_t outSpatialSize = prod_u32(hOutSpatial, rank);

  const size_t inElems = checked_channel_spatial_size(inC, inSpatialSize, "torchlean_cuda_conv_pool: input size overflow");
  const size_t outElems = checked_channel_spatial_size(inC, outSpatialSize, "torchlean_cuda_conv_pool: output size overflow");

  checkBufSize(input, inElems, "torchlean_cuda_maxpool_fwd: input.size mismatch");

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outElems);
  if (outElems == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  DeviceSpatialScratch scratch = alloc_spatial_scratch(rank);
  copy_spatial_scratch(rank, scratch, hInSpatial, hOutSpatial, hKSpatial, hStride, hPadding);

  const int grid = grid_for(outElems);
  maxpoolnd_fwd_kernel<<<grid, kBlock>>>(input->data, out->data,
                                        inC,
                                        scratch.inSpatial, scratch.outSpatial, scratch.kSpatial,
                                        scratch.stride, scratch.padding,
                                        rank, outSpatialSize, kSpatialSize);
  checkCuda(cudaGetLastError(), "maxpoolnd_fwd kernel launch failed");
  checkCuda(cudaDeviceSynchronize(), "maxpoolnd_fwd sync failed");

  free_spatial_scratch(rank, &scratch);

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
    hOutSpatial[ax] = poolOutDim(hInSpatial[ax], hKSpatial[ax], hStride[ax], hPadding[ax]);
  }

  const size_t inSpatialSize = prod_u32(hInSpatial, rank);
  const size_t kSpatialSize = prod_u32(hKSpatial, rank);
  const size_t outSpatialSize = prod_u32(hOutSpatial, rank);

  const size_t inElems = checked_channel_spatial_size(inC, inSpatialSize, "torchlean_cuda_conv_pool: input size overflow");
  const size_t outElems = checked_channel_spatial_size(inC, outSpatialSize, "torchlean_cuda_conv_pool: output size overflow");

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

  DeviceSpatialScratch scratch = alloc_spatial_scratch(rank);
  copy_spatial_scratch(rank, scratch, hInSpatial, hOutSpatial, hKSpatial, hStride, hPadding);

  if (det) {
    const int grid = grid_for(inElems);
    maxpoolnd_bwd_det_kernel<<<grid, kBlock>>>(input->data, gradOutput->data, dInput->data,
                                              inC,
                                              scratch.inSpatial, scratch.outSpatial, scratch.kSpatial,
                                              scratch.stride, scratch.padding,
                                              rank, inSpatialSize, outSpatialSize, kSpatialSize);
    checkCuda(cudaGetLastError(), "maxpoolnd_bwd (deterministic) kernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "maxpoolnd_bwd (deterministic) sync failed");
  } else {
    const int grid = grid_for(outElems);
    maxpoolnd_bwd_kernel<<<grid, kBlock>>>(input->data, gradOutput->data, dInput->data,
                                          inC,
                                          scratch.inSpatial, scratch.outSpatial, scratch.kSpatial,
                                          scratch.stride, scratch.padding,
                                          rank, outSpatialSize, kSpatialSize);
    checkCuda(cudaGetLastError(), "maxpoolnd_bwd kernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "maxpoolnd_bwd sync failed");
  }

  free_spatial_scratch(rank, &scratch);

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
    hOutSpatial[ax] = poolOutDim(hInSpatial[ax], hKSpatial[ax], hStride[ax], hPadding[ax]);
  }

  const size_t inSpatialSize = prod_u32(hInSpatial, rank);
  const size_t kSpatialSize = prod_u32(hKSpatial, rank);
  const size_t outSpatialSize = prod_u32(hOutSpatial, rank);

  const size_t inElems = checked_channel_spatial_size(inC, inSpatialSize, "torchlean_cuda_conv_pool: input size overflow");
  const size_t outElems = checked_channel_spatial_size(inC, outSpatialSize, "torchlean_cuda_conv_pool: output size overflow");

  checkBufSize(input, inElems, "torchlean_cuda_avgpool_fwd: input.size mismatch");

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outElems);
  if (outElems == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  DeviceSpatialScratch scratch = alloc_spatial_scratch(rank);
  copy_spatial_scratch(rank, scratch, hInSpatial, hOutSpatial, hKSpatial, hStride, hPadding);

  const int grid = grid_for(outElems);
  avgpoolnd_fwd_kernel<<<grid, kBlock>>>(input->data, out->data,
                                       inC,
                                       scratch.inSpatial, scratch.outSpatial, scratch.kSpatial,
                                       scratch.stride, scratch.padding,
                                       rank, outSpatialSize, kSpatialSize);
  checkCuda(cudaGetLastError(), "avgpoolnd_fwd kernel launch failed");
  checkCuda(cudaDeviceSynchronize(), "avgpoolnd_fwd sync failed");

  free_spatial_scratch(rank, &scratch);

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
    hOutSpatial[ax] = poolOutDim(hInSpatial[ax], hKSpatial[ax], hStride[ax], hPadding[ax]);
  }

  const size_t inSpatialSize = prod_u32(hInSpatial, rank);
  const size_t kSpatialSize = prod_u32(hKSpatial, rank);
  const size_t outSpatialSize = prod_u32(hOutSpatial, rank);

  const size_t inElems = checked_channel_spatial_size(inC, inSpatialSize, "torchlean_cuda_conv_pool: input size overflow");
  const size_t outElems = checked_channel_spatial_size(inC, outSpatialSize, "torchlean_cuda_conv_pool: output size overflow");

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

  DeviceSpatialScratch scratch = alloc_spatial_scratch(rank);
  copy_spatial_scratch(rank, scratch, hInSpatial, hOutSpatial, hKSpatial, hStride, hPadding);

  if (det) {
    const int grid = grid_for(inElems);
    avgpoolnd_bwd_det_kernel<<<grid, kBlock>>>(gradOutput->data, dInput->data,
                                              inC,
                                              scratch.inSpatial, scratch.outSpatial, scratch.kSpatial,
                                              scratch.stride, scratch.padding,
                                              rank, inSpatialSize, outSpatialSize, kSpatialSize);
    checkCuda(cudaGetLastError(), "avgpoolnd_bwd (deterministic) kernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "avgpoolnd_bwd (deterministic) sync failed");
  } else {
    const int grid = grid_for(outElems);
    avgpoolnd_bwd_kernel<<<grid, kBlock>>>(gradOutput->data, dInput->data,
                                          inC,
                                          scratch.inSpatial, scratch.outSpatial, scratch.kSpatial,
                                          scratch.stride, scratch.padding,
                                          rank, outSpatialSize, kSpatialSize);
    checkCuda(cudaGetLastError(), "avgpoolnd_bwd kernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "avgpoolnd_bwd sync failed");
  }

  free_spatial_scratch(rank, &scratch);

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
    hOutSpatial[ax] =
        poolOutDim(hInSpatial[ax], hKSpatial[ax], hStride[ax], hPadding[ax]);
  }

  const size_t inSpatialSize = prod_u32(hInSpatial, rank);
  const size_t kSpatialSize = prod_u32(hKSpatial, rank);
  const size_t outSpatialSize = prod_u32(hOutSpatial, rank);

  const size_t inElems = checked_channel_spatial_size(inC, inSpatialSize, "torchlean_cuda_conv_pool: input size overflow");
  const size_t outElems = checked_channel_spatial_size(inC, outSpatialSize, "torchlean_cuda_conv_pool: output size overflow");

  checkBufSize(input, inElems, "torchlean_cuda_smooth_maxpool_fwd: input.size mismatch");

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outElems);
  if (outElems == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  DeviceSpatialScratch scratch = alloc_spatial_scratch(rank);
  copy_spatial_scratch(rank, scratch, hInSpatial, hOutSpatial, hKSpatial, hStride, hPadding);

  const int grid = grid_for(outElems);
  const float betaF =
      checked_smoothmax_beta(beta, "torchlean_cuda_smooth_maxpool_fwd: beta must be finite and nonzero");
  smoothmaxpoolnd_fwd_kernel<<<grid, kBlock>>>(input->data, out->data,
                                              inC,
                                              scratch.inSpatial, scratch.outSpatial, scratch.kSpatial,
                                              scratch.stride, scratch.padding,
                                              rank, outSpatialSize, kSpatialSize,
                                              betaF);
  checkCuda(cudaGetLastError(), "smoothmaxpoolnd_fwd kernel launch failed");
  checkCuda(cudaDeviceSynchronize(), "smoothmaxpoolnd_fwd sync failed");

  free_spatial_scratch(rank, &scratch);

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
    hOutSpatial[ax] =
        poolOutDim(hInSpatial[ax], hKSpatial[ax], hStride[ax], hPadding[ax]);
  }

  const size_t inSpatialSize = prod_u32(hInSpatial, rank);
  const size_t kSpatialSize = prod_u32(hKSpatial, rank);
  const size_t outSpatialSize = prod_u32(hOutSpatial, rank);

  const size_t inElems = checked_channel_spatial_size(inC, inSpatialSize, "torchlean_cuda_conv_pool: input size overflow");
  const size_t outElems = checked_channel_spatial_size(inC, outSpatialSize, "torchlean_cuda_conv_pool: output size overflow");

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

  DeviceSpatialScratch scratch = alloc_spatial_scratch(rank);
  copy_spatial_scratch(rank, scratch, hInSpatial, hOutSpatial, hKSpatial, hStride, hPadding);

  if (det) {
    const int grid = grid_for(inElems);
    smoothmaxpoolnd_bwd_det_kernel<<<grid, kBlock>>>(input->data, gradOutput->data, dInput->data,
                                                    inC,
                                                    scratch.inSpatial, scratch.outSpatial,
                                                    scratch.kSpatial, scratch.stride, scratch.padding,
                                                    rank, inSpatialSize, outSpatialSize, kSpatialSize,
                                                    betaF);
    checkCuda(cudaGetLastError(), "smoothmaxpoolnd_bwd (deterministic) kernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "smoothmaxpoolnd_bwd (deterministic) sync failed");
  } else {
    const int grid = grid_for(outElems);
    smoothmaxpoolnd_bwd_kernel<<<grid, kBlock>>>(input->data, gradOutput->data, dInput->data,
                                                inC,
                                                scratch.inSpatial, scratch.outSpatial, scratch.kSpatial,
                                                scratch.stride, scratch.padding,
                                                rank, outSpatialSize, kSpatialSize,
                                                betaF);
    checkCuda(cudaGetLastError(), "smoothmaxpoolnd_bwd kernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "smoothmaxpoolnd_bwd sync failed");
  }

  free_spatial_scratch(rank, &scratch);

  return torchlean_cuda_buffer_box(dInput);
}
