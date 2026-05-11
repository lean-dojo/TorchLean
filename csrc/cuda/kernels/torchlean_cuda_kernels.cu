#include <lean/lean.h>

#include "torchlean_cuda_buffer.h"
#include "torchlean_cuda_common.h"
#include "torchlean_cublas_common.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>

#include <assert.h>
#include <math.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// CUDA implementation of TorchLean's general float32 tensor kernels.
//
// This file covers kernels that sit above raw elementwise buffer ops and below model-specific
// layers: reductions over axes, broadcasting, gathers/scatters, batched matmul, selective scan, and
// the correctness-first fused attention path. Host `LEAN_EXPORT` wrappers validate Lean-provided
// shape metadata before launching device kernels; device kernels assume that validation succeeded.
//
// Layout conventions:
// - all tensor buffers are flat row-major arrays;
// - rank-polymorphic kernels are capped at `kMaxRank` so stack coordinate arrays stay bounded;
// - atomic accumulation has deterministic alternatives where reproducibility matters;
// - cuBLAS calls use the row-major-as-transposed-column-major convention documented at call sites.
// - cuFFT R2C/C2R kernels expose spectra as packed real/imag float32 buffers:
//   `(batch, n/2+1, 2)`, last channel `[re, im]`.

static constexpr int kBlockSize = 256;
static constexpr int kMaxRank = 8;
static_assert(kBlockSize > 0 && (kBlockSize & (kBlockSize - 1)) == 0,
              "kBlockSize must remain a power of two for shared-memory reductions");

static inline dim3 blocks_for(size_t n) {
  size_t blocks = (n + (size_t)kBlockSize - 1) / (size_t)kBlockSize;
  if (blocks == 0) blocks = 1;
  if (blocks > 2147483647ULL) blocks = 2147483647ULL;
  return dim3((unsigned int)blocks);
}

static inline size_t checked_mul_size(size_t a, size_t b, const char* msg) {
  if (a != 0 && b > SIZE_MAX / a) {
    lean_internal_panic(msg);
  }
  return a * b;
}

static inline size_t checked_mul3_size(size_t a, size_t b, size_t c, const char* msg) {
  return checked_mul_size(checked_mul_size(a, b, msg), c, msg);
}

static inline void check_axis_grid_size(size_t blocks, const char* msg) {
  // These reductions launch one block per output row/column, so reject shapes outside the CUDA
  // x-grid range instead of relying on device-side behavior.
  if (blocks > 2147483647ULL) {
    lean_internal_panic(msg);
  }
}

static inline void checkCufft(cufftResult r, const char* msg) {
  if (r != CUFFT_SUCCESS) {
    lean_internal_panic(msg);
  }
}

static inline int checked_cufft_int(uint32_t x, const char* msg) {
  if (x > (uint32_t)INT_MAX) {
    lean_internal_panic(msg);
  }
  return (int)x;
}

__global__ void pack_cufft_complex_to_ri_f32(const cufftComplex* in, float* out, size_t count) {
  const size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (i >= count) return;
  out[2 * i] = in[i].x;
  out[2 * i + 1] = in[i].y;
}

__global__ void pack_ri_to_cufft_complex_f32(const float* in, cufftComplex* out, size_t count) {
  const size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (i >= count) return;
  out[i].x = in[2 * i];
  out[i].y = in[2 * i + 1];
}

__global__ void scale_f32(float* data, size_t n, float scale) {
  const size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (i >= n) return;
  data[i] *= scale;
}

__global__ void spectral_conv1d_mul_f32(const cufftComplex* X, const float* wRe,
                                        const float* wIm, cufftComplex* Z, uint32_t width,
                                        uint32_t modes) {
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t total = (size_t)modes * (size_t)width;
  if (idx >= total) return;
  const uint32_t o = (uint32_t)(idx % (size_t)width);
  const uint32_t k = (uint32_t)(idx / (size_t)width);

  float zr = 0.0f;
  float zi = 0.0f;
  for (uint32_t c = 0; c < width; ++c) {
    const cufftComplex x = X[(size_t)k * (size_t)width + (size_t)c];
    const size_t wIdx =
        ((size_t)k * (size_t)width + (size_t)c) * (size_t)width + (size_t)o;
    const float wr = wRe[wIdx];
    const float wi = wIm[wIdx];
    zr += x.x * wr - x.y * wi;
    zi += x.x * wi + x.y * wr;
  }
  Z[(size_t)k * (size_t)width + (size_t)o].x = zr;
  Z[(size_t)k * (size_t)width + (size_t)o].y = zi;
}

__global__ void spectral_conv1d_dz_from_dy_f32(const cufftComplex* G, cufftComplex* dZ,
                                              uint32_t grid, uint32_t width, uint32_t modes) {
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t total = (size_t)modes * (size_t)width;
  if (idx >= total) return;
  const uint32_t o = (uint32_t)(idx % (size_t)width);
  const uint32_t k = (uint32_t)(idx / (size_t)width);
  const uint32_t nyquist = grid / 2;
  const bool edge = (k == 0) || ((grid % 2 == 0) && (k == nyquist));
  const float scale = (edge ? 1.0f : 2.0f) / (float)grid;
  const cufftComplex g = G[(size_t)k * (size_t)width + (size_t)o];
  dZ[(size_t)k * (size_t)width + (size_t)o].x = scale * g.x;
  dZ[(size_t)k * (size_t)width + (size_t)o].y = scale * g.y;
}

__global__ void spectral_conv1d_bwd_weights_f32(const cufftComplex* X, const cufftComplex* dZ,
                                                float* dWRe, float* dWIm, uint32_t width,
                                                uint32_t modes) {
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t total = (size_t)modes * (size_t)width * (size_t)width;
  if (idx >= total) return;
  const uint32_t o = (uint32_t)(idx % (size_t)width);
  const uint32_t c = (uint32_t)((idx / (size_t)width) % (size_t)width);
  const uint32_t k = (uint32_t)(idx / ((size_t)width * (size_t)width));
  const cufftComplex x = X[(size_t)k * (size_t)width + (size_t)c];
  const cufftComplex dz = dZ[(size_t)k * (size_t)width + (size_t)o];
  dWRe[idx] = x.x * dz.x + x.y * dz.y;
  dWIm[idx] = x.x * dz.y - x.y * dz.x;
}

__global__ void spectral_conv1d_bwd_xspec_f32(const cufftComplex* dZ, const float* wRe,
                                              const float* wIm, cufftComplex* dX,
                                              uint32_t width, uint32_t modes) {
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t total = (size_t)modes * (size_t)width;
  if (idx >= total) return;
  const uint32_t c = (uint32_t)(idx % (size_t)width);
  const uint32_t k = (uint32_t)(idx / (size_t)width);
  float xr = 0.0f;
  float xi = 0.0f;
  for (uint32_t o = 0; o < width; ++o) {
    const cufftComplex dz = dZ[(size_t)k * (size_t)width + (size_t)o];
    const size_t wIdx =
        ((size_t)k * (size_t)width + (size_t)c) * (size_t)width + (size_t)o;
    const float wr = wRe[wIdx];
    const float wi = wIm[wIdx];
    xr += dz.x * wr + dz.y * wi;
    xi += dz.y * wr - dz.x * wi;
  }
  dX[(size_t)k * (size_t)width + (size_t)c].x = xr;
  dX[(size_t)k * (size_t)width + (size_t)c].y = xi;
}

__global__ void spectral_conv1d_irfft_adjoint_f32(const cufftComplex* dX, float* dx,
                                                 uint32_t grid, uint32_t width,
                                                 uint32_t modes) {
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t total = (size_t)grid * (size_t)width;
  if (idx >= total) return;
  const uint32_t c = (uint32_t)(idx % (size_t)width);
  const uint32_t t = (uint32_t)(idx / (size_t)width);
  float acc = 0.0f;
  for (uint32_t k = 0; k < modes; ++k) {
    const cufftComplex g = dX[(size_t)k * (size_t)width + (size_t)c];
    const float angle = 2.0f * 3.14159265358979323846f * (float)k * (float)t / (float)grid;
    acc += g.x * cosf(angle) - g.y * sinf(angle);
  }
  dx[idx] = acc;
}

__global__ void reduce_sum_axis0_f32(const float* in, float* out, uint32_t rows, uint32_t cols) {
  if (blockDim.x != kBlockSize) return;
  const uint32_t col = (uint32_t)blockIdx.x;
  if (col >= cols) return;

  __shared__ float sdata[kBlockSize];
  const int tid = threadIdx.x;

  float sum = 0.0f;
  for (uint32_t r = (uint32_t)tid; r < rows; r += (uint32_t)blockDim.x) {
    sum += in[(size_t)r * (size_t)cols + (size_t)col];
  }
  sdata[tid] = sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[col] = sdata[0];
  }
}

__global__ void reduce_sum_axis1_f32(const float* in, float* out, uint32_t rows, uint32_t cols) {
  if (blockDim.x != kBlockSize) return;
  const uint32_t row = (uint32_t)blockIdx.x;
  if (row >= rows) return;

  __shared__ float sdata[kBlockSize];
  const int tid = threadIdx.x;

  float sum = 0.0f;
  for (uint32_t c = (uint32_t)tid; c < cols; c += (uint32_t)blockDim.x) {
    sum += in[(size_t)row * (size_t)cols + (size_t)c];
  }
  sdata[tid] = sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[row] = sdata[0];
  }
}

__global__ void reduce_max_axis0_f32(const float* in, float* out, uint32_t rows, uint32_t cols) {
  if (blockDim.x != kBlockSize) return;
  const uint32_t col = (uint32_t)blockIdx.x;
  if (col >= cols) return;

  __shared__ float sdata[kBlockSize];
  const int tid = threadIdx.x;

  float m = -INFINITY;
  for (uint32_t r = (uint32_t)tid; r < rows; r += (uint32_t)blockDim.x) {
    float v = in[(size_t)r * (size_t)cols + (size_t)col];
    m = fmaxf(m, v);
  }
  sdata[tid] = m;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[col] = sdata[0];
  }
}

// ----------------------------
// General broadcast / permute / reduction kernels
// ----------------------------

__global__ void broadcast_to_f32(const float* in, float* out, size_t outSize,
                                 const uint32_t* inDims, int rankIn,
                                 const uint32_t* outDims, int rankOut,
                                 const uint32_t* axisMap) {
  size_t outIdx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (outIdx >= outSize) return;

  uint32_t inCoords[kMaxRank];
  for (int i = 0; i < kMaxRank; ++i) inCoords[i] = 0;

  size_t tmp = outIdx;
  for (int ax = rankOut - 1; ax >= 0; --ax) {
    const uint32_t od = outDims[ax];
    const uint32_t coord = (uint32_t)(tmp % (size_t)od);
    tmp /= (size_t)od;

    const uint32_t mv = axisMap[ax];
    if (mv == 0) continue;
    const int inAx = (int)mv - 1;
    const uint32_t id = inDims[inAx];
    if (id != 1) {
      inCoords[inAx] = coord;
    }
  }

  size_t inIdx = 0;
  for (int ax = 0; ax < rankIn; ++ax) {
    inIdx = inIdx * (size_t)inDims[ax] + (size_t)inCoords[ax];
  }
  out[outIdx] = in[inIdx];
}

__global__ void reduce_from_broadcast_f32(const float* dOut, size_t outSize,
                                         float* dIn,
                                         const uint32_t* inDims, int rankIn,
                                         const uint32_t* outDims, int rankOut,
                                         const uint32_t* axisMap) {
  size_t outIdx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (outIdx >= outSize) return;

  uint32_t inCoords[kMaxRank];
  for (int i = 0; i < kMaxRank; ++i) inCoords[i] = 0;

  size_t tmp = outIdx;
  for (int ax = rankOut - 1; ax >= 0; --ax) {
    const uint32_t od = outDims[ax];
    const uint32_t coord = (uint32_t)(tmp % (size_t)od);
    tmp /= (size_t)od;

    const uint32_t mv = axisMap[ax];
    if (mv == 0) continue;
    const int inAx = (int)mv - 1;
    const uint32_t id = inDims[inAx];
    if (id != 1) {
      inCoords[inAx] = coord;
    }
  }

  size_t inIdx = 0;
  for (int ax = 0; ax < rankIn; ++ax) {
    inIdx = inIdx * (size_t)inDims[ax] + (size_t)inCoords[ax];
  }
  atomicAdd(&dIn[inIdx], dOut[outIdx]);
}

// Deterministic reduction from broadcast:
// One thread computes one `dIn[inIdx]` by summing all `dOut[outIdx]` that map to it in a fixed
// (row-major) order. This avoids `atomicAdd` and is bit-stable across runs, but can be much slower
// than the atomic path when the broadcast expansion factor is large.
__global__ void reduce_from_broadcast_det_f32(const float* dOut, size_t outSize,
                                             float* dIn, size_t inSize,
                                             const uint32_t* inDims, int rankIn,
                                             const uint32_t* outDims, int rankOut,
                                             const uint32_t* axisMap) {
  const size_t inIdx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (inIdx >= inSize) return;

  // Decode `inIdx` into coordinates.
  uint32_t inCoords[kMaxRank];
  for (int i = 0; i < kMaxRank; ++i) inCoords[i] = 0;

  size_t tmpIn = inIdx;
  for (int ax = rankIn - 1; ax >= 0; --ax) {
    const uint32_t d = inDims[ax];
    const uint32_t coord = (d == 0) ? 0 : (uint32_t)(tmpIn % (size_t)d);
    tmpIn = (d == 0) ? 0 : (tmpIn / (size_t)d);
    inCoords[ax] = coord;
  }

  // Precompute row-major strides for the output shape.
  size_t outStride[kMaxRank];
  size_t stride = 1;
  for (int ax = rankOut - 1; ax >= 0; --ax) {
    outStride[ax] = stride;
    stride *= (size_t)outDims[ax];
  }

  // Determine which output axes are "free" (summed over) for this fixed input coordinate,
  // and compute the base output linear index for the fixed axes.
  int freeAxes[kMaxRank];
  int freeCount = 0;
  size_t freeTotal = 1;
  size_t baseOutIdx = 0;

  for (int ax = 0; ax < rankOut; ++ax) {
    const uint32_t od = outDims[ax];
    const uint32_t mv = axisMap[ax];

    bool isFree = false;
    uint32_t fixedCoord = 0;
    if (mv == 0) {
      isFree = true;
    } else {
      const int inAx = (int)mv - 1;
      const uint32_t id = inDims[inAx];
      if (id == 1) {
        isFree = true;
      } else {
        fixedCoord = inCoords[inAx];
        if (fixedCoord >= od) {
          // Defensive: shape mismatch; treat as no contribution.
          dIn[inIdx] = 0.0f;
          return;
        }
      }
    }

    if (isFree) {
      freeAxes[freeCount++] = ax;
      freeTotal *= (size_t)od;
    } else {
      baseOutIdx += (size_t)fixedCoord * outStride[ax];
    }
  }

  // Sum over the cartesian product of free axes in a fixed row-major order.
  float acc = 0.0f;
  for (size_t t = 0; t < freeTotal; ++t) {
    size_t tmp = t;
    size_t outIdx = baseOutIdx;
    for (int j = freeCount - 1; j >= 0; --j) {
      const int ax = freeAxes[j];
      const uint32_t d = outDims[ax];
      const size_t coord = (d == 0) ? 0 : (tmp % (size_t)d);
      tmp = (d == 0) ? 0 : (tmp / (size_t)d);
      outIdx += coord * outStride[ax];
    }
    if (outIdx < outSize) {
      acc += dOut[outIdx];
    }
  }

  dIn[inIdx] = acc;
}

__global__ void swap_adjacent_at_depth_f32(const float* in, float* out, size_t total,
                                          const uint32_t* dims, int rank, int depth) {
  size_t outIdx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (outIdx >= total) return;

  uint32_t coords[kMaxRank];
  size_t tmp = outIdx;
  for (int ax = rank - 1; ax >= 0; --ax) {
    uint32_t d = dims[ax];
    if (ax == depth) {
      d = dims[depth + 1];
    } else if (ax == depth + 1) {
      d = dims[depth];
    }
    coords[ax] = (d == 0) ? 0 : (uint32_t)(tmp % (size_t)d);
    tmp = (d == 0) ? 0 : (tmp / (size_t)d);
  }

  // `coords` are output coordinates. Swapping the adjacent coordinates maps them back to
  // the corresponding coordinates in the input layout.
  uint32_t t = coords[depth];
  coords[depth] = coords[depth + 1];
  coords[depth + 1] = t;

  size_t inIdx = 0;
  for (int ax = 0; ax < rank; ++ax) {
    inIdx = inIdx * (size_t)dims[ax] + (size_t)coords[ax];
  }
  out[outIdx] = in[inIdx];
}

__global__ void reduce_sum_axis_f32(const float* in, float* out, size_t inSize,
                                   const uint32_t* dims, int rank, int axis) {
  size_t inIdx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (inIdx >= inSize) return;

  uint32_t coords[kMaxRank];
  size_t tmp = inIdx;
  for (int ax = rank - 1; ax >= 0; --ax) {
    const uint32_t d = dims[ax];
    coords[ax] = (uint32_t)(tmp % (size_t)d);
    tmp /= (size_t)d;
  }

  size_t outIdx = 0;
  for (int ax = 0; ax < rank; ++ax) {
    if (ax == axis) continue;
    outIdx = outIdx * (size_t)dims[ax] + (size_t)coords[ax];
  }
  atomicAdd(&out[outIdx], in[inIdx]);
}

// Deterministic reduction along an axis:
// One thread computes one output element by looping over the reduced axis in a fixed order.
// This avoids `atomicAdd` and is bit-stable across runs, but is typically slower than the atomic
// accumulation kernel for large tensors.
__global__ void reduce_sum_axis_det_f32(const float* in, float* out, size_t outSize,
                                       const uint32_t* dims, int rank, int axis) {
  const size_t outIdx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (outIdx >= outSize) return;

  // Decode `outIdx` into coordinates, treating `axis` as a degenerate dimension of size 1.
  uint32_t coords[kMaxRank];
  for (int i = 0; i < kMaxRank; ++i) coords[i] = 0;

  size_t tmp = outIdx;
  for (int ax = rank - 1; ax >= 0; --ax) {
    if (ax == axis) {
      coords[ax] = 0;
      continue;
    }
    const uint32_t d = dims[ax];
    const uint32_t coord = (d == 0) ? 0 : (uint32_t)(tmp % (size_t)d);
    tmp = (d == 0) ? 0 : (tmp / (size_t)d);
    coords[ax] = coord;
  }

  // Compute the row-major stride for the reduced axis.
  size_t strideAxis = 1;
  for (int ax = axis + 1; ax < rank; ++ax) {
    strideAxis *= (size_t)dims[ax];
  }

  // Base linear index with the reduced coordinate set to 0.
  size_t base = 0;
  for (int ax = 0; ax < rank; ++ax) {
    const uint32_t c = (ax == axis) ? 0 : coords[ax];
    base = base * (size_t)dims[ax] + (size_t)c;
  }

  const uint32_t axisDim = dims[axis];
  float acc = 0.0f;
  for (uint32_t a = 0; a < axisDim; ++a) {
    acc += in[base + (size_t)a * strideAxis];
  }

  out[outIdx] = acc;
}

__global__ void gather_rows_f32(const float* mat, uint32_t rows, uint32_t cols,
                                const uint32_t* idx, uint32_t k, float* out) {
  size_t outIdx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t total = (size_t)k * (size_t)cols;
  if (outIdx >= total) return;

  const uint32_t r = (uint32_t)(outIdx / (size_t)cols);
  const uint32_t c = (uint32_t)(outIdx % (size_t)cols);
  const uint32_t srcRow = idx[r];
  if (srcRow < rows) {
    out[outIdx] = mat[(size_t)srcRow * (size_t)cols + (size_t)c];
  } else {
    out[outIdx] = 0.0f;
  }
}

__global__ void scatter_add_row_f32(const float* mat, const float* rowVec,
                                   uint32_t rows, uint32_t cols, uint32_t row,
                                   float* out) {
  size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t total = (size_t)rows * (size_t)cols;
  if (idx >= total) return;
  const uint32_t r = (uint32_t)(idx / (size_t)cols);
  const uint32_t c = (uint32_t)(idx % (size_t)cols);
  float v = mat[idx];
  if (r == row) {
    v += rowVec[c];
  }
  out[idx] = v;
}

__global__ void scatter_add_rows_f32(const float* values, const uint32_t* idx,
                                    uint32_t k, uint32_t rows, uint32_t cols,
                                    float* out) {
  size_t t = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t total = (size_t)k * (size_t)cols;
  if (t >= total) return;
  const uint32_t r = (uint32_t)(t / (size_t)cols);
  const uint32_t c = (uint32_t)(t % (size_t)cols);
  const uint32_t dstRow = idx[r];
  if (dstRow < rows) {
    atomicAdd(&out[(size_t)dstRow * (size_t)cols + (size_t)c], values[t]);
  }
}

// Deterministic row scatter-add:
// One thread computes one output element (row,col) by scanning all updates in increasing `r`
// order, accumulating a fixed-order sum. This avoids `atomicAdd` but is O(rows*cols*k).
__global__ void scatter_add_rows_det_f32(const float* mat, const float* values, const uint32_t* idx,
                                        uint32_t k, uint32_t rows, uint32_t cols, float* out) {
  const size_t t = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t total = (size_t)rows * (size_t)cols;
  if (t >= total) return;

  const uint32_t r = (uint32_t)(t / (size_t)cols);
  const uint32_t c = (uint32_t)(t % (size_t)cols);

  float acc = 0.0f;
  for (uint32_t j = 0; j < k; ++j) {
    if (idx[(size_t)j] == r) {
      acc += values[(size_t)j * (size_t)cols + (size_t)c];
    }
  }

  out[t] = mat[t] + acc;
}

__global__ void reduce_max_axis1_f32(const float* in, float* out, uint32_t rows, uint32_t cols) {
  if (blockDim.x != kBlockSize) return;
  const uint32_t row = (uint32_t)blockIdx.x;
  if (row >= rows) return;

  __shared__ float sdata[kBlockSize];
  const int tid = threadIdx.x;

  float m = -INFINITY;
  for (uint32_t c = (uint32_t)tid; c < cols; c += (uint32_t)blockDim.x) {
    float v = in[(size_t)row * (size_t)cols + (size_t)c];
    m = fmaxf(m, v);
  }
  sdata[tid] = m;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[row] = sdata[0];
  }
}

__global__ void concat1d_f32(const float* a, size_t n, const float* b, size_t m, float* out) {
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t total = n + m;
  if (idx >= total) return;
  out[idx] = (idx < n) ? a[idx] : b[idx - n];
}

__global__ void slice1d_f32(const float* in, size_t start, size_t len, float* out) {
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (idx >= len) return;
  out[idx] = in[start + idx];
}

__global__ void broadcast_vec_to_rows_f32(const float* vec, uint32_t rows, uint32_t cols,
                                         float* out) {
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t total = (size_t)rows * (size_t)cols;
  if (idx >= total) return;
  const uint32_t j = (uint32_t)(idx % (size_t)cols);
  out[idx] = vec[j];
}

__global__ void broadcast_vec_to_cols_f32(const float* vec, uint32_t rows, uint32_t cols,
                                         float* out) {
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t total = (size_t)rows * (size_t)cols;
  if (idx >= total) return;
  const uint32_t i = (uint32_t)(idx / (size_t)cols);
  out[idx] = vec[i];
}

__global__ void transpose2d_f32(const float* in, float* out, uint32_t rows, uint32_t cols) {
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t total = (size_t)rows * (size_t)cols;
  if (idx >= total) return;
  const uint32_t i = (uint32_t)(idx / (size_t)cols);
  const uint32_t j = (uint32_t)(idx % (size_t)cols);
  out[(size_t)j * (size_t)rows + (size_t)i] = in[(size_t)i * (size_t)cols + (size_t)j];
}

__global__ void gather_vec_f32(const float* vec, const uint32_t* idx, size_t k, uint32_t n,
                              float* out) {
  const size_t j = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (j >= k) return;
  const uint32_t i = idx[j];
  out[j] = (i < n) ? vec[(size_t)i] : 0.0f;
}

__global__ void scatter_add_f32(const float* values, const uint32_t* idx, size_t k, uint32_t n,
                               float* out) {
  const size_t j = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (j >= k) return;
  const uint32_t i = idx[j];
  if (i < n) {
    atomicAdd(&out[(size_t)i], values[j]);
  }
}

// Deterministic 1D scatter-add:
// One thread computes one output element by scanning all updates in increasing `j` order.
// This avoids `atomicAdd` but is O(n*k).
__global__ void scatter_add_det_f32(const float* x, const float* values, const uint32_t* idx,
                                   size_t k, uint32_t n, float* out) {
  const size_t i64 = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (i64 >= (size_t)n) return;
  const uint32_t i = (uint32_t)i64;

  float acc = 0.0f;
  for (size_t j = 0; j < k; ++j) {
    if (idx[j] == i) {
      acc += values[j];
    }
  }

  out[i64] = x[i64] + acc;
}

__global__ void selective_scan_diag_fwd_f32(const float* A, const float* B, const float* X,
                                           const float* h0, float* out, uint32_t seqLen,
                                           uint32_t stateDim) {
  const size_t j = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (j >= (size_t)stateDim) return;

  // One CUDA thread owns one diagonal channel. Time is recurrent; channels are independent.
  float h = h0[j];
  for (uint32_t t = 0; t < seqLen; ++t) {
    const size_t idx = (size_t)t * (size_t)stateDim + j;
    h = A[j] * h + B[j] * X[idx];
    out[idx] = h;
  }
}

__global__ void selective_scan_diag_var_fwd_f32(const float* A, const float* B, const float* X,
                                               const float* h0, float* out, uint32_t seqLen,
                                               uint32_t stateDim) {
  const size_t j = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (j >= (size_t)stateDim) return;

  // Full Mamba uses token-dependent affine summaries. One thread owns one flattened diagonal
  // channel; time remains recurrent, but all channels are independent.
  float h = h0[j];
  for (uint32_t t = 0; t < seqLen; ++t) {
    const size_t idx = (size_t)t * (size_t)stateDim + j;
    h = A[idx] * h + B[idx] * X[idx];
    out[idx] = h;
  }
}

__global__ void selective_scan_diag_bwd_f32(const float* A, const float* B, const float* X,
                                           const float* h0, const float* out, const float* dY,
                                           float* dA, float* dB, float* dX, float* dH0,
                                           uint32_t seqLen, uint32_t stateDim) {
  const size_t j = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (j >= (size_t)stateDim) return;

  float accA = 0.0f;
  float accB = 0.0f;
  float dhNext = 0.0f;
  const float a = A[j];
  const float b = B[j];

  for (uint32_t tr = 0; tr < seqLen; ++tr) {
    const uint32_t t = seqLen - 1u - tr;
    const size_t idx = (size_t)t * (size_t)stateDim + j;
    const float hPrev = (t == 0u) ? h0[j] : out[((size_t)t - 1u) * (size_t)stateDim + j];
    const float total = dY[idx] + dhNext;

    accA += total * hPrev;
    accB += total * X[idx];
    dX[idx] = total * b;
    dhNext = total * a;
  }

  dA[j] = accA;
  dB[j] = accB;
  dH0[j] = dhNext;
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_sum_axis0(b_lean_obj_arg BObj,
                                                                          uint32_t rows,
                                                                          uint32_t cols) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  const size_t R = (size_t)rows;
  const size_t C = (size_t)cols;
  const size_t total = checked_mul_size(R, C, "torchlean_cuda_buffer_reduce_sum_axis0: R*C overflow");
  if (b->size != total) {
    lean_internal_panic("torchlean_cuda_buffer_reduce_sum_axis0: size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(C);
  if (C == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  if (R == 0) {
    checkCuda(cudaMemset(out->data, 0, C * sizeof(float)), "cudaMemset axis0 out failed");
    return torchlean_cuda_buffer_box(out);
  }

  check_axis_grid_size(C, "torchlean_cuda_buffer_reduce_sum_axis0: cols exceed CUDA x-grid range");
  dim3 blocks = dim3((unsigned int)C);
  dim3 threads = dim3(kBlockSize);
  reduce_sum_axis0_f32<<<blocks, threads>>>(b->data, out->data, rows, cols);
  checkCuda(cudaGetLastError(), "cuda reduceSumAxis0 kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_sum_axis1(b_lean_obj_arg BObj,
                                                                          uint32_t rows,
                                                                          uint32_t cols) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  const size_t R = (size_t)rows;
  const size_t C = (size_t)cols;
  const size_t total = checked_mul_size(R, C, "torchlean_cuda_buffer_reduce_sum_axis1: R*C overflow");
  if (b->size != total) {
    lean_internal_panic("torchlean_cuda_buffer_reduce_sum_axis1: size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(R);
  if (R == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  if (C == 0) {
    checkCuda(cudaMemset(out->data, 0, R * sizeof(float)), "cudaMemset axis1 out failed");
    return torchlean_cuda_buffer_box(out);
  }

  check_axis_grid_size(R, "torchlean_cuda_buffer_reduce_sum_axis1: rows exceed CUDA x-grid range");
  dim3 blocks = dim3((unsigned int)R);
  dim3 threads = dim3(kBlockSize);
  reduce_sum_axis1_f32<<<blocks, threads>>>(b->data, out->data, rows, cols);
  checkCuda(cudaGetLastError(), "cuda reduceSumAxis1 kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_max_axis0(b_lean_obj_arg BObj,
                                                                          uint32_t rows,
                                                                          uint32_t cols) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  const size_t R = (size_t)rows;
  const size_t C = (size_t)cols;
  const size_t total = checked_mul_size(R, C, "torchlean_cuda_buffer_reduce_max_axis0: R*C overflow");
  if (b->size != total) {
    lean_internal_panic("torchlean_cuda_buffer_reduce_max_axis0: size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(C);
  if (C == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  if (R == 0) {
    // Not expected in TorchLean WellFormed shapes, but totalize.
    checkCuda(cudaMemset(out->data, 0, C * sizeof(float)), "cudaMemset max axis0 out failed");
    return torchlean_cuda_buffer_box(out);
  }

  check_axis_grid_size(C, "torchlean_cuda_buffer_reduce_max_axis0: cols exceed CUDA x-grid range");
  dim3 blocks = dim3((unsigned int)C);
  dim3 threads = dim3(kBlockSize);
  reduce_max_axis0_f32<<<blocks, threads>>>(b->data, out->data, rows, cols);
  checkCuda(cudaGetLastError(), "cuda reduceMaxAxis0 kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_max_axis1(b_lean_obj_arg BObj,
                                                                          uint32_t rows,
                                                                          uint32_t cols) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  const size_t R = (size_t)rows;
  const size_t C = (size_t)cols;
  const size_t total = checked_mul_size(R, C, "torchlean_cuda_buffer_reduce_max_axis1: R*C overflow");
  if (b->size != total) {
    lean_internal_panic("torchlean_cuda_buffer_reduce_max_axis1: size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(R);
  if (R == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  if (C == 0) {
    // Not expected in TorchLean WellFormed shapes, but totalize.
    checkCuda(cudaMemset(out->data, 0, R * sizeof(float)), "cudaMemset max axis1 out failed");
    return torchlean_cuda_buffer_box(out);
  }

  check_axis_grid_size(R, "torchlean_cuda_buffer_reduce_max_axis1: rows exceed CUDA x-grid range");
  dim3 blocks = dim3((unsigned int)R);
  dim3 threads = dim3(kBlockSize);
  reduce_max_axis1_f32<<<blocks, threads>>>(b->data, out->data, rows, cols);
  checkCuda(cudaGetLastError(), "cuda reduceMaxAxis1 kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_concat1d(b_lean_obj_arg AObj,
                                                                  b_lean_obj_arg BObj,
                                                                  uint32_t n, uint32_t m) {
  torchlean_cuda_buffer* a = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  const size_t N = (size_t)n;
  const size_t M = (size_t)m;
  if (a->size != N) {
    lean_internal_panic("torchlean_cuda_buffer_concat1d: a.size mismatch");
  }
  if (b->size != M) {
    lean_internal_panic("torchlean_cuda_buffer_concat1d: b.size mismatch");
  }

  const size_t total = N + M;
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(total);
  if (total == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  dim3 blocks = blocks_for(total);
  dim3 threads = dim3(kBlockSize);
  concat1d_f32<<<blocks, threads>>>(a->data, N, b->data, M, out->data);
  checkCuda(cudaGetLastError(), "cuda concat1d kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_slice1d(b_lean_obj_arg BObj,
                                                                 uint32_t n, uint32_t start,
                                                                 uint32_t len) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  const size_t N = (size_t)n;
  const size_t S = (size_t)start;
  const size_t L = (size_t)len;
  if (b->size != N) {
    lean_internal_panic("torchlean_cuda_buffer_slice1d: size mismatch");
  }
  if (S > N || S + L > N) {
    lean_internal_panic("torchlean_cuda_buffer_slice1d: start+len out of bounds");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(L);
  if (L == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  dim3 blocks = blocks_for(L);
  dim3 threads = dim3(kBlockSize);
  slice1d_f32<<<blocks, threads>>>(b->data, S, L, out->data);
  checkCuda(cudaGetLastError(), "cuda slice1d kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_broadcast_vec_to_rows(b_lean_obj_arg VObj,
                                                                              uint32_t rows,
                                                                              uint32_t cols) {
  torchlean_cuda_buffer* v = torchlean_cuda_buffer_unbox(VObj);
  const size_t C = (size_t)cols;
  if (v->size != C) {
    lean_internal_panic("torchlean_cuda_buffer_broadcast_vec_to_rows: vec.size mismatch");
  }

  const size_t total = (size_t)rows * (size_t)cols;
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(total);
  if (total == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  if (cols == 0) {
    // total == 0 handled above.
    return torchlean_cuda_buffer_box(out);
  }

  dim3 blocks = blocks_for(total);
  dim3 threads = dim3(kBlockSize);
  broadcast_vec_to_rows_f32<<<blocks, threads>>>(v->data, rows, cols, out->data);
  checkCuda(cudaGetLastError(), "cuda broadcastVecToRows kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_broadcast_vec_to_cols(b_lean_obj_arg VObj,
                                                                              uint32_t rows,
                                                                              uint32_t cols) {
  torchlean_cuda_buffer* v = torchlean_cuda_buffer_unbox(VObj);
  const size_t R = (size_t)rows;
  if (v->size != R) {
    lean_internal_panic("torchlean_cuda_buffer_broadcast_vec_to_cols: vec.size mismatch");
  }

  const size_t total = (size_t)rows * (size_t)cols;
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(total);
  if (total == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  if (cols == 0) {
    // total == 0 handled above.
    return torchlean_cuda_buffer_box(out);
  }

  dim3 blocks = blocks_for(total);
  dim3 threads = dim3(kBlockSize);
  broadcast_vec_to_cols_f32<<<blocks, threads>>>(v->data, rows, cols, out->data);
  checkCuda(cudaGetLastError(), "cuda broadcastVecToCols kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_bmm(b_lean_obj_arg AObj,
                                                             b_lean_obj_arg BObj,
                                                             uint32_t batch, uint32_t m,
                                                             uint32_t n, uint32_t p) {
  torchlean_cuda_buffer* A = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* B = torchlean_cuda_buffer_unbox(BObj);

  const size_t Batch = (size_t)batch;
  const size_t M = (size_t)m;
  const size_t N = (size_t)n;
  const size_t P = (size_t)p;

  const size_t aSz = checked_mul3_size(Batch, M, N, "torchlean_cuda_buffer_bmm: A size overflow");
  const size_t bSz = checked_mul3_size(Batch, N, P, "torchlean_cuda_buffer_bmm: B size overflow");
  const size_t cSz = checked_mul3_size(Batch, M, P, "torchlean_cuda_buffer_bmm: C size overflow");

  if (A->size != aSz) {
    lean_internal_panic("torchlean_cuda_buffer_bmm: A.size mismatch");
  }
  if (B->size != bSz) {
    lean_internal_panic("torchlean_cuda_buffer_bmm: B.size mismatch");
  }

  torchlean_cuda_buffer* C = torchlean_cuda_buffer_alloc(cSz);
  if (cSz == 0) {
    return torchlean_cuda_buffer_box(C);
  }

  if (N == 0) {
    checkCuda(cudaMemset(C->data, 0, cSz * sizeof(float)), "cudaMemset bmm out failed");
    return torchlean_cuda_buffer_box(C);
  }

  if (m > (uint32_t)INT_MAX || n > (uint32_t)INT_MAX || p > (uint32_t)INT_MAX ||
      batch > (uint32_t)INT_MAX) {
    lean_internal_panic("torchlean_cuda_buffer_bmm: dims too large for cuBLAS int API");
  }

  cublasHandle_t handle = getCublasHandle();

  const float alpha = 1.0f;
  const float beta = 0.0f;

  // Row-major trick (same as TorchLean DGEMM): treat row-major buffers as transposed column-major.
  // Compute C^T (p x m) = B^T (p x n) * A^T (n x m), without explicit transposes.
  const long long int strideA = (long long int)(N * P);  // B batch stride (treated as A in cuBLAS)
  const long long int strideB = (long long int)(M * N);  // A batch stride (treated as B in cuBLAS)
  const long long int strideC = (long long int)(M * P);

  checkCublas(
      cublasSgemmStridedBatched(handle,
                               CUBLAS_OP_N, CUBLAS_OP_N,
                               (int)P, (int)M, (int)N,
                               &alpha,
                               B->data, (int)P, strideA,
                               A->data, (int)N, strideB,
                               &beta,
                               C->data, (int)P, strideC,
                               (int)batch),
      "cublasSgemmStridedBatched failed");

  return torchlean_cuda_buffer_box(C);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_rfft1d_packed(b_lean_obj_arg XObj,
                                                                        uint32_t batch,
                                                                        uint32_t n) {
  torchlean_cuda_buffer* x = torchlean_cuda_buffer_unbox(XObj);
  if (n == 0) {
    lean_internal_panic("torchlean_cuda_buffer_rfft1d_packed: n must be positive");
  }

  const size_t Batch = (size_t)batch;
  const size_t N = (size_t)n;
  const size_t Freq = N / 2 + 1;
  const size_t inSz =
      checked_mul_size(Batch, N, "torchlean_cuda_buffer_rfft1d_packed: input size overflow");
  const size_t complexSz =
      checked_mul_size(Batch, Freq, "torchlean_cuda_buffer_rfft1d_packed: spectrum size overflow");
  const size_t outSz =
      checked_mul_size(complexSz, (size_t)2, "torchlean_cuda_buffer_rfft1d_packed: packed size overflow");
  if (x->size != inSz) {
    lean_internal_panic("torchlean_cuda_buffer_rfft1d_packed: size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outSz);
  if (Batch == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  const int nInt = checked_cufft_int(n, "torchlean_cuda_buffer_rfft1d_packed: n exceeds cuFFT int range");
  const int batchInt =
      checked_cufft_int(batch, "torchlean_cuda_buffer_rfft1d_packed: batch exceeds cuFFT int range");
  const int freqInt =
      checked_cufft_int((uint32_t)Freq, "torchlean_cuda_buffer_rfft1d_packed: freq exceeds cuFFT int range");

  cufftComplex* dSpec = nullptr;
  if (cudaMalloc((void**)&dSpec, complexSz * sizeof(cufftComplex)) != cudaSuccess) {
    lean_internal_panic("torchlean_cuda_buffer_rfft1d_packed: cudaMalloc spectrum failed");
  }

  cufftHandle plan;
  int nPlan[1] = {nInt};
  int inembed[1] = {nInt};
  int onembed[1] = {freqInt};
  cufftResult planRes =
      cufftPlanMany(&plan, 1, nPlan, inembed, 1, nInt, onembed, 1, freqInt, CUFFT_R2C, batchInt);
  if (planRes != CUFFT_SUCCESS) {
    torchlean_cuda_free_checked((void**)&dSpec, "torchlean_cuda_buffer_rfft1d_packed: cleanup spectrum after plan failure failed");
    lean_internal_panic("torchlean_cuda_buffer_rfft1d_packed: cufftPlanMany R2C failed");
  }

  cufftResult execRes = cufftExecR2C(plan, (cufftReal*)x->data, dSpec);
  if (execRes != CUFFT_SUCCESS) {
    cufftDestroy(plan);
    torchlean_cuda_free_checked((void**)&dSpec, "torchlean_cuda_buffer_rfft1d_packed: cleanup spectrum after exec failure failed");
    lean_internal_panic("torchlean_cuda_buffer_rfft1d_packed: cufftExecR2C failed");
  }

  pack_cufft_complex_to_ri_f32<<<blocks_for(complexSz), dim3(kBlockSize)>>>(dSpec, out->data,
                                                                            complexSz);
  if (cudaGetLastError() != cudaSuccess) {
    cufftDestroy(plan);
    torchlean_cuda_free_checked((void**)&dSpec, "torchlean_cuda_buffer_rfft1d_packed: cleanup spectrum after pack failure failed");
    lean_internal_panic("torchlean_cuda_buffer_rfft1d_packed: pack kernel launch failed");
  }

  checkCufft(cufftDestroy(plan), "torchlean_cuda_buffer_rfft1d_packed: cufftDestroy failed");
  checkCuda(cudaFree(dSpec), "torchlean_cuda_buffer_rfft1d_packed: cudaFree spectrum failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_irfft1d_packed(b_lean_obj_arg SpecObj,
                                                                         uint32_t batch,
                                                                         uint32_t n) {
  torchlean_cuda_buffer* spec = torchlean_cuda_buffer_unbox(SpecObj);
  if (n == 0) {
    lean_internal_panic("torchlean_cuda_buffer_irfft1d_packed: n must be positive");
  }

  const size_t Batch = (size_t)batch;
  const size_t N = (size_t)n;
  const size_t Freq = N / 2 + 1;
  const size_t complexSz =
      checked_mul_size(Batch, Freq, "torchlean_cuda_buffer_irfft1d_packed: spectrum size overflow");
  const size_t specSz =
      checked_mul_size(complexSz, (size_t)2, "torchlean_cuda_buffer_irfft1d_packed: packed size overflow");
  const size_t outSz =
      checked_mul_size(Batch, N, "torchlean_cuda_buffer_irfft1d_packed: output size overflow");
  if (spec->size != specSz) {
    lean_internal_panic("torchlean_cuda_buffer_irfft1d_packed: size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outSz);
  if (Batch == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  const int nInt = checked_cufft_int(n, "torchlean_cuda_buffer_irfft1d_packed: n exceeds cuFFT int range");
  const int batchInt =
      checked_cufft_int(batch, "torchlean_cuda_buffer_irfft1d_packed: batch exceeds cuFFT int range");
  const int freqInt =
      checked_cufft_int((uint32_t)Freq, "torchlean_cuda_buffer_irfft1d_packed: freq exceeds cuFFT int range");

  cufftComplex* dSpec = nullptr;
  if (cudaMalloc((void**)&dSpec, complexSz * sizeof(cufftComplex)) != cudaSuccess) {
    lean_internal_panic("torchlean_cuda_buffer_irfft1d_packed: cudaMalloc spectrum failed");
  }

  pack_ri_to_cufft_complex_f32<<<blocks_for(complexSz), dim3(kBlockSize)>>>(spec->data, dSpec,
                                                                            complexSz);
  if (cudaGetLastError() != cudaSuccess) {
    torchlean_cuda_free_checked((void**)&dSpec, "torchlean_cuda_buffer_irfft1d_packed: cleanup spectrum after pack failure failed");
    lean_internal_panic("torchlean_cuda_buffer_irfft1d_packed: pack kernel launch failed");
  }

  cufftHandle plan;
  int nPlan[1] = {nInt};
  int inembed[1] = {freqInt};
  int onembed[1] = {nInt};
  cufftResult planRes =
      cufftPlanMany(&plan, 1, nPlan, inembed, 1, freqInt, onembed, 1, nInt, CUFFT_C2R, batchInt);
  if (planRes != CUFFT_SUCCESS) {
    torchlean_cuda_free_checked((void**)&dSpec, "torchlean_cuda_buffer_irfft1d_packed: cleanup spectrum after plan failure failed");
    lean_internal_panic("torchlean_cuda_buffer_irfft1d_packed: cufftPlanMany C2R failed");
  }

  cufftResult execRes = cufftExecC2R(plan, dSpec, (cufftReal*)out->data);
  if (execRes != CUFFT_SUCCESS) {
    cufftDestroy(plan);
    torchlean_cuda_free_checked((void**)&dSpec, "torchlean_cuda_buffer_irfft1d_packed: cleanup spectrum after exec failure failed");
    lean_internal_panic("torchlean_cuda_buffer_irfft1d_packed: cufftExecC2R failed");
  }

  scale_f32<<<blocks_for(outSz), dim3(kBlockSize)>>>(out->data, outSz, 1.0f / (float)N);
  if (cudaGetLastError() != cudaSuccess) {
    cufftDestroy(plan);
    torchlean_cuda_free_checked((void**)&dSpec, "torchlean_cuda_buffer_irfft1d_packed: cleanup spectrum after scale failure failed");
    lean_internal_panic("torchlean_cuda_buffer_irfft1d_packed: scale kernel launch failed");
  }

  checkCufft(cufftDestroy(plan), "torchlean_cuda_buffer_irfft1d_packed: cufftDestroy failed");
  checkCuda(cudaFree(dSpec), "torchlean_cuda_buffer_irfft1d_packed: cudaFree spectrum failed");
  return torchlean_cuda_buffer_box(out);
}

static inline void validate_spectral_conv1d_sizes(torchlean_cuda_buffer* x,
                                                  torchlean_cuda_buffer* wRe,
                                                  torchlean_cuda_buffer* wIm,
                                                  const char* who, uint32_t grid,
                                                  uint32_t width, uint32_t modes,
                                                  size_t* xSzOut, size_t* wSzOut,
                                                  size_t* freqOut) {
  if (grid == 0 || width == 0) {
    lean_internal_panic("spectralConv1dRfft: grid and width must be positive");
  }
  const size_t Freq = (size_t)grid / 2 + 1;
  if ((size_t)modes > Freq) {
    lean_internal_panic("spectralConv1dRfft: modes exceeds rfft frequency count");
  }
  const size_t xSz = checked_mul_size((size_t)grid, (size_t)width, who);
  const size_t wSz =
      checked_mul3_size((size_t)modes, (size_t)width, (size_t)width, who);
  if (x->size != xSz) {
    lean_internal_panic("spectralConv1dRfft: x size mismatch");
  }
  if (wRe->size != wSz || wIm->size != wSz) {
    lean_internal_panic("spectralConv1dRfft: weight size mismatch");
  }
  *xSzOut = xSz;
  *wSzOut = wSz;
  *freqOut = Freq;
}

static inline cufftHandle make_spectral_conv1d_r2c_plan(uint32_t grid, uint32_t width,
                                                        uint32_t freq) {
  const int nInt = checked_cufft_int(grid, "spectralConv1dRfft: grid exceeds cuFFT int range");
  const int widthInt =
      checked_cufft_int(width, "spectralConv1dRfft: width exceeds cuFFT int range");
  const int freqInt = checked_cufft_int(freq, "spectralConv1dRfft: freq exceeds cuFFT int range");
  cufftHandle plan;
  int nPlan[1] = {nInt};
  int inembed[1] = {nInt};
  int onembed[1] = {freqInt};
  checkCufft(cufftPlanMany(&plan, 1, nPlan, inembed, widthInt, 1, onembed, widthInt, 1,
                           CUFFT_R2C, widthInt),
             "spectralConv1dRfft: cufftPlanMany R2C failed");
  return plan;
}

static inline cufftHandle make_spectral_conv1d_c2r_plan(uint32_t grid, uint32_t width,
                                                        uint32_t freq) {
  const int nInt = checked_cufft_int(grid, "spectralConv1dRfft: grid exceeds cuFFT int range");
  const int widthInt =
      checked_cufft_int(width, "spectralConv1dRfft: width exceeds cuFFT int range");
  const int freqInt = checked_cufft_int(freq, "spectralConv1dRfft: freq exceeds cuFFT int range");
  cufftHandle plan;
  int nPlan[1] = {nInt};
  int inembed[1] = {freqInt};
  int onembed[1] = {nInt};
  checkCufft(cufftPlanMany(&plan, 1, nPlan, inembed, widthInt, 1, onembed, widthInt, 1,
                           CUFFT_C2R, widthInt),
             "spectralConv1dRfft: cufftPlanMany C2R failed");
  return plan;
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_spectral_conv1d_rfft_fwd(
    b_lean_obj_arg XObj, b_lean_obj_arg WReObj, b_lean_obj_arg WImObj, uint32_t grid,
    uint32_t width, uint32_t modes) {
  torchlean_cuda_buffer* x = torchlean_cuda_buffer_unbox(XObj);
  torchlean_cuda_buffer* wRe = torchlean_cuda_buffer_unbox(WReObj);
  torchlean_cuda_buffer* wIm = torchlean_cuda_buffer_unbox(WImObj);
  size_t xSz = 0, wSz = 0, Freq = 0;
  validate_spectral_conv1d_sizes(x, wRe, wIm, "spectralConv1dRfft fwd size overflow", grid,
                                 width, modes, &xSz, &wSz, &Freq);
  const size_t specSz = checked_mul_size(Freq, (size_t)width, "spectralConv1dRfft: spec overflow");
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(xSz);
  cufftComplex* X = nullptr;
  cufftComplex* Z = nullptr;
  checkCuda(cudaMalloc((void**)&X, specSz * sizeof(cufftComplex)), "spectralConv1dRfft: malloc X failed");
  checkCuda(cudaMalloc((void**)&Z, specSz * sizeof(cufftComplex)), "spectralConv1dRfft: malloc Z failed");
  checkCuda(cudaMemset(Z, 0, specSz * sizeof(cufftComplex)), "spectralConv1dRfft: memset Z failed");

  cufftHandle r2c = make_spectral_conv1d_r2c_plan(grid, width, (uint32_t)Freq);
  cufftHandle c2r = make_spectral_conv1d_c2r_plan(grid, width, (uint32_t)Freq);
  checkCufft(cufftExecR2C(r2c, (cufftReal*)x->data, X), "spectralConv1dRfft: R2C failed");
  spectral_conv1d_mul_f32<<<blocks_for((size_t)modes * (size_t)width), dim3(kBlockSize)>>>(
      X, wRe->data, wIm->data, Z, width, modes);
  checkCuda(cudaGetLastError(), "spectralConv1dRfft: multiply kernel failed");
  checkCufft(cufftExecC2R(c2r, Z, (cufftReal*)out->data), "spectralConv1dRfft: C2R failed");
  scale_f32<<<blocks_for(xSz), dim3(kBlockSize)>>>(out->data, xSz, 1.0f / (float)grid);
  checkCuda(cudaGetLastError(), "spectralConv1dRfft: scale kernel failed");
  checkCufft(cufftDestroy(r2c), "spectralConv1dRfft: destroy R2C failed");
  checkCufft(cufftDestroy(c2r), "spectralConv1dRfft: destroy C2R failed");
  checkCuda(cudaFree(X), "spectralConv1dRfft: free X failed");
  checkCuda(cudaFree(Z), "spectralConv1dRfft: free Z failed");
  return torchlean_cuda_buffer_box(out);
}

static inline void spectral_conv1d_make_x_dz(torchlean_cuda_buffer* x, torchlean_cuda_buffer* dY,
                                             uint32_t grid, uint32_t width, uint32_t modes,
                                             size_t specSz, cufftComplex** XOut,
                                             cufftComplex** dZOut) {
  cufftComplex* X = nullptr;
  cufftComplex* G = nullptr;
  cufftComplex* dZ = nullptr;
  checkCuda(cudaMalloc((void**)&X, specSz * sizeof(cufftComplex)), "spectralConv1dRfft bwd: malloc X failed");
  checkCuda(cudaMalloc((void**)&G, specSz * sizeof(cufftComplex)), "spectralConv1dRfft bwd: malloc G failed");
  checkCuda(cudaMalloc((void**)&dZ, specSz * sizeof(cufftComplex)), "spectralConv1dRfft bwd: malloc dZ failed");
  checkCuda(cudaMemset(dZ, 0, specSz * sizeof(cufftComplex)), "spectralConv1dRfft bwd: memset dZ failed");
  cufftHandle r2c = make_spectral_conv1d_r2c_plan(grid, width, grid / 2 + 1);
  checkCufft(cufftExecR2C(r2c, (cufftReal*)x->data, X), "spectralConv1dRfft bwd: R2C x failed");
  checkCufft(cufftExecR2C(r2c, (cufftReal*)dY->data, G), "spectralConv1dRfft bwd: R2C dY failed");
  spectral_conv1d_dz_from_dy_f32<<<blocks_for((size_t)modes * (size_t)width), dim3(kBlockSize)>>>(
      G, dZ, grid, width, modes);
  checkCuda(cudaGetLastError(), "spectralConv1dRfft bwd: dZ kernel failed");
  checkCufft(cufftDestroy(r2c), "spectralConv1dRfft bwd: destroy R2C failed");
  checkCuda(cudaFree(G), "spectralConv1dRfft bwd: free G failed");
  *XOut = X;
  *dZOut = dZ;
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_spectral_conv1d_rfft_bwd_x(
    b_lean_obj_arg XObj, b_lean_obj_arg WReObj, b_lean_obj_arg WImObj, b_lean_obj_arg DYObj,
    uint32_t grid, uint32_t width, uint32_t modes) {
  torchlean_cuda_buffer* x = torchlean_cuda_buffer_unbox(XObj);
  torchlean_cuda_buffer* wRe = torchlean_cuda_buffer_unbox(WReObj);
  torchlean_cuda_buffer* wIm = torchlean_cuda_buffer_unbox(WImObj);
  torchlean_cuda_buffer* dY = torchlean_cuda_buffer_unbox(DYObj);
  size_t xSz = 0, wSz = 0, Freq = 0;
  validate_spectral_conv1d_sizes(x, wRe, wIm, "spectralConv1dRfft bwd_x size overflow", grid,
                                 width, modes, &xSz, &wSz, &Freq);
  if (dY->size != xSz) lean_internal_panic("spectralConv1dRfft bwd_x: dY size mismatch");
  const size_t specSz = checked_mul_size(Freq, (size_t)width, "spectralConv1dRfft bwd_x: spec overflow");
  cufftComplex* X = nullptr;
  cufftComplex* dZ = nullptr;
  spectral_conv1d_make_x_dz(x, dY, grid, width, modes, specSz, &X, &dZ);
  cufftComplex* dX = nullptr;
  checkCuda(cudaMalloc((void**)&dX, specSz * sizeof(cufftComplex)), "spectralConv1dRfft bwd_x: malloc dX failed");
  checkCuda(cudaMemset(dX, 0, specSz * sizeof(cufftComplex)), "spectralConv1dRfft bwd_x: memset dX failed");
  spectral_conv1d_bwd_xspec_f32<<<blocks_for((size_t)modes * (size_t)width), dim3(kBlockSize)>>>(
      dZ, wRe->data, wIm->data, dX, width, modes);
  checkCuda(cudaGetLastError(), "spectralConv1dRfft bwd_x: dXspec kernel failed");
  torchlean_cuda_buffer* dx = torchlean_cuda_buffer_alloc(xSz);
  spectral_conv1d_irfft_adjoint_f32<<<blocks_for(xSz), dim3(kBlockSize)>>>(dX, dx->data, grid,
                                                                            width, modes);
  checkCuda(cudaGetLastError(), "spectralConv1dRfft bwd_x: irfft adjoint kernel failed");
  checkCuda(cudaFree(X), "spectralConv1dRfft bwd_x: free X failed");
  checkCuda(cudaFree(dZ), "spectralConv1dRfft bwd_x: free dZ failed");
  checkCuda(cudaFree(dX), "spectralConv1dRfft bwd_x: free dX failed");
  return torchlean_cuda_buffer_box(dx);
}

static inline lean_obj_res spectral_conv1d_bwd_weight_common(
    b_lean_obj_arg XObj, b_lean_obj_arg WReObj, b_lean_obj_arg WImObj, b_lean_obj_arg DYObj,
    uint32_t grid, uint32_t width, uint32_t modes, bool imag) {
  torchlean_cuda_buffer* x = torchlean_cuda_buffer_unbox(XObj);
  torchlean_cuda_buffer* wRe = torchlean_cuda_buffer_unbox(WReObj);
  torchlean_cuda_buffer* wIm = torchlean_cuda_buffer_unbox(WImObj);
  torchlean_cuda_buffer* dY = torchlean_cuda_buffer_unbox(DYObj);
  size_t xSz = 0, wSz = 0, Freq = 0;
  validate_spectral_conv1d_sizes(x, wRe, wIm, "spectralConv1dRfft bwd_w size overflow", grid,
                                 width, modes, &xSz, &wSz, &Freq);
  if (dY->size != xSz) lean_internal_panic("spectralConv1dRfft bwd_w: dY size mismatch");
  const size_t specSz = checked_mul_size(Freq, (size_t)width, "spectralConv1dRfft bwd_w: spec overflow");
  cufftComplex* X = nullptr;
  cufftComplex* dZ = nullptr;
  spectral_conv1d_make_x_dz(x, dY, grid, width, modes, specSz, &X, &dZ);
  torchlean_cuda_buffer* dWRe = torchlean_cuda_buffer_alloc(wSz);
  torchlean_cuda_buffer* dWIm = torchlean_cuda_buffer_alloc(wSz);
  spectral_conv1d_bwd_weights_f32<<<blocks_for(wSz), dim3(kBlockSize)>>>(X, dZ, dWRe->data,
                                                                          dWIm->data, width, modes);
  checkCuda(cudaGetLastError(), "spectralConv1dRfft bwd_w: weights kernel failed");
  checkCuda(cudaFree(X), "spectralConv1dRfft bwd_w: free X failed");
  checkCuda(cudaFree(dZ), "spectralConv1dRfft bwd_w: free dZ failed");
  torchlean_cuda_buffer* keep = imag ? dWIm : dWRe;
  torchlean_cuda_buffer* drop = imag ? dWRe : dWIm;
  if (drop->data != nullptr) {
    checkCuda(cudaFree(drop->data), "spectralConv1dRfft bwd_w: free unused weight grad data failed");
  }
  free(drop);
  return torchlean_cuda_buffer_box(keep);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_spectral_conv1d_rfft_bwd_wre(
    b_lean_obj_arg XObj, b_lean_obj_arg WReObj, b_lean_obj_arg WImObj, b_lean_obj_arg DYObj,
    uint32_t grid, uint32_t width, uint32_t modes) {
  return spectral_conv1d_bwd_weight_common(XObj, WReObj, WImObj, DYObj, grid, width, modes, false);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_spectral_conv1d_rfft_bwd_wim(
    b_lean_obj_arg XObj, b_lean_obj_arg WReObj, b_lean_obj_arg WImObj, b_lean_obj_arg DYObj,
    uint32_t grid, uint32_t width, uint32_t modes) {
  return spectral_conv1d_bwd_weight_common(XObj, WReObj, WImObj, DYObj, grid, width, modes, true);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_selective_scan_diag_fwd(
    b_lean_obj_arg AObj, b_lean_obj_arg BObj, b_lean_obj_arg XObj, b_lean_obj_arg H0Obj,
    uint32_t seqLen, uint32_t stateDim) {
  torchlean_cuda_buffer* A = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* B = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* X = torchlean_cuda_buffer_unbox(XObj);
  torchlean_cuda_buffer* h0 = torchlean_cuda_buffer_unbox(H0Obj);

  const size_t T = (size_t)seqLen;
  const size_t D = (size_t)stateDim;
  if (D != 0 && T > SIZE_MAX / D) {
    lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_fwd: seqLen*state overflow");
  }
  const size_t total = T * D;

  if (A->size != D) {
    lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_fwd: A.size mismatch");
  }
  if (B->size != D) {
    lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_fwd: B.size mismatch");
  }
  if (h0->size != D) {
    lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_fwd: h0.size mismatch");
  }
  if (X->size != total) {
    lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_fwd: X.size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(total);
  if (total == 0 || D == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  dim3 blocks = blocks_for(D);
  dim3 threads = dim3(kBlockSize);
  selective_scan_diag_fwd_f32<<<blocks, threads>>>(A->data, B->data, X->data, h0->data,
                                                  out->data, seqLen, stateDim);
  checkCuda(cudaGetLastError(), "cuda selectiveScanDiagFwd kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_selective_scan_diag_bwd(
    b_lean_obj_arg AObj, b_lean_obj_arg BObj, b_lean_obj_arg XObj, b_lean_obj_arg H0Obj,
    b_lean_obj_arg OutObj, b_lean_obj_arg DYObj, uint32_t seqLen, uint32_t stateDim) {
  torchlean_cuda_buffer* A = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* B = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* X = torchlean_cuda_buffer_unbox(XObj);
  torchlean_cuda_buffer* h0 = torchlean_cuda_buffer_unbox(H0Obj);
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_unbox(OutObj);
  torchlean_cuda_buffer* dY = torchlean_cuda_buffer_unbox(DYObj);

  const size_t T = (size_t)seqLen;
  const size_t D = (size_t)stateDim;
  if (D != 0 && T > SIZE_MAX / D) {
    lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_bwd: seqLen*state overflow");
  }
  const size_t total = T * D;

  if (A->size != D) lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_bwd: A.size mismatch");
  if (B->size != D) lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_bwd: B.size mismatch");
  if (h0->size != D) lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_bwd: h0.size mismatch");
  if (X->size != total) lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_bwd: X.size mismatch");
  if (out->size != total) lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_bwd: out.size mismatch");
  if (dY->size != total) lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_bwd: dY.size mismatch");

  torchlean_cuda_buffer* dA = torchlean_cuda_buffer_alloc(D);
  torchlean_cuda_buffer* dB = torchlean_cuda_buffer_alloc(D);
  torchlean_cuda_buffer* dX = torchlean_cuda_buffer_alloc(total);
  torchlean_cuda_buffer* dH0 = torchlean_cuda_buffer_alloc(D);

  if (total != 0 && D != 0) {
    dim3 blocks = blocks_for(D);
    dim3 threads = dim3(kBlockSize);
    selective_scan_diag_bwd_f32<<<blocks, threads>>>(A->data, B->data, X->data, h0->data,
                                                     out->data, dY->data, dA->data, dB->data,
                                                     dX->data, dH0->data, seqLen, stateDim);
    checkCuda(cudaGetLastError(), "cuda selectiveScanDiagBwd kernel launch failed");
  }

  lean_object* inner2 = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(inner2, 0, torchlean_cuda_buffer_box(dX));
  lean_ctor_set(inner2, 1, torchlean_cuda_buffer_box(dH0));
  lean_object* inner1 = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(inner1, 0, torchlean_cuda_buffer_box(dB));
  lean_ctor_set(inner1, 1, inner2);
  lean_object* outer = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(outer, 0, torchlean_cuda_buffer_box(dA));
  lean_ctor_set(outer, 1, inner1);
  return outer;
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_selective_scan_diag_var_fwd(
    b_lean_obj_arg AObj, b_lean_obj_arg BObj, b_lean_obj_arg XObj, b_lean_obj_arg H0Obj,
    uint32_t seqLen, uint32_t stateDim) {
  torchlean_cuda_buffer* A = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* B = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* X = torchlean_cuda_buffer_unbox(XObj);
  torchlean_cuda_buffer* h0 = torchlean_cuda_buffer_unbox(H0Obj);

  const size_t T = (size_t)seqLen;
  const size_t D = (size_t)stateDim;
  if (D != 0 && T > SIZE_MAX / D) {
    lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_var_fwd: seqLen*state overflow");
  }
  const size_t total = T * D;

  if (A->size != total) {
    lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_var_fwd: A.size mismatch");
  }
  if (B->size != total) {
    lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_var_fwd: B.size mismatch");
  }
  if (X->size != total) {
    lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_var_fwd: X.size mismatch");
  }
  if (h0->size != D) {
    lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_var_fwd: h0.size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(total);
  if (total == 0 || D == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  dim3 blocks = blocks_for(D);
  dim3 threads = dim3(kBlockSize);
  selective_scan_diag_var_fwd_f32<<<blocks, threads>>>(A->data, B->data, X->data, h0->data,
                                                      out->data, seqLen, stateDim);
  checkCuda(cudaGetLastError(), "cuda selectiveScanDiagVarFwd kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

// Fused attention contract.
//
// These kernels compute the same mathematical operation as TorchLean's proof-facing
// scaled-dot-product attention contract: scores = QK^T * scale, optional dense Boolean mask, stable
// row softmax, then multiplication by V. The implementation is intentionally straightforward and
// correctness-first: it recomputes row softmax statistics in backward rather than storing the full
// attention matrix. It is not the production IO-tiled FlashAttention schedule; the website and trust
// docs describe that distinction explicitly.
//
// Mask convention: a zero mask entry is treated as disallowed and receives the same large-negative
// score used by the Lean spec. Backward skips disallowed score entries to match the hard-mask runtime
// contract used by the fused kernel tests.
__device__ inline bool flash_attention_allowed(const float* mask, uint32_t hasMask,
                                               uint32_t batchIdx, uint32_t i, uint32_t j,
                                               uint32_t n) {
  if (hasMask == 0) return true;
  const size_t idx = ((size_t)batchIdx * (size_t)n + (size_t)i) * (size_t)n + (size_t)j;
  return mask[idx] != 0.0f;
}

__device__ inline float flash_attention_score(const float* Q, const float* K, const float* mask,
                                              uint32_t hasMask, uint32_t batchIdx, uint32_t i,
                                              uint32_t j, uint32_t n, uint32_t d, float scale) {
  if (!flash_attention_allowed(mask, hasMask, batchIdx, i, j, n)) {
    return -1000.0f;
  }
  float dot = 0.0f;
  const size_t qBase = ((size_t)batchIdx * (size_t)n + (size_t)i) * (size_t)d;
  const size_t kBase = ((size_t)batchIdx * (size_t)n + (size_t)j) * (size_t)d;
  for (uint32_t k = 0; k < d; ++k) {
    dot += Q[qBase + (size_t)k] * K[kBase + (size_t)k];
  }
  return dot * scale;
}

__device__ inline void flash_attention_row_stats(const float* Q, const float* K,
                                                 const float* mask, uint32_t hasMask,
                                                 uint32_t batchIdx, uint32_t i, uint32_t n,
                                                 uint32_t d, float scale, float* rowMax,
                                                 float* denom) {
  float m = -INFINITY;
  for (uint32_t j = 0; j < n; ++j) {
    float s = flash_attention_score(Q, K, mask, hasMask, batchIdx, i, j, n, d, scale);
    if (s > m) m = s;
  }

  float z = 0.0f;
  for (uint32_t j = 0; j < n; ++j) {
    float s = flash_attention_score(Q, K, mask, hasMask, batchIdx, i, j, n, d, scale);
    z += expf(s - m);
  }
  *rowMax = m;
  *denom = z;
}

__device__ inline float flash_attention_prob(const float* Q, const float* K, const float* mask,
                                             uint32_t hasMask, uint32_t batchIdx, uint32_t i,
                                             uint32_t j, uint32_t n, uint32_t d, float scale,
                                             float rowMax, float denom) {
  float s = flash_attention_score(Q, K, mask, hasMask, batchIdx, i, j, n, d, scale);
  return expf(s - rowMax) / denom;
}

__device__ inline float flash_attention_d_attn(const float* V, const float* dOut,
                                               uint32_t batchIdx, uint32_t i, uint32_t j,
                                               uint32_t n, uint32_t d) {
  float acc = 0.0f;
  const size_t dOutBase = ((size_t)batchIdx * (size_t)n + (size_t)i) * (size_t)d;
  const size_t vBase = ((size_t)batchIdx * (size_t)n + (size_t)j) * (size_t)d;
  for (uint32_t k = 0; k < d; ++k) {
    acc += dOut[dOutBase + (size_t)k] * V[vBase + (size_t)k];
  }
  return acc;
}

__global__ void flash_attention_fwd_f32(const float* Q, const float* K, const float* V,
                                        const float* mask, uint32_t hasMask, uint32_t batch,
                                        uint32_t n, uint32_t d, float scale, float* out) {
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t total = (size_t)batch * (size_t)n * (size_t)d;
  if (idx >= total) return;

  const uint32_t dv = (uint32_t)(idx % (size_t)d);
  const uint32_t i = (uint32_t)((idx / (size_t)d) % (size_t)n);
  const uint32_t b = (uint32_t)(idx / ((size_t)n * (size_t)d));

  float rowMax = 0.0f;
  float denom = 0.0f;
  flash_attention_row_stats(Q, K, mask, hasMask, b, i, n, d, scale, &rowMax, &denom);

  float acc = 0.0f;
  for (uint32_t j = 0; j < n; ++j) {
    const float p = flash_attention_prob(Q, K, mask, hasMask, b, i, j, n, d, scale, rowMax, denom);
    const size_t vIdx = ((size_t)b * (size_t)n + (size_t)j) * (size_t)d + (size_t)dv;
    acc += p * V[vIdx];
  }
  out[idx] = acc;
}

__global__ void flash_attention_bwd_q_f32(const float* Q, const float* K, const float* V,
                                          const float* mask, const float* dOut, uint32_t hasMask,
                                          uint32_t batch, uint32_t n, uint32_t d, float scale,
                                          float* dQ) {
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t total = (size_t)batch * (size_t)n * (size_t)d;
  if (idx >= total) return;

  const uint32_t k = (uint32_t)(idx % (size_t)d);
  const uint32_t i = (uint32_t)((idx / (size_t)d) % (size_t)n);
  const uint32_t b = (uint32_t)(idx / ((size_t)n * (size_t)d));

  float rowMax = 0.0f;
  float denom = 0.0f;
  flash_attention_row_stats(Q, K, mask, hasMask, b, i, n, d, scale, &rowMax, &denom);

  float rowDot = 0.0f;
  for (uint32_t j = 0; j < n; ++j) {
    const float p = flash_attention_prob(Q, K, mask, hasMask, b, i, j, n, d, scale, rowMax, denom);
    rowDot += p * flash_attention_d_attn(V, dOut, b, i, j, n, d);
  }

  float acc = 0.0f;
  for (uint32_t j = 0; j < n; ++j) {
    if (!flash_attention_allowed(mask, hasMask, b, i, j, n)) continue;
    const float p = flash_attention_prob(Q, K, mask, hasMask, b, i, j, n, d, scale, rowMax, denom);
    const float dAttn = flash_attention_d_attn(V, dOut, b, i, j, n, d);
    const float dScore = p * (dAttn - rowDot) * scale;
    const size_t kIdx = ((size_t)b * (size_t)n + (size_t)j) * (size_t)d + (size_t)k;
    acc += dScore * K[kIdx];
  }
  dQ[idx] = acc;
}

__global__ void flash_attention_bwd_k_f32(const float* Q, const float* K, const float* V,
                                          const float* mask, const float* dOut, uint32_t hasMask,
                                          uint32_t batch, uint32_t n, uint32_t d, float scale,
                                          float* dK) {
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t total = (size_t)batch * (size_t)n * (size_t)d;
  if (idx >= total) return;

  const uint32_t k = (uint32_t)(idx % (size_t)d);
  const uint32_t j = (uint32_t)((idx / (size_t)d) % (size_t)n);
  const uint32_t b = (uint32_t)(idx / ((size_t)n * (size_t)d));

  float acc = 0.0f;
  for (uint32_t i = 0; i < n; ++i) {
    float rowMax = 0.0f;
    float denom = 0.0f;
    flash_attention_row_stats(Q, K, mask, hasMask, b, i, n, d, scale, &rowMax, &denom);

    float rowDot = 0.0f;
    for (uint32_t t = 0; t < n; ++t) {
      const float p = flash_attention_prob(Q, K, mask, hasMask, b, i, t, n, d, scale, rowMax,
                                           denom);
      rowDot += p * flash_attention_d_attn(V, dOut, b, i, t, n, d);
    }

    if (!flash_attention_allowed(mask, hasMask, b, i, j, n)) continue;
    const float p = flash_attention_prob(Q, K, mask, hasMask, b, i, j, n, d, scale, rowMax, denom);
    const float dAttn = flash_attention_d_attn(V, dOut, b, i, j, n, d);
    const float dScore = p * (dAttn - rowDot) * scale;
    const size_t qIdx = ((size_t)b * (size_t)n + (size_t)i) * (size_t)d + (size_t)k;
    acc += dScore * Q[qIdx];
  }
  dK[idx] = acc;
}

__global__ void flash_attention_bwd_v_f32(const float* Q, const float* K, const float* V,
                                          const float* mask, const float* dOut, uint32_t hasMask,
                                          uint32_t batch, uint32_t n, uint32_t d, float scale,
                                          float* dV) {
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t total = (size_t)batch * (size_t)n * (size_t)d;
  if (idx >= total) return;

  const uint32_t dv = (uint32_t)(idx % (size_t)d);
  const uint32_t j = (uint32_t)((idx / (size_t)d) % (size_t)n);
  const uint32_t b = (uint32_t)(idx / ((size_t)n * (size_t)d));

  float acc = 0.0f;
  for (uint32_t i = 0; i < n; ++i) {
    float rowMax = 0.0f;
    float denom = 0.0f;
    flash_attention_row_stats(Q, K, mask, hasMask, b, i, n, d, scale, &rowMax, &denom);
    const float p = flash_attention_prob(Q, K, mask, hasMask, b, i, j, n, d, scale, rowMax, denom);
    const size_t dOutIdx = ((size_t)b * (size_t)n + (size_t)i) * (size_t)d + (size_t)dv;
    acc += p * dOut[dOutIdx];
  }
  dV[idx] = acc;
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_flash_attention_fwd(
    b_lean_obj_arg QObj, b_lean_obj_arg KObj, b_lean_obj_arg VObj, b_lean_obj_arg MaskObj,
    uint32_t hasMask, uint32_t batch, uint32_t n, uint32_t d, double scaleHost) {
  torchlean_cuda_buffer* Q = torchlean_cuda_buffer_unbox(QObj);
  torchlean_cuda_buffer* K = torchlean_cuda_buffer_unbox(KObj);
  torchlean_cuda_buffer* V = torchlean_cuda_buffer_unbox(VObj);
  torchlean_cuda_buffer* mask = torchlean_cuda_buffer_unbox(MaskObj);
  const size_t qkvSz = (size_t)batch * (size_t)n * (size_t)d;
  const size_t maskSz = (size_t)batch * (size_t)n * (size_t)n;
  if (Q->size != qkvSz || K->size != qkvSz || V->size != qkvSz) {
    lean_internal_panic("torchlean_cuda_buffer_flash_attention_fwd: Q/K/V size mismatch");
  }
  if (hasMask != 0 && mask->size != maskSz) {
    lean_internal_panic("torchlean_cuda_buffer_flash_attention_fwd: mask size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(qkvSz);
  if (qkvSz == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  dim3 blocks = blocks_for(qkvSz);
  dim3 threads = dim3(kBlockSize);
  flash_attention_fwd_f32<<<blocks, threads>>>(Q->data, K->data, V->data, mask->data, hasMask,
                                               batch, n, d, (float)scaleHost, out->data);
  checkCuda(cudaGetLastError(), "cuda flashAttention forward kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

#define TORCHLEAN_FLASH_BWD_EXPORT(name, kernel, label)                                           \
  extern "C" LEAN_EXPORT lean_obj_res name(                                                       \
      b_lean_obj_arg QObj, b_lean_obj_arg KObj, b_lean_obj_arg VObj, b_lean_obj_arg MaskObj,       \
      b_lean_obj_arg DOutObj, uint32_t hasMask, uint32_t batch, uint32_t n, uint32_t d,            \
      double scaleHost) {                                                                          \
    torchlean_cuda_buffer* Q = torchlean_cuda_buffer_unbox(QObj);                                  \
    torchlean_cuda_buffer* K = torchlean_cuda_buffer_unbox(KObj);                                  \
    torchlean_cuda_buffer* V = torchlean_cuda_buffer_unbox(VObj);                                  \
    torchlean_cuda_buffer* mask = torchlean_cuda_buffer_unbox(MaskObj);                            \
    torchlean_cuda_buffer* dOut = torchlean_cuda_buffer_unbox(DOutObj);                            \
    const size_t qkvSz = (size_t)batch * (size_t)n * (size_t)d;                                    \
    const size_t maskSz = (size_t)batch * (size_t)n * (size_t)n;                                   \
    if (Q->size != qkvSz || K->size != qkvSz || V->size != qkvSz || dOut->size != qkvSz) {         \
      lean_internal_panic(label ": Q/K/V/dOut size mismatch");                                    \
    }                                                                                              \
    if (hasMask != 0 && mask->size != maskSz) {                                                    \
      lean_internal_panic(label ": mask size mismatch");                                          \
    }                                                                                              \
    torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(qkvSz);                               \
    if (qkvSz == 0) return torchlean_cuda_buffer_box(out);                                         \
    dim3 blocks = blocks_for(qkvSz);                                                               \
    dim3 threads = dim3(kBlockSize);                                                               \
    kernel<<<blocks, threads>>>(Q->data, K->data, V->data, mask->data, dOut->data, hasMask, batch, \
                                n, d, (float)scaleHost, out->data);                                \
    checkCuda(cudaGetLastError(), label " kernel launch failed");                                 \
    return torchlean_cuda_buffer_box(out);                                                         \
  }

TORCHLEAN_FLASH_BWD_EXPORT(torchlean_cuda_buffer_flash_attention_bwd_q,
                           flash_attention_bwd_q_f32,
                           "cuda flashAttention backward Q")
TORCHLEAN_FLASH_BWD_EXPORT(torchlean_cuda_buffer_flash_attention_bwd_k,
                           flash_attention_bwd_k_f32,
                           "cuda flashAttention backward K")
TORCHLEAN_FLASH_BWD_EXPORT(torchlean_cuda_buffer_flash_attention_bwd_v,
                           flash_attention_bwd_v_f32,
                           "cuda flashAttention backward V")

#undef TORCHLEAN_FLASH_BWD_EXPORT

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_transpose2d(b_lean_obj_arg BObj,
                                                                     uint32_t rows, uint32_t cols) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  const size_t R = (size_t)rows;
  const size_t C = (size_t)cols;
  const size_t total = checked_mul_size(R, C, "torchlean_cuda_buffer_transpose2d: R*C overflow");
  if (b->size != total) {
    lean_internal_panic("torchlean_cuda_buffer_transpose2d: size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(total);
  if (total == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  dim3 blocks = blocks_for(total);
  dim3 threads = dim3(kBlockSize);
  transpose2d_f32<<<blocks, threads>>>(b->data, out->data, rows, cols);
  checkCuda(cudaGetLastError(), "cuda transpose2d kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_gather_vec(b_lean_obj_arg VObj,
                                                                    uint32_t n,
                                                                    b_lean_obj_arg IdxObj,
                                                                    uint32_t k) {
  torchlean_cuda_buffer* v = torchlean_cuda_buffer_unbox(VObj);
  const size_t N = (size_t)n;
  const size_t K = (size_t)k;
  if (v->size != N) {
    lean_internal_panic("torchlean_cuda_buffer_gather_vec: vec.size mismatch");
  }
  if (!lean_is_array((lean_object*)IdxObj)) {
    lean_internal_panic("torchlean_cuda_buffer_gather_vec: expected Array Nat indices");
  }
  if (lean_array_size(IdxObj) != K) {
    lean_internal_panic("torchlean_cuda_buffer_gather_vec: indices.size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(K);
  if (K == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  uint32_t* hIdx = (uint32_t*)malloc(K * sizeof(uint32_t));
  if (!hIdx) {
    lean_internal_panic_out_of_memory();
  }
  for (size_t j = 0; j < K; ++j) {
    b_lean_obj_res idxNat = lean_array_get_core(IdxObj, j);
    hIdx[j] = nat_to_u32_or_panic(idxNat, "torchlean_cuda_buffer_gather_vec: bad index Nat");
  }

  uint32_t* dIdx = nullptr;
  checkCuda(cudaMalloc((void**)&dIdx, K * sizeof(uint32_t)), "cudaMalloc indices failed");
  checkCuda(cudaMemcpy(dIdx, hIdx, K * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy indices failed");
  free(hIdx);

  dim3 blocks = blocks_for(K);
  dim3 threads = dim3(kBlockSize);
  gather_vec_f32<<<blocks, threads>>>(v->data, dIdx, K, n, out->data);
  checkCuda(cudaGetLastError(), "cuda gatherVec kernel launch failed");

  torchlean_cuda_free_checked((void**)&dIdx, "cudaFree gatherVec indices failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_scatter_add(b_lean_obj_arg XObj,
                                                                     b_lean_obj_arg ValuesObj,
                                                                     uint32_t n,
                                                                     b_lean_obj_arg IdxObj,
                                                                     uint32_t k) {
  torchlean_cuda_buffer* x = torchlean_cuda_buffer_unbox(XObj);
  torchlean_cuda_buffer* values = torchlean_cuda_buffer_unbox(ValuesObj);
  const size_t N = (size_t)n;
  const size_t K = (size_t)k;

  if (x->size != N) {
    lean_internal_panic("torchlean_cuda_buffer_scatter_add: x.size mismatch");
  }
  if (values->size != K) {
    lean_internal_panic("torchlean_cuda_buffer_scatter_add: values.size mismatch");
  }
  if (!lean_is_array((lean_object*)IdxObj)) {
    lean_internal_panic("torchlean_cuda_buffer_scatter_add: expected Array Nat indices");
  }
  if (lean_array_size(IdxObj) != K) {
    lean_internal_panic("torchlean_cuda_buffer_scatter_add: indices.size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(N);
  if (N == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  if (K == 0) {
    checkCuda(cudaMemcpy(out->data, x->data, N * sizeof(float), cudaMemcpyDeviceToDevice),
              "cudaMemcpy D2D scatterAdd base copy failed");
    return torchlean_cuda_buffer_box(out);
  }

  uint32_t* hIdx = (uint32_t*)malloc(K * sizeof(uint32_t));
  if (!hIdx) {
    lean_internal_panic_out_of_memory();
  }
  for (size_t j = 0; j < K; ++j) {
    b_lean_obj_res idxNat = lean_array_get_core(IdxObj, j);
    hIdx[j] = nat_to_u32_or_panic(idxNat, "torchlean_cuda_buffer_scatter_add: bad index Nat");
  }

  uint32_t* dIdx = nullptr;
  checkCuda(cudaMalloc((void**)&dIdx, K * sizeof(uint32_t)), "cudaMalloc indices failed");
  checkCuda(cudaMemcpy(dIdx, hIdx, K * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy indices failed");
  free(hIdx);

  dim3 threads = dim3(kBlockSize);
  if (torchlean_cuda_get_deterministic_reductions()) {
    // Deterministic (fixed-order) accumulation.
    dim3 blocks = blocks_for(N);
    scatter_add_det_f32<<<blocks, threads>>>(x->data, values->data, dIdx, K, n, out->data);
    checkCuda(cudaGetLastError(), "cuda scatterAdd deterministic kernel launch failed");
  } else {
    // Fast atomic path.
    checkCuda(cudaMemcpy(out->data, x->data, N * sizeof(float), cudaMemcpyDeviceToDevice),
              "cudaMemcpy D2D scatterAdd base copy failed");
    dim3 blocks = blocks_for(K);
    scatter_add_f32<<<blocks, threads>>>(values->data, dIdx, K, n, out->data);
    checkCuda(cudaGetLastError(), "cuda scatterAdd kernel launch failed");
  }

  torchlean_cuda_free_checked((void**)&dIdx, "cudaFree scatterAdd indices failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_broadcast_to(b_lean_obj_arg XObj,
                                                                      b_lean_obj_arg InDimsObj,
                                                                      b_lean_obj_arg OutDimsObj,
                                                                      b_lean_obj_arg AxisMapObj) {
  torchlean_cuda_buffer* x = torchlean_cuda_buffer_unbox(XObj);
  if (!lean_is_array((lean_object*)InDimsObj) || !lean_is_array((lean_object*)OutDimsObj) ||
      !lean_is_array((lean_object*)AxisMapObj)) {
    lean_internal_panic("torchlean_cuda_buffer_broadcast_to: expected Array Nat dims/map");
  }

  const size_t rankIn = lean_array_size(InDimsObj);
  const size_t rankOut = lean_array_size(OutDimsObj);
  if (lean_array_size(AxisMapObj) != rankOut) {
    lean_internal_panic("torchlean_cuda_buffer_broadcast_to: axisMap.size mismatch");
  }
  if (rankIn > (size_t)kMaxRank || rankOut > (size_t)kMaxRank) {
    lean_internal_panic("torchlean_cuda_buffer_broadcast_to: rank too large");
  }

  uint32_t* hInDims = rankIn == 0 ? nullptr : (uint32_t*)malloc(rankIn * sizeof(uint32_t));
  uint32_t* hOutDims = rankOut == 0 ? nullptr : (uint32_t*)malloc(rankOut * sizeof(uint32_t));
  uint32_t* hMap = rankOut == 0 ? nullptr : (uint32_t*)malloc(rankOut * sizeof(uint32_t));
  if ((rankIn != 0 && !hInDims) || (rankOut != 0 && (!hOutDims || !hMap))) {
    lean_internal_panic_out_of_memory();
  }

  size_t inSize = 1;
  for (size_t i = 0; i < rankIn; ++i) {
    b_lean_obj_res dNat = lean_array_get_core(InDimsObj, i);
    uint32_t d = nat_to_u32_or_oob(dNat);
    if (d == UINT32_MAX) {
      lean_internal_panic("torchlean_cuda_buffer_broadcast_to: inDims contains big Nat");
    }
    hInDims[i] = d;
    inSize *= (size_t)d;
  }
  size_t outSize = 1;
  for (size_t i = 0; i < rankOut; ++i) {
    b_lean_obj_res dNat = lean_array_get_core(OutDimsObj, i);
    uint32_t d = nat_to_u32_or_oob(dNat);
    if (d == UINT32_MAX) {
      lean_internal_panic("torchlean_cuda_buffer_broadcast_to: outDims contains big Nat");
    }
    hOutDims[i] = d;
    outSize *= (size_t)d;

    b_lean_obj_res mNat = lean_array_get_core(AxisMapObj, i);
    uint32_t mv =
        nat_to_u32_or_panic(mNat, "torchlean_cuda_buffer_broadcast_to: axisMap contains big Nat");
    hMap[i] = mv;
  }

  if (x->size != inSize) {
    lean_internal_panic("torchlean_cuda_buffer_broadcast_to: input size mismatch");
  }

  // Sanity-check broadcast compatibility (proof-driven, but keep a runtime guard).
  for (size_t ax = 0; ax < rankOut; ++ax) {
    uint32_t mv = hMap[ax];
    if (mv == 0) continue;
    size_t inAx = (size_t)(mv - 1);
    if (inAx >= rankIn) {
      lean_internal_panic("torchlean_cuda_buffer_broadcast_to: axisMap out of range");
    }
    uint32_t id = hInDims[inAx];
    uint32_t od = hOutDims[ax];
    if (!(id == 1 || id == od)) {
      lean_internal_panic("torchlean_cuda_buffer_broadcast_to: incompatible broadcast dims");
    }
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outSize);
  if (outSize == 0) {
    free(hInDims);
    free(hOutDims);
    free(hMap);
    return torchlean_cuda_buffer_box(out);
  }

  uint32_t* dInDims = nullptr;
  uint32_t* dOutDims = nullptr;
  uint32_t* dMap = nullptr;
  if (rankIn != 0) {
    checkCuda(cudaMalloc((void**)&dInDims, rankIn * sizeof(uint32_t)), "cudaMalloc inDims failed");
    checkCuda(cudaMemcpy(dInDims, hInDims, rankIn * sizeof(uint32_t), cudaMemcpyHostToDevice),
              "cudaMemcpy inDims failed");
  }
  if (rankOut != 0) {
    checkCuda(cudaMalloc((void**)&dOutDims, rankOut * sizeof(uint32_t)), "cudaMalloc outDims failed");
    checkCuda(cudaMalloc((void**)&dMap, rankOut * sizeof(uint32_t)), "cudaMalloc axisMap failed");
    checkCuda(cudaMemcpy(dOutDims, hOutDims, rankOut * sizeof(uint32_t), cudaMemcpyHostToDevice),
              "cudaMemcpy outDims failed");
    checkCuda(cudaMemcpy(dMap, hMap, rankOut * sizeof(uint32_t), cudaMemcpyHostToDevice),
              "cudaMemcpy axisMap failed");
  }
  free(hInDims);
  free(hOutDims);
  free(hMap);

  dim3 blocks = blocks_for(outSize);
  dim3 threads = dim3(kBlockSize);
  broadcast_to_f32<<<blocks, threads>>>(x->data, out->data, outSize,
                                       dInDims, (int)rankIn,
                                       dOutDims, (int)rankOut,
                                       dMap);
  checkCuda(cudaGetLastError(), "cuda broadcastTo kernel launch failed");

  torchlean_cuda_free_checked((void**)&dInDims, "cudaFree broadcastTo inDims failed");
  torchlean_cuda_free_checked((void**)&dOutDims, "cudaFree broadcastTo outDims failed");
  torchlean_cuda_free_checked((void**)&dMap, "cudaFree broadcastTo axisMap failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_from_broadcast(
    b_lean_obj_arg DOutObj,
    b_lean_obj_arg InDimsObj,
    b_lean_obj_arg OutDimsObj,
    b_lean_obj_arg AxisMapObj) {
  torchlean_cuda_buffer* dOut = torchlean_cuda_buffer_unbox(DOutObj);
  if (!lean_is_array((lean_object*)InDimsObj) || !lean_is_array((lean_object*)OutDimsObj) ||
      !lean_is_array((lean_object*)AxisMapObj)) {
    lean_internal_panic("torchlean_cuda_buffer_reduce_from_broadcast: expected Array Nat dims/map");
  }

  const size_t rankIn = lean_array_size(InDimsObj);
  const size_t rankOut = lean_array_size(OutDimsObj);
  if (lean_array_size(AxisMapObj) != rankOut) {
    lean_internal_panic("torchlean_cuda_buffer_reduce_from_broadcast: axisMap.size mismatch");
  }
  if (rankIn > (size_t)kMaxRank || rankOut > (size_t)kMaxRank) {
    lean_internal_panic("torchlean_cuda_buffer_reduce_from_broadcast: rank too large");
  }

  uint32_t* hInDims = rankIn == 0 ? nullptr : (uint32_t*)malloc(rankIn * sizeof(uint32_t));
  uint32_t* hOutDims = rankOut == 0 ? nullptr : (uint32_t*)malloc(rankOut * sizeof(uint32_t));
  uint32_t* hMap = rankOut == 0 ? nullptr : (uint32_t*)malloc(rankOut * sizeof(uint32_t));
  if ((rankIn != 0 && !hInDims) || (rankOut != 0 && (!hOutDims || !hMap))) {
    lean_internal_panic_out_of_memory();
  }

  size_t inSize = 1;
  for (size_t i = 0; i < rankIn; ++i) {
    uint32_t d = nat_to_u32_or_oob(lean_array_get_core(InDimsObj, i));
    if (d == UINT32_MAX) {
      lean_internal_panic("torchlean_cuda_buffer_reduce_from_broadcast: inDims contains big Nat");
    }
    hInDims[i] = d;
    inSize *= (size_t)d;
  }
  size_t outSize = 1;
  for (size_t i = 0; i < rankOut; ++i) {
    uint32_t d = nat_to_u32_or_oob(lean_array_get_core(OutDimsObj, i));
    if (d == UINT32_MAX) {
      lean_internal_panic("torchlean_cuda_buffer_reduce_from_broadcast: outDims contains big Nat");
    }
    hOutDims[i] = d;
    outSize *= (size_t)d;

    uint32_t mv =
        nat_to_u32_or_panic(lean_array_get_core(AxisMapObj, i),
                            "torchlean_cuda_buffer_reduce_from_broadcast: axisMap contains big Nat");
    hMap[i] = mv;
  }

  if (dOut->size != outSize) {
    lean_internal_panic("torchlean_cuda_buffer_reduce_from_broadcast: dOut size mismatch");
  }

  // Sanity-check broadcast compatibility.
  for (size_t ax = 0; ax < rankOut; ++ax) {
    uint32_t mv = hMap[ax];
    if (mv == 0) continue;
    size_t inAx = (size_t)(mv - 1);
    if (inAx >= rankIn) {
      lean_internal_panic("torchlean_cuda_buffer_reduce_from_broadcast: axisMap out of range");
    }
    uint32_t id = hInDims[inAx];
    uint32_t od = hOutDims[ax];
    if (!(id == 1 || id == od)) {
      lean_internal_panic("torchlean_cuda_buffer_reduce_from_broadcast: incompatible broadcast dims");
    }
  }

  torchlean_cuda_buffer* dIn = torchlean_cuda_buffer_alloc(inSize);
  if (inSize == 0) {
    free(hInDims);
    free(hOutDims);
    free(hMap);
    return torchlean_cuda_buffer_box(dIn);
  }
  if (outSize == 0) {
    // Totalize empty output gradients as the all-zeros input gradient.
    checkCuda(cudaMemset(dIn->data, 0, inSize * sizeof(float)),
              "cudaMemset reduceFromBroadcast failed");
    free(hInDims);
    free(hOutDims);
    free(hMap);
    return torchlean_cuda_buffer_box(dIn);
  }

  uint32_t* dInDims = nullptr;
  uint32_t* dOutDims = nullptr;
  uint32_t* dMap = nullptr;
  if (rankIn != 0) {
    checkCuda(cudaMalloc((void**)&dInDims, rankIn * sizeof(uint32_t)), "cudaMalloc inDims failed");
    checkCuda(cudaMemcpy(dInDims, hInDims, rankIn * sizeof(uint32_t), cudaMemcpyHostToDevice),
              "cudaMemcpy inDims failed");
  }
  if (rankOut != 0) {
    checkCuda(cudaMalloc((void**)&dOutDims, rankOut * sizeof(uint32_t)), "cudaMalloc outDims failed");
    checkCuda(cudaMalloc((void**)&dMap, rankOut * sizeof(uint32_t)), "cudaMalloc axisMap failed");
    checkCuda(cudaMemcpy(dOutDims, hOutDims, rankOut * sizeof(uint32_t), cudaMemcpyHostToDevice),
              "cudaMemcpy outDims failed");
    checkCuda(cudaMemcpy(dMap, hMap, rankOut * sizeof(uint32_t), cudaMemcpyHostToDevice),
              "cudaMemcpy axisMap failed");
  }
  free(hInDims);
  free(hOutDims);
  free(hMap);

  dim3 threads = dim3(kBlockSize);
  if (torchlean_cuda_get_deterministic_reductions()) {
    // Deterministic: fixed-order reduction into each `dIn[inIdx]`.
    dim3 blocks = blocks_for(inSize);
    reduce_from_broadcast_det_f32<<<blocks, threads>>>(dOut->data, outSize,
                                                     dIn->data, inSize,
                                                     dInDims, (int)rankIn,
                                                     dOutDims, (int)rankOut,
                                                     dMap);
    checkCuda(cudaGetLastError(), "cuda reduceFromBroadcast deterministic kernel launch failed");
  } else {
    // Fast atomic accumulation.
    checkCuda(cudaMemset(dIn->data, 0, inSize * sizeof(float)),
              "cudaMemset reduceFromBroadcast failed");
    dim3 blocks = blocks_for(outSize);
    reduce_from_broadcast_f32<<<blocks, threads>>>(dOut->data, outSize,
                                                 dIn->data,
                                                 dInDims, (int)rankIn,
                                                 dOutDims, (int)rankOut,
                                                 dMap);
    checkCuda(cudaGetLastError(), "cuda reduceFromBroadcast kernel launch failed");
  }

  torchlean_cuda_free_checked((void**)&dInDims, "cudaFree reduceFromBroadcast inDims failed");
  torchlean_cuda_free_checked((void**)&dOutDims, "cudaFree reduceFromBroadcast outDims failed");
  torchlean_cuda_free_checked((void**)&dMap, "cudaFree reduceFromBroadcast axisMap failed");
  return torchlean_cuda_buffer_box(dIn);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_swap_adjacent_at_depth(
    b_lean_obj_arg XObj, b_lean_obj_arg DimsObj, uint32_t depth) {
  torchlean_cuda_buffer* x = torchlean_cuda_buffer_unbox(XObj);
  if (!lean_is_array((lean_object*)DimsObj)) {
    lean_internal_panic("torchlean_cuda_buffer_swap_adjacent_at_depth: expected Array Nat dims");
  }
  const size_t rank = lean_array_size(DimsObj);
  if (rank < 2 || (size_t)depth + 1 >= rank) {
    lean_internal_panic("torchlean_cuda_buffer_swap_adjacent_at_depth: invalid depth for rank");
  }
  if (rank > (size_t)kMaxRank) {
    lean_internal_panic("torchlean_cuda_buffer_swap_adjacent_at_depth: rank too large");
  }

  uint32_t hDims[kMaxRank];
  size_t total = 1;
  for (size_t i = 0; i < rank; ++i) {
    uint32_t d = nat_to_u32_or_oob(lean_array_get_core(DimsObj, i));
    if (d == UINT32_MAX) {
      lean_internal_panic("torchlean_cuda_buffer_swap_adjacent_at_depth: dims contains big Nat");
    }
    hDims[i] = d;
    total *= (size_t)d;
  }
  if (x->size != total) {
    lean_internal_panic("torchlean_cuda_buffer_swap_adjacent_at_depth: input size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(total);
  if (total == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  uint32_t* dDims = nullptr;
  checkCuda(cudaMalloc((void**)&dDims, rank * sizeof(uint32_t)), "cudaMalloc dims failed");
  checkCuda(cudaMemcpy(dDims, hDims, rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy dims failed");

  dim3 blocks = blocks_for(total);
  dim3 threads = dim3(kBlockSize);
  swap_adjacent_at_depth_f32<<<blocks, threads>>>(x->data, out->data, total, dDims, (int)rank,
                                                  (int)depth);
  checkCuda(cudaGetLastError(), "cuda swapAdjacentAtDepth kernel launch failed");

  torchlean_cuda_free_checked((void**)&dDims, "cudaFree swapAdjacentAtDepth dims failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_sum_axis(
    b_lean_obj_arg XObj, b_lean_obj_arg DimsObj, uint32_t axis) {
  torchlean_cuda_buffer* x = torchlean_cuda_buffer_unbox(XObj);
  if (!lean_is_array((lean_object*)DimsObj)) {
    lean_internal_panic("torchlean_cuda_buffer_reduce_sum_axis: expected Array Nat dims");
  }
  const size_t rank = lean_array_size(DimsObj);
  if (rank == 0) {
    // Scalar: identity.
    torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(1);
    checkCuda(cudaMemcpy(out->data, x->data, sizeof(float), cudaMemcpyDeviceToDevice),
              "cudaMemcpy reduceSumAxis scalar failed");
    return torchlean_cuda_buffer_box(out);
  }
  if ((size_t)axis >= rank) {
    lean_internal_panic("torchlean_cuda_buffer_reduce_sum_axis: invalid axis");
  }
  if (rank > (size_t)kMaxRank) {
    lean_internal_panic("torchlean_cuda_buffer_reduce_sum_axis: rank too large");
  }

  uint32_t hDims[kMaxRank];
  size_t inSize = 1;
  for (size_t i = 0; i < rank; ++i) {
    uint32_t d = nat_to_u32_or_oob(lean_array_get_core(DimsObj, i));
    if (d == UINT32_MAX) {
      lean_internal_panic("torchlean_cuda_buffer_reduce_sum_axis: dims contains big Nat");
    }
    hDims[i] = d;
    inSize *= (size_t)d;
  }
  if (x->size != inSize) {
    lean_internal_panic("torchlean_cuda_buffer_reduce_sum_axis: input size mismatch");
  }

  size_t outSize = 1;
  for (size_t i = 0; i < rank; ++i) {
    if (i != (size_t)axis) {
      outSize *= (size_t)hDims[i];
    }
  }
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outSize);
  if (outSize == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  if (inSize == 0) {
    checkCuda(cudaMemset(out->data, 0, outSize * sizeof(float)),
              "cudaMemset reduceSumAxis empty reduction failed");
    return torchlean_cuda_buffer_box(out);
  }

  uint32_t* dDims = nullptr;
  checkCuda(cudaMalloc((void**)&dDims, rank * sizeof(uint32_t)), "cudaMalloc dims failed");
  checkCuda(cudaMemcpy(dDims, hDims, rank * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy dims failed");

  dim3 threads = dim3(kBlockSize);
  if (torchlean_cuda_get_deterministic_reductions()) {
    // Deterministic: fixed-order loop over the reduced axis for each output element.
    dim3 blocks = blocks_for(outSize);
    reduce_sum_axis_det_f32<<<blocks, threads>>>(x->data, out->data, outSize, dDims, (int)rank,
                                                 (int)axis);
    checkCuda(cudaGetLastError(), "cuda reduceSumAxis deterministic kernel launch failed");
  } else {
    // Fast atomic accumulation.
    checkCuda(cudaMemset(out->data, 0, outSize * sizeof(float)),
              "cudaMemset reduceSumAxis out failed");
    dim3 blocks = blocks_for(inSize);
    reduce_sum_axis_f32<<<blocks, threads>>>(x->data, out->data, inSize, dDims, (int)rank,
                                             (int)axis);
    checkCuda(cudaGetLastError(), "cuda reduceSumAxis kernel launch failed");
  }

  torchlean_cuda_free_checked((void**)&dDims, "cudaFree reduceSumAxis dims failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_gather_rows(b_lean_obj_arg MObj,
                                                                     uint32_t rows, uint32_t cols,
                                                                     b_lean_obj_arg IdxObj,
                                                                     uint32_t k) {
  torchlean_cuda_buffer* m = torchlean_cuda_buffer_unbox(MObj);
  const size_t R = (size_t)rows;
  const size_t C = (size_t)cols;
  const size_t K = (size_t)k;
  const size_t matSz = checked_mul_size(R, C, "torchlean_cuda_buffer_gather_rows: rows*cols overflow");
  const size_t outSz = checked_mul_size(K, C, "torchlean_cuda_buffer_gather_rows: k*cols overflow");
  if (m->size != matSz) {
    lean_internal_panic("torchlean_cuda_buffer_gather_rows: mat.size mismatch");
  }
  if (!lean_is_array((lean_object*)IdxObj)) {
    lean_internal_panic("torchlean_cuda_buffer_gather_rows: expected Array Nat indices");
  }
  if (lean_array_size(IdxObj) != K) {
    lean_internal_panic("torchlean_cuda_buffer_gather_rows: indices.size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outSz);
  if (K == 0 || C == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  uint32_t* hIdx = (uint32_t*)malloc(K * sizeof(uint32_t));
  if (!hIdx) {
    lean_internal_panic_out_of_memory();
  }
  for (size_t j = 0; j < K; ++j) {
    hIdx[j] =
        nat_to_u32_or_panic(lean_array_get_core(IdxObj, j),
                            "torchlean_cuda_buffer_gather_rows: bad index Nat");
  }

  uint32_t* dIdx = nullptr;
  checkCuda(cudaMalloc((void**)&dIdx, K * sizeof(uint32_t)), "cudaMalloc indices failed");
  checkCuda(cudaMemcpy(dIdx, hIdx, K * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy indices failed");
  free(hIdx);

  dim3 blocks = blocks_for(outSz);
  dim3 threads = dim3(kBlockSize);
  gather_rows_f32<<<blocks, threads>>>(m->data, rows, cols, dIdx, k, out->data);
  checkCuda(cudaGetLastError(), "cuda gatherRows kernel launch failed");

  torchlean_cuda_free_checked((void**)&dIdx, "cudaFree gatherScalar indices failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_scatter_add_row(b_lean_obj_arg MObj,
                                                                         b_lean_obj_arg RowObj,
                                                                         uint32_t rows, uint32_t cols,
                                                                         uint32_t i) {
  torchlean_cuda_buffer* m = torchlean_cuda_buffer_unbox(MObj);
  torchlean_cuda_buffer* row = torchlean_cuda_buffer_unbox(RowObj);
  const size_t R = (size_t)rows;
  const size_t C = (size_t)cols;
  const size_t matSz =
      checked_mul_size(R, C, "torchlean_cuda_buffer_scatter_add_row: rows*cols overflow");
  if (m->size != matSz) {
    lean_internal_panic("torchlean_cuda_buffer_scatter_add_row: mat.size mismatch");
  }
  if (row->size != C) {
    lean_internal_panic("torchlean_cuda_buffer_scatter_add_row: rowVec.size mismatch");
  }
  if ((size_t)i >= R) {
    lean_internal_panic("torchlean_cuda_buffer_scatter_add_row: row index out of bounds");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(matSz);
  if (matSz == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  dim3 blocks = blocks_for(matSz);
  dim3 threads = dim3(kBlockSize);
  scatter_add_row_f32<<<blocks, threads>>>(m->data, row->data, rows, cols, i, out->data);
  checkCuda(cudaGetLastError(), "cuda scatterAddRow kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_scatter_add_rows(
    b_lean_obj_arg MObj,
    b_lean_obj_arg ValuesObj,
    uint32_t rows, uint32_t cols,
    b_lean_obj_arg IdxObj,
    uint32_t k) {
  torchlean_cuda_buffer* m = torchlean_cuda_buffer_unbox(MObj);
  torchlean_cuda_buffer* values = torchlean_cuda_buffer_unbox(ValuesObj);
  const size_t R = (size_t)rows;
  const size_t C = (size_t)cols;
  const size_t K = (size_t)k;
  const size_t matSz =
      checked_mul_size(R, C, "torchlean_cuda_buffer_scatter_add_rows: rows*cols overflow");
  const size_t valuesSz =
      checked_mul_size(K, C, "torchlean_cuda_buffer_scatter_add_rows: k*cols overflow");
  if (m->size != matSz) {
    lean_internal_panic("torchlean_cuda_buffer_scatter_add_rows: mat.size mismatch");
  }
  if (values->size != valuesSz) {
    lean_internal_panic("torchlean_cuda_buffer_scatter_add_rows: values.size mismatch");
  }
  if (!lean_is_array((lean_object*)IdxObj)) {
    lean_internal_panic("torchlean_cuda_buffer_scatter_add_rows: expected Array Nat indices");
  }
  if (lean_array_size(IdxObj) != K) {
    lean_internal_panic("torchlean_cuda_buffer_scatter_add_rows: indices.size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(matSz);
  if (matSz == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  if (K == 0 || C == 0) {
    checkCuda(cudaMemcpy(out->data, m->data, matSz * sizeof(float), cudaMemcpyDeviceToDevice),
              "cudaMemcpy D2D scatterAddRows base copy failed");
    return torchlean_cuda_buffer_box(out);
  }

  uint32_t* hIdx = (uint32_t*)malloc(K * sizeof(uint32_t));
  if (!hIdx) {
    lean_internal_panic_out_of_memory();
  }
  for (size_t j = 0; j < K; ++j) {
    hIdx[j] =
        nat_to_u32_or_panic(lean_array_get_core(IdxObj, j),
                            "torchlean_cuda_buffer_scatter_add_rows: bad index Nat");
  }
  uint32_t* dIdx = nullptr;
  checkCuda(cudaMalloc((void**)&dIdx, K * sizeof(uint32_t)), "cudaMalloc indices failed");
  checkCuda(cudaMemcpy(dIdx, hIdx, K * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy indices failed");
  free(hIdx);

  dim3 threads = dim3(kBlockSize);
  if (torchlean_cuda_get_deterministic_reductions()) {
    // Deterministic (fixed-order) accumulation.
    dim3 blocks = blocks_for(matSz);
    scatter_add_rows_det_f32<<<blocks, threads>>>(m->data, values->data, dIdx, k, rows, cols,
                                                  out->data);
    checkCuda(cudaGetLastError(), "cuda scatterAddRows deterministic kernel launch failed");
  } else {
    // Fast atomic path.
    checkCuda(cudaMemcpy(out->data, m->data, matSz * sizeof(float), cudaMemcpyDeviceToDevice),
              "cudaMemcpy D2D scatterAddRows base copy failed");
    dim3 blocks = blocks_for(valuesSz);
    scatter_add_rows_f32<<<blocks, threads>>>(values->data, dIdx, k, rows, cols, out->data);
    checkCuda(cudaGetLastError(), "cuda scatterAddRows kernel launch failed");
  }

  torchlean_cuda_free_checked((void**)&dIdx, "cudaFree scatterAddRow indices failed");
  return torchlean_cuda_buffer_box(out);
}
