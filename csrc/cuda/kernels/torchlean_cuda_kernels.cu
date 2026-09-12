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

// CUDA kernels above the raw buffer ops: reductions, broadcasting, gather/scatter, batched matmul,
// selective scan, FFT helpers, and the simple attention path.
// Everything is flat row-major; wrappers check ranks and shapes before launch.

static constexpr int kBlockSize = 256;
static constexpr int kMaxRank = 8;
static constexpr int kColumnTileCols = 32;
static constexpr int kColumnTileRows = kBlockSize / kColumnTileCols;
static constexpr uint32_t kColumnRowsPerTile = 64;
static constexpr size_t kMaxCudaGridY = 65535;
static_assert(kBlockSize > 0 && (kBlockSize & (kBlockSize - 1)) == 0,
              "kBlockSize must remain a power of two for shared-memory reductions");
static_assert(kColumnTileCols * kColumnTileRows == kBlockSize,
              "column-reduction tile must use one full CUDA block");

extern "C" void torchlean_cuda_kernels_flush_scratch_cache(void) {
  torchlean_cuda_scratch_flush();
}

static inline dim3 blocks_for(size_t n) {
  size_t blocks = (n + (size_t)kBlockSize - 1) / (size_t)kBlockSize;
  if (blocks == 0) blocks = 1;
  if (blocks > 2147483647ULL) blocks = 2147483647ULL;
  return dim3((unsigned int)blocks);
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

struct HostBroadcastArrays {
  uint32_t* inDims;
  uint32_t* outDims;
  uint32_t* axisMap;
};

static inline HostBroadcastArrays alloc_host_broadcast_arrays(size_t rankIn, size_t rankOut) {
  const size_t inBytes = checked_bytes_size(
      rankIn, sizeof(uint32_t), "alloc_host_broadcast_arrays: inDims byte size overflow");
  const size_t outBytes = checked_bytes_size(
      rankOut, sizeof(uint32_t), "alloc_host_broadcast_arrays: outDims byte size overflow");
  HostBroadcastArrays h = {
      rankIn == 0 ? nullptr : (uint32_t*)malloc(inBytes),
      rankOut == 0 ? nullptr : (uint32_t*)malloc(outBytes),
      rankOut == 0 ? nullptr : (uint32_t*)malloc(outBytes),
  };
  if ((rankIn != 0 && !h.inDims) || (rankOut != 0 && (!h.outDims || !h.axisMap))) {
    free(h.inDims);
    free(h.outDims);
    free(h.axisMap);
    lean_internal_panic_out_of_memory();
  }
  return h;
}

static inline void free_host_broadcast_arrays(HostBroadcastArrays* h) {
  if (!h) {
    return;
  }
  free(h->inDims);
  free(h->outDims);
  free(h->axisMap);
  h->inDims = nullptr;
  h->outDims = nullptr;
  h->axisMap = nullptr;
}

static inline uint32_t* upload_nat_indices(
    b_lean_obj_arg idxObj,
    size_t count,
    const char* natMsg,
    const char* mallocMsg,
    const char* memcpyMsg) {
  const size_t bytes =
      checked_bytes_size(count, sizeof(uint32_t), "upload_nat_indices: byte size overflow");
  uint32_t* hIdx = count == 0 ? nullptr : (uint32_t*)malloc(bytes);
  if (count != 0 && !hIdx) {
    lean_internal_panic_out_of_memory();
  }
  for (size_t j = 0; j < count; ++j) {
    hIdx[j] = nat_to_u32_or_panic(lean_array_get_core(idxObj, j), natMsg);
  }

  uint32_t* dIdx = torchlean_cuda_scratch_alloc<uint32_t>(count, mallocMsg);
  cudaError_t copyErr = cudaMemcpy(dIdx, hIdx, bytes, cudaMemcpyHostToDevice);
  free(hIdx);
  if (copyErr != cudaSuccess) {
    torchlean_cuda_scratch_free(&dIdx, count, "cudaFree indices after upload failure failed");
    checkCuda(copyErr, memcpyMsg);
  }
  return dIdx;
}

__global__ void pack_cufft_complex_to_ri_f32(const cufftComplex* in, float* out, size_t count) {
  const size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (i >= count) return;
  out[2 * i] = in[i].x;
  out[2 * i + 1] = in[i].y;
}

__global__ void pack_ri_to_cufft_complex_f32(
    const float* in, cufftComplex* out, size_t count, uint32_t n) {
  const size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (i >= count) return;
  const size_t k = i % ((size_t)n / 2 + 1);
  const bool endpoint = k == 0 || ((n % 2 == 0) && k == (size_t)n / 2);
  out[i].x = in[2 * i];
  // C2R requires real DC/Nyquist bins; match the host inverse's ignored coordinates.
  out[i].y = endpoint ? 0.0f : in[2 * i + 1];
}

__global__ void scale_f32(float* data, size_t n, float scale) {
  const size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (i >= n) return;
  data[i] *= scale;
}

__global__ void spectral_conv1d_mul_f32(const cufftComplex* X, const float* wRe,
                                        const float* wIm, cufftComplex* Z, uint32_t width,
                                        uint32_t modes, uint32_t grid) {
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
  // Complex weights can introduce endpoint imaginary values even for real FFT inputs.
  const bool endpoint = k == 0 || ((grid % 2 == 0) && k == grid / 2);
  Z[(size_t)k * (size_t)width + (size_t)o].y = endpoint ? 0.0f : zi;
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
  dZ[(size_t)k * (size_t)width + (size_t)o].y = edge ? 0.0f : scale * g.y;
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

// C2R supplies both members of each interior conjugate pair. The packed-real adjoint
// needs each stored coordinate only once, so halve interior bins before the unnormalized
// inverse. Endpoint imaginary coordinates are absent from the real transform.
__global__ void spectral_conv1d_prepare_rfft_adjoint_f32(cufftComplex* dX,
                                                         uint32_t grid, uint32_t width) {
  const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t total = ((size_t)grid / 2 + 1) * (size_t)width;
  if (idx >= total) return;
  const size_t k = idx / (size_t)width;
  const bool endpoint = k == 0 || ((grid % 2 == 0) && k == (size_t)grid / 2);
  if (endpoint) {
    dX[idx].y = 0.0f;
  } else {
    dX[idx].x *= 0.5f;
    dX[idx].y *= 0.5f;
  }
}

__global__ void reduce_sum_by_column_partial_f32(const float* in, float* partialOut,
                                                 uint32_t rows, uint32_t cols,
                                                 uint32_t rowsPerTile) {
  if (blockDim.x != kColumnTileCols || blockDim.y != kColumnTileRows) return;
  const uint32_t col =
      (uint32_t)blockIdx.x * (uint32_t)kColumnTileCols + (uint32_t)threadIdx.x;
  const uint32_t rowLane = (uint32_t)threadIdx.y;
  const size_t rowBegin = (size_t)blockIdx.y * (size_t)rowsPerTile;
  const size_t rowLimit = rowBegin + (size_t)rowsPerTile;
  const size_t rowEnd = rowLimit < (size_t)rows ? rowLimit : (size_t)rows;
  __shared__ float tileSums[kColumnTileRows][kColumnTileCols];
  float sum = 0.0f;
  if (col < cols) {
    for (size_t r = rowBegin + (size_t)rowLane; r < rowEnd;
         r += (size_t)kColumnTileRows) {
      sum += in[(size_t)r * (size_t)cols + (size_t)col];
    }
  }
  tileSums[rowLane][threadIdx.x] = sum;
  __syncthreads();

  if (rowLane == 0 && col < cols) {
    float total = tileSums[0][threadIdx.x];
    for (int lane = 1; lane < kColumnTileRows; ++lane) {
      total += tileSums[lane][threadIdx.x];
    }
    partialOut[(size_t)col * (size_t)gridDim.y + (size_t)blockIdx.y] = total;
  }
}

__global__ void reduce_sum_by_column_final_f32(const float* partial, float* out,
                                               uint32_t partialRows, uint32_t cols) {
  if (blockDim.x != kBlockSize) return;
  const uint32_t col = (uint32_t)blockIdx.x;
  if (col >= cols) return;

  __shared__ float sums[kBlockSize];
  const uint32_t tid = (uint32_t)threadIdx.x;
  float total = 0.0f;
  for (uint32_t r = tid; r < partialRows; r += (uint32_t)kBlockSize) {
    total += partial[(size_t)col * (size_t)partialRows + (size_t)r];
  }
  sums[tid] = total;
  __syncthreads();

  for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < (uint32_t)stride) {
      sums[tid] += sums[tid + (uint32_t)stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[col] = sums[0];
  }
}

static inline void launch_reduce_sum_by_column(const float* in, float* out,
                                               uint32_t rows, uint32_t cols,
                                               const char* gridError,
                                               const char* launchError) {
  if (cols == 0) return;
  if (rows == 0) {
    checkCuda(cudaMemset(out, 0, (size_t)cols * sizeof(float)), launchError);
    return;
  }
  const size_t columnTiles =
      ((size_t)cols + (size_t)kColumnTileCols - 1) / (size_t)kColumnTileCols;
  // The partial kernel uses `columnTiles` blocks, while the final kernel uses one block per
  // output column. Checking the larger launch bounds both grids.
  check_axis_grid_size((size_t)cols, gridError);

  size_t rowTiles =
      ((size_t)rows + (size_t)kColumnRowsPerTile - 1) / (size_t)kColumnRowsPerTile;
  if (rowTiles > kMaxCudaGridY) {
    rowTiles = kMaxCudaGridY;
  }
  const size_t rowsPerTileSize =
      ((size_t)rows + rowTiles - 1) / rowTiles;
  if (rowsPerTileSize > (size_t)UINT32_MAX) {
    lean_internal_panic("column reduction row tile exceeds UInt32");
  }
  const uint32_t rowsPerTile = (uint32_t)rowsPerTileSize;
  const size_t partialCount =
      checked_mul_size(rowTiles, (size_t)cols, "column reduction partial size overflow");
  float* partial =
      torchlean_cuda_scratch_alloc<float>(partialCount, "cudaMalloc column partials failed");

  reduce_sum_by_column_partial_f32<<<
      dim3((unsigned int)columnTiles, (unsigned int)rowTiles),
      dim3(kColumnTileCols, kColumnTileRows)>>>(
          in, partial, rows, cols, rowsPerTile);
  checkCuda(cudaGetLastError(), launchError);

  reduce_sum_by_column_final_f32<<<dim3((unsigned int)cols), dim3(kBlockSize)>>>(
      partial, out, (uint32_t)rowTiles, cols);
  checkCuda(cudaGetLastError(), launchError);
  torchlean_cuda_scratch_free(
      &partial, partialCount, "cudaFree column partials failed");
}

__global__ void reduce_sum_by_row_f32(const float* in, float* out, uint32_t rows, uint32_t cols) {
  if (blockDim.x != kBlockSize) return;
  const uint32_t row = (uint32_t)blockIdx.x;
  if (row >= rows) return;

  __shared__ float sdata[kBlockSize];
  const int tid = threadIdx.x;

  float sum = 0.0f;
  for (size_t c = (size_t)tid; c < (size_t)cols; c += (size_t)blockDim.x) {
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

__global__ void layer_norm_fwd_f32(
    const float* x,
    const float* gamma,
    const float* beta,
    float* out,
    float* normalized,
    float* invStdOut,
    uint32_t rows,
    uint32_t cols,
    double epsilon) {
  if (blockDim.x != kBlockSize) return;
  const uint32_t row = (uint32_t)blockIdx.x;
  if (row >= rows) return;

  // A row can have a large common offset and a small spread, even when every input is
  // exactly representable in float. Accumulate both passes in double and keep the mean
  // in double through subtraction: rounding it back to float would erase that spread.
  __shared__ double sums[kBlockSize];
  __shared__ double mean;
  __shared__ double std;
  const uint32_t tid = (uint32_t)threadIdx.x;
  const size_t rowOffset = (size_t)row * (size_t)cols;
  const double columnCount = (double)cols;

  double total = 0.0;
  for (size_t col = (size_t)tid; col < (size_t)cols; col += (size_t)kBlockSize) {
    total = __dadd_rn(total, (double)x[rowOffset + (size_t)col]);
  }
  sums[tid] = total;
  __syncthreads();
  for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < (uint32_t)stride) {
      sums[tid] = __dadd_rn(sums[tid], sums[tid + (uint32_t)stride]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    mean = __ddiv_rn(sums[0], columnCount);
  }
  __syncthreads();

  double squareTotal = 0.0;
  for (size_t col = (size_t)tid; col < (size_t)cols; col += (size_t)kBlockSize) {
    const double centered = __dsub_rn((double)x[rowOffset + (size_t)col], mean);
    squareTotal = __dadd_rn(squareTotal, __dmul_rn(centered, centered));
  }
  sums[tid] = squareTotal;
  __syncthreads();
  for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < (uint32_t)stride) {
      sums[tid] = __dadd_rn(sums[tid], sums[tid + (uint32_t)stride]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    const double variance = __ddiv_rn(sums[0], columnCount);
    std = __dsqrt_rn(__dadd_rn(variance, epsilon));
    invStdOut[row] = __double2float_rn(__ddiv_rn(1.0, std));
  }
  __syncthreads();

  for (size_t col = (size_t)tid; col < (size_t)cols; col += (size_t)kBlockSize) {
    const size_t index = rowOffset + (size_t)col;
    // Only the completed normalized coordinate is rounded to float. Backward retains
    // its existing float xhat and inverse-standard-deviation buffers.
    const double centered = __dsub_rn((double)x[index], mean);
    const float xHat = __double2float_rn(__ddiv_rn(centered, std));
    normalized[index] = xHat;
    out[index] = __fadd_rn(__fmul_rn(xHat, gamma[col]), beta[col]);
  }
}

__global__ void layer_norm_bwd_rows_f32(
    const float* dOut,
    const float* normalized,
    const float* invStd,
    const float* gamma,
    float* dX,
    float* dGammaPointwise,
    uint32_t rows,
    uint32_t cols,
    float colsScale,
    float invCols) {
  if (blockDim.x != kBlockSize) return;
  const uint32_t row = (uint32_t)blockIdx.x;
  if (row >= rows) return;

  __shared__ float dXhatSums[kBlockSize];
  __shared__ float dXhatXhatSums[kBlockSize];
  const uint32_t tid = (uint32_t)threadIdx.x;
  const size_t rowOffset = (size_t)row * (size_t)cols;

  float dXhatTotal = 0.0f;
  float dXhatXhatTotal = 0.0f;
  for (size_t col = (size_t)tid; col < (size_t)cols; col += (size_t)kBlockSize) {
    const size_t index = rowOffset + (size_t)col;
    const float dXhat = __fmul_rn(dOut[index], gamma[col]);
    dXhatTotal = __fadd_rn(dXhatTotal, dXhat);
    dXhatXhatTotal =
        __fadd_rn(dXhatXhatTotal, __fmul_rn(dXhat, normalized[index]));
    dGammaPointwise[index] = __fmul_rn(dOut[index], normalized[index]);
  }
  dXhatSums[tid] = dXhatTotal;
  dXhatXhatSums[tid] = dXhatXhatTotal;
  __syncthreads();
  for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < (uint32_t)stride) {
      dXhatSums[tid] =
          __fadd_rn(dXhatSums[tid], dXhatSums[tid + (uint32_t)stride]);
      dXhatXhatSums[tid] =
          __fadd_rn(dXhatXhatSums[tid], dXhatXhatSums[tid + (uint32_t)stride]);
    }
    __syncthreads();
  }

  const float sumDXhat = dXhatSums[0];
  const float sumDXhatXhat = dXhatXhatSums[0];
  const float rowInvStd = invStd[row];
  for (size_t col = (size_t)tid; col < (size_t)cols; col += (size_t)kBlockSize) {
    const size_t index = rowOffset + (size_t)col;
    const float dXhat = __fmul_rn(dOut[index], gamma[col]);
    const float scaledDXhat = __fmul_rn(dXhat, colsScale);
    const float centeredDXhat = __fsub_rn(scaledDXhat, sumDXhat);
    const float xHatSum =
        __fmul_rn(normalized[index], sumDXhatXhat);
    const float term = __fsub_rn(centeredDXhat, xHatSum);
    const float termInv = __fmul_rn(term, rowInvStd);
    dX[index] = __fmul_rn(termInv, invCols);
  }
}

__global__ void reduce_max_by_column_f32(const float* in, float* out, uint32_t rows, uint32_t cols) {
  if (blockDim.x != kBlockSize) return;
  const uint32_t col = (uint32_t)blockIdx.x;
  if (col >= cols) return;

  __shared__ float sdata[kBlockSize];
  const int tid = threadIdx.x;

  float m = -INFINITY;
  for (size_t r = (size_t)tid; r < (size_t)rows; r += (size_t)blockDim.x) {
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

  float acc = mat[t];
  for (uint32_t j = 0; j < k; ++j) {
    if (idx[(size_t)j] == r) {
      acc += values[(size_t)j * (size_t)cols + (size_t)c];
    }
  }

  out[t] = acc;
}

__global__ void reduce_max_by_row_f32(const float* in, float* out, uint32_t rows, uint32_t cols) {
  if (blockDim.x != kBlockSize) return;
  const uint32_t row = (uint32_t)blockIdx.x;
  if (row >= rows) return;

  __shared__ float sdata[kBlockSize];
  const int tid = threadIdx.x;

  float m = -INFINITY;
  for (size_t c = (size_t)tid; c < (size_t)cols; c += (size_t)blockDim.x) {
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

__global__ void hard_masked_softmax_by_row_f32(const float* scores, const float* mask, float* out,
                                                uint32_t rows, uint32_t cols) {
  if (blockDim.x != kBlockSize) return;
  const uint32_t row = (uint32_t)blockIdx.x;
  if (row >= rows) return;

  __shared__ float sdata[kBlockSize];
  const int tid = threadIdx.x;
  const size_t base = (size_t)row * (size_t)cols;

  float m = -INFINITY;
  for (size_t c = (size_t)tid; c < (size_t)cols; c += (size_t)blockDim.x) {
    if (mask[base + (size_t)c] != 0.0f) {
      m = fmaxf(m, scores[base + (size_t)c]);
    }
  }
  sdata[tid] = m;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
  const float rowMax = sdata[0];
  // Every thread must load the maximum before the shared buffer is reused for the sum reduction.
  __syncthreads();

  float z = 0.0f;
  if (rowMax != -INFINITY) {
    for (size_t c = (size_t)tid; c < (size_t)cols; c += (size_t)blockDim.x) {
      if (mask[base + (size_t)c] != 0.0f) {
        z += expf(scores[base + (size_t)c] - rowMax);
      }
    }
  }
  sdata[tid] = z;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  const float denom = sdata[0];

  for (size_t c = (size_t)tid; c < (size_t)cols; c += (size_t)blockDim.x) {
    const size_t idx = base + (size_t)c;
    out[idx] = (mask[idx] != 0.0f && denom != 0.0f)
        ? expf(scores[idx] - rowMax) / denom
        : 0.0f;
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

  float acc = x[i64];
  for (size_t j = 0; j < k; ++j) {
    if (idx[j] == i) {
      acc += values[j];
    }
  }

  out[i64] = acc;
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

// Reverse time within each independent state channel. A coefficient at time t multiplies
// h[t-1], so it is A[t], rather than A[t-1], that carries this step's cotangent backwards.
__global__ void selective_scan_diag_var_bwd_f32(
    const float* A, const float* B, const float* X, const float* h0,
    const float* out, const float* dY, float* dA, float* dB, float* dX, float* dH0,
    uint32_t seqLen, uint32_t stateDim) {
  const size_t j = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (j >= (size_t)stateDim) return;
  float carried = 0.0f;
  for (uint32_t remaining = seqLen; remaining > 0; --remaining) {
    const size_t t = (size_t)remaining - 1;
    const size_t idx = t * (size_t)stateDim + j;
    const float previous = t == 0 ? h0[j] : out[idx - (size_t)stateDim];
    const float gradient = dY[idx] + carried;
    dA[idx] = gradient * previous;
    dB[idx] = gradient * X[idx];
    dX[idx] = gradient * B[idx];
    carried = gradient * A[idx];
  }
  // This also initializes the initial-state cotangent when the sequence is empty.
  dH0[j] = carried;
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_sum_by_column(b_lean_obj_arg BObj,
                                                                          uint32_t rows,
                                                                          uint32_t cols) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  const size_t R = (size_t)rows;
  const size_t C = (size_t)cols;
  const size_t total = checked_mul_size(R, C, "torchlean_cuda_buffer_reduce_sum_by_column: R*C overflow");
  if (b->size != total) {
    lean_internal_panic("torchlean_cuda_buffer_reduce_sum_by_column: size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(C);
  if (C == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  if (R == 0) {
    checkCuda(cudaMemset(out->data, 0, C * sizeof(float)), "cudaMemset reduceSumByColumn out failed");
    return torchlean_cuda_buffer_box(out);
  }

  launch_reduce_sum_by_column(
      b->data, out->data, rows, cols,
      "torchlean_cuda_buffer_reduce_sum_by_column: column tiles exceed CUDA x-grid range",
      "cuda reduceSumByColumn kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_sum_by_row(b_lean_obj_arg BObj,
                                                                          uint32_t rows,
                                                                          uint32_t cols) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  const size_t R = (size_t)rows;
  const size_t C = (size_t)cols;
  const size_t total = checked_mul_size(R, C, "torchlean_cuda_buffer_reduce_sum_by_row: R*C overflow");
  if (b->size != total) {
    lean_internal_panic("torchlean_cuda_buffer_reduce_sum_by_row: size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(R);
  if (R == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  if (C == 0) {
    checkCuda(cudaMemset(out->data, 0, R * sizeof(float)), "cudaMemset reduceSumByRow out failed");
    return torchlean_cuda_buffer_box(out);
  }

  check_axis_grid_size(R, "torchlean_cuda_buffer_reduce_sum_by_row: rows exceed CUDA x-grid range");
  dim3 blocks = dim3((unsigned int)R);
  dim3 threads = dim3(kBlockSize);
  reduce_sum_by_row_f32<<<blocks, threads>>>(b->data, out->data, rows, cols);
  checkCuda(cudaGetLastError(), "cuda reduceSumByRow kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_max_by_column(b_lean_obj_arg BObj,
                                                                          uint32_t rows,
                                                                          uint32_t cols) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  const size_t R = (size_t)rows;
  const size_t C = (size_t)cols;
  const size_t total = checked_mul_size(R, C, "torchlean_cuda_buffer_reduce_max_by_column: R*C overflow");
  if (b->size != total) {
    lean_internal_panic("torchlean_cuda_buffer_reduce_max_by_column: size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(C);
  if (C == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  if (R == 0) {
    // Not expected in TorchLean WellFormed shapes, but totalize.
    checkCuda(cudaMemset(out->data, 0, C * sizeof(float)), "cudaMemset reduceMaxByColumn out failed");
    return torchlean_cuda_buffer_box(out);
  }

  check_axis_grid_size(C, "torchlean_cuda_buffer_reduce_max_by_column: cols exceed CUDA x-grid range");
  dim3 blocks = dim3((unsigned int)C);
  dim3 threads = dim3(kBlockSize);
  reduce_max_by_column_f32<<<blocks, threads>>>(b->data, out->data, rows, cols);
  checkCuda(cudaGetLastError(), "cuda reduceMaxByColumn kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_max_by_row(b_lean_obj_arg BObj,
                                                                          uint32_t rows,
                                                                          uint32_t cols) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  const size_t R = (size_t)rows;
  const size_t C = (size_t)cols;
  const size_t total = checked_mul_size(R, C, "torchlean_cuda_buffer_reduce_max_by_row: R*C overflow");
  if (b->size != total) {
    lean_internal_panic("torchlean_cuda_buffer_reduce_max_by_row: size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(R);
  if (R == 0) {
    return torchlean_cuda_buffer_box(out);
  }
  if (C == 0) {
    // Not expected in TorchLean WellFormed shapes, but totalize.
    checkCuda(cudaMemset(out->data, 0, R * sizeof(float)), "cudaMemset reduceMaxByRow out failed");
    return torchlean_cuda_buffer_box(out);
  }

  check_axis_grid_size(R, "torchlean_cuda_buffer_reduce_max_by_row: rows exceed CUDA x-grid range");
  dim3 blocks = dim3((unsigned int)R);
  dim3 threads = dim3(kBlockSize);
  reduce_max_by_row_f32<<<blocks, threads>>>(b->data, out->data, rows, cols);
  checkCuda(cudaGetLastError(), "cuda reduceMaxByRow kernel launch failed");
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_hard_masked_softmax_by_row(
    b_lean_obj_arg ScoresObj, b_lean_obj_arg MaskObj, uint32_t rows, uint32_t cols) {
  torchlean_cuda_buffer* scores = torchlean_cuda_buffer_unbox(ScoresObj);
  torchlean_cuda_buffer* mask = torchlean_cuda_buffer_unbox(MaskObj);
  const size_t R = (size_t)rows;
  const size_t C = (size_t)cols;
  const size_t total = checked_mul_size(
      R, C, "torchlean_cuda_buffer_hard_masked_softmax_by_row: R*C overflow");
  if (scores->size != total || mask->size != total) {
    lean_internal_panic("torchlean_cuda_buffer_hard_masked_softmax_by_row: size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(total);
  if (R == 0 || C == 0) return torchlean_cuda_buffer_box(out);

  check_axis_grid_size(
      R, "torchlean_cuda_buffer_hard_masked_softmax_by_row: rows exceed CUDA x-grid range");
  hard_masked_softmax_by_row_f32<<<dim3((unsigned int)R), dim3(kBlockSize)>>>(
      scores->data, mask->data, out->data, rows, cols);
  checkCuda(cudaGetLastError(), "cuda hardMaskedSoftmaxByRow kernel launch failed");
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

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_layer_norm_fwd(
    b_lean_obj_arg XObj,
    b_lean_obj_arg GammaObj,
    b_lean_obj_arg BetaObj,
    uint32_t rows,
    uint32_t cols,
    double invCols,
    double epsilon) {
  // Keep the FFI signature stable, but derive the forward divisor from integer cols.
  // Widening a pre-rounded reciprocal would retain its error in a large row mean.
  (void)invCols;
  torchlean_cuda_buffer* x = torchlean_cuda_buffer_unbox(XObj);
  torchlean_cuda_buffer* gamma = torchlean_cuda_buffer_unbox(GammaObj);
  torchlean_cuda_buffer* beta = torchlean_cuda_buffer_unbox(BetaObj);
  if (rows == 0 || cols == 0) {
    lean_internal_panic("torchlean_cuda_buffer_layer_norm_fwd: dimensions must be positive");
  }
  const size_t total = checked_mul_size(
      (size_t)rows, (size_t)cols, "torchlean_cuda_buffer_layer_norm_fwd: size overflow");
  if (x->size != total || gamma->size != (size_t)cols || beta->size != (size_t)cols) {
    lean_internal_panic("torchlean_cuda_buffer_layer_norm_fwd: buffer size mismatch");
  }
  check_axis_grid_size(
      (size_t)rows, "torchlean_cuda_buffer_layer_norm_fwd: rows exceed CUDA x-grid range");

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(total);
  torchlean_cuda_buffer* normalized = torchlean_cuda_buffer_alloc(total);
  torchlean_cuda_buffer* invStd = torchlean_cuda_buffer_alloc((size_t)rows);
  layer_norm_fwd_f32<<<dim3((unsigned int)rows), dim3(kBlockSize)>>>(
      x->data,
      gamma->data,
      beta->data,
      out->data,
      normalized->data,
      invStd->data,
      rows,
      cols,
      epsilon);
  checkCuda(cudaGetLastError(), "cuda layerNorm forward kernel launch failed");
  return torchlean_cuda_box_three_buffers(out, normalized, invStd);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_layer_norm_bwd(
    b_lean_obj_arg DOutObj,
    b_lean_obj_arg NormalizedObj,
    b_lean_obj_arg InvStdObj,
    b_lean_obj_arg GammaObj,
    uint32_t rows,
    uint32_t cols,
    double colsScale,
    double invCols) {
  torchlean_cuda_buffer* dOut = torchlean_cuda_buffer_unbox(DOutObj);
  torchlean_cuda_buffer* normalized = torchlean_cuda_buffer_unbox(NormalizedObj);
  torchlean_cuda_buffer* invStd = torchlean_cuda_buffer_unbox(InvStdObj);
  torchlean_cuda_buffer* gamma = torchlean_cuda_buffer_unbox(GammaObj);
  if (rows == 0 || cols == 0) {
    lean_internal_panic("torchlean_cuda_buffer_layer_norm_bwd: dimensions must be positive");
  }
  const size_t total = checked_mul_size(
      (size_t)rows, (size_t)cols, "torchlean_cuda_buffer_layer_norm_bwd: size overflow");
  if (dOut->size != total || normalized->size != total ||
      invStd->size != (size_t)rows || gamma->size != (size_t)cols) {
    lean_internal_panic("torchlean_cuda_buffer_layer_norm_bwd: buffer size mismatch");
  }
  check_axis_grid_size(
      (size_t)rows, "torchlean_cuda_buffer_layer_norm_bwd: rows exceed CUDA x-grid range");

  torchlean_cuda_buffer* dX = torchlean_cuda_buffer_alloc(total);
  torchlean_cuda_buffer* dGamma = torchlean_cuda_buffer_alloc((size_t)cols);
  torchlean_cuda_buffer* dBeta = torchlean_cuda_buffer_alloc((size_t)cols);
  float* dGammaPointwise = torchlean_cuda_scratch_alloc<float>(
      total, "cudaMalloc layerNorm dGamma workspace failed");
  layer_norm_bwd_rows_f32<<<dim3((unsigned int)rows), dim3(kBlockSize)>>>(
      dOut->data,
      normalized->data,
      invStd->data,
      gamma->data,
      dX->data,
      dGammaPointwise,
      rows,
      cols,
      (float)colsScale,
      (float)invCols);
  checkCuda(cudaGetLastError(), "cuda layerNorm backward row kernel launch failed");
  launch_reduce_sum_by_column(
      dGammaPointwise,
      dGamma->data,
      rows,
      cols,
      "torchlean_cuda_buffer_layer_norm_bwd: dGamma grid overflow",
      "cuda layerNorm dGamma reduction launch failed");
  launch_reduce_sum_by_column(
      dOut->data,
      dBeta->data,
      rows,
      cols,
      "torchlean_cuda_buffer_layer_norm_bwd: dBeta grid overflow",
      "cuda layerNorm dBeta reduction launch failed");
  torchlean_cuda_scratch_free(
      &dGammaPointwise, total, "cudaFree layerNorm dGamma workspace failed");
  return torchlean_cuda_box_three_buffers(dX, dGamma, dBeta);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_bmm_with_transpose(
    b_lean_obj_arg AObj, b_lean_obj_arg BObj, uint32_t batch, uint32_t m, uint32_t n,
    uint32_t p, uint32_t transposeA, uint32_t transposeB) {
  torchlean_cuda_buffer* A = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* B = torchlean_cuda_buffer_unbox(BObj);

  if (transposeA > 1 || transposeB > 1) {
    lean_internal_panic("torchlean_cuda_buffer_bmm_with_transpose: transpose flag must be 0 or 1");
  }

  const size_t Batch = (size_t)batch;
  const size_t M = (size_t)m;
  const size_t N = (size_t)n;
  const size_t P = (size_t)p;

  const size_t aSz = checked_mul3_size(
      Batch, M, N, "torchlean_cuda_buffer_bmm_with_transpose: A size overflow");
  const size_t bSz = checked_mul3_size(
      Batch, N, P, "torchlean_cuda_buffer_bmm_with_transpose: B size overflow");
  const size_t cSz = checked_mul3_size(
      Batch, M, P, "torchlean_cuda_buffer_bmm_with_transpose: C size overflow");

  if (A->size != aSz) {
    lean_internal_panic("torchlean_cuda_buffer_bmm_with_transpose: A.size mismatch");
  }
  if (B->size != bSz) {
    lean_internal_panic("torchlean_cuda_buffer_bmm_with_transpose: B.size mismatch");
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
    lean_internal_panic(
        "torchlean_cuda_buffer_bmm_with_transpose: dims too large for cuBLAS int API");
  }

  cublasHandle_t handle = getCublasHandle();

  const float alpha = 1.0f;
  const float beta = 0.0f;

  // Row-major buffers appear transposed to column-major cuBLAS. Reversing the operand order
  // computes C^T. The requested logical transpose maps directly to a cuBLAS transpose flag, so
  // no temporary matrix is needed.
  const cublasOperation_t opB = transposeB == 0 ? CUBLAS_OP_N : CUBLAS_OP_T;
  const cublasOperation_t opA = transposeA == 0 ? CUBLAS_OP_N : CUBLAS_OP_T;
  const int leadingB = transposeB == 0 ? (int)P : (int)N;
  const int leadingA = transposeA == 0 ? (int)N : (int)M;
  const long long int strideA = (long long int)(N * P);  // B batch stride (treated as A in cuBLAS)
  const long long int strideB = (long long int)(M * N);  // A batch stride (treated as B in cuBLAS)
  const long long int strideC = (long long int)(M * P);

  checkCublas(
      cublasSgemmStridedBatched(handle,
                               opB, opA,
                               (int)P, (int)M, (int)N,
                               &alpha,
                               B->data, leadingB, strideA,
                               A->data, leadingA, strideB,
                               &beta,
                               C->data, (int)P, strideC,
                               (int)batch),
      "cublasSgemmStridedBatched with logical transpose failed");

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
  dSpec = torchlean_cuda_scratch_alloc<cufftComplex>(
      complexSz, "torchlean_cuda_buffer_rfft1d_packed: cudaMalloc spectrum failed");

  cufftHandle plan;
  int nPlan[1] = {nInt};
  int inembed[1] = {nInt};
  int onembed[1] = {freqInt};
  cufftResult planRes =
      cufftPlanMany(&plan, 1, nPlan, inembed, 1, nInt, onembed, 1, freqInt, CUFFT_R2C, batchInt);
  if (planRes != CUFFT_SUCCESS) {
    torchlean_cuda_scratch_free(&dSpec, complexSz, "torchlean_cuda_buffer_rfft1d_packed: cleanup spectrum after plan failure failed");
    lean_internal_panic("torchlean_cuda_buffer_rfft1d_packed: cufftPlanMany R2C failed");
  }

  cufftResult execRes = cufftExecR2C(plan, (cufftReal*)x->data, dSpec);
  if (execRes != CUFFT_SUCCESS) {
    cufftDestroy(plan);
    torchlean_cuda_scratch_free(&dSpec, complexSz, "torchlean_cuda_buffer_rfft1d_packed: cleanup spectrum after exec failure failed");
    lean_internal_panic("torchlean_cuda_buffer_rfft1d_packed: cufftExecR2C failed");
  }

  pack_cufft_complex_to_ri_f32<<<blocks_for(complexSz), dim3(kBlockSize)>>>(dSpec, out->data,
                                                                            complexSz);
  if (cudaGetLastError() != cudaSuccess) {
    cufftDestroy(plan);
    torchlean_cuda_scratch_free(&dSpec, complexSz, "torchlean_cuda_buffer_rfft1d_packed: cleanup spectrum after pack failure failed");
    lean_internal_panic("torchlean_cuda_buffer_rfft1d_packed: pack kernel launch failed");
  }

  checkCufft(cufftDestroy(plan), "torchlean_cuda_buffer_rfft1d_packed: cufftDestroy failed");
  torchlean_cuda_scratch_free(&dSpec, complexSz, "torchlean_cuda_buffer_rfft1d_packed: cudaFree spectrum failed");
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
  dSpec = torchlean_cuda_scratch_alloc<cufftComplex>(
      complexSz, "torchlean_cuda_buffer_irfft1d_packed: cudaMalloc spectrum failed");

  pack_ri_to_cufft_complex_f32<<<blocks_for(complexSz), dim3(kBlockSize)>>>(spec->data, dSpec,
                                                                            complexSz, n);
  if (cudaGetLastError() != cudaSuccess) {
    torchlean_cuda_scratch_free(&dSpec, complexSz, "torchlean_cuda_buffer_irfft1d_packed: cleanup spectrum after pack failure failed");
    lean_internal_panic("torchlean_cuda_buffer_irfft1d_packed: pack kernel launch failed");
  }

  cufftHandle plan;
  int nPlan[1] = {nInt};
  int inembed[1] = {freqInt};
  int onembed[1] = {nInt};
  cufftResult planRes =
      cufftPlanMany(&plan, 1, nPlan, inembed, 1, freqInt, onembed, 1, nInt, CUFFT_C2R, batchInt);
  if (planRes != CUFFT_SUCCESS) {
    torchlean_cuda_scratch_free(&dSpec, complexSz, "torchlean_cuda_buffer_irfft1d_packed: cleanup spectrum after plan failure failed");
    lean_internal_panic("torchlean_cuda_buffer_irfft1d_packed: cufftPlanMany C2R failed");
  }

  cufftResult execRes = cufftExecC2R(plan, dSpec, (cufftReal*)out->data);
  if (execRes != CUFFT_SUCCESS) {
    cufftDestroy(plan);
    torchlean_cuda_scratch_free(&dSpec, complexSz, "torchlean_cuda_buffer_irfft1d_packed: cleanup spectrum after exec failure failed");
    lean_internal_panic("torchlean_cuda_buffer_irfft1d_packed: cufftExecC2R failed");
  }

  scale_f32<<<blocks_for(outSz), dim3(kBlockSize)>>>(out->data, outSz, 1.0f / (float)N);
  if (cudaGetLastError() != cudaSuccess) {
    cufftDestroy(plan);
    torchlean_cuda_scratch_free(&dSpec, complexSz, "torchlean_cuda_buffer_irfft1d_packed: cleanup spectrum after scale failure failed");
    lean_internal_panic("torchlean_cuda_buffer_irfft1d_packed: scale kernel launch failed");
  }

  checkCufft(cufftDestroy(plan), "torchlean_cuda_buffer_irfft1d_packed: cufftDestroy failed");
  torchlean_cuda_scratch_free(&dSpec, complexSz, "torchlean_cuda_buffer_irfft1d_packed: cudaFree spectrum failed");
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

static inline cufftResult try_make_spectral_conv1d_r2c_plan(uint32_t grid, uint32_t width,
                                                            uint32_t freq, cufftHandle* plan) {
  const int nInt = checked_cufft_int(grid, "spectralConv1dRfft: grid exceeds cuFFT int range");
  const int widthInt =
      checked_cufft_int(width, "spectralConv1dRfft: width exceeds cuFFT int range");
  const int freqInt = checked_cufft_int(freq, "spectralConv1dRfft: freq exceeds cuFFT int range");
  int nPlan[1] = {nInt};
  int inembed[1] = {nInt};
  int onembed[1] = {freqInt};
  return cufftPlanMany(plan, 1, nPlan, inembed, widthInt, 1, onembed, widthInt, 1,
                       CUFFT_R2C, widthInt);
}

static inline cufftResult try_make_spectral_conv1d_c2r_plan(uint32_t grid, uint32_t width,
                                                            uint32_t freq, cufftHandle* plan) {
  const int nInt = checked_cufft_int(grid, "spectralConv1dRfft: grid exceeds cuFFT int range");
  const int widthInt =
      checked_cufft_int(width, "spectralConv1dRfft: width exceeds cuFFT int range");
  const int freqInt = checked_cufft_int(freq, "spectralConv1dRfft: freq exceeds cuFFT int range");
  int nPlan[1] = {nInt};
  int inembed[1] = {freqInt};
  int onembed[1] = {nInt};
  return cufftPlanMany(plan, 1, nPlan, inembed, widthInt, 1, onembed, widthInt, 1,
                       CUFFT_C2R, widthInt);
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
  X = torchlean_cuda_scratch_alloc<cufftComplex>(specSz, "spectralConv1dRfft: malloc X failed");
  Z = torchlean_cuda_scratch_alloc<cufftComplex>(specSz, "spectralConv1dRfft: malloc Z failed");
  cudaError_t zErr = cudaMemset(
      Z, 0, checked_bytes_size(specSz, sizeof(cufftComplex), "spectralConv1dRfft: Z byte overflow"));
  if (zErr != cudaSuccess) {
    torchlean_cuda_scratch_free(&X, specSz, "spectralConv1dRfft: cleanup X after memset failed");
    torchlean_cuda_scratch_free(&Z, specSz, "spectralConv1dRfft: cleanup Z after memset failed");
    checkCuda(zErr, "spectralConv1dRfft: memset Z failed");
  }

  cufftHandle r2c = 0;
  cufftHandle c2r = 0;
  cufftResult r2cPlan = try_make_spectral_conv1d_r2c_plan(grid, width, (uint32_t)Freq, &r2c);
  if (r2cPlan != CUFFT_SUCCESS) {
    torchlean_cuda_scratch_free(&X, specSz, "spectralConv1dRfft: cleanup X after R2C plan failed");
    torchlean_cuda_scratch_free(&Z, specSz, "spectralConv1dRfft: cleanup Z after R2C plan failed");
    checkCufft(r2cPlan, "spectralConv1dRfft: cufftPlanMany R2C failed");
  }
  cufftResult c2rPlan = try_make_spectral_conv1d_c2r_plan(grid, width, (uint32_t)Freq, &c2r);
  if (c2rPlan != CUFFT_SUCCESS) {
    cufftDestroy(r2c);
    torchlean_cuda_scratch_free(&X, specSz, "spectralConv1dRfft: cleanup X after C2R plan failed");
    torchlean_cuda_scratch_free(&Z, specSz, "spectralConv1dRfft: cleanup Z after C2R plan failed");
    checkCufft(c2rPlan, "spectralConv1dRfft: cufftPlanMany C2R failed");
  }
  cufftResult execR2C = cufftExecR2C(r2c, (cufftReal*)x->data, X);
  if (execR2C != CUFFT_SUCCESS) {
    cufftDestroy(r2c);
    cufftDestroy(c2r);
    torchlean_cuda_scratch_free(&X, specSz, "spectralConv1dRfft: cleanup X after R2C failed");
    torchlean_cuda_scratch_free(&Z, specSz, "spectralConv1dRfft: cleanup Z after R2C failed");
    checkCufft(execR2C, "spectralConv1dRfft: R2C failed");
  }
  const size_t mulElems =
      checked_mul_size((size_t)modes, (size_t)width, "spectralConv1dRfft: mul launch size overflow");
  spectral_conv1d_mul_f32<<<blocks_for(mulElems), dim3(kBlockSize)>>>(
      X, wRe->data, wIm->data, Z, width, modes, grid);
  cudaError_t mulErr = cudaGetLastError();
  if (mulErr != cudaSuccess) {
    cufftDestroy(r2c);
    cufftDestroy(c2r);
    torchlean_cuda_scratch_free(&X, specSz, "spectralConv1dRfft: cleanup X after multiply failed");
    torchlean_cuda_scratch_free(&Z, specSz, "spectralConv1dRfft: cleanup Z after multiply failed");
    checkCuda(mulErr, "spectralConv1dRfft: multiply kernel failed");
  }
  cufftResult execC2R = cufftExecC2R(c2r, Z, (cufftReal*)out->data);
  if (execC2R != CUFFT_SUCCESS) {
    cufftDestroy(r2c);
    cufftDestroy(c2r);
    torchlean_cuda_scratch_free(&X, specSz, "spectralConv1dRfft: cleanup X after C2R failed");
    torchlean_cuda_scratch_free(&Z, specSz, "spectralConv1dRfft: cleanup Z after C2R failed");
    checkCufft(execC2R, "spectralConv1dRfft: C2R failed");
  }
  scale_f32<<<blocks_for(xSz), dim3(kBlockSize)>>>(out->data, xSz, 1.0f / (float)grid);
  cudaError_t scaleErr = cudaGetLastError();
  if (scaleErr != cudaSuccess) {
    cufftDestroy(r2c);
    cufftDestroy(c2r);
    torchlean_cuda_scratch_free(&X, specSz, "spectralConv1dRfft: cleanup X after scale failed");
    torchlean_cuda_scratch_free(&Z, specSz, "spectralConv1dRfft: cleanup Z after scale failed");
    checkCuda(scaleErr, "spectralConv1dRfft: scale kernel failed");
  }
  cufftResult destroyR2C = cufftDestroy(r2c);
  cufftResult destroyC2R = cufftDestroy(c2r);
  torchlean_cuda_scratch_free(&X, specSz, "spectralConv1dRfft: free X failed");
  torchlean_cuda_scratch_free(&Z, specSz, "spectralConv1dRfft: free Z failed");
  checkCufft(destroyR2C, "spectralConv1dRfft: destroy R2C failed");
  checkCufft(destroyC2R, "spectralConv1dRfft: destroy C2R failed");
  return torchlean_cuda_buffer_box(out);
}

static inline void spectral_conv1d_make_x_dz(torchlean_cuda_buffer* x, torchlean_cuda_buffer* dY,
                                             uint32_t grid, uint32_t width, uint32_t modes,
                                             size_t specSz, cufftComplex** XOut,
                                             cufftComplex** dZOut) {
  cufftComplex* X = nullptr;
  cufftComplex* G = nullptr;
  cufftComplex* dZ = nullptr;
  X = torchlean_cuda_scratch_alloc<cufftComplex>(specSz, "spectralConv1dRfft bwd: malloc X failed");
  G = torchlean_cuda_scratch_alloc<cufftComplex>(specSz, "spectralConv1dRfft bwd: malloc G failed");
  dZ = torchlean_cuda_scratch_alloc<cufftComplex>(specSz, "spectralConv1dRfft bwd: malloc dZ failed");
  cudaError_t dzMemset = cudaMemset(
      dZ, 0,
      checked_bytes_size(specSz, sizeof(cufftComplex), "spectralConv1dRfft bwd: dZ byte overflow"));
  if (dzMemset != cudaSuccess) {
    torchlean_cuda_scratch_free(&X, specSz, "spectralConv1dRfft bwd: cleanup X after memset failed");
    torchlean_cuda_scratch_free(&G, specSz, "spectralConv1dRfft bwd: cleanup G after memset failed");
    torchlean_cuda_scratch_free(&dZ, specSz, "spectralConv1dRfft bwd: cleanup dZ after memset failed");
    checkCuda(dzMemset, "spectralConv1dRfft bwd: memset dZ failed");
  }
  cufftHandle r2c = 0;
  cufftResult planRes = try_make_spectral_conv1d_r2c_plan(grid, width, grid / 2 + 1, &r2c);
  if (planRes != CUFFT_SUCCESS) {
    torchlean_cuda_scratch_free(&X, specSz, "spectralConv1dRfft bwd: cleanup X after R2C plan failed");
    torchlean_cuda_scratch_free(&G, specSz, "spectralConv1dRfft bwd: cleanup G after R2C plan failed");
    torchlean_cuda_scratch_free(&dZ, specSz, "spectralConv1dRfft bwd: cleanup dZ after R2C plan failed");
    checkCufft(planRes, "spectralConv1dRfft bwd: cufftPlanMany R2C failed");
  }
  cufftResult xExec = cufftExecR2C(r2c, (cufftReal*)x->data, X);
  if (xExec != CUFFT_SUCCESS) {
    cufftDestroy(r2c);
    torchlean_cuda_scratch_free(&X, specSz, "spectralConv1dRfft bwd: cleanup X after R2C x failed");
    torchlean_cuda_scratch_free(&G, specSz, "spectralConv1dRfft bwd: cleanup G after R2C x failed");
    torchlean_cuda_scratch_free(&dZ, specSz, "spectralConv1dRfft bwd: cleanup dZ after R2C x failed");
    checkCufft(xExec, "spectralConv1dRfft bwd: R2C x failed");
  }
  cufftResult dyExec = cufftExecR2C(r2c, (cufftReal*)dY->data, G);
  if (dyExec != CUFFT_SUCCESS) {
    cufftDestroy(r2c);
    torchlean_cuda_scratch_free(&X, specSz, "spectralConv1dRfft bwd: cleanup X after R2C dY failed");
    torchlean_cuda_scratch_free(&G, specSz, "spectralConv1dRfft bwd: cleanup G after R2C dY failed");
    torchlean_cuda_scratch_free(&dZ, specSz, "spectralConv1dRfft bwd: cleanup dZ after R2C dY failed");
    checkCufft(dyExec, "spectralConv1dRfft bwd: R2C dY failed");
  }
  const size_t modeWidth =
      checked_mul_size((size_t)modes, (size_t)width, "spectralConv1dRfft bwd: mode*width overflow");
  spectral_conv1d_dz_from_dy_f32<<<blocks_for(modeWidth), dim3(kBlockSize)>>>(
      G, dZ, grid, width, modes);
  cudaError_t dzKernel = cudaGetLastError();
  if (dzKernel != cudaSuccess) {
    cufftDestroy(r2c);
    torchlean_cuda_scratch_free(&X, specSz, "spectralConv1dRfft bwd: cleanup X after dZ failed");
    torchlean_cuda_scratch_free(&G, specSz, "spectralConv1dRfft bwd: cleanup G after dZ failed");
    torchlean_cuda_scratch_free(&dZ, specSz, "spectralConv1dRfft bwd: cleanup dZ after dZ failed");
    checkCuda(dzKernel, "spectralConv1dRfft bwd: dZ kernel failed");
  }
  checkCufft(cufftDestroy(r2c), "spectralConv1dRfft bwd: destroy R2C failed");
  torchlean_cuda_scratch_free(&G, specSz, "spectralConv1dRfft bwd: free G failed");
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
  dX = torchlean_cuda_scratch_alloc<cufftComplex>(specSz, "spectralConv1dRfft bwd_x: malloc dX failed");
  cudaError_t dxMemset = cudaMemset(
      dX, 0,
      checked_bytes_size(specSz, sizeof(cufftComplex), "spectralConv1dRfft bwd_x: dX byte overflow"));
  if (dxMemset != cudaSuccess) {
    torchlean_cuda_scratch_free(&X, specSz, "spectralConv1dRfft bwd_x: cleanup X after dX memset failed");
    torchlean_cuda_scratch_free(&dZ, specSz, "spectralConv1dRfft bwd_x: cleanup dZ after dX memset failed");
    torchlean_cuda_scratch_free(&dX, specSz, "spectralConv1dRfft bwd_x: cleanup dX after dX memset failed");
    checkCuda(dxMemset, "spectralConv1dRfft bwd_x: memset dX failed");
  }
  const size_t modeWidth =
      checked_mul_size((size_t)modes, (size_t)width, "spectralConv1dRfft bwd_x: mode*width overflow");
  spectral_conv1d_bwd_xspec_f32<<<blocks_for(modeWidth), dim3(kBlockSize)>>>(
      dZ, wRe->data, wIm->data, dX, width, modes);
  cudaError_t xSpecErr = cudaGetLastError();
  if (xSpecErr != cudaSuccess) {
    torchlean_cuda_scratch_free(&X, specSz, "spectralConv1dRfft bwd_x: cleanup X after dXspec failed");
    torchlean_cuda_scratch_free(&dZ, specSz, "spectralConv1dRfft bwd_x: cleanup dZ after dXspec failed");
    torchlean_cuda_scratch_free(&dX, specSz, "spectralConv1dRfft bwd_x: cleanup dX after dXspec failed");
    checkCuda(xSpecErr, "spectralConv1dRfft bwd_x: dXspec kernel failed");
  }
  spectral_conv1d_prepare_rfft_adjoint_f32<<<blocks_for(specSz), dim3(kBlockSize)>>>(
      dX, grid, width);
  cudaError_t prepareErr = cudaGetLastError();
  if (prepareErr != cudaSuccess) {
    torchlean_cuda_scratch_free(&X, specSz, "spectralConv1dRfft bwd_x: cleanup X after prepare failed");
    torchlean_cuda_scratch_free(&dZ, specSz, "spectralConv1dRfft bwd_x: cleanup dZ after prepare failed");
    torchlean_cuda_scratch_free(&dX, specSz, "spectralConv1dRfft bwd_x: cleanup dX after prepare failed");
    checkCuda(prepareErr, "spectralConv1dRfft bwd_x: prepare adjoint failed");
  }
  cufftHandle c2r = 0;
  cufftResult planResult = try_make_spectral_conv1d_c2r_plan(grid, width, (uint32_t)Freq, &c2r);
  if (planResult != CUFFT_SUCCESS) {
    torchlean_cuda_scratch_free(&X, specSz, "spectralConv1dRfft bwd_x: cleanup X after plan failed");
    torchlean_cuda_scratch_free(&dZ, specSz, "spectralConv1dRfft bwd_x: cleanup dZ after plan failed");
    torchlean_cuda_scratch_free(&dX, specSz, "spectralConv1dRfft bwd_x: cleanup dX after plan failed");
    checkCufft(planResult, "spectralConv1dRfft bwd_x: C2R plan failed");
  }
  torchlean_cuda_buffer* dx = torchlean_cuda_buffer_alloc(xSz);
  cufftResult inverseResult = cufftExecC2R(c2r, dX, (cufftReal*)dx->data);
  cufftResult destroyResult = cufftDestroy(c2r);
  if (inverseResult != CUFFT_SUCCESS || destroyResult != CUFFT_SUCCESS) {
    torchlean_cuda_scratch_free(&X, specSz, "spectralConv1dRfft bwd_x: cleanup X after C2R failed");
    torchlean_cuda_scratch_free(&dZ, specSz, "spectralConv1dRfft bwd_x: cleanup dZ after C2R failed");
    torchlean_cuda_scratch_free(&dX, specSz, "spectralConv1dRfft bwd_x: cleanup dX after C2R failed");
    checkCufft(inverseResult, "spectralConv1dRfft bwd_x: C2R failed");
    checkCufft(destroyResult, "spectralConv1dRfft bwd_x: destroy C2R failed");
  }
  torchlean_cuda_scratch_free(&X, specSz, "spectralConv1dRfft bwd_x: free X failed");
  torchlean_cuda_scratch_free(&dZ, specSz, "spectralConv1dRfft bwd_x: free dZ failed");
  torchlean_cuda_scratch_free(&dX, specSz, "spectralConv1dRfft bwd_x: free dX failed");
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
  cudaError_t weightsErr = cudaGetLastError();
  if (weightsErr != cudaSuccess) {
    torchlean_cuda_scratch_free(&X, specSz, "spectralConv1dRfft bwd_w: cleanup X after weights failed");
    torchlean_cuda_scratch_free(&dZ, specSz, "spectralConv1dRfft bwd_w: cleanup dZ after weights failed");
    torchlean_cuda_buffer_drop_unboxed(dWRe);
    torchlean_cuda_buffer_drop_unboxed(dWIm);
    checkCuda(weightsErr, "spectralConv1dRfft bwd_w: weights kernel failed");
  }
  torchlean_cuda_scratch_free(&X, specSz, "spectralConv1dRfft bwd_w: free X failed");
  torchlean_cuda_scratch_free(&dZ, specSz, "spectralConv1dRfft bwd_w: free dZ failed");
  torchlean_cuda_buffer* keep = imag ? dWIm : dWRe;
  torchlean_cuda_buffer* drop = imag ? dWRe : dWIm;
  torchlean_cuda_buffer_drop_unboxed(drop);
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
  const size_t total = checked_mul_size(
      T, D, "torchlean_cuda_buffer_selective_scan_diag_fwd: seqLen*state overflow");

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
  const size_t total = checked_mul_size(
      T, D, "torchlean_cuda_buffer_selective_scan_diag_bwd: seqLen*state overflow");

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

  if (D != 0) {
    dim3 blocks = blocks_for(D);
    dim3 threads = dim3(kBlockSize);
    selective_scan_diag_bwd_f32<<<blocks, threads>>>(A->data, B->data, X->data, h0->data,
                                                     out->data, dY->data, dA->data, dB->data,
                                                     dX->data, dH0->data, seqLen, stateDim);
    checkCuda(cudaGetLastError(), "cuda selectiveScanDiagBwd kernel launch failed");
  }

  return torchlean_cuda_box_four_buffers(dA, dB, dX, dH0);
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
  const size_t total = checked_mul_size(
      T, D, "torchlean_cuda_buffer_selective_scan_diag_var_fwd: seqLen*state overflow");

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

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_selective_scan_diag_var_bwd(
    b_lean_obj_arg AObj, b_lean_obj_arg BObj, b_lean_obj_arg XObj, b_lean_obj_arg H0Obj,
    b_lean_obj_arg OutObj, b_lean_obj_arg DYObj, uint32_t seqLen, uint32_t stateDim) {
  torchlean_cuda_buffer* A = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* B = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* X = torchlean_cuda_buffer_unbox(XObj);
  torchlean_cuda_buffer* h0 = torchlean_cuda_buffer_unbox(H0Obj);
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_unbox(OutObj);
  torchlean_cuda_buffer* dY = torchlean_cuda_buffer_unbox(DYObj);
  const size_t D = (size_t)stateDim;
  const size_t total = checked_mul_size(
      (size_t)seqLen, D, "selectiveScanDiagVarBwd: sequence size overflow");
  if (A->size != total || B->size != total || X->size != total ||
      out->size != total || dY->size != total || h0->size != D) {
    lean_internal_panic("selectiveScanDiagVarBwd: input or saved-state size mismatch");
  }
  torchlean_cuda_buffer* dA = torchlean_cuda_buffer_alloc(total);
  torchlean_cuda_buffer* dB = torchlean_cuda_buffer_alloc(total);
  torchlean_cuda_buffer* dX = torchlean_cuda_buffer_alloc(total);
  torchlean_cuda_buffer* dH0 = torchlean_cuda_buffer_alloc(D);
  if (D != 0) {
    selective_scan_diag_var_bwd_f32<<<blocks_for(D), dim3(kBlockSize)>>>(
        A->data, B->data, X->data, h0->data, out->data, dY->data,
        dA->data, dB->data, dX->data, dH0->data, seqLen, stateDim);
    checkCuda(cudaGetLastError(), "cuda selectiveScanDiagVarBwd kernel launch failed");
  }
  return torchlean_cuda_box_four_buffers(dA, dB, dX, dH0);
}

// Simple attention kernel.
//
// Match TorchLean's scaled-dot-product attention spec: scores = QK^T * scale, optional dense
// Boolean mask, stable row softmax, then multiplication by V. This is deliberately direct. Backward
// recomputes the row softmax statistics instead of storing the full attention matrix, so this is not
// the production IO-tiled FlashAttention schedule.
//
// Mask convention: zero means disallowed. Disallowed entries contribute literal zero softmax
// numerator, matching TorchLean's hard-mask spec rather than a finite sentinel.
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
    if (!flash_attention_allowed(mask, hasMask, batchIdx, i, j, n)) continue;
    float s = flash_attention_score(Q, K, mask, hasMask, batchIdx, i, j, n, d, scale);
    if (s > m) m = s;
  }

  float z = 0.0f;
  for (uint32_t j = 0; j < n; ++j) {
    if (!flash_attention_allowed(mask, hasMask, batchIdx, i, j, n)) continue;
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
  if (!flash_attention_allowed(mask, hasMask, batchIdx, i, j, n) || denom == 0.0f) {
    return 0.0f;
  }
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
  const size_t qkvSz =
      checked_mul3_size((size_t)batch, (size_t)n, (size_t)d,
                        "torchlean_cuda_buffer_flash_attention_fwd: Q/K/V size overflow");
  const size_t maskSz =
      checked_mul3_size((size_t)batch, (size_t)n, (size_t)n,
                        "torchlean_cuda_buffer_flash_attention_fwd: mask size overflow");
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

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_flash_attention_bwd(
    b_lean_obj_arg QObj, b_lean_obj_arg KObj, b_lean_obj_arg VObj, b_lean_obj_arg MaskObj,
    b_lean_obj_arg DOutObj, uint32_t hasMask, uint32_t batch, uint32_t n, uint32_t d,
    double scaleHost) {
  torchlean_cuda_buffer* Q = torchlean_cuda_buffer_unbox(QObj);
  torchlean_cuda_buffer* K = torchlean_cuda_buffer_unbox(KObj);
  torchlean_cuda_buffer* V = torchlean_cuda_buffer_unbox(VObj);
  torchlean_cuda_buffer* mask = torchlean_cuda_buffer_unbox(MaskObj);
  torchlean_cuda_buffer* dOut = torchlean_cuda_buffer_unbox(DOutObj);
  const size_t qkvSz = checked_mul3_size((size_t)batch, (size_t)n, (size_t)d,
                                         "cuda flashAttention backward: Q/K/V/dOut size overflow");
  const size_t maskSz = checked_mul3_size((size_t)batch, (size_t)n, (size_t)n,
                                          "cuda flashAttention backward: mask size overflow");
  if (Q->size != qkvSz || K->size != qkvSz || V->size != qkvSz || dOut->size != qkvSz) {
    lean_internal_panic("cuda flashAttention backward: Q/K/V/dOut size mismatch");
  }
  if (hasMask != 0 && mask->size != maskSz) {
    lean_internal_panic("cuda flashAttention backward: mask size mismatch");
  }
  torchlean_cuda_buffer* dQ = torchlean_cuda_buffer_alloc(qkvSz);
  torchlean_cuda_buffer* dK = torchlean_cuda_buffer_alloc(qkvSz);
  torchlean_cuda_buffer* dV = torchlean_cuda_buffer_alloc(qkvSz);
  if (qkvSz == 0) return torchlean_cuda_box_three_buffers(dQ, dK, dV);
  dim3 blocks = blocks_for(qkvSz);
  dim3 threads = dim3(kBlockSize);
  flash_attention_bwd_q_f32<<<blocks, threads>>>(Q->data, K->data, V->data, mask->data, dOut->data,
                                                 hasMask, batch, n, d, (float)scaleHost, dQ->data);
  checkCuda(cudaGetLastError(), "cuda flashAttention backward Q kernel launch failed");
  flash_attention_bwd_k_f32<<<blocks, threads>>>(Q->data, K->data, V->data, mask->data, dOut->data,
                                                 hasMask, batch, n, d, (float)scaleHost, dK->data);
  checkCuda(cudaGetLastError(), "cuda flashAttention backward K kernel launch failed");
  flash_attention_bwd_v_f32<<<blocks, threads>>>(Q->data, K->data, V->data, mask->data, dOut->data,
                                                 hasMask, batch, n, d, (float)scaleHost, dV->data);
  checkCuda(cudaGetLastError(), "cuda flashAttention backward V kernel launch failed");
  return torchlean_cuda_box_three_buffers(dQ, dK, dV);
}

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

  uint32_t* dIdx = upload_nat_indices(
      IdxObj, K,
      "torchlean_cuda_buffer_gather_vec: bad index Nat",
      "cudaMalloc indices failed",
      "cudaMemcpy indices failed");

  dim3 blocks = blocks_for(K);
  dim3 threads = dim3(kBlockSize);
  gather_vec_f32<<<blocks, threads>>>(v->data, dIdx, K, n, out->data);
  checkCuda(cudaGetLastError(), "cuda gatherVec kernel launch failed");

  torchlean_cuda_scratch_free(&dIdx, K, "cudaFree gatherVec indices failed");
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

  uint32_t* dIdx = upload_nat_indices(
      IdxObj, K,
      "torchlean_cuda_buffer_scatter_add: bad index Nat",
      "cudaMalloc indices failed",
      "cudaMemcpy indices failed");

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

  torchlean_cuda_scratch_free(&dIdx, K, "cudaFree scatterAdd indices failed");
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

  HostBroadcastArrays h = alloc_host_broadcast_arrays(rankIn, rankOut);

  size_t inSize = 1;
  for (size_t i = 0; i < rankIn; ++i) {
    b_lean_obj_res dNat = lean_array_get_core(InDimsObj, i);
    uint32_t d = 0;
    if (!nat_to_u32_checked(dNat, &d)) {
      lean_internal_panic("torchlean_cuda_buffer_broadcast_to: inDims contains big Nat");
    }
    h.inDims[i] = d;
    inSize = checked_mul_acc_size(
        inSize, (size_t)d, "torchlean_cuda_buffer_broadcast_to: input shape overflow");
  }
  size_t outSize = 1;
  for (size_t i = 0; i < rankOut; ++i) {
    b_lean_obj_res dNat = lean_array_get_core(OutDimsObj, i);
    uint32_t d = 0;
    if (!nat_to_u32_checked(dNat, &d)) {
      lean_internal_panic("torchlean_cuda_buffer_broadcast_to: outDims contains big Nat");
    }
    h.outDims[i] = d;
    outSize = checked_mul_acc_size(
        outSize, (size_t)d, "torchlean_cuda_buffer_broadcast_to: output shape overflow");

    b_lean_obj_res mNat = lean_array_get_core(AxisMapObj, i);
    uint32_t mv =
        nat_to_u32_or_panic(mNat, "torchlean_cuda_buffer_broadcast_to: axisMap contains big Nat");
    h.axisMap[i] = mv;
  }

  if (x->size != inSize) {
    lean_internal_panic("torchlean_cuda_buffer_broadcast_to: input size mismatch");
  }

  torchlean_cuda_require_broadcast_map(
      h.axisMap, rankIn, rankOut, "torchlean_cuda_buffer_broadcast_to");

  // Check broadcast shape agreement before launch.
  for (size_t ax = 0; ax < rankOut; ++ax) {
    uint32_t mv = h.axisMap[ax];
    if (mv == 0) continue;
    size_t inAx = (size_t)(mv - 1);
    if (inAx >= rankIn) {
      lean_internal_panic("torchlean_cuda_buffer_broadcast_to: axisMap out of range");
    }
    uint32_t id = h.inDims[inAx];
    uint32_t od = h.outDims[ax];
    if (!(id == 1 || id == od)) {
      lean_internal_panic("torchlean_cuda_buffer_broadcast_to: incompatible broadcast dims");
    }
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outSize);
  if (outSize == 0) {
    free_host_broadcast_arrays(&h);
    return torchlean_cuda_buffer_box(out);
  }

  uint32_t* dInDims = nullptr;
  uint32_t* dOutDims = nullptr;
  uint32_t* dMap = nullptr;
  if (rankIn != 0) {
    dInDims = torchlean_cuda_scratch_alloc<uint32_t>(rankIn, "cudaMalloc inDims failed");
    const size_t inDimBytes = checked_bytes_size(
        rankIn, sizeof(uint32_t), "torchlean_cuda_buffer_broadcast_to: inDims byte overflow");
    cudaError_t copyInDims = cudaMemcpy(dInDims, h.inDims, inDimBytes, cudaMemcpyHostToDevice);
    if (copyInDims != cudaSuccess) {
      free_host_broadcast_arrays(&h);
      torchlean_cuda_scratch_free(&dInDims, rankIn, "cudaFree broadcastTo inDims after copy failed");
      checkCuda(copyInDims, "cudaMemcpy inDims failed");
    }
  }
  if (rankOut != 0) {
    dOutDims = torchlean_cuda_scratch_alloc<uint32_t>(rankOut, "cudaMalloc outDims failed");
    dMap = torchlean_cuda_scratch_alloc<uint32_t>(rankOut, "cudaMalloc axisMap failed");
    const size_t outDimBytes = checked_bytes_size(
        rankOut, sizeof(uint32_t), "torchlean_cuda_buffer_broadcast_to: outDims byte overflow");
    cudaError_t copyOutDims = cudaMemcpy(dOutDims, h.outDims, outDimBytes, cudaMemcpyHostToDevice);
    if (copyOutDims != cudaSuccess) {
      free_host_broadcast_arrays(&h);
      torchlean_cuda_scratch_free(&dInDims, rankIn, "cudaFree broadcastTo inDims after outDims copy failed");
      torchlean_cuda_scratch_free(&dOutDims, rankOut, "cudaFree broadcastTo outDims after copy failed");
      torchlean_cuda_scratch_free(&dMap, rankOut, "cudaFree broadcastTo axisMap after outDims copy failed");
      checkCuda(copyOutDims, "cudaMemcpy outDims failed");
    }
    cudaError_t copyMap = cudaMemcpy(dMap, h.axisMap, outDimBytes, cudaMemcpyHostToDevice);
    if (copyMap != cudaSuccess) {
      free_host_broadcast_arrays(&h);
      torchlean_cuda_scratch_free(&dInDims, rankIn, "cudaFree broadcastTo inDims after axisMap copy failed");
      torchlean_cuda_scratch_free(&dOutDims, rankOut, "cudaFree broadcastTo outDims after axisMap copy failed");
      torchlean_cuda_scratch_free(&dMap, rankOut, "cudaFree broadcastTo axisMap after copy failed");
      checkCuda(copyMap, "cudaMemcpy axisMap failed");
    }
  }
  free_host_broadcast_arrays(&h);

  dim3 blocks = blocks_for(outSize);
  dim3 threads = dim3(kBlockSize);
  broadcast_to_f32<<<blocks, threads>>>(x->data, out->data, outSize,
                                       dInDims, (int)rankIn,
                                       dOutDims, (int)rankOut,
                                       dMap);
  checkCuda(cudaGetLastError(), "cuda broadcastTo kernel launch failed");

  torchlean_cuda_scratch_free(&dInDims, rankIn, "cudaFree broadcastTo inDims failed");
  torchlean_cuda_scratch_free(&dOutDims, rankOut, "cudaFree broadcastTo outDims failed");
  torchlean_cuda_scratch_free(&dMap, rankOut, "cudaFree broadcastTo axisMap failed");
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

  HostBroadcastArrays h = alloc_host_broadcast_arrays(rankIn, rankOut);

  size_t inSize = 1;
  for (size_t i = 0; i < rankIn; ++i) {
    uint32_t d = 0;
    if (!nat_to_u32_checked(lean_array_get_core(InDimsObj, i), &d)) {
      lean_internal_panic("torchlean_cuda_buffer_reduce_from_broadcast: inDims contains big Nat");
    }
    h.inDims[i] = d;
    inSize = checked_mul_acc_size(
        inSize, (size_t)d, "torchlean_cuda_buffer_reduce_from_broadcast: input shape overflow");
  }
  size_t outSize = 1;
  for (size_t i = 0; i < rankOut; ++i) {
    uint32_t d = 0;
    if (!nat_to_u32_checked(lean_array_get_core(OutDimsObj, i), &d)) {
      lean_internal_panic("torchlean_cuda_buffer_reduce_from_broadcast: outDims contains big Nat");
    }
    h.outDims[i] = d;
    outSize = checked_mul_acc_size(
        outSize, (size_t)d, "torchlean_cuda_buffer_reduce_from_broadcast: output shape overflow");

    uint32_t mv =
        nat_to_u32_or_panic(lean_array_get_core(AxisMapObj, i),
                            "torchlean_cuda_buffer_reduce_from_broadcast: axisMap contains big Nat");
    h.axisMap[i] = mv;
  }

  if (dOut->size != outSize) {
    lean_internal_panic("torchlean_cuda_buffer_reduce_from_broadcast: dOut size mismatch");
  }

  torchlean_cuda_require_broadcast_map(
      h.axisMap, rankIn, rankOut, "torchlean_cuda_buffer_reduce_from_broadcast");

  // Use the same broadcast checks as the forward path.
  for (size_t ax = 0; ax < rankOut; ++ax) {
    uint32_t mv = h.axisMap[ax];
    if (mv == 0) continue;
    size_t inAx = (size_t)(mv - 1);
    if (inAx >= rankIn) {
      lean_internal_panic("torchlean_cuda_buffer_reduce_from_broadcast: axisMap out of range");
    }
    uint32_t id = h.inDims[inAx];
    uint32_t od = h.outDims[ax];
    if (!(id == 1 || id == od)) {
      lean_internal_panic("torchlean_cuda_buffer_reduce_from_broadcast: incompatible broadcast dims");
    }
  }

  torchlean_cuda_buffer* dIn = torchlean_cuda_buffer_alloc(inSize);
  if (inSize == 0) {
    free_host_broadcast_arrays(&h);
    return torchlean_cuda_buffer_box(dIn);
  }
  if (outSize == 0) {
    // Totalize empty output gradients as the all-zeros input gradient.
    checkCuda(cudaMemset(dIn->data, 0, inSize * sizeof(float)),
              "cudaMemset reduceFromBroadcast failed");
    free_host_broadcast_arrays(&h);
    return torchlean_cuda_buffer_box(dIn);
  }

  // A scalar broadcast is common in dropout and scalar-valued model controls:
  //
  //   () -> (...).
  //
  // Its adjoint is one reduction over the entire output. Avoid sending every output element
  // through the generic atomic kernel when the caller has selected the faster reduction policy.
  if (!torchlean_cuda_get_deterministic_reductions() && rankIn == 0 &&
      outSize <= (size_t)UINT32_MAX) {
    free_host_broadcast_arrays(&h);
    launch_reduce_sum_by_column(
        dOut->data, dIn->data, (uint32_t)outSize, 1,
        "torchlean_cuda_buffer_reduce_from_broadcast: scalar reduction grid overflow",
        "cuda reduceFromBroadcast scalar kernel launch failed");
    return torchlean_cuda_buffer_box(dIn);
  }

  // A vector broadcast over leading axes is the common bias-add pattern:
  //
  //   (cols) -> (..., cols).
  //
  // Its adjoint is a column reduction after folding the leading axes into `rows`. The generic
  // atomic kernel below is correct for arbitrary broadcast maps, but it creates one atomic update
  // per output element and is especially costly for language-model output projections. Keep the
  // fixed row-major implementation when deterministic reductions are requested.
  bool vectorSuffixReduction =
      !torchlean_cuda_get_deterministic_reductions() &&
      rankIn == 1 && rankOut >= 1 &&
      h.inDims[0] == h.outDims[rankOut - 1] &&
      h.axisMap[rankOut - 1] == 1;
  for (size_t ax = 0; vectorSuffixReduction && ax + 1 < rankOut; ++ax) {
    vectorSuffixReduction = h.axisMap[ax] == 0;
  }
  if (vectorSuffixReduction) {
    const size_t cols = inSize;
    const size_t rows = outSize / cols;
    if (rows <= (size_t)UINT32_MAX && cols <= (size_t)UINT32_MAX) {
      free_host_broadcast_arrays(&h);
      launch_reduce_sum_by_column(
          dOut->data, dIn->data, (uint32_t)rows, (uint32_t)cols,
          "torchlean_cuda_buffer_reduce_from_broadcast: vector column tiles exceed CUDA x-grid range",
          "cuda reduceFromBroadcast vector-suffix kernel launch failed");
      return torchlean_cuda_buffer_box(dIn);
    }
  }

  uint32_t* dInDims = nullptr;
  uint32_t* dOutDims = nullptr;
  uint32_t* dMap = nullptr;
  if (rankIn != 0) {
    dInDims = torchlean_cuda_scratch_alloc<uint32_t>(rankIn, "cudaMalloc inDims failed");
    const size_t inDimBytes = checked_bytes_size(
        rankIn, sizeof(uint32_t),
        "torchlean_cuda_buffer_reduce_from_broadcast: inDims byte overflow");
    cudaError_t copyInDims = cudaMemcpy(dInDims, h.inDims, inDimBytes, cudaMemcpyHostToDevice);
    if (copyInDims != cudaSuccess) {
      free_host_broadcast_arrays(&h);
      torchlean_cuda_scratch_free(&dInDims, rankIn,
                                  "cudaFree reduceFromBroadcast inDims after copy failed");
      checkCuda(copyInDims, "cudaMemcpy inDims failed");
    }
  }
  if (rankOut != 0) {
    dOutDims = torchlean_cuda_scratch_alloc<uint32_t>(rankOut, "cudaMalloc outDims failed");
    dMap = torchlean_cuda_scratch_alloc<uint32_t>(rankOut, "cudaMalloc axisMap failed");
    const size_t outDimBytes = checked_bytes_size(
        rankOut, sizeof(uint32_t),
        "torchlean_cuda_buffer_reduce_from_broadcast: outDims byte overflow");
    cudaError_t copyOutDims = cudaMemcpy(dOutDims, h.outDims, outDimBytes, cudaMemcpyHostToDevice);
    if (copyOutDims != cudaSuccess) {
      free_host_broadcast_arrays(&h);
      torchlean_cuda_scratch_free(&dInDims, rankIn,
                                  "cudaFree reduceFromBroadcast inDims after outDims copy failed");
      torchlean_cuda_scratch_free(&dOutDims, rankOut,
                                  "cudaFree reduceFromBroadcast outDims after copy failed");
      torchlean_cuda_scratch_free(&dMap, rankOut,
                                  "cudaFree reduceFromBroadcast axisMap after outDims copy failed");
      checkCuda(copyOutDims, "cudaMemcpy outDims failed");
    }
    cudaError_t copyMap = cudaMemcpy(dMap, h.axisMap, outDimBytes, cudaMemcpyHostToDevice);
    if (copyMap != cudaSuccess) {
      free_host_broadcast_arrays(&h);
      torchlean_cuda_scratch_free(&dInDims, rankIn,
                                  "cudaFree reduceFromBroadcast inDims after axisMap copy failed");
      torchlean_cuda_scratch_free(&dOutDims, rankOut,
                                  "cudaFree reduceFromBroadcast outDims after axisMap copy failed");
      torchlean_cuda_scratch_free(&dMap, rankOut,
                                  "cudaFree reduceFromBroadcast axisMap after copy failed");
      checkCuda(copyMap, "cudaMemcpy axisMap failed");
    }
  }
  free_host_broadcast_arrays(&h);

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

  torchlean_cuda_scratch_free(&dInDims, rankIn, "cudaFree reduceFromBroadcast inDims failed");
  torchlean_cuda_scratch_free(&dOutDims, rankOut, "cudaFree reduceFromBroadcast outDims failed");
  torchlean_cuda_scratch_free(&dMap, rankOut, "cudaFree reduceFromBroadcast axisMap failed");
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
    uint32_t d = 0;
    if (!nat_to_u32_checked(lean_array_get_core(DimsObj, i), &d)) {
      lean_internal_panic("torchlean_cuda_buffer_swap_adjacent_at_depth: dims contains big Nat");
    }
    hDims[i] = d;
    total = checked_mul_acc_size(
        total, (size_t)d, "torchlean_cuda_buffer_swap_adjacent_at_depth: shape overflow");
  }
  if (x->size != total) {
    lean_internal_panic("torchlean_cuda_buffer_swap_adjacent_at_depth: input size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(total);
  if (total == 0) {
    return torchlean_cuda_buffer_box(out);
  }

  uint32_t* dDims = nullptr;
  dDims = torchlean_cuda_scratch_alloc<uint32_t>(rank, "cudaMalloc dims failed");
  const size_t dimBytes = checked_bytes_size(
      rank, sizeof(uint32_t), "torchlean_cuda_buffer_reduce_sum_axis: dims byte overflow");
  cudaError_t copyDims = cudaMemcpy(dDims, hDims, dimBytes, cudaMemcpyHostToDevice);
  if (copyDims != cudaSuccess) {
    torchlean_cuda_scratch_free(&dDims, rank, "cudaFree reduceSumAxis dims after copy failed");
    checkCuda(copyDims, "cudaMemcpy dims failed");
  }

  dim3 blocks = blocks_for(total);
  dim3 threads = dim3(kBlockSize);
  swap_adjacent_at_depth_f32<<<blocks, threads>>>(x->data, out->data, total, dDims, (int)rank,
                                                  (int)depth);
  checkCuda(cudaGetLastError(), "cuda swapAdjacentAtDepth kernel launch failed");

  torchlean_cuda_scratch_free(&dDims, rank, "cudaFree swapAdjacentAtDepth dims failed");
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
    uint32_t d = 0;
    if (!nat_to_u32_checked(lean_array_get_core(DimsObj, i), &d)) {
      lean_internal_panic("torchlean_cuda_buffer_reduce_sum_axis: dims contains big Nat");
    }
    hDims[i] = d;
    inSize = checked_mul_acc_size(
        inSize, (size_t)d, "torchlean_cuda_buffer_reduce_sum_axis: input shape overflow");
  }
  if (x->size != inSize) {
    lean_internal_panic("torchlean_cuda_buffer_reduce_sum_axis: input size mismatch");
  }

  size_t outSize = 1;
  for (size_t i = 0; i < rank; ++i) {
    if (i != (size_t)axis) {
      outSize = checked_mul_acc_size(
          outSize, (size_t)hDims[i], "torchlean_cuda_buffer_reduce_sum_axis: output shape overflow");
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
  dDims = torchlean_cuda_scratch_alloc<uint32_t>(rank, "cudaMalloc dims failed");
  const size_t dimBytes = checked_bytes_size(
      rank, sizeof(uint32_t), "torchlean_cuda_buffer_swap_adjacent_at_depth: dims byte overflow");
  cudaError_t copyDims = cudaMemcpy(dDims, hDims, dimBytes, cudaMemcpyHostToDevice);
  if (copyDims != cudaSuccess) {
    torchlean_cuda_scratch_free(&dDims, rank, "cudaFree swapAdjacentAtDepth dims after copy failed");
    checkCuda(copyDims, "cudaMemcpy dims failed");
  }

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

  torchlean_cuda_scratch_free(&dDims, rank, "cudaFree reduceSumAxis dims failed");
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

  uint32_t* dIdx = upload_nat_indices(
      IdxObj, K,
      "torchlean_cuda_buffer_gather_rows: bad index Nat",
      "cudaMalloc indices failed",
      "cudaMemcpy indices failed");

  dim3 blocks = blocks_for(outSz);
  dim3 threads = dim3(kBlockSize);
  gather_rows_f32<<<blocks, threads>>>(m->data, rows, cols, dIdx, k, out->data);
  checkCuda(cudaGetLastError(), "cuda gatherRows kernel launch failed");

  torchlean_cuda_scratch_free(&dIdx, K, "cudaFree gatherScalar indices failed");
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

  uint32_t* dIdx = upload_nat_indices(
      IdxObj, K,
      "torchlean_cuda_buffer_scatter_add_rows: bad index Nat",
      "cudaMalloc indices failed",
      "cudaMemcpy indices failed");

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

  torchlean_cuda_scratch_free(&dIdx, K, "cudaFree scatterAddRows indices failed");
  return torchlean_cuda_buffer_box(out);
}
