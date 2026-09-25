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

// CPU version of the conv/pool FFI symbols.
// This keeps non-CUDA builds working and gives the tests a plain reference for edge cases.

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

static inline int input_index_from_window(
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
    if (!window_input_coord(outCoord[ax], kCoord[ax], stride[ax], padding[ax],
                            inSpatial[ax], &pos)) {
      return 0;
    }
    inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)pos;
  }
  *inIdxOut = inIdx;
  return 1;
}

// -------------------------
// N-D conv + pooling exported functions (CPU stub)
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

  const size_t inElems = checked_channel_spatial_size(inC, inSpatialSize, "torchlean_cuda_conv_pool: input size overflow");
  const size_t kElems = checked_conv_kernel_size(outC, inC, kSpatialSize, "torchlean_cuda_conv_pool: kernel size overflow");
  const size_t bElems = (size_t)outC;
  const size_t outElems = checked_channel_spatial_size(outC, outSpatialSize, "torchlean_cuda_conv_pool: output size overflow");

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
            uint32_t pos;
            if (!window_input_coord(outCoord[ax], kCoord[ax], stride[ax], padding[ax],
                                    inSpatial[ax], &pos)) {
              ok = 0;
              break;
            }
            inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)pos;
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

  const size_t inElems = checked_channel_spatial_size(inC, inSpatialSize, "torchlean_cuda_conv_pool: input size overflow");
  const size_t kElems = checked_conv_kernel_size(outC, inC, kSpatialSize, "torchlean_cuda_conv_pool: kernel size overflow");
  const size_t outElems = checked_channel_spatial_size(outC, outSpatialSize, "torchlean_cuda_conv_pool: output size overflow");

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
            uint32_t pos;
            if (!window_input_coord(outCoord[ax], kCoord[ax], stride[ax], padding[ax],
                                    inSpatial[ax], &pos)) {
              ok = 0;
              break;
            }
            inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)pos;
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

  return torchlean_cuda_box_three_buffers(dKernel, dBias, dInput);
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

  const size_t inElems = checked_channel_spatial_size(inC, inSpatialSize, "torchlean_cuda_conv_pool: input size overflow");
  const size_t kElems = checked_conv_kernel_size(inC, outC, kSpatialSize, "torchlean_cuda_conv_pool: kernel size overflow");
  const size_t bElems = (size_t)outC;
  const size_t outElems = checked_channel_spatial_size(outC, outSpatialSize, "torchlean_cuda_conv_pool: output size overflow");

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
            uint32_t pos;
            if (!window_output_coord(outCoord[ax], kCoord[ax], stride[ax], padding[ax],
                                     inSpatial[ax], &pos)) {
              ok = 0;
              break;
            }
            inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)pos;
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

  const size_t inElems = checked_channel_spatial_size(inC, inSpatialSize, "torchlean_cuda_conv_pool: input size overflow");
  const size_t kElems = checked_conv_kernel_size(inC, outC, kSpatialSize, "torchlean_cuda_conv_pool: kernel size overflow");
  const size_t outElems = checked_channel_spatial_size(outC, outSpatialSize, "torchlean_cuda_conv_pool: output size overflow");

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
            uint32_t pos;
            if (!window_output_coord(outCoord[ax], kCoord[ax], stride[ax], padding[ax],
                                     inSpatial[ax], &pos)) {
              ok = 0;
              break;
            }
            inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)pos;
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

  return torchlean_cuda_box_three_buffers(dKernel, dBias, dInput);
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
    outSpatial[ax] = poolOutDim(inSpatial[ax], kSpatial[ax], stride[ax], padding[ax]);
  }

  const size_t inSpatialSize = prod_u32(inSpatial, rank);
  const size_t kSpatialSize = prod_u32(kSpatial, rank);
  const size_t outSpatialSize = prod_u32(outSpatial, rank);

  const size_t inElems = checked_channel_spatial_size(inC, inSpatialSize, "torchlean_cuda_conv_pool: input size overflow");
  const size_t outElems = checked_channel_spatial_size(inC, outSpatialSize, "torchlean_cuda_conv_pool: output size overflow");

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

      float best = 0.0f;
      int bestValid = 0;
      for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
        unflatten_coords(kCoord, kSpatial, rank, kIdx);
        size_t inIdx = 0;
        int ok =
            input_index_from_window(c, outCoord, kCoord, inSpatial, stride, padding, rank, &inIdx);

        if (ok) {
          const float v = input->data[inIdx];
          if (!bestValid || v > best) {
            best = v;
            bestValid = 1;
          }
        }
      }

      out->data[(size_t)c * outSpatialSize + outIdx] = bestValid ? best : 0.0f;
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
    outSpatial[ax] = poolOutDim(inSpatial[ax], kSpatial[ax], stride[ax], padding[ax]);
  }

  const size_t inSpatialSize = prod_u32(inSpatial, rank);
  const size_t kSpatialSize = prod_u32(kSpatial, rank);
  const size_t outSpatialSize = prod_u32(outSpatial, rank);

  const size_t inElems = checked_channel_spatial_size(inC, inSpatialSize, "torchlean_cuda_conv_pool: input size overflow");
  const size_t outElems = checked_channel_spatial_size(inC, outSpatialSize, "torchlean_cuda_conv_pool: output size overflow");

  checkBufSize(input, inElems, "torchlean_cuda_maxpool_bwd_stub: input.size mismatch");
  checkBufSize(gradOutput, outElems, "torchlean_cuda_maxpool_bwd_stub: gradOutput.size mismatch");

  torchlean_cuda_buffer* dInput = torchlean_cuda_buffer_alloc(inElems);
  const int det = torchlean_cpu_deterministic_reductions();
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

            float best = 0.0f;
            int bestValid = 0;
            for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
              unflatten_coords(kCoord, kSpatial, rank, kIdx);

              int ok = 1;
              for (int ax = 0; ax < rank; ++ax) {
                uint32_t pos;
                if (!window_input_coord(outCoord[ax], kCoord[ax], stride[ax], padding[ax],
                                        inSpatial[ax], &pos)) {
                  ok = 0;
                  break;
                }
                candCoord[ax] = pos;
              }

              if (ok) {
                size_t inIdx = (size_t)c;
                for (int ax = 0; ax < rank; ++ax) {
                  inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)candCoord[ax];
                }
                const float v = input->data[inIdx];
                if (!bestValid || v > best) {
                  best = v;
                  bestValid = 1;
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

        float best = 0.0f;
        int bestValid = 0;

        for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
          unflatten_coords(kCoord, kSpatial, rank, kIdx);

          int ok = 1;
          for (int ax = 0; ax < rank; ++ax) {
            uint32_t pos;
            if (!window_input_coord(outCoord[ax], kCoord[ax], stride[ax], padding[ax],
                                    inSpatial[ax], &pos)) {
              ok = 0;
              break;
            }
            candCoord[ax] = pos;
          }

          if (ok) {
            size_t inIdx = (size_t)c;
            for (int ax = 0; ax < rank; ++ax) {
              inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)candCoord[ax];
            }
            const float v = input->data[inIdx];
            if (!bestValid || v > best) {
              best = v;
              bestValid = 1;
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
    outSpatial[ax] = poolOutDim(inSpatial[ax], kSpatial[ax], stride[ax], padding[ax]);
  }

  const size_t inSpatialSize = prod_u32(inSpatial, rank);
  const size_t kSpatialSize = prod_u32(kSpatial, rank);
  const size_t outSpatialSize = prod_u32(outSpatial, rank);

  const size_t inElems = checked_channel_spatial_size(inC, inSpatialSize, "torchlean_cuda_conv_pool: input size overflow");
  const size_t outElems = checked_channel_spatial_size(inC, outSpatialSize, "torchlean_cuda_conv_pool: output size overflow");

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
        size_t inIdx = 0;
        int ok =
            input_index_from_window(c, outCoord, kCoord, inSpatial, stride, padding, rank, &inIdx);

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
    outSpatial[ax] = poolOutDim(inSpatial[ax], kSpatial[ax], stride[ax], padding[ax]);
  }

  const size_t inSpatialSize = prod_u32(inSpatial, rank);
  const size_t kSpatialSize = prod_u32(kSpatial, rank);
  const size_t outSpatialSize = prod_u32(outSpatial, rank);

  const size_t inElems = checked_channel_spatial_size(inC, inSpatialSize, "torchlean_cuda_conv_pool: input size overflow");
  const size_t outElems = checked_channel_spatial_size(inC, outSpatialSize, "torchlean_cuda_conv_pool: output size overflow");

  checkBufSize(gradOutput, outElems, "torchlean_cuda_avgpool_bwd_stub: gradOutput.size mismatch");

  torchlean_cuda_buffer* dInput = torchlean_cuda_buffer_alloc(inElems);
  const float denom = (float)kSpatialSize;
  const int det = torchlean_cpu_deterministic_reductions();
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
          size_t inIdx = 0;
          int ok = input_index_from_window(c, outCoord, kCoord, inSpatial, stride, padding, rank,
                                           &inIdx);
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
    outSpatial[ax] = poolOutDim(inSpatial[ax], kSpatial[ax], stride[ax], padding[ax]);
  }

  const size_t inSpatialSize = prod_u32(inSpatial, rank);
  const size_t kSpatialSize = prod_u32(kSpatial, rank);
  const size_t outSpatialSize = prod_u32(outSpatial, rank);

  const size_t inElems = checked_channel_spatial_size(inC, inSpatialSize, "torchlean_cuda_conv_pool: input size overflow");
  const size_t outElems = checked_channel_spatial_size(inC, outSpatialSize, "torchlean_cuda_conv_pool: output size overflow");

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

      // Match the CUDA kernel's input-space shift; multiplying beta only after subtraction avoids
      // overflow for large finite values.
      float pivot = (betaF > 0.0f) ? -INFINITY : INFINITY;
      for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
        unflatten_coords(kCoord, kSpatial, rank, kIdx);

        int ok = 1;
        size_t inIdx = (size_t)c;
        for (int ax = 0; ax < rank; ++ax) {
          uint32_t pos;
          if (!window_input_coord(outCoord[ax], kCoord[ax], stride[ax], padding[ax],
                                  inSpatial[ax], &pos)) {
            ok = 0;
            break;
          }
          inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)pos;
        }

        float v = 0.0f;
        if (ok) {
          v = input->data[inIdx];
        }
        pivot = (betaF > 0.0f) ? fmaxf(pivot, v) : fminf(pivot, v);
      }

      float sumExp = 0.0f;
      for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
        unflatten_coords(kCoord, kSpatial, rank, kIdx);

        int ok = 1;
        size_t inIdx = (size_t)c;
        for (int ax = 0; ax < rank; ++ax) {
          uint32_t pos;
          if (!window_input_coord(outCoord[ax], kCoord[ax], stride[ax], padding[ax],
                                  inSpatial[ax], &pos)) {
            ok = 0;
            break;
          }
          inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)pos;
        }

        float v = 0.0f;
        if (ok) {
          v = input->data[inIdx];
        }
        sumExp += expf(betaF * (v - pivot));
      }

      out->data[(size_t)c * outSpatialSize + outIdx] = pivot + logf(sumExp) / betaF;
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
    outSpatial[ax] = poolOutDim(inSpatial[ax], kSpatial[ax], stride[ax], padding[ax]);
  }

  const size_t inSpatialSize = prod_u32(inSpatial, rank);
  const size_t kSpatialSize = prod_u32(kSpatial, rank);
  const size_t outSpatialSize = prod_u32(outSpatial, rank);

  const size_t inElems = checked_channel_spatial_size(inC, inSpatialSize, "torchlean_cuda_conv_pool: input size overflow");
  const size_t outElems = checked_channel_spatial_size(inC, outSpatialSize, "torchlean_cuda_conv_pool: output size overflow");

  checkBufSize(input, inElems, "torchlean_cuda_smooth_maxpool_bwd_stub: input.size mismatch");
  checkBufSize(gradOutput, outElems, "torchlean_cuda_smooth_maxpool_bwd_stub: gradOutput.size mismatch");

  torchlean_cuda_buffer* dInput = torchlean_cuda_buffer_alloc(inElems);
  const float betaF =
      checked_smoothmax_beta(beta, "torchlean_cuda_smooth_maxpool_bwd_stub: beta must be finite and nonzero");
  const int det = torchlean_cpu_deterministic_reductions();
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

            /* Recompute the sign-aware pivot for this contributing output so the
               deterministic backward pass uses the same stable softmax weights. */
            float pivot = (betaF > 0.0f) ? -INFINITY : INFINITY;
            for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
              unflatten_coords(kCoord, kSpatial, rank, kIdx);

              int ok = 1;
              size_t inIdx = (size_t)c;
              for (int ax = 0; ax < rank; ++ax) {
                uint32_t pos;
                if (!window_input_coord(outCoord[ax], kCoord[ax], stride[ax], padding[ax],
                                        inSpatial[ax], &pos)) {
                  ok = 0;
                  break;
                }
                inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)pos;
              }

              float v = 0.0f;
              if (ok) {
                v = input->data[inIdx];
              }
              pivot = (betaF > 0.0f) ? fmaxf(pivot, v) : fminf(pivot, v);
            }

            float sumExp = 0.0f;
            for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
              unflatten_coords(kCoord, kSpatial, rank, kIdx);

              int ok = 1;
              size_t inIdx = (size_t)c;
              for (int ax = 0; ax < rank; ++ax) {
                uint32_t pos;
                if (!window_input_coord(outCoord[ax], kCoord[ax], stride[ax], padding[ax],
                                        inSpatial[ax], &pos)) {
                  ok = 0;
                  break;
                }
                inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)pos;
              }

              float v = 0.0f;
              if (ok) {
                v = input->data[inIdx];
              }
              sumExp += expf(betaF * (v - pivot));
            }

            const float w = expf(betaF * (vSelf - pivot)) / sumExp;
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

        // Recompute the forward input-space pivot before evaluating the gradient weights.
        float pivot = (betaF > 0.0f) ? -INFINITY : INFINITY;
        for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
          unflatten_coords(kCoord, kSpatial, rank, kIdx);

          int ok = 1;
          size_t inIdx = (size_t)c;
          for (int ax = 0; ax < rank; ++ax) {
            uint32_t pos;
            if (!window_input_coord(outCoord[ax], kCoord[ax], stride[ax], padding[ax],
                                    inSpatial[ax], &pos)) {
              ok = 0;
              break;
            }
            inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)pos;
          }

          float v = 0.0f;
          if (ok) {
            v = input->data[inIdx];
          }
          pivot = (betaF > 0.0f) ? fmaxf(pivot, v) : fminf(pivot, v);
        }

        float sumExp = 0.0f;
        for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
          unflatten_coords(kCoord, kSpatial, rank, kIdx);

          int ok = 1;
          size_t inIdx = (size_t)c;
          for (int ax = 0; ax < rank; ++ax) {
            uint32_t pos;
            if (!window_input_coord(outCoord[ax], kCoord[ax], stride[ax], padding[ax],
                                    inSpatial[ax], &pos)) {
              ok = 0;
              break;
            }
            inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)pos;
          }

          float v = 0.0f;
          if (ok) {
            v = input->data[inIdx];
          }
          sumExp += expf(betaF * (v - pivot));
        }

        for (size_t kIdx = 0; kIdx < kSpatialSize; ++kIdx) {
          unflatten_coords(kCoord, kSpatial, rank, kIdx);

          int ok = 1;
          size_t inIdx = (size_t)c;
          for (int ax = 0; ax < rank; ++ax) {
            uint32_t pos;
            if (!window_input_coord(outCoord[ax], kCoord[ax], stride[ax], padding[ax],
                                    inSpatial[ax], &pos)) {
              ok = 0;
              break;
            }
            inIdx = inIdx * (size_t)inSpatial[ax] + (size_t)pos;
          }
          if (!ok) continue;

          float v = input->data[inIdx];
          float e = expf(betaF * (v - pivot));
          float w = e / sumExp;
          dInput->data[inIdx] += g * w;
        }
      }
    }
  }

  return torchlean_cuda_buffer_box(dInput);
}
