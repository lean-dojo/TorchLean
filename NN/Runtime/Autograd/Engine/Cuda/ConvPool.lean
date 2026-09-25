/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

CUDA FFI: naive Float32 kernels for Conv + pooling (forward/backward).

Build:
  lake -R -K cuda=true build

Notes:
- APIs operate on `Cuda.Buffer` (opaque float32 device buffer).
- When built without LibTorch (`lake build` default), every call fails with an error that says
  how to rebuild with `-K cuda=true`.
- Layout conventions are channels-first and row-major within each tensor:
  - input:  (inC, spatial...)
  - kernel: (outC, inC, kernelSpatial...)
  - bias:   (outC)
  - output: (outC, outSpatial...)
  - pooling output: (inC, outSpatial...)
- The "ND" entrypoints (`torchlean_cuda_conv_fwd`, etc.) take per-axis shape/stride/padding
  as `Array Nat`, with `rank ≤ 8`.
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Trusted

/-!
# CUDA Conv/Pool FFI

Foreign-function declarations for TorchLean's float32 convolution and pooling kernels,
implemented in `csrc/libtorch/conv_pool.cpp`.

All buffers are contiguous `Cuda.Buffer` values and shape/stride/padding metadata is passed
explicitly through the FFI boundary.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

/--
Float32 N-D transposed convolution forward (channels-first, no batch).

Shapes/parameters:
- `inSpatial`: length `d` (input spatial dims)
- `kernelSpatial`: length `d` (kernel window)
- `stride`: length `d`
- `padding`: length `d`

All arrays must have the same length `d ≤ 8`.

Layout conventions:
- input:  `(inC, spatial...)`
- kernel: `(inC, outC, kernelSpatial...)`
- bias:   `(outC)`
- output: `(outC, outSpatial...)`, where
  `outSpatial[i] = (inSpatial[i] - 1) * stride[i] - 2*padding[i] + kernelSpatial[i]`.
-/
@[never_extract, extern "torchlean_cuda_convtranspose_fwd"]
opaque torchleanConvTransposeFwdCuda
    (input kernel bias : @& Buffer)
    (inSpatial kernelSpatial stride padding : @& Array Nat)
    (inC outC : UInt32) : Buffer

/--
Float32 N-D transposed convolution backward.

Returns `(dKernel, dBias, dInput)` as device buffers.
Array conventions match `torchleanConvTransposeFwdCuda`.
-/
@[never_extract, extern "torchlean_cuda_convtranspose_bwd"]
opaque torchleanConvTransposeBwdCuda
    (input kernel gradOutput : @& Buffer)
    (inSpatial kernelSpatial stride padding : @& Array Nat)
    (inC outC : UInt32) : Buffer × Buffer × Buffer

/--
Float32 N-D convolution forward (channels-first, no batch).

Shapes/parameters:
- `inSpatial`: length `d` (spatial dims)
- `kernelSpatial`: length `d` (kernel window)
- `stride`: length `d`
- `padding`: length `d`

All arrays must have the same length `d ≤ 8`.
-/
@[never_extract, extern "torchlean_cuda_conv_fwd"]
opaque torchleanConvFwdCuda
    (input kernel bias : @& Buffer)
    (inSpatial kernelSpatial stride padding : @& Array Nat)
    (inC outC : UInt32) : Buffer

/--
Float32 N-D convolution backward.

Returns `(dKernel, dBias, dInput)` as device buffers.
Array conventions match `torchleanConvFwdCuda`.
-/
@[never_extract, extern "torchlean_cuda_conv_bwd"]
opaque torchleanConvBwdCuda
    (input kernel gradOutput : @& Buffer)
    (inSpatial kernelSpatial stride padding : @& Array Nat)
    (inC outC : UInt32) : Buffer × Buffer × Buffer

/-- Float32 N-D max-pooling forward (channels preserved). -/
@[never_extract, extern "torchlean_cuda_maxpool_fwd"]
opaque torchleanMaxPoolFwdCuda
    (input : @& Buffer)
    (inSpatial kernel stride padding : @& Array Nat)
    (inC : UInt32) : Buffer

/-- Float32 N-D max-pooling backward: returns `dInput`. -/
@[never_extract, extern "torchlean_cuda_maxpool_bwd"]
opaque torchleanMaxPoolBwdCuda
    (input gradOutput : @& Buffer)
    (inSpatial kernel stride padding : @& Array Nat)
    (inC : UInt32) : Buffer

/-- Float32 N-D avg-pooling forward (channels preserved). -/
@[never_extract, extern "torchlean_cuda_avgpool_fwd"]
opaque torchleanAvgPoolFwdCuda
    (input : @& Buffer)
    (inSpatial kernel stride padding : @& Array Nat)
    (inC : UInt32) : Buffer

/-- Float32 N-D avg-pooling backward: returns `dInput`. -/
@[never_extract, extern "torchlean_cuda_avgpool_bwd"]
opaque torchleanAvgPoolBwdCuda
    (gradOutput : @& Buffer)
    (inSpatial kernel stride padding : @& Array Nat)
    (inC : UInt32) : Buffer

/--
Float32 N-D smooth max-pooling forward with channels preserved.

The native implementation requires finite nonzero `beta` and uses a maximum input pivot for
positive `beta` or a minimum input pivot for negative `beta`, matching the two-dimensional path.
-/
@[never_extract, extern "torchlean_cuda_smooth_maxpool_fwd"]
opaque torchleanSmoothMaxPoolFwdCuda
    (input : @& Buffer) (beta : Float)
    (inSpatial kernel stride padding : @& Array Nat)
    (inC : UInt32) : Buffer

/--
Float32 N-D smooth max-pooling backward, returning `dInput`.

It shares the forward operation's finite nonzero-`beta` contract and sign-aware max/min input pivot.
-/
@[never_extract, extern "torchlean_cuda_smooth_maxpool_bwd"]
opaque torchleanSmoothMaxPoolBwdCuda
    (input gradOutput : @& Buffer) (beta : Float)
    (inSpatial kernel stride padding : @& Array Nat)
    (inC : UInt32) : Buffer

end Cuda
end Autograd
end Runtime
