/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

CUDA FFI: float32 and float64 transposed convolution and pooling through LibTorch.

Build:
  scripts/lake.sh -Kcuda=true build

Notes:
- APIs operate on `LibTorch.Buffer` (opaque float32 or float64 device buffer).
- In the default build, without LibTorch, every call stops the program with a message that says
  how to rebuild with `-K cuda=true`.
- Layout conventions are channels-first and row-major within each tensor:
  - input:  (inC, spatial...)
  - kernel: (inC, outC, kernelSpatial...)
  - bias:   (outC)
  - output: (outC, outSpatial...)
  - pooling output: (inC, outSpatial...)
- The "ND" entry points (`torchlean_cuda_convtranspose_fwd`, etc.) take per-axis
  shape/stride/padding as `Array Nat`, with one to three spatial dimensions, matching LibTorch.
-/

module

public import TorchLean.CUDA.LibTorch.Trusted

/-!
# CUDA Conv/Pool FFI

Foreign-function declarations for TorchLean's transposed convolution and pooling over float32 or
float64 buffers, forward and backward, implemented in `csrc/libtorch/conv_pool.cpp`. Inputs and
outputs are unbatched channels-first row-major tensors in `LibTorch.Buffer` values, and every
result has the dtype of its input. Spatial extents, kernel extents, strides and padding cross the
FFI boundary as `Array Nat` values of one common length from one to three, with positive kernel
extents and strides. The grouped convolution over a batch is `Buffer.conv` with
`Buffer.convBackward`.

Each pooling output axis has the extent that `Pool.outSpatial` gives it: the floor-mode
window count, or zero for an empty input axis or for padding above half the kernel.
-/

@[expose] public section

namespace TorchLean
namespace LibTorch

/--
Transposed convolution forward of one unbatched channels-first input, with one group, unit dilation
and no output padding.

Shapes/parameters:
- `inSpatial`: length `d` (input spatial dims)
- `kernelSpatial`: length `d` (kernel window)
- `stride`: length `d`
- `padding`: length `d`

All arrays must have the same length `1 ≤ d ≤ 3`.

Layout conventions:
- input:  `(inC, spatial...)`
- kernel: `(inC, outC, kernelSpatial...)`
- bias:   `(outC)`
- output: `(outC, outSpatial...)`, where
  `outSpatial[i] = (inSpatial[i] - 1) * stride[i] + kernelSpatial[i] - 2*padding[i]` in `Nat`,
  and `outSpatial[i] = 0` when `inSpatial[i] = 0`.
-/
@[never_extract, extern "torchlean_cuda_convtranspose_fwd"]
opaque Buffer.convTranspose
    (input kernel bias : @& Buffer)
    (inSpatial kernelSpatial stride padding : @& Array Nat)
    (inC outC : UInt32) : Buffer

/--
Transposed convolution backward: the gradients `(dKernel, dBias, dInput)` for the output gradient
`gradOutput`, with the arrays and layouts of `convTranspose`.
-/
@[never_extract, extern "torchlean_cuda_convtranspose_bwd"]
opaque Buffer.convTransposeBackward
    (input kernel gradOutput : @& Buffer)
    (inSpatial kernelSpatial stride padding : @& Array Nat)
    (inC outC : UInt32) : Buffer × Buffer × Buffer

/-- Max-pooling forward, pooling each of the `inC` channels separately. -/
@[never_extract, extern "torchlean_cuda_maxpool_fwd"]
opaque Buffer.maxPool
    (input : @& Buffer)
    (inSpatial kernel stride padding : @& Array Nat)
    (inC : UInt32) : Buffer

/-- Max-pooling backward, returning `dInput`. -/
@[never_extract, extern "torchlean_cuda_maxpool_bwd"]
opaque Buffer.maxPoolBackward
    (input gradOutput : @& Buffer)
    (inSpatial kernel stride padding : @& Array Nat)
    (inC : UInt32) : Buffer

/-- Average-pooling forward, pooling each of the `inC` channels separately. Each window sum is
divided by the full kernel volume, padding included. -/
@[never_extract, extern "torchlean_cuda_avgpool_fwd"]
opaque Buffer.avgPool
    (input : @& Buffer)
    (inSpatial kernel stride padding : @& Array Nat)
    (inC : UInt32) : Buffer

/-- Average-pooling backward, returning `dInput`. The gradient depends only on `gradOutput` and the
geometry. -/
@[never_extract, extern "torchlean_cuda_avgpool_bwd"]
opaque Buffer.avgPoolBackward
    (gradOutput : @& Buffer)
    (inSpatial kernel stride padding : @& Array Nat)
    (inC : UInt32) : Buffer

/--
Smooth max-pooling forward, pooling each of the `inC` channels separately.

The native implementation requires finite nonzero `beta` and uses a maximum input pivot for
positive `beta` or a minimum input pivot for negative `beta`, matching the two-dimensional path,
and shifts by zero at an infinite pivot, as `TorchLean.Tensor.smoothMaxPoolShift` does.
-/
@[never_extract, extern "torchlean_cuda_smooth_maxpool_fwd"]
opaque Buffer.smoothMaxPool
    (input : @& Buffer) (beta : Float)
    (inSpatial kernel stride padding : @& Array Nat)
    (inC : UInt32) : Buffer

/--
Smooth max-pooling backward, returning `dInput`.

It shares the forward operation's finite nonzero-`beta` contract and sign-aware max/min input
pivot, which it keeps at an infinite pivot, so the weights of that window are NaN.
-/
@[never_extract, extern "torchlean_cuda_smooth_maxpool_bwd"]
opaque Buffer.smoothMaxPoolBackward
    (input gradOutput : @& Buffer) (beta : Float)
    (inSpatial kernel stride padding : @& Array Nat)
    (inC : UInt32) : Buffer

end LibTorch
end TorchLean
