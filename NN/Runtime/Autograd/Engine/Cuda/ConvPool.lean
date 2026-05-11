/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

CUDA FFI: naive Float32 kernels for Conv + pooling (forward/backward).

Build:
  lake build -K cuda=true

Notes:
- APIs operate on `Cuda.Buffer` (opaque float32 device buffer).
- When built without CUDA (`lake build` default), the stub implementation runs on CPU for
  portability.
- Layout conventions are channels-first and row-major within each tensor:
  - input:  (inC, spatial...)
  - kernel: (outC, inC, kernelSpatial...)
  - bias:   (outC)
  - output: (outC, outSpatial...)
  - pooling output: (inC, outSpatial...)
- The "ND" entrypoints (`torchlean_cuda_conv_fwd`, etc.) take per-axis shape/stride/padding
  as `Array Nat`, with `rank ≤ 8`.
- The `*2d*` entrypoints are concise rank-2 convenience wrappers with scalar stride/padding.
-/

module


public import NN.Runtime.Autograd.Engine.Cuda.Buffer

/-!
# CUDA Conv/Pool FFI

Foreign-function declarations for TorchLean's float32 convolution and pooling kernels. The real
CUDA implementation lives in `csrc/cuda/conv_pool/`; CPU stubs with the same symbols are used when
TorchLean is built without `-K cuda=true`.

All buffers are contiguous `Cuda.Buffer` values and shape/stride/padding metadata is passed
explicitly through the FFI boundary.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

/-- Float32 conv2d forward (device `Buffer` inputs/outputs). -/
@[extern "torchlean_cuda_conv2d_fwd"]
opaque torchleanConv2dFwdCuda
    (input kernel bias : Buffer)
    (inC inH inW outC kH kW stride padding : UInt32) : Buffer

/-- Float32 conv2d backward: returns `(dKernel, dBias, dInput)` device buffers. -/
@[extern "torchlean_cuda_conv2d_bwd"]
opaque torchleanConv2dBwdCuda
    (input kernel gradOutput : Buffer)
    (inC inH inW outC kH kW stride padding : UInt32) : Buffer × Buffer × Buffer

/-- Float32 conv-transpose2d forward (device `Buffer` inputs/outputs). -/
@[extern "torchlean_cuda_convtranspose2d_fwd"]
opaque torchleanConvTranspose2dFwdCuda
    (input kernel bias : Buffer)
    (inC inH inW outC kH kW stride padding : UInt32) : Buffer

/-- Float32 conv-transpose2d backward: returns `(dKernel, dBias, dInput)` device buffers. -/
@[extern "torchlean_cuda_convtranspose2d_bwd"]
opaque torchleanConvTranspose2dBwdCuda
    (input kernel gradOutput : Buffer)
    (inC inH inW outC kH kW stride padding : UInt32) : Buffer × Buffer × Buffer

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
@[extern "torchlean_cuda_convtranspose_fwd"]
opaque torchleanConvTransposeFwdCuda
    (input kernel bias : Buffer)
    (inSpatial kernelSpatial stride padding : Array Nat)
    (inC outC : UInt32) : Buffer

/--
Float32 N-D transposed convolution backward.

Returns `(dKernel, dBias, dInput)` as device buffers.
Array conventions match `torchleanConvTransposeFwdCuda`.
-/
@[extern "torchlean_cuda_convtranspose_bwd"]
opaque torchleanConvTransposeBwdCuda
    (input kernel gradOutput : Buffer)
    (inSpatial kernelSpatial stride padding : Array Nat)
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
@[extern "torchlean_cuda_conv_fwd"]
opaque torchleanConvFwdCuda
    (input kernel bias : Buffer)
    (inSpatial kernelSpatial stride padding : Array Nat)
    (inC outC : UInt32) : Buffer

/--
Float32 N-D convolution backward.

Returns `(dKernel, dBias, dInput)` as device buffers.
Array conventions match `torchleanConvFwdCuda`.
-/
@[extern "torchlean_cuda_conv_bwd"]
opaque torchleanConvBwdCuda
    (input kernel gradOutput : Buffer)
    (inSpatial kernelSpatial stride padding : Array Nat)
    (inC outC : UInt32) : Buffer × Buffer × Buffer

/-- Float32 max-pool2d forward (channels preserved). -/
@[extern "torchlean_cuda_maxpool2d_fwd"]
opaque torchleanMaxPool2dFwdCuda
    (input : Buffer)
    (inC inH inW kH kW stride padding : UInt32) : Buffer

/-- Float32 max-pool2d backward: returns `dInput`. -/
@[extern "torchlean_cuda_maxpool2d_bwd"]
opaque torchleanMaxPool2dBwdCuda
    (input gradOutput : Buffer)
    (inC inH inW kH kW stride padding : UInt32) : Buffer

/-- Float32 N-D max-pooling forward (channels preserved). -/
@[extern "torchlean_cuda_maxpool_fwd"]
opaque torchleanMaxPoolFwdCuda
    (input : Buffer)
    (inSpatial kernel stride padding : Array Nat)
    (inC : UInt32) : Buffer

/-- Float32 N-D max-pooling backward: returns `dInput`. -/
@[extern "torchlean_cuda_maxpool_bwd"]
opaque torchleanMaxPoolBwdCuda
    (input gradOutput : Buffer)
    (inSpatial kernel stride padding : Array Nat)
    (inC : UInt32) : Buffer

/-- Float32 avg-pool2d forward (channels preserved). -/
@[extern "torchlean_cuda_avgpool2d_fwd"]
opaque torchleanAvgPool2dFwdCuda
    (input : Buffer)
    (inC inH inW kH kW stride padding : UInt32) : Buffer

/-- Float32 avg-pool2d backward: returns `dInput`. -/
@[extern "torchlean_cuda_avgpool2d_bwd"]
opaque torchleanAvgPool2dBwdCuda
    (gradOutput : Buffer)
    (inC inH inW kH kW stride padding : UInt32) : Buffer

/-- Float32 N-D avg-pooling forward (channels preserved). -/
@[extern "torchlean_cuda_avgpool_fwd"]
opaque torchleanAvgPoolFwdCuda
    (input : Buffer)
    (inSpatial kernel stride padding : Array Nat)
    (inC : UInt32) : Buffer

/-- Float32 N-D avg-pooling backward: returns `dInput`. -/
@[extern "torchlean_cuda_avgpool_bwd"]
opaque torchleanAvgPoolBwdCuda
    (gradOutput : Buffer)
    (inSpatial kernel stride padding : Array Nat)
    (inC : UInt32) : Buffer

/--
Float32 smooth max-pool2d (log-sum-exp surrogate) forward.

This matches `Spec.smooth_max_pool2d_spec` for `Float`:
`y = log(sum(exp(beta*x))) / beta` computed per window, with `beta ≠ 0`.
-/
@[extern "torchlean_cuda_smooth_maxpool2d_fwd"]
opaque torchleanSmoothMaxPool2dFwdCuda
    (input : Buffer) (beta : Float)
    (inC inH inW kH kW stride padding : UInt32) : Buffer

/--
Float32 smooth max-pool2d backward: returns `dInput`.

VJP matches `Spec.smooth_max_pool2d_backward_spec` for `Float`:
`dx += dOut * exp(beta*x)/sum(exp(beta*x))` within each window.
-/
@[extern "torchlean_cuda_smooth_maxpool2d_bwd"]
opaque torchleanSmoothMaxPool2dBwdCuda
    (input gradOutput : Buffer) (beta : Float)
    (inC inH inW kH kW stride padding : UInt32) : Buffer

/-- Float32 N-D smooth max-pooling forward (channels preserved). -/
@[extern "torchlean_cuda_smooth_maxpool_fwd"]
opaque torchleanSmoothMaxPoolFwdCuda
    (input : Buffer) (beta : Float)
    (inSpatial kernel stride padding : Array Nat)
    (inC : UInt32) : Buffer

/-- Float32 N-D smooth max-pooling backward: returns `dInput`. -/
@[extern "torchlean_cuda_smooth_maxpool_bwd"]
opaque torchleanSmoothMaxPoolBwdCuda
    (input gradOutput : Buffer) (beta : Float)
    (inSpatial kernel stride padding : Array Nat)
    (inC : UInt32) : Buffer

end Cuda
end Autograd
end Runtime
