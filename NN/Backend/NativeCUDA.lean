/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Capsule

/-!
# Native CUDA Backend Capsules

Capsule metadata for TorchLean's native CUDA runtime provider.

These capsules describe kernels that currently live under `csrc/cuda/**` and are exposed to Lean
through `NN.Runtime.Autograd.Engine.Cuda.*`. The C/CUDA source still owns the implementation; this
module gives the planner a typed, inspectable contract layer for those implementation choices.
-/

@[expose] public section

namespace NN
namespace Backend
namespace NativeCUDA

/-- Build a checked native-CUDA capsule with explicit FFI, value, VJP, and layout contracts. -/
def nativeCapsule
    (name : String) (op : BackendOp) (valueSummary vjpSummary : String)
    (vjpMode : VJPMode := .backendVJP) : KernelCapsule :=
  { name
    op
    provider := .nativeCuda
    device := .cuda
    trustLevel := .checked
    supportsForward := true
    vjpMode
    shapeContract :=
      { claim := .shapeSafety op
        summary := "Inputs and outputs are checked against explicit UInt32 dimensions."
        evidence := .runtimeGuard "CUDA FFI size/rank checks at the Lean/native boundary" }
    layoutContract :=
      { claim := .layoutCompatibility op .flatRowMajor
        summary := "CUDA buffers are contiguous flat float32 buffers."
        evidence := .runtimeGuard "flat row-major Cuda.Buffer layout checks" }
    valueContract :=
      { claim := .valueRefinement op
        summary := valueSummary
        evidence := .testSuite "NN.Tests.Runtime.Cuda.Suite" }
    vjpContract :=
      match vjpMode with
      | .none => ContractDescriptor.vjpUnavailable op vjpSummary
      | mode => ContractDescriptor.tested
          (.vjpRefinement op mode) vjpSummary "NN.Tests.Runtime.Cuda.Suite"
    numericalPolicy := { reduction := .notApplicable } }

/-- Build the standard native-CUDA capsule for a pointwise operation. -/
def nativePointwiseCapsule (op : BackendOp) : KernelCapsule :=
  nativeCapsule
    s!"native_cuda.{op.name}"
    op
    s!"Native CUDA `{op.name}` follows the pointwise tensor contract."
    s!"Native CUDA `{op.name}` VJP is checked through runtime autograd tests."

/-- Build a native-CUDA reduction capsule with implementation-selected parallel reduction order. -/
def nativeReductionCapsule (op : BackendOp) : KernelCapsule :=
  { nativeCapsule
    s!"native_cuda.{op.name}"
    op
    s!"Native CUDA `{op.name}` follows the explicit reduction shape contract."
    s!"Native CUDA `{op.name}` adjoint is checked through runtime gradient tests." with
    numericalPolicy.reduction := .implementationDefined }

/-- Native CUDA kernel with an accumulation whose tree/order is selected by the implementation.

This covers matrix products, affine layers, convolutions, losses, and average pooling. CUDA may use
parallel trees, fused multiply-add, cuBLAS/cuDNN algorithms, or architecture-specific schedules;
the capsule therefore records the reduction as implementation-defined instead of pretending it is
the reference left fold. -/
def nativeAccumulationCapsule (name : String) (op : BackendOp) (valueSummary vjpSummary :
    String) (vjpMode : VJPMode := .backendVJP) : KernelCapsule :=
  { nativeCapsule name op valueSummary vjpSummary vjpMode with
    numericalPolicy.reduction := .implementationDefined }

/-- Build the standard native-CUDA capsule for a shape or layout transformation. -/
def nativeViewCapsule (op : BackendOp) : KernelCapsule :=
  nativeCapsule
    s!"native_cuda.{op.name}"
    op
    s!"Native CUDA `{op.name}` follows the explicit shape/layout contract."
    s!"Native CUDA `{op.name}` adjoint is checked through runtime gradient tests."

/-- Build a native-CUDA forward-only capsule with no registered reverse derivative. -/
def nativeForwardOnlyCapsule (op : BackendOp) (valueSummary : String) : KernelCapsule :=
  nativeCapsule
    s!"native_cuda.{op.name}"
    op
    valueSummary
    s!"Native CUDA `{op.name}` is a forward-only capsule with no registered VJP."
    .none

/-- Build a native-CUDA capsule for channel-first convolution or pooling. -/
def nativeConvPoolCapsule (op : BackendOp) : KernelCapsule :=
  nativeCapsule
    s!"native_cuda.{op.name}"
    op
    s!"Native CUDA `{op.name}` follows the channel-first runtime contract."
    s!"Native CUDA `{op.name}` VJP is checked by CUDA runtime coverage."

/-- Native window selection whose traversal and tie winner are chosen by the CUDA implementation. -/
def nativeSelectionCapsule (op : BackendOp) : KernelCapsule :=
  { nativeConvPoolCapsule op with
    numericalPolicy.reduction := .implementationDefined }

/-- Native CUDA batched/matrix multiplication, backed by CUDA/cuBLAS paths. -/
def matmul : KernelCapsule :=
  nativeAccumulationCapsule
    "native_cuda.matmul"
    .matmul
    "Matrix products agree with the row-major runtime contract."
    "Backward products are checked through autograd/runtime parity."

/-- Native CUDA ReLU activation. -/
def relu : KernelCapsule :=
  nativeCapsule
    "native_cuda.relu"
    .relu
    "ReLU forward follows the pointwise activation contract."
    "ReLU VJP is checked through runtime autograd tests."

/-- Native CUDA GELU activation. -/
def gelu : KernelCapsule :=
  nativeCapsule
    "native_cuda.gelu"
    .gelu
    "GELU forward follows the documented runtime approximation contract."
    "GELU VJP is checked through runtime autograd tests."

/-- Native-CUDA pointwise addition. -/
def add : KernelCapsule := nativePointwiseCapsule .add
/-- Native-CUDA pointwise subtraction. -/
def sub : KernelCapsule := nativePointwiseCapsule .sub
/-- Native-CUDA pointwise multiplication. -/
def mul : KernelCapsule := nativePointwiseCapsule .mul
/-- Native-CUDA scalar multiplication. -/
def scale : KernelCapsule := nativePointwiseCapsule .scale
/-- Native-CUDA pointwise absolute value. -/
def abs : KernelCapsule := nativePointwiseCapsule .abs
/-- Native-CUDA pointwise square root. -/
def sqrt : KernelCapsule := nativePointwiseCapsule .sqrt
/-- Native-CUDA pointwise interval clamp. -/
def clamp : KernelCapsule := nativePointwiseCapsule .clamp
/-- Native-CUDA pointwise maximum. -/
def max : KernelCapsule := nativePointwiseCapsule .max
/-- Native-CUDA pointwise minimum. -/
def min : KernelCapsule := nativePointwiseCapsule .min
/-- Native-CUDA pointwise sigmoid. -/
def sigmoid : KernelCapsule := nativePointwiseCapsule .sigmoid
/-- Native-CUDA pointwise hyperbolic tangent. -/
def tanh : KernelCapsule := nativePointwiseCapsule .tanh
/-- Native-CUDA pointwise softplus. -/
def softplus : KernelCapsule := nativePointwiseCapsule .softplus
/-- Native-CUDA pointwise exponential. -/
def exp : KernelCapsule := nativePointwiseCapsule .exp
/-- Native CUDA sine, with angles measured in radians. -/
def sin : KernelCapsule := nativePointwiseCapsule .sin
/-- Native CUDA cosine, with angles measured in radians. -/
def cos : KernelCapsule := nativePointwiseCapsule .cos
/-- Native-CUDA pointwise natural logarithm. -/
def log : KernelCapsule := nativePointwiseCapsule .log
/-- Native-CUDA pointwise reciprocal. -/
def inv : KernelCapsule := nativePointwiseCapsule .inv
/-- Native-CUDA guarded logarithm used by numerically defensive programs. -/
def safeLog : KernelCapsule := nativePointwiseCapsule .safeLog
/-- Native-CUDA log-softmax reduction and normalization. -/
def logSoftmax : KernelCapsule :=
  nativeAccumulationCapsule
    "native_cuda.log_softmax"
    .logSoftmax
    "Log-softmax kernels follow the stable row/axis normalization contract."
    "Log-softmax VJPs are checked through runtime autograd tests."

/-- Native CUDA row/axis softmax kernels. -/
def softmax : KernelCapsule :=
  nativeAccumulationCapsule
    "native_cuda.softmax"
    .softmax
    "Softmax kernels follow the row/axis normalization contract."
    "Softmax VJPs are checked through runtime autograd tests."

/-- Native CUDA hard-masked row softmax. -/
def hardMaskedSoftmax : KernelCapsule :=
  nativeAccumulationCapsule
    "native_cuda.hard_masked_softmax"
    .hardMaskedSoftmax
    ("The kernel normalizes over allowed entries and writes zeros at blocked coordinates. " ++
      "A fully blocked row returns zeros.")
    ("The local VJP uses the softmax Jacobian evaluated at the masked output; blocked " ++
      "coordinates therefore receive zero gradient.")

/-- Native-CUDA sum reduction. -/
def reduceSum : KernelCapsule := nativeReductionCapsule .reduceSum
/-- Native-CUDA arithmetic-mean reduction. -/
def reduceMean : KernelCapsule := nativeReductionCapsule .reduceMean

/-- Native-CUDA shape-preserving reshape view. -/
def reshape : KernelCapsule := nativeViewCapsule .reshape
/-- Native-CUDA axis permutation. -/
def permute : KernelCapsule := nativeViewCapsule .permute
/-- Native-CUDA tensor broadcasting. -/
def broadcast : KernelCapsule := nativeViewCapsule .broadcast
/-- Native-CUDA tensor concatenation. -/
def concat : KernelCapsule := nativeViewCapsule .concat
/-- Native-CUDA contiguous tensor slice. -/
def slice : KernelCapsule := nativeViewCapsule .slice
/-- Native-CUDA indexed gather. -/
def gather : KernelCapsule := nativeViewCapsule .gather
/-- Native-CUDA indexed scatter-add. -/
def scatterAdd : KernelCapsule :=
  nativeAccumulationCapsule
    "native_cuda.scatter_add"
    .scatterAdd
    "Indexed source values accumulate into the base tensor, including repeated indices."
    "The VJP gathers output gradients at the source indices and preserves the base gradient."

/-- Native-CUDA seeded uniform-random tensor generation. -/
def randUniform : KernelCapsule :=
  nativeForwardOnlyCapsule
    .randUniform
    "Native CUDA deterministic random-uniform buffers follow the seeded runtime contract."

/-- Native-CUDA seeded Bernoulli-mask generation. -/
def bernoulliMask : KernelCapsule :=
  nativeForwardOnlyCapsule
    .bernoulliMask
    "Native CUDA deterministic Bernoulli masks follow the seeded runtime contract."

/-- Native CUDA layer normalization. -/
def layerNorm : KernelCapsule :=
  nativeAccumulationCapsule
    "native_cuda.layer_norm"
    .layerNorm
    "LayerNorm follows the per-row normalization contract."
    "LayerNorm VJP is checked by CUDA runtime coverage."

/-- Native CUDA batch normalization. -/
def batchNorm : KernelCapsule :=
  nativeAccumulationCapsule
    "native_cuda.batch_norm"
    .batchNorm
    "BatchNorm follows the channel-first normalization contract."
    "BatchNorm VJP is checked by CUDA runtime coverage."

/-- Native CUDA generic channel-first convolution. -/
def conv : KernelCapsule :=
  nativeAccumulationCapsule
    "native_cuda.conv"
    .conv
    "Convolution follows the generic channel-first runtime contract."
    "Convolution VJP is checked by CUDA runtime coverage."

/-- Native CUDA generic channel-first transpose convolution. -/
def convTranspose : KernelCapsule :=
  nativeAccumulationCapsule
    "native_cuda.conv_transpose"
    .convTranspose
    "Transpose convolution follows the generic channel-first runtime contract."
    "Transpose-convolution VJP is checked by CUDA runtime coverage."

/-- Native CUDA max pooling. -/
def maxPool : KernelCapsule :=
  nativeSelectionCapsule .maxPool

/-- Native CUDA smooth max pooling. -/
def smoothMaxPool : KernelCapsule :=
  nativeAccumulationCapsule
    "native_cuda.smooth_max_pool"
    .smoothMaxPool
    "Smooth max pooling uses finite nonzero beta and stable max/min-shifted window weights."
    "Forward and VJP stability are checked against the reference runtime at overflow-scale inputs."

/-- Native CUDA average pooling. -/
def avgPool : KernelCapsule :=
  nativeAccumulationCapsule
    "native_cuda.avg_pool"
    .avgPool
    "Average-pooling follows the channel-first window contract."
    "Average-pooling VJP is checked by CUDA runtime coverage."

/-- Native CUDA linear layer. -/
def linear : KernelCapsule :=
  nativeAccumulationCapsule
    "native_cuda.linear"
    .linear
    "Linear layer kernels follow the matvec/matmul plus bias contract."
    "Linear VJP is checked by CUDA runtime coverage."

/-- Native CUDA mean-squared-error loss. -/
def mseLoss : KernelCapsule :=
  nativeAccumulationCapsule
    "native_cuda.mse_loss"
    .mseLoss
    "MSE loss follows the mean squared residual contract."
    "MSE VJP is checked by CUDA runtime coverage."

/-- Native CUDA FFT/FNO kernels. -/
def fftFno : KernelCapsule :=
  nativeAccumulationCapsule
    "native_cuda.fft_fno"
    .fftFno
    "Packed rFFT/irFFT and spectral convolution follow the documented half-spectrum contract."
    ("Packed transforms use the real-linear half-spectrum adjoints; spectral convolution " ++
      "propagates gradients to its input and both weight components.")

/-- Native CUDA selective scan kernels. -/
def selectiveScan : KernelCapsule :=
  nativeAccumulationCapsule
    "native_cuda.selective_scan"
    .selectiveScan
    "Shared and token-dependent coefficients follow the diagonal recurrence contract."
    ("Reverse recurrence differentiates coefficients, inputs, and the initial state; shared " ++
      "coefficient cotangents are accumulated across time.")

/-- Native CUDA capsules, excluding attention which has a dedicated semantic split. -/
def capsules : Array KernelCapsule :=
  #[ matmul
  , linear
  , mseLoss
  , relu
  , gelu
  , add
  , sub
  , mul
  , scale
  , abs
  , sqrt
  , clamp
  , max
  , min
  , sigmoid
  , tanh
  , softplus
  , exp
  , sin
  , cos
  , log
  , inv
  , safeLog
  , logSoftmax
  , softmax
  , hardMaskedSoftmax
  , reduceSum
  , reduceMean
  , reshape
  , permute
  , broadcast
  , concat
  , slice
  , gather
  , scatterAdd
  , randUniform
  , bernoulliMask
  , layerNorm
  , batchNorm
  , conv
  , convTranspose
  , maxPool
  , smoothMaxPool
  , avgPool
  , fftFno
  , selectiveScan
  ]

end NativeCUDA
end Backend
end NN
