/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Capsule

/-!
# LibTorch Backend Capsules

Maintained CUDA primitives implemented with LibTorch ATen tensor operations.

TorchLean owns tape traversal and selects each local VJP. Native forward and backward routines
execute without recording a LibTorch autograd graph. The wrappers remain a foreign implementation
boundary: shape/layout guards and named regression suites are engineering evidence, not Lean proofs
of ATen, its dependencies, or compiled execution. Numerical reductions retain implementation-defined
ordering. The CPU/reference capsules and their mathematical contracts are separate.
-/

@[expose] public section

namespace NN
namespace Backend
namespace LibTorch

/-- Describe a maintained LibTorch primitive and the evidence required by its runtime contract. -/
def capsule
    (name : String) (op : BackendOp) (valueSummary vjpSummary : String)
    (vjpMode : VJPMode := .backendVJP) : KernelCapsule :=
  { name
    op
    provider := .libTorch
    device := .cuda
    trustLevel := .checked
    supportsForward := true
    vjpMode
    shapeContract :=
      { claim := .shapeSafety op
        summary := "Inputs and outputs are checked against explicit UInt32 dimensions."
        evidence := .runtimeGuard "LibTorch bridge size/rank checks at the Lean/native boundary" }
    layoutContract :=
      { claim := .layoutCompatibility op .libTorchCudaView
        summary := "CUDA buffers contain contiguous row-major LibTorch float32 tensors."
        evidence := .runtimeGuard
          "LibTorch bridge dtype, device, contiguity, and element-count checks" }
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

/-- Build the standard LibTorch CUDA capsule for a pointwise operation. -/
def pointwiseCapsule (op : BackendOp) : KernelCapsule :=
  capsule
    s!"libtorch.{op.name}"
    op
    s!"LibTorch CUDA `{op.name}` follows the pointwise tensor contract."
    s!"LibTorch CUDA `{op.name}` VJP is checked through runtime autograd tests."

/-- Build a LibTorch CUDA reduction capsule with implementation-defined reduction order. -/
def reductionCapsule (op : BackendOp) : KernelCapsule :=
  { capsule
    s!"libtorch.{op.name}"
    op
    s!"LibTorch CUDA `{op.name}` follows the explicit reduction shape contract."
    s!"LibTorch CUDA `{op.name}` adjoint is checked through runtime gradient tests." with
    numericalPolicy.reduction := .implementationDefined }

/-- LibTorch CUDA kernel with an accumulation whose tree/order is selected by the implementation.

This covers matrix products, affine layers, convolutions, losses, and average pooling. ATen
selects the accumulation implementation; the capsule does not promise the reference left fold.
Deterministic execution settings do not strengthen this to a particular numerical schedule. -/
def accumulationCapsule (name : String) (op : BackendOp) (valueSummary vjpSummary :
    String) (vjpMode : VJPMode := .backendVJP) : KernelCapsule :=
  { capsule name op valueSummary vjpSummary vjpMode with
    numericalPolicy.reduction := .implementationDefined }

/-- Build the standard LibTorch CUDA capsule for a shape or layout transformation. -/
def viewCapsule (op : BackendOp) : KernelCapsule :=
  capsule
    s!"libtorch.{op.name}"
    op
    s!"LibTorch CUDA `{op.name}` follows the explicit shape/layout contract."
    s!"LibTorch CUDA `{op.name}` adjoint is checked through runtime gradient tests."

/-- Build a LibTorch CUDA forward-only capsule with no registered reverse derivative. -/
def forwardOnlyCapsule (op : BackendOp) (valueSummary : String) : KernelCapsule :=
  capsule
    s!"libtorch.{op.name}"
    op
    valueSummary
    s!"LibTorch CUDA `{op.name}` is a forward-only capsule with no registered VJP."
    .none

/-- Build a LibTorch CUDA capsule for channel-first convolution or pooling. -/
def convPoolCapsule (op : BackendOp) : KernelCapsule :=
  capsule
    s!"libtorch.{op.name}"
    op
    s!"LibTorch CUDA `{op.name}` follows the channel-first runtime contract."
    s!"LibTorch CUDA `{op.name}` VJP is checked by CUDA runtime coverage."

/-- Preserve window selection's tie convention.
Gradient accumulation order remains implementation-defined. -/
def selectionCapsule (op : BackendOp) : KernelCapsule :=
  { convPoolCapsule op with
    numericalPolicy.reduction := .implementationDefined }

/-- LibTorch CUDA batched/matrix multiplication. -/
def matmul : KernelCapsule :=
  accumulationCapsule
    "libtorch.matmul"
    .matmul
    "Matrix products agree with the row-major runtime contract."
    "Backward products are checked through autograd/runtime parity."

/-- LibTorch CUDA ReLU activation. -/
def relu : KernelCapsule :=
  capsule
    "libtorch.relu"
    .relu
    "ReLU forward follows the pointwise activation contract."
    "ReLU VJP is checked through runtime autograd tests."

/-- LibTorch CUDA GELU activation. -/
def gelu : KernelCapsule :=
  capsule
    "libtorch.gelu"
    .gelu
    "GELU forward follows the documented runtime approximation contract."
    "GELU VJP is checked through runtime autograd tests."

/-- LibTorch CUDA pointwise addition. -/
def add : KernelCapsule := pointwiseCapsule .add
/-- LibTorch CUDA pointwise subtraction. -/
def sub : KernelCapsule := pointwiseCapsule .sub
/-- LibTorch CUDA pointwise multiplication. -/
def mul : KernelCapsule := pointwiseCapsule .mul
/-- LibTorch CUDA scalar multiplication. -/
def scale : KernelCapsule := pointwiseCapsule .scale
/-- LibTorch CUDA pointwise absolute value. -/
def abs : KernelCapsule := pointwiseCapsule .abs
/-- LibTorch CUDA pointwise square root. -/
def sqrt : KernelCapsule := pointwiseCapsule .sqrt
/-- LibTorch CUDA pointwise interval clamp. -/
def clamp : KernelCapsule := pointwiseCapsule .clamp
/-- LibTorch CUDA pointwise maximum. -/
def max : KernelCapsule := pointwiseCapsule .max
/-- LibTorch CUDA pointwise minimum. -/
def min : KernelCapsule := pointwiseCapsule .min
/-- LibTorch CUDA pointwise sigmoid. -/
def sigmoid : KernelCapsule := pointwiseCapsule .sigmoid
/-- LibTorch CUDA pointwise hyperbolic tangent. -/
def tanh : KernelCapsule := pointwiseCapsule .tanh
/-- LibTorch CUDA pointwise softplus. -/
def softplus : KernelCapsule := pointwiseCapsule .softplus
/-- LibTorch CUDA pointwise exponential. -/
def exp : KernelCapsule := pointwiseCapsule .exp
/-- LibTorch CUDA sine, with angles measured in radians. -/
def sin : KernelCapsule := pointwiseCapsule .sin
/-- LibTorch CUDA cosine, with angles measured in radians. -/
def cos : KernelCapsule := pointwiseCapsule .cos
/-- LibTorch CUDA pointwise natural logarithm. -/
def log : KernelCapsule := pointwiseCapsule .log
/-- LibTorch CUDA pointwise reciprocal. -/
def inv : KernelCapsule := pointwiseCapsule .inv
/-- LibTorch CUDA smooth logarithm surrogate `log (softplus x + epsilon)`. -/
def safeLog : KernelCapsule := pointwiseCapsule .safeLog
/-- LibTorch CUDA log-softmax reduction and normalization. -/
def logSoftmax : KernelCapsule :=
  accumulationCapsule
    "libtorch.log_softmax"
    .logSoftmax
    "Log-softmax kernels follow the stable row/axis normalization contract."
    "Log-softmax VJPs are checked through runtime autograd tests."

/-- LibTorch CUDA row/axis softmax kernels. -/
def softmax : KernelCapsule :=
  accumulationCapsule
    "libtorch.softmax"
    .softmax
    "Softmax kernels follow the row/axis normalization contract."
    "Softmax VJPs are checked through runtime autograd tests."

/-- LibTorch CUDA hard-masked row softmax. -/
def hardMaskedSoftmax : KernelCapsule :=
  accumulationCapsule
    "libtorch.hard_masked_softmax"
    .hardMaskedSoftmax
    ("The kernel normalizes over allowed entries and writes zeros at blocked coordinates. " ++
      "A fully blocked row returns zeros.")
    ("The local VJP uses the softmax Jacobian evaluated at the masked output; blocked " ++
      "coordinates therefore receive zero gradient.")

/-- LibTorch CUDA sum reduction. -/
def reduceSum : KernelCapsule := reductionCapsule .reduceSum
/-- LibTorch CUDA arithmetic-mean reduction. -/
def reduceMean : KernelCapsule := reductionCapsule .reduceMean

/-- LibTorch CUDA shape-preserving reshape view. -/
def reshape : KernelCapsule := viewCapsule .reshape
/-- LibTorch CUDA axis permutation. -/
def permute : KernelCapsule := viewCapsule .permute
/-- LibTorch CUDA tensor broadcasting. -/
def broadcast : KernelCapsule := viewCapsule .broadcast
/-- LibTorch CUDA tensor concatenation. -/
def concat : KernelCapsule := viewCapsule .concat
/-- LibTorch CUDA contiguous tensor slice. -/
def slice : KernelCapsule := viewCapsule .slice
/-- LibTorch CUDA indexed gather. -/
def gather : KernelCapsule := viewCapsule .gather
/-- LibTorch CUDA indexed scatter-add. -/
def scatterAdd : KernelCapsule :=
  accumulationCapsule
    "libtorch.scatter_add"
    .scatterAdd
    "Indexed source values accumulate into the base tensor, including repeated indices."
    "The VJP gathers output gradients at the source indices and preserves the base gradient."

/-- LibTorch CUDA seeded uniform-random tensor generation. -/
def randUniform : KernelCapsule :=
  forwardOnlyCapsule
    .randUniform
    "LibTorch CUDA deterministic random-uniform buffers follow the seeded runtime contract."

/-- LibTorch CUDA seeded Bernoulli-mask generation. -/
def bernoulliMask : KernelCapsule :=
  forwardOnlyCapsule
    .bernoulliMask
    "LibTorch CUDA deterministic Bernoulli masks follow the seeded runtime contract."

/-- LibTorch CUDA layer normalization. -/
def layerNorm : KernelCapsule :=
  accumulationCapsule
    "libtorch.layer_norm"
    .layerNorm
    "LayerNorm follows the per-row normalization contract."
    "LayerNorm VJP is checked by CUDA runtime coverage."

/-- LibTorch CUDA batch normalization. -/
def batchNorm : KernelCapsule :=
  accumulationCapsule
    "libtorch.batch_norm"
    .batchNorm
    "BatchNorm follows the channel-first normalization contract."
    "BatchNorm VJP is checked by CUDA runtime coverage."

/-- LibTorch CUDA generic channel-first convolution. -/
def conv : KernelCapsule :=
  accumulationCapsule
    "libtorch.conv"
    .conv
    "Convolution follows the generic channel-first runtime contract."
    "Convolution VJP is checked by CUDA runtime coverage."

/-- LibTorch CUDA generic channel-first transpose convolution. -/
def convTranspose : KernelCapsule :=
  accumulationCapsule
    "libtorch.conv_transpose"
    .convTranspose
    "Transpose convolution follows the generic channel-first runtime contract."
    "Transpose-convolution VJP is checked by CUDA runtime coverage."

/-- LibTorch CUDA max pooling, skipping padded cells and retaining the first row-major winner. -/
def maxPool : KernelCapsule :=
  selectionCapsule .maxPool

/-- LibTorch CUDA smooth max pooling. -/
def smoothMaxPool : KernelCapsule :=
  accumulationCapsule
    "libtorch.smooth_max_pool"
    .smoothMaxPool
    "Smooth max pooling uses finite nonzero beta and stable max/min-shifted window weights."
    "Forward and VJP stability are checked against the reference runtime at overflow-scale inputs."

/-- LibTorch CUDA average pooling. -/
def avgPool : KernelCapsule :=
  accumulationCapsule
    "libtorch.avg_pool"
    .avgPool
    "Average-pooling follows the channel-first window contract."
    "Average-pooling VJP is checked by CUDA runtime coverage."

/-- LibTorch CUDA linear layer. -/
def linear : KernelCapsule :=
  accumulationCapsule
    "libtorch.linear"
    .linear
    "Linear layer kernels follow the matvec/matmul plus bias contract."
    "Linear VJP is checked by CUDA runtime coverage."

/-- LibTorch CUDA mean-squared-error loss. -/
def mseLoss : KernelCapsule :=
  accumulationCapsule
    "libtorch.mse_loss"
    .mseLoss
    "MSE loss follows the mean squared residual contract."
    "MSE VJP is checked by CUDA runtime coverage."

/-- LibTorch CUDA FFT/FNO kernels. -/
def fftFno : KernelCapsule :=
  accumulationCapsule
    "libtorch.fft_fno"
    .fftFno
    "Packed rFFT/irFFT and spectral convolution follow the documented half-spectrum contract."
    ("Packed transforms use the real-linear half-spectrum adjoints; spectral convolution " ++
      "propagates gradients to its input and both weight components.")

/-- LibTorch CUDA selective scan kernels. -/
def selectiveScan : KernelCapsule :=
  accumulationCapsule
    "libtorch.selective_scan"
    .selectiveScan
    "Shared and token-dependent coefficients follow the diagonal recurrence contract."
    ("Reverse recurrence differentiates coefficients, inputs, and the initial state; shared " ++
      "coefficient cotangents are accumulated across time.")

/-- LibTorch CUDA primitive capsules. Attention capsules live in `NN.Backend.Attention.capsules`. -/
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

end LibTorch
end Backend
end NN
