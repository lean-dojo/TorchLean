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

Local VJP ownership follows `Engine.LibTorch.Ops`: `.torchLeanTape` means Lean composes the
derivative from numerical primitives; `.backendVJP` means it calls a native backward primitive.
Both retain the LibTorch provider and TorchLean's global tape.
-/

@[expose] public section

namespace NN
namespace Backend
namespace LibTorch

namespace Internal

/-- Retained CUDA comparisons, by operation. An absent entry leaves an explicit trust boundary. -/
def coverage (op : BackendOp) (vjp : Bool := false) : Option String :=
  match op with
  | .add | .sub | .mul | .scale | .abs | .sqrt | .clamp | .max | .min | .relu
    | .sigmoid | .tanh | .gelu | .softplus | .exp | .log | .inv | .safeLog | .reduceSum =>
      some "NN.Tests.Runtime.Cuda.Elementwise"
  | .softmax | .logSoftmax => some "NN.Tests.Runtime.Cuda.Softmax"
  | .matmul | .hardMaskedSoftmax => some "NN.Tests.Runtime.Cuda.Attention"
  | .selectiveScan => some "NN.Tests.Runtime.Cuda.SelectiveScan"
  | .fftFno => some "NN.Tests.Runtime.Cuda.Fft"
  | .linear | .mseLoss => some "NN.Tests.Runtime.Cuda.Trainer"
  | .randUniform | .bernoulliMask | .reduceMean =>
      if vjp then none else some "NN.Tests.Runtime.Cuda.Stress"
  | _ => none

/-- Record the available comparison, or the untested native agreement being assumed. -/
def evidence (op : BackendOp) (vjp : Bool := false) : ContractEvidence :=
  match coverage op vjp with
  | some suite => .testSuite suite
  | none => .trustedBoundary <|
      s!"LibTorch `{op.name}` implementation; no retained dedicated CUDA " ++
        (if vjp then "VJP parity check" else "forward parity check")

/-- Describe a maintained LibTorch primitive and the evidence required by its runtime contract. -/
def capsule
    (name : String) (op : BackendOp) (valueSummary vjpSummary : String)
    (vjpMode : VJPMode) : KernelCapsule :=
  { name
    op
    provider := .libTorch
    device := .cuda
    trustLevel := if (coverage op).isSome &&
        (vjpMode == .none || (coverage op (vjp := true)).isSome) then
      .checked else .trustedExternal
    supportsForward := true
    vjpMode
    shapeContract :=
      { claim := .shapeSafety op
        summary := "Inputs and outputs are checked against explicit UInt32 dimensions."
        evidence := .runtimeGuard "LibTorch bridge size/rank checks at the Lean/native boundary" }
    layoutContract :=
      { claim := .layoutCompatibility op .libTorchCudaView
        summary := "CUDA buffers retain their dtype in contiguous row-major LibTorch tensors."
        evidence := .runtimeGuard
          "LibTorch bridge dtype, device, contiguity, and element-count checks" }
    valueContract :=
      { claim := .valueRefinement op
        summary := valueSummary
        evidence := evidence op }
    vjpContract :=
      match vjpMode with
      | .none => ContractDescriptor.vjpUnavailable op vjpSummary
      | mode =>
          { claim := .vjpRefinement op mode
            summary := vjpSummary
            evidence := evidence op (vjp := true) }
    numericalPolicy := { reduction := .notApplicable } }

/-- Build the standard LibTorch CUDA capsule for a pointwise operation. -/
def pointwiseCapsule (op : BackendOp) (vjpMode : VJPMode) : KernelCapsule :=
  capsule
    s!"libtorch.{op.name}"
    op
    s!"LibTorch CUDA `{op.name}` follows the pointwise tensor contract."
    s!"Local reverse rule for LibTorch CUDA `{op.name}`."
    vjpMode

/-- Build a LibTorch CUDA reduction capsule with implementation-defined reduction order. -/
def reductionCapsule (op : BackendOp) (vjpMode : VJPMode) : KernelCapsule :=
  { capsule
    s!"libtorch.{op.name}"
    op
    s!"LibTorch CUDA `{op.name}` follows the explicit reduction shape contract."
    s!"Local reduction adjoint for LibTorch CUDA `{op.name}`."
    vjpMode with
    numericalPolicy.reduction := .implementationDefined }

/-- LibTorch CUDA kernel with an accumulation whose tree/order is selected by the implementation.

This covers matrix products, affine layers, convolutions, losses, and average pooling. ATen
selects the accumulation implementation; the capsule does not promise the reference left fold.
Deterministic execution settings do not strengthen this to a particular numerical schedule. -/
def accumulationCapsule (name : String) (op : BackendOp) (valueSummary vjpSummary :
    String) (vjpMode : VJPMode) : KernelCapsule :=
  { capsule name op valueSummary vjpSummary vjpMode with
    numericalPolicy.reduction := .implementationDefined }

/-- Build the standard LibTorch CUDA capsule for a shape or layout transformation. -/
def viewCapsule (op : BackendOp) (vjpMode : VJPMode) : KernelCapsule :=
  capsule
    s!"libtorch.{op.name}"
    op
    s!"LibTorch CUDA `{op.name}` follows the explicit shape/layout contract."
    s!"Local shape/layout adjoint for LibTorch CUDA `{op.name}`."
    vjpMode

/-- Build a LibTorch CUDA forward-only capsule with no registered reverse derivative. -/
def forwardOnlyCapsule (op : BackendOp) (valueSummary : String) : KernelCapsule :=
  capsule
    s!"libtorch.{op.name}"
    op
    valueSummary
    s!"LibTorch CUDA `{op.name}` is a forward-only capsule with no registered VJP."
    .none

end Internal

open Internal

/-- LibTorch CUDA batched/matrix multiplication. -/
def matmul : KernelCapsule :=
  accumulationCapsule
    "libtorch.matmul"
    .matmul
    "Matrix products agree with the row-major runtime contract."
    "Backward products are checked through autograd/runtime parity."
    .torchLeanTape

/-- LibTorch CUDA ReLU activation. -/
def relu : KernelCapsule :=
  capsule
    "libtorch.relu"
    .relu
    "ReLU forward follows the pointwise activation contract."
    "ReLU VJP is checked through runtime autograd tests."
    .backendVJP

/-- LibTorch CUDA GELU activation. -/
def gelu : KernelCapsule :=
  capsule
    "libtorch.gelu"
    .gelu
    "GELU forward follows the documented runtime approximation contract."
    "GELU VJP is checked through runtime autograd tests."
    .backendVJP

/-- LibTorch CUDA pointwise addition. -/
def add : KernelCapsule := pointwiseCapsule .add .torchLeanTape
/-- LibTorch CUDA pointwise subtraction. -/
def sub : KernelCapsule := pointwiseCapsule .sub .torchLeanTape
/-- LibTorch CUDA pointwise multiplication. -/
def mul : KernelCapsule := pointwiseCapsule .mul .torchLeanTape
/-- LibTorch CUDA scalar multiplication. -/
def scale : KernelCapsule := pointwiseCapsule .scale .torchLeanTape
/-- LibTorch CUDA pointwise absolute value. -/
def abs : KernelCapsule := pointwiseCapsule .abs .backendVJP
/-- LibTorch CUDA pointwise square root. -/
def sqrt : KernelCapsule := pointwiseCapsule .sqrt .backendVJP
/-- LibTorch CUDA pointwise interval clamp. -/
def clamp : KernelCapsule := pointwiseCapsule .clamp .backendVJP
/-- LibTorch CUDA pointwise maximum. -/
def max : KernelCapsule := pointwiseCapsule .max .backendVJP
/-- LibTorch CUDA pointwise minimum. -/
def min : KernelCapsule := pointwiseCapsule .min .backendVJP
/-- LibTorch CUDA pointwise sigmoid. -/
def sigmoid : KernelCapsule := pointwiseCapsule .sigmoid .torchLeanTape
/-- LibTorch CUDA pointwise hyperbolic tangent. -/
def tanh : KernelCapsule := pointwiseCapsule .tanh .torchLeanTape
/-- LibTorch CUDA pointwise softplus. -/
def softplus : KernelCapsule := pointwiseCapsule .softplus .torchLeanTape
/-- LibTorch CUDA pointwise exponential. -/
def exp : KernelCapsule := pointwiseCapsule .exp .torchLeanTape
/-- LibTorch CUDA sine, with angles measured in radians. -/
def sin : KernelCapsule := pointwiseCapsule .sin .torchLeanTape
/-- LibTorch CUDA cosine, with angles measured in radians. -/
def cos : KernelCapsule := pointwiseCapsule .cos .torchLeanTape
/-- LibTorch CUDA pointwise natural logarithm. -/
def log : KernelCapsule := pointwiseCapsule .log .torchLeanTape
/-- LibTorch CUDA pointwise reciprocal. -/
def inv : KernelCapsule := pointwiseCapsule .inv .torchLeanTape

/-- Native quotient arithmetic with TorchLean's quotient-rule VJP. -/
def div : KernelCapsule := pointwiseCapsule .div .torchLeanTape

/-- Native scalar negation with TorchLean's sign-reversing VJP. -/
def neg : KernelCapsule := pointwiseCapsule .neg .torchLeanTape
/-- LibTorch CUDA smooth logarithm surrogate `log (softplus x + epsilon)`. -/
def safeLog : KernelCapsule := pointwiseCapsule .safeLog .torchLeanTape
/-- LibTorch CUDA log-softmax reduction and normalization. -/
def logSoftmax : KernelCapsule :=
  accumulationCapsule
    "libtorch.log_softmax"
    .logSoftmax
    "Log-softmax kernels follow the stable row/axis normalization contract."
    "Log-softmax VJPs are checked through runtime autograd tests."
    .torchLeanTape

/-- LibTorch CUDA row/axis softmax kernels. -/
def softmax : KernelCapsule :=
  accumulationCapsule
    "libtorch.softmax"
    .softmax
    "Softmax kernels follow the row/axis normalization contract."
    "Softmax VJPs are checked through runtime autograd tests."
    .torchLeanTape

/-- LibTorch CUDA hard-masked row softmax. -/
def hardMaskedSoftmax : KernelCapsule :=
  accumulationCapsule
    "libtorch.hard_masked_softmax"
    .hardMaskedSoftmax
    ("The kernel normalizes over allowed entries and writes zeros at blocked coordinates. " ++
      "A fully blocked row returns zeros.")
    ("The local VJP uses the softmax Jacobian evaluated at the masked output; blocked " ++
      "coordinates therefore receive zero gradient.")
    .torchLeanTape

/-- LibTorch CUDA sum reduction. -/
def reduceSum : KernelCapsule := reductionCapsule .reduceSum .torchLeanTape
/-- LibTorch CUDA arithmetic-mean reduction. -/
def reduceMean : KernelCapsule := reductionCapsule .reduceMean .torchLeanTape

/-- LibTorch CUDA shape-preserving reshape view. -/
def reshape : KernelCapsule := viewCapsule .reshape .torchLeanTape
/-- LibTorch CUDA axis permutation. -/
def permute : KernelCapsule := viewCapsule .permute .torchLeanTape
/-- LibTorch CUDA tensor broadcasting. -/
def broadcast : KernelCapsule := viewCapsule .broadcast .backendVJP
/-- LibTorch CUDA tensor concatenation. -/
def concat : KernelCapsule := viewCapsule .concat .torchLeanTape
/-- LibTorch CUDA contiguous tensor slice. -/
def slice : KernelCapsule := viewCapsule .slice .torchLeanTape
/-- LibTorch CUDA indexed gather. -/
def gather : KernelCapsule := viewCapsule .gather .torchLeanTape
/-- LibTorch CUDA indexed scatter-add. -/
def scatterAdd : KernelCapsule :=
  accumulationCapsule
    "libtorch.scatter_add"
    .scatterAdd
    "Indexed source values accumulate into the base tensor, including repeated indices."
    "The VJP gathers output gradients at the source indices and preserves the base gradient."
    .torchLeanTape

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
    "LayerNorm VJP calls the native backward primitive."
    .backendVJP

/-- LibTorch CUDA batch normalization. -/
def batchNorm : KernelCapsule :=
  accumulationCapsule
    "libtorch.batch_norm"
    .batchNorm
    "BatchNorm follows the channel-first normalization contract."
    "Lean composes the BatchNorm VJP from LibTorch numerical primitives."
    .torchLeanTape

/-- LibTorch CUDA generic channel-first convolution. -/
def conv : KernelCapsule :=
  accumulationCapsule
    "libtorch.conv"
    .conv
    "Convolution follows the generic channel-first runtime contract."
    "Convolution VJP calls the native backward primitive."
    .backendVJP

/-- LibTorch CUDA generic channel-first transpose convolution. -/
def convTranspose : KernelCapsule :=
  accumulationCapsule
    "libtorch.conv_transpose"
    .convTranspose
    "Transpose convolution follows the generic channel-first runtime contract."
    "Transpose-convolution VJP calls the native backward primitive."
    .backendVJP

/-- LibTorch CUDA max pooling, skipping padded cells and retaining the first row-major winner. -/
def maxPool : KernelCapsule :=
  let op := BackendOp.maxPool
  accumulationCapsule
    s!"libtorch.{op.name}" op
    s!"LibTorch CUDA `{op.name}` follows the channel-first runtime contract."
    s!"LibTorch CUDA `{op.name}` VJP calls the native backward primitive."
    .backendVJP

/-- LibTorch CUDA smooth max pooling. -/
def smoothMaxPool : KernelCapsule :=
  accumulationCapsule
    "libtorch.smooth_max_pool"
    .smoothMaxPool
    "Smooth max pooling uses finite nonzero beta and stable max/min-shifted window weights."
    "The native backward rule uses the shifted window weights."
    .backendVJP

/-- LibTorch CUDA average pooling. -/
def avgPool : KernelCapsule :=
  accumulationCapsule
    "libtorch.avg_pool"
    .avgPool
    "Average-pooling follows the channel-first window contract."
    "Average-pooling VJP calls the native backward primitive."
    .backendVJP

/-- LibTorch CUDA linear layer. -/
def linear : KernelCapsule :=
  accumulationCapsule
    "libtorch.linear"
    .linear
    "Linear layer kernels follow the matvec/matmul plus bias contract."
    "Linear VJP is checked by CUDA runtime coverage."
    .torchLeanTape

/-- LibTorch CUDA mean-squared-error loss. -/
def mseLoss : KernelCapsule :=
  accumulationCapsule
    "libtorch.mse_loss"
    .mseLoss
    "MSE loss follows the mean squared residual contract."
    "MSE VJP is checked by CUDA runtime coverage."
    .torchLeanTape

/-- LibTorch numerical FFTs with Lean adjoints and spectral-layer composition. -/
def fftFno : KernelCapsule :=
  accumulationCapsule
    "libtorch.fft_fno"
    .fftFno
    "Packed rFFT/irFFT follow the documented half-spectrum contract."
    ("Lean composes the real-linear transform adjoints and spectral channel mixing, " ++
      "propagating gradients to the input and both weight components.")
    .torchLeanTape

/-- LibTorch CUDA selective scan kernels. -/
def selectiveScan : KernelCapsule :=
  accumulationCapsule
    "libtorch.selective_scan"
    .selectiveScan
    "Shared and token-dependent coefficients follow the diagonal recurrence contract."
    ("Reverse recurrence differentiates coefficients, inputs, and the initial state; shared " ++
      "coefficient cotangents are accumulated across time.")
    .backendVJP

/--
Lean-composed attention over LibTorch numerical primitives.

TorchLean composes forward and the local VJP and owns the saved Q/K/V and probability buffers.
The existing capsule identity is retained for routing and audits. No native attention context or
fused-provider selection is involved; full score matrices require quadratic sequence memory.
Numerical primitives execute without a LibTorch autograd graph or a fixed reduction schedule.
-/
def attention : KernelCapsule :=
  { name := "libtorch.direct_attention"
    op := .attention
    provider := .libTorch
    device := .cuda
    trustLevel := .checked
    supportsForward := true
    vjpMode := .torchLeanTape
    shapeContract :=
      ContractDescriptor.guarded (.shapeSafety .attention)
        ("Q/K/V use a folded (batch, head, n, headDim) layout; the optional mask broadcasts " ++
          "over the folded batch-head axis.")
        "Buffer.attentionForward/attentionBackward dimension, scale, and buffer-size checks"
    layoutContract :=
      ContractDescriptor.guarded
        (.layoutCompatibility .attention .libTorchCudaView)
        "Row-major LibTorch tensors; the folded batch-head axis is the kernel batch axis."
        "LibTorch bridge dtype, device, contiguity, and element-count checks"
    valueContract :=
      ContractDescriptor.tested (.valueRefinement .attention)
        "Lean-composed attention with hard-mask zero numerators and zero fully blocked rows."
        "NN.Tests.Runtime.Cuda.Attention"
    vjpContract :=
      ContractDescriptor.tested
        (.vjpRefinement .attention .torchLeanTape)
        "Lean composes matrix products and the softmax VJP over saved probabilities for dQ/dK/dV."
        "NN.Tests.Runtime.Cuda.Attention"
    numericalPolicy := { reduction := .implementationDefined } }

/-- LibTorch CUDA primitive contracts. -/
def capsules : Array KernelCapsule :=
  #[ attention
  , matmul
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
  , div
  , neg
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
