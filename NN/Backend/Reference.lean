/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Capsule

/-!
# Reference and Portable Backend Capsules

Portable capsules for paths that do not require CUDA or LibTorch.

These are not meant to win benchmarks. They keep TorchLean runnable on CPU-only machines and give
the planner a clear fallback vocabulary for reference/spec-aligned execution.
-/

@[expose] public section

namespace NN
namespace Backend
namespace Reference

namespace Internal

/-- Build a checked portable CPU capsule with explicit value, VJP, shape, and layout contracts. -/
def capsule
    (name : String) (op : BackendOp) (valueSummary vjpSummary : String)
    (vjpMode : VJPMode := .torchLeanTape) : KernelCapsule :=
  { name
    op
    provider := .reference
    device := .cpu
    trustLevel := .checked
    supportsForward := true
    vjpMode
    shapeContract :=
      { claim := .shapeSafety op
        summary := "Shapes are checked in Lean before entering the portable runtime path."
        evidence := .runtimeGuard "portable runtime shape checks" }
    layoutContract :=
      { claim := .layoutCompatibility op .canonicalTensor
        summary := "Portable paths use TorchLean's canonical tensor representation."
        evidence := .runtimeGuard "typed tensor layout" }
    valueContract :=
      { claim := .valueRefinement op
        summary := valueSummary
        evidence := .testSuite "NN.Tests.Runtime.Floats.Suite" }
    vjpContract :=
      match vjpMode with
      | .none => ContractDescriptor.vjpUnavailable op vjpSummary
      | mode => ContractDescriptor.tested
          (.vjpRefinement op mode) vjpSummary "NN.Tests.Runtime.Floats.Suite"
    numericalPolicy := { reduction := .notApplicable } }

/-- Build the standard portable capsule for a pointwise operation. -/
def pointwiseCapsule (op : BackendOp) : KernelCapsule :=
  capsule
    s!"reference.{op.name}"
    op
    s!"Reference `{op.name}` follows the pointwise tensor contract."
    s!"TorchLean tape supplies the `{op.name}` VJP where differentiable."

/-- Build a portable reduction capsule with deterministic left-to-right accumulation. -/
def reductionCapsule (op : BackendOp) : KernelCapsule :=
  { capsule
    s!"reference.{op.name}"
    op
    s!"Reference `{op.name}` follows the explicit reduction shape contract."
    s!"TorchLean tape supplies the `{op.name}` adjoint where differentiable." with
    numericalPolicy.reduction := .fixedLeft }

/-- Reference kernel whose scalar result contains an explicit left-to-right accumulation.

Matrix products, affine layers, convolutions, and averaging operations all reduce several products
or samples into one output entry. Keeping this constructor separate from pointwise kernels prevents
the numerical audit from incorrectly reporting that reduction order is irrelevant. -/
def accumulationCapsule (name : String) (op : BackendOp) (valueSummary
    vjpSummary : String) : KernelCapsule :=
  { capsule name op valueSummary vjpSummary with
    numericalPolicy.reduction := .fixedLeft }

/-- Build the standard portable capsule for a shape or layout transformation. -/
def viewCapsule (op : BackendOp) : KernelCapsule :=
  capsule
    s!"reference.{op.name}"
    op
    s!"Reference `{op.name}` follows the explicit shape/layout contract."
    s!"TorchLean tape supplies the `{op.name}` adjoint where differentiable."

/-- Build a portable forward-only capsule with no registered reverse derivative. -/
def forwardOnlyCapsule (op : BackendOp) (valueSummary : String) : KernelCapsule :=
  capsule
    s!"reference.{op.name}"
    op
    valueSummary
    s!"Reference `{op.name}` is a forward-only capsule with no registered VJP."
    .none

end Internal

open Internal

/-- Reference ReLU activation. -/
def relu : KernelCapsule :=
  capsule
    "reference.relu"
    .relu
    "Reference ReLU follows pointwise tensor semantics."
    "TorchLean tape supplies the VJP."

/-- Reference GELU activation. -/
def gelu : KernelCapsule :=
  capsule
    "reference.gelu"
    .gelu
    "Reference GELU follows the documented runtime approximation contract."
    "TorchLean tape supplies the VJP."

/-- Reference pointwise addition. -/
def add : KernelCapsule := pointwiseCapsule .add
/-- Reference pointwise subtraction. -/
def sub : KernelCapsule := pointwiseCapsule .sub
/-- Reference pointwise multiplication. -/
def mul : KernelCapsule := pointwiseCapsule .mul
/-- Reference scalar multiplication. -/
def scale : KernelCapsule := pointwiseCapsule .scale
/-- Reference pointwise absolute value. -/
def abs : KernelCapsule := pointwiseCapsule .abs
/-- Reference pointwise square root. -/
def sqrt : KernelCapsule := pointwiseCapsule .sqrt
/-- Reference pointwise interval clamp. -/
def clamp : KernelCapsule := pointwiseCapsule .clamp
/-- Reference pointwise maximum. -/
def max : KernelCapsule := pointwiseCapsule .max
/-- Reference pointwise minimum. -/
def min : KernelCapsule := pointwiseCapsule .min
/-- Reference pointwise sigmoid. -/
def sigmoid : KernelCapsule := pointwiseCapsule .sigmoid
/-- SiLU calls the canonical value and VJP directly, without an intermediate sigmoid tensor. -/
def silu : KernelCapsule :=
  { pointwiseCapsule .silu with
    valueContract.evidence := .sharedDefinition "Activation.swishSpec"
    vjpContract.evidence := .sharedDefinition "Activation.swishDerivSpec" }
/-- Reference pointwise hyperbolic tangent. -/
def tanh : KernelCapsule := pointwiseCapsule .tanh
/-- Reference pointwise softplus. -/
def softplus : KernelCapsule := pointwiseCapsule .softplus
/-- Reference pointwise exponential. -/
def exp : KernelCapsule := pointwiseCapsule .exp
/-- Reference pointwise natural logarithm. -/
def log : KernelCapsule := pointwiseCapsule .log
/-- Reference pointwise sine. -/
def sin : KernelCapsule := pointwiseCapsule .sin
/-- Reference pointwise cosine. -/
def cos : KernelCapsule := pointwiseCapsule .cos
/-- Reference pointwise reciprocal. -/
def inv : KernelCapsule := pointwiseCapsule .inv

/-- Reference quotient arithmetic with the recorded quotient-rule VJP. -/
def div : KernelCapsule := pointwiseCapsule .div

/-- Reference scalar negation with the sign-reversing VJP. -/
def neg : KernelCapsule := pointwiseCapsule .neg
/-- Reference smooth logarithm surrogate `log (softplus x + epsilon)`. -/
def safeLog : KernelCapsule := pointwiseCapsule .safeLog
/-- Reference log-softmax reduction and normalization. -/
def logSoftmax : KernelCapsule :=
  accumulationCapsule
    "reference.log_softmax"
    .logSoftmax
    "Reference log-softmax follows the stable row/axis normalization contract."
    "TorchLean tape supplies the VJP."

/-- Reference softmax path. -/
def softmax : KernelCapsule :=
  accumulationCapsule
    "reference.softmax"
    .softmax
    "Reference softmax follows the row/axis normalization contract."
    "TorchLean tape supplies the VJP."

/-- Reference hard-masked softmax with exact zero weight at blocked coordinates. -/
def hardMaskedSoftmax : KernelCapsule :=
  accumulationCapsule
    "reference.hard_masked_softmax"
    .hardMaskedSoftmax
    ("Reference hard-masked softmax normalizes over allowed coordinates and returns zero for " ++
      "fully blocked rows.")
    "TorchLean tape supplies the masked-softmax VJP with the mask treated as constant."

/-- Reference sum reduction. -/
def reduceSum : KernelCapsule := reductionCapsule .reduceSum
/-- Reference arithmetic-mean reduction. -/
def reduceMean : KernelCapsule := reductionCapsule .reduceMean

/-- Reference shape-preserving reshape view. -/
def reshape : KernelCapsule := viewCapsule .reshape
/-- Reference axis permutation. -/
def permute : KernelCapsule := viewCapsule .permute
/-- Reference tensor broadcasting. -/
def broadcast : KernelCapsule := viewCapsule .broadcast
/-- Reference tensor concatenation. -/
def concat : KernelCapsule := viewCapsule .concat
/-- Reference contiguous tensor slice. -/
def slice : KernelCapsule := viewCapsule .slice
/-- Reference indexed gather. -/
def gather : KernelCapsule := viewCapsule .gather
/-- Reference indexed scatter-add. -/
def scatterAdd : KernelCapsule :=
  accumulationCapsule
    "reference.scatter_add"
    .scatterAdd
    "Indexed source values accumulate into the base tensor, including repeated indices."
    "The VJP gathers output gradients at the source indices and preserves the base gradient."

/-- Reference seeded uniform-random tensor generation. -/
def randUniform : KernelCapsule :=
  forwardOnlyCapsule
    .randUniform
    "Reference deterministic random-uniform tensors follow the seeded spec/runtime contract."

/-- Reference seeded Bernoulli-mask generation. -/
def bernoulliMask : KernelCapsule :=
  forwardOnlyCapsule
    .bernoulliMask
    "Reference deterministic Bernoulli masks follow the seeded spec/runtime contract."

/-- Reference top-`k` routing mask. -/
def topKMask : KernelCapsule :=
  forwardOnlyCapsule
    .topKMask
    ("Reference top-k masks evaluate `Spec.MoE.topKMaskSpec`: comparisons and counts, ties to " ++
      "the lower index.")

/-- Reference matmul path. -/
def matmul : KernelCapsule :=
  accumulationCapsule
    "reference.matmul"
    .matmul
    "Portable matmul follows the spec-level matrix product contract."
    "TorchLean tape supplies the VJP."

/-- Reference scaled contraction, distinguished from fixed-left ordinary matrix multiplication. -/
def scaledMatmul : KernelCapsule :=
  { capsule "reference.scaled_matmul" .scaledMatmul
      "A fixed scale remains inside the matrix contraction's finite-coefficient replay boundary."
      "TorchLean records the scaled contraction's local cotangents." with
    valueContract :=
      { claim := .valueRefinement .scaledMatmul
        summary := "CPU tape and IR evaluation reuse the canonical scaled contraction."
        evidence := .sharedDefinition "Spec.Internal.scaledProducts" }
    vjpContract :=
      { claim := .vjpRefinement .scaledMatmul .torchLeanTape
        summary := "Both cotangents retain the fixed scale through coefficient replay."
        evidence := .sharedDefinition "Spec.Internal.scaledMatmulBackward" }
    numericalPolicy := { reduction := .fixedLeftOrExact } }

/-- A compound pullback keeps original cotangents through row normalization. -/
def softmaxMatmul : KernelCapsule :=
  { capsule "reference.softmax_matmul" .softmaxMatmul
      "Row softmax, a pointwise multiplier, then ordinary matrix multiplication."
      "The centered pullback retains original contraction coefficients." with
    valueContract :=
      { claim := .valueRefinement .softmaxMatmul
        summary := "The CPU tape reuses the normalized matrix-product specification."
        evidence := .sharedDefinition
          "Spec.softmaxMatmulWeights; Spec.Internal.softmaxMatmulFromWeights" }
    vjpContract :=
      { claim := .vjpRefinement .softmaxMatmul .torchLeanTape
        summary := "The CPU tape reuses the retained centered pullback."
        evidence := .sharedDefinition "Spec.softmaxMatmulBackward" }
    numericalPolicy := { reduction := .implementationDefined } }

/-- Reference linear layer. -/
def linear : KernelCapsule :=
  accumulationCapsule
    "reference.linear"
    .linear
    "Reference linear follows the matvec/matmul plus bias contract."
    "TorchLean tape supplies the VJP."

/-- Reference mean-squared-error loss. -/
def mseLoss : KernelCapsule :=
  accumulationCapsule
    "reference.mse_loss"
    .mseLoss
    "Reference MSE follows the mean squared residual contract."
    "TorchLean tape supplies the VJP."

/-- Weighted logits cross-entropy with both pullbacks supplied by the shared specification. -/
def crossEntropy : KernelCapsule :=
  { Internal.accumulationCapsule "reference.cross_entropy" .crossEntropy
      "Spec.crossEntropyLogitsSpec retains original logits for overflowing weighted terms."
      "The shared logits and target-weight derivatives supply both pullbacks." with
    valueContract.evidence := .sharedDefinition "Spec.crossEntropyLogitsSpec"
    vjpContract.evidence := .sharedDefinition
      "Spec.crossEntropyLogitsDerivSpec and Spec.CrossEntropy.targetGradient" }

/-- Fixed-teacher cross-entropy using the shared loss and cancellation-resistant gradient. -/
def distill : KernelCapsule :=
  { Internal.accumulationCapsule "reference.distill" .distill
      "Spec.distillSpec evaluates cross-entropy from teacher logits."
      "Spec.distillDerivSpec supplies the student VJP; the teacher is fixed." with
    valueContract.evidence := .sharedDefinition "Spec.distillSpec"
    vjpContract.evidence := .sharedDefinition "Spec.distillDerivSpec" }

/-- Reference layer normalization. -/
def layerNorm : KernelCapsule :=
  accumulationCapsule
    "reference.layer_norm"
    .layerNorm
    "Reference LayerNorm follows the per-row normalization contract."
    "TorchLean tape supplies the VJP."

/-- Reference batch normalization. -/
def batchNorm : KernelCapsule :=
  accumulationCapsule
    "reference.batch_norm"
    .batchNorm
    "Reference BatchNorm follows the channel-first normalization contract."
    "TorchLean tape supplies the VJP."

/-- Reference generic channel-first convolution. -/
def conv : KernelCapsule :=
  accumulationCapsule
    "reference.conv"
    .conv
    "Reference convolution follows the generic channel-first contract."
    "TorchLean tape supplies the VJP."

/-- Reference generic channel-first transpose convolution. -/
def convTranspose : KernelCapsule :=
  accumulationCapsule
    "reference.conv_transpose"
    .convTranspose
    "Reference transpose convolution follows the generic channel-first contract."
    "TorchLean tape supplies the VJP."

/-- Reference max pooling. -/
def maxPool : KernelCapsule :=
  let op := BackendOp.maxPool
  accumulationCapsule
    s!"reference.{op.name}" op
    s!"Reference `{op.name}` follows the channel-first runtime contract."
    s!"TorchLean tape supplies the `{op.name}` VJP where differentiable."

/-- Reference smooth max pooling. -/
def smoothMaxPool : KernelCapsule :=
  accumulationCapsule
    "reference.smooth_max_pool"
    .smoothMaxPool
    "Reference smooth max pooling uses stable max/min-shifted window weights."
    "TorchLean tape supplies the VJP."

/-- Reference average pooling. -/
def avgPool : KernelCapsule :=
  accumulationCapsule
    "reference.avg_pool"
    .avgPool
    "Reference average-pooling follows the channel-first window contract."
    "TorchLean tape supplies the VJP."

/-- Reference attention with fixed-order arithmetic and finite-coefficient range repair. -/
def attention : KernelCapsule :=
  { name := "reference.attention"
    op := .attention
    provider := .reference
    device := .cpu
    trustLevel := .checked
    supportsForward := true
    vjpMode := .torchLeanTape
    shapeContract := ContractDescriptor.guarded (.shapeSafety .attention)
      "Q/K/V and mask shapes are checked by the typed tensor layer."
      "typed attention shapes"
    layoutContract := ContractDescriptor.guarded
      (.layoutCompatibility .attention .canonicalTensor)
      "Reference attention uses TorchLean tensor semantics rather than a foreign layout."
      "typed tensor layout"
    valueContract :=
      { claim := .valueRefinement .attention
        summary := "Composed reference attention uses hard-mask zero numerators and replays " ++
          "complete finite scaled contractions after range loss or an absorbed nonzero term."
        evidence := .sharedDefinition "Spec.MultiHeadAttention.forward" }
    vjpContract :=
      { claim := .vjpRefinement .attention .torchLeanTape
        summary := "TorchLean tape supplies the composed VJP."
        evidence := .sharedDefinition "Spec.multiHeadAttentionBackward" }
    numericalPolicy := { reduction := .implementationDefined } }

/-- Cross-platform reference capsules. -/
def capsules : Array KernelCapsule :=
  #[ matmul
  , scaledMatmul
  , softmaxMatmul
  , linear
  , mseLoss
  , crossEntropy
  , distill
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
  , silu
  , tanh
  , softplus
  , exp
  , log
  , sin
  , cos
  , inv
  , div
  , neg
  , safeLog
  , logSoftmax
  , softmax
  , hardMaskedSoftmax
  , topKMask
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
  , attention
  ]

end Reference
end Backend
end NN
