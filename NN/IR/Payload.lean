/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Conv
public import NN.IR.Graph
public import NN.Tensor.Conversion

/-!
# IR Payloads

Shared payload records for IR evaluators and verifier backends.

The graph stores operation names and edges. Tensor-valued constants, weights, convolution kernels,
and BatchNorm running statistics live in a separate payload keyed by node id, matching the way
formats such as ONNX keep graph structure separate from initializers.
-/

@[expose] public section

namespace NN.IR

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor

/--
Payload record for a `const` node.

Constants are stored in a flat representation so backends can use one vector container and let IR
evaluation reshape the data to the node's declared output shape.
-/
structure ConstFlat (α : Type) [TorchLean.Storage α] [Context α] where
  /-- Number of scalar entries stored in the flat constant payload. -/
  n : Nat
  /-- Constant values stored as a vector before evaluation reshapes them to the IR node shape. -/
  v : Tensor α [n]

/--
Payload record for a `linear` node: weight matrix `W` and bias vector `b`.

The node's input `x` comes from the graph edge; `W,b` live in the external `Payload`, similar to
ONNX initializers or a PyTorch `state_dict`.
-/
structure LinearWB (α : Type) [TorchLean.Storage α] [Context α] where
  /-- Output dimension. -/
  outDim : Nat
  /-- Input dimension. -/
  inDim  : Nat
  /-- Weight matrix in the PyTorch convention `outDim × inDim`. -/
  W : Tensor α [outDim, inDim]
  /-- Bias vector added after matrix-vector multiplication. -/
  b : Tensor α [outDim]

/-- Payload for an arbitrary-dimensional convolution node. -/
structure ConvParams (α : Type) [TorchLean.Storage α] [Context α] where
  /-- Number of spatial axes. -/
  spatialRank : Nat
  /-- Input channels. -/
  inChannels : Nat
  /-- Output channels. -/
  outChannels : Nat
  /-- Per-axis kernel extents. -/
  kernel : TorchLean.Tensor Nat [spatialRank]
  /-- Per-axis strides. -/
  stride : TorchLean.Tensor Nat [spatialRank]
  /-- Zero padding before each spatial axis. -/
  padding : TorchLean.Tensor Nat [spatialRank]
  /-- Spacing between adjacent kernel samples. -/
  dilation : TorchLean.Tensor Nat [spatialRank] := Tensor.full [spatialRank] 1
  /-- Zero padding after each spatial axis. `padding` is the padding before the input. -/
  paddingAfter : TorchLean.Tensor Nat [spatialRank] := padding
  /-- Number of independent channel groups. -/
  groups : Nat := 1
  /-- Spatial shape of one input sample. -/
  inputSpatial : TorchLean.Tensor Nat [spatialRank]
  /-- Every kernel extent is nonzero. -/
  kernelNonzero : ∀ i : Fin spatialRank, kernel.getScalar i ≠ 0
  /-- Every stride is nonzero. -/
  strideNonzero : ∀ i : Fin spatialRank, stride.getScalar i ≠ 0
  /-- Typed weights, bias, and convolution geometry. -/
  spec : Spec.ConvSpec spatialRank inChannels outChannels kernel stride padding α

namespace ConvParams

/-- Whether a convolution payload implements the geometry declared by an IR node. -/
def matchesConfig {α : Type} [TorchLean.Storage α] [Context α]
    (params : ConvParams α) (config : ConvConfig) : Bool :=
  params.spatialRank == config.spatialRank &&
    params.inChannels == config.inChannels &&
    params.outChannels == config.outChannels &&
    Tensor.to params.kernel (List Nat) == Tensor.to config.kernel (List Nat) &&
    Tensor.to params.stride (List Nat) == Tensor.to config.stride (List Nat) &&
    Tensor.to params.padding (List Nat) == Tensor.to config.padding (List Nat) &&
    Tensor.to params.paddingAfter (List Nat) == Tensor.to config.paddingAfter (List Nat) &&
    Tensor.to params.dilation (List Nat) == Tensor.to config.dilation (List Nat) &&
    params.groups == config.groups

/-- Input shape expected by a convolution payload after preserving the graph's leading axes. -/
def input {α : Type} [TorchLean.Storage α] [Context α]
    (params : ConvParams α) (leading : Shape) : Shape :=
  leading.concat
    (Shape.ofList (params.inChannels :: Tensor.to params.inputSpatial (List Nat)))

/-- Output shape produced by the typed convolution payload for the given leading axes. -/
def output {α : Type} [TorchLean.Storage α] [Context α]
    (params : ConvParams α) (leading : Shape) : Shape :=
  leading.concat <| Shape.ofList <|
    params.outChannels ::
      Tensor.to
        (Spec.convOutSpatialDilated params.inputSpatial params.kernel params.stride params.dilation
          params.padding params.paddingAfter)
        (List Nat)

end ConvParams

/-- Payload for eval-mode BatchNorm along a channel axis selected by the graph node. -/
structure BatchNormEvalParams (α : Type) [TorchLean.Storage α] [Context α] where
  /-- Channel count. -/
  c : Nat
  /-- Affine scale. -/
  gamma : Tensor α [c]
  /-- Affine bias. -/
  beta : Tensor α [c]
  /-- Running mean. -/
  mean : Tensor α [c]
  /-- Running variance. -/
  var : Tensor α [c]
  /-- Epsilon added to the running variance before taking the square root. -/
  eps : α

/-- Affine parameters and epsilon for LayerNorm over an arbitrary normalized suffix. -/
structure LayerNormParams (α : Type) [TorchLean.Storage α] [Context α] where
  /-- Shape of the suffix normalized by the corresponding node. -/
  normalizedShape : Shape
  /-- Learned elementwise scale over `normalizedShape`. -/
  gamma : Tensor α normalizedShape
  /-- Learned elementwise bias over `normalizedShape`. -/
  beta : Tensor α normalizedShape
  /-- Epsilon added to the variance before taking the square root. -/
  eps : α

/--
External parameter payloads keyed by IR node id.

This is focused on denotational IR evaluation. Runtime backends may store tensors differently, but
their proof layer semantics pass through this shape-indexed boundary.
-/
structure Payload (α : Type) [TorchLean.Storage α] [Context α] where
  /-- Flat constants keyed by the `const` node id. -/
  const?  : Nat → Option (ConstFlat α) := fun _ => none
  /-- Linear weights and bias keyed by the `linear` node id. -/
  linear? : Nat → Option (LinearWB α) := fun _ => none
  /-- Convolution parameters keyed by the `conv` node id. -/
  conv? : Nat → Option (ConvParams α) := fun _ => none
  /-- Eval-mode BatchNorm parameters keyed by the `batchNormEval` node id. -/
  batchNormEval? : Nat → Option (BatchNormEvalParams α) := fun _ => none
  /-- Affine LayerNorm parameters keyed by the `layernorm` node id. -/
  layerNorm? : Nat → Option (LayerNormParams α) := fun _ => none

end NN.IR
