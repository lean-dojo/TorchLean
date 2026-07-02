/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.OpContracts
public import NN.Runtime.Context
public import NN.Spec.Layers.Conv
public import NN.Spec.Layers.Normalization

/-!
# IR Payloads

Shared payload records for IR evaluators and verifier backends.

The graph stores operation tags and edges. Tensor-valued constants, weights, convolution kernels,
and BatchNorm running statistics live in a separate payload keyed by node id, matching the way
formats such as ONNX keep graph structure separate from initializers.
-/

@[expose] public section

namespace NN.IR

open _root_.Spec
open _root_.Spec.Tensor

/--
Payload record for a `const` node.

Constants are stored in a flat representation so backends can use one vector container and let IR
evaluation reshape the data to the node's declared output shape.
-/
structure ConstFlat (α : Type) [Context α] where
  /-- Number of scalar entries stored in the flat constant payload. -/
  n : Nat
  /-- Constant values stored as a vector before evaluation reshapes them to the IR node shape. -/
  v : Tensor α (.dim n .scalar)

/--
Payload record for a `linear` node: weight matrix `W` and bias vector `b`.

The node's input `x` comes from the graph edge; `W,b` live in the external `Payload`, similar to
ONNX initializers or a PyTorch `state_dict`.
-/
structure LinearWB (α : Type) [Context α] where
  /-- Output dimension. -/
  outDim : Nat
  /-- Input dimension. -/
  inDim  : Nat
  /-- Weight matrix in the PyTorch convention `outDim × inDim`. -/
  W : Tensor α (.dim outDim (.dim inDim .scalar))
  /-- Bias vector added after matrix-vector multiplication. -/
  b : Tensor α (.dim outDim .scalar)

/--
Payload record for a `conv2d` node.

The spec-layer `Conv2DSpec` carries the typed kernel and bias. The cached dimensions let verifier
passes reconstruct flat shapes without unpacking the spec package at every use site.
-/
structure Conv2DParams (α : Type) [Context α] where
  /-- Input channels. -/
  inC : Nat
  /-- Output channels. -/
  outC : Nat
  /-- Kernel height. -/
  kH : Nat
  /-- Kernel width. -/
  kW : Nat
  /-- Stride. -/
  stride : Nat
  /-- Padding size. -/
  padding : Nat
  /-- Input height. -/
  inH : Nat
  /-- Input width. -/
  inW : Nat
  /-- Proof that the input channel count is nonzero, required by the spec convolution layer. -/
  hIn : inC ≠ 0
  /-- Proof that the kernel height is nonzero. -/
  hKH : kH ≠ 0
  /-- Proof that the kernel width is nonzero. -/
  hKW : kW ≠ 0
  /-- Proof that the stride is nonzero. -/
  hStride : stride ≠ 0
  /-- Spec-layer convolution package containing weights, bias, and convolution metadata. -/
  spec : Spec.Conv2DSpec inC outC kH kW stride padding α hIn hKH hKW

/-- Payload record for eval-mode BatchNorm2d over `N×C×H×W` tensors. -/
structure BatchNorm2DNchwEvalParams (α : Type) [Context α] where
  /-- Channel count. -/
  c : Nat
  /-- Affine scale. -/
  gamma : Tensor α (.dim c .scalar)
  /-- Affine bias. -/
  beta : Tensor α (.dim c .scalar)
  /-- Running mean. -/
  mean : Tensor α (.dim c .scalar)
  /-- Running variance. -/
  var : Tensor α (.dim c .scalar)
  /-- Epsilon added to the running variance before taking the square root. -/
  eps : α

/--
External parameter payloads keyed by IR node id.

This is focused on denotational IR evaluation. Runtime backends may store tensors differently, but
their proof-facing semantics pass through this shape-indexed boundary.
-/
structure Payload (α : Type) [Context α] where
  /-- Flat constants keyed by the `const` node id. -/
  const?  : Nat → Option (ConstFlat α) := fun _ => none
  /-- Linear weights and bias keyed by the `linear` node id. -/
  linear? : Nat → Option (LinearWB α) := fun _ => none
  /-- Convolution parameters keyed by the `conv2d` node id. -/
  conv2d? : Nat → Option (Conv2DParams α) := fun _ => none
  /-- Eval-mode BatchNorm parameters keyed by the `batchNorm2dNchwEval` node id. -/
  batchNorm2dNchwEval? : Nat → Option (BatchNorm2DNchwEvalParams α) := fun _ => none

end NN.IR
