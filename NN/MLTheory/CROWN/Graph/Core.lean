/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Graph
public import NN.MLTheory.CROWN.Extras.BoundOpsIEEE32Exec
public import NN.MLTheory.CROWN.Flatbox
public import NN.MLTheory.CROWN.Operators.Arithmetic
public import NN.MLTheory.CROWN.Operators.Conv
public import NN.Spec.Layers.Pooling

/-!
# CROWN Graph

Graph-based LiRPA scaffolding (IBP + CROWN-style affine bounds).

This file is the "graph engine" counterpart to `NN.MLTheory.CROWN.Core`:
- `core` defines the scalar/box/affine primitives (`Box`, `AffineVec`, `IBP.linear`),
- `graph` lifts those primitives to arbitrary tensor DAGs (`NN.IR.Graph`).

What this module is for:
- Representing computation graphs over typed tensors.
- Propagating *interval* bounds forward (IBP).
- Propagating *affine* bounds (CROWN/DeepPoly style) w.r.t. a designated input node.
- Optionally running objective-dependent backward passes for tighter bounds.

Design notes:
- We store *flattened* bounds (`FlatBox`, `FlatAffine*`) so we can reuse `AffineVec` unchanged.
- Sequence models are represented by unrolling recurrent cells into a DAG (shared parameters,
  repeated ops).
- Transformer-style models rely on matmul/softmax/elementwise ops; relaxations are added
  incrementally.

References:
- auto_LiRPA: Xu et al.,
  "Automatic Perturbation Analysis for Scalable Certified Robustness and Beyond",
  NeurIPS 2020, arXiv:2002.12920 (https://arxiv.org/abs/2002.12920).
- CROWN: Zhang et al.,
  "Efficient Neural Network Robustness Certification with General Activation Functions",
  arXiv:1811.00866 (https://arxiv.org/abs/1811.00866).

PyTorch analogues (conceptual):
- `torch.fx` graphs as a user-facing DAG representation: https://pytorch.org/docs/stable/fx.html
- `auto_LiRPA` as a practical LiRPA implementation over PyTorch graphs.
-/

public section


namespace NN.MLTheory.CROWN

/-- Alias for the typed IR computation graph used by the CROWN/LiRPA engines. -/
abbrev Graph := NN.IR.Graph

end NN.MLTheory.CROWN

namespace NN.MLTheory.CROWN.Graph

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN
open NN.IR
open Std

/-- Alias for the IR node kind enumeration used by the graph engine. -/
abbrev OpKind := NN.IR.OpKind
/-- Alias for the IR node record used by the graph engine. -/
abbrev Node := NN.IR.Node

/--
Flattened affine form for a node output with respect to a fixed flattened input.

This represents `y ≈ A*x + c` for a chosen input node `x`.
-/
structure FlatAffine (α : Type) [Context α] where
  /-- Flattened input dimension. -/
  inDim  : Nat
  /-- Flattened output dimension. -/
  outDim : Nat
  /-- Affine form `A,c`. -/
  aff    : AffineVec α inDim outDim

/-- Flattened affine **lower and upper** bounds for a node output w.r.t. a fixed flattened input. -/
structure FlatAffineBounds (α : Type) [Context α] where
  /-- Flattened input dimension. -/
  inDim  : Nat
  /-- Flattened output dimension. -/
  outDim : Nat
  /-- Lower affine bound form. -/
  loAff  : AffineVec α inDim outDim
  /-- Upper affine bound form. -/
  hiAff  : AffineVec α inDim outDim

/--
Per-node bound state (flattened).

The option fields record which analyses have populated a node: an interval-only pass fills `ibp?`,
while affine CROWN passes additionally fill `aff?`.
-/
structure NodeState (α : Type) [Context α] where
  /-- Original (unflattened) tensor shape of the node output. -/
  shape : Shape
  /-- Interval bounds (IBP) if available. -/
  ibp?  : Option (FlatBox α)     := none
  /-- Affine form if available. -/
  aff?  : Option (FlatAffine α)  := none

/-- Propagation workspace across the whole graph. -/
structure PropState (α : Type) [Context α] where
  /-- Which node id is treated as the designated input for affine bounds. -/
  inputId   : Nat
  /-- Flattened input dimension. -/
  inputDim  : Nat
  /-- Per-node bound states. -/
  states    : Array (NodeState α)

/-
Coverage map for propagation rules:

Forward (IBP):
- add/sub: interval add/sub componentwise
- mul_elem: McCormick envelopes for elementwise product
- matmul/linear/conv2d: interval matrix multiplication as in `IBP.linear`/conv IBP
- relu/tanh/sigmoid/exp/log: elementwise monotone bounds (use activation-specific rules)
- softmax/layernorm: conservative last-axis interval bounds

Backward (CROWN):
- relu/tanh/sigmoid: per-neuron linear relaxations (like ReLU; tanh/sigmoid need convex hull)
- matmul/linear/conv2d: compose affine forms via matrix multiplication
- mul_elem: bilinear relaxation via McCormick (introduces additional linear terms)
- softmax/layernorm/mul_elem: conservative affine enclosures in the executable engine

Sequence models (RNN/GRU/LSTM):
- Unroll time steps as repeated nodes; gates are linear → nonlinearity → elementwise product
- Apply rules above for tanh/sigmoid and elementwise products.

Transformers:
- Provide ops for Q,K,V projections (linear), attention scores (matmul), softmax, and output matmul.
- Optional: bound sharing across heads; efficient batched propagation.
-/

end NN.MLTheory.CROWN.Graph
