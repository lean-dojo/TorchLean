/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.Base

/-!
# Derivative Transfers for Structured Operators

Convolution is affine in its input. With weights held fixed, every input derivative passes
through the same convolution with zero bias. This preserves the original groups, dilation,
padding, strides, and leading batch shape without materializing a dense matrix.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN NN.IR

variable {α : Type} [TorchLean.Storage α] [Context α] [BoundOps α]

/-- Apply the input differential of a convolution to a first or mixed-second derivative box. -/
def convDerivativeBox? (nodes : Array Node) (ps : ParamStore α)
    (derivatives : Array (Option (FlatBox α))) (id : Nat) (node : Node)
    (configuration : NN.IR.ConvConfig) : Option (FlatBox α) := do
  let parentId ← unaryParent? node.parents
  let parent ← nodes[parentId]?
  let direction ← (derivatives[parentId]?).join
  let parameters ← ps.convCfg[id]?
  let derivativeParameters :=
    { parameters with
      spec := { parameters.spec with bias := Tensor.full [parameters.outChannels] 0 } }
  let derivativeStore := { ps with convCfg := ps.convCfg.insert id derivativeParameters }
  ibpConvNode configuration parent.outShape node.outShape id derivativeStore direction

end NN.MLTheory.CROWN.Graph
