/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.NormNum.Inv
import Mathlib.Tactic.NormNum.Pow
import Mathlib.Tactic.Positivity.Finset
public import NN.Tensor.Internal.Elab.TensorLiteral
public import NN.Tensor.Operations
public import NN.Tensor -- shake: keep

/-!
# Low-Rank Adapters

LoRA represents a linear-weight update as two smaller matrices. For a base weight
$W : \mathbb{R}^{d_{in}\times d_{out}}$, an adapter of rank $r$ uses an input factor
$A : \mathbb{R}^{d_{in}\times r}$ and an output factor
$B : \mathbb{R}^{r\times d_{out}}$:

$$W_{eff}=W+sAB.$$

The matrix orientation agrees with TorchLean's row-batch linear layers. This module defines the
typed update and its action on a batch; the training code decides which parameters to optimize.

Reference: Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models” (2021),
https://arxiv.org/abs/2106.09685.
-/

@[expose] public section

namespace TorchLean.Adapters.LoRA

open Spec TorchLean
open TorchLean.Tensor

/-- LoRA factors for a linear weight of shape `inputWidth × outputWidth`. -/
structure Parameters (α : Type) [TorchLean.Storage α]
    (inputWidth rank outputWidth : Nat) where
  /-- Projection from the input dimension to the adapter rank. -/
  inputFactor : Tensor α [inputWidth, rank]
  /-- Projection from the adapter rank to the output dimension. -/
  outputFactor : Tensor α [rank, outputWidth]

/-- The scaled low-rank update $sAB$. -/
def weightUpdate {α : Type} [TorchLean.Storage α] [Add α] [Mul α] [Zero α]
    {inputWidth rank outputWidth : Nat}
    (parameters : Parameters α inputWidth rank outputWidth) (scale : α) :
    Tensor α [inputWidth, outputWidth] :=
  Tensor.scale (Tensor.matmul parameters.inputFactor parameters.outputFactor) scale

/-- Add a LoRA update to a base linear weight. -/
def effectiveWeight {α : Type} [TorchLean.Storage α] [Add α] [Mul α] [Sub α] [Zero α]
    {inputWidth rank outputWidth : Nat}
    (baseWeight : Tensor α [inputWidth, outputWidth])
    (parameters : Parameters α inputWidth rank outputWidth) (scale : α) :
    Tensor α [inputWidth, outputWidth] :=
  Tensor.add baseWeight (weightUpdate parameters scale)

/-- Apply a linear map whose weight is augmented by a LoRA update over an arbitrary batch shape. -/
def linear {α : Type} [TorchLean.Storage α] [Add α] [Mul α] [Sub α] [Zero α]
    {batchShape : Shape} {inputWidth rank outputWidth : Nat}
    (input : Tensor α (batchShape.appendDim inputWidth))
    (baseWeight : Tensor α [inputWidth, outputWidth])
    (parameters : Parameters α inputWidth rank outputWidth) (scale : α) :
    Tensor α (batchShape.appendDim outputWidth) := by
  let input' : Tensor α (batchShape.concat [inputWidth]) := by
    simpa only [Shape.appendDim_eq_concat] using input
  simpa only [Shape.appendDim_eq_concat] using
    TorchLean.Tensor.mapLeading batchShape
      (fun row =>
        Tensor.vecmat row (effectiveWeight baseWeight parameters scale))
      input'

end TorchLean.Adapters.LoRA
