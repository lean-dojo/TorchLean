/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.DAG.Syntax
public import NN.Runtime.Autograd.Model.Functional.ShapeOps

/-!
# DAG Nonlinear and Attention Primitives

Elementwise nonlinearities, softmax, and multi-head self-attention for typed DAG models.
-/

@[expose] public section

namespace NN
namespace GraphSpec
namespace DAG

open Spec TorchLean
open TorchLean.Tensor

namespace PrimOp

/-- Coordinatewise sigmoid. -/
def sigmoid (s : Shape) : PrimOp [s] s :=
  { name := "sigmoid"
    specFwd := fun {α} _ _ xs =>
      match xs with
      | .cons input .nil => Activation.sigmoidSpec (α := α) input
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input =>
        Runtime.Autograd.Model.sigmoid (m := m) (α := α) input }

/-- Pure evaluation of coordinatewise sigmoid. -/
@[simp] theorem sigmoid_specFwd {α : Type} [TorchLean.Storage α] [Context α] {s : Shape}
    (input : TorchLean.Tensor α s) :
    (sigmoid s).specFwd (.cons input .nil) = Activation.sigmoidSpec input := by
  rfl

/-- Coordinatewise SiLU, also called swish: `x * sigmoid x`. -/
def silu (s : Shape) : PrimOp [s] s :=
  { name := "silu"
    specFwd := fun {α} _ _ xs =>
      match xs with
      | .cons input .nil => Activation.swishSpec (α := α) input
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input =>
        Runtime.Autograd.Model.silu (m := m) (α := α) input }

/-- Pure evaluation of coordinatewise SiLU. -/
@[simp] theorem silu_specFwd {α : Type} [TorchLean.Storage α] [Context α] {s : Shape}
    (input : TorchLean.Tensor α s) :
    (silu s).specFwd (.cons input .nil) = Activation.swishSpec input := by
  rfl

/-- Coordinatewise Gaussian error linear unit using TorchLean's tanh approximation. -/
def gelu (s : Shape) : PrimOp [s] s :=
  { name := "gelu"
    specFwd := fun {α} _ _ xs =>
      match xs with
      | .cons input .nil => Activation.geluSpec (α := α) input
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input =>
        Runtime.Autograd.Model.gelu (m := m) (α := α) input }

/-- Pure evaluation of coordinatewise GELU. -/
@[simp] theorem gelu_specFwd {α : Type} [TorchLean.Storage α] [Context α] {s : Shape}
    (input : TorchLean.Tensor α s) :
    (gelu s).specFwd (.cons input .nil) = Activation.geluSpec input := by
  rfl

/-- Multi-head self-attention over an arbitrary leading shape.

The pure semantics applies `Spec.MultiHeadAttention.forward` independently at every leading index.
The executable path uses the prefix-polymorphic TorchLean operation and may flatten those axes for
backend execution without changing the mathematical operation.
-/
def multiHeadAttention (leading : Shape) (n numHeads dModel headDim : Nat) (hN : 0 < n) :
    PrimOp
      [ [dModel, numHeads * headDim],
        [dModel, numHeads * headDim],
        [dModel, numHeads * headDim],
        [numHeads * headDim, dModel],
        leading.concat [n, dModel] ]
      (leading.concat [n, dModel]) :=
  { name := s!"multiHeadAttention({n},{numHeads},{dModel},{headDim})"
    specFwd := fun {α} _ _ xs =>
      match xs with
      | .cons wq (.cons wk (.cons wv (.cons wo (.cons input .nil)))) =>
          let attention : Spec.MultiHeadAttention α numHeads dModel headDim :=
            { queryWeight := wq, keyWeight := wk, valueWeight := wv, outputWeight := wo }
          TorchLean.Tensor.mapLeading leading
            (fun inputs => attention.forward n (Nat.ne_of_gt hN) inputs none) input
    program := fun {α} _ _ =>
      fun {m} _ _ => fun wq wk wv wo input =>
        Runtime.Autograd.Model.multiHeadAttention (m := m) (α := α)
          (leadingShape := leading)
          (Nat.ne_of_gt hN) wq wk wv wo input none }

/-- Pure evaluation of prefix-polymorphic multi-head attention. -/
@[simp] theorem multiHeadAttention_specFwd
    {leading : Shape} {n numHeads dModel headDim : Nat} (hN : 0 < n)
    {α : Type} [TorchLean.Storage α] [Context α]
    (wq wk wv : TorchLean.Tensor α [dModel, numHeads * headDim])
    (wo : TorchLean.Tensor α [numHeads * headDim, dModel])
    (input : TorchLean.Tensor α (leading.concat [n, dModel])) :
    (multiHeadAttention leading n numHeads dModel headDim hN).specFwd
        (.cons wq (.cons wk (.cons wv (.cons wo (.cons input .nil))))) =
      TorchLean.Tensor.mapLeading leading (fun inputs =>
        ({ queryWeight := wq, keyWeight := wk, valueWeight := wv, outputWeight := wo } :
          Spec.MultiHeadAttention α numHeads dModel headDim).forward
          n (Nat.ne_of_gt hN) inputs none) input := by
  rfl


/-- Coordinatewise exponential. -/
def exp (s : Shape) : PrimOp [s] s :=
  { name := "exp"
    specFwd := fun {α} _ _ xs =>
      match xs with
      | .cons input .nil => TorchLean.Tensor.expSpec (α := α) input
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input =>
        Runtime.Autograd.Model.exp (m := m) (α := α) input }

/-- Coordinatewise multiplicative inverse. -/
def inv (s : Shape) : PrimOp [s] s :=
  { name := "inv"
    specFwd := fun {_α} _storage _ctx xs =>
      match xs with
      | .cons input .nil => TorchLean.Tensor.mapSpec (fun value => 1 / value) input
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input =>
        Runtime.Autograd.Model.inv (m := m) (α := α) input }

/-- The pure meaning of the DAG inverse node is coordinatewise reciprocal. -/
@[simp] theorem inv_specFwd {α : Type} [TorchLean.Storage α] [Context α] {s : Shape}
    (input : TorchLean.Tensor α s) :
    (inv s).specFwd (.cons input .nil) = TorchLean.Tensor.invSpec input := by
  rfl


/-- Coordinatewise hyperbolic tangent. -/
def tanh (s : Shape) : PrimOp [s] s :=
  { name := "tanh"
    specFwd := fun {α} _ _ xs =>
      match xs with
      | .cons input .nil => Activation.tanhSpec (α := α) input
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input =>
        Runtime.Autograd.Model.tanh (m := m) (α := α) input }

/-- Stable softmax along any in-bounds tensor dimension. -/
def softmax (s : Shape) (axis : Nat) [Spec.Shape.AxisInBounds axis s] : PrimOp [s] s :=
  { name := "softmax"
    specFwd := fun {α} _ _ xs =>
      match xs with
      | .cons input .nil => Activation.softmaxSpec (α := α) axis input
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input =>
        Runtime.Autograd.Model.F.softmax (m := m) (α := α) axis input }


end PrimOp

end DAG
end GraphSpec
end NN
