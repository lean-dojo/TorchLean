/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.TapeM
public import NN.Runtime.Autograd.Train.Dataset

/-!
# Training-facing TapeM helpers

The core tape builder lives in `NN.Runtime.Autograd.Engine.TapeM`; that file owns the operation
vocabulary and reverse-mode execution. This module is intentionally smaller: it contains the
training conveniences that make loss construction read cleanly without defining a second tape API.

The main helpers are:

- `param` for trainable leaves (`requires_grad := true`);
- `const` for data or frozen leaves (`requires_grad := false`);
- `meanScalarOver`, `meanScalarOverArray`, and `meanScalarOverDataset` for averaged scalar losses.

## Higher derivatives

`Tape.backwardScalar` is a first-order reverse pass over a completed tape. It returns gradient
values, but it does not record the backward pass itself as a differentiable graph. So this layer is
the right place for ordinary training losses, not for Hessians or differentiating-through-backward.

For higher derivatives, use the functional autodiff surface in `NN.Runtime.Autograd.TorchLean`
(`hvpInputs`, `hessian1`, and the public API wrappers). That path rebuilds the program over dual
numbers / compiled graph structure and is the correct architecture for JVP-over-VJP style
derivatives.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Train
namespace TapeM

open Spec
open Tensor

/--
Create a trainable leaf node.

This is the training alias for `Runtime.Autograd.TapeM.leaf` with `requires_grad := true`, matching
the role of a parameter tensor in a PyTorch-style eager tape.
-/
def param {a : Type} {s : Shape}
  (value : Tensor a s) (name : Option String := none) : Runtime.Autograd.TapeM a Nat :=
  Runtime.Autograd.TapeM.leaf value (name := name) (requires_grad := true)

/--
Create a constant/data leaf node.

Use this for minibatch inputs, labels, masks, and frozen tensors. The value is still used by the
forward computation, but `backwardScalar` will not accumulate a gradient for it as a leaf.
-/
def const {a : Type} {s : Shape}
  (value : Tensor a s) (name : Option String := none) : Runtime.Autograd.TapeM a Nat :=
  Runtime.Autograd.TapeM.leaf value (name := name) (requires_grad := false)

/-!
Compute the mean of a dataset of scalar-valued losses.

`lossOf x` must return a node id whose value has shape `.scalar`.
-/
/-- Mean reduction for a list of scalar-valued losses, written in `TapeM`.

This is a common pattern in training loops: compute a scalar loss per sample, sum, then scale by
`1/N`.
-/
def meanScalarOver {a b : Type}
  [Add a] [Mul a] [Div a] [One a] [Coe Nat a] [DecidableEq Shape]
  (tag : String) (xs : List b) (lossOf : b -> Runtime.Autograd.TapeM a Nat) :
  Runtime.Autograd.TapeM a Nat := do
  match xs with
  | [] =>
      throw (tagError tag "empty dataset")
  | x0 :: xs =>
      let firstLossId ← lossOf x0
      let sumLossId ← xs.foldlM (init := firstLossId) fun acc x => do
        let lossId ← lossOf x
        Runtime.Autograd.TapeM.add (s := Shape.scalar) acc lossId
      let n : Nat := xs.length.succ
      let invN : a := (1 : a) / (n : a)
      Runtime.Autograd.TapeM.scale (s := Shape.scalar) sumLossId invN

/--
Mean reduction for an array-backed batch.

Arrays are common at runtime boundaries, while `meanScalarOver` uses lists because `foldlM` over
lists keeps the implementation and error behavior simple. This wrapper makes that conversion
explicit and keeps call sites tidy.
-/
def meanScalarOverArray {a b : Type}
  [Add a] [Mul a] [Div a] [One a] [Coe Nat a] [DecidableEq Shape]
  (tag : String) (xs : Array b) (lossOf : b -> Runtime.Autograd.TapeM a Nat) :
  Runtime.Autograd.TapeM a Nat :=
  meanScalarOver (tag := tag) xs.toList lossOf

/--
Mean reduction for a `Dataset`.

This is the natural bridge from `Train.Dataset` batches to a scalar loss node. It does not shuffle,
batch, or otherwise mutate the dataset; it simply materializes the current dataset order as a list
and delegates to `meanScalarOver`.
-/
def meanScalarOverDataset {a b : Type}
  [Add a] [Mul a] [Div a] [One a] [Coe Nat a] [DecidableEq Shape]
  (tag : String) (xs : Dataset b) (lossOf : b -> Runtime.Autograd.TapeM a Nat) :
  Runtime.Autograd.TapeM a Nat :=
  meanScalarOver (tag := tag) xs.toList lossOf

/-- Array-batch mean loss is exactly list-batch mean loss after `Array.toList`. -/
@[simp] theorem meanScalarOverArray_eq_meanScalarOver {a b : Type}
    [Add a] [Mul a] [Div a] [One a] [Coe Nat a] [DecidableEq Shape]
    (tag : String) (xs : Array b) (lossOf : b -> Runtime.Autograd.TapeM a Nat) :
    meanScalarOverArray (a := a) (b := b) tag xs lossOf =
      meanScalarOver (tag := tag) xs.toList lossOf := rfl

/-- Dataset-batch mean loss is exactly list-batch mean loss after `Dataset.toList`. -/
@[simp] theorem meanScalarOverDataset_eq_meanScalarOver {a b : Type}
    [Add a] [Mul a] [Div a] [One a] [Coe Nat a] [DecidableEq Shape]
    (tag : String) (xs : Dataset b) (lossOf : b -> Runtime.Autograd.TapeM a Nat) :
    meanScalarOverDataset (a := a) (b := b) tag xs lossOf =
      meanScalarOver (tag := tag) xs.toList lossOf := rfl

end TapeM
end Train
end Autograd
end Runtime
