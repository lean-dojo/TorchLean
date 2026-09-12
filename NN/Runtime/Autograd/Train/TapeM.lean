/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.TapeM
public import NN.Data.SampleStream

/-!
# Training-facing TapeM helpers

The core tape builder lives in `NN.Runtime.Autograd.Engine.TapeM`; that file owns the operation
vocabulary and reverse-mode execution. This module is narrower: it contains the
training conveniences that make loss construction read cleanly without defining a second tape API.

The main helpers are:

- `param` for trainable leaves (`requiresGrad := true`);
- `const` for data or frozen leaves (`requiresGrad := false`);
- `meanScalarOver` and `meanScalarOverDataset` for averaged scalar losses.

## Higher derivatives

`Tape.backwardScalar` is a first-order reverse pass over a completed tape. It returns gradient
values, but it does not record the backward pass itself as a differentiable graph. So this layer is
the right place for ordinary training losses, not for Hessians or differentiating-through-backward.

For higher derivatives, use the functional autodiff surface in `NN.Runtime.Autograd.Model`
(`hvpInputs`, `hessianInput`, and the public API wrappers). That path rebuilds the program over dual
numbers and typed graph structure; it is the correct architecture for JVP-over-VJP style
derivatives.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Train
namespace TapeM

open Spec TorchLean
open TorchLean TorchLean.Tensor

/--
Create a trainable leaf node.

This constructor records a leaf with `requiresGrad := true`, matching the role of a parameter
tensor in a PyTorch-style eager tape.
-/
def param {a : Type} [TorchLean.Storage a] {s : Shape}
  (value : Tensor a s) (name : Option String := none) : Runtime.Autograd.TapeM a Nat :=
  Runtime.Autograd.TapeM.leaf value (name := name) (requiresGrad := true)

/--
Create a constant/data leaf node.

Use this for minibatch inputs, labels, masks, and frozen tensors. The value is still used by the
forward computation, but `backwardScalar` will not accumulate a gradient for it as a leaf.
-/
def const {a : Type} [TorchLean.Storage a] {s : Shape}
  (value : Tensor a s) (name : Option String := none) : Runtime.Autograd.TapeM a Nat :=
  Runtime.Autograd.TapeM.leaf value (name := name) (requiresGrad := false)

/-!
Compute the mean of a dataset of scalar-valued losses.

`lossOf x` must return a node id whose value has shape `.scalar`.
-/
/-- Mean reduction for an array of scalar-valued losses, written in `TapeM`.

This is a common pattern in training loops: compute a scalar loss per sample, sum, then scale by
`1/N`.
-/
def meanScalarOver {a b : Type}
  [TorchLean.Storage a] [Add a] [Mul a] [Div a] [One a] [NatCast a]
  (tag : String) (xs : Array b) (lossOf : b -> Runtime.Autograd.TapeM a Nat) :
  Runtime.Autograd.TapeM a Nat := do
  match xs[0]? with
  | none =>
      throw s!"{tag}: empty dataset"
  | some x0 =>
      let firstLossId ← lossOf x0
      let sumLossId ← (xs.drop 1).foldlM (init := firstLossId) fun acc x => do
        let lossId ← lossOf x
        Runtime.Autograd.TapeM.add (s := Shape.scalar) acc lossId
      let n : Nat := xs.size
      let invN : a := (1 : a) / (n : a)
      Runtime.Autograd.TapeM.scale (s := Shape.scalar) sumLossId invN

/--
Mean reduction for a finite `SampleStream`.

This is the natural bridge from `TorchLean.Data.SampleStream` batches to a scalar loss node. It
materializes the current dataset order as an array and delegates to `meanScalarOver`, without
shuffling, batching, or mutating the dataset.
-/
def meanScalarOverDataset {a b : Type}
  [TorchLean.Storage a] [Add a] [Mul a] [Div a] [One a] [NatCast a]
  (tag : String) (xs : TorchLean.Data.SampleStream b)
  (lossOf : b -> Runtime.Autograd.TapeM a Nat) :
  Runtime.Autograd.TapeM a Nat :=
  meanScalarOver (tag := tag) xs.toArray lossOf

end TapeM
end Train
end Autograd
end Runtime
