/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Core.Base

/-!
# Core training helpers

These are small, reusable utilities that keep training scripts short and readable.
They are pure and local.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Train

open Spec TorchLean
open TorchLean TorchLean.Tensor

/--
Prefix an error message with a caller-provided tag.

This is used throughout the training helpers to keep error messages readable when multiple
subsystems can fail (tape execution, dataset loading, shape checks, etc.).
-/
def tagError (tag msg : String) : String :=
  s!"{tag}: {msg}"

/-!
## Typed extraction helpers

The autograd engine stores values and gradients as `Spec.SomeTensor` (shape tag + tensor).
These helpers check shapes and give you back a typed tensor or a scalar value.
-/

/--
Read a typed gradient tensor `Tensor a s` from a gradient map keyed by node id.

The eager tape engine stores gradients in a shape-erased form (`Spec.SomeTensor a`), so this
helper performs the dynamic shape check and returns a typed tensor on success.
-/
def requireGradTensor {a : Type} [TorchLean.Storage a] {s : Shape}
  (tag : String) (grads : Std.HashMap Nat (Spec.SomeTensor a)) (id : Nat) :
  Result (Tensor a s) := by
  match grads.get? id with
  | none =>
      exact .error (tagError tag s!"missing gradient for node id {id}")
  | some packed =>
      if h : packed.shape = s then
        exact .ok (packed.cast h)
      else
        exact .error (tagError tag s!"gradient shape mismatch for node id {id}")

/--
Read a typed forward value `Tensor a s` from a tape node id.

This is the value-side analogue of `requireGradTensor`: it checks the shape stored in the packed
tensor before recovering a statically shaped value.
-/
def requireValueTensor {a : Type} [TorchLean.Storage a] {s : Shape}
  (tag : String) (t : Tape a) (id : Nat) : Result (Tensor a s) := by
  match t.getValue? id with
  | none =>
      exact .error (tagError tag s!"missing value for node id {id}")
  | some packed =>
      if h : packed.shape = s then
        exact .ok (packed.cast h)
      else
        exact .error (tagError tag s!"value shape mismatch for node id {id}")

/--
Read a scalar forward value from a tape node id.

This is a common pattern in training scripts where the loss is a scalar node.
-/
def requireScalarValue {a : Type} [TorchLean.Storage a]
  (tag : String) (t : Tape a) (id : Nat) : Result a := do
  let tScalar : Tensor a .scalar ←
    requireValueTensor (tag := tag) (s := Shape.scalar) t id
  pure (Tensor.item tScalar)

/-!
## SGD update helper

This is a small tensor-level update used by many tests.
-/
end Train
end Autograd
end Runtime
