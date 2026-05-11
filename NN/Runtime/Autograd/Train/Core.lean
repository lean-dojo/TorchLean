/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Core
public import NN.Runtime.Optim.Optimizers

/-!
# Core training helpers

These are small, reusable utilities that keep training scripts short and readable.
They are lightweight and pure.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Train

open Spec
open Tensor

/--
Prefix an error message with a caller-provided tag.

This is used throughout the training helpers to keep error messages readable when multiple
subsystems can fail (tape execution, dataset loading, shape checks, etc.).
-/
def tagError (tag msg : String) : String :=
  s!"{tag}: {msg}"

/-!
## Typed extraction helpers

The autograd engine stores values and gradients as `Runtime.AnyTensor` (shape tag + tensor).
These helpers check shapes and give you back a typed tensor or a scalar value.
-/

/--
Read a typed gradient tensor `Tensor a s` from a gradient map keyed by node id.

The eager tape engine stores gradients in a shape-erased form (`Runtime.AnyTensor a`), so this
helper performs the dynamic shape check and returns a typed tensor on success.
-/
def requireGradTensor {a : Type} [DecidableEq Shape] {s : Shape}
  (tag : String) (grads : Std.HashMap Nat (Runtime.AnyTensor a)) (id : Nat) :
  Result (Tensor a s) := by
  match grads.get? id with
  | none =>
      exact .error (tagError tag s!"missing gradient for node id {id}")
  | some any =>
      if h : any.s = s then
        exact .ok (Tensor.castShape any.t h)
      else
        exact .error (tagError tag s!"gradient shape mismatch for node id {id}")

/--
Read a typed forward value `Tensor a s` from a tape node id.

This is the value-side analogue of `requireGradTensor`: it performs a dynamic shape check on the
shape tag stored in `AnyTensor`.
-/
def requireValueTensor {a : Type} [DecidableEq Shape] {s : Shape}
  (tag : String) (t : Tape a) (id : Nat) : Result (Tensor a s) := by
  match t.getValue? id with
  | none =>
      exact .error (tagError tag s!"missing value for node id {id}")
  | some any =>
      if h : any.s = s then
        exact .ok (Tensor.castShape any.t h)
      else
        exact .error (tagError tag s!"value shape mismatch for node id {id}")

/--
Read a scalar forward value from a tape node id.

This is a common pattern in training scripts where the loss is a scalar node.
-/
def requireScalarValue {a : Type} [DecidableEq Shape]
  (tag : String) (t : Tape a) (id : Nat) : Result a := do
  let tScalar : Tensor a Shape.scalar ←
    requireValueTensor (tag := tag) (s := Shape.scalar) t id
  pure (Tensor.toScalar tScalar)

/-!
## SGD update helper

This is a small tensor-level update used by many tests.
-/
/--
Single-tensor SGD update rule.

Given a parameter tensor `param`, its gradient tensor `grad`, and a learning rate `lr`, compute:

`param - lr * grad`.

This is plain SGD (no momentum, weight decay, etc.); higher-level optimizers live in
`NN.Runtime.Autograd.Train.Optim`.
-/
def sgdUpdateTensor {a : Type} [Sub a] [Mul a] {s : Shape}
  (param grad : Tensor a s) (lr : a) : Tensor a s :=
  Tensor.subSpec param (Tensor.scaleSpec grad lr)

/--
The lightweight SGD helper is the same formula as the canonical pure optimizer.

We keep `sgdUpdateTensor` because it has a small algebraic signature (`Sub`/`Mul`) that is convenient
in tests, but this theorem pins it to the canonical optimizer equation used by the runtime
optimizer stack.
-/
theorem sgdUpdateTensor_eq_optimSGD {a : Type} [Context a]
    [DecidableRel ((· > ·) : a → a → Prop)] {s : Shape}
    (param grad : Tensor a s) (lr : a) :
    sgdUpdateTensor param grad lr =
      _root_.Optim.SGD.update (α := a) (s := s) { lr := lr } param grad := by
  rfl

end Train
end Autograd
end Runtime
