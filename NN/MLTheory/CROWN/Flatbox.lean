/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context
public import NN.Spec.Core.Tensor.Core
public import NN.Spec.Core.Tensor -- shake: keep

/-!
# Flattened interval bounds (`FlatBox`)

`FlatBox α` is a small container for interval bounds on a *flattened* tensor value.

It is used by the graph-based LiRPA/CROWN development (`NN.MLTheory.CROWN.Graph`) and by some
operator-level transfer rules that operate on flattened vectors (e.g. slice/reduce rules).

`dim` is the flattened size, and `lo`/`hi` are tensors of shape `.dim dim .scalar`.
-/

@[expose] public section


namespace NN.MLTheory.CROWN

open Spec TorchLean
open TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/--
Flattened interval bounds.

`dim` is the flattened size (number of scalar components).
-/
structure FlatBox (α : Type) [TorchLean.Storage α] [Context α] where
  /-- Flattened output dimension. -/
  dim : Nat
  /-- Lower bound vector (shape `.dim dim .scalar`). -/
  lo  : Tensor α [dim]
  /-- Upper bound vector (shape `.dim dim .scalar`). -/
  hi  : Tensor α [dim]

namespace FlatBox

/--
Componentwise validity of a flat interval box: `lo ≤ hi` for every coordinate.

This is a proof layer predicate; it uses the order carried by `Context α`, which keeps all CROWN
box predicates in the same scalar universe as the executable operators.
-/
def Valid (B : FlatBox α) : Prop :=
  ∀ i : Fin B.dim,
    TorchLean.Tensor.getScalar (α := α) B.lo i ≤ TorchLean.Tensor.getScalar (α := α) B.hi i

/--
Build a singleton `FlatBox` from an exact vector tensor `t` (set `lo = hi = t`).
-/
def ofTensor {n : Nat} (t : Tensor α [n]) : FlatBox α :=
  { dim := n, lo := t, hi := t }

/-- A singleton flat box is always valid (over any preorder). -/
theorem valid_ofTensor (le_refl : ∀ a : α, a ≤ a) {n : Nat}
    (t : Tensor α [n]) :
    (ofTensor (α := α) t).Valid := by
  intro i
  -- After unfolding `Valid`/`ofTensor`, both endpoints are the same scalar entry.
  dsimp [Valid, ofTensor]
  exact le_refl _

end FlatBox

end NN.MLTheory.CROWN
