/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Context
public import TorchLean.Tensor.Datatype

/-!
# Flattened interval bounds (`FlatBox`)

`FlatBox α` holds interval bounds on a flattened tensor value: `dim` is the number of scalar
entries, and `lo` and `hi` are vectors of that length. The graph-based CROWN development
(`CROWN.Graph`) uses it, and so do the operator-level transfer rules that work on
flattened vectors (slices, reductions). `FlatBox Float` is the output enclosure of the public
`Verification.Report`.

Besides the structure, the module holds the validity predicate `FlatBox.Valid` (`lo ≤ hi` at every
coordinate), the point box `FlatBox.ofTensor` of a vector and `FlatBox.maxOther?`, the competitor
bound of a top-label check. A point box is valid when `≤` is reflexive (`FlatBox.valid_ofTensor`).
-/

@[expose] public section


namespace TorchLean.CROWN

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

/--
Interval bounds on a flattened tensor with `dim` scalar entries.

The structure carries no `lo ≤ hi` invariant.
-/
structure FlatBox (α : Type) [Storage α] [Context α] where
  /-- Number of scalar entries in the flattened tensor. -/
  dim : Nat
  /-- Lower bound vector (shape `.dim dim .scalar`). -/
  lo  : Tensor α [dim]
  /-- Upper bound vector (shape `.dim dim .scalar`). -/
  hi  : Tensor α [dim]
  deriving Repr

namespace FlatBox

/-- Extensionality for the dependent record `FlatBox`: equal dimension and heterogeneously equal
endpoints. Stated by hand because the endpoint types depend on `dim`, so the `ext` tactic has no
automatically generated lemma to use. -/
@[ext] theorem ext {B1 B2 : FlatBox α} (hDim : B1.dim = B2.dim) (hLo : HEq B1.lo B2.lo)
    (hHi : HEq B1.hi B2.hi) : B1 = B2 := by
  cases B1
  cases B2
  cases hDim
  cases hLo
  cases hHi
  rfl

/-- Componentwise validity of a flat interval box: `lo ≤ hi` at every coordinate, in the order of
`Context α`. -/
def Valid (B : FlatBox α) : Prop :=
  ∀ i : Fin B.dim,
    Tensor.getScalar (α := α) B.lo i ≤ Tensor.getScalar (α := α) B.hi i

/--
Build a singleton `FlatBox` from an exact vector tensor `t` (set `lo = hi = t`).
-/
def ofTensor {n : Nat} (t : Tensor α [n]) : FlatBox α :=
  { dim := n, lo := t, hi := t }

/-- The singleton box `ofTensor t` is valid when `≤` on `α` is reflexive. -/
theorem valid_ofTensor (le_refl : ∀ a : α, a ≤ a) {n : Nat}
    (t : Tensor α [n]) :
    (ofTensor (α := α) t).Valid :=
  fun _ => le_refl _

/-- Largest upper endpoint of a host-Float box other than the one at `label`. -/
def maxOther? (box : FlatBox Float) (label : Nat) : Option Float :=
  (List.finRange box.dim).foldl
    (fun best index =>
      if index.val == label then best else
        match best with
        | none => some box.hi[index]
        | some current => some (max current box.hi[index]))
    none

end FlatBox

end TorchLean.CROWN
