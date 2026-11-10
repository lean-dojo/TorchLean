/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import Mathlib.Algebra.CharZero.Defs
public import Mathlib.Algebra.Field.Defs
public import Mathlib.Data.Finset.Max
public import Mathlib.Algebra.BigOperators.Group.Multiset.Basic -- shake: keep

/-!
# Scalar aggregation for tensor reduction

The einops aggregates without a canonical Mathlib name are defined here. Sum
and product reductions use `Multiset.sum` and `Multiset.prod`.

Boolean `any` and `all` are total, with the empty identities `false` and
`true`. Exact `mean`, `min` and `max` take evidence that their input multiset
is nonempty, and the lowering layer obtains that evidence from the checked
tensor shape without looking at tensor values.
-/

@[expose] public section

namespace TorchLean.Einops.Reduction

universe u

/-- Boolean disjunction over a multiset, with `false` for the empty multiset. -/
def any (values : Multiset Bool) : Bool :=
  values.fold Bool.or false

/-- Boolean conjunction over a multiset, with `true` for the empty multiset. -/
def all (values : Multiset Bool) : Bool :=
  values.fold Bool.and true

/--
The exact arithmetic mean of a nonempty multiset.

A `DivisionRing` suffices, commutative or not, because averaging only adds and
divides by the cardinality. `CharZero` keeps a positive cardinality nonzero in
the scalar type. The formula inspects neither the nonemptiness evidence nor
that instance, and both stay in the signature so that exactness is part of the
aggregate's public contract.
-/
def mean {α : Type u} [DivisionRing α] [CharZero α]
    (values : Multiset α) (_hValues : values ≠ 0) : α :=
  values.sum / (values.card : α)

/-- The least value in a nonempty multiset. -/
def min {α : Type u} [LinearOrder α]
    (values : Multiset α) (hValues : values ≠ 0) : α :=
  values.toFinset.min' (Multiset.toFinset_nonempty.mpr hValues)

/-- A minimum of a nonempty multiset is one of its values. -/
theorem min_mem {α : Type u} [LinearOrder α]
    (values : Multiset α) (hValues : values ≠ 0) :
    min values hValues ∈ values := by
  unfold min
  exact Multiset.mem_toFinset.mp (Finset.min'_mem _ _)

/-- The greatest value in a nonempty multiset. -/
def max {α : Type u} [LinearOrder α]
    (values : Multiset α) (hValues : values ≠ 0) : α :=
  values.toFinset.max' (Multiset.toFinset_nonempty.mpr hValues)

/-- A maximum of a nonempty multiset is one of its values. -/
theorem max_mem {α : Type u} [LinearOrder α]
    (values : Multiset α) (hValues : values ≠ 0) :
    max values hValues ∈ values := by
  unfold max
  exact Multiset.mem_toFinset.mp (Finset.max'_mem _ _)

end TorchLean.Einops.Reduction
