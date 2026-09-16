/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import Mathlib.Algebra.CharZero.Defs
public import Mathlib.Algebra.Field.Defs
public import Mathlib.Data.Finset.Max
public import Mathlib.Algebra.BigOperators.Group.Multiset.Basic -- shake: keep

/-!
# Scalar aggregation for tensor reduction

This module defines the einops aggregates that do not already have canonical
mathlib names. `Multiset.sum` and `Multiset.prod` are used directly.

Boolean `any` and `all` are total and use the usual empty identities `false`
and `true`. Exact `mean`, `min`, and `max` take evidence that their input
multiset is nonempty. The lowering layer obtains that evidence from the
checked tensor shape, independently of tensor values.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal.Reduction

universe u

/-- Boolean disjunction over a multiset; the empty result is `false`. -/
def any (values : Multiset Bool) : Bool :=
  values.fold Bool.or false

/-- Boolean conjunction over a multiset; the empty result is `true`. -/
def all (values : Multiset Bool) : Bool :=
  values.fold Bool.and true

/--
The exact arithmetic mean of a nonempty multiset.

`DivisionRing` permits noncommutative multiplication because averaging uses
only addition and division by the natural-number cardinality. `CharZero`
ensures a positive cardinality remains nonzero in the scalar type. Neither the
nonemptiness evidence nor that instance is inspected by the formula; both are
retained so that exactness is part of the aggregate's public contract rather
than only a later theorem.
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

end TorchLean.Tensor.Internal.Reduction
