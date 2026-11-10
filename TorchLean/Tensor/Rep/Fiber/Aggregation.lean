/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Rep.Reindex
public import TorchLean.Tensor.Rep.Fiber
public import TorchLean.Tensor.Einops.Reduction -- shake: keep
public import Mathlib.Algebra.BigOperators.Ring.Finset -- shake: keep

/-!
# Tensor Fiber Aggregation

`Rep.push`, `Rep.reduce` and `Rep.reduceNonempty` combine, for each output coordinate, every
source entry whose coordinate maps to it. `push` sums the fiber, and the other two hand the fiber's
multiset of values to a reducer. The laws hold for every coordinate map between shapes, whether or
not an einops pattern produced it.
-/

@[expose] public section

namespace TorchLean.Tensor.Rep

open scoped BigOperators

universe u v
section

open Shape (Coord)
open Einops (Fiber)

/-- A dependent aggregate agrees on equal nonempty multisets, whatever the nonemptiness proofs. -/
theorem nonemptyAggregate_congr {α : Type u} {β : Type v}
    (aggregate : (values : Multiset α) → values ≠ 0 → β)
    {leftValues rightValues : Multiset α}
    (hValues : leftValues = rightValues)
    (leftNonempty : leftValues ≠ 0) (rightNonempty : rightValues ≠ 0) :
    aggregate leftValues leftNonempty =
      aggregate rightValues rightNonempty := by
  subst rightValues
  rfl

/--
Aggregate a tensor along the fibers of a coordinate map.

The output at `j` is the sum of all input values whose coordinates map to
`j`. Empty fibers contribute the additive identity.
-/
def push {α : Type u} [Storage α] [AddCommMonoid α] {s t : Shape}
    (f : Coord s → Coord t) (x : Rep α s) : Rep α t :=
  Rep.ofFn fun j => Fiber.sum f j x

/--
Reduce every coordinate fiber with a function of the fiber's multiset of values, so the result
cannot depend on enumeration order.

The input and output scalar types may differ. A total reducer gives a value for the empty
multiset. A partial operation can return an `Option`, or go through `reduceNonempty` with a proof
that every fiber is nonempty.
-/
def reduce {α : Type u} {β : Type v}
    [Storage α] [Storage β] {s t : Shape}
    (aggregate : Multiset α → β)
    (f : Coord s → Coord t) (input : Rep α s) : Rep β t :=
  Rep.ofFn fun outputCoordinate =>
    aggregate (Fiber.values f outputCoordinate input)

/--
Reduce fibers with an operation that is defined only on nonempty multisets.

The premise `fiberNonempty` depends only on the coordinate map. A checked einops reduction
satisfies it when the product of its removed-axis lengths is positive, and it holds vacuously when
the output coordinate space is empty, even if that product is zero.
-/
def reduceNonempty {α : Type u} {β : Type v}
    [Storage α] [Storage β] {s t : Shape}
    (aggregate : (values : Multiset α) → values ≠ 0 → β)
    (f : Coord s → Coord t)
    (fiberNonempty :
      ∀ outputCoordinate, 0 < Fintype.card (Fiber f outputCoordinate))
    (input : Rep α s) : Rep β t :=
  Rep.ofFn fun outputCoordinate =>
    aggregate (Fiber.values f outputCoordinate input) <|
      Multiset.card_pos.mp <| by
        rw [Fiber.values_card]
        exact fiberNonempty outputCoordinate

/--
Reduction commutes with reindexing its input coordinate space.

The coordinate equivalence merely renames elements of each fiber, so the
multiset seen by the reducer is unchanged.
-/
@[grind =, einops =] theorem reduce_reindex_input {α : Type u} {β : Type v}
    [Storage α] [Storage β]
    {r s t : Shape} (aggregate : Multiset α → β)
    (e : Coord r ≃ Coord s) (f : Coord r → Coord t)
    (input : Rep α s) :
    reduce aggregate f (reindex e input) =
      reduce aggregate (f ∘ e.symm) input := by
  ext outputCoordinate
  simp only [reduce, get_ofFn]
  have hReindex : (reindex e input).get = input.get ∘ e := by
    funext coordinate
    exact reindex_apply e input coordinate
  rw [hReindex]
  exact congrArg aggregate
    (Fiber.values_comp_equiv e f outputCoordinate input)

/--
Reindexing the output of a reduction by `e` is the reduction along the fiber map `e.symm ∘ f`.
-/
@[grind =, einops =] theorem reindex_reduce {α : Type u} {β : Type v}
    [Storage α] [Storage β]
    {r s t : Shape} (aggregate : Multiset α → β)
    (e : Coord t ≃ Coord s) (f : Coord r → Coord s)
    (input : Rep α r) :
    reindex e (reduce aggregate f input) =
      reduce aggregate (e.symm ∘ f) input := by
  ext outputCoordinate
  simp only [reindex_apply, reduce, get_ofFn]
  exact congrArg aggregate
    (Fiber.values_target_equiv e.symm f outputCoordinate input)

/--
Nonempty reduction commutes with input reindexing. The two nonemptiness proofs may be built
independently, and proof irrelevance makes the choice immaterial.
-/
@[grind =, einops =] theorem reduceNonempty_reindex_input {α : Type u} {β : Type v}
    [Storage α] [Storage β]
    {r s t : Shape}
    (aggregate : (values : Multiset α) → values ≠ 0 → β)
    (e : Coord r ≃ Coord s) (f : Coord r → Coord t)
    (fiberNonempty :
      ∀ outputCoordinate, 0 < Fintype.card (Fiber f outputCoordinate))
    (reindexedFiberNonempty :
      ∀ outputCoordinate,
        0 < Fintype.card (Fiber (f ∘ e.symm) outputCoordinate))
    (input : Rep α s) :
    reduceNonempty aggregate f fiberNonempty (reindex e input) =
      reduceNonempty aggregate (f ∘ e.symm)
        reindexedFiberNonempty input := by
  ext outputCoordinate
  simp only [reduceNonempty, get_ofFn]
  have hReindex : (reindex e input).get = input.get ∘ e := by
    funext coordinate
    exact reindex_apply e input coordinate
  apply nonemptyAggregate_congr aggregate
  rw [hReindex]
  exact Fiber.values_comp_equiv e f outputCoordinate input

/--
Reindexing the output of a nonempty reduction by `e` is the nonempty reduction along
`e.symm ∘ f`.
-/
@[grind =, einops =] theorem reindex_reduceNonempty {α : Type u} {β : Type v}
    [Storage α] [Storage β]
    {r s t : Shape}
    (aggregate : (values : Multiset α) → values ≠ 0 → β)
    (e : Coord t ≃ Coord s) (f : Coord r → Coord s)
    (fiberNonempty :
      ∀ outputCoordinate, 0 < Fintype.card (Fiber f outputCoordinate))
    (reindexedFiberNonempty :
      ∀ outputCoordinate,
        0 < Fintype.card (Fiber (e.symm ∘ f) outputCoordinate))
    (input : Rep α r) :
    reindex e (reduceNonempty aggregate f fiberNonempty input) =
      reduceNonempty aggregate (e.symm ∘ f)
        reindexedFiberNonempty input := by
  ext outputCoordinate
  simp only [reindex_apply, reduceNonempty, get_ofFn]
  apply nonemptyAggregate_congr aggregate
  exact Fiber.values_target_equiv e.symm f outputCoordinate input

/-- A pushed tensor at `j` is the sum of the source entries in the fiber over `j`. -/
@[simp, grind =] theorem push_apply {α : Type u} [Storage α]
    [AddCommMonoid α] {s t : Shape}
    (f : Coord s → Coord t) (x : Rep α s) (j : Coord t) :
    push f x j = ∑ i : Fiber f j, x i.1 := by
  simp [push, Fiber.sum]

/-- Pushing along a coordinate equivalence is reindexing by its inverse. -/
@[grind =, einops_simps] theorem push_equiv {α : Type u} [Storage α]
    [AddCommMonoid α] {s t : Shape}
    (e : Coord s ≃ Coord t) (x : Rep α s) :
    push e x = reindex e.symm x := by
  ext j
  simp only [push_apply, reindex_apply]
  rw [Fintype.sum_unique]
  rfl

/-- Pushing along the identity coordinate map does nothing. -/
@[simp, grind =, einops_simps] theorem push_id {α : Type u} [Storage α]
    [AddCommMonoid α] {s : Shape}
    (x : Rep α s) : push id x = x := by
  simpa [reindex] using push_equiv (Equiv.refl (Coord s)) x

/--
Pushing along `f` and then along `g` is one push along `g ∘ f`.

The proof identifies an element of the composite fiber with a middle coordinate in the fiber of
`g` together with an input coordinate in the fiber of `f` over it. Commutativity of addition makes
the order of the two finite sums immaterial.
-/
@[einops_simps] theorem push_comp {α : Type u} [Storage α] [AddCommMonoid α]
    {r s t : Shape}
    (f : Coord r → Coord s) (g : Coord s → Coord t)
    (x : Rep α r) :
    push g (push f x) = push (g ∘ f) x := by
  classical
  ext outputCoordinate
  simp only [push_apply]
  let compositeFiberEquiv :
      (Σ middleCoordinate : Fiber g outputCoordinate,
        Fiber f middleCoordinate.1) ≃
        Fiber (g ∘ f) outputCoordinate :=
    Equiv.sigmaSubtypeFiberEquivSubtype f
      (p := fun inputCoordinate => g (f inputCoordinate) = outputCoordinate)
      (q := fun middleCoordinate => g middleCoordinate = outputCoordinate)
      (fun _ => Iff.rfl)
  change
    (∑ middleCoordinate : Fiber g outputCoordinate,
      ∑ inputCoordinate : Fiber f middleCoordinate.1,
        x inputCoordinate.1) =
      ∑ inputCoordinate : Fiber (g ∘ f) outputCoordinate,
        x inputCoordinate.1
  rw [← Fintype.sum_sigma']
  exact
    Fintype.sum_equiv compositeFiberEquiv _ _
      (fun coordinate => rfl)

/-- Fiber aggregation preserves the total sum over all coordinates. -/
@[grind =, einops_simps] theorem sum_push {α : Type u} [Storage α]
    [AddCommMonoid α] {s t : Shape}
    (f : Coord s → Coord t) (x : Rep α s) :
    (∑ j, push f x j) = ∑ i, x i := by
  simpa only [push_apply] using Fintype.sum_fiberwise f x

/--
Reindexing by a coordinate equivalence preserves the total sum of tensor entries, since an
equivalence neither drops nor duplicates coordinates.
-/
@[grind =, einops_simps] theorem sum_reindex {α : Type u} [Storage α]
    [AddCommMonoid α] {s t : Shape}
    (e : Coord t ≃ Coord s) (x : Rep α s) :
    (∑ outputCoordinate, reindex e x outputCoordinate) =
      ∑ inputCoordinate, x inputCoordinate := by
  simpa only [reindex_apply] using e.sum_comp x


end

end TorchLean.Tensor.Rep
