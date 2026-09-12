/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Representation.Basic.Reindex
public import NN.Tensor.Internal.Representation.Fiber.Basic
public import Mathlib.Data.Fintype.BigOperators
public import NN.Tensor.Internal.Representation.Fiber.Axis -- shake: keep
public import NN.Tensor.Internal.Representation.Reduction -- shake: keep
public import Mathlib.Algebra.BigOperators.Ring.Finset -- shake: keep

/-!
# Tensor Fiber Aggregation

The internal `Rep.push`, `Rep.reduce`, and `Rep.reduceNonempty` operations aggregate every
source coordinate mapping to one output coordinate. Their laws are stated for
arbitrary finite coordinate maps, not only maps parsed from einops patterns.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

open scoped BigOperators

universe u v
namespace Rep

/-- Transport a dependent aggregate across equality of its nonempty inputs. -/
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
Reduce every finite coordinate fiber with an order-independent multiset
function.

The input and output scalar types may differ. Total reducers specify a value
for the empty multiset; partial mathematical operations can instead return an
`Option` or use a separately proved nonempty-fiber precondition.
-/
def reduce {α : Type u} {β : Type v}
    [Storage α] [Storage β] {s t : Shape}
    (aggregate : Multiset α → β)
    (f : Coord s → Coord t) (input : Rep α s) : Rep β t :=
  Rep.ofFn fun outputCoordinate =>
    aggregate (Fiber.values f outputCoordinate input)

/--
Reduce fibers with an operation that is defined only on nonempty multisets.

The geometric premise is independent of tensor values. It is satisfied by a
checked einops reduction exactly when the product of its removed-axis lengths
is positive.
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
@[grind =] theorem reduce_reindex_input {α : Type u} {β : Type v}
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
Reindexing reduction outputs is equivalent to transporting the target of the
fiber map.
-/
@[grind =] theorem reindex_reduce {α : Type u} {β : Type v}
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
Nonempty reduction commutes with input reindexing. The two geometric
nonemptiness proofs may be constructed independently; proof irrelevance makes
their choice immaterial.
-/
@[grind =] theorem reduceNonempty_reindex_input {α : Type u} {β : Type v}
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
Output reindexing transports a nonempty reduction's target coordinates.
-/
@[grind =] theorem reindex_reduceNonempty {α : Type u} {β : Type v}
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

/-- Pushing a tensor sums exactly the entries in the selected coordinate fiber. -/
@[simp, grind =] theorem push_apply {α : Type u} [Storage α]
    [AddCommMonoid α] {s t : Shape}
    (f : Coord s → Coord t) (x : Rep α s) (j : Coord t) :
    push f x j = ∑ i : Fiber f j, x i.1 := by
  simp [push, Fiber.sum]

/-- Pushing along a coordinate equivalence is reindexing by its inverse. -/
@[grind =] theorem push_equiv {α : Type u} [Storage α]
    [AddCommMonoid α] {s t : Shape}
    (e : Coord s ≃ Coord t) (x : Rep α s) :
    push e x = reindex e.symm x := by
  ext j
  simp only [push_apply, reindex_apply]
  rw [Fintype.sum_unique]
  rfl

/-- Pushing along the identity coordinate map does nothing. -/
@[simp, grind =] theorem push_id {α : Type u} [Storage α]
    [AddCommMonoid α] {s : Shape}
    (x : Rep α s) : push id x = x := by
  simpa [reindex] using push_equiv (Equiv.refl (Coord s)) x

/--
Successive fiber aggregations compose to aggregation along the composite
coordinate map.

The proof identifies an element of a composite fiber with an intermediate
coordinate in the outer fiber together with an input coordinate in the
corresponding inner fiber. Commutativity is what makes the order of these two
finite sums immaterial.
-/
theorem push_comp {α : Type u} [Storage α] [AddCommMonoid α]
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
    { toFun := fun coordinate =>
        ⟨coordinate.2.1, by
          change g (f coordinate.2.1) = outputCoordinate
          rw [coordinate.2.2]
          exact coordinate.1.2⟩
      invFun := fun inputCoordinate =>
        ⟨⟨f inputCoordinate.1, inputCoordinate.2⟩,
          ⟨inputCoordinate.1, rfl⟩⟩
      left_inv := fun coordinate => by
        rcases coordinate with
          ⟨⟨middleCoordinate, middleProperty⟩,
            ⟨inputCoordinate, inputProperty⟩⟩
        change f inputCoordinate = middleCoordinate at inputProperty
        subst middleCoordinate
        rfl
      right_inv := fun inputCoordinate => by
        rcases inputCoordinate with ⟨inputCoordinate, inputProperty⟩
        rfl }
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
@[grind =] theorem sum_push {α : Type u} [Storage α]
    [AddCommMonoid α] {s t : Shape}
    (f : Coord s → Coord t) (x : Rep α s) :
    (∑ j, push f x j) = ∑ i, x i := by
  simpa only [push_apply] using Fintype.sum_fiberwise f x

/--
Reindexing by a coordinate equivalence preserves the total sum of tensor
entries. This is the aggregate form of the fact that an equivalence neither
drops nor duplicates coordinates.
-/
@[grind =] theorem sum_reindex {α : Type u} [Storage α]
    [AddCommMonoid α] {s t : Shape}
    (e : Coord t ≃ Coord s) (x : Rep α s) :
    (∑ outputCoordinate, reindex e x outputCoordinate) =
      ∑ inputCoordinate, x inputCoordinate := by
  simpa only [reindex_apply] using e.sum_comp x


end Rep

end TorchLean.Tensor.Internal
