/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Representation.Fiber.Aggregation

/-!
# Fiberwise differential identities

The finite tensor pairing makes push and pull adjoint. General fiberwise
directional derivatives and reverse maps are likewise adjoint when each local
fiber rule satisfies the corresponding scalar identity.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

open scoped BigOperators

universe u v
namespace Rep

/-- The standard bilinear pairing of two finite coordinate tensors. -/
def dot {R : Type u} [Storage R] [Semiring R] {s : Shape}
    (x y : Rep R s) : R :=
  ∑ i, x i * y i

/--
Stacking and leading-axis slicing are adjoint for the finite tensor pairing.

The pairing of a stacked family with a tensor is the sum of the pairings with
each corresponding leading-axis slice.
-/
@[grind =] theorem dot_stack {R : Type u} [Storage R] [Semiring R]
    {n : Nat} {s : Shape}
    (components : Fin n → Rep R s) (tensor : Rep R (n :: s)) :
    dot (stack components) tensor =
      ∑ component, dot (components component) (unstack tensor component) := by
  classical
  unfold dot
  simp only [stack_apply, unstack_apply]
  rw [show
    (Finset.univ : Finset (Coord (n :: s))) =
      (Finset.univ : Finset (Fin n)) ×ˢ
        (Finset.univ : Finset (Coord s)) by
    ext coordinate
    constructor
    · intro _
      exact Finset.mem_product.mpr
        ⟨Finset.mem_univ coordinate.1, Finset.mem_univ coordinate.2⟩
    · intro _
      exact Finset.mem_univ coordinate]
  exact Finset.sum_product _ _ _

/--
Reindexing and inverse reindexing are adjoint for the finite tensor pairing.

The proof changes the finite summation index along the coordinate
equivalence; it does not require an ordering or positivity assumption on
shape dimensions.
-/
@[grind =] theorem dot_reindex_eq_dot_reindex_symm {R : Type u}
    [Storage R] [Semiring R]
    {s t : Shape} (e : Coord t ≃ Coord s)
    (x : Rep R s) (y : Rep R t) :
    dot (reindex e x) y = dot x (reindex e.symm y) := by
  simpa only [dot, reindex_apply, Equiv.symm_apply_apply] using
    e.sum_comp (fun inputCoordinate => x inputCoordinate * y (e.symm inputCoordinate))

/-- Applying the same coordinate reindexing to both tensors preserves their pairing. -/
@[grind =] theorem dot_reindex_reindex {R : Type u} [Storage R]
    [Semiring R] {s t : Shape}
    (e : Coord t ≃ Coord s) (x y : Rep R s) :
    dot (reindex e x) (reindex e y) = dot x y := by
  rw [dot_reindex_eq_dot_reindex_symm, reindex_symm_reindex]

/--
Fiber aggregation is adjoint to pullback for the standard finite dot
product.
-/
@[grind =] theorem dot_push_eq_dot_pull {R : Type u} [Storage R]
    [Semiring R] {s t : Shape}
    (f : Coord s → Coord t) (x : Rep R s) (y : Rep R t) :
    dot (push f x) y = dot x (pull f y) := by
  classical
  simp only [dot, push_apply, pull_apply, Finset.sum_mul]
  calc
    (∑ j, ∑ i : Fiber f j, x i.1 * y j) =
        ∑ j, ∑ i : Fiber f j, x i.1 * y (f i.1) := by
      apply Finset.sum_congr rfl
      intro j _
      apply Finset.sum_congr rfl
      intro i _
      rw [i.property]
    _ = ∑ i, x i * y (f i) :=
      Fintype.sum_fiberwise f (fun i : Coord s => x i * y (f i))

/--
Lift a local differential/VJP adjunction on each reduction fiber to the
corresponding tensor-level adjunction.

The two operators are ordinary dependent functions rather than a certificate
structure. `fiberDifferential` receives the primal values and tangent values
in one output fiber. `fiberVjp` receives the same primal values and one output
cotangent, and returns one cotangent for every input in that fiber. The
`fiber_adjoint` hypothesis is the complete local proof obligation.

This theorem deliberately does not claim that `fiberDifferential` is the
derivative of a particular aggregate. A custom reducer establishes that fact
separately using the derivative notion appropriate to its scalar domain, then
uses this theorem to obtain the global VJP law. No permutation order is chosen:
both local functions act directly on the finite fiber.
-/
theorem dot_fiberwiseDifferential_eq_dot_fiberwiseVjp
    {R : Type u} [Storage R] [Semiring R] {s t : Shape}
    (f : Coord s → Coord t)
    (fiberDifferential :
      ∀ outputCoordinate,
        (Fiber f outputCoordinate → R) →
          (Fiber f outputCoordinate → R) → R)
    (fiberVjp :
      ∀ outputCoordinate,
        (Fiber f outputCoordinate → R) →
          R → Fiber f outputCoordinate → R)
    (fiber_adjoint :
      ∀ (outputCoordinate)
        (fiberInput fiberTangent : Fiber f outputCoordinate → R)
        (outputCotangent : R),
        fiberDifferential outputCoordinate fiberInput fiberTangent *
            outputCotangent =
          ∑ inputCoordinate,
            fiberTangent inputCoordinate *
              fiberVjp outputCoordinate fiberInput outputCotangent
                inputCoordinate)
    (inputTensor inputTangent : Rep R s)
    (outputCotangent : Rep R t) :
    dot
        (Rep.ofFn fun outputCoordinate =>
          fiberDifferential outputCoordinate
            (fun inputCoordinate => inputTensor inputCoordinate.1)
            (fun inputCoordinate => inputTangent inputCoordinate.1))
        outputCotangent =
      dot inputTangent
        (Rep.ofFn fun inputCoordinate =>
          fiberVjp (f inputCoordinate)
            (fun fiberCoordinate : Fiber f (f inputCoordinate) =>
              inputTensor fiberCoordinate.1)
            (outputCotangent (f inputCoordinate))
            ⟨inputCoordinate, rfl⟩) := by
  classical
  unfold dot
  simp only [get_ofFn]
  calc
    (∑ outputCoordinate,
        fiberDifferential outputCoordinate
            (fun inputCoordinate => inputTensor inputCoordinate.1)
            (fun inputCoordinate => inputTangent inputCoordinate.1) *
          outputCotangent outputCoordinate) =
        ∑ outputCoordinate,
          ∑ inputCoordinate : Fiber f outputCoordinate,
            inputTangent inputCoordinate.1 *
              fiberVjp outputCoordinate
                (fun fiberCoordinate => inputTensor fiberCoordinate.1)
                (outputCotangent outputCoordinate) inputCoordinate := by
      apply Finset.sum_congr rfl
      intro outputCoordinate _
      exact fiber_adjoint outputCoordinate _ _ _
    _ =
        ∑ inputCoordinate,
          inputTangent inputCoordinate *
            fiberVjp (f inputCoordinate)
              (fun fiberCoordinate : Fiber f (f inputCoordinate) =>
                inputTensor fiberCoordinate.1)
              (outputCotangent (f inputCoordinate))
              ⟨inputCoordinate, rfl⟩ := by
      rw [← Fintype.sum_fiberwise f]
      apply Finset.sum_congr rfl
      intro outputCoordinate _
      apply Finset.sum_congr rfl
      rintro ⟨inputCoordinate, rfl⟩ _
      rfl

end Rep

end TorchLean.Tensor.Internal
