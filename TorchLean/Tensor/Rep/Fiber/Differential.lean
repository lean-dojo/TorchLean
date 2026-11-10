/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Rep.Fiber.Aggregation

/-!
# Fiberwise differential identities

Under the finite tensor pairing `dot`, push is adjoint to pull. A fiberwise directional
derivative and a fiberwise reverse map are adjoint on whole tensors as soon as each fiber's pair
satisfies the scalar identity.
-/

@[expose] public section

namespace TorchLean.Tensor.Rep

open scoped BigOperators

universe u v
section

open Einops (Fiber)
open Shape (Coord)

/--
Fiber aggregation is adjoint to pullback for the standard finite dot
product.
-/
@[grind =, einops_simps] theorem dot_push_eq_dot_pull {R : Type u} [Storage R]
    [NonUnitalNonAssocSemiring R] {s t : Shape}
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
A differential/VJP adjunction on every reduction fiber lifts to the tensor-level adjunction.

Both operators are plain dependent functions. `fiberDifferential` receives the primal and tangent
values of one output fiber, and `fiberVjp` receives the same primal values with one output
cotangent and returns a cotangent for every input in the fiber. `fiber_adjoint` is the whole local
proof obligation. Both functions act on the finite fiber itself, so no permutation order is
chosen.

That `fiberDifferential` is the derivative of a particular aggregate is a separate fact. A custom
reducer proves it with the derivative notion suited to its scalar domain, then uses this theorem
to obtain the global VJP law.
-/
theorem dot_fiberwise_eq_dot_vjp
    {R : Type u} [Storage R] [AddCommMonoid R] [Mul R] {s t : Shape}
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

end

end TorchLean.Tensor.Rep
