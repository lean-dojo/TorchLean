/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Rep.Fiber.Differential

/-!
# Product-reduction differentiation

The leave-one-out product formula defines an algebraic differential for product reduction and its
adjoint reverse map. The reverse rule never divides by a primal entry, so it is correct at zeros.
The dual-number derivation is in `TorchLean.Tensor.Rep.Fiber.Product.DualNumber`.
-/

@[expose] public section

namespace TorchLean.Tensor.Rep

open scoped BigOperators

universe u v
section

open Shape (Coord)
open Einops (Fiber)

/--
The algebraic directional derivative of product reduction.

By the Leibniz rule, each input coordinate in a fiber contributes its tangent times every other
primal value in that fiber, and the output sums these terms. The formula uses neither subtraction
nor division, so it is valid in any commutative semiring.
-/
def Product.differential {R : Type u} [Storage R]
    [CommSemiring R] {s t : Shape}
    (f : Coord s → Coord t)
    (inputTensor inputTangent : Rep R s) : Rep R t :=
  Rep.ofFn fun outputCoordinate =>
    ∑ selectedCoordinate : Fiber f outputCoordinate,
      inputTangent selectedCoordinate.1 *
        (Finset.univ.erase selectedCoordinate).prod
          (fun otherCoordinate => inputTensor otherCoordinate.1)

/--
The zero-aware reverse map for product reduction.

Each input coordinate receives its fiber's output cotangent times the product of the other primal
values in the fiber. The product leaves that coordinate out and never divides the forward product
by it, so the rule is correct when one or more primal values are zero.
-/
def Product.vjp {R : Type u} [Storage R]
    [CommSemiring R] {s t : Shape}
    (f : Coord s → Coord t)
    (inputTensor : Rep R s)
    (outputCotangent : Rep R t) : Rep R s :=
  Rep.ofFn fun inputCoordinate =>
    (Finset.univ.erase
        (⟨inputCoordinate, rfl⟩ :
          Fiber f (f inputCoordinate))).prod
      (fun otherCoordinate => inputTensor otherCoordinate.1) *
      outputCotangent (f inputCoordinate)

/--
The product-reduction differential and its zero-aware reverse map are
adjoint under the finite tensor pairing.

The proof partitions the input pairing by reduction fibers, then chooses the
same omitted-coordinate product on both sides. It therefore covers empty
fibers and every pattern of zero primal values without side conditions.
-/
@[einops_simps] theorem Product.dot_differential_eq_dot_vjp
    {R : Type u} [Storage R] [CommSemiring R] {s t : Shape}
    (f : Coord s → Coord t)
    (inputTensor inputTangent : Rep R s)
    (outputCotangent : Rep R t) :
    dot
        (differential f inputTensor inputTangent)
        outputCotangent =
      dot inputTangent
        (vjp f inputTensor outputCotangent) := by
  classical
  unfold dot
  simp only [differential, vjp, get_ofFn,
    Finset.sum_mul, mul_assoc]
  rw [← Fintype.sum_fiberwise f]
  apply Finset.sum_congr rfl
  intro outputCoordinate _
  apply Finset.sum_congr rfl
  rintro ⟨inputCoordinate, rfl⟩ _
  rfl

end

end TorchLean.Tensor.Rep
