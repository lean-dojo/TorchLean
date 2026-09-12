/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Representation.Fiber.Differential
public import NN.Tensor.Internal.Representation.Fiber.Axis -- shake: keep
public import NN.Tensor.Internal.Representation.Fiber.Basic -- shake: keep
public import NN.Tensor.Internal.Representation.Fiber.Aggregation -- shake: keep

/-!
# Product-reduction differentiation

Dual numbers derive the exact leave-one-out differential of product reduction.
The reverse rule is zero-aware and requires no division by primal entries.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

open scoped BigOperators

universe u v
namespace Rep

/--
The algebraic directional derivative of product reduction.

For each possible selected input coordinate, the Leibniz rule multiplies its
tangent by every other primal value in the same reduction fiber. This formula
is valid in a commutative semiring: it uses neither subtraction nor division.
-/
def productReduceDifferential {R : Type u} [Storage R]
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

The product explicitly omits the selected input coordinate. Unlike a formula
written as the forward product divided by that input, it remains correct when
one or more primal values are zero.
-/
def productReduceVjp {R : Type u} [Storage R]
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
theorem dot_productReduceDifferential_eq_dot_productReduceVjp
    {R : Type u} [Storage R] [CommSemiring R] {s t : Shape}
    (f : Coord s → Coord t)
    (inputTensor inputTangent : Rep R s)
    (outputCotangent : Rep R t) :
    dot
        (productReduceDifferential f inputTensor inputTangent)
        outputCotangent =
      dot inputTangent
        (productReduceVjp f inputTensor outputCotangent) := by
  classical
  unfold dot
  simp only [productReduceDifferential, productReduceVjp, get_ofFn,
    Finset.sum_mul, mul_assoc]
  rw [← Fintype.sum_fiberwise f]
  apply Finset.sum_congr rfl
  intro outputCoordinate _
  apply Finset.sum_congr rfl
  rintro ⟨inputCoordinate, rfl⟩ _
  rfl

end Rep

end TorchLean.Tensor.Internal
