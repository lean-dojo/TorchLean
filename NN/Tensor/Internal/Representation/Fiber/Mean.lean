/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Representation.Fiber.Differential
public import NN.Tensor.Internal.Representation.Reduction
public import NN.Tensor.Internal.Representation.Fiber.Axis -- shake: keep
public import Mathlib.Algebra.Field.Defs -- shake: keep
public import Mathlib.Algebra.CharZero.Defs -- shake: keep

/-!
# Mean-reduction adjoint

Mean reduction and its reverse map are adjoint for every finite coordinate
map whose fibers are nonempty.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

open scoped BigOperators

universe u v
namespace Rep

/--
The reverse map for mean reduction divides each output cotangent by the
cardinality of the input fiber to which it is broadcast.

The definition is geometric: it depends only on the coordinate map and not
on the primal tensor. Nonemptiness and characteristic zero are needed by the
forward mean and its adjoint theorem, not to evaluate this map.
-/
def meanReduceVjp {R : Type u} [Storage R] [DivisionRing R]
    {s t : Shape}
    (f : Coord s → Coord t)
    (outputCotangent : Rep R t) : Rep R s :=
  Rep.ofFn fun inputCoordinate =>
    (Fintype.card (Fiber f (f inputCoordinate)) : R)⁻¹ *
      outputCotangent (f inputCoordinate)

/--
Mean reduction and `meanReduceVjp` are adjoint under the finite tensor
pairing.

This is a theorem for every finite coordinate map, not only maps produced by
einops. The positive-cardinality premise rules out empty means. The proof
partitions the input pairing into the same fibers used by the forward
reduction and does not assume commutative multiplication.
-/
theorem dot_reduceNonempty_mean_eq_dot_meanReduceVjp
    {R : Type u} [Storage R] [DivisionRing R] [CharZero R]
    {s t : Shape}
    (f : Coord s → Coord t)
    (fiberNonempty :
      ∀ outputCoordinate, 0 < Fintype.card (Fiber f outputCoordinate))
    (inputTangent : Rep R s)
    (outputCotangent : Rep R t) :
    dot
        (reduceNonempty Reduction.mean f fiberNonempty inputTangent)
        outputCotangent =
      dot inputTangent (meanReduceVjp f outputCotangent) := by
  classical
  unfold dot
  simp only [reduceNonempty, Reduction.mean, meanReduceVjp, get_ofFn,
    Fiber.values_card, Fiber.values_sum]
  change
    (∑ outputCoordinate,
      ((∑ inputCoordinate : Fiber f outputCoordinate,
          inputTangent inputCoordinate.1) /
        (Fintype.card (Fiber f outputCoordinate) : R)) *
        outputCotangent outputCoordinate) =
      ∑ inputCoordinate,
        inputTangent inputCoordinate *
          ((Fintype.card (Fiber f (f inputCoordinate)) : R)⁻¹ *
            outputCotangent (f inputCoordinate))
  calc
    _ =
        ∑ outputCoordinate,
          ∑ inputCoordinate : Fiber f outputCoordinate,
            inputTangent inputCoordinate.1 *
              ((Fintype.card (Fiber f outputCoordinate) : R)⁻¹ *
                outputCotangent outputCoordinate) := by
      apply Finset.sum_congr rfl
      intro outputCoordinate _
      simp only [div_eq_mul_inv, Finset.sum_mul, mul_assoc]
    _ =
        ∑ inputCoordinate,
          inputTangent inputCoordinate *
            ((Fintype.card (Fiber f (f inputCoordinate)) : R)⁻¹ *
              outputCotangent (f inputCoordinate)) := by
      rw [← Fintype.sum_fiberwise f]
      apply Finset.sum_congr rfl
      intro outputCoordinate _
      apply Finset.sum_congr rfl
      intro inputCoordinate _
      rw [inputCoordinate.property]

end Rep

end TorchLean.Tensor.Internal
