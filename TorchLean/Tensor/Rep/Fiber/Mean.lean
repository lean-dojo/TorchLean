/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Rep.Fiber.Differential

/-!
# Mean-reduction adjoint

Over a division ring of characteristic zero, mean reduction and its reverse map are adjoint for
every coordinate map whose fibers are nonempty.
-/

@[expose] public section

namespace TorchLean.Tensor.Rep

open scoped BigOperators

universe u v
section

open Einops
open Shape (Coord)

/--
The reverse map for mean reduction divides each output cotangent by the cardinality of the input
fiber to which it is broadcast.

The definition is geometric, depending on the coordinate map and never on the primal tensor.
Nonemptiness and characteristic zero matter for the forward mean and the adjoint theorem, and
evaluating this map needs neither.
-/
def Mean.vjp {R : Type u} [Storage R] [DivisionRing R]
    {s t : Shape}
    (f : Coord s → Coord t)
    (outputCotangent : Rep R t) : Rep R s :=
  Rep.ofFn fun inputCoordinate =>
    (Fintype.card (Fiber f (f inputCoordinate)) : R)⁻¹ *
      outputCotangent (f inputCoordinate)

/--
Mean reduction and `vjp` are adjoint under the finite tensor pairing.

The theorem covers every coordinate map, whether or not einops produced it, and the
positive-cardinality premise rules out empty means. The proof partitions the input pairing into
the fibers of the forward reduction and needs no commutativity of multiplication.
-/
theorem Mean.dot_reduceNonempty_eq_dot_vjp
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
      dot inputTangent (vjp f outputCotangent) := by
  classical
  unfold dot
  simp only [reduceNonempty, Reduction.mean, vjp, get_ofFn,
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

end

end TorchLean.Tensor.Rep
