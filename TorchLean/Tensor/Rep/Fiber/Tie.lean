/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Rep.Fiber.Differential

/-!
# Equal-share tie rules

For attained-value reducers such as minimum and maximum, the equal-share convention splits each
output cotangent evenly among the tied coordinates of its fiber, the ones whose value equals the
reduced value. Over a division ring of characteristic zero the weights in each fiber sum to one,
and for any reducer the resulting differential and VJP are adjoint.
-/

@[expose] public section

namespace TorchLean.Tensor.Rep

open scoped BigOperators

universe u v
section

open Einops (Fiber)
open Shape (Coord)

/--
The equal-share weight of one input coordinate in a reducer's fiber.

Each coordinate whose primal value equals the reduced value gets the reciprocal of the number of
such coordinates, and every other coordinate gets zero. The reducer is an ordinary function
argument, so the same convention serves minimum, maximum or a custom attained-value reducer.
-/
def Tie.weight {R : Type u} [Storage R]
    [DivisionRing R] [DecidableEq R]
    {s t : Shape}
    (aggregate : (values : Multiset R) → values ≠ 0 → R)
    (f : Coord s → Coord t)
    (fiberNonempty :
      ∀ outputCoordinate, 0 < Fintype.card (Fiber f outputCoordinate))
    (inputTensor : Rep R s)
    (outputCoordinate : Coord t)
    (inputCoordinate : Fiber f outputCoordinate) : R :=
  let selectedCoordinates :=
    (Finset.univ : Finset (Fiber f outputCoordinate)).filter fun coordinate =>
      inputTensor coordinate.1 =
        reduceNonempty aggregate f fiberNonempty inputTensor outputCoordinate
  if inputCoordinate ∈ selectedCoordinates then
    (selectedCoordinates.card : R)⁻¹
  else
    0

/--
The equal-sharing linearization of a nonempty reduction at a fixed primal tensor.

When several coordinates attain the reduced value, their input tangents are averaged. For minimum
and maximum, averaging is the convention we adopt at a tie.
-/
def Tie.differential {R : Type u} [Storage R]
    [DivisionRing R] [DecidableEq R]
    {s t : Shape}
    (aggregate : (values : Multiset R) → values ≠ 0 → R)
    (f : Coord s → Coord t)
    (fiberNonempty :
      ∀ outputCoordinate, 0 < Fintype.card (Fiber f outputCoordinate))
    (inputTensor inputTangent : Rep R s) : Rep R t :=
  Rep.ofFn fun outputCoordinate =>
    ∑ inputCoordinate : Fiber f outputCoordinate,
      weight aggregate f fiberNonempty inputTensor
          outputCoordinate inputCoordinate *
        inputTangent inputCoordinate.1

/--
The reverse map adjoint to `differential`.

Each selected input, one whose value attains the reduced value, receives the same fraction of its
fiber's output cotangent, and every other input receives zero.
-/
def Tie.vjp {R : Type u} [Storage R]
    [DivisionRing R] [DecidableEq R]
    {s t : Shape}
    (aggregate : (values : Multiset R) → values ≠ 0 → R)
    (f : Coord s → Coord t)
    (fiberNonempty :
      ∀ outputCoordinate, 0 < Fintype.card (Fiber f outputCoordinate))
    (inputTensor : Rep R s)
    (outputCotangent : Rep R t) : Rep R s :=
  Rep.ofFn fun inputCoordinate =>
    weight aggregate f fiberNonempty inputTensor
        (f inputCoordinate) ⟨inputCoordinate, rfl⟩ *
      outputCotangent (f inputCoordinate)

/--
Over a division ring of characteristic zero, if a reducer returns one of its input values, the
equal-share weights in every nonempty fiber sum to one.

`Einops.Reduction.min_mem` and `Einops.Reduction.max_mem` provide the premise for minimum and
maximum, so a tie among `n` extrema puts weight `1 / n` on each selected coordinate, for a total of
one.
-/
theorem Tie.sum_weight
    {R : Type u} [Storage R]
    [DivisionRing R] [CharZero R] [DecidableEq R]
    {s t : Shape}
    (aggregate : (values : Multiset R) → values ≠ 0 → R)
    (aggregate_mem :
      ∀ (values : Multiset R) (hValues : values ≠ 0),
        aggregate values hValues ∈ values)
    (f : Coord s → Coord t)
    (fiberNonempty :
      ∀ outputCoordinate, 0 < Fintype.card (Fiber f outputCoordinate))
    (inputTensor : Rep R s)
    (outputCoordinate : Coord t) :
    (∑ inputCoordinate : Fiber f outputCoordinate,
        weight aggregate f fiberNonempty inputTensor
          outputCoordinate inputCoordinate) =
      1 := by
  classical
  let selectedCoordinates :=
    (Finset.univ : Finset (Fiber f outputCoordinate)).filter fun coordinate =>
      inputTensor coordinate.1 =
        reduceNonempty aggregate f fiberNonempty inputTensor outputCoordinate
  have hReducedValue :
      reduceNonempty aggregate f fiberNonempty inputTensor outputCoordinate ∈
        Fiber.values f outputCoordinate inputTensor := by
    simp only [reduceNonempty, get_ofFn]
    exact aggregate_mem _ _
  obtain ⟨selectedCoordinate, hSelectedValue⟩ :
      ∃ coordinate : Fiber f outputCoordinate,
        inputTensor coordinate.1 =
          reduceNonempty aggregate f fiberNonempty inputTensor
            outputCoordinate := by
    simpa [Fiber.values] using hReducedValue
  have hSelectedCoordinates : selectedCoordinates.Nonempty := by
    exact ⟨selectedCoordinate,
      Finset.mem_filter.mpr ⟨Finset.mem_univ _, hSelectedValue⟩⟩
  change
    (∑ inputCoordinate : Fiber f outputCoordinate,
      if inputCoordinate ∈ selectedCoordinates then
        (selectedCoordinates.card : R)⁻¹
      else
        0) =
      1
  rw [← Finset.sum_filter]
  simp only [Finset.filter_mem_eq_inter, Finset.univ_inter,
    Finset.sum_const, nsmul_eq_mul]
  have hCard : (selectedCoordinates.card : R) ≠ 0 :=
    Nat.cast_ne_zero.mpr <| Finset.card_ne_zero.mpr hSelectedCoordinates
  exact mul_inv_cancel₀ hCard

/--
The equal-sharing differential and VJP are adjoint under the finite tensor pairing.

The identity is algebraic and holds for every reducer. At a minimum or maximum tie the pair
describes one symmetric generalized derivative, and the ordinary derivative need not exist there.
-/
@[einops_simps] theorem Tie.dot_differential_eq_dot_vjp
    {R : Type u} [Storage R] [DivisionRing R] [DecidableEq R] {s t : Shape}
    (aggregate : (values : Multiset R) → values ≠ 0 → R)
    (f : Coord s → Coord t)
    (fiberNonempty :
      ∀ outputCoordinate, 0 < Fintype.card (Fiber f outputCoordinate))
    (inputTensor inputTangent : Rep R s)
    (outputCotangent : Rep R t) :
    dot
        (differential aggregate f fiberNonempty inputTensor
          inputTangent)
        outputCotangent =
      dot inputTangent
        (vjp aggregate f fiberNonempty inputTensor
          outputCotangent) := by
  classical
  unfold dot
  simp only [differential, vjp, get_ofFn,
    Finset.sum_mul, mul_assoc]
  rw [← Fintype.sum_fiberwise f]
  apply Finset.sum_congr rfl
  intro outputCoordinate _
  apply Finset.sum_congr rfl
  rintro ⟨inputCoordinate, rfl⟩ _
  dsimp only [weight]
  split
  · rw [← mul_assoc, (Nat.cast_commute _ (inputTangent inputCoordinate)).inv_left₀.eq,
      mul_assoc]
  · simp

end

end TorchLean.Tensor.Rep
