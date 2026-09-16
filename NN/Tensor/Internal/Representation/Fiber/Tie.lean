/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Representation.Fiber.Differential

/-!
# Equal-share tie rules

For attained-value reducers such as minimum and maximum, this module specifies
and proves an explicit convention that distributes cotangents equally among
tied coordinates.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

open scoped BigOperators

universe u v
namespace Rep

/--
The equal-share weight assigned to one coordinate tied at a reducer's output.

All coordinates in the fiber whose primal value equals the reduced value
receive the reciprocal of their count; every other coordinate receives zero.
The reducer is kept as an ordinary function, so the same convention can be
used for minimum, maximum, or a custom attained-value reducer.
-/
def equalShareTieWeight {R : Type u} [Storage R]
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
The equal-sharing linearization of a nonempty reduction at a fixed primal
tensor.

When several coordinates attain the reduced value, their input tangents are
averaged. For minimum and maximum this is an explicit tie convention, not a
claim that the classical derivative is unique at a tie.
-/
def equalShareTieDifferential {R : Type u} [Storage R]
    [DivisionRing R] [DecidableEq R]
    {s t : Shape}
    (aggregate : (values : Multiset R) → values ≠ 0 → R)
    (f : Coord s → Coord t)
    (fiberNonempty :
      ∀ outputCoordinate, 0 < Fintype.card (Fiber f outputCoordinate))
    (inputTensor inputTangent : Rep R s) : Rep R t :=
  Rep.ofFn fun outputCoordinate =>
    ∑ inputCoordinate : Fiber f outputCoordinate,
      equalShareTieWeight aggregate f fiberNonempty inputTensor
          outputCoordinate inputCoordinate *
        inputTangent inputCoordinate.1

/--
The reverse map adjoint to `equalShareTieDifferential`.

Each selected input receives the same fraction of its fiber's output
cotangent; nonselected inputs receive zero.
-/
def equalShareTieVjp {R : Type u} [Storage R]
    [DivisionRing R] [DecidableEq R]
    {s t : Shape}
    (aggregate : (values : Multiset R) → values ≠ 0 → R)
    (f : Coord s → Coord t)
    (fiberNonempty :
      ∀ outputCoordinate, 0 < Fintype.card (Fiber f outputCoordinate))
    (inputTensor : Rep R s)
    (outputCotangent : Rep R t) : Rep R s :=
  Rep.ofFn fun inputCoordinate =>
    equalShareTieWeight aggregate f fiberNonempty inputTensor
        (f inputCoordinate) ⟨inputCoordinate, rfl⟩ *
      outputCotangent (f inputCoordinate)

/--
If a reducer returns one of its input values, the equal-share weights in
every nonempty fiber sum to one.

`Reduction.min_mem` and `Reduction.max_mem` provide the premise for minimum
and maximum. Thus a tie among `n` extrema contributes exactly `1 / n` at each
selected coordinate and contributes total weight one.
-/
theorem sum_equalShareTieWeight
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
        equalShareTieWeight aggregate f fiberNonempty inputTensor
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
The equal-sharing differential and VJP are adjoint under the finite tensor
pairing.

This is an algebraic identity for the named tie policy. At a minimum or
maximum tie it describes one symmetric generalized derivative; it does not
assert that the ordinary derivative exists there.
-/
theorem dot_equalShareTieDifferential_eq_dot_equalShareTieVjp
    {R : Type u} [Storage R] [Field R] [DecidableEq R] {s t : Shape}
    (aggregate : (values : Multiset R) → values ≠ 0 → R)
    (f : Coord s → Coord t)
    (fiberNonempty :
      ∀ outputCoordinate, 0 < Fintype.card (Fiber f outputCoordinate))
    (inputTensor inputTangent : Rep R s)
    (outputCotangent : Rep R t) :
    dot
        (equalShareTieDifferential aggregate f fiberNonempty inputTensor
          inputTangent)
        outputCotangent =
      dot inputTangent
        (equalShareTieVjp aggregate f fiberNonempty inputTensor
          outputCotangent) := by
  classical
  unfold dot
  simp only [equalShareTieDifferential, equalShareTieVjp, get_ofFn,
    Finset.sum_mul, mul_assoc]
  rw [← Fintype.sum_fiberwise f]
  apply Finset.sum_congr rfl
  intro outputCoordinate _
  apply Finset.sum_congr rfl
  rintro ⟨inputCoordinate, rfl⟩ _
  ac_rfl

end Rep

end TorchLean.Tensor.Internal
