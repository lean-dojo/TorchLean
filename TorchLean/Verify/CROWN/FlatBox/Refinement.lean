/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.FlatBox.Transfer
public import TorchLean.Tensor.Context.Real

/-!
# Coverage of input subdivision

`splitAt_covers` proves that subdividing a real box loses no point. The split sets one upper
endpoint and one lower endpoint to the same cut, and every point of the original box lies in at
least one child, points on the cut included. The cut is arbitrary, so the result holds whatever
midpoint arithmetic chose it and whatever layers are evaluated afterwards. It does not establish
soundness of an arbitrary enclosure procedure passed to `Refinement.bound`.
-/

public section

namespace TorchLean.CROWN.Graph.Refinement

/-- Subdividing at a shared coordinate cut never loses an original real input. -/
theorem splitAt_covers (box : FlatBox ℝ) (axis : Fin box.dim) (cut : ℝ)
    (x : Fin box.dim → ℝ)
    (hx : ∀ i, box.lo.getScalar i ≤ x i ∧ x i ≤ box.hi.getScalar i) :
    (∀ i, (splitAt box axis cut).1.lo.getScalar i ≤ x i ∧
      x i ≤ (splitAt box axis cut).1.hi.getScalar i) ∨
    (∀ i, (splitAt box axis cut).2.lo.getScalar i ≤ x i ∧
      x i ≤ (splitAt box axis cut).2.hi.getScalar i) := by
  dsimp only [splitAt]
  by_cases hcut : x axis ≤ cut
  · left
    intro i
    by_cases hi : i = axis
    · subst i
      simpa [splitAt] using And.intro (hx axis).1 hcut
    · simpa only [splitAt, Tensor.getScalar_ofFn, ite_eq_right hi] using hx i
  · right
    intro i
    by_cases hi : i = axis
    · subst i
      simpa [splitAt] using And.intro (le_of_not_ge hcut) (hx axis).2
    · simpa only [splitAt, Tensor.getScalar_ofFn, ite_eq_right hi] using hx i

end TorchLean.CROWN.Graph.Refinement
