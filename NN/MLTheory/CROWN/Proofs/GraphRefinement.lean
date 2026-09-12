/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.Refinement
public import NN.MLTheory.CROWN.BoundOps.Lawful

/-!
# Coverage of input subdivision

The split changes one upper endpoint and one lower endpoint to the same cut. Every original point
belongs to at least one child, including points on the boundary. This is independent of midpoint
arithmetic and of the layers subsequently evaluated. The result does not establish soundness of
an arbitrary enclosure procedure passed to `Refinement.bound`.
-/

public section

namespace NN.MLTheory.CROWN.Graph.Refinement

open Spec TorchLean

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
    · simpa only [splitAt, Tensor.getScalar_ofFn, if_neg hi] using hx i
  · right
    intro i
    by_cases hi : i = axis
    · subst i
      simpa [splitAt] using And.intro (le_of_lt (lt_of_not_ge hcut)) (hx axis).2
    · simpa only [splitAt, Tensor.getScalar_ofFn, if_neg hi] using hx i

end NN.MLTheory.CROWN.Graph.Refinement
