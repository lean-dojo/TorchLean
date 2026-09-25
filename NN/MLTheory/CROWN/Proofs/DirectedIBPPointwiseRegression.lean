/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import NN.MLTheory.CROWN.Extras.FP32
import NN.MLTheory.CROWN.Proofs.DirectedIBPPointwise

/-!
# Absolute-value endpoint regression

The FP32 proof carrier permits stored real values outside its rounding grid. This regression
checks that the directed abs transfer preserves the positive lower endpoint of a `1/3` singleton
and encloses its exact absolute value.
-/

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean TorchLean.Tensor

noncomputable section

private def thirdBox : FlatBox FP32 :=
  { dim := 1
    lo := Tensor.full [1] ⟨(1 : ℝ) / 3⟩
    hi := Tensor.full [1] ⟨(1 : ℝ) / 3⟩ }

/-- A positive FP32 comparison endpoint is preserved even when it is not a grid value. -/
theorem boxAbs_fp32_third_lower :
    LawfulBoundOps.toReal
      ((boxAbs thirdBox).lo.getScalar ⟨0, by simp [boxAbs, thirdBox]⟩) = (1 : ℝ) / 3 := by
  have hnonneg : ¬ (⟨(1 : ℝ) / 3⟩ : FP32) < 0 := by
    rw [LawfulBoundOps.lt_iff, (LawfulBoundOps.toReal_zero (α := FP32))]
    change ¬ (1 : ℝ) / 3 < 0
    norm_num
  simp only [thirdBox, boxAbs, getScalar_ofFn, getScalar_full, hnonneg, ↓reduceIte]
  rfl

/-- The actual FP32 abs box encloses the exact value of the singleton. -/
theorem boxAbs_fp32_third_encloses :
    RowEncloses (boxAbs thirdBox) 1 (fun _ => (1 : ℝ) / 3) := by
  have hinput : RowEncloses thirdBox 1 (fun _ => (1 : ℝ) / 3) := by
    rw [thirdBox, rowEncloses_iff]
    intro i
    simp only [getScalar_full]
    exact ⟨le_rfl, le_rfl⟩
  have h := boxAbs_encloses hinput
  simpa only [abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 1 / 3)] using h

end

end NN.MLTheory.CROWN.Graph.DirectedBackward
