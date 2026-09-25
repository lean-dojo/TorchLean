/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedIBPNormalization
public import NN.MLTheory.CROWN.Extras.FP32

/-!
# Normalization scalar facts and domain regression

The default LayerNorm shortcut needs a nonnegative stored epsilon. For the real row `[-1, 1]`,
epsilon `-3/4` produces `[-2, 2]`, which lies outside the width-two uniform bound.
The actual Real and rounded-real FP32 constants satisfy the required nonnegativity.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean TorchLean.Tensor
open _root_.Proofs.Autograd.Norm

noncomputable section

/-- The exact-real default normalization stabilizer is nonnegative. -/
theorem normalizationEpsilon_nonneg_real :
    0 ≤ LawfulBoundOps.toReal (TorchLean.normalizationEpsilon : ℝ) := by
  change (0 : ℝ) ≤ ((1 / 100000 : ℚ) : ℝ)
  norm_num

/-- FP32 rounds the positive rational stabilizer once; valid rounding preserves nonnegativity. -/
theorem normalizationEpsilon_nonneg_fp32 :
    0 ≤ LawfulBoundOps.toReal
      (TorchLean.normalizationEpsilon : NN.MLTheory.CROWN.FP32) := by
  change (0 : ℝ) ≤ FloatLib.Floats.Formats.Flocq.round
    (β := FloatLib.Numerics.binaryRadix) (fexp := TorchLean.Floats.fexp32)
    TorchLean.Floats.rnd32 ((1 / 100000 : ℚ) : ℝ)
  apply FloatLib.Floats.Formats.Flocq.round_nonneg
  norm_num

/-- A negative stabilizer invalidates the square-root-width bound for the actual real Spec. -/
theorem negative_epsilon_exceeds_uniform_layerNorm_bound :
    let x : Tensor ℝ [1, 2] :=
      Tensor.dim fun _ : Fin 1 => Tensor.ofFn fun j : Fin 2 => if j = 0 then -1 else 1
    Real.sqrt 2 <
      |Spec.get2 (Spec.layerNorm x (Tensor.full [2] 1) (Tensor.full [2] 0)
        (by decide) (by decide) (-3 / 4)) 0 1| := by
  intro x
  have hx (j : Fin 2) : Spec.get2 x 0 j = if j = 0 then -1 else 1 := by
    simp only [x, Spec.get2_eq_apply, Tensor.dim, TorchLean.Tensor.Internal.Rep.stack_apply,
      Tensor.ofFn_apply]
  have hm : rowMeanE x 0 = 0 := by
    norm_num [rowMeanE, Fin.sum_univ_two, hx]
  have hv : rowVarE x 0 = 1 := by
    norm_num [rowVarE, Fin.sum_univ_two, hm, hx]
  rw [get2_layerNorm]
  simp only [hx, hm, hv, Tensor.getScalar_full]
  have hs : Real.sqrt ((1 : ℝ) / 4) = 1 / 2 := by
    rw [show (1 : ℝ) / 4 = (1 / 2) ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
  norm_num [hs]

end

end NN.MLTheory.CROWN.Graph.DirectedBackward
