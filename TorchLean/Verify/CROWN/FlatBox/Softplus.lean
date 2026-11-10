/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Gradients.Activation

/-!
# Softplus envelope over the reals

Two scalar facts about softplus over `ℝ`: `softplus_tail_bounds_real` bounds the logarithmic
tail `log (1 + exp z)` between 0 and 1 for `z ≤ 0`, and `Activation.softplus_envelope_real` gives
`max x 0 ≤ softplus x ≤ max x 0 + 1`, the envelope behind the softplus interval transfer.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean

open TorchLean.Tensor

noncomputable section

/-- For `z ≤ 0`, the logarithmic tail `Real.log (1 + Real.exp z)` of the stable softplus branch lies
between zero and one. -/
theorem CROWN.Graph.softplus_tail_bounds_real {z : ℝ} (hz : z ≤ 0) :
    0 ≤ Real.log (1 + Real.exp z) ∧ Real.log (1 + Real.exp z) ≤ 1 := by
  have hexp : Real.exp z ≤ 1 := Real.exp_le_one_iff.mpr hz
  have hpositive : 0 < 1 + Real.exp z := by positivity
  refine ⟨Real.log_nonneg (by linarith [Real.exp_pos z]), ?_⟩
  have hlog := Real.log_le_sub_one_of_pos hpositive
  linarith

/-- Softplus lies between `max x 0` and `max x 0 + 1`.

The proof follows the sign branch of the specification. The exponential argument is nonpositive in
both branches, so the logarithmic tail lies in `[0, 1]` and the interval rule built on this envelope
evaluates no exponential. -/
theorem Activation.softplus_envelope_real (x : ℝ) :
    max x 0 ≤ Activation.softplus x ∧
      Activation.softplus x ≤ max x 0 + 1 := by
  simp only [Activation.softplus, Context.log1p_eq_log_one_add, MathFunctions.exp]
  by_cases hx : x > 0
  · simp only [ite_eq_left hx, max_eq_left (le_of_lt hx)]
    have htail := CROWN.Graph.softplus_tail_bounds_real (neg_nonpos.mpr (le_of_lt hx))
    constructor <;> linarith
  · have hnonpos : x ≤ 0 := le_of_not_gt hx
    simpa only [ite_eq_right hx, max_eq_right hnonpos, zero_add] using
      CROWN.Graph.softplus_tail_bounds_real hnonpos

end

end TorchLean
