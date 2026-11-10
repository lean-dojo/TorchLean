/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.Activation
public import TorchLean.Tensor.Context.Real

/-!
# Scalar ReLU Upper Bound

The CROWN upper relaxation `relaxScalar l u` lies above ReLU on the whole interval `[l, u]`
(`relaxScalar_sound`). The graph and phase-aware certificate proofs use this scalar
inequality.
-/

@[expose] public section

namespace TorchLean.CROWN

/-- For `x` in `[l, u]`, ReLU at `x` is at most the CROWN upper line `relaxScalar l u` at `x`. -/
theorem ReLU.relaxScalar_sound
    (l u x : ℝ) (hlx : l ≤ x) (hxu : x ≤ u) :
    let rp := CROWN.ReLU.relaxScalar (α := ℝ) l u
    Activation.relu (α := ℝ) x ≤ rp.slope * x + rp.bias := by
  unfold CROWN.ReLU.relaxScalar
  by_cases hu : u > 0
  · by_cases hlpos : l > 0
    · have hxpos : 0 < x := lt_of_lt_of_le hlpos hlx
      have hxnonneg : 0 ≤ x := le_of_lt hxpos
      simp [hu, hlpos, Activation.relu_eq_max, max_eq_left hxnonneg]
    · have hle0 : l ≤ 0 := le_of_not_gt hlpos
      have hden : 0 < (u - l) := by linarith
      simp only [hu, hlpos, ite_true, ite_false]
      by_cases hxpos : 0 < x
      · have hxnonneg : 0 ≤ x := le_of_lt hxpos
        simp [Activation.relu_eq_max, max_eq_left hxnonneg]
        -- Triangular relaxation: after clearing `u - l > 0` the claim is `l * (u - x) ≤ 0`.
        have hx_to_goal : x ≤ u / (u - l) * (x - l) := by
          rw [div_mul_eq_mul_div, le_div_iff₀ hden]
          linarith [mul_nonpos_of_nonpos_of_nonneg hle0 (sub_nonneg.mpr hxu)]
        have h2 : u / (u - l) * (x - l) = u / (u - l) * x + -(u / (u - l)) * l := by ring
        simpa [h2] using hx_to_goal
      · have hxle : x ≤ 0 := le_of_not_gt hxpos
        have h1 : u / (u - l) * x + -(u / (u - l) * l) = u / (u - l) * (x - l) := by ring
        have : 0 ≤ u / (u - l) * (x - l) := by
          apply mul_nonneg
          · have : 0 ≤ u := le_of_lt hu
            exact div_nonneg this (le_of_lt hden)
          · linarith
        simpa [Activation.relu_eq_max, max_eq_right hxle, h1] using this
  · have hxle : x ≤ 0 := le_trans hxu (le_of_not_gt hu)
    simp [hu, Activation.relu_eq_max, max_eq_right hxle]

end TorchLean.CROWN
