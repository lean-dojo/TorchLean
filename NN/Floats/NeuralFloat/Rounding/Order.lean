/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Rounding.Core
public import NN.Floats.NeuralFloat.Rounding.Predicates
public import NN.Floats.NeuralFloat.Rounding.Generic

/-!
# Order Theory for Generic Rounding

Monotonicity is subtle because two inputs may be scaled with different canonical exponents.  The
proof separates equal-grid inputs from inputs in different magnitude bins and uses the bounds from
`GenericRound` in the latter case.
-/

@[expose] public section

namespace TorchLean.Floats

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]

/-- Rounding is monotone when both inputs use the same canonical exponent. -/
theorem neuralRound_le_of_cexp_eq (rnd : ℝ → ℤ) [NeuralValidRnd rnd]
    {x y : ℝ} (hxy : x ≤ y) (he : neuralCexp β fexp x = neuralCexp β fexp y) :
    neuralRound (β := β) (fexp := fexp) rnd x ≤
      neuralRound (β := β) (fexp := fexp) rnd y := by
  have hs : neuralScaledMantissa β fexp x ≤ neuralScaledMantissa β fexp y := by
    rw [neuralScaledMantissa_eq_div, neuralScaledMantissa_eq_div, he]
    exact div_le_div_of_nonneg_right hxy (neuralBpow.nonneg β _)
  have hm : rnd (neuralScaledMantissa β fexp x) ≤
      rnd (neuralScaledMantissa β fexp y) := NeuralValidRnd.monotone _ _ hs
  have hmR : (rnd (neuralScaledMantissa β fexp x) : ℝ) ≤
      (rnd (neuralScaledMantissa β fexp y) : ℝ) := by exact_mod_cast hm
  unfold neuralRound neuralToReal
  rw [he]
  exact mul_le_mul_of_nonneg_right hmR (neuralBpow.nonneg β _)

/-- Positive generic rounding is monotone. -/
theorem neuralRound_mono_pos (rnd : ℝ → ℤ) [NeuralValidRnd rnd]
    {x y : ℝ} (hx : 0 < x) (hxy : x ≤ y) :
    neuralRound (β := β) (fexp := fexp) rnd x ≤
      neuralRound (β := β) (fexp := fexp) rnd y := by
  have hy : 0 < y := hx.trans_le hxy
  let ex := neuralMagnitude β x
  let ey := neuralMagnitude β y
  have hmag : ex ≤ ey := by
    have hxLower := (neuralMagnitude_spec β x hx.ne').1
    have hyUpper := (neuralMagnitude_spec β y hy.ne').2
    have hpowers : neuralBpow β (ex - 1) < neuralBpow β ey := by
      exact hxLower.trans (by simpa [abs_of_pos hx, abs_of_pos hy] using hxy) |>.trans_lt hyUpper
    have : ex - 1 < ey := (neuralBpow_lt_neuralBpow_iff β _ _).mp hpowers
    linarith
  by_cases hySmall : ey ≤ fexp ey
  · have hconst := ((NeuralValidExp.flocq_valid (fexp := fexp) ey).2 hySmall).2
    have hf : fexp ex = fexp ey := hconst ex (hmag.trans hySmall)
    apply neuralRound_le_of_cexp_eq rnd hxy
    simp [neuralCexp, ex, ey, hf]
  · have hyLarge : fexp ey < ey := lt_of_not_ge hySmall
    by_cases heq : ex = ey
    · apply neuralRound_le_of_cexp_eq rnd hxy
      simp [neuralCexp, ex, ey, heq]
    · have hmagLt : ex < ey := lt_of_le_of_ne hmag heq
      have hxUpper : neuralRound (β := β) (fexp := fexp) rnd x ≤
          neuralBpow β (ey - 1) := by
        by_cases hxSmall : ex ≤ fexp ex
        · rcases neural_round_pos_small_cases
            (β := β) (fexp := fexp) rnd x hx (by simpa [ex] using hxSmall) with hz | hp
          · rw [hz]
            exact neuralBpow.nonneg β _
          · rw [hp]
            apply (neuralBpow_le_neuralBpow_iff β _ _).2
            have hfLt : fexp ex < ey := by
              by_contra hnot
              have hey : ey ≤ fexp ex := le_of_not_gt hnot
              have hconst := ((NeuralValidExp.flocq_valid (fexp := fexp) ex).2 hxSmall).2
              have := hconst ey hey
              linarith
            linarith
        · have hxLarge : fexp ex < ex := lt_of_not_ge hxSmall
          have hb := (neural_round_pos_large_bounds_and_generic
            (β := β) (fexp := fexp) rnd x hx (by simpa [ex] using hxLarge)).1.2
          exact hb.trans ((neuralBpow_le_neuralBpow_iff β _ _).2 (by linarith))
      have hyLower := (neural_round_pos_large_bounds_and_generic
        (β := β) (fexp := fexp) rnd y hy (by simpa [ey] using hyLarge)).1.1
      exact hxUpper.trans (by simpa [ey] using hyLower)

/-- Rounding a nonnegative value with a valid mode produces a nonnegative value. -/
theorem neuralRound_nonneg (rnd : ℝ → ℤ) [NeuralValidRnd rnd] {x : ℝ} (hx : 0 ≤ x) :
    0 ≤ neuralRound (β := β) (fexp := fexp) rnd x := by
  have hs : 0 ≤ neuralScaledMantissa β fexp x := by
    rw [neuralScaledMantissa_eq_div]
    exact div_nonneg hx (neuralBpow.nonneg β _)
  have hm : 0 ≤ rnd (neuralScaledMantissa β fexp x) := by
    have := NeuralValidRnd.monotone (rnd := rnd) 0 (neuralScaledMantissa β fexp x) hs
    have hr0 : rnd 0 = 0 := by simpa using NeuralValidRnd.id (rnd := rnd) (0 : ℤ)
    rwa [hr0] at this
  unfold neuralRound neuralToReal
  exact mul_nonneg (by exact_mod_cast hm) (neuralBpow.nonneg β _)

/-- Rounding a nonpositive value with a valid mode produces a nonpositive value. -/
theorem neuralRound_nonpos (rnd : ℝ → ℤ) [NeuralValidRnd rnd] {x : ℝ} (hx : x ≤ 0) :
    neuralRound (β := β) (fexp := fexp) rnd x ≤ 0 := by
  have hs : neuralScaledMantissa β fexp x ≤ 0 := by
    rw [neuralScaledMantissa_eq_div]
    exact div_nonpos_of_nonpos_of_nonneg hx (neuralBpow.nonneg β _)
  have hm : rnd (neuralScaledMantissa β fexp x) ≤ 0 := by
    have := NeuralValidRnd.monotone (rnd := rnd) (neuralScaledMantissa β fexp x) 0 hs
    have hr0 : rnd 0 = 0 := by simpa using NeuralValidRnd.id (rnd := rnd) (0 : ℤ)
    rwa [hr0] at this
  unfold neuralRound neuralToReal
  exact mul_nonpos_of_nonpos_of_nonneg (by exact_mod_cast hm) (neuralBpow.nonneg β _)

/-- Generic rounding is monotone on all real inputs. -/
theorem neuralRound_mono (rnd : ℝ → ℤ) [NeuralValidRnd rnd] {x y : ℝ} (hxy : x ≤ y) :
    neuralRound (β := β) (fexp := fexp) rnd x ≤
      neuralRound (β := β) (fexp := fexp) rnd y := by
  by_cases hx : 0 < x
  · exact neuralRound_mono_pos rnd hx hxy
  · have hx0 : x ≤ 0 := le_of_not_gt hx
    by_cases hy : 0 < y
    · exact (neuralRound_nonpos rnd hx0).trans (neuralRound_nonneg rnd hy.le)
    · have hy0 : y ≤ 0 := le_of_not_gt hy
      by_cases hyz : y = 0
      · subst y
        exact (neuralRound_nonpos rnd hx0).trans (by
          have hr0 : neuralRound (β := β) (fexp := fexp) rnd 0 = 0 := by
            have hgeneric : neuralGenericFormat β fexp 0 := neural_generic_format_zero
            exact neural_round_preserves_generic rnd 0 hgeneric
          rw [hr0])
      · have hny : 0 < -y := neg_pos.mpr (lt_of_le_of_ne hy0 hyz)
        have hnegxy : -y ≤ -x := neg_le_neg hxy
        have hmono := neuralRound_mono_pos (β := β) (fexp := fexp)
          (neuralNegRound rnd) hny hnegxy
        have hxround := neuralRound_neg (β := β) (fexp := fexp) rnd (-x)
        have hyround := neuralRound_neg (β := β) (fexp := fexp) rnd (-y)
        simp only [neg_neg] at hxround hyround
        rw [hxround, hyround]
        exact neg_le_neg hmono

/-- A representable lower value remains below rounding of any larger input. -/
theorem neural_generic_le_round (rnd : ℝ → ℤ) [NeuralValidRnd rnd] {x y : ℝ}
    (hx : neuralGenericFormat β fexp x) (hxy : x ≤ y) :
    x ≤ neuralRound (β := β) (fexp := fexp) rnd y := by
  rw [← neural_round_preserves_generic rnd x hx]
  exact neuralRound_mono rnd hxy

/-- Rounding of a smaller input remains below a representable upper value. -/
theorem neural_round_le_generic (rnd : ℝ → ℤ) [NeuralValidRnd rnd] {x y : ℝ}
    (hy : neuralGenericFormat β fexp y) (hxy : x ≤ y) :
    neuralRound (β := β) (fexp := fexp) rnd x ≤ y := by
  rw [← neural_round_preserves_generic rnd y hy]
  exact neuralRound_mono rnd hxy

/-- Directed-down generic rounding selects the greatest representable value below its input. -/
theorem neuralRound_floor_point (x : ℝ) :
    NeuralRoundDownPoint (neuralGenericFormat β fexp) x
      (neuralRound (β := β) (fexp := fexp) neuralFloorRound x) := by
  refine ⟨neural_generic_format_round neuralFloorRound x, neural_round_floor_le x, ?_⟩
  intro g hg hgx
  exact neural_generic_le_round neuralFloorRound hg hgx

/-- Directed-up generic rounding selects the least representable value above its input. -/
theorem neuralRound_ceil_point (x : ℝ) :
    NeuralRoundUpPoint (neuralGenericFormat β fexp) x
      (neuralRound (β := β) (fexp := fexp) neuralCeilRound x) := by
  refine ⟨neural_generic_format_round neuralCeilRound x, le_neural_round_ceil x, ?_⟩
  intro g hg hxg
  exact neural_round_le_generic neuralCeilRound hg hxg

/-- Generic truncation satisfies the toward-zero rounding specification. -/
theorem neuralRound_trunc_point (x : ℝ) :
    NeuralRoundTowardZeroPoint (neuralGenericFormat β fexp) x
      (neuralRound (β := β) (fexp := fexp) neuralTruncRound x) := by
  constructor
  · intro hx
    have hs : 0 ≤ neuralScaledMantissa β fexp x := by
      rw [neuralScaledMantissa_eq_div]
      exact div_nonneg hx (neuralBpow.nonneg β _)
    have hr : neuralRound (β := β) (fexp := fexp) neuralTruncRound x =
        neuralRound (β := β) (fexp := fexp) neuralFloorRound x := by
      unfold neuralRound
      rw [neuralTruncRound_eq_floor hs]
    rw [hr]
    exact neuralRound_floor_point x
  · intro hx
    have hs : neuralScaledMantissa β fexp x ≤ 0 := by
      rw [neuralScaledMantissa_eq_div]
      exact div_nonpos_of_nonpos_of_nonneg hx (neuralBpow.nonneg β _)
    have hr : neuralRound (β := β) (fexp := fexp) neuralTruncRound x =
        neuralRound (β := β) (fexp := fexp) neuralCeilRound x := by
      unfold neuralRound
      rw [neuralTruncRound_eq_ceil hs]
    rw [hr]
    exact neuralRound_ceil_point x

/-- Nearest-even generic rounding is no farther from the input than any other integer rounding. -/
theorem neuralRound_nearestEven_error_le (rnd : ℝ → ℤ) (x : ℝ) :
    abs (neuralRound (β := β) (fexp := fexp) neuralNearestEven x - x) ≤
      abs (neuralRound (β := β) (fexp := fexp) rnd x - x) := by
  let s := neuralScaledMantissa β fexp x
  let b := neuralBpow β (neuralCexp β fexp x)
  have hxb : x = s * b := by
    simpa [s, b] using (neural_scaled_mantissa_mul_bpow (β := β) (fexp := fexp) x).symm
  have hlocal := neuralNearestEven_is_nearest_integer s (rnd s)
  have hb : 0 ≤ b := by simp [b, neuralBpow.nonneg]
  have hscaled := mul_le_mul_of_nonneg_right hlocal hb
  have hnear : neuralRound (β := β) (fexp := fexp) neuralNearestEven x - x =
      ((neuralNearestEven s : ℝ) - s) * b := by
    unfold neuralRound neuralToReal
    change (neuralNearestEven s : ℝ) * b - x = ((neuralNearestEven s : ℝ) - s) * b
    rw [hxb]
    ring
  have hrnd : neuralRound (β := β) (fexp := fexp) rnd x - x =
      ((rnd s : ℝ) - s) * b := by
    unfold neuralRound neuralToReal
    change (rnd s : ℝ) * b - x = ((rnd s : ℝ) - s) * b
    rw [hxb]
    ring
  rw [hnear, hrnd, abs_mul, abs_mul, abs_of_nonneg hb]
  exact hscaled

/-- Nearest-even rounding selects a globally nearest representable value. -/
theorem neuralRound_nearestEven_point (x : ℝ) :
    NeuralRoundNearestPoint (neuralGenericFormat β fexp) x
      (neuralRound (β := β) (fexp := fexp) neuralNearestEven x) := by
  let d := neuralRound (β := β) (fexp := fexp) neuralFloorRound x
  let u := neuralRound (β := β) (fexp := fexp) neuralCeilRound x
  let f := neuralRound (β := β) (fexp := fexp) neuralNearestEven x
  apply neuralRoundNearestPoint_of_down_up
    (neuralRound_floor_point (β := β) (fexp := fexp) x)
    (neuralRound_ceil_point (β := β) (fexp := fexp) x)
  · exact neural_generic_format_round neuralNearestEven x
  · exact neuralRound_nearestEven_error_le neuralFloorRound x
  · exact neuralRound_nearestEven_error_le neuralCeilRound x

end TorchLean.Floats
