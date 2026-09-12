/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Error.Bounds
public import NN.Floats.NeuralFloat.Format.Formats
public import NN.Floats.NeuralFloat.Format.Magnitude

/-!
# Relative Error in the FLX Format

The unbounded-exponent `FLXExp prec` format has a uniform relative-error bound.  For a nonzero
input, its ULP is `β^(magnitude x - prec)`, while the magnitude lower bound gives
`β^(magnitude x - 1) ≤ |x|`.  Their ratio is therefore at most `β^(1 - prec)`.
-/

@[expose] public section

namespace TorchLean.Floats

variable {β : NeuralRadix}

/-- The relative size of one FLX ULP is at most `β^(1-prec)`. -/
theorem neuralUlp_div_abs_le_FLX (prec : ℤ) (hprec : 0 < prec) (x : ℝ) (hx : x ≠ 0) :
    @neuralUlp β (FLXExp prec) (flxValidExp prec hprec) x / abs x ≤
      neuralBpow β (1 - prec) := by
  let _ : NeuralValidExp (FLXExp prec) := flxValidExp prec hprec
  rw [div_le_iff₀ (abs_pos.mpr hx)]
  rw [neuralUlp.of_ne_zero β (FLXExp prec) x hx]
  have hlower := neuralBpow_magnitude_sub_one_le β x hx
  calc
    neuralBpow β (neuralCexp β (FLXExp prec) x) =
        neuralBpow β (1 - prec) * neuralBpow β (neuralMagnitude β x - 1) := by
      rw [← neuralBpow.add_exp]
      congr 1
      simp [neuralCexp, FLXExp]
    _ ≤ neuralBpow β (1 - prec) * abs x :=
      mul_le_mul_of_nonneg_left hlower (neuralBpow.nonneg β _)

/-- Nearest FLX rounding has the standard uniform relative-error bound. -/
theorem relative_error_round_FLX (prec : ℤ) (hprec : 0 < prec)
    (rnd : ℝ → ℤ) [NeuralValidRndToNearest rnd] (x : ℝ) (hx : x ≠ 0) :
    ErrorBounds.relativeError x
        (@neuralRound β (FLXExp prec) (flxValidExp prec hprec) rnd x) hx ≤
      neuralBpow β (1 - prec) / 2 := by
  let _ : NeuralValidExp (FLXExp prec) := flxValidExp prec hprec
  calc
    ErrorBounds.relativeError x (neuralRound (β := β) (fexp := FLXExp prec) rnd x) hx ≤
        neuralUlp β (FLXExp prec) x / (2 * abs x) :=
      ErrorBounds.relative_error_round_ulp rnd x hx
    _ = (neuralUlp β (FLXExp prec) x / abs x) / 2 := by
      field_simp [abs_ne_zero.mpr hx]
    _ ≤ neuralBpow β (1 - prec) / 2 :=
      div_le_div_of_nonneg_right (neuralUlp_div_abs_le_FLX prec hprec x hx) (by norm_num)

/--
Nearest FLX rounding admits the usual multiplicative model
`round x = x * (1 + δ)` with `|δ| ≤ β^(1-prec)/2`.
-/
theorem neural_round_relative_error_FLX (prec : ℤ) (hprec : 0 < prec)
    (rnd : ℝ → ℤ) [NeuralValidRndToNearest rnd] (x : ℝ) (hx : x ≠ 0) :
    ∃ δ : ℝ,
      abs δ ≤ neuralBpow β (1 - prec) / 2 ∧
      @neuralRound β (FLXExp prec) (flxValidExp prec hprec) rnd x = x * (1 + δ) := by
  let _ : NeuralValidExp (FLXExp prec) := flxValidExp prec hprec
  obtain ⟨δ, hδ, hround⟩ :=
    ErrorBounds.neural_round_relative_error_ulp
      (β := β) (fexp := FLXExp prec) rnd x hx
  refine ⟨δ, hδ.trans ?_, hround⟩
  calc
    neuralUlp β (FLXExp prec) x / (2 * abs x) =
        (neuralUlp β (FLXExp prec) x / abs x) / 2 := by
      field_simp [abs_ne_zero.mpr hx]
    _ ≤ neuralBpow β (1 - prec) / 2 :=
      div_le_div_of_nonneg_right (neuralUlp_div_abs_le_FLX prec hprec x hx) (by norm_num)

/-- In the normal range, one FLT ULP has relative size at most `β^(1-prec)`. -/
theorem neuralUlp_div_abs_le_FLT_normal (emin prec : ℤ) (hprec : 0 < prec)
    (x : ℝ) (hx : x ≠ 0)
    (hnormal : neuralBpow β (emin + prec - 1) ≤ abs x) :
    @neuralUlp β (FLTExp emin prec) (fltValidExp emin prec hprec) x / abs x ≤
      neuralBpow β (1 - prec) := by
  let _ : NeuralValidExp (FLTExp emin prec) := fltValidExp emin prec hprec
  have hupper := abs_lt_neuralBpow_magnitude β x hx
  have hpowLt : neuralBpow β (emin + prec - 1) <
      neuralBpow β (neuralMagnitude β x) := hnormal.trans_lt hupper
  have hnormalExp : emin ≤ neuralMagnitude β x - prec := by
    have := (neuralBpow_lt_neuralBpow_iff β _ _).mp hpowLt
    linarith
  rw [div_le_iff₀ (abs_pos.mpr hx)]
  rw [neuralUlp.of_ne_zero β (FLTExp emin prec) x hx]
  have hlower := neuralBpow_magnitude_sub_one_le β x hx
  calc
    neuralBpow β (neuralCexp β (FLTExp emin prec) x) =
        neuralBpow β (1 - prec) * neuralBpow β (neuralMagnitude β x - 1) := by
      rw [← neuralBpow.add_exp]
      congr 1
      simp [neuralCexp, FLTExp, max_eq_left hnormalExp]
    _ ≤ neuralBpow β (1 - prec) * abs x :=
      mul_le_mul_of_nonneg_left hlower (neuralBpow.nonneg β _)

/-- Nearest FLT rounding has the FLX relative bound throughout the normal range. -/
theorem relative_error_round_FLT_normal (emin prec : ℤ) (hprec : 0 < prec)
    (rnd : ℝ → ℤ) [NeuralValidRndToNearest rnd] (x : ℝ) (hx : x ≠ 0)
    (hnormal : neuralBpow β (emin + prec - 1) ≤ abs x) :
    ErrorBounds.relativeError x
        (@neuralRound β (FLTExp emin prec) (fltValidExp emin prec hprec) rnd x) hx ≤
      neuralBpow β (1 - prec) / 2 := by
  let _ : NeuralValidExp (FLTExp emin prec) := fltValidExp emin prec hprec
  calc
    ErrorBounds.relativeError x
        (neuralRound (β := β) (fexp := FLTExp emin prec) rnd x) hx ≤
        neuralUlp β (FLTExp emin prec) x / (2 * abs x) :=
      ErrorBounds.relative_error_round_ulp rnd x hx
    _ = (neuralUlp β (FLTExp emin prec) x / abs x) / 2 := by
      field_simp [abs_ne_zero.mpr hx]
    _ ≤ neuralBpow β (1 - prec) / 2 :=
      div_le_div_of_nonneg_right
        (neuralUlp_div_abs_le_FLT_normal emin prec hprec x hx hnormal) (by norm_num)

end TorchLean.Floats
