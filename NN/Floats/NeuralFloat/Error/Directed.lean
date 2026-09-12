/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Error.Bounds
public import NN.Floats.NeuralFloat.Analysis.Ulp
public import NN.Floats.NeuralFloat.Format.Formats
public import NN.Floats.NeuralFloat.Rounding.Generic
public import NN.Floats.NeuralFloat.Error.Relative

/-!
# Error Bounds for General Rounding Modes

A valid rounding mode selects one of the two adjacent representable values.  Consequently its
absolute error is at most one ULP.  Nearest rounding sharpens this to half an ULP; this file records
the one-ULP result needed for directed and toward-zero modes.
-/

@[expose] public section

namespace TorchLean.Floats

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]

/-- Every valid generic rounding mode has absolute error at most one ULP. -/
theorem neural_round_abs_error_le_ulp (rnd : ℝ → ℤ) [NeuralValidRnd rnd] (x : ℝ) :
    abs (neuralRound (β := β) (fexp := fexp) rnd x - x) ≤ neuralUlp β fexp x := by
  by_cases hx : neuralGenericFormat β fexp x
  · rw [neural_round_preserves_generic rnd x hx, sub_self, abs_zero]
    exact neuralUlp.nonneg (β := β) (fexp := fexp) x
  · let d := neuralRound (β := β) (fexp := fexp) neuralFloorRound x
    let u := neuralRound (β := β) (fexp := fexp) neuralCeilRound x
    let r := neuralRound (β := β) (fexp := fexp) rnd x
    have hd : d ≤ x := neural_round_floor_le x
    have hu : x ≤ u := le_neural_round_ceil x
    have hdr : d ≤ r := neural_round_floor_le_round rnd x
    have hru : r ≤ u := neural_round_le_ceil rnd x
    have hgap : u = d + neuralUlp β fexp x :=
      neuralRound_ceil_eq_floor_add_ulp hx
    rcases le_total r x with hrx | hxr
    · rw [abs_of_nonpos (sub_nonpos.mpr hrx)]
      simp only [neg_sub]
      linarith
    · rw [abs_of_nonneg (sub_nonneg.mpr hxr)]
      linarith

/-- A non-exact valid rounding has error strictly smaller than one ULP. -/
theorem neural_round_abs_error_lt_ulp_of_inexact (rnd : ℝ → ℤ) [NeuralValidRnd rnd]
    {x : ℝ} (hinexact : neuralRound (β := β) (fexp := fexp) rnd x ≠ x) :
    abs (neuralRound (β := β) (fexp := fexp) rnd x - x) < neuralUlp β fexp x := by
  have hx : ¬neuralGenericFormat β fexp x := by
    intro hx
    exact hinexact (neural_round_preserves_generic rnd x hx)
  let d := neuralRound (β := β) (fexp := fexp) neuralFloorRound x
  let u := neuralRound (β := β) (fexp := fexp) neuralCeilRound x
  let r := neuralRound (β := β) (fexp := fexp) rnd x
  have hdle : d ≤ x := neural_round_floor_le x
  have hxleu : x ≤ u := le_neural_round_ceil x
  have hdlt : d < x := lt_of_le_of_ne hdle (by
    intro h
    apply hx
    rw [← h]
    exact neural_generic_format_round neuralFloorRound x)
  have hxlt : x < u := lt_of_le_of_ne hxleu (by
    intro h
    apply hx
    rw [h]
    exact neural_generic_format_round neuralCeilRound x)
  have hgap : u = d + neuralUlp β fexp x := neuralRound_ceil_eq_floor_add_ulp hx
  rcases neuralRound_eq_floor_or_ceil (β := β) (fexp := fexp) rnd x with hr | hr
  · change abs (r - x) < neuralUlp β fexp x
    have hr' : r = d := hr
    rw [hr']
    rw [abs_of_nonpos (sub_nonpos.mpr hdle), neg_sub]
    linarith
  · change abs (r - x) < neuralUlp β fexp x
    have hr' : r = u := hr
    rw [hr']
    rw [abs_of_nonneg (sub_nonneg.mpr hxleu)]
    linarith

/-- Every valid FLX rounding mode has relative error at most `β^(1-prec)`. -/
theorem relative_error_round_FLX_of_valid (prec : ℤ) (hprec : 0 < prec)
    (rnd : ℝ → ℤ) [NeuralValidRnd rnd] (x : ℝ) (hx : x ≠ 0) :
    ErrorBounds.relativeError x
        (@neuralRound β (FLXExp prec) (flxValidExp prec hprec) rnd x) hx ≤
      neuralBpow β (1 - prec) := by
  let _ : NeuralValidExp (FLXExp prec) := flxValidExp prec hprec
  unfold ErrorBounds.relativeError
  calc
    abs (neuralRound (β := β) (fexp := FLXExp prec) rnd x - x) / abs x ≤
        neuralUlp β (FLXExp prec) x / abs x :=
      div_le_div_of_nonneg_right (neural_round_abs_error_le_ulp rnd x) (abs_nonneg x)
    _ ≤ neuralBpow β (1 - prec) := neuralUlp_div_abs_le_FLX prec hprec x hx

/-- General FLX rounding admits a multiplicative error model with a one-ULP relative bound. -/
theorem neural_round_relative_error_FLX_of_valid (prec : ℤ) (hprec : 0 < prec)
    (rnd : ℝ → ℤ) [NeuralValidRnd rnd] (x : ℝ) (hx : x ≠ 0) :
    ∃ δ : ℝ,
      abs δ ≤ neuralBpow β (1 - prec) ∧
      @neuralRound β (FLXExp prec) (flxValidExp prec hprec) rnd x = x * (1 + δ) := by
  let _ : NeuralValidExp (FLXExp prec) := flxValidExp prec hprec
  let rounded := neuralRound (β := β) (fexp := FLXExp prec) rnd x
  refine ⟨(rounded - x) / x, ?_, ?_⟩
  · rw [abs_div]
    exact relative_error_round_FLX_of_valid prec hprec rnd x hx
  · dsimp [rounded]
    field_simp [hx]
    ring

end TorchLean.Floats
