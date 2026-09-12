/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Error.Exactness
public import NN.Floats.NeuralFloat.Error.Directed

/-!
# Division and Square-Root Residuals

This file develops exact representability results for arithmetic residuals in FLX.  A division
residual is controlled both relative to the dividend and relative to the product of the rounded
quotient with the divisor; the two bounds place it on their common exponent grid.
-/

@[expose] public section

namespace TorchLean.Floats

variable {β : NeuralRadix}

/-- Positive FLX rounding cannot move a nonzero result into a lower magnitude bin. -/
private theorem neuralMagnitude_le_round_FLX_of_pos (prec : ℤ) (hprec : 0 < prec)
    (rnd : ℝ → ℤ) [NeuralValidRnd rnd] {x : ℝ} (hx : 0 < x) :
    neuralMagnitude β x ≤
      neuralMagnitude β
        (@neuralRound β (FLXExp prec) (flxValidExp prec hprec) rnd x) := by
  let _ : NeuralValidExp (FLXExp prec) := flxValidExp prec hprec
  have hlarge : FLXExp prec (neuralMagnitude β x) < neuralMagnitude β x := by
    simp [FLXExp]
    linarith
  have hb := (neural_round_pos_large_bounds_and_generic
    (β := β) (fexp := FLXExp prec) rnd x hx hlarge).1.1
  let q := neuralRound (β := β) (fexp := FLXExp prec) rnd x
  have hqpos : 0 < q := (neuralBpow.pos β _).trans_le hb
  have hqUpper := abs_lt_neuralBpow_magnitude β q hqpos.ne'
  have hpowers : neuralBpow β (neuralMagnitude β x - 1) <
      neuralBpow β (neuralMagnitude β q) := by
    exact lt_of_le_of_lt hb (by simpa [abs_of_pos hqpos] using hqUpper)
  have hexp : neuralMagnitude β x - 1 < neuralMagnitude β q :=
    (neuralBpow_lt_neuralBpow_iff β _ _).mp hpowers
  linarith

/-- A nonzero FLX rounded result has magnitude at least that of its exact input. -/
theorem neuralMagnitude_le_round_FLX (prec : ℤ) (hprec : 0 < prec)
    (rnd : ℝ → ℤ) [NeuralValidRnd rnd] {x : ℝ}
    (hround : @neuralRound β (FLXExp prec) (flxValidExp prec hprec) rnd x ≠ 0) :
    neuralMagnitude β x ≤
      neuralMagnitude β
        (@neuralRound β (FLXExp prec) (flxValidExp prec hprec) rnd x) := by
  let _ : NeuralValidExp (FLXExp prec) := flxValidExp prec hprec
  rcases lt_trichotomy x 0 with hx | hx | hx
  · let nrnd := neuralNegRound rnd
    have hpos : 0 < -x := neg_pos.mpr hx
    have hmag := neuralMagnitude_le_round_FLX_of_pos
      (β := β) prec hprec nrnd hpos
    have hneg := neuralRound_neg (β := β) (fexp := FLXExp prec) rnd (-x)
    simp only [neg_neg] at hneg
    simpa [hneg] using hmag
  · subst x
    simp [neuralMagnitude]
  · exact neuralMagnitude_le_round_FLX_of_pos (β := β) prec hprec rnd hx

/-- The residual `x - round(x / y) * y` is exactly FLX-representable. -/
theorem neural_div_round_residual_FLX (prec : ℤ) (hprec : 0 < prec)
    (rnd : ℝ → ℤ) [NeuralValidRnd rnd] {x y : ℝ}
    (hx : @neuralGenericFormat β (FLXExp prec) (flxValidExp prec hprec) x)
    (hy : @neuralGenericFormat β (FLXExp prec) (flxValidExp prec hprec) y) :
    @neuralGenericFormat β (FLXExp prec) (flxValidExp prec hprec)
      (x - @neuralRound β (FLXExp prec) (flxValidExp prec hprec) rnd (x / y) * y) := by
  let _ : NeuralValidExp (FLXExp prec) := flxValidExp prec hprec
  let z := x / y
  let q := neuralRound (β := β) (fexp := FLXExp prec) rnd z
  let residual := x - q * y
  by_cases hy0 : y = 0
  · subst y
    simpa [residual, q] using hx
  by_cases hq0 : q = 0
  · change neuralGenericFormat β (FLXExp prec) residual
    simpa [residual, hq0] using hx
  have hx0 : x ≠ 0 := by
    intro hx0
    subst x
    have hz : z = 0 := by simp [z]
    have hq : q = 0 := by
      unfold q
      rw [hz]
      exact neural_round_preserves_generic rnd 0 neural_generic_format_zero
    exact hq0 hq
  have hz0 : z ≠ 0 := div_ne_zero hx0 hy0
  obtain ⟨fx, hfx, cfx⟩ := neural_canonical_exists_of_generic hx
  obtain ⟨fy, hfy, cfy⟩ := neural_canonical_exists_of_generic hy
  have hqFmt : neuralGenericFormat β (FLXExp prec) q := by
    exact neural_generic_format_round rnd z
  obtain ⟨fq, hfq, cfq⟩ := neural_canonical_exists_of_generic hqFmt
  let negProduct : NeuralFloat β :=
    { mantissa := -(fq.mantissa * fy.mantissa)
      exponent := fq.exponent + fy.exponent }
  have hnegProduct : -(q * y) = neuralToReal negProduct := by
    rw [hfq, hfy]
    unfold negProduct neuralToReal
    rw [neuralBpow.add_exp]
    push_cast
    ring
  have hresidualAdd : residual = x + -(q * y) := by simp [residual, sub_eq_add_neg]
  by_cases hinexact : q = z
  · change neuralGenericFormat β (FLXExp prec) residual
    have hres0 : residual = 0 := by
      simp [residual, hinexact, z, hy0]
    rw [hres0]
    exact neural_generic_format_zero
  have hinexact' : q ≠ z := hinexact
  have herrorStrict : abs (q - z) < neuralUlp β (FLXExp prec) z := by
    exact neural_round_abs_error_lt_ulp_of_inexact rnd hinexact'
  have hrelativeUlp : neuralUlp β (FLXExp prec) z ≤
      neuralBpow β (1 - prec) * abs z := by
    have h := neuralUlp_div_abs_le_FLX (β := β) prec hprec z hz0
    rw [div_le_iff₀ (abs_pos.mpr hz0)] at h
    simpa [mul_comm] using h
  have hresAbs : abs residual = abs y * abs (q - z) := by
    have halg : residual = -(y * (q - z)) := by
      unfold residual z
      field_simp [hy0]
      ring
    rw [halg, abs_neg, abs_mul]
  have hboundX : abs residual < neuralBpow β (prec + fx.exponent) := by
    have hxy : abs y * abs z = abs x := by
      dsimp [z]
      rw [abs_div]
      field_simp [abs_ne_zero.mpr hy0]
    have heps : neuralBpow β (1 - prec) ≤ 1 := by
      calc
        neuralBpow β (1 - prec) ≤ neuralBpow β 0 :=
          (neuralBpow_le_neuralBpow_iff β _ _).2 (by linarith)
        _ = 1 := by simp [neuralBpow]
    have hlt : abs residual < abs y *
        (neuralBpow β (1 - prec) * abs z) := by
      rw [hresAbs]
      exact mul_lt_mul_of_pos_left
        (herrorStrict.trans_le hrelativeUlp) (abs_pos.mpr hy0)
    have hle : abs y * (neuralBpow β (1 - prec) * abs z) ≤ abs x := by
      calc
        abs y * (neuralBpow β (1 - prec) * abs z) =
            neuralBpow β (1 - prec) * (abs y * abs z) := by ring
        _ ≤ 1 * (abs y * abs z) :=
          mul_le_mul_of_nonneg_right heps (mul_nonneg (abs_nonneg y) (abs_nonneg z))
        _ = abs x := by simp [hxy]
    have hxUpper := abs_lt_neuralBpow_magnitude β x hx0
    have hfxexp : fx.exponent = neuralCexp β (FLXExp prec) x := by
      rw [hfx]
      exact cfx
    calc
      abs residual < abs x := hlt.trans_le hle
      _ < neuralBpow β (prec + fx.exponent) := by
        rw [hfxexp]
        simpa [neuralCexp, FLXExp] using hxUpper
  have hcexpZQ : neuralCexp β (FLXExp prec) z ≤
      neuralCexp β (FLXExp prec) q := by
    simp only [neuralCexp, FLXExp]
    exact sub_le_sub_right (neuralMagnitude_le_round_FLX
      (β := β) prec hprec rnd hq0) prec
  have hboundProduct : abs residual <
      neuralBpow β (prec + negProduct.exponent) := by
    have hyUpper := abs_lt_neuralBpow_magnitude β y hy0
    have herrBpow : abs (q - z) <
        neuralBpow β (neuralCexp β (FLXExp prec) z) := by
      simpa [neuralUlp, hz0] using herrorStrict
    have herrBpowQ : abs (q - z) <
        neuralBpow β (neuralCexp β (FLXExp prec) q) :=
      herrBpow.trans_le ((neuralBpow_le_neuralBpow_iff β _ _).2 hcexpZQ)
    have hprod : abs y * abs (q - z) <
        neuralBpow β (neuralMagnitude β y) *
          neuralBpow β (neuralCexp β (FLXExp prec) q) :=
      mul_lt_mul hyUpper herrBpowQ.le (abs_pos.mpr (sub_ne_zero.mpr hinexact'))
        (neuralBpow.nonneg β _)
    rw [← neuralBpow.add_exp] at hprod
    rw [hresAbs]
    apply hprod.trans_le
    apply (neuralBpow_le_neuralBpow_iff β _ _).2
    have hfqexp : fq.exponent = neuralCexp β (FLXExp prec) q := by
      rw [hfq]
      exact cfq
    have hfyexp : fy.exponent = neuralCexp β (FLXExp prec) y := by
      rw [hfy]
      exact cfy
    simp [negProduct, hfqexp, hfyexp, neuralCexp, FLXExp]
    linarith
  change neuralGenericFormat β (FLXExp prec) residual
  rw [hresidualAdd]
  apply neural_generic_format_FLX_add_of_repr_bounds prec hprec fx negProduct x (-(q * y))
  · exact hfx
  · exact hnegProduct
  · simpa [hresidualAdd] using hboundX
  · simpa [hresidualAdd] using hboundProduct

/-- For precision greater than one, the nearest-rounded square-root residual is FLX-representable. -/
theorem neural_sqrt_round_residual_FLX (prec : ℤ) (hprec : 1 < prec) {x : ℝ}
    (hxFmt : @neuralGenericFormat β (FLXExp prec) (flxValidExp prec (by linarith)) x) :
    @neuralGenericFormat β (FLXExp prec) (flxValidExp prec (by linarith))
      (x - (@neuralRound β (FLXExp prec) (flxValidExp prec (by linarith))
        neuralNearestEven (Real.sqrt x)) ^ 2) := by
  have hp0 : 0 < prec := by linarith
  let _ : NeuralValidExp (FLXExp prec) := flxValidExp prec hp0
  let s := Real.sqrt x
  let q := neuralRound (β := β) (fexp := FLXExp prec) neuralNearestEven s
  let residual := x - q ^ 2
  rcases lt_trichotomy x 0 with hx | hx | hx
  · have hs0 : s = 0 := by
      simp [s, Real.sqrt_eq_zero_of_nonpos hx.le]
    have hq0 : q = 0 := by
      unfold q
      rw [hs0]
      exact neural_round_preserves_generic neuralNearestEven 0 neural_generic_format_zero
    change neuralGenericFormat β (FLXExp prec) residual
    simpa [residual, hq0] using hxFmt
  · subst x
    change neuralGenericFormat β (FLXExp prec) residual
    have hs0 : s = 0 := by simp [s]
    have hq0 : q = 0 := by
      unfold q
      rw [hs0]
      exact neural_round_preserves_generic neuralNearestEven 0 neural_generic_format_zero
    have hres0 : residual = 0 := by simp [residual, hq0]
    rw [hres0]
    exact neural_generic_format_zero
  have hspos : 0 < s := by simpa [s] using Real.sqrt_pos.2 hx
  have hsSq : s ^ 2 = x := by simpa [s] using Real.sq_sqrt hx.le
  have hlarge : FLXExp prec (neuralMagnitude β s) < neuralMagnitude β s := by
    simp [FLXExp]
    linarith
  have hqBounds := (neural_round_pos_large_bounds_and_generic
    (β := β) (fexp := FLXExp prec) neuralNearestEven s hspos hlarge).1
  have hqpos : 0 < q := (neuralBpow.pos β _).trans_le hqBounds.1
  obtain ⟨fx, hfx, cfx⟩ := neural_canonical_exists_of_generic hxFmt
  have hqFmt : neuralGenericFormat β (FLXExp prec) q :=
    neural_generic_format_round neuralNearestEven s
  obtain ⟨fq, hfq, cfq⟩ := neural_canonical_exists_of_generic hqFmt
  let negSquare : NeuralFloat β :=
    { mantissa := -(fq.mantissa * fq.mantissa)
      exponent := fq.exponent + fq.exponent }
  have hnegSquare : -(q ^ 2) = neuralToReal negSquare := by
    rw [hfq]
    unfold negSquare neuralToReal
    rw [neuralBpow.add_exp]
    push_cast
    ring
  have hresidualAdd : residual = x + -(q ^ 2) := by simp [residual, sub_eq_add_neg]
  by_cases hqs : q = s
  · change neuralGenericFormat β (FLXExp prec) residual
    have hres0 : residual = 0 := by simp [residual, hqs, hsSq]
    rw [hres0]
    exact neural_generic_format_zero
  have herrorUlp : abs (q - s) ≤ neuralUlp β (FLXExp prec) s / 2 :=
    neural_error_bound_ulp neuralNearestEven s
  have herrorBpow : abs (q - s) ≤
      neuralBpow β (neuralCexp β (FLXExp prec) s) / 2 := by
    simpa [neuralUlp, hspos.ne'] using herrorUlp
  let u := neuralBpow β (1 - prec) / 2
  have hrelative : abs (q - s) ≤ u * s := by
    have h := relative_error_round_FLX
      (β := β) prec hp0 neuralNearestEven s hspos.ne'
    unfold ErrorBounds.relativeError at h
    rw [div_le_iff₀ (abs_pos.mpr hspos.ne')] at h
    simpa [u, abs_of_pos hspos, mul_comm] using h
  have hbHalf : neuralBpow β (-1) ≤ (1 : ℝ) / 2 := by
    have hb : (2 : ℝ) ≤ β.toReal := by
      change (2 : ℝ) ≤ (β.base : ℝ)
      exact_mod_cast β.base_valid
    calc
      neuralBpow β (-1) = β.toReal⁻¹ := by simp [neuralBpow]
      _ ≤ (2 : ℝ)⁻¹ := by
        simpa [one_div] using one_div_le_one_div_of_le (by norm_num : (0 : ℝ) < 2) hb
      _ = 1 / 2 := by norm_num
  have hepsHalf : neuralBpow β (1 - prec) ≤ (1 : ℝ) / 2 := by
    calc
      neuralBpow β (1 - prec) ≤ neuralBpow β (-1) :=
        (neuralBpow_le_neuralBpow_iff β _ _).2 (by linarith)
      _ ≤ 1 / 2 := hbHalf
  have huNonneg : 0 ≤ u := div_nonneg (neuralBpow.nonneg β _) (by norm_num)
  have huQuarter : u ≤ (1 : ℝ) / 4 := by
    dsimp [u]
    linarith
  have hqLe : q ≤ (1 + u) * s := by
    have habsLower : q - s ≤ abs (q - s) := le_abs_self (q - s)
    nlinarith
  have hsumPos : 0 < q + s := add_pos hqpos hspos
  have hsumLe : q + s ≤ (9 / 4 : ℝ) * s := by
    nlinarith
  have hresAbs : abs residual = abs (q - s) * (q + s) := by
    have halg : residual = -(q - s) * (q + s) := by
      unfold residual
      rw [← hsSq]
      ring
    rw [halg, abs_mul, abs_neg, abs_of_pos hsumPos]
  have hboundXRaw : abs residual < x := by
    have hmul : abs residual ≤ (u * s) * ((9 / 4 : ℝ) * s) := by
      rw [hresAbs]
      exact mul_le_mul hrelative hsumLe hsumPos.le (mul_nonneg huNonneg hspos.le)
    have huSq : u * s ^ 2 ≤ (1 / 4 : ℝ) * s ^ 2 :=
      mul_le_mul_of_nonneg_right huQuarter (sq_nonneg s)
    rw [← hsSq]
    calc
      abs residual ≤ (u * s) * ((9 / 4 : ℝ) * s) := hmul
      _ = (9 / 4 : ℝ) * (u * s ^ 2) := by ring
      _ ≤ (9 / 4 : ℝ) * ((1 / 4 : ℝ) * s ^ 2) :=
        mul_le_mul_of_nonneg_left huSq (by norm_num)
      _ < s ^ 2 := by nlinarith [sq_pos_of_pos hspos]
  have hboundX : abs residual < neuralBpow β (prec + fx.exponent) := by
    have hxUpper := abs_lt_neuralBpow_magnitude β x hx.ne'
    have hfxexp : fx.exponent = neuralCexp β (FLXExp prec) x := by
      rw [hfx]
      exact cfx
    calc
      abs residual < x := hboundXRaw
      _ = abs x := (abs_of_pos hx).symm
      _ < neuralBpow β (prec + fx.exponent) := by
        rw [hfxexp]
        simpa [neuralCexp, FLXExp] using hxUpper
  have hcexpSQ : neuralCexp β (FLXExp prec) s ≤
      neuralCexp β (FLXExp prec) q := by
    simp only [neuralCexp, FLXExp]
    exact sub_le_sub_right
      (neuralMagnitude_le_round_FLX (β := β) prec hp0 neuralNearestEven hqpos.ne') prec
  have hsUpper := abs_lt_neuralBpow_magnitude β s hspos.ne'
  have hsumStrict : q + s < 2 * neuralBpow β (neuralMagnitude β s) := by
    have hqUpper := hqBounds.2
    simpa [q, two_mul, abs_of_pos hspos] using add_lt_add_of_le_of_lt hqUpper hsUpper
  have hresProduct : abs residual <
      neuralBpow β (neuralCexp β (FLXExp prec) s + neuralMagnitude β s) := by
    rw [hresAbs]
    calc
      abs (q - s) * (q + s) ≤
          (neuralBpow β (neuralCexp β (FLXExp prec) s) / 2) * (q + s) :=
        mul_le_mul_of_nonneg_right herrorBpow hsumPos.le
      _ <
          (neuralBpow β (neuralCexp β (FLXExp prec) s) / 2) *
            (2 * neuralBpow β (neuralMagnitude β s)) :=
        mul_lt_mul_of_pos_left hsumStrict
          (div_pos (neuralBpow.pos β _) (by norm_num))
      _ = neuralBpow β
          (neuralCexp β (FLXExp prec) s + neuralMagnitude β s) := by
        rw [neuralBpow.add_exp]
        ring
  have hboundSquare : abs residual < neuralBpow β (prec + negSquare.exponent) := by
    apply hresProduct.trans_le
    apply (neuralBpow_le_neuralBpow_iff β _ _).2
    have hfqexp : fq.exponent = neuralCexp β (FLXExp prec) q := by
      rw [hfq]
      exact cfq
    simp [negSquare, hfqexp, neuralCexp, FLXExp] at hcexpSQ ⊢
    linarith
  change neuralGenericFormat β (FLXExp prec) residual
  rw [hresidualAdd]
  apply neural_generic_format_FLX_add_of_repr_bounds prec hp0 fx negSquare x (-(q ^ 2))
  · exact hfx
  · exact hnegSquare
  · simpa [hresidualAdd] using hboundX
  · simpa [hresidualAdd] using hboundSquare

end TorchLean.Floats
