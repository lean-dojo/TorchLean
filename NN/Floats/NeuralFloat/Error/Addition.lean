/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Error.Exactness
public import NN.Floats.NeuralFloat.Format.Theorems
public import NN.Floats.NeuralFloat.Rounding.Order

/-!
# Exactness of Addition Errors

For nearest rounding, the error of adding two representable values is itself representable.  The
proof aligns both operands on the smaller canonical grid, represents the rounded result on that
same grid, and uses nearestness to control the canonical exponent of the error.
-/

@[expose] public section

namespace TorchLean.Floats

variable {β : NeuralRadix} {fexp : ℤ → ℤ}
  [NeuralValidExp fexp] [NeuralMonotoneExp fexp]

/-- Addition-error exactness when the first operand has the smaller canonical exponent. -/
private theorem neural_add_round_error_generic_aux {x y : ℝ}
    (hxy : neuralCexp β fexp x ≤ neuralCexp β fexp y)
    (hx : neuralGenericFormat β fexp x) (hy : neuralGenericFormat β fexp y) :
    neuralGenericFormat β fexp
      (neuralRound (β := β) (fexp := fexp) neuralNearestEven (x + y) - (x + y)) := by
  let ex := neuralCexp β fexp x
  let ey := neuralCexp β fexp y
  obtain ⟨mx, hmx⟩ :=
    (neural_generic_format_iff_scaled_mantissa_int (β := β) (fexp := fexp) x).mp hx
  obtain ⟨my, hmy⟩ :=
    (neural_generic_format_iff_scaled_mantissa_int (β := β) (fexp := fexp) y).mp hy
  have hxrepr : x = (mx : ℝ) * neuralBpow β ex := by
    have h := neural_scaled_mantissa_mul_bpow (β := β) (fexp := fexp) x
    rw [hmx] at h
    exact h.symm
  have hyrepr : y = (my : ℝ) * neuralBpow β ey := by
    have h := neural_scaled_mantissa_mul_bpow (β := β) (fexp := fexp) y
    rw [hmy] at h
    exact h.symm
  obtain ⟨scale, hscale⟩ := neuralBpow_eq_natCast_of_nonneg β (ey - ex)
    (sub_nonneg.mpr hxy)
  let commonMantissa : ℤ := mx + my * Int.ofNat scale
  let common : NeuralFloat β := { mantissa := commonMantissa, exponent := ex }
  have hscaleCast : (((Int.ofNat scale : ℤ) : ℝ)) = (scale : ℝ) := by norm_num
  have hsum : x + y = neuralToReal common := by
    rw [hxrepr, hyrepr]
    unfold common commonMantissa neuralToReal
    rw [show ey = (ey - ex) + ex by ring, neuralBpow.add_exp, hscale]
    simp only [Int.cast_add, Int.cast_mul, hscaleCast]
    ring
  obtain ⟨mr, hrounded⟩ :=
    neuralRound_toReal_exists_same_exponent
      (β := β) (fexp := fexp) neuralNearestEven common
  let errorMantissa : ℤ := mr - commonMantissa
  let errorFloat : NeuralFloat β := { mantissa := errorMantissa, exponent := ex }
  let err := neuralRound (β := β) (fexp := fexp) neuralNearestEven (x + y) - (x + y)
  have herr : err = neuralToReal errorFloat := by
    unfold err errorFloat errorMantissa neuralToReal
    rw [hsum, hrounded]
    unfold common
    simp only [neuralToReal]
    push_cast
    ring
  by_cases herr0 : err = 0
  · change neuralGenericFormat β fexp err
    rw [herr0]
    exact neural_generic_format_zero
  have hnear := (neuralRound_nearestEven_point
    (β := β) (fexp := fexp) (x + y)).2 y hy
  have herrLe : abs err ≤ abs x := by
    simpa [err] using hnear
  have hcexp : neuralCexp β fexp err ≤ ex := by
    exact neuralCexp_mono_abs β herr0 herrLe
  apply neural_generic_format_of_toReal_of_cexp_le errorFloat err herr
  exact hcexp

/-- The nearest-rounded addition error of two representable values is representable. -/
theorem neural_add_round_error_generic {x y : ℝ}
    (hx : neuralGenericFormat β fexp x) (hy : neuralGenericFormat β fexp y) :
    neuralGenericFormat β fexp
      (neuralRound (β := β) (fexp := fexp) neuralNearestEven (x + y) - (x + y)) := by
  rcases le_total (neuralCexp β fexp x) (neuralCexp β fexp y) with hxy | hyx
  · exact neural_add_round_error_generic_aux hxy hx hy
  · simpa [add_comm] using
      (neural_add_round_error_generic_aux (β := β) (fexp := fexp) hyx hy hx)

/-- Every FLT value lies on its minimum-exponent FIX grid. -/
theorem neural_generic_format_FLT_to_FIX (emin prec : ℤ) (hprec : 0 < prec) {x : ℝ}
    (hx : @neuralGenericFormat β (FLTExp emin prec) (fltValidExp emin prec hprec) x) :
    neuralGenericFormat β (FIXExp emin) x := by
  let _ : NeuralValidExp (FLTExp emin prec) := fltValidExp emin prec hprec
  let _ : NeuralValidExp (FIXExp emin) := fixValidExp emin
  obtain ⟨m, hm⟩ :=
    (neural_generic_format_iff_scaled_mantissa_int
      (β := β) (fexp := FLTExp emin prec) x).mp hx
  let e := neuralCexp β (FLTExp emin prec) x
  have herepr : x = (m : ℝ) * neuralBpow β e := by
    have h := neural_scaled_mantissa_mul_bpow (β := β) (fexp := FLTExp emin prec) x
    rw [hm] at h
    exact h.symm
  have hemin : emin ≤ e := by simp [e, neuralCexp, FLTExp]
  obtain ⟨scale, hscale⟩ := neuralBpow_eq_natCast_of_nonneg β (e - emin)
    (sub_nonneg.mpr hemin)
  let f : NeuralFloat β := { mantissa := m * Int.ofNat scale, exponent := emin }
  have hxf : x = neuralToReal f := by
    rw [herepr, show e = (e - emin) + emin by ring, neuralBpow.add_exp, hscale]
    unfold f neuralToReal
    have hcast : (((Int.ofNat scale : ℤ) : ℝ)) = (scale : ℝ) := by norm_num
    rw [Int.cast_mul, hcast]
    ring
  apply neural_generic_format_of_toReal_of_cexp_le f x hxf
  simp [f, neuralCexp, FIXExp]

/-- A bounded FIX value is FLT-representable, including the radix-power boundary. -/
theorem neural_generic_format_FIX_to_FLT_of_abs_le (emin prec : ℤ) (hprec : 0 < prec)
    {x : ℝ} (hx : neuralGenericFormat β (FIXExp emin) x)
    (hbound : abs x ≤ neuralBpow β (prec + emin)) :
    @neuralGenericFormat β (FLTExp emin prec) (fltValidExp emin prec hprec) x := by
  let _ : NeuralValidExp (FIXExp emin) := fixValidExp emin
  let _ : NeuralValidExp (FLTExp emin prec) := fltValidExp emin prec hprec
  by_cases hx0 : x = 0
  · subst x
    exact neural_generic_format_zero
  rcases hbound.eq_or_lt with heq | hlt
  · have habsFmt : neuralGenericFormat β (FLTExp emin prec) (abs x) := by
      rw [heq]
      apply neural_generic_format_bpow
      unfold FLTExp
      rw [max_eq_left (by linarith)]
      linarith
    exact (neural_generic_format_abs_iff (β := β) (fexp := FLTExp emin prec) x).mp habsFmt
  · obtain ⟨f, hxf, hfe⟩ :=
      (generic_format_FIX_iff (β := β) emin x).mp hx
    apply neural_generic_format_of_toReal_of_cexp_le f x hxf
    have hmag : neuralMagnitude β x ≤ prec + emin :=
      neuralMagnitude_le_of_abs_lt_bpow β x (prec + emin) hx0 hlt
    rw [hfe]
    simp [neuralCexp, FLTExp]
    linarith

/-- A sufficiently small sum of two FLT values is exactly FLT-representable. -/
theorem neural_generic_format_FLT_add_small (emin prec : ℤ) (hprec : 0 < prec)
    {x y : ℝ}
    (hx : @neuralGenericFormat β (FLTExp emin prec) (fltValidExp emin prec hprec) x)
    (hy : @neuralGenericFormat β (FLTExp emin prec) (fltValidExp emin prec hprec) y)
    (hbound : abs (x + y) ≤ neuralBpow β (prec + emin)) :
    @neuralGenericFormat β (FLTExp emin prec) (fltValidExp emin prec hprec) (x + y) := by
  let _ : NeuralValidExp (FIXExp emin) := fixValidExp emin
  have hxFix := neural_generic_format_FLT_to_FIX (β := β) emin prec hprec hx
  have hyFix := neural_generic_format_FLT_to_FIX (β := β) emin prec hprec hy
  obtain ⟨fx, hfx, ex⟩ := (generic_format_FIX_iff (β := β) emin x).mp hxFix
  obtain ⟨fy, hfy, ey⟩ := (generic_format_FIX_iff (β := β) emin y).mp hyFix
  let fsum : NeuralFloat β :=
    { mantissa := fx.mantissa + fy.mantissa, exponent := emin }
  have hsum : x + y = neuralToReal fsum := by
    rw [hfx, hfy]
    unfold fsum neuralToReal
    rw [ex, ey]
    push_cast
    ring
  have hsumFix : neuralGenericFormat β (FIXExp emin) (x + y) := by
    apply (generic_format_FIX_iff (β := β) emin (x + y)).2
    exact ⟨fsum, hsum, rfl⟩
  exact neural_generic_format_FIX_to_FLT_of_abs_le
    (β := β) emin prec hprec hsumFix hbound

end TorchLean.Floats
