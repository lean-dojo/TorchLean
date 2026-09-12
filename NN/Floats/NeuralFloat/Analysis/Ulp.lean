/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Format.Generic

/-!
# Unit in the Last Place

Format-generic properties of `neuralUlp`, including the lower-exponent witness used to define
`ulp 0`.  The definitions and hypotheses follow Flocq's `Core/Ulp.v`.
-/

@[expose] public section

namespace TorchLean.Floats

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]

/-- Two negligible-exponent witnesses select the same format exponent. -/
theorem neuralNegligibleExp_value_unique {n m : ℤ}
    (hn : IsNeuralNegligibleExp fexp n) (hm : IsNeuralNegligibleExp fexp m) :
    fexp n = fexp m := by
  rcases le_total m (fexp n) with hmn | hnm
  · exact (((NeuralValidExp.flocq_valid (fexp := fexp) n).2 hn).2 m hmn).symm
  · have hnfm : n ≤ fexp m := hn.trans (hnm.trans hm)
    exact ((NeuralValidExp.flocq_valid (fexp := fexp) m).2 hm).2 n hnfm

omit [NeuralValidExp fexp] in
/-- Absence of a negligible exponent means `fexp n < n` at every exponent. -/
theorem neuralNegligibleExp_none_iff_forall_lt :
    neuralNegligibleExp fexp = none ↔ ∀ n, fexp n < n := by
  rw [neuralNegligibleExp_eq_none_iff]
  constructor
  · intro h n
    exact lt_of_not_ge (fun hn => h ⟨n, hn⟩)
  · intro h
    rintro ⟨n, hn⟩
    exact (not_le_of_gt (h n)) hn

/-- ULP is invariant under negation. -/
@[simp] theorem neuralUlp_neg (x : ℝ) : neuralUlp β fexp (-x) = neuralUlp β fexp x := by
  by_cases hx : x = 0
  · subst x
    simp
  · simp [neuralUlp, hx, neuralCexp]

/-- ULP is invariant under absolute value. -/
@[simp] theorem neuralUlp_abs (x : ℝ) : neuralUlp β fexp (abs x) = neuralUlp β fexp x := by
  rcases le_total 0 x with hx | hx
  · simp [abs_of_nonneg hx]
  · rw [abs_of_nonpos hx, neuralUlp_neg]

/-- The ULP of a radix power is selected at the next magnitude. -/
theorem neuralUlp_bpow (e : ℤ) :
    neuralUlp β fexp (neuralBpow β e) = neuralBpow β (fexp (e + 1)) := by
  rw [neuralUlp.of_ne_zero β fexp _ (neuralBpow.ne_zero β e)]
  simp [neuralCexp]

/-- Exponent functions for which ULP values themselves remain representable. -/
class NeuralExpNotFlushToZero (fexp : ℤ → ℤ) : Prop where
  ulpExponent : ∀ e, fexp (fexp e + 1) ≤ fexp e

/-- The zero ULP is representable, including the FLX case where it equals zero. -/
theorem neural_generic_format_ulp_zero : neuralGenericFormat β fexp (neuralUlp β fexp 0) := by
  rw [neuralUlp.zero]
  cases hopt : neuralNegligibleExp fexp with
  | none =>
      simp
  | some n =>
      have hn := neuralNegligibleExp_spec hopt
      apply neural_generic_format_bpow
      exact ((NeuralValidExp.flocq_valid (fexp := fexp) n).2 hn).1

/-- Under the non-flush-to-zero condition, every ULP is representable. -/
theorem neural_generic_format_ulp [NeuralExpNotFlushToZero fexp] (x : ℝ) :
    neuralGenericFormat β fexp (neuralUlp β fexp x) := by
  by_cases hx : x = 0
  · subst x
    exact neural_generic_format_ulp_zero
  · rw [neuralUlp.of_ne_zero β fexp x hx]
    exact neural_generic_format_bpow _
      (NeuralExpNotFlushToZero.ulpExponent (fexp := fexp) (neuralMagnitude β x))

/-- For a nonrepresentable input, directed-up and directed-down rounding differ by one ULP. -/
theorem neuralRound_ceil_eq_floor_add_ulp {x : ℝ} (hx : ¬neuralGenericFormat β fexp x) :
    neuralRound (β := β) (fexp := fexp) neuralCeilRound x =
      neuralRound (β := β) (fexp := fexp) neuralFloorRound x + neuralUlp β fexp x := by
  have hx0 : x ≠ 0 := by
    intro hzero
    subst x
    exact hx neural_generic_format_zero
  have hsnot : neuralScaledMantissa β fexp x ∉ Set.range ((↑·) : ℤ → ℝ) := by
    rintro ⟨n, hn⟩
    apply hx
    apply neural_generic_format_of_scaled_mantissa_int (n := n)
    exact hn.symm
  have hceil : neuralCeilRound (neuralScaledMantissa β fexp x) =
      neuralFloorRound (neuralScaledMantissa β fexp x) + 1 := by
    exact (Int.ceil_eq_floor_add_one_iff_notMem _).2 hsnot
  rw [neuralUlp.of_ne_zero β fexp x hx0]
  unfold neuralRound neuralToReal
  rw [hceil]
  push_cast
  ring

/-- One ULP is no larger than the absolute value of a nonzero representable number. -/
theorem neuralUlp_le_abs_of_generic {x : ℝ} (hx0 : x ≠ 0)
    (hx : neuralGenericFormat β fexp x) : neuralUlp β fexp x ≤ abs x := by
  obtain ⟨n, hn⟩ := neural_scaled_mantissa_int_of_generic
    (β := β) (fexp := fexp) x hx
  have hn0 : n ≠ 0 := by
    intro hnzero
    have hs0 : neuralScaledMantissa β fexp x = 0 := by simpa [hnzero] using hn
    have hrepr := neural_scaled_mantissa_mul_bpow (β := β) (fexp := fexp) x
    rw [hs0, zero_mul] at hrepr
    exact hx0 hrepr.symm
  have habsn : (1 : ℝ) ≤ abs (n : ℝ) := by
    exact_mod_cast (Int.one_le_abs hn0)
  have hb := neuralBpow.nonneg β (neuralCexp β fexp x)
  rw [neuralUlp.of_ne_zero β fexp x hx0]
  have hrepr := neural_scaled_mantissa_mul_bpow (β := β) (fexp := fexp) x
  rw [hn] at hrepr
  calc
    neuralBpow β (neuralCexp β fexp x) =
        1 * neuralBpow β (neuralCexp β fexp x) := by ring
    _ ≤ abs (n : ℝ) * neuralBpow β (neuralCexp β fexp x) :=
      mul_le_mul_of_nonneg_right habsn hb
    _ = abs ((n : ℝ) * neuralBpow β (neuralCexp β fexp x)) := by
      rw [abs_mul, abs_of_nonneg hb]
    _ = abs x := by rw [hrepr]

/-- ULP is monotone on positive inputs when the exponent selector is monotone. -/
theorem neuralUlp_mono_pos [NeuralMonotoneExp fexp] {x y : ℝ}
    (hx : 0 < x) (hxy : x ≤ y) : neuralUlp β fexp x ≤ neuralUlp β fexp y := by
  have hy : 0 < y := hx.trans_le hxy
  have hmag : neuralMagnitude β x ≤ neuralMagnitude β y := by
    have hxLower := (neuralMagnitude_spec β x hx.ne').1
    have hyUpper := (neuralMagnitude_spec β y hy.ne').2
    have hpowers : neuralBpow β (neuralMagnitude β x - 1) <
        neuralBpow β (neuralMagnitude β y) := by
      exact hxLower.trans (by simpa [abs_of_pos hx, abs_of_pos hy] using hxy) |>.trans_lt hyUpper
    have hexp : neuralMagnitude β x - 1 < neuralMagnitude β y :=
      (neuralBpow_lt_neuralBpow_iff β _ _).mp hpowers
    linarith
  rw [neuralUlp.of_ne_zero β fexp x hx.ne', neuralUlp.of_ne_zero β fexp y hy.ne']
  exact (neuralBpow_le_neuralBpow_iff β _ _).2
    (NeuralMonotoneExp.monotone _ _ hmag)

end TorchLean.Floats
