/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Format.Generic
public import NN.Floats.NeuralFloat.Rounding.Core
public import NN.Floats.NeuralFloat.Rounding.Properties

/-!
# Rounding Into A Generic Format

This file develops the converse of `neural_round_preserves_generic`: rounding an arbitrary real
produces a value in the selected generic format. The proof follows the small/large magnitude split
used by Flocq rather than assuming the conclusion as part of the rounding definition.
-/

@[expose] public section

namespace TorchLean.Floats

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]

/-- The `Valid_exp` consequence used when the input magnitude is below the selected exponent. -/
theorem neuralValidExp_small {e : ℤ} (h : e ≤ fexp e) : fexp (fexp e + 1) ≤ fexp e :=
  ((NeuralValidExp.flocq_valid (fexp := fexp) e).2 h).1

/-- The `Valid_exp` consequence used when the selected exponent is below the input magnitude. -/
theorem neuralValidExp_large {e : ℤ} (h : fexp e < e) : fexp (e + 1) ≤ e :=
  (NeuralValidExp.flocq_valid (fexp := fexp) e).1 h

/-- Positive inputs in the small-magnitude regime round either to zero or to one radix power. -/
theorem neural_round_pos_small_cases (rnd : ℝ → ℤ) [NeuralValidRnd rnd]
    (x : ℝ) (hx : 0 < x) (hsmall : neuralMagnitude β x ≤ fexp (neuralMagnitude β x)) :
    neuralRound (β := β) (fexp := fexp) rnd x = 0 ∨
      neuralRound (β := β) (fexp := fexp) rnd x =
        neuralBpow β (fexp (neuralMagnitude β x)) := by
  let ex := neuralMagnitude β x
  let e := fexp ex
  let s := neuralScaledMantissa β fexp x
  have hxne : x ≠ 0 := ne_of_gt hx
  have hbpowPos : 0 < neuralBpow β e := neuralBpow.pos β e
  have hcexp : neuralCexp β fexp x = e := rfl
  have hsdiv : s = x / neuralBpow β e := by
    simpa [s, hcexp] using neuralScaledMantissa_eq_div (β := β) (fexp := fexp) x
  have hspos : 0 < s := by rw [hsdiv]; positivity
  have hxUpper : x < neuralBpow β ex := by
    simpa [ex, abs_of_pos hx] using (neuralMagnitude_spec β x hxne).2
  have hbpowMono : neuralBpow β ex ≤ neuralBpow β e := by
    exact (neuralBpow_le_neuralBpow_iff β ex e).2 hsmall
  have hslt : s < 1 := by
    rw [hsdiv, div_lt_one hbpowPos]
    exact hxUpper.trans_le hbpowMono
  have hfloor : neuralFloorRound s = 0 := by
    unfold neuralFloorRound
    exact Int.floor_eq_iff.mpr ⟨by exact_mod_cast hspos.le, by simpa using hslt⟩
  have hceil : neuralCeilRound s = 1 := by
    unfold neuralCeilRound
    exact Int.ceil_eq_iff.mpr ⟨by simpa using hspos, by exact_mod_cast hslt.le⟩
  have hm0 : 0 ≤ rnd s := by
    simpa [hfloor] using neural_floor_le_valid_round (rnd := rnd) s
  have hm1 : rnd s ≤ 1 := by
    simpa [hceil] using neural_valid_round_le_ceil (rnd := rnd) s
  have hm : rnd s = 0 ∨ rnd s = 1 := by
    by_cases hzero : rnd s = 0
    · exact Or.inl hzero
    · right
      have hpos : 0 < rnd s := lt_of_le_of_ne hm0 (Ne.symm hzero)
      exact le_antisymm hm1 (by simpa using Int.add_one_le_iff.mpr hpos)
  rcases hm with hm | hm
  · have hy : neuralRound (β := β) (fexp := fexp) rnd x = 0 := by
      simp [neuralRound, neuralToReal, s, hm]
    exact Or.inl hy
  · have hy : neuralRound (β := β) (fexp := fexp) rnd x = neuralBpow β e := by
      simp [neuralRound, neuralToReal, s, hm, hcexp]
    exact Or.inr (by simpa [e, ex] using hy)

/-- Rounding in the small-magnitude regime produces a generic-format value. -/
theorem neural_generic_format_round_pos_small (rnd : ℝ → ℤ) [NeuralValidRnd rnd]
    (x : ℝ) (hx : 0 < x) (hsmall : neuralMagnitude β x ≤ fexp (neuralMagnitude β x)) :
    neuralGenericFormat β fexp (neuralRound (β := β) (fexp := fexp) rnd x) := by
  rcases neural_round_pos_small_cases (β := β) (fexp := fexp) rnd x hx hsmall with hy | hy
  · rw [hy]
    exact neural_generic_format_zero
  · rw [hy]
    exact neural_generic_format_bpow _ (neuralValidExp_small (fexp := fexp) hsmall)

/--
In the large-magnitude regime, rounding stays inside the input magnitude bin and produces a
generic-format value.
-/
theorem neural_round_pos_large_bounds_and_generic (rnd : ℝ → ℤ) [NeuralValidRnd rnd]
    (x : ℝ) (hx : 0 < x) (hlarge : fexp (neuralMagnitude β x) < neuralMagnitude β x) :
    (neuralBpow β (neuralMagnitude β x - 1) ≤
        neuralRound (β := β) (fexp := fexp) rnd x ∧
      neuralRound (β := β) (fexp := fexp) rnd x ≤
        neuralBpow β (neuralMagnitude β x)) ∧
      neuralGenericFormat β fexp (neuralRound (β := β) (fexp := fexp) rnd x) := by
  let ex := neuralMagnitude β x
  let e := fexp ex
  let s := neuralScaledMantissa β fexp x
  let y := neuralRound (β := β) (fexp := fexp) rnd x
  have hxne : x ≠ 0 := ne_of_gt hx
  have hbpowPos : 0 < neuralBpow β e := neuralBpow.pos β e
  have hcexp : neuralCexp β fexp x = e := rfl
  have helt : e < ex := by simpa [e, ex] using hlarge
  have hsdiv : s = x / neuralBpow β e := by
    simpa [s, hcexp] using neuralScaledMantissa_eq_div (β := β) (fexp := fexp) x
  have hxLower : neuralBpow β (ex - 1) ≤ x := by
    simpa [ex, abs_of_pos hx] using (neuralMagnitude_spec β x hxne).1
  have hxUpper : x < neuralBpow β ex := by
    simpa [ex, abs_of_pos hx] using (neuralMagnitude_spec β x hxne).2
  have hdLower : 0 ≤ ex - 1 - e := by linarith
  have hdUpper : 0 ≤ ex - e := by linarith
  obtain ⟨nLower, hnLower⟩ := neuralBpow_eq_natCast_of_nonneg β (ex - 1 - e) hdLower
  obtain ⟨nUpper, hnUpper⟩ := neuralBpow_eq_natCast_of_nonneg β (ex - e) hdUpper
  have hsLower : neuralBpow β (ex - 1 - e) ≤ s := by
    calc
      neuralBpow β (ex - 1 - e) = neuralBpow β (ex - 1) / neuralBpow β e := by
        rw [neuralBpow.sub_exp]
      _ ≤ x / neuralBpow β e := div_le_div_of_nonneg_right hxLower hbpowPos.le
      _ = s := hsdiv.symm
  have hsUpper : s < neuralBpow β (ex - e) := by
    calc
      s = x / neuralBpow β e := hsdiv
      _ < neuralBpow β ex / neuralBpow β e :=
        (div_lt_div_iff_of_pos_right hbpowPos).2 hxUpper
      _ = neuralBpow β (ex - e) := (neuralBpow.sub_exp β ex e).symm
  have hmLower : (nLower : ℤ) ≤ rnd s := by
    have hsLower' : (nLower : ℝ) ≤ s := by simpa [hnLower] using hsLower
    have hmono := NeuralValidRnd.monotone (rnd := rnd) (nLower : ℝ) s hsLower'
    have hid : rnd (nLower : ℝ) = (nLower : ℤ) := by
      simpa using NeuralValidRnd.id (rnd := rnd) (nLower : ℤ)
    rw [hid] at hmono
    exact hmono
  have hmUpper : rnd s ≤ (nUpper : ℤ) := by
    have hsUpper' : s ≤ (nUpper : ℝ) := by exact (by simpa [hnUpper] using hsUpper.le)
    exact (neural_valid_round_le_ceil (rnd := rnd) s).trans (Int.ceil_le.mpr hsUpper')
  have hnLowerOne : 1 ≤ nLower := by
    have hbOne : (1 : ℝ) ≤ neuralBpow β (ex - 1 - e) := by
      change (1 : ℝ) ≤ β.toReal ^ (ex - 1 - e)
      exact one_le_zpow₀ (le_of_lt (NeuralRadix.gt_one β)) hdLower
    exact_mod_cast (show (1 : ℝ) ≤ nLower by simpa [hnLower] using hbOne)
  have hmPos : 0 < rnd s := by linarith
  have hy : y = (rnd s : ℝ) * neuralBpow β e := by
    rfl
  have hyPos : 0 < y := by rw [hy]; positivity
  have hyLower : neuralBpow β (ex - 1) ≤ y := by
    have hmLowerR : (nLower : ℝ) ≤ (rnd s : ℝ) := by exact_mod_cast hmLower
    calc
      neuralBpow β (ex - 1) =
          neuralBpow β (ex - 1 - e) * neuralBpow β e := by
        rw [← neuralBpow.add_exp]
        congr 1
        linarith
      _ = (nLower : ℝ) * neuralBpow β e := by rw [hnLower]
      _ ≤ (rnd s : ℝ) * neuralBpow β e :=
        mul_le_mul_of_nonneg_right hmLowerR hbpowPos.le
      _ = y := hy.symm
  have hyUpper : y ≤ neuralBpow β ex := by
    have hmUpperR : (rnd s : ℝ) ≤ (nUpper : ℝ) := by exact_mod_cast hmUpper
    calc
      y = (rnd s : ℝ) * neuralBpow β e := hy
      _ ≤ (nUpper : ℝ) * neuralBpow β e :=
        mul_le_mul_of_nonneg_right hmUpperR hbpowPos.le
      _ = neuralBpow β (ex - e) * neuralBpow β e := by rw [hnUpper]
      _ = neuralBpow β ex := by
        rw [← neuralBpow.add_exp]
        congr 1
        linarith
  refine ⟨⟨by simpa [y, ex] using hyLower, by simpa [y, ex] using hyUpper⟩, ?_⟩
  rcases hyUpper.eq_or_lt with hyEq | hyLt
  · change neuralGenericFormat β fexp y
    rw [hyEq]
    exact neural_generic_format_bpow (e := ex) (neuralValidExp_large (fexp := fexp) hlarge)
  · have hmagY : neuralMagnitude β y = ex :=
      neuralMagnitude_eq_of_bpow_bounds β y ex (ne_of_gt hyPos)
        (by simpa [abs_of_pos hyPos] using hyLower)
        (by simpa [abs_of_pos hyPos] using hyLt)
    change neuralGenericFormat β fexp y
    apply neural_generic_format_of_scaled_mantissa_int (n := rnd s)
    rw [neuralScaledMantissa_eq_div]
    have hcexpY : neuralCexp β fexp y = e := by simp [neuralCexp, hmagY, e]
    rw [hcexpY, hy]
    exact mul_div_cancel_right₀ (rnd s : ℝ) (neuralBpow.ne_zero β e)

/-- Rounding in the large-magnitude regime produces a generic-format value. -/
theorem neural_generic_format_round_pos_large (rnd : ℝ → ℤ) [NeuralValidRnd rnd]
    (x : ℝ) (hx : 0 < x) (hlarge : fexp (neuralMagnitude β x) < neuralMagnitude β x) :
    neuralGenericFormat β fexp (neuralRound (β := β) (fexp := fexp) rnd x) :=
  (neural_round_pos_large_bounds_and_generic
    (β := β) (fexp := fexp) rnd x hx hlarge).2

/-- Every positive real rounds into the selected generic format. -/
theorem neural_generic_format_round_pos (rnd : ℝ → ℤ) [NeuralValidRnd rnd]
    (x : ℝ) (hx : 0 < x) :
    neuralGenericFormat β fexp (neuralRound (β := β) (fexp := fexp) rnd x) := by
  by_cases hsmall : neuralMagnitude β x ≤ fexp (neuralMagnitude β x)
  · exact neural_generic_format_round_pos_small rnd x hx hsmall
  · exact neural_generic_format_round_pos_large rnd x hx (lt_of_not_ge hsmall)

/-- Conjugate a rounding rule by negation. -/
def neuralNegRound (rnd : ℝ → ℤ) : ℝ → ℤ := fun x => -rnd (-x)

/-- Negation-conjugating a valid rounding rule leaves it valid, which is how the "down" and "up"
directions are obtained from one another rather than proved twice. -/
instance neuralNegRoundValid (rnd : ℝ → ℤ) [NeuralValidRnd rnd] :
    NeuralValidRnd (neuralNegRound rnd) where
  monotone := by
    intro x y hxy
    exact neg_le_neg (NeuralValidRnd.monotone (rnd := rnd) (-y) (-x) (neg_le_neg hxy))
  id := by
    intro n
    have h : rnd (-(n : ℝ)) = -n := by
      simpa using NeuralValidRnd.id (rnd := rnd) (-n)
    simp [neuralNegRound, h]

/-- Rounding a negated input is negated rounding with the conjugate integer rule. -/
theorem neuralRound_neg (rnd : ℝ → ℤ) [NeuralValidRnd rnd] (x : ℝ) :
    neuralRound (β := β) (fexp := fexp) rnd (-x) =
      -neuralRound (β := β) (fexp := fexp) (neuralNegRound rnd) x := by
  simp [neuralRound, neuralToReal, neuralNegRound]

/-- Rounding any real produces a value in the selected generic format. -/
theorem neural_generic_format_round (rnd : ℝ → ℤ) [NeuralValidRnd rnd] (x : ℝ) :
    neuralGenericFormat β fexp (neuralRound (β := β) (fexp := fexp) rnd x) := by
  rcases lt_trichotomy x 0 with hx | hx | hx
  · have hpos : 0 < -x := neg_pos.mpr hx
    have hgeneric :=
      neural_generic_format_round_pos (β := β) (fexp := fexp) (neuralNegRound rnd) (-x) hpos
    have hround := neuralRound_neg (β := β) (fexp := fexp) rnd (-x)
    simp only [neg_neg] at hround
    rw [hround]
    exact neural_generic_format_neg _ hgeneric
  · subst x
    have hr0 : rnd 0 = 0 := by simpa using NeuralValidRnd.id (rnd := rnd) (0 : ℤ)
    have hround : neuralRound (β := β) (fexp := fexp) rnd 0 = 0 := by
      simp [neuralRound, neuralScaledMantissa, neuralToReal, hr0]
    rw [hround]
    exact neural_generic_format_zero
  · exact neural_generic_format_round_pos rnd x hx

end TorchLean.Floats
