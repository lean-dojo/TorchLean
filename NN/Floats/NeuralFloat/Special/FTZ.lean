/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Format.Theorems
public import NN.Floats.NeuralFloat.Analysis.Ulp

/-!
# Abrupt Underflow

`FTZExp emin prec` is the Flocq abrupt-underflow exponent selector.  Below the smallest normal
magnitude it selects the normal threshold `emin + prec - 1`; above that threshold it agrees with
the unbounded precision-`prec` selector.  A matching rounding mode can therefore flush values below
the normal range directly to zero.
-/

@[expose] public section

namespace TorchLean.Floats

variable {β : NeuralRadix}

/-- Exponent selection for a precision-`prec` format with abrupt underflow at exponent `emin`. -/
def FTZExp (emin prec : ℤ) (e : ℤ) : ℤ :=
  if e - prec < emin then emin + prec - 1 else e - prec

/-- The abrupt-underflow threshold is the smallest normal magnitude exponent. -/
def FTZThreshold (emin prec : ℤ) : ℤ := emin + prec - 1

/-- Below the threshold, `FTZExp` selects the threshold itself. -/
theorem FTZExp_eq_threshold {emin prec e : ℤ} (h : e ≤ FTZThreshold emin prec) :
    FTZExp emin prec e = FTZThreshold emin prec := by
  rw [FTZExp, if_pos]
  · rfl
  · simp [FTZThreshold] at h ⊢
    linarith

/-- Above the threshold, `FTZExp` agrees with `e - prec`. -/
theorem FTZExp_eq_sub {emin prec e : ℤ} (h : FTZThreshold emin prec < e) :
    FTZExp emin prec e = e - prec := by
  rw [FTZExp, if_neg]
  simp [FTZThreshold] at h ⊢
  linarith

/-- Positive-precision abrupt-underflow exponent selection satisfies Flocq validity. -/
abbrev ftzValidExp (emin prec : ℤ) (hprec : 0 < prec) :
    NeuralValidExp (FTZExp emin prec) where
  flocq_valid := by
    intro k
    constructor
    · intro hk
      have hlarge : FTZThreshold emin prec < k := by
        by_contra hnot
        have hsmall : k ≤ FTZThreshold emin prec := le_of_not_gt hnot
        rw [FTZExp_eq_threshold hsmall] at hk
        exact (not_lt_of_ge hsmall) hk
      rw [FTZExp_eq_sub hlarge] at hk
      by_cases hnext : k + 1 ≤ FTZThreshold emin prec
      · rw [FTZExp_eq_threshold hnext]
        linarith
      · rw [FTZExp_eq_sub (lt_of_not_ge hnext)]
        linarith
    · intro hk
      have hsmall : k ≤ FTZThreshold emin prec := by
        by_contra hnot
        have hlarge := lt_of_not_ge hnot
        rw [FTZExp_eq_sub hlarge] at hk
        linarith
      have hkEq := FTZExp_eq_threshold hsmall
      rw [hkEq] at hk ⊢
      constructor
      · have hnext : FTZThreshold emin prec < FTZThreshold emin prec + 1 := by linarith
        rw [FTZExp_eq_sub hnext]
        simp [FTZThreshold]
        linarith
      · intro l hl
        rw [FTZExp_eq_threshold hl]

/-- `FTZExp emin prec` satisfies the exponent axioms exactly for positive precision. -/
theorem neuralValidExp_FTZ_iff (emin prec : ℤ) : NeuralValidExp (FTZExp emin prec) ↔ 0 < prec := by
  constructor
  · intro hvalid
    by_contra hprec
    have hnonpos : prec ≤ 0 := le_of_not_gt hprec
    have hlarge : FTZThreshold emin prec < emin + 1 := by
      simp [FTZThreshold]
      linarith
    have hexp : FTZExp emin prec (emin + 1) = emin + 1 - prec :=
      FTZExp_eq_sub hlarge
    have hk : emin + 1 ≤ FTZExp emin prec (emin + 1) := by
      rw [hexp]
      linarith
    have hnext := ((hvalid.flocq_valid (emin + 1)).2 hk).1
    have hnextLarge :
        FTZThreshold emin prec < FTZExp emin prec (emin + 1) + 1 := by
      rw [hexp]
      simp [FTZThreshold]
      linarith
    rw [FTZExp_eq_sub hnextLarge, hexp] at hnext
    linarith
  · exact ftzValidExp emin prec

namespace NeuralFormatPrecision

/-- The abrupt-underflow exponent selector associated with a checked precision. -/
def ftzExp (precision : NeuralFormatPrecision) (emin : ℤ) : ℤ → ℤ :=
  FTZExp emin precision.toInt

/-- A checked precision automatically discharges the abrupt-underflow validity obligation. -/
instance ftzExpValid (precision : NeuralFormatPrecision) (emin : ℤ) :
    NeuralValidExp (precision.ftzExp emin) :=
  ftzValidExp emin precision.toInt precision.toInt_pos

end NeuralFormatPrecision

/-- The threshold is a negligible exponent for the abrupt-underflow format. -/
theorem FTZThreshold_negligible (emin prec : ℤ) :
    IsNeuralNegligibleExp (FTZExp emin prec) (FTZThreshold emin prec) := by
  unfold IsNeuralNegligibleExp
  rw [FTZExp_eq_threshold le_rfl]

/-- The ULP at zero in the abrupt-underflow format is the smallest normal magnitude. -/
theorem neuralUlp_zero_FTZ (emin prec : ℤ) (hprec : 0 < prec) :
    @neuralUlp β (FTZExp emin prec) (ftzValidExp emin prec hprec) 0 =
      neuralBpow β (FTZThreshold emin prec) := by
  let _ : NeuralValidExp (FTZExp emin prec) := ftzValidExp emin prec hprec
  rw [neuralUlp.zero]
  cases hopt : neuralNegligibleExp (FTZExp emin prec) with
  | none =>
      have hnone := (neuralNegligibleExp_eq_none_iff (FTZExp emin prec)).mp hopt
      exact (hnone ⟨FTZThreshold emin prec, FTZThreshold_negligible emin prec⟩).elim
  | some n =>
      have hn := neuralNegligibleExp_spec hopt
      have heq := neuralNegligibleExp_value_unique hn
        (FTZThreshold_negligible emin prec)
      change neuralBpow β (FTZExp emin prec n) = neuralBpow β (FTZThreshold emin prec)
      rw [heq, FTZExp_eq_threshold le_rfl]

/-- Flush an inexact scaled mantissa to zero when its magnitude is below one. -/
noncomputable def neuralFTZRound (rnd : ℝ → ℤ) (x : ℝ) : ℤ :=
  if 1 ≤ abs x then rnd x else 0

/-- Flushing around `(-1,1)` preserves monotonicity and exact integer values. -/
instance neuralFTZRoundValid (rnd : ℝ → ℤ) [NeuralValidRnd rnd] :
    NeuralValidRnd (neuralFTZRound rnd) where
  id := by
    intro n
    by_cases hn0 : n = 0
    · subst n
      simp [neuralFTZRound]
    · have habs : (1 : ℝ) ≤ abs (n : ℝ) := by
        exact_mod_cast (Int.one_le_abs hn0)
      simp [neuralFTZRound, habs, NeuralValidRnd.id (rnd := rnd)]
  monotone := by
    intro x y hxy
    by_cases hx : 1 ≤ abs x
    · by_cases hy : 1 ≤ abs y
      · simp [neuralFTZRound, hx, hy, NeuralValidRnd.monotone (rnd := rnd) x y hxy]
      · have hyAbs : abs y < 1 := lt_of_not_ge hy
        have hx0 : x ≤ 0 := by
          by_contra hnot
          have hxpos : 0 < x := lt_of_not_ge hnot
          have hypos : 0 < y := hxpos.trans_le hxy
          rw [abs_of_pos hxpos] at hx
          rw [abs_of_pos hypos] at hyAbs
          linarith
        have hrx : rnd x ≤ 0 := by
          have hmono := NeuralValidRnd.monotone (rnd := rnd) x 0 hx0
          have hr0 : rnd 0 = 0 := by
            simpa using NeuralValidRnd.id (rnd := rnd) (0 : ℤ)
          rwa [hr0] at hmono
        simp [neuralFTZRound, hx, hy, hrx]
    · by_cases hy : 1 ≤ abs y
      · have hxAbs : abs x < 1 := lt_of_not_ge hx
        have hy0 : 0 ≤ y := by
          by_contra hnot
          have hyneg : y < 0 := lt_of_not_ge hnot
          have hxneg : x < 0 := lt_of_le_of_lt hxy hyneg
          rw [abs_of_neg hxneg] at hxAbs
          rw [abs_of_neg hyneg] at hy
          linarith
        have hry : 0 ≤ rnd y := by
          have hmono := NeuralValidRnd.monotone (rnd := rnd) 0 y hy0
          have hr0 : rnd 0 = 0 := by
            simpa using NeuralValidRnd.id (rnd := rnd) (0 : ℤ)
          rwa [hr0] at hmono
        simp [neuralFTZRound, hx, hy, hry]
      · simp [neuralFTZRound, hx, hy]

/-- Values below the smallest-normal threshold flush exactly to zero. -/
theorem neuralRound_FTZ_small (emin prec : ℤ) (hprec : 0 < prec)
    (rnd : ℝ → ℤ) [NeuralValidRnd rnd] (x : ℝ)
    (hsmall : abs x < neuralBpow β (FTZThreshold emin prec)) :
    @neuralRound β (FTZExp emin prec) (ftzValidExp emin prec hprec)
      (neuralFTZRound rnd) x = 0 := by
  let _ : NeuralValidExp (FTZExp emin prec) := ftzValidExp emin prec hprec
  by_cases hx : x = 0
  · subst x
    simp [neuralRound, neuralScaledMantissa, neuralFTZRound, neuralToReal]
  have hmag : neuralMagnitude β x ≤ FTZThreshold emin prec :=
    neuralMagnitude_le_of_abs_lt_bpow β x (FTZThreshold emin prec) hx hsmall
  have hcexp : neuralCexp β (FTZExp emin prec) x = FTZThreshold emin prec := by
    simp [neuralCexp, FTZExp_eq_threshold hmag]
  have hb : 0 < neuralBpow β (FTZThreshold emin prec) := neuralBpow.pos β _
  have hscaled : abs (neuralScaledMantissa β (FTZExp emin prec) x) < 1 := by
    rw [neuralScaledMantissa_eq_div, hcexp, abs_div, abs_of_pos hb, div_lt_one hb]
    exact hsmall
  unfold neuralRound neuralToReal
  rw [show neuralFTZRound rnd (neuralScaledMantissa β (FTZExp emin prec) x) = 0 by
    simp [neuralFTZRound, not_le.mpr hscaled]]
  simp

/-- At normal magnitudes, abrupt-underflow rounding agrees with FLX rounding. -/
theorem neuralRound_FTZ_eq_FLX_of_normal (emin prec : ℤ) (hprec : 0 < prec)
    (rnd : ℝ → ℤ) [NeuralValidRnd rnd] (x : ℝ)
    (hnormal : neuralBpow β (FTZThreshold emin prec) ≤ abs x) :
    @neuralRound β (FTZExp emin prec) (ftzValidExp emin prec hprec)
        (neuralFTZRound rnd) x =
      @neuralRound β (FLXExp prec) (flxValidExp prec hprec) rnd x := by
  let _ : NeuralValidExp (FTZExp emin prec) := ftzValidExp emin prec hprec
  let _ : NeuralValidExp (FLXExp prec) := flxValidExp prec hprec
  have hx : x ≠ 0 := by
    intro hx0
    rw [hx0, abs_zero] at hnormal
    exact (not_le_of_gt (neuralBpow.pos β _)) hnormal
  have hupper := abs_lt_neuralBpow_magnitude β x hx
  have hmag : FTZThreshold emin prec < neuralMagnitude β x :=
    (neuralBpow_lt_neuralBpow_iff β _ _).mp (hnormal.trans_lt hupper)
  have hcexp : neuralCexp β (FTZExp emin prec) x =
      neuralCexp β (FLXExp prec) x := by
    simp [neuralCexp, FTZExp_eq_sub hmag, FLXExp]
  have hscaled : neuralScaledMantissa β (FTZExp emin prec) x =
      neuralScaledMantissa β (FLXExp prec) x := by
    simp [neuralScaledMantissa, hcexp]
  have hb : 0 < neuralBpow β (neuralMagnitude β x - prec) := neuralBpow.pos β _
  have hlower := neuralBpow_magnitude_sub_one_le β x hx
  have hp : 0 ≤ prec - 1 := by linarith
  have hone : (1 : ℝ) ≤ neuralBpow β (prec - 1) := by
    change (1 : ℝ) ≤ β.toReal ^ (prec - 1)
    exact one_le_zpow₀ (NeuralRadix.gt_one β).le hp
  have hsone : 1 ≤ abs (neuralScaledMantissa β (FTZExp emin prec) x) := by
    rw [neuralScaledMantissa_eq_div, hcexp]
    simp only [neuralCexp, FLXExp]
    rw [abs_div, abs_of_pos hb]
    calc
      1 ≤ neuralBpow β (prec - 1) := hone
      _ = neuralBpow β ((neuralMagnitude β x - 1) -
          (neuralMagnitude β x - prec)) := by
        congr 1
        linarith
      _ = neuralBpow β (neuralMagnitude β x - 1) /
          neuralBpow β (neuralMagnitude β x - prec) :=
        neuralBpow.sub_exp β _ _
      _ ≤ abs x / neuralBpow β (neuralMagnitude β x - prec) :=
        div_le_div_of_nonneg_right hlower hb.le
  have hsoneFLX : 1 ≤ abs (neuralScaledMantissa β (FLXExp prec) x) := by
    simpa [hscaled] using hsone
  unfold neuralRound neuralToReal
  rw [hcexp, hscaled]
  simp [neuralFTZRound, hsoneFLX]

/-- Exact abrupt-underflow values are zero or normal-range FLX values. -/
def FTZFormat (emin prec : ℤ) (x : ℝ) : Prop :=
  FLXFormat (β := β) prec x ∧
    (x = 0 ∨ neuralBpow β (FTZThreshold emin prec) ≤ abs x)

/-- Nonpositive precision is rejected by the explicit abrupt-underflow format predicate. -/
theorem not_FTZFormat_of_nonpos (emin prec : ℤ) (hprec : prec ≤ 0) (x : ℝ) :
    ¬FTZFormat (β := β) emin prec x := by
  intro h
  exact not_FLXFormat_of_nonpos prec hprec x h.1

/-- The generic format generated by `FTZExp` satisfies the explicit abrupt-underflow predicate. -/
theorem FTZFormat_of_generic (emin prec : ℤ) (hprec : 0 < prec) {x : ℝ}
    (hx : @neuralGenericFormat β (FTZExp emin prec) (ftzValidExp emin prec hprec) x) :
    FTZFormat (β := β) emin prec x := by
  let _ : NeuralValidExp (FTZExp emin prec) := ftzValidExp emin prec hprec
  let _ : NeuralValidExp (FLXExp prec) := flxValidExp prec hprec
  constructor
  · rw [← generic_format_FLX_iff prec hprec]
    apply neural_generic_inclusion (fexp₁ := FTZExp emin prec) (fexp₂ := FLXExp prec)
    · intro e
      by_cases he : e ≤ FTZThreshold emin prec
      · rw [FTZExp_eq_threshold he]
        change e - prec ≤ FTZThreshold emin prec
        linarith
      · rw [FTZExp_eq_sub (lt_of_not_ge he)]
        rfl
    · exact hx
  · by_cases hx0 : x = 0
    · exact Or.inl hx0
    · right
      by_cases hmag : neuralMagnitude β x ≤ FTZThreshold emin prec
      · obtain ⟨n, hn⟩ := neural_scaled_mantissa_int_of_generic
          (β := β) (fexp := FTZExp emin prec) x hx
        have hn0 : n ≠ 0 := by
          intro hnzero
          have hrepr := neural_scaled_mantissa_mul_bpow
            (β := β) (fexp := FTZExp emin prec) x
          rw [hn, hnzero, Int.cast_zero, zero_mul] at hrepr
          exact hx0 hrepr.symm
        have habsn : (1 : ℝ) ≤ abs (n : ℝ) := by
          exact_mod_cast (Int.one_le_abs hn0)
        have hcexp : neuralCexp β (FTZExp emin prec) x = FTZThreshold emin prec := by
          simp [neuralCexp, FTZExp_eq_threshold hmag]
        have hrepr := neural_scaled_mantissa_mul_bpow
          (β := β) (fexp := FTZExp emin prec) x
        rw [hn, hcexp] at hrepr
        have hb := neuralBpow.nonneg β (FTZThreshold emin prec)
        calc
          neuralBpow β (FTZThreshold emin prec) =
              1 * neuralBpow β (FTZThreshold emin prec) := by ring
          _ ≤ abs (n : ℝ) * neuralBpow β (FTZThreshold emin prec) :=
            mul_le_mul_of_nonneg_right habsn hb
          _ = abs ((n : ℝ) * neuralBpow β (FTZThreshold emin prec)) := by
            rw [abs_mul, abs_of_nonneg hb]
          _ = abs x := by rw [hrepr]
      · have hmagLt : FTZThreshold emin prec < neuralMagnitude β x := lt_of_not_ge hmag
        have hlower := neuralBpow_magnitude_sub_one_le β x hx0
        exact ((neuralBpow_le_neuralBpow_iff β _ _).2 (by linarith)).trans hlower

/-- Every explicit abrupt-underflow value belongs to the generic `FTZExp` format. -/
theorem generic_of_FTZFormat (emin prec : ℤ) (hprec : 0 < prec) {x : ℝ}
    (hx : FTZFormat (β := β) emin prec x) :
    @neuralGenericFormat β (FTZExp emin prec) (ftzValidExp emin prec hprec) x := by
  let _ : NeuralValidExp (FTZExp emin prec) := ftzValidExp emin prec hprec
  let _ : NeuralValidExp (FLXExp prec) := flxValidExp prec hprec
  rcases hx with ⟨hxFLX, hxRange⟩
  rcases hxRange with rfl | hnormal
  · exact neural_generic_format_zero
  · have hx0 : x ≠ 0 := by
      intro hxzero
      rw [hxzero, abs_zero] at hnormal
      exact (not_le_of_gt (neuralBpow.pos β _)) hnormal
    have hupper := abs_lt_neuralBpow_magnitude β x hx0
    have hmag : FTZThreshold emin prec < neuralMagnitude β x :=
      (neuralBpow_lt_neuralBpow_iff β _ _).mp (hnormal.trans_lt hupper)
    apply neural_generic_inclusion_mag (fexp₁ := FLXExp prec) (fexp₂ := FTZExp emin prec)
    · intro _
      rw [FTZExp_eq_sub hmag]
      rfl
    · exact (generic_format_FLX_iff prec hprec x).2 hxFLX

/-- `FTZFormat` is exactly the generic format generated by `FTZExp`. -/
theorem generic_format_FTZ_iff (emin prec : ℤ) (hprec : 0 < prec) (x : ℝ) :
    @neuralGenericFormat β (FTZExp emin prec) (ftzValidExp emin prec hprec) x ↔
      FTZFormat (β := β) emin prec x :=
  ⟨FTZFormat_of_generic emin prec hprec, generic_of_FTZFormat emin prec hprec⟩

end TorchLean.Floats
