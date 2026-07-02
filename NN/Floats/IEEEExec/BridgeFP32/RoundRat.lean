/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.BridgeFP32.RoundDyadic

/-!
# IEEE32Exec and FP32: Rational Rounder Correctness
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

/-!
## Rounding rationals (finite/no-overflow)

Some operations (notably division and parts of transcendental approximations) naturally produce
rationals `num / den`. The executable kernel rounds those rationals to float32 by:

- classifying magnitude (normal/subnormal/underflow/overflow),
- computing a scaled mantissa,
- applying nearest-even,
- and assembling the output bits.

The lemmas below connect that algorithm to the `FP32` real rounding model.
-/

lemma neural_magnitude_signedRat (sign : Bool) (num den : Nat) (hnum : num ≠ 0) (hden : den
  ≠ 0) :
    TorchLean.Floats.neuralMagnitude binaryRadix ((if sign then (-1 : ℝ) else 1) * ((num : ℝ) /
      (den : ℝ))) =
      floorLog2Rat num den + 1 := by
  classical
  set k : Int := floorLog2Rat num den
  set r : ℝ := (num : ℝ) / (den : ℝ)
  set x : ℝ := (if sign then (-1 : ℝ) else 1) * r
  have hdenpos : (0 : ℝ) < (den : ℝ) := by
    exact_mod_cast Nat.pos_of_ne_zero hden
  have hnumpos : (0 : ℝ) < (num : ℝ) := by
    exact_mod_cast Nat.pos_of_ne_zero hnum
  have hrpos : 0 < r := div_pos hnumpos hdenpos
  have hx : x ≠ 0 := by
    have hr0 : r ≠ 0 := ne_of_gt hrpos
    cases sign <;> simp [x, hr0]
  have habs : _root_.abs x = r := by
    cases sign <;> simp [x, r, abs_of_pos hrpos]

  have hbounds := floorLog2Rat_bounds (num := num) (den := den) hnum hden
  have hk_le : neuralBpow binaryRadix k ≤ r := by
    simpa [k, r] using hbounds.1
  have hk_lt : r < neuralBpow binaryRadix (k + 1) := by
    simpa [k, r] using hbounds.2

  have hkpow_le : (2 : ℝ) ^ (k : ℝ) ≤ r := by
    have : (2 : ℝ) ^ k ≤ r := by
      simpa [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal, k, r] using hk_le
    have hEq : (2 : ℝ) ^ k = (2 : ℝ) ^ ((k : ℤ) : ℝ) := by
      simp
    have : (2 : ℝ) ^ ((k : ℤ) : ℝ) ≤ r := le_of_eq_of_le hEq.symm this
    simpa using this
  have hkpow_lt : r < (2 : ℝ) ^ ((k + 1) : ℝ) := by
    have : r < (2 : ℝ) ^ (k + 1) := by
      simpa [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal, k, r] using hk_lt
    have hEq : (2 : ℝ) ^ (k + 1) = (2 : ℝ) ^ (((k + 1 : ℤ)) : ℝ) := by
      simpa using (Real.rpow_intCast (x := (2 : ℝ)) (n := k + 1)).symm
    have : r < (2 : ℝ) ^ (((k + 1 : ℤ)) : ℝ) := lt_of_lt_of_eq this hEq
    -- Rewrite `((k+1 : ℤ) : ℝ)` as `((k : ℤ) : ℝ) + 1`.
    simpa [Int.cast_add, Int.cast_one] using this

  have hk_logb : (k : ℝ) ≤ Real.logb 2 r := by
    have hb : 1 < (2 : ℝ) := by norm_num
    exact (Real.le_logb_iff_rpow_le (b := (2 : ℝ)) (x := (k : ℝ)) (y := r) hb hrpos).2 hkpow_le
  have hk_logb_lt : Real.logb 2 r < (k : ℝ) + 1 := by
    have hb : 1 < (2 : ℝ) := by norm_num
    have hr_lt' : r < (2 : ℝ) ^ ((k : ℝ) + 1) := by
      -- Rewrite `(k+1 : ℝ)` as `k + 1`.
      simpa [Int.cast_add, Int.cast_one, add_assoc] using hkpow_lt
    exact (Real.logb_lt_iff_lt_rpow (b := (2 : ℝ)) (x := r) (y := (k : ℝ) + 1) hb hrpos).2 hr_lt'

  have hfloor : (⌊Real.logb 2 r⌋ : Int) = k := by
    refine (Int.floor_eq_iff).2 ?_
    constructor
    · exact hk_logb
    · exact hk_logb_lt

  have hb2 : binaryRadix.toReal = (2 : ℝ) := by rfl
  have hfloor' : (⌊Real.logb (binaryRadix.toReal) r⌋ : Int) = k := by
    simpa [hb2] using hfloor
  have : TorchLean.Floats.neuralMagnitude binaryRadix x = k + 1 := by
    -- Unfold and reduce `neural_magnitude` to the floor-log identity already proved as `hfloor`.
    simp [TorchLean.Floats.neuralMagnitude, hx, Real.log_div_log, habs, hb2]
    simpa [k] using hfloor
  simpa [x, k] using this

-- Shared dyadic/rational scaling lemmas live in `NN.Floats.IEEEExec.RatScaling`.

/--
Refinement theorem (finite/no-overflow): rounding an exact rational with the executable IEEE32
  kernel
agrees with the Flocq-style `FP32` rounding-on-`ℝ` model.

The hypothesis `isFinite (roundRatToIEEE32 sign num den) = true` rules out the overflow-to-`±Inf`
branches.
-/
theorem toReal_roundRatToIEEE32_eq_fp32Round (sign : Bool) (num den : Nat) (hden : den ≠ 0)
    (hfin : isFinite (roundRatToIEEE32 sign num den) = true) :
    toReal (roundRatToIEEE32 sign num den) =
      fp32Round ((if sign then (-1 : ℝ) else 1) * ((num : ℝ) / (den : ℝ))) := by
  classical
  by_cases hnum : num = 0
  · -- Both sides are real `0`.
    have hto : toReal (roundRatToIEEE32 sign num den) = 0 := by
      have hround0 : roundRatToIEEE32 sign num den = (if sign then negZero else posZero) := by
        simp [roundRatToIEEE32, hnum]
      rw [hround0]
      simpa using (toReal_signedZero sign)
    have hfp : fp32Round ((if sign then (-1 : ℝ) else 1) * ((num : ℝ) / (den : ℝ))) = 0 := by
      have hx0 :
          (if sign then (-1 : ℝ) else 1) * ((num : ℝ) / (den : ℝ)) = 0 := by
        simp [hnum]
      have hrnd0 : TorchLean.Floats.rnd32
          (TorchLean.Floats.neuralScaledMantissa binaryRadix TorchLean.Floats.fexp32 0) = 0 := by
        -- scaled mantissa is `0`, and nearest-even rounds `0` to `0`.
        have hAbs : _root_.abs (0 : ℝ) < (1 / 2 : ℝ) := by simp
        simpa [TorchLean.Floats.rnd32, TorchLean.Floats.neuralScaledMantissa] using
          (neural_nearest_even_eq_zero_of_abs_lt_half (x := (0 : ℝ)) hAbs)
      -- Now compute `fp32Round` at `0`.
      -- Keep `rnd32`/`neural_scaled_mantissa` folded so `hrnd0` can rewrite the mantissa to `0`.
      simp [fp32Round, hx0, TorchLean.Floats.neuralRound, TorchLean.Floats.neuralToReal, hrnd0]
    rw [hto, hfp]
  · have hnumbeq : (num == 0) = false := (beq_eq_false_iff_ne).2 hnum
    have hdenpos : (0 : ℝ) < (den : ℝ) := by exact_mod_cast Nat.pos_of_ne_zero hden
    have hnumpos : (0 : ℝ) < (num : ℝ) := by exact_mod_cast Nat.pos_of_ne_zero hnum
    set r : ℝ := (num : ℝ) / (den : ℝ)
    set x : ℝ := (if sign then (-1 : ℝ) else 1) * r
    have hrpos : 0 < r := div_pos hnumpos hdenpos
    have hx : x ≠ 0 := by
      have : r ≠ 0 := ne_of_gt hrpos
      cases sign <;> simp [x, this]

    set k : Int := floorLog2Rat num den
    -- Eliminate the overflow-to-Inf branch.
    by_cases hkHi : k > 127
    · have hround : roundRatToIEEE32 sign num den = (if sign then negInf else posInf) := by
        simp [roundRatToIEEE32, hnumbeq, k, hkHi]
      have hfalse : isFinite (roundRatToIEEE32 sign num den) = false := by
        rw [hround]
        cases sign <;> decide
      cases (hfalse.symm.trans hfin)
    · -- Non-overflowing exponent range: `k ≤ 127`.
      by_cases hkUnder : k < -150
      · -- Underflow-to-zero: show `FP32` rounding also yields `0`.
        have hround : roundRatToIEEE32 sign num den = (if sign then negZero else posZero) := by
          simp [roundRatToIEEE32, hnumbeq, k, hkHi, hkUnder]
        have hto : toReal (roundRatToIEEE32 sign num den) = 0 := by
          simpa [hround] using (toReal_signedZero sign)

        have hmag :
            TorchLean.Floats.neuralMagnitude binaryRadix x = k + 1 := by
          simpa [x, r, k] using
            (neural_magnitude_signedRat (sign := sign) (num := num) (den := den) hnum hden)
        have hcexp : TorchLean.Floats.neuralCexp binaryRadix TorchLean.Floats.fexp32 x = (-149 :
          Int) := by
          have hk1_le : k + 1 ≤ (-150 : Int) := by linarith
          have hk1_le' : k + 1 - 24 ≤ (-149 : Int) := by linarith
          simp [TorchLean.Floats.neuralCexp, TorchLean.Floats.fexp32, TorchLean.Floats.FLTExp,
            hmag,
            max_eq_right hk1_le']

        have hAbsBpow : _root_.abs x < neuralBpow binaryRadix (k + 1) := by
          have hbounds := floorLog2Rat_bounds (num := num) (den := den) hnum hden
          have habs : _root_.abs x = r := by
            cases sign <;> simp [x, r, abs_of_pos hrpos]
          simpa [habs, r, k] using hbounds.2
        have hk1_le : k + 1 ≤ (-150 : Int) := by linarith
        have hBpow_le :
            neuralBpow binaryRadix (k + 1) ≤ neuralBpow binaryRadix (-150 : Int) := by
          simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal]
          exact zpow_le_zpow_right₀ (by norm_num : (1 : ℝ) ≤ 2) hk1_le
        have hAbs150 : _root_.abs x < neuralBpow binaryRadix (-150 : Int) :=
          lt_of_lt_of_le hAbsBpow hBpow_le
        have hbpos149 : 0 < neuralBpow binaryRadix (149 : Int) := neuralBpow.pos binaryRadix 149
        have hAbsScaled :
            _root_.abs (x * neuralBpow binaryRadix (149 : Int)) < (1 / 2 : ℝ) := by
          have habs_mul :
              _root_.abs (x * neuralBpow binaryRadix (149 : Int)) =
                _root_.abs x * neuralBpow binaryRadix (149 : Int) := by
            have hnonneg : 0 ≤ neuralBpow binaryRadix (149 : Int) := le_of_lt hbpos149
            simp [abs_mul, abs_of_nonneg hnonneg]
          have hmul :
              _root_.abs x * neuralBpow binaryRadix (149 : Int) <
                neuralBpow binaryRadix (-150 : Int) * neuralBpow binaryRadix (149 : Int) :=
            (mul_lt_mul_of_pos_right hAbs150 hbpos149)
          have hprod :
              neuralBpow binaryRadix (-150 : Int) * neuralBpow binaryRadix (149 : Int) = (1 / 2
                : ℝ) := by
            have := (neuralBpow.add_exp binaryRadix (-150 : Int) (149 : Int))
            simpa [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal] using this.symm
          simpa [habs_mul, hprod, mul_assoc] using hmul
        have hRnd0 :
            TorchLean.Floats.neuralNearestEven (x * neuralBpow binaryRadix (149 : Int)) = 0 :=
          neural_nearest_even_eq_zero_of_abs_lt_half _ hAbsScaled
        have hfp : fp32Round x = 0 := by
          have hscaled :
              TorchLean.Floats.neuralScaledMantissa binaryRadix TorchLean.Floats.fexp32 x =
                x * neuralBpow binaryRadix (149 : Int) := by
            simp [TorchLean.Floats.neuralScaledMantissa, hcexp]
          have hmant :
              TorchLean.Floats.rnd32
                  (TorchLean.Floats.neuralScaledMantissa binaryRadix TorchLean.Floats.fexp32 x) =
                0 := by
            simpa [TorchLean.Floats.rnd32, hscaled] using hRnd0
          -- With mantissa rounded to `0`, `fp32Round x` is `0`.
          have hdisj :
              TorchLean.Floats.neuralNearestEven (x * neuralBpow binaryRadix (149 : Int)) = 0 ∨
                neuralBpow binaryRadix (-149 : Int) = 0 :=
            Or.inl hRnd0
          simpa [fp32Round, TorchLean.Floats.neuralRound, TorchLean.Floats.neuralToReal,
            TorchLean.Floats.neuralScaledMantissa, TorchLean.Floats.rnd32, hmant, hcexp, hscaled]
              using hdisj
        rw [hto, hfp]
      · -- Remaining cases (`k ≥ -150`): subnormal or normal rounding.
        by_cases hkSub : k < -126
        · -- Subnormal rounding.
          set frac : Nat := roundQuotEven (Nat.shiftLeft num 149) den
          have hround :
              roundRatToIEEE32 sign num den =
                if frac == 0 then
                  (if sign then negZero else posZero)
                else
                  match Nat.decLe (pow2 23) frac with
                  | isTrue _ => ofBits (mkBits sign 1 0)
                  | isFalse _ => ofBits (mkBits sign 0 frac) := by
            simp (config := { zeta := true }) [roundRatToIEEE32, hnumbeq, k, hkHi, hkUnder, hkSub,
              frac]
            rfl
          have hmag :
              TorchLean.Floats.neuralMagnitude binaryRadix x = k + 1 := by
            simpa [x, r, k] using
              (neural_magnitude_signedRat (sign := sign) (num := num) (den := den) hnum hden)
          have hcexp : TorchLean.Floats.neuralCexp binaryRadix TorchLean.Floats.fexp32 x = (-149 :
            Int) := by
            have hk1_le : k + 1 ≤ (-125 : Int) := by linarith [hkSub]
            have hk1_le' : k + 1 - 24 ≤ (-149 : Int) := by linarith [hk1_le]
            simp [TorchLean.Floats.neuralCexp, TorchLean.Floats.fexp32, TorchLean.Floats.FLTExp,
              hmag,
              max_eq_right hk1_le']

          have hscaled :
              TorchLean.Floats.neuralScaledMantissa binaryRadix TorchLean.Floats.fexp32 x =
                x * neuralBpow binaryRadix (149 : Int) := by
            simp [TorchLean.Floats.neuralScaledMantissa, hcexp]
          have hScaleRat :
              r * neuralBpow binaryRadix (Int.ofNat 149) =
                ((Nat.shiftLeft num 149 : Nat) : ℝ) / (den : ℝ) := by
            -- `r = num/den`
            dsimp [r]
            exact scaleRat_ofNat (num := num) (den := den) (sh := 149)

          have hRndFracPos :
              TorchLean.Floats.neuralNearestEven (r * neuralBpow binaryRadix (Int.ofNat 149)) =
                Int.ofNat frac := by
            have hden' : den ≠ 0 := hden
            have h :=
              neural_nearest_even_div_eq_roundQuotEven (num := Nat.shiftLeft num 149) (den := den)
                hden'
            -- rewrite the argument using `hScaleRat`
            calc
              TorchLean.Floats.neuralNearestEven (r * neuralBpow binaryRadix (Int.ofNat 149))
                  =
                  TorchLean.Floats.neuralNearestEven (((Nat.shiftLeft num 149 : Nat) : ℝ) / (den :
                    ℝ)) := by
                    rw [hScaleRat]
              _ = Int.ofNat (roundQuotEven (Nat.shiftLeft num 149) den) := h
              _ = Int.ofNat frac := by simp [frac]

          have hRndFrac :
              TorchLean.Floats.neuralNearestEven (x * neuralBpow binaryRadix (149 : Int)) =
                if sign then -Int.ofNat frac else Int.ofNat frac := by
            cases hs : sign
            · -- positive
              have hx' : x = r := by simp [x, hs]
              have hb149 : neuralBpow binaryRadix (149 : Int) = neuralBpow binaryRadix
                (Int.ofNat 149) := by rfl
              simp [hx', hb149, hRndFracPos]
            · -- negative
              have hx' : x = -r := by simp [x, hs]
              have hb149 : neuralBpow binaryRadix (149 : Int) = neuralBpow binaryRadix
                (Int.ofNat 149) := by rfl
              have hneg :
                  TorchLean.Floats.neuralNearestEven (- (r * neuralBpow binaryRadix (Int.ofNat
                    149))) =
                    -TorchLean.Floats.neuralNearestEven (r * neuralBpow binaryRadix (Int.ofNat
                      149)) := by
                simpa using (neural_nearest_even_neg (x := r * neuralBpow binaryRadix (Int.ofNat
                  149)))
              have hscale :
                  x * neuralBpow binaryRadix (149 : Int) = - (r * neuralBpow binaryRadix
                    (Int.ofNat 149)) := by
                simp [hx', hb149]
              -- Reduce to the positive case via oddness.
              calc
                TorchLean.Floats.neuralNearestEven (x * neuralBpow binaryRadix (Int.ofNat 149))
                    = TorchLean.Floats.neuralNearestEven (- (r * neuralBpow binaryRadix
                      (Int.ofNat 149))) := by
                      simpa [hb149] using congrArg TorchLean.Floats.neuralNearestEven hscale
                _ = -TorchLean.Floats.neuralNearestEven (r * neuralBpow binaryRadix (Int.ofNat
                  149)) := hneg
                _ = -Int.ofNat frac := by simpa [hRndFracPos]

          have hfp :
              fp32Round x =
                (if sign then (-1 : ℝ) else 1) * (frac : ℝ) * neuralBpow binaryRadix (-149 : Int)
                  := by
            have hround :
                fp32Round x =
                  (TorchLean.Floats.rnd32
                      (TorchLean.Floats.neuralScaledMantissa binaryRadix TorchLean.Floats.fexp32
                        x) : ℝ) *
                    neuralBpow binaryRadix (-149 : Int) := by
              simp [fp32Round, TorchLean.Floats.neuralRound, TorchLean.Floats.neuralToReal,
                hcexp]
            have hrndInt :
                TorchLean.Floats.rnd32 (TorchLean.Floats.neuralScaledMantissa binaryRadix
                  TorchLean.Floats.fexp32 x) =
                  (if sign then -Int.ofNat frac else Int.ofNat frac) := by
              -- `rnd32` is nearest-even, and `neural_scaled_mantissa = x * 2^149` in this branch.
              simpa [TorchLean.Floats.rnd32, hscaled] using hRndFrac
            have hrnd :
                (TorchLean.Floats.rnd32
                      (TorchLean.Floats.neuralScaledMantissa binaryRadix TorchLean.Floats.fexp32
                        x) : ℝ) =
                  (if sign then (-1 : ℝ) else 1) * (frac : ℝ) := by
              have hrnd' := congrArg (fun z : Int => (z : ℝ)) hrndInt
              cases sign <;> simp [hrnd']
            calc
              fp32Round x =
                  (TorchLean.Floats.rnd32
                      (TorchLean.Floats.neuralScaledMantissa binaryRadix TorchLean.Floats.fexp32
                        x) : ℝ) *
                    neuralBpow binaryRadix (-149 : Int) := hround
              _ = ((if sign then (-1 : ℝ) else 1) * (frac : ℝ)) * neuralBpow binaryRadix (-149 :
                Int) := by
                  simp [hrnd]
              _ = (if sign then (-1 : ℝ) else 1) * (frac : ℝ) * neuralBpow binaryRadix (-149 :
                Int) := by
                  simp []

          -- Compute `toReal` for the executable result.
          have hto :
              toReal (roundRatToIEEE32 sign num den) =
                (if sign then (-1 : ℝ) else 1) * (frac : ℝ) * neuralBpow binaryRadix (-149 : Int)
                  := by
            -- split on `frac == 0` and the `pow2 23 ≤ frac` test.
            by_cases hF0 : frac = 0
            · have hF0b : (frac == 0) = true := by simp [hF0]
              have hres : roundRatToIEEE32 sign num den = (if sign then negZero else posZero) := by
                simp [hround, hF0b]
              calc
                toReal (roundRatToIEEE32 sign num den) = 0 := by
                  rw [hres]
                  simpa using (toReal_signedZero sign)
                _ =
                    (if sign then (-1 : ℝ) else 1) * (frac : ℝ) *
                      neuralBpow binaryRadix (-149 : Int) := by
                  simp [hF0]
            · have hF0b : (frac == 0) = false := (beq_eq_false_iff_ne).2 hF0
              have hres : roundRatToIEEE32 sign num den =
                  match Nat.decLe (pow2 23) frac with
                  | isTrue _ => ofBits (mkBits sign 1 0)
                  | isFalse _ => ofBits (mkBits sign 0 frac) := by
                simp [hround, hF0b]
              rw [hres]
              -- `frac` is nonzero, so the output is either smallest normal or a true subnormal.
              cases hlt : Nat.decLe (pow2 23) frac with
              | isTrue hle =>
                  -- Output is the smallest normal, which corresponds to `frac = 2^23`.
                  have hle' : pow2 23 ≤ frac := hle
                  have hfrac_le : frac ≤ pow2 23 := by
                    -- Bound `r * 2^149 < 2^23`, then use `neural_nearest_even_bounds`.
                    have hk_lt : r < neuralBpow binaryRadix (k + 1) := by
                      have hbounds := floorLog2Rat_bounds (num := num) (den := den) hnum hden
                      simpa [r, k] using hbounds.2
                    have hk1_le : k + 1 ≤ (-126 : Int) := by linarith [hkSub]
                    have hbpow_le :
                        neuralBpow binaryRadix (k + 1) ≤ neuralBpow binaryRadix (-126 : Int) :=
                          by
                      simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal]
                      exact zpow_le_zpow_right₀ (by norm_num : (1 : ℝ) ≤ 2) hk1_le
                    have hr_lt126 : r < neuralBpow binaryRadix (-126 : Int) :=
                      lt_of_lt_of_le hk_lt hbpow_le
                    have hbpos149 : 0 < neuralBpow binaryRadix (149 : Int) := neuralBpow.pos
                      binaryRadix 149
                    have hscaled_lt : r * neuralBpow binaryRadix (149 : Int) < (pow2 23 : ℝ) := by
                      have hsum : (-126 : Int) + 149 = 23 := by norm_num
                      have hmul :
                          neuralBpow binaryRadix (-126 : Int) * neuralBpow binaryRadix (149 :
                            Int) =
                            neuralBpow binaryRadix (23 : Int) := by
                        simpa [hsum] using (neuralBpow.add_exp binaryRadix (-126 : Int) (149 :
                          Int)).symm
                      have h23 : neuralBpow binaryRadix (23 : Int) = (pow2 23 : ℝ) := by
                        simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal,
                          pow2_eq_two_pow]
                        norm_num
                      have hprod :
                          neuralBpow binaryRadix (-126 : Int) * neuralBpow binaryRadix (149 :
                            Int) = (pow2 23 : ℝ) :=
                        hmul.trans h23
                      have : r * neuralBpow binaryRadix (149 : Int) <
                          neuralBpow binaryRadix (-126 : Int) * neuralBpow binaryRadix (149 :
                            Int) :=
                        mul_lt_mul_of_pos_right hr_lt126 hbpos149
                      simpa [mul_assoc, hprod] using this
                    have hbound :
                        TorchLean.Floats.neuralNearestEven (r * neuralBpow binaryRadix
                          (Int.ofNat 149)) ≤
                          Int.ofNat (pow2 23) := by
                      have hbnds := TorchLean.Floats.neural_nearest_even_bounds (r * neuralBpow
                        binaryRadix (Int.ofNat 149))
                      have hfloor_lt :
                          (⌊r * neuralBpow binaryRadix (Int.ofNat 149)⌋ : Int) < Int.ofNat (pow2
                            23) :=
                        Int.floor_lt.2 (by simpa using hscaled_lt)
                      have hfloor_le :
                          (⌊r * neuralBpow binaryRadix (Int.ofNat 149)⌋ : Int) + 1 ≤ Int.ofNat
                            (pow2 23) :=
                        Int.add_one_le_iff.mpr hfloor_lt
                      exact le_trans hbnds.2 hfloor_le
                    have hint : Int.ofNat frac ≤ Int.ofNat (pow2 23) := by
                      -- Rewrite the LHS into the bounded nearest-even term.
                      rw [← hRndFracPos]
                      exact hbound
                    exact (Int.ofNat_le).1 hint
                  have hEq : frac = pow2 23 := le_antisymm hfrac_le hle'
                  have hexp1 : (1 : Nat) < 255 := by decide
                  have hfrac0 : (0 : Nat) < 2 ^ 23 := by decide
                  simp [hEq, toReal_eq,
                    toDyadic?_ofBits_mkBits_fin (sign := sign) (exp := 1) (frac := 0)
                      (hexp := hexp1) (hfrac := hfrac0),
                    dyadicToReal, TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal,
                    pow2_eq_two_pow, Nat.cast_ofNat, mul_comm]
              | isFalse hlt' =>
                  -- True subnormal: decode `mkBits sign 0 frac`.
                  have hfrac_lt : frac < 2 ^ 23 := by
                    -- `¬ pow2 23 ≤ frac` implies `frac < pow2 23 = 2^23`.
                    have : frac < pow2 23 := Nat.lt_of_not_ge hlt'
                    simpa [pow2_eq_two_pow] using this
                  simp [toReal_eq,
                    toDyadic?_ofBits_mkBits_fin (sign := sign) (exp := 0) (frac := frac)
                      (hexp := (by decide : (0 : Nat) < 255)) (hfrac := hfrac_lt),
                    dyadicToReal, TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal,
                      hF0,
                    ]

          -- Combine the executable and FP32 computations.
          rw [hto, hfp]
        · -- Normal rounding.
          -- Mirror the executable kernel definitions.
          set shift : Int := 23 - k
          set numden :=
            match shift with
            | .ofNat sh => (Nat.shiftLeft num sh, den)
            | .negSucc sh => (num, Nat.shiftLeft den (sh + 1))
          set num' : Nat := numden.1
          set den' : Nat := numden.2
          set m : Nat := roundQuotEven num' den'
          set k' : Int := if m == pow2 24 then k + 1 else k
          set m' : Nat := if m == pow2 24 then pow2 23 else m
          have hround :
              roundRatToIEEE32 sign num den =
                if k' > 127 then
                  (if sign then negInf else posInf)
                else
                  let expNat : Nat := Int.toNat (k' + 127)
                  let fracNat : Nat := m' - pow2 23
                  ofBits (mkBits sign expNat fracNat) := by
            simp (config := { zeta := true }) [roundRatToIEEE32, hnumbeq, k, hkHi, hkUnder, hkSub,
              shift, numden, num',
              den', m, k', m']
            rfl
          -- If the carry-adjusted exponent overflows, the result is `±Inf`, contradicting `hfin`.
          by_cases hk'Hi : k' > 127
          · have hInf : roundRatToIEEE32 sign num den = (if sign then negInf else posInf) := by
              simp [hround, hk'Hi]
            have hfalse : isFinite (roundRatToIEEE32 sign num den) = false := by
              rw [hInf]
              cases sign <;> decide
            cases (hfalse.symm.trans hfin)
          · have hk'le : k' ≤ 127 := le_of_not_gt hk'Hi
            -- Name the fields for decoding.
            set expNat : Nat := Int.toNat (k' + 127)
            set fracNat : Nat := m' - pow2 23
            have hroundBits : roundRatToIEEE32 sign num den = ofBits (mkBits sign expNat fracNat) :=
              by
              simp [hround, hk'Hi, expNat, fracNat]

            have hkge : (-126 : Int) ≤ k := (not_lt).1 hkSub
            have hk'ge : (-126 : Int) ≤ k' := by
              by_cases hcarry : m == pow2 24
              · simp [k', hcarry]
                linarith [hkge]
              · simpa [k', hcarry] using hkge
            have hk'exp_nonneg : 0 ≤ k' + 127 := by linarith [hk'ge]
            have hk'exp_lt : k' + 127 < (255 : Int) := by linarith [hk'le]
            have hexp : expNat < 255 := by
              have h := (Int.toNat_lt_of_ne_zero (m := k' + 127) (n := 255) (by decide)).2 hk'exp_lt
              simpa [expNat] using h

            -- Compute the FP32 canonical exponent `k - 23`.
            have hmag :
                TorchLean.Floats.neuralMagnitude binaryRadix x = k + 1 := by
              simpa [x, r, k] using
                (neural_magnitude_signedRat (sign := sign) (num := num) (den := den) hnum hden)
            have hcexp :
                TorchLean.Floats.neuralCexp binaryRadix TorchLean.Floats.fexp32 x = k - 23 := by
              have hk1_ge : (-149 : Int) ≤ k + 1 - 24 := by linarith [hkge]
              have hk123 : k + 1 - 24 = k - 23 := by linarith
              have h' :
                  TorchLean.Floats.neuralCexp binaryRadix TorchLean.Floats.fexp32 x = k + 1 - 24
                    := by
                simp [TorchLean.Floats.neuralCexp, TorchLean.Floats.fexp32,
                  TorchLean.Floats.FLTExp, hmag,
                  max_eq_left hk1_ge]
              simpa [hk123] using h'
            have hscaled :
                TorchLean.Floats.neuralScaledMantissa binaryRadix TorchLean.Floats.fexp32 x =
                  x * neuralBpow binaryRadix (23 - k) := by
              simp [TorchLean.Floats.neuralScaledMantissa, hcexp]

            -- The positive scaled mantissa is a nonnegative rational `num'/den'`.
            have hden' : den' ≠ 0 := by
              cases hshift : shift with
              | ofNat sh =>
                  have hden'eq : den' = den := by
                    simp [den', numden, hshift]
                  simpa [hden'eq] using hden
              | negSucc sh =>
                  have hden'eq : den' = Nat.shiftLeft den (sh + 1) := by
                    simp [den', numden, hshift]
                  intro h0
                  have h0' : Nat.shiftLeft den (sh + 1) = 0 := by
                    simpa [hden'eq] using h0
                  have hmul : den * 2 ^ (sh + 1) = 0 := by
                    simpa [Nat.shiftLeft_eq] using h0'
                  have : den = 0 := by
                    have : den = 0 ∨ 2 ^ (sh + 1) = 0 := Nat.mul_eq_zero.mp hmul
                    cases this with
                    | inl h => exact h
                    | inr hpow =>
                        have hpos : 0 < 2 ^ (sh + 1) :=
                          Nat.pow_pos (a := 2) (n := sh + 1) (by decide : 0 < (2 : Nat))
                        have : False := (Nat.ne_of_gt hpos) hpow
                        exact False.elim this
                  exact (hden this).elim

            have hScalePos :
                r * neuralBpow binaryRadix (23 - k) = (num' : ℝ) / (den' : ℝ) := by
              cases hshift : shift with
              | ofNat sh =>
                  have hk : 23 - k = (Int.ofNat sh) := by
                    simpa [shift] using hshift
                  have hnumden : numden = (Nat.shiftLeft num sh, den) := by
                    simp [numden, hshift]
                  -- `r * 2^sh = (num <<< sh) / den`
                  have hbpow : neuralBpow binaryRadix (Int.ofNat sh) = (2 : ℝ) ^ sh := by
                    simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal]
                  calc
                    r * neuralBpow binaryRadix (23 - k)
                        = (num : ℝ) / (den : ℝ) * neuralBpow binaryRadix (Int.ofNat sh) := by
                          simp [r, hk]
                    _ = (num : ℝ) / (den : ℝ) * (2 : ℝ) ^ sh := by
                          rw [hbpow]
                    _ = ((num : ℝ) * (2 : ℝ) ^ sh) / (den : ℝ) := by
                          simp [div_mul_eq_mul_div]
                    _ = (num' : ℝ) / (den' : ℝ) := by
                          have hnum' : (num' : ℝ) = (num : ℝ) * (2 : ℝ) ^ sh := by
                            -- `num' = num <<< sh`.
                            simp [hnumden, num', Nat.shiftLeft_eq, Nat.cast_mul, Nat.cast_pow]
                          have hden'' : (den' : ℝ) = (den : ℝ) := by
                            simp [hnumden, den']
                          simp [hnum', hden'']
              | negSucc sh =>
                  have hk : 23 - k = (Int.negSucc sh) := by
                    simpa [shift] using hshift
                  have hnumden : numden = (num, Nat.shiftLeft den (sh + 1)) := by
                    simp [numden, hshift]
                  have hbpow :
                      neuralBpow binaryRadix (Int.negSucc sh) = ((2 : ℝ) ^ (sh + 1))⁻¹ := by
                    simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal]
                  calc
                    r * neuralBpow binaryRadix (23 - k)
                        = (num : ℝ) / (den : ℝ) * neuralBpow binaryRadix (Int.negSucc sh) := by
                          simp [r, hk]
                    _ = (num : ℝ) / (den : ℝ) * ((2 : ℝ) ^ (sh + 1))⁻¹ := by simp [hbpow]
                    _ = ((num : ℝ) / (den : ℝ)) / ((2 : ℝ) ^ (sh + 1)) := by
                          simp [div_eq_mul_inv]
                    _ = (num : ℝ) / ((den : ℝ) * (2 : ℝ) ^ (sh + 1)) := by
                          simp [div_div]
                    _ = (num' : ℝ) / (den' : ℝ) := by
                          simp [hnumden, num', den', Nat.shiftLeft_eq, Nat.cast_mul, Nat.cast_pow]

            have hRndPos :
                TorchLean.Floats.neuralNearestEven (r * neuralBpow binaryRadix (23 - k)) =
                  Int.ofNat m := by
              have h :=
                neural_nearest_even_div_eq_roundQuotEven (num := num') (den := den') hden'
              simpa [m, hScalePos] using h

            have hRnd :
                TorchLean.Floats.neuralNearestEven (x * neuralBpow binaryRadix (23 - k)) =
                  if sign then -Int.ofNat m else Int.ofNat m := by
              cases hs : sign
              · have hx' : x = r := by simp [x, r, hs]
                simp [hx', hRndPos]
              · have hx' : x = -r := by simp [x, r, hs]
                have hscale : x * neuralBpow binaryRadix (23 - k) = -(r * neuralBpow binaryRadix
                  (23 - k)) := by
                  simp [hx']
                have hneg :
                    TorchLean.Floats.neuralNearestEven (-(r * neuralBpow binaryRadix (23 - k)))
                      =
                      -TorchLean.Floats.neuralNearestEven (r * neuralBpow binaryRadix (23 - k))
                        := by
                  simpa using (neural_nearest_even_neg (x := r * neuralBpow binaryRadix (23 - k)))
                simp [hscale, hneg, hRndPos]

            -- Bound the mantissa to show `fracNat < 2^23` in the non-carry branch.
            have hm_ge : pow2 23 ≤ m := by
              -- `2^23 ≤ r * 2^(23-k)` from `2^k ≤ r`.
              have hbounds := floorLog2Rat_bounds (num := num) (den := den) hnum hden
              have hbpos : 0 < neuralBpow binaryRadix (23 - k) := neuralBpow.pos binaryRadix (23
                - k)
              have hk_le_r : neuralBpow binaryRadix k ≤ r := by simpa [k, r] using hbounds.1
              have hmul := mul_le_mul_of_nonneg_right hk_le_r (le_of_lt hbpos)
              have hprod :
                  neuralBpow binaryRadix k * neuralBpow binaryRadix (23 - k) = neuralBpow
                    binaryRadix (23 : Int) := by
                have hadd := (neuralBpow.add_exp binaryRadix k (23 - k))
                have hsum : k + (23 - k) = (23 : Int) := by linarith
                simpa [hsum] using hadd.symm
              have hle : neuralBpow binaryRadix (23 : Int) ≤ r * neuralBpow binaryRadix (23 - k)
                := by
                simpa [hprod] using hmul
              have hmono :=
                (NeuralValidRnd.monotone (rnd := TorchLean.Floats.neuralNearestEven)
                  (x := (neuralBpow binaryRadix (23 : Int))) (y := r * neuralBpow binaryRadix
                    (23 - k)) hle)
              have hid :
                  TorchLean.Floats.neuralNearestEven (neuralBpow binaryRadix (23 : Int)) =
                    Int.ofNat (pow2 23) := by
                -- `bpow 23 = 2^23 = pow2 23`.
                have hbpow : neuralBpow binaryRadix (23 : Int) = (pow2 23 : ℝ) := by
                  have hcast : ((2 ^ 23 : Nat) : ℝ) = (pow2 23 : ℝ) := by
                    simpa using congrArg (fun n : Nat => (n : ℝ)) (pow2_eq_two_pow 23).symm
                  have hb : neuralBpow binaryRadix (23 : Int) = ((2 ^ 23 : Nat) : ℝ) := by
                    have hb' : neuralBpow binaryRadix (23 : Int) = (2 : ℝ) ^ 23 := by
                      simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal,
                        zpow_ofNat]
                    have hcastpow : ((2 ^ 23 : Nat) : ℝ) = (2 : ℝ) ^ 23 := by
                      simpa using (Nat.cast_pow (α := ℝ) 2 23)
                    exact hb'.trans hcastpow.symm
                  exact hb.trans hcast
                have : TorchLean.Floats.neuralNearestEven (pow2 23 : ℝ) = Int.ofNat (pow2 23) :=
                  by
                  simpa using
                    (TorchLean.Floats.NeuralValidRnd.id (rnd :=
                      TorchLean.Floats.neuralNearestEven) (Int.ofNat (pow2 23)))
                simpa [hbpow] using this
              have : Int.ofNat (pow2 23) ≤ Int.ofNat m := by
                simpa [hid, hRndPos] using hmono
              exact (Int.ofNat_le).1 this

            have hm_le : m ≤ pow2 24 := by
              -- `r * 2^(23-k) < 2^24` from `r < 2^(k+1)`.
              have hbounds := floorLog2Rat_bounds (num := num) (den := den) hnum hden
              have hbpos : 0 < neuralBpow binaryRadix (23 - k) := neuralBpow.pos binaryRadix (23
                - k)
              have hr_lt : r < neuralBpow binaryRadix (k + 1) := by simpa [k, r] using hbounds.2
              have hmul := mul_lt_mul_of_pos_right hr_lt hbpos
              have hprod :
                  neuralBpow binaryRadix (k + 1) * neuralBpow binaryRadix (23 - k) = neuralBpow
                    binaryRadix (24 : Int) := by
                have hadd := (neuralBpow.add_exp binaryRadix (k + 1) (23 - k))
                have hsum : (k + 1) + (23 - k) = (24 : Int) := by linarith
                simpa [hsum] using hadd.symm
              have hlt : r * neuralBpow binaryRadix (23 - k) < neuralBpow binaryRadix (24 : Int)
                := by
                simpa [hprod] using hmul
              have hle : r * neuralBpow binaryRadix (23 - k) ≤ neuralBpow binaryRadix (24 : Int)
                := le_of_lt hlt
              have hmono :=
                (NeuralValidRnd.monotone (rnd := TorchLean.Floats.neuralNearestEven)
                  (x := r * neuralBpow binaryRadix (23 - k)) (y := neuralBpow binaryRadix (24 :
                    Int)) hle)
              have hid :
                  TorchLean.Floats.neuralNearestEven (neuralBpow binaryRadix (24 : Int)) =
                    Int.ofNat (pow2 24) := by
                have hbpow : neuralBpow binaryRadix (24 : Int) = (pow2 24 : ℝ) := by
                  have hcast : ((2 ^ 24 : Nat) : ℝ) = (pow2 24 : ℝ) := by
                    simpa using congrArg (fun n : Nat => (n : ℝ)) (pow2_eq_two_pow 24).symm
                  have hb : neuralBpow binaryRadix (24 : Int) = ((2 ^ 24 : Nat) : ℝ) := by
                    have hb' : neuralBpow binaryRadix (24 : Int) = (2 : ℝ) ^ 24 := by
                      simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal,
                        zpow_ofNat]
                    have hcastpow : ((2 ^ 24 : Nat) : ℝ) = (2 : ℝ) ^ 24 := by
                      simpa using (Nat.cast_pow (α := ℝ) 2 24)
                    exact hb'.trans hcastpow.symm
                  exact hb.trans hcast
                have : TorchLean.Floats.neuralNearestEven (pow2 24 : ℝ) = Int.ofNat (pow2 24) :=
                  by
                  simpa using
                    (TorchLean.Floats.NeuralValidRnd.id (rnd :=
                      TorchLean.Floats.neuralNearestEven) (Int.ofNat (pow2 24)))
                simpa [hbpow] using this
              have : Int.ofNat m ≤ Int.ofNat (pow2 24) := by
                simpa [hid, hRndPos] using hmono
              exact (Int.ofNat_le).1 this

            have hfrac_lt : fracNat < 2 ^ 23 := by
              by_cases hcarryEq : m = pow2 24
              · have hb : (m == pow2 24) = true := by simp [hcarryEq]
                simp [fracNat, m', hb]
              · have hb : (m == pow2 24) = false := by
                  apply (Bool.eq_false_iff).2
                  intro hb'
                  have : m = pow2 24 := (beq_iff_eq).1 hb'
                  exact hcarryEq this
                have hm_lt : m < pow2 24 := lt_of_le_of_ne hm_le hcarryEq
                have hm'_eq : m' = m := by simp [m', hb]
                have hdiff : fracNat = m - pow2 23 := by simp [fracNat, hm'_eq]
                have hm_lt' : m - pow2 23 < pow2 23 := by
                  have hm_lt24 : m < 2 * pow2 23 := by
                    simpa [pow2_eq_two_pow, Nat.pow_succ] using hm_lt
                  -- `m < 2*2^23` implies `m - 2^23 < 2^23`.
                  have : m < pow2 23 + pow2 23 := by
                    simpa [two_mul, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using hm_lt24
                  exact Nat.sub_lt_left_of_lt_add hm_ge this
                simpa [hdiff, pow2_eq_two_pow] using hm_lt'

            -- Compute `toReal` for the executable result.
            have hto :
                toReal (roundRatToIEEE32 sign num den) =
                  (if sign then (-1 : ℝ) else 1) * (m' : ℝ) * neuralBpow binaryRadix (k' - 23) :=
                    by
              -- Decode the produced bits.
              have hdy :
                  toDyadic? (ofBits (mkBits sign expNat fracNat)) =
                    some { sign := sign, mant := pow2 23 + fracNat, exp := (Int.ofNat expNat) - 150
                      } := by
                have hexp' : expNat < 255 := by simpa using hexp
                have hfrac' : fracNat < 2 ^ 23 := by simpa using hfrac_lt
                have hexpNat0 : expNat ≠ 0 := by
                  intro h0
                  have hk127pos : 0 < k' + 127 := by linarith [hk'ge]
                  have hk127le : k' + 127 ≤ 0 := by
                    have : (k' + 127).toNat = 0 := by simpa [expNat] using h0
                    exact (Int.toNat_eq_zero).1 this
                  linarith
                simp [toDyadic?_ofBits_mkBits_fin (sign := sign) (exp := expNat) (frac := fracNat)
                  (hexp := hexp') (hfrac := hfrac'), hexpNat0]
              have hm'_mant : pow2 23 + fracNat = m' := by
                have hm'_ge : pow2 23 ≤ m' := by
                  by_cases hcarry : m == pow2 24
                  · simp [m', hcarry]
                  · have : m' = m := by simp [m', hcarry]
                    simpa [this] using hm_ge
                simp [fracNat, Nat.add_sub_of_le hm'_ge]
              have hexpInt : (expNat : Int) - 150 = k' - 23 := by
                -- `expNat = (k' + 127).toNat`, and `k' + 127 ≥ 0`.
                have hk'127 : (expNat : Int) = k' + 127 := by
                  simpa [expNat] using (Int.toNat_of_nonneg hk'exp_nonneg)
                linarith [hk'127]
              have htoBits : toReal (ofBits (mkBits sign expNat fracNat)) =
                  (if sign then (-1 : ℝ) else 1) * (m' : ℝ) * neuralBpow binaryRadix (k' - 23) :=
                    by
                -- Avoid `simp` timeouts by rewriting stepwise.
                rw [toReal_eq]
                simp [hdy, dyadicToReal, hm'_mant, hexpInt]
              simpa [hroundBits] using htoBits

            have hfp :
                fp32Round x =
                  (if sign then (-1 : ℝ) else 1) * (m : ℝ) * neuralBpow binaryRadix (k - 23) := by
              -- Unfold `fp32Round` using the computed mantissa and exponent.
              have hrnd :
                  ((TorchLean.Floats.rnd32
                          (TorchLean.Floats.neuralScaledMantissa binaryRadix
                            TorchLean.Floats.fexp32 x) : ℤ) : ℝ) =
                    (if sign then -(m : ℝ) else (m : ℝ)) := by
                have h0 :
                    TorchLean.Floats.neuralNearestEven
                        (TorchLean.Floats.neuralScaledMantissa binaryRadix
                          TorchLean.Floats.fexp32 x) =
                      if sign then -(m : ℤ) else (m : ℤ) := by
                  simpa [hscaled] using hRnd
                have h0' :
                    TorchLean.Floats.rnd32
                        (TorchLean.Floats.neuralScaledMantissa binaryRadix
                          TorchLean.Floats.fexp32 x) =
                      if sign then -(m : ℤ) else (m : ℤ) := by
                  simpa [TorchLean.Floats.rnd32] using h0
                cases hs : sign
                · -- `sign = false`
                  have h0'' :
                      ((TorchLean.Floats.rnd32
                              (TorchLean.Floats.neuralScaledMantissa binaryRadix
                                TorchLean.Floats.fexp32 x) : ℤ) : ℝ) =
                        (m : ℝ) := by
                    have : TorchLean.Floats.rnd32
                            (TorchLean.Floats.neuralScaledMantissa binaryRadix
                              TorchLean.Floats.fexp32 x) =
                          (m : ℤ) := by
                      simpa [hs] using h0'
                    simpa using congrArg (fun z : ℤ => (z : ℝ)) this
                  simpa [hs] using h0''
                · -- `sign = true`
                  have h0'' :
                      ((TorchLean.Floats.rnd32
                              (TorchLean.Floats.neuralScaledMantissa binaryRadix
                                TorchLean.Floats.fexp32 x) : ℤ) : ℝ) =
                        -(m : ℝ) := by
                    have : TorchLean.Floats.rnd32
                            (TorchLean.Floats.neuralScaledMantissa binaryRadix
                              TorchLean.Floats.fexp32 x) =
                          -(m : ℤ) := by
                      simpa [hs] using h0'
                    simpa using congrArg (fun z : ℤ => (z : ℝ)) this
                  simpa [hs] using h0''
              -- Finish without simp-canceling products.
              calc
                fp32Round x =
                    ((TorchLean.Floats.rnd32
                            (TorchLean.Floats.neuralScaledMantissa binaryRadix
                              TorchLean.Floats.fexp32 x) : ℤ) : ℝ) *
                      neuralBpow binaryRadix (k - 23) := by
                      simp [fp32Round, TorchLean.Floats.neuralRound,
                        TorchLean.Floats.neuralToReal, hcexp]
                _ =
                    (if sign then -(m : ℝ) else (m : ℝ)) * neuralBpow binaryRadix (k - 23) := by
                      simp [hrnd]
                _ =
                    (if sign then (-1 : ℝ) else 1) * (m : ℝ) * neuralBpow binaryRadix (k - 23) :=
                      by
                      cases sign <;> simp []

            -- Carry adjustment: `m' * 2^(k'-23) = m * 2^(k-23)` as reals.
            have hadj :
                (m' : ℝ) * neuralBpow binaryRadix (k' - 23) =
                  (m : ℝ) * neuralBpow binaryRadix (k - 23) := by
              by_cases hcarryEq : m = pow2 24
              · have hb : (m == pow2 24) = true := by simp [hcarryEq]
                have hm' : m' = pow2 23 := by simp [m', hb]
                have hk' : k' = k + 1 := by simp [k', hb]
                -- Rewrite everything into the canonical carry form.
                rw [hm', hk', hcarryEq]
                have hk : (k + 1 : Int) - 23 = (k - 23) + 1 := by linarith
                rw [hk, neuralBpow.add_exp]
                have htwo : neuralBpow binaryRadix (1 : Int) = (2 : ℝ) := by
                  simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal]
                rw [htwo]
                have hp : ((pow2 24 : Nat) : ℝ) = (2 : ℝ) * (pow2 23 : ℝ) := by
                  simp [pow2_eq_two_pow, Nat.pow_succ]
                  norm_num
                -- Now it's pure algebra.
                simp [hp, mul_assoc, mul_left_comm, mul_comm]
              · have hb : (m == pow2 24) = false := by
                  apply (Bool.eq_false_iff).2
                  intro hb'
                  have : m = pow2 24 := (beq_iff_eq).1 hb'
                  exact hcarryEq this
                simp [k', m', hb]

            rw [hto, hfp]
            -- Push the carry adjustment through the sign factor.
            cases hs : sign <;> simp [hadj, mul_comm]
end IEEE32Exec

end TorchLean.Floats.IEEE754
