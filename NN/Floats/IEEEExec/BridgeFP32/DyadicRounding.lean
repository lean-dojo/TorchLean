/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.BridgeFP32.Core

/-!
# IEEE32Exec and FP32: Dyadic Rounding Helpers
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

/-!
## Exact dyadic addition (used by op-level refinement theorems)

IEEE-754 addition is “exact add, then round”. To connect an executable `add` to the `FP32` model, we
factor the refinement into two steps:

1. perform an *exact* addition in an unbounded format (here: dyadics with integer exponents), and
2. apply float32 rounding.

`addDyadic` is our exact step (it aligns exponents, adds signed mantissas, and normalizes), and the
the lemmas below show that its real interpretation is literally real addition.
-/

noncomputable def signedMant (sign : Bool) (m : Nat) : Int :=
  if sign then -(Int.ofNat m) else Int.ofNat m

lemma dyadicToReal_eq_signedMant (d : Dyadic) :
    dyadicToReal d = (signedMant d.sign d.mant : ℝ) * neuralBpow binaryRadix d.exp := by
  by_cases hs : d.sign <;> simp [dyadicToReal, signedMant, hs]

lemma signedMant_shiftLeft (sign : Bool) (m sh : Nat) :
    ((signedMant sign (Nat.shiftLeft m sh) : Int) : ℝ) =
      ((signedMant sign m : Int) : ℝ) * neuralBpow binaryRadix (Int.ofNat sh) := by
  by_cases hs : sign
  · simp [signedMant, hs, TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal,
    Nat.shiftLeft_eq,
      Nat.cast_mul, Nat.cast_pow]
  · simp [signedMant, hs, TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal,
    Nat.shiftLeft_eq,
      Nat.cast_mul, Nat.cast_pow]

lemma signFactor_natAbs_int (s : Int) :
    (if decide (s < 0) then (-1 : ℝ) else (1 : ℝ)) * (Int.natAbs s : ℝ) = (s : ℝ) := by
  cases s with
  | ofNat n =>
      simp
  | negSucc n =>
      simp

lemma dyadicToReal_ofNatAbs (s : Int) (e : Int) :
    dyadicToReal { sign := decide (s < 0), mant := Int.natAbs s, exp := e } =
      (s : ℝ) * neuralBpow binaryRadix e := by
  dsimp [dyadicToReal]
  rw [signFactor_natAbs_int (s := s)]

lemma dyadicToReal_zero (sign : Bool) :
    dyadicToReal { sign := sign, mant := 0, exp := 0 } = (0 : ℝ) := by
  by_cases hs : sign <;> simp [dyadicToReal, hs, TorchLean.Floats.neuralBpow, binaryRadix,
    NeuralRadix.toReal]

/--
`addDyadic` is exact with respect to `dyadicToReal`.

Informal: `addDyadic` aligns exponents, adds signed mantissas, and normalizes; decoding the result
gives the sum of the decoded inputs.
-/
theorem dyadicToReal_addDyadic_exact (a b : Dyadic) :
    dyadicToReal (addDyadic a b) = dyadicToReal a + dyadicToReal b := by
  classical
  by_cases hab : a.exp ≤ b.exp
  · -- align to `a.exp`
    let sh : Nat := Int.toNat (b.exp - a.exp)
    have hdiff_nonneg : 0 ≤ b.exp - a.exp := sub_nonneg.mpr hab
    have hdiff : (b.exp - a.exp) = (sh : Int) := by
      have := (Int.toNat_of_nonneg (a := b.exp - a.exp) hdiff_nonneg)
      simpa [sh] using this.symm
    have hbexp : b.exp = a.exp + (sh : Int) := by
      have hb : a.exp + (b.exp - a.exp) = b.exp := by
        simp [sub_eq_add_neg]
      have hb' : a.exp + (sh : Int) = b.exp := by
        simpa [hdiff] using hb
      exact hb'.symm

    let m1 : Int := signedMant a.sign a.mant
    let m2 : Int := signedMant b.sign (Nat.shiftLeft b.mant sh)
    let s : Int := m1 + m2

    have hadd :
        addDyadic a b =
          if s == 0 then { sign := a.sign && b.sign, mant := 0, exp := 0 }
          else { sign := decide (s < 0), mant := Int.natAbs s, exp := a.exp } := by
      simp (config := { zeta := true }) [addDyadic, hab, sh, m1, m2, s, signedMant]

    have ha : dyadicToReal a = (m1 : ℝ) * neuralBpow binaryRadix a.exp := by
      simp [m1, dyadicToReal_eq_signedMant, signedMant]

    have hb : dyadicToReal b = (m2 : ℝ) * neuralBpow binaryRadix a.exp := by
      have hb0 := dyadicToReal_eq_signedMant (d := b)
      rw [hb0, hbexp]
      rw [neuralBpow.add_exp binaryRadix a.exp (sh : Int)]
      calc
        (signedMant b.sign b.mant : ℝ) *
            (neuralBpow binaryRadix a.exp * neuralBpow binaryRadix (sh : Int)) =
            ((signedMant b.sign b.mant : ℝ) * neuralBpow binaryRadix (sh : Int)) *
              neuralBpow binaryRadix a.exp := by
              ring
        _ = (m2 : ℝ) * neuralBpow binaryRadix a.exp := by
            have hm2 : (m2 : ℝ) = (signedMant b.sign b.mant : ℝ) * neuralBpow binaryRadix (sh :
              Int) := by
              simpa [m2] using (signedMant_shiftLeft (sign := b.sign) (m := b.mant) (sh := sh))
            simp [hm2]

    have hsum : dyadicToReal a + dyadicToReal b = (s : ℝ) * neuralBpow binaryRadix a.exp := by
      rw [ha, hb]
      have hfactor :
          (m1 : ℝ) * neuralBpow binaryRadix a.exp + (m2 : ℝ) * neuralBpow binaryRadix a.exp =
            ((m1 : ℝ) + (m2 : ℝ)) * neuralBpow binaryRadix a.exp := by
        simpa using (add_mul (m1 : ℝ) (m2 : ℝ) (neuralBpow binaryRadix a.exp)).symm
      have hcast : ((m1 : ℝ) + (m2 : ℝ)) = (s : ℝ) := by
        simp [s, Int.cast_add]
      calc
        (m1 : ℝ) * neuralBpow binaryRadix a.exp + (m2 : ℝ) * neuralBpow binaryRadix a.exp =
            ((m1 : ℝ) + (m2 : ℝ)) * neuralBpow binaryRadix a.exp := hfactor
        _ = (s : ℝ) * neuralBpow binaryRadix a.exp := by simp [hcast]

    by_cases hs0 : s = 0
    · rw [hadd]
      simp [hs0, dyadicToReal_zero, hsum]
    · have hs0b : (s == 0) = false := (beq_eq_false_iff_ne).2 hs0
      have hres :
          dyadicToReal { sign := decide (s < 0), mant := Int.natAbs s, exp := a.exp } =
            (s : ℝ) * neuralBpow binaryRadix a.exp := by
        simpa using (dyadicToReal_ofNatAbs (s := s) (e := a.exp))
      rw [hadd]
      simp [hs0b, hres, hsum]

  · -- align to `b.exp`
    have hba : b.exp ≤ a.exp := le_of_not_ge hab
    let sh : Nat := Int.toNat (a.exp - b.exp)
    have hdiff_nonneg : 0 ≤ a.exp - b.exp := sub_nonneg.mpr hba
    have hdiff : (a.exp - b.exp) = (sh : Int) := by
      have := (Int.toNat_of_nonneg (a := a.exp - b.exp) hdiff_nonneg)
      simpa [sh] using this.symm
    have haexp : a.exp = b.exp + (sh : Int) := by
      have hb : b.exp + (a.exp - b.exp) = a.exp := by
        simp [sub_eq_add_neg]
      have hb' : b.exp + (sh : Int) = a.exp := by
        simpa [hdiff] using hb
      exact hb'.symm

    let m1 : Int := signedMant a.sign (Nat.shiftLeft a.mant sh)
    let m2 : Int := signedMant b.sign b.mant
    let s : Int := m1 + m2

    have hadd :
        addDyadic a b =
          if s == 0 then { sign := a.sign && b.sign, mant := 0, exp := 0 }
          else { sign := decide (s < 0), mant := Int.natAbs s, exp := b.exp } := by
      simp (config := { zeta := true }) [addDyadic, hab, sh, m1, m2, s, signedMant]

    have hb' : dyadicToReal b = (m2 : ℝ) * neuralBpow binaryRadix b.exp := by
      simp [m2, dyadicToReal_eq_signedMant, signedMant]

    have ha' : dyadicToReal a = (m1 : ℝ) * neuralBpow binaryRadix b.exp := by
      have ha0 := dyadicToReal_eq_signedMant (d := a)
      rw [ha0, haexp]
      rw [neuralBpow.add_exp binaryRadix b.exp (sh : Int)]
      calc
        (signedMant a.sign a.mant : ℝ) *
            (neuralBpow binaryRadix b.exp * neuralBpow binaryRadix (sh : Int)) =
            ((signedMant a.sign a.mant : ℝ) * neuralBpow binaryRadix (sh : Int)) *
              neuralBpow binaryRadix b.exp := by
              ring
        _ = (m1 : ℝ) * neuralBpow binaryRadix b.exp := by
            have hm1 : (m1 : ℝ) = (signedMant a.sign a.mant : ℝ) * neuralBpow binaryRadix (sh :
              Int) := by
              simpa [m1] using (signedMant_shiftLeft (sign := a.sign) (m := a.mant) (sh := sh))
            simp [hm1]

    have hsum : dyadicToReal a + dyadicToReal b = (s : ℝ) * neuralBpow binaryRadix b.exp := by
      rw [ha', hb']
      have hfactor :
          (m1 : ℝ) * neuralBpow binaryRadix b.exp + (m2 : ℝ) * neuralBpow binaryRadix b.exp =
            ((m1 : ℝ) + (m2 : ℝ)) * neuralBpow binaryRadix b.exp := by
        simpa using (add_mul (m1 : ℝ) (m2 : ℝ) (neuralBpow binaryRadix b.exp)).symm
      have hcast : ((m1 : ℝ) + (m2 : ℝ)) = (s : ℝ) := by
        simp [s, Int.cast_add]
      calc
        (m1 : ℝ) * neuralBpow binaryRadix b.exp + (m2 : ℝ) * neuralBpow binaryRadix b.exp =
            ((m1 : ℝ) + (m2 : ℝ)) * neuralBpow binaryRadix b.exp := hfactor
        _ = (s : ℝ) * neuralBpow binaryRadix b.exp := by simp [hcast]

    by_cases hs0 : s = 0
    · rw [hadd]
      simp [hs0, dyadicToReal_zero, hsum]
    · have hs0b : (s == 0) = false := (beq_eq_false_iff_ne).2 hs0
      have hres :
          dyadicToReal { sign := decide (s < 0), mant := Int.natAbs s, exp := b.exp } =
            (s : ℝ) * neuralBpow binaryRadix b.exp := by
        simpa using (dyadicToReal_ofNatAbs (s := s) (e := b.exp))
      rw [hadd]
      simp [hs0b, hres, hsum]

/--
For a nonzero dyadic, `neural_magnitude` matches the expected “power-of-two interval”
characterization: `mag = ⌊logb 2 (mant)⌋ + exp + 1`.

We use this as a key link between the `FP32` rounding model (defined using `neural_magnitude`) and
the executable kernel (which naturally computes `Nat.log2 mant + exp` from the decoded dyadic).
-/
theorem neural_magnitude_dyadic (d : Dyadic) (hm : d.mant ≠ 0) :
    neuralMagnitude binaryRadix (dyadicToReal d) =
      (Int.ofNat (Nat.log 2 d.mant)) + d.exp + 1 := by
  have hx : dyadicToReal d ≠ 0 := by
    have hs : (if d.sign then (-1 : ℝ) else (1 : ℝ)) ≠ 0 := by
      by_cases h : d.sign <;> simp [h]
    have hm' : (d.mant : ℝ) ≠ 0 := by
      exact_mod_cast hm
    have hb : neuralBpow binaryRadix d.exp ≠ 0 := neuralBpow.ne_zero binaryRadix d.exp
    -- `s * mant * 2^exp ≠ 0`.
    have : (if d.sign then (-1 : ℝ) else (1 : ℝ)) * (d.mant : ℝ) * neuralBpow binaryRadix d.exp ≠
      0 := by
      exact mul_ne_zero (mul_ne_zero hs hm') hb
    simpa [dyadicToReal, mul_assoc] using this

  -- Expand `neural_magnitude` and rewrite the log ratio as `Real.logb`.
  simp [TorchLean.Floats.neuralMagnitude, hx, Real.log_div_log, abs_dyadicToReal d]

  -- Use `logb_mul` to split `logb 2 (mant * 2^exp)` as `logb 2 mant + logb 2 (2^exp)`.
  have hmpos : (0 : ℝ) < (d.mant : ℝ) := by
    have : 0 < d.mant := Nat.pos_of_ne_zero hm
    exact_mod_cast this
  have hbpos : (0 : ℝ) < neuralBpow binaryRadix d.exp := neuralBpow.pos binaryRadix d.exp

  have hlogb_mul :
      Real.logb (binaryRadix.toReal) ((d.mant : ℝ) * neuralBpow binaryRadix d.exp) =
        Real.logb (binaryRadix.toReal) (d.mant : ℝ) +
        Real.logb (binaryRadix.toReal) (neuralBpow binaryRadix d.exp) := by
    -- `logb_mul` needs nonzero arguments.
    have hm0 : (d.mant : ℝ) ≠ 0 := (ne_of_gt hmpos)
    have hb0 : neuralBpow binaryRadix d.exp ≠ 0 := (ne_of_gt hbpos)
    simpa [binaryRadix, NeuralRadix.toReal] using
      (Real.logb_mul (b := (binaryRadix.toReal)) (x := (d.mant : ℝ)) (y := neuralBpow
        binaryRadix d.exp) hm0 hb0)

  -- `logb 2 (2^e) = e`.
  have hlogb_bpow : Real.logb (binaryRadix.toReal) (neuralBpow binaryRadix d.exp) = (d.exp : ℝ)
    := by
    -- `logb 2 (2^e) = e` using `Real.log_zpow`.
    have hlog2 : Real.log (2 : ℝ) ≠ 0 := by
      have h2 : (2 : ℝ) ≠ 0 := by norm_num
      have h21 : (2 : ℝ) ≠ 1 := by norm_num
      have h2m1 : (2 : ℝ) ≠ -1 := by norm_num
      exact Real.log_ne_zero.mpr ⟨h2, h21, h2m1⟩
    -- Unfold `neural_bpow` and reduce to division by `log 2`.
    simp [Real.logb, neuralBpow, binaryRadix, NeuralRadix.toReal, Real.log_zpow, hlog2]

  -- Now take floors: `floor (a + z) = floor a + z`.
  -- First rewrite the base to `2` (a Nat) so we can use `Real.floor_logb_natCast`.
  have hb2 : (binaryRadix.toReal) = (2 : ℝ) := by rfl
  -- Reduce to `⌊logb 2 mant⌋`.
  have hfloor_mant :
      ⌊Real.logb (binaryRadix.toReal) (d.mant : ℝ)⌋ = Int.ofNat (Nat.log 2 d.mant) := by
    have hr : (0 : ℝ) ≤ (d.mant : ℝ) := Nat.cast_nonneg _
    -- `⌊logb 2 n⌋ = Int.log 2 n = Nat.log 2 n`
    have h :=
      (Real.floor_logb_natCast (b := 2) (r := (d.mant : ℝ)) hr)
    -- Rewrite base `binary_radix.to_real` to `2`, then simplify `Int.log` on a nat cast.
    simpa [hb2, Int.log_natCast] using h

  -- Put it together.
  -- `floor (logb 2 (mant*2^exp)) = floor (logb 2 mant + exp) = floor(logb2 mant) + exp`.
  have hfloor_total :
      ⌊Real.logb (binaryRadix.toReal) ((d.mant : ℝ) * neuralBpow binaryRadix d.exp)⌋ =
        Int.ofNat (Nat.log 2 d.mant) + d.exp := by
    -- rewrite using `hlogb_mul` and `hlogb_bpow`
    rw [hlogb_mul, hlogb_bpow]
    -- `floor (a + z) = floor a + z`
    simp [hfloor_mant]

  -- `simp` reduced the goal to a statement about the floored `logb`; discharge it.
  exact hfloor_total

lemma neural_magnitude_eq_of_bpow_bounds (x : ℝ) (k : Int)
    (hx0 : x ≠ 0)
    (hlo : neuralBpow binaryRadix k ≤ _root_.abs x)
    (hhi : _root_.abs x < neuralBpow binaryRadix (k + 1)) :
    neuralMagnitude binaryRadix x = k + 1 := by
  have hxabs : 0 < _root_.abs x := abs_pos.mpr hx0
  have hb : (1 : ℝ) < (2 : ℝ) := by norm_num

  have hlo_z : (2 : ℝ) ^ (k : Int) ≤ _root_.abs x := by
    simpa [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal] using hlo
  have hlo_r : (2 : ℝ) ^ (k : ℝ) ≤ _root_.abs x := by
    calc
      (2 : ℝ) ^ (k : ℝ) = (2 : ℝ) ^ (k : Int) := by
        simp
      _ ≤ _root_.abs x := hlo_z

  have hhi_z : _root_.abs x < (2 : ℝ) ^ (k + 1 : Int) := by
    simpa [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal] using hhi
  have hhi_r : _root_.abs x < (2 : ℝ) ^ ((k + 1 : Int) : ℝ) := by
    calc
      _root_.abs x < (2 : ℝ) ^ (k + 1 : Int) := hhi_z
      _ = (2 : ℝ) ^ ((k + 1 : Int) : ℝ) := by
        simpa using (Real.rpow_intCast (2 : ℝ) (k + 1)).symm

  have hlo' : (k : ℝ) ≤ Real.logb (2 : ℝ) (_root_.abs x) :=
    (Real.le_logb_iff_rpow_le (b := (2 : ℝ)) (x := (k : ℝ)) (y := _root_.abs x) hb hxabs).2 hlo_r
  have hhi' : Real.logb (2 : ℝ) (_root_.abs x) < ((k + 1 : Int) : ℝ) :=
    (Real.logb_lt_iff_lt_rpow (b := (2 : ℝ)) (x := _root_.abs x) (y := ((k + 1 : Int) : ℝ)) hb
      hxabs).2 hhi_r

  have hfloor : (⌊Real.logb (2 : ℝ) (_root_.abs x)⌋ : Int) = k := by
    have hhi'' : Real.logb (2 : ℝ) (_root_.abs x) < (k : ℝ) + 1 := by
      simpa [Int.cast_add, Int.cast_one] using hhi'
    exact (Int.floor_eq_iff).2 ⟨hlo', hhi''⟩

  unfold TorchLean.Floats.neuralMagnitude
  rw [if_neg hx0]
  have hfloor_ratio_abs :
      (⌊Real.log (_root_.abs x) / Real.log (binaryRadix.toReal)⌋ : Int) = k := by
    simpa [Real.logb, binaryRadix, NeuralRadix.toReal] using hfloor
  have hfloor_ratio :
      (⌊Real.log x / Real.log (binaryRadix.toReal)⌋ : Int) = k := by
    simpa [Real.log_abs] using hfloor_ratio_abs
  simp [hfloor_ratio]

/-
## Nearest-even on rationals (core bridge lemma)

To relate the executable kernel’s integer rounding (`roundQuotEven` / `roundShiftRightEven`) to the
proof-relevant model’s rounding (`neural_nearest_even`), we need a lemma that computes nearest-even
rounding of a nonnegative rational `num/den` using Euclidean division.
-/

lemma fract_real_div_nat (n den : Nat) (hden : den ≠ 0) :
    (n : ℝ) / (den : ℝ) - ((n / den : Nat) : ℝ) = ((n % den : Nat) : ℝ) / (den : ℝ) := by
  have hdiv : den * (n / den) + n % den = n := Nat.div_add_mod n den
  have hdivR : ((den * (n / den) : Nat) : ℝ) + ((n % den : Nat) : ℝ) = (n : ℝ) := by
    exact_mod_cast hdiv
  have hdenR : (den : ℝ) ≠ 0 := by exact_mod_cast hden
  have hsplit :
      (n : ℝ) / (den : ℝ) =
        ((n / den : Nat) : ℝ) + ((n % den : Nat) : ℝ) / (den : ℝ) := by
    have := congrArg (fun t : ℝ => t / (den : ℝ)) hdivR
    -- `simp` knows `((den*q)/den) = q` given `hdenR`.
    simp [add_div, hdenR] at this
    simpa using this.symm
  calc
    (n : ℝ) / (den : ℝ) - ((n / den : Nat) : ℝ)
        = (((n / den : Nat) : ℝ) + ((n % den : Nat) : ℝ) / (den : ℝ)) - ((n / den : Nat) : ℝ) := by
            simp [hsplit]
    _ = ((n % den : Nat) : ℝ) / (den : ℝ) := by ring

lemma floor_real_nat_div (n den : Nat) :
    (⌊(n : ℝ) / (den : ℝ)⌋ : Int) = (n / den : Nat) := by
  -- Reduce to `Rat` where the floor/div lemma exists, then cast back to `ℝ`.
  calc
    (⌊(n : ℝ) / (den : ℝ)⌋ : Int)
        = (⌊(((n : ℚ) / (den : ℚ)) : ℝ)⌋ : Int) := by
            simp
    _ = (⌊((n : ℚ) / (den : ℚ))⌋ : Int) := by
            simpa using (Rat.floor_cast (α := ℝ) ((n : ℚ) / (den : ℚ)))
    _ = (n / den : Nat) := by
            simpa using (Rat.floor_natCast_div_natCast n den)

lemma div_lt_half_iff (r den : Nat) (hden : den ≠ 0) :
    ((r : ℝ) / (den : ℝ) < (2⁻¹ : ℝ)) ↔ (2 * r < den) := by
  have hdenpos : (0 : ℝ) < (den : ℝ) := by
    exact_mod_cast (Nat.pos_of_ne_zero hden)
  have h2pos : (0 : ℝ) < (2 : ℝ) := by norm_num
  have h1 :
      ((r : ℝ) / (den : ℝ) < (2⁻¹ : ℝ)) ↔ (r : ℝ) < (2⁻¹ : ℝ) * (den : ℝ) := by
    simpa using (div_lt_iff₀ hdenpos)
  have hscale : (2 : ℝ) * ((2⁻¹ : ℝ) * (den : ℝ)) = (den : ℝ) := by
    calc
      (2 : ℝ) * ((2⁻¹ : ℝ) * (den : ℝ))
          = ((2 : ℝ) * (2⁻¹ : ℝ)) * (den : ℝ) := by simp []
      _ = (1 : ℝ) * (den : ℝ) := by
          have h : (2 : ℝ) * (2⁻¹ : ℝ) = (1 : ℝ) := by
            simp
          simp []
      _ = (den : ℝ) := by simp
  constructor
  · intro h
    have hr : (r : ℝ) < (2⁻¹ : ℝ) * (den : ℝ) := (h1.mp h)
    have hmul : (2 : ℝ) * (r : ℝ) < (2 : ℝ) * ((2⁻¹ : ℝ) * (den : ℝ)) :=
      mul_lt_mul_of_pos_left hr h2pos
    have hmul' : (2 : ℝ) * (r : ℝ) < (den : ℝ) := by simpa [hscale] using hmul
    have : ((2 * r : Nat) : ℝ) < (den : ℝ) := by
      simpa [Nat.cast_mul, Nat.cast_ofNat, mul_assoc, mul_left_comm, mul_comm] using hmul'
    exact (by exact_mod_cast this)
  · intro h
    have hR : ((2 * r : Nat) : ℝ) < (den : ℝ) := by exact_mod_cast h
    have hmul : (2 : ℝ) * (r : ℝ) < (den : ℝ) := by
      simpa [Nat.cast_mul, Nat.cast_ofNat, mul_assoc, mul_left_comm, mul_comm] using hR
    have hmul' : (2 : ℝ) * (r : ℝ) < (2 : ℝ) * ((2⁻¹ : ℝ) * (den : ℝ)) := by
      simpa [hscale] using hmul
    have hr : (r : ℝ) < (2⁻¹ : ℝ) * (den : ℝ) := (mul_lt_mul_iff_right₀ h2pos).1 hmul'
    exact h1.mpr hr

lemma half_lt_div_iff (r den : Nat) (hden : den ≠ 0) :
    ((2⁻¹ : ℝ) < (r : ℝ) / (den : ℝ)) ↔ (den < 2 * r) := by
  have hdenpos : (0 : ℝ) < (den : ℝ) := by
    exact_mod_cast (Nat.pos_of_ne_zero hden)
  have h2pos : (0 : ℝ) < (2 : ℝ) := by norm_num
  have h1 :
      ((2⁻¹ : ℝ) < (r : ℝ) / (den : ℝ)) ↔ (2⁻¹ : ℝ) * (den : ℝ) < (r : ℝ) := by
    simpa using (lt_div_iff₀ hdenpos)
  have hscale : (2 : ℝ) * ((2⁻¹ : ℝ) * (den : ℝ)) = (den : ℝ) := by
    calc
      (2 : ℝ) * ((2⁻¹ : ℝ) * (den : ℝ))
          = ((2 : ℝ) * (2⁻¹ : ℝ)) * (den : ℝ) := by simp []
      _ = (1 : ℝ) * (den : ℝ) := by
          have h : (2 : ℝ) * (2⁻¹ : ℝ) = (1 : ℝ) := by
            simp
          simp []
      _ = (den : ℝ) := by simp
  constructor
  · intro h
    have hr : (2⁻¹ : ℝ) * (den : ℝ) < (r : ℝ) := (h1.mp h)
    have hmul : (2 : ℝ) * ((2⁻¹ : ℝ) * (den : ℝ)) < (2 : ℝ) * (r : ℝ) :=
      mul_lt_mul_of_pos_left hr h2pos
    have hmul' : (den : ℝ) < (2 : ℝ) * (r : ℝ) := by simpa [hscale] using hmul
    have : (den : ℝ) < ((2 * r : Nat) : ℝ) := by
      simpa [Nat.cast_mul, Nat.cast_ofNat, mul_assoc, mul_left_comm, mul_comm] using hmul'
    exact (by exact_mod_cast this)
  · intro h
    have hR : (den : ℝ) < ((2 * r : Nat) : ℝ) := by exact_mod_cast h
    have hmul : (den : ℝ) < (2 : ℝ) * (r : ℝ) := by
      simpa [Nat.cast_mul, Nat.cast_ofNat, mul_assoc, mul_left_comm, mul_comm] using hR
    have hmul' : (2 : ℝ) * ((2⁻¹ : ℝ) * (den : ℝ)) < (2 : ℝ) * (r : ℝ) := by
      simpa [hscale] using hmul
    have hr : (2⁻¹ : ℝ) * (den : ℝ) < (r : ℝ) := (mul_lt_mul_iff_right₀ h2pos).1 hmul'
    exact h1.mpr hr

lemma neural_nearest_even_div_eq_roundQuotEven (num den : Nat) (hden : den ≠ 0) :
    TorchLean.Floats.neuralNearestEven ((num : ℝ) / (den : ℝ)) =
      Int.ofNat (roundQuotEven num den) := by
  classical
  set q : Nat := num / den
  set r : Nat := num % den
  have hfloor : (⌊((num : ℝ) / (den : ℝ))⌋ : Int) = q := by
    simpa [q] using (floor_real_nat_div (n := num) (den := den))
  have hfract : ((num : ℝ) / (den : ℝ)) - (q : ℝ) = (r : ℝ) / (den : ℝ) := by
    simpa [q, r] using (fract_real_div_nat (n := num) (den := den) hden)

  unfold TorchLean.Floats.neuralNearestEven
  simp [hfloor]
  rw [hfract]
  simp [roundQuotEven, q, r]

  have hlt :
      (((num % den : Nat) : ℝ) / (den : ℝ) < (2⁻¹ : ℝ)) ↔ (2 * (num % den) < den) := by
    simpa using (div_lt_half_iff (r := num % den) (den := den) hden)
  have hgt :
      ((2⁻¹ : ℝ) < ((num % den : Nat) : ℝ) / (den : ℝ)) ↔ (den < 2 * (num % den)) := by
    simpa using (half_lt_div_iff (r := num % den) (den := den) hden)

  by_cases h2lt : (2 * (num % den) < den)
  · have hrlt : (((num % den : Nat) : ℝ) / (den : ℝ) < (2⁻¹ : ℝ)) := (hlt.mpr h2lt)
    simp [h2lt, hrlt]
  · have hrlt : ¬(((num % den : Nat) : ℝ) / (den : ℝ) < (2⁻¹ : ℝ)) := by
      intro hr; exact h2lt (hlt.mp hr)
    by_cases h2gt : (den < 2 * (num % den))
    · have hrgt : ((2⁻¹ : ℝ) < ((num % den : Nat) : ℝ) / (den : ℝ)) := (hgt.mpr h2gt)
      simp [h2lt, hrlt, h2gt, hrgt]
    · have hrgt : ¬((2⁻¹ : ℝ) < ((num % den : Nat) : ℝ) / (den : ℝ)) := by
        intro hr; exact h2gt (hgt.mp hr)
      simp [h2lt, hrlt, h2gt, hrgt, Nat.even_iff]
end IEEE32Exec

end TorchLean.Floats.IEEE754
