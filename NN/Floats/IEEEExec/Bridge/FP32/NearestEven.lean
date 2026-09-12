/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Bridge.FP32.DyadicRounding

/-!
# IEEE32Exec and FP32: Nearest-Even Quotient Lemmas
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

/-!
## Signed nearest-even (and power-of-two specialization)

`FP32` uses “round to nearest, ties to even” (IEEE 754's default). The executable kernel implements
the same policy, but at the level of integer arithmetic on mantissas.

The lemmas below establish basic algebraic properties of nearest-even rounding that we can reuse
later when relating the `IEEE32Exec` rounding code to `fp32Round`.
-/

/-- Nearest-even integer rounding commutes with negation. -/
lemma neural_nearest_even_neg (x : ℝ) :
    TorchLean.Floats.neuralNearestEven (-x) = -TorchLean.Floats.neuralNearestEven x := by
  classical
  -- Use `r = x - ⌊x⌋` for the case split (integer vs non-integer).
  set r : ℝ := x - (⌊x⌋ : ℝ)
  by_cases hr0 : r = 0
  · -- Integer case: both `x` and `-x` have fractional part `0`, so both round to their floors.
    have hx_eq : x = (⌊x⌋ : ℝ) := by linarith [hr0]
    have hceil : ⌈x⌉ = ⌊x⌋ := by
      have hx_le : x ≤ (⌊x⌋ : ℝ) := by linarith [hx_eq]
      have hceil_le : ⌈x⌉ ≤ ⌊x⌋ := (Int.ceil_le).2 hx_le
      exact le_antisymm hceil_le (Int.floor_le_ceil x)
    have hfloor_neg : ⌊-x⌋ = -⌊x⌋ := by
      calc
        ⌊-x⌋ = -⌈x⌉ := by simpa using (Int.floor_neg (a := x))
        _ = -⌊x⌋ := by simp [hceil]
    have hx_lt_half : x - (⌊x⌋ : ℝ) < (1 / 2 : ℝ) := by
      -- `x - ⌊x⌋ = 0`.
      simpa [r] using (by linarith [hr0] : r < (1 / 2 : ℝ))
    have hneg_lt_half : (-x) - (⌊-x⌋ : ℝ) < (1 / 2 : ℝ) := by
      have hcast : ((↑(-⌊x⌋) : ℝ)) = -((⌊x⌋ : ℤ) : ℝ) := by
        -- `Int.cast_neg` with explicit result type.
        simp
      have : (-x) - (⌊-x⌋ : ℝ) = 0 := by
        rw [hfloor_neg]
        rw [hcast]
        ring_nf
        linarith [hx_eq]
      linarith [this]
    have hx_round :
        TorchLean.Floats.neuralNearestEven x = ⌊x⌋ :=
      TorchLean.Floats.neural_nearest_even_eq_floor_of_frac_lt_half x hx_lt_half
    have hy_round :
        TorchLean.Floats.neuralNearestEven (-x) = ⌊-x⌋ :=
      TorchLean.Floats.neural_nearest_even_eq_floor_of_frac_lt_half (-x) hneg_lt_half
    -- Replace `⌊-x⌋` with `-⌊x⌋` and finish.
    simp [hx_round, hy_round, hfloor_neg]
  · -- Non-integer case: `⌈x⌉ = ⌊x⌋ + 1` and `⌊-x⌋ = -⌊x⌋ - 1`, and `fract(-x) = 1 - r`.
    have hx_ne : (⌊x⌋ : ℝ) ≠ x := by
      intro hx_eq
      apply hr0
      have : x - (⌊x⌋ : ℝ) = 0 := by linarith [hx_eq]
      simpa [r] using this
    have hx_lt : (⌊x⌋ : ℝ) < x :=
      lt_of_le_of_ne (Int.floor_le x) hx_ne
    have hceil : ⌈x⌉ = ⌊x⌋ + 1 := by
      have hx_le' : x ≤ ((⌊x⌋ + 1 : ℤ) : ℝ) := by
        have hx_lt_add : x < (⌊x⌋ : ℝ) + 1 := Int.lt_floor_add_one x
        have : x < ((⌊x⌋ : ℤ) : ℝ) + 1 := by simp
        have : x < ((⌊x⌋ + 1 : ℤ) : ℝ) := by
          simp [Int.cast_add, Int.cast_one]
        exact le_of_lt this
      apply (Int.ceil_eq_iff).2
      constructor
      · -- `((⌊x⌋+1):ℝ) - 1 = ⌊x⌋ < x`
        have : ((⌊x⌋ : ℤ) : ℝ) < x := hx_lt
        simpa [Int.cast_add, Int.cast_one, sub_eq_add_neg, add_assoc] using this
      · simpa using hx_le'
    have hfloor_neg : ⌊-x⌋ = -⌊x⌋ - 1 := by
      calc
        ⌊-x⌋ = -⌈x⌉ := by simpa using (Int.floor_neg (a := x))
        _ = -(⌊x⌋ + 1) := by simp [hceil]
        _ = -⌊x⌋ - 1 := by ring
    have hfract_neg : (-x) - (⌊-x⌋ : ℝ) = 1 - r := by
      -- Expand `⌊-x⌋ = -⌊x⌋ - 1`, then normalize.
      rw [hfloor_neg]
      have hcast : ((-⌊x⌋ - 1 : ℤ) : ℝ) = -((⌊x⌋ : ℤ) : ℝ) - 1 := by
        simp [Int.cast_neg, Int.cast_one]
      rw [hcast]
      dsimp [r]
      -- If `Int.fract` appears, unfold it explicitly (avoid `simp [Int.fract]` loops).
      rw [Int.fract]
      ring

    by_cases hlt : r < (1 / 2 : ℝ)
    · -- `r < 1/2`: `x` rounds down, `-x` rounds up.
      have hx_round :
          TorchLean.Floats.neuralNearestEven x = ⌊x⌋ := by
        have : x - (⌊x⌋ : ℝ) < (1 / 2 : ℝ) := by simpa [r] using hlt
        exact TorchLean.Floats.neural_nearest_even_eq_floor_of_frac_lt_half x this
      have hneg_gt : (-x) - (⌊-x⌋ : ℝ) > (1 / 2 : ℝ) := by
        -- `1 - r > 1/2`.
        have : (1 / 2 : ℝ) < 1 - r := by linarith [hlt]
        -- rewrite using `hfract_neg`
        have : (1 / 2 : ℝ) < (-x) - (⌊-x⌋ : ℝ) := by simpa [hfract_neg] using this
        linarith
      have hy_round :
          TorchLean.Floats.neuralNearestEven (-x) = ⌊-x⌋ + 1 :=
        TorchLean.Floats.neural_nearest_even_eq_ceil_of_frac_gt_half (-x) hneg_gt
      have hfloor_succ : ⌊-x⌋ + 1 = -⌊x⌋ := by linarith [hfloor_neg]
      simp [hx_round, hy_round, hfloor_succ]
    · by_cases hgt : r > (1 / 2 : ℝ)
      · -- `r > 1/2`: `x` rounds up, `-x` rounds down.
        have hx_round :
            TorchLean.Floats.neuralNearestEven x = ⌊x⌋ + 1 := by
          have : x - (⌊x⌋ : ℝ) > (1 / 2 : ℝ) := by simpa [r] using hgt
          exact TorchLean.Floats.neural_nearest_even_eq_ceil_of_frac_gt_half x this
        have hneg_lt : (-x) - (⌊-x⌋ : ℝ) < (1 / 2 : ℝ) := by
          have : 1 - r < (1 / 2 : ℝ) := by linarith [hgt]
          simpa [hfract_neg] using this
        have hy_round :
            TorchLean.Floats.neuralNearestEven (-x) = ⌊-x⌋ :=
          TorchLean.Floats.neural_nearest_even_eq_floor_of_frac_lt_half (-x) hneg_lt
        -- Reduce to an integer identity via `hfloor_neg`.
        rw [hy_round, hx_round, hfloor_neg]
        ring
      · -- Tie: `r = 1/2`; reduce to parity of the floor.
        have hr_eq : r = (1 / 2 : ℝ) := by
          have hge : (1 / 2 : ℝ) ≤ r := le_of_not_gt hlt
          have hle : r ≤ (1 / 2 : ℝ) := le_of_not_gt (by intro h; exact hgt h)
          exact le_antisymm hle hge
        have hneg_eq : (-x) - (⌊-x⌋ : ℝ) = (1 / 2 : ℝ) := by
          have : 1 - r = (1 / 2 : ℝ) := by linarith [hr_eq]
          simp [hfract_neg, this]

        have heven_floor_neg : Even (⌊-x⌋) ↔ ¬Even (⌊x⌋) := by
          -- `⌊-x⌋ = -(⌊x⌋ + 1)` and parity toggles under `+1`.
          have h1 : ⌊-x⌋ = -(⌊x⌋ + 1) := by linarith [hfloor_neg]
          -- `Even (-(a)) ↔ Even a` and `Even (a+1) ↔ ¬Even a`.
          have : Even (-(⌊x⌋ + 1)) ↔ ¬Even (⌊x⌋) := by
            exact (Iff.trans (even_neg (a := (⌊x⌋ + 1 : ℤ))) (Int.even_add_one (n := ⌊x⌋)))
          simpa [h1] using this

        by_cases hf : Even (⌊x⌋)
        · -- `x` rounds to `⌊x⌋`, `-x` rounds to `⌊-x⌋ + 1 = -⌊x⌋`.
          have hx_round :
              TorchLean.Floats.neuralNearestEven x = ⌊x⌋ := by
            have : x - (⌊x⌋ : ℝ) = (1 / 2 : ℝ) := by simpa [r] using hr_eq
            exact TorchLean.Floats.neural_nearest_even_eq_floor_of_frac_half_even x this hf
          have hfloor_odd : ¬Even (⌊-x⌋) := by
            intro hef
            have : ¬Even (⌊x⌋) := (heven_floor_neg.mp hef)
            exact this hf
          have hy_round :
              TorchLean.Floats.neuralNearestEven (-x) = ⌊-x⌋ + 1 :=
            TorchLean.Floats.neural_nearest_even_eq_ceil_of_frac_half_odd (-x) hneg_eq hfloor_odd
          have hfloor_succ : ⌊-x⌋ + 1 = -⌊x⌋ := by linarith [hfloor_neg]
          simp [hx_round, hy_round, hfloor_succ]
        · -- `x` rounds to `⌊x⌋+1`, `-x` rounds to `⌊-x⌋ = -⌊x⌋-1`.
          have hx_round :
              TorchLean.Floats.neuralNearestEven x = ⌊x⌋ + 1 := by
            have : x - (⌊x⌋ : ℝ) = (1 / 2 : ℝ) := by simpa [r] using hr_eq
            exact TorchLean.Floats.neural_nearest_even_eq_ceil_of_frac_half_odd x this hf
          have hfloor_even : Even (⌊-x⌋) := by
            exact (heven_floor_neg.mpr hf)
          have hy_round :
              TorchLean.Floats.neuralNearestEven (-x) = ⌊-x⌋ :=
            TorchLean.Floats.neural_nearest_even_eq_floor_of_frac_half_even (-x) hneg_eq hfloor_even
          -- `-(⌊x⌋+1) = -⌊x⌋-1`.
          rw [hy_round, hx_round, hfloor_neg]
          ring

/-- Same agreement for negative quotients, obtained from the nonnegative case by oddness.

Nearest-even is symmetric about zero, so the negative branch needs no separate tie analysis. -/
lemma neural_nearest_even_neg_div_eq_roundQuotEven (num den : Nat) (hden : den ≠ 0) :
    TorchLean.Floats.neuralNearestEven (-((num : ℝ) / (den : ℝ))) =
      -Int.ofNat (roundQuotEven num den) := by
  have hpos :=
    neural_nearest_even_div_eq_roundQuotEven (num := num) (den := den) hden
  -- Use oddness and then the nonnegative rational lemma.
  have hodd :=
    neural_nearest_even_neg (x := (num : ℝ) / (den : ℝ))
  -- `neural_nearest_even (-x) = -neural_nearest_even x`.
  simpa [hpos] using hodd

/-- A rounding right shift is division by a power of two, both rounded to nearest even. This is what
lets the hardware-shaped implementation be verified against the arithmetic specification. -/
lemma roundShiftRightEven_eq_roundQuotEven_pow2 (n shift : Nat) :
    roundShiftRightEven n shift = roundQuotEven n (pow2 shift) := by
  classical
  cases shift with
  | zero =>
      -- `pow2 0 = 1`, and `roundQuotEven n 1 = n`.
      simp [roundShiftRightEven, pow2, roundQuotEven, Nat.div_one, Nat.mod_one]
  | succ s =>
      -- Abbreviations.
      let den : Nat := pow2 (Nat.succ s)
      let half : Nat := pow2 s
      have hden : den = 2 * half := by
        -- `2^(s+1) = 2 * 2^s`.
        simp [den, half, pow2_eq_two_pow, Nat.pow_succ, Nat.mul_comm]

      -- `Nat.shiftRight`/`Nat.shiftLeft` simp to `>>>`/`<<<`, so work in that notation.
      have hq : n >>> (Nat.succ s) = n / den := by
        -- `n >>> k = n / 2^k` and `den = 2^k`.
        have : n >>> (Nat.succ s) = n / (2 ^ Nat.succ s) := Nat.shiftRight_eq_div_pow n (Nat.succ s)
        simpa [den, pow2_eq_two_pow] using this

      have hrem : n - (n >>> (Nat.succ s) <<< (Nat.succ s)) = n % den := by
        -- Replace shifts with `/` and `*`, then use the division algorithm.
        have hshiftLeft : (n >>> (Nat.succ s) <<< (Nat.succ s)) = (n / den) * den := by
          -- `a <<< k = a * 2^k`.
          simp [Nat.shiftLeft_eq, hq, den, pow2_eq_two_pow, Nat.mul_comm]
        rw [hshiftLeft]
        -- `((n/den)*den) + (n%den) = n`.
        have hdiv : (n / den) * den + n % den = n := by
          simpa [Nat.mul_comm] using (Nat.div_add_mod n den)
        calc
          n - (n / den) * den = ((n / den) * den + n % den) - (n / den) * den := by simp [hdiv]
          _ = n % den := Nat.add_sub_cancel_left _ _

      -- Now both rounders are the same case analysis, just phrased differently.
      have hshift_def :
          roundShiftRightEven n (Nat.succ s) =
            (let q := n / den
             let r := n % den
             if r < half then q
             else if half < r then q + 1
             else if q % 2 == 0 then q else q + 1) := by
        have hrem_div : n - (n / den) <<< (Nat.succ s) = n % den := by
          -- `simp` will rewrite `n >>> _` into `n / den`, so rewrite `hrem` first.
          simpa [hq] using hrem
        -- Unfold and rewrite `q`/`rem`.
        simp [roundShiftRightEven, den, half, hq, hrem_div]

      have hquot_def :
          roundQuotEven n den =
            (let q := n / den
             let r := n % den
             let twice := 2 * r
             if twice < den then q
             else if den < twice then q + 1
             else if q % 2 == 0 then q else q + 1) := by
        simp [roundQuotEven, den]

      -- Finish by splitting on `r` relative to `half`.
      -- `den = 2 * half`, so comparisons against `half` match comparisons of `2*r` against `den`.
      -- Expand both sides and do a 3-way case split.
      rw [hshift_def]
      -- Rewrite RHS goal to use `roundQuotEven n den`.
      -- `pow2 (succ s)` is `den` by definition.
      have : roundQuotEven n (pow2 (Nat.succ s)) = roundQuotEven n den := by rfl
      -- Unfold `roundQuotEven` with `den`.
      rw [this, hquot_def]

      -- Now compare the conditionals.
      -- Let-bindings: keep `q`/`r` as in both sides.
      -- A small local `simp` step exposes the shared structure.
      simp only
      -- Case split on the remainder.
      by_cases hrlt : n % den < half
      · -- `r < half` ⇒ `2*r < den`.
        have htw_lt : 2 * (n % den) < den := by
          -- `2*r < 2*half`.
          have : 2 * (n % den) < 2 * half :=
            (Nat.mul_lt_mul_left (by decide : 0 < (2 : Nat))).2 hrlt
          simpa [hden, Nat.mul_assoc] using this
        simp [hrlt, htw_lt]
      · by_cases hrgt : half < n % den
        · -- `half < r` ⇒ `den < 2*r`.
          have htw_gt : den < 2 * (n % den) := by
            have : 2 * half < 2 * (n % den) :=
              (Nat.mul_lt_mul_left (by decide : 0 < (2 : Nat))).2 hrgt
            -- Rewrite `2*half` as `den`.
            exact lt_of_eq_of_lt hden this
          have htw_lt' : ¬(2 * (n % den) < den) := by
            intro h; exact (not_lt_of_ge (le_of_lt htw_gt)) h
          simp [hrlt, hrgt, htw_lt', htw_gt]
        · -- Tie: `r = half`, so `2*r = den` and both use the parity branch.
          have hre : n % den = half := by
            exact le_antisymm (le_of_not_gt hrgt) (le_of_not_gt hrlt)
          have hre₂ : n % (2 * half) = half := by
            -- Useful because simp will rewrite `den` using `hden`.
            simpa [hden] using hre
          have htw_eq : 2 * (n % den) = den := by
            -- `2*half = den`
            calc
              2 * (n % den) = 2 * half := by simp [hre]
              _ = den := hden.symm
          have htw_lt' : ¬(2 * (n % den) < den) := by
            simp [htw_eq]
          have htw_gt' : ¬(den < 2 * (n % den)) := by
            simp [htw_eq]
          -- Give `simp` the rewritten remainder for `den = 2*half`.
          simp [hrlt, hrgt, htw_lt', htw_gt']

/-- Composite of the previous two: the spec rounding of a division by `2 ^ shift` is exactly the
executable rounding shift. -/
lemma neural_nearest_even_div_pow2_eq_roundShiftRightEven (num shift : Nat) :
    TorchLean.Floats.neuralNearestEven ((num : ℝ) / (pow2 shift : ℝ)) =
      Int.ofNat (roundShiftRightEven num shift) := by
  have hden : pow2 shift ≠ 0 := by
    have : 0 < pow2 shift := by
      simp [pow2_eq_two_pow]
    exact Nat.ne_of_gt this
  -- Reduce to the generic rational lemma via `roundQuotEven`, then rewrite to
  -- `roundShiftRightEven`.
  have h :=
    neural_nearest_even_div_eq_roundQuotEven (num := num) (den := pow2 shift) hden
  simpa [roundShiftRightEven_eq_roundQuotEven_pow2 (n := num) (shift := shift)] using h

/-- Nearest-even rounding of `√n` is decided by comparing the remainder `n - ⌊√n⌋²` against `⌊√n⌋`.

That comparison is exactly the midpoint test: `q + 1` wins precisely when `n` exceeds `(q + ½)²`,
and since `(q + ½)² = q² + q + ¼` the quarter never matters for integer `n`, so the tie case cannot
arise and no even-mantissa rule is needed here. -/
lemma neural_nearest_even_sqrt_nat (n : Nat) :
    TorchLean.Floats.neuralNearestEven (Real.sqrt (n : ℝ)) =
      Int.ofNat
        (let q : Nat := Nat.sqrt n
         let r : Nat := n - q * q
         if r > q then q + 1 else q) := by
  classical
  set q : Nat := Nat.sqrt n
  set r : Nat := n - q * q
  have hqle : q * q ≤ n := by
    simpa [q, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using (Nat.sqrt_le n)
  have hn_eq : n = q * q + r := by
    have : (n - q * q) + q * q = n := Nat.sub_add_cancel hqle
    calc
      n = (n - q * q) + q * q := this.symm
      _ = r + q * q := by simp [r]
      _ = q * q + r := by ac_rfl
  have hfloor : (⌊Real.sqrt (n : ℝ)⌋ : Int) = q := by
    simp [q, Real.floor_real_sqrt_eq_nat_sqrt (a := n)]

  by_cases hgt : r > q
  · have hrge : q + 1 ≤ r := Nat.succ_le_of_lt hgt
    have hn_ge : q * q + (q + 1) ≤ n := by
      have : q * q + (q + 1) ≤ q * q + r := Nat.add_le_add_left hrge (q * q)
      simpa [hn_eq] using this
    have hmid_lt : ((q : ℝ) + (1 / 2 : ℝ)) ^ 2 < (n : ℝ) := by
      have h1 : ((q : ℝ) + (1 / 2 : ℝ)) ^ 2 < (q * q + (q + 1) : ℝ) := by
        have hR : (q * q + (q + 1) : ℝ) = (q : ℝ) ^ 2 + (q : ℝ) + 1 := by
          simp [pow_two]
          ring
        have hL : ((q : ℝ) + (1 / 2 : ℝ)) ^ 2 = (q : ℝ) ^ 2 + (q : ℝ) + (1 / 4 : ℝ) := by
          ring
        rw [hL, hR]
        linarith
      have h2 : (q * q + (q + 1) : ℝ) ≤ (n : ℝ) := by
        exact_mod_cast hn_ge
      exact lt_of_lt_of_le h1 h2
    have hx0 : 0 ≤ (q : ℝ) + (1 / 2 : ℝ) := by
      have : 0 ≤ (q : ℝ) := by exact_mod_cast (Nat.zero_le q)
      linarith
    have hgt_mid : (q : ℝ) + (1 / 2 : ℝ) < Real.sqrt (n : ℝ) := by
      exact
        (Real.lt_sqrt (x := (q : ℝ) + (1 / 2 : ℝ)) (y := (n : ℝ)) hx0).2 (by
          simpa using hmid_lt)
    have hfrac_gt : Real.sqrt (n : ℝ) - (⌊Real.sqrt (n : ℝ)⌋ : ℝ) > (1 / 2 : ℝ) := by
      have hfloorR : (⌊Real.sqrt (n : ℝ)⌋ : ℝ) = (q : ℝ) := by
        have := congrArg (fun z : Int => (z : ℝ)) hfloor
        simpa using this
      have : (1 / 2 : ℝ) < Real.sqrt (n : ℝ) - (q : ℝ) := by linarith [hgt_mid]
      simpa [hfloorR] using this
    have hround :
        TorchLean.Floats.neuralNearestEven (Real.sqrt (n : ℝ)) =
          (⌊Real.sqrt (n : ℝ)⌋ : Int) + 1 :=
      TorchLean.Floats.neural_nearest_even_eq_ceil_of_frac_gt_half (Real.sqrt (n : ℝ)) (by
        simpa using hfrac_gt)
    have hLHS : TorchLean.Floats.neuralNearestEven (Real.sqrt (n : ℝ)) = Int.ofNat (q + 1) := by
      calc
        TorchLean.Floats.neuralNearestEven (Real.sqrt (n : ℝ)) = (⌊Real.sqrt (n : ℝ)⌋ : Int) + 1
          := hround
        _ = (q : Int) + 1 := by simp [hfloor]
        _ = Int.ofNat (q + 1) := by simp
    simpa [r, hgt] using hLHS
  · have hle : r ≤ q := Nat.le_of_not_gt hgt
    have hn_le : n ≤ q * q + q := by
      have : q * q + r ≤ q * q + q := Nat.add_le_add_left hle (q * q)
      simpa [hn_eq] using this
    have hmid_gt : (n : ℝ) < ((q : ℝ) + (1 / 2 : ℝ)) ^ 2 := by
      have h1 : (n : ℝ) ≤ (q * q + q : ℝ) := by
        exact_mod_cast hn_le
      have h2 : (q * q + q : ℝ) < ((q : ℝ) + (1 / 2 : ℝ)) ^ 2 := by
        have hR : (q * q + q : ℝ) = (q : ℝ) ^ 2 + (q : ℝ) := by
          simp [pow_two]
        have hL : ((q : ℝ) + (1 / 2 : ℝ)) ^ 2 = (q : ℝ) ^ 2 + (q : ℝ) + (1 / 4 : ℝ) := by
          ring
        rw [hR, hL]
        linarith
      exact lt_of_le_of_lt h1 h2
    have hy0 : 0 < (q : ℝ) + (1 / 2 : ℝ) := by
      have : 0 ≤ (q : ℝ) := by exact_mod_cast (Nat.zero_le q)
      linarith
    have hlt_mid : Real.sqrt (n : ℝ) < (q : ℝ) + (1 / 2 : ℝ) := by
      exact
        (Real.sqrt_lt' (x := (n : ℝ)) (y := (q : ℝ) + (1 / 2 : ℝ)) hy0).2 (by
          simpa using hmid_gt)
    have hfrac_lt : Real.sqrt (n : ℝ) - (⌊Real.sqrt (n : ℝ)⌋ : ℝ) < (1 / 2 : ℝ) := by
      have hfloorR : (⌊Real.sqrt (n : ℝ)⌋ : ℝ) = (q : ℝ) := by
        have := congrArg (fun z : Int => (z : ℝ)) hfloor
        simpa using this
      have : Real.sqrt (n : ℝ) - (q : ℝ) < (1 / 2 : ℝ) := by linarith [hlt_mid]
      simpa [hfloorR] using this
    have hround :
        TorchLean.Floats.neuralNearestEven (Real.sqrt (n : ℝ)) =
          (⌊Real.sqrt (n : ℝ)⌋ : Int) :=
      TorchLean.Floats.neural_nearest_even_eq_floor_of_frac_lt_half (Real.sqrt (n : ℝ)) (by
        simpa using hfrac_lt)
    have hLHS : TorchLean.Floats.neuralNearestEven (Real.sqrt (n : ℝ)) = Int.ofNat q := by
      calc
        TorchLean.Floats.neuralNearestEven (Real.sqrt (n : ℝ)) = (⌊Real.sqrt (n : ℝ)⌋ : Int) :=
          hround
        _ = q := hfloor
        _ = Int.ofNat q := by simp
    simpa [r, hgt] using hLHS
end IEEE32Exec

end TorchLean.Floats.IEEE754
