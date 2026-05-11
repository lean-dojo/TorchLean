/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Init.Data.Nat.Bitwise.Lemmas
public import Mathlib.Data.Nat.Log
public import NN.Floats.IEEEExec.ERealSemantics
public import NN.Floats.IEEEExec.MkBitsToReal
public import NN.Floats.IEEEExec.Negation
public import NN.Floats.IEEEExec.NatLemmas

/-!
# Directed rounding soundness for `IEEE32Exec`

`NN/Floats/IEEEExec/Exec32.lean` defines executable *directed rounding* on the IEEE-754 binary32
  grid:

- `roundDyadicDown` / `roundDyadicUp` round an exact dyadic toward `-∞` / `+∞`.
- `addDown`/`addUp`, `mulDown`/`mulUp` lift this to outward-rounded arithmetic for interval
  endpoints.

This file proves (one core piece of) the enclosure direction (“golden theorem” style): the
*directed rounding down* kernel for positive dyadics is a sound **lower bound** for the exact real
value.

Scope (what is proved here):
- `roundDyadicPosDown` / `roundDyadicPosUp` soundness (in `EReal`, to handle overflow to `±∞`).
- Full dyadic rounding soundness: `roundDyadicDown` is a lower bound and `roundDyadicUp` is an
  upper bound for the exact dyadic real semantics.
- Interval-endpoint soundness for `addDown/addUp/mulDown/mulUp` on finite inputs, as inequalities
  in `EReal` between the executable endpoint operation and the exact real operation.

Non-goals (not proved here yet):
- Correctly-rounded/outward-rounded transcendental functions w.r.t. a standard libm. IEEE-754 does
  not specify libm results; for rigorous transcendentals, prefer the Arb oracle backend.
- Division/sqrt interval primitives (substantially larger; requires separate algorithms + proofs).

References:
- IEEE 754-2019 (floating-point arithmetic): doi:10.1109/IEEESTD.2019.8766229
- Goldberg (1991), “What Every Computer Scientist Should Know About Floating-Point Arithmetic”:
  doi:10.1145/103162.103163
-/

@[expose] public section


namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

noncomputable section

/-! ## Basic helpers -/

abbrev bpow (e : Int) : ℝ :=
  neuralBpow binaryRadix e

/-- `bpow e = 2^e` is strictly positive. -/
lemma bpow_pos (e : Int) : 0 < bpow e :=
  neuralBpow.pos binaryRadix e

/-- `bpow e = 2^e` is nonnegative. -/
lemma bpow_nonneg (e : Int) : 0 ≤ bpow e :=
  neuralBpow.nonneg binaryRadix e

/-- Exponent law for `bpow`: `bpow (e₁+e₂) = bpow e₁ * bpow e₂`. -/
lemma bpow_add (e1 e2 : Int) : bpow (e1 + e2) = bpow e1 * bpow e2 := by
  -- `simp` can close this goal directly once `bpow` is unfolded.
  simp [bpow, neuralBpow.add_exp]

/--
`pow2` is monotone in the exponent, in the simple form `pow2 k ≤ pow2 (k+1)`.

We use this for small exponent-range arguments (e.g. `2^23 ≤ 2^24`) without importing heavier
monotonicity machinery.
-/
lemma pow2_le_pow2_succ (k : Nat) : pow2 k ≤ pow2 (k + 1) := by
  -- `pow2 (k+1) = 2 * pow2 k` by the shift-left recursion.
  simpa [pow2, Nat.shiftLeft_succ] using
    (Nat.le_mul_of_pos_left (pow2 k) (by decide : 0 < 2))

/-- Strict growth of `pow2`: `pow2 k < pow2 (k+1)`. -/
lemma pow2_lt_pow2_succ (k : Nat) : pow2 k < pow2 (k + 1) := by
  have hkpos : 0 < pow2 k := pow2_pos k
  have h' : 1 * pow2 k < 2 * pow2 k :=
    Nat.mul_lt_mul_of_pos_right (by decide : (1 : Nat) < 2) hkpos
  -- `1 * pow2 k = pow2 k` and `pow2 (k+1) = 2 * pow2 k`.
  simpa [pow2, Nat.shiftLeft_succ] using h'

lemma pow2_add (a b : Nat) : pow2 (a + b) = pow2 a * pow2 b := by
  -- `pow2 k = 2^k`, so this is `2^(a+b) = 2^a * 2^b`.
  simp [pow2_eq_two_pow, Nat.pow_add]

/-- Commuted form of `pow2_add`: `pow2 a * pow2 b = pow2 (a + b)`. -/
lemma pow2_mul (a b : Nat) : pow2 a * pow2 b = pow2 (a + b) :=
  (pow2_add a b).symm

/-- Interpret `bpow` at a natural exponent in terms of `pow2`. -/
lemma bpow_ofNat (n : Nat) : bpow (Int.ofNat n) = (pow2 n : ℝ) := by
  -- `binary_radix.to_real = 2`, so `neural_bpow` is `2^n`.
  simp [bpow, TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal, pow2_eq_two_pow,
    Nat.cast_pow]

/-- Interpret `bpow` at a negative successor exponent as an inverse power of two. -/
lemma bpow_negSucc (n : Nat) : bpow (Int.negSucc n) = ((pow2 (n + 1) : Nat) : ℝ)⁻¹ := by
  -- `2^(-(n+1)) = (2^(n+1))⁻¹`.
  simp [bpow, TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal, pow2_eq_two_pow]

/-- Real semantics of a nonnegative dyadic `(mant, exp)` (sign bit false). -/
lemma dyadicToReal_pos (mant : Nat) (exp : Int) :
    dyadicToReal { sign := false, mant := mant, exp := exp } = (mant : ℝ) * bpow exp := by
  simp [dyadicToReal, bpow]

/-- Real semantics of a negative dyadic `(mant, exp)` (sign bit true). -/
lemma dyadicToReal_signTrue (mant : Nat) (exp : Int) :
    dyadicToReal { sign := true, mant := mant, exp := exp } = -((mant : ℝ) * bpow exp) := by
  simp [dyadicToReal, bpow]

/-! ## A few float32 constants as reals -/

@[simp] private lemma toReal_posZero' : toReal (posZero : IEEE32Exec) = 0 := by
  have hexp : (0 : Nat) < 255 := by decide
  have hfrac : (0 : Nat) < 2 ^ 23 := by decide
  have hdy : toDyadic? posZero = some { sign := false, mant := 0, exp := (0 : Int) } := by
    -- `posZero = ofBits 0 = ofBits (mkBits false 0 0)`.
    simpa [posZero, mkBits] using
      (toDyadic?_ofBits_mkBits_fin (sign := false) (exp := 0) (frac := 0) hexp hfrac)
  simp [toReal_eq, hdy, dyadicToReal]

/-- `toEReal` maps `+0` to `0`. -/
@[simp] private lemma toEReal_posZero : toEReal (posZero : IEEE32Exec) = (0 : EReal) := by
  simpa using (toEReal_signedZero (s := false))

/-- `toEReal` maps `-0` to `0`. -/
@[simp] private lemma toEReal_negZero : toEReal (negZero : IEEE32Exec) = (0 : EReal) := by
  simpa using (toEReal_signedZero (s := true))

private lemma toReal_posMinSubnormal :
    toReal (posMinSubnormal : IEEE32Exec) = bpow (-149) := by
  have hexp : (0 : Nat) < 255 := by decide
  have hfrac : (1 : Nat) < 2 ^ 23 := by decide
  have hdy :
      toDyadic? posMinSubnormal = some { sign := false, mant := 1, exp := (-149 : Int) } := by
    simpa [posMinSubnormal] using
      (toDyadic?_ofBits_mkBits_fin (sign := false) (exp := 0) (frac := 1) hexp hfrac)
  simp [toReal_eq, hdy, dyadicToReal]

/-- `toEReal` maps the smallest positive subnormal float32 to `2^{-149}`. -/
@[simp] private lemma toEReal_posMinSubnormal :
    toEReal (posMinSubnormal : IEEE32Exec) = (bpow (-149 : Int) : EReal) := by
  have hfin : isFinite (posMinSubnormal : IEEE32Exec) = true := by decide
  have hcoe :
      toEReal (posMinSubnormal : IEEE32Exec) =
        (toReal (posMinSubnormal : IEEE32Exec) : EReal) :=
    toEReal_eq_coe_toReal_of_isFinite (x := posMinSubnormal) hfin
  simp [hcoe, toReal_posMinSubnormal]

private lemma toReal_posMaxFinite_lt_bpow128 :
    toReal (posMaxFinite : IEEE32Exec) < bpow (128 : Int) := by
  -- `posMaxFinite = mkBits false 254 (2^23-1)` so its real value is `(2^24-1)*2^104 < 2^128`.
  have hexp : (254 : Nat) < 255 := by decide
  have hfrac : (pow2 23 - 1) < 2 ^ 23 := by
    have h : (pow2 23 - 1) < pow2 23 := Nat.sub_lt (pow2_pos 23) (by decide)
    exact lt_of_lt_of_eq h (pow2_eq_two_pow 23)
  have hdy :
      toDyadic? posMaxFinite =
        some { sign := false, mant := pow2 23 + (pow2 23 - 1), exp := (Int.ofNat 254) - 150 } := by
    simpa [posMaxFinite] using
      (toDyadic?_ofBits_mkBits_fin (sign := false) (exp := 254) (frac := (pow2 23 - 1)) hexp hfrac)
  have hto :
      toReal posMaxFinite = ((pow2 24 - 1 : Nat) : ℝ) * bpow (104 : Int) := by
    simp [toReal_eq, hdy, dyadicToReal, bpow, pow2, Nat.shiftLeft_eq]
  have hltMant : ((pow2 24 - 1 : Nat) : ℝ) < (pow2 24 : ℝ) := by
    exact_mod_cast (Nat.sub_lt (pow2_pos 24) (by decide))
  have hbpos : 0 < bpow (104 : Int) := bpow_pos (104 : Int)
  have hmul_lt :
      ((pow2 24 - 1 : Nat) : ℝ) * bpow (104 : Int) < (pow2 24 : ℝ) * bpow (104 : Int) :=
    mul_lt_mul_of_pos_right hltMant hbpos
  have hpow : (pow2 24 : ℝ) * bpow (104 : Int) = bpow (128 : Int) := by
    calc
      (pow2 24 : ℝ) * bpow (104 : Int)
          = bpow (Int.ofNat 24) * bpow (104 : Int) := by
              simpa using congrArg (fun t : ℝ => t * bpow (104 : Int)) (bpow_ofNat 24).symm
      _ = bpow ((Int.ofNat 24) + 104) := (bpow_add (Int.ofNat 24) (104 : Int)).symm
      _ = bpow (128 : Int) := by norm_num
  -- Finish by chaining the inequalities.
  calc
    toReal posMaxFinite = ((pow2 24 - 1 : Nat) : ℝ) * bpow (104 : Int) := hto
    _ < (pow2 24 : ℝ) * bpow (104 : Int) := hmul_lt
    _ = bpow (128 : Int) := hpow

/-! ## Shift helpers (floor/ceil division by powers of two) -/

private lemma shiftRight_mul_pow2_le (n k : Nat) : Nat.shiftRight n k * pow2 k ≤ n := by
  have : (n / (2 ^ k)) * (2 ^ k) ≤ n := Nat.div_mul_le_self n (2 ^ k)
  simpa [Nat.shiftRight_eq_div_pow, pow2_eq_two_pow, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
    using this

private lemma shiftRightCeilPow2_mul_pow2_ge (n k : Nat) :
    n ≤ shiftRightCeilPow2 n k * pow2 k := by
  classical
  -- This lemma is used when proving soundness of `roundDyadicPosUp` (ceil rounding): shifting right
  -- by `k` and then multiplying back by `2^k` produces a value `≥ n` when we round the quotient up.
  cases k with
  | zero =>
      simp [shiftRightCeilPow2, pow2_eq_two_pow]
  | succ k =>
      -- Use `d = 2^(k+1)`, `q = n / d`, `r = n % d`.
      set d : Nat := pow2 (Nat.succ k)
      have hdpos : 0 < d := pow2_pos (Nat.succ k)
      set q : Nat := n / d
      set r : Nat := n % d
      have hn : n = q * d + r := by
        -- `Nat.div_add_mod` is stated with `d * (n / d)`, so we commute.
        simpa [q, r, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using (Nat.div_add_mod n
          d).symm
      have hrlt : r < d := Nat.mod_lt n hdpos
      have hshift : Nat.shiftRight n (Nat.succ k) = q := by
        simp [q, d, Nat.shiftRight_eq_div_pow, pow2_eq_two_pow]
      have hrem : n - Nat.shiftLeft q (Nat.succ k) = r := by
        have : Nat.shiftLeft q (Nat.succ k) = q * d := by
          simp [Nat.shiftLeft_eq, d, pow2_eq_two_pow]
        rw [this, hn]
        simp
      have hshift' : n >>> (k + 1) = q := by
        simpa [Nat.succ_eq_add_one] using hshift
      have hrem' : n - q <<< (k + 1) = r := by
        simpa [Nat.succ_eq_add_one] using hrem
      -- Now split on `r = 0`.
      by_cases hr0 : r = 0
      · -- exact division
        have hn' : n = q * d := by simp [hn, hr0]
        -- `shiftRightCeilPow2` returns `q` since the remainder is zero.
        have hceil : shiftRightCeilPow2 n (Nat.succ k) = q := by
          have hk0 : (Nat.succ k == 0) = false := by simp
          -- Expand the definition and rewrite the quotient/remainder.
          simp (config := { zeta := true }) [shiftRightCeilPow2, hk0]
          rw [hshift', hrem']
          simp [hr0]
        have hceil' : shiftRightCeilPow2 (q * d) (Nat.succ k) = q := by
          simpa [hn'] using hceil
        simp [hn', hceil', d]
      · -- nonzero remainder: ceil is `q+1`
        have hnle : n ≤ (q + 1) * d := by
          -- `n = q*d + r` and `r < d` imply `n ≤ q*d + d = (q+1)*d`.
          have h1 : q * d + r ≤ q * d + d := Nat.add_le_add_left (Nat.le_of_lt hrlt) (q * d)
          have h2 : (q + 1) * d = q * d + d := by
            simp [Nat.add_mul]
          have : n ≤ q * d + d := by simpa [hn] using h1
          simpa [h2] using this
        have hceil : shiftRightCeilPow2 n (Nat.succ k) = q + 1 := by
          have hk0 : (Nat.succ k == 0) = false := by simp
          simp (config := { zeta := true }) [shiftRightCeilPow2, hk0]
          rw [hshift', hrem']
          simp [hr0]
        simp [hceil, hnle, d]

/-!
`shiftRightCeilPow2` is also controlled *from above* by the corresponding floor quotient:
ceil-division can overshoot by at most `1`.
-/

/--
`shiftRightCeilPow2 n k ≤ (n >>> k) + 1`.

This is the standard fact that `ceil(n / 2^k)` is either the floor quotient or that quotient plus
one, depending on whether the remainder is zero.
-/
private lemma shiftRightCeilPow2_le_shiftRight_add1 (n k : Nat) :
    shiftRightCeilPow2 n k ≤ Nat.shiftRight n k + 1 := by
  classical
  cases k with
  | zero =>
      -- `k = 0`: both sides are `n` up to the `+1`.
      simp [shiftRightCeilPow2]
  | succ k =>
      -- `k > 0`: unfold the definition and split on whether the remainder is zero.
      have hk0 : (Nat.succ k == 0) = false := by simp
      -- After unfolding, `shiftRightCeilPow2` returns `q` or `q+1`, where `q := n >>> (k+1)`.
      -- Both are `≤ q+1`.
      simp (config := { zeta := true }) [shiftRightCeilPow2, hk0]
      by_cases hrem : n - n >>> (k + 1) <<< (k + 1) = 0 <;> simp [hrem]

/-! ## Soundness: directed rounding on positive dyadics -/

private lemma pow2_log2_le (mant : Nat) (hm : mant ≠ 0) : pow2 (Nat.log2 mant) ≤ mant := by
  have h : 2 ^ Nat.log2 mant ≤ mant := (Nat.le_log2 hm).1 le_rfl
  simpa [pow2_eq_two_pow] using h

private lemma mant_lt_pow2_log2_add1 (mant : Nat) (hm : mant ≠ 0) : mant < pow2 (Nat.log2 mant + 1)
  := by
  have : mant.log2 < mant.log2 + 1 := Nat.lt_succ_self _
  have h : mant < 2 ^ (mant.log2 + 1) := (Nat.log2_lt hm).1 this
  simpa [pow2_eq_two_pow] using h

/-- `roundDyadicPosDown` is a real lower bound for a positive dyadic. -/
theorem toReal_roundDyadicPosDown_le (mant : Nat) (exp : Int) (hm : mant ≠ 0) :
    toReal (roundDyadicPosDown mant exp) ≤ (mant : ℝ) * bpow exp := by
  classical
  -- Proof idea:
  -- 1) Split `k := log2(mant) + exp` into the four float32 magnitude regimes:
  --    overflow / underflow / subnormal / normal.
  -- 2) In the normal branch, we show the 24-bit mantissa `m24` produced by shifting satisfies
  -- `2^23 ≤ m24`, and that `(2^23 + frac) ≤ m24`, hence the produced float is ≤ the exact dyadic.
  -- 3) In subnormal branches we reduce to (shifted) floor division by powers of two.
  --
  -- The arithmetic facts about `log2` are provided by `Mathlib.Data.Nat.Log`.
  set log2m : Nat := Nat.log2 mant with hlog2m
  set k : Int := (Int.ofNat log2m) + exp with hk
  have hmantPos : 0 ≤ (mant : ℝ) * bpow exp :=
    mul_nonneg (Nat.cast_nonneg _) (bpow_nonneg exp)
  by_cases hkHi : k > 127
  · -- overflow: return `posMaxFinite`, which is still ≤ the exact dyadic (since `k ≥ 128`).
    have hk128 : (128 : Int) ≤ k := by
      have hk127 : (127 : Int) < k := by simpa using hkHi
      -- `Int` is discrete: `a < b` implies `a+1 ≤ b`.
      simpa using (Int.add_one_le_of_lt hk127)
    have hbpow_le : bpow (128 : Int) ≤ (mant : ℝ) * bpow exp := by
      have hpow : (pow2 log2m : ℝ) ≤ (mant : ℝ) := by
        exact_mod_cast pow2_log2_le mant hm
      have hbpowk :
          bpow (Int.ofNat log2m + exp) ≤ (mant : ℝ) * bpow exp := by
        -- `2^log2m ≤ mant`, multiply by `2^exp`.
        have hbpos : 0 ≤ bpow exp := bpow_nonneg exp
        have : (pow2 log2m : ℝ) * bpow exp ≤ (mant : ℝ) * bpow exp := by
          exact mul_le_mul_of_nonneg_right hpow hbpos
        -- Rewrite `bpow (log2m+exp) = 2^log2m * 2^exp`.
        have hb : bpow (Int.ofNat log2m + exp) = (pow2 log2m : ℝ) * bpow exp := by
          calc
            bpow (Int.ofNat log2m + exp) = bpow (Int.ofNat log2m) * bpow exp :=
              bpow_add (Int.ofNat log2m) exp
            _ = (pow2 log2m : ℝ) * bpow exp := by
              -- Avoid simp-cancellation on the right factor.
              rw [bpow_ofNat log2m]
        simpa [hb.symm] using this
      -- monotonicity in the exponent: `2^128 ≤ 2^k`.
      have hbpow128k : bpow (128 : Int) ≤ bpow k := by
        -- `bpow` is `2^•`.
        have hbase : (1 : ℝ) ≤ (2 : ℝ) := by norm_num
        -- rewrite `bpow` to `zpow`.
        have : (2 : ℝ) ^ (128 : Int) ≤ (2 : ℝ) ^ k := by
          exact zpow_le_zpow_right₀ hbase hk128
        simpa [bpow, TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal] using this
      have hkdef : k = Int.ofNat log2m + exp := hk
      -- Combine.
      have : bpow (128 : Int) ≤ bpow (Int.ofNat log2m + exp) := by
        simpa [hkdef] using hbpow128k
      exact this.trans hbpowk
    have hmax_le : toReal posMaxFinite ≤ (mant : ℝ) * bpow exp :=
      (le_of_lt toReal_posMaxFinite_lt_bpow128).trans hbpow_le
    have hOut : roundDyadicPosDown mant exp = posMaxFinite := by
      have hkHiLt : (127 : Int) < (↑mant.log2 + exp) := by
        simpa [hk, hlog2m] using hkHi
      conv_lhs =>
        simp (config := { zeta := true }) [roundDyadicPosDown]
        rw [if_pos hkHiLt]
    simpa [hOut] using hmax_le
  · have hkHi' : ¬ k > 127 := hkHi
    by_cases hkUnder : k < -149
    · -- underflow-to-zero: `0 ≤ exact`.
      have hOut : roundDyadicPosDown mant exp = posZero := by
        have hkHiNot : ¬ (127 : Int) < (↑mant.log2 + exp) := by
          simpa [hk, hlog2m] using hkHi'
        have hkUnder' : (↑mant.log2 + exp) < -149 := by
          simpa [hk, hlog2m] using hkUnder
        conv_lhs =>
          simp (config := { zeta := true }) [roundDyadicPosDown]
          rw [if_neg hkHiNot]
          rw [if_pos hkUnder']
      -- Avoid unfolding `toReal` through `toReal_eq`.
      rw [hOut, toReal_posZero']
      exact hmantPos
    · have hkUnder' : ¬ k < -149 := hkUnder
      by_cases hkSub : k < -126
      · -- subnormal: output is (masked) `floor(mant * 2^(exp+149)) * 2^-149`.
        set fracNat : Nat :=
          match exp + 149 with
          | .ofNat sh => Nat.shiftLeft mant sh
          | .negSucc sh => Nat.shiftRight mant (sh + 1)
        set frac : Nat := fracNat % pow2 23
        have hfrac_lt : frac < 2 ^ 23 := by
          have : frac < pow2 23 := Nat.mod_lt _ (pow2_pos 23)
          simpa [pow2_eq_two_pow] using this
        by_cases hfrac0 : fracNat = 0
        · -- returns `0`
          have hOut : roundDyadicPosDown mant exp = posZero := by
            have hkHiNot : ¬ (127 : Int) < (↑mant.log2 + exp) := by
              simpa [hk, hlog2m] using hkHi'
            have hkUnder'' : ¬ (↑mant.log2 + exp) < -149 := by
              simpa [hk, hlog2m] using hkUnder'
            have hkSub' : (↑mant.log2 + exp) < -126 := by
              simpa [hk, hlog2m] using hkSub
            have hmatch0 :
                (match exp + 149 with
                  | Int.ofNat sh => mant <<< sh
                  | Int.negSucc sh => mant >>> (sh + 1)) = 0 := by
              simpa [fracNat] using hfrac0
            conv_lhs =>
              simp (config := { zeta := true }) [roundDyadicPosDown, fracNat, frac]
              rw [if_neg hkHiNot]
              rw [if_neg hkUnder'']
              rw [if_pos hkSub']
            -- Now the remaining `if` is exactly the `fracNat == 0` test; discharge it using
            -- `hmatch0`.
            by_cases hc :
                (match exp + 149 with
                  | Int.ofNat sh => mant <<< sh
                  | Int.negSucc sh => mant >>> (sh + 1)) = 0
            · exact if_pos hc
            · exfalso; exact hc hmatch0
          rw [hOut, toReal_posZero']
          exact hmantPos
        · -- returns `ofBits (mkBits .. frac)`
          have hexp : (0 : Nat) < 255 := by decide
          have hto : toReal (ofBits (mkBits false 0 frac)) = (frac : ℝ) * bpow (-149) := by
            have h :=
              toReal_ofBits_mkBits_fin (sign := false) (exp := 0) (frac := frac) (hexp := hexp)
                (hfrac := hfrac_lt)
            have h' :
                toReal (ofBits (mkBits false 0 frac)) =
                  (if frac = 0 then 0 else (frac : ℝ) * bpow (-149)) := by
              simpa [bpow] using h
            rw [h']
            by_cases hf : frac = 0 <;> simp [hf]
          have hfrac_le : (frac : ℝ) ≤ (fracNat : ℝ) := by
            have : frac ≤ fracNat := Nat.mod_le _ _
            exact_mod_cast this
          -- Show `fracNat * 2^-149 ≤ exact`.
          have hfracNat_le :
              (fracNat : ℝ) * bpow (-149) ≤ (mant : ℝ) * bpow exp := by
            cases hexp149 : exp + 149 with
            | ofNat sh =>
                -- `exp = sh - 149`, `fracNat = mant * 2^sh`.
                have hfracNat : fracNat = Nat.shiftLeft mant sh := by simp [fracNat, hexp149]
                have hexp : exp = (Int.ofNat sh) - 149 := by
                  -- subtract `149` from both sides of `exp + 149 = sh`
                  have h := congrArg (fun z : Int => z - 149) hexp149
                  simpa using h
                -- Rewrite to a shape where `bpow_add` applies cleanly.
                rw [hfracNat, hexp]
                have hb :
                    bpow ((Int.ofNat sh) - 149) = (pow2 sh : ℝ) * bpow (-149 : Int) := by
                  -- `2^(sh-149) = 2^sh * 2^-149`.
                  calc
                    bpow ((Int.ofNat sh) - 149)
                        = bpow (Int.ofNat sh + (-149 : Int)) := by simp [sub_eq_add_neg]
                    _ = bpow (Int.ofNat sh) * bpow (-149 : Int) := bpow_add (Int.ofNat sh) (-149 :
                      Int)
                    _ = (pow2 sh : ℝ) * bpow (-149 : Int) := by
                      -- avoid cancellation on the common factor
                      rw [bpow_ofNat sh]
                have hShift :
                    ((Nat.shiftLeft mant sh : Nat) : ℝ) = (mant : ℝ) * (pow2 sh : ℝ) := by
                  simp [Nat.shiftLeft_eq, pow2_eq_two_pow, Nat.cast_mul, Nat.cast_pow]
                -- Now both sides are definitionally equal after rewriting.
                have hEq :
                    ((Nat.shiftLeft mant sh : Nat) : ℝ) * bpow (-149 : Int) =
                      (mant : ℝ) * bpow ((Int.ofNat sh) - 149) := by
                  -- rewrite the RHS into the product form and then unfold the shift on the LHS
                  rw [hb]
                  have hShiftMul :
                      ((Nat.shiftLeft mant sh : Nat) : ℝ) * bpow (-149 : Int) =
                        (mant : ℝ) * (pow2 sh : ℝ) * bpow (-149 : Int) := by
                    simpa [mul_assoc] using congrArg (fun t : ℝ => t * bpow (-149 : Int)) hShift
                  rw [hShiftMul]
                  simp [mul_assoc]
                exact le_of_eq hEq
            | negSucc sh =>
                -- `fracNat = floor(mant / 2^(sh+1))`, `exp = -(sh+1) - 149`.
                have hfracNat : fracNat = Nat.shiftRight mant (sh + 1) := by simp [fracNat, hexp149]
                have hexp : exp = (Int.negSucc sh) - 149 := by
                  have h := congrArg (fun z : Int => z - 149) hexp149
                  simpa using h
                have hfloor : (fracNat : ℝ) * (pow2 (sh + 1) : ℝ) ≤ (mant : ℝ) := by
                  have : Nat.shiftRight mant (sh + 1) * pow2 (sh + 1) ≤ mant :=
                    shiftRight_mul_pow2_le mant (sh + 1)
                  -- use the explicit value of `fracNat`
                  simpa [hfracNat] using (show ((Nat.shiftRight mant (sh + 1) * pow2 (sh + 1) : Nat)
                    : ℝ) ≤ mant by
                    exact_mod_cast this)
                have hdenPos : 0 < (pow2 (sh + 1) : ℝ) := by
                  exact_mod_cast (pow2_pos (sh + 1))
                have hfracNat_le_div : (fracNat : ℝ) ≤ (mant : ℝ) * (bpow (Int.negSucc sh)) := by
                  -- divide `hfloor` by `2^(sh+1)`.
                  have hdiv :
                      (fracNat : ℝ) ≤ (mant : ℝ) / (pow2 (sh + 1) : ℝ) :=
                    (le_div_iff₀ hdenPos).2 (by
                      simpa [mul_assoc, mul_left_comm, mul_comm] using hfloor)
                  simpa [div_eq_mul_inv, bpow_negSucc, mul_assoc, mul_left_comm, mul_comm] using
                    hdiv
                -- now multiply by `2^-149`
                have hbpos : 0 ≤ bpow (-149 : Int) := bpow_nonneg (-149 : Int)
                have : (fracNat : ℝ) * bpow (-149 : Int) ≤ (mant : ℝ) * bpow (Int.negSucc sh) * bpow
                  (-149 : Int) := by
                  exact mul_le_mul_of_nonneg_right hfracNat_le_div hbpos
                -- rewrite RHS to `mant * bpow exp`
                have hb :
                    bpow (Int.negSucc sh) * bpow (-149 : Int) = bpow ((Int.negSucc sh) - 149) := by
                  calc
                    bpow (Int.negSucc sh) * bpow (-149 : Int)
                        = bpow (Int.negSucc sh + (-149 : Int)) := (bpow_add (Int.negSucc sh) (-149 :
                          Int)).symm
                    _ = bpow ((Int.negSucc sh) - 149) := by simp [sub_eq_add_neg]
                -- Use `hb` and `hexp`.
                simpa [mul_assoc, mul_left_comm, mul_comm, hb, hexp] using this
          have : (frac : ℝ) * bpow (-149) ≤ (mant : ℝ) * bpow exp := by
            have hbpos : 0 ≤ bpow (-149 : Int) := bpow_nonneg (-149 : Int)
            have : (frac : ℝ) * bpow (-149 : Int) ≤ (fracNat : ℝ) * bpow (-149 : Int) := by
              exact mul_le_mul_of_nonneg_right hfrac_le hbpos
            exact this.trans hfracNat_le
          have htoLe : toReal (ofBits (mkBits false 0 frac)) ≤ (mant : ℝ) * bpow exp := by
            rw [hto]
            exact this
          have hOut : roundDyadicPosDown mant exp = ofBits (mkBits false 0 frac) := by
            have hkHiNot : ¬ (127 : Int) < (↑mant.log2 + exp) := by
              simpa [hk, hlog2m] using hkHi'
            have hkUnder'' : ¬ (↑mant.log2 + exp) < -149 := by
              simpa [hk, hlog2m] using hkUnder'
            have hkSub' : (↑mant.log2 + exp) < -126 := by
              simpa [hk, hlog2m] using hkSub
            have hmatch0 :
                ¬ (match exp + 149 with
                    | Int.ofNat sh => mant <<< sh
                    | Int.negSucc sh => mant >>> (sh + 1)) = 0 := by
              simpa [fracNat] using hfrac0
            conv_lhs =>
              simp (config := { zeta := true }) [roundDyadicPosDown, fracNat, frac]
              rw [if_neg hkHiNot]
              rw [if_neg hkUnder'']
              rw [if_pos hkSub']
            by_cases hc :
                (match exp + 149 with
                  | Int.ofNat sh => mant <<< sh
                  | Int.negSucc sh => mant >>> (sh + 1)) = 0
            · exfalso; exact hmatch0 hc
            ·
              -- The `fracNat == 0` test is false, so we return the masked subnormal bits.
              exact if_neg hc
          rw [hOut]
          exact htoLe
      · -- normal: scale mantissa to 24 bits by floor, then interpret as a normal float.
        have hkSub' : ¬ k < -126 := hkSub
        set m24 : Nat :=
          if log2m ≥ 23 then Nat.shiftRight mant (log2m - 23) else Nat.shiftLeft mant (23 - log2m)
        set expNat : Nat := Int.toNat (k + 127)
        set frac : Nat := (m24 - pow2 23) % pow2 23
        have hexpNat : expNat < 255 := by
          have hk_nonneg : 0 ≤ k + 127 := by
            have hk_ge : (-126 : Int) ≤ k := (not_lt).1 hkSub'
            linarith
          have hk_lt : k + 127 < 255 := by
            have hk_le : k ≤ 127 := (not_lt).1 (by simpa using hkHi')
            linarith
          have : (expNat : Int) < (255 : Int) := by
            -- `↑(toNat z) = z` for `z ≥ 0`.
            have hz : (expNat : Int) = k + 127 := by
              simpa [expNat] using (Int.toNat_of_nonneg hk_nonneg)
            simpa [hz] using hk_lt
          exact (Int.ofNat_lt).1 this
        have hfrac_lt : frac < 2 ^ 23 := by
          have : frac < pow2 23 := Nat.mod_lt _ (pow2_pos 23)
          simpa [pow2_eq_two_pow] using this
        -- in the normal branch, `expNat ≠ 0`
        have hexp0 : expNat ≠ 0 := by
          have hk_pos : (0 : Int) < k + 127 := by
            have hk_ge : (-126 : Int) ≤ k := (not_lt).1 hkSub'
            linarith
          have hk_nonneg : 0 ≤ k + 127 := le_of_lt hk_pos
          have hz : (expNat : Int) = k + 127 := by
            simpa [expNat] using (Int.toNat_of_nonneg hk_nonneg)
          have : (0 : Int) < (expNat : Int) := by simpa [hz] using hk_pos
          exact Nat.ne_of_gt ((Int.ofNat_lt).1 (by simpa using this))
        have hto :
            toReal (ofBits (mkBits false expNat frac)) =
              ((pow2 23 + frac : Nat) : ℝ) * bpow ((Int.ofNat expNat) - 150) := by
          have h :=
            toReal_ofBits_mkBits_fin (sign := false) (exp := expNat) (frac := frac) hexpNat hfrac_lt
          by_cases h0 : expNat = 0
          · exact False.elim (hexp0 h0)
          ·
            have h' :
                toReal (ofBits (mkBits false expNat frac)) =
                  ((pow2 23 + frac : Nat) : ℝ) * neuralBpow binaryRadix ((Int.ofNat expNat) - 150)
                    := by
              -- Avoid `simp` here: we only need to pick the `exp ≠ 0` branch and then evaluate
              -- the remaining `if false then _ else _`.
              have h1 := h
              -- Drop the outer `if expNat = 0` using `h0 : expNat ≠ 0`.
              rw [if_neg h0] at h1
              -- Evaluate the remaining `if false then _ else _` by definitional reduction.
              exact h1.trans (by rfl)
            -- `bpow` is just `neural_bpow binary_radix`, so this is definitional.
            dsimp [bpow]
            exact h'
        have hmant_le_m24 : (pow2 23 + frac : Nat) ≤ m24 := by
          -- `frac ≤ m24 - 2^23`, and in this branch we indeed have `2^23 ≤ m24`.
          have hm24_ge : pow2 23 ≤ m24 := by
            -- `m24` is the “24-bit mantissa” obtained by shifting `mant` to have its top bit at
            -- position 23.
            -- This is standard: `2^log2 mant ≤ mant`, and shifting by `|log2 mant - 23|` places the
            -- leading 1.
            have hpow : pow2 log2m ≤ mant := pow2_log2_le mant hm
            by_cases hlog : log2m ≥ 23
            ·
              set sh : Nat := log2m - 23
              have hm24' : m24 = Nat.shiftRight mant sh := by
                simp [m24, hlog, sh]
              -- Divide `2^log2m ≤ mant` by `2^sh` to get `2^23 ≤ mant >> sh`.
              have hdiv : (pow2 log2m) / (pow2 sh) ≤ mant / (pow2 sh) :=
                Nat.div_le_div_right hpow
              have hpowSplit : pow2 log2m = pow2 sh * pow2 23 := by
                -- `log2m = sh + 23` since `sh = log2m - 23`.
                have hlogEq : log2m = sh + 23 := by
                  simpa [sh] using (Nat.sub_add_cancel hlog).symm
                calc
                  pow2 log2m = pow2 (sh + 23) := by simp [hlogEq]
                  _ = pow2 sh * pow2 23 := pow2_add sh 23
              have hcalc : (pow2 log2m) / (pow2 sh) = pow2 23 := by
                calc
                  (pow2 log2m) / (pow2 sh) = (pow2 sh * pow2 23) / (pow2 sh) := by
                    simp [hpowSplit]
                  _ = (pow2 23 * pow2 sh) / (pow2 sh) := by
                    simp [Nat.mul_comm]
                  _ = pow2 23 := by
                    simpa using (Nat.mul_div_left (pow2 23) (n := pow2 sh) (pow2_pos sh))
              -- `mant / 2^sh = mant >> sh`.
              have : pow2 23 ≤ Nat.shiftRight mant sh := by
                -- from `hdiv` and `hcalc`
                have : pow2 23 ≤ mant / pow2 sh := by
                  simpa [hcalc] using hdiv
                simpa [Nat.shiftRight_eq_div_pow, pow2_eq_two_pow] using this
              simpa [hm24'] using this
            ·
              -- `log2m < 23`, so we shift left by `23 - log2m`.
              set sh : Nat := 23 - log2m
              have hm24' : m24 = Nat.shiftLeft mant sh := by
                have : ¬ log2m ≥ 23 := hlog
                simp [m24, this, sh]
              -- Multiply `2^log2m ≤ mant` by `2^sh` to get `2^23 ≤ mant << sh`.
              have hmul : pow2 log2m * pow2 sh ≤ mant * pow2 sh := Nat.mul_le_mul_right (pow2 sh)
                hpow
              have hpow23 : pow2 log2m * pow2 sh = pow2 23 := by
                -- `log2m + sh = 23` by definition of `sh`.
                have hle : log2m ≤ 23 := le_of_lt (Nat.lt_of_not_ge hlog)
                have : log2m + sh = 23 := by simpa [sh] using (Nat.add_sub_of_le hle)
                calc
                  pow2 log2m * pow2 sh = pow2 (log2m + sh) := by
                    simpa using (pow2_mul log2m sh)
                  _ = pow2 23 := by simp [this]
              have hmul' : pow2 23 ≤ mant * pow2 sh := by
                simpa [hpow23] using hmul
              have : pow2 23 ≤ Nat.shiftLeft mant sh := by
                simpa [Nat.shiftLeft_eq, pow2_eq_two_pow, Nat.mul_assoc, Nat.mul_left_comm,
                  Nat.mul_comm] using hmul'
              simpa [hm24'] using this
          have hle : frac ≤ m24 - pow2 23 := Nat.mod_le _ _
          have h1 : pow2 23 + frac ≤ pow2 23 + (m24 - pow2 23) := Nat.add_le_add_left hle (pow2 23)
          have : pow2 23 + (m24 - pow2 23) = m24 := by
            simpa using (Nat.add_sub_of_le hm24_ge)
          simpa [this] using h1
        have hmant_le_m24R : ((pow2 23 + frac : Nat) : ℝ) ≤ (m24 : ℝ) := by
          exact_mod_cast hmant_le_m24
        have hk_nonneg : 0 ≤ k + 127 := by
          have hk_ge : (-126 : Int) ≤ k := (not_lt).1 hkSub'
          linarith
        have hexpInt : (Int.ofNat expNat) = k + 127 := by
          simpa [expNat] using (Int.toNat_of_nonneg hk_nonneg)
        have hexpPow : bpow ((↑expNat : Int) - 150) = bpow (k - 23) := by
          have hexpInt' : (↑expNat : Int) = k + 127 := by simpa using hexpInt
          have hExp : (↑expNat : Int) - 150 = k - 23 := by
            calc
              (↑expNat : Int) - 150 = (k + 127) - 150 := by simp [hexpInt']
              _ = k - 23 := by ring
          simp [hExp]
        have hbpos : 0 ≤ bpow (k - 23) := bpow_nonneg (k - 23)
        have hto_le_m24 :
            ((pow2 23 + frac : Nat) : ℝ) * bpow (k - 23) ≤ (m24 : ℝ) * bpow (k - 23) := by
          exact mul_le_mul_of_nonneg_right hmant_le_m24R hbpos
        have hm24_le_exact : (m24 : ℝ) * bpow (k - 23) ≤ (mant : ℝ) * bpow exp := by
          by_cases hlog : log2m ≥ 23
          · -- `m24 = mant >> (log2m-23)`, and `k-23 = exp + (log2m-23)`.
            set sh : Nat := log2m - 23
            have hsh : (Int.ofNat log2m) = 23 + (Int.ofNat sh) := by
              have hNat : 23 + sh = log2m := by
                simpa [sh] using (Nat.add_sub_of_le hlog)
              have hNat' : log2m = 23 + sh := hNat.symm
              calc
                Int.ofNat log2m = Int.ofNat (23 + sh) := by simp [hNat']
                _ = 23 + Int.ofNat sh := by simp [Nat.cast_add]
            have hm24 : m24 = Nat.shiftRight mant sh := by
              simp [m24, hlog, sh]
            have hk23 : k - 23 = exp + (Int.ofNat sh) := by
              calc
                k - 23 = (Int.ofNat log2m + exp) - 23 := by simp [k]
                _ = ((23 + Int.ofNat sh) + exp) - 23 := by simpa [hsh]
                _ = exp + Int.ofNat sh := by ring
            have hfloorNat : (Nat.shiftRight mant sh : Nat) * pow2 sh ≤ mant :=
              shiftRight_mul_pow2_le mant sh
            have hfloor : ((Nat.shiftRight mant sh : Nat) : ℝ) * (pow2 sh : ℝ) ≤ (mant : ℝ) := by
              exact_mod_cast hfloorNat
            have hbposExp : 0 ≤ bpow exp := bpow_nonneg exp
            have : ((Nat.shiftRight mant sh : Nat) : ℝ) * bpow (exp + (Int.ofNat sh)) ≤ (mant : ℝ) *
              bpow exp := by
              have hb : bpow (exp + Int.ofNat sh) = bpow exp * (pow2 sh : ℝ) := by
                calc
                  bpow (exp + Int.ofNat sh) = bpow exp * bpow (Int.ofNat sh) := bpow_add exp
                    (Int.ofNat sh)
                  _ = bpow exp * (pow2 sh : ℝ) := by
                        -- rewrite only the right factor to avoid `simp` turning `a*b=a*c` into `b=c
                        -- ∨ a=0`
                        rw [bpow_ofNat sh]
              have hfloor' :
                  ((Nat.shiftRight mant sh : Nat) : ℝ) * (pow2 sh : ℝ) * bpow exp ≤ (mant : ℝ) *
                    bpow exp := by
                exact mul_le_mul_of_nonneg_right hfloor hbposExp
              -- Rewrite `bpow (exp+sh)` and commute factors to match `hfloor'`.
              calc
                ((Nat.shiftRight mant sh : Nat) : ℝ) * bpow (exp + Int.ofNat sh)
                    = ((Nat.shiftRight mant sh : Nat) : ℝ) * (bpow exp * (pow2 sh : ℝ)) := by
                        rw [hb]
                _ = ((Nat.shiftRight mant sh : Nat) : ℝ) * (pow2 sh : ℝ) * bpow exp := by
                        ac_rfl
                _ ≤ (mant : ℝ) * bpow exp := hfloor'
            simpa [hm24, hk23] using this
          · -- `m24 = mant << (23-log2m)`, and this path is exact scaling (no rounding).
            set sh : Nat := 23 - log2m
            have hNat : log2m + sh = 23 := by
              have hle : log2m ≤ 23 := le_of_lt (Nat.lt_of_not_ge hlog)
              simpa [sh] using (Nat.add_sub_of_le hle)
            have hInt : (log2m : Int) + (sh : Int) = (23 : Int) := by
              exact_mod_cast hNat
            have hsh : (Int.ofNat log2m) - 23 = - (Int.ofNat sh) := by
              -- rewrite `hInt` into the `Int.ofNat` form for `linarith`
              have hInt' : (Int.ofNat log2m) + (Int.ofNat sh) = (23 : Int) := by
                simpa using hInt
              linarith [hInt']
            have hm24 : m24 = Nat.shiftLeft mant sh := by
              have : ¬ log2m ≥ 23 := hlog
              simp [m24, this, sh]
            have hk23 : k - 23 = exp - (Int.ofNat sh) := by
              calc
                k - 23 = (Int.ofNat log2m + exp) - 23 := by simp [k]
                _ = exp + ((Int.ofNat log2m) - 23) := by ring
                _ = exp - (Int.ofNat sh) := by
                    simpa [sub_eq_add_neg] using congrArg (fun z => exp + z) hsh
            -- Evaluate the real value: `(mant<<sh) * 2^(exp-sh) = mant * 2^exp`.
            have hb : bpow exp = bpow (exp - Int.ofNat sh) * bpow (Int.ofNat sh) := by
              -- `bpow ((exp - sh) + sh) = bpow (exp - sh) * bpow sh`, and the exponent simplifies
              -- to `exp`.
              simpa [sub_eq_add_neg, add_assoc] using (bpow_add (exp - Int.ofNat sh) (Int.ofNat sh))
            have hShift :
                ((Nat.shiftLeft mant sh : Nat) : ℝ) = (mant : ℝ) * (pow2 sh : ℝ) := by
              simp [Nat.shiftLeft_eq, pow2_eq_two_pow, Nat.cast_mul, Nat.cast_pow]
            -- Convert `pow2 sh` to `bpow sh` and finish by rearranging factors.
            have hbsh : (pow2 sh : ℝ) = bpow (Int.ofNat sh) := by
              simpa [bpow_ofNat] using (bpow_ofNat sh).symm
            have : (m24 : ℝ) * bpow (k - 23) = (mant : ℝ) * bpow exp := by
              -- rewrite `m24` and `k-23`, then use `hb`.
              rw [hm24, hk23, hShift, hbsh]
              -- `hb : bpow exp = bpow (exp - sh) * bpow sh`
              -- so `bpow (exp - sh) * bpow sh = bpow exp`.
              calc
                (mant : ℝ) * (bpow (Int.ofNat sh)) * bpow (exp - Int.ofNat sh)
                    = (mant : ℝ) * (bpow (exp - Int.ofNat sh) * bpow (Int.ofNat sh)) := by
                        ring_nf
                _ = (mant : ℝ) * bpow exp := by
                        exact congrArg (fun t : ℝ => (mant : ℝ) * t) hb.symm
            exact (le_of_eq this)
        have hfinal : toReal (ofBits (mkBits false expNat frac)) ≤ (mant : ℝ) * bpow exp := by
          -- chain: `toReal = (pow2 23 + frac) * 2^(k-23) ≤ m24 * 2^(k-23) ≤ exact`.
          have : ((pow2 23 + frac : Nat) : ℝ) * bpow (k - 23) ≤ (mant : ℝ) * bpow exp :=
            (hto_le_m24.trans hm24_le_exact)
          -- rewrite `bpow ((↑expNat)-150)` to `bpow (k-23)` via `hexpPow`
          calc
            toReal (ofBits (mkBits false expNat frac))
                = ((pow2 23 + frac : Nat) : ℝ) * bpow ((Int.ofNat expNat) - 150) := hto
            _ = ((pow2 23 + frac : Nat) : ℝ) * bpow (k - 23) := by
                exact congrArg (fun t => ((pow2 23 + frac : Nat) : ℝ) * t) hexpPow
            _ ≤ (mant : ℝ) * bpow exp := this
        -- Avoid unfolding `toReal`/`toDyadic?` in the final step: rewrite `roundDyadicPosDown`
        -- directly.
        have hkHiNot : ¬ (127 : Int) < (↑mant.log2 + exp) := by
          simpa [hk, hlog2m] using hkHi'
        have hkUnder'' : ¬ (↑mant.log2 + exp) < -149 := by
          simpa [hk, hlog2m] using hkUnder'
        have hkNorm : ¬ (↑mant.log2 + exp) < -126 := by
          simpa [hk, hlog2m] using hkSub'
        have hOut : roundDyadicPosDown mant exp = ofBits (mkBits false expNat frac) := by
          conv_lhs =>
            simp (config := { zeta := true }) [roundDyadicPosDown]
            rw [if_neg hkHiNot]
            rw [if_neg hkUnder'']
            rw [if_neg hkNorm]
          -- Rewrite our local abbreviations to match the unfolded normal-branch expression.
          simp (config := { zeta := true }) [expNat, frac, m24, hk, log2m]
        rw [hOut]
        exact hfinal

/-
## Soundness: directed rounding up

For interval arithmetic, the executable “round toward `+∞`” routine must be a **provable upper
bound** for the corresponding exact real/dyadic value.

Since `roundDyadicPosUp` can overflow to `+∞`, we phrase the theorem in `EReal`.
-/

/--
`roundDyadicPosUp mant exp` is an `EReal` upper bound for the positive dyadic `(mant : ℝ) * 2^exp`.

This is the “ceil” counterpart to `toReal_roundDyadicPosDown_le`.
-/
theorem toEReal_roundDyadicPosUp_ge (mant : Nat) (exp : Int) (hm : mant ≠ 0) :
    ((mant : ℝ) * bpow exp : EReal) ≤ toEReal (roundDyadicPosUp mant exp) := by
  classical
  set log2m : Nat := Nat.log2 mant with hlog2m
  set k : Int := (Int.ofNat log2m) + exp with hk
  -- Split by magnitude regime (matching `roundDyadicPosUp`'s executable case split).
  by_cases hkHi : k > 127
  · -- overflow: `roundDyadicPosUp` returns `+∞`, which is trivially an upper bound.
    have hkHiLt : (127 : Int) < (↑mant.log2 + exp) := by
      simpa [hk, hlog2m] using hkHi
    have hOut : roundDyadicPosUp mant exp = posInf := by
      conv_lhs =>
        simp (config := { zeta := true }) [roundDyadicPosUp]
        rw [if_pos hkHiLt]
    -- Any real is ≤ `⊤`.
    rw [hOut]
    exact le_top
  ·
    have hkHi' : ¬ k > 127 := hkHi
    by_cases hkUnder : k < -149
    ·
      -- Subnormal positives: `roundDyadicPosUp` returns the smallest subnormal `2^-149`.
      have hkHiNot : ¬ (127 : Int) < (↑mant.log2 + exp) := by
        simpa [hk, hlog2m] using hkHi'
      have hkUnder' : (↑mant.log2 + exp) < -149 := by
        simpa [hk, hlog2m] using hkUnder
      have hOut : roundDyadicPosUp mant exp = posMinSubnormal := by
        conv_lhs =>
          simp (config := { zeta := true }) [roundDyadicPosUp]
          rw [if_neg hkHiNot]
          rw [if_pos hkUnder']
      -- Show the exact dyadic is ≤ `2^-149`.
      have hmant_lt : (mant : ℝ) < (pow2 (log2m + 1) : ℝ) := by
        exact_mod_cast (mant_lt_pow2_log2_add1 mant hm)
      have hbpos : 0 < bpow exp := bpow_pos exp
      have hmul_lt :
          (mant : ℝ) * bpow exp < (pow2 (log2m + 1) : ℝ) * bpow exp :=
        mul_lt_mul_of_pos_right hmant_lt hbpos
      have hpow_as_bpow : (pow2 (log2m + 1) : ℝ) = bpow (Int.ofNat (log2m + 1)) := by
        simpa using (bpow_ofNat (log2m + 1)).symm
      have hmul_lt' :
          (mant : ℝ) * bpow exp < bpow (Int.ofNat (log2m + 1) + exp) := by
        -- Rewrite the RHS using `bpow_add`.
        calc
          (mant : ℝ) * bpow exp < (pow2 (log2m + 1) : ℝ) * bpow exp := hmul_lt
          _ = bpow (Int.ofNat (log2m + 1)) * bpow exp := by simp [hpow_as_bpow, mul_comm]
          _ = bpow (Int.ofNat (log2m + 1) + exp) := by
                simpa [mul_comm] using (bpow_add (Int.ofNat (log2m + 1)) exp).symm
      have hmul_le_k1 : (mant : ℝ) * bpow exp ≤ bpow (k + 1) := by
        -- Rewrite `Int.ofNat (log2m+1)` as `Int.ofNat log2m + 1`, then use `hk : k = ofNat log2m +
        -- exp`.
        have hmul_lt'' : (mant : ℝ) * bpow exp < bpow ((Int.ofNat log2m) + 1 + exp) := by
          -- `Int.ofNat (log2m+1) = Int.ofNat log2m + 1`.
          simpa [Nat.succ_eq_add_one, add_assoc, add_left_comm, add_comm] using hmul_lt'
        have hEq : (Int.ofNat log2m) + 1 + exp = k + 1 := by
          -- This is a rearrangement of addition plus `hk : k = ofNat log2m + exp`.
          -- We keep the orientation that rewrites the `bpow` exponent on the RHS of `hmul_lt''`.
          have : (Int.ofNat log2m) + 1 + exp = (Int.ofNat log2m + exp) + 1 := by
            ac_rfl
          simpa [hk] using this
        have hb : bpow ((Int.ofNat log2m) + 1 + exp) = bpow (k + 1) := by
          simpa using congrArg bpow hEq
        have : (mant : ℝ) * bpow exp < bpow (k + 1) := by
          exact lt_of_lt_of_eq hmul_lt'' hb
        exact le_of_lt this
      have hk1 : k + 1 ≤ (-149 : Int) :=
        Int.add_one_le_of_lt hkUnder
      have hbpow_le : bpow (k + 1) ≤ bpow (-149 : Int) := by
        have hbase : (1 : ℝ) ≤ (2 : ℝ) := by norm_num
        have : (2 : ℝ) ^ (k + 1) ≤ (2 : ℝ) ^ (-149 : Int) :=
          zpow_le_zpow_right₀ hbase hk1
        simpa [bpow, TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal] using this
      have hle_real : (mant : ℝ) * bpow exp ≤ bpow (-149 : Int) :=
        hmul_le_k1.trans hbpow_le
      -- Convert to `EReal` and rewrite the output.
      have hle : ((mant : ℝ) * bpow exp : EReal) ≤ (bpow (-149 : Int) : EReal) := by
        exact_mod_cast hle_real
      simpa [hOut, toEReal_posMinSubnormal] using hle
    ·
      have hkUnder' : ¬ k < -149 := hkUnder
      by_cases hkSub : k < -126
      ·
        -- subnormal: compute `fracNat = ceil(mant * 2^(exp+149))` and scale back by `2^-149`.
        -- The subnormal path computes `fracNat = ceil(mant * 2^(exp+149))` as a natural number.
        -- We name it once here so we can reason about positivity/nonzero-ness and then fold it back
        -- into the unfolded executable definition.
        set fracNat : Nat :=
          match exp + 149 with
          | .ofNat sh => mant <<< sh
          | .negSucc sh => shiftRightCeilPow2 mant (sh + 1)
        with hfracNat
        have hfracNat_pos : 0 < fracNat := by
          -- `mant ≠ 0` implies `mant > 0`, and the ceil division/shift keeps it positive.
          have hmant_pos : 0 < mant := Nat.pos_of_ne_zero hm
          cases h : (exp + 149) with
          | ofNat sh =>
              simp [fracNat, h, Nat.shiftLeft_eq, hmant_pos]
          | negSucc sh =>
              -- `shiftRightCeilPow2 mant (sh+1)` is a ceil-division of a positive number, so it is
              -- positive.
              have hge : mant ≤ shiftRightCeilPow2 mant (sh + 1) * pow2 (sh + 1) :=
                shiftRightCeilPow2_mul_pow2_ge mant (sh + 1)
              have hne : shiftRightCeilPow2 mant (sh + 1) ≠ 0 := by
                intro hz
                have : mant ≤ 0 := by simpa [hz] using hge
                exact (Nat.not_le_of_gt hmant_pos) this
              have : 0 < shiftRightCeilPow2 mant (sh + 1) := Nat.pos_of_ne_zero hne
              simpa [fracNat, h] using this
        have hfracNat_ne0 : fracNat ≠ 0 := Nat.ne_of_gt hfracNat_pos
        have hkHiNot : ¬ (127 : Int) < (↑mant.log2 + exp) := by
          simpa [hk, hlog2m] using hkHi'
        have hkUnder'' : ¬ (↑mant.log2 + exp) < -149 := by
          simpa [hk, hlog2m] using hkUnder'
        have hkSub' : (↑mant.log2 + exp) < -126 := by
          simpa [hk, hlog2m] using hkSub
        -- Simplify the executable output (the `fracNat == 0` case is impossible here).
        have hOut :
            roundDyadicPosUp mant exp =
              (match Nat.decLe (pow2 23) fracNat with
               | isTrue _ => ofBits (mkBits false 1 0)
               | isFalse _ => ofBits (mkBits false 0 fracNat)) := by
          -- In this branch of `roundDyadicPosUp`, the executable code computes
          -- `fracNat := ceil(mant * 2^(exp+149))` and then returns either:
          -- - the smallest normal (if `2^23 ≤ fracNat`), or
          -- - the subnormal with fraction bits `fracNat`.
          --
          -- Since `fracNat > 0` here, the `fracNat == 0` guard is dead code.
          -- Reduce `roundDyadicPosUp` to the subnormal branch, then rewrite the computed
          -- `fracNat` into our local constant and eliminate the dead `fracNat = 0` guard.
          -- We do this by rewriting the outer magnitude-branch `if`s, and then eliminating the
          -- dead `fracNat == 0` guard using `fracNat ≠ 0`.
          conv_lhs =>
            simp (config := { zeta := true }) [roundDyadicPosUp]
            rw [if_neg hkHiNot]
            rw [if_neg hkUnder'']
            rw [if_pos hkSub']
          -- Rewrite the RHS to use the *same* `fracNat` computation as the unfolded LHS.
          -- (We keep everything in terms of the executable `match ...` form here; this avoids
          -- subtle mismatches between notations like `Nat.shiftLeft` vs `<<<`.)
          rw [hfracNat]
          -- Eliminate the dead `fracNat = 0` guard using `fracNat ≠ 0`.
          have hmatch_ne0 :
              (match exp + 149 with
                | Int.ofNat sh => mant <<< sh
                | Int.negSucc sh => shiftRightCeilPow2 mant (sh + 1)) ≠ 0 := by
            intro h0
            have : fracNat = 0 := by simpa [hfracNat] using h0
            exact hfracNat_ne0 this
          -- The remaining goal is an `if` over the (propositional) equality to `0`.
          -- Since `fracNat > 0`, this branch is unreachable, so the `if` reduces to the `else`.
          by_cases h0 :
              (match exp + 149 with
                | Int.ofNat sh => mant <<< sh
                | Int.negSucc sh => shiftRightCeilPow2 mant (sh + 1)) = 0
          · exfalso
            exact hmatch_ne0 h0
          ·
            exact if_neg h0
        -- Establish the core numeric inequality in `ℝ`: exact dyadic ≤ `fracNat * 2^-149`.
        have hcore : (mant : ℝ) * bpow exp ≤ (fracNat : ℝ) * bpow (-149 : Int) := by
          cases h : (exp + 149) with
          | ofNat sh =>
              -- Here `exp = sh - 149` and `fracNat = mant << sh`, so it's exact.
              have hexp : exp = (Int.ofNat sh) - 149 := by
                have h' := congrArg (fun z : Int => z - 149) h
                simpa using h'
              have hfrac : fracNat = Nat.shiftLeft mant sh := by simp [fracNat, h]
              -- Rewrite the exact dyadic into `(mant*2^sh) * 2^-149`.
              have hb :
                  bpow exp = bpow (Int.ofNat sh) * bpow (-149 : Int) := by
                -- `bpow (sh-149) = bpow sh * bpow (-149)`.
                simpa [hexp, sub_eq_add_neg] using (bpow_add (Int.ofNat sh) (-149 : Int))
              -- Now just unfold the shift-left and finish.
              rw [hb, hfrac]
              -- `bpow (ofNat sh) = 2^sh` and `mant << sh = mant * 2^sh`.
              rw [bpow_ofNat sh]
              simp [Nat.shiftLeft_eq, Nat.cast_mul, pow2_eq_two_pow, mul_assoc]
          | negSucc sh =>
              -- Here `exp + 149 = -(sh+1)`, so `mant * 2^exp = mant * 2^(-(sh+1)) * 2^-149`.
              have hexp : exp = (Int.negSucc sh) - 149 := by
                have h' := congrArg (fun z : Int => z - 149) h
                simpa using h'
              have hfrac : fracNat = shiftRightCeilPow2 mant (sh + 1) := by simp [fracNat, h]
              -- Use `mant ≤ fracNat * 2^(sh+1)` and divide by `2^(sh+1)`.
              have hgeNat : mant ≤ fracNat * pow2 (sh + 1) := by
                simpa [hfrac] using shiftRightCeilPow2_mul_pow2_ge mant (sh + 1)
              have hge : (mant : ℝ) ≤ (fracNat : ℝ) * (pow2 (sh + 1) : ℝ) := by
                exact_mod_cast hgeNat
              have hpow_pos : 0 < (pow2 (sh + 1) : ℝ) := by
                exact_mod_cast (pow2_pos (sh + 1))
              have hdiv :
                  (mant : ℝ) * ((pow2 (sh + 1) : ℝ)⁻¹) ≤ (fracNat : ℝ) := by
                -- Multiply `hge` by the inverse.
                have : (mant : ℝ) * ((pow2 (sh + 1) : ℝ)⁻¹) ≤
                    ((fracNat : ℝ) * (pow2 (sh + 1) : ℝ)) * ((pow2 (sh + 1) : ℝ)⁻¹) :=
                  mul_le_mul_of_nonneg_right hge (by exact inv_nonneg.2 (le_of_lt hpow_pos))
                -- simplify the RHS
                simpa [mul_assoc, hpow_pos.ne'] using this
              -- Convert `bpow (Int.negSucc sh)` to the inverse power of two.
              have hbneg : bpow (Int.negSucc sh) = ((pow2 (sh + 1) : Nat) : ℝ)⁻¹ := by
                simpa using (bpow_negSucc sh)
              have hb :
                  bpow exp = bpow (Int.negSucc sh) * bpow (-149 : Int) := by
                simpa [hexp, sub_eq_add_neg] using (bpow_add (Int.negSucc sh) (-149 : Int))
              -- Now combine.
              rw [hb, hbneg, hfrac]
              -- `mant * inv ≤ fracNat` gives the desired inequality after multiplying by
              -- `bpow(-149)`.
              have hb149_nonneg : 0 ≤ bpow (-149 : Int) := bpow_nonneg (-149 : Int)
              simpa [mul_assoc, hfrac] using (mul_le_mul_of_nonneg_right hdiv hb149_nonneg)
        -- Now finish by cases on whether we must round up into the smallest normal.
        cases hle : Nat.decLe (pow2 23) fracNat with
        | isTrue hge23 =>
            -- Output is the smallest normal, with value `2^-126`.
            have hOut' : roundDyadicPosUp mant exp = ofBits (mkBits false 1 0) := by
              simp [hOut, hle]
            have hto : toReal (ofBits (mkBits false 1 0)) = (pow2 23 : ℝ) * bpow (-149 : Int) := by
              have hexp : (1 : Nat) < 255 := by decide
              have hfrac : (0 : Nat) < 2 ^ 23 := by decide
              have h := toReal_ofBits_mkBits_fin (sign := false) (exp := 1) (frac := 0) hexp hfrac
              simpa [bpow] using h
            -- In this branch we only know `pow2 23 ≤ fracNat`, so we cannot use `hcore` to compare
            -- to the
            -- smallest normal. Instead, use a direct magnitude bound from `k < -126` to show the
            -- exact
            -- dyadic is ≤ `2^-126`.
            have hex_le : (mant : ℝ) * bpow exp ≤ (pow2 23 : ℝ) * bpow (-149 : Int) := by
              have hk1 : k + 1 ≤ (-126 : Int) :=
                Int.add_one_le_of_lt hkSub
              have hmant_lt' : (mant : ℝ) < (pow2 (log2m + 1) : ℝ) := by
                exact_mod_cast (mant_lt_pow2_log2_add1 mant hm)
              have hbpos : 0 < bpow exp := bpow_pos exp
              have hmul_lt :
                  (mant : ℝ) * bpow exp < (pow2 (log2m + 1) : ℝ) * bpow exp :=
                mul_lt_mul_of_pos_right hmant_lt' hbpos
              have hpow_as_bpow : (pow2 (log2m + 1) : ℝ) = bpow (Int.ofNat (log2m + 1)) := by
                simpa using (bpow_ofNat (log2m + 1)).symm
              have hmul_lt_k1 :
                  (mant : ℝ) * bpow exp < bpow (k + 1) := by
                have hEq : (Int.ofNat (log2m + 1)) + exp = k + 1 := by
                  -- `k = ofNat log2m + exp` by definition.
                  -- The goal is a rearrangement of addition on `Int`.
                  simp [k, add_assoc, add_comm]
                have hb : bpow (Int.ofNat (log2m + 1) + exp) = bpow (k + 1) := by
                  simpa using congrArg bpow hEq
                calc
                  (mant : ℝ) * bpow exp < (pow2 (log2m + 1) : ℝ) * bpow exp := hmul_lt
                  _ = bpow (Int.ofNat (log2m + 1) + exp) := by
                        simp [hpow_as_bpow, bpow_add, mul_comm]
                  _ = bpow (k + 1) := by simpa [hb]
              have hbpow_le : bpow (k + 1) ≤ bpow (-126 : Int) := by
                have hbase : (1 : ℝ) ≤ (2 : ℝ) := by norm_num
                have : (2 : ℝ) ^ (k + 1) ≤ (2 : ℝ) ^ (-126 : Int) :=
                  zpow_le_zpow_right₀ hbase hk1
                simpa [bpow, TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal] using
                  this
              have hle126 : (mant : ℝ) * bpow exp ≤ bpow (-126 : Int) :=
                (le_of_lt hmul_lt_k1).trans hbpow_le
              have hb126 : bpow (-126 : Int) = (pow2 23 : ℝ) * bpow (-149 : Int) := by
                calc
                  bpow (-126 : Int)
                      = bpow ((Int.ofNat 23) + (-149 : Int)) := by norm_num
                  _ = bpow (Int.ofNat 23) * bpow (-149 : Int) := by
                        simpa using (bpow_add (Int.ofNat 23) (-149 : Int))
                  _ = (pow2 23 : ℝ) * bpow (-149 : Int) := by
                        rw [bpow_ofNat 23]
              simpa [hb126] using hle126
            have hE : toEReal (ofBits (mkBits false 1 0) : IEEE32Exec) =
                ((pow2 23 : ℝ) * bpow (-149 : Int) : EReal) := by
              -- Compute `toEReal?` directly from the dyadic decode (avoids unfolding `toReal`).
              have hexp : (1 : Nat) < 255 := by decide
              have hfrac : (0 : Nat) < 2 ^ 23 := by decide
              have hdy :
                  toDyadic? (ofBits (mkBits false 1 0) : IEEE32Exec) =
                    some { sign := false, mant := pow2 23, exp := (Int.ofNat 1) - 150 } := by
                simpa using (toDyadic?_ofBits_mkBits_fin (sign := false) (exp := 1) (frac := 0) hexp
                  hfrac)
              have hnan : isNaN (ofBits (mkBits false 1 0) : IEEE32Exec) = false :=
                isNaN_eq_false_of_toDyadic?_some (hx := hdy)
              have hinf : isInf (ofBits (mkBits false 1 0) : IEEE32Exec) = false :=
                isInf_eq_false_of_toDyadic?_some (hx := hdy)
              have hE' : toEReal? (ofBits (mkBits false 1 0) : IEEE32Exec) =
                  some (((pow2 23 : ℝ) * bpow (-149 : Int)) : EReal) := by
                -- `exp = 1` corresponds to dyadic exponent `-149`, so the value is `2^23 * 2^-149 =
                -- 2^-126`.
                simp [IEEE32Exec.toEReal?, hnan, hinf, toReal_eq, hdy, dyadicToReal, bpow]
              simp [toEReal, hE']
            have hleE : ((mant : ℝ) * bpow exp : EReal) ≤ ((pow2 23 : ℝ) * bpow (-149 : Int) :
              EReal) := by
              exact_mod_cast hex_le
            simpa [hOut', hE] using hleE
        | isFalse hlt23 =>
            -- Output is the subnormal float with `fracNat` as fraction bits.
            have hOut' : roundDyadicPosUp mant exp = ofBits (mkBits false 0 fracNat) := by
              simp [hOut, hle]
            have hto : toReal (ofBits (mkBits false 0 fracNat)) = (fracNat : ℝ) * bpow (-149 : Int)
              := by
              have hexp : (0 : Nat) < 255 := by decide
              have hfrac : fracNat < 2 ^ 23 := by
                -- use `¬(2^23 ≤ fracNat)`.
                have : fracNat < pow2 23 := Nat.lt_of_not_ge hlt23
                simpa [pow2_eq_two_pow] using this
              -- `fracNat ≠ 0` in this subnormal branch, so the decode is exactly `fracNat *
              -- 2^-149`.
              have h := toReal_ofBits_mkBits_fin (sign := false) (exp := 0) (frac := fracNat) hexp
                hfrac
              simpa [hfracNat_ne0, bpow] using h
            have hE : toEReal (ofBits (mkBits false 0 fracNat) : IEEE32Exec) =
                ((fracNat : ℝ) * bpow (-149 : Int) : EReal) := by
              have hexp : (0 : Nat) < 255 := by decide
              have hfrac : fracNat < 2 ^ 23 := by
                have : fracNat < pow2 23 := Nat.lt_of_not_ge hlt23
                simpa [pow2_eq_two_pow] using this
              have hdy :
                  toDyadic? (ofBits (mkBits false 0 fracNat) : IEEE32Exec) =
                    some { sign := false, mant := fracNat, exp := (-149 : Int) } := by
                simpa [hfracNat_ne0] using
                  (toDyadic?_ofBits_mkBits_fin (sign := false) (exp := 0) (frac := fracNat) hexp
                    hfrac)
              have hnan : isNaN (ofBits (mkBits false 0 fracNat) : IEEE32Exec) = false :=
                isNaN_eq_false_of_toDyadic?_some (hx := hdy)
              have hinf : isInf (ofBits (mkBits false 0 fracNat) : IEEE32Exec) = false :=
                isInf_eq_false_of_toDyadic?_some (hx := hdy)
              have hE' : toEReal? (ofBits (mkBits false 0 fracNat) : IEEE32Exec) =
                  some (((fracNat : ℝ) * bpow (-149 : Int)) : EReal) := by
                simp [IEEE32Exec.toEReal?, hnan, hinf, toReal_eq, hdy, dyadicToReal, bpow]
              simp [toEReal, hE']
            have hleE : ((mant : ℝ) * bpow exp : EReal) ≤ ((fracNat : ℝ) * bpow (-149 : Int) :
              EReal) := by
              exact_mod_cast hcore
            simpa [hOut', hE] using hleE
      ·
        -- normal: shift to 24-bit mantissa using ceil when shifting right, then interpret as a
        -- normal float.
        have hkSub' : ¬ k < -126 := hkSub
        set m24 : Nat :=
          if log2m ≥ 23 then shiftRightCeilPow2 mant (log2m - 23) else Nat.shiftLeft mant (23 -
            log2m)
        set k' : Int := if m24 == pow2 24 then k + 1 else k
        set m24' : Nat := if m24 == pow2 24 then pow2 23 else m24
        -- If `k' > 127`, we again overflow to `+∞`.
        by_cases hkHi2 : k' > 127
        ·
          have hkHiNot : ¬ (127 : Int) < (↑mant.log2 + exp) := by
            simpa [hk, hlog2m] using hkHi'
          have hkUnder'' : ¬ (↑mant.log2 + exp) < -149 := by
            simpa [hk, hlog2m] using hkUnder'
          have hkNorm : ¬ (↑mant.log2 + exp) < -126 := by
            simpa [hk, hlog2m] using hkSub'
          have hOut : roundDyadicPosUp mant exp = posInf := by
            -- In the normal branch, overflow is detected by the computed exponent `k'`.
            --
            -- We use `conv_lhs` to rewrite the *expression* step-by-step, avoiding `simp` lemmas
            -- that
            -- turn `ite` goals into implications.
            have hkHi2_beq :
                (127 : Int) <
                  (if m24 == pow2 24 then (↑mant.log2 + exp + 1) else (↑mant.log2 + exp)) := by
              simpa [k', hk, hlog2m, add_assoc, add_left_comm, add_comm] using hkHi2
            -- The executable uses `m24 == 2^24` (a boolean test). The simplifier will sometimes
            -- rewrite this into a propositional test `m24 = 2^24`, so we prepare that form too.
            have hkHi2_eq :
                (127 : Int) <
                  (if m24 = pow2 24 then (↑mant.log2 + exp + 1) else (↑mant.log2 + exp)) := by
              by_cases hm24 : m24 = pow2 24
              ·
                -- `m24 == 2^24` is `true` in this branch.
                have hbeq : (m24 == pow2 24) = true := (beq_iff_eq).2 hm24
                simpa [hm24, hbeq] using hkHi2_beq
              ·
                have hbeq : (m24 == pow2 24) = false := (beq_eq_false_iff_ne (a := m24) (b := pow2
                  24)).2 hm24
                simpa [hm24, hbeq] using hkHi2_beq
            have hkHi2_expanded :
                (127 : Int) <
                  (if
                      (if 23 ≤ mant.log2 then shiftRightCeilPow2 mant (mant.log2 - 23) else mant <<<
                        (23 - mant.log2)) =
                        pow2 24 then
                    (↑mant.log2 + exp + 1)
                  else (↑mant.log2 + exp)) := by
              simpa [m24] using hkHi2_eq
            conv_lhs =>
              simp (config := { zeta := true }) [roundDyadicPosUp]
              rw [if_neg hkHiNot]
              rw [if_neg hkUnder'']
              rw [if_neg hkNorm]
              simp (config := { zeta := true }) [m24, k', m24']
              rw [if_pos hkHi2_expanded]
          have hE : toEReal (posInf : IEEE32Exec) = (⊤ : EReal) := by
            simp
          simp [hOut, hE]
        ·
          have hkHi2' : ¬ k' > 127 := hkHi2
          -- In the finite normal branch, the returned float decodes to the normal real value.
          set expNat : Nat := Int.toNat (k' + 127)
          set frac : Nat := m24' - pow2 23
          have hexpNat : expNat < 255 := by
            have hk_nonneg : 0 ≤ k' + 127 := by
              have hk_ge : (-126 : Int) ≤ k := (not_lt).1 hkSub'
              by_cases hcarry : m24 == pow2 24 <;> simp [k', hcarry] <;> linarith
            have hk_lt : k' + 127 < 255 := by
              have hk_le : k' ≤ 127 := (not_lt).1 (by simpa using hkHi2')
              linarith
            have : (expNat : Int) < (255 : Int) := by
              have hz : (expNat : Int) = k' + 127 := by
                simpa [expNat] using (Int.toNat_of_nonneg hk_nonneg)
              simpa [hz] using hk_lt
            exact (Int.ofNat_lt).1 this
          have hfrac_lt : frac < 2 ^ 23 := by
            -- The normal float format stores `frac := m24' - 2^23`, so we need `frac < 2^23`.
            --
            -- It suffices to show `m24' < 2^24 = 2^23 + 2^23`, because then
            -- `m24' - 2^23 < 2^23` by `Nat.sub_lt_left_of_lt_add`.
            have hm24_le : m24 ≤ pow2 24 := by
              -- Bound the 24-bit mantissa `m24` produced by either (ceil) right-shifting or
              -- left-shifting.
              by_cases hlog : log2m ≥ 23
              ·
                -- Right shift (with ceil): `m24 = ceil(mant / 2^(log2m-23))`.
                set sh : Nat := log2m - 23
                have hm24_def : m24 = shiftRightCeilPow2 mant sh := by
                  simp [m24, hlog, sh]
                -- Let `q := mant >>> sh = floor(mant / 2^sh)`. We show `q < 2^24`.
                have hmant_lt : mant < pow2 (log2m + 1) := mant_lt_pow2_log2_add1 mant hm
                have hsh : sh + 24 = log2m + 1 := by
                  -- `sh = log2m - 23` and `23 ≤ log2m`, so `sh + 23 = log2m`.
                  have h23 : 23 ≤ log2m := hlog
                  have h0 : sh + 23 = log2m := by
                    simpa [sh] using (Nat.sub_add_cancel h23)
                  have h1 : sh + 23 + 1 = log2m + 1 :=
                    congrArg (fun n => n + 1) h0
                  simpa [Nat.add_assoc] using h1
                have hmul : mant < (2 ^ sh) * pow2 24 := by
                  -- `pow2 (log2m+1) = pow2 (sh+24) = pow2 sh * pow2 24`.
                  have hpow : pow2 (log2m + 1) = pow2 sh * pow2 24 := by
                    calc
                      pow2 (log2m + 1) = pow2 (sh + 24) := by simp [hsh]
                      _ = pow2 sh * pow2 24 := by simp [pow2_add]
                  -- Rewrite `2^sh` as `pow2 sh`.
                  simpa [hpow, pow2_eq_two_pow, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc]
                    using hmant_lt
                have hq_lt : Nat.shiftRight mant sh < pow2 24 := by
                  -- `q = mant / 2^sh`, and `mant < 2^sh * 2^24`.
                  have : mant / (2 ^ sh) < pow2 24 := Nat.div_lt_of_lt_mul hmul
                  simpa [Nat.shiftRight_eq_div_pow] using this
                have hq1_le : Nat.shiftRight mant sh + 1 ≤ pow2 24 :=
                  Nat.succ_le_of_lt hq_lt
                -- `ceil ≤ floor + 1`.
                have hceil_le :
                    shiftRightCeilPow2 mant sh ≤ Nat.shiftRight mant sh + 1 :=
                  shiftRightCeilPow2_le_shiftRight_add1 mant sh
                -- Combine.
                exact (le_trans (by simpa [hm24_def] using hceil_le) hq1_le)
              ·
                -- Left shift (exact): `m24 = mant * 2^(23-log2m)`.
                set sh : Nat := 23 - log2m
                have hm24_def : m24 = Nat.shiftLeft mant sh := by
                  have : ¬ log2m ≥ 23 := hlog
                  simp [m24, this, sh]
                have hmant_lt : mant < pow2 (log2m + 1) := mant_lt_pow2_log2_add1 mant hm
                have hle : log2m ≤ 23 := Nat.le_of_lt (Nat.lt_of_not_ge hlog)
                have hsum : (log2m + 1) + sh = 24 := by
                  have h0 : log2m + sh = 23 := by
                    simpa [sh] using (Nat.add_sub_of_le hle)
                  have h1 : (log2m + sh) + 1 = 24 := by simp [h0]
                  have : (log2m + 1) + sh = (log2m + sh) + 1 := by
                    ac_rfl
                  simpa [this] using h1
                have hmul_lt : mant * pow2 sh < pow2 (log2m + 1) * pow2 sh :=
                  Nat.mul_lt_mul_of_pos_right hmant_lt (pow2_pos sh)
                have hpow : pow2 (log2m + 1) * pow2 sh = pow2 24 := by
                  calc
                    pow2 (log2m + 1) * pow2 sh = pow2 ((log2m + 1) + sh) := by
                      simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using (pow2_mul (log2m
                        + 1) sh)
                    _ = pow2 24 := by simp [hsum]
                have hm24_lt : m24 < pow2 24 := by
                  -- `shiftLeft` is multiplication by `2^sh = pow2 sh`.
                  have hshift : (mant <<< sh) = mant * pow2 sh := by
                    simp [Nat.shiftLeft_eq, pow2_eq_two_pow]
                  have hmul_lt24 : mant * pow2 sh < pow2 24 := by
                    simpa [hpow] using hmul_lt
                  have : mant <<< sh < pow2 24 := by
                    simpa [hshift] using hmul_lt24
                  simpa [hm24_def] using this
                exact le_of_lt hm24_lt
            have hm24'_lt : m24' < pow2 24 := by
              by_cases hcarry : m24 == pow2 24
              ·
                -- In the carry case we set `m24' := 2^23`.
                have hm24' : m24' = pow2 23 := by simp [m24', hcarry]
                simpa [hm24'] using (pow2_lt_pow2_succ 23)
              ·
                -- Otherwise `m24' = m24` and we already have `m24 ≤ 2^24` plus `m24 ≠ 2^24`.
                have hm24' : m24' = m24 := by simp [m24', hcarry]
                have hne : m24 ≠ pow2 24 := by
                  intro hEq
                  have : (m24 == pow2 24) = true := by simp [hEq]
                  exact hcarry this
                have : m24 < pow2 24 := lt_of_le_of_ne hm24_le hne
                simpa [hm24'] using this
            have hpow : pow2 24 = pow2 23 + pow2 23 := by
              -- `2^24 = 2 * 2^23 = 2^23 + 2^23`.
              calc
                pow2 24 = 2 * pow2 23 := by
                  -- `pow2 24 = 1 <<< 24 = 2 * (1 <<< 23) = 2 * pow2 23`.
                  simp [pow2, Nat.shiftLeft_succ]
                _ = pow2 23 + pow2 23 := by simpa using (Nat.two_mul (pow2 23))
            have : m24' - pow2 23 < pow2 23 := by
              have hlt : m24' < pow2 23 + pow2 23 := by simpa [hpow] using hm24'_lt
              by_cases hle : pow2 23 ≤ m24'
              · exact Nat.sub_lt_left_of_lt_add hle hlt
              ·
                have hle' : m24' ≤ pow2 23 := le_of_not_ge hle
                have hzero : m24' - pow2 23 = 0 := Nat.sub_eq_zero_of_le hle'
                -- `0 < 2^23`.
                have hpos : 0 < pow2 23 := pow2_pos 23
                simpa [hzero] using hpos
            -- Convert `pow2 23` to `2^23`.
            simpa [frac, pow2_eq_two_pow] using this
          have hto :
              toReal (ofBits (mkBits false expNat frac)) =
                ((pow2 23 + frac : Nat) : ℝ) * bpow ((Int.ofNat expNat) - 150) := by
            have h :=
              toReal_ofBits_mkBits_fin (sign := false) (exp := expNat) (frac := frac) hexpNat
                hfrac_lt
            -- `expNat ≠ 0` here since `k' ≥ -126`.
            have hexp0 : expNat ≠ 0 := by
              have hk_ge : (-126 : Int) ≤ k := (not_lt).1 hkSub'
              have hk'_ge : (-126 : Int) ≤ k' := by
                by_cases hcarry : m24 == pow2 24
                ·
                  have : (-126 : Int) ≤ k + 1 := by linarith [hk_ge]
                  simpa [k', hcarry] using this
                ·
                  simpa [k', hcarry] using hk_ge
              have hk_pos : (0 : Int) < k' + 127 := by linarith [hk'_ge]
              have hk_nonneg : 0 ≤ k' + 127 := le_of_lt hk_pos
              have hz : (expNat : Int) = k' + 127 := by
                simpa [expNat] using (Int.toNat_of_nonneg hk_nonneg)
              have : (0 : Int) < (expNat : Int) := by simpa [hz] using hk_pos
              exact Nat.ne_of_gt ((Int.ofNat_lt).1 (by simpa using this))
            by_cases h0 : expNat = 0
            · exact False.elim (hexp0 h0)
            ·
              have h' :
                  toReal (ofBits (mkBits false expNat frac)) =
                    ((pow2 23 + frac : Nat) : ℝ) * neuralBpow binaryRadix ((Int.ofNat expNat) -
                      150) := by
                have h1 := h
                -- Keep the `exp ≠ 0` branch of the decoding lemma.
                rw [if_neg h0] at h1
                -- Evaluate the remaining `if false then _ else _` by definitional reduction.
                exact h1.trans (by rfl)
              -- `bpow` is definitional (`neural_bpow binary_radix`).
              dsimp [bpow]
              exact h'
          -- Show the exact dyadic is ≤ `m24 * 2^(k-23)` and relate this to the returned float.
          have h_exact_le_m24 :
              (mant : ℝ) * bpow exp ≤ (m24 : ℝ) * bpow (k - 23) := by
            by_cases hlog : log2m ≥ 23
            ·
              set sh : Nat := log2m - 23
              have hm24 : m24 = shiftRightCeilPow2 mant sh := by
                simp [m24, hlog, sh]
              have hgeNat : mant ≤ m24 * pow2 sh := by
                -- `mant ≤ ceil(mant/2^sh) * 2^sh`.
                simpa [hm24] using shiftRightCeilPow2_mul_pow2_ge mant sh
              have hge : (mant : ℝ) ≤ (m24 : ℝ) * (pow2 sh : ℝ) := by exact_mod_cast hgeNat
              have hbsh : (pow2 sh : ℝ) = bpow (Int.ofNat sh) :=
                (bpow_ofNat sh).symm
              have hk23 : k - 23 = exp + (Int.ofNat sh) := by
                have hNat : log2m = 23 + sh := by
                  -- `sh = log2m - 23`, so `log2m - 23 + 23 = log2m` rearranges to `log2m = 23 +
                  -- sh`.
                  simpa [sh, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using
                    (Nat.sub_add_cancel hlog).symm
                have : (Int.ofNat log2m) = 23 + (Int.ofNat sh) := by
                  simp [hNat, Nat.cast_add]
                -- expand `k` and cancel the `23`
                calc
                  k - 23 = (Int.ofNat log2m + exp) - 23 := by simp [k]
                  _ = ((23 + Int.ofNat sh) + exp) - 23 := by simpa [this]
                  _ = exp + Int.ofNat sh := by ring
              have hb : bpow (k - 23) = bpow exp * bpow (Int.ofNat sh) := by
                -- `bpow (exp + sh)`.
                simpa [hk23, add_comm, add_left_comm, add_assoc] using (bpow_add exp (Int.ofNat sh))
              -- Multiply `hge` by `bpow exp` and rewrite.
              have hbpos : 0 ≤ bpow exp := bpow_nonneg exp
              have : (mant : ℝ) * bpow exp ≤ ((m24 : ℝ) * (pow2 sh : ℝ)) * bpow exp :=
                mul_le_mul_of_nonneg_right hge hbpos
              -- rearrange and rewrite `pow2 sh` as `bpow sh`.
              calc
                (mant : ℝ) * bpow exp ≤ ((m24 : ℝ) * (pow2 sh : ℝ)) * bpow exp := this
                _ = (m24 : ℝ) * (bpow exp * bpow (Int.ofNat sh)) := by
                      simp [hbsh, mul_assoc, mul_comm]
                _ = (m24 : ℝ) * bpow (k - 23) := by
                      simp [hb]
            ·
              -- left shift case is exact scaling.
              set sh : Nat := 23 - log2m
              have hm24 : m24 = Nat.shiftLeft mant sh := by
                have : ¬ log2m ≥ 23 := hlog
                simp [m24, this, sh]
              have hk23 : k - 23 = exp - (Int.ofNat sh) := by
                have hNat : log2m + sh = 23 := by
                  have hle : log2m ≤ 23 := le_of_lt (Nat.lt_of_not_ge hlog)
                  simpa [sh] using (Nat.add_sub_of_le hle)
                have hInt : (Int.ofNat log2m) + (Int.ofNat sh) = (23 : Int) := by
                  -- Cast the natural-number identity `log2m + sh = 23` into `Int`.
                  -- We keep it in the expanded “`ofNat` + `ofNat`” form for later rewriting.
                  have := congrArg (fun n : Nat => (Int.ofNat n)) hNat
                  simpa [Int.natCast_add] using this
                calc
                  k - 23 = (Int.ofNat log2m + exp) - 23 := by simp [hk]
                  _ = (Int.ofNat log2m + exp) - (Int.ofNat log2m + Int.ofNat sh) := by
                        -- Replace the `23` using `hInt` (in the needed direction).
                        simpa using
                          congrArg (fun t : Int => (Int.ofNat log2m + exp) - t) hInt.symm
                  _ = exp - Int.ofNat sh := by
                        abel
              -- `mant*2^exp = (mant<<sh) * 2^(exp-sh)`.
              have hShift : ((Nat.shiftLeft mant sh : Nat) : ℝ) = (mant : ℝ) * bpow (Int.ofNat sh)
                := by
                have : ((Nat.shiftLeft mant sh : Nat) : ℝ) = (mant : ℝ) * (pow2 sh : ℝ) := by
                  simp [Nat.shiftLeft_eq, pow2_eq_two_pow, Nat.cast_mul, Nat.cast_pow]
                have hbsh : (pow2 sh : ℝ) = bpow (Int.ofNat sh) := by
                  simpa using (bpow_ofNat sh).symm
                simpa [hbsh] using this
              have hb : bpow exp = bpow (exp - Int.ofNat sh) * bpow (Int.ofNat sh) := by
                simpa [sub_eq_add_neg, add_assoc] using (bpow_add (exp - Int.ofNat sh) (Int.ofNat
                  sh))
              -- Finish by rewriting to equality.
              have : (mant : ℝ) * bpow exp = (m24 : ℝ) * bpow (k - 23) := by
                rw [hm24, hk23, hShift, hb]
                ac_rfl
              exact le_of_eq this
          -- Relate the returned float value to `m24`/`k`.
          have hcarry : (m24 == pow2 24) = true ∨ (m24 == pow2 24) = false := by
            cases h : (m24 == pow2 24) <;> simp
          -- In either case, the returned float is `m24 * 2^(k-23)` as an `EReal` (or `⊤` handled
          -- above).
          have hret_eq :
              ((pow2 23 + frac : Nat) : ℝ) * bpow (k' - 23) = (m24 : ℝ) * bpow (k - 23) := by
            -- First, show `pow2 23 ≤ m24'` so that `pow2 23 + (m24' - pow2 23) = m24'`.
            have hm24_ge : pow2 23 ≤ m24 := by
              -- `m24` is the “24-bit mantissa” obtained by shifting `mant` so its leading bit sits
              -- at position 23.
              -- Since `2^log2m ≤ mant`, the shift (either right with ceil, or left) ensures `2^23 ≤
              -- m24`.
              have hpow : pow2 log2m ≤ mant := pow2_log2_le mant hm
              by_cases hlog : log2m ≥ 23
              ·
                set sh : Nat := log2m - 23
                have hm24' : m24 = shiftRightCeilPow2 mant sh := by
                  simp [m24, hlog, sh]
                have hsh : sh + 23 = log2m := by
                  -- `sh = log2m - 23`.
                  simpa [sh] using (Nat.sub_add_cancel hlog)
                -- Prove `2^23 ≤ ceil(mant / 2^sh)` by contradiction: otherwise the reconstructed
                -- value
                -- would be < `2^log2m ≤ mant`.
                have : pow2 23 ≤ shiftRightCeilPow2 mant sh := by
                  by_contra hlt
                  have hlt' : shiftRightCeilPow2 mant sh < pow2 23 := Nat.lt_of_not_ge hlt
                  have hmul_lt :
                      shiftRightCeilPow2 mant sh * pow2 sh < pow2 23 * pow2 sh :=
                    Nat.mul_lt_mul_of_pos_right hlt' (pow2_pos sh)
                  have hpowMul : pow2 23 * pow2 sh = pow2 log2m := by
                    calc
                      pow2 23 * pow2 sh = pow2 (23 + sh) := by
                        simpa using (pow2_mul 23 sh)
                      _ = pow2 (sh + 23) := by simp [Nat.add_comm]
                      _ = pow2 log2m := by simp [hsh]
                  have hmul_lt' :
                      shiftRightCeilPow2 mant sh * pow2 sh < pow2 log2m := by
                    simpa [hpowMul] using hmul_lt
                  have hge : mant ≤ shiftRightCeilPow2 mant sh * pow2 sh :=
                    shiftRightCeilPow2_mul_pow2_ge mant sh
                  have : mant < pow2 log2m := lt_of_le_of_lt hge hmul_lt'
                  exact (not_lt_of_ge hpow) this
                simpa [hm24'] using this
              ·
                -- `log2m < 23`, so we shift left by `23 - log2m`.
                set sh : Nat := 23 - log2m
                have hm24' : m24 = Nat.shiftLeft mant sh := by
                  have : ¬ log2m ≥ 23 := hlog
                  simp [m24, this, sh]
                have hmul : pow2 log2m * pow2 sh ≤ mant * pow2 sh :=
                  Nat.mul_le_mul_right (pow2 sh) hpow
                have hle : log2m ≤ 23 := le_of_lt (Nat.lt_of_not_ge hlog)
                have hsum : log2m + sh = 23 := by
                  simpa [sh] using (Nat.add_sub_of_le hle)
                have hpow23 : pow2 log2m * pow2 sh = pow2 23 := by
                  calc
                    pow2 log2m * pow2 sh = pow2 (log2m + sh) := by
                      simpa using (pow2_mul log2m sh)
                    _ = pow2 23 := by simp [hsum]
                have hmul' : pow2 23 ≤ mant * pow2 sh := by
                  simpa [hpow23] using hmul
                have : pow2 23 ≤ Nat.shiftLeft mant sh := by
                  simpa [Nat.shiftLeft_eq, pow2_eq_two_pow, Nat.mul_assoc, Nat.mul_left_comm,
                    Nat.mul_comm] using hmul'
                simpa [hm24'] using this
            have hm24'_ge : pow2 23 ≤ m24' := by
              by_cases hcarry' : m24 == pow2 24
              · simp [m24', hcarry']
              · simpa [m24', hcarry'] using hm24_ge
            have hmNat : (pow2 23 + frac : Nat) = m24' := by
              -- `frac = m24' - 2^23`.
              simpa [frac] using (Nat.add_sub_of_le hm24'_ge)
            have hmR : ((pow2 23 + frac : Nat) : ℝ) = (m24' : ℝ) := by
              exact_mod_cast hmNat
            -- Rewrite the LHS to expose the carry logic.
            rw [hmR]
            by_cases hcarry' : m24 == pow2 24
            ·
              -- carry: `k' = k+1`, `m24' = 2^23`, and `m24 = 2^24`.
              have hk' : k' = k + 1 := by simp [k', hcarry']
              have hm24' : m24' = pow2 23 := by simp [m24', hcarry']
              have hm24Nat : m24 = pow2 24 := (beq_iff_eq).1 hcarry'
              -- Reduce to a simple power-of-two identity.
              have hExp : (k + 1) - 23 = (k - 23) + 1 := by ring
              have hbpow :
                  bpow ((k + 1) - 23) = bpow (k - 23) * bpow (1 : Int) := by
                simpa [hExp] using (bpow_add (k - 23) (1 : Int))
              have hb1 : bpow (1 : Int) = (2 : ℝ) := by
                -- `bpow 1 = 2^1 = 2`.
                simpa [pow2_eq_two_pow] using (bpow_ofNat 1)
              have hpow24Nat : pow2 24 = 2 * pow2 23 := by
                simp [pow2, Nat.shiftLeft_succ]
              have hpow24 : ((pow2 24 : Nat) : ℝ) = (2 : ℝ) * (pow2 23 : ℝ) := by
                exact_mod_cast hpow24Nat
              -- Rewrite everything and finish by commutativity/associativity.
              -- (The `ac_rfl` avoids brittle `simp` normalization.)
              simp [hk', hm24', hm24Nat, hbpow, hb1, hpow24, mul_assoc, mul_left_comm, mul_comm]
            ·
              -- no carry: `k'=k`, `m24'=m24`.
              have hk' : k' = k := by simp [k', hcarry']
              have hm24' : m24' = m24 := by simp [m24', hcarry']
              simp [hk', hm24']
          have hle_final :
              ((mant : ℝ) * bpow exp : EReal) ≤
                toEReal (ofBits (mkBits false expNat frac) : IEEE32Exec) := by
            -- `toEReal` agrees with `toReal` as soon as the value is not NaN/Inf.
            have hE :
                toEReal (ofBits (mkBits false expNat frac) : IEEE32Exec) =
                  (((pow2 23 + frac : Nat) : ℝ) * bpow ((Int.ofNat expNat) - 150) : EReal) := by
              -- Get a `toDyadic?` witness (hence not NaN/Inf) for the concrete `ofBits (mkBits
              -- ...)`.
              have hdy_exists :
                  ∃ d, toDyadic? (ofBits (mkBits false expNat frac)) = some d := by
                have hdy :=
                  toDyadic?_ofBits_mkBits_fin (sign := false) (exp := expNat) (frac := frac) hexpNat
                    hfrac_lt
                by_cases h0 : expNat = 0
                ·
                  by_cases hf0 : frac = 0
                  ·
                    refine ⟨{ sign := false, mant := 0, exp := 0 }, ?_⟩
                    simpa [h0, hf0] using hdy
                  ·
                    refine ⟨{ sign := false, mant := frac, exp := (-149 : Int) }, ?_⟩
                    simpa [h0, hf0] using hdy
                ·
                  refine ⟨{ sign := false, mant := pow2 23 + frac, exp := (Int.ofNat expNat) - 150
                    }, ?_⟩
                  simpa [h0] using hdy
              rcases hdy_exists with ⟨dout, hdout⟩
              have hnan : isNaN (ofBits (mkBits false expNat frac) : IEEE32Exec) = false :=
                isNaN_eq_false_of_toDyadic?_some (hx := hdout)
              have hinf : isInf (ofBits (mkBits false expNat frac) : IEEE32Exec) = false :=
                isInf_eq_false_of_toDyadic?_some (hx := hdout)
              have hE? :
                  IEEE32Exec.toEReal? (ofBits (mkBits false expNat frac) : IEEE32Exec) =
                    some (toReal (ofBits (mkBits false expNat frac) : IEEE32Exec) : EReal) := by
                simp [IEEE32Exec.toEReal?, hnan, hinf]
              have hE0 :
                  toEReal (ofBits (mkBits false expNat frac) : IEEE32Exec) =
                    (toReal (ofBits (mkBits false expNat frac) : IEEE32Exec) : EReal) :=
                toEReal_of_toEReal? hE?
              -- Rewrite `toReal` using the previously-established real decoding equation `hto`.
              have htoE :
                  (toReal (ofBits (mkBits false expNat frac) : IEEE32Exec) : EReal) =
                    ((((pow2 23 + frac : Nat) : ℝ) * bpow ((Int.ofNat expNat) - 150)) : EReal) :=
                congrArg (fun r : ℝ => (r : EReal)) hto
              -- Assemble without `simp` (avoids `maxRecDepth` blowups).
              exact hE0.trans htoE
            have hk_nonneg : 0 ≤ k' + 127 := by
              have hk_ge : (-126 : Int) ≤ k := (not_lt).1 hkSub'
              have hk'_ge : (-126 : Int) ≤ k' := by
                by_cases hcarry : m24 == pow2 24
                ·
                  have : (-126 : Int) ≤ k + 1 := by linarith [hk_ge]
                  simpa [k', hcarry] using this
                ·
                  simpa [k', hcarry] using hk_ge
              linarith [hk'_ge]
            have hexpPow : bpow ((↑expNat : Int) - 150) = bpow (k' - 23) := by
              have hexpInt' : (↑expNat : Int) = k' + 127 := by
                simpa [expNat] using (Int.toNat_of_nonneg hk_nonneg)
              have hExp : (↑expNat : Int) - 150 = k' - 23 := by
                calc
                  (↑expNat : Int) - 150 = (k' + 127) - 150 := by simp [hexpInt']
                  _ = k' - 23 := by ring
              simp [hExp]
            -- chain in ℝ then cast to EReal
            have hleR :
                (mant : ℝ) * bpow exp ≤
                  ((pow2 23 + frac : Nat) : ℝ) * bpow (k' - 23) := by
              -- from exact ≤ m24*2^(k-23) and the equality `hret_eq`
              have htmp : (mant : ℝ) * bpow exp ≤ (m24 : ℝ) * bpow (k - 23) := h_exact_le_m24
              -- rewrite RHS
              have :
                  (m24 : ℝ) * bpow (k - 23) =
                    ((pow2 23 + frac : Nat) : ℝ) * bpow (k' - 23) := by
                exact hret_eq.symm
              exact htmp.trans_eq this
            have hleR' :
                (mant : ℝ) * bpow exp ≤
                  ((pow2 23 + frac : Nat) : ℝ) * bpow ((Int.ofNat expNat) - 150) := by
              -- Rewrite the RHS of `hleR` using `hexpPow`.
              calc
                (mant : ℝ) * bpow exp ≤ ((pow2 23 + frac : Nat) : ℝ) * bpow (k' - 23) := hleR
                _ = ((pow2 23 + frac : Nat) : ℝ) * bpow ((Int.ofNat expNat) - 150) := by
                      -- `hexpPow` is stated in the opposite direction, so use `.symm`.
                      exact
                        congrArg (fun t : ℝ => ((pow2 23 + frac : Nat) : ℝ) * t) hexpPow.symm
            have hleE :
                ((mant : ℝ) * bpow exp : EReal) ≤
                  ((((pow2 23 + frac : Nat) : ℝ) * bpow ((Int.ofNat expNat) - 150)) : EReal) := by
              exact_mod_cast hleR'
            -- Rewrite the RHS using `hE`.
            -- (We avoid `simp` here; `rw` is more robust and avoids deep simp recursion.)
            rw [hE]
            exact hleE
          -- Finish by rewriting the executable output directly.
          have hkHiNot : ¬ (127 : Int) < (↑mant.log2 + exp) := by
            simpa [hk, hlog2m] using hkHi'
          have hkUnder'' : ¬ (↑mant.log2 + exp) < -149 := by
            simpa [hk, hlog2m] using hkUnder'
          have hkNorm : ¬ (↑mant.log2 + exp) < -126 := by
            simpa [hk, hlog2m] using hkSub'
          -- After `simp (config := {zeta := true}) [roundDyadicPosUp]`, the last guard is the
          -- overflow check
          -- `if k' > 127 then posInf else ...`, but `k'` and `m24` have already been zeta-expanded,
          -- so we
          -- record a matching “not-overflow” fact in that expanded shape for `rw [if_neg ...]`.
          have hkHi2_guard :
              ¬ (127 : Int) <
                (if
                    (if 23 ≤ mant.log2 then
                        shiftRightCeilPow2 mant (mant.log2 - 23)
                      else
                        mant <<< (23 - mant.log2)) = pow2 24 then
                  ↑mant.log2 + exp + 1
                else
                  ↑mant.log2 + exp) := by
            have hkHi2_guard0 : ¬ (127 : Int) < k' := by
              simpa [gt_iff_lt] using hkHi2'
            -- Expand `k'` and `m24` and rewrite `log2m` to `mant.log2`.
            simpa [k', hk, hlog2m, m24] using hkHi2_guard0
          have hOut : roundDyadicPosUp mant exp = ofBits (mkBits false expNat frac) := by
            conv_lhs =>
              simp (config := { zeta := true }) [roundDyadicPosUp]
              rw [if_neg hkHiNot]
              rw [if_neg hkUnder'']
              rw [if_neg hkNorm]
              -- Unfold `m24/k'/m24'` and then discharge the overflow guard in its expanded form.
              simp (config := { zeta := true }) [m24, k', m24']
              rw [if_neg hkHi2_guard]
            simp (config := { zeta := true }) [expNat, frac, k', m24', m24, hk, log2m]
          simpa [hOut] using hle_final

/-
## Signed dyadics and endpoint ops

Once we have soundness for the positive directed-rounding kernels, we can lift them to the signed
versions (`roundDyadicDown` / `roundDyadicUp`) and then to the endpoint operations
(`addDown/addUp/mulDown/mulUp`), under finiteness assumptions on the operands.
-/

/--
`toEReal` respects negation on the finite/dyadic branch.

We phrase this lemma using the dyadic decode witness `toDyadic? x = some d`, which guarantees:
- `x` is finite (hence `toEReal` agrees with `toReal`), and
- `toReal (neg x) = -toReal x` (via `toReal_neg_eq_neg`).
-/
theorem toEReal_neg_of_toDyadic?_some (x : IEEE32Exec) {d : Dyadic} (hx : toDyadic? x = some d) :
    toEReal (neg x) = -toEReal x := by
  have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hx)
  have hxInf : isInf x = false := isInf_eq_false_of_toDyadic?_some (hx := hx)
  have hfin : isFinite x = true :=
    isFinite_eq_true_of_isNaN_eq_false_of_isInf_eq_false (x := x) hxNaN hxInf
  have hdyNeg : toDyadic? (neg x) = some { sign := (!d.sign), mant := d.mant, exp := d.exp } :=
    toDyadic?_neg_of_toDyadic?_some (x := x) (d := d) hx
  have hfinNeg : isFinite (neg x) = true := by
    have hnan : isNaN (neg x) = false := isNaN_eq_false_of_toDyadic?_some (hx := hdyNeg)
    have hinf : isInf (neg x) = false := isInf_eq_false_of_toDyadic?_some (hx := hdyNeg)
    exact isFinite_eq_true_of_isNaN_eq_false_of_isInf_eq_false (x := neg x) hnan hinf
  have hxE : toEReal? x = some (toReal x : EReal) :=
    toEReal?_eq_some_toReal_of_isFinite_eq_true (x := x) hfin
  have hxNegE : toEReal? (neg x) = some (toReal (neg x) : EReal) :=
    toEReal?_eq_some_toReal_of_isFinite_eq_true (x := neg x) hfinNeg
  have hxReal : toReal (neg x) = -toReal x := toReal_neg_eq_neg (x := x) (d := d) hx
  have hxE0 : toEReal x = (toReal x : EReal) := toEReal_of_toEReal? hxE
  have hxNegE0 : toEReal (neg x) = (toReal (neg x) : EReal) := toEReal_of_toEReal? hxNegE
  have hxRealE : (toReal (neg x) : EReal) = (-toReal x : EReal) :=
    congrArg (fun r : ℝ => (r : EReal)) hxReal
  calc
    toEReal (neg x) = (toReal (neg x) : EReal) := hxNegE0
    _ = (-toReal x : EReal) := hxRealE
    _ = -(toReal x : EReal) := by simp
    _ = -toEReal x := by simp [hxE0]

/-
`IEEE32Exec.neg` flips only the sign bit. For our `EReal` semantics, we will often want to push
negation through `toEReal`. This is automatic in the finite/dyadic case
(`toEReal_neg_of_toDyadic?_some`), but we also need a lemma for `±∞`.

To avoid relying on private bitfield lemmas from other files, we prove the small bitwise facts we
need directly using the `Nat.testBit` API from Lean's core bitwise theory.
-/

/-- `signMask` is the single bit `2^31` (as a `Nat`). -/
@[simp] private lemma signMask_toNat : (signMask : UInt32).toNat = 2 ^ 31 := by decide

/-- `expAllOnes` is the mask `2^8-1` (as a `Nat`). -/
@[simp] private lemma expAllOnes_toNat : (expAllOnes : UInt32).toNat = 2 ^ 8 - 1 := by decide

/-- `fracMask` is the mask `2^23-1` (as a `Nat`). -/
@[simp] private lemma fracMask_toNat : (fracMask : UInt32).toNat = 2 ^ 23 - 1 := by decide

/--
`signBit x` is exactly the `31`-st bit of `x.bits` (viewed as a `Nat`).

This is the bridge we use to reason about sign-bit toggling under `neg` using the core lemma
`Nat.testBit_xor`.
-/
private lemma signBit_eq_testBit31 (x : IEEE32Exec) :
    signBit x = x.bits.toNat.testBit 31 := by
  classical
  have hSignMask : signMask.toNat = 2 ^ 31 := signMask_toNat
  by_cases hb : x.bits.toNat.testBit 31
  · -- bit 31 is set, so `x.bits &&& signMask` is nonzero.
    have hnat : x.bits.toNat &&& signMask.toNat = 2 ^ 31 := by
      simpa [hSignMask, Nat.and_two_pow, hb] using (Nat.and_two_pow x.bits.toNat 31)
    have hne : (x.bits &&& signMask) ≠ 0 := by
      intro h0
      have h0' : (x.bits &&& signMask).toNat = 0 := by simp [h0]
      have hAnd0 : (x.bits.toNat &&& signMask.toNat) = 0 := by
        simpa [UInt32.toNat_and] using h0'
      have hPow0 : (2 ^ 31 : Nat) = 0 := by
        -- `x.bits.toNat &&& signMask.toNat` is simultaneously `2^31` (because bit 31 is set) and
        -- `0`
        -- (because the underlying `UInt32` result is `0`), contradiction.
        exact hnat.symm.trans hAnd0
      exact (Nat.ne_of_gt (Nat.pow_pos (a := 2) (n := 31) (by decide : 0 < (2 : Nat)))) hPow0
    have hbne : (x.bits &&& signMask != 0) = true := (bne_iff_ne).2 hne
    simp [IEEE32Exec.signBit, hb, hbne]
  · -- bit 31 is not set, so `x.bits &&& signMask = 0`.
    have hnat : x.bits.toNat &&& signMask.toNat = 0 := by
      simpa [hSignMask, Nat.and_two_pow, hb] using (Nat.and_two_pow x.bits.toNat 31)
    have heq : (x.bits &&& signMask) = 0 := by
      apply (UInt32.toNat_inj).1
      simpa [UInt32.toNat_and, hnat]
    simp [IEEE32Exec.signBit, hb, heq]

/--
`signBit` toggles under `neg` (on the non-NaN branch).

This is the only fact about `neg` we need in the `±∞` case of `toEReal_neg_of_isNaN_eq_false`.
-/
private lemma signBit_neg_of_isNaN_eq_false (x : IEEE32Exec) (hnan : isNaN x = false) :
    signBit (neg x) = (!signBit x) := by
  have hnegBits : (neg x).bits = x.bits ^^^ signMask := by
    simp [IEEE32Exec.neg, hnan, ofBits]
  have hNeg : signBit (neg x) = (neg x).bits.toNat.testBit 31 :=
    signBit_eq_testBit31 (x := neg x)
  have hOrig : signBit x = x.bits.toNat.testBit 31 :=
    signBit_eq_testBit31 (x := x)
  -- Reduce everything to the nat-level `testBit` statement, then use `Nat.testBit_xor`.
  rw [hNeg, hOrig, hnegBits]
  have hx :
      ((x.bits ^^^ signMask).toNat).testBit 31 =
        Bool.xor (x.bits.toNat.testBit 31) (signMask.toNat.testBit 31) := by
    simp [UInt32.toNat_xor, Nat.testBit_xor]
  rw [hx]
  have hmask : signMask.toNat.testBit 31 = true := by
    simpa [signMask_toNat] using (Nat.testBit_two_pow_self (n := 31))
  rw [hmask]
  cases hb : x.bits.toNat.testBit 31 <;> simp

/-! ## Negation / sign-bit flip helpers -/

-- The core bitfield lemmas about sign-bit flips live in `NN.Floats.IEEEExec.Negation`.

/-! ## Negation preserves “not NaN” -/

/--
If `x` is not a NaN then `neg x` is not a NaN.

We only state the direction we actually need in the directed-rounding development: negative
directed rounding is implemented as `neg` of a positive kernel, so we need to know this cannot
introduce a NaN when starting from a non-NaN value.
-/
theorem isNaN_neg_eq_false_of_isNaN_eq_false (x : IEEE32Exec) (hnan : isNaN x = false) :
    isNaN (neg x) = false := by
  -- In the non-NaN branch, `neg` is `ofBits (x.bits ^^^ signMask)`.
  have hexp : expField (neg x) = expField x := by
    simpa [IEEE32Exec.neg, hnan, ofBits] using (expField_ofBits_xor_signMask (b := x.bits))
  have hfrac : fracField (neg x) = fracField x := by
    simpa [IEEE32Exec.neg, hnan, ofBits] using (fracField_ofBits_xor_signMask (b := x.bits))
  -- `isNaN` depends only on exponent+fraction fields.
  have hEq : isNaN (neg x) = isNaN x := by
    unfold IEEE32Exec.isNaN
    simp [hexp, hfrac]
  -- Rewrite the goal using `hEq`.
  simpa [hEq] using hnan

/--
`toEReal` commutes with `neg` as long as we are not in the NaN case.

This is a small but important glue lemma: it lets us lift the positive rounding kernels to the
signed directed-rounding functions (`roundDyadicDown` / `roundDyadicUp`) without introducing new
floating-point corner cases.
-/
theorem toEReal_neg_of_isNaN_eq_false (x : IEEE32Exec) (hnan : isNaN x = false) :
    toEReal (neg x) = -toEReal x := by
  classical
  cases hx : toDyadic? x with
  | some d =>
      exact toEReal_neg_of_toDyadic?_some (x := x) (d := d) hx
  | none =>
      have hxInf : isInf x = true :=
        isInf_eq_true_of_toDyadic?_eq_none_of_isNaN_eq_false (x := x) hx hnan
      -- Compute `toEReal x` and `toEReal (neg x)` in the `±∞` branch.
      have hxE : toEReal? x = some (if signBit x then (⊥ : EReal) else (⊤ : EReal)) := by
        simp [IEEE32Exec.toEReal?, hnan, hxInf]
      -- `neg` xors only the sign bit, so exponent+fraction fields are unchanged.
      have hexp : expField (neg x) = expField x := by
        simpa [IEEE32Exec.neg, hnan, ofBits] using (expField_ofBits_xor_signMask (b := x.bits))
      have hfrac : fracField (neg x) = fracField x := by
        simpa [IEEE32Exec.neg, hnan, ofBits] using (fracField_ofBits_xor_signMask (b := x.bits))
      have hxNegInf : isInf (neg x) = true := by
        have hEq : isInf (neg x) = isInf x := by
          unfold IEEE32Exec.isInf
          simp [hexp, hfrac]
        simpa [hEq] using hxInf
      have hxNegNaN : isNaN (neg x) = false := by
        have hEq : isNaN (neg x) = isNaN x := by
          unfold IEEE32Exec.isNaN
          simp [hexp, hfrac]
        simpa [hEq] using hnan
      have hxNegE : toEReal? (neg x) = some (if signBit (neg x) then (⊥ : EReal) else (⊤ : EReal))
        := by
        simp [IEEE32Exec.toEReal?, hxNegNaN, hxNegInf]
      have hs : signBit (neg x) = (!signBit x) := signBit_neg_of_isNaN_eq_false (x := x) hnan
      -- Reduce to the two `±∞` cases by splitting on the sign bit.
      cases hsb : signBit x <;> simp [toEReal, hxE, hxNegE, hs, hsb]

/--
Soundness of `roundDyadicDown`: the result is ≤ the exact dyadic real value (in `EReal`).

This is the key lemma needed to justify `addDown/mulDown` as interval *lower endpoints*.
-/
theorem isNaN_roundDyadicPosUp_eq_false (mant : Nat) (exp : Int) (hm : mant ≠ 0) :
    isNaN (roundDyadicPosUp mant exp) = false := by
  -- If `roundDyadicPosUp` were a NaN, then `toEReal?` would be `none`, hence `toEReal = 0`.
  -- But `toEReal_roundDyadicPosUp_ge` shows it is ≥ the strictly positive exact value.
  cases hnan' : isNaN (roundDyadicPosUp mant exp) with
  | false =>
      simp
  | true =>
      have hE? : toEReal? (roundDyadicPosUp mant exp) = none := by
        simp [IEEE32Exec.toEReal?, hnan']
      have hE : toEReal (roundDyadicPosUp mant exp) = 0 := by
        simp [toEReal, hE?]
      have hge :
          ((mant : ℝ) * bpow exp : EReal) ≤ toEReal (roundDyadicPosUp mant exp) :=
        toEReal_roundDyadicPosUp_ge (mant := mant) (exp := exp) hm
      have hle0 : ((mant : ℝ) * bpow exp : EReal) ≤ (0 : EReal) := by
        simpa [hE] using hge
      have hle0R : (mant : ℝ) * bpow exp ≤ (0 : ℝ) := by
        have hle0' : ((mant : ℝ) * bpow exp : EReal) ≤ ((0 : ℝ) : EReal) := by
          simpa using hle0
        exact (EReal.coe_le_coe_iff).1 hle0'
      have hmpos : (0 : ℝ) < (mant : ℝ) := Nat.cast_pos.2 (Nat.pos_of_ne_zero hm)
      have hbpos : (0 : ℝ) < bpow exp := bpow_pos exp
      have hpos : (0 : ℝ) < (mant : ℝ) * bpow exp := mul_pos hmpos hbpos
      have : False := (not_le_of_gt hpos) hle0R
      cases this

/--
`roundDyadicPosDown` (directed rounding toward `-∞` for *positive* dyadics) never produces a
NaN/Inf value.

We package this as an existence statement: the output always has a dyadic decode.
This is the bridge we use later to turn `toEReal` into `toReal` (as an `EReal`) without unfolding
all bit-level definitions at call sites.
-/
theorem toDyadic?_roundDyadicPosDown_some (mant : Nat) (exp : Int) :
    ∃ d, toDyadic? (roundDyadicPosDown mant exp) = some d := by
  classical
  set log2m : Nat := Nat.log2 mant with hlog2m
  set k : Int := (Int.ofNat log2m) + exp with hk
  by_cases hkHi : k > 127
  ·
    -- Overflow branch: `roundDyadicPosDown` saturates to the maximum finite float.
    refine ⟨{ sign := false, mant := pow2 23 + (pow2 23 - 1), exp := (Int.ofNat 254) - 150 }, ?_⟩
    have hOut : roundDyadicPosDown mant exp = posMaxFinite := by
      have hkHiExp : (127 : Int) < ((log2m : Int) + exp) := by
        have : (127 : Int) < k := hkHi
        simpa [hk] using this
      simp (config := { zeta := true }) [roundDyadicPosDown, hkHiExp, hlog2m.symm]
    have hexp : (254 : Nat) < 255 := by decide
    have hfrac : (pow2 23 - 1) < 2 ^ 23 := by
      have h : (pow2 23 - 1) < pow2 23 := Nat.sub_lt (pow2_pos 23) (by decide)
      -- Rewrite the goal's RHS to `pow2 23`, then discharge it using `Nat.sub_lt`.
      -- (We avoid `simp` on the whole inequality here, since it can reduce it to a decidable
      -- proposition.)
      rw [← pow2_eq_two_pow 23]
      exact h
    have hdy :
        toDyadic? (posMaxFinite : IEEE32Exec) =
          some { sign := false, mant := pow2 23 + (pow2 23 - 1), exp := (Int.ofNat 254) - 150 } :=
            by
      simpa [posMaxFinite] using
        (toDyadic?_ofBits_mkBits_fin (sign := false) (exp := 254) (frac := pow2 23 - 1) hexp hfrac)
    simpa [hOut] using hdy
  ·
    have hkHi' : ¬ k > 127 := hkHi
    by_cases hkUnder : k < -149
    ·
      -- Underflow: exact value is < 2^-149, so the floor is `+0`.
      refine ⟨{ sign := false, mant := 0, exp := 0 }, ?_⟩
      have hOut : roundDyadicPosDown mant exp = posZero := by
        have hkHiExp : ¬ (127 : Int) < ((log2m : Int) + exp) := by
          simpa [hk] using hkHi'
        have hkUnderExp : ((log2m : Int) + exp) < -149 := by
          simpa [hk] using hkUnder
        simp (config := { zeta := true }) [roundDyadicPosDown, hkHiExp, hkUnderExp, hlog2m.symm]
      have hexp : (0 : Nat) < 255 := by decide
      have hfrac : (0 : Nat) < 2 ^ 23 := by decide
      have hdy : toDyadic? (posZero : IEEE32Exec) = some { sign := false, mant := 0, exp := 0 } :=
        by
        simpa [posZero] using
          (toDyadic?_ofBits_mkBits_fin (sign := false) (exp := 0) (frac := 0) hexp hfrac)
      simpa [hOut] using hdy
    ·
      have hkUnder' : ¬ k < -149 := hkUnder
      by_cases hkSub : k < -126
      ·
        -- Subnormal branch: exponent field is `0`, so it's always finite.
        set fracNat : Nat :=
            match exp + 149 with
            | .ofNat sh => Nat.shiftLeft mant sh
            | .negSucc sh => Nat.shiftRight mant (sh + 1) with hfracNat
        set frac : Nat := fracNat % pow2 23 with hfrac
        have hfrac_lt : frac < 2 ^ 23 := by
          have : frac < pow2 23 := Nat.mod_lt _ (pow2_pos 23)
          simpa [hfrac, pow2_eq_two_pow] using this
        have hexp0 : (0 : Nat) < 255 := by decide
        -- The output is either `posZero` or `ofBits (mkBits false 0 frac)`. Both have a dyadic
        -- decode.
        cases hZero : (fracNat == 0) with
        | true =>
            -- In this sub-branch `roundDyadicPosDown` returns `posZero`.
            refine ⟨{ sign := false, mant := 0, exp := 0 }, ?_⟩
            have hOut : roundDyadicPosDown mant exp = posZero := by
              have hkHiExp : ¬ (127 : Int) < ((log2m : Int) + exp) := by
                simpa [hk] using hkHi'
              have hkUnderExp : ¬ ((log2m : Int) + exp) < -149 := by
                simpa [hk] using hkUnder'
              have hkSubExp : ((log2m : Int) + exp) < -126 := by
                simpa [hk] using hkSub
              have hZeroEq :
                  (match exp + 149 with
                    | Int.ofNat sh => mant <<< sh
                    | Int.negSucc sh => mant >>> (sh + 1)) =
                    0 := by
                have : fracNat = 0 := (beq_iff_eq).1 hZero
                simpa [hfracNat] using this
              -- Reduce to the final subnormal `if` and discharge the remaining obligation by
              -- contradiction.
              simp (config := { zeta := true }) [roundDyadicPosDown, hkHiExp, hkUnderExp, hkSubExp,
                hlog2m.symm, posZero]
              intro hne
              exact False.elim (hne hZeroEq)
            have hdy : toDyadic? (posZero : IEEE32Exec) = some { sign := false, mant := 0, exp := 0
              } := by
              have hfrac0 : (0 : Nat) < 2 ^ 23 := by decide
              simpa [posZero] using
                (toDyadic?_ofBits_mkBits_fin (sign := false) (exp := 0) (frac := 0) hexp0 hfrac0)
            simpa [hOut] using hdy
        | false =>
            -- Otherwise `roundDyadicPosDown` returns the subnormal float `mkBits false 0 frac`.
            by_cases hf0 : frac = 0
            ·
              -- Even if `fracNat ≠ 0`, masking can still yield `frac = 0`; then the output is
              -- `posZero`.
              refine ⟨{ sign := false, mant := 0, exp := 0 }, ?_⟩
              have hOut : roundDyadicPosDown mant exp = posZero := by
                have hkHiExp : ¬ (127 : Int) < ((log2m : Int) + exp) := by
                  simpa [hk] using hkHi'
                have hkUnderExp : ¬ ((log2m : Int) + exp) < -149 := by
                  simpa [hk] using hkUnder'
                have hkSubExp : ((log2m : Int) + exp) < -126 := by
                  simpa [hk] using hkSub
                have hZeroNe :
                    (match exp + 149 with
                      | Int.ofNat sh => mant <<< sh
                      | Int.negSucc sh => mant >>> (sh + 1)) ≠
                      0 := by
                  have : fracNat ≠ 0 := (beq_eq_false_iff_ne (a := fracNat) (b := 0)).1 hZero
                  simpa [hfracNat] using this
                -- Reduce to an implication goal, then show the masked fraction is `0` using `hf0`.
                simp (config := { zeta := true }) [roundDyadicPosDown, hkHiExp, hkUnderExp,
                  hkSubExp, hlog2m.symm, posZero]
                intro _
                have hmod0 : fracNat % pow2 23 = 0 := by
                  simpa [hf0] using hfrac.symm
                -- Keep `mkBits` opaque and discharge the remaining bit-level equality by `decide`.
                have hmod0' :
                    (match exp + 149 with
                      | Int.ofNat sh => mant <<< sh
                      | Int.negSucc sh => mant >>> (sh + 1)) %
                        pow2 23 =
                      0 := by
                  simpa [hfracNat] using hmod0
                have hbits : mkBits false 0 0 = (0 : UInt32) := by decide
                have hmk :
                    mkBits false 0
                        ((match exp + 149 with
                          | Int.ofNat sh => mant <<< sh
                          | Int.negSucc sh => mant >>> (sh + 1)) %
                          pow2 23) =
                      mkBits false 0 0 := by
                  simpa using congrArg (fun n : Nat => mkBits false 0 n) hmod0'
                calc
                  ofBits
                        (mkBits false 0
                          ((match exp + 149 with
                            | Int.ofNat sh => mant <<< sh
                            | Int.negSucc sh => mant >>> (sh + 1)) %
                            pow2 23)) =
                    ofBits (mkBits false 0 0) := by
                          simpa using congrArg ofBits hmk
                    _ = ofBits 0 := by
                          simp [hbits]
              have hdy : toDyadic? (posZero : IEEE32Exec) = some { sign := false, mant := 0, exp :=
                0 } := by
                have hfrac0 : (0 : Nat) < 2 ^ 23 := by decide
                simpa [posZero] using
                  (toDyadic?_ofBits_mkBits_fin (sign := false) (exp := 0) (frac := 0) hexp0 hfrac0)
              simpa [hOut] using hdy
            ·
              refine ⟨{ sign := false, mant := frac, exp := -149 }, ?_⟩
              have hOut : roundDyadicPosDown mant exp = ofBits (mkBits false 0 frac) := by
                have hkHiExp : ¬ (127 : Int) < ((log2m : Int) + exp) := by
                  simpa [hk] using hkHi'
                have hkUnderExp : ¬ ((log2m : Int) + exp) < -149 := by
                  simpa [hk] using hkUnder'
                have hkSubExp : ((log2m : Int) + exp) < -126 := by
                  simpa [hk] using hkSub
                have hZeroNe :
                    (match exp + 149 with
                      | Int.ofNat sh => mant <<< sh
                      | Int.negSucc sh => mant >>> (sh + 1)) ≠
                      0 := by
                  have : fracNat ≠ 0 := (beq_eq_false_iff_ne (a := fracNat) (b := 0)).1 hZero
                  simpa [hfracNat] using this
                -- Reduce to the final subnormal `if` and select the `else` branch using `hZeroNe`.
                simp (config := { zeta := true })
                  [roundDyadicPosDown, hkHiExp, hkUnderExp, hkSubExp, hlog2m.symm]
                have hmodEq' :
                    (match exp + 149 with
                      | Int.ofNat sh => mant <<< sh
                      | Int.negSucc sh => mant >>> (sh + 1)) %
                        pow2 23 =
                      frac := by
                  simpa [hfracNat] using hfrac.symm
                have hmk :
                    mkBits false 0
                        ((match exp + 149 with
                          | Int.ofNat sh => mant <<< sh
                          | Int.negSucc sh => mant >>> (sh + 1)) %
                          pow2 23) =
                      mkBits false 0 frac := by
                  simpa using congrArg (fun n : Nat => mkBits false 0 n) hmodEq'
                -- Discharge the `if` by a case split on whether the (inlined) subnormal `fracNat`
                -- is `0`.
                by_cases hEq :
                    (match exp + 149 with
                      | Int.ofNat sh => mant <<< sh
                      | Int.negSucc sh => mant >>> (sh + 1)) =
                      0
                · exact False.elim (hZeroNe hEq)
                ·
                  -- Select the `else` branch and rewrite the masked fraction using `hmodEq'`.
                  calc
                    (if
                          (match exp + 149 with
                            | Int.ofNat sh => mant <<< sh
                            | Int.negSucc sh => mant >>> (sh + 1)) =
                            0 then
                        posZero
                      else
                        ofBits
                          (mkBits false 0
                            ((match exp + 149 with
                              | Int.ofNat sh => mant <<< sh
                              | Int.negSucc sh => mant >>> (sh + 1)) %
                              pow2 23))) =
                        ofBits
                          (mkBits false 0
                            ((match exp + 149 with
                              | Int.ofNat sh => mant <<< sh
                              | Int.negSucc sh => mant >>> (sh + 1)) %
                              pow2 23)) := by
                        simpa using
                          (if_neg (h := (inferInstance :
                            Decidable ((match exp + 149 with
                              | Int.ofNat sh => mant <<< sh
                              | Int.negSucc sh => mant >>> (sh + 1)) = 0))) hEq)
                    _ = ofBits (mkBits false 0 frac) := congrArg ofBits hmk
              have hdy :
                  toDyadic? (ofBits (mkBits false 0 frac)) =
                    some { sign := false, mant := frac, exp := -149 } := by
                have h := toDyadic?_ofBits_mkBits_fin (sign := false) (exp := 0) (frac := frac)
                  hexp0 hfrac_lt
                simpa [hf0] using h
              simpa [hOut] using hdy
      ·
        -- Normal branch: exponent field is `expNat = toNat (k+127)`, which is in `[1,254]`.
        set expNat : Nat := Int.toNat (k + 127) with hexpNat
        set m24 : Nat :=
            if log2m >= 23 then Nat.shiftRight mant (log2m - 23) else Nat.shiftLeft mant (23 -
              log2m) with hm24
        set fracNat : Nat := (m24 - pow2 23) % pow2 23 with hfracNat
        have hfrac_lt : fracNat < 2 ^ 23 := by
          have : fracNat < pow2 23 := Nat.mod_lt _ (pow2_pos 23)
          simpa [hfracNat, pow2_eq_two_pow] using this
        have hk_ge : (-126 : Int) ≤ k := le_of_not_gt hkSub
        have hk_le : k ≤ 127 := le_of_not_gt hkHi'
        have hk_pos : (0 : Int) < k + 127 := by
          linarith [hk_ge]
        have hk_nonneg : 0 ≤ k + 127 := le_of_lt hk_pos
        have hz : (expNat : Int) = k + 127 := by
          simpa [expNat] using (Int.toNat_of_nonneg hk_nonneg)
        have hexpNat_lt : expNat < 255 := by
          have hk_lt : k + 127 < 255 := by
            linarith [hk_le]
          have : (expNat : Int) < (255 : Int) := by simpa [hz] using hk_lt
          exact (Int.ofNat_lt).1 this
        have expNat_ne0 : expNat ≠ 0 := by
          have : (0 : Int) < (expNat : Int) := by simpa [hz] using hk_pos
          have : 0 < expNat := (Int.natCast_pos).1 this
          exact Nat.ne_of_gt this
        have hOut : roundDyadicPosDown mant exp = ofBits (mkBits false expNat fracNat) := by
          have hkHiExp : ¬ (127 : Int) < ((log2m : Int) + exp) := by
            simpa [hk] using hkHi'
          have hkUnderExp : ¬ ((log2m : Int) + exp) < -149 := by
            simpa [hk] using hkUnder'
          have hkSubExp : ¬ ((log2m : Int) + exp) < -126 := by
            simpa [hk] using hkSub
          -- Reduce the outer branches using the translated hypotheses, then expand local
          -- abbreviations on the RHS.
          simp (config := { zeta := true })
            [roundDyadicPosDown, hkHiExp, hkUnderExp, hkSubExp, hlog2m.symm, expNat, m24, fracNat,
              hk]
        refine ⟨{ sign := false, mant := pow2 23 + fracNat, exp := (Int.ofNat expNat) - 150 }, ?_⟩
        have hdy :
            toDyadic? (ofBits (mkBits false expNat fracNat)) =
              some { sign := false, mant := pow2 23 + fracNat, exp := (Int.ofNat expNat) - 150 } :=
                by
          have h := toDyadic?_ofBits_mkBits_fin (sign := false) (exp := expNat) (frac := fracNat)
            hexpNat_lt hfrac_lt
          simpa [expNat_ne0] using h
        simpa [hOut] using hdy

/--
Soundness of `roundDyadicDown`: it produces an `EReal` lower bound for the exact dyadic value.

Informal: rounding down is monotone towards `-∞`, so the rounded result is always `≤` the exact
real meaning.
-/
theorem toEReal_roundDyadicDown_le (d : Dyadic) :
    toEReal (roundDyadicDown d) ≤ (dyadicToReal d : EReal) := by
  classical
  by_cases h0 : d.mant = 0
  ·
    -- Signed zero: exact value is `0`, and `roundDyadicDown` returns ±0.
    have hrd : roundDyadicDown d = (if d.sign then negZero else posZero) := by
      simp [roundDyadicDown, h0]
    -- Both sides are exactly `0` in `EReal`.
    rw [hrd]
    cases hs : d.sign <;> simp [hs, dyadicToReal, h0, toEReal_posZero, toEReal_negZero]
  ·
    have hm : d.mant ≠ 0 := h0
    by_cases hs : d.sign = true
    ·
      -- Negative: rounding down makes it more negative, i.e. negate the positive rounding up.
      have hOut : roundDyadicDown d = neg (roundDyadicPosUp d.mant d.exp) := by
        simp [roundDyadicDown, hm, hs]
      have hpos : ((d.mant : ℝ) * bpow d.exp : EReal) ≤ toEReal (roundDyadicPosUp d.mant d.exp) :=
        toEReal_roundDyadicPosUp_ge (mant := d.mant) (exp := d.exp) hm
      -- Negate the inequality (order-reversing).
      have hneg : toEReal (neg (roundDyadicPosUp d.mant d.exp)) ≤ (-( (d.mant : ℝ) * bpow d.exp) :
        EReal) := by
        have hnan : isNaN (roundDyadicPosUp d.mant d.exp) = false :=
          isNaN_roundDyadicPosUp_eq_false (mant := d.mant) (exp := d.exp) hm
        -- `toEReal (neg x) = -toEReal x`.
        have htoNeg :
            toEReal (neg (roundDyadicPosUp d.mant d.exp)) =
              -toEReal (roundDyadicPosUp d.mant d.exp) :=
          toEReal_neg_of_isNaN_eq_false (x := roundDyadicPosUp d.mant d.exp) hnan
        -- Negation on `EReal` is order-reversing: `b ≤ a` implies `-a ≤ -b`.
        have hneg' :
            (-toEReal (roundDyadicPosUp d.mant d.exp)) ≤ (-( (d.mant : ℝ) * bpow d.exp : EReal)) :=
          (EReal.neg_le_neg_iff).2 hpos
        simpa [htoNeg] using hneg'
      -- Rewrite the exact dyadic semantics for a negative sign.
      have hd : (dyadicToReal d : EReal) = (-( (d.mant : ℝ) * bpow d.exp) : EReal) := by
        simp [dyadicToReal, hs, bpow]
      simpa [hOut, hd] using hneg
    ·
      -- Positive: reduce to `toReal_roundDyadicPosDown_le` and cast to `EReal`.
      have hs' : d.sign = false := by
        cases h : d.sign with
        | false => rfl
        | true => cases (hs h)
      have hOut : roundDyadicDown d = roundDyadicPosDown d.mant d.exp := by
        simp [roundDyadicDown, hm, hs']
      have hleR : toReal (roundDyadicPosDown d.mant d.exp) ≤ (d.mant : ℝ) * bpow d.exp :=
        toReal_roundDyadicPosDown_le (mant := d.mant) (exp := d.exp) hm
      have hleE : (toReal (roundDyadicPosDown d.mant d.exp) : EReal) ≤ ((d.mant : ℝ) * bpow d.exp :
        EReal) := by
        exact_mod_cast hleR
      have hE : toEReal (roundDyadicPosDown d.mant d.exp) = (toReal (roundDyadicPosDown d.mant
        d.exp) : EReal) := by
        -- `roundDyadicPosDown` always decodes to a dyadic, hence is finite.
        have ⟨dd, hdd⟩ := toDyadic?_roundDyadicPosDown_some (mant := d.mant) (exp := d.exp)
        have hnan : isNaN (roundDyadicPosDown d.mant d.exp) = false :=
          isNaN_eq_false_of_toDyadic?_some (hx := hdd)
        have hinf : isInf (roundDyadicPosDown d.mant d.exp) = false :=
          isInf_eq_false_of_toDyadic?_some (hx := hdd)
        have hfin : isFinite (roundDyadicPosDown d.mant d.exp) = true :=
          isFinite_eq_true_of_isNaN_eq_false_of_isInf_eq_false (x := roundDyadicPosDown d.mant
            d.exp) hnan hinf
        have hE? :
            toEReal? (roundDyadicPosDown d.mant d.exp) =
              some (toReal (roundDyadicPosDown d.mant d.exp) : EReal) :=
          toEReal?_eq_some_toReal_of_isFinite_eq_true (x := roundDyadicPosDown d.mant d.exp) hfin
        exact toEReal_of_toEReal? hE?
      have hd : (dyadicToReal d : EReal) = ((d.mant : ℝ) * bpow d.exp : EReal) := by
        simp [dyadicToReal, hs']
      simpa [hOut, hE, hd] using hleE

/--
Soundness of `roundDyadicUp`: the result is ≥ the exact dyadic real value (in `EReal`).

This is the key lemma needed to justify `addUp/mulUp` as interval *upper endpoints*.
-/
theorem toEReal_roundDyadicUp_ge (d : Dyadic) :
    (dyadicToReal d : EReal) ≤ toEReal (roundDyadicUp d) := by
  classical
  by_cases h0 : d.mant = 0
  ·
    have hru : roundDyadicUp d = (if d.sign then negZero else posZero) := by
      simp [roundDyadicUp, h0]
    -- Both `dyadicToReal d` and `toEReal` of a signed zero are exactly `0`.
    rw [hru]
    cases hs : d.sign <;> simp [dyadicToReal, h0, hs, toEReal_posZero, toEReal_negZero]
  ·
    have hm : d.mant ≠ 0 := h0
    by_cases hs : d.sign = true
    ·
      -- Negative: rounding up makes it less negative, i.e. negate the positive rounding down.
      have hOut : roundDyadicUp d = neg (roundDyadicPosDown d.mant d.exp) := by
        simp [roundDyadicUp, hm, hs]
      have hpos : toReal (roundDyadicPosDown d.mant d.exp) ≤ (d.mant : ℝ) * bpow d.exp :=
        toReal_roundDyadicPosDown_le (mant := d.mant) (exp := d.exp) hm
      have hposE : (toReal (roundDyadicPosDown d.mant d.exp) : EReal) ≤ ((d.mant : ℝ) * bpow d.exp :
        EReal) :=
        by exact_mod_cast hpos
      have hnan : isNaN (roundDyadicPosDown d.mant d.exp) = false := by
        have ⟨dd, hdd⟩ := toDyadic?_roundDyadicPosDown_some (mant := d.mant) (exp := d.exp)
        exact isNaN_eq_false_of_toDyadic?_some (hx := hdd)
      have htoNeg :
          toEReal (neg (roundDyadicPosDown d.mant d.exp)) =
            -toEReal (roundDyadicPosDown d.mant d.exp) :=
        toEReal_neg_of_isNaN_eq_false (x := roundDyadicPosDown d.mant d.exp) hnan
      have hE : toEReal (roundDyadicPosDown d.mant d.exp) = (toReal (roundDyadicPosDown d.mant
        d.exp) : EReal) := by
        have ⟨dd, hdd⟩ := toDyadic?_roundDyadicPosDown_some (mant := d.mant) (exp := d.exp)
        have hinf : isInf (roundDyadicPosDown d.mant d.exp) = false :=
          isInf_eq_false_of_toDyadic?_some (hx := hdd)
        have hfin : isFinite (roundDyadicPosDown d.mant d.exp) = true :=
          isFinite_eq_true_of_isNaN_eq_false_of_isInf_eq_false (x := roundDyadicPosDown d.mant
            d.exp) hnan hinf
        have hE? :
            toEReal? (roundDyadicPosDown d.mant d.exp) =
              some (toReal (roundDyadicPosDown d.mant d.exp) : EReal) :=
          toEReal?_eq_some_toReal_of_isFinite_eq_true (x := roundDyadicPosDown d.mant d.exp) hfin
        exact toEReal_of_toEReal? hE?
      -- Negate the inequality.
      have : (-( (d.mant : ℝ) * bpow d.exp) : EReal) ≤ toEReal (neg (roundDyadicPosDown d.mant
        d.exp)) := by
        -- `toReal ≤ exact` implies `-exact ≤ -toReal`.
        have hneg : (-( (d.mant : ℝ) * bpow d.exp) : EReal) ≤ (-(toReal (roundDyadicPosDown d.mant
          d.exp) : EReal)) :=
          (EReal.neg_le_neg_iff).2 hposE
        -- Rewrite the RHS as `toEReal (neg ...)`.
        simpa [htoNeg, hE] using hneg
      have hd : (dyadicToReal d : EReal) = (-( (d.mant : ℝ) * bpow d.exp) : EReal) := by
        simp [dyadicToReal, hs, bpow]
      simpa [hOut, hd] using this
    ·
      have hs' : d.sign = false := by
        cases h : d.sign with
        | false => rfl
        | true => cases (hs h)
      have hOut : roundDyadicUp d = roundDyadicPosUp d.mant d.exp := by
        simp [roundDyadicUp, hm, hs']
      have hge : ((d.mant : ℝ) * bpow d.exp : EReal) ≤ toEReal (roundDyadicPosUp d.mant d.exp) :=
        toEReal_roundDyadicPosUp_ge (mant := d.mant) (exp := d.exp) hm
      have hd : (dyadicToReal d : EReal) = ((d.mant : ℝ) * bpow d.exp : EReal) := by
        simp [dyadicToReal, hs']
      simpa [hOut, hd] using hge

/--
Lower-endpoint soundness for `addDown` on finite inputs:
the result is ≤ the exact real sum (in `EReal`), even in overflow-to-`-∞` scenarios.
-/
theorem toEReal_addDown_le (x y : IEEE32Exec) (hx : isFinite x = true) (hy : isFinite y = true) :
    toEReal (addDown x y) ≤ ((toReal x + toReal y : ℝ) : EReal) := by
  classical
  -- Decode both operands to dyadics.
  cases hdx : toDyadic? x with
  | none =>
      have hfinFalse : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      have : False := by
        have h := hfinFalse
        simp [hx] at h
      exact False.elim this
  | some dx =>
      cases hdy : toDyadic? y with
      | none =>
          have hfinFalse : isFinite y = false := isFinite_eq_false_of_toDyadic?_eq_none (x := y) hdy
          have : False := by
            have h := hfinFalse
            simp [hy] at h
          exact False.elim this
      | some dy =>
          have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hdx)
          have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx := hdy)
          have hchoose : chooseNaN2 x y = none := chooseNaN2_none_of_not_isNaN x y hxNaN hyNaN
          have hxInf : isInf x = false := isInf_eq_false_of_toDyadic?_some (hx := hdx)
          have hyInf : isInf y = false := isInf_eq_false_of_toDyadic?_some (hx := hdy)
          have hadd :
              addDown x y = roundDyadicDown (addDyadic dx dy) := by
            simp [addDown, hchoose, hxInf, hyInf, hdx, hdy]
          have hsum :
              (dyadicToReal (addDyadic dx dy) : EReal) = ((toReal x + toReal y : ℝ) : EReal) := by
            -- exactness of dyadic addition + decode of operands
            simp [dyadicToReal_addDyadic_exact (a := dx) (b := dy), toReal_eq, hdx, hdy]
          have hle :
              toEReal (roundDyadicDown (addDyadic dx dy)) ≤ (dyadicToReal (addDyadic dx dy) : EReal)
                :=
            toEReal_roundDyadicDown_le (d := addDyadic dx dy)
          simpa [hadd, hsum] using hle

/--
Upper-endpoint soundness for `addUp` on finite inputs:
the exact real sum is ≤ the result (in `EReal`), with overflow rounding to `+∞`.
-/
theorem toEReal_addUp_ge (x y : IEEE32Exec) (hx : isFinite x = true) (hy : isFinite y = true) :
    ((toReal x + toReal y : ℝ) : EReal) ≤ toEReal (addUp x y) := by
  classical
  cases hdx : toDyadic? x with
  | none =>
      have hfinFalse : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      have : False := by
        have h := hfinFalse
        simp [hx] at h
      exact False.elim this
  | some dx =>
      cases hdy : toDyadic? y with
      | none =>
          have hfinFalse : isFinite y = false := isFinite_eq_false_of_toDyadic?_eq_none (x := y) hdy
          have : False := by
            have h := hfinFalse
            simp [hy] at h
          exact False.elim this
      | some dy =>
          have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hdx)
          have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx := hdy)
          have hchoose : chooseNaN2 x y = none := chooseNaN2_none_of_not_isNaN x y hxNaN hyNaN
          have hxInf : isInf x = false := isInf_eq_false_of_toDyadic?_some (hx := hdx)
          have hyInf : isInf y = false := isInf_eq_false_of_toDyadic?_some (hx := hdy)
          have hadd :
              addUp x y = roundDyadicUp (addDyadic dx dy) := by
            simp [addUp, hchoose, hxInf, hyInf, hdx, hdy]
          have hsum :
              (dyadicToReal (addDyadic dx dy) : EReal) = ((toReal x + toReal y : ℝ) : EReal) := by
            simp [dyadicToReal_addDyadic_exact (a := dx) (b := dy), toReal_eq, hdx, hdy]
          have hge :
              (dyadicToReal (addDyadic dx dy) : EReal) ≤ toEReal (roundDyadicUp (addDyadic dx dy))
                :=
            toEReal_roundDyadicUp_ge (d := addDyadic dx dy)
          simpa [hadd, hsum] using hge

/-- Lower-endpoint soundness for `mulDown` on finite inputs. -/
theorem toEReal_mulDown_le (x y : IEEE32Exec) (hx : isFinite x = true) (hy : isFinite y = true) :
    toEReal (mulDown x y) ≤ ((toReal x * toReal y : ℝ) : EReal) := by
  classical
  cases hdx : toDyadic? x with
  | none =>
      have hfinFalse : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      have : False := by
        have h := hfinFalse
        simp [hx] at h
      exact False.elim this
  | some dx =>
      cases hdy : toDyadic? y with
      | none =>
          have hfinFalse : isFinite y = false := isFinite_eq_false_of_toDyadic?_eq_none (x := y) hdy
          have : False := by
            have h := hfinFalse
            simp [hy] at h
          exact False.elim this
      | some dy =>
          have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hdx)
          have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx := hdy)
          have hchoose : chooseNaN2 x y = none := chooseNaN2_none_of_not_isNaN x y hxNaN hyNaN
          have hxInf : isInf x = false := isInf_eq_false_of_toDyadic?_some (hx := hdx)
          have hyInf : isInf y = false := isInf_eq_false_of_toDyadic?_some (hx := hdy)
          have hxR : (toReal x : ℝ) = dyadicToReal dx := by simp [toReal_eq, hdx]
          have hyR : (toReal y : ℝ) = dyadicToReal dy := by simp [toReal_eq, hdy]
          set s : Bool := Bool.xor dx.sign dy.sign
          set prod : Dyadic := { sign := s, mant := dx.mant * dy.mant, exp := dx.exp + dy.exp }
          have hmul :
              mulDown x y =
                if dx.mant == 0 || dy.mant == 0 then
                  if s then negZero else posZero
                else
                  roundDyadicDown prod := by
            simp [mulDown, hchoose, hxInf, hyInf, hdx, hdy, s, prod]
          cases hz : (dx.mant == 0 || dy.mant == 0) with
          | true =>
              -- At least one factor is zero, so both the exact product and the endpoint are zero.
              have hz' : dx.mant = 0 ∨ dy.mant = 0 := by
                have hzProp : (dx.mant == 0) = true ∨ (dy.mant == 0) = true :=
                  (Eq.mp (Bool.or_eq_true (dx.mant == 0) (dy.mant == 0)) hz)
                cases hzProp with
                | inl hx0 => exact Or.inl ((beq_iff_eq).1 hx0)
                | inr hy0 => exact Or.inr ((beq_iff_eq).1 hy0)
              have hprod0 : (toReal x * toReal y : ℝ) = 0 := by
                cases hz' with
                | inl hx0 =>
                    have hx0R : (toReal x : ℝ) = 0 := by
                      -- Keep `toReal x` opaque and rewrite via the extracted dyadic.
                      rw [hxR]
                      simp [dyadicToReal, hx0]
                    rw [hx0R]
                    simp
                | inr hy0 =>
                    have hy0R : (toReal y : ℝ) = 0 := by
                      rw [hyR]
                      simp [dyadicToReal, hy0]
                    rw [hy0R]
                    simp
              have hout0 : toEReal (mulDown x y) = (0 : EReal) := by
                -- `mulDown` returns a signed zero in this branch.
                by_cases hs : s <;> simp [hmul, hz, hs, toEReal_posZero, toEReal_negZero]
              -- Both the returned endpoint and the exact product are `0`.
              -- We rewrite explicitly (instead of `simp`) to avoid unfolding `toReal` into matches.
              rw [hout0]
              have hprod0E : ((toReal x * toReal y : ℝ) : EReal) = (0 : EReal) := by
                simpa using congrArg (fun r : ℝ => (r : EReal)) hprod0
              -- Rewrite the exact product to `0`.
              rw [hprod0E]
          | false =>
              -- Nonzero mantissas: reduce to `roundDyadicDown` applied to the exact dyadic product.
              have hmul' : mulDown x y = roundDyadicDown prod := by
                simp [hmul, hz]
              have hle : toEReal (roundDyadicDown prod) ≤ (dyadicToReal prod : EReal) :=
                toEReal_roundDyadicDown_le (d := prod)
              have hprod :
                  (dyadicToReal prod : EReal) = ((toReal x * toReal y : ℝ) : EReal) := by
                have hR : dyadicToReal prod = dyadicToReal dx * dyadicToReal dy := by
                  simpa [prod, s] using (dyadicToReal_mul_exact (a := dx) (b := dy))
                -- Work in `ℝ`, then cast the equality into `EReal`.
                have hReal : dyadicToReal prod = (toReal x : ℝ) * (toReal y : ℝ) := by
                  calc
                    dyadicToReal prod = dyadicToReal dx * dyadicToReal dy := hR
                    _ = (toReal x : ℝ) * (toReal y : ℝ) := by
                      -- Reduce `toReal` by rewriting `toDyadic?` via the pattern matches `hdx/hdy`.
                      simp [toReal_eq, hdx, hdy]
                simpa using congrArg (fun r : ℝ => (r : EReal)) hReal
              simpa [hmul', hprod] using hle

/-- Upper-endpoint soundness for `mulUp` on finite inputs. -/
theorem toEReal_mulUp_ge (x y : IEEE32Exec) (hx : isFinite x = true) (hy : isFinite y = true) :
    ((toReal x * toReal y : ℝ) : EReal) ≤ toEReal (mulUp x y) := by
  classical
  cases hdx : toDyadic? x with
  | none =>
      have hfinFalse : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      have : False := by
        have h := hfinFalse
        simp [hx] at h
      exact False.elim this
  | some dx =>
      cases hdy : toDyadic? y with
      | none =>
          have hfinFalse : isFinite y = false := isFinite_eq_false_of_toDyadic?_eq_none (x := y) hdy
          have : False := by
            have h := hfinFalse
            simp [hy] at h
          exact False.elim this
      | some dy =>
          have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hdx)
          have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx := hdy)
          have hchoose : chooseNaN2 x y = none := chooseNaN2_none_of_not_isNaN x y hxNaN hyNaN
          have hxInf : isInf x = false := isInf_eq_false_of_toDyadic?_some (hx := hdx)
          have hyInf : isInf y = false := isInf_eq_false_of_toDyadic?_some (hx := hdy)
          have hxR : (toReal x : ℝ) = dyadicToReal dx := by simp [toReal_eq, hdx]
          have hyR : (toReal y : ℝ) = dyadicToReal dy := by simp [toReal_eq, hdy]
          set s : Bool := Bool.xor dx.sign dy.sign
          set prod : Dyadic := { sign := s, mant := dx.mant * dy.mant, exp := dx.exp + dy.exp }
          have hmul :
              mulUp x y =
                if dx.mant == 0 || dy.mant == 0 then
                  if s then negZero else posZero
                else
                  roundDyadicUp prod := by
            simp [mulUp, hchoose, hxInf, hyInf, hdx, hdy, s, prod]
          cases hz : (dx.mant == 0 || dy.mant == 0) with
          | true =>
              -- At least one factor is zero, so both the exact product and the endpoint are zero.
              have hz' : dx.mant = 0 ∨ dy.mant = 0 := by
                have hzProp : (dx.mant == 0) = true ∨ (dy.mant == 0) = true :=
                  (Eq.mp (Bool.or_eq_true (dx.mant == 0) (dy.mant == 0)) hz)
                cases hzProp with
                | inl hx0 => exact Or.inl ((beq_iff_eq).1 hx0)
                | inr hy0 => exact Or.inr ((beq_iff_eq).1 hy0)
              have hprod0 : (toReal x * toReal y : ℝ) = 0 := by
                cases hz' with
                | inl hx0 =>
                    have hx0R : (toReal x : ℝ) = 0 := by
                      rw [hxR]
                      simp [dyadicToReal, hx0]
                    rw [hx0R]
                    simp
                | inr hy0 =>
                    have hy0R : (toReal y : ℝ) = 0 := by
                      rw [hyR]
                      simp [dyadicToReal, hy0]
                    rw [hy0R]
                    simp
              have hout0 : toEReal (mulUp x y) = (0 : EReal) := by
                by_cases hs : s <;>
                  simp [hmul, hz, hs, toEReal_posZero, toEReal_negZero]
              -- `0 ≤ 0`.
              rw [hout0]
              rw [hprod0]
              simp
          | false =>
              have hmul' : mulUp x y = roundDyadicUp prod := by
                simp [hmul, hz]
              have hge : (dyadicToReal prod : EReal) ≤ toEReal (roundDyadicUp prod) :=
                toEReal_roundDyadicUp_ge (d := prod)
              have hprod :
                  (dyadicToReal prod : EReal) = ((toReal x * toReal y : ℝ) : EReal) := by
                have hR : dyadicToReal prod = dyadicToReal dx * dyadicToReal dy := by
                  simpa [prod, s] using (dyadicToReal_mul_exact (a := dx) (b := dy))
                have hReal : dyadicToReal prod = (toReal x : ℝ) * (toReal y : ℝ) := by
                  calc
                    dyadicToReal prod = dyadicToReal dx * dyadicToReal dy := hR
                    _ = (toReal x : ℝ) * (toReal y : ℝ) := by
                          have hxR' : dyadicToReal dx = (toReal x : ℝ) := by simpa using hxR.symm
                          have hyR' : dyadicToReal dy = (toReal y : ℝ) := by simpa using hyR.symm
                          simp [hxR', hyR']
                simpa using congrArg (fun r : ℝ => (r : EReal)) hReal
              -- Rewrite the exact product into the dyadic product and apply the rounding theorem.
              have : ((toReal x * toReal y : ℝ) : EReal) ≤ toEReal (roundDyadicUp prod) := by
                simpa [hprod] using hge
              rw [hmul']
              exact this

end

end IEEE32Exec

end TorchLean.Floats.IEEE754
