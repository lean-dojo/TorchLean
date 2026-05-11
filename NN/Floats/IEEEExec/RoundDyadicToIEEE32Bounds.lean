/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.DirectedRoundingSoundness
public import NN.Floats.IEEEExec.Exec32
public import NN.Floats.IEEEExec.MkBitsToReal
public import NN.Floats.IEEEExec.NatLemmas
public import NN.Floats.IEEEExec.RoundShiftRightEven

public import Mathlib.Data.EReal.Basic
public import Mathlib.Data.Nat.Log
public import Mathlib.Data.Real.Basic

/-!
# `roundDyadicToIEEE32` is sandwiched by directed rounding (the “full IEEE” bridge)

Our codebase has **two** floating-point execution models for binary32:

1. `roundDyadicToIEEE32` (in `NN/Floats/IEEEExec/Exec32.lean`)
   - models IEEE-754 **round-to-nearest, ties-to-even** when rounding an *exact* dyadic
     `(-1)^sign * mant * 2^exp` back to float32.
2. `roundDyadicDown` / `roundDyadicUp`
   - implement **directed rounding** toward `-∞` / `+∞`.
   - these are the primitives used by interval arithmetic endpoints (`addDown/addUp`, etc.).

To justify “IBP/CROWN bounds are sound w.r.t. full IEEE execution”, we need a bridge that relates
these two rounding policies.

The key fact is:

> For every exact dyadic `d`, nearest-even rounding lies between the directed endpoints:
>
> `roundDyadicDown d ≤ roundDyadicToIEEE32 d ≤ roundDyadicUp d`
>
> (with the order taken in `EReal` via `toEReal` so that `±∞` behavior remains well-defined).

Intuition:
- Directed rounding computes the **lower / upper neighbor** of the exact real value on the binary32
  grid (“floor” / “ceil” on the float lattice).
- Nearest-even rounding must choose **one of these two neighbors**.

So the proof splits into:
1. a *nonnegative* lemma showing the nearest-even implementation returns either the directed `down`
   result or the directed `up` result, and
2. a sign-flip lemma for `toEReal` allowing us to lift the result to signed dyadics.

We keep the argument executable-model-specific but mathematically elementary:
the only “real math” we use is that `Nat.log2` controls the bit-length, and that shifting right is
division by a power of two.
-/

@[expose] public section


namespace TorchLean.Floats.IEEE754
namespace IEEE32Exec

open TorchLean.Floats

noncomputable section

/-! ## Small Nat helpers -/

private lemma shiftRight_eq_div_pow (n k : Nat) : Nat.shiftRight n k = n / 2 ^ k := by
  simp [Nat.shiftRight_eq_div_pow]

/--
`shiftRightCeilPow2 n shift` is always either `n >>> shift` or `(n >>> shift) + 1`.

This is the “two-point” characterization of the ceil-quotient; it complements
`roundShiftRightEven_eq_shiftRight_or_shiftRight_add1` for nearest-even rounding.
-/
private lemma shiftRightCeilPow2_eq_shiftRight_or_shiftRight_add1 (n shift : Nat) :
    shiftRightCeilPow2 n shift = Nat.shiftRight n shift ∨
      shiftRightCeilPow2 n shift = Nat.shiftRight n shift + 1 := by
  classical
  by_cases hs : shift = 0
  · subst hs
    simp [shiftRightCeilPow2]
  ·
    have hk0 : (shift == 0) = false := (beq_eq_false_iff_ne).2 hs
    -- `shift > 0` branch: unfold and split on the remainder test.
    set q : Nat := Nat.shiftRight n shift
    set rem : Nat := n - Nat.shiftLeft q shift
    have hdef : shiftRightCeilPow2 n shift = (if rem == 0 then q else q + 1) := by
      simp (config := { zeta := true }) [shiftRightCeilPow2, hk0, q, rem]
    by_cases hrem : rem == 0
    · left; simp [hdef, hrem, q]
    · right; simp [hdef, hrem, q]

private lemma shiftRightCeilPow2_le_shiftRight_add1 (n shift : Nat) :
    shiftRightCeilPow2 n shift ≤ Nat.shiftRight n shift + 1 := by
  have hor := shiftRightCeilPow2_eq_shiftRight_or_shiftRight_add1 (n := n) (shift := shift)
  rcases hor with hq | hq <;> simp [hq]

/-! ### Working around `Nat.decLe` (Lean 4.26)

`Nat.decLe` is implemented as a nested dependent `dite` over `Nat.ble`, and rewriting under the
scrutinee of a `match` on `Decidable` is very brittle in Lean 4.26.

To keep our main proofs readable, we use a small lemma that reduces the *specific* `decLe`-match
used in the binary32 subnormal “round up to the smallest normal” check.
-/

private lemma match_pow2_23_decLe_isFalse (x : Nat) (h : ¬ pow2 23 ≤ x) :
    (match (pow2 23).decLe x with
    | isTrue _ => ofBits (mkBits false 1 0)
    | isFalse _ => ofBits (mkBits false 0 x)) =
      ofBits (mkBits false 0 x) := by
  -- Unfold `Nat.decLe` and discharge the `Nat.ble`-equality branch using `h`.
  have hcond : ¬ Eq (Nat.ble (pow2 23) x) true := by
    intro hEq
    have hxle : pow2 23 ≤ x :=
      (eq_iff_iff.mp (Nat.ble_eq (x := pow2 23) (y := x))).1 hEq
    exact h hxle
  simp [Nat.decLe, hcond]

private lemma match_pow2_23_decLe_isFalse_const1 (x : Nat) (h : ¬ pow2 23 ≤ x) :
    (match (pow2 23).decLe x with
    | isTrue _ => ofBits (mkBits false 1 0)
    | isFalse _ => ofBits (mkBits false 0 1)) =
      ofBits (mkBits false 0 1) := by
  -- Same as `match_pow2_23_decLe_isFalse`, but the `isFalse` branch is a constant (`1`).
  have hcond : ¬ Eq (Nat.ble (pow2 23) x) true := by
    intro hEq
    have hxle : pow2 23 ≤ x :=
      (eq_iff_iff.mp (Nat.ble_eq (x := pow2 23) (y := x))).1 hEq
    exact h hxle
  simp [Nat.decLe, hcond]

/-!
## Bit-length bounds used to rule out impossible carries

In the normal regime, `roundDyadicToIEEE32` computes a 24-bit mantissa `m24`.

If `m24 = 2^24` we carry into the exponent (this is the IEEE rule when rounding pushes us across a
power-of-two boundary).

However, the **floor quotient** produced by `Nat.shiftRight` can never be `2^24` in that regime: it
always fits strictly below `2^24`.

We prove the corresponding statement as a reusable lemma.
-/

private lemma shiftRight_lt_pow2_of_lt_pow (n k t : Nat) (hn : n < 2 ^ (k + t)) :
    Nat.shiftRight n k < 2 ^ t := by
  -- `n >>> k = n / 2^k`, and `2^(k+t) = 2^k * 2^t`.
  have hn' : n < (2 ^ k) * (2 ^ t) := by
    simpa [Nat.pow_add, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm] using hn
  have hdiv : n / 2 ^ k < 2 ^ t := Nat.div_lt_of_lt_mul hn'
  simpa [Nat.shiftRight_eq_div_pow] using hdiv

private lemma shiftRight_log2_sub_lt_pow2_24 (mant : Nat) (_hm : mant ≠ 0) (hlog : 23 ≤ mant.log2) :
    Nat.shiftRight mant (mant.log2 - 23) < pow2 24 := by
  -- Use the generic `lt_pow_succ_log_self` bound with base 2.
  have hlt : mant < 2 ^ (Nat.log 2 mant).succ := Nat.lt_pow_succ_log_self Nat.one_lt_two mant
  have hlt' : mant < 2 ^ (mant.log2 + 1) := by
    -- Rewrite `Nat.log 2 mant` as `mant.log2`.
    simpa [Nat.log2_eq_log_two, Nat.succ_eq_add_one, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc]
      using hlt
  -- `mant.log2 ≥ 23` implies `mant.log2 - 23 + 24 = mant.log2 + 1`.
  have hpow : mant < 2 ^ ((mant.log2 - 23) + 24) := by
    -- `mant.log2 - 23 + 24 = mant.log2 + 1` since `23 ≤ mant.log2`.
    have hcancel : mant.log2 - 23 + 23 = mant.log2 := Nat.sub_add_cancel hlog
    have : (mant.log2 - 23) + 24 = mant.log2 + 1 := by
      -- rewrite `24` as `23+1` and reassociate.
      have h24 : (24 : Nat) = 23 + 1 := by decide
      calc
        (mant.log2 - 23) + 24 = (mant.log2 - 23) + (23 + 1) := by simp [h24]
        _ = ((mant.log2 - 23) + 23) + 1 := by simp [Nat.add_assoc]
        _ = mant.log2 + 1 := by simp [hcancel]
    simpa [this] using hlt'
  have hq : Nat.shiftRight mant (mant.log2 - 23) < 2 ^ 24 :=
    shiftRight_lt_pow2_of_lt_pow (n := mant) (k := mant.log2 - 23) (t := 24) hpow
  simpa [pow2_eq_two_pow] using hq

private lemma shiftLeft_lt_pow2_24_of_log2_lt (mant : Nat) (_hm : mant ≠ 0) (hlog : mant.log2 < 23)
  :
    Nat.shiftLeft mant (23 - mant.log2) < pow2 24 := by
  -- As above, use `mant < 2^(mant.log2+1)` and shift left by `23 - log2`.
  have hlt : mant < 2 ^ (Nat.log 2 mant).succ := Nat.lt_pow_succ_log_self Nat.one_lt_two mant
  have hlt' : mant < 2 ^ (mant.log2 + 1) := by
    simpa [Nat.log2_eq_log_two, Nat.succ_eq_add_one, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc]
      using hlt
  -- `mant <<< (23 - log2) = mant * 2^(23 - log2)`, so it is `< 2^(log2+1) * 2^(23-log2) = 2^24`.
  -- Rewrite `shiftLeft` and compare using the `mant < 2^(log2m+1)` bound.
  have hpos : 0 < 2 ^ (23 - mant.log2) :=
    Nat.pow_pos (n := (23 - mant.log2)) (by decide : 0 < (2 : Nat))
  have hltmul : mant * 2 ^ (23 - mant.log2) < (2 ^ (mant.log2 + 1)) * 2 ^ (23 - mant.log2) :=
    Nat.mul_lt_mul_of_pos_right hlt' hpos
  have hsimp :
      (2 ^ (mant.log2 + 1)) * 2 ^ (23 - mant.log2) = 2 ^ 24 := by
    -- `2^(a) * 2^(b) = 2^(a+b)` and `a+b = 24` since `mant.log2 < 23`.
    have hle22 : mant.log2 ≤ 22 := Nat.le_of_lt_succ hlog
    have hle : mant.log2 ≤ 23 := le_trans hle22 (by decide : (22 : Nat) ≤ 23)
    have hadd : (mant.log2 + 1) + (23 - mant.log2) = 24 := by
      have hsum : mant.log2 + (23 - mant.log2) = 23 := Nat.add_sub_of_le hle
      calc
        (mant.log2 + 1) + (23 - mant.log2) = (mant.log2 + (23 - mant.log2)) + 1 := by
          simp [Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]
        _ = 23 + 1 := by simp [hsum]
        _ = 24 := by decide
    -- Rewrite the product into `2^(a+b)`.
    have hprod :
        (2 ^ (mant.log2 + 1)) * 2 ^ (23 - mant.log2) =
          2 ^ ((mant.log2 + 1) + (23 - mant.log2)) := by
      -- `pow_add` gives `2^(a+b) = 2^a * 2^b`.
      simp [Nat.pow_add, Nat.mul_assoc, Nat.mul_comm]
    simpa [hadd] using hprod
  have : Nat.shiftLeft mant (23 - mant.log2) < 2 ^ 24 := by
    -- `mant <<< a = mant * 2^a`.
    have : mant * 2 ^ (23 - mant.log2) < 2 ^ 24 := by
      simpa [hsimp] using hltmul
    simpa [Nat.shiftLeft_eq] using this
  simpa [pow2_eq_two_pow] using this

/-!
## Nonnegative case: nearest-even picks `down` or `up`

For `sign = false`, `roundDyadicDown` and `roundDyadicUp` reduce to the positive kernels
`roundDyadicPosDown` / `roundDyadicPosUp`.

In each magnitude regime, the only difference between the three rounders is the choice of Nat
quotient when shifting right:

- floor quotient: `Nat.shiftRight`
- nearest-even:  `roundShiftRightEven`
- ceil quotient: `shiftRightCeilPow2`

Since each of these returns either `q` or `q+1`, nearest-even must coincide with one of the
directed candidates.
-/

private theorem roundDyadicToIEEE32_eq_roundDyadicDown_or_roundDyadicUp_pos (mant : Nat) (exp : Int)
  :
    let d : Dyadic := { sign := false, mant := mant, exp := exp }
    roundDyadicToIEEE32 d = roundDyadicDown d ∨ roundDyadicToIEEE32 d = roundDyadicUp d := by
  classical
  intro d
  by_cases hm : mant = 0
  · subst hm
    simp [d, roundDyadicToIEEE32, roundDyadicDown, roundDyadicUp]
  ·
    have hmbeq : (mant == 0) = false := (beq_eq_false_iff_ne).2 hm
    let log2m : Nat := Nat.log2 mant
    let k : Int := (Int.ofNat log2m) + exp

    -- We run the same case split as the executable code, but we keep `k` explicit.
    by_cases hkHi : k > 127
    · -- overflow: nearest and `up` both return `+∞`.
      right
      have hkHi' : (Int.ofNat log2m) + exp > 127 := by simpa [k] using hkHi
      -- In this branch the *first* magnitude guard is true, so both functions return `+∞`.
      have hkHiLt : (127 : Int) < (Int.ofNat log2m) + exp := by
        simpa [gt_iff_lt] using hkHi'
      have hkHiLt' : (127 : Int) < (Int.ofNat mant.log2) + exp := by
        simpa [log2m] using hkHiLt
      have hnear : roundDyadicToIEEE32 d = posInf := by
        -- `simp` reduces `roundDyadicToIEEE32` to an implication (`k ≤ 127 → ...`) in the overflow
        -- case.
        -- Discharge it by contradiction using `hkHiLt' : 127 < k`.
        simp (config := { zeta := true }) [d, roundDyadicToIEEE32, hmbeq]
        intro hkLe
        exact False.elim ((not_le_of_gt hkHiLt') hkLe)
      have hup : roundDyadicUp d = posInf := by
        simp (config := { zeta := true }) [d, roundDyadicUp, roundDyadicPosUp, hmbeq]
        intro hkLe
        exact False.elim ((not_le_of_gt hkHiLt') hkLe)
      simp [hnear, hup]
    ·
      have hkHi' : ¬ k > 127 := hkHi
      by_cases hkUnder : k < -150
      · -- Underflow-to-zero: nearest and `down` are both `+0`.
        left
        have hkHiExp : ¬ ((Int.ofNat log2m) + exp > 127) := by simpa [k] using hkHi'
        have hkHiGuardFalse : ¬ (127 : Int) < (Int.ofNat mant.log2) + exp := by
          have : ¬ (127 : Int) < (Int.ofNat log2m) + exp := by
            simpa [gt_iff_lt] using hkHiExp
          simpa [log2m] using this
        have hkUnderExp : (Int.ofNat log2m) + exp < -150 := by simpa [k] using hkUnder
        have hkUnderExp' : (Int.ofNat mant.log2) + exp < -150 := by
          simpa [log2m] using hkUnderExp
        have hkHiGuardFalse' : ¬ (127 : Int) < ((mant.log2 : Nat) : Int) + exp := by
          simpa using hkHiGuardFalse
        have hkUnderExp'' : ((mant.log2 : Nat) : Int) + exp < -150 := by
          simpa using hkUnderExp'
        have hnear : roundDyadicToIEEE32 d = posZero := by
          simp (config := { zeta := true })
            [d, roundDyadicToIEEE32, hmbeq, hkHiGuardFalse', hkUnderExp'']
        -- `k < -150` implies `k < -149`, so directed-down also yields `+0`.
        have hkUnder149 : (Int.ofNat log2m) + exp < -149 := by
          have : k < -149 := lt_trans hkUnder (by decide : (-150 : Int) < -149)
          simpa [k] using this
        have hkUnder149' : (Int.ofNat mant.log2) + exp < -149 := by
          simpa [log2m] using hkUnder149
        have hkUnder149'' : ((mant.log2 : Nat) : Int) + exp < -149 := by
          simpa using hkUnder149'
        have hdown : roundDyadicDown d = posZero := by
          simp (config := { zeta := true })
            [d, roundDyadicDown, roundDyadicPosDown, hmbeq, hkHiGuardFalse', hkUnder149'']
        simp [hnear, hdown]
      ·
        have hkUnder' : ¬ k < -150 := hkUnder
        by_cases hkSub : k < -126
        · -- Subnormal regime.
          have hkHiExp : ¬ ((Int.ofNat log2m) + exp > 127) := by simpa [k] using hkHi'
          have hkUnderExp : ¬ ((Int.ofNat log2m) + exp < -150) := by simpa [k] using hkUnder'
          have hkSubExp : (Int.ofNat log2m) + exp < -126 := by simpa [k] using hkSub

          cases hsh : (exp + 149) with
          | ofNat sh =>
              -- Left-shift case: all three compute the same `fracNat = mant <<< sh`.
              -- In the subnormal regime this lies below `2^23`, so no masking/carry happens.
              have hfrac_ne0 : (Nat.shiftLeft mant sh) ≠ 0 := by
                -- `mant ≠ 0` and `2^sh ≠ 0` imply the product is nonzero.
                have hpowne : (2 ^ sh : Nat) ≠ 0 := pow_ne_zero sh (by decide : (2 : Nat) ≠ 0)
                have hmulne : mant * 2 ^ sh ≠ 0 := Nat.mul_ne_zero hm hpowne
                simpa [Nat.shiftLeft_eq] using hmulne
              have hfrac_lt : Nat.shiftLeft mant sh < pow2 23 := by
                -- From `k < -126` we get `log2m + sh < 23`, hence `mant <<< sh < 2^23`.
                -- Convert `k < -126` into `log2m + exp + 149 < 23`.
                have hk23 : k + 149 < 23 := by
                    -- `k < -126` ⇒ `k + 149 < 23`.
                    have := add_lt_add_right hkSub 149
                    -- `-126 + 149 = 23`.
                    -- Reassociate/commute the LHS into the `log2m + (exp+149)` shape.
                    simpa [k, Int.add_assoc, Int.add_left_comm, Int.add_comm] using this
                have hk23' : (Int.ofNat log2m) + (exp + 149) < 23 := by
                  -- reassociate: `(a + exp) + 149 = a + (exp + 149)`.
                  simpa [k, Int.add_assoc, Int.add_left_comm, Int.add_comm] using hk23
                -- `exp+149 = sh` in this branch.
                have hlog : (Int.ofNat log2m) + (Int.ofNat sh) < 23 := by
                  simpa [hsh] using hk23'
                -- convert to Nat inequality `log2m + sh < 23`.
                have hlogNat : log2m + sh < 23 := by
                  -- `Int.ofNat` is order embedding on naturals.
                  exact (Int.ofNat_lt.1 (by
                    simpa [Int.natCast_add] using hlog))
                -- Now use bit-length bound: `mant <<< sh < 2^(log2m+1+sh) ≤ 2^23`.
                -- A simple way is: `mant <<< sh ≤ 2^(log2(mant <<< sh)+1)` but we avoid that.
                -- Instead, use `mant < 2^(log2m+1)` and multiply by `2^sh`.
                have hlt : mant < 2 ^ (Nat.log 2 mant).succ :=
                  Nat.lt_pow_succ_log_self Nat.one_lt_two mant
                have hlt' : mant < 2 ^ (log2m + 1) := by
                  simpa [Nat.log2_eq_log_two, Nat.succ_eq_add_one, log2m] using hlt
                have hpos : 0 < 2 ^ sh := Nat.pow_pos (n := sh) (by decide : 0 < (2 : Nat))
                have hmul : mant * 2 ^ sh < (2 ^ (log2m + 1)) * 2 ^ sh :=
                  Nat.mul_lt_mul_of_pos_right hlt' hpos
                have hsimp : (2 ^ (log2m + 1)) * 2 ^ sh = 2 ^ ((log2m + 1) + sh) := by
                  simp [Nat.pow_add, Nat.mul_comm, Nat.mul_left_comm]
                have hpowle : 2 ^ ((log2m + 1) + sh) ≤ 2 ^ 23 := by
                  have h1 : (log2m + sh) + 1 ≤ 23 := Nat.succ_le_of_lt hlogNat
                  have : (log2m + 1) + sh ≤ 23 := by
                    -- reassociate: `(log2m+1)+sh = (log2m+sh)+1`.
                    simpa [Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using h1
                  exact Nat.pow_le_pow_right (by decide : 1 ≤ (2 : Nat)) this
                -- Combine everything and rewrite `shiftLeft`.
                have hmul' : mant * 2 ^ sh < 2 ^ ((log2m + 1) + sh) := by
                  simpa [hsimp] using hmul
                have hmul'' : mant * 2 ^ sh < 2 ^ 23 := lt_of_lt_of_le hmul' hpowle
                have : Nat.shiftLeft mant sh < 2 ^ 23 := by
                  simpa [Nat.shiftLeft_eq] using hmul''
                simpa [pow2_eq_two_pow] using this

              -- With these bounds, all three rounders produce the same subnormal `mkBits false 0
              -- frac`.
              left
              -- Work with an explicit name for the subnormal fraction. We choose the `<<<` form
              -- to avoid definitional-equality issues between `Nat.shiftLeft` and `<<<`.
              set frac : Nat := mant <<< sh
              have hfrac_lt' : frac < pow2 23 := by
                -- `hfrac_lt` was proven for `Nat.shiftLeft`; rewrite it to the `<<<` form.
                simpa [frac, Nat.shiftLeft_eq] using hfrac_lt
              have hfrac_ne0' : frac ≠ 0 := by
                simpa [frac, Nat.shiftLeft_eq] using hfrac_ne0
              have hzFrac : (frac == 0) = false := (beq_eq_false_iff_ne).2 hfrac_ne0'
              have hdecFrac : ¬ pow2 23 ≤ frac := Nat.not_le_of_lt hfrac_lt'
              have hmodFrac : frac % pow2 23 = frac := Nat.mod_eq_of_lt hfrac_lt'

              -- The magnitude guards that force both rounders into the same subnormal branch.
              have hkHiGuardFalse : ¬ (127 : Int) < ((mant.log2 : Nat) : Int) + exp := by
                have : ¬ (127 : Int) < (Int.ofNat log2m) + exp := by
                  simpa [gt_iff_lt] using hkHiExp
                simpa [log2m] using this
              have hkUnderGuardFalse : ¬ ((mant.log2 : Nat) : Int) + exp < -150 := by
                have : ¬ (Int.ofNat log2m) + exp < -150 := hkUnderExp
                simpa [log2m] using this
              have hkSubGuardTrue : ((mant.log2 : Nat) : Int) + exp < -126 := by
                have : (Int.ofNat log2m) + exp < -126 := hkSubExp
                simpa [log2m] using this
              have hkUnder149False : ¬ ((mant.log2 : Nat) : Int) + exp < -149 := by
                -- `k < -149` is impossible in this left-shift subnormal branch:
                -- `(k + 149) = mant.log2 + sh` is a nonnegative integer.
                have hk149nonneg : (0 : Int) ≤ (((mant.log2 : Nat) : Int) + exp) + 149 := by
                  -- Rewrite `k+149` as `mant.log2 + (exp+149) = mant.log2 + sh`.
                  have hEq :
                      (((mant.log2 : Nat) : Int) + exp) + 149 =
                        ((mant.log2 : Nat) : Int) + (exp + 149) := by
                    simp [Int.add_assoc, Int.add_left_comm, Int.add_comm]
                  -- Now `exp+149 = sh`, and both summands are nonnegative integers.
                  have h0 : (0 : Int) ≤ ((mant.log2 : Nat) : Int) + (Int.ofNat sh) := by
                    exact add_nonneg (Int.natCast_nonneg _) (Int.natCast_nonneg _)
                  -- Put the rewrites together.
                  simpa [hEq, hsh] using h0
                have hkge : (-149 : Int) ≤ ((mant.log2 : Nat) : Int) + exp := by
                  -- Add 149 to both sides and use `hk149nonneg`.
                  have : (-149 : Int) + 149 ≤ (((mant.log2 : Nat) : Int) + exp) + 149 := by
                    simpa using hk149nonneg
                  exact (Int.add_le_add_iff_right 149).1 this
                exact not_lt_of_ge hkge

              have hnear' : roundDyadicToIEEE32 d = ofBits (mkBits false 0 frac) := by
                -- Unfold to the subnormal branch and split on the *actual* `decLe` in the code.
                -- (Split on `mant <<< sh` rather than `frac` to avoid `simp`-mismatch
                -- between `Nat.shiftLeft` and the `<<<` notation.)
                simp (config := { zeta := true })
                  [d, roundDyadicToIEEE32, hm, hmbeq, hkHiGuardFalse, hkUnderGuardFalse,
                    hkSubGuardTrue, hsh]
                -- At this point the only remaining conditional is the `decLe` check on the exact
                -- fraction.
                have hdec :
                    ¬ pow2 23 ≤
                        (match exp + 149 with
                        | Int.ofNat sh => mant <<< sh
                        | Int.negSucc sh => roundShiftRightEven mant (sh + 1)) := by
                  -- In this `ofNat` branch, the fraction is exactly `frac = mant <<< sh`.
                  simpa [hsh, frac] using hdecFrac
                -- Reduce the `decLe` match using the helper lemma above.
                have hx :
                    (match exp + 149 with
                    | Int.ofNat sh => mant <<< sh
                    | Int.negSucc sh => roundShiftRightEven mant (sh + 1)) = mant <<< sh := by
                  simp [hsh]
                have hEq :=
                  match_pow2_23_decLe_isFalse
                    (x :=
                      (match exp + 149 with
                      | Int.ofNat sh => mant <<< sh
                      | Int.negSucc sh => roundShiftRightEven mant (sh + 1)))
                    hdec
                -- Rewrite the `x` occurrences to `mant <<< sh` and fold back to `frac`.
                simpa [hx, frac]
                  using hEq

              have hdown' : roundDyadicDown d = ofBits (mkBits false 0 frac) := by
                -- Directed-down uses the same `fracNat = mant <<< sh`, then masks by `% 2^23`.
                -- Here the mask is a no-op because `frac < 2^23`.
                have hmod' : (mant <<< sh) % pow2 23 = mant <<< sh := by
                  simpa [frac] using hmodFrac
                simp (config := { zeta := true })
                  [d, roundDyadicDown, roundDyadicPosDown, hmbeq, hkHiGuardFalse, hkUnder149False,
                    hkSubGuardTrue, hsh, hzFrac, frac, hmod']

              simp [hnear', hdown']
          | negSucc sh =>
              -- Right-shift case: nearest-even chooses `q` or `q+1`.
              set shift : Nat := sh + 1
              set q : Nat := Nat.shiftRight mant shift

              have hkHiExp : ¬ ((Int.ofNat log2m) + exp > 127) := by simpa [k] using hkHi'
              have hkUnderExp : ¬ ((Int.ofNat log2m) + exp < -150) := by simpa [k] using hkUnder'
              have hkSubExp : (Int.ofNat log2m) + exp < -126 := by simpa [k] using hkSub

              have hnear_cases :
                  roundShiftRightEven mant shift = q ∨ roundShiftRightEven mant shift = q + 1 := by
                simpa [q] using
                  (roundShiftRightEven_eq_shiftRight_or_shiftRight_add1 (n := mant) (shift :=
                    shift))

              -- A small bound: in the subnormal branch, the *exact* scaled fraction is `< 2^23`,
              -- hence the floor quotient `q` is `< 2^23`.
              have hq_lt : q < pow2 23 := by
                -- From `k < -126` deduce `log2m < sh + 24`.
                have hk23 : k + 149 < 23 := by
                    have := add_lt_add_right hkSub 149
                    -- Reassociate/commute the LHS into the `log2m + (exp+149)` shape.
                    simpa [k, Int.add_assoc, Int.add_left_comm, Int.add_comm] using this
                have hk23' : (Int.ofNat log2m) + (exp + 149) < 23 := by
                  simpa [k, Int.add_assoc, Int.add_left_comm, Int.add_comm] using hk23
                have hk23'' : (Int.ofNat log2m) + (Int.negSucc sh) < 23 := by
                  simpa [hsh] using hk23'
                -- `Int.negSucc sh = - (sh+1)` and rearrange.
                have hklog : (Int.ofNat log2m) < 23 + (Int.ofNat (sh + 1)) := by
                  -- Add `(sh+1)` to both sides.
                  have := add_lt_add_right hk23'' (Int.ofNat (sh + 1))
                  -- `a + negSucc sh + (sh+1) = a` (since `negSucc` is `-(sh+1)`).
                  simpa [Int.negSucc_eq, Int.natCast_add, Int.add_assoc, Int.add_left_comm,
                    Int.add_comm] using this
                have hkNat : log2m < sh + 24 := by
                  -- Rewrite the RHS as an `Int.ofNat` so we can use `Int.ofNat_lt`.
                  have hR : (23 : Int) + Int.ofNat (sh + 1) = Int.ofNat (sh + 24) := by
                    -- `23 = Int.ofNat 23`, and `23 + (sh+1) = sh+24` in `Nat`.
                    have hNat : (23 + (sh + 1) : Nat) = sh + 24 := by
                      simp [Nat.add_assoc, Nat.add_comm]
                    -- Push the `Nat` equality through `Int.ofNat` and expand the casted sum.
                    have hI : Int.ofNat (23 + (sh + 1)) = Int.ofNat (sh + 24) :=
                      congrArg Int.ofNat hNat
                    have : (Int.ofNat 23) + Int.ofNat (sh + 1) = Int.ofNat (sh + 24) := by
                      simpa [Int.natCast_add] using hI
                    simpa using this
                  have hk' : (Int.ofNat log2m) < Int.ofNat (sh + 24) :=
                    lt_of_lt_of_eq hklog (by simpa using hR)
                  exact Int.ofNat_lt.1 hk'
                have hle : log2m + 1 ≤ sh + 24 := Nat.succ_le_of_lt hkNat
                -- Now `mant < 2^(log2m+1) ≤ 2^(sh+24)`.
                have hlt : mant < 2 ^ (Nat.log 2 mant).succ :=
                  Nat.lt_pow_succ_log_self Nat.one_lt_two mant
                have hlt' : mant < 2 ^ (log2m + 1) := by
                  simpa [Nat.log2_eq_log_two, Nat.succ_eq_add_one, log2m] using hlt
                have hlt'' : mant < 2 ^ (sh + 24) := by
                  have hp : 2 ^ (log2m + 1) ≤ 2 ^ (sh + 24) :=
                    Nat.pow_le_pow_right (by decide : 1 ≤ (2 : Nat)) hle
                  exact lt_of_lt_of_le hlt' hp
                -- Apply the generic shift-right bound.
                have : Nat.shiftRight mant shift < 2 ^ 23 := by
                  -- `shift = sh+1`, and `sh+24 = shift+23`.
                  have : sh + 24 = shift + 23 := by simp [shift, Nat.add_assoc]
                  have hpow : mant < 2 ^ (shift + 23) := by simpa [this] using hlt''
                  exact shiftRight_lt_pow2_of_lt_pow (n := mant) (k := shift) (t := 23) hpow
                simpa [q, pow2_eq_two_pow] using this

              -- Case split on the nearest-even quotient.
              rcases hnear_cases with hqNear | hqNear
              · -- nearest chose `q` (the floor quotient), so it matches directed-down.
                left
                -- Directed-down has an additional `k < -149` underflow guard that nearest-even
                -- does not have. The only possible way to trigger it here is `k = -150` (since we
                -- already know `¬k < -150`).
                by_cases hkUnder149 : k < -149
                · have hkGe : (-150 : Int) ≤ k := by
                    exact le_of_not_gt (by simpa [gt_iff_lt] using hkUnderExp)
                  have hkLe : k ≤ (-150 : Int) := by
                    have : k < (-150 : Int) + 1 := by
                      simpa using hkUnder149
                    exact (Int.lt_add_one_iff).1 this
                  have hkEq : k = (-150 : Int) := le_antisymm hkLe hkGe
                  -- From `k = -150` and `exp+149 = -(sh+1)` we get `log2m = sh`, hence
                  -- `q = mant >>> (sh+1) = mant >>> (log2m+1) = 0`.
                  have hkEqExp : (Int.ofNat log2m) + exp = (-150 : Int) := by
                    simpa [k] using hkEq
                  have hk1 : (Int.ofNat log2m) + (exp + 149) = (-1 : Int) := by
                    have := congrArg (fun t => t + 149) hkEqExp
                    -- reassociate to `ofNat log2m + (exp+149)` and simplify the RHS.
                    simpa [Int.add_assoc, Int.add_left_comm, Int.add_comm] using this
                  have hk2 : (Int.ofNat log2m) + Int.negSucc sh = (-1 : Int) := by
                    simpa [hsh] using hk1
                  have hlog2m : log2m = sh := by
                    -- `ofNat log2m + (-(sh+1)) = -1` implies `ofNat log2m = ofNat sh`.
                    have hk3 : (Int.ofNat log2m) - Int.ofNat (sh + 1) = (-1 : Int) := by
                      simpa [Int.negSucc_eq, Int.sub_eq_add_neg, Int.add_assoc] using hk2
                    have hk4 := congrArg (fun t => t + Int.ofNat (sh + 1)) hk3
                    have hk5 : Int.ofNat log2m = Int.ofNat sh := by
                      -- simplify `a - b + b = a` and `-1 + (sh+1) = sh`.
                      simpa [Int.sub_eq_add_neg, Int.add_assoc, Int.add_left_comm, Int.add_comm]
                        using hk4
                    exact (Int.ofNat_inj.1 hk5)
                  have hq0 : q = 0 := by
                    -- `q = mant >>> (sh+1)` and `sh = log2m`, so we shift by `log2m+1`,
                    -- which is strictly larger than the bit-length.
                    have hlt : mant < 2 ^ (log2m + 1) := by
                      have hlt0 : mant < 2 ^ (Nat.log 2 mant).succ :=
                        Nat.lt_pow_succ_log_self Nat.one_lt_two mant
                      simpa [Nat.log2_eq_log_two, Nat.succ_eq_add_one, log2m] using hlt0
                    have hlt' : mant < 2 ^ shift := by
                      -- `shift = sh+1 = log2m+1`.
                      have : shift = log2m + 1 := by simp [shift, hlog2m]
                      simpa [this] using hlt
                    have hdiv : mant / 2 ^ shift = 0 := Nat.div_eq_of_lt hlt'
                    simpa [q, Nat.shiftRight_eq_div_pow] using hdiv
                  have hnear0 : roundDyadicToIEEE32 d = posZero := by
                    have hkHiGuardFalse : ¬ (127 : Int) < ((mant.log2 : Nat) : Int) + exp := by
                      have : ¬ (127 : Int) < (Int.ofNat log2m) + exp := by
                        simpa [gt_iff_lt] using hkHiExp
                      simpa [log2m] using this
                    have hkUnderGuardFalse : ¬ ((mant.log2 : Nat) : Int) + exp < -150 := by
                      simpa [log2m] using hkUnderExp
                    have hkSubGuardTrue : ((mant.log2 : Nat) : Int) + exp < -126 := by
                      simpa [log2m] using hkSubExp
                    have hshift0 : mant >>> (sh + 1) = 0 := by
                      -- `q = mant >>> (sh+1)` by definition.
                      simpa [q, shift] using hq0
                    simp (config := { zeta := true })
                      [d, roundDyadicToIEEE32, hmbeq,
                        hkHiGuardFalse, hkUnderGuardFalse, hkSubGuardTrue,
                        hsh, shift, q, hqNear, hshift0]
                  have hdown0 : roundDyadicDown d = posZero := by
                    have hkHiGuardFalse : ¬ (127 : Int) < ((mant.log2 : Nat) : Int) + exp := by
                      have : ¬ (127 : Int) < (Int.ofNat log2m) + exp := by
                        simpa [gt_iff_lt] using hkHiExp
                      simpa [log2m] using this
                    have hkUnder149Guard : ((mant.log2 : Nat) : Int) + exp < -149 := by
                      simpa [k, log2m] using hkUnder149
                    simp (config := { zeta := true })
                      [d, roundDyadicDown, roundDyadicPosDown, hmbeq, hkHiGuardFalse,
                        hkUnder149Guard]
                  simp [hnear0, hdown0]
                ·
                -- Proper subnormal regime (`¬k < -149`): both are computed from the floor quotient
                -- `q`.
                  by_cases hq0 : q = 0
                  · have hkHiGuardFalse : ¬ (127 : Int) < ((mant.log2 : Nat) : Int) + exp := by
                      have : ¬ (127 : Int) < (Int.ofNat log2m) + exp := by
                        simpa [gt_iff_lt] using hkHiExp
                      simpa [log2m] using this
                    have hkUnderGuardFalse : ¬ ((mant.log2 : Nat) : Int) + exp < -150 := by
                      simpa [log2m] using hkUnderExp
                    have hkSubGuardTrue : ((mant.log2 : Nat) : Int) + exp < -126 := by
                      simpa [log2m] using hkSubExp
                    have hshift0 : mant >>> (sh + 1) = 0 := by
                      simpa [q, shift] using hq0
                    have hnear0 : roundDyadicToIEEE32 d = posZero := by
                      simp (config := { zeta := true })
                        [d, roundDyadicToIEEE32, hmbeq,
                          hkHiGuardFalse, hkUnderGuardFalse, hkSubGuardTrue,
                          hsh, shift, q, hqNear, hshift0]
                    have hdown0 : roundDyadicDown d = posZero := by
                      simp (config := { zeta := true })
                        [d, roundDyadicDown, roundDyadicPosDown, hmbeq,
                          hkHiGuardFalse, hkSubGuardTrue, hsh, hshift0]
                    simp [hnear0, hdown0]
                  · have hz : (q == 0) = false := (beq_eq_false_iff_ne).2 hq0
                    have hkHiGuardFalse : ¬ (127 : Int) < ((mant.log2 : Nat) : Int) + exp := by
                      have : ¬ (127 : Int) < (Int.ofNat log2m) + exp := by
                        simpa [gt_iff_lt] using hkHiExp
                      simpa [log2m] using this
                    have hkUnderGuardFalse : ¬ ((mant.log2 : Nat) : Int) + exp < -150 := by
                      simpa [log2m] using hkUnderExp
                    have hkSubGuardTrue : ((mant.log2 : Nat) : Int) + exp < -126 := by
                      simpa [log2m] using hkSubExp
                    have hshift_ne0 : mant >>> (sh + 1) ≠ 0 := by
                      intro h0
                      apply hq0
                      simpa [q, shift] using h0
                    have hkUnder149GuardFalse : ¬ ((mant.log2 : Nat) : Int) + exp < -149 := by
                      simpa [k, log2m] using hkUnder149
                    have hdec : ¬ pow2 23 ≤ q := Nat.not_le_of_lt hq_lt
                    have hmod : q % pow2 23 = q := Nat.mod_eq_of_lt hq_lt

                    have hnear' : roundDyadicToIEEE32 d = ofBits (mkBits false 0 q) := by
                      -- In this branch `fracNat = roundShiftRightEven mant (sh+1) = q`,
                      -- and `q < 2^23`, so the `decLe` match takes the `isFalse` branch.
                      have hq' : roundShiftRightEven mant (sh + 1) = q := by
                        simpa [shift] using hqNear
                      simp (config := { zeta := true })
                        [d, roundDyadicToIEEE32, hmbeq,
                          hkHiGuardFalse, hkUnderGuardFalse, hkSubGuardTrue,
                          hsh, hq', hz]
                      -- Discharge the remaining `decLe` match by forcing the `isFalse` outcome.
                      have hx :
                          (match exp + 149 with
                          | Int.ofNat sh => mant <<< sh
                          | Int.negSucc sh => roundShiftRightEven mant (sh + 1)) = q := by
                        simpa [hsh, shift] using hqNear
                      have hdecx :
                          ¬ pow2 23 ≤
                              (match exp + 149 with
                              | Int.ofNat sh => mant <<< sh
                              | Int.negSucc sh => roundShiftRightEven mant (sh + 1)) := by
                        simpa [hx] using hdec
                      have hx' :
                          q =
                            (match exp + 149 with
                            | Int.ofNat sh => mant <<< sh
                            | Int.negSucc sh => roundShiftRightEven mant (sh + 1)) := hx.symm
                      have hEq :=
                        match_pow2_23_decLe_isFalse
                          (x :=
                            (match exp + 149 with
                            | Int.ofNat sh => mant <<< sh
                            | Int.negSucc sh => roundShiftRightEven mant (sh + 1)))
                          hdecx
                      -- The goal uses `q` in the payload position; rewrite it to `fracNat` using
                      -- `hx'`.
                      simpa [hx'] using hEq

                    have hdown' : roundDyadicDown d = ofBits (mkBits false 0 q) := by
                      -- Directed-down uses the exact floor quotient `q = mant >>> (sh+1)` and then
                      -- masks.
                      have hsr : mant >>> (sh + 1) = q := by
                        simp [q, shift]
                      have hzsr : (mant >>> (sh + 1) == 0) = false := by
                        simpa [hsr] using hz
                      -- Unfold to the subnormal directed-down branch; then use `hzsr` and `hmod`.
                      simp (config := { zeta := true })
                        [d, roundDyadicDown, roundDyadicPosDown, hmbeq,
                          hkHiGuardFalse, hkUnder149GuardFalse, hkSubGuardTrue,
                          hsh, hsr, hmod]
                      intro hqEq
                      -- If the quotient is 0, the masked fraction is 0, hence the result is `+0`.
                      simp [hqEq, posZero, mkBits]

                    simp [hnear', hdown']
              · -- nearest chose `q+1`, so it must match directed-up (ceil quotient).
                right
                -- First show `shiftRightCeilPow2 mant shift = q+1`.
                have hceil_ge : q + 1 ≤ shiftRightCeilPow2 mant shift := by
                  -- `roundShiftRightEven ≤ ceil`, and we are in the `q+1` branch.
                  have hle := roundShiftRightEven_le_shiftRightCeilPow2 (n := mant) (shift := shift)
                  -- Rewrite `roundShiftRightEven` and `q`.
                  simpa [q, hqNear] using hle
                have hceil_le : shiftRightCeilPow2 mant shift ≤ q + 1 := by
                  -- ceil is either `q` or `q+1`.
                  have := shiftRightCeilPow2_le_shiftRight_add1 (n := mant) (shift := shift)
                  simpa [q] using this
                have hceil : shiftRightCeilPow2 mant shift = q + 1 := le_antisymm hceil_le hceil_ge

                -- We also have the bound `q < 2^23`, so `q+1 ≤ 2^23`.
                have hq1_le : q + 1 ≤ pow2 23 := by
                  have : q + 1 ≤ 2 ^ 23 := Nat.succ_le_of_lt (by simpa [pow2_eq_two_pow] using
                    hq_lt)
                  simpa [pow2_eq_two_pow] using this

                -- As in the floor case, handle the `k < -149` guard in directed-up explicitly.
                by_cases hkUnder149 : k < -149
                ·
                -- Here (since `¬k < -150`) we are forced to `k = -150`, hence `q = 0` and nearest
                -- yields
                  -- the minimum subnormal (`frac = 1`), matching the `k < -149` branch of `up`.
                  have hkGe : (-150 : Int) ≤ k := by
                    exact le_of_not_gt (by simpa [gt_iff_lt] using hkUnderExp)
                  have hkLe : k ≤ (-150 : Int) := by
                    have : k < (-150 : Int) + 1 := by
                      simpa using hkUnder149
                    exact (Int.lt_add_one_iff).1 this
                  have hkEq : k = (-150 : Int) := le_antisymm hkLe hkGe
                  have hkEqExp : (Int.ofNat log2m) + exp = (-150 : Int) := by
                    simpa [k] using hkEq
                  have hk1 : (Int.ofNat log2m) + (exp + 149) = (-1 : Int) := by
                    have := congrArg (fun t => t + 149) hkEqExp
                    simpa [Int.add_assoc, Int.add_left_comm, Int.add_comm] using this
                  have hk2 : (Int.ofNat log2m) + Int.negSucc sh = (-1 : Int) := by
                    simpa [hsh] using hk1
                  have hlog2m : log2m = sh := by
                    have hk3 : (Int.ofNat log2m) - Int.ofNat (sh + 1) = (-1 : Int) := by
                      simpa [Int.negSucc_eq, Int.sub_eq_add_neg, Int.add_assoc] using hk2
                    have hk4 := congrArg (fun t => t + Int.ofNat (sh + 1)) hk3
                    have hk5 : Int.ofNat log2m = Int.ofNat sh := by
                      simpa [Int.sub_eq_add_neg, Int.add_assoc, Int.add_left_comm, Int.add_comm]
                        using hk4
                    exact (Int.ofNat_inj.1 hk5)
                  have hq0 : q = 0 := by
                    have hlt : mant < 2 ^ (log2m + 1) := by
                      have hlt0 : mant < 2 ^ (Nat.log 2 mant).succ :=
                        Nat.lt_pow_succ_log_self Nat.one_lt_two mant
                      simpa [Nat.log2_eq_log_two, Nat.succ_eq_add_one, log2m] using hlt0
                    have hlt' : mant < 2 ^ shift := by
                      have : shift = log2m + 1 := by simp [shift, hlog2m]
                      simpa [this] using hlt
                    have hdiv : mant / 2 ^ shift = 0 := Nat.div_eq_of_lt hlt'
                    simpa [q, Nat.shiftRight_eq_div_pow] using hdiv
                  have hqNear1 : roundShiftRightEven mant shift = 1 := by
                    -- `roundShiftRightEven = q+1` and `q = 0`.
                    simpa [hq0] using hqNear
                  have hz1 : (1 == 0) = false := by decide
                  have hdec1 : ¬ pow2 23 ≤ (1 : Nat) := by
                    -- `2^23` is huge; this is trivial.
                    have : (1 : Nat) < pow2 23 := by
                      -- `pow2 23 = 2^23` and `1 < 2^23`.
                      simp [pow2_eq_two_pow]
                    exact Nat.not_le_of_lt this
                  have hkHiGuardFalse : ¬ (127 : Int) < ((mant.log2 : Nat) : Int) + exp := by
                    have : ¬ (127 : Int) < (Int.ofNat log2m) + exp := by
                      simpa [gt_iff_lt] using hkHiExp
                    simpa [log2m] using this
                  have hkUnderGuardFalse : ¬ ((mant.log2 : Nat) : Int) + exp < -150 := by
                    simpa [log2m] using hkUnderExp
                  have hkSubGuardTrue : ((mant.log2 : Nat) : Int) + exp < -126 := by
                    simpa [log2m] using hkSubExp
                  have hxRS : roundShiftRightEven mant (sh + 1) = 1 := by
                    simpa [shift] using hqNear1
                  have hdecMatch1 :
                      (match (pow2 23).decLe (roundShiftRightEven mant (sh + 1)) with
                      | isTrue _ => ofBits (mkBits false 1 0)
                      | isFalse _ =>
                        ofBits (mkBits false 0 (roundShiftRightEven mant (sh + 1)))) =
                        ofBits (mkBits false 0 1) := by
                    have hdecRS : ¬ pow2 23 ≤ roundShiftRightEven mant (sh + 1) := by
                      simpa [hxRS] using hdec1
                    cases h :
                        (pow2 23).decLe (roundShiftRightEven mant (sh + 1)) with
                    | isTrue hle' =>
                        exact False.elim (hdecRS hle')
                    | isFalse _ =>
                        simp [hxRS]
                  have hnear1 : roundDyadicToIEEE32 d = ofBits (mkBits false 0 1) := by
                    -- Unfold to the subnormal branch, then reduce the remaining `decLe` match.
                    simp (config := { zeta := true })
                      [d, roundDyadicToIEEE32, hmbeq,
                        hkHiGuardFalse, hkUnderGuardFalse, hkSubGuardTrue,
                        hsh, hxRS]
                    -- Remaining goal: the `decLe` match must take the `isFalse` branch.
                    let fracNat : Nat :=
                      (match exp + 149 with
                      | Int.ofNat sh => mant <<< sh
                      | Int.negSucc sh => roundShiftRightEven mant (sh + 1))
                    have hfrac : fracNat = 1 := by
                      simp [fracNat, hsh, hxRS]
                    have hdecFrac : ¬ pow2 23 ≤ fracNat := by
                      simpa [hfrac] using hdec1
                    simpa [fracNat] using
                      (match_pow2_23_decLe_isFalse_const1 (x := fracNat) hdecFrac)
                  have hup1 : roundDyadicUp d = posMinSubnormal := by
                    have hkUnder149GuardTrue : ((mant.log2 : Nat) : Int) + exp < (-149 : Int) := by
                      -- In this branch we have `k = -150`, hence certainly `k < -149`.
                      have hkEqExp' : ((mant.log2 : Nat) : Int) + exp = (-150 : Int) := by
                        -- `log2m = mant.log2` by definition.
                        simpa [log2m] using hkEqExp
                      have hlt : (-150 : Int) < (-149 : Int) := by decide
                      exact lt_of_eq_of_lt hkEqExp' hlt
                    simp (config := { zeta := true })
                      [d, roundDyadicUp, roundDyadicPosUp, hmbeq,
                        hkHiGuardFalse, hkUnder149GuardTrue]
                  -- `posMinSubnormal` is exactly `ofBits (mkBits false 0 1)`.
                  have hnear1' : roundDyadicToIEEE32 d = posMinSubnormal := by
                    simpa [posMinSubnormal] using hnear1
                  simp [hnear1', hup1]
                ·
                -- Proper subnormal regime (`¬k < -149`): both are computed from the ceil quotient
                -- `q+1`.
                  -- Split depending on whether we hit the boundary `q+1 = 2^23` (round up to
                  -- smallest normal).
                  have hkHiGuardFalse : ¬ (127 : Int) < ((mant.log2 : Nat) : Int) + exp := by
                    have : ¬ (127 : Int) < (Int.ofNat log2m) + exp := by
                      -- `hkHiExp : ¬((ofNat log2m)+exp > 127)` is the same as `¬(127 < k)`.
                      simpa [gt_iff_lt] using hkHiExp
                    simpa [log2m] using this
                  have hkUnderGuardFalse : ¬ ((mant.log2 : Nat) : Int) + exp < (-150 : Int) := by
                    simpa [log2m] using hkUnderExp
                  have hkUnder149GuardFalse : ¬ ((mant.log2 : Nat) : Int) + exp < (-149 : Int) := by
                    -- This branch assumption is `¬(k < -149)` with `k = log2m + exp`.
                    simpa [k, log2m] using hkUnder149
                  have hkSubGuardTrue : ((mant.log2 : Nat) : Int) + exp < (-126 : Int) := by
                    simpa [log2m] using hkSubExp
                  by_cases hbound : pow2 23 ≤ q + 1
                  · -- Boundary: `q+1 = 2^23`, so both nearest and `up` return the smallest normal.
                    have hEq : q + 1 = pow2 23 := le_antisymm hq1_le hbound
                    have hdec : pow2 23 ≤ q + 1 := hbound
                    have hle : (pow2 23).decLe (q + 1) = isTrue hdec := Subsingleton.elim _ _
                    -- `simp` discharges the magnitude guards; we then reduce the remaining `decLe`
                    -- matches.
                    simp (config := { zeta := true })
                      [d, roundDyadicToIEEE32, roundDyadicUp, roundDyadicPosUp,
                        hmbeq, hkHiGuardFalse, hkUnderGuardFalse, hkUnder149GuardFalse,
                          hkSubGuardTrue, hsh]
                    have hqNear' : roundShiftRightEven mant (sh + 1) = q + 1 := by
                      simpa [shift] using hqNear
                    have hceil' : shiftRightCeilPow2 mant (sh + 1) = q + 1 := by
                      simpa [shift] using hceil
                    -- Force both `exp+149` matches into the `negSucc sh` branch, then rewrite to
                    -- `q+1`.
                    rw [hsh]
                    have hdecRS : pow2 23 ≤ roundShiftRightEven mant (sh + 1) := by
                      simpa [hqNear'] using hdec
                    have hdecCeil : pow2 23 ≤ shiftRightCeilPow2 mant (sh + 1) := by
                      simpa [hceil'] using hdec
                    have hleRS :
                        (pow2 23).decLe (roundShiftRightEven mant (sh + 1)) = isTrue hdecRS :=
                      Subsingleton.elim _ _
                    have hleCeil :
                        (pow2 23).decLe (shiftRightCeilPow2 mant (sh + 1)) = isTrue hdecCeil :=
                      Subsingleton.elim _ _
                    have hrs_ne0 : roundShiftRightEven mant (sh + 1) ≠ 0 := by
                      simp [hqNear']
                    have hceil_ne0 : shiftRightCeilPow2 mant (sh + 1) ≠ 0 := by
                      simp [hceil']
                    simp [hleRS, hleCeil, hrs_ne0, hceil_ne0]
                  ·
                  -- Proper subnormal: `q+1 < 2^23`, so both nearest and `up` return `mkBits false 0
                  -- (q+1)`.
                    have hq1_lt : q + 1 < pow2 23 := lt_of_le_of_ne hq1_le (Ne.symm (by
                      intro hEq
                      apply hbound
                      simp [hEq]))
                    have hdec : ¬ pow2 23 ≤ q + 1 := hbound
                    have hle : (pow2 23).decLe (q + 1) = isFalse hdec := Subsingleton.elim _ _
                    simp (config := { zeta := true })
                      [d, roundDyadicToIEEE32, roundDyadicUp, roundDyadicPosUp,
                        hmbeq, hkHiGuardFalse, hkUnderGuardFalse, hkUnder149GuardFalse,
                          hkSubGuardTrue, hsh]
                    have hqNear' : roundShiftRightEven mant (sh + 1) = q + 1 := by
                      simpa [shift] using hqNear
                    have hceil' : shiftRightCeilPow2 mant (sh + 1) = q + 1 := by
                      simpa [shift] using hceil
                    rw [hsh]
                    have hdecRS : ¬ pow2 23 ≤ roundShiftRightEven mant (sh + 1) := by
                      simpa [hqNear'] using hdec
                    have hdecCeil : ¬ pow2 23 ≤ shiftRightCeilPow2 mant (sh + 1) := by
                      simpa [hceil'] using hdec
                    have hleRS :
                        (pow2 23).decLe (roundShiftRightEven mant (sh + 1)) = isFalse hdecRS :=
                      Subsingleton.elim _ _
                    have hleCeil :
                        (pow2 23).decLe (shiftRightCeilPow2 mant (sh + 1)) = isFalse hdecCeil :=
                      Subsingleton.elim _ _
                    have hrs_ne0 : roundShiftRightEven mant (sh + 1) ≠ 0 := by
                      simp [hqNear']
                    have hceil_ne0 : shiftRightCeilPow2 mant (sh + 1) ≠ 0 := by
                      simp [hceil']
                    simp [hleRS, hleCeil, hqNear', hceil']
        ·
          -- Normal regime (`k ≥ -126`): same structure, but with the 24-bit mantissa `m24`.
          have hkHiExp : ¬ ((Int.ofNat log2m) + exp > 127) := by simpa [k] using hkHi'
          have hkUnderExp : ¬ ((Int.ofNat log2m) + exp < -150) := by
            -- `k < -150` is false and `k = (log2m+exp)`.
            simpa [k] using hkUnder'
          have hkSubExp : ¬ ((Int.ofNat log2m) + exp < -126) := by simpa [k] using hkSub
          have hkHiGuardFalse : ¬ (127 : Int) < ((mant.log2 : Nat) : Int) + exp := by
            have : ¬ (127 : Int) < (Int.ofNat log2m) + exp := by
              simpa [gt_iff_lt] using hkHiExp
            simpa [log2m] using this
          have hkUnderGuardFalse : ¬ ((mant.log2 : Nat) : Int) + exp < (-150 : Int) := by
            simpa [log2m] using hkUnderExp
          have hkSubGuardFalse : ¬ ((mant.log2 : Nat) : Int) + exp < (-126 : Int) := by
            simpa [log2m] using hkSubExp
          have hkUnder149GuardFalse : ¬ ((mant.log2 : Nat) : Int) + exp < (-149 : Int) := by
            intro hk
            have h149 : (-149 : Int) < (-126 : Int) := by decide
            exact hkSubGuardFalse (lt_trans hk h149)

          by_cases hlog : 23 ≤ log2m
          · -- shift-right case (mantissa shrink).
            have hlogM : 23 ≤ mant.log2 := by simpa [log2m] using hlog
            set shift : Nat := log2m - 23
            set q : Nat := Nat.shiftRight mant shift

            have hq_lt : q < pow2 24 :=
              shiftRight_log2_sub_lt_pow2_24 (mant := mant) (_hm := hm) (hlog := hlog)

            have hnear_cases :
                roundShiftRightEven mant shift = q ∨ roundShiftRightEven mant shift = q + 1 := by
              simpa [q] using
                (roundShiftRightEven_eq_shiftRight_or_shiftRight_add1 (n := mant) (shift := shift))

            rcases hnear_cases with hqNear | hqNear
            · -- nearest chose floor quotient `q`, so it matches directed-down.
              left
              have hneq : (q == pow2 24) = false := by
                exact (beq_eq_false_iff_ne).2 (ne_of_lt hq_lt)
              have hqNear' : roundShiftRightEven mant (log2m - 23) = q := by
                simpa [shift] using hqNear
              have hfrac_lt : q - pow2 23 < pow2 23 := by
                -- `q < 2^24` implies `q - 2^23 < 2^23` (with a case split for `q < 2^23`).
                have hp23 : 0 < pow2 23 := by
                  -- `pow2 23 = 2^23` and `2^23 > 0`.
                  simp [pow2_eq_two_pow]
                have hpow : pow2 24 = pow2 23 + pow2 23 := by
                  -- `2^24 = 2^23 * 2 = 2^23 + 2^23`.
                  simp [pow2_eq_two_pow, Nat.pow_succ, Nat.mul_two, Nat.mul_comm]
                by_cases hqsmall : q < pow2 23
                · have hsub : q - pow2 23 = 0 := Nat.sub_eq_zero_of_le (le_of_lt hqsmall)
                  simpa [hsub] using hp23
                · have hqge : pow2 23 ≤ q := le_of_not_gt hqsmall
                  have hlt : q < pow2 23 + pow2 23 := by
                    simpa [hpow] using hq_lt
                  exact Nat.sub_lt_left_of_lt_add hqge hlt
              have hmod : (q - pow2 23) % pow2 23 = q - pow2 23 := Nat.mod_eq_of_lt hfrac_lt
              have hsr : mant >>> (mant.log2 - 23) = q := by
                simp [q, shift, log2m]
              have hnear :
                  roundDyadicToIEEE32 d =
                    ofBits (mkBits false (Int.toNat (k + 127)) (q - pow2 23)) := by
                -- Discharge the magnitude guards and specialize the normal mantissa logic.
                simp (config := { zeta := true })
                  [d, roundDyadicToIEEE32, hmbeq, log2m, k,
                    hkHiGuardFalse, hkUnderGuardFalse, hkSubGuardFalse,
                    hlog, hqNear', hneq]
              have hdown :
                  roundDyadicDown d =
                    ofBits (mkBits false (Int.toNat (k + 127)) (q - pow2 23)) := by
                simp (config := { zeta := true })
                  [d, roundDyadicDown, roundDyadicPosDown, hmbeq, log2m, k,
                    hkHiGuardFalse, hkUnder149GuardFalse, hkSubGuardFalse,
                    hlog, hmod, hsr]
              simp [hnear, hdown]
            · -- nearest chose `q+1`, so it matches directed-up (ceil quotient).
              right
              -- First show `shiftRightCeilPow2 mant shift = q+1`.
              have hceil_ge : q + 1 ≤ shiftRightCeilPow2 mant shift := by
                have hle :=
                  roundShiftRightEven_le_shiftRightCeilPow2 (n := mant) (shift := shift)
                simpa [q, hqNear] using hle
              have hceil_le : shiftRightCeilPow2 mant shift ≤ q + 1 := by
                have := shiftRightCeilPow2_le_shiftRight_add1 (n := mant) (shift := shift)
                simpa [q] using this
              have hceil : shiftRightCeilPow2 mant shift = q + 1 := le_antisymm hceil_le hceil_ge
              simp (config := { zeta := true })
                [d, roundDyadicToIEEE32, roundDyadicUp, roundDyadicPosUp,
                  hmbeq, log2m, hkHiGuardFalse, hkUnderGuardFalse, hkSubGuardFalse,
                  hkUnder149GuardFalse, hlogM, shift, q, hqNear, hceil]
          ·
              -- shift-left case (log2m < 23): all three compute the same exact 24-bit mantissa.
              have hlog' : log2m < 23 := Nat.lt_of_not_ge hlog
              set m24 : Nat := Nat.shiftLeft mant (23 - log2m)
              have hm24_lt : m24 < pow2 24 :=
                shiftLeft_lt_pow2_24_of_log2_lt (mant := mant) (_hm := hm) (hlog := hlog')
              left
              have hneq : (m24 == pow2 24) = false := (beq_eq_false_iff_ne).2 (ne_of_lt hm24_lt)
              -- As in the shift-right case, keep the unfolding structured to avoid `simp` blowups.
              have hkHi' : ¬ k > 127 := by
                simpa [k] using hkHiExp
              have hkUnder' : ¬ k < (-150 : Int) := by
                simpa [k] using hkUnderExp
              have hkSub' : ¬ k < (-126 : Int) := by
                simpa [k] using hkSubExp
              have hkUnder149' : ¬ k < (-149 : Int) := by
                -- In the normal regime (`k ≥ -126`), we are certainly not below `-149`.
                intro hk
                have h149 : (-149 : Int) < (-126 : Int) := by decide
                exact hkSub' (lt_trans hk h149)
              have hfrac_lt : m24 - pow2 23 < pow2 23 := by
                -- `m24 < 2^24` implies `m24 - 2^23 < 2^23` (with a split on `m24 < 2^23`).
                have hp23 : 0 < pow2 23 := by
                  -- `pow2 23 = 2^23` and `2^23 > 0`.
                  simp [pow2_eq_two_pow]
                have hpow : pow2 24 = pow2 23 + pow2 23 := by
                  simp [pow2_eq_two_pow, Nat.pow_succ, Nat.mul_two, Nat.mul_comm]
                by_cases hsmall : m24 < pow2 23
                · have hsub : m24 - pow2 23 = 0 := Nat.sub_eq_zero_of_le (le_of_lt hsmall)
                  simpa [hsub] using hp23
                · have hge : pow2 23 ≤ m24 := le_of_not_gt hsmall
                  have hlt : m24 < pow2 23 + pow2 23 := by
                    simpa [hpow] using hm24_lt
                  exact Nat.sub_lt_left_of_lt_add hge hlt
              have hmod : (m24 - pow2 23) % pow2 23 = m24 - pow2 23 := Nat.mod_eq_of_lt hfrac_lt
              have hnear :
                  roundDyadicToIEEE32 d =
                    ofBits (mkBits false (Int.toNat (k + 127)) (m24 - pow2 23)) := by
                have hne24' : ¬ (mant <<< (23 - mant.log2)) = pow2 24 := by
                  -- `m24 < 2^24`, hence it cannot be exactly `2^24`.
                  have hne24 : m24 ≠ pow2 24 := ne_of_lt hm24_lt
                  simpa [m24, log2m] using hne24
                -- Discharge the magnitude guards and the `m24 = 2^24` overflow-carry check.
                have htmp :
                    roundDyadicToIEEE32 d =
                      ofBits (mkBits false ((↑mant.log2 + exp + 127).toNat) (m24 - pow2 23)) := by
                    simp (config := { zeta := true })
                      [d, roundDyadicToIEEE32, hmbeq, log2m,
                        hkHiGuardFalse, hkUnderGuardFalse, hkSubGuardFalse, hlog, hne24', m24]
                -- Rewrite the exponent field back to the local `k`.
                simpa [k, log2m, Int.toNat] using htmp
              have hdown :
                  roundDyadicDown d =
                    ofBits (mkBits false (Int.toNat (k + 127)) (m24 - pow2 23)) := by
                have hmod' :
                    ((mant <<< (23 - mant.log2) - pow2 23) % pow2 23) =
                      (mant <<< (23 - mant.log2) - pow2 23) := by
                  simpa [m24, log2m] using hmod
                have htmp :
                    roundDyadicDown d =
                      ofBits (mkBits false ((↑mant.log2 + exp + 127).toNat) (m24 - pow2 23)) := by
                  simp (config := { zeta := true })
                    [d, roundDyadicDown, roundDyadicPosDown, hmbeq, log2m,
                      hkHiGuardFalse, hkUnder149GuardFalse, hkSubGuardFalse, hlog, m24, hmod']
                simpa [k, log2m, Int.toNat] using htmp
              simp [hnear, hdown]

/-! ## Main theorem: nearest-even lies between directed endpoints (in `EReal`) -/

private lemma toEReal_roundDyadicDown_le_roundDyadicUp (d : Dyadic) :
    toEReal (roundDyadicDown d) ≤ toEReal (roundDyadicUp d) := by
  exact le_trans (toEReal_roundDyadicDown_le (d := d)) (toEReal_roundDyadicUp_ge (d := d))

@[simp] private lemma toEReal_posZero : toEReal (posZero : IEEE32Exec) = (0 : EReal) := by
  have hexp : (0 : Nat) < 255 := by decide
  have hfrac : (0 : Nat) < 2 ^ 23 := by decide
  have hdy :
      toDyadic? (posZero : IEEE32Exec) = some { sign := false, mant := 0, exp := (0 : Int) } := by
    -- `posZero = ofBits 0 = ofBits (mkBits false 0 0)`.
    simpa [posZero, mkBits] using
      (toDyadic?_ofBits_mkBits_fin (sign := false) (exp := 0) (frac := 0) hexp hfrac)
  have hnan : isNaN (posZero : IEEE32Exec) = false := by
    simpa [posZero, mkBits] using (isNaN_eq_false_of_toDyadic?_some (hx := hdy))
  have hinf : isInf (posZero : IEEE32Exec) = false := by
    simpa [posZero, mkBits] using (isInf_eq_false_of_toDyadic?_some (hx := hdy))
  have hfin : isFinite (posZero : IEEE32Exec) = true :=
    isFinite_eq_true_of_isNaN_eq_false_of_isInf_eq_false (x := posZero) hnan hinf
  have hE? : toEReal? (posZero : IEEE32Exec) = some (toReal (posZero : IEEE32Exec) : EReal) :=
    toEReal?_eq_some_toReal_of_isFinite_eq_true (x := posZero) hfin
  have hE : toEReal (posZero : IEEE32Exec) = (toReal (posZero : IEEE32Exec) : EReal) :=
    toEReal_of_toEReal? hE?
  -- `toReal posZero = 0`.
  have hto : toReal (posZero : IEEE32Exec) = 0 := by
    -- `toReal` follows the `toDyadic?` decode.
    simp [toReal_eq, hdy, dyadicToReal]
  simp [hE, hto]

@[simp] private lemma toEReal_negZero : toEReal (negZero : IEEE32Exec) = (0 : EReal) := by
  have hexp : (0 : Nat) < 255 := by decide
  have hfrac : (0 : Nat) < 2 ^ 23 := by decide
  have hdy :
      toDyadic? (negZero : IEEE32Exec) = some { sign := true, mant := 0, exp := (0 : Int) } := by
    -- `negZero = ofBits signMask = ofBits (mkBits true 0 0)`.
    simpa [negZero, signMask, mkBits] using
      (toDyadic?_ofBits_mkBits_fin (sign := true) (exp := 0) (frac := 0) hexp hfrac)
  have hnan : isNaN (negZero : IEEE32Exec) = false := by
    simpa [negZero, mkBits] using (isNaN_eq_false_of_toDyadic?_some (hx := hdy))
  have hinf : isInf (negZero : IEEE32Exec) = false := by
    simpa [negZero, mkBits] using (isInf_eq_false_of_toDyadic?_some (hx := hdy))
  have hfin : isFinite (negZero : IEEE32Exec) = true :=
    isFinite_eq_true_of_isNaN_eq_false_of_isInf_eq_false (x := negZero) hnan hinf
  have hE? : toEReal? (negZero : IEEE32Exec) = some (toReal (negZero : IEEE32Exec) : EReal) :=
    toEReal?_eq_some_toReal_of_isFinite_eq_true (x := negZero) hfin
  have hE : toEReal (negZero : IEEE32Exec) = (toReal (negZero : IEEE32Exec) : EReal) :=
    toEReal_of_toEReal? hE?
  have hto : toReal (negZero : IEEE32Exec) = 0 := by
    simp [toReal_eq, hdy, dyadicToReal]
  simp [hE, hto]

private lemma toEReal_ofBits_mkBits_fin_eq_toReal (sign : Bool) (exp frac : Nat)
    (hexp : exp < 255) (hfrac : frac < 2 ^ 23) :
    toEReal (ofBits (mkBits sign exp frac) : IEEE32Exec) =
      (toReal (ofBits (mkBits sign exp frac) : IEEE32Exec) : EReal) := by
  let x : IEEE32Exec := ofBits (mkBits sign exp frac)
  have hdy0 : toDyadic? x =
      (if exp = 0 then
        if frac = 0 then some { sign := sign, mant := 0, exp := 0 }
        else some { sign := sign, mant := frac, exp := (-149 : Int) }
      else some { sign := sign, mant := (pow2 23 + frac), exp := (Int.ofNat exp - 150) }) := by
    simpa [x] using (toDyadic?_ofBits_mkBits_fin (sign := sign) (exp := exp) (frac := frac) hexp
      hfrac)
  have hdySome : ∃ d : Dyadic, toDyadic? x = some d := by
    by_cases hexp0 : exp = 0
    · by_cases hfrac0 : frac = 0
      · refine ⟨{ sign := sign, mant := 0, exp := 0 }, ?_⟩
        simp [hdy0, hexp0, hfrac0]
      · refine ⟨{ sign := sign, mant := frac, exp := (-149 : Int) }, ?_⟩
        simp [hdy0, hexp0, hfrac0]
    · refine ⟨{ sign := sign, mant := (pow2 23 + frac), exp := (Int.ofNat exp - 150) }, ?_⟩
      simp [hdy0, hexp0]
  rcases hdySome with ⟨d, hdy⟩
  have hnan : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hdy)
  have hinf : isInf x = false := isInf_eq_false_of_toDyadic?_some (hx := hdy)
  have hfin : isFinite x = true :=
    isFinite_eq_true_of_isNaN_eq_false_of_isInf_eq_false (x := x) hnan hinf
  have hE? : toEReal? x = some (toReal x : EReal) :=
    toEReal?_eq_some_toReal_of_isFinite_eq_true (x := x) hfin
  simpa [toEReal, x] using (toEReal_of_toEReal? hE?)

private lemma toReal_ofBits_mkBits_fin_signFlip (exp frac : Nat) (hexp : exp < 255) (hfrac : frac <
  2 ^ 23) :
    toReal (ofBits (mkBits true exp frac) : IEEE32Exec) =
      -toReal (ofBits (mkBits false exp frac) : IEEE32Exec) := by
  have htTrue := toReal_ofBits_mkBits_fin (sign := true) (exp := exp) (frac := frac) hexp hfrac
  have htFalse := toReal_ofBits_mkBits_fin (sign := false) (exp := exp) (frac := frac) hexp hfrac
  rw [htTrue, htFalse]
  by_cases hexp0 : exp = 0 <;> by_cases hfrac0 : frac = 0 <;>
    simp [hexp0, hfrac0]

private lemma toEReal_ofBits_mkBits_fin_signFlip (exp frac : Nat) (hexp : exp < 255) (hfrac : frac <
  2 ^ 23) :
    toEReal (ofBits (mkBits true exp frac) : IEEE32Exec) =
      -toEReal (ofBits (mkBits false exp frac) : IEEE32Exec) := by
  have hETrue :=
    toEReal_ofBits_mkBits_fin_eq_toReal (sign := true) (exp := exp) (frac := frac) hexp hfrac
  have hEFalse :=
    toEReal_ofBits_mkBits_fin_eq_toReal (sign := false) (exp := exp) (frac := frac) hexp hfrac
  have htoReal :
      toReal (ofBits (mkBits true exp frac) : IEEE32Exec) =
        -toReal (ofBits (mkBits false exp frac) : IEEE32Exec) :=
    toReal_ofBits_mkBits_fin_signFlip (exp := exp) (frac := frac) hexp hfrac
  have htoRealE :
      (toReal (ofBits (mkBits true exp frac) : IEEE32Exec) : EReal) =
        ((-toReal (ofBits (mkBits false exp frac) : IEEE32Exec)) : EReal) :=
    congrArg (fun r : ℝ => (r : EReal)) htoReal
  -- Rewrite `toEReal` into `toReal`, then use the `toReal` sign-flip lemma.
  rw [hETrue, hEFalse]
  calc
    (toReal (ofBits (mkBits true exp frac) : IEEE32Exec) : EReal) =
        ((-toReal (ofBits (mkBits false exp frac) : IEEE32Exec)) : EReal) := htoRealE
    _ = -(toReal (ofBits (mkBits false exp frac) : IEEE32Exec) : EReal) := by simp

/--
Nearest-even rounding is sandwiched between directed roundings (in `EReal`).

Informal: rounding a dyadic to float32 using round-to-nearest-even yields a value that lies between
rounding down and rounding up.
-/
theorem toEReal_roundDyadicDown_le_roundDyadicToIEEE32_le_roundDyadicUp (d : Dyadic) :
    toEReal (roundDyadicDown d) ≤ toEReal (roundDyadicToIEEE32 d) ∧
      toEReal (roundDyadicToIEEE32 d) ≤ toEReal (roundDyadicUp d) := by
  classical
  -- We prove both inequalities at once to avoid circular dependencies in the signed case.
  cases d with
  | mk sign mant exp =>
      cases sign with
      | false =>
          -- Nonnegative dyadics: nearest-even matches one of the directed endpoints.
          have hcases :=
            roundDyadicToIEEE32_eq_roundDyadicDown_or_roundDyadicUp_pos (mant := mant) (exp := exp)
          -- `hcases` is stated for `d := {sign:=false, mant, exp}`; unfold that `let`.
          have hcases' :
              roundDyadicToIEEE32 { sign := false, mant := mant, exp := exp } =
                  roundDyadicDown { sign := false, mant := mant, exp := exp } ∨
                roundDyadicToIEEE32 { sign := false, mant := mant, exp := exp } =
                  roundDyadicUp { sign := false, mant := mant, exp := exp } := by
            simpa using hcases
          rcases hcases' with hEq | hEq
          · -- `near = down`
            refine ⟨?_, ?_⟩
            · simp [hEq]
            ·
              have hdu :
                  toEReal (roundDyadicDown { sign := false, mant := mant, exp := exp }) ≤
                    toEReal (roundDyadicUp { sign := false, mant := mant, exp := exp }) :=
                toEReal_roundDyadicDown_le_roundDyadicUp (d := { sign := false, mant := mant, exp :=
                  exp })
              simpa [hEq] using hdu
          · -- `near = up`
            refine ⟨?_, ?_⟩
            ·
              have hdu :
                  toEReal (roundDyadicDown { sign := false, mant := mant, exp := exp }) ≤
                    toEReal (roundDyadicUp { sign := false, mant := mant, exp := exp }) :=
                toEReal_roundDyadicDown_le_roundDyadicUp (d := { sign := false, mant := mant, exp :=
                  exp })
              simpa [hEq] using hdu
            · simp [hEq]
      | true =>
          -- Negative dyadics: reduce to the nonnegative case by negation in `EReal`.
          by_cases hm0 : mant = 0
          · subst hm0
            simp [roundDyadicDown, roundDyadicUp, roundDyadicToIEEE32]
          ·
            have hm' : mant ≠ 0 := hm0
            let dpos : Dyadic := { sign := false, mant := mant, exp := exp }
            have hpos :
                toEReal (roundDyadicDown dpos) ≤ toEReal (roundDyadicToIEEE32 dpos) ∧
                  toEReal (roundDyadicToIEEE32 dpos) ≤ toEReal (roundDyadicUp dpos) :=
              by
                have hcases :=
                  roundDyadicToIEEE32_eq_roundDyadicDown_or_roundDyadicUp_pos (mant := mant) (exp :=
                    exp)
                have hcases' :
                    roundDyadicToIEEE32 dpos = roundDyadicDown dpos ∨
                      roundDyadicToIEEE32 dpos = roundDyadicUp dpos := by
                  simpa [dpos] using hcases
                rcases hcases' with hEq | hEq
                · refine ⟨?_, ?_⟩
                  · simp [hEq]
                  ·
                    have hdu :
                        toEReal (roundDyadicDown dpos) ≤ toEReal (roundDyadicUp dpos) :=
                      toEReal_roundDyadicDown_le_roundDyadicUp (d := dpos)
                    simpa [hEq] using hdu
                · refine ⟨?_, ?_⟩
                  ·
                    have hdu :
                        toEReal (roundDyadicDown dpos) ≤ toEReal (roundDyadicUp dpos) :=
                      toEReal_roundDyadicDown_le_roundDyadicUp (d := dpos)
                    simpa [hEq] using hdu
                  · simp [hEq]
            have hnanPosUp : isNaN (roundDyadicPosUp mant exp) = false :=
              isNaN_roundDyadicPosUp_eq_false (mant := mant) (exp := exp) hm'
            have hnanPosDown : isNaN (roundDyadicPosDown mant exp) = false := by
              have ⟨dd, hdd⟩ := toDyadic?_roundDyadicPosDown_some (mant := mant) (exp := exp)
              exact isNaN_eq_false_of_toDyadic?_some (hx := hdd)
            have hnegDown :
                toEReal (roundDyadicDown { sign := true, mant := mant, exp := exp }) =
                  -toEReal (roundDyadicUp dpos) := by
              -- `roundDyadicDown` on negative dyadics is `neg (roundDyadicPosUp ...)`.
              simp [roundDyadicDown, roundDyadicUp, dpos, hm', toEReal_neg_of_isNaN_eq_false,
                hnanPosUp]
            have hnegUp :
                toEReal (roundDyadicUp { sign := true, mant := mant, exp := exp }) =
                  -toEReal (roundDyadicDown dpos) := by
              -- `roundDyadicUp` on negative dyadics is `neg (roundDyadicPosDown ...)`.
              simp [roundDyadicUp, roundDyadicDown, dpos, hm', toEReal_neg_of_isNaN_eq_false,
                hnanPosDown]
            have hnegNear :
                toEReal (roundDyadicToIEEE32 { sign := true, mant := mant, exp := exp }) =
                  -toEReal (roundDyadicToIEEE32 dpos) := by
              -- Unfold `roundDyadicToIEEE32` and use the explicit `mkBits sign ...` constructor in
              -- the
              -- finite cases to show the sign flip negates the real value.
              by_cases hkHiGuard : (127 : Int) < ((mant.log2 : Nat) : Int) + exp
              ·
                -- Overflow branch: `roundDyadicToIEEE32` returns `±∞`, and `toEReal` is `⊤/⊥`.
                simp (config := { zeta := true })
                  [roundDyadicToIEEE32, dpos, hm', hkHiGuard, toEReal_posInf, toEReal_negInf]
              ·
                -- In all remaining branches the output is finite or `±∞`/signed-zero, never NaN.
                -- Split the remaining magnitude guards so `simp` can compute `toEReal` directly.
                have hkHiGuardGT : ¬ (((mant.log2 : Nat) : Int) + exp > (127 : Int)) := by
                  simpa [gt_iff_lt] using hkHiGuard
                by_cases hkUnder : ((mant.log2 : Nat) : Int) + exp < (-150 : Int)
                ·
                  simp (config := { zeta := true })
                    [roundDyadicToIEEE32, dpos, hm', hkHiGuardGT, hkUnder, toEReal_posZero,
                      toEReal_negZero]
                ·
                    by_cases hkSub : ((mant.log2 : Nat) : Int) + exp < (-126 : Int)
                    ·
                      -- Subnormal branch: sign only affects the sign bit (and the signed-zero/±∞
                      -- conventions).
                      classical
                      have hmpos : dpos.mant ≠ 0 := by
                        simpa [dpos] using hm'
                      let fracNat : Nat :=
                        match exp + 149 with
                        | .ofNat sh => mant <<< sh
                        | .negSucc sh => roundShiftRightEven mant (sh + 1)
                      cases hzFrac : (fracNat == 0) with
                      | true =>
                        have hzFracExp :
                            ((match exp + 149 with
                                  | .ofNat sh => mant <<< sh
                                  | .negSucc sh => roundShiftRightEven mant (sh + 1)) ==
                                0) =
                              true := by
                          simpa [fracNat] using hzFrac
                        have hfracEq :
                            (match exp + 149 with
                                | .ofNat sh => mant <<< sh
                                | .negSucc sh => roundShiftRightEven mant (sh + 1)) =
                              0 :=
                          (beq_iff_eq (a :=
                            match exp + 149 with
                            | .ofNat sh => mant <<< sh
                            | .negSucc sh => roundShiftRightEven mant (sh + 1)) (b := 0)).1
                              hzFracExp
                        have hnegOut :
                            roundDyadicToIEEE32 { sign := true, mant := mant, exp := exp } = negZero
                              := by
                          simp (config := { zeta := true })
                            [roundDyadicToIEEE32, hm', hkHiGuardGT, hkUnder, hkSub]
                          intro hne
                          exact (hne hfracEq).elim
                        have hposOut : roundDyadicToIEEE32 dpos = posZero := by
                          simp (config := { zeta := true })
                            [roundDyadicToIEEE32, dpos, hmpos, hkHiGuardGT, hkUnder, hkSub]
                          intro hne
                          exact (hne hfracEq).elim
                        simp [hnegOut, hposOut, toEReal_posZero, toEReal_negZero]
                      | false =>
                        have hzFracExp :
                            ((match exp + 149 with
                                  | .ofNat sh => mant <<< sh
                                  | .negSucc sh => roundShiftRightEven mant (sh + 1)) ==
                                0) =
                              false := by
                          simpa [fracNat] using hzFrac
                        have hfracNe :
                            (match exp + 149 with
                                | .ofNat sh => mant <<< sh
                                | .negSucc sh => roundShiftRightEven mant (sh + 1)) ≠
                              0 :=
                          (beq_eq_false_iff_ne (a :=
                            match exp + 149 with
                            | .ofNat sh => mant <<< sh
                            | .negSucc sh => roundShiftRightEven mant (sh + 1)) (b := 0)).1
                              hzFracExp
                        cases hle : (pow2 23).decLe fracNat with
                        | isTrue hle' =>
                          have hleExp :
                              (pow2 23).decLe
                                  (match exp + 149 with
                                    | .ofNat sh => mant <<< sh
                                    | .negSucc sh => roundShiftRightEven mant (sh + 1)) =
                                isTrue hle' := by
                            simpa [fracNat] using hle
                          have hnegOut :
                              roundDyadicToIEEE32 { sign := true, mant := mant, exp := exp } =
                                ofBits (mkBits true 1 0) := by
                            simp (config := { zeta := true })
                              [roundDyadicToIEEE32, hm', hkHiGuardGT, hkUnder, hkSub]
                            cases h : (exp + 149) with
                            | ofNat sh =>
                              have hfracNe' : mant <<< sh ≠ 0 := by
                                simpa [h] using hfracNe
                              have hle_sh : pow2 23 ≤ mant <<< sh := by
                                simpa [fracNat, h] using hle'
                              simp (config := { failIfUnchanged := false }) [hfracNe']
                              cases hdec : (pow2 23).decLe (mant <<< sh) with
                              | isTrue hle2 => rfl
                              | isFalse hnle => exact (hnle hle_sh).elim
                            | negSucc sh =>
                              have hfracNe' : roundShiftRightEven mant (sh + 1) ≠ 0 := by
                                simpa [h] using hfracNe
                              have hle_sh : pow2 23 ≤ roundShiftRightEven mant (sh + 1) := by
                                simpa [fracNat, h] using hle'
                              simp (config := { failIfUnchanged := false }) [hfracNe']
                              cases hdec : (pow2 23).decLe (roundShiftRightEven mant (sh + 1)) with
                              | isTrue hle2 => rfl
                              | isFalse hnle => exact (hnle hle_sh).elim
                          have hposOut :
                              roundDyadicToIEEE32 dpos = ofBits (mkBits false 1 0) := by
                            simp (config := { zeta := true })
                              [roundDyadicToIEEE32, dpos, hmpos, hkHiGuardGT, hkUnder, hkSub]
                            cases h : (exp + 149) with
                            | ofNat sh =>
                              have hfracNe' : mant <<< sh ≠ 0 := by
                                simpa [h] using hfracNe
                              have hle_sh : pow2 23 ≤ mant <<< sh := by
                                simpa [fracNat, h] using hle'
                              simp (config := { failIfUnchanged := false }) [hfracNe']
                              cases hdec : (pow2 23).decLe (mant <<< sh) with
                              | isTrue hle2 => rfl
                              | isFalse hnle => exact (hnle hle_sh).elim
                            | negSucc sh =>
                              have hfracNe' : roundShiftRightEven mant (sh + 1) ≠ 0 := by
                                simpa [h] using hfracNe
                              have hle_sh : pow2 23 ≤ roundShiftRightEven mant (sh + 1) := by
                                simpa [fracNat, h] using hle'
                              simp (config := { failIfUnchanged := false }) [hfracNe']
                              cases hdec : (pow2 23).decLe (roundShiftRightEven mant (sh + 1)) with
                              | isTrue hle2 => rfl
                              | isFalse hnle => exact (hnle hle_sh).elim
                          have hflip :
                              toEReal (ofBits (mkBits true 1 0) : IEEE32Exec) =
                                -toEReal (ofBits (mkBits false 1 0) : IEEE32Exec) :=
                            toEReal_ofBits_mkBits_fin_signFlip (exp := 1) (frac := 0) (by decide)
                              (by decide)
                          simpa [hnegOut, hposOut] using hflip
                        | isFalse hle' =>
                          have hfracLt : fracNat < 2 ^ 23 := by
                            have : fracNat < pow2 23 := Nat.lt_of_not_ge hle'
                            simpa [pow2_eq_two_pow 23] using this
                          have hleExp :
                              (pow2 23).decLe
                                  (match exp + 149 with
                                    | .ofNat sh => mant <<< sh
                                    | .negSucc sh => roundShiftRightEven mant (sh + 1)) =
                                isFalse hle' := by
                            simpa [fracNat] using hle
                          have hnegOut :
                              roundDyadicToIEEE32 { sign := true, mant := mant, exp := exp } =
                                ofBits
                                  (mkBits true 0
                                    (match exp + 149 with
                                      | .ofNat sh => mant <<< sh
                                      | .negSucc sh => roundShiftRightEven mant (sh + 1))) := by
                            simp (config := { zeta := true })
                              [roundDyadicToIEEE32, hm', hkHiGuardGT, hkUnder, hkSub]
                            cases h : (exp + 149) with
                            | ofNat sh =>
                              have hfracNe' : mant <<< sh ≠ 0 := by
                                simpa [h] using hfracNe
                              have hnle_sh : ¬pow2 23 ≤ mant <<< sh := by
                                simpa [fracNat, h] using hle'
                              simp (config := { failIfUnchanged := false }) [hfracNe']
                              cases hdec : (pow2 23).decLe (mant <<< sh) with
                              | isTrue hle2 => exact (hnle_sh hle2).elim
                              | isFalse hnle => rfl
                            | negSucc sh =>
                              have hfracNe' : roundShiftRightEven mant (sh + 1) ≠ 0 := by
                                simpa [h] using hfracNe
                              have hnle_sh : ¬pow2 23 ≤ roundShiftRightEven mant (sh + 1) := by
                                simpa [fracNat, h] using hle'
                              simp (config := { failIfUnchanged := false }) [hfracNe']
                              cases hdec : (pow2 23).decLe (roundShiftRightEven mant (sh + 1)) with
                              | isTrue hle2 => exact (hnle_sh hle2).elim
                              | isFalse hnle => rfl
                          have hposOut :
                              roundDyadicToIEEE32 dpos =
                                ofBits
                                  (mkBits false 0
                                    (match exp + 149 with
                                      | .ofNat sh => mant <<< sh
                                      | .negSucc sh => roundShiftRightEven mant (sh + 1))) := by
                            simp (config := { zeta := true })
                              [roundDyadicToIEEE32, dpos, hmpos, hkHiGuardGT, hkUnder, hkSub]
                            cases h : (exp + 149) with
                            | ofNat sh =>
                              have hfracNe' : mant <<< sh ≠ 0 := by
                                simpa [h] using hfracNe
                              have hnle_sh : ¬pow2 23 ≤ mant <<< sh := by
                                simpa [fracNat, h] using hle'
                              simp (config := { failIfUnchanged := false }) [hfracNe']
                              cases hdec : (pow2 23).decLe (mant <<< sh) with
                              | isTrue hle2 => exact (hnle_sh hle2).elim
                              | isFalse hnle => rfl
                            | negSucc sh =>
                              have hfracNe' : roundShiftRightEven mant (sh + 1) ≠ 0 := by
                                simpa [h] using hfracNe
                              have hnle_sh : ¬pow2 23 ≤ roundShiftRightEven mant (sh + 1) := by
                                simpa [fracNat, h] using hle'
                              simp (config := { failIfUnchanged := false }) [hfracNe']
                              cases hdec : (pow2 23).decLe (roundShiftRightEven mant (sh + 1)) with
                              | isTrue hle2 => exact (hnle_sh hle2).elim
                              | isFalse hnle => rfl
                          have hflip :
                              toEReal
                                  (ofBits
                                      (mkBits true 0
                                        (match exp + 149 with
                                          | .ofNat sh => mant <<< sh
                                          | .negSucc sh => roundShiftRightEven mant (sh + 1))) :
                                            IEEE32Exec) =
                                -toEReal
                                    (ofBits
                                        (mkBits false 0
                                          (match exp + 149 with
                                            | .ofNat sh => mant <<< sh
                                            | .negSucc sh => roundShiftRightEven mant (sh + 1))) :
                                              IEEE32Exec) :=
                            toEReal_ofBits_mkBits_fin_signFlip
                              (exp := 0)
                              (frac :=
                                match exp + 149 with
                                | .ofNat sh => mant <<< sh
                                | .negSucc sh => roundShiftRightEven mant (sh + 1))
                              (by decide)
                              (by
                                -- `hfracLt` is stated over `fracNat`, which is definitional to this
                                -- `match`.
                                simpa [fracNat] using hfracLt)
                          simpa [hnegOut, hposOut] using hflip
                    ·
                      -- Normal rounding branch: the only dependence on `sign` is the final sign bit
                      -- / ±∞ choice.
                      classical
                      have hmbeq : (mant == 0) = false := (beq_eq_false_iff_ne).2 hm'
                      let log2m : Nat := mant.log2
                      let k : Int := (Int.ofNat log2m) + exp
                      let m24 : Nat :=
                        if log2m >= 23 then
                          roundShiftRightEven mant (log2m - 23)
                        else
                          mant <<< (23 - log2m)
                      let k' : Int := if m24 = pow2 24 then k + 1 else k
                      let m24' : Nat := if m24 = pow2 24 then pow2 23 else m24
                      by_cases hk'Hi : k' > (127 : Int)
                      ·
                        have hk'HiExp : (if m24 = pow2 24 then k + 1 else k) > (127 : Int) := by
                          simpa [k'] using hk'Hi
                        have hk'HiLt : (127 : Int) < if m24 = pow2 24 then k + 1 else k := by
                          simpa [gt_iff_lt] using hk'HiExp
                        have hk'HiLtExp :
                            (127 : Int) <
                                (if
                                    (if 23 ≤ mant.log2 then roundShiftRightEven mant (mant.log2 -
                                      23)
                                      else mant <<< (23 - mant.log2)) =
                                      pow2 24 then
                                  ↑mant.log2 + exp + 1
                                else ↑mant.log2 + exp) := by
                          simpa [log2m, k, m24] using hk'HiLt
                        have hnegOut :
                            roundDyadicToIEEE32 { sign := true, mant := mant, exp := exp } = negInf
                              := by
                          -- Overflow-after-carry case: the final exponent-range check triggers ±∞.
                          simp (config := { zeta := true })
                            [roundDyadicToIEEE32, hm0, hkHiGuard, hkUnder, hkSub, hk'HiLtExp]
                        have hposOut : roundDyadicToIEEE32 dpos = posInf := by
                          -- Overflow-after-carry case: the final exponent-range check triggers ±∞.
                          simp (config := { zeta := true })
                            [roundDyadicToIEEE32, dpos, hm0, hkHiGuard, hkUnder, hkSub, hk'HiLtExp]
                        simp [hnegOut, hposOut, toEReal_posInf, toEReal_negInf]
                      ·
                        have hk'HiExp : ¬(if m24 = pow2 24 then k + 1 else k) > (127 : Int) := by
                          simpa [k'] using hk'Hi
                        have hk'HiLt : ¬(127 : Int) < if m24 = pow2 24 then k + 1 else k := by
                          simpa [gt_iff_lt] using hk'HiExp
                        have hk'HiLtExp :
                            ¬(127 : Int) <
                                (if
                                    (if 23 ≤ mant.log2 then roundShiftRightEven mant (mant.log2 -
                                      23)
                                      else mant <<< (23 - mant.log2)) =
                                      pow2 24 then
                                  ↑mant.log2 + exp + 1
                                else ↑mant.log2 + exp) := by
                          -- `k'` is syntactically the above `if ...` after unfolding `k` and `m24`.
                          simpa [log2m, k, m24] using hk'HiLt
                        let expNat : Nat := Int.toNat (k' + 127)
                        let fracNat : Nat := m24' - pow2 23
                        have hnegOut :
                            roundDyadicToIEEE32 { sign := true, mant := mant, exp := exp } =
                              ofBits (mkBits true expNat fracNat) := by
                          -- Unfold `roundDyadicToIEEE32` and discharge the guard `if`s using our
                          -- hypotheses.
                          simp (config := { zeta := true })
                            [roundDyadicToIEEE32, hm0, hkHiGuard, hkUnder, hkSub, hk'HiLtExp]
                          simp [expNat, fracNat, k', log2m, k, m24, m24']
                        have hposOut :
                            roundDyadicToIEEE32 dpos = ofBits (mkBits false expNat fracNat) := by
                          -- Unfold `roundDyadicToIEEE32` and discharge the guard `if`s using our
                          -- hypotheses.
                          simp (config := { zeta := true })
                            [roundDyadicToIEEE32, dpos, hm0, hkHiGuard, hkUnder, hkSub, hk'HiLtExp]
                          simp [expNat, fracNat, k', log2m, k, m24, m24']
                        have hexpNat : expNat < 255 := by
                          have hk'le : k' ≤ (127 : Int) := le_of_not_gt hk'Hi
                          have hle : k' + 127 ≤ (254 : Int) := by
                            have := Int.add_le_add_right hk'le 127
                            simpa using this
                          have hleNat : Int.toNat (k' + 127) ≤ 254 :=
                            (Int.toNat_le).2 (by simpa using hle)
                          -- `expNat` is just a name for `Int.toNat (k' + 127)`.
                          simpa [expNat] using (lt_of_le_of_lt hleNat (by decide : 254 < 255))
                        have hm24'_lt : m24' < pow2 24 := by
                          by_cases hcarry : m24 = pow2 24
                          ·
                            have hp : pow2 23 < pow2 24 := by
                              simp [pow2_eq_two_pow]
                            simpa [m24', hcarry] using hp
                          ·
                            have hm24_le : m24 ≤ pow2 24 := by
                              by_cases hlog : 23 ≤ log2m
                              ·
                                set shift : Nat := log2m - 23
                                set q : Nat := Nat.shiftRight mant shift
                                have hq_lt : q < pow2 24 := by
                                  have :
                                      Nat.shiftRight mant (mant.log2 - 23) < pow2 24 :=
                                    shiftRight_log2_sub_lt_pow2_24 (mant := mant) (_hm := hm') (hlog
                                      := by
                                      simpa [log2m] using hlog)
                                  simpa [q, shift, log2m] using this
                                have hnear_cases :
                                    roundShiftRightEven mant shift = q ∨
                                      roundShiftRightEven mant shift = q + 1 := by
                                  simpa [q] using
                                    (roundShiftRightEven_eq_shiftRight_or_shiftRight_add1 (n :=
                                      mant)
                                    (shift := shift))
                                rcases hnear_cases with hqNear | hqNear
                                ·
                                  have : m24 = q := by
                                    simpa [m24, log2m, hlog, shift, q] using hqNear
                                  exact this ▸ le_of_lt hq_lt
                                ·
                                  have : m24 = q + 1 := by
                                    simpa [m24, log2m, hlog, shift, q] using hqNear
                                  have hq1_le : q + 1 ≤ pow2 24 := Nat.succ_le_of_lt hq_lt
                                  exact this ▸ hq1_le
                              ·
                                have hlog' : log2m < 23 := Nat.lt_of_not_ge hlog
                                have hm24_lt :
                                    Nat.shiftLeft mant (23 - mant.log2) < pow2 24 := by
                                  exact shiftLeft_lt_pow2_24_of_log2_lt (mant := mant) (_hm := hm')
                                    (hlog := by
                                    simpa [log2m] using hlog')
                                have hm24_lt' : m24 < pow2 24 := by
                                  -- Reduce `m24` to the shift-left branch and apply the bound
                                  -- lemma.
                                  simpa [m24, hlog, log2m] using (by simpa [log2m] using hm24_lt)
                                exact le_of_lt hm24_lt'
                            have hm24_lt : m24 < pow2 24 :=
                              lt_of_le_of_ne hm24_le hcarry
                            simpa [m24', hcarry] using hm24_lt
                        have hfrac_lt : fracNat < 2 ^ 23 := by
                          -- `m24' < 2^24` implies `m24' - 2^23 < 2^23` (with a split on `m24' <
                          -- 2^23`).
                          have hp23 : 0 < pow2 23 := by
                            simp [pow2_eq_two_pow]
                          have hpow : pow2 24 = pow2 23 + pow2 23 := by
                            simp [pow2_eq_two_pow, Nat.pow_succ, Nat.mul_two, Nat.mul_comm]
                          have hfrac_lt' : m24' - pow2 23 < pow2 23 := by
                            by_cases hsmall : m24' < pow2 23
                            ·
                              have hsub : m24' - pow2 23 = 0 :=
                                Nat.sub_eq_zero_of_le (le_of_lt hsmall)
                              simpa [fracNat, hsub] using hp23
                            ·
                              have hge : pow2 23 ≤ m24' := le_of_not_gt hsmall
                              have hlt : m24' < pow2 23 + pow2 23 := by
                                simpa [hpow] using hm24'_lt
                              have : m24' - pow2 23 < pow2 23 :=
                                Nat.sub_lt_left_of_lt_add hge hlt
                              simpa [fracNat] using this
                          simpa [pow2_eq_two_pow] using hfrac_lt'
                        have hflip :
                            toEReal (ofBits (mkBits true expNat fracNat) : IEEE32Exec) =
                              -toEReal (ofBits (mkBits false expNat fracNat) : IEEE32Exec) :=
                          toEReal_ofBits_mkBits_fin_signFlip (exp := expNat) (frac := fracNat)
                            hexpNat hfrac_lt
                        simpa [hnegOut, hposOut] using hflip
            have hlo :
                toEReal (roundDyadicDown { sign := true, mant := mant, exp := exp }) ≤
                  toEReal (roundDyadicToIEEE32 { sign := true, mant := mant, exp := exp }) := by
              -- Negate `nearPos ≤ upPos` to obtain `-upPos ≤ -nearPos`.
              have hneg :
                  -toEReal (roundDyadicUp dpos) ≤ -toEReal (roundDyadicToIEEE32 dpos) := by
                simpa using (EReal.neg_le_neg_iff).2 hpos.2
              simpa [hnegDown, hnegNear] using hneg
            have hhi :
                toEReal (roundDyadicToIEEE32 { sign := true, mant := mant, exp := exp }) ≤
                  toEReal (roundDyadicUp { sign := true, mant := mant, exp := exp }) := by
              -- Negate `downPos ≤ nearPos` to obtain `-nearPos ≤ -downPos`.
              have hneg :
                  -toEReal (roundDyadicToIEEE32 dpos) ≤ -toEReal (roundDyadicDown dpos) := by
                simpa using (EReal.neg_le_neg_iff).2 hpos.1
              simpa [hnegNear, hnegUp] using hneg
            exact ⟨hlo, hhi⟩

/-- Lower bound half of `toEReal_roundDyadicDown_le_roundDyadicToIEEE32_le_roundDyadicUp`. -/
theorem toEReal_roundDyadicDown_le_roundDyadicToIEEE32 (d : Dyadic) :
    toEReal (roundDyadicDown d) ≤ toEReal (roundDyadicToIEEE32 d) :=
  (toEReal_roundDyadicDown_le_roundDyadicToIEEE32_le_roundDyadicUp (d := d)).1

/-- Upper bound half of `toEReal_roundDyadicDown_le_roundDyadicToIEEE32_le_roundDyadicUp`. -/
theorem toEReal_roundDyadicToIEEE32_le_roundDyadicUp (d : Dyadic) :
    toEReal (roundDyadicToIEEE32 d) ≤ toEReal (roundDyadicUp d) :=
  (toEReal_roundDyadicDown_le_roundDyadicToIEEE32_le_roundDyadicUp (d := d)).2

end
end IEEE32Exec
end TorchLean.Floats.IEEE754
