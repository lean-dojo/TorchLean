/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.Nat.Bitwise
public import NN.Floats.IEEEExec.Exec32

/-!
# Special-value rules for `IEEE32Exec`

`IEEE32Exec` is the executable, proof-relevant bit-level IEEE-754 binary32 kernel. The operations
are defined by case splits (`chooseNaN2`/`chooseNaN3`, Inf rules, then exact+rounded finite core).

We collect the "special-value" part of those definitions as reusable theorems: how NaNs (including
payload selection/quieting) and infinities propagate.

These lemmas are *fully proved* in Lean because they are consequences of the executable definitions
in `NN/Floats/IEEEExec/Exec32.lean`.

Why keep these lemmas in a separate file?

- The executable definitions in `Exec32.lean` are necessarily a bit long: they split on NaN/Inf/0
  cases, decode to exact intermediates, and then round back to float32.
- In proofs (and in higher-level specs), we usually want to *use* those definitions without
  unfolding them over and over. The lemmas here are the small, reliable rewrite rules we reach for.

## Binary32 reminder (math we use repeatedly)

IEEE-754 binary32 is stored as a 32-bit word split into fields:

- 1 sign bit,
- 8 exponent bits,
- 23 fraction ("mantissa") bits.

If we write `e` for the exponent field and `f` for the fraction field, then:

- `x` is **finite**  iff `e ≠ 255`,
- `x` is `±∞`        iff `e = 255 ∧ f = 0`,
- `x` is a **NaN**   iff `e = 255 ∧ f ≠ 0`.

`isFinite`, `isInf`, and `isNaN` in `Exec32.lean` are just executable versions of those predicates.

Finally, a **signaling NaN** is a NaN whose fraction's designated “quiet bit” is *not* set.
Our `quietNaN` turns any NaN into a quiet NaN by setting that one bit, leaving the exponent field
unchanged.

## References (background)

- IEEE Standard for Floating-Point Arithmetic, IEEE 754-2019.
- David Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic",
  *ACM Computing Surveys* (1991). DOI: 10.1145/103162.103163
- Jean-Michel Muller et al., *Handbook of Floating-Point Arithmetic*, 2nd ed. (2018).
-/

@[expose] public section


namespace TorchLean.Floats.IEEE754

namespace IEEE32Exec

/-! ## NaN selection helpers -/

/-! ## Bitfield facts used by special-case lemmas -/

private lemma expField_ofBits_or_quietBit (b : UInt32) :
    expField (ofBits (b ||| quietBit)) = expField (ofBits b) := by
  -- `quietNaN` is implemented by OR-ing a fixed bit (`quietBit`) into the fraction field.
  -- This lemma records the key invariant: the exponent field is unchanged.
  --
  -- Proof idea: show the exponent fields have the same `toNat` value by proving that all exponent
  -- `testBit`s agree.
  apply UInt32.toNat_inj.1
  have hQuiet : quietBit.toNat = 2 ^ 22 := by decide
  have hExpMask : expAllOnes.toNat = 2 ^ 8 - 1 := by decide
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases hi : i < 8
  · have hmask : Nat.testBit expAllOnes.toNat i = true := by
      -- `expAllOnes` is `2^8 - 1`, so bits 0..7 are set.
      simpa [hExpMask, hi] using (Nat.testBit_two_pow_sub_one 8 i)
    have hQuietBit :
        Nat.testBit quietBit.toNat (23 + i) = false := by
      have hlt : 22 < 23 + i := by
        have : 23 ≤ 23 + i := Nat.le_add_right 23 i
        exact lt_of_lt_of_le (by decide : 22 < 23) this
      have hne : 22 ≠ 23 + i := Nat.ne_of_lt hlt
      simpa [hQuiet] using (Nat.testBit_two_pow_of_ne (n := 22) (m := 23 + i) hne)
    calc
      Nat.testBit (((((b ||| quietBit).toNat >>> 23) &&& expAllOnes.toNat))) i
          = (Nat.testBit (((b ||| quietBit).toNat >>> 23)) i && Nat.testBit expAllOnes.toNat i) :=
            by
              simp []
      _ = Nat.testBit (((b ||| quietBit).toNat >>> 23)) i := by simp [hmask]
      _ = Nat.testBit ((b ||| quietBit).toNat) (23 + i) := by
            simp [Nat.testBit_shiftRight]
      _ = Nat.testBit (b.toNat ||| quietBit.toNat) (23 + i) := by
            simp [UInt32.toNat_or]
      _ = (Nat.testBit b.toNat (23 + i) || Nat.testBit quietBit.toNat (23 + i)) := by
            simp []
      _ = Nat.testBit b.toNat (23 + i) := by simp [hQuietBit]
      _ = Nat.testBit (b.toNat >>> 23) i := by
            simp [Nat.testBit_shiftRight]
      _ = Nat.testBit ((b.toNat >>> 23) &&& expAllOnes.toNat) i := by
            simp [hmask]
      _ = Nat.testBit (expField (ofBits b)).toNat i := by
            simp [expField, ofBits, UInt32.toNat_and, UInt32.toNat_shiftRight]
  · have hmask : Nat.testBit expAllOnes.toNat i = false := by
      simpa [hExpMask, hi] using (Nat.testBit_two_pow_sub_one 8 i)
    simp [expField, ofBits, UInt32.toNat_and, UInt32.toNat_shiftRight, UInt32.toNat_or, hmask]

/--
Quieting a NaN does not change the exponent field.

Informal: `quietNaN` only ORs `quietBit` into the fraction payload; exponent bits are preserved.
-/
theorem expField_quietNaN_eq (x : IEEE32Exec) :
    expField (quietNaN x) = expField x := by
  by_cases hx : isNaN x = true
  · -- `quietNaN` ORs `quietBit` into the fraction field; exponent bits are unchanged.
    unfold quietNaN
    simp [hx]
    -- Reduce to a raw-bit statement.
    simpa using (expField_ofBits_or_quietBit (b := x.bits))
  · simp [quietNaN, hx]

/-- A shorter name for `expField_quietNaN_eq` (keeps the original name stable). -/
theorem expField_quietNaN (x : IEEE32Exec) : expField (quietNaN x) = expField x :=
  expField_quietNaN_eq x

/-- If `x` is a NaN, then its exponent field is all ones. -/
theorem expField_eq_expAllOnes_of_isNaN (x : IEEE32Exec) (hx : isNaN x = true) :
    expField x = expAllOnes := by
  -- By definition, `isNaN x` checks `(expField x == expAllOnes) && (fracField x != 0)`.
  -- We only need the exponent-field part of that check.
  have hx' : (expField x == expAllOnes && fracField x != 0) = true := by
    simpa [isNaN] using hx
  have hexp : (expField x == expAllOnes) = true := by
    have : (expField x == expAllOnes) = true ∧ (fracField x != 0) = true := by
      simpa [Bool.and_eq_true] using hx'
    exact this.1
  exact (beq_iff_eq).1 hexp

/-- A shorter name for `expField_eq_expAllOnes_of_isNaN`. -/
theorem expField_allOnes_of_isNaN (x : IEEE32Exec) (hx : isNaN x = true) :
    expField x = expAllOnes :=
  expField_eq_expAllOnes_of_isNaN x hx

/-- If `x` is an infinity, then its exponent field is all ones. -/
theorem expField_eq_expAllOnes_of_isInf (x : IEEE32Exec) (hx : isInf x = true) :
    expField x = expAllOnes := by
  -- By definition, `isInf x` checks `(expField x == expAllOnes) && (fracField x == 0)`.
  have hx' : (expField x == expAllOnes && fracField x == 0) = true := by
    simpa [isInf] using hx
  have hexp : (expField x == expAllOnes) = true := by
    have : (expField x == expAllOnes) = true ∧ (fracField x == 0) = true := by
      simpa [Bool.and_eq_true] using hx'
    exact this.1
  exact (beq_iff_eq).1 hexp

/-- A shorter name for `expField_eq_expAllOnes_of_isInf`. -/
theorem expField_allOnes_of_isInf (x : IEEE32Exec) (hx : isInf x = true) :
    expField x = expAllOnes :=
  expField_eq_expAllOnes_of_isInf x hx

/-- If the exponent field is all ones, the value is not finite (it is either `Inf` or `NaN`). -/
theorem isFinite_eq_false_of_expField_eq_expAllOnes (x : IEEE32Exec) (hx : expField x = expAllOnes)
  :
    isFinite x = false := by
  -- `isFinite` is the executable version of "exponent field is not all ones".
  unfold isFinite
  simp [hx]

/-!
## "Choosing a NaN" always yields a non-finite result

The `chooseNaN*` helpers return a NaN operand (quieted). This makes it easy to prove that "once a
NaN is selected, the result is not finite" without redoing bitfield reasoning at every call site.
-/

/--
If `chooseNaN1 x = some nan`, then `nan` is not finite.

Mathematically: `chooseNaN1` only returns `some _` when `x` is a NaN, and in that case it returns
`quietNaN x`. Quieting sets a fraction bit but keeps the exponent field equal to all ones, so the
result is not finite.
-/
theorem isFinite_eq_false_of_chooseNaN1_some (x nan : IEEE32Exec) (h : chooseNaN1 x = some nan) :
    isFinite nan = false := by
  unfold chooseNaN1 at h
  cases hx : isNaN x with
  | true =>
      have h' : some (quietNaN x) = some nan := by
        simpa [hx] using h
      have hnan : nan = quietNaN x := by
        simpa using (Option.some.inj h').symm
      have hexp : expField nan = expAllOnes := by
        simp [hnan, expField_quietNaN_eq, expField_eq_expAllOnes_of_isNaN x hx]
      exact isFinite_eq_false_of_expField_eq_expAllOnes (x := nan) hexp
  | false =>
      have : (none : Option IEEE32Exec) = some nan := by
        simp [hx] at h
      cases this

/--
If `chooseNaN2 x y = some nan`, then `nan` is not finite.

This is the core invariant behind the op-level facts like
`add_eq_of_chooseNaN2_some` / `mul_eq_of_chooseNaN2_some`: once a NaN is selected, we know it is a
NaN value at the bit level (exponent field is all ones), hence not finite.
-/
theorem isFinite_eq_false_of_chooseNaN2_some (x y nan : IEEE32Exec) (h : chooseNaN2 x y = some nan)
  :
    isFinite nan = false := by
  -- `chooseNaN2` returns a (quieted) NaN operand, hence its exponent field is all ones.
  cases hxS : isSNaN x with
  | true =>
      have hxNaN : isNaN x = true := by
        have : isNaN x = true ∧ ((x.bits &&& quietBit) == 0) = true := by
          simpa [isSNaN, Bool.and_eq_true] using hxS
        exact this.1
      have h' : some (quietNaN x) = some nan := by
        simpa [chooseNaN2, hxS] using h
      have hnan : nan = quietNaN x := by
        simpa using (Option.some.inj h').symm
      have hexp : expField nan = expAllOnes := by
        simp [hnan, expField_quietNaN_eq, expField_eq_expAllOnes_of_isNaN x hxNaN]
      exact isFinite_eq_false_of_expField_eq_expAllOnes (x := nan) hexp
  | false =>
      cases hyS : isSNaN y with
      | true =>
          have hyNaN : isNaN y = true := by
            have : isNaN y = true ∧ ((y.bits &&& quietBit) == 0) = true := by
              simpa [isSNaN, Bool.and_eq_true] using hyS
            exact this.1
          have h' : some (quietNaN y) = some nan := by
            simpa [chooseNaN2, hxS, hyS] using h
          have hnan : nan = quietNaN y := by
            simpa using (Option.some.inj h').symm
          have hexp : expField nan = expAllOnes := by
            simp [hnan, expField_quietNaN_eq, expField_eq_expAllOnes_of_isNaN y hyNaN]
          exact isFinite_eq_false_of_expField_eq_expAllOnes (x := nan) hexp
      | false =>
          cases hxN : isNaN x with
          | true =>
              have h' : some (quietNaN x) = some nan := by
                simpa [chooseNaN2, hxS, hyS, hxN] using h
              have hnan : nan = quietNaN x := by
                simpa using (Option.some.inj h').symm
              have hexp : expField nan = expAllOnes := by
                simp [hnan, expField_quietNaN_eq, expField_eq_expAllOnes_of_isNaN x hxN]
              exact isFinite_eq_false_of_expField_eq_expAllOnes (x := nan) hexp
          | false =>
              cases hyN : isNaN y with
              | true =>
                  have h' : some (quietNaN y) = some nan := by
                    simpa [chooseNaN2, hxS, hyS, hxN, hyN] using h
                  have hnan : nan = quietNaN y := by
                    simpa using (Option.some.inj h').symm
                  have hexp : expField nan = expAllOnes := by
                    simp [hnan, expField_quietNaN_eq, expField_eq_expAllOnes_of_isNaN y hyN]
                  exact isFinite_eq_false_of_expField_eq_expAllOnes (x := nan) hexp
              | false =>
                  -- Contradiction: `chooseNaN2 x y = none`.
                  have : (none : Option IEEE32Exec) = some nan := by
                    simp [chooseNaN2, hxS, hyS, hxN, hyN] at h
                  cases this

/--
If `chooseNaN3 x y z = some nan`, then `nan` is not finite.

This is the ternary analogue of `isFinite_eq_false_of_chooseNaN2_some`, used for `fma`.
-/
theorem isFinite_eq_false_of_chooseNaN3_some (x y z nan : IEEE32Exec) (h : chooseNaN3 x y z = some
  nan) :
    isFinite nan = false := by
  -- `chooseNaN3` returns a (quieted) NaN operand, hence its exponent field is all ones.
  cases hxS : isSNaN x with
  | true =>
      have hxNaN : isNaN x = true := by
        have : isNaN x = true ∧ ((x.bits &&& quietBit) == 0) = true := by
          simpa [isSNaN, Bool.and_eq_true] using hxS
        exact this.1
      have h' : some (quietNaN x) = some nan := by
        simpa [chooseNaN3, hxS] using h
      have hnan : nan = quietNaN x := by
        simpa using (Option.some.inj h').symm
      have hexp : expField nan = expAllOnes := by
        simp [hnan, expField_quietNaN_eq, expField_eq_expAllOnes_of_isNaN x hxNaN]
      exact isFinite_eq_false_of_expField_eq_expAllOnes (x := nan) hexp
  | false =>
      cases hyS : isSNaN y with
      | true =>
          have hyNaN : isNaN y = true := by
            have : isNaN y = true ∧ ((y.bits &&& quietBit) == 0) = true := by
              simpa [isSNaN, Bool.and_eq_true] using hyS
            exact this.1
          have h' : some (quietNaN y) = some nan := by
            simpa [chooseNaN3, hxS, hyS] using h
          have hnan : nan = quietNaN y := by
            simpa using (Option.some.inj h').symm
          have hexp : expField nan = expAllOnes := by
            simp [hnan, expField_quietNaN_eq, expField_eq_expAllOnes_of_isNaN y hyNaN]
          exact isFinite_eq_false_of_expField_eq_expAllOnes (x := nan) hexp
      | false =>
          cases hzS : isSNaN z with
          | true =>
              have hzNaN : isNaN z = true := by
                have : isNaN z = true ∧ ((z.bits &&& quietBit) == 0) = true := by
                  simpa [isSNaN, Bool.and_eq_true] using hzS
                exact this.1
              have h' : some (quietNaN z) = some nan := by
                simpa [chooseNaN3, hxS, hyS, hzS] using h
              have hnan : nan = quietNaN z := by
                simpa using (Option.some.inj h').symm
              have hexp : expField nan = expAllOnes := by
                simp [hnan, expField_quietNaN_eq, expField_eq_expAllOnes_of_isNaN z hzNaN]
              exact isFinite_eq_false_of_expField_eq_expAllOnes (x := nan) hexp
          | false =>
              cases hxN : isNaN x with
              | true =>
                  have h' : some (quietNaN x) = some nan := by
                    simpa [chooseNaN3, hxS, hyS, hzS, hxN] using h
                  have hnan : nan = quietNaN x := by
                    simpa using (Option.some.inj h').symm
                  have hexp : expField nan = expAllOnes := by
                    simp [hnan, expField_quietNaN_eq, expField_eq_expAllOnes_of_isNaN x hxN]
                  exact isFinite_eq_false_of_expField_eq_expAllOnes (x := nan) hexp
              | false =>
                  cases hyN : isNaN y with
                  | true =>
                      have h' : some (quietNaN y) = some nan := by
                        simpa [chooseNaN3, hxS, hyS, hzS, hxN, hyN] using h
                      have hnan : nan = quietNaN y := by
                        simpa using (Option.some.inj h').symm
                      have hexp : expField nan = expAllOnes := by
                        simp [hnan, expField_quietNaN_eq, expField_eq_expAllOnes_of_isNaN y hyN]
                      exact isFinite_eq_false_of_expField_eq_expAllOnes (x := nan) hexp
                  | false =>
                      cases hzN : isNaN z with
                      | true =>
                          have h' : some (quietNaN z) = some nan := by
                            simpa [chooseNaN3, hxS, hyS, hzS, hxN, hyN, hzN] using h
                          have hnan : nan = quietNaN z := by
                            simpa using (Option.some.inj h').symm
                          have hexp : expField nan = expAllOnes := by
                            simp [hnan, expField_quietNaN_eq, expField_eq_expAllOnes_of_isNaN z hzN]
                          exact isFinite_eq_false_of_expField_eq_expAllOnes (x := nan) hexp
                      | false =>
                          have : (none : Option IEEE32Exec) = some nan := by
                            simp [chooseNaN3, hxS, hyS, hzS, hxN, hyN, hzN] at h
                          cases this

/-!
## NaN selection rules (`chooseNaN2` / `chooseNaN3`)

These are "one-step" lemmas for the NaN-selection helpers themselves. They let you prove facts
about `chooseNaN2`/`chooseNaN3` without unfolding their nested `if` chains.

### What `chooseNaN*` does (informal spec)

Both helpers implement the NaN-propagation part of IEEE-754-style arithmetic:

- If any operand is a signaling NaN (sNaN), return that operand *quieted*.
- Otherwise, if any operand is a (quiet) NaN, return one of them *quieted*.
- Otherwise return `none` (meaning: "no NaN special-case, continue with Inf/finite rules").

The order is left-to-right: if two operands are both NaNs, we keep the first one. That is not a
deep mathematical choice (IEEE leaves payload selection somewhat implementation-defined), but it is
useful for reproducibility and it matches the kind of behavior you typically see when running
PyTorch models on IEEE hardware (payloads propagate in a deterministic, operand-order-dependent
  way).
-/

/-- If the left operand is a signaling NaN, `chooseNaN2` returns it (quieted). -/
theorem chooseNaN2_of_isSNaN_left (x y : IEEE32Exec) (hx : isSNaN x = true) :
    chooseNaN2 x y = some (quietNaN x) := by
  simp [chooseNaN2, hx]

/-- If the right operand is a signaling NaN (and the left one is not), `chooseNaN2` returns it. -/
theorem chooseNaN2_of_isSNaN_right (x y : IEEE32Exec)
    (hx : isSNaN x = false) (hy : isSNaN y = true) :
    chooseNaN2 x y = some (quietNaN y) := by
  simp [chooseNaN2, hx, hy]

/-- If `x` is a quiet NaN and neither operand is a signaling NaN, `chooseNaN2` returns `x`. -/
theorem chooseNaN2_of_isNaN_left (x y : IEEE32Exec)
    (hxS : isSNaN x = false) (hyS : isSNaN y = false) (hx : isNaN x = true) :
    chooseNaN2 x y = some (quietNaN x) := by
  simp [chooseNaN2, hxS, hyS, hx]

/-- If only `y` is a quiet NaN, `chooseNaN2` returns `y`. -/
theorem chooseNaN2_of_isNaN_right (x y : IEEE32Exec)
    (hxS : isSNaN x = false) (hyS : isSNaN y = false)
    (hx : isNaN x = false) (hy : isNaN y = true) :
    chooseNaN2 x y = some (quietNaN y) := by
  simp [chooseNaN2, hxS, hyS, hx, hy]

/-- If neither operand is a NaN, `chooseNaN2` returns `none`. -/
theorem chooseNaN2_none_of_not_isNaN (x y : IEEE32Exec)
    (hx : isNaN x = false) (hy : isNaN y = false) :
    chooseNaN2 x y = none := by
  have hxS : isSNaN x = false := by simp [isSNaN, hx]
  have hyS : isSNaN y = false := by simp [isSNaN, hy]
  simp [chooseNaN2, hxS, hyS, hx, hy]

/-- If `x` is a signaling NaN, `chooseNaN3` returns it (quieted), regardless of `y` and `z`. -/
theorem chooseNaN3_of_isSNaN_x (x y z : IEEE32Exec) (hx : isSNaN x = true) :
    chooseNaN3 x y z = some (quietNaN x) := by
  simp [chooseNaN3, hx]

/--
If `y` is a signaling NaN and `x` is not, `chooseNaN3` returns `y` (quieted).

This is the same precedence rule as `chooseNaN2`: signaling NaNs win (and get quieted).
-/
theorem chooseNaN3_of_isSNaN_y (x y z : IEEE32Exec)
    (hx : isSNaN x = false) (hy : isSNaN y = true) :
    chooseNaN3 x y z = some (quietNaN y) := by
  simp [chooseNaN3, hx, hy]

/-- If only `z` is a signaling NaN, `chooseNaN3` returns `z` (quieted). -/
theorem chooseNaN3_of_isSNaN_z (x y z : IEEE32Exec)
    (hx : isSNaN x = false) (hy : isSNaN y = false) (hz : isSNaN z = true) :
    chooseNaN3 x y z = some (quietNaN z) := by
  simp [chooseNaN3, hx, hy, hz]

-- Alternative names for the `chooseNaN3` signaling-NaN rules (left/mid/right wording).
/-- Alias for `chooseNaN3_of_isSNaN_x` (left operand is a signaling NaN). -/
theorem chooseNaN3_of_isSNaN_left (x y z : IEEE32Exec) (hx : isSNaN x = true) :
    chooseNaN3 x y z = some (quietNaN x) :=
  chooseNaN3_of_isSNaN_x x y z hx

/-- Alias for `chooseNaN3_of_isSNaN_y` (middle operand is a signaling NaN). -/
theorem chooseNaN3_of_isSNaN_mid (x y z : IEEE32Exec)
    (hx : isSNaN x = false) (hy : isSNaN y = true) :
    chooseNaN3 x y z = some (quietNaN y) :=
  chooseNaN3_of_isSNaN_y x y z hx hy

/-- Alias for `chooseNaN3_of_isSNaN_z` (right operand is a signaling NaN). -/
theorem chooseNaN3_of_isSNaN_right (x y z : IEEE32Exec)
    (hx : isSNaN x = false) (hy : isSNaN y = false) (hz : isSNaN z = true) :
    chooseNaN3 x y z = some (quietNaN z) :=
  chooseNaN3_of_isSNaN_z x y z hx hy hz

/-!
## NaN propagation for ops (short-circuit lemmas)

Every IEEE32Exec op starts by calling `chooseNaN*`. If a NaN is selected, the definition returns it
immediately. These lemmas expose that "short-circuit" behavior so higher-level proofs can avoid
unfolding the full op definitions.
-/

/-- NaN short-circuit for addition: `add x y` returns the NaN chosen by `chooseNaN2 x y`. -/
theorem add_eq_of_chooseNaN2_some (x y nan : IEEE32Exec) (h : chooseNaN2 x y = some nan) :
    add x y = nan := by
  simp [add, h]

/-- NaN short-circuit for multiplication: `mul x y` returns the NaN chosen by `chooseNaN2 x y`. -/
theorem mul_eq_of_chooseNaN2_some (x y nan : IEEE32Exec) (h : chooseNaN2 x y = some nan) :
    mul x y = nan := by
  simp [mul, h]

/-- NaN short-circuit for division: `div x y` returns the NaN chosen by `chooseNaN2 x y`. -/
theorem div_eq_of_chooseNaN2_some (x y nan : IEEE32Exec) (h : chooseNaN2 x y = some nan) :
    div x y = nan := by
  simp [div, h]

/-- NaN short-circuit for fused multiply-add: `fma x y z` returns the NaN chosen by `chooseNaN3`. -/
theorem fma_eq_of_chooseNaN3_some (x y z nan : IEEE32Exec) (h : chooseNaN3 x y z = some nan) :
    fma x y z = nan := by
  simp [fma, h]

/-- NaN short-circuit for square root: `sqrt x` returns the NaN chosen by `chooseNaN1 x`. -/
theorem sqrt_eq_of_chooseNaN1_some (x nan : IEEE32Exec) (h : chooseNaN1 x = some nan) :
    sqrt x = nan := by
  simp [sqrt, h]

/-- NaN short-circuit for `minimum`: propagate NaNs, like `torch.minimum`. -/
theorem minimum_eq_of_chooseNaN2_some (x y nan : IEEE32Exec) (h : chooseNaN2 x y = some nan) :
    minimum x y = nan := by
  simp [minimum, h]

/-- NaN short-circuit for `maximum`: propagate NaNs, like `torch.maximum`. -/
theorem maximum_eq_of_chooseNaN2_some (x y nan : IEEE32Exec) (h : chooseNaN2 x y = some nan) :
    maximum x y = nan := by
  simp [maximum, h]

/-!
### Invalid operations (canonical NaN)

Some combinations of infinities and zeros are specified as *invalid* in IEEE-754 and return a NaN
(while also raising an invalid-operation flag in hardware). In our executable kernel we model the
return value: after confirming there is no NaN operand (i.e. `chooseNaN2 = none`), we return the
fixed bit-pattern `canonicalNaN`.

These lemmas expose those branches directly.
-/

/-- `(+∞) + (-∞)` (or `(-∞) + (+∞)`) returns `canonicalNaN` once we know there is no NaN operand. -/
theorem add_eq_canonicalNaN_of_opposite_infinities (x y : IEEE32Exec)
    (hx : isInf x = true) (hy : isInf y = true) (hs : (signBit x == signBit y) = false)
    (hchoose : chooseNaN2 x y = none) :
    add x y = canonicalNaN := by
  simp [add, hchoose, hx, hy, hs]

/-- `∞ / ∞` returns `canonicalNaN` once we know there is no NaN operand. -/
theorem div_eq_canonicalNaN_of_inf_inf (x y : IEEE32Exec)
    (hx : isInf x = true) (hy : isInf y = true) (hchoose : chooseNaN2 x y = none) :
    div x y = canonicalNaN := by
  simp [div, hchoose, hx, hy]

/-- `∞ * 0` returns `canonicalNaN` once we know there is no NaN operand. -/
theorem mul_eq_canonicalNaN_of_inf_zero_left (x y : IEEE32Exec)
    (hx : isInf x = true) (hy : isZero y = true) (hchoose : chooseNaN2 x y = none) :
    mul x y = canonicalNaN := by
  simp [mul, hchoose, hx, hy]

/-- `0 * ∞` returns `canonicalNaN` once we know there is no NaN operand. -/
theorem mul_eq_canonicalNaN_of_inf_zero_right (x y : IEEE32Exec)
    (hx : isZero x = true) (hy : isInf y = true) (hchoose : chooseNaN2 x y = none) :
    mul x y = canonicalNaN := by
  -- `isZero x = true` gives `expField x = 0` and `fracField x = 0`, hence `x` cannot be infinite.
  have hfields : (expField x == 0) = true ∧ (fracField x == 0) = true := by
    simpa [IEEE32Exec.isZero, Bool.and_eq_true] using hx
  have hxExp0 : expField x = 0 := (beq_iff_eq).1 hfields.1
  have hxInf : isInf x = false := by
    have hexp : (expField x == expAllOnes) = false := by
      simp [hxExp0, expAllOnes]
    simp [IEEE32Exec.isInf, hexp]
  simp [mul, hchoose, hxInf, hy, hx]

/--
IEEE comparisons are *unordered* in the presence of NaNs.

Our executable comparator returns `Option Ordering`:

- `some ord` for the normal cases, and
- `none` when the comparison is unordered (because at least one side is NaN).

This matches what you see in practice in PyTorch/NumPy: e.g. comparisons with NaN are false, and
sorting has to pick an explicit policy for NaNs.
-/
theorem compare_eq_none_of_isNaN_left (x y : IEEE32Exec) (hx : isNaN x = true) :
    compare x y = none := by
  simp [compare, hx]

/-- Symmetric version of `compare_eq_none_of_isNaN_left`. -/
theorem compare_eq_none_of_isNaN_right (x y : IEEE32Exec) (hy : isNaN y = true) :
    compare x y = none := by
  by_cases hx : isNaN x = true
  · simp [compare, hx, hy]
  · simp [compare, hx, hy]

/-!
## `minNum` / `maxNum` (quiet-NaN ignoring)

IEEE-754 distinguishes two related families of min/max:

- `minimum` / `maximum`, which propagate NaNs, and
- `minNum` / `maxNum`, which (roughly) ignore quiet NaNs when the other operand is numeric.

Our definitions in `Exec32.lean` follow this common convention, and the lemmas below let proofs
reason about those branches directly.

In PyTorch terms, `minimum`/`maximum` behave like `torch.minimum`/`torch.maximum`, while
`minNum`/`maxNum` are closer to `torch.fmin`/`torch.fmax` (the "ignore NaN when possible" variants).
-/

/-- If the left operand is a signaling NaN, `minNum` returns it (quieted). -/
theorem minNum_eq_of_isSNaN_left (x y : IEEE32Exec) (hx : isSNaN x = true) :
    minNum x y = quietNaN x := by
  simp [minNum, hx]

/-- If the right operand is a signaling NaN (and the left is not), `minNum` returns it (quieted). -/
theorem minNum_eq_of_isSNaN_right (x y : IEEE32Exec)
    (hx : isSNaN x = false) (hy : isSNaN y = true) :
    minNum x y = quietNaN y := by
  simp [minNum, hx, hy]

/--
If `x` is a quiet NaN and `y` is numeric, `minNum x y` returns the numeric operand.

This is the defining difference from `minimum`: `minNum` tries to return a number when one is
available.
-/
theorem minNum_eq_right_of_quietNaN_left (x y : IEEE32Exec)
    (hxS : isSNaN x = false) (hyS : isSNaN y = false)
    (hx : isNaN x = true) (hy : isNaN y = false) :
    minNum x y = y := by
  simp [minNum, hxS, hyS, hx, hy]

/-- If only `y` is a quiet NaN, `minNum` returns `x`. -/
theorem minNum_eq_left_of_quietNaN_right (x y : IEEE32Exec)
    (hxS : isSNaN x = false) (hyS : isSNaN y = false)
    (hx : isNaN x = false) (hy : isNaN y = true) :
    minNum x y = x := by
  simp [minNum, hxS, hyS, hx, hy]

/-- If the left operand is a signaling NaN, `maxNum` returns it (quieted). -/
theorem maxNum_eq_of_isSNaN_left (x y : IEEE32Exec) (hx : isSNaN x = true) :
    maxNum x y = quietNaN x := by
  simp [maxNum, hx]

/-- If the right operand is a signaling NaN (and the left is not), `maxNum` returns it (quieted). -/
theorem maxNum_eq_of_isSNaN_right (x y : IEEE32Exec)
    (hx : isSNaN x = false) (hy : isSNaN y = true) :
    maxNum x y = quietNaN y := by
  simp [maxNum, hx, hy]

/-- If `x` is a quiet NaN and `y` is numeric, `maxNum x y` returns the numeric operand. -/
theorem maxNum_eq_right_of_quietNaN_left (x y : IEEE32Exec)
    (hxS : isSNaN x = false) (hyS : isSNaN y = false)
    (hx : isNaN x = true) (hy : isNaN y = false) :
    maxNum x y = y := by
  simp [maxNum, hxS, hyS, hx, hy]

/-- If only `y` is a quiet NaN, `maxNum` returns `x`. -/
theorem maxNum_eq_left_of_quietNaN_right (x y : IEEE32Exec)
    (hxS : isSNaN x = false) (hyS : isSNaN y = false)
    (hx : isNaN x = false) (hy : isNaN y = true) :
    maxNum x y = x := by
  simp [maxNum, hxS, hyS, hx, hy]

end IEEE32Exec

end TorchLean.Floats.IEEE754
