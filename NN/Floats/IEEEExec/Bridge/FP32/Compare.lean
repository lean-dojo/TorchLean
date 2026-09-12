/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Bridge.FP32.RoundDyadic
public import NN.Floats.IEEEExec.Exec32.Compare
/-!
# IEEE32Exec and FP32: Comparisons and Min/Max
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

/-!
## Comparisons and min/max (finite refinement)

Comparisons are subtle in IEEE-754 because of NaNs and signed zeros. Since `FP32` is a finite-only
model, we only bridge the *finite* behavior here: comparisons/min/max agree with the corresponding
real comparisons once NaNs/Infs are ruled out.
-/

lemma dyadicToReal_eq_signedMant_shiftLeft_toExp (d : Dyadic) (e : Int) (he : e ≤ d.exp) :
    dyadicToReal d =
      (signedMant d.sign (Nat.shiftLeft d.mant (Int.toNat (d.exp - e))) : ℝ) *
        neuralBpow binaryRadix e := by
  have hnonneg : 0 ≤ d.exp - e := sub_nonneg.mpr he
  have htoNat : (Int.ofNat (Int.toNat (d.exp - e))) = d.exp - e := by
    simpa using (Int.toNat_of_nonneg hnonneg)
  have hexp : e + (d.exp - e) = d.exp := by
    simp [sub_eq_add_neg]
  -- Expand the `bpow` product at `d.exp = e + (d.exp - e)`.
  have hbpow : neuralBpow binaryRadix d.exp = neuralBpow binaryRadix e * neuralBpow
    binaryRadix (d.exp - e) := by
    simpa [hexp] using (neuralBpow.add_exp binaryRadix e (d.exp - e))
  -- Turn `bpow (d.exp - e)` into an `ofNat` exponent and absorb it into a `shiftLeft`.
  let sh : Nat := Int.toNat (d.exp - e)
  have hbpow' : neuralBpow binaryRadix (d.exp - e) = neuralBpow binaryRadix (Int.ofNat sh) := by
    simpa [sh] using congrArg (fun t : Int => neuralBpow binaryRadix t) htoNat.symm
  have hshift :
      (signedMant d.sign (Nat.shiftLeft d.mant sh) : ℝ) =
        (signedMant d.sign d.mant : ℝ) * neuralBpow binaryRadix (d.exp - e) := by
    have hshift0 :
        (signedMant d.sign (Nat.shiftLeft d.mant sh) : ℝ) =
          (signedMant d.sign d.mant : ℝ) * neuralBpow binaryRadix (Int.ofNat sh) := by
      simpa [sh] using (signedMant_shiftLeft (sign := d.sign) (m := d.mant) (sh := sh))
    -- Replace `bpow (ofNat sh)` with `bpow (d.exp - e)`.
    simpa [hbpow'] using hshift0
  calc
    dyadicToReal d = (signedMant d.sign d.mant : ℝ) * neuralBpow binaryRadix d.exp := by
      simpa using (dyadicToReal_eq_signedMant (d := d))
    _ =
        (signedMant d.sign d.mant : ℝ) *
          (neuralBpow binaryRadix e * neuralBpow binaryRadix (d.exp - e)) := by
      simp [hbpow]
    _ =
        ((signedMant d.sign d.mant : ℝ) * neuralBpow binaryRadix (d.exp - e)) *
          neuralBpow binaryRadix e := by
      ring_nf
    _ = (signedMant d.sign (Nat.shiftLeft d.mant sh) : ℝ) * neuralBpow binaryRadix e := by
      -- Replace the first factor using `hshift`.
      simpa [mul_assoc] using congrArg (fun z : ℝ => z * neuralBpow binaryRadix e) hshift.symm

/--
Dyadic comparison correctness (lt case).

`cmpDyadic` compares two decoded dyadics by aligning exponents and comparing signed integers.
This theorem states that the `.lt` result agrees with the real-ordering of `dyadicToReal`.
-/
theorem cmpDyadic_lt_iff (a b : Dyadic) :
    cmpDyadic a b = .lt ↔ dyadicToReal a < dyadicToReal b := by
  classical
  unfold cmpDyadic
  cases hzero : (a.mant == 0 && b.mant == 0) with
  | true =>
      -- Both are real zero.
      have hab : (a.mant == 0) = true ∧ (b.mant == 0) = true := by
        simpa [Bool.and_eq_true] using (show (a.mant == 0 && b.mant == 0) = true from hzero)
      have ha0 : a.mant = 0 := (beq_iff_eq).1 hab.1
      have hb0 : b.mant = 0 := (beq_iff_eq).1 hab.2
      simp [dyadicToReal, ha0, hb0]
  | false =>
      -- Reduce to comparing aligned signed integers, then map to ℝ via `dyadicToReal`.
      let e : Int := if a.exp ≤ b.exp then a.exp else b.exp
      have heA : e ≤ a.exp := by
        by_cases hab : a.exp ≤ b.exp
        · simp [e, hab]
        · have : b.exp ≤ a.exp := le_of_lt (lt_of_not_ge (show ¬ b.exp ≥ a.exp by simpa using hab))
          simp [e, hab, this]
      have heB : e ≤ b.exp := by
        by_cases hab : a.exp ≤ b.exp
        · simp [e, hab]
        · simp [e, hab]
      let shA : Nat := Int.toNat (a.exp - e)
      let shB : Nat := Int.toNat (b.exp - e)
      let aNat : Nat := Nat.shiftLeft a.mant shA
      let bNat : Nat := Nat.shiftLeft b.mant shB
      let aInt : Int := signedMant a.sign aNat
      let bInt : Int := signedMant b.sign bNat
      have hcmp : cmpDyadic a b = Ord.compare aInt bInt := by
        simp (config := { zeta := true }) [cmpDyadic, hzero, e, shA, shB, aNat, bNat, aInt, bInt,
          signedMant]
      have ha : dyadicToReal a = (aInt : ℝ) * neuralBpow binaryRadix e := by
        simp [aInt, aNat, shA, e, dyadicToReal_eq_signedMant_shiftLeft_toExp (d := a) (e := e) heA]
      have hb : dyadicToReal b = (bInt : ℝ) * neuralBpow binaryRadix e := by
        simp [bInt, bNat, shB, e, dyadicToReal_eq_signedMant_shiftLeft_toExp (d := b) (e := e) heB]
      have hbpos : 0 < neuralBpow binaryRadix e := neuralBpow.pos binaryRadix e
      -- Cancel the positive scaling factor `2^e`.
      have hlt :
          aInt < bInt ↔ dyadicToReal a < dyadicToReal b := by
        constructor
        · intro hab
          have habR : (aInt : ℝ) < (bInt : ℝ) := by
            exact_mod_cast hab
          have habScaled :
              (aInt : ℝ) * neuralBpow binaryRadix e < (bInt : ℝ) * neuralBpow binaryRadix e :=
            mul_lt_mul_of_pos_right habR hbpos
          simpa [ha, hb] using habScaled
        · intro h
          have habScaled :
              (aInt : ℝ) * neuralBpow binaryRadix e < (bInt : ℝ) * neuralBpow binaryRadix e :=
                by
            simpa [ha, hb] using h
          have habR : (aInt : ℝ) < (bInt : ℝ) :=
            lt_of_mul_lt_mul_right habScaled (le_of_lt hbpos)
          exact (by exact_mod_cast habR)
      -- Finish.
      simpa [hcmp, compare_lt_iff_lt, hlt]

/--
Dyadic comparison correctness (eq case).

This is the equality variant of `cmpDyadic_lt_iff`.
-/
theorem cmpDyadic_eq_iff (a b : Dyadic) :
    cmpDyadic a b = .eq ↔ dyadicToReal a = dyadicToReal b := by
  classical
  unfold cmpDyadic
  cases hzero : (a.mant == 0 && b.mant == 0) with
  | true =>
      have hab : (a.mant == 0) = true ∧ (b.mant == 0) = true := by
        simpa [Bool.and_eq_true] using (show (a.mant == 0 && b.mant == 0) = true from hzero)
      have ha0 : a.mant = 0 := (beq_iff_eq).1 hab.1
      have hb0 : b.mant = 0 := (beq_iff_eq).1 hab.2
      simp [dyadicToReal, ha0, hb0]
  | false =>
      let e : Int := if a.exp ≤ b.exp then a.exp else b.exp
      have heA : e ≤ a.exp := by
        by_cases hab : a.exp ≤ b.exp
        · simp [e, hab]
        · have : b.exp ≤ a.exp := le_of_lt (lt_of_not_ge (show ¬ b.exp ≥ a.exp by simpa using hab))
          simp [e, hab, this]
      have heB : e ≤ b.exp := by
        by_cases hab : a.exp ≤ b.exp
        · simp [e, hab]
        · simp [e, hab]
      let shA : Nat := Int.toNat (a.exp - e)
      let shB : Nat := Int.toNat (b.exp - e)
      let aNat : Nat := Nat.shiftLeft a.mant shA
      let bNat : Nat := Nat.shiftLeft b.mant shB
      let aInt : Int := signedMant a.sign aNat
      let bInt : Int := signedMant b.sign bNat
      have hcmp : cmpDyadic a b = Ord.compare aInt bInt := by
        simp (config := { zeta := true }) [cmpDyadic, hzero, e, shA, shB, aNat, bNat, aInt, bInt,
          signedMant]
      have ha : dyadicToReal a = (aInt : ℝ) * neuralBpow binaryRadix e := by
        simp [aInt, aNat, shA, e, dyadicToReal_eq_signedMant_shiftLeft_toExp (d := a) (e := e) heA]
      have hb : dyadicToReal b = (bInt : ℝ) * neuralBpow binaryRadix e := by
        simp [bInt, bNat, shB, e, dyadicToReal_eq_signedMant_shiftLeft_toExp (d := b) (e := e) heB]
      have hbpos : 0 < neuralBpow binaryRadix e := neuralBpow.pos binaryRadix e
      have heq :
          aInt = bInt ↔ dyadicToReal a = dyadicToReal b := by
        calc
          aInt = bInt ↔ (aInt : ℝ) = (bInt : ℝ) := by
            simp
          _ ↔ (aInt : ℝ) * neuralBpow binaryRadix e = (bInt : ℝ) * neuralBpow binaryRadix e :=
            by
            have hbne : neuralBpow binaryRadix e ≠ 0 := ne_of_gt hbpos
            constructor
            · intro h
              simp [h]
            · intro h
              exact mul_right_cancel₀ hbne h
          _ ↔ dyadicToReal a = dyadicToReal b := by
            simp [ha, hb]
      -- `compare = .eq` ↔ equality.
      simpa [hcmp, compare_eq_iff_eq, heq]

/--
Dyadic comparison correctness (gt case).

This is the greater-than variant of `cmpDyadic_lt_iff`, phrased as `dyadicToReal b < dyadicToReal
  a`.
-/
theorem cmpDyadic_gt_iff (a b : Dyadic) :
    cmpDyadic a b = .gt ↔ dyadicToReal b < dyadicToReal a := by
  classical
  unfold cmpDyadic
  cases hzero : (a.mant == 0 && b.mant == 0) with
  | true =>
      have hab : (a.mant == 0) = true ∧ (b.mant == 0) = true := by
        simpa [Bool.and_eq_true] using (show (a.mant == 0 && b.mant == 0) = true from hzero)
      have ha0 : a.mant = 0 := (beq_iff_eq).1 hab.1
      have hb0 : b.mant = 0 := (beq_iff_eq).1 hab.2
      simp [dyadicToReal, ha0, hb0]
  | false =>
      let e : Int := if a.exp ≤ b.exp then a.exp else b.exp
      have heA : e ≤ a.exp := by
        by_cases hab : a.exp ≤ b.exp
        · simp [e, hab]
        · have : b.exp ≤ a.exp := le_of_lt (lt_of_not_ge (show ¬ b.exp ≥ a.exp by simpa using hab))
          simp [e, hab, this]
      have heB : e ≤ b.exp := by
        by_cases hab : a.exp ≤ b.exp
        · simp [e, hab]
        · simp [e, hab]
      let shA : Nat := Int.toNat (a.exp - e)
      let shB : Nat := Int.toNat (b.exp - e)
      let aNat : Nat := Nat.shiftLeft a.mant shA
      let bNat : Nat := Nat.shiftLeft b.mant shB
      let aInt : Int := signedMant a.sign aNat
      let bInt : Int := signedMant b.sign bNat
      have hcmp : cmpDyadic a b = Ord.compare aInt bInt := by
        simp (config := { zeta := true }) [cmpDyadic, hzero, e, shA, shB, aNat, bNat, aInt, bInt,
          signedMant]
      have ha : dyadicToReal a = (aInt : ℝ) * neuralBpow binaryRadix e := by
        simp [aInt, aNat, shA, e, dyadicToReal_eq_signedMant_shiftLeft_toExp (d := a) (e := e) heA]
      have hb : dyadicToReal b = (bInt : ℝ) * neuralBpow binaryRadix e := by
        simp [bInt, bNat, shB, e, dyadicToReal_eq_signedMant_shiftLeft_toExp (d := b) (e := e) heB]
      have hbpos : 0 < neuralBpow binaryRadix e := neuralBpow.pos binaryRadix e
      have hgt :
          bInt < aInt ↔ dyadicToReal b < dyadicToReal a := by
        constructor
        · intro hab
          have habR : (bInt : ℝ) < (aInt : ℝ) := by
            exact_mod_cast hab
          have habScaled :
              (bInt : ℝ) * neuralBpow binaryRadix e < (aInt : ℝ) * neuralBpow binaryRadix e :=
            mul_lt_mul_of_pos_right habR hbpos
          simpa [ha, hb] using habScaled
        · intro h
          have habScaled :
              (bInt : ℝ) * neuralBpow binaryRadix e < (aInt : ℝ) * neuralBpow binaryRadix e :=
                by
            simpa [ha, hb] using h
          have habR : (bInt : ℝ) < (aInt : ℝ) :=
            lt_of_mul_lt_mul_right habScaled (le_of_lt hbpos)
          exact (by exact_mod_cast habR)
      simpa [hcmp, compare_gt_iff_gt, hgt]

/--
Bridge for `IEEE32Exec.compare` on finite values.

When both operands decode to dyadics (`toDyadic? = some`), `compare` returns a result and it is
exactly `cmpDyadic` of those dyadics.
-/
theorem compare_eq_some_cmpDyadic_of_toDyadic? (x y : IEEE32Exec) {dx dy : Dyadic}
    (hx : toDyadic? x = some dx) (hy : toDyadic? y = some dy) :
    compare x y = some (cmpDyadic dx dy) := by
  unfold compare
  have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hx)
  have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx := hy)
  have hxInf : isInf x = false := isInf_eq_false_of_toDyadic?_some (hx := hx)
  have hyInf : isInf y = false := isInf_eq_false_of_toDyadic?_some (hx := hy)
  simp [hxNaN, hyNaN, hxInf, hyInf, hx, hy]

/--
`compare x y = .lt` if and only if `toReal x < toReal y` (finite path).

This is the user-facing ordering theorem that lets downstream reasoning switch between
`IEEE32Exec.compare` and `<` on reals.
-/
theorem compare_eq_some_lt_iff_toReal_lt (x y : IEEE32Exec) {dx dy : Dyadic}
    (hx : toDyadic? x = some dx) (hy : toDyadic? y = some dy) :
    compare x y = some .lt ↔ toReal x < toReal y := by
  have hcmp : compare x y = some (cmpDyadic dx dy) :=
    compare_eq_some_cmpDyadic_of_toDyadic? (x := x) (y := y) (hx := hx) (hy := hy)
  have hto : toReal x = dyadicToReal dx := by simp [toReal_eq, hx]
  have hto' : toReal y = dyadicToReal dy := by simp [toReal_eq, hy]
  -- Reduce to the dyadic comparison.
  simpa [hcmp, hto, hto'] using
    (cmpDyadic_lt_iff (a := dx) (b := dy))

/--
`compare x y = .eq` if and only if `toReal x = toReal y` (finite path).

Note: this equality is on the decoded real values; it ignores NaN payloads and
signed-zero distinctions (those are handled explicitly elsewhere).
-/
theorem compare_eq_some_eq_iff_toReal_eq (x y : IEEE32Exec) {dx dy : Dyadic}
    (hx : toDyadic? x = some dx) (hy : toDyadic? y = some dy) :
    compare x y = some .eq ↔ toReal x = toReal y := by
  have hcmp : compare x y = some (cmpDyadic dx dy) :=
    compare_eq_some_cmpDyadic_of_toDyadic? (x := x) (y := y) (hx := hx) (hy := hy)
  have hto : toReal x = dyadicToReal dx := by simp [toReal_eq, hx]
  have hto' : toReal y = dyadicToReal dy := by simp [toReal_eq, hy]
  simpa [hcmp, hto, hto'] using
    (cmpDyadic_eq_iff (a := dx) (b := dy))

/--
`compare x y = .gt` if and only if `toReal y < toReal x` (finite path).

This is the greater-than companion to `compare_eq_some_lt_iff_toReal_lt`.
-/
theorem compare_eq_some_gt_iff_toReal_gt (x y : IEEE32Exec) {dx dy : Dyadic}
    (hx : toDyadic? x = some dx) (hy : toDyadic? y = some dy) :
    compare x y = some .gt ↔ toReal y < toReal x := by
  have hcmp : compare x y = some (cmpDyadic dx dy) :=
    compare_eq_some_cmpDyadic_of_toDyadic? (x := x) (y := y) (hx := hx) (hy := hy)
  have hto : toReal x = dyadicToReal dx := by simp [toReal_eq, hx]
  have hto' : toReal y = dyadicToReal dy := by simp [toReal_eq, hy]
  simpa [hcmp, hto, hto'] using
    (cmpDyadic_gt_iff (a := dx) (b := dy))

/-- A bit pattern the encoding calls zero really does decode to the real number `0`.

Both signed zeros land here, which is why the statement is about `toReal` rather than about the
dyadic: the two dyadics differ in their sign bit but not in the value they denote. -/
lemma toReal_eq_zero_of_isZero (x : IEEE32Exec) {d : Dyadic}
    (hx : toDyadic? x = some d) (hz : isZero x = true) : toReal x = 0 := by
  -- Extract bitfield facts from `isZero`.
  unfold isZero at hz
  have hfields : (expField x == 0) = true ∧ (fracField x == 0) = true := by
    simpa [Bool.and_eq_true] using hz
  -- `toDyadic?` returns the canonical dyadic `0` in the `isZero` case.
  have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hx)
  have hxInf : isInf x = false := isInf_eq_false_of_toDyadic?_some (hx := hx)
  have hnaninf : (isNaN x || isInf x) = false := by simp [hxNaN, hxInf]
  have hdy :
      { sign := signBit x, mant := 0, exp := 0 } = d := by
    unfold toDyadic? at hx
    simp (config := { zeta := true }) [hnaninf, hfields.1, hfields.2] at hx
    simpa using hx
  -- Hence `toReal x = 0`.
  have hd : d = { sign := signBit x, mant := 0, exp := 0 } := by
    simpa using hdy.symm
  simp [toReal_eq, hx, hd, dyadicToReal, TorchLean.Floats.neuralBpow, binaryRadix,
    NeuralRadix.toReal]

/--
Bridge for `IEEE32Exec.minimum` on finite values: its real meaning is `min (toReal x) (toReal y)`.

This proof follows IEEE-754 style rules (including NaN propagation and signed-zero handling), but
the statement is on `toReal`, which erases the sign of zero.
-/
theorem toReal_minimum_eq_min (x y : IEEE32Exec) {dx dy : Dyadic}
    (hx : toDyadic? x = some dx) (hy : toDyadic? y = some dy) :
    toReal (minimum x y) = min (toReal x) (toReal y) := by
  classical
  have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hx)
  have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx := hy)
  have hchoose : chooseNaN2 x y = none := by
    simp [chooseNaN2, isSNaN, hxNaN, hyNaN]
  -- Reduce to dyadic `compare`.
  have hcmp : compare x y = some (cmpDyadic dx dy) :=
    compare_eq_some_cmpDyadic_of_toDyadic? (x := x) (y := y) (hx := hx) (hy := hy)
  cases hord : cmpDyadic dx dy with
  | lt =>
      have hcmp' : compare x y = some .lt := by simpa [hord] using hcmp
      have hlt : toReal x < toReal y :=
        (compare_eq_some_lt_iff_toReal_lt (x := x) (y := y) (hx := hx) (hy := hy)).1 hcmp'
      -- `minimum` returns `x`.
      have hmin : toReal (minimum x y) = toReal x := by
        simp [minimum, hchoose, hcmp']
      calc
        toReal (minimum x y) = toReal x := hmin
        _ = min (toReal x) (toReal y) := by
          simpa using (min_eq_left (le_of_lt hlt)).symm
  | gt =>
      have hcmp' : compare x y = some .gt := by simpa [hord] using hcmp
      have hlt : toReal y < toReal x :=
        (compare_eq_some_gt_iff_toReal_gt (x := x) (y := y) (hx := hx) (hy := hy)).1 hcmp'
      have hmin : toReal (minimum x y) = toReal y := by
        simp [minimum, hchoose, hcmp']
      calc
        toReal (minimum x y) = toReal y := hmin
        _ = min (toReal x) (toReal y) := by
          simpa using (min_eq_right (le_of_lt hlt)).symm
  | eq =>
      have hcmp' : compare x y = some .eq := by simpa [hord] using hcmp
      have heq : toReal x = toReal y :=
        (compare_eq_some_eq_iff_toReal_eq (x := x) (y := y) (hx := hx) (hy := hy)).1 hcmp'
      cases hzeros : (isZero x && isZero y) with
      | true =>
          have hxz : isZero x = true := by
            have : (isZero x = true) ∧ (isZero y = true) := by
              simpa [Bool.and_eq_true] using (show (isZero x && isZero y) = true from hzeros)
            exact this.1
          have hyz : isZero y = true := by
            have : (isZero x = true) ∧ (isZero y = true) := by
              simpa [Bool.and_eq_true] using (show (isZero x && isZero y) = true from hzeros)
            exact this.2
          have hx0 : toReal x = 0 := toReal_eq_zero_of_isZero (x := x) (hx := hx) (hz := hxz)
          have hy0 : toReal y = 0 := toReal_eq_zero_of_isZero (x := y) (hx := hy) (hz := hyz)
          have hmin : toReal (minimum x y) = 0 := by
            -- `minimum` returns a signed zero; `toReal` erases the sign.
              cases hs : (signBit x || signBit y) with
              | true =>
                  simp [minimum, hchoose, hcmp', hzeros, hs]
              | false =>
                  simp [minimum, hchoose, hcmp', hzeros, hs]
          calc
            toReal (minimum x y) = 0 := hmin
            _ = min (toReal x) (toReal y) := by
              simp [hx0, hy0]
      | false =>
          have hmin : toReal (minimum x y) = toReal x := by
            simp [minimum, hchoose, hcmp', hzeros]
          calc
            toReal (minimum x y) = toReal x := hmin
            _ = min (toReal x) (toReal y) := by
              simpa using (min_eq_left (le_of_eq heq)).symm

/--
Bridge for `IEEE32Exec.maximum` on finite values: its real meaning is `max (toReal x) (toReal y)`.

This is the companion of `toReal_minimum_eq_min`. As above, the conclusion is phrased in terms of
`toReal`, so signed zeros are identified.
-/
theorem toReal_maximum_eq_max (x y : IEEE32Exec) {dx dy : Dyadic}
    (hx : toDyadic? x = some dx) (hy : toDyadic? y = some dy) :
    toReal (maximum x y) = max (toReal x) (toReal y) := by
  classical
  have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hx)
  have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx := hy)
  have hchoose : chooseNaN2 x y = none := by
    simp [chooseNaN2, isSNaN, hxNaN, hyNaN]
  have hcmp : compare x y = some (cmpDyadic dx dy) :=
    compare_eq_some_cmpDyadic_of_toDyadic? (x := x) (y := y) (hx := hx) (hy := hy)
  cases hord : cmpDyadic dx dy with
  | lt =>
      have hcmp' : compare x y = some .lt := by simpa [hord] using hcmp
      have hlt : toReal x < toReal y :=
        (compare_eq_some_lt_iff_toReal_lt (x := x) (y := y) (hx := hx) (hy := hy)).1 hcmp'
      have hmax : toReal (maximum x y) = toReal y := by
        simp [maximum, hchoose, hcmp']
      calc
        toReal (maximum x y) = toReal y := hmax
        _ = max (toReal x) (toReal y) := by
          simpa using (max_eq_right (le_of_lt hlt)).symm
  | gt =>
      have hcmp' : compare x y = some .gt := by simpa [hord] using hcmp
      have hlt : toReal y < toReal x :=
        (compare_eq_some_gt_iff_toReal_gt (x := x) (y := y) (hx := hx) (hy := hy)).1 hcmp'
      have hmax : toReal (maximum x y) = toReal x := by
        simp [maximum, hchoose, hcmp']
      calc
        toReal (maximum x y) = toReal x := hmax
        _ = max (toReal x) (toReal y) := by
          simpa using (max_eq_left (le_of_lt hlt)).symm
  | eq =>
      have hcmp' : compare x y = some .eq := by simpa [hord] using hcmp
      have heq : toReal x = toReal y :=
        (compare_eq_some_eq_iff_toReal_eq (x := x) (y := y) (hx := hx) (hy := hy)).1 hcmp'
      cases hzeros : (isZero x && isZero y) with
      | true =>
          have hxz : isZero x = true := by
            have : (isZero x = true) ∧ (isZero y = true) := by
              simpa [Bool.and_eq_true] using (show (isZero x && isZero y) = true from hzeros)
            exact this.1
          have hyz : isZero y = true := by
            have : (isZero x = true) ∧ (isZero y = true) := by
              simpa [Bool.and_eq_true] using (show (isZero x && isZero y) = true from hzeros)
            exact this.2
          have hx0 : toReal x = 0 := toReal_eq_zero_of_isZero (x := x) (hx := hx) (hz := hxz)
          have hy0 : toReal y = 0 := toReal_eq_zero_of_isZero (x := y) (hx := hy) (hz := hyz)
          have hmax : toReal (maximum x y) = 0 := by
            -- `maximum` returns a signed zero; `toReal` erases the sign.
              cases hs : ((!signBit x) || (!signBit y)) with
              | true =>
                  simp [maximum, hchoose, hcmp', hzeros, hs]
              | false =>
                  simp [maximum, hchoose, hcmp', hzeros, hs]
          calc
            toReal (maximum x y) = 0 := hmax
            _ = max (toReal x) (toReal y) := by
              simp [hx0, hy0]
      | false =>
          have hmax : toReal (maximum x y) = toReal x := by
            simp [maximum, hchoose, hcmp', hzeros]
          calc
            toReal (maximum x y) = toReal x := hmax
            _ = max (toReal x) (toReal y) := by
              simpa using (max_eq_left (le_of_eq heq.symm)).symm
end IEEE32Exec

end TorchLean.Floats.IEEE754

