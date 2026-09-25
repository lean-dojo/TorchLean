/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Extras.BoundOpsIEEE32Exec
public import NN.Floats.Interval.Comparison

/-!
# Finite binary32 interval regressions

These checks exercise the actual nonlinear backend, including exceptional endpoints, overflow,
subnormal values, zero-crossing square roots, and interior trigonometric extrema. All numerical
comparisons decode endpoints exactly as rationals. The theorem below separately checks that the
backend's containment interface applies at the irrational input `π / 2`.
-/

open FloatLib.Floats.ExecFloat (Binary)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)
open FloatLib.Numerics (Interval RationalInterval)
open NN.MLTheory.CROWN
open TorchLean.Floats.Interval.Comparison (interval32ToRat?)

namespace NN.Tests.Floats.IEEE32IntervalBounds

/-- A successful sine transfer includes an interior maximum whenever its input includes `π / 2`. -/
public theorem sinBounds_contains_pi_div_two {lo hi outLo outHi : Binary 8 23}
    (h : NonlinearBoundOps.sinBounds lo hi = some (outLo, outHi))
    (hx : (⟨lo, hi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? (Real.pi / 2)) :
    (⟨outLo, outHi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? 1 := by
  simpa only [Real.sin_pi_div_two] using IEEE32ExecBounds.sinBounds_containsReal h hx

private abbrev Endpoint := Binary 8 23

private def binary32 (q : ℚ) : Endpoint :=
  Binary.ofModel (Model.roundRatQ FloatFormat.binary32 q)

private def decodedBounds? (result : Option (Endpoint × Endpoint)) :
    Option RationalInterval :=
  result.bind fun (lo, hi) => interval32ToRat? ⟨lo, hi⟩

private def satisfies (result : Option (Endpoint × Endpoint))
    (p : RationalInterval → Bool) : Bool :=
  (decodedBounds? result).any p

private def encloses (result : Option (Endpoint × Endpoint)) (lo hi : ℚ) : Bool :=
  satisfies result fun I => decide (I.lo ≤ lo ∧ hi ≤ I.hi)

private def check (label : String) (passed : Bool) : IO Unit := do
  unless passed do throw <| IO.userError s!"IEEE32 interval regression failed: {label}"

/-- Run exact-rational boundary and enclosure checks in the native Float suite. -/
public def run : IO Unit := do
  let zero := binary32 0
  let one := binary32 1
  let two := binary32 2
  let three := binary32 3
  let four := binary32 4
  let negOne := binary32 (-1)
  let negTwo := binary32 (-2)
  let negZero : Endpoint := Binary.ofBits32 0x80000000
  let nan : Endpoint := Binary.ofBits32 0x7fc00000
  let posInf : Endpoint := Binary.ofBits32 0x7f800000
  let negInf : Endpoint := Binary.ofBits32 0xff800000
  let maxFinite : Endpoint := Binary.ofBits32 0x7f7fffff
  let negMaxFinite : Endpoint := Binary.ofBits32 0xff7fffff
  let minSubnormal : Endpoint := Binary.ofBits32 1
  let invalid := #[(one, zero), (nan, one), (zero, nan), (negInf, one),
    (zero, posInf), (posInf, posInf), (negInf, negInf)]
  let unary : Array (String × (Endpoint → Endpoint → Option (Endpoint × Endpoint))) := #[
    ("exp", NonlinearBoundOps.expBounds), ("log", NonlinearBoundOps.logBounds),
    ("sqrt", NonlinearBoundOps.sqrtBounds), ("sigmoid", NonlinearBoundOps.sigmoidBounds),
    ("tanh", NonlinearBoundOps.tanhBounds), ("sin", NonlinearBoundOps.sinBounds),
    ("cos", NonlinearBoundOps.cosBounds)]
  for (name, op) in unary do
    for (lo, hi) in invalid do
      check s!"{name}: reject invalid input" (op lo hi).isNone
  for (lo, hi) in invalid do
    check "div: reject invalid numerator" (NonlinearBoundOps.divBounds lo hi one two).isNone
    check "div: reject invalid denominator" (NonlinearBoundOps.divBounds one two lo hi).isNone
  for (lo, hi) in #[(negOne, one), (zero, one), (negOne, zero), (negZero, zero)] do
    check "div: reject zero in denominator" (NonlinearBoundOps.divBounds one two lo hi).isNone
  check "div: positive denominator"
    (encloses (NonlinearBoundOps.divBounds one two three four) (1 / 4) (2 / 3))
  check "div: mixed-sign numerator"
    (encloses (NonlinearBoundOps.divBounds negTwo one two four) (-1) (1 / 2))
  check "div: negative denominator"
    (encloses (NonlinearBoundOps.divBounds one two (binary32 (-4)) (binary32 (-3)))
      (-2 / 3) (-1 / 4))
  check "div: two negative intervals"
    (encloses (NonlinearBoundOps.divBounds negTwo negOne (binary32 (-4)) (binary32 (-3)))
      (1 / 4) (2 / 3))
  check "div: finite input overflow"
    (NonlinearBoundOps.divBounds maxFinite maxFinite minSubnormal minSubnormal).isNone
  check "exp: nontrivial interval"
    (satisfies (NonlinearBoundOps.expBounds negOne one) fun I =>
      decide (1 / 3 < I.lo ∧ I.lo ≤ 1 ∧ 1 ≤ I.hi ∧ I.hi < 3))
  check "exp: directed-rounding overflow"
    (NonlinearBoundOps.expBounds (binary32 100) (binary32 100)).isNone
  check "exp: extreme positive input returns promptly"
    (NonlinearBoundOps.expBounds maxFinite maxFinite).isNone
  check "exp: extreme negative input underflows outward"
    (satisfies (NonlinearBoundOps.expBounds negMaxFinite negMaxFinite) fun I =>
      decide (I.lo = 0 ∧ I.hi = (1 / (2 : ℚ) ^ 149)))
  check "exp: interval spanning the negative clamp"
    (satisfies (NonlinearBoundOps.expBounds negMaxFinite one) fun I =>
      decide (I.lo = 0 ∧ 2 < I.hi ∧ I.hi < 3))
  check "log: nontrivial interval"
    (satisfies (NonlinearBoundOps.logBounds one four) fun I =>
      decide (I.lo = 0 ∧ 1 < I.hi ∧ I.hi < 3 / 2))
  for (lo, hi) in #[(zero, one), (negOne, one), (negTwo, negOne)] do
    check "log: reject nonpositive domain" (NonlinearBoundOps.logBounds lo hi).isNone
  check "log: subnormal input"
    (satisfies (NonlinearBoundOps.logBounds minSubnormal minSubnormal) fun I =>
      decide (-104 < I.lo ∧ I.lo ≤ I.hi ∧ I.hi < -103))
  check "log: largest finite input"
    (satisfies (NonlinearBoundOps.logBounds maxFinite maxFinite) fun I =>
      decide (88 < I.lo ∧ I.lo ≤ I.hi ∧ I.hi < 89))
  check "sqrt: negative interval fails" (NonlinearBoundOps.sqrtBounds negTwo negOne).isNone
  check "sqrt: zero crossing is preserved"
    (satisfies (NonlinearBoundOps.sqrtBounds negOne four) fun I =>
      decide (I.lo = 0 ∧ I.hi = 2))
  check "sqrt: signed zero"
    (satisfies (NonlinearBoundOps.sqrtBounds negZero zero) fun I =>
      decide (I.lo = 0 ∧ I.hi = 0))
  check "sqrt: irrational endpoints rounded outward"
    (satisfies (NonlinearBoundOps.sqrtBounds two three) fun I =>
      decide (0 < I.lo ∧ I.lo ^ 2 ≤ 2 ∧ 3 ≤ I.hi ^ 2 ∧ I.hi < 7 / 4))
  check "sqrt: subnormal input retains precision"
    (satisfies (NonlinearBoundOps.sqrtBounds minSubnormal minSubnormal) fun I =>
      decide (0 < I.lo ∧ I.lo ^ 2 ≤ 1 / (2 : ℚ) ^ 149 ∧
        1 / (2 : ℚ) ^ 149 ≤ I.hi ^ 2 ∧ I.hi < 2 * I.lo))
  check "sin: interior maximum"
    (satisfies (NonlinearBoundOps.sinBounds one two) fun I =>
      decide (0 < I.lo ∧ I.hi = 1))
  check "cos: interior maximum"
    (satisfies (NonlinearBoundOps.cosBounds negOne one) fun I =>
      decide (I.lo = 0 ∧ I.hi = 1))
  for op in #[NonlinearBoundOps.sinBounds, NonlinearBoundOps.cosBounds] do
    check "trigonometric: wide interval covers both extrema"
      (satisfies (op negMaxFinite maxFinite) fun I => decide (I.lo = -1 ∧ I.hi = 1))
  check "sigmoid: finite fallback"
    (satisfies (NonlinearBoundOps.sigmoidBounds negMaxFinite maxFinite) fun I =>
      decide (I.lo = 0 ∧ I.hi = 1))
  check "tanh: finite fallback"
    (satisfies (NonlinearBoundOps.tanhBounds negMaxFinite maxFinite) fun I =>
      decide (I.lo = -1 ∧ I.hi = 1))
  IO.println "IEEE32 interval regressions passed."

end NN.Tests.Floats.IEEE32IntervalBounds
