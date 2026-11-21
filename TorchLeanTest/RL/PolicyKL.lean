/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.PolicyGradient

/-!
# Policy KL divergences against mpmath

`TorchLean.RL.PolicyGradient.kl` and `klFromLogits` use the tilted form of Amos (2026, Section 4.1)
with `expm1x`, which does not subtract the nearly equal logarithms of the textbook sum. The expected
values were computed independently with 100-digit mpmath from the exact dyadic inputs below. The
checks cover these inputs only. `klFromLogits`'s docstring states the logit range and two limits
of the rounding, which `checkNormalizerAndLimits` checks.
-/

@[expose] public section

namespace TorchLean.Test.PolicyKL

open FloatLib.Floats
open TorchLean.RL.PolicyGradient

/-- Fail with `label` unless `condition` holds. -/
def expect (label : String) (condition : Bool) : IO Unit := do
  unless condition do
    throw <| IO.userError s!"policy KL: {label}"

/-- `|value - expected| / expected`, for a positive `expected`. -/
def relativeError (value expected : Float) : Float :=
  Float.abs (value - expected) / expected

/-- `expm1x` rounds `exp x - 1 - x` once on binary64 and on the native `Float`. Each expected word
is the nearest-even rounding of the real value from 100-digit mpmath. At `2 ^ (-30)` the default
`expm1 x - x` would cancel all but a few bits. -/
def checkExpm1x : IO Unit := do
  let w64 (x : Binary64) : Nat := ExecFloat.Binary.toNatBits x
  let tiny : Binary64 := 9.31322574615478515625e-10
  expect "binary64 expm1x at 2^-30, -2^-30 and 1"
    (w64 (Context.expm1x tiny) == 0x3c20000000155555 &&
      w64 (Context.expm1x (-tiny)) == 0x3c1fffffffd55555 &&
      w64 (Context.expm1x (1 : Binary64)) == 0x3fe6fc2a2c515da5)
  let tinyFloat : Float := 9.31322574615478515625e-10
  expect "Float expm1x at 2^-30, -2^-30 and 1"
    ((Context.expm1x tinyFloat).toBits == 0x3c20000000155555 &&
      (Context.expm1x (-tinyFloat)).toBits == 0x3c1fffffffd55555 &&
      (Context.expm1x (1 : Float)).toBits == 0x3fe6fc2a2c515da5)

/-- `klFromLogits` of two logit vectors `2 ^ (-13)` apart, a divergence near `5.6e-9`. The old
logits are `[0.5, -1.25, 2, 0]` and the new ones add `2 ^ (-13) * [1, -2, 0.5, 3]`, all exact in
binary32. The binary32 value is within one unit in the last place of the divergence, `2 ^ (-51)`
at this magnitude. -/
def checkKlFromLogits : IO Unit := do
  let expected : Float := 5.62472991606012020293808598126e-9
  let old : Tensor Float [4] := [0.5, -1.25, 2, 0]
  let new : Tensor Float [4] :=
    [0.5001220703125, -1.250244140625, 2.00006103515625, 0.0003662109375]
  let value := klFromLogits old new
  expect s!"Float klFromLogits {value} is within 1e-12 relative of {expected}"
    (relativeError value expected < 1e-12)
  let old32 : Tensor Float32 [4] := [0.5, -1.25, 2, 0]
  let new32 : Tensor Float32 [4] :=
    [0.5001220703125, -1.250244140625, 2.00006103515625, 0.0003662109375]
  let value32 := (klFromLogits old32 new32).toFloat
  expect s!"Float32 klFromLogits {value32} is within 2 ^ (-51) of {expected}"
    (Float.abs (value32 - expected) < 4.44089209850062616169452667236e-16)

/-- `kl` of `p = [1/4, 1/4, 1/2]` and `q = p + 2 ^ (-14) * [1, 1, -2]`, a divergence near `3e-8`,
on `Float32`. The binary32 logarithms of `p` and `q` round by about `1e-7`, a relative error near
`4e-4` in the log-ratios, so the bound is `1e-2`. -/
def checkKl : IO Unit := do
  let expected : Float := 2.980232327587376749316627899e-8
  let p : Tensor Float32 [3] := [0.25, 0.25, 0.5]
  let q : Tensor Float32 [3] := [0.25006103515625, 0.25006103515625, 0.4998779296875]
  let value := (kl p q).toFloat
  expect s!"Float32 kl {value} is within 1e-2 relative of {expected}"
    (relativeError value expected < 1e-2)

/-- Policies far apart. Binary32 logits `[0, 0]` and `[-100, 100]` have the divergence
`100 - log 2`, about `99.307`; centered at the mean log-ratio, `expm1 100` overflowed to `∞`. The
logits `[0, -95]` and `[0, 0]` put the largest log-ratio, `95`, on the old probability `exp (-95)`,
a binary32 subnormal far below the rounding unit, which enters as its logarithm `-95`. The sum
`Σ p * expm1 d` rounds to `-1`, and `logWeightedMass` takes its log-sum-exp branch; the divergence
is about `log 2`. -/
def checkFarApart : IO Unit := do
  let expectedFar : Float := 99.3068528194400546905827678785
  let value := (klFromLogits ([0, 0] : Tensor Float32 [2]) [-100, 100]).toFloat
  expect s!"Float32 klFromLogits [0, 0] [-100, 100] {value} is within 1e-6 of {expectedFar}"
    (relativeError value expectedFar < 1e-6)
  let expectedTiny : Float := 0.693147180559945309417232121458
  let valueTiny := (klFromLogits ([0, -95] : Tensor Float32 [2]) [0, 0]).toFloat
  expect s!"Float32 klFromLogits [0, -95] [0, 0] {valueTiny} is within 1e-6 of {expectedTiny}"
    (relativeError valueTiny expectedTiny < 1e-6)

/-- The binary32 cases of the release reviews of b67c4f73, with `A = 2^127` and `M` the largest
finite binary32 value. `klFromLogits` centers a quarter of each logit vector at its own largest
entry before the difference and takes the old weights from the log-softmax:
- `[A, A]` against `[-A, -A]`: both policies are uniform and the divergence is `0`; the
  difference of the uncentered logits overflowed and gave NaN;
- `[1, 1]` against `[0, 2^(-26)]`: the divergence `log cosh (2^(-27))` rounds to `2^(-55)`; both
  uncentered differences rounded to `-1` and gave `0`;
- `[0, -110]` against `[0, 0]`: the old probability `exp (-110)` underflows to `0`, and the
  divergence is `log 2` within `1e-6`; the rounded probability gave `0`;
- `[0, 0]` against `[-A/4, A/4]`: the divergence `A/4 - log 2` rounds to `A/4`;
- `[0, 0]` against `[-A, A]`: the divergence `A - log 2` rounds to `A`; the centered new logit
  `-2A` overflowed and gave `+∞`;
- `[0, 0]` against `[-M, M]`: the divergence `M - log 2` rounds to `M`, while `d` in
  `klTilted` has the entry `-2M`, which overflows to `-∞`;
- `[-A, A]` against `[0, 0]`: the old logits are more than `M` apart, their log-softmax has the
  entry `-∞`, and the result is NaN, outside the range that `klFromLogits` states. -/
def checkReleaseCases : IO Unit := do
  let a : Float32 := 1.7014118346046923e38
  let minusA : Float32 := -a
  let quarter : Float32 := 4.2535295865117308e37
  let minusQuarter : Float32 := -quarter
  let allA : Tensor Float32 [2] := [a, a]
  let allMinusA : Tensor Float32 [2] := [minusA, minusA]
  let equal := (klFromLogits allA allMinusA).toFloat
  expect s!"Float32 klFromLogits [A, A] [-A, -A] {equal} is 0" (equal == 0)
  let small : Float32 := 1.490116119384765625e-8
  let expectedSmall : Float := 2.77555756156289132538001324506e-17
  let nearOne : Tensor Float32 [2] := [0, small]
  let valueSmall := (klFromLogits ([1, 1] : Tensor Float32 [2]) nearOne).toFloat
  expect s!"Float32 klFromLogits [1, 1] [0, 2^-26] {valueSmall} is within 1e-6 of {expectedSmall}"
    (relativeError valueSmall expectedSmall < 1e-6)
  let expectedUnder : Float := 0.693147180559945309417232121458
  let valueUnder := (klFromLogits ([0, -110] : Tensor Float32 [2]) [0, 0]).toFloat
  expect s!"Float32 klFromLogits [0, -110] [0, 0] {valueUnder} is within 1e-6 of {expectedUnder}"
    (relativeError valueUnder expectedUnder < 1e-6)
  let quarters : Tensor Float32 [2] := [minusQuarter, quarter]
  let valueQuarter := klFromLogits ([0, 0] : Tensor Float32 [2]) quarters
  expect s!"Float32 klFromLogits [0, 0] [-A/4, A/4] {valueQuarter.toFloat} is A/4"
    (valueQuarter == quarter)
  let far : Tensor Float32 [2] := [minusA, a]
  let valueFar := klFromLogits ([0, 0] : Tensor Float32 [2]) far
  expect s!"Float32 klFromLogits [0, 0] [-A, A] {valueFar.toFloat} is A" (valueFar == a)
  let m : Float32 := 3.4028234663852886e38
  let minusM : Float32 := -m
  let widest : Tensor Float32 [2] := [minusM, m]
  let valueWidest := klFromLogits ([0, 0] : Tensor Float32 [2]) widest
  expect s!"Float32 klFromLogits [0, 0] [-M, M] {valueWidest.toFloat} is M" (valueWidest == m)
  let valueOldFar := (klFromLogits far ([0, 0] : Tensor Float32 [2])).toFloat
  expect s!"Float32 klFromLogits [-A, A] [0, 0] {valueOldFar} is NaN" valueOldFar.isNaN

/-- The binary32 cases of the release review of 0a6008a7, with `A = 2^127`:
- `[0, -2^27]` and `[0, -A]` against `[0, 0]`: the divergence is `log 2`. The log-sum-exp branch
  of `logWeightedMass` has the parts `-2^27` and `log 2`, whose rounded sum is `-2^27`, and
  `klTilted` gave `1` when it subtracted that sum;
- the two limits that `klFromLogits`'s docstring names. `[0, 1]` against `[2^(-26), 1]` gives `0`,
  where the divergence is about `2.18e-17`, since `2^(-28) - 1/4` rounds to `-1/4`.
  Against `[0, 5]`, where the divergence is about `5.0067`, `[0, -2^24]` gives `log (1 + e^4)`,
  since `-2^24 - 5` rounds to `-2^24 - 4`, and `[0, -2^100]` gives `log 2`, since `-2^100 - 5`
  rounds to `-2^100`. -/
def checkNormalizerAndLimits : IO Unit := do
  let expected : Float := 0.693147180559945309417232121458
  let zeros : Tensor Float32 [2] := [0, 0]
  let gap : Float32 := -134217728
  let gapOld : Tensor Float32 [2] := [0, gap]
  let valueGap := (klFromLogits gapOld zeros).toFloat
  expect s!"Float32 klFromLogits [0, -2^27] [0, 0] {valueGap} is within 1e-6 of {expected}"
    (relativeError valueGap expected < 1e-6)
  let minusA : Float32 := -1.7014118346046923e38
  let farOld : Tensor Float32 [2] := [0, minusA]
  let valueFar := (klFromLogits farOld zeros).toFloat
  expect s!"Float32 klFromLogits [0, -A] [0, 0] {valueFar} is within 1e-6 of {expected}"
    (relativeError valueFar expected < 1e-6)
  let small : Float32 := 1.490116119384765625e-8
  let nearOld : Tensor Float32 [2] := [0, 1]
  let nearNew : Tensor Float32 [2] := [small, 1]
  let valueNear := (klFromLogits nearOld nearNew).toFloat
  expect s!"Float32 klFromLogits [0, 1] [2^-26, 1] {valueNear} is 0" (valueNear == 0)
  let five : Float32 := 5
  let wideNew : Tensor Float32 [2] := [0, five]
  let expectedFour : Float := 4.01814992791780939
  let gapRound : Float32 := -16777216
  let roundOld : Tensor Float32 [2] := [0, gapRound]
  let valueRound := (klFromLogits roundOld wideNew).toFloat
  expect s!"Float32 klFromLogits [0, -2^24] [0, 5] {valueRound} is within 1e-6 of {expectedFour}"
    (relativeError valueRound expectedFour < 1e-6)
  let wide : Float32 := -1.2676506002282294e30
  let wideOld : Tensor Float32 [2] := [0, wide]
  let valueWide := (klFromLogits wideOld wideNew).toFloat
  expect s!"Float32 klFromLogits [0, -2^100] [0, 5] {valueWide} is within 1e-6 of {expected}"
    (relativeError valueWide expected < 1e-6)

/-- The binary32 case of the release review of 91de1165, with `E = 2^32`: `[0, -384, -2E]` against
`[0, 2E, E]`, where the divergence `2E - 384 e^(-384)` rounds to `2E`. In `klTilted`, `d` is
`[-3E, -E + 512, 0]`, and rounding `d + l` absorbs the old log-probability `-384`, so the largest
entry `t` of `d + l` is `-E`. The term's exponent formed as `(d - t) + l - r` was `128`, and the
result `+∞`. Now `e = (d - t) + l` is `[-2E, 128, -E]`, and the exponents subtract its largest
entry `128`. -/
def checkTermExponent : IO Unit := do
  let e : Float32 := 4294967296
  let old : Tensor Float32 [3] := [0, -384, -(e + e)]
  let new : Tensor Float32 [3] := [0, e + e, e]
  let expected : Float := 8589934592
  let value := (klFromLogits old new).toFloat
  expect s!"Float32 klFromLogits [0, -384, -2^33] [0, 2^33, 2^32] {value} is within 1e-6 of 2^33"
    (relativeError value expected < 1e-6)

/-- The binary32 case where the exponent `(d + l - t) - r` failed: `[0.5625, 0, -2^26]` against
`[0.5625, 2.25, -2^25]`. In `klTilted`, `d` is `[-2^25 + 2, -2^25 + 4, 0]`, and rounding `d + l`
puts its first two entries at `t = -2^25 + 2`, though `d - t` is `2` at the second. That exponent
lost the old log-probability `-log (1 + e^(9/16))`, about `-1.016`, so the term's exponential was
below its linear part, and the result about `-0.214`. With `e = (d - t) + l`, the result is the
value of the form at the shift `[0, 2, 2^25 - 2]` the quarters round to, about `0.4737`, where
the divergence is about `0.5896`. -/
def checkExponentOrder : IO Unit := do
  let big : Float32 := 33554432
  let old : Tensor Float32 [3] := [0.5625, 0, -(big + big)]
  let new : Tensor Float32 [3] := [0.5625, 2.25, -big]
  let expected : Float := 0.473734346196317368355019185014
  let value := (klFromLogits old new).toFloat
  expect (s!"Float32 klFromLogits [0.5625, 0, -2^26] [0.5625, 2.25, -2^25] {value} is within " ++
    s!"1e-6 of {expected}") (relativeError value expected < 1e-6)

/-- Binary16 `kl [1, 0] [0, 1]` with the default `epsilon`, binary16's nearest value to `1e-6`,
`17 * 2 ^ (-24)`. The guarded old probability `epsilon` is far below binary16's rounding unit, and
`exp` of its log-ratio, about `13.8`, overflows binary16. The divergence of the guarded vectors is
about `13.8023`, the word `0x4ae7`; the check allows the words `0x4ae3` to `0x4aea`, within `0.2%`.
-/
def checkBinary16DefaultEpsilon : IO Unit := do
  let value : Binary16 := kl ([1, 0] : Tensor Binary16 [2]) [0, 1]
  let word := ExecFloat.Binary.toNatBits value
  expect s!"binary16 kl [1, 0] [0, 1] is the word {word}, within 0x4ae3 to 0x4aea"
    (0x4ae3 ≤ word && word ≤ 0x4aea)

/-- Run the policy KL checks. -/
def run : IO Unit := do
  checkExpm1x
  checkKlFromLogits
  checkKl
  checkFarApart
  checkReleaseCases
  checkNormalizerAndLimits
  checkTermExponent
  checkExponentOrder
  checkBinary16DefaultEpsilon
  IO.println "  policy KL divergences: passed"

end TorchLean.Test.PolicyKL
