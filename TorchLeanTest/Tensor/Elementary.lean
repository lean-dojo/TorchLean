/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Operations

/-!
# Elementary functions against mpmath, CPython and numpy

The configured binary16, binary32 and binary64 formats round `tan`, `exp2`, `log2`, `log10`,
`asinh`, `acosh` and `atanh` to the words of 400-digit mpmath, and `Float` and `Float32` keep the
signed zeros and exact powers of their C library. The complex `sqrt` and `acosh` agree to the bit
with C99's `csqrt` and `cacosh`, as CPython's `cmath` computes them, on both sides of the branch
cut and at infinities. The binary32 GELU constant `√(2 / π)` is numpy's word.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean.Test.Elementary

open FloatLib.Floats

/-- Fail with `label` unless `condition` holds. -/
def expect (label : String) (condition : Bool) : IO Unit := do
  unless condition do
    throw <| IO.userError s!"elementary function: {label}"

/-- Configured formats round `tan`, `exp2`, `log2`, `log10`, `asinh`, `acosh` and `atanh` once,
with the special values of C99 Annex F, while the host types call their C library. Each expected
word is the nearest-even rounding of the real value, computed independently with 400-digit mpmath:
binary16 `asinh 300` and `acosh 300` are `0x4666` (about 6.398) although `300 * 300` overflows
binary16, binary32 `asinh 1e-8` is its input, binary16 `tan 355` is the subnormal `0x1fa`,
`exp2 (-25)` is half the least subnormal and ties to `+0`, and exponent width `2` cannot hold
`tan 1.5` or `log2` of the least subnormal. -/
def checkElementary : IO Unit := do
  let w16 (x : Binary16) : Nat := ExecFloat.Binary.toNatBits x
  let w32 (x : Binary32) : Nat := ExecFloat.Binary.toNatBits x
  let w64 (x : Binary64) : Nat := ExecFloat.Binary.toNatBits x
  let inf : Binary16 := 1 / 0
  let negZero : Binary16 := -0
  expect "binary16 asinh 300 and acosh 300 are 6.398 rounded, not an overflow of 300 * 300"
    (w16 (Context.asinh 300) == 0x4666 && w16 (Context.asinh (-300)) == 0xc666 &&
      w16 (Context.acosh 300) == 0x4666)
  expect "binary16 asinh and acosh of the largest finite value are finite"
    (w16 (Context.asinh 65504) == 0x49e4 && w16 (Context.acosh 65504) == 0x49e4)
  expect "binary32 and binary64 asinh 300 and acosh 300 are correctly rounded"
    (w32 (Context.asinh 300) == 0x40ccb3ac && w32 (Context.acosh 300) == 0x40ccb3a0 &&
      w64 (Context.asinh 300) == 0x4019967574a9860c &&
      w64 (Context.acosh 300) == 0x40199673ffd5ce52)
  expect "binary32 asinh 1e-8 is its input, not zero"
    (w32 (Context.asinh 1e-8) == 0x322bcc77)
  expect "asinh, atanh and tan keep a negative zero"
    ([Context.asinh negZero, Context.atanh negZero, Context.tan negZero].all (w16 · == 0x8000) &&
      [Context.asinh (-0 : Binary32), Context.atanh (-0), Context.tan (-0)].all
        (w32 · == 0x80000000) &&
      [Context.asinh (-0 : Binary64), Context.atanh (-0), Context.tan (-0)].all
        (w64 · == 0x8000000000000000))
  expect "acosh 1 is +0 and acosh 0.5 is a NaN"
    (w16 (Context.acosh 1) == 0 && w32 (Context.acosh 1) == 0 && w64 (Context.acosh 1) == 0 &&
      ExecFloat.Binary.isNaN (Context.acosh (0.5 : Binary16)) &&
      ExecFloat.Binary.isNaN (Context.acosh (0.5 : Binary32)) &&
      ExecFloat.Binary.isNaN (Context.acosh (0.5 : Binary64)))
  expect "exp2 3 is 8, log10 1000 is 3, log2 2^15 and 2^29 are 15 and 29"
    (w16 (Context.exp2 3) == 0x4800 && w32 (Context.exp2 3) == 0x41000000 &&
      w64 (Context.exp2 3) == 0x4020000000000000 && w16 (Context.log10 1000) == 0x4200 &&
      w32 (Context.log10 1000) == 0x40400000 && w64 (Context.log10 1000) == 0x4008000000000000 &&
      w64 (Context.log10 1e22) == 0x4036000000000000 && w16 (Context.log2 32768) == 0x4b80 &&
      w32 (Context.log2 536870912) == 0x41e80000 &&
      w64 (Context.log2 536870912) == 0x403d000000000000)
  expect "binary64 tan 1, atanh 0.5, log2 3 and exp2 0.5 are correctly rounded"
    (w64 (Context.tan 1) == 0x3ff8eb245cbee3a6 && w64 (Context.atanh 0.5) == 0x3fe193ea7aad030b &&
      w64 (Context.log2 3) == 0x3ff95c01a39fbd68 && w64 (Context.exp2 0.5) == 0x3ff6a09e667f3bcd)
  expect "tan reduces large arguments: binary16 355, binary64 1e22 and the largest binary32"
    (w16 (Context.tan 355) == 0x1fa && w64 (Context.tan 1e22) == 0xbffa0f79c1b6b257 &&
      w32 (Context.tan (ExecFloat.Binary.ofNatBits 0x7f7fffff)) == 0xbf1c9eca)
  expect "exp2 overflows at 16 in binary16, is finite below, and ties to +0 at half a subnormal"
    (w16 (Context.exp2 16) == 0x7c00 && w16 (Context.exp2 15.9921875) == 0x7bf5 &&
      w16 (Context.exp2 (-25)) == 0 && w16 (Context.exp2 (-24.5)) == 1 &&
      w16 (Context.exp2 (-24)) == 1 && w64 (Context.exp2 (-1075)) == 0 &&
      w64 (Context.exp2 (-1074)) == 1)
  expect "binary16 infinities, poles and exact zeros"
    (w16 (Context.asinh inf) == 0x7c00 && w16 (Context.asinh (-inf)) == 0xfc00 &&
      w16 (Context.acosh inf) == 0x7c00 && w16 (Context.exp2 inf) == 0x7c00 &&
      w16 (Context.exp2 (-inf)) == 0 && w16 (Context.log2 inf) == 0x7c00 &&
      w16 (Context.log10 0) == 0xfc00 && w16 (Context.log2 negZero) == 0xfc00 &&
      w16 (Context.atanh 1) == 0x7c00 && w16 (Context.atanh (-1)) == 0xfc00 &&
      w16 (Context.log2 1) == 0 && w16 (Context.log10 1) == 0)
  expect "binary16 inputs outside the domains give NaNs"
    ([Context.tan inf, Context.tan (-inf), Context.acosh (-inf), Context.atanh 2,
      Context.atanh inf, Context.log2 (-1), Context.log10 (-inf)].all ExecFloat.Binary.isNaN)
  let signaling : Binary16 := ExecFloat.Binary.ofNatBits 0x7d00
  expect "a signaling NaN comes back quiet with its payload"
    ([Context.tan signaling, Context.exp2 signaling, Context.log2 signaling,
      Context.log10 signaling, Context.asinh signaling, Context.acosh signaling,
      Context.atanh signaling].all (w16 · == 0x7f00))
  let narrow (x : ExecFloat.Binary 2 10) : Nat := ExecFloat.Binary.toNatBits x
  let wide (x : ExecFloat.Binary 2 20) : Nat := ExecFloat.Binary.toNatBits x
  expect "exponent width two overflows tan 1.5, log2 2^-10 and atanh (1 - 2^-20)"
    (narrow (Context.tan 1.5) == 0xc00 &&
      narrow (Context.log2 (ExecFloat.Binary.ofNatBits 1)) == 0x1c00 &&
      wide (Context.atanh (ExecFloat.Binary.ofNatBits 0xfffff)) == 0x300000)
  let biased (x : ExecFloat.Binary 2 1 (bias := 2)) : Nat := ExecFloat.Binary.toNatBits x
  expect "exp2 and log2 handle a custom bias that cannot represent two"
    (biased (Context.exp2 (-1)) == 0x2 && biased (Context.log2 0.5) == 0xc &&
      biased (Context.log2 1.5) == 0x2)
  expect "Float and Float32 keep a negative zero in asinh, atanh and tan; acosh 1 is +0"
    ([Context.asinh (-0.0 : Float), Context.atanh (-0.0), Context.tan (-0.0)].all
        (·.toBits == 0x8000000000000000) &&
      [Context.asinh (-0.0 : Float32), Context.atanh (-0.0), Context.tan (-0.0)].all
        (·.toBits == 0x80000000) &&
      (Context.acosh (1 : Float)).toBits == 0 && (Context.acosh (1 : Float32)).toBits == 0 &&
      (Context.acosh (0.5 : Float)).isNaN && (Context.acosh (0.5 : Float32)).isNaN)
  expect "Float and Float32 exp2 3 is 8 and log2 2^29 is 29"
    ((Context.exp2 (3 : Float)).toBits == 0x4020000000000000 &&
      (Context.exp2 (3 : Float32)).toBits == 0x41000000 &&
      (Context.log2 (536870912 : Float)).toBits == 0x403d000000000000 &&
      (Context.log2 (536870912 : Float32)).toBits == 0x41e80000)

/-- `√(2 / π)` computed in binary32, as numpy computes it, is the word `0x3f4c4229`, one below the
nearest `0x3f4c422a`, in the configured format and in `Float32`. -/
def checkGeluBeta : IO Unit := do
  let configured : Binary32 := Activation.geluBeta
  let host : Float32 := Activation.geluBeta
  expect "binary32 and Float32 GELU beta are numpy's word"
    (ExecFloat.Binary.toNatBits configured == 0x3f4c4229 && host.toBits == 0x3f4c4229)

/-- The principal complex `acosh` and `sqrt` agree with C99's `cacosh` and `csqrt` (CPython's
`cmath`) to the bit at these points: the sign of a zero imaginary part selects the side of the cut,
also at `1 - 0i` and `0 - 0i`, and the real part of `acosh` is `+0` on `(-1, 1)`. The pullback of
`Op.acosh` at `-2 + i` is the principal derivative `1 / (√(z - 1) √(z + 1)) = -0.402 - 0.249i`,
within two ulps of `cmath`'s. -/
def checkComplexAcosh : IO Unit := do
  let acosh (re im : Float) : Complex Float := Context.acosh ⟨re, im⟩
  let words (z : Complex Float) : UInt64 × UInt64 := (z.re.toBits, z.im.toBits)
  expect "complex acosh at -2 + 0i is 1.317 + pi i"
    (words (acosh (-2) 0) == (0x3ff5124271980434, 0x400921fb54442d18))
  expect "complex acosh at -2 - 0i is 1.317 - pi i"
    (words (acosh (-2) (-0.0)) == (0x3ff5124271980434, 0xc00921fb54442d18))
  expect "complex acosh at 2 - 0i is 1.317 - 0i"
    (words (acosh 2 (-0.0)) == (0x3ff5124271980434, 0x8000000000000000))
  expect "complex acosh at -0.5 + 0i is 0 + 2.094i"
    (words (acosh (-0.5) 0) == (0, 0x4000c152382d7365))
  let sqrt (re im : Float) : Complex Float := MathFunctions.sqrt ⟨re, im⟩
  expect "complex sqrt at 1 +- 0i, -2 +- 0i and 0 +- 0i keeps the sign of the zero"
    (words (sqrt 1 0) == (0x3ff0000000000000, 0) &&
      words (sqrt 1 (-0.0)) == (0x3ff0000000000000, 0x8000000000000000) &&
      words (sqrt (-2) 0) == (0, 0x3ff6a09e667f3bcd) &&
      words (sqrt (-2) (-0.0)) == (0, 0xbff6a09e667f3bcd) &&
      words (sqrt 0 0) == (0, 0) && words (sqrt 0 (-0.0)) == (0, 0x8000000000000000))
  expect "complex acosh at 1 +- 0i is +0 +- 0i and at 0 +- 0i is +-pi/2 i"
    (words (acosh 1 0) == (0, 0) && words (acosh 1 (-0.0)) == (0, 0x8000000000000000) &&
      words (acosh 0 0) == (0, 0x3ff921fb54442d18) &&
      words (acosh 0 (-0.0)) == (0, 0xbff921fb54442d18))
  let sqrt32 (re im : Float32) : Complex Float32 := MathFunctions.sqrt ⟨re, im⟩
  let acosh32 (re im : Float32) : Complex Float32 := Context.acosh ⟨re, im⟩
  let words32 (z : Complex Float32) : UInt32 × UInt32 := (z.re.toBits, z.im.toBits)
  expect "binary32 complex sqrt and acosh at 1 - 0i, 0 - 0i and -2 - 0i"
    (words32 (sqrt32 1 (-0.0)) == (0x3f800000, 0x80000000) &&
      words32 (sqrt32 0 (-0.0)) == (0, 0x80000000) &&
      words32 (sqrt32 (-2) (-0.0)) == (0, 0xbfb504f3) &&
      words32 (acosh32 1 (-0.0)) == (0, 0x80000000) && words32 (acosh32 1 0) == (0, 0))
  let z : Tensor (Complex Float) [1] := Tensor.ofFn fun _ => ⟨-2, 1⟩
  expect "complex acosh pullback at -2 + i"
    (words (Tensor.acoshBackward z (Tensor.ofFn fun _ => 1) 0) ==
      (0xbfd9be6e1c4eec69, 0xbfcfd23839948190))

/-- At an infinite real part and a zero imaginary one the complex `sqrt` and `acosh` agree with
C99's `csqrt` and `cacosh` (CPython's `cmath`, numpy's `complex64`) to the bit, the sign of the
zero selecting the side of the cut: `sqrt (∞ ± 0i) = ∞ ± 0i`, `sqrt (-∞ ± 0i) = 0 ± ∞i`,
`acosh (∞ ± 0i) = ∞ ± 0i` and `acosh (-∞ ± 0i) = ∞ ± πi`, in binary64 and binary32. At a NaN
real part `acosh` is a NaN in both parts, as `cacosh` is. -/
def checkComplexInfinities : IO Unit := do
  let sqrt (re im : Float) : Complex Float := MathFunctions.sqrt ⟨re, im⟩
  let acosh (re im : Float) : Complex Float := Context.acosh ⟨re, im⟩
  let words (z : Complex Float) : UInt64 × UInt64 := (z.re.toBits, z.im.toBits)
  let inf : Float := 1 / 0
  expect "complex sqrt at +-inf +- 0i"
    (words (sqrt inf 0) == (0x7ff0000000000000, 0) &&
      words (sqrt inf (-0.0)) == (0x7ff0000000000000, 0x8000000000000000) &&
      words (sqrt (-inf) 0) == (0, 0x7ff0000000000000) &&
      words (sqrt (-inf) (-0.0)) == (0, 0xfff0000000000000))
  expect "complex acosh at +-inf +- 0i"
    (words (acosh inf 0) == (0x7ff0000000000000, 0) &&
      words (acosh inf (-0.0)) == (0x7ff0000000000000, 0x8000000000000000) &&
      words (acosh (-inf) 0) == (0x7ff0000000000000, 0x400921fb54442d18) &&
      words (acosh (-inf) (-0.0)) == (0x7ff0000000000000, 0xc00921fb54442d18))
  expect "complex acosh at NaN +- 0i is a NaN in both parts"
    ((#[0, -0.0] : Array Float).all fun im => (acosh (0 / 0) im).re.isNaN &&
      (acosh (0 / 0) im).im.isNaN)
  let sqrt32 (re im : Float32) : Complex Float32 := MathFunctions.sqrt ⟨re, im⟩
  let acosh32 (re im : Float32) : Complex Float32 := Context.acosh ⟨re, im⟩
  let words32 (z : Complex Float32) : UInt32 × UInt32 := (z.re.toBits, z.im.toBits)
  let inf32 : Float32 := 1 / 0
  expect "binary32 complex sqrt and acosh at +-inf +- 0i"
    (words32 (sqrt32 inf32 0) == (0x7f800000, 0) &&
      words32 (sqrt32 inf32 (-0.0)) == (0x7f800000, 0x80000000) &&
      words32 (sqrt32 (-inf32) 0) == (0, 0x7f800000) &&
      words32 (sqrt32 (-inf32) (-0.0)) == (0, 0xff800000) &&
      words32 (acosh32 inf32 0) == (0x7f800000, 0) &&
      words32 (acosh32 inf32 (-0.0)) == (0x7f800000, 0x80000000) &&
      words32 (acosh32 (-inf32) 0) == (0x7f800000, 0x40490fdb) &&
      words32 (acosh32 (-inf32) (-0.0)) == (0x7f800000, 0xc0490fdb))
  expect "binary32 complex acosh at NaN +- 0i is a NaN in both parts"
    ((#[0, -0.0] : Array Float32).all fun im => (acosh32 (0 / 0) im).re.isNaN &&
      (acosh32 (0 / 0) im).im.isNaN)

/-- Run `checkElementary`, `checkGeluBeta`, `checkComplexAcosh` and `checkComplexInfinities`. -/
def run : IO Unit := do
  checkElementary
  checkGeluBeta
  checkComplexAcosh
  checkComplexInfinities
  IO.println "  elementary functions: passed"

end TorchLean.Test.Elementary
