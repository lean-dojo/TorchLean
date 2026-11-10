/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Context
public import TorchLean.Tensor.Storage
public import FloatLib.Floats.Formats.BinaryInterchange.Configured
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Transcendentals
public import TorchLean.Tensor.ConfiguredBinary.InverseTrigonometric
public import TorchLean.Tensor.ConfiguredBinary.Elementary

/-!
# Configured binary scalars for TorchLean specifications

FloatLib owns the binary formats, representations, arithmetic, and elementary-function class. The
`Context` dictionary `configuredBinaryContext` makes those formats the scalars of TorchLean's
tensors and models. Select a format with `FloatLib.Floats.ExecFloat.Binary`, including
formats wider than binary64. The descriptor carries proofs that the exponent width is at least two,
the fraction width and bias are positive, and the bias is at most the encoding's largest finite
exponent field. Concrete valid widths discharge these requirements automatically, while
parameterized formats must provide the proofs.

Integer and rational casts round their exact value once in the destination format. The `pow`
field is FloatLib's `ExecFloat.Binary.pow`, the model operation lifted without converting through
a native `Float`. For an IEEE descriptor, `exp`, `log`, `log1p`, `expm1` and `expMul` are
FloatLib's correctly rounded functions, and the last three round `log (1 + x)`, `exp x - 1` and
`exp x * w` once. Other descriptors use FloatLib's deterministic approximations for `exp` and
`log` and the compensated defaults for `log1p` and `expm1`. These, like `sin`, `cos`, `sinh`,
`cosh` and `tanh` in every format, carry no accuracy theorem. `tan`, `exp2`, `log2`, `log10`,
`asinh`, `acosh` and `atanh` are the real functions rounded once to nearest even, with the special
values of C99 Annex F (`TorchLean.Tensor.ConfiguredBinary.Elementary`).

Dimension roots use the ordinary cast-then-root schedule unless the count overflows. That branch
rounds the exact rational square root, so a representable root is not lost with the cast.
Division by a count (`divNat`) rounds the exact quotient once without casting the count, so a
count beyond the format's precision or range is used exactly. The pullbacks of means divide each
entry the same way.

The operations in this instance use FloatLib's CPU software arithmetic. Custom computations
can select configured GPU arithmetic through `Kernel`. This dictionary alone supplies neither
that execution path nor a `LawfulContext` or trainer/checkpoint encoding.

At a configured format the registered operators (`TorchLean.Op`) use FloatLib's correctly rounded
division and square root, `Rounding` (FloatLib's integral rounding in each direction and C's `fmod`
from the exact IEEE remainder), FloatLib's power function, the `InverseTrig` instance of
`TorchLean.Tensor.ConfiguredBinary.InverseTrigonometric`, whose functions round the real
function once in an IEEE format, FloatLib's `rootN` as the cube root (IEEE 754-2019 `rootn(x, 3)`,
which in an IEEE format rounds the real cube root once at a finite input with a finite result) and
its `nextUp` and `nextDown` in `nextafter`. `hypot` is FloatLib's, which rounds the exact norm once.
Flooring division (`Tensor.floorDivide`) runs `divFloorSchedule` with the format's rounded
operations and `Rounding` functions, which gives the floor of the IEEE quotient at an infinite
dividend. Over the reals the schedule computes `⌊a / b⌋` (`divFloorSchedule_real`). FloatLib has no
error function, so a configured format has no `Erf` instance, and `Op.erf` and `Op.gelu .none` have
no rule there.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean

open FloatLib.Floats
open FloatLib.Floats.Formats.BinaryInterchange

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

-- On IEEE descriptors, TorchLean uses FloatLib's correctly rounded `exp` and `log`.
attribute [instance high] ExecFloat.Binary.CorrectlyRounded.mathFunctions

/-- Storage that compares configured scalars by their encodings, as saved reverse programs need.
Numerical equality is too coarse there: signed zeros and NaNs must keep their distinct storage
representations. -/
instance configuredBinaryStorage : Storage Value :=
  { instArrayStorage Value with decEq? := some inferInstance }

namespace ConfiguredBinary

/-- The square root of the count `n`: the root of the cast count, or, where the cast overflows,
FloatLib's exact rational root rounded once, since the root can be representable when the count is
not. That branch also covers custom exponent biases and finite encodings. -/
def sqrtNat (n : Nat) : Value :=
  let input := Model.ofRat format (n : Rat) (status := true)
  if input.status.overflow then
    ExecFloat.Binary.ofModel (Model.AlgebraicRounding.sqrtRat format false (n : Rat))
  else
    ExecFloat.Binary.ofModel (Model.sqrt input.value)

/-- `x` divided by the count `n`, without casting `n`. For an IEEE descriptor, a finite `x` and a
positive `n`, the exact quotient `value(x) / n` is rounded once to nearest-even, and a zero `x`
keeps its sign. The count `0` gives `x / 0`, a nonfinite `x` over a positive `n` gives `x / 1`,
and the result carries no exception status. -/
def divNat (x : Value) (n : Nat) : Value :=
  if n = 0 then x / ExecFloat.Binary.ofModel (Model.ofRat format 0) else
    match ExecFloat.Binary.toRat? x with
    | some q => if q = 0 then x else ExecFloat.Binary.ofRat (q / n)
    | none => x / ExecFloat.Binary.ofModel (Model.ofRat format 1)

end ConfiguredBinary

/-- FloatLib's integral rounding in each direction, with ties to even for `round`, and C's `fmod`
from FloatLib's exact IEEE remainder. -/
instance configuredBinaryRounding : Rounding Value where
  floor x := ExecFloat.Binary.roundToIntegral x .down
  ceil x := ExecFloat.Binary.roundToIntegral x .up
  trunc x := ExecFloat.Binary.roundToIntegral x .towardZero
  round x := ExecFloat.Binary.roundToIntegral x .nearest
  fmod := fmodSchedule (ExecFloat.Binary.remainder · ·) ExecFloat.Binary.isZero
    ExecFloat.Binary.signBit ExecFloat.Binary.copySign

namespace ConfiguredBinary

/--
C's `nextafter`, which `torch.nextafter` (PyTorch 2.8 docs) and StableHLO's `next_after` follow: a
quiet NaN (the sum of the operands) where either is a NaN, `y` where `x == y` (so
`nextafter (-0) (+0)` is `+0`), and otherwise FloatLib's IEEE 754-2019 `nextUp` or `nextDown` of
`x` toward `y`, subnormals included.

We wrote it by hand from C99 7.12.11.3 (WG14 N1256), and its NaN result `x + y` is the one glibc's
`nextafterf` returns. PyTorch 2.8.0's CPU kernel calls `std::nextafter` in its scalar loop and
SLEEF's `nextafter` in the vectorized loop it runs on AVX2 and AVX-512 CPUs.
`test_integer_bitwise_operators_and_shifts` in `scripts/checks/test_ir_capture.py` checks seven
binary32 results, `nextafter 3 (-∞) = 0x403fffff` among them.
-/
def nextafter (x y : Value) : Value :=
  if ExecFloat.Binary.isNaN x || ExecFloat.Binary.isNaN y then x + y
  else if x == y then y
  else if x < y then ExecFloat.Binary.nextUp x else ExecFloat.Binary.nextDown x

end ConfiguredBinary

/-- FloatLib's `rootN x 3`, IEEE 754-2019 `rootn(x, 3)`: in an IEEE format, at a finite input
with a finite result, the real cube root rounded once (`toReal_rootN_eq_roundAt`).
`test_cube_root_and_nextafter` in `scripts/checks/test_ir_capture.py` checks it in binary32,
bfloat16 and binary16 against mpmath roots certified by the cubes of the midpoints. -/
instance configuredBinaryCbrt : Cbrt Value := ⟨(ExecFloat.Binary.rootN · 3)⟩

/-- C's `nextafter`, `ConfiguredBinary.nextafter`. -/
instance configuredBinaryNextAfter : NextAfter Value := ⟨ConfiguredBinary.nextafter⟩

/-- FloatLib's complete interchange word of `format.bitWidth` bits, a bijection with the values. -/
instance configuredBinaryBits : Bits Value where
  toBits_ofBits b := congrArg Model.bits (ExecFloat.Binary.toModel_ofModel ⟨b⟩)

/--
The `Context` of any configured binary format, as a tensor and model scalar.

The default safeguard is `1e-6`, rounded once. A coarse format that rounds it to zero uses its
smallest positive subnormal (encoding word `1`): exponent width `3` and fraction width `2`
give `1/16`, and the minimal default-bias format with widths `2` and `1` gives `1/2`. Such a large
safeguard can materially change a guarded formula. It is a nonzero default, neither machine
epsilon nor an error bound, and callers should choose tolerances for their format and problem.
-/
instance configuredBinaryContext : Context Value where
  natCast n := ExecFloat.Binary.ofModel (Model.ofRat format (n : Rat))
  ratCast value := ExecFloat.Binary.ofModel (Model.ofRat format value)
  sqrtNat := ConfiguredBinary.sqrtNat
  divNat := ConfiguredBinary.divNat
  pow := ExecFloat.Binary.pow
  defaultEpsilon :=
    let candidate : Value := ExecFloat.Binary.ofModel
      (Model.ofRat format (1 / 1000000 : Rat))
    if ExecFloat.Binary.isZero candidate then ExecFloat.Binary.ofNatBits 1 else candidate
  decidableGT := fun x y => inferInstanceAs (Decidable (x > y))
  log1p x := if format.isIEEE then ExecFloat.Binary.CorrectlyRounded.logPlus1 x
    else Numeric.log1p x
  expm1 x := if format.isIEEE then ExecFloat.Binary.CorrectlyRounded.expMinus1 x
    else Numeric.expm1 x
  expMul x w := if format.isIEEE then ExecFloat.Binary.CorrectlyRounded.expMul x w
    else MathFunctions.exp x * w
  copysign := ExecFloat.Binary.copySign
  hypot := ExecFloat.Binary.hypot
  isNaN := ExecFloat.Binary.isNaN
  isInf := ExecFloat.Binary.isInf
  quietNaN x := ExecFloat.Binary.ofModel (Model.quietNaN (ExecFloat.Binary.toModel x))
  tan := ConfiguredBinary.tan
  exp2 := ConfiguredBinary.exp2
  log2 := ConfiguredBinary.logb 2
  log10 := ConfiguredBinary.logb 10
  asinh := ConfiguredBinary.asinh
  acosh := ConfiguredBinary.acosh
  atanh := ConfiguredBinary.atanh

namespace ConfiguredBinary

/-- A configured power decodes to `Model.pow` of the decoded operands. -/
theorem toModel_hPow (base exponent : Value) :
    ExecFloat.Binary.toModel (base ^ exponent : Value) =
      Model.pow (ExecFloat.Binary.toModel base) (ExecFloat.Binary.toModel exponent) :=
  ExecFloat.Binary.toModel_pow base exponent

/-! ## Correctly rounded transcendentals

Operators reach these functions through their `Context`. For an IEEE descriptor, each is
FloatLib's correctly rounded function. -/

/-- The exponential is FloatLib's correctly rounded one. -/
theorem exp_of_isIEEE (hformat : format.isIEEE = true) (x : Value) :
    MathFunctions.exp (self := Context.toMathFunctions) x =
      ExecFloat.Binary.CorrectlyRounded.exp x :=
  ExecFloat.Binary.CorrectlyRounded.mathFunctions_exp_of_isIEEE x

/-- The logarithm is FloatLib's correctly rounded one. -/
theorem log_of_isIEEE (hformat : format.isIEEE = true) (x : Value) :
    MathFunctions.log (self := Context.toMathFunctions) x =
      ExecFloat.Binary.CorrectlyRounded.log x :=
  ExecFloat.Binary.CorrectlyRounded.mathFunctions_log_of_isIEEE x

/-- `log1p` is FloatLib's `logPlus1`, which rounds `log (1 + x)` once. -/
theorem log1p_of_isIEEE (hformat : format.isIEEE = true) (x : Value) :
    Context.log1p x = ExecFloat.Binary.CorrectlyRounded.logPlus1 x :=
  ite_eq_left hformat

/-- `expm1` is FloatLib's `expMinus1`, which rounds `exp x - 1` once. -/
theorem expm1_of_isIEEE (hformat : format.isIEEE = true) (x : Value) :
    Context.expm1 x = ExecFloat.Binary.CorrectlyRounded.expMinus1 x :=
  ite_eq_left hformat

/-- `expMul` is FloatLib's `expMul`, which rounds `exp x * w` once. -/
theorem expMul_of_isIEEE (hformat : format.isIEEE = true) (x w : Value) :
    Context.expMul x w = ExecFloat.Binary.CorrectlyRounded.expMul x w :=
  ite_eq_left hformat

end ConfiguredBinary

end TorchLean
