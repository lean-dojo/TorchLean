/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.ComplexFn
public import FloatLib.Floats.Formats.BinaryInterchange.Configured
public import FloatLib.Floats.Formats.BinaryInterchange.Complex.Elementary.Algebraic
public import FloatLib.Floats.Formats.BinaryInterchange.Complex.Elementary.Inverse
public import FloatLib.Floats.Formats.BinaryInterchange.Complex.Elementary.Power
public import FloatLib.Floats.Formats.BinaryInterchange.Complex.Elementary.Tangent

/-!
# The complex functions and the power in a configured binary format

A configured format applies FloatLib's `ExecComplex` functions to the pair of parts
(`configuredBinaryComplex`): `ExecComplex.exp`, `expm1`, `log`, `log1p`, `sqrt`, `rsqrt`, `sin`,
`cos`, `tan`, `sinh`, `cosh`, `tanh`, `asin`, `acos`, `atan`, `asinh`, `acosh` and `atanh`, the
functions of C23 7.3 and Annex G, rounded to nearest even. FloatLib proves that in an IEEE format,
at a finite argument, each part of the result is the part of an exact result rounded once
(`ExecComplex.exp_eq` to `ExecComplex.atanh_eq`), and that the exact result is `ComplexFn.value`
of the argument (`ExecComplex.expExact_value` and the others, in
`FloatLib.Floats.Formats.BinaryInterchange.Complex.Elementary.ValueProof`):

* `exp`, `expm1`, `sin`, `cos`, `tan`, `sinh`, `cosh` and `tanh` at every finite argument;
* `sqrt`, `rsqrt`, `log`, `log1p`, `atan` and `atanh` off the branch cut, and on it where the
  sign bit that selects its side is clear, away from the poles `-i` of `atan` and `1` of `atanh`.
  With that bit set the result is the mirror image, the limit from the other side (C23 7.3.3);
* `asinh` where both sign bits are clear (`ExecComplex.asinhExact_value`). FloatLib extends it to
  the other quadrants by oddness and conjugation, and the principal expression is odd and commutes
  with conjugation off `{i y : 1 < |y|}` (`ComplexFn.value_asinh_neg`,
  `ComplexFn.value_asinh_conj`), so `asinh` is the value where the sign bits agree or off that
  cut, and `asin`, which is `-i asinh (i z)` (`ComplexFn.value_asin`), where they differ or off
  `{x : 1 < |x|}`;
* `acosh`: where the imaginary sign bit is clear (`ExecComplex.acoshExact_value`), and where it is
  set off `(-∞, 1)`, by conjugation (`ComplexFn.value_acosh_conj`);
* `acos`: where the imaginary sign bit is clear `-i` times the expression of `acosh`
  (`ExecComplex.acosExact_value`), which is `π / 2 - asin` off `(1, ∞)` (`ComplexFn.value_acos`),
  and where it is set the conjugate of that at `conj z`, which is `π / 2 - asin` off `(-∞, -1)`
  (`ComplexFn.value_acos_lower`). On `(1, ∞)` the principal expression is the limit from below.

The power is FloatLib's `ExecComplex.pow`, C23's `cpow` (7.3.8.2, G.6.5.1), `exp (w log z)` with
the cut of `log` on the negative real axis of the base. FloatLib proves that in an IEEE format, at
a finite base other than zero and a finite exponent, a result it certifies
(`ExecComplex.powCertified`) has each part of the exact result rounded once (`ExecComplex.pow_eq`);
the exact result (`ExecComplex.powExact`) is Mathlib's `z ^ w` where the imaginary sign bit of the
base is clear and `conj (conj z ^ conj w)`, the limit from below the cut, where it is set. It
certifies a zero exponent, the base one, the exponent `1 / 2` (`ExecComplex.sqrt`), exact powers of
a rational root, powers on an axis, and every other result whose rounding enclosures of increasing
precision decide; an uncertified part is the rounding of the midpoint of its last enclosure, or a
NaN where the exponent `c L - d θ` is beyond `ExecComplex.powExponentCap`. `z ^ 0` is `1 ± 0i` at
every `z`, a NaN too; a zero base gives `+0 + 0i`, `∞ + NaN i` or `NaN + NaN i` as the real part of
the exponent is positive, negative or zero; any other NaN or infinite part gives `NaN + NaN i`.

The derivative is the closed form of each complex derivative evaluated in `ExecComplex`
arithmetic, each operation rounded: `exp z` for `exp` and `expm1`, `1 / z`, `1 / (1 + z)`,
`rsqrt z / 2` and `-rsqrt z / (2 z)` for `log`, `log1p`, `sqrt` and `rsqrt`, `cos z`, `-sin z`,
`1 + tan² z`, `cosh z`, `sinh z` and `1 - tanh² z` for the trigonometric and hyperbolic
functions, and `rsqrt (1 - z²)`, `-rsqrt (1 - z²)`, `1 / (1 + z²)`, `rsqrt (1 + z²)`,
`rsqrt (z - 1) rsqrt (z + 1)` and `1 / (1 - z²)` for the inverses, and `w z ^ (w - 1)` and
`z ^ w log z` for the power. No theorem bounds its error.
-/

@[expose] public section

namespace TorchLean

open FloatLib.Floats
open FloatLib.Floats.Formats.BinaryInterchange

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

namespace ConfiguredBinary

/-- Each function of FloatLib's complex numbers, rounded to nearest even. -/
def complexApply : ComplexFn → ExecComplex format → ExecComplex format
  | .exp => fun z => ExecComplex.exp z
  | .expm1 => fun z => ExecComplex.expm1 z
  | .log => fun z => ExecComplex.log z
  | .log1p => fun z => ExecComplex.log1p z
  | .sqrt => fun z => ExecComplex.sqrt z
  | .rsqrt => fun z => ExecComplex.rsqrt z
  | .sin => fun z => ExecComplex.sin z
  | .cos => fun z => ExecComplex.cos z
  | .tan => fun z => ExecComplex.tan z
  | .sinh => fun z => ExecComplex.sinh z
  | .cosh => fun z => ExecComplex.cosh z
  | .tanh => fun z => ExecComplex.tanh z
  | .asin => fun z => ExecComplex.asin z
  | .acos => fun z => ExecComplex.acos z
  | .atan => fun z => ExecComplex.atan z
  | .asinh => fun z => ExecComplex.asinh z
  | .acosh => fun z => ExecComplex.acosh z
  | .atanh => fun z => ExecComplex.atanh z

/-- The complex derivative of each function, its closed form in `ExecComplex` arithmetic. -/
def complexDeriv (f : ComplexFn) (z : ExecComplex format) : ExecComplex format :=
  let one : ExecComplex format := ⟨Model.ofRat format 1, Model.ofRat format 0⟩
  let two : ExecComplex format := ⟨Model.ofRat format 2, Model.ofRat format 0⟩
  match f with
  | .exp | .expm1 => ExecComplex.exp z
  | .log => one / z
  | .log1p => one / (one + z)
  | .sqrt => ExecComplex.rsqrt z / two
  | .rsqrt => -(ExecComplex.rsqrt z / (two * z))
  | .sin => ExecComplex.cos z
  | .cos => -ExecComplex.sin z
  | .tan => one + ExecComplex.tan z * ExecComplex.tan z
  | .sinh => ExecComplex.cosh z
  | .cosh => ExecComplex.sinh z
  | .tanh => one - ExecComplex.tanh z * ExecComplex.tanh z
  | .asin => ExecComplex.rsqrt (one - z * z)
  | .acos => -ExecComplex.rsqrt (one - z * z)
  | .atan => one / (one + z * z)
  | .asinh => ExecComplex.rsqrt (one + z * z)
  | .acosh => ExecComplex.rsqrt (z - one) * ExecComplex.rsqrt (z + one)
  | .atanh => one / (one - z * z)

/-- The partial complex derivatives of `z ^ w` in `z` and in `w`, `w z ^ (w - 1)` and
`z ^ w log z`, in `ExecComplex` arithmetic. -/
def complexPowDeriv (z w : ExecComplex format) : ExecComplex format × ExecComplex format :=
  let one : ExecComplex format := ⟨Model.ofRat format 1, Model.ofRat format 0⟩
  (w * ExecComplex.pow z (w - one), ExecComplex.pow z w * ExecComplex.log z)

/-- The complex number whose parts are a pair of scalars. -/
def toExecComplex (z : Value × Value) : ExecComplex format :=
  ⟨ExecFloat.Binary.toModel z.1, ExecFloat.Binary.toModel z.2⟩

/-- The pair of parts of a complex number. -/
def ofExecComplex (w : ExecComplex format) : Value × Value :=
  (ExecFloat.Binary.ofModel w.re, ExecFloat.Binary.ofModel w.im)

end ConfiguredBinary

/-- FloatLib's `ExecComplex` functions and power, and the closed forms of their derivatives
(`ConfiguredBinary.complexApply`, `ConfiguredBinary.complexDeriv`,
`ConfiguredBinary.complexPowDeriv`). -/
instance configuredBinaryComplex : ComplexFunctions Value where
  apply f z := ConfiguredBinary.ofExecComplex (ConfiguredBinary.complexApply f
    (ConfiguredBinary.toExecComplex z))
  deriv f z := ConfiguredBinary.ofExecComplex (ConfiguredBinary.complexDeriv f
    (ConfiguredBinary.toExecComplex z))
  pow z w := ConfiguredBinary.ofExecComplex (ExecComplex.pow (ConfiguredBinary.toExecComplex z)
    (ConfiguredBinary.toExecComplex w))
  powDeriv z w :=
    let d := ConfiguredBinary.complexPowDeriv (ConfiguredBinary.toExecComplex z)
      (ConfiguredBinary.toExecComplex w)
    (ConfiguredBinary.ofExecComplex d.1, ConfiguredBinary.ofExecComplex d.2)

end TorchLean
