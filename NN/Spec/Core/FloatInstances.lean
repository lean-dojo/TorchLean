/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context
public import NN.Tensor.Internal.Representation.Storage
public import FloatLib.Floats.Formats.BinaryInterchange.Configured
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Transcendentals

/-!
# Configured binary scalars for TorchLean specifications

FloatLib owns the binary formats, representations, arithmetic, and elementary-function class.
This module supplies the `Context` dictionary used by TorchLean's tensors and models. Select a
format directly with `FloatLib.Floats.ExecFloat.Binary`, including formats wider than binary64.
The descriptor carries proofs that the exponent width is at least two, the fraction width and bias
are positive, and the bias is at most the encoding's largest finite exponent field. Concrete valid
widths discharge these requirements automatically; parameterized formats must provide the proofs.

Integer and rational casts round their exact value once in the destination format. The `pow`
field lifts FloatLib's model operation without converting through a native `Float`. FloatLib's
transcendentals are deterministic approximations; this dictionary adds no general accuracy or
correct-rounding theorem for them.

The operations in this instance use FloatLib's CPU software arithmetic. Custom computations
can select configured GPU arithmetic through `NN.Kernel`; this dictionary alone supplies
neither that execution path nor a `LawfulContext` or trainer/checkpoint encoding.

The pointwise operators of `Spec.PointwiseOp` use FloatLib's correctly rounded division and
square root, its integral rounding and IEEE remainder, its power function and its `tanh`.
Flooring division and the tanh GELU follow PyTorch's CPU kernels operation by operation, the GELU
as they run at the AVX2 and AVX-512 capability levels, measured on x86-64. FloatLib has no error
function, so `erf` and the exact GELU are unsupported here.
-/

@[expose] public section

namespace TorchLean

open FloatLib.Floats
open FloatLib.Floats.Formats.BinaryInterchange

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- Compare configured scalar encodings in saved reverse programs. Numerical equality is not
enough here: signed zeros and NaNs must keep their distinct storage representations. -/
instance configuredBinaryStorage : Storage Value :=
  { instArrayStorage Value with decEq? := some inferInstance }

namespace ConfiguredBinary

/-- FloatLib's power function, the dictionary's `pow`. -/
def pow (base exponent : Value) : Value :=
  ExecFloat.ModelCodec.liftBinary (Model := Model format) (plan := plan) Model.pow base exponent

/--
C's `fmod`: the remainder of truncating division, with the sign of the dividend.

The IEEE remainder rounds the quotient to the nearest integer instead. Both remainders are exact.
When their signs differ, `fmod` is the IEEE remainder plus the divisor's magnitude with the
dividend's sign; that sum is the exact remainder, so the rounded addition returns it unchanged.
-/
def fmod (dividend divisor : Value) : Value :=
  let remainder := ExecFloat.Binary.remainder dividend divisor
  if !ExecFloat.Binary.isZero remainder &&
      ExecFloat.Binary.signBit remainder != ExecFloat.Binary.signBit dividend then
    remainder + ExecFloat.Binary.copySign divisor dividend
  else
    remainder

/--
PyTorch's flooring division, `c10::div_floor_floating`, one FloatLib operation per step.

A zero divisor returns the IEEE quotient. Otherwise the exact remainder corrects the rounded
quotient, as Python's `//` does; `1 // 0.1` is `9`, while the floor of the rounded quotient
`1 / 0.1 = 10` would be `10`. A zero result takes the sign of the IEEE quotient.
-/
def divFloor (dividend divisor : Value) : Value :=
  Spec.divFloorSchedule fmod (ExecFloat.Binary.roundToIntegral · .down)
    (ExecFloat.Binary.copySign 0) (ExecFloat.Binary.ofModel (Model.ofRat format (1 / 2 : Rat)))
    dividend divisor

/--
PyTorch's tanh GELU, `0.5 * x * (1 + tanh (β * (x + κ * (x * x * x))))`, one FloatLib operation
per step in the order of PyTorch's kernel, with `x + κ x³` rounded once as a fused multiply-add.
One PyTorch binary runs each CPU kernel at the DEFAULT, AVX2 or AVX-512 capability level, chosen at
run time from the processor or `ATEN_CPU_CAPABILITY`. On x86-64 (a Xeon Platinum 8275CL), both of
its loops compute that sum so at the AVX2 and AVX-512 levels, and round `κ x³` and the sum
separately at DEFAULT. PyTorch on aarch64, on Apple processors and on CUDA was not measured.
`β` and `κ` are PyTorch's binary64 constants `kGeluBeta` and `kGeluKappa`, the binary64 values
nearest `√(2 / π)` and `0.044715`, each rounded once to the format as the kernel's conversion to its
computation type does.
-/
def geluTanh (x : Value) : Value :=
  let constant (value : Rat) : Value := ExecFloat.Binary.ofModel (Model.ofRat format value)
  let beta := constant (7186705221432913 / 9007199254740992)
  let kappa := constant (6444110634811895 / 144115188075855872)
  let inner := beta * ExecFloat.Binary.fma kappa (x * x * x) x .nearest
  constant (1 / 2) * x * (constant 1 + MathFunctions.tanh inner)

/-- FloatLib meanings of the pointwise operators that depend on the scalar type. -/
def pointwise : Spec.PointwiseTable Value
  | .rsqrt => some (Spec.PointwiseFn.unary fun x => .ok (1 / MathFunctions.sqrt x))
  | .div .none => some (Spec.PointwiseFn.binary fun x y => .ok (x / y))
  | .div .trunc => some (Spec.PointwiseFn.binary fun x y =>
      .ok (ExecFloat.Binary.roundToIntegral (x / y) .towardZero))
  | .div .floor => some (Spec.PointwiseFn.binary fun x y => .ok (divFloor x y))
  | .pow => some (Spec.PointwiseFn.binary fun x y => .ok (pow x y))
  | .gelu .tanh => some (Spec.PointwiseFn.unary fun x => .ok (geluTanh x))
  | .erf | .neg | .gelu .none | .clampMin => none

end ConfiguredBinary

/--
Use any configured binary format as a tensor/model scalar.

The default safeguard is `1e-6`, rounded once. If a coarse format rounds it to zero, use its
smallest positive subnormal (encoding word `1`) instead. For example, exponent width `3` and
fraction width `2` use `1/16`; the minimal default-bias format with widths `2` and `1` uses `1/2`.
Such a large safeguard can materially change a guarded formula. This is a nonzero default, not
machine epsilon or an error bound; callers should choose tolerances for their format and problem.
-/
instance configuredBinaryContext : Context Value where
  natCast n := ExecFloat.Binary.ofModel (Model.ofRat format (n : Rat))
  ratCast value := ExecFloat.Binary.ofModel (Model.ofRat format value)
  pow := ConfiguredBinary.pow
  defaultEpsilon :=
    let candidate : Value := ExecFloat.Binary.ofModel
      (Model.ofRat format (1 / 1000000 : Rat))
    if ExecFloat.Binary.isZero candidate then ExecFloat.Binary.ofNatBits 1 else candidate
  decidableGT := fun x y => inferInstanceAs (Decidable (x > y))
  pointwise := ConfiguredBinary.pointwise

end TorchLean
