/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.IEEE754.Native
public import TorchLean.Tensor.Complex

/-!
# Conversion from Lean `Float` constants

Initial parameters, optimizer settings and data loaded from files reach the runtime as Lean
`Float` values, which are IEEE binary64. `FromFloat α` rounds such a constant once into
the element type `α`, in a chosen rounding mode. `ofFloat` rounds to nearest-even by
default. The class also gives the binary64 value of a constant rounded to nearest in `α`, so
optimizer and scheduler settings can be checked in the precision they run in. Instances cover
`Float`, `Float32` and every configured FloatLib binary format, and lift to `TorchLean.Complex α`
when `α` has an instance and a `Zero`.
-/

@[expose] public section

namespace TorchLean

open FloatLib.Floats
open FloatLib.Floats.Formats.BinaryInterchange

/--
Conversion of Lean `Float` constants into a runtime element type `α`, in a chosen rounding mode.
-/
class FromFloat (α : Type) where
  /-- Convert a Lean binary64 `Float` into `α`, rounding once in `mode` (IEEE 754-2019
  convertFormat). On overflow, the round-to-nearest modes and any mode that rounds the magnitude
  up give the format's overflow value, and the other modes give the finite value of largest
  magnitude with the input's sign. The overflow value is an infinity when the format has
  infinities, and otherwise a NaN or the largest finite value, depending on the encoding. A NaN
  stays a NaN unless the format has none, and then it becomes `+0`. -/
  ofFloat : Float → Model.IEEERoundingMode → α
  /--
  Round a binary64 constant to nearest in `α` and read the result back as a `Float`, for
  parameter-domain checks.

  The default `id` skips the rounding, so a custom scalar without an override gets only the
  binary64 checks. The `Float32` and configured binary instances override it, which lets a check
  reject a setting that leaves its domain after conversion, such as a positive stabilizer that
  rounds to zero. The field works on `Float` alone and needs none of the arithmetic or order
  operations of `Context`.
  -/
  roundForValidation : Float → Float := id

/--
Convert a Lean `Float` into the runtime element type `α`, rounding once in `mode`, nearest-even by
default.

Interval code rounds a lower bound toward negative infinity with `(mode := .down)` and an upper
bound toward positive infinity with `(mode := .up)`, so that the converted interval encloses the
original.
-/
def ofFloat {α : Type} [FromFloat α] (x : Float)
    (mode : Model.IEEERoundingMode := .nearest) : α :=
  FromFloat.ofFloat x mode

/-- On `Float` the conversion is the identity in every rounding mode. -/
instance : FromFloat Float where
  ofFloat x _ := x

/-- Round binary64 values to Lean's native binary32 representation: the native conversion to
nearest, FloatLib's binary32 cast in the other modes. The native one is the C cast `(float)a` of
`lean_float_to_float32` in Lean 4.35.0's `lean.h`, which C99 F.3 (WG14 N1256) makes IEC 60559's
conversion between precisions. `TorchLean.Test.DirectedConversion.run` checks that `0.1` lands on
`0x3DCCCCCD`. -/
instance : FromFloat Float32 where
  ofFloat x mode := match mode with
    | .nearest => x.toFloat32
    | mode => ExecFloat.Binary.toFloat32 (ExecFloat.Binary.ofFloat x mode : Binary32)
  roundForValidation x := x.toFloat32.toFloat

/--
Round the exact binary64 source value once into any configured binary format, in the requested
mode.

Binary32 to nearest goes through Lean's native conversion and stores its interchange word, and the
other modes and formats use FloatLib's cast. Validation rounds to nearest in the destination and
converts the result back to binary64, so the checks see overflow, underflow and any change of
domain. Tensors and state supplied as values of the configured scalar skip this conversion.
-/
instance (priority := 100) configuredBinaryFromFloat
    {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code] :
    FromFloat (ExecFloat (Configured.Family format code plan)) :=
  let convert (value : Float) (mode : Model.IEEERoundingMode) :
      ExecFloat (Configured.Family format code plan) :=
    if format = .binary32 ∧ mode = .nearest then
      ExecFloat.Binary.ofNatBits value.toFloat32.toBits.toNat
    else
      ExecFloat.Binary.ofModel
        (Model.cast _ format (ExecFloat.Binary.toModel
          (ExecFloat.Binary.ofFloat value : FloatLib.Floats.Binary64)) mode)
  { ofFloat := convert
    roundForValidation value :=
      if format = .binary32 then value.toFloat32.toFloat
      else
        ExecFloat.Binary.toFloat (ExecFloat.Binary.ofModel
          (Model.cast format .binary64 (ExecFloat.Binary.toModel (convert value .nearest))) :
          FloatLib.Floats.Binary64) }

/-- Inject binary64 literals into TorchLean's complex representation with zero imaginary part. -/
instance {α : Type} [FromFloat α] [Zero α] : FromFloat (TorchLean.Complex α) where
  ofFloat x mode := ⟨ofFloat (α := α) x mode, 0⟩
  roundForValidation := FromFloat.roundForValidation (α := α)

end TorchLean
