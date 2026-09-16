/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Numerics.Capabilities.Elementary

/-!
# Foundational Numeric Interfaces

This module re-exports FloatLib's elementary-function interface and supplies TorchLean's native
binary32 instance and natural-number casts. It knows nothing about tensors, models, or runtimes.

`MathFunctions` names the transcendental operations used by numerical code. Integer and rational
constants use the standard numerical interfaces. The broader neural-model interface, `Context`,
lives in `NN.Spec.Core.Context`.

FloatLib owns the single `MathFunctions` class and its `Float` and `ℝ` instances. Its elementary
capability module imports real analysis; this facade does not define a second class or duplicate
those instances.
-/

@[expose] public section

export FloatLib.Numerics (MathFunctions)

namespace MathFunctions

export FloatLib.Numerics.MathFunctions (exp tanh cosh sqrt abs log pi cos sin sinh)

end MathFunctions

namespace TorchLean

/-- Default normalization stabilizer, obtained by casting the exact rational `1e-5` once.

Native and configured binary contexts round the fraction without first casting its denominator.
This preserves the representable binary16 tolerance even though `100000` itself overflows there.
The result can still be zero in a format that cannot represent a nearby positive value; callers
using such a format must choose an explicit positive tolerance.
-/
def normalizationEpsilon {α : Type} [RatCast α] : α :=
  Rat.cast (1 / 100000 : Rat)

end TorchLean

/-- Native binary32 implementations of the scalar transcendental interface. -/
instance : MathFunctions Float32 where
  exp := Float32.exp
  tanh := Float32.tanh
  cosh := Float32.cosh
  sqrt := Float32.sqrt
  abs := Float32.abs
  log := Float32.log
  pi := (3.14159265358979323846 : Float).toFloat32
  cos := Float32.cos
  sin := Float32.sin
  sinh := Float32.sinh

/-- Cast naturals into Lean's host `Float`. -/
instance : NatCast Float where
  natCast := Float.ofNat

/-- Round naturals directly to binary32, without an intermediate binary64 rounding.

Machine-sized inputs use Lean's native integer conversion; larger naturals use its binary32 model.
-/
instance : NatCast Float32 where
  natCast n :=
    if n < UInt64.size then
      (UInt64.ofNat n).toFloat32
    else
      Float32.ofModel (Float32.Model.ofNat n)
