/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# Foundational Numeric Interfaces

This module contains the small scalar interfaces shared by TorchLean's floating-point library and
its tensor specifications. It deliberately knows nothing about tensors, models, runtimes, or
verification.

`MathFunctions` names the transcendental operations used by numerical code. Integer and rational
constants use the standard numerical interfaces. The broader neural-model interface, `Context`,
lives in `NN.Spec.Core.Context`.

Nothing here depends on Mathlib. That is deliberate: most of the library is scalar-polymorphic or
runs on `Float`, and only the specification layer ever needs the exact reals. The `ℝ` instances
therefore live in `NN.Core.Numeric.Real`, so that modules which never mention `ℝ` do not pay for
loading the real-analysis hierarchy.
-/

@[expose] public section

/-- Scalar transcendental functions shared by numerical and model code. -/
class MathFunctions (α : Type) where
  exp : α → α
  tanh : α → α
  cosh : α → α
  sqrt : α → α
  abs : α → α
  log : α → α
  pi : α
  cos : α → α
  sin : α → α
  sinh : α → α

namespace TorchLean

/-- Default normalization stabilizer, `1e-5`, evaluated in the selected scalar arithmetic.

The natural denominator is cast once, then division uses the backend. Arbitrary coarse grids need
not agree with repeated multiplication by ten or with rounding the exact real value `1e-5` once.
-/
def normalizationEpsilon {α : Type} [One α] [NatCast α] [Div α] : α :=
  1 / ((100000 : Nat) : α)

end TorchLean

/-- Host implementations of the scalar transcendental interface. -/
instance : MathFunctions Float where
  exp := Float.exp
  tanh := Float.tanh
  cosh := Float.cosh
  sqrt := Float.sqrt
  abs := Float.abs
  log := Float.log
  pi := 3.14159265358979323846
  cos := Float.cos
  sin := Float.sin
  sinh := Float.sinh

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
