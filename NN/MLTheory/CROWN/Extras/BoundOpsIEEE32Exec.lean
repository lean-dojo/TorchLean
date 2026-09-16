/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.BoundOps
public import NN.Spec.Core.FloatInstances
public import FloatLib.Floats.Formats.BinaryInterchange.Configured
public import FloatLib.Floats.Formats.BinaryInterchange.Conversion.Cast.Runtime
public import FloatLib.Floats.Formats.BinaryInterchange.Model.RealSemantics
public import FloatLib.Floats.Formats.BinaryInterchange.Model.ERealSemantics
public import FloatLib.Floats.Formats.IEEE754.Native

/-!
# `BoundOps` instance for `ExecFloat.Binary 8 23`

This instance plugs FloatLib's configured binary32 directed-rounding primitives into the
IBP/CROWN endpoint propagation code.

With this, IBP code written in terms of `BoundOps` can use `α := ExecFloat.Binary 8 23` to get
float32-grid, outward-rounded interval propagation (subject to the usual finiteness preconditions).
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


namespace NN.MLTheory.CROWN

/-- `BoundOps` for `ExecFloat.Binary 8 23`, using the executable directed-rounding endpoint
primitives. -/
instance (priority := 1000) : BoundOps (ExecFloat.Binary 8 23) where
  addDown := (ExecFloat.Binary.add (rounding := .towardNegativeInfinity))
  addUp   := (ExecFloat.Binary.add (rounding := .towardPositiveInfinity))
  subDown := (ExecFloat.Binary.sub (rounding := .towardNegativeInfinity))
  subUp   := (ExecFloat.Binary.sub (rounding := .towardPositiveInfinity))
  mulDown := (ExecFloat.Binary.mul (rounding := .towardNegativeInfinity))
  mulUp   := (ExecFloat.Binary.mul (rounding := .towardPositiveInfinity))

/--
Nonlinear enclosures backed by the proved directed binary32 division and square-root operations.

FloatLib's `ExecFloat.Binary.exp` is a deterministic approximation, not yet a proved
enclosure of real exponentiation, so exponential and logarithmic transfers are intentionally absent.
-/
instance (priority := 1000) : NonlinearBoundOps (ExecFloat.Binary 8 23) where
  divBounds aLo aHi bLo bHi :=
    if NonlinearBoundOps.denominatorAvoidsZero bLo bHi then
      let d1 := (ExecFloat.Binary.div (rounding := .towardNegativeInfinity)) aLo bLo
      let d2 := (ExecFloat.Binary.div (rounding := .towardNegativeInfinity)) aLo bHi
      let d3 := (ExecFloat.Binary.div (rounding := .towardNegativeInfinity)) aHi bLo
      let d4 := (ExecFloat.Binary.div (rounding := .towardNegativeInfinity)) aHi bHi
      let u1 := (ExecFloat.Binary.div (rounding := .towardPositiveInfinity)) aLo bLo
      let u2 := (ExecFloat.Binary.div (rounding := .towardPositiveInfinity)) aLo bHi
      let u3 := (ExecFloat.Binary.div (rounding := .towardPositiveInfinity)) aHi bLo
      let u4 := (ExecFloat.Binary.div (rounding := .towardPositiveInfinity)) aHi bHi
      some (NonlinearBoundOps.min4 d1 d2 d3 d4, NonlinearBoundOps.max4 u1 u2 u3 u4)
    else
      none
  expBounds := fun _ _ => none
  logBounds := fun _ _ => none
  sqrtBounds lo hi :=
    if hi < 0 then
      none
    else
      let lo' := if lo > 0 then lo else 0
      some ((ExecFloat.Binary.sqrt (rounding := .towardNegativeInfinity)) lo',
        (ExecFloat.Binary.sqrt (rounding := .towardPositiveInfinity)) hi)
  sigmoidBounds := fun _ _ => some (0, 1)
  tanhBounds := fun _ _ => some ((-1), 1)
  sinBounds := fun _ _ => some ((-1), 1)
  cosBounds := fun _ _ => some ((-1), 1)
  layerNormAbsBound := fun _ => none
  supportsIdealCoupledDerivatives := false

end NN.MLTheory.CROWN
