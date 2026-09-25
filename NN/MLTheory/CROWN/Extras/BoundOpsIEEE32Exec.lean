/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.BoundOps
public import NN.Spec.Core.FloatInstances

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
  addDown := (ExecFloat.Binary.addWithRounding (rounding := .towardNegativeInfinity))
  addUp   := (ExecFloat.Binary.addWithRounding (rounding := .towardPositiveInfinity))
  subDown := (ExecFloat.Binary.subWithRounding (rounding := .towardNegativeInfinity))
  subUp   := (ExecFloat.Binary.subWithRounding (rounding := .towardPositiveInfinity))
  mulDown := (ExecFloat.Binary.mulWithRounding (rounding := .towardNegativeInfinity))
  mulUp   := (ExecFloat.Binary.mulWithRounding (rounding := .towardPositiveInfinity))

/--
Nonlinear enclosures backed by the proved directed binary32 division and square-root operations.

FloatLib's `ExecFloat.Binary.exp` is a deterministic approximation, not yet a proved
enclosure of real exponentiation, so exponential and logarithmic transfers are intentionally absent.
-/
instance (priority := 1000) : NonlinearBoundOps (ExecFloat.Binary 8 23) where
  divBounds aLo aHi bLo bHi :=
    if NonlinearBoundOps.denominatorAvoidsZero bLo bHi then
      let d1 := (ExecFloat.Binary.divWithRounding (rounding := .towardNegativeInfinity)) aLo bLo
      let d2 := (ExecFloat.Binary.divWithRounding (rounding := .towardNegativeInfinity)) aLo bHi
      let d3 := (ExecFloat.Binary.divWithRounding (rounding := .towardNegativeInfinity)) aHi bLo
      let d4 := (ExecFloat.Binary.divWithRounding (rounding := .towardNegativeInfinity)) aHi bHi
      let u1 := (ExecFloat.Binary.divWithRounding (rounding := .towardPositiveInfinity)) aLo bLo
      let u2 := (ExecFloat.Binary.divWithRounding (rounding := .towardPositiveInfinity)) aLo bHi
      let u3 := (ExecFloat.Binary.divWithRounding (rounding := .towardPositiveInfinity)) aHi bLo
      let u4 := (ExecFloat.Binary.divWithRounding (rounding := .towardPositiveInfinity)) aHi bHi
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
      some ((ExecFloat.Binary.sqrtWithRounding (rounding := .towardNegativeInfinity)) lo',
        (ExecFloat.Binary.sqrtWithRounding (rounding := .towardPositiveInfinity)) hi)
  sigmoidBounds := fun _ _ => some (0, 1)
  tanhBounds := fun _ _ => some ((-1), 1)
  sinBounds := fun _ _ => some ((-1), 1)
  cosBounds := fun _ _ => some ((-1), 1)
  layerNormAbsBound := fun _ => none
  supportsIdealCoupledDerivatives := false

end NN.MLTheory.CROWN
