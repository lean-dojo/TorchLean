/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.BoundOps
public import NN.Spec.Core.FloatInstances
public import NN.Floats.IEEEExec.Exec32.Directed

/-!
# `BoundOps` instance for `IEEE32Exec`

This instance plugs the executable float32 directed-rounding primitives from
`NN/Floats/IEEEExec/Exec32.lean` into the IBP/CROWN endpoint propagation code.

With this, any IBP code written in terms of `BoundOps` can be run with `α := IEEE32Exec` to get
float32-grid, outward-rounded interval propagation (subject to the usual finiteness preconditions).
-/

@[expose] public section


namespace NN.MLTheory.CROWN

open TorchLean.Floats.IEEE754

/-- `BoundOps` for `IEEE32Exec`, using the executable directed-rounding endpoint primitives. -/
instance (priority := 1000) : BoundOps IEEE32Exec where
  addDown := IEEE32Exec.addDown
  addUp   := IEEE32Exec.addUp
  subDown := IEEE32Exec.subDown
  subUp   := IEEE32Exec.subUp
  mulDown := IEEE32Exec.mulDown
  mulUp   := IEEE32Exec.mulUp

/--
Nonlinear enclosures backed by the proved directed binary32 division and square-root operations.

The executable `IEEE32Exec.exp` implementation is a deterministic approximation, not yet a proved
enclosure of real exponentiation, so exponential and logarithmic transfers are intentionally absent.
-/
instance (priority := 1000) : NonlinearBoundOps IEEE32Exec where
  divBounds aLo aHi bLo bHi :=
    if NonlinearBoundOps.denominatorAvoidsZero bLo bHi then
      let d1 := IEEE32Exec.divDown aLo bLo
      let d2 := IEEE32Exec.divDown aLo bHi
      let d3 := IEEE32Exec.divDown aHi bLo
      let d4 := IEEE32Exec.divDown aHi bHi
      let u1 := IEEE32Exec.divUp aLo bLo
      let u2 := IEEE32Exec.divUp aLo bHi
      let u3 := IEEE32Exec.divUp aHi bLo
      let u4 := IEEE32Exec.divUp aHi bHi
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
      some (IEEE32Exec.sqrtDown lo', IEEE32Exec.sqrtUp hi)
  sigmoidBounds := fun _ _ => some (0, 1)
  tanhBounds := fun _ _ => some ((-1), 1)
  sinBounds := fun _ _ => some ((-1), 1)
  cosBounds := fun _ _ => some ((-1), 1)
  layerNormAbsBound := fun _ => none
  supportsIdealCoupledDerivatives := false

end NN.MLTheory.CROWN
