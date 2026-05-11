/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Exec32
public import NN.MLTheory.CROWN.BoundOps

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

end NN.MLTheory.CROWN
