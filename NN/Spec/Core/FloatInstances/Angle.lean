/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Core.Numeric.Angle
public import NN.Floats.IEEEExec.Exec32.Compare

/-!
# Host polar-angle adapter for executable binary32

Complex logarithms use the host binary64 `atan2` and round its result to binary32. Finite inputs
embed exactly in binary64. This adapter is a host transcendental boundary; it is not a certified or
platform-independent binary32 arctangent implementation.
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754.IEEE32Exec

/-- Host polar angle rounded from binary64 to executable binary32. -/
instance : Atan2 IEEE32Exec :=
  ⟨fun y x => ofFloat (Float.atan2 (toFloat y) (toFloat x))⟩

end TorchLean.Floats.IEEE754.IEEE32Exec
