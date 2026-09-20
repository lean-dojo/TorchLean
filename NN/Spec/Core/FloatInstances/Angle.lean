/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Core.Numeric.Angle
public import FloatLib.Floats.Formats.IEEE754.Native

/-!
# Host polar-angle adapter for executable binary32

Complex logarithms use the host binary64 `atan2` and round its result to binary32. Finite inputs
embed exactly in binary64. This adapter is a host transcendental boundary; it is not a certified or
platform-independent binary32 arctangent implementation.
-/

@[expose] public section

namespace TorchLean

open FloatLib.Floats

/-- Host polar angle rounded from binary64 to executable binary32. -/
instance : Atan2 (ExecFloat.Binary (exponentBits := 8) (fractionBits := 23)) :=
  ⟨fun y x => ExecFloat.Binary.ofFloat32
    (Float.atan2 (ExecFloat.Binary.toFloat32 y).toFloat
      (ExecFloat.Binary.toFloat32 x).toFloat).toFloat32⟩

end TorchLean
