/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module -- shake: keep-all

public import NN.Floats.FP32.Core
public import NN.Floats.FP32.Error
public import NN.Floats.FP32.Notation
public import NN.Floats.FP32.Sterbenz

/-!
# TorchLean's rounded-real binary32 specialization

`FP32` specializes FloatLib's `NF` to binary32 precision and gradual underflow, with no upper
exponent bound. These modules retain TorchLean's error bounds, notation, and exact-subtraction
corollaries. Encoded binary32 values and exceptional arithmetic use FloatLib's configured
`ExecFloat.Binary 8 23` format.
-/

@[expose] public section
