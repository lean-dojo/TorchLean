/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module -- shake: keep-all

public import FloatLib.Floats.Interval
public import NN.Floats.Interval.FP32
public import NN.Floats.Interval.IEEEExec32
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

/-!
# Interval adapters for TorchLean scalar formats

FloatLib supplies the generic interval theory and directed arithmetic. These modules connect it
to TorchLean's rounded-real `FP32` and configured `ExecFloat.Binary 8 23` endpoints. The Arb
enclosure
adapter remains a separate import because it uses an external oracle.
-/

@[expose] public section
