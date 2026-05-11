/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.Arb
public import NN.Floats.FP32
public import NN.Floats.Float32
public import NN.Floats.IEEEExec
public import NN.Floats.Interval
public import NN.Floats.NeuralFloat

/-!
# Floats entrypoint

This is the “all floating-point semantics in one import” entrypoint:

- proof-oriented real-valued models (`FP32`, `NeuralFloat`),
- the executable bit-level model (`IEEEExec`),
- interval/enclosure utilities (`Interval`),
- and the external Arb oracle integration (`Arb`),
- and the shared error-bound vocabulary used across the library.

This module is the chapter boundary for the floating-point surface. It imports the focused
`NN.Floats.*` subsystems directly so downstream users have one stable entrypoint.
-/

@[expose] public section
