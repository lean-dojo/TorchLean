/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.Interval.Comparison
public import NN.Floats.Interval.FP32
public import NN.Floats.Interval.IEEEExec32
public import NN.Floats.Interval.IEEEExec32AddSoundness
public import NN.Floats.Interval.IEEEExec32ArbTrans
public import NN.Floats.Interval.IEEEExec32DivSoundness
public import NN.Floats.Interval.IEEEExec32MinMaxSoundness
public import NN.Floats.Interval.IEEEExec32MulSoundness
public import NN.Floats.Interval.IEEEExec32NoNaN
public import NN.Floats.Interval.IEEEExec32Soundness
public import NN.Floats.Interval.Quantized
public import NN.Floats.Interval.RealBounds
public import NN.Floats.Interval.Rounders

/-!
# `NN.Floats.Interval`

This folder collects **interval / enclosure** utilities used across TorchLean:

- proof-friendly interval enclosures for rounding-on-`ℝ` formats (e.g. `FP32`),
- quantized intervals (endpoints snapped to a chosen NeuralFloat/Flocq-style grid),
- executable IEEE-754 endpoint intervals (`IEEE32Exec`) with Lean proofs for the supported
  arithmetic operations.

The optional Arb-backed transcendental enclosure adapter is an explicit import:
`NN.Floats.Interval.IEEEExec32ArbTrans`.

Design note: we keep the *interval API* here in `NN/Floats` so it can be reused by both
verification (`NN/MLTheory/CROWN/*`) and numerical-model code (`NN/Floats/*`) without creating
unnecessary dependency cycles.

## References
- IEEE 1788-2015 (interval arithmetic standard).
- Moore, Kearfott, Cloud, *Introduction to Interval Analysis* (2009).
- Rump, “INTLAB — INTerval LABoratory” (1999).
- Boldo & Melquiond, “Flocq” (ARITH 2011) for rounded-arithmetic-on-`ℝ` modeling.
-/

@[expose] public section
