/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.Interval.IEEEExec32
public import NN.Floats.Interval.IEEEExec32NoNaN
public import NN.Floats.Interval.RealBounds

public import NN.Floats.Interval.IEEEExec32AddSoundness
public import NN.Floats.Interval.IEEEExec32DivSoundness
public import NN.Floats.Interval.IEEEExec32MinMaxSoundness
public import NN.Floats.Interval.IEEEExec32MulSoundness

/-!
# `NN.Floats.Interval.IEEEExec32Soundness`

Umbrella import for the soundness theory of `IEEE32Exec.Interval32`.

The underlying interval type lives in `NN.Floats.Interval.IEEEExec32` and is *executable*:
endpoints are `IEEE32Exec` (a bit-level binary32 kernel).

This module re-exports the main enclosure theorems for:
- addition/subtraction/negation,
- multiplication (4-corner rule),
- division/reciprocal (with a conservative whole-interval fallback on division-by-zero),
- auxiliary lemmas about `min4/max4` and non-NaN behavior.

We keep these as separate files for navigation, and provide this umbrella so downstream code can
`import NN.Floats.Interval.IEEEExec32Soundness` without hunting for filenames.
-/

@[expose] public section

