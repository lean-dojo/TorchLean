/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.Order.AbsoluteValue.Basic
public import Mathlib.Algebra.Order.Field.Basic

/-!
# `NN.Floats.Interval.IEEEExec32Soundness`

Umbrella import for the soundness theory of `IEEE32Exec.Interval32`.

The underlying interval type lives in `NN.Floats.Interval.IEEEExec32` and is *executable*:
endpoints are `IEEE32Exec` (a bit-level binary32 kernel).

This module re-exports the main enclosure theorems for:
- addition/subtraction/negation,
- multiplication (4-corner rule),
- division/reciprocal (with a conservative whole-interval fallback on division-by-zero),
- auxiliary lemmas about `minOfFour/maxOfFour` and non-NaN behavior.

We keep these as separate files for navigation, and provide this umbrella so downstream code can
`import NN.Floats.Interval.IEEEExec32Soundness` without hunting for filenames.
-/

@[expose] public section
