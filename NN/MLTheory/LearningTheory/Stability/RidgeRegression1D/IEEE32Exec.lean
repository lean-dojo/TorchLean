/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.LearningTheory.Stability.RidgeRegression1D.IEEE32Exec.Core
public import NN.MLTheory.LearningTheory.Stability.RidgeRegression1D.IEEE32Exec.ExampleDataset

/-!
# 1D ridge regression under `IEEE32Exec` (option A bridge)

This module provides an *executable* 1D ridge-regression implementation using the bit-level
IEEE-754 binary32 kernel `IEEE32Exec`, and a refinement statement to the proof-level `FP32`
rounding model.

Implementation details live in:

- `NN.MLTheory.LearningTheory.Stability.RidgeRegression1D.IEEE32Exec.Core` (definitions + bridge),
- `NN.MLTheory.LearningTheory.Stability.RidgeRegression1D.IEEE32Exec.ExampleDataset`
  (small concrete dataset).

This is the “Option A” bridge:

1. Prove the learning-theory theorem over `ℝ` (see
  `NN.MLTheory.LearningTheory.Stability.RidgeRegression1D`).
2. Provide an executable `IEEE32Exec` implementation.
3. Connect `IEEE32Exec` execution to a proof-level float32 rounding semantics (`FP32`) via a bridge.

## Scope of this bridge

This file proves the executable-to-rounding-semantics bridge. A full floating-point stability
theorem additionally needs hypotheses and a numerical analysis layer bounding the gap between:

- the real-valued ridge solution `ŵ : ℝ`, and
- the computed float32 value produced by `IEEE32Exec`.

Instead, this file establishes the first correctness link in that pipeline:

> *the executable IEEE32 run agrees with a proof-level “round-after-each-primitive” semantics*,
> provided the evaluation stays finite (no NaN/Inf/div-by-zero along the way).

The bridge lemma is local: it shows that `toReal` of the executable IEEE run agrees with the
`FP32`-style “round-after-each-primitive” semantics, under an explicit finiteness assumption
(`FiniteEval`) ruling out NaN/Inf and division-by-zero.

## Why this file exists

The learning-theory theorem is about an idealized real-valued algorithm. In practice, we run the
algorithm in finite precision. This file is the first piece of the numerics bridge:

- implement the algorithm using an *executable* IEEE-754 kernel, and
- relate it to a proof-level rounding model via an explicit bridge lemma.

## Datasets

This file reuses the stability dataset type from `Stability.Core`:

`Dataset N Z := Spec.Vec N Z`

so our executable ridge regression runs on the same tensor-based dataset representation as the rest
of the learning-theory layer.

## References

- IEEE floating point: IEEE Std 754-2019.
- Floating-point pitfalls/background: Goldberg (1991), “What Every Computer Scientist Should Know
  About Floating-Point Arithmetic”.
- For a broader modern reference on floating-point and rounding analysis, see:
  Muller et al. (2018), “Handbook of Floating-Point Arithmetic” (2nd ed.).
-/

@[expose] public section
