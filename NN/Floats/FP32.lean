/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.FP32.Error
public import NN.Floats.FP32.Notation
public import NN.Floats.FP32.RuntimeApprox
import Mathlib.Algebra.Order.Algebra

/-!
# `NN.Floats.FP32`

`FP32` umbrella import.

`import NN.Floats.FP32` is what we reach for when we want “float32 semantics for proofs”:
- the canonical binary32-like rounding configuration (`fexp32`, `rnd32`, `FP32`),
- the real-level helper operators (`round32`/`ulp32`/`eps32`, plus unicode aliases),
- per-op absolute error bounds (`*_abs_error`),
- interval-style enclosure corollaries (`*_mem_Icc`).
- convenient wrappers that restate those bounds using the generic `≈[t]` tolerance relation.

Most of the implementation lives under `NN/Floats/FP32/*` to keep the code navigable.
-/

@[expose] public section
