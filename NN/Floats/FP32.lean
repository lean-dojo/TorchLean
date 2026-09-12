/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.FP32.Core
public import NN.Floats.FP32.Error
public import NN.Floats.FP32.Notation
public import NN.Floats.FP32.Sterbenz

/-!
# `NN.Floats.FP32`

`FP32` umbrella import.

`import NN.Floats.FP32` is what we reach for when we want “float32 semantics for proofs”:
- the canonical binary32-like rounding configuration (`fexp32`, `rnd32`, `FP32`),
- the real-level helper operators `round32`, `ulp32`, and `eps32`,
- per-op absolute error bounds (`*_abs_error`),
- interval-style enclosure corollaries (`*_mem_Icc`).
- exact subtraction for nearby representable operands via Sterbenz's lemma.

The TorchLean-specific `≈[t]` adapter lives in `NN.Proofs.RuntimeApprox.FP32`; it is intentionally
not part of this numerical umbrella.

Most of the implementation lives under `NN/Floats/FP32/*` to keep the code navigable.
-/

@[expose] public section
