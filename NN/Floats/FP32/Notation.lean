/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.Flocq

public import NN.Floats.FP32.Core
import Mathlib.Algebra.Order.Algebra

/-!
# Notation for TorchLean's FP32 model

TorchLean uses a proof-oriented float32 model (`FP32`) defined by:

- a radix $\beta=2$,
- the canonical IEEE-754 binary32 exponent function (`fexp32`), and
- round-to-nearest, ties-to-even (`rnd32`).

This file names the corresponding real-level operators:

- `round32 x`: round $x\in\mathbb{R}$ to the binary32 grid.
- `ulp32 x` and `eps32 x`: the ULP scale (and half-ULP) associated with $x$.

We keep these under `TorchLean.Floats` so they are available where float semantics are in focus,
without polluting unrelated namespaces.
-/

@[expose] public section

open FloatLib.Numerics FloatLib.Floats.Formats.Flocq

namespace TorchLean.Floats

noncomputable section

/--
Real-level binary32 rounding operator for the canonical `fexp32`/`rnd32` configuration.

This is definitionally the same rounding operator used in the `NF`/`FP32` semantics, but phrased as
a function $\mathbb{R}\to\mathbb{R}$ (useful for bridge theorems and error bounds).
-/
noncomputable abbrev round32 (x : ℝ) : ℝ :=
  round (β := binaryRadix) (fexp := fexp32) rnd32 x

/-- One ULP at `x` for the canonical binary32 exponent configuration. -/
noncomputable abbrev ulp32 (x : ℝ) : ℝ :=
  ulp binaryRadix fexp32 x

/-- Convenience abbreviation: half an ULP at `x`. -/
noncomputable abbrev eps32 (x : ℝ) : ℝ := ulp32 x / 2

/-- Binary32 has a smallest grid step, so its ULP at zero is $2^{-149}$. -/
@[simp] theorem ulp32_zero : ulp32 0 = bpow binaryRadix (-149) := by
  exact ulp_zero_FLT (-149) 24 (by norm_num)

end

end TorchLean.Floats
