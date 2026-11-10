/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Numeric.InverseTrig
public import Mathlib.Analysis.SpecialFunctions.Complex.Arg
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Arctan

/-!
# Exact real inverse trigonometric functions

The `InverseTrig ℝ` instance takes the polar angle from Mathlib's principal complex argument in
`(-π, π]`, which is zero at the origin, and uses `Real.arctan`, `Real.arcsin` and `Real.arccos`.
Outside `[-1, 1]` the last two return Mathlib's clamped values. The instance has its own module so
that the executable scalar interfaces do not import complex analysis.

Compared with C99's Annex F (WG14 N1256), the zero at the origin is C's `atan2 (+0, +0)`
(F.9.1.4), and the `±π` that C returns against `-0` has no real counterpart, since a real zero
carries no sign. The clamp gives `arcsin 2 = π/2` (`Real.arcsin_of_one_le`), where C's `asin`
returns a NaN (F.9.1.2).
-/

@[expose] public section

namespace TorchLean

/-- The real polar angle and inverse trigonometric functions. -/
noncomputable instance : InverseTrig ℝ :=
  ⟨fun y x => _root_.Complex.arg ⟨x, y⟩, Real.arctan, Real.arcsin, Real.arccos⟩

end TorchLean
