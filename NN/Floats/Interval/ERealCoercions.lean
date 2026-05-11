/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.EReal.Basic
public import Mathlib.Data.Real.Basic

/-!
# Coercion lemmas for `ℝ → EReal`

Several IEEE32Exec interval soundness proofs move between real bounds (proved in `ℝ`) and
overflow-safe endpoint reasoning (done in `EReal`).

This file centralizes small “coe distributes over min/max” rewrite lemmas for the interval
soundness modules.

We do **not** mark these as simp lemmas: they are intended for targeted rewriting in interval
proofs, and we want to avoid surprising `simp` behavior in unrelated developments.
-/

@[expose] public section

namespace TorchLean.Floats.Interval

/-- Coercion distributes over `min` for reals embedded into `EReal`. -/
theorem coe_min (a b : ℝ) : ((min a b : ℝ) : EReal) = min (a : EReal) (b : EReal) := by
  by_cases h : a ≤ b
  · have hE : (a : EReal) ≤ (b : EReal) := by
      simpa [EReal.coe_le_coe_iff] using h
    simp [min_eq_left h, min_eq_left hE]
  · have h' : b ≤ a := le_of_not_ge h
    have hE : (b : EReal) ≤ (a : EReal) := by
      simpa [EReal.coe_le_coe_iff] using h'
    simp [min_eq_right h', min_eq_right hE]

/-- Coercion distributes over `max` for reals embedded into `EReal`. -/
theorem coe_max (a b : ℝ) : ((max a b : ℝ) : EReal) = max (a : EReal) (b : EReal) := by
  by_cases h : a ≤ b
  · have hE : (a : EReal) ≤ (b : EReal) := by
      simpa [EReal.coe_le_coe_iff] using h
    simp [max_eq_right h, max_eq_right hE]
  · have h' : b ≤ a := le_of_not_ge h
    have hE : (b : EReal) ≤ (a : EReal) := by
      simpa [EReal.coe_le_coe_iff] using h'
    simp [max_eq_left h', max_eq_left hE]

end TorchLean.Floats.Interval
