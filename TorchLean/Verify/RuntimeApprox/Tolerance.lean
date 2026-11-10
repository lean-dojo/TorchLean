/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Basic.NNReal.Defs

/-!
# Tolerance

An approximation tolerance `ApproxTol` combines an absolute tolerance in the units of the quantity,
a dimensionless relative tolerance and a multiplier for the whole budget, all nonnegative. It is
independent of any backend, IBP and FP32 included. `ApproxTol.bound` uses `max |x| |y|` as the
relative scale, so the bound is symmetric in `x` and `y`.
-/

@[expose] public section

namespace TorchLean

open scoped Real NNReal

/-- Absolute/relative tolerance with a nonnegative budget multiplier. -/
structure ApproxTol where
  /-- Absolute part of the bound, which is what keeps the tolerance meaningful near zero. -/
  abs : ℝ≥0
  /-- Relative part, multiplied by the maximum magnitude of the compared values. -/
  rel : ℝ≥0
  /-- Multiplier for the whole budget: `1` leaves it unchanged, values above `1` loosen it,
  and values below `1` tighten it. `0` requires exact agreement. -/
  slack : ℝ≥0

namespace ApproxTol

/-- Build a tolerance from reals, clamping negatives to 0 via `Real.toNNReal`. -/
def ofReal (abs rel slack : ℝ) : ApproxTol :=
  { abs := Real.toNNReal abs
    rel := Real.toNNReal rel
    slack := Real.toNNReal slack }

/-- Absolute-only tolerance (relative = 0, slack = 1). -/
def absOnly (eps : ℝ) : ApproxTol :=
  ofReal eps 0 1

end ApproxTol

/-- The error budget `slack * (abs + rel * max |x| |y|)` of a tolerance at `x` and `y`. -/
def ApproxTol.bound (t : ApproxTol) (x y : ℝ) : ℝ :=
  (t.slack : ℝ) * ((t.abs : ℝ) + (t.rel : ℝ) * max (_root_.abs x) (_root_.abs y))

/-- `ApproxR x y t`: the error `|y - x|` is at most the budget `ApproxTol.bound t x y`. -/
def ApproxR (x y : ℝ) (t : ApproxTol) : Prop :=
  abs (y - x) ≤ ApproxTol.bound t x y

/-- The budget before the slack factor, `abs + rel * max |x| |y|`, is nonnegative.

The fields are `NNReal`, so the proof is bookkeeping. `ApproxTol.bound_mono` peels off the slack
factor first and needs this part on its own. -/
theorem ApproxTol.bound_inner_nonneg (t : ApproxTol) (x y : ℝ) :
    0 ≤ (t.abs : ℝ) + (t.rel : ℝ) * max (_root_.abs x) (_root_.abs y) :=
  add_nonneg t.abs.coe_nonneg
    (mul_nonneg t.rel.coe_nonneg ((abs_nonneg x).trans (le_max_left _ _)))

/-- The error budget is nonnegative, so every value approximates itself (`approxR_refl`). -/
theorem ApproxTol.bound_nonneg (t : ApproxTol) (x y : ℝ) : 0 ≤ bound t x y := by
  unfold bound
  exact mul_nonneg t.slack.coe_nonneg (bound_inner_nonneg t x y)

/-- The budget is monotone in each tolerance field.

A composite bound can therefore be stated with one loose tolerance that dominates the tolerances
its sub-proofs produce. -/
theorem ApproxTol.bound_mono {t₁ t₂ : ApproxTol} (habs : t₁.abs ≤ t₂.abs) (hrel : t₁.rel ≤ t₂.rel)
    (hslack : t₁.slack ≤ t₂.slack) (x y : ℝ) :
    bound t₁ x y ≤ bound t₂ x y := by
  have hmax : 0 ≤ max (_root_.abs x) (_root_.abs y) := (abs_nonneg x).trans (le_max_left _ _)
  have hinner : (t₁.abs : ℝ) + (t₁.rel : ℝ) * max (_root_.abs x) (_root_.abs y) ≤
      (t₂.abs : ℝ) + (t₂.rel : ℝ) * max (_root_.abs x) (_root_.abs y) :=
    add_le_add (by exact_mod_cast habs) (mul_le_mul_of_nonneg_right (by exact_mod_cast hrel) hmax)
  unfold bound
  refine (mul_le_mul_of_nonneg_right ?_ (bound_inner_nonneg t₁ x y)).trans
    (mul_le_mul_of_nonneg_left hinner t₂.slack.coe_nonneg)
  exact_mod_cast hslack

/-- Approximation is preserved when the tolerance is weakened. -/
theorem ApproxR.mono {x y : ℝ} {t₁ t₂ : ApproxTol} (habs : t₁.abs ≤ t₂.abs) (hrel : t₁.rel ≤ t₂.rel)
    (hslack : t₁.slack ≤ t₂.slack) (h : ApproxR x y t₁) : ApproxR x y t₂ :=
  le_trans h (ApproxTol.bound_mono (t₁ := t₁) (t₂ := t₂) habs hrel hslack x y)

/-- An absolute-only budget is the constant `Real.toNNReal eps`, whatever the compared values. -/
@[simp] theorem ApproxTol.bound_absOnly (eps x y : ℝ) :
    bound (ApproxTol.absOnly eps) x y = (Real.toNNReal eps : ℝ) := by
  simp [bound, ApproxTol.absOnly, ApproxTol.ofReal]

/-- For a nonnegative `eps`, absolute-only approximation is plain `|y - x| ≤ eps`.

The hypothesis is needed because `ApproxTol.absOnly` stores `Real.toNNReal eps`, which clamps a
negative `eps` to zero: then `ApproxR x x (ApproxTol.absOnly eps)` holds while `|x - x| ≤ eps`
fails. -/
theorem approxR_absOnly_iff {x y eps : ℝ} (heps : 0 ≤ eps) :
    ApproxR x y (ApproxTol.absOnly eps) ↔ abs (y - x) ≤ eps := by
  simp [ApproxR, ApproxTol.bound_absOnly, Real.coe_toNNReal eps heps]

/-- For nonnegative `eps₁` and `eps₂`, absolute errors add along a chain, by the triangle
inequality.

Composite runtime bounds have this shape: each layer contributes its own `eps`, and the network's
error is the sum. The relative term has no rule this clean, so the composition lemmas are
absolute-only. -/
theorem ApproxR.absOnly_trans {x y z eps₁ eps₂ : ℝ} (h₁ : 0 ≤ eps₁) (h₂ : 0 ≤ eps₂)
    (hxy : ApproxR x y (ApproxTol.absOnly eps₁)) (hyz : ApproxR y z (ApproxTol.absOnly eps₂)) :
    ApproxR x z (ApproxTol.absOnly (eps₁ + eps₂)) := by
  rw [approxR_absOnly_iff h₁] at hxy
  rw [approxR_absOnly_iff h₂] at hyz
  rw [approxR_absOnly_iff (add_nonneg h₁ h₂)]
  linarith [abs_sub_le z y x]

/-- Every value approximates itself, at any tolerance. -/
@[simp] theorem approxR_refl (x : ℝ) (t : ApproxTol) : ApproxR x x t := by
  simpa [ApproxR] using ApproxTol.bound_nonneg t x x

/-- The relation is symmetric because its scale `max |x| |y|` treats both arguments alike.

Scaling by one argument would make the relation asymmetric and force every proof to fix which side
is the reference value. -/
theorem approxR_comm (x y : ℝ) (t : ApproxTol) : ApproxR x y t ↔ ApproxR y x t := by
  constructor <;> intro h <;> simpa [ApproxR, ApproxTol.bound, abs_sub_comm, max_comm] using h

/-! ## Notation

Use `open scoped ApproxTol` to enable:

`x ≈[t] y`  meaning: `ApproxR x y t`.
-/

scoped[TorchLean.ApproxTol] notation:50 x " ≈[" t "] " y => ApproxR x y t

end TorchLean
