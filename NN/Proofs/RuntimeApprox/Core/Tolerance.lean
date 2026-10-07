/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Basic.NNReal.Defs

/-!
# Tolerance

Approximation tolerances (absolute + relative).

This file defines a small, reusable tolerance object for "close enough" reasoning:

- absolute tolerance (units of the quantity),
- relative tolerance (dimensionless), and
- a nonnegative multiplier for the whole budget.

It is small and explicit and independent of any specific backend (IBP/FP32/etc.).

## PyTorch correspondence / citations
PyTorch (and NumPy) commonly expose *absolute* + *relative* tolerances (often called `atol`/`rtol`)
in APIs like `torch.allclose` / `torch.testing.assert_allclose`. Our `approxBound` follows the same
pattern, but uses `max |x| |y|` as the relative scale so the bound is symmetric in `x` and `y`.
https://pytorch.org/docs/stable/generated/torch.allclose.html
https://pytorch.org/docs/stable/testing.html
-/

@[expose] public section

namespace Proofs
namespace RuntimeApprox

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

/-- Scalar abs+rel error budget using `max |x| |y|` as the scale. -/
def approxBound (t : ApproxTol) (x y : ℝ) : ℝ :=
  (t.slack : ℝ) * ((t.abs : ℝ) + (t.rel : ℝ) * max (abs x) (abs y))

/-- Scalar approximation under an abs+rel tolerance. -/
def approxR (x y : ℝ) (t : ApproxTol) : Prop :=
  abs (y - x) ≤ approxBound t x y

/-- The absolute-plus-relative part of the budget is never negative.

All three tolerance fields are `NNReal`, so this is really just bookkeeping, but it is needed
separately from `approxBound_nonneg` because the slack factor is peeled off first in the
monotonicity proof below. -/
theorem approxBound_inner_nonneg (t : ApproxTol) (x y : ℝ) :
    0 ≤ (t.abs : ℝ) + (t.rel : ℝ) * max (abs x) (abs y) :=
  add_nonneg t.abs.coe_nonneg
    (mul_nonneg t.rel.coe_nonneg ((abs_nonneg x).trans (le_max_left _ _)))

/-- The full error budget is never negative, so `approxR` is always satisfiable at equality. -/
theorem approxBound_nonneg (t : ApproxTol) (x y : ℝ) : 0 ≤ approxBound t x y := by
  unfold approxBound
  exact mul_nonneg t.slack.coe_nonneg (approxBound_inner_nonneg t x y)

/-- The budget grows when any of the three tolerance fields grows.

Monotonicity in every field is what lets a composite bound be stated with one loose tolerance rather
than tracking the exact tolerance each sub-proof happened to produce. -/
theorem approxBound_mono {t₁ t₂ : ApproxTol} (habs : t₁.abs ≤ t₂.abs) (hrel : t₁.rel ≤ t₂.rel)
    (hslack : t₁.slack ≤ t₂.slack) (x y : ℝ) :
    approxBound t₁ x y ≤ approxBound t₂ x y := by
  have hmax : 0 ≤ max (abs x) (abs y) := (abs_nonneg x).trans (le_max_left _ _)
  have hinner : (t₁.abs : ℝ) + (t₁.rel : ℝ) * max (abs x) (abs y) ≤
      (t₂.abs : ℝ) + (t₂.rel : ℝ) * max (abs x) (abs y) :=
    add_le_add (by exact_mod_cast habs) (mul_le_mul_of_nonneg_right (by exact_mod_cast hrel) hmax)
  unfold approxBound
  refine (mul_le_mul_of_nonneg_right ?_ (approxBound_inner_nonneg t₁ x y)).trans
    (mul_le_mul_of_nonneg_left hinner t₂.slack.coe_nonneg)
  exact_mod_cast hslack

/-- Approximation is preserved when the tolerance is weakened. -/
theorem approxR_mono {x y : ℝ} {t₁ t₂ : ApproxTol} (habs : t₁.abs ≤ t₂.abs) (hrel : t₁.rel ≤ t₂.rel)
    (hslack : t₁.slack ≤ t₂.slack) (h : approxR x y t₁) : approxR x y t₂ :=
  le_trans h (approxBound_mono (t₁ := t₁) (t₂ := t₂) habs hrel hslack x y)

/-- With no relative term the budget is the constant `eps`, independent of the compared values. -/
@[simp] theorem approxBound_absOnly (eps x y : ℝ) :
    approxBound (ApproxTol.absOnly eps) x y = (Real.toNNReal eps : ℝ) := by
  simp [approxBound, ApproxTol.absOnly, ApproxTol.ofReal]

/-- For a nonnegative `eps`, absolute-only approximation is plain `|y - x| ≤ eps`.

The nonnegativity hypothesis is not decoration: `ApproxTol` stores `Real.toNNReal eps`, which clamps
a negative input to zero, and the clamped statement would be strictly stronger than intended. -/
theorem approxR_absOnly_iff {x y eps : ℝ} (heps : 0 ≤ eps) :
    approxR x y (ApproxTol.absOnly eps) ↔ abs (y - x) ≤ eps := by
  simp [approxR, approxBound_absOnly, Real.coe_toNNReal eps heps]

/-- Absolute errors add along a chain, by the triangle inequality.

Composite runtime bounds have this shape: each layer contributes its own `eps`, and the network's
error is the sum. There is no analogous clean rule for the relative term, which is why the
composition lemmas stay absolute-only. -/
theorem approxR_absOnly_trans {x y z eps₁ eps₂ : ℝ} (h₁ : 0 ≤ eps₁) (h₂ : 0 ≤ eps₂)
    (hxy : approxR x y (ApproxTol.absOnly eps₁)) (hyz : approxR y z (ApproxTol.absOnly eps₂)) :
    approxR x z (ApproxTol.absOnly (eps₁ + eps₂)) := by
  rw [approxR_absOnly_iff h₁] at hxy
  rw [approxR_absOnly_iff h₂] at hyz
  rw [approxR_absOnly_iff (add_nonneg h₁ h₂)]
  linarith [abs_sub_le z y x]

/-- Every value approximates itself, at any tolerance. -/
@[simp] theorem approxR_refl (x : ℝ) (t : ApproxTol) : approxR x x t := by
  simpa [approxR] using approxBound_nonneg t x x

/-- The relation is symmetric, because the scale is `max |x| |y|` rather than `|x|`.

That choice is deliberate. Scaling by one argument only would make the relation asymmetric and would
force every proof to fix which side is the reference value. -/
theorem approxR_symm (x y : ℝ) (t : ApproxTol) : approxR x y t ↔ approxR y x t := by
  constructor <;> intro h <;> simpa [approxR, approxBound, abs_sub_comm, max_comm] using h

/-! ## Notation

Use `open scoped ApproxTol` to enable:

`x ≈[t] y`  meaning: `approxR x y t`.
-/

scoped[ApproxTol] notation:50 x " ≈[" t "] " y => Proofs.RuntimeApprox.approxR x y t

end RuntimeApprox
end Proofs
