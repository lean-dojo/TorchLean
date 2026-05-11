/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.NNReal.Defs

/-!
# Tolerance

Approximation tolerances (absolute + relative).

This file defines a small, reusable tolerance object for "close enough" reasoning:

- absolute tolerance (units of the quantity),
- relative tolerance (dimensionless), and
- a nonnegative slack factor to scale the budget.

It is intentionally lightweight and independent of any specific backend (IBP/FP32/etc.).

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

/-- Absolute/relative tolerance with an extra nonnegative slack factor. -/
structure ApproxTol where
  /-- abs. -/
  abs : ℝ≥0
  /-- rel. -/
  rel : ℝ≥0
  /-- slack. -/
  slack : ℝ≥0

namespace ApproxTol

/-- Build a tolerance from reals, clamping negatives to 0 via `Real.toNNReal`. -/
def ofReal (abs rel slack : ℝ) : ApproxTol :=
  { abs := Real.toNNReal abs
    rel := Real.toNNReal rel
    slack := Real.toNNReal slack }

/-- Default slack = 1. -/
def ofReal' (abs rel : ℝ) : ApproxTol :=
  ofReal abs rel 1

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

lemma approxBound_inner_nonneg (t : ApproxTol) (x y : ℝ) :
    0 ≤ (t.abs : ℝ) + (t.rel : ℝ) * max (abs x) (abs y) := by
  nlinarith [t.abs.coe_nonneg, t.rel.coe_nonneg, le_max_left (abs x) (abs y),
    le_max_right (abs x) (abs y), abs_nonneg x, abs_nonneg y]

lemma approxBound_nonneg (t : ApproxTol) (x y : ℝ) : 0 ≤ approxBound t x y := by
  have hinner : 0 ≤ (t.abs : ℝ) + (t.rel : ℝ) * max (abs x) (abs y) :=
    approxBound_inner_nonneg t x y
  have : 0 ≤ (t.slack : ℝ) * ((t.abs : ℝ) + (t.rel : ℝ) * max (abs x) (abs y)) :=
    mul_nonneg t.slack.coe_nonneg hinner
  simpa [approxBound] using this

lemma approxBound_mono {t₁ t₂ : ApproxTol} (habs : t₁.abs ≤ t₂.abs) (hrel : t₁.rel ≤ t₂.rel)
    (hslack : t₁.slack ≤ t₂.slack) (x y : ℝ) :
    approxBound t₁ x y ≤ approxBound t₂ x y := by
  have hslack' : (t₁.slack : ℝ) ≤ (t₂.slack : ℝ) := by exact_mod_cast hslack
  have habs' : (t₁.abs : ℝ) ≤ (t₂.abs : ℝ) := by exact_mod_cast habs
  have hrel' : (t₁.rel : ℝ) ≤ (t₂.rel : ℝ) := by exact_mod_cast hrel
  have hmax : 0 ≤ max (abs x) (abs y) := by
    exact le_trans (abs_nonneg x) (le_max_left _ _)
  have hinner_nonneg : 0 ≤ (t₁.abs : ℝ) + (t₁.rel : ℝ) * max (abs x) (abs y) :=
    approxBound_inner_nonneg t₁ x y
  have hinner : (t₁.abs : ℝ) + (t₁.rel : ℝ) * max (abs x) (abs y) ≤
      (t₂.abs : ℝ) + (t₂.rel : ℝ) * max (abs x) (abs y) := by
    have hmul : (t₁.rel : ℝ) * max (abs x) (abs y) ≤ (t₂.rel : ℝ) * max (abs x) (abs y) :=
      mul_le_mul_of_nonneg_right hrel' hmax
    exact add_le_add habs' hmul
  have h1 :
      (t₁.slack : ℝ) * ((t₁.abs : ℝ) + (t₁.rel : ℝ) * max (abs x) (abs y)) ≤
        (t₂.slack : ℝ) * ((t₁.abs : ℝ) + (t₁.rel : ℝ) * max (abs x) (abs y)) :=
    mul_le_mul_of_nonneg_right hslack' hinner_nonneg
  have h2 :
      (t₂.slack : ℝ) * ((t₁.abs : ℝ) + (t₁.rel : ℝ) * max (abs x) (abs y)) ≤
        (t₂.slack : ℝ) * ((t₂.abs : ℝ) + (t₂.rel : ℝ) * max (abs x) (abs y)) :=
    mul_le_mul_of_nonneg_left hinner t₂.slack.coe_nonneg
  exact le_trans (by simpa [approxBound] using h1) (by simpa [approxBound] using h2)

lemma approxR_mono {x y : ℝ} {t₁ t₂ : ApproxTol} (habs : t₁.abs ≤ t₂.abs) (hrel : t₁.rel ≤ t₂.rel)
    (hslack : t₁.slack ≤ t₂.slack) (h : approxR x y t₁) : approxR x y t₂ :=
  le_trans h (approxBound_mono (t₁ := t₁) (t₂ := t₂) habs hrel hslack x y)

@[simp] lemma approxBound_absOnly (eps x y : ℝ) :
    approxBound (ApproxTol.absOnly eps) x y = (Real.toNNReal eps : ℝ) := by
  simp [approxBound, ApproxTol.absOnly, ApproxTol.ofReal]

lemma approxR_absOnly_iff {x y eps : ℝ} (heps : 0 ≤ eps) :
    approxR x y (ApproxTol.absOnly eps) ↔ abs (y - x) ≤ eps := by
  have hcoe : (Real.toNNReal eps : ℝ) = eps := by
    simp [Real.toNNReal_of_nonneg heps]
  simp [approxR, approxBound_absOnly, hcoe]

lemma approxR_absOnly_trans {x y z eps₁ eps₂ : ℝ} (h₁ : 0 ≤ eps₁) (h₂ : 0 ≤ eps₂)
    (hxy : approxR x y (ApproxTol.absOnly eps₁)) (hyz : approxR y z (ApproxTol.absOnly eps₂)) :
    approxR x z (ApproxTol.absOnly (eps₁ + eps₂)) := by
  have hxy' : abs (y - x) ≤ eps₁ := (approxR_absOnly_iff (x := x) (y := y) (eps := eps₁) h₁).1 hxy
  have hyz' : abs (z - y) ≤ eps₂ := (approxR_absOnly_iff (x := y) (y := z) (eps := eps₂) h₂).1 hyz
  have hzx : abs (z - x) ≤ eps₁ + eps₂ := by
    have := abs_sub_le z y x
    -- `|z - x| ≤ |z - y| + |y - x|`.
    exact le_trans this (by linarith)
  have h12 : 0 ≤ eps₁ + eps₂ := add_nonneg h₁ h₂
  exact (approxR_absOnly_iff (x := x) (y := z) (eps := eps₁ + eps₂) h12).2 (by simpa [abs_sub_comm]
    using hzx)

@[simp] lemma approxR_refl (x : ℝ) (t : ApproxTol) : approxR x x t := by
  have hinner : 0 ≤ (t.abs : ℝ) + (t.rel : ℝ) * abs x := by
    nlinarith [t.abs.coe_nonneg, t.rel.coe_nonneg, abs_nonneg x]
  have hbound : 0 ≤ approxBound t x x := by
    have : 0 ≤ (t.slack : ℝ) * ((t.abs : ℝ) + (t.rel : ℝ) * abs x) :=
      mul_nonneg t.slack.coe_nonneg hinner
    -- `max (abs x) (abs x) = abs x`
    simpa [approxBound, max_self] using this
  simpa [approxR, sub_self] using hbound

lemma approxR_symm (x y : ℝ) (t : ApproxTol) : approxR x y t ↔ approxR y x t := by
  constructor <;> intro h
  · simpa [approxR, approxBound, abs_sub_comm, max_comm] using h
  · simpa [approxR, approxBound, abs_sub_comm, max_comm] using h

/-! ## Notation

Use `open scoped ApproxTol` to enable:

`x ≈[t] y`  meaning: `approxR x y t`.
-/

scoped[ApproxTol] notation:50 x " ≈[" t "] " y => Proofs.RuntimeApprox.approxR x y t

end RuntimeApprox
end Proofs
