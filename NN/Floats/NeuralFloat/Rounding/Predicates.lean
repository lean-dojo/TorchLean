/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.Real.Basic

/-!
# Rounding Predicates

Semantic specifications for directed, toward-zero, and nearest rounding.  They are independent of
any radix or concrete format: a predicate `F : ℝ → Prop` identifies the representable values, and
the point predicates characterize the required output among those values.

These definitions correspond to Flocq's `Rnd_DN_pt`, `Rnd_UP_pt`, `Rnd_ZR_pt`, and `Rnd_N_pt`.
-/

@[expose] public section

namespace TorchLean.Floats

/-- `f` is the greatest `F`-value no larger than `x`. -/
def NeuralRoundDownPoint (F : ℝ → Prop) (x f : ℝ) : Prop :=
  F f ∧ f ≤ x ∧ ∀ g, F g → g ≤ x → g ≤ f

/-- `f` is the least `F`-value no smaller than `x`. -/
def NeuralRoundUpPoint (F : ℝ → Prop) (x f : ℝ) : Prop :=
  F f ∧ x ≤ f ∧ ∀ g, F g → x ≤ g → f ≤ g

/-- Toward-zero rounding uses directed down on nonnegative inputs and directed up on negative ones. -/
def NeuralRoundTowardZeroPoint (F : ℝ → Prop) (x f : ℝ) : Prop :=
  (0 ≤ x → NeuralRoundDownPoint F x f) ∧ (x ≤ 0 → NeuralRoundUpPoint F x f)

/-- `f` is an `F`-value at least as close to `x` as every other representable value. -/
def NeuralRoundNearestPoint (F : ℝ → Prop) (x f : ℝ) : Prop :=
  F f ∧ ∀ g, F g → abs (f - x) ≤ abs (g - x)

/-- A downward rounding point is unique. -/
theorem neuralRoundDownPoint_unique {F : ℝ → Prop} {x f g : ℝ}
    (hf : NeuralRoundDownPoint F x f) (hg : NeuralRoundDownPoint F x g) : f = g := by
  exact le_antisymm (hg.2.2 f hf.1 hf.2.1) (hf.2.2 g hg.1 hg.2.1)

/-- An upward rounding point is unique. -/
theorem neuralRoundUpPoint_unique {F : ℝ → Prop} {x f g : ℝ}
    (hf : NeuralRoundUpPoint F x f) (hg : NeuralRoundUpPoint F x g) : f = g := by
  exact le_antisymm (hf.2.2 g hg.1 hg.2.1) (hg.2.2 f hf.1 hf.2.1)

/-- A representable value is its own downward rounding point. -/
theorem neuralRoundDownPoint_refl {F : ℝ → Prop} {x : ℝ} (hx : F x) :
    NeuralRoundDownPoint F x x := by
  exact ⟨hx, le_rfl, fun _ _ hg => hg⟩

/-- A representable value is its own upward rounding point. -/
theorem neuralRoundUpPoint_refl {F : ℝ → Prop} {x : ℝ} (hx : F x) :
    NeuralRoundUpPoint F x x := by
  exact ⟨hx, le_rfl, fun _ _ hg => hg⟩

/-- Negation turns a downward point into an upward point for a symmetric format. -/
theorem neuralRoundUpPoint_neg {F : ℝ → Prop}
    (hneg : ∀ x, F x → F (-x)) {x f : ℝ} (hf : NeuralRoundDownPoint F x f) :
    NeuralRoundUpPoint F (-x) (-f) := by
  refine ⟨hneg f hf.1, neg_le_neg hf.2.1, ?_⟩
  intro g hg hxg
  have hng : F (-g) := hneg g hg
  have hngx : -g ≤ x := by simpa using neg_le_neg hxg
  simpa using neg_le_neg (hf.2.2 (-g) hng hngx)

/-- Negation turns an upward point into a downward point for a symmetric format. -/
theorem neuralRoundDownPoint_neg {F : ℝ → Prop}
    (hneg : ∀ x, F x → F (-x)) {x f : ℝ} (hf : NeuralRoundUpPoint F x f) :
    NeuralRoundDownPoint F (-x) (-f) := by
  refine ⟨hneg f hf.1, neg_le_neg hf.2.1, ?_⟩
  intro g hg hgx
  have hng : F (-g) := hneg g hg
  have hxng : x ≤ -g := by simpa using neg_le_neg hgx
  simpa using neg_le_neg (hf.2.2 (-g) hng hxng)

/-- Any representable value lies below the downward point or above the upward point. -/
theorem neuralRoundDownUpPoint_split {F : ℝ → Prop} {x d u f : ℝ}
    (hd : NeuralRoundDownPoint F x d) (hu : NeuralRoundUpPoint F x u) (hf : F f) :
    f ≤ d ∨ u ≤ f := by
  rcases le_total f x with hfx | hxf
  · exact Or.inl (hd.2.2 f hf hfx)
  · exact Or.inr (hu.2.2 f hf hxf)

/--
A representable value no farther than both directed neighbors is globally nearest.  Every other
representable value lies outside the interval between those neighbors.
-/
theorem neuralRoundNearestPoint_of_down_up {F : ℝ → Prop} {x d u f : ℝ}
    (hd : NeuralRoundDownPoint F x d) (hu : NeuralRoundUpPoint F x u)
    (hf : F f) (hfd : abs (f - x) ≤ abs (d - x))
    (hfu : abs (f - x) ≤ abs (u - x)) : NeuralRoundNearestPoint F x f := by
  refine ⟨hf, ?_⟩
  intro g hg
  rcases neuralRoundDownUpPoint_split hd hu hg with hgd | hug
  · calc
      abs (f - x) ≤ abs (d - x) := hfd
      _ = x - d := by simpa [neg_sub] using abs_of_nonpos (sub_nonpos.mpr hd.2.1)
      _ ≤ x - g := sub_le_sub_left hgd x
      _ = abs (g - x) := by
        simpa [neg_sub] using
          (abs_of_nonpos (sub_nonpos.mpr (hgd.trans hd.2.1))).symm
  · calc
      abs (f - x) ≤ abs (u - x) := hfu
      _ = u - x := abs_of_nonneg (sub_nonneg.mpr hu.2.1)
      _ ≤ g - x := sub_le_sub_right hug x
      _ = abs (g - x) := (abs_of_nonneg (sub_nonneg.mpr (hu.2.1.trans hug))).symm

end TorchLean.Floats
