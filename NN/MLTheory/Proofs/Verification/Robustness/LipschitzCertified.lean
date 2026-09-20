/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import Mathlib.Tactic.Positivity.Finset
public import NN.Proofs.Analysis.Lipschitz.Norm

/-!
# Lipschitz-based robustness lemmas

This file connects Lipschitz continuity assumptions to the robustness specifications from
`NN.MLTheory.Robustness.Spec`.

Main results (over $\mathbb{R}$):

* An $L$-Lipschitz map is adversarially robust: $\varepsilon$ input perturbations imply
  $L\varepsilon$ output
  perturbations.
* If a logits vector has a positive logit margin, then its `argmax` classifier is stable under
  sufficiently small $\ell_\infty$ perturbations; combined with an output-$\ell_\infty$ Lipschitz
  bound, this yields
  certified robustness radii.
-/

@[expose] public section

namespace NN.MLTheory.Proofs.Verification.Robustness

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open NN.MLTheory.Robustness.Spec

/-! ## Lipschitz continuity implies adversarial robustness -/

/--
If $f$ is $L$-Lipschitz and $L\geq 0$, then $f$ is adversarially robust at $x_0$:
every input within distance $\varepsilon$ of $x_0$ maps within distance $L\varepsilon$ of $f(x_0)$.
-/
theorem is_adversarially_robust_of_lipschitz
    {s₁ s₂ : Shape}
    {f : Tensor ℝ s₁ → Tensor ℝ s₂}
    {norm₁ norm₂ : ∀ {s : Shape}, Tensor ℝ s → ℝ}
    {L : ℝ} (hL : 0 ≤ L)
    (hLip : isLipschitzContinuous f norm₁ norm₂ L)
    (x₀ : Tensor ℝ s₁) (ε : ℝ) :
    IsAdversariallyRobust f norm₁ norm₂ x₀ ε (L * ε) := by
  intro x hx
  have h1 : tensorDistance norm₂ (f x₀) (f x) ≤ L * tensorDistance norm₁ x₀ x :=
    hLip x₀ x
  have h2 : L * tensorDistance norm₁ x₀ x ≤ L * ε :=
    mul_le_mul_of_nonneg_left hx hL
  exact le_trans h1 h2

/-! ## Bridging $\ell_2$ and $\ell_\infty$ Lipschitz predicates -/

/--
$\ell_2$-Lipschitz implies $\ell_\infty$-Lipschitz into logits, with the same constant.

This is the standard norm comparison $\lVert v\rVert_\infty\leq\lVert v\rVert_2$ (proved as
`Proofs.tensor_linf_norm_le_tensor_l2_norm`).
-/
theorem is_lipschitz_continuous_linf_of_l2
    {s₁ : Shape} {n : Nat}
    {f : Tensor ℝ s₁ → Tensor ℝ [n]}
    {normIn : ∀ {s : Shape}, Tensor ℝ s → ℝ}
    {L : ℝ}
    (hLip2 : isLipschitzContinuous f normIn Proofs.tensorL2Norm L) :
    isLipschitzContinuous f normIn (tensorLinfNorm (α := ℝ)) L := by
  intro x y
  have hnorm :
      tensorDistance (α := ℝ) (tensorLinfNorm (α := ℝ)) (f x) (f y) ≤
        tensorDistance (α := ℝ) Proofs.tensorL2Norm (f x) (f y) := by
    simpa [tensorDistance] using
      (Proofs.tensor_linf_norm_le_tensor_l2_norm
        (y := tensorDistance.tensorSub (α := ℝ) (f x) (f y)))
  exact le_trans hnorm (hLip2 x y)

/-! ## Argmax stability from a positive logit margin -/

/--
The `argmax` classifier on logits vectors, breaking ties by the earliest index.

When $n=0$, this returns $0$ by convention.
-/
noncomputable def argmaxClassifier {n : Nat} (y : Tensor ℝ [n]) : Nat :=
  match List.argmax (fun i : Fin n => Tensor.getScalar y i) (List.finRange n) with
  | some i => i.val
  | none => 0

/--
`HasLogitMargin y c m` means class $c$ beats every competitor by at least $m$ in logit value.
-/
def HasLogitMargin {n : Nat} (y : Tensor ℝ [n]) (c : Fin n) (m : ℝ) : Prop :=
  ∀ k : Fin n, k ≠ c → Tensor.getScalar y k ≤ Tensor.getScalar y c - m

private theorem abs_getScalar_le_tensor_linf_norm {n : Nat} (y : Tensor ℝ [n]) (i : Fin n) :
    |Tensor.getScalar y i| ≤ tensorLinfNorm (α := ℝ) y := by
  have hi : i ∈ List.finRange n := List.mem_finRange i
  have hle :
      tensorLinfNorm (α := ℝ) (y.unstack i) ≤
        (List.finRange n).foldl
          (fun acc j => max acc (tensorLinfNorm (α := ℝ) (y.unstack j))) 0 :=
    List.le_foldl_max_of_mem (List.finRange n)
      (fun j => tensorLinfNorm (α := ℝ) (y.unstack j)) (acc := (0 : ℝ)) hi
  simpa [Tensor.getScalar, Spec.get, tensorLinfNorm, MathFunctions.abs] using hle

private theorem abs_getScalar_sub_le_of_linf_distance
    {n : Nat} {y₀ y : Tensor ℝ [n]} {δ : ℝ} (i : Fin n)
    (h : tensorDistance (α := ℝ) (tensorLinfNorm (α := ℝ)) y₀ y ≤ δ) :
    |Tensor.getScalar y₀ i - Tensor.getScalar y i| ≤ δ := by
  -- `tensor_distance` is the `L∞` norm of the entrywise difference.
  have hcoord :
      |Tensor.getScalar (tensorDistance.tensorSub (α := ℝ) y₀ y) i| ≤
        tensorLinfNorm (α := ℝ) (tensorDistance.tensorSub (α := ℝ) y₀ y) := by
    simpa using
      (abs_getScalar_le_tensor_linf_norm (y := tensorDistance.tensorSub (α := ℝ) y₀ y) (i := i))
  have : |Tensor.getScalar (tensorDistance.tensorSub (α := ℝ) y₀ y) i| ≤ δ :=
    le_trans hcoord (by simpa [tensorDistance] using h)
  simpa [NN.MLTheory.Robustness.Spec.tensor_distance_tensor_sub_eq_sub_spec,
    TorchLean.Tensor.subSpec] using this

private theorem le_getScalar_add_of_abs_sub_le
    {n : Nat} {y₀ y : Tensor ℝ [n]} {δ : ℝ} (i : Fin n)
    (h : |Tensor.getScalar y₀ i - Tensor.getScalar y i| ≤ δ) :
    Tensor.getScalar y i ≤ Tensor.getScalar y₀ i + δ := by
  have h' := abs_sub_le_iff.1 (by simpa [abs_sub_comm] using h)
  linarith

private theorem ge_getScalar_sub_of_abs_sub_le
    {n : Nat} {y₀ y : Tensor ℝ [n]} {δ : ℝ} (i : Fin n)
    (h : |Tensor.getScalar y₀ i - Tensor.getScalar y i| ≤ δ) :
    Tensor.getScalar y₀ i - δ ≤ Tensor.getScalar y i := by
  have h' := abs_sub_le_iff.1 (by simpa using h)
  linarith

private theorem argmax_eq_some_of_strictMax
    {n : Nat} {y : Tensor ℝ [n]} {c : Fin n}
    (hstrict : ∀ k : Fin n, k ≠ c → Tensor.getScalar y k < Tensor.getScalar y c) :
    List.argmax (fun i : Fin n => Tensor.getScalar y i) (List.finRange n) = some c := by
  classical
  have hc_mem : c ∈ List.finRange n := List.mem_finRange c
  refine (List.argmax_eq_some_iff (f := fun i : Fin n => Tensor.getScalar y i)
    (l := List.finRange n) (m := c)).2 ?_
  refine ⟨hc_mem, ?_, ?_⟩
  · intro a ha
    by_cases hEq : a = c
    · subst hEq
      exact le_rfl
    · exact le_of_lt (hstrict a hEq)
  · intro a ha hca
    by_cases hEq : a = c
    · subst hEq
      exact le_rfl
    · exact False.elim (not_le_of_gt (hstrict a hEq) hca)

/--
If $y_0$ has a positive logit margin $m$ for class $c$, then any $y$ within $\ell_\infty$
distance $\delta$ with $2\delta<m$ has the same `argmax` class.
-/
theorem argmaxClassifier_eq_of_linf_distance_lt_half_margin
    {n : Nat} {y₀ y : Tensor ℝ [n]} {c : Fin n} {m δ : ℝ}
    (hmargin : HasLogitMargin (n := n) y₀ c m)
    (hδ : 2 * δ < m)
    (hdist : tensorDistance (α := ℝ) (tensorLinfNorm (α := ℝ)) y₀ y ≤ δ) :
    argmaxClassifier (n := n) y = c.val := by
  have hstrict : ∀ k : Fin n, k ≠ c → Tensor.getScalar y k < Tensor.getScalar y c := by
    intro k hk
    have habs_k : |Tensor.getScalar y₀ k - Tensor.getScalar y k| ≤ δ :=
      abs_getScalar_sub_le_of_linf_distance (i := k) hdist
    have habs_c : |Tensor.getScalar y₀ c - Tensor.getScalar y c| ≤ δ :=
      abs_getScalar_sub_le_of_linf_distance (i := c) hdist
    have hk_le : Tensor.getScalar y k ≤ Tensor.getScalar y₀ k + δ :=
      le_getScalar_add_of_abs_sub_le (i := k) habs_k
    have hc_ge : Tensor.getScalar y₀ c - δ ≤ Tensor.getScalar y c :=
      ge_getScalar_sub_of_abs_sub_le (i := c) habs_c
    have hcomp : Tensor.getScalar y₀ k ≤ Tensor.getScalar y₀ c - m := hmargin k hk
    -- Combine inequalities; `2*δ < m` yields strictness.
    have : Tensor.getScalar y k < Tensor.getScalar y c := by
      linarith
    exact this
  have harg : List.argmax (fun i : Fin n => Tensor.getScalar y i) (List.finRange n) = some c :=
    argmax_eq_some_of_strictMax (n := n) (y := y) (c := c) hstrict
  simp [argmaxClassifier, harg]

/--
If $y_0$ has margin $m>0$ for class $c$, then $c$ is the `argmaxClassifier` of $y_0$.
-/
theorem argmaxClassifier_eq_of_hasLogitMargin
    {n : Nat} {y₀ : Tensor ℝ [n]} {c : Fin n} {m : ℝ}
    (hm : 0 < m) (hmargin : HasLogitMargin (n := n) y₀ c m) :
    argmaxClassifier (n := n) y₀ = c.val := by
  have hstrict : ∀ k : Fin n, k ≠ c → Tensor.getScalar y₀ k < Tensor.getScalar y₀ c := by
    intro k hk
    have hcomp : Tensor.getScalar y₀ k ≤ Tensor.getScalar y₀ c - m := hmargin k hk
    linarith
  have harg : List.argmax (fun i : Fin n => Tensor.getScalar y₀ i) (List.finRange n) = some c :=
    argmax_eq_some_of_strictMax (n := n) (y := y₀) (c := c) hstrict
  simp [argmaxClassifier, harg]

/-! ## Certified robustness from a Lipschitz bound and a logit margin -/

/--
If $f$ is $L$-Lipschitz into $\ell_\infty$ logits and the reference logits $f(x_0)$ have margin
$m$ for class $c$, then any input perturbation of radius $\varepsilon$ with
$2L\varepsilon<m$ preserves the predicted class.

This is a standard “margin over Lipschitz constant” certified robustness lemma.
-/
theorem is_certified_robust_of_lipschitz_of_logitMargin
    {s₁ : Shape} {n : Nat}
    {f : Tensor ℝ s₁ → Tensor ℝ [n]}
    {normIn : ∀ {s : Shape}, Tensor ℝ s → ℝ}
    {L : ℝ} (hL : 0 ≤ L)
    (hLip : isLipschitzContinuous f normIn (tensorLinfNorm (α := ℝ)) L)
    {x₀ : Tensor ℝ s₁} {ε m : ℝ} {c : Fin n}
    (hRadius : 0 ≤ ε)
    (hm : 0 < m)
    (hmargin : HasLogitMargin (n := n) (f x₀) c m)
    (hε : 2 * (L * ε) < m) :
    IsCertifiedRobust (classifier := fun x => argmaxClassifier (n := n) (f x))
      (norm := normIn) x₀ ε := by
  refine ⟨hRadius, ?_⟩
  intro x hx
  have hx' : tensorDistance (α := ℝ) normIn x₀ x ≤ ε := by
    simpa [tensorBall] using hx
  have hdist₁ : tensorDistance (α := ℝ) (tensorLinfNorm (α := ℝ)) (f x₀) (f x) ≤
      L * tensorDistance (α := ℝ) normIn x₀ x :=
    hLip x₀ x
  have hdist₂ : L * tensorDistance (α := ℝ) normIn x₀ x ≤ L * ε :=
    mul_le_mul_of_nonneg_left hx' hL
  have hdist : tensorDistance (α := ℝ) (tensorLinfNorm (α := ℝ)) (f x₀) (f x) ≤ L * ε :=
    le_trans hdist₁ hdist₂
  have hx_class : argmaxClassifier (n := n) (f x) = c.val :=
    argmaxClassifier_eq_of_linf_distance_lt_half_margin (n := n) (y₀ := f x₀) (y := f x)
      (c := c) (m := m) (δ := L * ε) hmargin hε hdist
  have hx0_class : argmaxClassifier (n := n) (f x₀) = c.val :=
    argmaxClassifier_eq_of_hasLogitMargin (n := n) (y₀ := f x₀) (c := c) (m := m) hm hmargin
  calc
    argmaxClassifier (n := n) (f x) = c.val := hx_class
    _ = argmaxClassifier (n := n) (f x₀) := by simp [hx0_class]

/--
Certified robustness, but starting from an output-$\ell_2$ Lipschitz assumption.

This avoids requiring the user to manually insert the norm-equivalence step
$\lVert\cdot\rVert_\infty\leq\lVert\cdot\rVert_2$.
-/
  theorem is_certified_robust_of_l2_lipschitz_of_logitMargin
    {s₁ : Shape} {n : Nat}
    {f : Tensor ℝ s₁ → Tensor ℝ [n]}
    {normIn : ∀ {s : Shape}, Tensor ℝ s → ℝ}
    {L : ℝ} (hL : 0 ≤ L)
    (hLip2 : isLipschitzContinuous f normIn Proofs.tensorL2Norm L)
    {x₀ : Tensor ℝ s₁} {ε m : ℝ} {c : Fin n}
    (hRadius : 0 ≤ ε)
    (hm : 0 < m)
    (hmargin : HasLogitMargin (n := n) (f x₀) c m)
    (hε : 2 * (L * ε) < m) :
    IsCertifiedRobust (classifier := fun x => argmaxClassifier (n := n) (f x))
      (norm := normIn) x₀ ε := by
  have hLipLinf : isLipschitzContinuous f normIn (tensorLinfNorm (α := ℝ)) L :=
    is_lipschitz_continuous_linf_of_l2 (hLip2 := hLip2)
  exact
    is_certified_robust_of_lipschitz_of_logitMargin (n := n) (hL := hL) (hLip := hLipLinf)
      (x₀ := x₀) (ε := ε) (m := m) (c := c) hRadius hm hmargin hε

end NN.MLTheory.Proofs.Verification.Robustness
