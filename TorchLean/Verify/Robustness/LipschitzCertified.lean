/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Lipschitz.TensorNorm

/-!
# Lipschitz-based robustness lemmas

Lipschitz continuity over $\mathbb{R}$ implies the robustness properties specified in
`Robustness`. An $L$-Lipschitz map with $L\geq 0$ sends every input within
$\varepsilon$ of $x_0$ to an output within $L\varepsilon$ of $f(x_0)$. When class $c$ leads every
other logit by a margin $m$, the `argmax` class stays $c$ under every $\ell_\infty$ perturbation
of size at most $\delta$ with $2\delta<m$. Together with an output-$\ell_\infty$ or output-$\ell_2$
Lipschitz bound, the two facts certify the radius $\varepsilon$ whenever $2L\varepsilon<m$.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean.Robustness

open _root_.TorchLean.Tensor

/-! ## Lipschitz continuity implies adversarial robustness -/

/--
If $f$ is $L$-Lipschitz and $L\geq 0$, then $f$ is adversarially robust at $x_0$:
every input within distance $\varepsilon$ of $x_0$ maps within distance $L\varepsilon$ of $f(x_0)$.
-/
theorem IsAdversariallyRobust.of_lipschitz
    {s₁ s₂ : Shape}
    {f : Tensor ℝ s₁ → Tensor ℝ s₂}
    {norm₁ norm₂ : ∀ {s : Shape}, Tensor ℝ s → ℝ}
    {L : ℝ} (hL : 0 ≤ L)
    (hLip : IsLipschitzContinuous f norm₁ norm₂ L)
    (x₀ : Tensor ℝ s₁) (ε : ℝ) :
    IsAdversariallyRobust f norm₁ norm₂ x₀ ε (L * ε) := by
  intro x hx
  have h1 : distance norm₂ (f x₀) (f x) ≤ L * distance norm₁ x₀ x :=
    hLip x₀ x
  have h2 : L * distance norm₁ x₀ x ≤ L * ε :=
    mul_le_mul_of_nonneg_left hx hL
  exact le_trans h1 h2

/-! ## Bridging $\ell_2$ and $\ell_\infty$ Lipschitz predicates -/

/--
$\ell_2$-Lipschitz implies $\ell_\infty$-Lipschitz into logits, with the same constant.

The proof applies the standard comparison $\lVert v\rVert_\infty\leq\lVert v\rVert_2$, proved as
`linfNorm_le_norm`, to the difference of the two outputs.
-/
theorem IsLipschitzContinuous.linf_of_l2
    {s₁ : Shape} {n : Nat}
    {f : Tensor ℝ s₁ → Tensor ℝ [n]}
    {normIn : ∀ {s : Shape}, Tensor ℝ s → ℝ}
    {L : ℝ}
    (hLip2 : IsLipschitzContinuous f normIn (‖·‖) L) :
    IsLipschitzContinuous f normIn (linfNorm (α := ℝ)) L := by
  intro x y
  have hnorm :
      distance (α := ℝ) (linfNorm (α := ℝ)) (f x) (f y) ≤
        distance (α := ℝ) (‖·‖) (f x) (f y) := by
    simpa [distance] using
      (linfNorm_le_norm
        (y := f x - f y))
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

private theorem abs_getScalar_le_linfNorm {n : Nat} (y : Tensor ℝ [n]) (i : Fin n) :
    |Tensor.getScalar y i| ≤ linfNorm (α := ℝ) y := by
  have hi : i ∈ List.finRange n := List.mem_finRange i
  have hle :
      linfNorm (α := ℝ) (y.unstack i) ≤
        (List.finRange n).foldl
          (fun acc j => max acc (linfNorm (α := ℝ) (y.unstack j))) 0 :=
    List.le_foldl_max_of_mem (List.finRange n)
      (fun j => linfNorm (α := ℝ) (y.unstack j)) (acc := (0 : ℝ)) hi
  simpa [Tensor.getScalar, linfNorm, MathFunctions.abs] using hle

private theorem abs_getScalar_sub_le
    {n : Nat} {y₀ y : Tensor ℝ [n]} {δ : ℝ} (i : Fin n)
    (h : distance (α := ℝ) (linfNorm (α := ℝ)) y₀ y ≤ δ) :
    |Tensor.getScalar y₀ i - Tensor.getScalar y i| ≤ δ := by
  -- Under `linfNorm`, `distance` is the largest absolute entry of `y₀ - y`.
  have hcoord :
      |Tensor.getScalar (y₀ - y) i| ≤
        linfNorm (α := ℝ) (y₀ - y) := by
    simpa using
      (abs_getScalar_le_linfNorm (y := y₀ - y) (i := i))
  have : |Tensor.getScalar (y₀ - y) i| ≤ δ :=
    le_trans hcoord (by simpa [distance] using h)
  simpa [Tensor.sub_def] using this

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
If class $c$ has logit margin $m$ in $y_0$, then every $y$ within $\ell_\infty$ distance $\delta$
of $y_0$, where $2\delta<m$, has `argmax` class $c$.
-/
theorem argmaxClassifier_eq_of_distance_le
    {n : Nat} {y₀ y : Tensor ℝ [n]} {c : Fin n} {m δ : ℝ}
    (hmargin : HasLogitMargin (n := n) y₀ c m)
    (hδ : 2 * δ < m)
    (hdist : distance (α := ℝ) (linfNorm (α := ℝ)) y₀ y ≤ δ) :
    argmaxClassifier (n := n) y = c.val := by
  have hstrict : ∀ k : Fin n, k ≠ c → Tensor.getScalar y k < Tensor.getScalar y c := by
    intro k hk
    have habs_k : |Tensor.getScalar y₀ k - Tensor.getScalar y k| ≤ δ :=
      abs_getScalar_sub_le (i := k) hdist
    have habs_c : |Tensor.getScalar y₀ c - Tensor.getScalar y c| ≤ δ :=
      abs_getScalar_sub_le (i := c) hdist
    have hk' := abs_sub_le_iff.1 habs_k
    have hc' := abs_sub_le_iff.1 habs_c
    have hcomp : Tensor.getScalar y₀ k ≤ Tensor.getScalar y₀ c - m := hmargin k hk
    -- `y k ≤ y₀ k + δ ≤ y₀ c - m + δ < y₀ c - δ ≤ y c`, using `2δ < m`.
    linarith [hk'.1, hk'.2, hc'.1, hc'.2]
  have harg : List.argmax (fun i : Fin n => Tensor.getScalar y i) (List.finRange n) = some c :=
    argmax_eq_some_of_strictMax (n := n) (y := y) (c := c) hstrict
  simp [argmaxClassifier, harg]

/--
If $y_0$ has margin $m>0$ for class $c$, then $c$ is the `argmaxClassifier` of $y_0$.
-/
theorem HasLogitMargin.argmaxClassifier_eq
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

The argument is the standard margin-over-Lipschitz-constant certificate. The logits move by at
most $L\varepsilon$ in $\ell_\infty$, and twice that stays below the margin.
-/
theorem IsCertifiedRobust.of_lipschitz
    {s₁ : Shape} {n : Nat}
    {f : Tensor ℝ s₁ → Tensor ℝ [n]}
    {normIn : ∀ {s : Shape}, Tensor ℝ s → ℝ}
    {L : ℝ} (hL : 0 ≤ L)
    (hLip : IsLipschitzContinuous f normIn (linfNorm (α := ℝ)) L)
    {x₀ : Tensor ℝ s₁} {ε m : ℝ} {c : Fin n}
    (hRadius : 0 ≤ ε)
    (hmargin : HasLogitMargin (n := n) (f x₀) c m)
    (hε : 2 * (L * ε) < m) :
    IsCertifiedRobust (classifier := fun x => argmaxClassifier (n := n) (f x))
      (norm := normIn) x₀ ε := by
  have hm : 0 < m := lt_of_le_of_lt (by positivity) hε
  refine ⟨hRadius, ?_⟩
  intro x hx
  have hx' : distance (α := ℝ) normIn x₀ x ≤ ε := by
    simpa [ball] using hx
  have hdist₁ : distance (α := ℝ) (linfNorm (α := ℝ)) (f x₀) (f x) ≤
      L * distance (α := ℝ) normIn x₀ x :=
    hLip x₀ x
  have hdist₂ : L * distance (α := ℝ) normIn x₀ x ≤ L * ε :=
    mul_le_mul_of_nonneg_left hx' hL
  have hdist : distance (α := ℝ) (linfNorm (α := ℝ)) (f x₀) (f x) ≤ L * ε :=
    le_trans hdist₁ hdist₂
  have hx_class : argmaxClassifier (n := n) (f x) = c.val :=
    argmaxClassifier_eq_of_distance_le (n := n) (y₀ := f x₀) (y := f x)
      (c := c) (m := m) (δ := L * ε) hmargin hε hdist
  have hx0_class : argmaxClassifier (n := n) (f x₀) = c.val :=
    HasLogitMargin.argmaxClassifier_eq (n := n) (y₀ := f x₀) (c := c) (m := m) hm hmargin
  exact hx_class.trans hx0_class.symm

/--
Certified robustness from an output-$\ell_2$ Lipschitz assumption.

The norm comparison $\lVert\cdot\rVert_\infty\leq\lVert\cdot\rVert_2$ turns the hypothesis into
the output-$\ell_\infty$ form of `Robustness.IsCertifiedRobust.of_lipschitz`, with the same
constant $L$.
-/
theorem IsCertifiedRobust.of_l2_lipschitz
    {s₁ : Shape} {n : Nat}
    {f : Tensor ℝ s₁ → Tensor ℝ [n]}
    {normIn : ∀ {s : Shape}, Tensor ℝ s → ℝ}
    {L : ℝ} (hL : 0 ≤ L)
    (hLip2 : IsLipschitzContinuous f normIn (‖·‖) L)
    {x₀ : Tensor ℝ s₁} {ε m : ℝ} {c : Fin n}
    (hRadius : 0 ≤ ε)
    (hmargin : HasLogitMargin (n := n) (f x₀) c m)
    (hε : 2 * (L * ε) < m) :
    IsCertifiedRobust (classifier := fun x => argmaxClassifier (n := n) (f x))
      (norm := normIn) x₀ ε := by
  have hLipLinf : IsLipschitzContinuous f normIn (linfNorm (α := ℝ)) L :=
    IsLipschitzContinuous.linf_of_l2 (hLip2 := hLip2)
  exact
    of_lipschitz (n := n) (hL := hL) (hLip := hLipLinf)
      (x₀ := x₀) (ε := ε) (m := m) (c := c) hRadius hmargin hε

end TorchLean.Robustness
