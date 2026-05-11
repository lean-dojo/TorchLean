/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximation
import Mathlib.Data.Real.Archimedean
import Mathlib.Tactic.Linarith

/-!
# Universal approximation (1D, explicit rate)

This file strengthens `relu_universal_approximation_Icc` by choosing an explicit hidden width in
terms of the Lipschitz constant `L`, the interval length `(b-a)`, and the target accuracy `ε`.

The bound is the standard `O(1 / hidDim)` rate coming from piecewise-linear interpolation:
we pick

`hidDim = ⌈(2 * L * (b - a)) / ε⌉ + 1`,

which guarantees uniform approximation error `< ε` on `Set.Icc a b`.

Mathematically, this is the quantitative sibling of the constructive one-dimensional ReLU
universal approximation proof in `UniversalApproximation`: sample a Lipschitz function on a
uniform grid, interpolate linearly by hinge functions, and choose the grid fine enough that the
Lipschitz modulus controls the interpolation error.  The style is classical approximation theory
(Pinkus) and agrees with the first-order rate used in modern ReLU-network approximation analyses
such as Yarotsky's quantitative bounds.
-/

@[expose] public section

namespace NN.MLTheory.Proofs.UniversalApproximation

open _root_.Spec
open _root_.Spec.Tensor
open Examples

noncomputable section

/-- Explicit hidden width for the 1D Lipschitz ReLU approximation construction. -/
def reluApproximationWidth (L a b ε : ℝ) : ℕ :=
  Nat.ceil (2 * L * (b - a) / ε) + 1

/-- The explicit ReLU approximation width is always positive. -/
lemma relu_approximation_width_pos (L a b ε : ℝ) : 0 < reluApproximationWidth L a b ε := by
  simp [reluApproximationWidth]

/--
The chosen width makes the mesh-size error term smaller than the target accuracy.

This is the arithmetic heart of the explicit-rate theorem: the ceiling construction ensures
`N > 2L(b-a)/ε`, hence `2L(b-a)/N < ε`.
-/
lemma two_mul_mul_sub_div_relu_approximation_width_lt {L a b ε : ℝ} (hε : 0 < ε) :
    (2 * L * (b - a)) / (reluApproximationWidth L a b ε : ℝ) < ε := by
  classical
  let N : ℕ := reluApproximationWidth L a b ε
  have hNpos_nat : 0 < N := relu_approximation_width_pos L a b ε
  have hNpos : 0 < (N : ℝ) := by exact_mod_cast hNpos_nat
  have hr_lt : (2 * L * (b - a) / ε : ℝ) < (N : ℝ) := by
    have hr_le :
        (2 * L * (b - a) / ε : ℝ) ≤ (Nat.ceil (2 * L * (b - a) / ε) : ℝ) :=
      Nat.le_ceil _
    have : (2 * L * (b - a) / ε : ℝ) < (Nat.ceil (2 * L * (b - a) / ε) : ℝ) + 1 := by
      linarith
    simpa [N, reluApproximationWidth, Nat.cast_add, Nat.cast_one, add_assoc] using this
  have hmul : ε * (2 * L * (b - a) / ε) < ε * (N : ℝ) := mul_lt_mul_of_pos_left hr_lt hε
  have hεne : (ε : ℝ) ≠ 0 := ne_of_gt hε
  have hleft : ε * (2 * L * (b - a) / ε) = 2 * L * (b - a) := by
    calc
      ε * (2 * L * (b - a) / ε) = ε * (2 * L * (b - a)) / ε := by
        simp [mul_div_assoc']
      _ = 2 * L * (b - a) := by
        simpa using (mul_div_cancel_left₀ (2 * L * (b - a)) hεne)
  have hnum : 2 * L * (b - a) < ε * (N : ℝ) := by
    simpa [hleft] using hmul
  exact (div_lt_iff₀ hNpos).2 (by simpa [mul_comm, mul_assoc] using hnum)

/--
Universal approximation (1D, hinge form) with an explicit width choice.

This is a quantitative variant of `relu_universal_approximation_Icc_hinge` where the hidden width
is fixed to `reluApproximationWidth L a b ε`.
-/
theorem relu_universal_approximation_Icc_hinge_rate {f : ℝ → ℝ} {a b L : ℝ}
    (h_ab : a < b) (hL : 0 < L)
    (h_lip : ∀ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, |f x - f y| ≤ L * |x - y|) :
    ∀ ε > 0,
      ∃ (t : Fin (reluApproximationWidth L a b ε) → ℝ)
        (c : Fin (reluApproximationWidth L a b ε) → ℝ),
        ∀ x ∈ Set.Icc a b,
          |f x - hingeFun (reluApproximationWidth L a b ε) t c (f a) x| < ε := by
  intro ε hε
  classical
  have hba : 0 < b - a := sub_pos.mpr h_ab

  -- Fix the width explicitly and define the grid spacing `δ`.
  let N : ℕ := reluApproximationWidth L a b ε
  have hNpos_nat : 0 < N := relu_approximation_width_pos L a b ε
  have hNpos : 0 < (N : ℝ) := by exact_mod_cast hNpos_nat
  have hNne : (N : ℝ) ≠ 0 := ne_of_gt hNpos
  let δ : ℝ := (b - a) / (N : ℝ)
  have hδpos : 0 < δ := div_pos hba hNpos
  have hδnonneg : 0 ≤ δ := le_of_lt hδpos
  have h2mesh : 2 * L * δ < ε := by
    have hdiv : (2 * L * (b - a)) / (N : ℝ) < ε := by
      simpa [N] using
        (two_mul_mul_sub_div_relu_approximation_width_lt (L := L) (a := a) (b := b) (ε := ε) hε)
    simpa [δ, mul_div_assoc', mul_assoc] using hdiv

  let grid : ℕ → ℝ := fun k => a + (k : ℝ) * δ
  have hgrid0 : grid 0 = a := by simp [grid]
  have hgridN : grid N = b := by
    calc
      grid N = a + (N : ℝ) * δ := by simp [grid]
      _ = a + (N : ℝ) * (b - a) / (N : ℝ) := by simp [δ, mul_div_assoc']
      _ = a + (b - a) := by simp [mul_div_cancel_left₀, hNne]
      _ = b := by ring

  let mNat : ℕ → ℝ := fun k => (f (grid (k + 1)) - f (grid k)) / δ
  let cNat : ℕ → ℝ
    | 0 => mNat 0
    | k + 1 => mNat (k + 1) - mNat k
  let g : ℝ → ℝ := fun x => f a + ∑ i ∈ Finset.range N, cNat i * relu (x - grid i)

  have prefix_sum_cNat_eq_mNat : ∀ k : ℕ, (∑ i ∈ Finset.range (k + 1), cNat i) = mNat k := by
    intro k
    induction k with
    | zero =>
      simp [cNat, mNat]
    | succ k ih =>
      calc
        (∑ i ∈ Finset.range (k + 2), cNat i)
            = (∑ i ∈ Finset.range (k + 1), cNat i) + cNat (k + 1) := by
              simpa using (Finset.sum_range_succ (f := fun i => cNat i) (n := k + 1))
        _ = mNat k + cNat (k + 1) := by simp [ih]
        _ = mNat k + (mNat (k + 1) - mNat k) := by simp [cNat]
        _ = mNat (k + 1) := by ring

  have grid_mono : Monotone grid := by
    intro m n hmn
    dsimp [grid]
    have : (m : ℝ) ≤ (n : ℝ) := by exact_mod_cast hmn
    nlinarith

  have g_affine_on_segment :
      ∀ {k : ℕ}, k + 1 ≤ N → ∀ {x : ℝ}, grid k ≤ x → x ≤ grid (k + 1) →
        g x = g (grid k) + mNat k * (x - grid k) := by
    intro k hkN x hx0 hx1
    classical
    let F : ℕ → ℝ := fun i => cNat i * relu (x - grid i)
    let G : ℕ → ℝ := fun i => cNat i * relu (grid k - grid i)
    have hsub : Finset.range (k + 1) ⊆ Finset.range N := by
      intro i hi
      have hi' : i < k + 1 := Finset.mem_range.mp hi
      have : i < N := lt_of_lt_of_le hi' hkN
      exact Finset.mem_range.mpr this
    have hFzero : ∀ i ∈ Finset.range N, i ∉ Finset.range (k + 1) → F i = 0 := by
      intro i hiN hik
      have hik' : k + 1 ≤ i := by
        have : ¬ i < k + 1 := by
          exact fun hlt => hik (Finset.mem_range.mpr hlt)
        exact Nat.le_of_not_gt this
      have hgi : grid (k + 1) ≤ grid i := grid_mono hik'
      have hxle : x ≤ grid i := le_trans hx1 hgi
      simp [F, relu_sub_eq_zero_of_le (x := x) (t := grid i) hxle]
    have hGzero : ∀ i ∈ Finset.range N, i ∉ Finset.range (k + 1) → G i = 0 := by
      intro i hiN hik
      have hik' : k + 1 ≤ i := by
        have : ¬ i < k + 1 := by
          exact fun hlt => hik (Finset.mem_range.mpr hlt)
        exact Nat.le_of_not_gt this
      have hgi : grid k ≤ grid i := by
        have : k ≤ i := le_trans (Nat.le_succ k) hik'
        exact grid_mono this
      simp [G, relu_sub_eq_zero_of_le (x := grid k) (t := grid i) hgi]
    have sumF : (∑ i ∈ Finset.range N, F i) = (∑ i ∈ Finset.range (k + 1), F i) := by
      symm
      exact Finset.sum_subset hsub hFzero
    have sumG : (∑ i ∈ Finset.range N, G i) = (∑ i ∈ Finset.range (k + 1), G i) := by
      symm
      exact Finset.sum_subset hsub hGzero
    have gx : g x = f a + ∑ i ∈ Finset.range (k + 1), F i := by
      simp [g, F, sumF]
    have gk : g (grid k) = f a + ∑ i ∈ Finset.range (k + 1), G i := by
      simp [g, G, sumG]
    have hF' : ∀ i ∈ Finset.range (k + 1), F i = cNat i * (x - grid i) := by
      intro i hi
      have hi' : i ≤ k := Nat.le_of_lt_succ (Finset.mem_range.mp hi)
      have hgi : grid i ≤ grid k := grid_mono hi'
      have : grid i ≤ x := le_trans hgi hx0
      simp [F, relu_sub_eq_of_le (x := x) (t := grid i) this]
    have hG' : ∀ i ∈ Finset.range (k + 1), G i = cNat i * (grid k - grid i) := by
      intro i hi
      have hi' : i ≤ k := Nat.le_of_lt_succ (Finset.mem_range.mp hi)
      have : grid i ≤ grid k := grid_mono hi'
      simp [G, relu_sub_eq_of_le (x := grid k) (t := grid i) this]
    have gx' : g x = f a + ∑ i ∈ Finset.range (k + 1), cNat i * (x - grid i) := by
      refine gx.trans ?_
      congr 1
      exact Finset.sum_congr rfl hF'
    have gk' : g (grid k) = f a + ∑ i ∈ Finset.range (k + 1), cNat i * (grid k - grid i) := by
      refine gk.trans ?_
      congr 1
      exact Finset.sum_congr rfl hG'
    have hdiff :
        (∑ i ∈ Finset.range (k + 1), cNat i * (x - grid i)) -
          (∑ i ∈ Finset.range (k + 1), cNat i * (grid k - grid i)) =
            (∑ i ∈ Finset.range (k + 1), cNat i) * (x - grid k) := by
      calc
        (∑ i ∈ Finset.range (k + 1), cNat i * (x - grid i)) -
            (∑ i ∈ Finset.range (k + 1), cNat i * (grid k - grid i)) =
            ∑ i ∈ Finset.range (k + 1),
              (cNat i * (x - grid i) - cNat i * (grid k - grid i)) := by
              exact
                (Finset.sum_sub_distrib (s := Finset.range (k + 1))
                  (f := fun i => cNat i * (x - grid i))
                  (g := fun i => cNat i * (grid k - grid i))).symm
        _ = ∑ i ∈ Finset.range (k + 1), cNat i * (x - grid k) := by
              apply Finset.sum_congr rfl
              intro i hi
              have hmul :
                  cNat i * (x - grid i) - cNat i * (grid k - grid i) =
                    cNat i * ((x - grid i) - (grid k - grid i)) := by
                simpa using (mul_sub (cNat i) (x - grid i) (grid k - grid i)).symm
              have hinner : (x - grid i) - (grid k - grid i) = x - grid k := by ring
              simp [hmul, hinner]
        _ = (∑ i ∈ Finset.range (k + 1), cNat i) * (x - grid k) := by
              simpa using
                (Finset.sum_mul (s := Finset.range (k + 1))
                  (f := fun i => cNat i)
                  (a := (x - grid k))).symm
    have hsub' : g x - g (grid k) = mNat k * (x - grid k) := by
      calc
        g x - g (grid k) =
            (f a + ∑ i ∈ Finset.range (k + 1), cNat i * (x - grid i)) -
              (f a + ∑ i ∈ Finset.range (k + 1), cNat i * (grid k - grid i)) := by
              simp [gx', gk']
        _ =
            (∑ i ∈ Finset.range (k + 1), cNat i * (x - grid i)) -
              (∑ i ∈ Finset.range (k + 1), cNat i * (grid k - grid i)) := by ring
        _ = (∑ i ∈ Finset.range (k + 1), cNat i) * (x - grid k) := hdiff
        _ = mNat k * (x - grid k) := by simp [prefix_sum_cNat_eq_mNat k]
    linarith

  have g_grid_eq_f_grid : ∀ k : ℕ, k ≤ N → g (grid k) = f (grid k) := by
    intro k hk
    induction k with
    | zero =>
      -- g(a) = f(a)
      have ha_le : ∀ i ∈ Finset.range N, a ≤ grid i := by
        intro i hi
        have : grid 0 ≤ grid i := grid_mono (Nat.zero_le i)
        simpa [hgrid0] using this
      have hsum :
          (∑ i ∈ Finset.range N, cNat i * relu (a - grid i)) = 0 := by
        refine Finset.sum_eq_zero ?_
        intro i hi
        have : a ≤ grid i := ha_le i hi
        simp [relu_sub_eq_zero_of_le (x := a) (t := grid i) this]
      simp [g, hgrid0, hsum]
    | succ k ih =>
      have hkN : k + 1 ≤ N := hk
      have hk_le : k ≤ N := le_trans (Nat.le_succ k) hkN
      have hx0 : grid k ≤ grid (k + 1) := grid_mono (Nat.le_succ k)
      have haff := g_affine_on_segment (k := k) hkN (x := grid (k + 1)) hx0 le_rfl
      have hstep : grid (k + 1) - grid k = δ := by
        dsimp [grid]
        simp [Nat.cast_add, Nat.cast_one, add_assoc, add_comm, sub_eq_add_neg]
        ring
      have hm : mNat k * δ = f (grid (k + 1)) - f (grid k) := by
        have hδne : δ ≠ 0 := ne_of_gt hδpos
        dsimp [mNat]
        field_simp [hδne]
      have hk_eq : g (grid k) = f (grid k) := ih hk_le
      have : g (grid (k + 1)) = f (grid (k + 1)) := by
        calc
          g (grid (k + 1)) = g (grid k) + mNat k * (grid (k + 1) - grid k) := haff
          _ = f (grid k) + mNat k * δ := by simp [hk_eq, hstep]
          _ = f (grid k) + (f (grid (k + 1)) - f (grid k)) := by simp [hm]
          _ = f (grid (k + 1)) := by ring
      simpa using this

  -- Build the width-`N` hinge network corresponding to the polygonal interpolant.
  let t : Fin N → ℝ := fun i => grid i.1
  let c : Fin N → ℝ := fun i => cNat i.1
  refine ⟨t, c, ?_⟩
  intro x hx
  have hhinge : hingeFun N t c (f a) x = g x := by
    unfold hingeFun g t c
    congr 1
    simpa using
      (Fin.sum_univ_eq_sum_range
        (f := fun i : ℕ => cNat i * relu (x - grid i)) (n := N))
  -- Prove the approximation bound on `[a,b]`.
  have hxle_gridN : x ≤ grid N := by simpa [hgridN] using hx.2
  have hexists : ∃ k : ℕ, x ≤ grid k := ⟨N, hxle_gridN⟩
  -- Two cases: x = a (`Nat.find hexists = 0`) or x lies in some segment `[grid k, grid (k+1)]`.
  cases hjcases : Nat.find hexists with
  | zero =>
    have hx_eq : x = a := by
      have hxle : x ≤ grid 0 := by
        simpa [hjcases] using Nat.find_spec hexists
      have hxle' : x ≤ a := by simpa [hgrid0] using hxle
      exact le_antisymm hxle' hx.1
    have hg_a : g a = f a := by
      have := g_grid_eq_f_grid 0 (Nat.zero_le N)
      simpa [hgrid0] using this
    have : |f x - g x| < ε := by
      simpa [hx_eq, hg_a, abs_zero] using hε
    simpa [N, hhinge] using this
  | succ k =>
    have hj : x ≤ grid (k + 1) := by
      simpa [hjcases] using Nat.find_spec hexists
    have hkN : k + 1 ≤ N := by
      have hmin : Nat.find hexists ≤ N := Nat.find_min' hexists hxle_gridN
      simpa [hjcases] using hmin
    have hx_not_le_prev : ¬ x ≤ grid k := by
      intro hxle
      have hmin : Nat.find hexists ≤ k := Nat.find_min' hexists hxle
      rw [hjcases] at hmin
      exact Nat.not_succ_le_self k hmin
    have hx0 : grid k ≤ x := le_of_lt (lt_of_not_ge hx_not_le_prev)
    have hx1 : x ≤ grid (k + 1) := hj
    have hk_le : k ≤ N := le_trans (Nat.le_succ k) hkN
    have hgk : g (grid k) = f (grid k) := g_grid_eq_f_grid k hk_le
    have haff : g x = g (grid k) + mNat k * (x - grid k) :=
      g_affine_on_segment (k := k) hkN (x := x) hx0 hx1
    have hfx : |f x - g x| < ε := by
      -- bound via Lipschitz: |f x - g x| ≤ 2 L δ < ε
      have hgridk_mem : grid k ∈ Set.Icc a b := by
        have ha : a ≤ grid k := by
          have : grid 0 ≤ grid k := grid_mono (Nat.zero_le k)
          simpa [hgrid0] using this
        have hb : grid k ≤ b := by
          have : grid k ≤ grid N := grid_mono hk_le
          simpa [hgridN] using this
        exact ⟨ha, hb⟩
      have hgridkp1_mem : grid (k + 1) ∈ Set.Icc a b := by
        have ha : a ≤ grid (k + 1) := by
          have : grid 0 ≤ grid (k + 1) := grid_mono (Nat.zero_le (k + 1))
          simpa [hgrid0] using this
        have hb : grid (k + 1) ≤ b := by
          have : grid (k + 1) ≤ grid N := grid_mono hkN
          simpa [hgridN] using this
        exact ⟨ha, hb⟩
      have hx_dist : |x - grid k| ≤ δ := by
        have hx0' : 0 ≤ x - grid k := sub_nonneg.mpr hx0
        have hxle : x - grid k ≤ δ := by
          have : x - grid k ≤ grid (k + 1) - grid k := sub_le_sub_right hx1 (grid k)
          have hstep : grid (k + 1) - grid k = δ := by
            dsimp [grid]
            simp [Nat.cast_add, Nat.cast_one, add_assoc, add_comm, sub_eq_add_neg]
            ring
          simpa [hstep] using this
        have habs : |x - grid k| = x - grid k := abs_of_nonneg hx0'
        simpa [habs] using hxle
      have h1 : |f x - f (grid k)| ≤ L * δ := by
        have := h_lip x hx (grid k) hgridk_mem
        exact le_trans this (mul_le_mul_of_nonneg_left hx_dist hL.le)
      have hmδ : mNat k * δ = f (grid (k + 1)) - f (grid k) := by
        have hδne : δ ≠ 0 := ne_of_gt hδpos
        dsimp [mNat]
        field_simp [hδne]
      have h2 : |f (grid k) - g x| ≤ L * δ := by
        have habs' : |f (grid k) - g x| = |mNat k * (x - grid k)| := by
          have hgx : g x = f (grid k) + mNat k * (x - grid k) := by
            simpa [hgk] using haff
          have hs : f (grid k) - g x = -(mNat k * (x - grid k)) := by
            simp [hgx]
          have : |f (grid k) - g x| = |-(mNat k * (x - grid k))| := by
            simpa using congrArg abs hs
          simpa [abs_neg] using this
        have hmul_le : |mNat k * (x - grid k)| ≤ |mNat k| * δ := by
          have := mul_le_mul_of_nonneg_left hx_dist (abs_nonneg (mNat k))
          simpa [abs_mul] using this
        have hmul_eq : |mNat k| * δ = |mNat k * δ| := by
          simp [abs_mul, abs_of_nonneg hδnonneg, mul_comm]
        have hendpoint : |mNat k * δ| ≤ L * δ := by
          have hdiff := h_lip (grid (k + 1)) hgridkp1_mem (grid k) hgridk_mem
          have hstep' : grid (k + 1) - grid k = δ := by
            dsimp [grid]
            simp [Nat.cast_add, Nat.cast_one, add_assoc, add_comm, sub_eq_add_neg]
            ring
          have hnonneg : 0 ≤ grid (k + 1) - grid k := by
            have : grid k ≤ grid (k + 1) := grid_mono (Nat.le_succ k)
            exact sub_nonneg.mpr this
          have hstep : |grid (k + 1) - grid k| = δ := by
            calc
              |grid (k + 1) - grid k| = grid (k + 1) - grid k := abs_of_nonneg hnonneg
              _ = δ := hstep'
          have : |mNat k * δ| = |f (grid (k + 1)) - f (grid k)| := by simp [hmδ]
          simpa [this, hstep] using hdiff
        have hbound : |mNat k * (x - grid k)| ≤ |mNat k * δ| := by
          exact le_trans hmul_le (le_of_eq hmul_eq)
        have hfinal : |mNat k * (x - grid k)| ≤ L * δ := le_trans hbound hendpoint
        simpa [habs'] using hfinal
      have : |f x - g x| ≤ 2 * L * δ := by
        have htri : |f x - g x| ≤ |f x - f (grid k)| + |f (grid k) - g x| := by
          have htri0 :
              |(f x - f (grid k)) + (f (grid k) - g x)| ≤
                |f x - f (grid k)| + |f (grid k) - g x| :=
            abs_add_le (f x - f (grid k)) (f (grid k) - g x)
          have hrew' : (f x - f (grid k)) + (f (grid k) - g x) = f x - g x := by ring
          simpa [hrew'] using htri0
        have hsum : |f x - f (grid k)| + |f (grid k) - g x| ≤ L * δ + L * δ :=
          add_le_add h1 (by simpa [abs_sub_comm] using h2)
        have hsum' : L * δ + L * δ = 2 * L * δ := by ring
        exact le_trans htri (by simpa [hsum'] using hsum)
      exact lt_of_le_of_lt this h2mesh
    simpa [N, hhinge] using hfx

/--
Universal approximation (1D, explicit rate) for a 2-layer ReLU MLP.

This is the MLP-packaged version of `relu_universal_approximation_Icc_hinge_rate`.
-/
theorem relu_universal_approximation_Icc_rate {f : ℝ → ℝ} {a b L : ℝ}
    (h_ab : a < b) (hL : 0 < L)
    (h_lip : ∀ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, |f x - f y| ≤ L * |x - y|) :
    ∀ ε > 0,
      ∃ (l1 : LinearSpec ℝ 1 (reluApproximationWidth L a b ε))
        (l2 : LinearSpec ℝ (reluApproximationWidth L a b ε) 1),
        ∀ x ∈ Set.Icc a b,
          |f x - mlpEval1d (reluApproximationWidth L a b ε) l1 l2 x| < ε := by
  intro ε hε
  classical
  rcases
      relu_universal_approximation_Icc_hinge_rate (f := f) (a := a) (b := b) (L := L)
        h_ab hL h_lip ε hε with
    ⟨t, c, happx⟩
  refine ⟨hingeLayer1 (reluApproximationWidth L a b ε) t,
    hingeLayer2 (reluApproximationWidth L a b ε) c (f a),
    ?_⟩
  intro x hx
  have hnet :
      mlpEval1d (reluApproximationWidth L a b ε)
          (hingeLayer1 (reluApproximationWidth L a b ε) t)
          (hingeLayer2 (reluApproximationWidth L a b ε) c (f a)) x =
        hingeFun (reluApproximationWidth L a b ε) t c (f a) x := by
    simpa using (mlp_eval_1d_hinge (reluApproximationWidth L a b ε) t c (f a) x)
  simpa [hnet] using happx x hx

end

end NN.MLTheory.Proofs.UniversalApproximation
