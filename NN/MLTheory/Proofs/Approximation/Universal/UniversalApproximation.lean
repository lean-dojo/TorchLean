/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models.Mlp
public import NN.Proofs.Tensor.Basic.Folds
public import NN.Spec.Core.Context.Real

/-!
# Universal approximation (1D, constructive)

On a compact interval $I=[a,b]$, any Lipschitz function $f:\mathbb{R}\to\mathbb{R}$ can be
uniformly approximated
by a single-hidden-layer ReLU network (a 2-layer MLP).

This file formalizes the classic constructive proof strategy:

- approximate $f$ by a polygonal function on a uniform grid,
- express that polygonal function as an affine term plus a finite sum of hinges
  $\operatorname{ReLU}(x-t_i)$,
- package the hinge representation as TorchLean's spec-level 2-layer MLP (`NN.Spec.Models.Mlp`).

## Main result

- `relu_universal_approximation_Icc`: existence of a 2-layer ReLU MLP approximator on `Set.Icc a b`.

## References

- Leshno, Lin, Pinkus, Schocken (1993), *Multilayer feedforward networks with a nonpolynomial
  activation function can approximate any function*.
- Yarotsky (2017), *Error bounds for approximations with deep ReLU networks*.
- Pinkus (1999), *Approximation theory of the MLP model in neural networks*.
-/

@[expose] public section

namespace NN.MLTheory.Proofs.UniversalApproximation

open _root_.Spec
open _root_.TorchLean
open _root_.TorchLean.Tensor
open Examples

/-- Shorthand for `relu` in this development, using TorchLean’s spec semantics. -/
noncomputable abbrev relu (x : ℝ) : ℝ := Activation.Math.reluSpec x

/-- If the knot $t$ is to the left of $x$, then $\operatorname{ReLU}(x-t)=x-t$. -/
theorem relu_sub_eq_of_le {x t : ℝ} (h : t ≤ x) : relu (x - t) = x - t := by
  have : 0 ≤ x - t := sub_nonneg.mpr h
  rw [relu, Activation.Math.reluSpec_eq_max, max_eq_left this]

/-- If $x$ is to the left of the knot $t$, then $\operatorname{ReLU}(x-t)=0$. -/
theorem relu_sub_eq_zero_of_le {x t : ℝ} (h : x ≤ t) : relu (x - t) = 0 := by
  have : x - t ≤ 0 := sub_nonpos.mpr h
  rw [relu, Activation.Math.reluSpec_eq_max, max_eq_right this]

/-- Extract the unique entry of a length-`1` tensor. -/
def extractScalarOutput {α : Type} [TorchLean.Storage α] (t : Tensor α [1]) : α :=
  t.getScalar ⟨0, by norm_num⟩

/-- Evaluate a 2-layer ReLU MLP on a scalar input. -/
noncomputable def mlpEvalScalar (hidDim : ℕ)
    (l1 : LinearSpec ℝ 1 hidDim) (l2 : LinearSpec ℝ hidDim 1) (x : ℝ) : ℝ :=
  extractScalarOutput (Examples.mlpForward l1 l2 (Tensor.singleton x))

/--
TorchLean's MLP forward pass is exactly
$\operatorname{linear}\circ\operatorname{ReLU}\circ\operatorname{linear}$.
-/
theorem mlp_forward_eq_linear_relu_linear {hidDim : ℕ}
    (l1 : LinearSpec ℝ 1 hidDim) (l2 : LinearSpec ℝ hidDim 1) (x : Tensor ℝ [1]) :
    Examples.mlpForward l1 l2 x =
      let z1 := Spec.linearSpec (α := ℝ) l1 x
      let a1 := Activation.reluSpec z1
      Spec.linearSpec (α := ℝ) l2 a1 := by
  simpa [Examples.mlpForward] using (Examples.mlp_spec_forward_eq (α := ℝ) l1 l2 x)

/-- First real hinge layer: hidden unit $i$ computes $x-t_i$ before ReLU. -/
noncomputable def hingeLayer1 (n : ℕ) (t : Fin n → ℝ) : LinearSpec ℝ 1 n :=
  { weights := Tensor.matrix (m := n) (n := 1) (fun _ _ => (1 : ℝ))
    bias := Tensor.ofFn (n := n) (fun i => -t i) }

/-- Second real hinge layer: sum hidden activations with coefficients $c_i$ and bias $b$. -/
noncomputable def hingeLayer2 (n : ℕ) (c : Fin n → ℝ) (b : ℝ) : LinearSpec ℝ n 1 :=
  { weights := Tensor.matrix (m := 1) (n := n) (fun _ j => c j)
    bias := Tensor.ofFn (n := 1) (fun _ => b) }

/-- Real hinge network $b+\sum_i c_i\operatorname{ReLU}(x-t_i)$. -/
noncomputable def hingeFun (n : ℕ) (t : Fin n → ℝ) (c : Fin n → ℝ) (b x : ℝ) : ℝ :=
  b + ∑ i : Fin n, c i * relu (x - t i)

/-- Matrix-vector multiply for a one-row matrix is the expected finite dot product. -/
theorem mat_vec_mul_spec_matrix_vector (n : ℕ) (c v : Fin n → ℝ) :
    matVecMulSpec (Tensor.matrix (m := 1) (n := n) (fun _ j => c j))
        (Tensor.dim (fun j => Tensor.scalar (v j))) =
      Tensor.dim (fun _ => Tensor.scalar (∑ j : Fin n, c j * v j)) := by
  classical
  apply Tensor.ext_vector
  intro i
  fin_cases i
  simp [getScalar_mat_vec_mul_spec, Tensor.matrix, Spec.get2]

/-- Matrix-vector multiply by the all-ones column extracts the scalar input into every hidden
unit. -/
theorem mat_vec_mul_spec_matrix_singleton (n : ℕ) (x : ℝ) :
    matVecMulSpec (Tensor.matrix (m := n) (n := 1) (fun _ _ => (1 : ℝ)))
        (Tensor.singleton x) =
      Tensor.dim (fun _ : Fin n => Tensor.scalar x) := by
  classical
  apply Tensor.ext_vector
  intro i
  simp [getScalar_mat_vec_mul_spec, Tensor.matrix, Spec.get2, Tensor.singleton]

/--
The explicit two-layer network built from `hingeLayer1` and `hingeLayer2` computes `hingeFun`.

This is the main semantic bridge from the approximation-theory hinge representation to TorchLean's
spec-level MLP model.
-/
theorem mlp_eval_scalar_hinge (n : ℕ) (t : Fin n → ℝ) (c : Fin n → ℝ) (b x : ℝ) :
    mlpEvalScalar n (hingeLayer1 n t) (hingeLayer2 n c b) x = hingeFun n t c b x := by
  classical
  unfold mlpEvalScalar hingeFun extractScalarOutput
  -- Avoid `simp` here (it unfolds too much under matchers); rewrite `mlpForward` explicitly.
  rw [mlp_forward_eq_linear_relu_linear (l1 := hingeLayer1 n t) (l2 := hingeLayer2 n c b)
    (x := Tensor.singleton x)]
  -- Reduce the let-bindings introduced by `mlp_forward_eq_linear_relu_linear`.
  dsimp
  -- Compute the first linear layer: `x ↦ (x - t i)`.
  have hz1 :
      Spec.linearSpec (α := ℝ) (hingeLayer1 n t) (Tensor.singleton x) =
        Tensor.dim (fun i : Fin n => Tensor.scalar (x - t i)) := by
    unfold hingeLayer1 Spec.linearSpec
    -- `matVecMulSpec` yields the constant vector `x`; then add the bias `-t`.
    rw [mat_vec_mul_spec_matrix_singleton]
    apply Tensor.ext_vector
    intro i
    rw [congrFun (Spec.getScalar_add_spec _ _) i]
    simp [Tensor.ofFn, sub_eq_add_neg]
  -- Apply ReLU pointwise.
  have ha1 :
      Activation.reluSpec (α := ℝ) (s := .dim n .scalar)
          (Spec.linearSpec (α := ℝ) (hingeLayer1 n t) (Tensor.singleton x)) =
        Tensor.dim (fun i : Fin n => Tensor.scalar (relu (x - t i))) := by
    simp [hz1, Activation.reluSpec, relu]
  -- Compute the second linear layer as a sum of hinges.
  have hy :
      Spec.linearSpec (α := ℝ) (hingeLayer2 n c b)
          (Activation.reluSpec (α := ℝ) (s := .dim n .scalar)
            (Spec.linearSpec (α := ℝ) (hingeLayer1 n t) (Tensor.singleton x))) =
        Tensor.dim (fun _ : Fin 1 => Tensor.scalar ((∑ i : Fin n, c i * relu (x - t i)) + b)) := by
    -- Rewrite the activation output first, so unfolding `Spec.linearSpec` doesn't unfold the inner
    -- layer.
    rw [ha1]
    unfold hingeLayer2 Spec.linearSpec
    -- `mat_vec_mul_spec_matrix_vector` gives the dot product as a `Finset.univ` sum.
    have hmv :
        matVecMulSpec (Tensor.matrix (m := 1) (n := n) (fun _ j => c j))
            (Tensor.dim (fun j => Tensor.scalar (relu (x - t j)))) =
          Tensor.dim (fun _ : Fin 1 => Tensor.scalar (∑ j : Fin n, c j * relu (x - t j))) := by
      simpa using mat_vec_mul_spec_matrix_vector n c (fun j => relu (x - t j))
    rw [hmv]
    apply Tensor.ext_vector
    intro i
    fin_cases i
    simp [Spec.getScalar_add_spec, Tensor.ofFn, add_comm]
  -- Extract the scalar output (the unique element of `Fin 1`) and reorder `b + sum`.
  -- `Fin 1` has a unique element, so `fin_cases` reduces the extracted component.
  simp [hy, Tensor.item, add_comm]

namespace Internal.UniformInterpolation

/-- The uniform grid with left endpoint `a` and spacing `δ`. -/
noncomputable def grid (a δ : ℝ) (k : ℕ) : ℝ := a + (k : ℝ) * δ

/-- The uniform grid starts at its left endpoint. -/
@[simp] theorem grid_zero (a δ : ℝ) : grid a δ 0 = a := by simp [grid]

/-- Consecutive grid points are one mesh width apart. -/
theorem grid_step (a δ : ℝ) (k : ℕ) : grid a δ (k + 1) - grid a δ k = δ := by
  simp only [grid, Nat.cast_add, Nat.cast_one]
  ring

/-- Nonnegative spacing gives an ordered grid. -/
theorem grid_mono (a : ℝ) {δ : ℝ} (hδ : 0 ≤ δ) : Monotone (grid a δ) := by
  intro m n hmn
  exact add_le_add_right (mul_le_mul_of_nonneg_right (Nat.cast_le.mpr hmn) hδ) a

/-- The slope between consecutive, arbitrary sample values. -/
noncomputable def slope (δ : ℝ) (y : ℕ → ℝ) (k : ℕ) : ℝ :=
  (y (k + 1) - y k) / δ

/-- Hinge coefficients are the initial slope followed by successive slope differences. -/
noncomputable def coefficient (δ : ℝ) (y : ℕ → ℝ) : ℕ → ℝ
  | 0 => slope δ y 0
  | k + 1 => slope δ y (k + 1) - slope δ y k

/-- The slope times a nonzero mesh width is the corresponding sample increment. -/
theorem slope_mul_step {δ : ℝ} (hδ : δ ≠ 0) (y : ℕ → ℝ) (k : ℕ) :
    slope δ y k * δ = y (k + 1) - y k := by
  exact div_mul_cancel₀ _ hδ

/-- Summing the slope differences recovers the slope of the last segment. -/
theorem sum_coefficient (δ : ℝ) (y : ℕ → ℝ) (k : ℕ) :
    (∑ i ∈ Finset.range (k + 1), coefficient δ y i) = slope δ y k := by
  induction k with
  | zero => simp [coefficient]
  | succ k ih =>
    rw [Finset.sum_range_succ, ih]
    simp [coefficient]

/-- The width-`N` hinge interpolant of the samples `y 0, …, y N`. -/
noncomputable def interpolant (N : ℕ) (a δ : ℝ) (y : ℕ → ℝ) (x : ℝ) : ℝ :=
  y 0 + ∑ i ∈ Finset.range N, coefficient δ y i * relu (x - grid a δ i)

/-- On each grid cell, the hinge sum is affine with the prescribed sample slope. -/
theorem affine_on_segment {N k : ℕ} {a δ x : ℝ} (y : ℕ → ℝ) (hδ : 0 ≤ δ)
    (hkN : k + 1 ≤ N) (hx0 : grid a δ k ≤ x) (hx1 : x ≤ grid a δ (k + 1)) :
    interpolant N a δ y x = interpolant N a δ y (grid a δ k) +
      slope δ y k * (x - grid a δ k) := by
  classical
  have hmono := grid_mono a hδ
  have hsum :
      (∑ i ∈ Finset.range N, coefficient δ y i *
        (relu (x - grid a δ i) - relu (grid a δ k - grid a δ i))) =
          slope δ y k * (x - grid a δ k) := by
    calc
      _ = ∑ i ∈ Finset.range (k + 1), coefficient δ y i *
          (relu (x - grid a δ i) - relu (grid a δ k - grid a δ i)) := by
        symm
        refine Finset.sum_subset (Finset.range_mono hkN) ?_
        intro i _ hi
        have hki : k + 1 ≤ i := Nat.le_of_not_gt (by simpa using hi)
        rw [relu_sub_eq_zero_of_le (hx1.trans (hmono hki)),
          relu_sub_eq_zero_of_le (hmono ((Nat.le_succ k).trans hki))]
        simp
      _ = ∑ i ∈ Finset.range (k + 1), coefficient δ y i * (x - grid a δ k) := by
        apply Finset.sum_congr rfl
        intro i hi
        have hik : grid a δ i ≤ grid a δ k :=
          hmono (Nat.le_of_lt_succ (Finset.mem_range.mp hi))
        rw [relu_sub_eq_of_le (hik.trans hx0), relu_sub_eq_of_le hik]
        ring
      _ = _ := by rw [← Finset.sum_mul, sum_coefficient]
  have hdiff : interpolant N a δ y x - interpolant N a δ y (grid a δ k) =
      slope δ y k * (x - grid a δ k) := by
    simpa only [interpolant, add_sub_add_left_eq_sub, ← Finset.sum_sub_distrib,
      ← mul_sub] using hsum
  linarith

/-- Positive spacing makes the hinge interpolant exact at every bounded grid point. -/
theorem interpolant_grid {N : ℕ} (a : ℝ) {δ : ℝ} (y : ℕ → ℝ) (hδ : 0 < δ)
    (k : ℕ) (hk : k ≤ N) : interpolant N a δ y (grid a δ k) = y k := by
  classical
  induction k with
  | zero =>
    have hsum : (∑ i ∈ Finset.range N,
        coefficient δ y i * relu (grid a δ 0 - grid a δ i)) = 0 := by
      apply Finset.sum_eq_zero
      intro i _
      rw [relu_sub_eq_zero_of_le (grid_mono a hδ.le (Nat.zero_le i)), mul_zero]
    simp only [interpolant, hsum, add_zero]
  | succ k ih =>
    rw [affine_on_segment y hδ.le hk (grid_mono a hδ.le (Nat.le_succ k)) le_rfl,
      ih ((Nat.le_succ k).trans hk), grid_step, slope_mul_step (ne_of_gt hδ)]
    ring

/-- The finite-index hinge network and the range-sum interpolant have the same value. -/
theorem hinge_fun_eq_interpolant (N : ℕ) (a δ : ℝ) (y : ℕ → ℝ) (x : ℝ) :
    hingeFun N (fun i => grid a δ i.1) (fun i => coefficient δ y i.1) (y 0) x =
      interpolant N a δ y x := by
  unfold hingeFun interpolant
  congr 1
  exact Fin.sum_univ_eq_sum_range
    (f := fun i => coefficient δ y i * relu (x - grid a δ i)) (n := N)

end Internal.UniformInterpolation

/-- A uniform mesh with `N` cells gives a width-`N` hinge approximation whenever
`2 * L * ((b - a) / N)` is below the requested error. The same interpolant serves both the
qualitative existence theorem and the theorem with an explicit width. -/
theorem relu_hinge_approximation_Icc_of_mesh {f : ℝ → ℝ} {a b L ε : ℝ} {N : ℕ}
    (h_ab : a < b) (hL : 0 < L)
    (h_lip : ∀ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, |f x - f y| ≤ L * |x - y|)
    (hNpos_nat : 0 < N)
    (hmesh : 2 * L * ((b - a) / (N : ℝ)) < ε) :
    ∃ (t : Fin N → ℝ) (c : Fin N → ℝ),
      ∀ x ∈ Set.Icc a b, |f x - hingeFun N t c (f a) x| < ε := by
  classical
  have hba : 0 < b - a := sub_pos.mpr h_ab
  have hNpos : 0 < (N : ℝ) := by exact_mod_cast hNpos_nat
  have hNne : (N : ℝ) ≠ 0 := ne_of_gt hNpos
  let δ : ℝ := (b - a) / (N : ℝ)
  have hδpos : 0 < δ := div_pos hba hNpos
  have hδnonneg : 0 ≤ δ := le_of_lt hδpos
  have hδne : δ ≠ 0 := ne_of_gt hδpos
  have h2mesh : 2 * L * δ < ε := hmesh
  have hε : 0 < ε := (mul_pos (mul_pos two_pos hL) hδpos).trans h2mesh

  let grid := Internal.UniformInterpolation.grid a δ
  let y : ℕ → ℝ := fun k => f (grid k)
  let mNat := Internal.UniformInterpolation.slope δ y
  let cNat := Internal.UniformInterpolation.coefficient δ y
  let g := Internal.UniformInterpolation.interpolant N a δ y
  have hgrid0 : grid 0 = a := Internal.UniformInterpolation.grid_zero a δ
  have hgridN : grid N = b := by
    dsimp [grid, Internal.UniformInterpolation.grid, δ]
    rw [mul_div_cancel₀ _ hNne]
    ring
  have hstep : ∀ k : ℕ, grid (k + 1) - grid k = δ :=
    Internal.UniformInterpolation.grid_step a δ
  have hmδ : ∀ k : ℕ, mNat k * δ = f (grid (k + 1)) - f (grid k) :=
    Internal.UniformInterpolation.slope_mul_step hδne y
  have grid_mono : Monotone grid := Internal.UniformInterpolation.grid_mono a hδnonneg
  have g_affine_on_segment :
      ∀ {k : ℕ}, k + 1 ≤ N → ∀ {x : ℝ}, grid k ≤ x → x ≤ grid (k + 1) →
        g x = g (grid k) + mNat k * (x - grid k) := by
    intro k hkN x hx0 hx1
    exact Internal.UniformInterpolation.affine_on_segment y hδnonneg hkN hx0 hx1
  have g_grid_eq_f_grid : ∀ k : ℕ, k ≤ N → g (grid k) = f (grid k) :=
    Internal.UniformInterpolation.interpolant_grid a y hδpos

  -- Build the network corresponding to the polygonal interpolant.
  let t : Fin N → ℝ := fun i => grid i.1
  let c : Fin N → ℝ := fun i => cNat i.1
  refine ⟨t, c, ?_⟩
  intro x hx
  have hhinge : hingeFun N t c (f a) x = g x := by
    simpa only [y, hgrid0] using
      Internal.UniformInterpolation.hinge_fun_eq_interpolant N a δ y x
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
      -- at `x = a`, `g a = f a`
      simpa [hx_eq, hg_a, abs_zero] using hε
    simpa [hhinge] using this
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
      have hgrid_mem (i : ℕ) (hi : i ≤ N) : grid i ∈ Set.Icc a b := by
        have ha : a ≤ grid i := by
          have : grid 0 ≤ grid i := grid_mono (Nat.zero_le i)
          simpa [hgrid0] using this
        have hb : grid i ≤ b := by
          have : grid i ≤ grid N := grid_mono hi
          simpa [hgridN] using this
        exact ⟨ha, hb⟩
      have hx_dist : |x - grid k| ≤ δ := by
        have hx0' : 0 ≤ x - grid k := sub_nonneg.mpr hx0
        have hxle : x - grid k ≤ δ := by
          have : x - grid k ≤ grid (k + 1) - grid k := sub_le_sub_right hx1 (grid k)
          simpa [hstep k] using this
        have habs : |x - grid k| = x - grid k := abs_of_nonneg hx0'
        simpa [habs] using hxle
      have h1 : |f x - f (grid k)| ≤ L * δ := by
        have := h_lip x hx (grid k) (hgrid_mem k hk_le)
        exact le_trans this (mul_le_mul_of_nonneg_left hx_dist hL.le)
      have h2 : |f (grid k) - g x| ≤ L * δ := by
        -- g x is between the endpoints; bound by the endpoint difference.
        have habs' : |f (grid k) - g x| = |mNat k * (x - grid k)| := by
          have hgx : g x = f (grid k) + mNat k * (x - grid k) := by
            simpa [hgk] using haff
          have hs : f (grid k) - g x = -(mNat k * (x - grid k)) := by
            simp [hgx]
          have : |f (grid k) - g x| = |-(mNat k * (x - grid k))| := by
            simpa using congrArg abs hs
          simpa [abs_neg] using this
        -- use |x-grid k| ≤ δ to bound
        have hmul_le : |mNat k * (x - grid k)| ≤ |mNat k| * δ := by
          have := mul_le_mul_of_nonneg_left hx_dist (abs_nonneg (mNat k))
          simpa [abs_mul] using this
        have hmul_eq : |mNat k| * δ = |mNat k * δ| := by
          simp [abs_mul, abs_of_nonneg hδnonneg, mul_comm]
        have hendpoint : |mNat k * δ| ≤ L * δ := by
          -- from Lipschitz on grid points
          have hdiff := h_lip (grid (k + 1)) (hgrid_mem (k + 1) hkN)
            (grid k) (hgrid_mem k hk_le)
          have hstepAbs : |grid (k + 1) - grid k| = δ := by
            rw [hstep k]
            exact abs_of_nonneg hδnonneg
          -- rewrite |mNat k * δ| as |f(grid(k+1)) - f(grid k)|
          have : |mNat k * δ| = |f (grid (k + 1)) - f (grid k)| := by simp [hmδ k]
          simpa [this, hstepAbs] using hdiff
        -- combine bounds
        have hbound : |mNat k * (x - grid k)| ≤ |mNat k * δ| := by
          exact le_trans hmul_le (le_of_eq hmul_eq)
        have hfinal : |mNat k * (x - grid k)| ≤ L * δ := le_trans hbound hendpoint
        simpa [habs'] using hfinal
      have : |f x - g x| ≤ 2 * L * δ := by
        have htri : |f x - g x| ≤ |f x - f (grid k)| + |f (grid k) - g x| := by
          exact abs_sub_le (f x) (f (grid k)) (g x)
        have hsum : |f x - f (grid k)| + |f (grid k) - g x| ≤ L * δ + L * δ :=
          add_le_add h1 (by simpa [abs_sub_comm] using h2)
        have hsum' : L * δ + L * δ = 2 * L * δ := by ring
        exact le_trans htri (by simpa [hsum'] using hsum)
      exact lt_of_le_of_lt this h2mesh
    simpa [hhinge] using hfx

/-- Explicit hidden width for the 1D Lipschitz ReLU approximation construction. -/
noncomputable def reluApproximationWidth (L a b ε : ℝ) : ℕ :=
  Nat.ceil (2 * L * (b - a) / ε) + 1

/-- The explicit ReLU approximation width is always positive. -/
theorem relu_approximation_width_pos (L a b ε : ℝ) : 0 < reluApproximationWidth L a b ε :=
  Nat.succ_pos _

/--
The chosen width makes the mesh-size error term smaller than the target accuracy.

This is the arithmetic heart of the explicit-rate theorem: the ceiling construction ensures
$N>2L(b-a)/\varepsilon$, hence $2L(b-a)/N<\varepsilon$.
-/
theorem two_mul_mul_sub_div_relu_approximation_width_lt {L a b ε : ℝ} (hε : 0 < ε) :
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
  -- Clear the `ε` denominator in `hr_lt`, then the `N` denominator in the goal.
  have hnum : 2 * L * (b - a) < (N : ℝ) * ε := (div_lt_iff₀ hε).1 hr_lt
  exact (div_lt_iff₀ hNpos).2 (by simpa [mul_comm, mul_assoc] using hnum)

/--
1D Universal Approximation (ReLU, one hidden layer).

This is the classic constructive proof:
Lipschitz continuity on $[a,b]$, a uniform partition, and piecewise-linear interpolation,
then represent the interpolant as a finite linear combination of hinges
$\operatorname{ReLU}(x-t_i)$.
-/
theorem relu_universal_approximation_Icc_hinge {f : ℝ → ℝ} {a b L : ℝ}
    (h_ab : a < b) (hL : 0 < L)
    (h_lip : ∀ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, |f x - f y| ≤ L * |x - y|) :
    ∀ ε > 0, ∃ (hidDim : ℕ) (t : Fin hidDim → ℝ) (c : Fin hidDim → ℝ),
      ∀ x ∈ Set.Icc a b, |f x - hingeFun hidDim t c (f a) x| < ε := by
  intro ε hε
  obtain ⟨t, c, happrox⟩ :=
    relu_hinge_approximation_Icc_of_mesh h_ab hL h_lip
      (relu_approximation_width_pos L a b ε)
      (by simpa [mul_div_assoc', mul_assoc] using
        two_mul_mul_sub_div_relu_approximation_width_lt (L := L) (a := a) (b := b) hε)
  exact ⟨_, t, c, happrox⟩

/--
1D Universal Approximation (ReLU, one hidden layer), stated as an existence theorem for a 2-layer
MLP.

This is a wrapper around `relu_universal_approximation_Icc_hinge` that instantiates the linear
layers as the explicit hinge construction.
-/
theorem relu_universal_approximation_Icc {f : ℝ → ℝ} {a b L : ℝ}
    (h_ab : a < b) (hL : 0 < L)
    (h_lip : ∀ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, |f x - f y| ≤ L * |x - y|) :
    ∀ ε > 0, ∃ (hidDim : ℕ) (l1 : LinearSpec ℝ 1 hidDim) (l2 : LinearSpec ℝ hidDim 1),
      ∀ x ∈ Set.Icc a b, |f x - mlpEvalScalar hidDim l1 l2 x| < ε := by
  intro ε hε
  classical
  rcases
      relu_universal_approximation_Icc_hinge (f := f) (a := a) (b := b) (L := L)
        h_ab hL h_lip ε hε with
    ⟨hidDim, t, c, happx⟩
  refine ⟨hidDim, hingeLayer1 hidDim t, hingeLayer2 hidDim c (f a), ?_⟩
  intro x hx
  have hnet :
      mlpEvalScalar hidDim (hingeLayer1 hidDim t) (hingeLayer2 hidDim c (f a)) x =
        hingeFun hidDim t c (f a) x := by
    simpa using (mlp_eval_scalar_hinge hidDim t c (f a) x)
  simpa [hnet] using happx x hx

end NN.MLTheory.Proofs.UniversalApproximation
