/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Models.MLP
public import TorchLean.Tensor.Reduce.Real
public import TorchLean.Tensor.Context.Real
public import TorchLean.Tensor.Contract

/-!
# Universal approximation (1D, constructive)

On a compact interval $I=[a,b]$, every function $f:\mathbb{R}\to\mathbb{R}$ that is Lipschitz on
$I$ is uniformly approximated by a single-hidden-layer ReLU network (a 2-layer MLP).

The proof is the classic construction:

- approximate $f$ by a polygonal function on a uniform grid,
- write that polygonal function as the constant $f(a)$ plus a weighted sum of hinges
  $\operatorname{ReLU}(x-t_i)$, with the first knot at $a$,
- package the hinge representation as TorchLean's spec-level 2-layer MLP
  (`TorchLean.NN.Models.MLP`).

## Main result

- `relu_Icc`: existence of a 2-layer ReLU MLP approximator on `Set.Icc a b`.

## References

- Leshno, Lin, Pinkus, Schocken (1993), *Multilayer feedforward networks with a nonpolynomial
  activation function can approximate any function*.
- Yarotsky (2017), *Error bounds for approximations with deep ReLU networks*.
- Pinkus (1999), *Approximation theory of the MLP model in neural networks*.
-/

@[expose] public section

namespace TorchLean.UniversalApproximation

open _root_.TorchLean.Tensor
open Activation (relu)

/-- If the knot $t$ is to the left of $x$, then $\operatorname{ReLU}(x-t)=x-t$. -/
theorem relu_sub_eq_of_le {x t : ℝ} (h : t ≤ x) : Activation.relu (x - t) = x - t := by
  have : 0 ≤ x - t := sub_nonneg.mpr h
  rw [Activation.relu_eq_max, max_eq_left this]

/-- If $x$ is to the left of the knot $t$, then $\operatorname{ReLU}(x-t)=0$. -/
theorem relu_sub_eq_zero_of_le {x t : ℝ} (h : x ≤ t) : Activation.relu (x - t) = 0 := by
  have : x - t ≤ 0 := sub_nonpos.mpr h
  rw [Activation.relu_eq_max, max_eq_right this]

/-- Extract the unique entry of a length-`1` tensor. -/
def extractScalarOutput {α : Type} [Storage α] (t : Tensor α [1]) : α :=
  t.getScalar ⟨0, by norm_num⟩

/-- Evaluate a 2-layer ReLU MLP on a scalar input. -/
def mlpEvalScalar {α : Type} [Storage α] [Context α] (hidDim : ℕ)
    (l1 : Linear α 1 hidDim) (l2 : Linear α hidDim 1) (x : α) : α :=
  extractScalarOutput (MLP.forward l1 l2 (Tensor.singleton x))

/--
TorchLean's MLP forward pass is definitionally
$\operatorname{linear}\circ\operatorname{ReLU}\circ\operatorname{linear}$.
-/
theorem _root_.TorchLean.MLP.forward_def {n hidDim : ℕ}
    (l1 : Linear ℝ n hidDim) (l2 : Linear ℝ hidDim 1) (x : Tensor ℝ [n]) :
    MLP.forward l1 l2 x =
      let z1 := Linear.forward (α := ℝ) l1 x
      let a1 := Tensor.relu z1
      Linear.forward (α := ℝ) l2 a1 := by
  rfl

/-- First real hinge layer: hidden unit $i$ computes $x-t_i$ before ReLU. -/
noncomputable def hingeLayer1 (n : ℕ) (t : Fin n → ℝ) : Linear ℝ 1 n :=
  { weights := Tensor.matrix (m := n) (n := 1) (fun _ _ => (1 : ℝ))
    bias := Tensor.ofFn (n := n) (fun i => -t i) }

/-- Second real hinge layer: sum hidden activations with coefficients $c_i$ and bias $b$. -/
noncomputable def hingeLayer2 (n : ℕ) (c : Fin n → ℝ) (b : ℝ) : Linear ℝ n 1 :=
  { weights := Tensor.matrix (m := 1) (n := n) (fun _ j => c j)
    bias := Tensor.ofFn (n := 1) (fun _ => b) }

/-- The hinge term $c_i\operatorname{ReLU}(x-t_i)$. -/
def hingeTerm {α : Type} [Context α] {n : ℕ} (c t : Fin n → α) (x : α) (i : Fin n) : α :=
  c i * Activation.relu (x - t i)

/-- The hinge terms added to `0` in ascending index order. -/
def hingeSum {α : Type} [Context α] {n : ℕ} (c t : Fin n → α) (x : α) : α :=
  Fin.foldl n (fun acc i => acc + hingeTerm c t x i) 0

/-- The hinge network: the hinge sum plus the bias `b`. -/
def hingeFun {α : Type} [Context α] {n : ℕ} (t c : Fin n → α) (b x : α) : α :=
  hingeSum c t x + b

/-- Over `ℝ` the hinge sum is the finite sum of the hinge terms. -/
theorem hingeSum_eq_sum {n : ℕ} (c t : Fin n → ℝ) (x : ℝ) :
    hingeSum c t x = ∑ i, c i * Activation.relu (x - t i) := by
  induction n with
  | zero => simp [hingeSum]
  | succ n ih =>
    rw [hingeSum, Fin.foldl_succ_last, Fin.sum_univ_castSucc]
    exact congrArg (· + hingeTerm c t x (Fin.last n))
      (ih (fun i => c i.castSucc) (fun i => t i.castSucc))

/-- Over `ℝ` the hinge network is $b+\sum_i c_i\operatorname{ReLU}(x-t_i)$. -/
theorem hingeFun_eq_add_sum {n : ℕ} (t c : Fin n → ℝ) (b x : ℝ) :
    hingeFun t c b x = b + ∑ i, c i * Activation.relu (x - t i) := by
  rw [hingeFun, hingeSum_eq_sum, add_comm]

/-- A one-row matrix times a vector is the finite dot product $\sum_j c_j v_j$. -/
theorem contract_matVec_matrix_row (n : ℕ) (c v : Fin n → ℝ) :
    Tensor.contract (Tensor.matrix (m := 1) (n := n) (fun _ j => c j))
        (Tensor.dim (fun j => Tensor.scalar (v j))) _ _ [1] .matVec =
      Tensor.dim (fun _ => Tensor.scalar (∑ j : Fin n, c j * v j)) := by
  classical
  apply Tensor.ext_vector
  intro i
  fin_cases i
  simp [Tensor.getScalar_contract_matVec_eq_sum, Tensor.matrix, Tensor.get2]

/-- The all-ones column times a length-1 input copies the scalar `x` into every hidden unit. -/
theorem contract_matVec_ones (n : ℕ) (x : ℝ) :
    Tensor.contract (Tensor.matrix (m := n) (n := 1) (fun _ _ => (1 : ℝ)))
        (Tensor.singleton x) _ _ [n] .matVec =
      Tensor.dim (fun _ : Fin n => Tensor.scalar x) := by
  classical
  apply Tensor.ext_vector
  intro i
  simp [Tensor.getScalar_contract_matVec_eq_sum, Tensor.matrix, Tensor.get2,
    Tensor.singleton]

/--
The two-layer network built from `hingeLayer1` and `hingeLayer2` computes `hingeFun`.

The approximation argument runs on hinge sums, and this equation carries its result over to
TorchLean's spec-level MLP model.
-/
theorem mlpEvalScalar_hinge (n : ℕ) (t : Fin n → ℝ) (c : Fin n → ℝ) (b x : ℝ) :
    mlpEvalScalar n (hingeLayer1 n t) (hingeLayer2 n c b) x = hingeFun t c b x := by
  classical
  rw [hingeFun_eq_add_sum]
  unfold mlpEvalScalar extractScalarOutput
  -- `simp` unfolds too much under matchers here, so rewrite `MLP.forward` with the lemma.
  rw [MLP.forward_def (l1 := hingeLayer1 n t) (l2 := hingeLayer2 n c b)
    (x := Tensor.singleton x)]
  -- Reduce the let-bindings introduced by `MLP.forward_def`.
  dsimp
  -- Compute the first linear layer: `x ↦ (x - t i)`.
  have hz1 :
      Linear.forward (α := ℝ) (hingeLayer1 n t) (Tensor.singleton x) =
        Tensor.dim (fun i : Fin n => Tensor.scalar (x - t i)) := by
    unfold hingeLayer1 Linear.forward
    -- The contraction gives the constant vector `x`, and the bias adds `-t`.
    rw [contract_matVec_ones]
    apply Tensor.ext_vector
    intro i
    rw [congrFun (Tensor.getScalar_add _ _) i]
    simp [Tensor.ofFn, sub_eq_add_neg]
  -- Apply ReLU pointwise.
  have ha1 :
      Tensor.relu (α := ℝ) (s := .dim n .scalar)
          (Linear.forward (α := ℝ) (hingeLayer1 n t) (Tensor.singleton x)) =
        Tensor.dim (fun i : Fin n => Tensor.scalar (Activation.relu (x - t i))) := by
    simp [hz1, Tensor.relu]
  -- Compute the second linear layer as a sum of hinges.
  have hy :
      Linear.forward (α := ℝ) (hingeLayer2 n c b)
          (Tensor.relu (α := ℝ) (s := .dim n .scalar)
            (Linear.forward (α := ℝ) (hingeLayer1 n t) (Tensor.singleton x))) =
        Tensor.dim (fun _ : Fin 1 =>
          Tensor.scalar ((∑ i : Fin n, c i * Activation.relu (x - t i)) + b)) := by
    -- Rewrite the activation output first, so unfolding `Linear.forward` doesn't unfold the inner
    -- layer.
    rw [ha1]
    unfold hingeLayer2 Linear.forward
    -- `contract_matVec_matrix_row` gives the dot product as a `Finset.univ` sum.
    have hmv :
        Tensor.contract (Tensor.matrix (m := 1) (n := n) (fun _ j => c j))
            (Tensor.dim (fun j => Tensor.scalar (Activation.relu (x - t j)))) _ _ [1] .matVec =
          Tensor.dim (fun _ : Fin 1 => Tensor.scalar
            (∑ j : Fin n, c j * Activation.relu (x - t j))) := by
      simpa using contract_matVec_matrix_row n c (fun j => Activation.relu (x - t j))
    rw [hmv]
    apply Tensor.ext_vector
    intro i
    fin_cases i
    simp [Tensor.getScalar_add, Tensor.ofFn, add_comm]
  -- Read the single output entry and reorder `sum + b` as `b + sum`.
  simp [hy, Tensor.item, add_comm]

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

/-- The slope between consecutive sample values `y k` and `y (k + 1)` at spacing `δ`. -/
noncomputable def gridSlope (δ : ℝ) (y : ℕ → ℝ) (k : ℕ) : ℝ :=
  (y (k + 1) - y k) / δ

/-- Hinge coefficients are the initial slope followed by successive slope differences. -/
noncomputable def hingeCoeff (δ : ℝ) (y : ℕ → ℝ) : ℕ → ℝ
  | 0 => gridSlope δ y 0
  | k + 1 => gridSlope δ y (k + 1) - gridSlope δ y k

/-- The slope times a nonzero mesh width is the corresponding sample increment. -/
theorem gridSlope_mul {δ : ℝ} (hδ : δ ≠ 0) (y : ℕ → ℝ) (k : ℕ) :
    gridSlope δ y k * δ = y (k + 1) - y k := by
  exact div_mul_cancel₀ _ hδ

/-- Summing the slope differences recovers the slope of the last segment. -/
theorem sum_hingeCoeff (δ : ℝ) (y : ℕ → ℝ) (k : ℕ) :
    (∑ i ∈ Finset.range (k + 1), hingeCoeff δ y i) = gridSlope δ y k := by
  induction k with
  | zero => simp [hingeCoeff]
  | succ k ih =>
    rw [Finset.sum_range_succ, ih]
    simp [hingeCoeff]

/-- The width-`N` hinge interpolant of the samples `y 0, …, y N`. -/
noncomputable def interpolant (N : ℕ) (a δ : ℝ) (y : ℕ → ℝ) (x : ℝ) : ℝ :=
  y 0 + ∑ i ∈ Finset.range N, hingeCoeff δ y i * Activation.relu (x - grid a δ i)

/-- On each grid cell, the hinge sum is affine with the prescribed sample slope. -/
theorem interpolant_affine {N k : ℕ} {a δ x : ℝ} (y : ℕ → ℝ) (hδ : 0 ≤ δ)
    (hkN : k + 1 ≤ N) (hx0 : grid a δ k ≤ x) (hx1 : x ≤ grid a δ (k + 1)) :
    interpolant N a δ y x = interpolant N a δ y (grid a δ k) +
      gridSlope δ y k * (x - grid a δ k) := by
  classical
  have hmono := grid_mono a hδ
  have hsum :
      (∑ i ∈ Finset.range N, hingeCoeff δ y i *
        (Activation.relu (x - grid a δ i) - Activation.relu (grid a δ k - grid a δ i))) =
          gridSlope δ y k * (x - grid a δ k) := by
    calc
      _ = ∑ i ∈ Finset.range (k + 1), hingeCoeff δ y i *
          (Activation.relu (x - grid a δ i) - Activation.relu (grid a δ k - grid a δ i)) := by
        symm
        refine Finset.sum_subset (Finset.range_mono hkN) ?_
        intro i _ hi
        have hki : k + 1 ≤ i := Nat.le_of_not_gt (by simpa using hi)
        rw [relu_sub_eq_zero_of_le (hx1.trans (hmono hki)),
          relu_sub_eq_zero_of_le (hmono ((Nat.le_succ k).trans hki))]
        simp
      _ = ∑ i ∈ Finset.range (k + 1), hingeCoeff δ y i * (x - grid a δ k) := by
        apply Finset.sum_congr rfl
        intro i hi
        have hik : grid a δ i ≤ grid a δ k :=
          hmono (Nat.le_of_lt_succ (Finset.mem_range.mp hi))
        rw [relu_sub_eq_of_le (hik.trans hx0), relu_sub_eq_of_le hik]
        ring
      _ = _ := by rw [← Finset.sum_mul, sum_hingeCoeff]
  have hdiff : interpolant N a δ y x - interpolant N a δ y (grid a δ k) =
      gridSlope δ y k * (x - grid a δ k) := by
    simpa only [interpolant, add_sub_add_left_eq_sub, ← Finset.sum_sub_distrib,
      ← mul_sub] using hsum
  linarith

/-- With positive spacing, the hinge interpolant takes the sample value `y k` at every grid point
with `k ≤ N`. -/
theorem interpolant_grid {N : ℕ} (a : ℝ) {δ : ℝ} (y : ℕ → ℝ) (hδ : 0 < δ)
    (k : ℕ) (hk : k ≤ N) : interpolant N a δ y (grid a δ k) = y k := by
  classical
  induction k with
  | zero =>
    have hsum : (∑ i ∈ Finset.range N,
        hingeCoeff δ y i * Activation.relu (grid a δ 0 - grid a δ i)) = 0 := by
      apply Finset.sum_eq_zero
      intro i _
      rw [relu_sub_eq_zero_of_le (grid_mono a hδ.le (Nat.zero_le i)), mul_zero]
    simp only [interpolant, hsum, add_zero]
  | succ k ih =>
    rw [interpolant_affine y hδ.le hk (grid_mono a hδ.le (Nat.le_succ k)) le_rfl,
      ih ((Nat.le_succ k).trans hk), grid_step, gridSlope_mul (ne_of_gt hδ)]
    ring

/-- The finite-index hinge network and the range-sum interpolant have the same value. -/
theorem hingeFun_eq_interpolant (N : ℕ) (a δ : ℝ) (y : ℕ → ℝ) (x : ℝ) :
    hingeFun (fun i : Fin N => grid a δ i.1) (fun i => hingeCoeff δ y i.1) (y 0) x =
      interpolant N a δ y x := by
  rw [hingeFun_eq_add_sum]
  unfold interpolant
  congr 1
  exact Fin.sum_univ_eq_sum_range
    (f := fun i => hingeCoeff δ y i * Activation.relu (x - grid a δ i)) (n := N)

/-- Let `f` be `L`-Lipschitz on $[a,b]$ with `a < b` and `0 < L`. If
`2 * L * ((b - a) / N) < ε` with `0 < N`, the hinge interpolant on the uniform mesh with `N` cells
is a width-`N` hinge network within `ε` of `f` on $[a,b]$. The existence theorem and the
fixed-width rate theorem both use it. -/
theorem relu_Icc_hinge_of_mesh {f : ℝ → ℝ} {a b L ε : ℝ} {N : ℕ}
    (h_ab : a < b) (hL : 0 < L)
    (h_lip : ∀ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, |f x - f y| ≤ L * |x - y|)
    (hNpos_nat : 0 < N)
    (hmesh : 2 * L * ((b - a) / (N : ℝ)) < ε) :
    ∃ (t : Fin N → ℝ) (c : Fin N → ℝ),
      ∀ x ∈ Set.Icc a b, |f x - hingeFun t c (f a) x| < ε := by
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

  let grid := UniversalApproximation.grid a δ
  let y : ℕ → ℝ := fun k => f (grid k)
  let mNat := gridSlope δ y
  let cNat := hingeCoeff δ y
  let g := interpolant N a δ y
  have hgrid0 : grid 0 = a := grid_zero a δ
  have hgridN : grid N = b := by
    dsimp [grid, UniversalApproximation.grid, δ]
    rw [mul_div_cancel₀ _ hNne]
    ring
  have hstep : ∀ k : ℕ, grid (k + 1) - grid k = δ :=
    grid_step a δ
  have hmδ : ∀ k : ℕ, mNat k * δ = f (grid (k + 1)) - f (grid k) :=
    gridSlope_mul hδne y
  have grid_mono : Monotone grid := UniversalApproximation.grid_mono a hδnonneg
  have g_affine_on_segment :
      ∀ {k : ℕ}, k + 1 ≤ N → ∀ {x : ℝ}, grid k ≤ x → x ≤ grid (k + 1) →
        g x = g (grid k) + mNat k * (x - grid k) := by
    intro k hkN x hx0 hx1
    exact interpolant_affine y hδnonneg hkN hx0 hx1
  have g_grid_eq_f_grid : ∀ k : ℕ, k ≤ N → g (grid k) = f (grid k) :=
    interpolant_grid a y hδpos

  -- Build the network corresponding to the polygonal interpolant.
  let t : Fin N → ℝ := fun i => grid i.1
  let c : Fin N → ℝ := fun i => cNat i.1
  refine ⟨t, c, ?_⟩
  intro x hx
  have hhinge : hingeFun t c (f a) x = g x := by
    simpa only [y, hgrid0] using
      hingeFun_eq_interpolant N a δ y x
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
      -- At `x = a` the interpolant equals `f a`.
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
        -- `g x` lies between the cell's endpoint values, so their difference bounds this term.
        have habs' : |f (grid k) - g x| = |mNat k * (x - grid k)| := by
          have hgx : g x = f (grid k) + mNat k * (x - grid k) := by
            simpa [hgk] using haff
          have hs : f (grid k) - g x = -(mNat k * (x - grid k)) := by
            simp [hgx]
          have : |f (grid k) - g x| = |-(mNat k * (x - grid k))| := by
            simpa using congrArg _root_.abs hs
          simpa [abs_neg] using this
        -- `|x - grid k| ≤ δ` bounds the slope term by `|mNat k| * δ`.
        have hmul_le : |mNat k * (x - grid k)| ≤ |mNat k| * δ := by
          have := mul_le_mul_of_nonneg_left hx_dist (abs_nonneg (mNat k))
          simpa [abs_mul] using this
        have hmul_eq : |mNat k| * δ = |mNat k * δ| := by
          simp [abs_mul, abs_of_nonneg hδnonneg, mul_comm]
        have hendpoint : |mNat k * δ| ≤ L * δ := by
          -- Lipschitz at the two grid endpoints.
          have hdiff := h_lip (grid (k + 1)) (hgrid_mem (k + 1) hkN)
            (grid k) (hgrid_mem k hk_le)
          have hstepAbs : |grid (k + 1) - grid k| = δ := by
            rw [hstep k]
            exact abs_of_nonneg hδnonneg
          -- rewrite |mNat k * δ| as |f(grid(k+1)) - f(grid k)|
          have : |mNat k * δ| = |f (grid (k + 1)) - f (grid k)| := by simp [hmδ k]
          simpa [this, hstepAbs] using hdiff
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

/-- Hidden width $\lceil 2L(b-a)/\varepsilon\rceil+1$ for the 1D Lipschitz ReLU construction. -/
noncomputable def width (L a b ε : ℝ) : ℕ :=
  Nat.ceil (2 * L * (b - a) / ε) + 1

/-- `width L a b ε` is positive for all arguments. -/
theorem width_pos (L a b ε : ℝ) : 0 < width L a b ε :=
  Nat.succ_pos _

/--
For $\varepsilon>0$, the width $N$ = `width L a b ε` makes the mesh error term
$2L(b-a)/N$ smaller than $\varepsilon$.

The ceiling gives $N>2L(b-a)/\varepsilon$, and clearing denominators gives the bound.
-/
theorem div_width_lt {L a b ε : ℝ} (hε : 0 < ε) :
    (2 * L * (b - a)) / (width L a b ε : ℝ) < ε := by
  classical
  let N : ℕ := width L a b ε
  have hNpos_nat : 0 < N := width_pos L a b ε
  have hNpos : 0 < (N : ℝ) := by exact_mod_cast hNpos_nat
  have hr_lt : (2 * L * (b - a) / ε : ℝ) < (N : ℝ) := by
    have hr_le :
        (2 * L * (b - a) / ε : ℝ) ≤ (Nat.ceil (2 * L * (b - a) / ε) : ℝ) :=
      Nat.le_ceil _
    have : (2 * L * (b - a) / ε : ℝ) < (Nat.ceil (2 * L * (b - a) / ε) : ℝ) + 1 := by
      linarith
    simpa [N, width, Nat.cast_add, Nat.cast_one, add_assoc] using this
  -- Clear the `ε` denominator in `hr_lt`, then the `N` denominator in the goal.
  have hnum : 2 * L * (b - a) < (N : ℝ) * ε := (div_lt_iff₀ hε).1 hr_lt
  exact (div_lt_iff₀ hNpos).2 (by simpa [mul_comm, mul_assoc] using hnum)

/--
1D universal approximation (ReLU, one hidden layer) in hinge form: if $a<b$ and $f$ is
$L$-Lipschitz on $[a,b]$ with $L>0$, then for every $\varepsilon>0$ some hinge network with bias
$f(a)$ is within $\varepsilon$ of $f$ on $[a,b]$.

The proof is the classic construction: sample $f$ on a uniform partition, interpolate piecewise
linearly, and write the interpolant as a finite linear combination of hinges
$\operatorname{ReLU}(x-t_i)$.
-/
theorem relu_Icc_hinge {f : ℝ → ℝ} {a b L : ℝ}
    (h_ab : a < b) (hL : 0 < L)
    (h_lip : ∀ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, |f x - f y| ≤ L * |x - y|) :
    ∀ ε > 0, ∃ (hidDim : ℕ) (t : Fin hidDim → ℝ) (c : Fin hidDim → ℝ),
      ∀ x ∈ Set.Icc a b, |f x - hingeFun t c (f a) x| < ε := by
  intro ε hε
  obtain ⟨t, c, happrox⟩ :=
    relu_Icc_hinge_of_mesh h_ab hL h_lip
      (width_pos L a b ε)
      (by simpa [mul_div_assoc', mul_assoc] using
        div_width_lt (L := L) (a := a) (b := b) hε)
  exact ⟨_, t, c, happrox⟩

/--
1D universal approximation by a 2-layer ReLU MLP: if $f$ is $L$-Lipschitz on $[a,b]$ with $L>0$,
then for every $\varepsilon>0$ some hidden width and pair of linear layers give an MLP within
$\varepsilon$ of $f$ on $[a,b]$.

The layers are `hingeLayer1` and `hingeLayer2` loaded with the hinge network of
`relu_Icc_hinge`.
-/
theorem relu_Icc {f : ℝ → ℝ} {a b L : ℝ}
    (h_ab : a < b) (hL : 0 < L)
    (h_lip : ∀ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, |f x - f y| ≤ L * |x - y|) :
    ∀ ε > 0, ∃ (hidDim : ℕ) (l1 : Linear ℝ 1 hidDim) (l2 : Linear ℝ hidDim 1),
      ∀ x ∈ Set.Icc a b, |f x - mlpEvalScalar hidDim l1 l2 x| < ε := by
  intro ε hε
  classical
  rcases
      relu_Icc_hinge (f := f) (a := a) (b := b) (L := L)
        h_ab hL h_lip ε hε with
    ⟨hidDim, t, c, happx⟩
  refine ⟨hidDim, hingeLayer1 hidDim t, hingeLayer2 hidDim c (f a), ?_⟩
  intro x hx
  have hnet :
      mlpEvalScalar hidDim (hingeLayer1 hidDim t) (hingeLayer2 hidDim c (f a)) x =
        hingeFun t c (f a) x := by
    simpa using (mlpEvalScalar_hinge hidDim t c (f a) x)
  simpa [hnet] using happx x hx

end TorchLean.UniversalApproximation
