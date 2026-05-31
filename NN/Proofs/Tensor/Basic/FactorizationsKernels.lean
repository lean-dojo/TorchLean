/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Basic.Factorizations
public import NN.Proofs.Tensor.Basic.FactorizationsOrthonormal
public import Mathlib.Data.Real.StarOrdered
public import Mathlib.Analysis.Matrix.Order
public import Mathlib.Analysis.Normed.Algebra.Exponential
public import Mathlib.Analysis.SpecialFunctions.Exponential
public import Mathlib.Topology.Instances.Matrix

/-!
# CHD mode kernels are symmetric positive-semidefinite

The entire verified CHD solve / `find_gamma` / `Z_test` development takes the kernel matrix `K` as
input under the hypothesis `(Matrix.of K).PosSemidef`. CHD does not receive `K`; it *builds* it from
data (`Modes/kernels.py`). This file discharges that standing hypothesis for the **linear mode** — the
first and simplest of CHD's kernels — exactly as the positive-pivot keystone discharged the Cholesky
success condition.

The linear-mode kernel is `K[i,j] = 1 + scale · ⟨Φ i, Φ j⟩` with `Φ` the column-masked data, i.e.

`K = 𝟙𝟙ᵀ + scale · Φ·Φᵀ`,

a sum of the all-ones matrix (a rank-one Gram, PSD) and a scaled Gram matrix `Φ·Φᵀ` (PSD for
`scale ≥ 0` by `posSemidef_self_mul_conjTranspose`). `PosSemidef.add` / `PosSemidef.smul` finish it.

* `linearKernelFn_posSemidef` — `(Matrix.of (linearKernelFn X w scale)).PosSemidef` for `0 ≤ scale`.
* `linearKernelFn_symm` — `K` is symmetric (a corollary, via `PosSemidef.isHermitian`).
* `linearKernelSpec_posSemidef` — the tensor-level statement, the form the solve theorems consume.

The **quadratic mode** is `K[i,j] = scale·(alpha + ⟨Φ i, Φ j⟩)² + (1 − alpha²·scale)`, which expands
algebraically to

`K = 𝟙𝟙ᵀ + (2·scale·alpha)·Φ·Φᵀ + scale·(Φ·Φᵀ ⊙ Φ·Φᵀ)`,

a sum of: the all-ones Gram (PSD), a nonnegative multiple of the Gram `Φ·Φᵀ` (PSD), and a nonnegative
multiple of the **Hadamard square** of that Gram — PSD by the **Schur product theorem**
(`PosSemidef.hadamard`). So `K` is PSD whenever `scale ≥ 0` and `alpha ≥ 0`.

* `quadraticKernelFn_posSemidef` — `(Matrix.of (quadraticKernelFn X w scale alpha)).PosSemidef` for
  `0 ≤ scale` and `0 ≤ alpha`.
* `quadraticKernelFn_symm` / `quadraticKernelSpec_posSemidef` — symmetry and the tensor-level form.

The **Gaussian mode** product kernel
`K[i,j] = scale · ∏_dim (1 + w[dim]·exp(−(X[i,dim]−X[j,dim])²/2l²))` is also discharged here, *without*
Bochner/Schoenberg (absent from Mathlib v4.30.0), via an elementary Hadamard-exponential route that
reuses the same Schur product theorem:

* `posSemidef_of_tendsto` — the PSD cone is **closed under entrywise limits** (the one genuinely new,
  independently-useful lemma: the quadratic form is continuous in the entries, `{≥0}` is closed).
* `posSemidef_map_exp` — the **entrywise exponential** of a PSD matrix is PSD: `exp∘G = Σₖ G^∘k/k!`,
  each Hadamard power `G^∘k` PSD by the Schur product theorem, the partial sums PSD, the limit PSD.
* `posSemidef_gaussianCol` — a single Gaussian matrix `exp(−c·(yᵢ−yⱼ)²)` is PSD (`c ≥ 0`), by writing it
  as `D·(exp∘(2c·yyᵀ))·Dᵀ` — a diagonal congruence of an entrywise-exponential of the rank-one Gram.
* `gaussianKernelFn_posSemidef` — each feature factor `𝟙𝟙ᵀ + w[dim]·Gaussian` is PSD, and the product
  over features is PSD by the Schur product theorem; so `K` is PSD for `scale ≥ 0` and a mask `w ≥ 0`.
  `gaussianKernelFn_symm` / `gaussianKernelSpec_posSemidef` give symmetry and the tensor-level form.

All three CHD non-interpolatory/nonlinear modes (linear, quadratic, Gaussian) are now PSD-discharged.
-/

@[expose] public section

namespace Spec.Factorization

open Matrix
open scoped BigOperators
open Spec.Factorization.Reconstruction

variable {n d : Nat}

/-- Over `ℝ`, `Φᴴ = Φᵀ` (the star is trivial), for any rectangular matrix. -/
private theorem conjTranspose_eq_transpose {m k : Nat} (Φ : Matrix (Fin m) (Fin k) ℝ) :
    (Φᴴ : Matrix (Fin k) (Fin m) ℝ) = Φᵀ := by
  ext a b; simp [Matrix.conjTranspose_apply, Matrix.transpose_apply]

/-- The Gram matrix `Φ·Φᵀ` is positive-semidefinite (real form of
`posSemidef_self_mul_conjTranspose`). -/
private theorem posSemidef_mul_transpose_self {m k : Nat} (Φ : Matrix (Fin m) (Fin k) ℝ) :
    (Φ * Φᵀ).PosSemidef := by
  have h := Matrix.posSemidef_self_mul_conjTranspose Φ
  rwa [conjTranspose_eq_transpose Φ] at h

/-- **The linear-mode kernel is symmetric positive-semidefinite.** For data `X`, selection mask `w`,
and `scale ≥ 0`, `K = 𝟙𝟙ᵀ + scale·Φ·Φᵀ` is PSD — discharging the `PosSemidef` hypothesis of the CHD
solve / `find_gamma` development for the real linear kernel. -/
theorem linearKernelFn_posSemidef (X : Fin n → Fin d → ℝ) (w : Fin d → ℝ) {scale : ℝ}
    (hscale : 0 ≤ scale) : (Matrix.of (Spec.linearKernelFn X w scale)).PosSemidef := by
  -- the masked data as a matrix, and the all-ones column
  set Φ : Matrix (Fin n) (Fin d) ℝ := Matrix.of (Spec.maskColsFn X w) with hΦ
  set Ψ : Matrix (Fin n) (Fin 1) ℝ := Matrix.of (fun _ _ => 1) with hΨ
  -- `K = Ψ·Ψᵀ + scale • (Φ·Φᵀ)`
  have hKeq : Matrix.of (Spec.linearKernelFn X w scale) = Ψ * Ψᵀ + scale • (Φ * Φᵀ) := by
    ext i j
    simp only [Matrix.of_apply, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul,
      Matrix.mul_apply, Matrix.transpose_apply, Spec.linearKernelFn, hΦ, hΨ]
    rw [dotFn_eq_sum, Fin.sum_univ_one]
    simp only [Spec.maskColsFn]
    ring
  rw [hKeq]
  exact (posSemidef_mul_transpose_self Ψ).add ((posSemidef_mul_transpose_self Φ).smul hscale)

/-- The linear-mode kernel is symmetric: `K[i,j] = K[j,i]`. -/
theorem linearKernelFn_symm (X : Fin n → Fin d → ℝ) (w : Fin d → ℝ) {scale : ℝ}
    (hscale : 0 ≤ scale) (i j : Fin n) :
    Spec.linearKernelFn X w scale i j = Spec.linearKernelFn X w scale j i := by
  have h := (linearKernelFn_posSemidef X w hscale).isHermitian
  have e : (Matrix.of (Spec.linearKernelFn X w scale))ᴴ i j
      = (Matrix.of (Spec.linearKernelFn X w scale)) i j := by rw [h]
  simpa [Matrix.conjTranspose_apply, Matrix.of_apply] using e.symm

/-- **Tensor-level: the linear-mode kernel is positive-semidefinite.** The form the verified solve
consumes: `(Matrix.of (toMatFn (linearKernelSpec X w scale))).PosSemidef` for `scale ≥ 0`, so e.g.
`solveRidgeSpec (linearKernelSpec X w scale) γ b` is the exact regularized solve for any `γ > 0`. -/
theorem linearKernelSpec_posSemidef (X : Spec.Tensor ℝ (.dim n (.dim d .scalar)))
    (w : Spec.Tensor ℝ (.dim d .scalar)) {scale : ℝ} (hscale : 0 ≤ scale) :
    (Matrix.of (Spec.toMatFn (Spec.linearKernelSpec X w scale))).PosSemidef := by
  have hround : Spec.toMatFn (Spec.linearKernelSpec X w scale)
      = Spec.linearKernelFn (Spec.toMatFn X) (Spec.toVecFn w) scale := by
    funext i j; rfl
  rw [hround]
  exact linearKernelFn_posSemidef _ _ hscale

/-- **The quadratic-mode kernel is positive-semidefinite.** For data `X`, selection mask `w`, and
`scale ≥ 0`, `alpha ≥ 0`, `K[i,j] = scale·(alpha + ⟨Φ i, Φ j⟩)² + (1 − alpha²·scale)` is PSD. The proof
expands `K = 𝟙𝟙ᵀ + (2·scale·alpha)·Φ·Φᵀ + scale·(Φ·Φᵀ ⊙ Φ·Φᵀ)` and adds three PSD pieces, the last via
the **Schur product theorem** `PosSemidef.hadamard`. -/
theorem quadraticKernelFn_posSemidef (X : Fin n → Fin d → ℝ) (w : Fin d → ℝ) {scale alpha : ℝ}
    (hscale : 0 ≤ scale) (halpha : 0 ≤ alpha) :
    (Matrix.of (Spec.quadraticKernelFn X w scale alpha)).PosSemidef := by
  -- the masked data as a matrix, the all-ones column, and the data Gram `M = Φ·Φᵀ`
  set Φ : Matrix (Fin n) (Fin d) ℝ := Matrix.of (Spec.maskColsFn X w) with hΦ
  set Ψ : Matrix (Fin n) (Fin 1) ℝ := Matrix.of (fun _ _ => 1) with hΨ
  -- `K = Ψ·Ψᵀ + (2·scale·alpha)·(Φ·Φᵀ) + scale·((Φ·Φᵀ) ⊙ (Φ·Φᵀ))`
  have hKeq : Matrix.of (Spec.quadraticKernelFn X w scale alpha)
      = Ψ * Ψᵀ + (2 * scale * alpha) • (Φ * Φᵀ) + scale • ((Φ * Φᵀ) ⊙ (Φ * Φᵀ)) := by
    ext i j
    simp only [Matrix.of_apply, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul,
      Matrix.mul_apply, Matrix.transpose_apply, Matrix.hadamard_apply, Spec.quadraticKernelFn, hΦ, hΨ]
    rw [dotFn_eq_sum, Fin.sum_univ_one]
    simp only [Spec.maskColsFn]
    ring
  rw [hKeq]
  have hM : (Φ * Φᵀ).PosSemidef := posSemidef_mul_transpose_self Φ
  have hc : (0 : ℝ) ≤ 2 * scale * alpha := by positivity
  exact ((posSemidef_mul_transpose_self Ψ).add (hM.smul hc)).add ((hM.hadamard hM).smul hscale)

/-- The quadratic-mode kernel is symmetric: `K[i,j] = K[j,i]`. -/
theorem quadraticKernelFn_symm (X : Fin n → Fin d → ℝ) (w : Fin d → ℝ) {scale alpha : ℝ}
    (hscale : 0 ≤ scale) (halpha : 0 ≤ alpha) (i j : Fin n) :
    Spec.quadraticKernelFn X w scale alpha i j = Spec.quadraticKernelFn X w scale alpha j i := by
  have h := (quadraticKernelFn_posSemidef X w hscale halpha).isHermitian
  have e : (Matrix.of (Spec.quadraticKernelFn X w scale alpha))ᴴ i j
      = (Matrix.of (Spec.quadraticKernelFn X w scale alpha)) i j := by rw [h]
  simpa [Matrix.conjTranspose_apply, Matrix.of_apply] using e.symm

/-- **Tensor-level: the quadratic-mode kernel is positive-semidefinite.** The form the verified solve
consumes, so e.g. `solveRidgeSpec (quadraticKernelSpec X w scale alpha) γ b` is the exact regularized
solve for any `γ > 0` whenever `scale ≥ 0` and `alpha ≥ 0`. -/
theorem quadraticKernelSpec_posSemidef (X : Spec.Tensor ℝ (.dim n (.dim d .scalar)))
    (w : Spec.Tensor ℝ (.dim d .scalar)) {scale alpha : ℝ} (hscale : 0 ≤ scale) (halpha : 0 ≤ alpha) :
    (Matrix.of (Spec.toMatFn (Spec.quadraticKernelSpec X w scale alpha))).PosSemidef := by
  have hround : Spec.toMatFn (Spec.quadraticKernelSpec X w scale alpha)
      = Spec.quadraticKernelFn (Spec.toMatFn X) (Spec.toVecFn w) scale alpha := by
    funext i j; rfl
  rw [hround]
  exact quadraticKernelFn_posSemidef _ _ hscale halpha

/-! ## The Gaussian mode: an elementary Hadamard-exponential PSD proof

CHD's Gaussian (fully-nonlinear) kernel introduces `exp(−Δ²/2l²)`, which has no *finite* algebraic
PSD identity. We discharge it without Bochner/Schoenberg by the classical Schur route: the entrywise
exponential of a PSD matrix is PSD (Hadamard-power series), and the Gaussian is a diagonal congruence
of such an exponential. The PSD-cone-closed-under-limits lemma is the only genuinely new ingredient. -/

open scoped Topology
open Filter

variable {N : Nat}

/-- The all-ones matrix `𝟙𝟙ᵀ` is positive-semidefinite (a rank-one Gram). -/
private theorem posSemidef_ones : (Matrix.of (fun _ _ : Fin N => (1 : ℝ))).PosSemidef := by
  have h := Matrix.posSemidef_vecMulVec_self_star (fun _ : Fin N => (1 : ℝ))
  have he : Matrix.vecMulVec (fun _ : Fin N => (1 : ℝ)) (star (fun _ : Fin N => (1 : ℝ)))
      = Matrix.of (fun _ _ : Fin N => (1 : ℝ)) := by
    ext i j; simp [Matrix.vecMulVec_apply, Pi.star_apply]
  rwa [he] at h

/-- **The PSD cone is closed under entrywise limits.** If real positive-semidefinite matrices `A k`
converge entrywise to `B`, then `B` is positive-semidefinite. The quadratic form `xᵀ·M·x` is continuous
in `M`'s entries, and `{y | 0 ≤ y}` is closed. -/
private theorem posSemidef_of_tendsto {A : ℕ → Matrix (Fin N) (Fin N) ℝ}
    {B : Matrix (Fin N) (Fin N) ℝ} (hA : ∀ k, (A k).PosSemidef)
    (hlim : Tendsto A atTop (𝓝 B)) : B.PosSemidef := by
  have hentry : ∀ i j, Tendsto (fun k => A k i j) atTop (𝓝 (B i j)) :=
    fun i j => (hlim.apply_nhds i).apply_nhds j
  -- entry symmetry of any real Hermitian matrix
  have hsymm_entry : ∀ (M : Matrix (Fin N) (Fin N) ℝ), M.IsHermitian → ∀ i j, M i j = M j i := by
    intro M hM i j
    have e : Mᴴ i j = M i j := congrFun (congrFun hM i) j
    rw [Matrix.conjTranspose_apply, star_trivial] at e
    exact e.symm
  -- B is Hermitian (symmetric over ℝ)
  have hBsymm : B.IsHermitian := by
    ext i j
    rw [Matrix.conjTranspose_apply, star_trivial]
    refine tendsto_nhds_unique (hentry j i) ?_
    have hfun : (fun k => A k j i) = (fun k => A k i j) :=
      funext fun k => (hsymm_entry (A k) (hA k).isHermitian j i)
    rw [hfun]; exact hentry i j
  refine Matrix.PosSemidef.of_dotProduct_mulVec_nonneg hBsymm (fun x => ?_)
  have hquad : ∀ (M : Matrix (Fin N) (Fin N) ℝ),
      star x ⬝ᵥ (M *ᵥ x) = ∑ i, ∑ j, star (x i) * (M i j * x j) := by
    intro M
    simp only [dotProduct, Matrix.mulVec, Pi.star_apply, Finset.mul_sum]
  have key : Tendsto (fun k => star x ⬝ᵥ (A k *ᵥ x)) atTop (𝓝 (star x ⬝ᵥ (B *ᵥ x))) := by
    simp only [hquad]
    refine tendsto_finsetSum _ (fun i _ => ?_)
    refine tendsto_finsetSum _ (fun j _ => ?_)
    exact tendsto_const_nhds.mul ((hentry i j).mul tendsto_const_nhds)
  exact ge_of_tendsto' key (fun k => (hA k).dotProduct_mulVec_nonneg x)

/-- The `k`-fold Hadamard (entrywise) power of `G`, with `G^∘0 = 𝟙𝟙ᵀ` (the all-ones matrix). -/
private def hadamardPow (G : Matrix (Fin N) (Fin N) ℝ) : ℕ → Matrix (Fin N) (Fin N) ℝ
  | 0 => Matrix.of (fun _ _ => 1)
  | (k + 1) => G ⊙ hadamardPow G k

private theorem hadamardPow_apply (G : Matrix (Fin N) (Fin N) ℝ) (k : ℕ) (i j : Fin N) :
    hadamardPow G k i j = (G i j) ^ k := by
  induction k with
  | zero => simp [hadamardPow]
  | succ k ih =>
    rw [hadamardPow, Matrix.hadamard_apply, ih, pow_succ]; ring

private theorem posSemidef_hadamardPow {G : Matrix (Fin N) (Fin N) ℝ} (hG : G.PosSemidef) (k : ℕ) :
    (hadamardPow G k).PosSemidef := by
  induction k with
  | zero => exact posSemidef_ones
  | succ k ih => exact hG.hadamard ih

/-- **The entrywise exponential of a PSD matrix is PSD.** `exp∘G = Σₖ G^∘k/k!`: each Hadamard power is
PSD by the Schur product theorem, the partial sums are PSD, and the PSD cone is closed under limits. -/
private theorem posSemidef_map_exp {G : Matrix (Fin N) (Fin N) ℝ} (hG : G.PosSemidef) :
    (G.map Real.exp).PosSemidef := by
  set S : ℕ → Matrix (Fin N) (Fin N) ℝ :=
    fun n => ∑ k ∈ Finset.range n, ((k.factorial : ℝ)⁻¹) • hadamardPow G k with hS
  have hSpsd : ∀ n, (S n).PosSemidef := by
    intro n
    refine Matrix.posSemidef_sum _ (fun k _ => ?_)
    exact (posSemidef_hadamardPow hG k).smul (by positivity)
  have hlim : Tendsto S atTop (𝓝 (G.map Real.exp)) := by
    refine tendsto_pi_nhds.mpr (fun i => ?_)
    refine tendsto_pi_nhds.mpr (fun j => ?_)
    have hentry : (fun n => S n i j)
        = (fun n => ∑ k ∈ Finset.range n, ((k.factorial : ℝ)⁻¹) * (G i j) ^ k) := by
      funext n
      simp only [hS, Matrix.sum_apply, Matrix.smul_apply, smul_eq_mul, hadamardPow_apply]
    rw [hentry]
    have hsum : HasSum (fun k => ((k.factorial : ℝ)⁻¹) * (G i j) ^ k) (Real.exp (G i j)) := by
      have h := NormedSpace.exp_series_hasSum_exp' (𝕂 := ℝ) (G i j)
      simp only [smul_eq_mul] at h
      rwa [← Real.exp_eq_exp_ℝ] at h
    have hmap : (G.map Real.exp) i j = Real.exp (G i j) := by simp [Matrix.map_apply]
    rw [hmap]
    exact hsum.tendsto_sum_nat
  exact posSemidef_of_tendsto hSpsd hlim

/-- **A single Gaussian matrix is positive-semidefinite.** For `c ≥ 0`, the matrix
`exp(−c·(yᵢ−yⱼ)²)` is PSD: writing the exponent as `−c·yᵢ² + 2c·yᵢyⱼ − c·yⱼ²`, it is the diagonal
congruence `D·(exp∘(2c·yyᵀ))·Dᵀ` of the entrywise exponential of the (PSD) rank-one Gram `yyᵀ`. -/
private theorem posSemidef_gaussianCol (y : Fin N → ℝ) {c : ℝ} (hc : 0 ≤ c) :
    (Matrix.of (fun i j => Real.exp (-(c * ((y i - y j) * (y i - y j)))))).PosSemidef := by
  set G : Matrix (Fin N) (Fin N) ℝ := (2 * c) • Matrix.vecMulVec y y with hG
  have hGpsd : G.PosSemidef := by
    have hv : (Matrix.vecMulVec y (star y)).PosSemidef := Matrix.posSemidef_vecMulVec_self_star y
    have hstar : Matrix.vecMulVec y (star y) = Matrix.vecMulVec y y := by
      ext i j; simp [Matrix.vecMulVec_apply]
    rw [hstar] at hv
    rw [hG]; exact hv.smul (mul_nonneg (by norm_num) hc)
  have hMpsd : (G.map Real.exp).PosSemidef := posSemidef_map_exp hGpsd
  set D : Matrix (Fin N) (Fin N) ℝ := Matrix.diagonal (fun i => Real.exp (-(c * (y i * y i)))) with hD
  have hcong : (D * (G.map Real.exp) * Dᴴ).PosSemidef := hMpsd.mul_mul_conjTranspose_same D
  have hDH : (Dᴴ : Matrix (Fin N) (Fin N) ℝ) = D := by
    rw [hD]; simp
  rw [hDH] at hcong
  have heq : D * (G.map Real.exp) * D
      = Matrix.of (fun i j => Real.exp (-(c * ((y i - y j) * (y i - y j))))) := by
    ext i j
    rw [hD, Matrix.mul_diagonal, Matrix.diagonal_mul]
    simp only [Matrix.of_apply, hG, Matrix.map_apply, Matrix.smul_apply, Matrix.vecMulVec_apply,
      smul_eq_mul]
    rw [← Real.exp_add, ← Real.exp_add]
    congr 1; ring
  rwa [heq] at hcong

/-- Folding scalar multiplication over a list is the product of the mapped list. -/
private theorem foldl_mul_eq_prod {ι : Type} (l : List ι) (g : ι → ℝ) (a : ℝ) :
    l.foldl (fun acc x => acc * g x) a = a * (l.map g).prod := by
  induction l generalizing a with
  | nil => simp
  | cons x xs ih => simp only [List.foldl_cons, List.map_cons, List.prod_cons, ih]; ring

/-- A Hadamard product (over a finset) of positive-semidefinite matrices is positive-semidefinite —
the Schur product theorem, iterated. -/
private theorem posSemidef_prod_hadamard {ι : Type} [DecidableEq ι]
    (F : ι → Matrix (Fin N) (Fin N) ℝ) (s : Finset ι) (hF : ∀ k ∈ s, (F k).PosSemidef) :
    (Matrix.of (fun i j => ∏ k ∈ s, (F k) i j)).PosSemidef := by
  induction s using Finset.induction with
  | empty => simpa only [Finset.prod_empty] using (posSemidef_ones (N := N))
  | @insert a s ha ih =>
    rw [show (Matrix.of (fun i j => ∏ k ∈ insert a s, (F k) i j))
        = (F a) ⊙ Matrix.of (fun i j => ∏ k ∈ s, (F k) i j) from by
          ext i j; simp only [Matrix.hadamard_apply, Matrix.of_apply, Finset.prod_insert ha]]
    exact (hF a (Finset.mem_insert_self a s)).hadamard
      (ih (fun k hk => hF k (Finset.mem_insert_of_mem hk)))

variable {n d : Nat}

/-- **The Gaussian-mode product kernel is positive-semidefinite.** For data `X`, a nonnegative
selection mask `w ≥ 0`, and `scale ≥ 0`,
`K[i,j] = scale · ∏_dim (1 + w[dim]·exp(−(X[i,dim]−X[j,dim])²/2l²))` is PSD. Each feature factor
`𝟙𝟙ᵀ + w[dim]·Gaussian` is PSD (`posSemidef_ones` + `posSemidef_gaussianCol`), and the product over
features is PSD by the **Schur product theorem** (`posSemidef_prod_hadamard`). -/
theorem gaussianKernelFn_posSemidef (X : Fin n → Fin d → ℝ) (w : Fin d → ℝ) {scale l : ℝ}
    (hscale : 0 ≤ scale) (hw : ∀ k, 0 ≤ w k) :
    (Matrix.of (Spec.gaussianKernelFn X w scale l)).PosSemidef := by
  -- the per-feature factor matrices
  set F : Fin d → Matrix (Fin n) (Fin n) ℝ :=
    fun k => Matrix.of (fun i j => 1 + w k *
      Real.exp (-((X i k - X j k) * (X i k - X j k)) / ((1 + 1) * l * l))) with hF
  have hFpsd : ∀ k, (F k).PosSemidef := by
    intro k
    -- the per-feature Gaussian is PSD via `posSemidef_gaussianCol`
    have hGauss : (Matrix.of (fun i j =>
        Real.exp (-((X i k - X j k) * (X i k - X j k)) / ((1 + 1) * l * l)))).PosSemidef := by
      have h := posSemidef_gaussianCol (fun i => X i k)
        (c := ((1 + 1) * l * l)⁻¹) (inv_nonneg.mpr (by nlinarith [mul_self_nonneg l]))
      have he : (Matrix.of (fun i j =>
          Real.exp (-(((1 + 1) * l * l)⁻¹ * ((X i k - X j k) * (X i k - X j k))))))
          = Matrix.of (fun i j =>
            Real.exp (-((X i k - X j k) * (X i k - X j k)) / ((1 + 1) * l * l))) := by
        ext i j
        show Real.exp _ = Real.exp _
        congr 1; ring
      rwa [he] at h
    -- F k = 𝟙𝟙ᵀ + w k • Gaussian
    have hsplit : F k = Matrix.of (fun _ _ : Fin n => (1 : ℝ))
        + (w k) • Matrix.of (fun i j =>
            Real.exp (-((X i k - X j k) * (X i k - X j k)) / ((1 + 1) * l * l))) := by
      rw [hF]; ext i j
      simp only [Matrix.add_apply, Matrix.smul_apply, Matrix.of_apply, smul_eq_mul]
    rw [hsplit]
    exact posSemidef_ones.add (hGauss.smul (hw k))
  -- the product matrix is PSD
  have hPpsd : (Matrix.of (fun i j => ∏ k, (F k) i j)).PosSemidef :=
    posSemidef_prod_hadamard F Finset.univ (fun k _ => hFpsd k)
  -- the kernel is `scale • (product matrix)`
  have hKeq : Matrix.of (Spec.gaussianKernelFn X w scale l)
      = scale • Matrix.of (fun i j => ∏ k, (F k) i j) := by
    ext i j
    rw [Matrix.smul_apply, Matrix.of_apply, Matrix.of_apply, smul_eq_mul, Spec.gaussianKernelFn,
      foldl_mul_eq_prod, one_mul, ← List.ofFn_eq_map, List.prod_ofFn]
    rfl
  rw [hKeq]
  exact hPpsd.smul hscale

/-- The Gaussian-mode product kernel is symmetric: `K[i,j] = K[j,i]`. -/
theorem gaussianKernelFn_symm (X : Fin n → Fin d → ℝ) (w : Fin d → ℝ) {scale l : ℝ}
    (hscale : 0 ≤ scale) (hw : ∀ k, 0 ≤ w k) (i j : Fin n) :
    Spec.gaussianKernelFn X w scale l i j = Spec.gaussianKernelFn X w scale l j i := by
  have h := (gaussianKernelFn_posSemidef (scale := scale) (l := l) X w hscale hw).isHermitian
  have e : (Matrix.of (Spec.gaussianKernelFn X w scale l))ᴴ i j
      = (Matrix.of (Spec.gaussianKernelFn X w scale l)) i j := by rw [h]
  simpa [Matrix.conjTranspose_apply, Matrix.of_apply] using e.symm

/-- **Tensor-level: the Gaussian-mode product kernel is positive-semidefinite.** The form the verified
solve consumes, so e.g. `solveRidgeSpec (gaussianKernelSpec X w scale l) γ b` is the exact regularized
solve for any `γ > 0` whenever `scale ≥ 0` and the mask `w ≥ 0`. -/
theorem gaussianKernelSpec_posSemidef (X : Spec.Tensor ℝ (.dim n (.dim d .scalar)))
    (w : Spec.Tensor ℝ (.dim d .scalar)) {scale l : ℝ} (hscale : 0 ≤ scale)
    (hw : ∀ k, 0 ≤ Spec.toVecFn w k) :
    (Matrix.of (Spec.toMatFn (Spec.gaussianKernelSpec X w scale l))).PosSemidef := by
  have hround : Spec.toMatFn (Spec.gaussianKernelSpec X w scale l)
      = Spec.gaussianKernelFn (Spec.toMatFn X) (Spec.toVecFn w) scale l := by
    funext i j; rfl
  rw [hround]
  exact gaussianKernelFn_posSemidef _ _ hscale hw

end Spec.Factorization
