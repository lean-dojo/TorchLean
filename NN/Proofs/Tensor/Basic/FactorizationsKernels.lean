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

Gaussian mode (Bochner / Schoenberg positive-definiteness, not in Mathlib v4.30.0) is the natural
remaining follow-on.
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

end Spec.Factorization
