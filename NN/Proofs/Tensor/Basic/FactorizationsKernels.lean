/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Basic.Factorizations
public import NN.Proofs.Tensor.Basic.FactorizationsOrthonormal
public import Mathlib.Data.Real.StarOrdered

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

Quadratic mode (`PosSemidef.hadamard`, the Schur product theorem) and Gaussian mode (Bochner /
Schoenberg, not in Mathlib v4.30.0) are the natural follow-ons.
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

end Spec.Factorization
