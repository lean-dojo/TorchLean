/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Basic.Factorizations
public import NN.Proofs.Tensor.Basic.FactorizationsSolve

/-!
# CHD `solve_variationnal`, `find_gamma`, and `Z_test` (eigendecomposition form)

[`Factorizations`](./Factorizations.lean) proved the *predicate-level* spectral facts CHD consumes —
the regularized inverse `(K + γ·I)⁻¹ = V·diag(1/(λ+γ))·Vᵀ` (`IsSymEig.add_smul_inv`), the trace/det
sums, and the SVD ⟹ Gram-eigendecomposition bridge. This file closes the gap up to the three concrete
CHD routines built on those facts (`interpolatory.py`): `solve_variationnal`, `find_gamma`, `Z_test`.

All three are computed from `eigh(K)` and share one arithmetic core: the projected data
`Pga = Vᵀ·ga` and the shrinkage coefficients `rᵢ = γ/(λᵢ + γ)`. The executable definitions
(`Spec.variationalSolveFn`, `Spec.varNoiseFn`, …) mirror `interpolatory.py` verbatim. The theorems
here identify what they compute:

* **`variationalSolveFn_eq_neg_inv_mulVec`** — the eigendecomposition-form variational solution
  `yb = -V·(Pga/(λ+γ))` *is* the regularized-inverse solve `-(K + γ·I)⁻¹·ga` (from `add_smul_inv`).
* **`variationalSolveFn_eq_neg_solveRidgeFn`** — hence the eig route and the *Cholesky* route
  (`solveRidgeFn`, verified in `FactorizationsSolve`) compute the **same** `solve_variationnal`, up to
  CHD's sign convention. Two independent implementations, one closed form.
* **`varNoiseFn_eq_ratio`** — the `noise` level (= the `find_gamma` loss = the `Z_test` per-sample
  statistic) is the spectral ratio `Σ (Pgaᵢ·rᵢ)² / Σ Pgaᵢ²·rᵢ`.
* **`varNoiseFn_nonneg` / `varNoiseFn_le_one`** — for a PSD spectrum (`λᵢ ≥ 0`) and `γ > 0` the noise
  lies in `[0, 1]`, because each shrinkage coefficient does (`ridgeCoeffFn_pos`, `ridgeCoeffFn_le_one`).
  This is the meaningful invariant: the CHD noise level is a genuine fraction.
* **`projFn_mulVec_self` / `varNoiseFn_projFn_mulVec`** — feeding data `ga = V·z` makes `V` drop out of
  the statistic, so it depends on the kernel only through its spectrum. This is the deterministic
  content of "the `Z_test` null distribution depends only on the eigenvalues" (the *distributional*
  step — Gaussian sampling and percentiles — is out of scope here, exercised numerically instead).

`IsSymEig.eigenvalues_nonneg` supplies the `λᵢ ≥ 0` hypothesis from a positive-semidefinite kernel.

Scope honesty: everything here is exact over `ℝ`, proved from the *specification* `IsSymEig` (so it
holds for whatever eigendecomposition the solver returns), not from the asymptotic Jacobi convergence.
-/

@[expose] public section

namespace Spec.Factorization

open Matrix
open scoped BigOperators
open Spec.Factorization.Reconstruction

variable {n : Nat}

/-! ## Bridge: the projection is `Vᵀ·ga` -/

/-- `Spec.projFn V ga = Vᵀ *ᵥ ga`: the executable projection is multiplication by `Vᵀ`. -/
theorem projFn_eq_mulVec (V : Matrix (Fin n) (Fin n) ℝ) (ga : Fin n → ℝ) :
    Spec.projFn V ga = Vᵀ *ᵥ ga := by
  funext i
  rw [Spec.projFn, dotFn_eq_sum]
  show ∑ k, V k i * ga k = ∑ k, Vᵀ i k * ga k
  exact Finset.sum_congr rfl (fun k _ => by rw [Matrix.transpose_apply])

/-- Feeding `ga = V·z` recovers `z`: `projFn V (V *ᵥ z) = z` when `Vᵀ·V = 1`. The change of variables
that makes the `Z_test` statistic depend on the kernel only through its spectrum. -/
theorem projFn_mulVec_self {V : Matrix (Fin n) (Fin n) ℝ} (hV : Vᵀ * V = 1) (z : Fin n → ℝ) :
    Spec.projFn V (V *ᵥ z) = z := by
  rw [projFn_eq_mulVec, Matrix.mulVec_mulVec, hV, Matrix.one_mulVec]

/-! ## The variational solution is the regularized inverse -/

/-- **The eigendecomposition-form `solve_variationnal` is the regularized-inverse solve.** Given an
eigendecomposition `IsSymEig A Λ V` and `γ` avoiding every `-λᵢ`, the CHD solution
`yb = -V·(Pga/(λ+γ))` equals `-(A + γ·I)⁻¹·ga`. Proved directly from `add_smul_inv`. -/
theorem variationalSolveFn_eq_neg_inv_mulVec
    {A V : Matrix (Fin n) (Fin n) ℝ} {Λ : Fin n → ℝ}
    (h : IsSymEig A Λ V) (γ : ℝ) (hγ : ∀ i, Λ i + γ ≠ 0) (ga : Fin n → ℝ) :
    Spec.variationalSolveFn Λ V γ ga
      = -((A + γ • (1 : Matrix (Fin n) (Fin n) ℝ))⁻¹ *ᵥ ga) := by
  rw [h.add_smul_inv γ hγ]
  funext i
  simp only [Spec.variationalSolveFn, Pi.neg_apply]
  congr 1
  rw [dotFn_eq_sum]
  rw [show (V * Matrix.diagonal (fun j => (Λ j + γ)⁻¹) * Vᵀ) *ᵥ ga
        = V *ᵥ (fun j => (Λ j + γ)⁻¹ * Spec.projFn V ga j) from by
        rw [← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec]
        congr 1
        funext j
        rw [Matrix.mulVec_diagonal, ← projFn_eq_mulVec]]
  show ∑ j, V i j * (Spec.projFn V ga j / (Λ j + γ))
      = ∑ j, V i j * ((Λ j + γ)⁻¹ * Spec.projFn V ga j)
  exact Finset.sum_congr rfl (fun j _ => by rw [div_eq_mul_inv]; ring)

/-! ## PSD kernels have nonnegative eigenvalues -/

/-- For a positive-semidefinite `A`, every eigenvalue in *any* `IsSymEig` decomposition is `≥ 0`. The
`i`-th eigenvalue is the quadratic form `vᵢᵀ A vᵢ` of the `i`-th eigenvector, which PSD makes
nonnegative. -/
theorem IsSymEig.eigenvalues_nonneg {A V : Matrix (Fin n) (Fin n) ℝ} {Λ : Fin n → ℝ}
    (h : IsSymEig A Λ V) (hA : A.PosSemidef) (i : Fin n) : 0 ≤ Λ i := by
  obtain ⟨hV, hAeq⟩ := h
  -- `Vᵀ A V = diag Λ` (orthogonal conjugation collapses to the diagonal)
  have hconj : Vᵀ * A * V = Matrix.diagonal Λ := by
    rw [hAeq,
      show Vᵀ * (V * Matrix.diagonal Λ * Vᵀ) * V
          = (Vᵀ * V) * Matrix.diagonal Λ * (Vᵀ * V) by simp [Matrix.mul_assoc],
      hV, Matrix.one_mul, Matrix.mul_one]
  -- over ℝ, `Vᴴ = Vᵀ`, so PSD-congruence `Vᵀ A V` is PSD, i.e. `diag Λ` is PSD
  have hVH : (Vᴴ : Matrix (Fin n) (Fin n) ℝ) = Vᵀ := by
    ext a b; simp [Matrix.conjTranspose_apply, Matrix.transpose_apply]
  have hps : (Matrix.diagonal Λ).PosSemidef := by
    have hcong := hA.conjTranspose_mul_mul_same V
    rwa [hVH, hconj] at hcong
  have hdiag := hps.diag_nonneg (i := i)
  rwa [Matrix.diagonal_apply_eq] at hdiag

/-- **The eig route and the Cholesky route agree.** For a PSD kernel `K` and `γ > 0`, the
eigendecomposition-form `variationalSolveFn` equals `-solveRidgeFn` (the verified Cholesky solve of
`FactorizationsSolve`): two independent implementations of CHD `solve_variationnal`, both equal to
`-(K + γ·I)⁻¹·ga`. -/
theorem variationalSolveFn_eq_neg_solveRidgeFn
    {K : Fin n → Fin n → ℝ} {Λ : Fin n → ℝ} {V : Matrix (Fin n) (Fin n) ℝ}
    (h : IsSymEig (Matrix.of K) Λ V) (hK : (Matrix.of K).PosSemidef) {γ : ℝ} (hγ : 0 < γ)
    (ga : Fin n → ℝ) :
    Spec.variationalSolveFn Λ V γ ga = -(Spec.solveRidgeFn K γ ga) := by
  have hΛ : ∀ i, 0 ≤ Λ i := h.eigenvalues_nonneg hK
  have hγne : ∀ i, Λ i + γ ≠ 0 := fun i => (by have := hΛ i; linarith : (0:ℝ) < Λ i + γ).ne'
  rw [variationalSolveFn_eq_neg_inv_mulVec h γ hγne ga,
    show Spec.solveRidgeFn K γ ga = (Matrix.of (Spec.addScaledIdFn K γ))⁻¹ *ᵥ ga from
      solveRidgeFn_eq_inv_mulVec K γ ga hK hγ,
    of_addScaledIdFn]

/-! ## The noise / `find_gamma` loss / `Z_test` statistic -/

/-- **The noise functional as a spectral ratio.** `varNoiseFn` (the CHD `noise`, the `find_gamma` loss,
and the `Z_test` per-sample statistic) is `Σᵢ (Pgaᵢ·rᵢ)² / Σᵢ Pgaᵢ²·rᵢ`, with `rᵢ = γ/(λᵢ + γ)`. -/
theorem varNoiseFn_eq_ratio (Λ : Fin n → ℝ) (γ : ℝ) (Pga : Fin n → ℝ) :
    Spec.varNoiseFn Λ γ Pga
      = (∑ i, (Pga i * (γ / (Λ i + γ))) ^ 2) / (∑ i, Pga i ^ 2 * (γ / (Λ i + γ))) := by
  simp only [Spec.varNoiseFn, Spec.ridgeCoeffFn]
  rw [dotFn_eq_sum, dotFn_eq_sum]
  congr 1
  · exact Finset.sum_congr rfl (fun i _ => by ring)
  · exact Finset.sum_congr rfl (fun i _ => by ring)

/-- A shrinkage coefficient is strictly positive for a PSD spectrum and `γ > 0`. -/
theorem ridgeCoeffFn_pos {Λ : Fin n → ℝ} (hΛ : ∀ i, 0 ≤ Λ i) {γ : ℝ} (hγ : 0 < γ) (i : Fin n) :
    0 < Spec.ridgeCoeffFn Λ γ i := by
  rw [Spec.ridgeCoeffFn]; exact div_pos hγ (by have := hΛ i; linarith)

/-- A shrinkage coefficient is at most `1` for a PSD spectrum and `γ > 0`. -/
theorem ridgeCoeffFn_le_one {Λ : Fin n → ℝ} (hΛ : ∀ i, 0 ≤ Λ i) {γ : ℝ} (hγ : 0 < γ) (i : Fin n) :
    Spec.ridgeCoeffFn Λ γ i ≤ 1 := by
  rw [Spec.ridgeCoeffFn, div_le_one (by have := hΛ i; linarith)]
  have := hΛ i; linarith

/-- **The noise level is nonnegative** for a PSD spectrum and `γ > 0`. -/
theorem varNoiseFn_nonneg {Λ : Fin n → ℝ} (hΛ : ∀ i, 0 ≤ Λ i) {γ : ℝ} (hγ : 0 < γ)
    (Pga : Fin n → ℝ) : 0 ≤ Spec.varNoiseFn Λ γ Pga := by
  rw [varNoiseFn_eq_ratio]
  apply div_nonneg
  · exact Finset.sum_nonneg (fun i _ => sq_nonneg _)
  · refine Finset.sum_nonneg (fun i _ => ?_)
    have hd : (0:ℝ) < Λ i + γ := by have := hΛ i; linarith
    exact mul_nonneg (sq_nonneg _) (div_nonneg hγ.le hd.le)

/-- **The noise level is at most `1`** for a PSD spectrum and `γ > 0`: each squared shrinkage
coefficient `rᵢ²` is dominated by `rᵢ` (since `0 ≤ rᵢ ≤ 1`), so the numerator is at most the
denominator. The CHD `noise` is therefore a genuine fraction in `[0, 1]`. -/
theorem varNoiseFn_le_one {Λ : Fin n → ℝ} (hΛ : ∀ i, 0 ≤ Λ i) {γ : ℝ} (hγ : 0 < γ)
    (Pga : Fin n → ℝ) : Spec.varNoiseFn Λ γ Pga ≤ 1 := by
  rw [varNoiseFn_eq_ratio]
  have hdenom_nonneg : 0 ≤ ∑ i, Pga i ^ 2 * (γ / (Λ i + γ)) :=
    Finset.sum_nonneg (fun i _ => by
      have hd : (0:ℝ) < Λ i + γ := by have := hΛ i; linarith
      exact mul_nonneg (sq_nonneg _) (div_nonneg hγ.le hd.le))
  have hle : (∑ i, (Pga i * (γ / (Λ i + γ))) ^ 2) ≤ ∑ i, Pga i ^ 2 * (γ / (Λ i + γ)) := by
    refine Finset.sum_le_sum (fun i _ => ?_)
    have hd : (0:ℝ) < Λ i + γ := by have := hΛ i; linarith
    have hr0 : 0 ≤ γ / (Λ i + γ) := div_nonneg hγ.le hd.le
    have hr1 : γ / (Λ i + γ) ≤ 1 := by rw [div_le_one hd]; have := hΛ i; linarith
    rw [show (Pga i * (γ / (Λ i + γ))) ^ 2 = Pga i ^ 2 * (γ / (Λ i + γ)) ^ 2 by ring]
    apply mul_le_mul_of_nonneg_left _ (sq_nonneg _)
    nlinarith [mul_nonneg hr0 (sub_nonneg.mpr hr1)]
  rcases hdenom_nonneg.lt_or_eq with hpos | h0
  · rw [div_le_one hpos]; exact hle
  · rw [← h0, div_zero]; exact zero_le_one

/-- **`Z_test` spectral invariance.** Replacing the data by `ga = V·z` removes `V` from the statistic:
`varNoiseFn Λ γ (projFn V (V·z)) = varNoiseFn Λ γ z`. So the functional `Z_test` samples depends on the
kernel only through its eigenvalues. -/
theorem varNoiseFn_projFn_mulVec {V : Matrix (Fin n) (Fin n) ℝ} (hV : Vᵀ * V = 1)
    (Λ : Fin n → ℝ) (γ : ℝ) (z : Fin n → ℝ) :
    Spec.varNoiseFn Λ γ (Spec.projFn V (V *ᵥ z)) = Spec.varNoiseFn Λ γ z := by
  rw [projFn_mulVec_self hV]

end Spec.Factorization
