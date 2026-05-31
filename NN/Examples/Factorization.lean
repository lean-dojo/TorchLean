/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Factorization.Common
public import NN.Examples.Factorization.Cholesky
public import NN.Examples.Factorization.QR
public import NN.Examples.Factorization.SymEig
public import NN.Examples.Factorization.SVD
public import NN.Examples.Factorization.JacobiDecrease
public import NN.Examples.Factorization.JacobiRate
public import NN.Examples.Factorization.RidgeSolve
public import NN.Examples.Factorization.Variational

/-!
# Matrix factorization examples

Executable sanity checks for the spec-layer matrix factorizations in
`NN.Spec.Core.Tensor.Factorizations`, designed to corroborate the formal correctness theorems in
`NN.Proofs.Tensor.Basic.{Factorizations, FactorizationsReconstruction, FactorizationsOrthonormal,
FactorizationsJacobi}`. Each check runs through compiled `#eval` assertions, so the build fails if a
factorization misbehaves.

- `Cholesky` — `A = L · Lᵀ`; **negative control**: an indefinite `A` correctly fails (no SPD factor).
- `QR`       — `A = Q · R`, `Qᵀ·Q = I`; **negative control**: a rank-deficient `A` still reconstructs
  but `Qᵀ Q ≠ I`, separating the two guarantees and showing full column rank is needed.
- `SymEig`   — `A = V · diag(λ) · Vᵀ`; orthogonality `Vᵀ V = I` is exact at *any* sweep count (witness
  of the a-priori `jacobi_orthogonal`), diagonalization is asymptotic, and the **exact residual
  certificate** `‖A − V·diag(λ)·Vᵀ‖² = ‖offDiag(VᵀAV)‖²` (`symEigJacobi_frobenius_residual`) is
  verified numerically.
- `SVD`      — `A = U · diag(σ) · Vᵀ`, `Vᵀ V = I`; **negative control**: a permuted `σ` fails to
  reconstruct.
- `JacobiDecrease` — the per-rotation progress identity `‖offDiag(Jᵀ A J)‖² = ‖offDiag A‖² − 2·A[p,q]²`
  (`jacobi_off_decrease`) and Frobenius-mass invariance; **negative controls**: a wrong-angle rotation
  misses the decrease, a non-orthogonal one breaks mass invariance.
- `JacobiRate` — the *aggregate* linear-contraction rate of the classical largest-pivot strategy:
  `‖offDiag(Jᵀ A J)‖² ≤ (1 − 2/(n²−n))·‖offDiag A‖²` (`jacobi_off_decrease_classical`); **negative
  control**: annihilating a non-largest (tiny) pivot misses the guaranteed factor, so the rate is
  specific to the largest-pivot choice.
- `RidgeSolve` — the kernel-ridge (Tikhonov) linear solve `(K + γ·I)·x = b` via Cholesky +
  forward/back substitution (`solveRidgeFn_mulVec_of_posSemidef`, the verified core of CHD
  `solve_variationnal`, now *unconditional* for PSD `K` and `γ > 0`): for a rank-deficient Gram kernel
  `K = G·Gᵀ` and `γ > 0`, `solveRidgeFn` reconstructs `b` to machine precision; **negative control**:
  with `γ = 0` the singular `K` has a zero Cholesky pivot and the solve diverges (`NaN`), so
  regularization is necessary. Also exhibits the **keystone** `choleskyFn_diag_pos_of_posDef`: the SPD
  `K + γ·I` has all-positive Cholesky pivots, while the singular `K` has a zero pivot (PosDef needed);
  and the two **capstones** — `cholesky_posDef` (the SPD Cholesky reconstructs `L·Lᵀ = K + γ·I`
  exactly, while an *indefinite* matrix fails with a `NaN` pivot) and `solveRidgeFn_eq_inv_mulVec` (the
  solve *is* the regularized inverse: its columns assemble into `(K + γ·I)⁻¹` with
  `(K + γ·I)·(K + γ·I)⁻¹ = I`).
- `Variational` — the *eigendecomposition* form of CHD `perform_regression_and_find_gamma`
  (`interpolatory.py`): from `eigh(K)`, the variational solve `yb = -(K + γ·I)⁻¹·ga`, the agreement of
  the eig and Cholesky routes (`variationalSolveFn_eq_neg_solveRidgeFn`), the
  `noise`/`find_gamma`-loss/`Z_test` statistic as a spectral ratio bounded in `[0,1]`
  (`varNoiseFn_nonneg`, `varNoiseFn_le_one`), and `Z_test` spectral invariance
  (`varNoiseFn_projFn_mulVec`); **negative controls**: wrong eigenvectors break the solve, and `γ < 0`
  pushes the noise outside `[0,1]`.

Both **positive** checks (a valid factorization reconstructs to `err ≈ 0`) and **negative controls**
(the same metric reports a large error / `NaN` when a hypothesis is violated) are included, so a
reviewer can see the checks are not vacuous.
-/

@[expose] public section
