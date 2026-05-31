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

Both **positive** checks (a valid factorization reconstructs to `err ≈ 0`) and **negative controls**
(the same metric reports a large error / `NaN` when a hypothesis is violated) are included, so a
reviewer can see the checks are not vacuous.
-/

@[expose] public section
