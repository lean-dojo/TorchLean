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
`NN.Spec.Core.Tensor.Factorizations`:

- `Cholesky` — `A = L · Lᵀ`
- `QR`       — `A = Q · R`, `Qᵀ·Q = I`
- `SymEig`   — full symmetric eigendecomposition `A = V · diag(λ) · Vᵀ`
- `SVD`      — `A = U · diag(σ) · Vᵀ`

Each example reconstructs the original matrix and asserts (via `#guard`) that the maximum
reconstruction error is below `tol`, so the build fails if a factorization is incorrect.
-/

@[expose] public section
