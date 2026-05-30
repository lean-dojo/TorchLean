/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Factorization.Common
meta import NN.Examples.Factorization.Common

/-!
# Example: symmetric eigendecomposition (cyclic Jacobi)

`symEigJacobiSpec A sweeps` returns `(eigenvalues, V)` for a symmetric `A`, where the columns of
`V` are the (orthonormal) eigenvectors. Unlike the power-iteration `eigendecompSpec`, this recovers
**all** eigenpairs. We check the spectral reconstruction `A = V · diag(λ) · Vᵀ` and orthogonality
`Vᵀ · V = I`.
-/

@[expose] public section


namespace NN.Examples.Factorization.SymEig

/-- A symmetric test matrix (eigenvalues ≈ {1.3249, 2.4608, 5.2143}). -/
def A : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) :=
  mkMat [[2, 1, 1],
         [1, 3, 1],
         [1, 1, 4]]

/-- Eigenvalues (diagonal after Jacobi sweeps) and eigenvector matrix `V`. -/
def eig : Spec.Tensor Float (.dim 3 .scalar) × Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) :=
  Spec.symEigJacobiSpec A 8

/-- Eigenvalues. -/
def evals : Spec.Tensor Float (.dim 3 .scalar) := eig.1
/-- Eigenvector matrix (columns are eigenvectors). -/
def V : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) := eig.2

/-- Spectral reconstruction error `‖A - V·diag(λ)·Vᵀ‖_max`. -/
def reconErr : Float := maxMatErr A (mm (mm V (diagFromVec evals)) (tr V))
/-- Orthogonality error `‖Vᵀ·V - I‖_max`. -/
def orthoErr : Float := maxMatErr (mm (tr V) V) (Spec.identityTensorSpec 3)

#eval vecToList evals

-- Compiled assertions (fail the build otherwise).
#eval assertLt "SymEig A = V·diag(λ)·Vᵀ" reconErr
#eval assertLt "SymEig Vᵀ·V = I" orthoErr

end NN.Examples.Factorization.SymEig
