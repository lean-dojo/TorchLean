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
**all** eigenpairs.

These checks are designed to give a reviewer confidence in the matching formal development
(`NN.Proofs.Tensor.Basic.FactorizationsJacobi`), and in particular to exhibit the precise boundary
between what is proved *exactly / a-priori* and what is only *asymptotic*:

* **Spectral reconstruction** `A = V · diag(λ) · Vᵀ` and orthogonality `Vᵀ V = I` hold at high sweep
  counts (positive checks).
* **Orthogonality is exact at *any* sweep count** — even after a single sweep `Vᵀ V = I` to machine
  precision. This is the numeric witness of `jacobi_orthogonal`, which is an a-priori theorem (no
  convergence hypothesis).
* **Diagonalization is only asymptotic**: one sweep leaves a genuine off-diagonal residual that more
  sweeps drive to zero. This is the "rate" that remains a-posteriori (`What remains` in the blueprint).
* **The exact residual certificate** `‖A − V·diag(λ)·Vᵀ‖_F² = ‖offDiag(VᵀAV)‖_F²`
  (`symEigJacobi_frobenius_residual`) is checked numerically at a *low* sweep count, where both sides
  are large and equal — the two sides are computed by independent routines, so the match is evidence
  the identity is real and not a tautology of the code.
-/

@[expose] public section


namespace NN.Examples.Factorization.SymEig

/-- A symmetric test matrix (eigenvalues ≈ {1.3249, 2.4608, 5.2143}). -/
def A : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) :=
  mkMat [[2, 1, 1],
         [1, 3, 1],
         [1, 1, 4]]

/-- The 3×3 identity (target for the orthogonality checks). -/
def I3 : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) := Spec.identityTensorSpec 3

/-- Eigendecomposition after 8 sweeps (converged) and after 1 sweep (not yet converged). -/
def eig8 : Spec.Tensor Float (.dim 3 .scalar) × Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) :=
  Spec.symEigJacobiSpec A 8
def eig1 : Spec.Tensor Float (.dim 3 .scalar) × Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) :=
  Spec.symEigJacobiSpec A 1

/-- Eigenvalues and eigenvector matrix `V` (columns are eigenvectors) at 8 sweeps. -/
def evals8 : Spec.Tensor Float (.dim 3 .scalar) := eig8.1
def V8 : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) := eig8.2
/-- Eigenvector matrix after a single sweep. -/
def V1 : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) := eig1.2

/-- Rotated matrices `Af = Vᵀ A V` after 1 and 8 sweeps (diagonal in the limit). -/
def Af1 : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) := mm (mm (tr V1) A) V1
def Af8 : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) := mm (mm (tr V8) A) V8

/-- Spectral reconstruction error `‖A - V·diag(λ)·Vᵀ‖_max` at 8 sweeps. -/
def reconErr8 : Float := maxMatErr A (mm (mm V8 (diagFromVec evals8)) (tr V8))
/-- Orthogonality error `‖Vᵀ·V - I‖_max` at 8 and at 1 sweep. -/
def orthoErr8 : Float := maxMatErr (mm (tr V8) V8) I3
def orthoErr1 : Float := maxMatErr (mm (tr V1) V1) I3

/-- Off-diagonal mass of `Af` after 1 and 8 sweeps (the squared reconstruction residual). -/
def offResid1 : Float := offDiagFrobSq Af1
def offResid8 : Float := offDiagFrobSq Af8

/-- Reconstruction side of the exact certificate, computed independently at 1 sweep. -/
def reconFrobSq1 : Float := frobSqErr A (mm (mm V1 (diagFromVec (diagOf Af1))) (tr V1))

#eval vecToList evals8
#eval IO.println s!"off-diagonal mass: 1 sweep = {offResid1}, 8 sweeps = {offResid8}"

-- Positive checks at convergence.
#eval assertLt "SymEig(8) A = V·diag(λ)·Vᵀ" reconErr8
#eval assertLt "SymEig(8) Vᵀ·V = I" orthoErr8

-- Orthogonality is EXACT after a single sweep (numeric witness of the a-priori `jacobi_orthogonal`).
#eval assertLt "SymEig(1) Vᵀ·V = I  (orthogonality exact at any sweep count)" orthoErr1

-- Diagonalization is only asymptotic: 1 sweep leaves a real residual, 8 sweeps remove it.
#eval assertGe "SymEig(1) off-diagonal residual is non-negligible" offResid1 0.01
#eval assertLt "SymEig(8) off-diagonal residual ≈ 0" offResid8

-- The EXACT residual certificate `‖A - V·diag(λ)·Vᵀ‖² = ‖offDiag(VᵀAV)‖²`, at a sweep count where
-- both sides are large — independent computations agree (witness of `symEigJacobi_frobenius_residual`).
#eval assertApproxEq "SymEig residual certificate ‖A-V·diagΛ·Vᵀ‖² = ‖offDiag(VᵀAV)‖²"
  reconFrobSq1 offResid1

end NN.Examples.Factorization.SymEig
