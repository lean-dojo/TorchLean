/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Factorization.Common

/-!
# Cholesky Factorization

`Tensor.cholesky A` returns a lower-triangular factor candidate. Here we factor a 3×3 symmetric
positive-definite matrix and check that its `Float` reconstruction error is small.
-/

@[expose] public section

namespace NN.Examples.Factorization.Cholesky

open TorchLean
open TorchLean.Tensor

/-- Symmetric positive-definite matrix used for the positive Cholesky check. -/
def positiveDefiniteMatrix : Tensor Float [3, 3] :=
  [[4, 2, 2],
   [2, 5, 3],
   [2, 3, 6]]

/-- Lower-triangular Cholesky factor. -/
def lowerFactor := Tensor.cholesky positiveDefiniteMatrix

/-- Reconstruction error $\lVert A-LL^\mathsf{T}\rVert_{\max}$. -/
def reconstructionError : Float :=
  Tensor.maxAbsDiff positiveDefiniteMatrix <|
    einsum lowerFactor, lowerFactor "row contracted, column contracted -> row column"

/-! ## Negative Control

Cholesky requires positive pivots. The matrix below is symmetric but not positive-definite
(eigenvalues `3` and `-1`), so the `Float` computation reaches the square root of a negative value
and the reconstruction error becomes `NaN`.
-/

/-- A symmetric but **indefinite** matrix (eigenvalues `{3, -1}`), outside Cholesky's domain. -/
def indefiniteMatrix : Tensor Float [2, 2] :=
  [[1, 2],
   [2, 1]]

/--
The negative pivot produces a `NaN` factor entry, making reconstruction fail.
-/
def indefiniteFactor := Tensor.cholesky indefiniteMatrix

-- Squaring and summing also preserves the invalid square root as a NaN error.
/-- Reconstruction error for the indefinite case, which should come out `NaN`. -/
def indefiniteReconstructionError : Float :=
  Tensor.sum <| Tensor.square <| indefiniteMatrix -
    einsum indefiniteFactor, indefiniteFactor "row contracted, column contracted -> row column"

/-- Run the positive reconstruction check and its indefinite-matrix negative control. -/
def check : IO Unit := do
  assertBelow "Cholesky A = L·Lᵀ" reconstructionError
  assertNotBelow "Cholesky on indefinite A correctly fails (no SPD ⇒ no factor)"
    indefiniteReconstructionError

end NN.Examples.Factorization.Cholesky
