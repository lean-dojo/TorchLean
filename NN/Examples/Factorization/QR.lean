/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Factorization.Common

/-!
# QR Factorization

Check whether `Tensor.qr A` reconstructs $A$ as $QR$ and gives orthonormal columns in $Q$.
The checks use `Float` with an explicit tolerance. Reduced QR uses
`Q : Tensor Float [rows, min rows columns]` and
`R : Tensor Float [min rows columns, columns]`.

Run `lake exe torchlean factorizations`. The rank-deficient negative control still reconstructs
the input, but fails orthonormality: reconstructing a matrix alone does not establish both QR
properties. This file tests the implementation; it does not prove a real-arithmetic QR theorem.
-/

@[expose] public section

namespace NN.Examples.Factorization.QR

open TorchLean
open TorchLean.Tensor

/-- Full-rank matrix used for the positive QR check. -/
def fullRankMatrix : Tensor Float [3, 3] :=
  [[12, -51, 4],
   [6, 167, -68],
   [-4, 24, -41]]

/-- Compute both factors once. -/
def fullRankFactors := Tensor.qr fullRankMatrix

/-- Reconstruction error $\lVert A-QR\rVert_{\max}$. -/
def reconstructionError : Float :=
  Tensor.maxAbsDiff fullRankMatrix <|
    einsum fullRankFactors.q, fullRankFactors.r
      "row contracted, contracted column -> row column"

/-- Orthonormality error $\lVert Q^\mathsf{T}Q-I\rVert_{\max}$. -/
def orthonormalityError : Float :=
  let qtq : Tensor Float [3, 3] :=
    einsum fullRankFactors.q, fullRankFactors.q
      "contracted row, contracted column -> row column"
  Tensor.maxAbsDiff qtq (Tensor.identity 3)

/-! ## Wide Matrix

This case also places a dependent column before a later independent column. A reduced
implementation that merely truncates the old square factors loses that later basis direction.
-/

def wideMatrix : Tensor Float [2, 3] :=
  [[1, 2, 0],
   [0, 0, 1]]

/-- Reduced QR of a wide matrix: `Q` is `2 x 2` and `R` is `2 x 3`. -/
def wideFactors := Tensor.qr wideMatrix

/-- How far `Q R` is from the original matrix; should be at rounding level. -/
def wideReconstructionError : Float :=
  Tensor.maxAbsDiff wideMatrix <|
    einsum wideFactors.q, wideFactors.r
      "row contracted, contracted column -> row column"

/-- How far `Qᵀ Q` is from the identity, the other half of what QR promises. -/
def wideOrthonormalityError : Float :=
  let qtq : Tensor Float [2, 2] :=
    einsum wideFactors.q, wideFactors.q
      "contracted row, contracted column -> row column"
  Tensor.maxAbsDiff qtq (Tensor.identity 2)

/-! ## Negative Control

The orthonormality property requires full column rank. The following matrix has one dependent
column. Gram-Schmidt still reconstructs it, but the corresponding column of `Q` vanishes and
$Q^\mathsf{T}Q\ne I$.
-/

/-- A matrix whose second column is twice its first. -/
def rankDeficientMatrix : Tensor Float [3, 3] :=
  [[1, 2, 0],
   [2, 4, 1],
   [1, 2, 0]]

/--
QR of a rank-deficient matrix. Reconstruction survives; orthonormality does not, because the
dependent column contributes a zero column to `Q`.
-/
def rankDeficientFactors := Tensor.qr rankDeficientMatrix

/-- Reconstruction still holds without full rank. -/
def rankDeficientReconstructionError : Float :=
  Tensor.maxAbsDiff rankDeficientMatrix <|
    einsum rankDeficientFactors.q, rankDeficientFactors.r
      "row contracted, contracted column -> row column"

/-- Orthonormality fails because `Q` has a zero column. -/
def rankDeficientOrthonormalityError : Float :=
  let qtq : Tensor Float [3, 3] :=
    einsum rankDeficientFactors.q, rankDeficientFactors.q
      "contracted row, contracted column -> row column"
  Tensor.maxAbsDiff qtq (Tensor.identity 3)

/-- Run square and wide full-rank checks plus the dependent-column negative control. -/
def check : IO Unit := do
  assertBelow "QR A = Q·R" reconstructionError
  assertBelow "QR Qᵀ·Q = I" orthonormalityError
  assertBelow "QR(wide) A = Q·R" wideReconstructionError
  assertBelow "QR(wide) Qᵀ·Q = I" wideOrthonormalityError
  assertBelow "QR(rank-deficient) A = Q·R still reconstructs"
    rankDeficientReconstructionError
  assertAtLeast "QR(rank-deficient) Qᵀ·Q = I correctly fails (needs full column rank)"
    rankDeficientOrthonormalityError

end NN.Examples.Factorization.QR
