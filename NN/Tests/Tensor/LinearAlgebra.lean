/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor

/-!
# Public Tensor Linear-Algebra Regression Tests

Numerical checks for the user-facing QR, Cholesky, and ridge-solve API. The tests cover reduced
shape inference, tall and wide reconstruction, structural zeros, rank-deficient QR behavior, empty
axes, and a solved linear system.
-/

@[expose] public section

namespace NN.Tests.Tensor.LinearAlgebra

open TorchLean
open TorchLean.Tensor

def expect (label : String) (condition : Bool) : IO Unit := do
  unless condition do
    throw <| IO.userError s!"tensor linear-algebra check failed: {label}"

/-- Invalid columns must not alter the active-basis policy or contaminate unused tensor slots. -/
def checkExceptionalWideQR : IO Unit := do
  let nan : Float := 0.0 / 0.0
  let firstInvalid : Tensor Float [2, 3] := [[nan, 1.0, 0.0], [0.0, 0.0, 1.0]]
  let firstFactors := Tensor.qr firstInvalid
  expect "invalid first column leaves room for later independent columns"
    (Tensor.maxAbsDiff firstFactors.q (Tensor.identity 2) == 0)
  expect "invalid first column has no active projection coefficients"
    (Tensor.maxAbsDiff firstFactors.r
      ([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] : Tensor Float [2, 3]) == 0)
  let lastInvalid : Tensor Float [2, 3] := [[1.0, 0.0, nan], [0.0, 1.0, 0.0]]
  let lastFactors := Tensor.qr lastInvalid
  expect "invalid later column preserves the completed basis"
    (Tensor.maxAbsDiff lastFactors.q (Tensor.identity 2) == 0)
  expect "active projections preserve IEEE invalid products"
    (lastFactors.r[0][2].isNaN && lastFactors.r[1][2].isNaN)

def run : IO Unit := do
  checkExceptionalWideQR
  let rectangular : Tensor Float [3, 2] :=
    [[1.0, 1.0],
     [1.0, 0.0],
     [0.0, 1.0]]
  let factors := Tensor.qr rectangular
  let rectangularQ : Tensor Float [3, 2] := factors.q
  let rectangularR : Tensor Float [2, 2] := factors.r
  let reconstructed : Tensor Float [3, 2] :=
    einsum rectangularQ, rectangularR
      "row contracted, contracted column -> row column"
  expect "rectangular QR reconstructs its input"
    (Tensor.maxAbsDiff rectangular reconstructed < 1e-6)

  let qtq : Tensor Float [2, 2] :=
    einsum rectangularQ, rectangularQ
      "contracted row, contracted column -> row column"
  expect "full-rank rectangular QR has orthonormal columns"
    (Tensor.maxAbsDiff qtq (Tensor.identity 2) < 1e-6)
  expect "QR R factor is upper triangular"
    (rectangularR[1][0].abs < 1e-6)

  let wide : Tensor Float [2, 3] :=
    [[1.0, 2.0, 0.0],
     [0.0, 0.0, 1.0]]
  let wideFactors := Tensor.qr wide
  let wideQ : Tensor Float [2, 2] := wideFactors.q
  let wideR : Tensor Float [2, 3] := wideFactors.r
  let wideReconstructed : Tensor Float [2, 3] :=
    einsum wideQ, wideR
      "row contracted, contracted column -> row column"
  expect "wide reduced QR reconstructs after an early dependent column"
    (Tensor.maxAbsDiff wide wideReconstructed < 1e-6)
  let wideQtq : Tensor Float [2, 2] :=
    einsum wideQ, wideQ
      "contracted row, contracted column -> row column"
  expect "wide reduced QR retains a complete row-space basis"
    (Tensor.maxAbsDiff wideQtq (Tensor.identity 2) < 1e-6)

  let deficient : Tensor Float [3, 2] :=
    [[1.0, 2.0],
     [2.0, 4.0],
     [3.0, 6.0]]
  let deficientFactors := Tensor.qr deficient
  let deficientReconstructed : Tensor Float [3, 2] :=
    einsum deficientFactors.q, deficientFactors.r
      "row contracted, contracted column -> row column"
  expect "rank-deficient QR still reconstructs its input"
    (Tensor.maxAbsDiff deficient deficientReconstructed < 1e-6)
  let deficientSecondColumn : Tensor Float [3] :=
    rearrange deficientFactors.q "row column -> column row" |>.get 1
  expect "rank-deficient QR emits a zero dependent column"
    (Tensor.maxAbs deficientSecondColumn < 1e-6)

  let noRows : Tensor Float [0, 3] := Tensor.zeros [0, 3]
  let noRowsFactors := Tensor.qr noRows
  expect "zero-row QR has an empty Q buffer"
    ((Tensor.to noRowsFactors.q (Array Float)).isEmpty)
  expect "zero-row QR has an empty R buffer"
    ((Tensor.to noRowsFactors.r (Array Float)).isEmpty)

  let noColumns : Tensor Float [3, 0] := Tensor.zeros [3, 0]
  let noColumnsFactors := Tensor.qr noColumns
  expect "zero-column QR has an empty Q buffer"
    ((Tensor.to noColumnsFactors.q (Array Float)).isEmpty)
  expect "zero-column QR has an empty R buffer"
    ((Tensor.to noColumnsFactors.r (Array Float)).isEmpty)

  let positiveDefinite : Tensor Float [3, 3] :=
    [[4.0, 2.0, 2.0],
     [2.0, 5.0, 3.0],
     [2.0, 3.0, 6.0]]
  let lower := Tensor.cholesky positiveDefinite
  let choleskyReconstructed : Tensor Float [3, 3] :=
    einsum lower, lower
      "row contracted, column contracted -> row column"
  expect "Cholesky reconstructs a positive-definite matrix"
    (Tensor.maxAbsDiff positiveDefinite choleskyReconstructed < 1e-6)
  expect "Cholesky factor is lower triangular"
    (lower[0][1].abs < 1e-6 && lower[0][2].abs < 1e-6 &&
      lower[1][2].abs < 1e-6)

  let kernel : Tensor Float [2, 2] :=
    [[2.0, 0.0],
     [0.0, 3.0]]
  let target : Tensor Float [2] := [6.0, 8.0]
  let solution := Tensor.solveRidge kernel 1.0 target
  expect "ridge solve returns the known diagonal-system solution"
    (Tensor.maxAbsDiff solution ([2.0, 2.0] : Tensor Float [2]) < 1e-6)

  IO.println "  public tensor linear algebra: passed"

end NN.Tests.Tensor.LinearAlgebra
