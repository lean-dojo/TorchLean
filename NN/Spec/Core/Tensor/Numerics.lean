/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Logic.Function.Iterate
public import NN.Spec.Core.TensorReductionShape.LinearAlgebra

/-!
# Tensor Numerical Algorithms

Reference implementations shared by classical models, graph specifications, and runtime checks:

- matrix minors, determinants, inverses, and power iteration;
- distances between vectors;
- vector normalization.

## Intent / tradeoffs

These definitions prioritize:
- **mathematical clarity**, and
- **shape safety** (via `TorchLean.Tensor`),
over performance.

In particular, `determinantSpec` uses Laplace expansion, which is exponentially expensive and is
only meant for small matrices (for example, 2 x 2 or 3 x 3) and proof-oriented reference code. For
large-scale linear algebra, use the runtime layer with array-backed kernels.
-/

@[expose] public section


open TorchLean

namespace Spec

open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

-- Matrix operations used by classical model specifications.

/-- The index in `Fin n` corresponding to `i : Fin (n - 1)` after skipping one position. -/
theorem minor_index_lt {n : ℕ} (skip : Fin n) (i : Fin (n - 1)) :
    (if i.val < skip.val then i.val else i.val + 1) < n := by
  by_cases h : i.val < skip.val <;> simp [h]
  · exact Nat.lt_of_lt_of_le i.isLt (Nat.pred_le n)
  · grind

/--
Matrix minor: delete `row` and `col` from an `n × n` matrix, producing an `(n-1) × (n-1)` matrix.

This is used by `determinantSpec` (Laplace expansion) and the adjugate-based inverse below.
-/
def matrixMinorSpec {α : Type} [TorchLean.Storage α] {n : Nat}
    (matrix : Tensor α [n, n])
    (row col : Fin n) :
    Tensor α [n - 1, n - 1] :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      let actualI := if i.val < row.val then i.val else i.val + 1
      let actualJ := if j.val < col.val then j.val else j.val + 1
      Tensor.scalar (
        get2 matrix
          ⟨actualI, minor_index_lt row i⟩
          ⟨actualJ, minor_index_lt col j⟩
      )
    )
  )


/--
Determinant of an `n × n` matrix (spec-level reference implementation).

This uses Laplace expansion (cofactor expansion) along the first row, with special-cased base cases
for `n = 0, 1, 2`. It is mathematically clear but exponentially slow, so it is intended only for
very small `n` and/or proof-oriented reference code.
-/
def determinantSpec {α : Type} [TorchLean.Storage α] [Context α] :
  ∀ {n : Nat}, Tensor α [n, n] → Tensor α .scalar
| 0, _ => Tensor.scalar 1
| 1, A =>
  Tensor.scalar (get2 A ⟨0, Nat.zero_lt_succ 0⟩ ⟨0, Nat.zero_lt_succ 0⟩)
| 2, A =>
  let zero : Fin 2 := ⟨0, Nat.zero_lt_succ 1⟩
  let one : Fin 2 := ⟨1, Nat.succ_lt_succ (Nat.zero_lt_succ 0)⟩
  Tensor.scalar (get2 A zero zero * get2 A one one - get2 A zero one * get2 A one zero)
| n+2, A =>
  let laplaceTerm (j : Fin (n+2)) :=
    let minor := matrixMinorSpec A ⟨0, Nat.zero_lt_succ (n + 1)⟩ j
    let cofactor : α := if j.val % 2 = 0 then 1 else -1
    let element := get2 A ⟨0, Nat.zero_lt_succ (n+1)⟩ j
    cofactor * element * Tensor.item (determinantSpec minor)
  let sum := (List.finRange (n+2)).foldl (fun acc j => acc + laplaceTerm j) 0
  Tensor.scalar sum


-- Matrix inverse (implemented using adjugate method)
/--
Matrix inverse via the adjugate formula (spec-level reference implementation).

The result is `none` when the backend comparison `det == 0` succeeds. No conditioning or finiteness
check is performed; on floating-point backends, a nonzero or NaN determinant can therefore produce
nonfinite entries.

PyTorch analogue: `torch.linalg.inv`, with failure represented explicitly by `Option`.
-/
def inverseSpec? {n : Nat}
  (matrix : Tensor α [n, n]) :
  Option (Tensor α [n, n]) :=
  let det := Tensor.item (determinantSpec matrix)
  if det == 0 then
    none
  else
    -- Compute the cofactor matrix `C` and then transpose it to get the adjugate `adj(A) = Cᵀ`.
    --
    -- Note: the transpose matters. Without it you'd get the cofactor matrix, not the adjugate.
    let cofactors :=
      Tensor.dim (fun i =>
        Tensor.dim (fun j =>
          let cofactor := if (i.val + j.val) % 2 = 0 then 1 else -1
          let minor := matrixMinorSpec matrix i j
          let minorDet := Tensor.item (determinantSpec minor)
          Tensor.scalar (cofactor * minorDet)))
    let adjugate := swapAdjacentAxes cofactors 0
    -- Scale by 1/det
    some (scaleSpec adjugate (1 / det))

/--
Approximate the leading eigenpair by a caller-selected number of power-iteration steps.

The scalar is the final dot product `v · (matrix v)`, which is the Rayleigh quotient when `v` has
unit norm. Each normalization is attempted only when its computed norm is positive; finite-precision
execution does not guarantee a unit vector. Convergence to a dominant eigenvector requires spectral
assumptions on `matrix` and a suitable initial vector.

The default `initial` is the normalized all-ones vector. An explicitly supplied tensor is used
as-is, so callers can retain their own initialization and normalization convention.
-/
def powerIterationLeadingEigenpairSpec {n : Nat}
    (matrix : Tensor α [n, n]) (iterations : Nat)
    (initial : Tensor α [n] :=
      let raw : Tensor α [n] := Tensor.ones [n]
      let norm := MathFunctions.sqrt (sumSpec (squareSpec raw))
      if norm > 0 then scaleSpec raw (1 / norm) else raw) :
    α × Tensor α [n] :=
  -- Retain the last iterate when its next computed norm is not positive.
  let step (v : Tensor α [n]) : Tensor α [n] :=
    let next := matVecMulSpec matrix v
    let norm := MathFunctions.sqrt (sumSpec (squareSpec next))
    if norm > 0 then Tensor.map (fun value => value / norm) next else v
  let direction := step^[iterations] initial
  (dotSpec direction (matVecMulSpec matrix direction), direction)


-- Distance functions used by nearest-neighbor, clustering, and metric-learning specs.

/--
Euclidean (L2) distance between two feature vectors.

PyTorch analogue: `torch.linalg.vector_norm(x - y)` or `torch.cdist` (batched).
-/
def euclideanDistanceSpec {nFeatures : Nat}
  (x y : Tensor α [nFeatures]) : α :=
  let diff := subSpec x y
  let sumSquared := sumSpec (squareSpec diff)
  let finiteDifferences := foldlSpec (fun finite value =>
    finite && value - value == 0) true diff
  -- Preserve nonfinite differences before considering a scale: an overflowing subtraction
  -- must remain an infinite distance, and a NaN must not become an exact match.
  if !finiteDifferences || (sumSquared - sumSquared == 0 && !(sumSquared == 0)) then
    MathFunctions.sqrt sumSquared
  else
    -- Finite differences can lose their norm when squaring overflows or rounds to zero.
    -- Dividing by the largest magnitude keeps each square at most one; restore the scale
    -- after the square root. Coincident inputs retain the original square-root convention.
    let scale := foldlSpec (fun largest value => max largest (MathFunctions.abs value)) 0 diff
    if scale == 0 then MathFunctions.sqrt sumSquared
    else
      let scaledSum := foldlSpec (fun total value =>
        let ratio := value / scale
        total + ratio * ratio) 0 diff
      scale * MathFunctions.sqrt scaledSum

/--
L2-normalize a vector with an additive regularizer under the square root.

The denominator is

`sqrt (sumᵢ vector[i] ^ 2 + regularizer)`.

This operation has no zero-norm branch. Callers are responsible for
choosing a regularizer that makes the denominator meaningful in their scalar context. This is the
normalization convention used by several attention and recurrent architectures, where the exact
regularizer is part of the model specification.
-/
def normalizeL2RegularizedSpec {n : Nat}
    (vector : Tensor α [n]) (regularizer : α) :
    Tensor α [n] :=
  let norm := MathFunctions.sqrt (sumSpec (squareSpec vector) + regularizer)
  Tensor.map (fun value => value / norm) vector

/--
Z-score normalization: subtract the mean and divide by the population standard deviation.

When `n = 0`, the empty input is returned before forming a division. Otherwise the mean and
variance divide by the scalar cast `(n : α)` without testing that cast for zero or finiteness.
The final division occurs only when the computed standard deviation is positive; otherwise the
mean-centered vector is returned. These guards do not guarantee finite output on every backend.
-/
def normalizeZscoreSpec {n : Nat} (vector : Tensor α [n]) :
  Tensor α [n] :=
  if n = 0 then
    vector
  else
    let count : α := (n : α)
    let mean := sumSpec vector / count
    let centered := Tensor.map (fun value => value - mean) vector
    let variance := sumSpec (squareSpec centered) / count
    let std := MathFunctions.sqrt variance
    if std > 0 then
      Tensor.map (fun value => value / std) centered
    else
      centered

end Spec
