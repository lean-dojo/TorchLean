/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Logic.Function.Iterate
public import TorchLean.Tensor.AdjacentSwap
public import TorchLean.Tensor.Contract

/-!
# Tensor Numerical Algorithms

Reference implementations for the classical model specifications and for runtime code such as PPO
advantage normalization: matrix minors, determinants and inverses (`minor`,
`determinant`, `inverse?`), power iteration (`powerIteration`), the
Euclidean distance (`euclideanDistance`) and two vector normalizations
(`normalizeL2`, `normalizeZscore`).

The definitions put mathematical clarity and shape safety (through `TorchLean.Tensor`) ahead of
speed. `determinant` uses Laplace expansion, which costs exponential time, so it is meant for
small matrices (for example, 2 x 2 or 3 x 3) and for reference code in proofs. For large-scale
linear algebra, use the runtime layer and its array-backed kernels.
-/

@[expose] public section


open FloatLib.Numerics (MathFunctions)

namespace TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

-- Matrix operations used by classical model specifications.

/-- Skipping position `skip` sends `i : Fin (n - 1)` to a natural number below `n`: to `i` when
`i < skip`, and to `i + 1` otherwise. -/
theorem minor_index_lt {n : ℕ} (skip : Fin n) (i : Fin (n - 1)) :
    (if i.val < skip.val then i.val else i.val + 1) < n := by
  by_cases h : i.val < skip.val <;> simp [h]
  · exact Nat.lt_of_lt_of_le i.isLt (Nat.pred_le n)
  · grind

/--
The minor of an `n × n` matrix with `row` and `col` deleted, an `(n-1) × (n-1)` matrix.

`determinant` uses these minors for Laplace expansion, and `Tensor.inverse?` uses them for the
adjugate.
-/
def minor {α : Type} [Storage α] {n : Nat}
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
Determinant of an `n × n` matrix by cofactor (Laplace) expansion along the first row, with the
cases `n = 0, 1, 2` written out.

The expansion is clear but takes exponential time, so it is meant for very small `n` and for
reference code in proofs.
-/
def determinant {α : Type} [Storage α] [Context α] :
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
    let minor := Tensor.minor A ⟨0, Nat.zero_lt_succ (n + 1)⟩ j
    let cofactor : α := if j.val % 2 = 0 then 1 else -1
    let element := get2 A ⟨0, Nat.zero_lt_succ (n+1)⟩ j
    cofactor * element * Tensor.item (determinant minor)
  let sum := (List.finRange (n+2)).foldl (fun acc j => acc + laplaceTerm j) 0
  Tensor.scalar sum


/--
Matrix inverse by the adjugate formula, or `none` when the backend comparison `det == 0` succeeds.

No conditioning or finiteness check is made, so on a floating-point backend a near-zero or NaN
determinant can give nonfinite entries. PyTorch analogue: `torch.linalg.inv`, with failure
returned as `none`.
-/
def inverse? {n : Nat}
  (matrix : Tensor α [n, n]) :
  Option (Tensor α [n, n]) :=
  let det := Tensor.item (determinant matrix)
  if det == 0 then
    none
  else
    -- Build the cofactor matrix `C`. The adjugate is its transpose, `adj(A) = Cᵀ`.
    let cofactors :=
      Tensor.dim (fun i =>
        Tensor.dim (fun j =>
          let cofactor := if (i.val + j.val) % 2 = 0 then 1 else -1
          let minor := Tensor.minor matrix i j
          let minorDet := Tensor.item (determinant minor)
          Tensor.scalar (cofactor * minorDet)))
    let adjugate := Tensor.permute cofactors (Shape.adjacentSwap [n, n] 0) [n, n]
    some (Tensor.scale adjugate (1 / det))

/--
Approximate the leading eigenpair with `iterations` steps of power iteration from `initial`,
returning the dot product `v · (matrix v)` and the final iterate `v`.

Each step divides `matrix v` by its largest magnitude (`Tensor.maxAbs`) before squaring, so in
floating point no square overflows, and then by the computed norm. It keeps the last iterate when
`matrix v` is zero or has a NaN or infinite entry, and in finite precision the normalized vector
need not have unit norm. The scalar is the Rayleigh quotient when `v` does have unit norm.
Convergence to a dominant eigenvector needs spectral assumptions on `matrix` and a suitable initial
vector.

The default `initial` is the normalized all-ones vector. A supplied tensor is used as is, so callers
keep their own initialization and normalization convention.
-/
def powerIteration {n : Nat}
    (matrix : Tensor α [n, n]) (iterations : Nat)
    (initial : Tensor α [n] :=
      let raw : Tensor α [n] := Tensor.ones [n]
      let norm := MathFunctions.sqrt (Tensor.sum (Tensor.square raw))
      if norm > 0 then Tensor.scale raw (1 / norm) else raw) :
    α × Tensor α [n] :=
  -- Retain the last iterate when the largest magnitude or the norm is not positive.
  let step (v : Tensor α [n]) : Tensor α [n] :=
    let next := Tensor.contract matrix v _ _ [n] .matVec
    let scale := Tensor.maxAbs next
    if scale > 0 then
      let scaled := Tensor.map (fun value => value / scale) next
      let norm := MathFunctions.sqrt (Tensor.sum (Tensor.square scaled))
      if norm > 0 then Tensor.map (fun value => value / norm) scaled else v
    else v
  let direction := step^[iterations] initial
  (dot direction (Tensor.contract matrix direction _ _ [n] .matVec), direction)


-- The Euclidean distance, used by the k-nearest-neighbor specification.

/--
Euclidean (L2) distance between two feature vectors.

PyTorch analogue: `torch.linalg.vector_norm(x - y)` or `torch.cdist` (batched). When the sum of
squares overflows, or every square rounds to zero, the differences are divided by their largest
magnitude before squaring. A subnormal nonzero sum is not rescaled, as in
`torch.linalg.vector_norm`, and keeps only its few significant bits: a difference
`[1.6e-162, 1.6e-162]` gives `3.14e-162`, not `2.26e-162`.
-/
def euclideanDistance {nFeatures : Nat}
  (x y : Tensor α [nFeatures]) : α :=
  let diff := x - y
  let sumSquared := Tensor.sum (Tensor.square diff)
  let finiteDifferences := Tensor.foldl (fun finite value =>
    finite && value - value == 0) true diff
  -- Nonfinite differences skip the rescaling below: an overflowing subtraction has to give an
  -- infinite distance, and a NaN must not become an exact match.
  if !finiteDifferences || (sumSquared - sumSquared == 0 && !(sumSquared == 0)) then
    MathFunctions.sqrt sumSquared
  else
    -- The sum of squares overflowed or every square rounded to zero. Dividing by the largest
    -- magnitude keeps each square at most one, and the scale comes back after the square root.
    -- Coincident inputs (scale `0`) take the unscaled square root.
    let scale := Tensor.foldl (fun largest value => max largest (MathFunctions.abs value)) 0 diff
    if scale == 0 then MathFunctions.sqrt sumSquared
    else
      let scaledSum := Tensor.foldl (fun total value =>
        let ratio := value / scale
        total + ratio * ratio) 0 diff
      scale * MathFunctions.sqrt scaledSum

/--
L2 normalization with an additive regularizer under the square root: every entry is divided by
`sqrt (sumᵢ vector[i] ^ 2 + regularizer)`, with the mean square in place of the sum when
`mean := true`.

The entries are divided by a shared positive scale before squaring and the regularizer by the
square of that scale, so for a nonzero finite scale every scaled entry has magnitude at most one.
The scale is detached: its value cancels from the real function, so it contributes no derivative
and no max-selection kink.

Safe scaling of sums of squares is classical (see Edward Anderson (2017), "Algorithm 978: Safe
Scaling in the Level 1 BLAS", ACM Transactions on Mathematical Software 44(1), 1-28,
[doi:10.1145/3061665](https://doi.org/10.1145/3061665)), and here the regularizer is scaled too,
to preserve the epsilon-bearing formula. Finite output is not guaranteed in every format: the
reduction, its element-count cast or a gradient can exceed that format's range.

There is no zero-norm branch, so callers choose a regularizer that makes the denominator meaningful
for their scalar type. Several attention and recurrent architectures normalize this way, and the
exact regularizer is part of their model specification.
-/
def normalizeL2 {n : Nat}
    (vector : Tensor α [n]) (regularizer : α) (mean : Bool := false) :
    Tensor α [n] :=
  let scale := Context.stopGradient
    (max (Tensor.maxAbs vector) (MathFunctions.sqrt (MathFunctions.abs regularizer)))
  let scaled := Tensor.map (fun value => value / scale) vector
  let squares := Tensor.square scaled
  let squaredNorm :=
    if mean then Tensor.sum squares / (meanDenominator [n] : α) else Tensor.sum squares
  let norm := MathFunctions.sqrt (squaredNorm + (regularizer / scale) / scale)
  Tensor.map (fun value => value / norm) scaled

/--
Z-score normalization: subtract the mean and divide by the population standard deviation.

An empty input (`n = 0`) is returned as it is. Otherwise the mean is `Context.divNat` of the sum.
The centered vector is divided by its gradient-stopped largest magnitude (`Tensor.maxAbs`) before
squaring, so no square overflows, and the scaled vector over its standard deviation is the z-score.
When that magnitude is not positive, as when every centered entry is zero or one is NaN, the
mean-centered vector is returned. The sum of the entries and the centering can overflow.
We divide by `n`, as `torch.std(x, correction=0)` does, where the `torch.std` default (PyTorch 2.8
docs) is Bessel's `correction=1`: `[1, 2, 3]` maps to `[-1.2247, 0, 1.2247]` here and to
`[-1, 0, 1]` under `(x - x.mean()) / x.std()`.
-/
def normalizeZscore {n : Nat} (vector : Tensor α [n]) :
  Tensor α [n] :=
  if n = 0 then
    vector
  else
    let mean := Context.divNat (Tensor.sum vector) n
    let centered := Tensor.map (fun value => value - mean) vector
    let scale := Context.stopGradient (Tensor.maxAbs centered)
    if scale > 0 then
      let scaled := Tensor.map (fun value => value / scale) centered
      let std := MathFunctions.sqrt (Context.divNat (Tensor.sum (Tensor.square scaled)) n)
      Tensor.map (fun value => value / std) scaled
    else
      centered

end TorchLean.Tensor
