/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape

/-!
# Common helpers (spec models)

This file centralizes small utilities used across multiple spec‑level models:

- simple matrix operations (minor, determinant, inverse),
- distance functions for KNN,
- normalization helpers,
- and other "model glue" functions.

## Intent / tradeoffs

These definitions prioritize:
- **mathematical clarity**, and
- **shape safety** (via `Spec.Tensor`),
over performance.

In particular, `determinant_spec` uses Laplace expansion, which is exponentially expensive and is
only meant for small matrices (e.g. 2×2, 3×3) and/or proof‑oriented reference code. If you need
large‑scale linear algebra, use the runtime layer with array‑backed kernels.
-/

@[expose] public section


namespace Spec

open Tensor

variable {α : Type} [Context α]

-- ============================================================================
-- MATRIX OPERATIONS
-- ============================================================================

/--
Helper lemma for minor/index computations.

When forming a matrix minor, we "skip" a row/column and map an index `i : Fin (n-1)` to the
corresponding original index in `Fin n` by either leaving it unchanged (if it is before the skipped
index) or shifting it by `+1` (if it is at/after the skipped index). This lemma proves the resulting
index is still `< n`.
-/
lemma actual_index_lt {n : ℕ} (skip : ℕ) (i : Fin (n-1)) :
  (if i.val < skip then i.val else i.val + 1) < n := by
  by_cases h : i.val < skip <;> simp [h]
  · exact Nat.lt_of_lt_of_le i.isLt (Nat.pred_le n)
  · grind

-- Minor (delete row and col)
/--
Matrix minor: delete `row` and `col` from an `n × n` matrix, producing an `(n-1) × (n-1)` matrix.

This is used by `determinant_spec` (Laplace expansion) and the adjugate-based inverse below.
-/
def getMinorSpec {α : Type} {n : Nat}
    (matrix : Tensor α (.dim n (.dim n .scalar)))
    (row col : Nat) :
    Tensor α (.dim (n - 1) (.dim (n - 1) .scalar)) :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      let actual_i := if i.val < row then i.val else i.val + 1
      let actual_j := if j.val < col then j.val else j.val + 1
      Tensor.scalar (
        get2 matrix
          ⟨actual_i, actual_index_lt row i⟩
          ⟨actual_j, actual_index_lt col j⟩
      )
    )
  )


-- Matrix determinant (implemented using Laplace expansion)
-- Determinant using Laplace expansion along first row
/--
Determinant of an `n × n` matrix (spec-level reference implementation).

This uses Laplace expansion (cofactor expansion) along the first row, with special-cased base cases
for `n = 0, 1, 2`. It is mathematically clear but exponentially slow, so it is intended only for
very small `n` and/or proof-oriented reference code.
-/
def determinantSpec {α : Type} [Context α]:
  ∀ {n : Nat}, Tensor α (.dim n (.dim n .scalar)) → Tensor α .scalar
| 0, _ => Tensor.scalar 1
| 1, A =>
  match A with
  | Tensor.dim rows =>
    match rows ⟨0, Nat.zero_lt_succ 0⟩ with
    | Tensor.dim cols =>
      match cols ⟨0, Nat.zero_lt_succ 0⟩ with
      | Tensor.scalar val => Tensor.scalar val
| 2, A =>
  match A with
  | Tensor.dim rows =>
    match rows ⟨0, Nat.zero_lt_succ 1⟩, rows ⟨1, Nat.succ_lt_succ (Nat.zero_lt_succ 0)⟩ with
    | Tensor.dim row0, Tensor.dim row1 =>
      match row0 ⟨0, Nat.zero_lt_succ 1⟩, row0 ⟨1, Nat.succ_lt_succ (Nat.zero_lt_succ 0)⟩,
            row1 ⟨0, Nat.zero_lt_succ 1⟩, row1 ⟨1, Nat.succ_lt_succ (Nat.zero_lt_succ 0)⟩ with
      | Tensor.scalar a, Tensor.scalar b, Tensor.scalar c, Tensor.scalar d =>
        Tensor.scalar (a * d - b * c)
| n+2, A =>
  let laplace_expansion (j : Fin (n+2)) :=
    let minor := getMinorSpec A 0 j
    let cofactor := if j.val % 2 = 0 then 1 else Numbers.neg_one
    let element := get2 A ⟨0, Nat.zero_lt_succ (n+1)⟩ j
    cofactor * element * Tensor.toScalar (determinantSpec minor)
  let sum := (List.finRange (n+2)).foldl (fun acc j => acc + laplace_expansion j) 0
  Tensor.scalar sum


-- Matrix inverse (implemented using adjugate method)
/--
Matrix inverse via the adjugate formula (spec-level reference implementation).

If `det(A) == 0`, this returns the identity matrix as a "safe default" for singular matrices.
Otherwise it computes `adj(A) / det(A)` using cofactors and a transpose.

PyTorch analogue: `torch.linalg.inv` (but note the singular-case behavior differs).
-/
def inverseSpec {n : Nat}
  (matrix : Tensor α (.dim n (.dim n .scalar))) :
  Tensor α (.dim n (.dim n .scalar)) :=
  let det := Tensor.toScalar (determinantSpec matrix)
  if det == 0 then
    -- Singular matrix, return identity
    identityTensorSpec n
  else
    -- Compute the cofactor matrix `C` and then transpose it to get the adjugate `adj(A) = Cᵀ`.
    --
    -- Note: the transpose matters. Without it you'd get the cofactor matrix, not the adjugate.
    let cofactors :=
      Tensor.dim (fun i =>
        Tensor.dim (fun j =>
          let cofactor := if (i.val + j.val) % 2 = 0 then 1 else -1
          let minor := getMinorSpec matrix i.val j.val
          let minor_det := Tensor.toScalar (determinantSpec minor)
          Tensor.scalar (cofactor * minor_det)))
    let adjugate := matrixTransposeSpec cofactors
    -- Scale by 1/det
    scaleSpec adjugate (1 / det)

-- Eigendecomposition using basic power iteration for the largest eigenvalue
/--
Reference eigendecomposition via power iteration (largest eigenpair only).

This returns a pair `(eigenvalues, eigenvectors)` where only the first eigenvalue/eigenvector slot
is populated (the rest are zeros). It is intended as simple, proof-friendly reference code rather
than a full-featured numerical linear algebra routine.
-/
def eigendecompSpec {n : Nat}
  (matrix : Tensor α (.dim n (.dim n .scalar))) :
  (Tensor α (.dim n .scalar) × Tensor α (.dim n (.dim n .scalar))) :=
  -- Power iteration: returns normalized eigenvector and its Rayleigh quotient
  let rec power_iteration (v : Tensor α (.dim n .scalar)) (iter : Nat) :
    (Tensor α (.dim n .scalar) × α) :=
    if iter = 0 then
      let Av := matVecMulSpec matrix v
      let eigenvalue := dotSpec v Av
      (v, eigenvalue)
    else
      let Av := matVecMulSpec matrix v
      let norm := MathFunctions.sqrt (sumSpec (squareSpec Av))
      let normalized := if norm > 0 then
        Tensor.dim (fun i =>
          match get Av i with
          | Tensor.scalar val => Tensor.scalar (val / norm)
        )
      else v
      power_iteration normalized (iter - 1)

  -- Initialize vector with all ones
  let initial_v := Tensor.dim (fun _ => Tensor.scalar 1)
  let (eigenvector, eigenvalue) := power_iteration initial_v 20
  -- more iterations for better approximation

  -- Construct eigenvalues tensor (first eigenvalue filled, rest zeros)
  let eigenvalues := Tensor.dim (fun i =>
    if i.val = 0 then Tensor.scalar eigenvalue else Tensor.scalar 0)

  -- Construct eigenvectors tensor (first column is eigenvector, rest zero vectors)
  let eigenvectors := Tensor.dim (fun i =>
    if i.val = 0 then eigenvector else Tensor.dim (fun _ => Tensor.scalar 0))

  (eigenvalues, eigenvectors)


-- ============================================================================
-- DISTANCE FUNCTIONS
-- ============================================================================

-- Euclidean distance between two points
/--
Euclidean (L2) distance between two feature vectors.

PyTorch analogue: `torch.linalg.vector_norm(x - y)` or `torch.cdist` (batched).
-/
def euclideanDistanceSpec {nFeatures : Nat}
  (x y : Tensor α (.dim nFeatures .scalar)) : α :=
  let diff := subSpec x y
  let squared_diff := squareSpec diff
  let sum_squared := sumSpec squared_diff
  MathFunctions.sqrt sum_squared

-- Squared Euclidean distance (more efficient for comparisons)
/-- Squared Euclidean distance (avoids the final square root). -/
def squaredEuclideanDistanceSpec {nFeatures : Nat}
  (x y : Tensor α (.dim nFeatures .scalar)) : α :=
  let diff := subSpec x y
  let squared_diff := squareSpec diff
  sumSpec squared_diff

-- Manhattan distance between two points
/-- Manhattan (L1) distance between two feature vectors. -/
def manhattanDistanceSpec {nFeatures : Nat}
  (x y : Tensor α (.dim nFeatures .scalar)) : α :=
  let diff := subSpec x y
  let abs_diff := mapSpec MathFunctions.abs diff
  sumSpec abs_diff

-- Cosine distance between two points
/--
Cosine distance `1 - cos(theta)` between two feature vectors.

If either vector has zero norm, this returns `1`.
-/
def cosineDistanceSpec {nFeatures : Nat}
  (x y : Tensor α (.dim nFeatures .scalar)) : α :=
  let dot_product := dotSpec x y
  let norm_x := MathFunctions.sqrt (sumSpec (squareSpec x))
  let norm_y := MathFunctions.sqrt (sumSpec (squareSpec y))
  let denominator := norm_x * norm_y
  if denominator == 0 then 1 else 1 - (dot_product / denominator)

-- Minkowski distance (generalization of Euclidean and Manhattan)
/--
Minkowski distance of order `p` between two feature vectors.

This generalizes L1 (Manhattan) and L2 (Euclidean). For `p = 1` this is the L1 norm, and for
`p = 2` it is the L2 norm.
-/
def minkowskiDistanceSpec {nFeatures : Nat}
  (p : α) (x y : Tensor α (.dim nFeatures .scalar)) : α :=
  let diff := subSpec x y
  let abs_diff := mapSpec MathFunctions.abs diff
  let powered := mapSpec (fun a => a ^ p) abs_diff
  let sum_powered := sumSpec powered
  sum_powered ^ (1 / p)

-- ============================================================================
-- NORMALIZATION AND UTILITY FUNCTIONS
-- ============================================================================

-- Normalize a tensor to sum to 1 (probability distribution)
/--
Normalize a vector of (nonnegative) scores into a probability distribution.

If the total is `0`, this returns the uniform distribution.

PyTorch analogue: `probs / probs.sum()` (with an explicit zero-sum guard).
-/
def normalizeProbsSpec {n : Nat} (probs : Tensor α (.dim n .scalar)) :
  Tensor α (.dim n .scalar) :=
  let total := sumSpec probs
  if total > 0 then
    Tensor.dim (fun i =>
      match get probs i with
      | Tensor.scalar p => Tensor.scalar (p / total)
    )
  else
    Tensor.dim (fun _ => Tensor.scalar (1 / n))

-- Normalize a tensor to have unit L2 norm
/--
L2-normalize a vector.

If the norm is `0`, this returns the input unchanged.
-/
def normalizeL2Spec {n : Nat} (vector : Tensor α (.dim n .scalar)) :
  Tensor α (.dim n .scalar) :=
  let norm := MathFunctions.sqrt (sumSpec (squareSpec vector))
  if norm > 0 then
    Tensor.dim (fun i =>
      match get vector i with
      | Tensor.scalar v => Tensor.scalar (v / norm)
    )
  else
    vector

-- Normalize a tensor to have zero mean and unit variance
/--
Z-score normalization: subtract mean and divide by standard deviation.

If the standard deviation is `0`, this returns the mean-centered vector.
-/
def normalizeZscoreSpec {n : Nat} (vector : Tensor α (.dim n .scalar)) :
  Tensor α (.dim n .scalar) :=
  let mean := sumSpec vector / n
  let centered := Tensor.dim (fun i =>
    match get vector i with
    | Tensor.scalar v => Tensor.scalar (v - mean)
  )
  let variance := sumSpec (squareSpec centered) / n
  let std := MathFunctions.sqrt variance
  if std > 0 then
    Tensor.dim (fun i =>
      match get centered i with
      | Tensor.scalar v => Tensor.scalar (v / std)
    )
  else
    centered

-- ============================================================================
-- SORTING AND INDEXING FUNCTIONS
-- ============================================================================

-- Argsort in descending order (returns indices)
/--
Argsort in descending order (returns indices as a `Nat` tensor).

PyTorch analogue: `torch.argsort(values, descending=True)`.
-/
def argsortDescendingSpec {n : Nat}
  (values : Tensor α (.dim n .scalar)) :
  Tensor Nat (.dim n .scalar) :=
  -- Create list of (index, value) pairs
  let indexed_values := (List.finRange n).map (fun i =>
    match get values i with
    | Tensor.scalar val => (i.val, val)
  )
  -- Sort by value in descending order
  let sorted_indices := indexed_values.mergeSort (fun a b => a.2 > b.2)
  -- Extract indices
  let indices := sorted_indices.map (fun (idx, _) => idx)
  -- Convert back to tensor
  Tensor.dim (fun i => Tensor.scalar (indices.getD i.val 0))

-- Gather values using indices
/--
Gather elements from a 1-D tensor using a 1-D tensor of indices.

Out-of-bounds indices produce `0`.

PyTorch analogue: `values[indices]` (with an explicit out-of-bounds guard).
-/
def gatherSpec {n : Nat}
  (values : Tensor α (.dim n .scalar))
  (indices : Tensor Nat (.dim n .scalar)) :
  Tensor α (.dim n .scalar) :=
  Tensor.dim (fun i =>
    match get indices i with
    | Tensor.scalar idx =>
      if h : idx < n then
        get values ⟨idx, h⟩
      else
        Tensor.scalar (0 : α)
  )

-- Gather columns using indices
/--
Gather columns of an `m × n` matrix using a length-`n` index vector.

Out-of-bounds indices produce `0` entries.
-/
def gatherColumnsSpec {m n : Nat}
  (matrix : Tensor α (.dim m (.dim n .scalar)))
  (indices : Tensor Nat (.dim n .scalar)) :
  Tensor α (.dim m (.dim n .scalar)) :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      match get indices j with
      | Tensor.scalar idx =>
        if h : idx < n then
          match get matrix i with
          | Tensor.dim row => row ⟨idx, h⟩
        else
          Tensor.scalar (0 : α)
    )
  )

-- Slice columns from start to end
/--
Slice a contiguous block of columns from an `m × n` matrix.

The resulting matrix has `end_p - start` columns. Entries outside the `[start, end_p)` range are
filled with `0` (this matches the "safe default" style used by other helpers in this file).
-/
def sliceColumnsSpec {m n : Nat}
  (matrix : Tensor α (.dim m (.dim n .scalar)))
  (start end_p : Nat) (h : end_p ≤ n) :
  Tensor α (.dim m (.dim (end_p - start) .scalar)) :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      let col_idx := start + j.val
      if h_col : col_idx < end_p then
        match get matrix i with
        | Tensor.dim row => row ⟨col_idx, Nat.lt_of_lt_of_le h_col h⟩
      else
        Tensor.scalar (0 : α)
    )
  )

-- Slice values from start to end
/--
Slice a contiguous subvector from `values`, returning length `end_p - start`.

Entries outside the `[start, end_p)` range are filled with `0`.
-/
def sliceValuesSpec {n : Nat}
  (values : Tensor α (.dim n .scalar))
  (start end_p : Nat) (h : end_p ≤ n) :
  Tensor α (.dim (end_p - start) .scalar) :=
  Tensor.dim (fun i =>
    let idx := start + i.val
    if h_idx : idx < end_p then
      get values ⟨idx, Nat.lt_of_lt_of_le h_idx h⟩
    else
      Tensor.scalar (0 : α)
  )

-- Orient components (ensure deterministic sign)
/--
Orient component vectors to have a deterministic sign.

Many decompositions (PCA, ICA, eigenvectors) are sign-ambiguous: both `v` and `-v` are valid.
This helper flips each component so its first entry is nonnegative, making the result stable for
display/comparison in small specs.
-/
def orientComponentsSpec {m n : Nat}
  (components : Tensor α (.dim m (.dim n .scalar))) (h : 0 < n) :
  Tensor α (.dim m (.dim n .scalar)) :=
  Tensor.dim (fun i =>
    match get components i with
    | Tensor.dim component =>
      -- check first element
      match component (Fin.mk 0 h) with
      | Tensor.scalar first_val =>
        let sign := if first_val < 0 then -1 else 1
        Tensor.dim (fun j =>
          match component j with
          | Tensor.scalar val => Tensor.scalar (sign * val)
        )
  )
end Spec
