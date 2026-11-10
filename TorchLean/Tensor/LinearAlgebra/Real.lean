/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Reduce.Real
public import TorchLean.Tensor.Contract

/-!
# Linear algebra for dependent tensors

Over `ℝ`, matrix dot products are finite sums, matrix-vector products associate
(`contract_matVec_assoc`) and the transpose reverses products (`transpose_matmul`). Under the
Frobenius inner product, multiplying by a matrix on either side is adjoint to multiplying by its
transpose on that side (`dot_matmul_right_adjoint`, `dot_matmul_left_adjoint`). Autograd,
runtime-approximation and model proofs use these facts. The entry rules `get2_identity`, `get2_add`,
`get2_scale` and `get2_sub` hold over any scalar type with the operations involved.

`Rep.matrixEquiv` reads a rank-two tensor as a Mathlib `Matrix`. Matrix-vector
associativity, the transpose of a transpose and of a product, the identity entries and right
multiplication by the identity follow from Mathlib's matrix identities through it.
-/

@[expose] public section

namespace TorchLean.Tensor.Rep

open Shape (Coord)

universe u

/--
The canonical equivalence between a rank-two coordinate tensor and a mathlib
matrix with the same row and column index types.
-/
def matrixEquiv (α : Type u) [Storage α] (rows columns : Nat) :
    Rep α [rows, columns] ≃ Matrix (Fin rows) (Fin columns) α where
  toFun tensor row column :=
    tensor (row, column, PUnit.unit)
  invFun matrix :=
    Rep.ofFn fun coordinate =>
      matrix coordinate.1 coordinate.2.1
  left_inv tensor := by
    ext coordinate
    rcases coordinate with ⟨row, column, scalarCoordinate⟩
    cases scalarCoordinate
    exact Rep.get_ofFn
      (fun coordinate : Coord [rows, columns] =>
        tensor (coordinate.1, coordinate.2.1, PUnit.unit))
      (row, column, PUnit.unit)
  right_inv matrix := by
    ext row column
    exact Rep.get_ofFn
      (fun coordinate : Coord [rows, columns] =>
        matrix coordinate.1 coordinate.2.1)
      (row, column, PUnit.unit)

variable {α : Type u} [Storage α] {rows columns : Nat}

/-- The matrix view reads the same row and column entry as the tensor view. -/
@[simp] theorem matrixEquiv_apply (tensor : Rep α [rows, columns])
    (row : Fin rows) (column : Fin columns) :
    matrixEquiv α rows columns tensor row column =
      tensor (row, column, PUnit.unit) :=
  rfl

/-- The inverse matrix view rebuilds the corresponding rank-two tensor entry. -/
@[simp] theorem matrixEquiv_symm_apply
    (matrix : Matrix (Fin rows) (Fin columns) α)
    (coordinate : Coord [rows, columns]) :
    (matrixEquiv α rows columns).symm matrix coordinate =
      matrix coordinate.1 coordinate.2.1 :=
  Rep.get_ofFn _ _

end TorchLean.Tensor.Rep

namespace TorchLean.Tensor

open scoped BigOperators
open Rep (dot)

/-- `Tensor.sum` on a 1D tensor equals the `Finset` sum of its coordinates (`getScalar`). -/
theorem sum_vec {n : Nat} (v : Tensor ℝ [n]) :
  Tensor.sum v = ∑ i : Fin n, getScalar v i := by
  classical
  rw [sum_dim]
  apply Finset.sum_congr rfl
  intro i _
  rw [sum_eq_coord_sum]
  simp [getScalar_eq_apply]

/-- `getScalar` of `*` is pointwise multiplication of coordinate functions. -/
theorem getScalar_mul {n : Nat} (a b : Tensor ℝ [n]) (i : Fin n) :
  getScalar (a * b) i = getScalar a i * getScalar b i := by
  simp [getScalar_eq_apply, mul_def]

/-! ## Mathlib matrix view

`matrixEquiv_matmul`, `matrixEquiv_permute_adjacentSwap`, `getScalar_contract_matVec_eq_mulVec` and
`matrixEquiv_identity` carry the matrix product, transpose, matrix-vector product and identity
across `Rep.matrixEquiv`. `contract_matVec_assoc`, `transpose_transpose`,
`transpose_matmul`, `get2_identity` and `matmul_identity` are proved from the Mathlib matrix
identities through these rules.
-/

/-- `Tensor.matmul` is the Mathlib matrix product under `Rep.matrixEquiv`. -/
theorem matrixEquiv_matmul {m n p : Nat} (A : Tensor ℝ [m, n]) (B : Tensor ℝ [n, p]) :
    Rep.matrixEquiv ℝ m p (Tensor.matmul A B) =
      Rep.matrixEquiv ℝ m n A * Rep.matrixEquiv ℝ n p B := by
  ext i j
  simpa [Matrix.mul_apply, get2_eq_apply] using get2_matmul_eq_sum A B i j

/-- The matrix transpose `permute A (Shape.adjacentSwap [m, n] 0) [n, m]` is the Mathlib
transpose under `Rep.matrixEquiv`. -/
theorem matrixEquiv_permute_adjacentSwap {m n : Nat} (A : Tensor ℝ [m, n]) :
    Rep.matrixEquiv ℝ n m (Tensor.permute A (Shape.adjacentSwap [m, n] 0) [n, m]) =
      Matrix.transpose (Rep.matrixEquiv ℝ m n A) := by
  ext i j
  simp only [Rep.matrixEquiv_apply, Matrix.transpose_apply]
  exact permute_adjacentSwap_zero_apply A _ (i, j, PUnit.unit)

/-- `Tensor.contract` with `.matVec` is `Matrix.mulVec` of the matrix view, read through
`getScalar`. -/
theorem getScalar_contract_matVec_eq_mulVec {m n : Nat} (A : Tensor ℝ [m, n])
    (x : Tensor ℝ [n]) :
    getScalar (Tensor.contract A x _ _ [m] .matVec) =
      Matrix.mulVec (Rep.matrixEquiv ℝ m n A) (getScalar x) := by
  funext i
  simp [getScalar_contract_matVec_eq_sum, Matrix.mulVec, dotProduct,
    get2_eq_apply]

/-- `Tensor.identity n` is the Mathlib identity matrix under `Rep.matrixEquiv`. -/
theorem matrixEquiv_identity {α : Type} [Storage α] [Zero α] [One α] (n : Nat) :
    Rep.matrixEquiv α n n (Tensor.identity (α := α) n) = 1 := by
  ext i j
  simp [Tensor.identity, Matrix.one_apply, Fin.ext_iff]

/-- `shape t` is the shape `s` tracked in the type of `t : Tensor α s`. -/
theorem shape_eq {α : Type} [Storage α]
    {s : Shape} (t : Tensor α s) :
  shape t = s := by
  rfl

/-! ## Matrix and vector algebra -/

/-- Associativity of matrix-vector multiplication: `A (B x) = (A B) x`. -/
theorem contract_matVec_assoc {m n p : Nat}
  (A : Tensor ℝ [m, n])
  (B : Tensor ℝ [n, p])
  (x : Tensor ℝ [p]) :
  Tensor.contract A (Tensor.contract B x _ _ [n] .matVec) _ _ [m] .matVec =
  Tensor.contract (Tensor.matmul A B) x _ _ [m] .matVec := by
  apply Tensor.ext_vector
  intro i
  simp only [getScalar_contract_matVec_eq_mulVec, matrixEquiv_matmul, Matrix.mulVec_mulVec]

/-- Coordinate rule for the matrix transpose `permute A (Shape.adjacentSwap [m, n] 0) [n, m]`:
`(Aᵀ)[i,j] = A[j,i]`. -/
theorem get2_transpose {m n : Nat}
  (A : Tensor ℝ [m, n]) (i : Fin n) (j : Fin m) :
  get2 (Tensor.permute A (Shape.adjacentSwap [m, n] 0) [n, m]) i j = get2 A j i := by
  rw [permute_adjacentSwap_zero]
  simp [get2, Tensor.getScalar]

/-- Matrix extensionality: matrices are equal when all their entries are equal. -/
theorem ext_get2 {m n : Nat} {A B : Tensor ℝ [m, n]} :
  (∀ i : Fin m, ∀ j : Fin n, get2 A i j = get2 B i j) → A = B := by
  intro h
  apply Tensor.Rep.ext
  intro coordinate
  rcases coordinate with ⟨i, j, ⟨⟩⟩
  simpa [get2, Tensor.getScalar, Tensor.item] using h i j

/-- Matrix transpose is an involution. -/
theorem transpose_transpose {m n : Nat}
    (A : Tensor ℝ [m, n]) :
    Tensor.permute (Tensor.permute A (Shape.adjacentSwap [m, n] 0) [n, m])
      (Shape.adjacentSwap [n, m] 0) [m, n] = A := by
  apply (Rep.matrixEquiv ℝ m n).injective
  rw [matrixEquiv_permute_adjacentSwap, matrixEquiv_permute_adjacentSwap,
    Matrix.transpose_transpose]

/-- Transpose of a product: `(A ⬝ B)ᵀ = Bᵀ ⬝ Aᵀ`. -/
theorem transpose_matmul {m n p : Nat}
  (A : Tensor ℝ [m, n])
  (B : Tensor ℝ [n, p]) :
  Tensor.permute (Tensor.matmul A B) (Shape.adjacentSwap [m, p] 0) [p, m] =
  Tensor.matmul (Tensor.permute B (Shape.adjacentSwap [n, p] 0) [p, n])
    (Tensor.permute A (Shape.adjacentSwap [m, n] 0) [n, m]) := by
  apply (Rep.matrixEquiv ℝ p m).injective
  simp only [matrixEquiv_permute_adjacentSwap, matrixEquiv_matmul, Matrix.transpose_mul]

/-! ## Frobenius dot and matmul adjointness -/

/-- The `dot` of two matrices is the Frobenius inner product `∑ i, ∑ j, A i j * B i j`. -/
theorem dot_mat_eq_sum {m n : Nat}
  (A B : Tensor ℝ [m, n]) :
  Rep.dot A B = ∑ i : Fin m, ∑ j : Fin n, (get2 A i j) * (get2 B i j) := by
  rw [dot_eq_sum_unstack]
  apply Finset.sum_congr rfl
  intro i _
  rw [dot_vec_eq_sum]
  rfl

/-- Right adjointness of matrix multiplication under the Frobenius `dot`:
`⟪A ⬝ B, C⟫ = ⟪A, C ⬝ Bᵀ⟫`. -/
theorem dot_matmul_right_adjoint
  {m n p : Nat}
  (A : Tensor ℝ [m, n])
  (B : Tensor ℝ [n, p])
  (C : Tensor ℝ [m, p]) :
  Rep.dot (Tensor.matmul A B) C =
    Rep.dot A (Tensor.matmul C (Tensor.permute B (Shape.adjacentSwap [n, p] 0) [p, n])) := by
  classical
  -- Interchange the two inner indices of the finite triple sum.
  -- LHS: ∑ i ∑ j (∑ k Aik*Bkj) * Cij
  -- RHS: ∑ i ∑ k Aik * (∑ j Cij*Bkj)
  rw [dot_mat_eq_sum (A := Tensor.matmul A B) (B := C)]
  rw [dot_mat_eq_sum (A := A)
    (B := Tensor.matmul C (Tensor.permute B (Shape.adjacentSwap [n, p] 0) [p, n]))]
  simp [get2_matmul_eq_sum, get2_transpose, Finset.mul_sum, Finset.sum_mul]
  refine Finset.sum_congr rfl ?_
  intro i _
  simpa [mul_assoc, mul_left_comm, mul_comm] using
    (Finset.sum_comm (s := (Finset.univ : Finset (Fin p))) (t := (Finset.univ : Finset (Fin n)))
      (f := fun j k => get2 A i k * (get2 C i j * get2 B k j)))

/-- Transpose invariance of the Frobenius dot-product: `⟪Aᵀ, Bᵀ⟫ = ⟪A, B⟫`. -/
theorem dot_transpose {m n : Nat}
  (A B : Tensor ℝ [m, n]) :
  Rep.dot (Tensor.permute A (Shape.adjacentSwap [m, n] 0) [n, m])
    (Tensor.permute B (Shape.adjacentSwap [m, n] 0) [n, m]) = Rep.dot A B := by
  classical
  calc
    Rep.dot (Tensor.permute A (Shape.adjacentSwap [m, n] 0) [n, m])
        (Tensor.permute B (Shape.adjacentSwap [m, n] 0) [n, m]) =
        ∑ i : Fin n, ∑ j : Fin m,
          get2 (Tensor.permute A (Shape.adjacentSwap [m, n] 0) [n, m]) i j *
            get2 (Tensor.permute B (Shape.adjacentSwap [m, n] 0) [n, m]) i j :=
      dot_mat_eq_sum _ _
    _ = ∑ i : Fin n, ∑ j : Fin m, get2 A j i * get2 B j i := by
      simp [get2_transpose]
    _ = ∑ j : Fin m, ∑ i : Fin n, get2 A j i * get2 B j i := Finset.sum_comm
    _ = Rep.dot A B := (dot_mat_eq_sum A B).symm

/-- Left adjointness of matrix multiplication under the Frobenius `dot`:
`⟪A ⬝ B, C⟫ = ⟪B, Aᵀ ⬝ C⟫`. -/
theorem dot_matmul_left_adjoint
  {m n p : Nat}
  (A : Tensor ℝ [m, n])
  (B : Tensor ℝ [n, p])
  (C : Tensor ℝ [m, p]) :
  Rep.dot (Tensor.matmul A B) C =
    Rep.dot B (Tensor.matmul (Tensor.permute A (Shape.adjacentSwap [m, n] 0) [n, m]) C) := by
  classical
  -- Transpose, apply the right-adjoint lemma, and transpose back:
  -- ⟪A·B, C⟫ = ⟪(A·B)ᵀ, Cᵀ⟫ = ⟪Bᵀ·Aᵀ, Cᵀ⟫ = ⟪Bᵀ, Cᵀ·A⟫ = ⟪B, (Cᵀ·A)ᵀ⟫ = ⟪B, Aᵀ·C⟫.
  have htrans :=
    (dot_transpose (m := m) (n := p) (A := Tensor.matmul A B) (B := C)).symm
  have hmulT :
      Tensor.permute (Tensor.matmul A B) (Shape.adjacentSwap [m, p] 0) [p, m] =
        Tensor.matmul (Tensor.permute B (Shape.adjacentSwap [n, p] 0) [p, n])
          (Tensor.permute A (Shape.adjacentSwap [m, n] 0) [n, m]) :=
    transpose_matmul (A := A) (B := B)
  have hadj :
      Rep.dot (Tensor.matmul (Tensor.permute B (Shape.adjacentSwap [n, p] 0) [p, n])
          (Tensor.permute A (Shape.adjacentSwap [m, n] 0) [n, m]))
          (Tensor.permute C (Shape.adjacentSwap [m, p] 0) [p, m])
        =
      Rep.dot (Tensor.permute B (Shape.adjacentSwap [n, p] 0) [p, n])
        (Tensor.matmul (Tensor.permute C (Shape.adjacentSwap [m, p] 0) [p, m])
          (Tensor.permute (Tensor.permute A (Shape.adjacentSwap [m, n] 0) [n, m])
            (Shape.adjacentSwap [n, m] 0) [m, n]))
          := by
    simpa using
      (dot_matmul_right_adjoint (A := Tensor.permute B (Shape.adjacentSwap [n, p] 0) [p, n])
        (B := Tensor.permute A (Shape.adjacentSwap [m, n] 0) [n, m])
        (C := Tensor.permute C (Shape.adjacentSwap [m, p] 0) [p, m]))
  have hAinv : Tensor.permute (Tensor.permute A (Shape.adjacentSwap [m, n] 0) [n, m])
      (Shape.adjacentSwap [n, m] 0) [m, n] = A :=
    transpose_transpose (A := A)
  have hCinv : Tensor.permute (Tensor.permute C (Shape.adjacentSwap [m, p] 0) [p, m])
      (Shape.adjacentSwap [p, m] 0) [m, p] = C :=
    transpose_transpose (A := C)
  -- `dot (Bᵀ) D = dot B (Dᵀ)` for matching shapes.
  have hdot_swap :
      Rep.dot (Tensor.permute B (Shape.adjacentSwap [n, p] 0) [p, n])
          (Tensor.matmul (Tensor.permute C (Shape.adjacentSwap [m, p] 0) [p, m]) A)
        =
      Rep.dot B (Tensor.permute
          (Tensor.matmul (Tensor.permute C (Shape.adjacentSwap [m, p] 0) [p, m]) A)
          (Shape.adjacentSwap [p, n] 0) [n, p]) := by
    have := dot_transpose (m := n) (n := p)
      (A := B) (B := Tensor.permute
        (Tensor.matmul (Tensor.permute C (Shape.adjacentSwap [m, p] 0) [p, m]) A)
        (Shape.adjacentSwap [p, n] 0) [n, p])
    simpa [transpose_transpose, hCinv] using this
  calc
    Rep.dot (Tensor.matmul A B) C
        = Rep.dot (Tensor.permute (Tensor.matmul A B) (Shape.adjacentSwap [m, p] 0) [p, m])
            (Tensor.permute C (Shape.adjacentSwap [m, p] 0) [p, m]) := htrans
    _ = Rep.dot (Tensor.matmul (Tensor.permute B (Shape.adjacentSwap [n, p] 0) [p, n])
          (Tensor.permute A (Shape.adjacentSwap [m, n] 0) [n, m]))
        (Tensor.permute C (Shape.adjacentSwap [m, p] 0) [p, m]) := by
          simp [hmulT]
    _ = Rep.dot (Tensor.permute B (Shape.adjacentSwap [n, p] 0) [p, n])
          (Tensor.matmul (Tensor.permute C (Shape.adjacentSwap [m, p] 0) [p, m]) A) := by
          simpa [hAinv] using hadj
    _ = Rep.dot B (Tensor.permute
          (Tensor.matmul (Tensor.permute C (Shape.adjacentSwap [m, p] 0) [p, m]) A)
          (Shape.adjacentSwap [p, n] 0) [n, p]) := hdot_swap
    _ = Rep.dot B (Tensor.matmul (Tensor.permute A (Shape.adjacentSwap [m, n] 0) [n, m]) C) := by
          simp [transpose_matmul, hCinv]

/-! ## Entry rules for matrix-shaped tensor operations -/

section MatrixEntries

variable {α : Type} [Storage α]

/-- Entries of the identity matrix, in the `get2` form the certificate proofs consume. -/
theorem get2_identity [Zero α] [One α] {n : Nat} (i j : Fin n) :
    get2 (Tensor.identity (α := α) n) i j = if i = j then 1 else 0 := by
  rw [get2_eq_apply, ← Rep.matrixEquiv_apply, matrixEquiv_identity, Matrix.one_apply]

/-- Entry rule for matrix-shaped tensor addition. -/
theorem get2_add [Add α] {m n : Nat} (A B : Tensor α [m, n]) (i : Fin m) (j : Fin n) :
    get2 (A + B) i j = get2 A i j + get2 B i j := by
  simp [add_def]

/-- Entry rule for matrix-shaped tensor scaling. -/
theorem get2_scale [Mul α] {m n : Nat} (A : Tensor α [m, n]) (c : α) (i : Fin m) (j : Fin n) :
    get2 (Tensor.scale A c) i j = get2 A i j * c := by
  simp [Tensor.scale]

/-- Entry rule for matrix-shaped tensor subtraction. -/
theorem get2_sub [Sub α] {m n : Nat} (A B : Tensor α [m, n]) (i : Fin m) (j : Fin n) :
    get2 (A - B) i j = get2 A i j - get2 B i j := by
  simp [sub_def]

end MatrixEntries

/-! ## Real matrix identities -/

/-- Right multiplication by the identity matrix leaves a real matrix unchanged. -/
theorem matmul_identity {m n : Nat} (A : Tensor ℝ [m, n]) :
    Tensor.matmul A (Tensor.identity (α := ℝ) n) = A := by
  apply (Rep.matrixEquiv ℝ m n).injective
  rw [matrixEquiv_matmul, matrixEquiv_identity, Matrix.mul_one]

/-- Scaling a real matrix by `1` returns the matrix. -/
theorem scale_one {m n : Nat} (Q : Tensor ℝ [m, n]) : Tensor.scale Q 1 = Q := by
  apply ext_get2
  intro i j
  rw [get2_scale, mul_one]

end TorchLean.Tensor
