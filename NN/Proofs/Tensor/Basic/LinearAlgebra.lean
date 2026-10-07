/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Basic.Folds

/-!
Linear-algebra facts for dependent tensors.

The results here cover dot products, matrix-vector structure, and linearity facts used by
autograd, runtime approximation, and model proofs.
-/

@[expose] public section

open TorchLean

namespace Spec

open TorchLean TorchLean.Tensor
open scoped BigOperators

/-- `sumSpec` on a 1D tensor equals the `Finset` sum of its coordinates (`getScalar`). -/
theorem sum_spec_vec {n : Nat} (v : Tensor ℝ [n]) :
  sumSpec v = ∑ i : Fin n, getScalar v i := by
  classical
  rw [sum_spec_dim]
  apply Finset.sum_congr rfl
  intro i _
  rw [sum_spec_eq_coord_sum]
  simp [getScalar_eq_apply, get, Tensor.unstack]

/-- `getScalar` of `mulSpec` is pointwise multiplication of coordinate functions. -/
theorem getScalar_mul_spec {n : Nat} (a b : Tensor ℝ [n]) (i : Fin n) :
  getScalar (mulSpec a b) i = getScalar a i * getScalar b i := by
  simp [getScalar_eq_apply, mulSpec, map2Spec]

/-- Dot product of vectors is the coordinate-wise sum `∑ i, a[i] * b[i]`. -/
theorem dot_vec_eq_sum {n : Nat} (a b : Tensor ℝ [n]) :
  dot a b = ∑ i : Fin n, getScalar a i * getScalar b i := by
  calc
    dot a b = Proofs.TensorAlgebra.dot (α := ℝ) a b := by
      exact dot_eq_tensorAlgebra_dot (a := a) (b := b)
    _ = ∑ i : Fin n, getScalar a i * getScalar b i := by
      simpa using Proofs.TensorAlgebra.dot_vec_eq_sum (α := ℝ) (a := a) (b := b)

/--
Adjointness of matrix-vector and vector-matrix multiplication under the `dot` product:
`⟪y, W x⟫ = ⟪y W, x⟫` (a.k.a. `⟪y, W x⟫ = ⟪Wᵀ y, x⟫` depending on conventions).

This is the algebraic heart of the linear-layer gradient rule.
-/
theorem dot_mat_linear_adjoint
  {inDim outDim : Nat}
  (W : Tensor ℝ [outDim, inDim])
  (dLdy : Tensor ℝ [outDim])
  (dx : Tensor ℝ [inDim]) :
  dot dLdy (matVecMulSpec W dx)
  = dot (vecMatMulSpec dLdy W) dx := by
  calc
    dot dLdy (matVecMulSpec W dx)
        = Proofs.TensorAlgebra.dot (α := ℝ) dLdy (matVecMulSpec W dx) := by
          exact dot_eq_tensorAlgebra_dot (a := dLdy) (b := matVecMulSpec W dx)
    _ = Proofs.TensorAlgebra.dot (α := ℝ) (vecMatMulSpec dLdy W) dx := by
          exact Proofs.TensorAlgebra.dot_mat_linear_adjoint (α := ℝ) (W := W) (dLdy := dLdy)
            (dx := dx)
    _ = dot (vecMatMulSpec dLdy W) dx := by
          exact (dot_eq_tensorAlgebra_dot (a := vecMatMulSpec dLdy W) (b := dx)).symm

/--
`shapeOf` recovers the shape already tracked in the tensor type.

This is a small bridge for proofs that move between value-level shape computations and type-indexed
tensor operations.
-/
theorem shapeOf_eq_shape {α : Type} [TorchLean.Storage α]
    {s : Shape} (t : Tensor α s) :
  shapeOf t = s := by
  rfl

/-! ## Matrix and vector algebra -/

/-- Associativity of matrix-vector multiplication: `A (B x) = (A B) x`. -/
theorem mat_vec_assoc {m n p : Nat}
  (A : Tensor ℝ [m, n])
  (B : Tensor ℝ [n, p])
  (x : Tensor ℝ [p]) :
  matVecMulSpec A (matVecMulSpec B x) =
  matVecMulSpec (matMulSpec A B) x := by
  apply Tensor.ext_vector
  intro i
  -- Both coordinates are the double sum `∑ k, ∑ j, A i k * (B k j * x j)`; only the order of
  -- summation differs.
  simp only [getScalar_mat_vec_mul_spec, get2_mat_mul_spec, Finset.mul_sum, Finset.sum_mul,
    mul_assoc]
  exact Finset.sum_comm

/-- Coordinate rule for the matrix transpose `swapAdjacentAxes A 0`: `(Aᵀ)[i,j] = A[j,i]`. -/
theorem get2_matrix_transpose_spec {m n : Nat}
  (A : Tensor ℝ [m, n]) (i : Fin n) (j : Fin m) :
  get2 (swapAdjacentAxes A 0) i j = get2 A j i := by
  rw [swapAdjacentAxes_zero]
  simp [get2, get, Tensor.getScalar]

/-- Matrix extensionality: matrices are equal when all their entries are equal. -/
theorem matrix_ext {m n : Nat} {A B : Tensor ℝ [m, n]} :
  (∀ i : Fin m, ∀ j : Fin n, get2 A i j = get2 B i j) → A = B := by
  intro h
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  rcases coordinate with ⟨i, j, ⟨⟩⟩
  simpa [get2, get, Tensor.getScalar, Tensor.unstack, Tensor.item] using h i j

/-- Matrix transpose is an involution. -/
theorem matrix_transpose_involution {m n : Nat}
    (A : Tensor ℝ [m, n]) :
    swapAdjacentAxes (swapAdjacentAxes A 0) 0 = A := by
  apply matrix_ext
  intro i j
  rw [get2_matrix_transpose_spec, get2_matrix_transpose_spec]

/-- Transpose of a product: `(A ⬝ B)ᵀ = Bᵀ ⬝ Aᵀ`. -/
theorem matrix_transpose_mul {m n p : Nat}
  (A : Tensor ℝ [m, n])
  (B : Tensor ℝ [n, p]) :
  swapAdjacentAxes (matMulSpec A B) 0 =
  matMulSpec (swapAdjacentAxes B 0) (swapAdjacentAxes A 0) := by
  classical
  apply matrix_ext
  intro j i
  calc
    get2 (swapAdjacentAxes (matMulSpec A B) 0) j i
        = get2 (matMulSpec A B) i j := by
            simpa using (get2_matrix_transpose_spec (A := matMulSpec A B) (i := j) (j := i))
    _ = ∑ k : Fin n, (get2 A i k) * (get2 B k j) := by
          simpa using (get2_mat_mul_spec (A := A) (B := B) (i := i) (j := j))
    _ = ∑ k : Fin n, (get2 B k j) * (get2 A i k) := by
          refine Finset.sum_congr rfl ?_
          intro k _
          simp [mul_comm]
    _ = ∑ k : Fin n, (get2 (swapAdjacentAxes B 0) j k) * (get2 (swapAdjacentAxes A 0) k i) :=
      by
          refine Finset.sum_congr rfl ?_
          intro k _
          simp [get2_matrix_transpose_spec]
    _ = get2 (matMulSpec (swapAdjacentAxes B 0) (swapAdjacentAxes A 0)) j i := by
          symm
          simpa using
            (get2_mat_mul_spec (A := swapAdjacentAxes B 0) (B := swapAdjacentAxes A 0) (i :=
              j) (j := i))

/-! ## Frobenius dot and matmul adjointness -/

/-- Expand the matrix dot-product as a double sum over entries (Frobenius inner product). -/
theorem dot_mat_eq_sum {m n : Nat}
  (A B : Tensor ℝ [m, n]) :
  dot A B = ∑ i : Fin m, ∑ j : Fin n, (get2 A i j) * (get2 B i j) := by
  classical
  rw [dot, sum_spec_dim]
  apply Finset.sum_congr rfl
  intro i _
  rw [sum_spec_vec]
  apply Finset.sum_congr rfl
  intro j _
  have hRow :
      get (mulSpec A B) i = mulSpec (get A i) (get B i) := by
    simp [mulSpec]
  rw [hRow, getScalar_mul_spec]
  rfl

/-- Right-adjointness of matrix multiplication under the Frobenius dot-product.

Informally: `⟪A ⬝ B, C⟫ = ⟪A, C ⬝ Bᵀ⟫`.
-/
theorem dot_mat_mul_right_adjoint
  {m n p : Nat}
  (A : Tensor ℝ [m, n])
  (B : Tensor ℝ [n, p])
  (C : Tensor ℝ [m, p]) :
  dot (matMulSpec A B) C = dot A (matMulSpec C (swapAdjacentAxes B 0)) := by
  classical
  -- Interchange the two inner indices of the finite triple sum.
  -- LHS: ∑ i ∑ j (∑ k Aik*Bkj) * Cij
  -- RHS: ∑ i ∑ k Aik * (∑ j Cij*Bkj)
  rw [dot_mat_eq_sum (A := matMulSpec A B) (B := C)]
  rw [dot_mat_eq_sum (A := A) (B := matMulSpec C (swapAdjacentAxes B 0))]
  simp [get2_mat_mul_spec, get2_matrix_transpose_spec, Finset.mul_sum, Finset.sum_mul]
  refine Finset.sum_congr rfl ?_
  intro i _
  simpa [mul_assoc, mul_left_comm, mul_comm] using
    (Finset.sum_comm (s := (Finset.univ : Finset (Fin p))) (t := (Finset.univ : Finset (Fin n)))
      (f := fun j k => get2 A i k * (get2 C i j * get2 B k j)))

/-- Transpose invariance of the Frobenius dot-product: `⟪Aᵀ, Bᵀ⟫ = ⟪A, B⟫`. -/
theorem dot_mat_transpose {m n : Nat}
  (A B : Tensor ℝ [m, n]) :
  dot (swapAdjacentAxes A 0) (swapAdjacentAxes B 0) = dot A B := by
  classical
  calc
    dot (swapAdjacentAxes A 0) (swapAdjacentAxes B 0) =
        ∑ i : Fin n, ∑ j : Fin m,
          get2 (swapAdjacentAxes A 0) i j * get2 (swapAdjacentAxes B 0) i j :=
      dot_mat_eq_sum _ _
    _ = ∑ i : Fin n, ∑ j : Fin m, get2 A j i * get2 B j i := by
      simp [get2_matrix_transpose_spec]
    _ = ∑ j : Fin m, ∑ i : Fin n, get2 A j i * get2 B j i := Finset.sum_comm
    _ = dot A B := (dot_mat_eq_sum A B).symm

/-- Left-adjointness of matrix multiplication under the Frobenius dot-product.

Informally: `⟪A ⬝ B, C⟫ = ⟪B, Aᵀ ⬝ C⟫`.
-/
theorem dot_mat_mul_left_adjoint
  {m n p : Nat}
  (A : Tensor ℝ [m, n])
  (B : Tensor ℝ [n, p])
  (C : Tensor ℝ [m, p]) :
  dot (matMulSpec A B) C = dot B (matMulSpec (swapAdjacentAxes A 0) C) := by
  classical
  -- Reduce to the right-adjoint lemma via transpose.
  -- ⟪A·B, C⟫ = ⟪(A·B)ᵀ, Cᵀ⟫ = ⟪Bᵀ·Aᵀ, Cᵀ⟫ = ⟪B, (Cᵀ·A)ᵀ⟫ = ⟪B, Aᵀ·C⟫.
  have htrans :=
    (dot_mat_transpose (m := m) (n := p) (A := matMulSpec A B) (B := C)).symm
  have hmulT :
      swapAdjacentAxes (matMulSpec A B) 0 =
        matMulSpec (swapAdjacentAxes B 0) (swapAdjacentAxes A 0) :=
    matrix_transpose_mul (A := A) (B := B)
  have hadj :
      dot (matMulSpec (swapAdjacentAxes B 0) (swapAdjacentAxes A 0)) (swapAdjacentAxes
        C 0)
        =
      dot (swapAdjacentAxes B 0)
        (matMulSpec (swapAdjacentAxes C 0) (swapAdjacentAxes (swapAdjacentAxes A 0) 0))
          := by
    simpa using
      (dot_mat_mul_right_adjoint (A := swapAdjacentAxes B 0) (B := swapAdjacentAxes A 0)
        (C := swapAdjacentAxes C 0))
  have hAinv : swapAdjacentAxes (swapAdjacentAxes A 0) 0 = A :=
    matrix_transpose_involution (A := A)
  have hCinv : swapAdjacentAxes (swapAdjacentAxes C 0) 0 = C :=
    matrix_transpose_involution (A := C)
  -- `dot (Bᵀ) D = dot B (Dᵀ)` for matching shapes.
  have hdot_swap :
      dot (swapAdjacentAxes B 0) (matMulSpec (swapAdjacentAxes C 0) A)
        =
      dot B (swapAdjacentAxes (matMulSpec (swapAdjacentAxes C 0) A) 0) := by
    have := dot_mat_transpose (m := n) (n := p)
      (A := B) (B := swapAdjacentAxes (matMulSpec (swapAdjacentAxes C 0) A) 0)
    simpa [matrix_transpose_involution, hCinv] using this
  calc
    dot (matMulSpec A B) C
        = dot (swapAdjacentAxes (matMulSpec A B) 0) (swapAdjacentAxes C 0) := htrans
    _ = dot (matMulSpec (swapAdjacentAxes B 0) (swapAdjacentAxes A 0))
      (swapAdjacentAxes C 0) := by
          simp [hmulT]
    _ = dot (swapAdjacentAxes B 0) (matMulSpec (swapAdjacentAxes C 0) A) := by
          simpa [hAinv] using hadj
    _ = dot B (swapAdjacentAxes (matMulSpec (swapAdjacentAxes C 0) A) 0) := hdot_swap
    _ = dot B (matMulSpec (swapAdjacentAxes A 0) C) := by
          simp [matrix_transpose_mul, hCinv]

/-! ## Entry rules for matrix-shaped tensor operations -/

section MatrixEntries

variable {α : Type} [Storage α]

/-- Entries of the identity matrix, in the `get2` form the certificate proofs consume. -/
theorem get2_identityTensorSpec [Zero α] [One α] {n : Nat} (i j : Fin n) :
    get2 (identityTensorSpec (α := α) n) i j = if i = j then 1 else 0 := by
  by_cases h : i = j
  · subst j
    simp [identityTensorSpec, get2, Tensor.getScalar, Spec.get, Tensor.unstack,
      Tensor.item]
  · have hval : i.val ≠ j.val := fun hval => h (Fin.ext hval)
    simp [identityTensorSpec, get2, Tensor.getScalar, Spec.get, Tensor.unstack,
      Tensor.item, h, hval]

/-- Entry rule for matrix-shaped tensor addition. -/
theorem get2_addSpec [Add α] {m n : Nat} (A B : Tensor α [m, n]) (i : Fin m) (j : Fin n) :
    get2 (addSpec A B) i j = get2 A i j + get2 B i j := by
  simp [addSpec]

/-- Entry rule for matrix-shaped tensor scaling. -/
theorem get2_scaleSpec [Mul α] {m n : Nat} (A : Tensor α [m, n]) (c : α) (i : Fin m) (j : Fin n) :
    get2 (scaleSpec A c) i j = get2 A i j * c := by
  simp [scaleSpec]

/-- Entry rule for matrix-shaped tensor subtraction. -/
theorem get2_subSpec [Sub α] {m n : Nat} (A B : Tensor α [m, n]) (i : Fin m) (j : Fin n) :
    get2 (subSpec A B) i j = get2 A i j - get2 B i j := by
  simp [subSpec]

end MatrixEntries

/-! ## Real matrix identities -/

/-- Right multiplication by the identity matrix leaves a real matrix unchanged. -/
theorem matMulSpec_identityTensorSpec {m n : Nat} (A : Tensor ℝ [m, n]) :
    matMulSpec A (identityTensorSpec (α := ℝ) n) = A := by
  classical
  apply matrix_ext
  intro i j
  rw [get2_mat_mul_spec]
  simp [get2_identityTensorSpec]

/-- Scaling a real matrix by `1` returns the matrix. -/
theorem scaleSpec_one {m n : Nat} (Q : Tensor ℝ [m, n]) : scaleSpec Q 1 = Q := by
  apply matrix_ext
  intro i j
  rw [get2_scaleSpec, mul_one]

end Spec
