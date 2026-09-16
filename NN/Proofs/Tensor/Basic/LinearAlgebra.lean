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

-- Pointwise product of vectors under `getScalar`.
/-- `getScalar` of `mulSpec` is pointwise multiplication of coordinate functions. -/
theorem getScalar_mul_spec {n : Nat} (a b : Tensor ℝ [n]) (i : Fin n) :
  getScalar (mulSpec a b) i = getScalar a i * getScalar b i := by
  simp [getScalar_eq_apply, mulSpec, map2Spec]

-- Dot product of vectors as a `Finset` sum over coordinates.
/-- Dot product of vectors is the coordinate-wise sum `∑ i, a[i] * b[i]`. -/
theorem dot_vec_eq_sum {n : Nat} (a b : Tensor ℝ [n]) :
  dot a b = ∑ i : Fin n, getScalar a i * getScalar b i := by
  calc
    dot a b = Proofs.TensorAlgebra.dot (α := ℝ) a b := by
      exact dot_eq_tensorAlgebra_dot (a := a) (b := b)
    _ = ∑ i : Fin n, getScalar a i * getScalar b i := by
      simpa using Proofs.TensorAlgebra.dot_vec_eq_sum (α := ℝ) (a := a) (b := b)

-- Converting the spec-level `List.finRange` fold for `vec_mat_mul_spec` into a `Finset.univ` sum.
/-- Coordinate formula for `vecMatMulSpec` as a `Finset` sum: `(v @ A)[j] = ∑ i, v[i] * A[i,j]`.
  -/
theorem getScalar_vec_mat_mul_spec {m n : Nat}
  (v : Tensor ℝ [m])
  (A : Tensor ℝ [m, n]) (j : Fin n) :
  getScalar (vecMatMulSpec v A) j = ∑ i : Fin m, (getScalar v i) * (get2 A i j) := by
  simpa using
    (Proofs.TensorAlgebra.getScalar_vec_mat_mul_spec (α := ℝ) (v := v) (A := A) (j := j))

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


/-- Indexing the outer dimension of a tensor exposes a subtensor with the declared inner shape. -/
theorem get_preserves_inner_shape {n : Nat} {s : Shape}
  (t : Tensor ℝ (.dim n s)) (i : Fin n) :
  shapeOf (get t i) = s := by
  rfl

/-! ## Map and elementwise operation laws -/

/-- Functor identity law for `mapSpec`: mapping `id` is a no-op. -/
theorem map_spec_id {s : Shape} (t : Tensor ℝ s) :
  mapSpec id t = t := by
  exact TorchLean.Tensor.Internal.Rep.map_id t

/-- Functor law for `mapSpec`: mapping `g` then `f` equals mapping `f ∘ g`. -/
theorem map_spec_comp {s : Shape} (f g : ℝ → ℝ) (t : Tensor ℝ s) :
  mapSpec f (mapSpec g t) = mapSpec (f ∘ g) t := by
  exact TorchLean.Tensor.Internal.Rep.map_map f g t

/-- A scalar additivity law lifts pointwise through `mapSpec` and `addSpec`. -/
theorem map_spec_add_distrib {s : Shape} (f : ℝ → ℝ) (a b : Tensor ℝ s)
  (h : ∀ x y, f (x + y) = f x + f y) :
  mapSpec f (addSpec a b) = addSpec (mapSpec f a) (mapSpec f b) := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp only [mapSpec, Tensor.map, addSpec, map2Spec,
    TorchLean.Tensor.Internal.Rep.map_apply, TorchLean.Tensor.Internal.Rep.zipWith_apply]
  exact h _ _

/-- Commutativity transfer: if `f` is commutative, then `map2_spec f` is commutative on tensors. -/
theorem map2_spec_comm {s : Shape} (f : ℝ → ℝ → ℝ) (a b : Tensor ℝ s)
  (h : ∀ x y, f x y = f y x) :
  map2Spec f a b = map2Spec f b a := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp [map2Spec, h]

/-! ## Matrix and vector algebra -/

/-- Associativity of matrix-vector multiplication: `A (B x) = (A B) x`. -/
theorem mat_vec_assoc {m n p : Nat}
  (A : Tensor ℝ [m, n])
  (B : Tensor ℝ [n, p])
  (x : Tensor ℝ [p]) :
  matVecMulSpec A (matVecMulSpec B x) =
  matVecMulSpec (matMulSpec A B) x := by
  classical
  have hto :
      getScalar (matVecMulSpec A (matVecMulSpec B x)) =
        getScalar (matVecMulSpec (matMulSpec A B) x) := by
    funext i
    have hBx : ∀ k : Fin n,
        getScalar (matVecMulSpec B x) k = ∑ j : Fin p, (get2 B k j) * (getScalar x j) := by
      intro k
      simpa using (getScalar_mat_vec_mul_spec (A := B) (v := x) (i := k))

    -- Expand both sides into finite sums and use a Fubini-style swap.
    have h_expand :
        (∑ k : Fin n, (get2 A i k) * (∑ j : Fin p, (get2 B k j) * (getScalar x j))) =
          (∑ j : Fin p, (∑ k : Fin n, (get2 A i k) * (get2 B k j)) * (getScalar x j)) := by
      -- This is a finite-dimensional distributivity/commutation identity.
      -- We follow the standard pattern: expand, swap sums, factor.
      classical
      -- Expand `get2 A i k * (∑ j, ...)` into a double sum.
      have h1 :
          (∑ k : Fin n, (get2 A i k) * (∑ j : Fin p, (get2 B k j) * (getScalar x j))) =
            (∑ k : Fin n, ∑ j : Fin p, (get2 A i k) * ((get2 B k j) * (getScalar x j))) := by
        simp [Finset.mul_sum]
      -- Swap the order of summation.
      have h2 :
          (∑ k : Fin n, ∑ j : Fin p, (get2 A i k) * ((get2 B k j) * (getScalar x j))) =
            (∑ j : Fin p, ∑ k : Fin n, (get2 A i k) * ((get2 B k j) * (getScalar x j))) := by
        simpa using
          (Finset.sum_comm (s := (Finset.univ : Finset (Fin n))) (t := (Finset.univ : Finset (Fin
            p)))
            (f := fun k j => (get2 A i k) * ((get2 B k j) * (getScalar x j))))
      -- Factor `(getScalar x j)` out of the inner sum.
      have h3 :
          (∑ j : Fin p, ∑ k : Fin n, (get2 A i k) * ((get2 B k j) * (getScalar x j))) =
            (∑ j : Fin p, (∑ k : Fin n, (get2 A i k) * (get2 B k j)) * (getScalar x j)) := by
        refine Finset.sum_congr rfl ?_
        intro j _
        have h_reassoc :
            (∑ k : Fin n, (get2 A i k) * ((get2 B k j) * (getScalar x j))) =
              (∑ k : Fin n, ((get2 A i k) * (get2 B k j)) * (getScalar x j)) := by
          refine Finset.sum_congr rfl ?_
          intro k _
          simpa using (mul_assoc (get2 A i k) (get2 B k j) (getScalar x j)).symm
        have h_pull :
            (∑ k : Fin n, ((get2 A i k) * (get2 B k j)) * (getScalar x j)) =
              (∑ k : Fin n, (get2 A i k) * (get2 B k j)) * (getScalar x j) := by
          simp [Finset.sum_mul]
        exact h_reassoc.trans h_pull

      exact h1.trans (h2.trans h3)

    -- Turn the vector components into the needed sum forms, then apply `h_expand`.
    have lhs :
        getScalar (matVecMulSpec A (matVecMulSpec B x)) i =
          ∑ k : Fin n, (get2 A i k) * (∑ j : Fin p, (get2 B k j) * (getScalar x j)) := by
      -- start from `getScalar_mat_vec_mul_spec` and rewrite each inner component via `hBx`
      have hA :
          getScalar (matVecMulSpec A (matVecMulSpec B x)) i =
            ∑ k : Fin n, (get2 A i k) * (getScalar (matVecMulSpec B x) k) := by
        simpa using (getScalar_mat_vec_mul_spec (A := A) (v := matVecMulSpec B x) (i := i))
      -- rewrite `getScalar (mat_vec_mul_spec B x) k`
      classical
      refine hA.trans ?_
      refine Finset.sum_congr rfl ?_
      intro k _
      simp [hBx k]

    have rhs :
        getScalar (matVecMulSpec (matMulSpec A B) x) i =
          ∑ j : Fin p, (∑ k : Fin n, (get2 A i k) * (get2 B k j)) * (getScalar x j) := by
      -- rewrite the matrix multiplication entry via `get2_mat_mul_spec`
      have hR :
          getScalar (matVecMulSpec (matMulSpec A B) x) i =
            ∑ j : Fin p, (get2 (matMulSpec A B) i j) * (getScalar x j) := by
        simpa using (getScalar_mat_vec_mul_spec (A := matMulSpec A B) (v := x) (i := i))
      -- now rewrite `get2 (mat_mul_spec A B) i j`
      classical
      refine hR.trans ?_
      refine Finset.sum_congr rfl ?_
      intro j _
      simp [get2_mat_mul_spec]

    -- Combine.
    simpa [lhs, rhs] using h_expand

  -- Lift pointwise equality back to tensors via `ofFn`.
  have h := congrArg ofFn hto
  -- `ofFn (getScalar t) = t` for vectors.
  simpa [ofFn_getScalar] using h

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
  -- Prove equality by `get2`-extensionality on matrix entries.
  apply matrix_ext
  intro j i
  -- Compare the `(j,i)` entry of both sides.
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

-- ---------------------------------------------------------------------------
-- Frobenius dot / matmul adjointness
-- ---------------------------------------------------------------------------

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
  -- Expand both sides into entry sums; then it's just rearranging a finite triple sum.
  -- LHS: ∑ i ∑ j (∑ k Aik*Bkj) * Cij
  -- RHS: ∑ i ∑ k Aik * (∑ j Cij*Bkj)
  rw [dot_mat_eq_sum (A := matMulSpec A B) (B := C)]
  rw [dot_mat_eq_sum (A := A) (B := matMulSpec C (swapAdjacentAxes B 0))]
  -- Rewrite matrix products and transpose entries.
  simp [get2_mat_mul_spec, get2_matrix_transpose_spec, Finset.mul_sum, Finset.sum_mul]
  -- The goal is now exactly `Finset.sum_comm` on the two inner indices.
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
  -- rewrite `(A·B)ᵀ`
  have hmulT :
      swapAdjacentAxes (matMulSpec A B) 0 =
        matMulSpec (swapAdjacentAxes B 0) (swapAdjacentAxes A 0) :=
    matrix_transpose_mul (A := A) (B := B)
  -- apply the right-adjoint lemma to `Bᵀ·Aᵀ` against `Cᵀ`
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
  -- simplify involutions and transpose the last dot back
  have hAinv : swapAdjacentAxes (swapAdjacentAxes A 0) 0 = A :=
    matrix_transpose_involution (A := A)
  have hCinv : swapAdjacentAxes (swapAdjacentAxes C 0) 0 = C :=
    matrix_transpose_involution (A := C)
  -- `dot (Bᵀ) D = dot B (Dᵀ)` for matching shapes.
  have hdot_swap :
      dot (swapAdjacentAxes B 0) (matMulSpec (swapAdjacentAxes C 0) A)
        =
      dot B (swapAdjacentAxes (matMulSpec (swapAdjacentAxes C 0) A) 0) := by
    -- Apply `dot_mat_transpose` to `B` and `((Cᵀ·A)ᵀ)`.
    have := dot_mat_transpose (m := n) (n := p)
      (A := B) (B := swapAdjacentAxes (matMulSpec (swapAdjacentAxes C 0) A) 0)
    -- Rewrite involutions.
    simpa [matrix_transpose_involution, hCinv] using this
  -- Finish by rewriting `transpose (Cᵀ·A) = Aᵀ·C`.
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
          -- `transpose (Cᵀ·A) = Aᵀ·C`
          simp [matrix_transpose_mul, hCinv]

/--
Outer product properties.
Essential for proving weight gradient correctness.
-/
theorem outer_product_transpose {m n : Nat}
  (a : Tensor ℝ [m])
  (b : Tensor ℝ [n]) :
  swapAdjacentAxes (outerProductSpec a b) 0 = outerProductSpec b a := by
  apply matrix_ext
  intro i j
  rw [get2_matrix_transpose_spec]
  simp [mul_comm]

/-! ## Reductions and aggregation -/


end Spec
