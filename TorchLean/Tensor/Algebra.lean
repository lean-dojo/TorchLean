/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Arithmetic
public import TorchLean.Tensor.List
public import TorchLean.Tensor.Contract

/-!
# Tensor algebra proofs

Backend-generic algebraic lemmas for TorchLean's single packed tensor type. Proofs observe tensors
through `item`, `unstack` and typed indexing, so they hold for whichever physical buffer `Storage`
selects.

The pairing is `TorchLean.Tensor.Rep.dot`, the finite sum of matching entry
products, which callers name `dot` after `open TorchLean.Tensor.Rep (dot)`.
It is Mathlib's `dotProduct` of the two coordinate functions (`dot_eq_dotProduct`, by `rfl`),
so over a commutative semiring we prove its symmetry, linearity and matrix adjoint by
transporting the `dotProduct` lemmas. `dot_eq_sum_unstack` splits it along the leading axis,
and `dot_elemwise_adjoint` moves a pointwise factor across it.
-/

@[expose] public section

namespace TorchLean
namespace Tensor

open Rep (dot)

/-! ## Pointwise multiplication

Tensor `*` multiplies coordinatewise, so it is associative, commutative or unital whenever the
scalar multiplication is, and it distributes over tensor `+` whenever the scalar operations do.
The unit is the all-ones tensor `Tensor.full s 1`, and `a ^ n` raises every coordinate of `a` to
the power `n`. These instances let `mul_assoc`, `mul_comm`, `one_mul`, `mul_add`, `add_mul` and
`ac_rfl` act on tensor products.
-/

/-- Coordinatewise tensor multiplication is associative when scalar multiplication is. -/
instance {α : Type} [Storage α] [Semigroup α] {s : Shape} :
    Semigroup (Tensor α s) where
  mul_assoc a b c := by
    apply Rep.ext
    intro coordinate
    simp [mul_def, mul_assoc]

/-- Coordinatewise tensor multiplication is commutative when scalar multiplication is. -/
instance {α : Type} [Storage α] [CommSemigroup α] {s : Shape} :
    CommSemigroup (Tensor α s) where
  mul_comm a b := by
    apply Rep.ext
    intro coordinate
    simp [mul_def, mul_comm]

/-- The tensor `1` is the all-ones tensor `Tensor.full s 1`. -/
instance {α : Type} [Storage α] [One α] {s : Shape} : One (Tensor α s) :=
  ⟨Tensor.full s 1⟩

/-- The all-ones tensor is a two-sided unit for coordinatewise multiplication when the scalar `1`
is a unit for scalar multiplication. -/
instance {α : Type} [Storage α] [MulOneClass α] {s : Shape} :
    MulOneClass (Tensor α s) where
  one_mul a := by
    show Tensor.full s 1 * a = a
    apply Rep.ext
    intro coordinate
    simp [mul_def]
  mul_one a := by
    show a * Tensor.full s 1 = a
    apply Rep.ext
    intro coordinate
    simp [mul_def]

/-- Coordinatewise tensor multiplication is a monoid when scalar multiplication is, and `a ^ n`
is `Tensor.map (· ^ n) a`. -/
instance {α : Type} [Storage α] [Monoid α] {s : Shape} : Monoid (Tensor α s) where
  one_mul := one_mul
  mul_one := mul_one
  npow n a := Tensor.map (· ^ n) a
  npow_zero a := by
    show Tensor.map (· ^ 0) a = Tensor.full s 1
    apply Rep.ext
    intro coordinate
    simp
  npow_succ n a := by
    show Tensor.map (· ^ (n + 1)) a = Tensor.map (· ^ n) a * a
    apply Rep.ext
    intro coordinate
    simp [mul_def, pow_succ]

/-- Coordinatewise tensor multiplication is a commutative monoid when scalar multiplication is. -/
instance {α : Type} [Storage α] [CommMonoid α] {s : Shape} :
    CommMonoid (Tensor α s) where
  mul_comm := mul_comm

/-- Coordinatewise tensor multiplication distributes over coordinatewise tensor addition when
scalar multiplication distributes over scalar addition. -/
instance {α : Type} [Storage α] [Distrib α] {s : Shape} : Distrib (Tensor α s) where
  left_distrib a b c := by
    apply Rep.ext
    intro coordinate
    simp [mul_def, left_distrib]
  right_distrib a b c := by
    apply Rep.ext
    intro coordinate
    simp [mul_def, right_distrib]

/-! ## Dot-product algebra -/

section

variable {α : Type} [Storage α] [AddCommMonoid α] [Mul α]

/-- The pairing is Mathlib's `dotProduct` of the two coordinate functions. -/
theorem dot_eq_dotProduct {s : Shape} (left right : Tensor α s) :
    Rep.dot left right = (fun i => left i) ⬝ᵥ fun i => right i :=
  rfl

/-- The pairing of two scalar tensors is the product of their entries. -/
@[simp] theorem dot_scalar (left right : Tensor α .scalar) :
    Rep.dot left right = left.item * right.item := by
  simp [Rep.dot, Tensor.item]

/-- The pairing of two tensors is the sum of the pairings of their leading-axis slices. -/
theorem dot_eq_sum_unstack {n : Nat} {s : Shape} (left right : Tensor α (.dim n s)) :
    Rep.dot left right = ∑ index : Fin n, Rep.dot (left.unstack index) (right.unstack index) := by
  simp only [Rep.dot, Tensor.unstack_apply]
  exact Fintype.sum_prod_type _

end

section

variable {α : Type} [Storage α] [CommSemiring α]

/-- Scaling the right argument by `scalar` multiplies the dot product by `scalar`. -/
theorem dot_scale_right {s : Shape}
    (left right : Tensor α s) (scalar : α) :
    Rep.dot left (Tensor.scale right scalar) = Rep.dot left right * scalar := by
  simp only [dot_eq_dotProduct, Tensor.scale, Tensor.map_apply]
  exact dotProduct_smul (MulOpposite.op scalar) (fun i => left i) fun i => right i

/-- Dot is symmetric over a commutative semiring. -/
theorem dot_comm {s : Shape} (left right : Tensor α s) :
    Rep.dot left right = Rep.dot right left :=
  dotProduct_comm _ _

/-- Scaling the left argument by `scalar` multiplies the dot product by `scalar`. -/
theorem dot_scale_left {s : Shape}
    (left right : Tensor α s) (scalar : α) :
    Rep.dot (Tensor.scale left scalar) right = Rep.dot left right * scalar := by
  rw [dot_comm, dot_scale_right, dot_comm]

/-- Dot distributes over addition in its left argument. -/
theorem dot_add_left {s : Shape}
    (left middle right : Tensor α s) :
    Rep.dot (left + middle) right = Rep.dot left right + Rep.dot middle right := by
  simp only [dot_eq_dotProduct, Tensor.add_apply]
  exact add_dotProduct (fun i => left i) (fun i => middle i) fun i => right i

/-- Dot distributes over addition in its right argument. -/
theorem dot_add_right {s : Shape}
    (left middle right : Tensor α s) :
    Rep.dot left (middle + right) = Rep.dot left middle + Rep.dot left right := by
  simp only [dot_eq_dotProduct, Tensor.add_apply]
  exact dotProduct_add (fun i => left i) (fun i => middle i) fun i => right i

/-- The dot product with an all-zero tensor is zero. -/
theorem dot_full_zero_right {s : Shape} (tensor : Tensor α s) :
    Rep.dot tensor (Tensor.full s 0) = 0 := by
  simp only [dot_eq_dotProduct, full_apply, dotProduct_zero']

/-- The dot product of an all-zero tensor with any tensor is zero. -/
theorem dot_full_zero_left {s : Shape} (tensor : Tensor α s) :
    Rep.dot (Tensor.full s 0) tensor = 0 := by
  simp only [dot_eq_dotProduct, full_apply, zero_dotProduct']

/-- Pointwise multiplication by a fixed tensor `df` is self-adjoint for the pairing `dot`:
$\langle dx\odot df,\delta\rangle=\langle dx,df\odot\delta\rangle$. -/
theorem dot_elemwise_adjoint {s : Shape} (dx df δ : Tensor α s) :
    Rep.dot (dx * df) δ = Rep.dot dx (df * δ) := by
  simp only [Rep.dot, Tensor.mul_apply, mul_assoc]

/-- A vector dot product is the finite sum of the products of matching coordinates. -/
theorem dot_vec_eq_sum {n : Nat} (a b : Tensor α [n]) :
    Rep.dot a b =
      ∑ index : Fin n,
        a.getScalar index * b.getScalar index := by
  rw [dot_eq_sum_unstack]
  simp only [dot_scalar]
  rfl

/-! ## Indexing and matrix/vector bridges -/

omit [CommSemiring α] in
/-- Matrix indexing is two successive packed leading-axis selections. -/
theorem get2_eq_item_unstack {m n : Nat} (A : Tensor α [m, n])
    (i : Fin m) (j : Fin n) :
    get2 A i j =
      (Tensor.unstack (Tensor.unstack A i) j).item :=
  rfl

/-- Coordinate expansion of matrix-vector multiplication. -/
theorem getScalar_contract_matVec_eq_sum {m n : Nat}
    (A : Tensor α [m, n]) (v : Tensor α [n])
    (i : Fin m) :
    getScalar (Tensor.contract A v _ _ [m] .matVec) i =
      ∑ k : Fin n, get2 A i k * getScalar v k := by
  rw [getScalar_contract_matVec]
  rw [Fin.sum_univ_def, List.sum_eq_foldl, List.foldl_map]

/-- Coordinate expansion of vector-matrix multiplication. -/
theorem getScalar_contract_vecMat_eq_sum {m n : Nat}
    (v : Tensor α [m]) (A : Tensor α [m, n])
    (j : Fin n) :
    getScalar (Tensor.contract v A _ _ [n] .vecMat) j =
      ∑ i : Fin m, getScalar v i * get2 A i j := by
  rw [getScalar_contract_vecMat]
  rw [Fin.sum_univ_def, List.sum_eq_foldl, List.foldl_map]

/-- Matrix-vector and vector-matrix multiplication are adjoint under `dot`. -/
theorem dot_contract_matVec {inDim outDim : Nat}
    (W : Tensor α [outDim, inDim])
    (dLdy : Tensor α [outDim])
    (dx : Tensor α [inDim]) :
    Rep.dot dLdy (Tensor.contract W dx _ _ [outDim] .matVec) =
      Rep.dot (Tensor.contract dLdy W _ _ [inDim] .vecMat) dx := by
  rw [dot_vec_eq_sum, dot_vec_eq_sum]
  simp only [getScalar_contract_matVec_eq_sum, getScalar_contract_vecMat_eq_sum]
  exact Matrix.dotProduct_mulVec (fun output => getScalar dLdy output)
    (Matrix.of fun output input => get2 W output input) fun input => getScalar dx input

end

end Tensor
end TorchLean
