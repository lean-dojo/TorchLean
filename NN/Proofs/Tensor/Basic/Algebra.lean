/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Basic.Folds
public import NN.Proofs.Tensor.Basic.Core
public import NN.Spec.Core.Context.Real

/-!
# Tensor Algebra Lemmas

This file collects foundational algebraic lemmas for `TorchLean.Tensor`: extensionality, map/fold
rewrites, and pointwise arithmetic facts used throughout the autograd and runtime-correctness proof
files.
-/

@[expose] public section

open TorchLean

namespace Spec

open TorchLean TorchLean.Tensor
open scoped BigOperators

/-- Tensor extensionality over a generic element type: equal `getSpec` views imply equal tensors. -/
theorem tensor_ext {α : Type} [TorchLean.Storage α]
    {s : Shape} {x y : Tensor α s} :
  (∀ idxs : List Nat, getSpec x idxs = getSpec y idxs) → x = y := by
  exact Tensor.ext_getSpec
/-- Elementwise addition is associative (over $\mathbb R$ tensors). -/
theorem add_spec_assoc {s : Shape}
  (a b c : Tensor ℝ s) :
  addSpec (addSpec a b) c = addSpec a (addSpec b c) := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp [addSpec, map2Spec, add_assoc]

/-- Elementwise subtraction distributes over addition on the right. -/
theorem sub_spec_add_right {s : Shape}
  (a b c : Tensor ℝ s) :
  subSpec a (addSpec b c) = addSpec (subSpec a b) (negSpec c) := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp [subSpec, addSpec, negSpec, map2Spec, mapSpec, Tensor.map]
  ring

/-- Elementwise multiplication distributes over addition on the right. -/
theorem mul_spec_add_right {s : Shape}
  (a b c : Tensor ℝ s) :
  mulSpec a (addSpec b c) = addSpec (mulSpec a b) (mulSpec a c) := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp [mulSpec, addSpec, map2Spec, mul_add]

/-- Elementwise multiplication distributes over addition on the left. -/
theorem mul_spec_add_left {s : Shape}
  (a b c : Tensor ℝ s) :
  mulSpec (addSpec a b) c = addSpec (mulSpec a c) (mulSpec b c) := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp [mulSpec, addSpec, map2Spec, add_mul]

/-- Bias cancellation for tensor subtraction: `(a + c) - (b + c) = a - b`. -/
theorem sub_spec_bias_cancel {s : Shape} (a b c : Tensor ℝ s) :
  subSpec (addSpec a c) (addSpec b c) = subSpec a b := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp [subSpec, addSpec, map2Spec]

/-- Linearity of matrix-vector multiplication in the vector argument (addition). -/
theorem mat_vec_add {m n : Nat}
  (W : Tensor ℝ [m, n])
  (x y : Tensor ℝ [n]) :
  matVecMulSpec W (addSpec x y) =
  addSpec (matVecMulSpec W x) (matVecMulSpec W y) := by
  classical
  have hToVec :
      getScalar (matVecMulSpec W (addSpec x y)) =
        getScalar (addSpec (matVecMulSpec W x) (matVecMulSpec W y)) := by
    funext i
    -- Rewrite all mat-vec outputs as sums.
    rw [getScalar_mat_vec_mul_spec (A := W) (v := addSpec x y) (i := i)]
    -- Expand the elementwise addition on the right (without unfolding `getScalar` itself).
    simp [getScalar_add_spec]
    rw [getScalar_mat_vec_mul_spec (A := W) (v := x) (i := i)]
    rw [getScalar_mat_vec_mul_spec (A := W) (v := y) (i := i)]
    -- Distribute `*` over `+` inside the sum and split the sum.
    simp [mul_add, Finset.sum_add_distrib]

  have hTensor :
      ofFn (getScalar (matVecMulSpec W (addSpec x y))) =
        ofFn (getScalar (addSpec (matVecMulSpec W x) (matVecMulSpec W y))) :=
    congrArg ofFn hToVec

  -- `ofFn ∘ getScalar` is identity.
  simpa using
    (Eq.trans (ofFn_getScalar (t := matVecMulSpec W (addSpec x y))).symm
      (Eq.trans hTensor (ofFn_getScalar (t := addSpec (matVecMulSpec W x) (matVecMulSpec W
        y)))))

/-- Linearity of matrix-vector multiplication in the vector argument (scaling). -/
theorem mat_vec_scale {m n : Nat}
  (W : Tensor ℝ [m, n])
  (x : Tensor ℝ [n]) (c : ℝ) :
  matVecMulSpec W (scaleSpec x c) =
  scaleSpec (matVecMulSpec W x) c := by
  classical
  have hToVec :
      getScalar (matVecMulSpec W (scaleSpec x c)) =
        getScalar (scaleSpec (matVecMulSpec W x) c) := by
    funext i
    rw [getScalar_mat_vec_mul_spec (A := W) (v := scaleSpec x c) (i := i)]
    -- `getScalar (scale_spec _ c)` is pointwise scaling.
    simp [getScalar_scale_spec]
    rw [getScalar_mat_vec_mul_spec (A := W) (v := x) (i := i)]
    -- Pull out the scalar `c` from the sum.
    -- (Reassociate `*` so `Finset.sum_mul` applies.)
    have hassoc :
        (∑ k : Fin n, get2 W i k * (getScalar x k * c)) =
          ∑ k : Fin n, (get2 W i k * getScalar x k) * c := by
      refine Finset.sum_congr rfl ?_
      intro k _
      ring
    -- Now use `Finset.sum_mul` to factor `c` to the right.
    -- (`Finset.sum_mul` gives the reverse direction, so use symmetry.)
    simpa [hassoc, mul_assoc] using
      (Finset.sum_mul (s := (Finset.univ : Finset (Fin n)))
        (f := fun k : Fin n => get2 W i k * getScalar x k) (a := c)).symm

  have hTensor :
      ofFn (getScalar (matVecMulSpec W (scaleSpec x c))) =
        ofFn (getScalar (scaleSpec (matVecMulSpec W x) c)) :=
    congrArg ofFn hToVec

  simpa using
    (Eq.trans (ofFn_getScalar (t := matVecMulSpec W (scaleSpec x c))).symm
      (Eq.trans hTensor (ofFn_getScalar (t := scaleSpec (matVecMulSpec W x) c))))

/-- Full linearity of matrix-vector multiplication in the vector argument. -/
theorem mat_vec_linear_combination {m n : Nat}
  (W : Tensor ℝ [m, n])
  (x y : Tensor ℝ [n]) (a b : ℝ) :
  matVecMulSpec W (addSpec (scaleSpec x a) (scaleSpec y b)) =
  addSpec (scaleSpec (matVecMulSpec W x) a)
           (scaleSpec (matVecMulSpec W y) b) := by
  -- Combine mat_vec_add and mat_vec_scale
  rw [mat_vec_add, mat_vec_scale, mat_vec_scale]

/-- Mapping `0 + ·` over an `Option` is the identity. -/
theorem option_zero_add (o : Option ℝ) : o.map (fun x => 0 + x) = o := by
  cases o
  · rfl
  · simp [zero_add]

-- add_spec with zero tensor on the left
/-- Left identity for `addSpec`: adding the all-zero tensor does nothing. -/
@[simp]
theorem add_spec_zero_left {s : Shape} : ∀ (t : Tensor ℝ s),
  addSpec (Tensor.full s 0) t = t
| t => by
    apply TorchLean.Tensor.Internal.Rep.ext
    intro coordinate
    simp [addSpec, map2Spec, Tensor.full]

-- add_spec with zero tensor on the right
/-- Right identity for `addSpec`: adding the all-zero tensor does nothing. -/
@[simp]
theorem add_spec_zero_right {s : Shape} : ∀ (t : Tensor ℝ s),
  addSpec t (Tensor.full s (0 : ℝ)) = t
  | t => by
      apply TorchLean.Tensor.Internal.Rep.ext
      intro coordinate
      simp [addSpec, map2Spec, Tensor.full]

-- mul_spec with one tensor on the left
/-- Left identity for `mulSpec`: multiplying by the all-ones tensor does nothing. -/
@[simp]
theorem mul_spec_one_left {s : Shape} : ∀ (t : Tensor ℝ s),
  mulSpec (Tensor.full s (1 : ℝ)) t = t
| t => by
    apply TorchLean.Tensor.Internal.Rep.ext
    intro coordinate
    simp [mulSpec, map2Spec, Tensor.full]

-- mul_spec with one tensor on the right
/-- Right identity for `mulSpec`: multiplying by the all-ones tensor does nothing. -/
@[simp]
theorem mul_spec_one_right {s : Shape} : ∀ (t : Tensor ℝ s),
  mulSpec t (Tensor.full s (1 : ℝ)) = t
  | t => by
      apply TorchLean.Tensor.Internal.Rep.ext
      intro coordinate
      simp [mulSpec, map2Spec, Tensor.full]

/-- Adding scalar tensors in a left fold agrees with folding their scalar values. -/
theorem foldl_add_scalar {ι : Type} (values : ι → Tensor ℝ .scalar) (items : List ι)
    (initial : ℝ) :
    items.foldl (fun total item => total + values item) (Tensor.scalar initial) =
      Tensor.scalar (items.foldl (fun total item => total + (values item).item) initial) := by
  induction items generalizing initial with
  | nil => rfl
  | cons item rest ih =>
      simp only [List.foldl_cons]
      rw [show Tensor.scalar initial + values item =
        Tensor.scalar (initial + (values item).item) by
          apply Tensor.ext_scalar
          change Tensor.item
              (TorchLean.Tensor.Internal.Rep.zipWith (· + ·)
                (Tensor.scalar initial) (values item)) =
            initial + (values item).item
          unfold Tensor.item Tensor.scalar
          rw [TorchLean.Tensor.Internal.Rep.zipWith_apply, TorchLean.Tensor.Internal.Rep.get_ofFn]]
      exact ih (initial := initial + (values item).item)

end Spec
