/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorOps
public import NN.Proofs.Utils.List

/-!
# Tensor algebra proofs

Backend-generic algebraic lemmas for TorchLean's single packed tensor type.
Proofs observe tensors through `item`, `unstack`, and typed indexing; they do
not depend on the physical buffer selected by `Storage`.
-/

@[expose] public section

namespace Proofs
namespace TensorAlgebra

open Spec TorchLean TorchLean.Tensor

noncomputable section

/-- Multiply matching scalar entries and add them in row-major shape order. -/
def dot {α : Type} [TorchLean.Storage α] [Zero α] [Add α] [Mul α] :
    {shape : Shape} → Tensor α shape → Tensor α shape → α
  | .scalar, left, right => left.item * right.item
  | .dim length inner, left, right =>
      (List.finRange length).foldl
        (fun accumulator index =>
          accumulator +
            dot (shape := inner) (left.unstack index) (right.unstack index))
        0

/-! ## Dot-product algebra -/

section

variable {α : Type} [TorchLean.Storage α] [CommSemiring α]

/-- The dot product of two scalars is their product: the base case of the recursion. -/
@[simp] theorem dot_scalar (a b : α) :
    dot (shape := Shape.scalar) (Tensor.scalar a) (Tensor.scalar b) =
      a * b := by
  simp [dot]

/-- Dot is multiplicative with scalar scaling in its right argument. -/
theorem dot_scale_right {s : Shape}
    (left right : Tensor α s) (scalar : α) :
    dot left (scaleSpec right scalar) = dot left right * scalar := by
  induction s with
  | scalar =>
      simp [dot, scaleSpec, Tensor.item_map, mul_assoc]
  | dim length inner inductionHypothesis =>
      simp only [dot]
      calc
        _ = (List.finRange length).foldl (fun accumulator index =>
            accumulator + dot (left.unstack index) (right.unstack index) * scalar) 0 := by
          apply List.foldl_add_congr
          intro index
          change dot (left.unstack index)
            (Internal.Rep.unstack (Internal.Rep.map (· * scalar) right) index) = _
          rw [← Internal.Rep.map_unstack]
          exact inductionHypothesis _ _
        _ = _ := by
          simpa only [zero_mul] using
            List.foldl_add_mul_right (List.finRange length)
              (fun index => dot (left.unstack index) (right.unstack index)) 0 scalar

/-- Dot is symmetric over a commutative semiring. -/
theorem dot_comm {s : Shape} (left right : Tensor α s) :
    dot left right = dot right left := by
  induction s with
  | scalar => simp [dot, mul_comm]
  | dim length inner inductionHypothesis =>
      apply List.foldl_add_congr
      intro index
      exact inductionHypothesis _ _

/-- Dot is multiplicative with scalar scaling in its left argument. -/
theorem dot_scale_left {s : Shape}
    (left right : Tensor α s) (scalar : α) :
    dot (scaleSpec left scalar) right = dot left right * scalar := by
  calc
    dot (scaleSpec left scalar) right =
        dot right (scaleSpec left scalar) := dot_comm _ _
    _ = dot right left * scalar := dot_scale_right _ _ _
    _ = dot left right * scalar := by rw [dot_comm right left]

/-- Dot distributes over addition in its left argument. -/
theorem dot_add_left {s : Shape}
    (left middle right : Tensor α s) :
    dot (addSpec left middle) right = dot left right + dot middle right := by
  induction s with
  | scalar =>
      simp [dot, addSpec, map2Spec, Tensor.item,
        TorchLean.Tensor.Internal.Rep.zipWith_apply, add_mul]
  | dim length inner inductionHypothesis =>
      simp only [dot]
      calc
        _ = (List.finRange length).foldl (fun accumulator index =>
            accumulator + (dot (left.unstack index) (right.unstack index) +
              dot (middle.unstack index) (right.unstack index))) 0 := by
          apply List.foldl_add_congr
          intro index
          change dot
            (Internal.Rep.unstack (Internal.Rep.zipWith (· + ·) left middle) index)
            (right.unstack index) = _
          rw [← Internal.Rep.zipWith_unstack]
          exact inductionHypothesis _ _ _
        _ = _ := by
          simpa only [zero_add] using
            List.foldl_add_distrib2 (List.finRange length)
              (fun index => dot (left.unstack index) (right.unstack index))
              (fun index => dot (middle.unstack index) (right.unstack index)) 0 0

/-- Dot distributes over addition in its right argument. -/
theorem dot_add_right {s : Shape}
    (left middle right : Tensor α s) :
    dot left (addSpec middle right) = dot left middle + dot left right := by
  calc
    dot left (addSpec middle right) =
        dot (addSpec middle right) left := dot_comm _ _
    _ = dot middle left + dot right left := dot_add_left _ _ _
    _ = dot left middle + dot left right := by
      rw [dot_comm middle left, dot_comm right left]

/-- The dot product with an all-zero tensor is zero. -/
theorem dot_full_zero_right {s : Shape} (tensor : Tensor α s) :
    dot tensor (Tensor.full s 0) = 0 := by
  induction s with
  | scalar => simp [dot, Tensor.item, Tensor.full_apply]
  | dim length inner inductionHypothesis =>
      simp only [dot]
      calc
        dot tensor (Tensor.full (.dim length inner) 0) = (List.finRange length).foldl
            (fun accumulator (_ : Fin length) => accumulator + 0) 0 := by
          apply List.foldl_add_congr (l := List.finRange length)
            (f := fun index => dot (tensor.unstack index)
              ((Tensor.full (.dim length inner) (0 : α)).unstack index))
            (g := fun _ => (0 : α)) (a := (0 : α))
          intro index
          have hUnstack :
              (Tensor.full (.dim length inner) (0 : α)).unstack index =
                Tensor.full inner 0 := by
            simpa only [Spec.get] using get_full length inner (0 : α) index
          rw [hUnstack]
          exact inductionHypothesis _
        _ = 0 := by simp

/-- Vector dot is the ordinary finite sum of matching scalar coordinates. -/
theorem dot_vec_eq_sum {n : Nat} (a b : Tensor α [n]) :
    dot a b =
      ∑ index : Fin n,
        a.getScalar index * b.getScalar index := by
  change
    (List.finRange n).foldl
        (fun accumulator index =>
          accumulator + a.getScalar index * b.getScalar index)
        0 =
      ∑ index : Fin n, a.getScalar index * b.getScalar index
  exact List.finRange_foldl_add_eq_finset_sum _

/-! ## Indexing and matrix/vector bridges -/

omit [CommSemiring α] in
/-- Matrix indexing is two successive packed leading-axis selections. -/
theorem get2_eq {m n : Nat} (A : Tensor α [m, n])
    (i : Fin m) (j : Fin n) :
    get2 A i j =
      (Tensor.unstack (Tensor.unstack A i) j).item :=
  rfl

omit [CommSemiring α] in
/-- `Spec.get` is the public spelling of packed leading-axis selection. -/
theorem get_eq {m : Nat} {shape : Shape}
    (t : Tensor α (.dim m shape)) (i : Fin m) :
    Spec.get t i = Tensor.unstack t i :=
  rfl

/-- Coordinate expansion of matrix-vector multiplication. -/
theorem getScalar_mat_vec_mul_spec {m n : Nat}
    (A : Tensor α [m, n]) (v : Tensor α [n])
    (i : Fin m) :
    getScalar (matVecMulSpec A v) i =
      ∑ k : Fin n, get2 A i k * getScalar v k := by
  rw [getScalar_eq_apply]
  simp only [matVecMulSpec, TorchLean.Tensor.Internal.Rep.get_ofFn]
  exact List.finRange_foldl_add_eq_finset_sum _

/-- Coordinate expansion of vector-matrix multiplication. -/
theorem getScalar_vec_mat_mul_spec {m n : Nat}
    (v : Tensor α [m]) (A : Tensor α [m, n])
    (j : Fin n) :
    getScalar (vecMatMulSpec v A) j =
      ∑ i : Fin m, getScalar v i * get2 A i j := by
  rw [getScalar_eq_apply]
  simp only [vecMatMulSpec, TorchLean.Tensor.Internal.Rep.get_ofFn]
  exact List.finRange_foldl_add_eq_finset_sum _

/-- Matrix-vector and vector-matrix multiplication are adjoint under `dot`. -/
theorem dot_mat_linear_adjoint {inDim outDim : Nat}
    (W : Tensor α [outDim, inDim])
    (dLdy : Tensor α [outDim])
    (dx : Tensor α [inDim]) :
    dot dLdy (matVecMulSpec W dx) =
      dot (vecMatMulSpec dLdy W) dx := by
  classical
  rw [dot_vec_eq_sum, dot_vec_eq_sum]
  simp only [getScalar_mat_vec_mul_spec, getScalar_vec_mat_mul_spec]
  calc
    (∑ output : Fin outDim,
        getScalar dLdy output *
          ∑ input : Fin inDim, get2 W output input * getScalar dx input) =
        ∑ output : Fin outDim, ∑ input : Fin inDim,
          getScalar dLdy output * (get2 W output input * getScalar dx input) := by
      apply Finset.sum_congr rfl
      intro output _
      rw [Finset.mul_sum]
    _ = ∑ input : Fin inDim, ∑ output : Fin outDim,
          getScalar dLdy output * (get2 W output input * getScalar dx input) := by
      exact Finset.sum_comm
    _ = ∑ input : Fin inDim,
          (∑ output : Fin outDim, getScalar dLdy output * get2 W output input) *
            getScalar dx input := by
      apply Finset.sum_congr rfl
      intro input _
      rw [Finset.sum_mul]
      apply Finset.sum_congr rfl
      intro output _
      simp [mul_assoc]

end

end

end TensorAlgebra
end Proofs
