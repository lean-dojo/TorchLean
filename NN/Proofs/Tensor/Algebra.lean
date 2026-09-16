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

open Spec TorchLean
open TorchLean TorchLean.Tensor

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

/-! ## Fold bridges -/

/-- Push an accumulator into an additive fold that starts at zero. -/
theorem add_finRange_foldl_add_zero {α : Type} [AddMonoid α] {n : Nat}
    (f : Fin n → α) (accumulator : α) :
    accumulator + (List.finRange n).foldl (fun sum i => sum + f i) 0 =
      (List.finRange n).foldl (fun sum i => sum + f i) accumulator := by
  simpa using
    List.add_foldl_add0
      (l := List.finRange n) (f := f) (acc := accumulator)

/-- A fold over scalar tensors is the corresponding fold over scalar values. -/
theorem foldl_tensorScalar_mulAdd {α : Type} [TorchLean.Storage α]
    [Add α] [Mul α] {n : Nat}
    (columns values : Fin n → Tensor α .scalar)
    (indices : List (Fin n)) (initial : α) :
    indices.foldl
        (fun accumulator index =>
          Tensor.scalar
            (accumulator.item +
              (columns index).item * (values index).item))
        (Tensor.scalar initial) =
      Tensor.scalar
        (indices.foldl
          (fun accumulator index =>
            accumulator +
              (columns index).item * (values index).item)
          initial) := by
  induction indices generalizing initial with
  | nil => rfl
  | cons index indices inductionHypothesis =>
      simp only [List.foldl_cons, Tensor.item_scalar]
      exact inductionHypothesis _

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
    (a b : Tensor α s) (k : α) :
    dot a (scaleSpec b k) = dot a b * k := by
  let left := a
  let right := b
  let scalar := k
  change dot left (scaleSpec right scalar) = dot left right * scalar
  induction s with
  | scalar =>
      simp [dot, scaleSpec, Tensor.toScalar_mapSpec, mul_assoc]
  | dim length inner inductionHypothesis =>
      have hTerm : ∀ index : Fin length,
          dot (left.unstack index)
              ((scaleSpec right scalar).unstack index) =
            dot (left.unstack index) (right.unstack index) * scalar := by
        intro index
        have hUnstack :
            (scaleSpec right scalar).unstack index =
              scaleSpec (right.unstack index) scalar := by
          apply TorchLean.Tensor.Internal.Rep.ext
          intro coordinate
          simp [Tensor.unstack, scaleSpec, mapSpec, Tensor.map]
        rw [hUnstack]
        exact inductionHypothesis _ _
      have hCongruence :
          (List.finRange length).foldl
              (fun accumulator index =>
                accumulator +
                  dot (left.unstack index)
                    ((scaleSpec right scalar).unstack index))
              0 =
            (List.finRange length).foldl
              (fun accumulator index =>
                accumulator +
                  dot (left.unstack index) (right.unstack index) * scalar)
              0 := by
        simpa using
          (List.foldl_add_congr
            (l := List.finRange length)
            (f := fun index =>
              dot (left.unstack index)
                ((scaleSpec right scalar).unstack index))
            (g := fun index =>
              dot (left.unstack index) (right.unstack index) * scalar)
            (a := (0 : α)) (h := hTerm))
      have hScaleFold :
          (List.finRange length).foldl
              (fun accumulator index =>
                accumulator +
                  dot (left.unstack index) (right.unstack index) * scalar)
              0 =
            (List.finRange length).foldl
                (fun accumulator index =>
                  accumulator +
                    dot (left.unstack index) (right.unstack index))
                0 * scalar := by
        simpa [zero_mul] using
          (List.foldl_add_mul_right
            (α := α) (l := List.finRange length)
            (g := fun index =>
              dot (left.unstack index) (right.unstack index))
            (a := (0 : α)) (k := scalar))
      exact hCongruence.trans hScaleFold

/-- Dot is symmetric over a commutative semiring. -/
theorem dot_comm {s : Shape} (a b : Tensor α s) :
    dot a b = dot b a := by
  let left := a
  let right := b
  change dot left right = dot right left
  induction s with
  | scalar => simp [dot, mul_comm]
  | dim length inner inductionHypothesis =>
      apply List.foldl_add_congr
      intro index
      exact inductionHypothesis _ _

/-- Dot is multiplicative with scalar scaling in its left argument. -/
theorem dot_scale_left {s : Shape}
    (a b : Tensor α s) (k : α) :
    dot (scaleSpec a k) b = dot a b * k := by
  let left := a
  let right := b
  let scalar := k
  change dot (scaleSpec left scalar) right = dot left right * scalar
  calc
    dot (scaleSpec left scalar) right =
        dot right (scaleSpec left scalar) := dot_comm _ _
    _ = dot right left * scalar := dot_scale_right _ _ _
    _ = dot left right * scalar := by rw [dot_comm right left]

/-- Dot distributes over addition in its left argument. -/
theorem dot_add_left {s : Shape}
    (a b c : Tensor α s) :
    dot (addSpec a b) c = dot a c + dot b c := by
  let left := a
  let middle := b
  let right := c
  change dot (addSpec left middle) right =
    dot left right + dot middle right
  induction s with
  | scalar =>
      simp [dot, addSpec, map2Spec, Tensor.item,
        TorchLean.Tensor.Internal.Rep.zipWith_apply, add_mul]
  | dim length inner inductionHypothesis =>
      have hTerm : ∀ index : Fin length,
          dot ((addSpec left middle).unstack index)
              (right.unstack index) =
            dot (left.unstack index) (right.unstack index) +
              dot (middle.unstack index) (right.unstack index) := by
        intro index
        have hUnstack :
            (addSpec left middle).unstack index =
              addSpec (left.unstack index) (middle.unstack index) := by
          apply TorchLean.Tensor.Internal.Rep.ext
          intro coordinate
          simp [Tensor.unstack, addSpec, map2Spec]
        rw [hUnstack]
        exact inductionHypothesis _ _ _
      have hCongruence :
          (List.finRange length).foldl
              (fun accumulator index =>
                accumulator +
                  dot ((addSpec left middle).unstack index)
                    (right.unstack index))
              0 =
            (List.finRange length).foldl
              (fun accumulator index =>
                accumulator +
                  (dot (left.unstack index) (right.unstack index) +
                    dot (middle.unstack index) (right.unstack index)))
              0 := by
        simpa using
          (List.foldl_add_congr
            (l := List.finRange length)
            (f := fun index =>
              dot ((addSpec left middle).unstack index)
                (right.unstack index))
            (g := fun index =>
              dot (left.unstack index) (right.unstack index) +
                dot (middle.unstack index) (right.unstack index))
            (a := (0 : α)) (h := hTerm))
      have hSplit :=
        List.foldl_add_distrib2
          (l := List.finRange length)
          (g1 := fun index =>
            dot (left.unstack index) (right.unstack index))
          (g2 := fun index =>
            dot (middle.unstack index) (right.unstack index))
          (a1 := (0 : α)) (a2 := (0 : α))
      exact hCongruence.trans (by
        simpa only [dot, zero_add, add_assoc, add_left_comm, add_comm] using hSplit)

/-- Dot distributes over addition in its right argument. -/
theorem dot_add_right {s : Shape}
    (a b c : Tensor α s) :
    dot a (addSpec b c) = dot a b + dot a c := by
  let left := a
  let middle := b
  let right := c
  change dot left (addSpec middle right) =
    dot left middle + dot left right
  calc
    dot left (addSpec middle right) =
        dot (addSpec middle right) left := dot_comm _ _
    _ = dot middle left + dot right left := dot_add_left _ _ _
    _ = dot left middle + dot left right := by
      rw [dot_comm middle left, dot_comm right left]

/-- The dot product with an all-zero tensor is zero. -/
theorem dot_full_zero_right {s : Shape} (a : Tensor α s) :
    dot a (Tensor.full s 0) = 0 := by
  let tensor := a
  change dot tensor (Tensor.full s 0) = 0
  induction s with
  | scalar =>
      simp only [dot]
      have hFill : (Tensor.full .scalar (0 : α)).item = 0 := by
        exact Tensor.full_apply .scalar 0 PUnit.unit
      calc
        tensor.item * (Tensor.full .scalar 0).item = tensor.item * 0 :=
          congrArg (tensor.item * ·) hFill
        _ = 0 := mul_zero tensor.item
  | dim length inner inductionHypothesis =>
      have hTerm : ∀ index : Fin length,
          dot (tensor.unstack index)
              ((Tensor.full (.dim length inner) 0).unstack index) = 0 := by
        intro index
        have hUnstack :
            (Tensor.full (.dim length inner) (0 : α)).unstack index = Tensor.full inner 0 := by
          simpa only [Spec.get] using get_full length inner (0 : α) index
        calc
          dot (tensor.unstack index)
              ((Tensor.full (.dim length inner) 0).unstack index) =
              dot (tensor.unstack index) (Tensor.full inner 0) := by
                exact congrArg (dot (tensor.unstack index)) hUnstack
          _ = 0 := inductionHypothesis _
      have hCongruence :
          (List.finRange length).foldl
              (fun accumulator index =>
                accumulator +
                  dot (tensor.unstack index)
                    ((Tensor.full (.dim length inner) 0).unstack index))
              0 =
            (List.finRange length).foldl
              (fun accumulator (_ : Fin length) => accumulator + 0)
              0 := by
        simpa using
          (List.foldl_add_congr
            (l := List.finRange length)
            (f := fun index =>
              dot (tensor.unstack index)
                ((Tensor.full (.dim length inner) 0).unstack index))
            (g := fun _ => (0 : α)) (a := (0 : α)) (h := hTerm))
      exact hCongruence.trans (by simp)

/-- Vector dot is the ordinary finite sum of matching scalar coordinates. -/
theorem dot_vec_eq_sum {n : Nat} (a b : Tensor α [n]) :
    dot a b =
      ∑ index : Fin n,
        a.getScalar index * b.getScalar index := by
  let left := a
  let right := b
  change dot left right =
    ∑ index : Fin n, left.getScalar index * right.getScalar index
  change
    (List.finRange n).foldl
        (fun accumulator index =>
          accumulator +
            left.getScalar index * right.getScalar index)
        0 =
      ∑ index : Fin n,
        left.getScalar index * right.getScalar index
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

/-- Scalar-tensor multiply-add folds agree with their scalar-value folds. -/
theorem foldl_matvec_scalar {n : Nat} (l : List (Fin n))
    (a : α) (cols vals : Fin n → Tensor α .scalar) :
    l.foldl
        (fun accumulator index =>
          Tensor.scalar
            (accumulator.item +
              (cols index).item * (vals index).item))
        (Tensor.scalar a) =
      Tensor.scalar
        (l.foldl
          (fun accumulator index =>
            accumulator +
              (cols index).item * (vals index).item)
          a) :=
  foldl_tensorScalar_mulAdd cols vals l a

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
  let weights := W
  let outputGradient := dLdy
  let inputTangent := dx
  change dot outputGradient (matVecMulSpec weights inputTangent) =
    dot (vecMatMulSpec outputGradient weights) inputTangent
  classical
  rw [dot_vec_eq_sum, dot_vec_eq_sum]
  simp only [getScalar_mat_vec_mul_spec, getScalar_vec_mat_mul_spec]
  calc
    (∑ output : Fin outDim,
        getScalar outputGradient output *
          ∑ input : Fin inDim,
            get2 weights output input * getScalar inputTangent input) =
        ∑ output : Fin outDim, ∑ input : Fin inDim,
          getScalar outputGradient output *
            (get2 weights output input * getScalar inputTangent input) := by
      apply Finset.sum_congr rfl
      intro output _
      rw [Finset.mul_sum]
    _ = ∑ input : Fin inDim, ∑ output : Fin outDim,
          getScalar outputGradient output *
            (get2 weights output input * getScalar inputTangent input) := by
      exact Finset.sum_comm
    _ = ∑ input : Fin inDim,
          (∑ output : Fin outDim,
            getScalar outputGradient output *
              get2 weights output input) *
            getScalar inputTangent input := by
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
