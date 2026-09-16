/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context.Real
public import NN.Proofs.Tensor.Basic.LinearAlgebra

/-!
Bounds and norm facts for dependent tensors.

The proofs use coordinate extensionality and finite sums. They therefore describe the mathematical
tensor independently of its packed physical storage.
-/

@[expose] public section

open TorchLean

namespace Spec

open TorchLean TorchLean.Tensor
open scoped BigOperators

/-- Sum distributes over elementwise addition. -/
theorem sum_spec_add_distrib {s : Shape} (a b : Tensor ℝ s) :
    sumSpec (addSpec a b) = sumSpec a + sumSpec b := by
  rw [sum_spec_eq_coord_sum, sum_spec_eq_coord_sum, sum_spec_eq_coord_sum]
  simp only [addSpec, map2Spec, TorchLean.Tensor.Internal.Rep.zipWith_apply]
  exact Finset.sum_add_distrib

/-- The dot product is symmetric. -/
theorem dot_comm {s : Shape} (a b : Tensor ℝ s) :
    dot a b = dot b a := by
  simp [dot, mul_spec_comm]

/-- Dot-product distributes over addition in the left argument. -/
theorem dot_add_left {s : Shape} (a b c : Tensor ℝ s) :
    dot (addSpec a b) c = dot a c + dot b c := by
  have hmul : mulSpec (addSpec a b) c = addSpec (mulSpec a c) (mulSpec b c) := by
    apply TorchLean.Tensor.Internal.Rep.ext
    intro coordinate
    simp [mulSpec, addSpec, map2Spec, add_mul]
  rw [dot, hmul, sum_spec_add_distrib]
  rfl

/-- Scaling a tensor scales its dot-product. -/
theorem dot_scale_left {s : Shape} (a b : Tensor ℝ s) (k : ℝ) :
    dot (scaleSpec a k) b = k * dot a b := by
  rw [dot, dot, sum_spec_eq_coord_sum, sum_spec_eq_coord_sum, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro coordinate _
  simp [scaleSpec, mapSpec, Tensor.map, mulSpec, map2Spec, mul_assoc]
  ring

/-!
## Shape preservation
-/

/-- Elementwise addition does not change the statically tracked shape size. -/
theorem shape_size_add {s : Shape} (a b : Tensor ℝ s) :
    Shape.size (shapeOf (addSpec a b)) = Shape.size s := by
  rw [shapeOf_eq_shape]

/-- Elementwise multiplication does not change the statically tracked shape size. -/
theorem shape_size_mul {s : Shape} (a b : Tensor ℝ s) :
    Shape.size (shapeOf (mulSpec a b)) = Shape.size s := by
  rw [shapeOf_eq_shape]

/-!
## Uniform finite bounds
-/

/-- Every coordinate is at most `bound`. -/
private def tensorAllLE {s : Shape} (bound : ℝ) (tensor : Tensor ℝ s) : Prop :=
  ∀ coordinate, tensor coordinate ≤ bound

/-- A finite, representation-independent upper bound for all tensor entries. -/
private noncomputable def tensorMax {s : Shape} (tensor : Tensor ℝ s) : ℝ :=
  ∑ coordinate : s.Coord, |tensor coordinate|

private theorem tensorAllLE_mono {s : Shape} {lower upper : ℝ}
    (tensor : Tensor ℝ s) (hTensor : tensorAllLE lower tensor) (hBounds : lower ≤ upper) :
    tensorAllLE upper tensor := by
  intro coordinate
  exact (hTensor coordinate).trans hBounds

private theorem tensorAllLE_tensorMax {s : Shape} (tensor : Tensor ℝ s) :
    tensorAllLE (tensorMax tensor) tensor := by
  intro coordinate
  apply (le_abs_self (tensor coordinate)).trans
  exact Finset.single_le_sum
    (fun index _ => abs_nonneg (tensor index))
    (Finset.mem_univ coordinate)

private theorem map_min_eq_self_of_tensorAllLE {s : Shape} {bound : ℝ}
    (tensor : Tensor ℝ s) (hTensor : tensorAllLE bound tensor) :
    mapSpec (fun value => min value bound) tensor = tensor := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp only [mapSpec, Tensor.map, TorchLean.Tensor.Internal.Rep.map_apply]
  exact min_eq_left (hTensor coordinate)

/-- A finite tensor admits a uniform bound, expressed by an idempotent minimum clamp. -/
theorem safediv_bound {s : Shape} (a b : Tensor ℝ s) :
    ∀ _i : Fin s.size, (Context.defaultEpsilon : ℝ) > 0 →
      ∃ bound,
        absSpec (safedivSpec a b) =
          mapSpec (fun value => min value bound) (absSpec (safedivSpec a b)) := by
  intro _ _
  let tensor := absSpec (safedivSpec a b)
  refine ⟨tensorMax tensor, ?_⟩
  exact (map_min_eq_self_of_tensorAllLE tensor (tensorAllLE_tensorMax tensor)).symm

/-!
## Squared norm
-/

/-- The squared Euclidean norm of all coordinates of a tensor. -/
noncomputable def tensorNormSquared {s : Shape} (tensor : Tensor ℝ s) : ℝ :=
  dot tensor tensor

/-- A tensor squared norm is nonnegative. -/
theorem tensor_norm_squared_nonneg {s : Shape} (tensor : Tensor ℝ s) :
    0 ≤ tensorNormSquared tensor := by
  rw [tensorNormSquared, dot, sum_spec_eq_coord_sum]
  apply Finset.sum_nonneg
  intro coordinate _
  simp only [mulSpec, map2Spec, TorchLean.Tensor.Internal.Rep.zipWith_apply]
  exact mul_self_nonneg (tensor coordinate)

/-- A tensor has zero squared norm exactly when every coordinate is zero. -/
theorem tensor_norm_squared_zero_iff {s : Shape} (tensor : Tensor ℝ s) :
    tensorNormSquared tensor = 0 ↔ tensor = Tensor.full s (0 : ℝ) := by
  rw [tensorNormSquared, dot, sum_spec_eq_coord_sum]
  constructor
  · intro hSum
    have hTerms :
        ∀ coordinate ∈ (Finset.univ : Finset s.Coord),
          tensor coordinate * tensor coordinate = 0 := by
      exact
        (Finset.sum_eq_zero_iff_of_nonneg
          (fun coordinate _ => mul_self_nonneg (tensor coordinate))).mp
          (by simpa only [mulSpec, map2Spec, TorchLean.Tensor.Internal.Rep.zipWith_apply]
            using hSum)
    apply TorchLean.Tensor.Internal.Rep.ext
    intro coordinate
    have hZero : tensor coordinate = 0 :=
      mul_self_eq_zero.mp (hTerms coordinate (Finset.mem_univ coordinate))
    simpa [Tensor.full] using hZero
  · intro hZero
    subst tensor
    simp [Tensor.full, mulSpec, map2Spec]

end Spec
