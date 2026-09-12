/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.InnerProductSpace.PiL2
public import NN.Spec.Core.Tensor.Core
public import NN.Spec.Core.Tensor -- shake: keep

/-!
# Real tensors as Euclidean space

A real tensor of shape `s` is a finite family of reals indexed by `s.Coord`. This file transports
the Euclidean inner-product structure of `EuclideanSpace ℝ s.Coord` along the coordinate view, so
that `Tensor ℝ s` is a `NormedAddCommGroup` and an `InnerProductSpace ℝ` with

- `‖x‖ = √(∑ i, x i ^ 2)` (`TorchLean.Tensor.norm_eq_sqrt_sum`),
- `⟪x, y⟫_ℝ = ∑ i, x i * y i` (`TorchLean.Tensor.inner_eq_sum`).

All Mathlib facts about inner product spaces (Cauchy-Schwarz, the triangle inequality,
`norm_smul`, `LipschitzWith`, finite dimensionality) then apply to real tensors directly. The
metric topology this induces is the one every analytic statement about real tensors should use;
`continuous_vectorEquiv` shows the rank-one coordinate view is a homeomorphism for it.
-/

@[expose] public section

open Spec TorchLean
open scoped RealInnerProductSpace

namespace TorchLean.Tensor

/-!
The instances are stated on the native representation `Internal.Rep ℝ s` (with `s : List Nat`)
rather than on `Tensor ℝ shape`: the public type is a reducible abbreviation whose shape argument
unfolds to a dimension list, so only representation-level instances are found for literal shapes
such as `Tensor ℝ [n]`.
-/

variable {s : Internal.Shape}

/-- The coordinate view of a real tensor as a point of Euclidean space. -/
noncomputable def toEuclidean (s : Internal.Shape) :
    Internal.Rep ℝ s ≃ₗ[ℝ] EuclideanSpace ℝ (Internal.Coord s) where
  toFun x := WithLp.toLp 2 fun i => x i
  invFun v := Internal.Rep.ofFn fun i => v.ofLp i
  map_add' x y := by
    apply PiLp.ext
    intro i
    simp only [Internal.Rep.hAdd_apply, PiLp.add_apply]
  map_smul' c x := by
    apply PiLp.ext
    intro i
    simp only [Internal.Rep.smul_apply, PiLp.smul_apply, RingHom.id_apply, smul_eq_mul]
  left_inv x := by
    apply Internal.Rep.ext
    intro i
    simp only [Internal.Rep.get_ofFn]
  right_inv v := by
    apply PiLp.ext
    intro i
    simp only [Internal.Rep.get_ofFn]

/-- Coordinates of the Euclidean view are the tensor's coordinates. -/
@[simp] theorem toEuclidean_apply (x : Internal.Rep ℝ s) (i : Internal.Coord s) :
    (toEuclidean s x).ofLp i = x i :=
  rfl

/-- Coordinates of a tensor rebuilt from Euclidean space are the point's coordinates. -/
@[simp] theorem toEuclidean_symm_apply (v : EuclideanSpace ℝ (Internal.Coord s))
    (i : Internal.Coord s) : (toEuclidean s).symm v i = v.ofLp i :=
  Internal.Rep.get_ofFn _ i

/-- The Euclidean norm on real tensors. -/
noncomputable instance : NormedAddCommGroup (Internal.Rep ℝ s) :=
  NormedAddCommGroup.induced (Internal.Rep ℝ s) (EuclideanSpace ℝ (Internal.Coord s))
    (toEuclidean s) (toEuclidean s).injective

/-- The Euclidean inner product on real tensors. -/
noncomputable instance : InnerProductSpace ℝ (Internal.Rep ℝ s) :=
  InnerProductSpace.induced (toEuclidean s)

/-- Real tensors of a fixed shape form a finite-dimensional real vector space. -/
instance : FiniteDimensional ℝ (Internal.Rep ℝ s) :=
  (toEuclidean s).symm.finiteDimensional

/-- The tensor norm is the norm of its Euclidean view. -/
theorem norm_eq_norm_toEuclidean (x : Internal.Rep ℝ s) : ‖x‖ = ‖toEuclidean s x‖ :=
  rfl

/-- The tensor inner product is the inner product of the Euclidean views. -/
theorem inner_eq_inner_toEuclidean (x y : Internal.Rep ℝ s) :
    ⟪x, y⟫ = ⟪toEuclidean s x, toEuclidean s y⟫ :=
  rfl

/-- The inner product of two real tensors is the sum of coordinatewise products. -/
theorem inner_eq_sum (x y : Internal.Rep ℝ s) : ⟪x, y⟫ = ∑ i, x i * y i := by
  rw [inner_eq_inner_toEuclidean, PiLp.inner_apply]
  simp only [toEuclidean_apply, RCLike.inner_apply, conj_trivial, mul_comm]

/-- The norm of a real tensor is the square root of its sum of squares. -/
theorem norm_eq_sqrt_sum (x : Internal.Rep ℝ s) : ‖x‖ = Real.sqrt (∑ i, x i ^ 2) := by
  rw [norm_eq_norm_toEuclidean, EuclideanSpace.norm_eq]
  simp only [toEuclidean_apply, Real.norm_eq_abs, sq_abs]

/-- The squared norm of a real tensor is its sum of squares. -/
theorem norm_sq_eq_sum (x : Internal.Rep ℝ s) : ‖x‖ ^ 2 = ∑ i, x i ^ 2 := by
  rw [norm_eq_norm_toEuclidean, EuclideanSpace.norm_sq_eq]
  simp only [toEuclidean_apply, Real.norm_eq_abs, sq_abs]

/-- Every coordinate of a real tensor is bounded by the Euclidean norm. -/
theorem abs_apply_le_norm (x : Internal.Rep ℝ s) (i : Internal.Coord s) : |x i| ≤ ‖x‖ := by
  have h := PiLp.norm_apply_le (toEuclidean s x) i
  simpa only [toEuclidean_apply, Real.norm_eq_abs, norm_eq_norm_toEuclidean] using h

/-- The distance between real tensors is the square root of the summed squared differences. -/
theorem dist_eq_sqrt_sum (x y : Internal.Rep ℝ s) :
    dist x y = Real.sqrt (∑ i, (x i - y i) ^ 2) := by
  rw [dist_eq_norm, norm_eq_sqrt_sum]
  simp only [Internal.Rep.hSub_apply]

/-- The rank-one coordinate view as a real linear equivalence. -/
noncomputable def vectorLinearEquiv (n : Nat) : Tensor ℝ [n] ≃ₗ[ℝ] (Fin n → ℝ) :=
  { vectorEquiv (α := ℝ) n with
    map_add' := fun x y => by
      funext i
      show getScalar (x + y) i = getScalar x i + getScalar y i
      simp only [getScalar_eq_apply, Internal.Rep.hAdd_apply]
    map_smul' := fun c x => by
      funext i
      show getScalar (c • x) i = c • getScalar x i
      simp only [getScalar_eq_apply, Internal.Rep.smul_apply] }

/-- The rank-one coordinate view is continuous for the Euclidean topology. -/
theorem continuous_vectorEquiv (n : Nat) : Continuous (vectorEquiv (α := ℝ) n) :=
  (vectorLinearEquiv n).toLinearMap.continuous_of_finiteDimensional

/-- Rebuilding a rank-one tensor from its coordinates is continuous. -/
theorem continuous_vectorEquiv_symm (n : Nat) :
    Continuous (vectorEquiv (α := ℝ) n).symm :=
  (vectorLinearEquiv n).symm.toLinearMap.continuous_of_finiteDimensional

end TorchLean.Tensor
