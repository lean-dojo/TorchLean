/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Euclidean
public import NN.Proofs.Tensor.Basic.Folds
public import Mathlib.Analysis.Calculus.FDeriv.Linear
public import Mathlib.Analysis.InnerProductSpace.Adjoint

/-!
# Derivatives of tensor coordinate maps

Slicing, gathering, and broadcasting all read input coordinates through a fixed map. The map
need not be injective: repeated reads are precisely why the reverse pass must sum contributions.

These statements use `Rep.pull`, the coordinate kernel underlying the tensor representation, and
the existing Euclidean topology on real tensors. A derivative is therefore stated on the actual
tensor type. The coordinate formulas also apply to empty shapes; there is no chosen first element
or hidden positivity assumption.
-/

@[expose] public section

namespace Proofs.Autograd.Reindex

open Spec TorchLean
open TorchLean.Tensor
open scoped BigOperators

noncomputable section

/-- The native coordinate pullback, bundled as a continuous linear map. -/
def pullCLM {source target : Shape} (index : target.Coord → source.Coord) :
    Tensor ℝ source →L[ℝ] Tensor ℝ target := by
  let linear : Tensor ℝ source →ₗ[ℝ] Tensor ℝ target :=
    { toFun := Internal.Rep.pull index
      map_add' := by
        intro x y
        apply Internal.Rep.ext
        intro coordinate
        simp
      map_smul' := by
        intro c x
        apply Internal.Rep.ext
        intro coordinate
        simp }
  exact ⟨linear, linear.continuous_of_finiteDimensional⟩

/-- The bundled map reads exactly the coordinates selected by the native kernel. -/
@[simp] theorem pullCLM_apply {source target : Shape}
    (index : target.Coord → source.Coord) (x : Tensor ℝ source) :
    pullCLM index x = Internal.Rep.pull index x := rfl

/-- Reindexing is linear, so the same coordinate map acts on every input tangent. -/
theorem hasFDerivAt_pull {source target : Shape}
    (index : target.Coord → source.Coord) (x : Tensor ℝ source) :
    HasFDerivAt (Internal.Rep.pull index) (pullCLM index) x :=
  (pullCLM index).hasFDerivAt

/-- A source coordinate receives the sum over all output coordinates which read it.

There is deliberately no distinctness hypothesis on `index`. For a gather with repeated indices,
every occurrence contributes; for a broadcast, every replicated batch contributes. -/
def sumFibers {source target : Shape} (index : target.Coord → source.Coord)
    (gradient : Tensor ℝ target) : Tensor ℝ source :=
  Internal.Rep.ofFn fun sourceCoordinate =>
    ∑ targetCoordinate, if index targetCoordinate = sourceCoordinate then
      gradient targetCoordinate else 0

/-- Coordinate pullback and summation over its fibers are adjoint. -/
theorem inner_pull_sumFibers {source target : Shape}
    (index : target.Coord → source.Coord)
    (x : Tensor ℝ source) (gradient : Tensor ℝ target) :
    inner ℝ (Internal.Rep.pull index x) gradient =
      inner ℝ x (sumFibers index gradient) := by
  classical
  simp only [Tensor.inner_eq_sum, Internal.Rep.pull_apply, sumFibers,
    Internal.Rep.get_ofFn, Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro coordinate _
  simp only [mul_ite, mul_zero]
  simp [eq_comm]

/-- The analytic adjoint is the concrete accumulation formula, not an unspecified reduction. -/
theorem pullCLM_adjoint {source target : Shape}
    (index : target.Coord → source.Coord) (gradient : Tensor ℝ target) :
    (pullCLM index).adjoint gradient = sumFibers index gradient := by
  apply ext_inner_left ℝ
  intro x
  rw [ContinuousLinearMap.adjoint_inner_right, pullCLM_apply]
  exact inner_pull_sumFibers index x gradient

/-- The real spec dot product uses the same inner product as tensor Fréchet derivatives. -/
theorem dot_eq_inner {shape : Shape} (x y : Tensor ℝ shape) :
    Spec.dot x y = inner ℝ x y := by
  rw [Spec.dot, Spec.sum_spec_eq_coord_sum, Tensor.inner_eq_sum]
  apply Finset.sum_congr rfl
  intro coordinate _
  exact Internal.Rep.zipWith_apply (· * ·) x y coordinate

/-- The Euclidean inner product separates over the leading tensor axis. -/
theorem inner_eq_sum_unstack {n : Nat} {shape : Shape}
    (x y : Tensor ℝ (.dim n shape)) :
    inner ℝ x y = ∑ i : Fin n, inner ℝ (x.unstack i) (y.unstack i) := by
  simp only [Tensor.inner_eq_sum, Fintype.sum_prod_type,
    Tensor.unstack, Internal.Rep.unstack_apply]

end

end Proofs.Autograd.Reindex
