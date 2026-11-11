/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Euclidean
public import TorchLean.Tensor.Reduce.Real
public import Mathlib.Analysis.Calculus.FDeriv.Linear
public import Mathlib.Analysis.InnerProductSpace.Adjoint
public import TorchLean.NN.FDeriv.DerivCert

/-!
# Derivatives of tensor coordinate maps

Slicing, gathering, and broadcasting all read input coordinates through a fixed map. The map
need not be injective, and repeated reads are why the reverse pass must sum contributions.

These statements use `Tensor.Rep.pull`, the coordinate kernel underlying the tensor
representation, and the Euclidean topology already defined on real tensors, so each derivative is
stated on the tensor type itself. The coordinate formulas also apply to empty shapes, since they
pick no first element and make no hidden positivity assumption.
-/

@[expose] public section

namespace TorchLean.Tensor

open scoped BigOperators

noncomputable section

/-- The native coordinate pullback, bundled as a continuous linear map. -/
def Rep.pullCLM {source target : Shape} (index : target.Coord → source.Coord) :
    Tensor ℝ source →L[ℝ] Tensor ℝ target := by
  let linear : Tensor ℝ source →ₗ[ℝ] Tensor ℝ target :=
    { toFun := Rep.pull index
      map_add' := by
        intro x y
        apply Rep.ext
        intro coordinate
        simp
      map_smul' := by
        intro c x
        apply Rep.ext
        intro coordinate
        simp }
  exact ⟨linear, linear.continuous_of_finiteDimensional⟩

/-- The bundled map reads exactly the coordinates selected by the native kernel. -/
@[simp] theorem Rep.pullCLM_apply {source target : Shape}
    (index : target.Coord → source.Coord) (x : Tensor ℝ source) :
    pullCLM index x = Rep.pull index x := rfl

/-- A source coordinate receives the sum over all output coordinates which read it.

`index` deliberately carries no distinctness hypothesis: for a gather with repeated indices every
occurrence contributes, and for a broadcast every replicated batch contributes. -/
def Rep.sumFibers {source target : Shape} (index : target.Coord → source.Coord)
    (gradient : Tensor ℝ target) : Tensor ℝ source :=
  Rep.ofFn fun sourceCoordinate =>
    ∑ targetCoordinate, if index targetCoordinate = sourceCoordinate then
      gradient targetCoordinate else 0

/-- Coordinate pullback and summation over its fibers are adjoint. -/
theorem Rep.inner_pull_sumFibers {source target : Shape}
    (index : target.Coord → source.Coord)
    (x : Tensor ℝ source) (gradient : Tensor ℝ target) :
    inner ℝ (Rep.pull index x) gradient =
      inner ℝ x (sumFibers index gradient) := by
  classical
  simp only [Tensor.inner_eq_sum, Rep.pull_apply, sumFibers,
    Rep.get_ofFn, Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro coordinate _
  simp only [mul_ite, mul_zero]
  simp [eq_comm]

/-- The analytic adjoint of `pullCLM index` is the concrete accumulation `sumFibers index`. -/
theorem Rep.pullCLM_adjoint {source target : Shape}
    (index : target.Coord → source.Coord) (gradient : Tensor ℝ target) :
    (pullCLM index).adjoint gradient = sumFibers index gradient := by
  apply ext_inner_left ℝ
  intro x
  rw [ContinuousLinearMap.adjoint_inner_right, pullCLM_apply]
  exact inner_pull_sumFibers index x gradient

/-- The Euclidean inner product separates over the leading tensor axis. -/
theorem inner_eq_sum_unstack {n : Nat} {shape : Shape}
    (x y : Tensor ℝ (.dim n shape)) :
    inner ℝ x y = ∑ i : Fin n, inner ℝ (x.unstack i) (y.unstack i) := by
  simp only [Tensor.inner_eq_sum, Fintype.sum_prod_type,
    Tensor.unstack_apply]

end

end TorchLean.Tensor
