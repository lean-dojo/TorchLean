/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape.LinearAlgebra
public import Mathlib.Analysis.Calculus.FDeriv.Linear
public import Mathlib.Analysis.Normed.Module.FiniteDimension

/-!
# Linear axis permutations

The softmax axis wrapper moves the selected axis to the last position, applies the row kernel,
and restores the axes. Each move only changes which input coordinate an output coordinate reads.
Its derivative therefore applies the same permutation to the input direction.

The coordinate maps below follow `swapAdjacentAxes` and `permuteByAdjacentSwaps` exactly. They
also retain the executable convention for an invalid swap depth: that swap leaves the tensor
unchanged. No nonempty-axis assumption is needed.
-/

@[expose] public section

namespace Proofs.TensorAxis

open Spec TorchLean
open TorchLean.Tensor

noncomputable section

/-- Source coordinate read by one adjacent-axis swap.

The derivative proofs and the compiled tensor operation use the same coordinate
map. This name keeps the coordinate description available in the proof API. -/
abbrev swapCoordinate (s : Shape) (depth : Nat) :
    Shape.Coord (s.swapAdjacentAtDepth depth) → Shape.Coord s :=
  TorchLean.Tensor.Internal.swapAdjacentAxesCoordinate s depth

/-- Source coordinate read after a sequence of adjacent-axis swaps. -/
abbrev permutedCoordinate (s : Shape) (swaps : List Nat) :
    Shape.Coord (s.applyAdjacentSwaps swaps) → Shape.Coord s :=
  TorchLean.Tensor.Internal.permutedCoordinate s swaps

/-- The coordinate description follows the actual tensor permutation at every rank. -/
theorem permuteByAdjacentSwaps_apply {α : Type} [TorchLean.Storage α] {s : Shape}
    (x : Tensor α s) (swaps : List Nat) (p : Shape.Coord (s.applyAdjacentSwaps swaps)) :
    permuteByAdjacentSwaps x swaps p = x (permutedCoordinate s swaps p) :=
  TorchLean.Tensor.permuteByAdjacentSwaps_apply x swaps p

/-- Axis permutation on finite coordinate functions, bundled as a continuous linear map.

This is the pullback `x ↦ x ∘ permutedCoordinate s swaps`, which is linear for free
(`LinearMap.funLeft`) and continuous because the domain is finite-dimensional. -/
def permuteCLM {s : Shape} (swaps : List Nat) :
    (Shape.Coord s → ℝ) →L[ℝ] (Shape.Coord (s.applyAdjacentSwaps swaps) → ℝ) :=
  LinearMap.toContinuousLinearMap (LinearMap.funLeft ℝ ℝ (permutedCoordinate s swaps))

/-- The continuous linear map computes the library's tensor permutation. -/
theorem permuteCLM_apply {s : Shape} (swaps : List Nat) (x : Shape.Coord s → ℝ) :
    permuteCLM swaps x =
      fun p => permuteByAdjacentSwaps (TorchLean.Tensor.Internal.Rep.ofFn x) swaps p := by
  funext p
  rw [permuteByAdjacentSwaps_apply]
  exact (TorchLean.Tensor.Internal.Rep.get_ofFn x _).symm

/-- Differentiating the actual tensor permutation permutes its input direction. -/
theorem hasFDerivAt_permuteByAdjacentSwaps {s : Shape}
    (swaps : List Nat) (x : Shape.Coord s → ℝ) :
    HasFDerivAt
      (fun z : Shape.Coord s → ℝ =>
        fun p => permuteByAdjacentSwaps (TorchLean.Tensor.Internal.Rep.ofFn z) swaps p)
      (permuteCLM swaps) x := by
  have hfun :
      (fun z : Shape.Coord s → ℝ =>
        fun p => permuteByAdjacentSwaps (TorchLean.Tensor.Internal.Rep.ofFn z) swaps p) =
      (permuteCLM swaps : (Shape.Coord s → ℝ) →
        (Shape.Coord (s.applyAdjacentSwaps swaps) → ℝ)) := by
    funext z
    exact (permuteCLM_apply swaps z).symm
  rw [hfun]
  exact (permuteCLM (s := s) swaps).hasFDerivAt (x := x)

end

end Proofs.TensorAxis
