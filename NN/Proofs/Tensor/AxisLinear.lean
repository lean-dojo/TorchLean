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

/-- Source coordinate read by one adjacent-axis swap. -/
def swapCoordinate : (s : Shape) → (depth : Nat) →
    Shape.Coord (s.swapAdjacentAtDepth depth) → Shape.Coord s
  | .dim _ (.dim _ _), 0, p => (p.2.1, p.1, p.2.2)
  | .dim _ s, depth + 1, p => (p.1, swapCoordinate s depth p.2)
  | .scalar, 0, p => p
  | .scalar, _ + 1, p => p
  | .dim _ .scalar, 0, p => p

/-- The tensor swap reads exactly the coordinate selected by `swapCoordinate`. -/
theorem swapAdjacentAxes_apply {α : Type} [TorchLean.Storage α] {s : Shape}
    (x : Tensor α s) (depth : Nat) (p : Shape.Coord (s.swapAdjacentAtDepth depth)) :
    swapAdjacentAxes x depth p = x (swapCoordinate s depth p) := by
  induction depth generalizing s with
  | zero =>
      cases s with
      | scalar => rfl
      | dim n s =>
          cases s with
          | scalar => rfl
          | dim m s =>
              simp only [swapAdjacentAxes, Tensor.dim,
                TorchLean.Tensor.Internal.Rep.stack_apply, Tensor.unstack,
                TorchLean.Tensor.Internal.Rep.unstack_apply, swapCoordinate]
  | succ depth ih =>
      cases s with
      | scalar => rfl
      | dim n s =>
          change Tensor.dim (fun i => swapAdjacentAxes (x.unstack i) depth) p = _
          simp only [Tensor.dim, TorchLean.Tensor.Internal.Rep.stack_apply]
          rw [ih]
          simpa only [Tensor.unstack, swapCoordinate] using
            TorchLean.Tensor.Internal.Rep.unstack_apply x p.1 (swapCoordinate s depth p.2)

/-- Source coordinate read after a sequence of adjacent-axis swaps. -/
def permutedCoordinate : (s : Shape) → (swaps : List Nat) →
    Shape.Coord (s.applyAdjacentSwaps swaps) → Shape.Coord s
  | _, [], p => p
  | s, depth :: swaps, p =>
      swapCoordinate s depth
        (permutedCoordinate (s.swapAdjacentAtDepth depth) swaps p)

/-- The coordinate description follows the actual tensor permutation at every rank. -/
theorem permuteByAdjacentSwaps_apply {α : Type} [TorchLean.Storage α] {s : Shape}
    (x : Tensor α s) (swaps : List Nat) (p : Shape.Coord (s.applyAdjacentSwaps swaps)) :
    permuteByAdjacentSwaps x swaps p = x (permutedCoordinate s swaps p) := by
  induction swaps generalizing s with
  | nil => rfl
  | cons depth swaps ih =>
      change Shape.Coord ((s.swapAdjacentAtDepth depth).applyAdjacentSwaps swaps) at p
      change permuteByAdjacentSwaps (swapAdjacentAxes x depth) swaps p =
        x (swapCoordinate s depth
          (permutedCoordinate (s.swapAdjacentAtDepth depth) swaps p))
      exact (ih (swapAdjacentAxes x depth) p).trans (swapAdjacentAxes_apply x depth _)

/-- Axis permutation on finite coordinate functions, bundled as a continuous linear map. -/
def permuteCLM {s : Shape} (swaps : List Nat) :
    (Shape.Coord s → ℝ) →L[ℝ] (Shape.Coord (s.applyAdjacentSwaps swaps) → ℝ) := by
  let linear :
      (Shape.Coord s → ℝ) →ₗ[ℝ] (Shape.Coord (s.applyAdjacentSwaps swaps) → ℝ) :=
    { toFun := fun x p => x (permutedCoordinate s swaps p)
      map_add' := fun _ _ => rfl
      map_smul' := fun _ _ => rfl }
  exact ⟨linear, LinearMap.continuous_of_finiteDimensional (f := linear)⟩

/-- The continuous linear map computes the library's tensor permutation. -/
theorem permuteCLM_apply {s : Shape} (swaps : List Nat) (x : Shape.Coord s → ℝ) :
    permuteCLM swaps x =
      fun p => permuteByAdjacentSwaps (TorchLean.Tensor.Internal.Rep.ofFn x) swaps p := by
  funext p
  rw [permuteByAdjacentSwaps_apply]
  exact (TorchLean.Tensor.Internal.Rep.get_ofFn x _).symm

/-- Transporting coordinate functions through a shape equality is continuous and linear. -/
def castCLM {s t : Shape} (h : s = t) :
    (Shape.Coord s → ℝ) →L[ℝ] (Shape.Coord t → ℝ) := by
  cases h
  exact ContinuousLinearMap.id ℝ _

/-- The coordinate transport agrees with the shape transport used by the tensor API. -/
theorem castCLM_apply {s t : Shape} (h : s = t) (x : Tensor ℝ s) :
    castCLM h (fun p => x p) = fun p => (h ▸ x) p := by
  cases h
  rfl

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
