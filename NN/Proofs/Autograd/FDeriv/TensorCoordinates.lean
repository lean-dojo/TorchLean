/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.FDeriv.Reindex
public import Mathlib.Analysis.Calculus.FDeriv.Prod

/-!
# Coordinate derivatives on real tensors

The tensor representation already carries its Euclidean norm. Reading its coordinates is a
continuous linear equivalence with a finite function space, so coordinatewise derivative proofs
give Fréchet derivatives on the tensor itself. In particular, a matrix proof can keep its row and
column indices instead of introducing a second flattened layout.

The coordinate function space is only used to assemble the derivative. The result has the native
tensor type and the Euclidean topology used by the tensor inner-product and adjoint theorems.
-/

@[expose] public section

namespace Proofs.Autograd.TensorCoordinates

open Spec TorchLean
open TorchLean.Tensor

noncomputable section

/-- The native coordinate view, with continuity supplied by finite dimensionality. -/
def coordinateEquiv (shape : Shape) : Tensor ℝ shape ≃L[ℝ] (shape.Coord → ℝ) :=
  (Tensor.toEuclidean shape.toList).toContinuousLinearEquiv.trans
    (EuclideanSpace.equiv (𝕜 := ℝ) (ι := shape.Coord))

/-- Reading the coordinate view performs the same lookup as the tensor representation. -/
@[simp] theorem coordinateEquiv_apply {shape : Shape}
    (x : Tensor ℝ shape) (i : shape.Coord) :
    coordinateEquiv shape x i = x i := rfl

/-- Rebuilding a tensor preserves each supplied coordinate. -/
@[simp] theorem coordinateEquiv_symm_apply {shape : Shape}
    (x : shape.Coord → ℝ) (i : shape.Coord) :
    (coordinateEquiv shape).symm x i = x i := by
  exact Internal.Rep.get_ofFn _ i

/-- A single tensor coordinate is a continuous linear functional. -/
def coordinateCLM {shape : Shape} (i : shape.Coord) : Tensor ℝ shape →L[ℝ] ℝ :=
  (ContinuousLinearMap.proj i).comp (coordinateEquiv shape).toContinuousLinearMap

/-- The coordinate functional uses the native lookup. -/
@[simp] theorem coordinateCLM_apply {shape : Shape}
    (i : shape.Coord) (x : Tensor ℝ shape) :
    coordinateCLM i x = x i := rfl

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Assemble one scalar derivative for each output coordinate into a tensor derivative. -/
def assemble {shape : Shape} (derivatives : shape.Coord → E →L[ℝ] ℝ) :
    E →L[ℝ] Tensor ℝ shape :=
  (coordinateEquiv shape).symm.toContinuousLinearMap.comp
    (ContinuousLinearMap.pi derivatives)

/-- Each coordinate of the assembled tangent is given by its supplied scalar derivative. -/
@[simp] theorem assemble_apply {shape : Shape}
    (derivatives : shape.Coord → E →L[ℝ] ℝ) (dx : E) (i : shape.Coord) :
    assemble derivatives dx i = derivatives i dx := by
  exact coordinateEquiv_symm_apply _ i

/-- Scalar coordinate derivatives determine the full tensor Fréchet derivative.

Finiteness of the shape matters here: assembling the coordinates is a continuous linear map.
The hypotheses therefore imply a Fréchet derivative in the tensor norm, not just separate
directional derivatives of its entries. -/
theorem hasFDerivAt_of_coordinates {shape : Shape} (f : E → Tensor ℝ shape)
    (derivatives : shape.Coord → E →L[ℝ] ℝ) (x : E)
    (h : ∀ i, HasFDerivAt (fun y => f y i) (derivatives i) x) :
    HasFDerivAt f (assemble derivatives) x := by
  have hpi := (hasFDerivAt_pi (𝕜 := ℝ)
    (φ := fun i y => f y i) (φ' := derivatives) (x := x)).2 h
  have hcomp :=
    (coordinateEquiv shape).symm.toContinuousLinearMap.hasFDerivAt.comp x hpi
  have hfun :
      f = (coordinateEquiv shape).symm ∘ (fun y i => f y i) := by
    funext y
    apply Internal.Rep.ext
    intro i
    exact (coordinateEquiv_symm_apply _ i).symm
  exact hcomp.congr_of_eventuallyEq hfun.eventuallyEq

end

end Proofs.Autograd.TensorCoordinates
