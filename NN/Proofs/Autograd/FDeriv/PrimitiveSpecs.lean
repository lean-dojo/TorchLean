/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.FDeriv.Elementwise
public import NN.Proofs.Autograd.Tape.Core.FDeriv

/-!
# Coordinatewise primitive specifications

A captured tensor can supply a different scalar function at every coordinate. Its derivative is
still diagonal: each input tangent is multiplied by the corresponding scalar derivative. Over the
reals this diagonal map is self-adjoint, which gives the reverse rule for the same scalar formula.

The final lemmas apply to an actual `OpSpec`. They require its forward and backward coordinate
formulas, as well as scalar derivative evidence at the input being differentiated. This keeps the
analytic step separate from an algebraic pairing identity and permits the necessary domain
conditions for reciprocals and piecewise functions. Flattening uses the existing tensor/vector
equivalence, so the statements apply to arbitrary tensor shapes, including empty ones.
-/

@[expose] public section

namespace Proofs.Autograd.PrimitiveSpecs

open Spec TorchLean
open scoped BigOperators

noncomputable section

/-- Apply a possibly different scalar function at each coordinate. -/
def coordinatewise {n : Nat} (f : Fin n → ℝ → ℝ) (x : Vec n) : Vec n :=
  (euclideanEquiv n).symm fun i => f i (x i)

/-- The diagonal linear map with the supplied scalar coefficients. -/
def coordinateDeriv {n : Nat} (coefficients : Fin n → ℝ) : Vec n →L[ℝ] Vec n :=
  (euclideanEquiv n).symm.toContinuousLinearMap.comp
    (ContinuousLinearMap.pi fun i => coefficients i • evalCLM i)

/-- A diagonal derivative multiplies each tangent by its own coefficient. -/
@[simp] theorem coordinateDeriv_apply {n : Nat} (coefficients : Fin n → ℝ)
    (dx : Vec n) (i : Fin n) :
    coordinateDeriv coefficients dx i = coefficients i * dx i := by
  simp [coordinateDeriv, euclideanEquiv, ContinuousLinearMap.comp_apply]

/-- Scalar derivatives at the input coordinates determine the derivative of the whole vector. -/
theorem hasFDerivAt_coordinatewise {n : Nat} (f : Fin n → ℝ → ℝ)
    (coefficients : Fin n → ℝ) (x : Vec n)
    (hf : ∀ i, HasDerivAt (f i) (coefficients i) (x i)) :
    HasFDerivAt (coordinatewise f) (coordinateDeriv coefficients) x := by
  have hcoord : ∀ i : Fin n,
      HasFDerivAt (fun y : Vec n => f i (y i)) (coefficients i • evalCLM i) x := by
    intro i
    simpa only [Function.comp_def, evalCLM_apply] using
      (hf i).comp_hasFDerivAt x ((evalCLM i).hasFDerivAt (x := x))
  have hpi :
      HasFDerivAt (fun y : Vec n => fun i => f i (y i))
        (ContinuousLinearMap.pi fun i => coefficients i • evalCLM i) x :=
    (hasFDerivAt_pi (𝕜 := ℝ)).2 hcoord
  exact (euclideanEquiv n).symm.toContinuousLinearMap.hasFDerivAt.comp x hpi

/-- Real diagonal maps are self-adjoint, independently of differentiability. -/
theorem coordinateDeriv_adjoint_apply {n : Nat} (coefficients : Fin n → ℝ) (δ : Vec n) :
    (coordinateDeriv coefficients).adjoint δ = coordinateDeriv coefficients δ := by
  apply ext_inner_left ℝ
  intro dx
  rw [ContinuousLinearMap.adjoint_inner_right]
  simp only [inner_eq_sum_mul]
  apply Finset.sum_congr rfl
  intro i _
  simp only [coordinateDeriv_apply]
  ring

/-- Transport coordinatewise calculus to the exact forward function stored by an `OpSpec`. -/
theorem hasFDerivAt_of_coordinates {s : Shape} (op : Spec.OpSpec ℝ s s)
    (f : Fin (Shape.size s) → ℝ → ℝ) (coefficients : Fin (Shape.size s) → ℝ)
    (x : Vec (Shape.size s))
    (hforward : ∀ y, tensorToVec (op.forward (vecToTensor y)) = coordinatewise f y)
    (hf : ∀ i, HasDerivAt (f i) (coefficients i) (x i)) :
    HasFDerivAt (fun y => tensorToVec (op.forward (vecToTensor y)))
      (coordinateDeriv coefficients) x := by
  have heq :
      (fun y => tensorToVec (op.forward (vecToTensor y))) = coordinatewise f :=
    funext hforward
  rw [heq]
  exact hasFDerivAt_coordinatewise f coefficients x hf

/-- Identify the stored backward with the adjoint of the actual forward derivative.

The backward formula is checked at the same input as the scalar derivative hypotheses. In
particular, supplying a selected slope at a kink does not establish a classical derivative there.
-/
theorem backward_eq_adjoint_of_coordinates {s : Shape} (op : Spec.OpSpec ℝ s s)
    (f : Fin (Shape.size s) → ℝ → ℝ) (coefficients : Fin (Shape.size s) → ℝ)
    (x : Vec (Shape.size s))
    (hforward : ∀ y, tensorToVec (op.forward (vecToTensor y)) = coordinatewise f y)
    (hf : ∀ i, HasDerivAt (f i) (coefficients i) (x i))
    (hbackward : ∀ δ, tensorToVec (op.backward (vecToTensor x) δ) =
      coordinateDeriv coefficients (tensorToVec δ))
    (δ : Tensor ℝ s) :
    tensorToVec (op.backward (vecToTensor x) δ) =
      (fderiv ℝ (fun y => tensorToVec (op.forward (vecToTensor y))) x).adjoint
        (tensorToVec δ) := by
  rw [(hasFDerivAt_of_coordinates op f coefficients x hforward hf).fderiv]
  rw [coordinateDeriv_adjoint_apply]
  exact hbackward δ

end

end Proofs.Autograd.PrimitiveSpecs
