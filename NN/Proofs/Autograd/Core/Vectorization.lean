/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.InnerProductSpace.PiL2
public import NN.Spec.Core.Context.Real
public import NN.Spec.Core.Tensor.Constructors

/-!
# Vectorization

Shared Euclidean-space vectorization utilities for analytic autograd proofs.

This module centralizes the `Vec` alias (`EuclideanSpace ℝ (Fin n)`) and the basic
`Tensor ℝ [n]` ↔ `Vec n` conversions used across multiple proof files.

## PyTorch correspondence / citations
This plays the same role as treating a length-`n` tensor as an element of $\mathbb R^n$ when using
standard analysis results (mean value theorem, operator norms, etc.).
https://pytorch.org/docs/stable/linalg.html
-/

@[expose] public section


namespace Proofs
namespace Autograd

open Spec TorchLean
open TorchLean TorchLean.Tensor

open scoped BigOperators

noncomputable section

/-- Euclidean vectors over `ℝ`. -/
abbrev Vec (n : Nat) := EuclideanSpace ℝ (Fin n)

/--
Convert a rank-one tensor (`Tensor ℝ [n]`) into a Euclidean vector `Vec n`.

This is the “analysis-friendly” view of a length-`n` tensor as an element of $\mathbb R^n$.
-/
def getScalarE {n : Nat} (t : Tensor ℝ [n]) : Vec n :=
  (EuclideanSpace.equiv (𝕜 := ℝ) (ι := Fin n)).symm (TorchLean.Tensor.getScalar t)

/-- Coordinate evaluation of the Euclidean view of a rank-one tensor. -/
@[simp] theorem getScalarE_ofLp {n : Nat} (t : Tensor ℝ [n]) (i : Fin n) :
    (getScalarE t).ofLp i = TorchLean.Tensor.getScalar t i := by
  simp [getScalarE, EuclideanSpace.equiv]

/--
Convert a Euclidean vector `Vec n` back into a rank-one tensor (`Tensor ℝ [n]`).

This is the inverse direction of `getScalarE`.
-/
def ofFnE {n : Nat} (v : Vec n) : Tensor ℝ [n] :=
  TorchLean.Tensor.ofFn ((EuclideanSpace.equiv (𝕜 := ℝ) (ι := Fin n)) v)

/-- `getScalarE` is a left inverse of `ofFnE`. -/
@[simp] theorem getScalarE_ofFnE {n : Nat} (v : Vec n) : getScalarE (ofFnE v) = v := by
  classical
  let e := EuclideanSpace.equiv (𝕜 := ℝ) (ι := Fin n)
  change e.symm (fun i => (TorchLean.Tensor.ofFn (e v)).getScalar i) = v
  rw [show (fun i => (TorchLean.Tensor.ofFn (e v)).getScalar i) = e v by
    funext i
    simp]
  exact e.symm_apply_apply v

/-- Coordinate evaluation commutes with conversion from a Euclidean vector. -/
@[simp] theorem getScalar_ofFnE {n : Nat} (v : Vec n) (i : Fin n) :
    TorchLean.Tensor.getScalar (ofFnE v) i = v.ofLp i := by
  rw [← getScalarE_ofLp, getScalarE_ofFnE]

/-- `ofFnE` is a left inverse of `getScalarE`. -/
@[simp] theorem ofFnE_getScalarE {n : Nat} (t : Tensor ℝ [n]) : ofFnE (getScalarE t) = t := by
  classical
  let e := EuclideanSpace.equiv (𝕜 := ℝ) (ι := Fin n)
  change TorchLean.Tensor.ofFn (e (e.symm (fun i => t.getScalar i))) = t
  rw [e.apply_symm_apply]
  exact TorchLean.Tensor.ofFn_getScalar t

/--
Coordinate formula for the Euclidean inner product on `Vec n`.

This is the statement $\langle x,y\rangle=\sum_i x_i y_i$ specialized to
`EuclideanSpace ℝ (Fin n)`.
-/
theorem inner_eq_sum_mul {n : Nat} (x y : Vec n) :
    inner ℝ x y = ∑ i : Fin n, x i * y i := by
  classical
  simpa [Vec, dotProduct, mul_comm] using
    (EuclideanSpace.inner_eq_star_dotProduct (x := x) (y := y))

end
end Autograd
end Proofs
