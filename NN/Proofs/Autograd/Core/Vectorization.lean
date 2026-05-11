/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Basic

public import Mathlib.Algebra.BigOperators.Ring.Finset
public import Mathlib.Analysis.InnerProductSpace.PiL2

/-!
# Vectorization

Shared Euclidean-space vectorization utilities for analytic autograd proofs.

This module centralizes the `Vec` alias (`EuclideanSpace ℝ (Fin n)`) and the basic
`Tensor ℝ (.dim n .scalar)` ↔ `Vec n` conversions used across multiple proof files.

## PyTorch correspondence / citations
This plays the same role as treating a length-`n` tensor as an element of `ℝ^n` when using
standard analysis results (mean value theorem, operator norms, etc.).
https://pytorch.org/docs/stable/linalg.html
-/

@[expose] public section


namespace Proofs
namespace Autograd

open Spec
open Tensor

open scoped BigOperators

noncomputable section

/-- Euclidean vectors over `ℝ`. -/
abbrev Vec (n : Nat) := EuclideanSpace ℝ (Fin n)

/--
Convert a 1D scalar tensor (`Tensor ℝ (.dim n .scalar)`) into a Euclidean vector `Vec n`.

This is the “analysis-friendly” view of a length-`n` tensor as an element of `ℝ^n`.
-/
def toVecE {n : Nat} (t : Tensor ℝ (.dim n .scalar)) : Vec n :=
  (EuclideanSpace.equiv (𝕜 := ℝ) (ι := Fin n)).symm (Spec.toVec t)

/--
Convert a Euclidean vector `Vec n` back into a 1D scalar tensor (`Tensor ℝ (.dim n .scalar)`).

This is the inverse direction of `toVecE`.
-/
def ofVecE {n : Nat} (v : Vec n) : Tensor ℝ (.dim n .scalar) :=
  Spec.ofVec ((EuclideanSpace.equiv (𝕜 := ℝ) (ι := Fin n)) v)

/-- `toVecE` is a left inverse of `ofVecE`. -/
@[simp] lemma toVecE_ofVecE {n : Nat} (v : Vec n) : toVecE (ofVecE v) = v := by
  classical
  let e := EuclideanSpace.equiv (𝕜 := ℝ) (ι := Fin n)
  have h :=
      congrArg e.symm (Spec.toVec_ofVec (n := n) (v := e v)) |>
        (·.trans (e.symm_apply_apply v))
  simp [toVecE, ofVecE]

/-- `ofVecE` is a left inverse of `toVecE`. -/
@[simp] lemma ofVecE_toVecE {n : Nat} (t : Tensor ℝ (.dim n .scalar)) : ofVecE (toVecE t) = t := by
  classical
  let e := EuclideanSpace.equiv (𝕜 := ℝ) (ι := Fin n)
  have h :=
      congrArg Spec.ofVec (e.apply_symm_apply (Spec.toVec t)) |>
        (·.trans (Spec.ofVec_toVec (t := t)))
  simpa [toVecE, ofVecE, e] using h

/--
Coordinate formula for the Euclidean inner product on `Vec n`.

This is the statement “`⟪x,y⟫ = ∑ᵢ xᵢ*yᵢ`” specialized to `EuclideanSpace ℝ (Fin n)`.
-/
lemma inner_eq_sum_mul {n : Nat} (x y : Vec n) :
    inner ℝ x y = ∑ i : Fin n, x i * y i := by
  classical
  simpa [Vec, dotProduct, mul_comm] using
    (EuclideanSpace.inner_eq_star_dotProduct (x := x) (y := y))

end
end Autograd
end Proofs
