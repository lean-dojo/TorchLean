/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Ops.Norm.MatrixEntries

/-!
# LayerNorm JVP and VJP adjointness

`Spec.layerNormJvp` is the forward-mode rule and `Spec.layerNormBackward` the reverse-mode rule
shipped with `Spec.layerNorm`. This file proves that they are adjoint: pairing an input and
parameter tangent with the JVP gives the same scalar as pairing the upstream gradient with the
three backward outputs. `LayerNormFDeriv` identifies this JVP with the derivative of the actual
specification for positive epsilon, then uses this pairing to prove that the backward rule is
the adjoint derivative. The pairing itself is algebraic and allows any real epsilon.

The proof is entrywise. Both spec functions are unfolded to explicit matrix forms (`rfl`), every
entry is rewritten as a real expression, and the per-row identity `row_adjoint` finishes with
`ring`.
-/

@[expose] public section

namespace Proofs
namespace Autograd
namespace LayerNorm

open Spec TorchLean
open TorchLean.Tensor
open Proofs.Autograd.Norm

open scoped BigOperators

noncomputable section

variable {m n : Nat}

/-! ## Entry lemmas -/

/-- Entries of a pointwise vector product. -/
theorem getScalar_mulSpec' {k : Nat} (a b : Tensor ℝ [k]) (i : Fin k) :
    getScalar (mulSpec a b) i = getScalar a i * getScalar b i := by
  simp [mulSpec]

/-- Column sums along the sequence axis. -/
theorem getScalar_reduceSum_zero (x : Tensor ℝ [m, n])
    (h : Shape.NonemptyAxis 0 (.dim m (.dim n .scalar))) (j : Fin n) :
    getScalar (reduceSum 0 x h) j = ∑ i : Fin m, Spec.get2 x i j := by
  rw [← Tensor.dim_unstack x]
  simp only [reduceSum, reduceDim, TorchLean.Tensor.Reduction.Internal.reduceDimCore_dim_zero,
    TorchLean.Tensor.Reduction.Internal.reduceOuterAxis, TorchLean.Tensor.getScalar_dim_entry,
    Tensor.item_scalar, Tensor.unstack_dim]
  rw [sum_spec_vec]
  apply Finset.sum_congr rfl
  intro i _
  simp [Spec.get2, Spec.get, TorchLean.Tensor.getScalar]

/-! ## Explicit matrix forms -/

/-- Row broadcast of a length-`m` vector to an `m × n` matrix. -/
abbrev bAS (v : Tensor ℝ [m]) : Tensor ℝ [m, n] :=
  broadcastAfterSum (.dim m (.dim n .scalar)) 1 v

/-- `1 / std`, as computed by the spec. -/
def lnInvStd (x : Tensor ℝ [m, n]) (h1 : Shape.NonemptyAxis 1 (.dim m (.dim n .scalar)))
    (ε : ℝ) : Tensor ℝ [m] :=
  divSpec (Tensor.full (.dim m .scalar) 1) (lnStd x h1 ε)

/-- `Spec.layerNormJvp` with its axis, broadcast, and cast evidence spelled out. -/
def layerNormJvpMat (x tangent : Tensor ℝ [m, n]) (gamma dgamma dbeta : Tensor ℝ [n])
    (h1 : Shape.NonemptyAxis 1 (.dim m (.dim n .scalar))) (ε : ℝ) : Tensor ℝ [m, n] :=
  let inv_std_b : Tensor ℝ [m, n] := bAS (lnInvStd x h1 ε)
  let norm := mulSpec (lnCentered x h1) inv_std_b
  let mean_tangent := divSpec (reduceSum 1 tangent h1) (Tensor.full (.dim m .scalar) (n : ℝ))
  let mean_tangent_norm :=
    divSpec (reduceSum 1 (mulSpec tangent norm) h1) (Tensor.full (.dim m .scalar) (n : ℝ))
  let dnorm :=
    mulSpec inv_std_b
      (subSpec (subSpec tangent (bAS mean_tangent)) (mulSpec norm (bAS mean_tangent_norm)))
  addSpec
    (addSpec (mulSpec dnorm (broadcastTo (colBroadcast (m := m) (n := n)) gamma))
      (mulSpec norm (broadcastTo (colBroadcast (m := m) (n := n)) dgamma)))
    (broadcastTo (colBroadcast (m := m) (n := n)) dbeta)

/-- `Spec.layerNormBackward` with its axis, broadcast, and cast evidence spelled out. -/
def layerNormBackwardMat (x : Tensor ℝ [m, n]) (gamma : Tensor ℝ [n]) (grad : Tensor ℝ [m, n])
    (h1 : Shape.NonemptyAxis 1 (.dim m (.dim n .scalar)))
    (h0 : Shape.NonemptyAxis 0 (.dim m (.dim n .scalar))) (ε : ℝ) :
    NormalizationGradients ℝ [m, n] [n] :=
  let norm := divSpec (lnCentered x h1) (bAS (lnStd x h1 ε))
  let biasGradient := reduceSum 0 grad h0
  let scaleGradient := reduceSum 0 (mulSpec grad norm) h0
  let dy_gamma := mulSpec grad (broadcastTo (colBroadcast (m := m) (n := n)) gamma)
  let mean_dy_gamma := divSpec (reduceSum 1 dy_gamma h1) (Tensor.full (.dim m .scalar) (n : ℝ))
  let mean_dy_gamma_xhat :=
    divSpec (reduceSum 1 (mulSpec dy_gamma norm) h1) (Tensor.full (.dim m .scalar) (n : ℝ))
  let inputGradient :=
    mulSpec (bAS (lnInvStd x h1 ε))
      (subSpec (subSpec dy_gamma (bAS mean_dy_gamma)) (mulSpec norm (bAS mean_dy_gamma_xhat)))
  { inputGradient, scaleGradient, biasGradient }

/-- `Spec.layerNormJvp` is `layerNormJvpMat` by unfolding. -/
theorem layerNormJvp_eq_mat (hm : 0 < m) (hn : 0 < n) (x tangent : Tensor ℝ [m, n])
    (gamma dgamma beta dbeta : Tensor ℝ [n]) (ε : ℝ) :
    Spec.layerNormJvp hm hn x tangent gamma dgamma beta dbeta ε =
      layerNormJvpMat x tangent gamma dgamma dbeta (nonemptyAxis_one hn) ε :=
  rfl

/-- `Spec.layerNormBackward` is `layerNormBackwardMat` by unfolding. -/
theorem layerNormBackward_eq_mat (hm : 0 < m) (hn : 0 < n) (x : Tensor ℝ [m, n])
    (gamma : Tensor ℝ [n]) (grad : Tensor ℝ [m, n]) (ε : ℝ) :
    Spec.layerNormBackward hm hn x gamma grad ε =
      layerNormBackwardMat x gamma grad (nonemptyAxis_one hn)
        (Shape.hasNonemptyAxisZeroOfPos hm).proof ε :=
  rfl

/-! ## The per-row identity -/

/-- Scalar adjointness of the LayerNorm rules on one row: `s` is the inverse standard deviation,
`xh` the normalized row, `γ` the per-column scale. -/
theorem row_adjoint (t g xh γ dγ dβ : Fin n → ℝ) (s : ℝ) :
    ∑ j : Fin n,
        (s * (t j - (∑ k : Fin n, t k) / n - xh j * ((∑ k : Fin n, t k * xh k) / n)) * γ j +
            xh j * dγ j + dβ j) * g j =
      ∑ j : Fin n,
          t j *
            (s * (g j * γ j - (∑ k : Fin n, g k * γ k) / n -
              xh j * ((∑ k : Fin n, g k * γ k * xh k) / n))) +
        ∑ j : Fin n, dγ j * (g j * xh j) + ∑ j : Fin n, dβ j * g j := by
  have hL :
      ∑ j : Fin n,
          (s * (t j - (∑ k : Fin n, t k) / n - xh j * ((∑ k : Fin n, t k * xh k) / n)) * γ j +
              xh j * dγ j + dβ j) * g j =
        s * ∑ j : Fin n, t j * γ j * g j -
          s * ((∑ k : Fin n, t k) / n) * ∑ j : Fin n, γ j * g j -
          s * ((∑ k : Fin n, t k * xh k) / n) * ∑ j : Fin n, xh j * γ j * g j +
          ∑ j : Fin n, xh j * dγ j * g j + ∑ j : Fin n, dβ j * g j := by
    calc
      _ = ∑ j : Fin n,
            (s * (t j * γ j * g j) - (s * ((∑ k : Fin n, t k) / n)) * (γ j * g j) -
              (s * ((∑ k : Fin n, t k * xh k) / n)) * (xh j * γ j * g j) +
              xh j * dγ j * g j + dβ j * g j) := by
            apply Finset.sum_congr rfl
            intro j _
            ring
      _ = _ := by
        simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]
  have hR :
      ∑ j : Fin n,
          t j *
            (s * (g j * γ j - (∑ k : Fin n, g k * γ k) / n -
              xh j * ((∑ k : Fin n, g k * γ k * xh k) / n))) =
        s * ∑ j : Fin n, t j * γ j * g j -
          s * ((∑ k : Fin n, g k * γ k) / n) * ∑ j : Fin n, t j -
          s * ((∑ k : Fin n, g k * γ k * xh k) / n) * ∑ j : Fin n, t j * xh j := by
    calc
      _ = ∑ j : Fin n,
            (s * (t j * γ j * g j) - (s * ((∑ k : Fin n, g k * γ k) / n)) * t j -
              (s * ((∑ k : Fin n, g k * γ k * xh k) / n)) * (t j * xh j)) := by
            apply Finset.sum_congr rfl
            intro j _
            ring
      _ = _ := by
        simp only [Finset.sum_sub_distrib, ← Finset.mul_sum]
  have hγg : ∑ j : Fin n, γ j * g j = ∑ k : Fin n, g k * γ k :=
    Finset.sum_congr rfl fun j _ => mul_comm _ _
  have hxhγg : ∑ j : Fin n, xh j * γ j * g j = ∑ k : Fin n, g k * γ k * xh k :=
    Finset.sum_congr rfl fun j _ => by ring
  have hdγ : ∑ j : Fin n, xh j * dγ j * g j = ∑ j : Fin n, dγ j * (g j * xh j) :=
    Finset.sum_congr rfl fun j _ => by ring
  rw [hL, hR, hγg, hxhγg, hdγ]
  ring

/-! ## Adjointness -/

/-- Entries of the JVP output. -/
theorem get2_layerNormJvpMat (x tangent : Tensor ℝ [m, n]) (gamma dgamma dbeta : Tensor ℝ [n])
    (h1 : Shape.NonemptyAxis 1 (.dim m (.dim n .scalar))) (ε : ℝ) (i : Fin m) (j : Fin n) :
    Spec.get2 (layerNormJvpMat x tangent gamma dgamma dbeta h1 ε) i j =
      (getScalar (lnInvStd x h1 ε) i *
            (Spec.get2 tangent i j - (∑ k : Fin n, Spec.get2 tangent i k) / n -
              (Spec.get2 (lnCentered x h1) i j * getScalar (lnInvStd x h1 ε) i) *
                ((∑ k : Fin n,
                  Spec.get2 tangent i k *
                    (Spec.get2 (lnCentered x h1) i k * getScalar (lnInvStd x h1 ε) i)) / n)) *
            getScalar gamma j +
          Spec.get2 (lnCentered x h1) i j * getScalar (lnInvStd x h1 ε) i * getScalar dgamma j) +
        getScalar dbeta j := by
  simp only [layerNormJvpMat, get2_addSpec, get2_mulSpec, get2_subSpec,
    get2_broadcastAfterSum_one, get2_broadcastTo_col, getScalar_divSpec, getScalar_full,
    getScalar_reduceSum_one]

/-- Entries of the backward input gradient. -/
theorem get2_inputGradient (x : Tensor ℝ [m, n]) (gamma : Tensor ℝ [n]) (grad : Tensor ℝ [m, n])
    (h1 : Shape.NonemptyAxis 1 (.dim m (.dim n .scalar)))
    (h0 : Shape.NonemptyAxis 0 (.dim m (.dim n .scalar))) (ε : ℝ) (i : Fin m) (j : Fin n) :
    Spec.get2 (layerNormBackwardMat x gamma grad h1 h0 ε).inputGradient i j =
      getScalar (lnInvStd x h1 ε) i *
        (Spec.get2 grad i j * getScalar gamma j -
          (∑ k : Fin n, Spec.get2 grad i k * getScalar gamma k) / n -
          Spec.get2 (lnCentered x h1) i j / getScalar (lnStd x h1 ε) i *
            ((∑ k : Fin n,
              Spec.get2 grad i k * getScalar gamma k *
                (Spec.get2 (lnCentered x h1) i k / getScalar (lnStd x h1 ε) i)) / n)) := by
  simp only [layerNormBackwardMat, get2_mulSpec, get2_subSpec, get2_divSpec,
    get2_broadcastAfterSum_one, get2_broadcastTo_col, getScalar_divSpec, getScalar_full,
    getScalar_reduceSum_one]

/-- Entries of the backward scale gradient. -/
theorem getScalar_scaleGradient (x : Tensor ℝ [m, n]) (gamma : Tensor ℝ [n])
    (grad : Tensor ℝ [m, n]) (h1 : Shape.NonemptyAxis 1 (.dim m (.dim n .scalar)))
    (h0 : Shape.NonemptyAxis 0 (.dim m (.dim n .scalar))) (ε : ℝ) (j : Fin n) :
    getScalar (layerNormBackwardMat x gamma grad h1 h0 ε).scaleGradient j =
      ∑ i : Fin m,
        Spec.get2 grad i j * (Spec.get2 (lnCentered x h1) i j / getScalar (lnStd x h1 ε) i) := by
  simp only [layerNormBackwardMat, getScalar_reduceSum_zero, get2_mulSpec, get2_divSpec,
    get2_broadcastAfterSum_one]

/-- Entries of the backward bias gradient. -/
theorem getScalar_biasGradient (x : Tensor ℝ [m, n]) (gamma : Tensor ℝ [n])
    (grad : Tensor ℝ [m, n]) (h1 : Shape.NonemptyAxis 1 (.dim m (.dim n .scalar)))
    (h0 : Shape.NonemptyAxis 0 (.dim m (.dim n .scalar))) (ε : ℝ) (j : Fin n) :
    getScalar (layerNormBackwardMat x gamma grad h1 h0 ε).biasGradient j =
      ∑ i : Fin m, Spec.get2 grad i j := by
  simp only [layerNormBackwardMat, getScalar_reduceSum_zero]

/-- `1 / std` times `centered` is `centered / std`. -/
theorem centered_mul_invStd (x : Tensor ℝ [m, n])
    (h1 : Shape.NonemptyAxis 1 (.dim m (.dim n .scalar))) (ε : ℝ) (i : Fin m) (j : Fin n) :
    Spec.get2 (lnCentered x h1) i j * getScalar (lnInvStd x h1 ε) i =
      Spec.get2 (lnCentered x h1) i j / getScalar (lnStd x h1 ε) i := by
  rw [lnInvStd, getScalar_divSpec, getScalar_full]
  ring

/--
The LayerNorm reverse rule is adjoint to its forward differential.

Pairing the input tangent and both parameter tangents with `Spec.layerNormJvp` gives the same
scalar as pairing the upstream gradient with the three outputs of `Spec.layerNormBackward`.
-/
theorem layerNormJvp_layerNormBackward_adjoint (hm : 0 < m) (hn : 0 < n)
    (x tangent gradOutput : Tensor ℝ [m, n]) (gamma dgamma beta dbeta : Tensor ℝ [n]) (ε : ℝ) :
    dot (Spec.layerNormJvp hm hn x tangent gamma dgamma beta dbeta ε) gradOutput =
      dot tangent (Spec.layerNormBackward hm hn x gamma gradOutput ε).inputGradient +
        dot dgamma (Spec.layerNormBackward hm hn x gamma gradOutput ε).scaleGradient +
        dot dbeta (Spec.layerNormBackward hm hn x gamma gradOutput ε).biasGradient := by
  rw [layerNormJvp_eq_mat, layerNormBackward_eq_mat, dot_mat_eq_sum, dot_mat_eq_sum,
    dot_vec_eq_sum, dot_vec_eq_sum]
  simp only [get2_layerNormJvpMat, get2_inputGradient, getScalar_scaleGradient,
    getScalar_biasGradient, centered_mul_invStd]
  simp only [Finset.mul_sum]
  rw [Finset.sum_comm (f := fun j i => getScalar dgamma j * _),
    Finset.sum_comm (f := fun j i => getScalar dbeta j * _), ← Finset.sum_add_distrib,
    ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i _
  exact row_adjoint (fun j => Spec.get2 tangent i j) (fun j => Spec.get2 gradOutput i j)
    (fun j => Spec.get2 (lnCentered x (nonemptyAxis_one hn)) i j /
      getScalar (lnStd x (nonemptyAxis_one hn) ε) i)
    (fun j => getScalar gamma j) (fun j => getScalar dgamma j) (fun j => getScalar dbeta j)
    (getScalar (lnInvStd x (nonemptyAxis_one hn) ε) i)

end

end LayerNorm
end Autograd
end Proofs
