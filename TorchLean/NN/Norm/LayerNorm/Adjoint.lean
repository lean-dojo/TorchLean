/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Norm.MatrixEntries

/-!
# LayerNorm JVP and VJP adjointness

`jvp` is the forward-mode rule and `backward` the reverse-mode rule
shipped with `Tensor.layerNorm`, and the two are adjoint: pairing the JVP of an input and parameter
tangent with the upstream gradient gives the same scalar as pairing each tangent with the matching
backward output. `TorchLean.NN.Norm.LayerNorm.FDeriv` identifies this JVP with the derivative of
the spec for positive epsilon, then uses the pairing to prove that the backward rule is the
adjoint of that derivative.
The pairing itself is algebraic and holds for every real epsilon.

The proof is entrywise. Both spec functions are rewritten to matrix forms (over the reals the rows
scaled by `Tensor.normalizationScale` are the centered rows divided by `lnStd`), every entry is
rewritten as a real expression, and the per-row identity `row_adjoint` finishes with `ring`.
-/

@[expose] public section

namespace TorchLean
namespace LayerNorm

open Tensor Normalization
open Rep (dot)

open scoped BigOperators

noncomputable section

variable {m n : Nat}

/-! ## Matrix forms -/

/-- `1 / std`, as computed by the spec. -/
def invStd (x : Tensor ℝ [m, n]) (ε : ℝ) : Tensor ℝ [m] :=
  Tensor.full (.dim m .scalar) 1 / lnStd x ε

/-- `jvp` with its axis, broadcast, and cast evidence spelled out. -/
def jvpMat (x tangent : Tensor ℝ [m, n]) (gamma dgamma dbeta : Tensor ℝ [n])
    (ε : ℝ) : Tensor ℝ [m, n] :=
  let inv_std_b : Tensor ℝ [m, n] := rowBroadcast (invStd x ε)
  let norm := lnCentered x * inv_std_b
  let mean_tangent := Tensor.map (Context.divNat · n) (rowSum tangent)
  let mean_tangent_norm := Tensor.map (Context.divNat · n) (rowSum (tangent * norm))
  let dnorm :=
    inv_std_b *
      ((tangent - rowBroadcast mean_tangent) -
        (norm * rowBroadcast mean_tangent_norm))
  ((dnorm * (Tensor.broadcast gamma _ (canBroadcastTo_col (m := m) (n := n)).dims
        (canBroadcastTo_col (m := m) (n := n)).broadcastDims)) +
      (norm * (Tensor.broadcast dgamma _ (canBroadcastTo_col (m := m) (n := n)).dims
        (canBroadcastTo_col
          (m := m) (n := n)).broadcastDims))) +
    (Tensor.broadcast dbeta _ (canBroadcastTo_col (m := m) (n := n)).dims
      (canBroadcastTo_col (m := m)
        (n := n)).broadcastDims)

/-- `backward` with its axis, broadcast, and cast evidence spelled out. -/
def backwardMat (x : Tensor ℝ [m, n]) (gamma : Tensor ℝ [n]) (grad : Tensor ℝ [m, n])
    (ε : ℝ) :
    Gradients ℝ [m, n] [n] :=
  let norm := lnCentered x / rowBroadcast (lnStd x ε)
  let biasGradient := colSum grad
  let scaleGradient := colSum (grad * norm)
  let dy_gamma := grad * (Tensor.broadcast gamma _ (canBroadcastTo_col (m := m) (n := n)).dims
      (canBroadcastTo_col (m := m) (n := n)).broadcastDims)
  let mean_dy_gamma := Tensor.map (Context.divNat · n) (rowSum dy_gamma)
  let mean_dy_gamma_xhat := Tensor.map (Context.divNat · n) (rowSum (dy_gamma * norm))
  let inputGradient :=
    rowBroadcast (invStd x ε) *
      ((dy_gamma - rowBroadcast mean_dy_gamma) -
        (norm * rowBroadcast mean_dy_gamma_xhat))
  { inputGradient, scaleGradient, biasGradient }

/-- For `0 < n`, the spec's normalized rows `(Normalization.normalizedRows x ε true).1` are, over
the reals, the centered rows times `invStd`. -/
theorem normalizedRows_fst_eq_mul (hn : 0 < n) (x : Tensor ℝ [m, n]) (ε : ℝ) :
    (Normalization.normalizedRows x ε true).1 = lnCentered x * rowBroadcast (invStd x ε) := by
  rw [normalizedRows_fst hn]
  exact Tensor.ext_get2 fun i j => by
    rw [get2_div, get2_mul, get2_rowBroadcast, get2_rowBroadcast, invStd, getScalar_div,
      getScalar_full]
    exact div_eq_mul_one_div _ _

/-- `jvp` is `jvpMat`: over the reals its normalized rows are the
centered rows times `invStd`. -/
theorem jvp_eq_jvpMat (hm : 0 < m) (hn : 0 < n) (x tangent : Tensor ℝ [m, n])
    (gamma dgamma beta dbeta : Tensor ℝ [n]) (ε : ℝ) :
    jvp hm hn x tangent gamma dgamma beta dbeta ε =
      jvpMat x tangent gamma dgamma dbeta ε := by
  simp only [jvp]
  rw [normalizedRows_fst_eq_mul hn, normalizedRows_snd hn]
  rfl

/-- `backward` is `backwardMat`: over the reals its normalized rows are
the centered rows divided by `lnStd`. -/
theorem backward_eq_backwardMat (hm : 0 < m) (hn : 0 < n) (x : Tensor ℝ [m, n])
    (gamma : Tensor ℝ [n]) (grad : Tensor ℝ [m, n]) (ε : ℝ) :
    backward hm hn x gamma grad ε =
      backwardMat x gamma grad ε := by
  simp only [backward]
  rw [normalizedRows_fst hn, normalizedRows_snd hn]
  rfl

/-! ## The per-row identity -/

/-- Scalar adjointness of the LayerNorm rules on one row, with input tangent `t`, upstream gradient
`g`, normalized row `xh`, inverse standard deviation `s`, per-column scale `γ` and parameter
tangents `dγ`, `dβ`. -/
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
theorem get2_jvpMat (x tangent : Tensor ℝ [m, n]) (gamma dgamma dbeta : Tensor ℝ [n])
    (ε : ℝ) (i : Fin m) (j : Fin n) :
    Tensor.get2 (jvpMat x tangent gamma dgamma dbeta ε) i j =
      (getScalar (invStd x ε) i *
            (Tensor.get2 tangent i j - (∑ k : Fin n, Tensor.get2 tangent i k) / n -
              (Tensor.get2 (lnCentered x) i j * getScalar (invStd x ε) i) *
                ((∑ k : Fin n,
                  Tensor.get2 tangent i k *
                    (Tensor.get2 (lnCentered x) i k * getScalar (invStd x ε) i)) / n)) *
            getScalar gamma j +
          Tensor.get2 (lnCentered x) i j * getScalar (invStd x ε) i * getScalar dgamma j) +
        getScalar dbeta j := by
  simp only [jvpMat, get2_add, get2_mul, get2_sub,
    get2_rowBroadcast, get2_broadcast_col, getScalar_map, Context.divNat_real,
    getScalar_rowSum]

/-- Entries of the backward input gradient. -/
theorem get2_inputGradient (x : Tensor ℝ [m, n]) (gamma : Tensor ℝ [n]) (grad : Tensor ℝ [m, n])
    (ε : ℝ) (i : Fin m) (j : Fin n) :
    Tensor.get2 (backwardMat x gamma grad ε).inputGradient i j =
      getScalar (invStd x ε) i *
        (Tensor.get2 grad i j * getScalar gamma j -
          (∑ k : Fin n, Tensor.get2 grad i k * getScalar gamma k) / n -
          Tensor.get2 (lnCentered x) i j / getScalar (lnStd x ε) i *
            ((∑ k : Fin n,
              Tensor.get2 grad i k * getScalar gamma k *
                (Tensor.get2 (lnCentered x) i k / getScalar (lnStd x ε) i)) / n)) := by
  simp only [backwardMat, get2_mul, get2_sub, get2_div,
    get2_rowBroadcast, get2_broadcast_col, getScalar_map, Context.divNat_real,
    getScalar_rowSum]

/-- Entries of the backward scale gradient. -/
theorem getScalar_scaleGradient (x : Tensor ℝ [m, n]) (gamma : Tensor ℝ [n])
    (grad : Tensor ℝ [m, n]) (ε : ℝ) (j : Fin n) :
    getScalar (backwardMat x gamma grad ε).scaleGradient j =
      ∑ i : Fin m,
        Tensor.get2 grad i j * (Tensor.get2 (lnCentered x) i j / getScalar (lnStd x ε) i) := by
  simp only [backwardMat, getScalar_colSum, get2_mul, get2_div,
    get2_rowBroadcast]

/-- Entries of the backward bias gradient. -/
theorem getScalar_biasGradient (x : Tensor ℝ [m, n]) (gamma : Tensor ℝ [n])
    (grad : Tensor ℝ [m, n]) (ε : ℝ) (j : Fin n) :
    getScalar (backwardMat x gamma grad ε).biasGradient j =
      ∑ i : Fin m, Tensor.get2 grad i j := by
  simp only [backwardMat, getScalar_colSum]

/-- `1 / std` times `centered` is `centered / std`. -/
theorem centered_mul_invStd (x : Tensor ℝ [m, n])
    (ε : ℝ) (i : Fin m) (j : Fin n) :
    Tensor.get2 (lnCentered x) i j * getScalar (invStd x ε) i =
      Tensor.get2 (lnCentered x) i j / getScalar (lnStd x ε) i := by
  rw [invStd, getScalar_div, getScalar_full]
  ring

/--
The LayerNorm reverse rule is adjoint to its forward differential, for every `ε`.

Pairing the JVP of the input and parameter tangents with `gradOutput` gives the sum of the three
outputs of `backward`, each paired with its tangent.
-/
theorem jvp_backward_adjoint (hm : 0 < m) (hn : 0 < n)
    (x tangent gradOutput : Tensor ℝ [m, n]) (gamma dgamma beta dbeta : Tensor ℝ [n]) (ε : ℝ) :
    Rep.dot (jvp hm hn x tangent gamma dgamma beta dbeta ε) gradOutput =
      Rep.dot tangent (backward hm hn x gamma gradOutput ε).inputGradient +
        Rep.dot dgamma (backward hm hn x gamma gradOutput ε).scaleGradient +
        Rep.dot dbeta (backward hm hn x gamma gradOutput ε).biasGradient := by
  rw [jvp_eq_jvpMat, backward_eq_backwardMat, dot_mat_eq_sum, dot_mat_eq_sum,
    Tensor.dot_vec_eq_sum, Tensor.dot_vec_eq_sum]
  simp only [get2_jvpMat, get2_inputGradient, getScalar_scaleGradient,
    getScalar_biasGradient, centered_mul_invStd]
  simp only [Finset.mul_sum]
  rw [Finset.sum_comm (f := fun j i => getScalar dgamma j * _),
    Finset.sum_comm (f := fun j i => getScalar dbeta j * _), ← Finset.sum_add_distrib,
    ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i _
  exact row_adjoint (fun j => Tensor.get2 tangent i j) (fun j => Tensor.get2 gradOutput i j)
    (fun j => Tensor.get2 (lnCentered x) i j /
      getScalar (lnStd x ε) i)
    (fun j => getScalar gamma j) (fun j => getScalar dgamma j) (fun j => getScalar dbeta j)
    (getScalar (invStd x ε) i)

end

end LayerNorm
end TorchLean
