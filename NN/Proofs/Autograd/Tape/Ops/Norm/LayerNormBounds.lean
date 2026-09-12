/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Ops.Norm.LayerNormFDeriv

/-!
# Bounds for LayerNorm derivatives

Positive epsilon gives a lower bound `sqrt ε` for every row's standard deviation. We use it
to bound the actual row differential, retaining the contribution from the changing variance.
The variance contribution contains three factors of `1 / sqrt ε`; replacing their product
by `1 / ε` would lose the bound when epsilon is smaller than one.

The hypotheses below bound the centered input and the input direction coordinatewise. They
make no assumption about the derivative being bounded. The final theorem also includes the
scale and bias directions, using the Fréchet derivative of the actual LayerNorm specification.
-/

@[expose] public section

namespace Proofs.Autograd

open Spec TorchLean
open TorchLean.Tensor
open TapeNodes.Matmul
open scoped BigOperators

noncomputable section

namespace RowNorm

variable {m n : Nat}

/-- The inverse standard deviation is nonnegative. -/
theorem invStd_nonneg (X : Vec (matSize m n)) (ε : ℝ) (i : Fin m) :
    0 ≤ invStd X ε i :=
  inv_nonneg.mpr (Real.sqrt_nonneg _)

/-- Epsilon alone bounds the inverse standard deviation, including for constant rows. -/
theorem invStd_le_inv_sqrt {ε : ℝ} (hε : 0 < ε)
    (X : Vec (matSize m n)) (i : Fin m) :
    invStd X ε i ≤ (Real.sqrt ε)⁻¹ := by
  apply inv_anti₀ (Real.sqrt_pos.mpr hε)
  exact Real.sqrt_le_sqrt (le_add_of_nonneg_left (rowVar_nonneg X i))

/-- Coordinate bounds on a row give a bound on the absolute value of its mean. -/
theorem abs_rowMean_le (X : Vec (matSize m n)) (i : Fin m)
    (bounds : Fin n → ℝ)
    (hbounds : ∀ j, |X (idxMN (m := m) (n := n) i j)| ≤ bounds j) :
    |rowMean X i| ≤ (∑ j, bounds j) / n := by
  have hn0 : (0 : ℝ) ≤ n := Nat.cast_nonneg n
  rw [rowMean, abs_div, abs_of_nonneg hn0]
  apply div_le_div_of_nonneg_right _ hn0
  exact (Finset.abs_sum_le_sum_abs _ _).trans
    (Finset.sum_le_sum fun j _ => hbounds j)

/-- A centered-input bound gives a normalized-input bound without a variance lower estimate. -/
theorem abs_nrm_le {ε : ℝ} (hε : 0 < ε)
    (X : Vec (matSize m n)) (i : Fin m) (j : Fin n)
    {bound : ℝ} (hbound : |centered X i j| ≤ bound) :
    |nrm X ε i j| ≤ bound * (Real.sqrt ε)⁻¹ := by
  rw [nrm, abs_mul, abs_of_nonneg (invStd_nonneg X ε i)]
  exact mul_le_mul hbound (invStd_le_inv_sqrt hε X i)
    (invStd_nonneg X ε i) ((abs_nonneg _).trans hbound)

/--
Radius obtained from centered-input bounds `u` and input-direction bounds `d`.

The first two terms bound the direction and its row mean. The third bounds the projection
along the normalized input. All sums are over this row's feature axis.
-/
def derivativeRadius {n : Nat} (ε : ℝ) (u d : Fin n → ℝ) (j : Fin n) : ℝ :=
  let t := (Real.sqrt ε)⁻¹
  t * (d j + (∑ k, d k) / n +
    (u j * t) * ((∑ k, d k * (u k * t)) / n))

/-- The row differential is bounded by the centered-input and direction intervals. -/
theorem abs_nrmJvp_le {ε : ℝ} (hε : 0 < ε)
    (X dX : Vec (matSize m n)) (i : Fin m)
    (u d : Fin n → ℝ)
    (hu : ∀ k, |centered X i k| ≤ u k)
    (hd : ∀ k, |dX (idxMN (m := m) (n := n) i k)| ≤ d k)
    (j : Fin n) :
    |nrmJvp X ε dX i j| ≤ derivativeRadius ε u d j := by
  let t := (Real.sqrt ε)⁻¹
  have ht : 0 ≤ t := inv_nonneg.mpr (Real.sqrt_nonneg ε)
  have hu0 (k) : 0 ≤ u k := (abs_nonneg _).trans (hu k)
  have hd0 (k) : 0 ≤ d k := (abs_nonneg _).trans (hd k)
  have hn0 : (0 : ℝ) ≤ n := Nat.cast_nonneg n
  have hnorm (k) : |nrm X ε i k| ≤ u k * t :=
    abs_nrm_le hε X i k (hu k)
  have hmean := abs_rowMean_le dX i d hd
  have hsub (a b : ℝ) : |a - b| ≤ |a| + |b| := by
    simpa only [sub_eq_add_neg, abs_neg] using abs_add_le a (-b)
  have hproduct :
      |(∑ k, dX (idxMN (m := m) (n := n) i k) * nrm X ε i k) / n| ≤
        (∑ k, d k * (u k * t)) / n := by
    rw [abs_div, abs_of_nonneg hn0]
    apply div_le_div_of_nonneg_right _ hn0
    refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum ?_)
    intro k _
    rw [abs_mul]
    exact mul_le_mul (hd k) (hnorm k) (abs_nonneg _) (hd0 k)
  have hbracket :
      |dX (idxMN (m := m) (n := n) i j) - rowMean dX i -
          nrm X ε i j *
            ((∑ k, dX (idxMN (m := m) (n := n) i k) * nrm X ε i k) / n)| ≤
        d j + (∑ k, d k) / n + (u j * t) * ((∑ k, d k * (u k * t)) / n) := by
    refine (hsub _ _).trans (add_le_add ?_ ?_)
    · exact (hsub _ _).trans (add_le_add (hd j) hmean)
    · rw [abs_mul]
      exact mul_le_mul (hnorm j) hproduct (abs_nonneg _) (mul_nonneg (hu0 j) ht)
  rw [nrmJvp, abs_mul, abs_of_nonneg (invStd_nonneg X ε i)]
  exact mul_le_mul (invStd_le_inv_sqrt hε X i) hbracket
    (abs_nonneg _) ht

end RowNorm

namespace LayerNorm

variable {m n : Nat}

/--
Coordinate bound for the full LayerNorm derivative, including both affine parameter directions.

The scale multiplies the normalized-input differential. Its own direction contributes the
normalized input times `dgamma`, and the bias direction contributes directly. Gamma and beta
remain arbitrary exact parameters throughout.
-/
theorem abs_fderiv_specLayerNormVec_le (hm : 0 < m) (hn : 0 < n)
    {ε : ℝ} (hε : 0 < ε) (q dq : CtxVec (ΓLN m n))
    (i : Fin m) (j : Fin n) (u d : Fin n → ℝ)
    (hu : ∀ k, |RowNorm.centered (valX q) i k| ≤ u k)
    (hd : ∀ k, |valX dq (idxMN (m := m) (n := n) i k)| ≤ d k) :
    |fderiv ℝ (specLayerNormVec hm hn ε) q dq (idxMN (m := m) (n := n) i j)| ≤
      RowNorm.derivativeRadius ε u d j * |valGamma q j| +
        (u j * (Real.sqrt ε)⁻¹) * |valGamma dq j| + |valBeta dq j| := by
  rw [fderiv_specLayerNormVec_idxMN hm hn hε]
  refine (abs_add_le _ _).trans (add_le_add ?_ (le_refl |valBeta dq j|))
  refine (abs_add_le _ _).trans (add_le_add ?_ ?_)
  · rw [abs_mul]
    exact mul_le_mul_of_nonneg_right
      (RowNorm.abs_nrmJvp_le hε (valX q) (valX dq) i u d hu hd j) (abs_nonneg _)
  · rw [abs_mul]
    exact mul_le_mul_of_nonneg_right
      (RowNorm.abs_nrm_le hε (valX q) i j (hu j)) (abs_nonneg _)

end LayerNorm

end

end Proofs.Autograd
