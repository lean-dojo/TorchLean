/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Norm.LayerNorm.FDeriv

/-!
# Bounds for LayerNorm derivatives

Positive epsilon bounds every row's standard deviation below by `sqrt ε`, and the theorems here
apply that bound to the full row differential `nrmJVP`, keeping the contribution of the changing
variance. That contribution carries three factors of `1 / sqrt ε`, and replacing their product by
`1 / ε` would lose the bound when epsilon is smaller than one.

The hypotheses bound the centered input and the input direction coordinatewise, and nothing is
assumed about the derivative. The final theorem adds the scale and bias directions, through the
Fréchet derivative of the LayerNorm specification `LayerNorm.forwardVec`.
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean

open TorchLean.Tensor
open Matmul
open scoped BigOperators

noncomputable section

namespace RowNorm

variable {m n : Nat}

/-- The inverse standard deviation is nonnegative. -/
theorem invStd_nonneg (X : Vec (Shape.size [m, n])) (ε : ℝ) (i : Fin m) :
    0 ≤ invStd X ε i :=
  inv_nonneg.mpr (Real.sqrt_nonneg _)

/-- For positive `ε`, the inverse standard deviation is at most `(sqrt ε)⁻¹`, constant rows
included. -/
theorem invStd_le_inv_sqrt {ε : ℝ} (hε : 0 < ε)
    (X : Vec (Shape.size [m, n])) (i : Fin m) :
    invStd X ε i ≤ (Real.sqrt ε)⁻¹ := by
  apply inv_anti₀ (Real.sqrt_pos.mpr hε)
  exact Real.sqrt_le_sqrt
    (le_add_of_nonneg_left (var_nonneg fun j => X (idxMN (m := m) (n := n) i j)))

/-- Termwise absolute bounds give a bound on the absolute value of a mean over `Fin n`. -/
theorem abs_sum_div_le (f bounds : Fin n → ℝ) (h : ∀ k, |f k| ≤ bounds k) :
    |(∑ k, f k) / n| ≤ (∑ k, bounds k) / n := by
  rw [abs_div, abs_of_nonneg (Nat.cast_nonneg n : (0 : ℝ) ≤ n)]
  exact div_le_div_of_nonneg_right
    ((Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun k _ => h k))
    (Nat.cast_nonneg n)

/-- Coordinate bounds on a row give a bound on the absolute value of its mean. -/
theorem abs_rowMean_le (X : Vec (Shape.size [m, n])) (i : Fin m)
    (bounds : Fin n → ℝ)
    (hbounds : ∀ j, |X (idxMN (m := m) (n := n) i j)| ≤ bounds j) :
    |𝔼 j, X (idxMN (m := m) (n := n) i j)| ≤ (∑ j, bounds j) / n := by
  rw [Fintype.expect_eq_sum_div_card, Fintype.card_fin]
  exact abs_sum_div_le _ bounds hbounds

/-- A centered-input bound gives a normalized-input bound without a variance lower estimate. -/
theorem abs_nrm_le {ε : ℝ} (hε : 0 < ε)
    (X : Vec (Shape.size [m, n])) (i : Fin m) (j : Fin n)
    {bound : ℝ} (hbound : |centered X i j| ≤ bound) :
    |nrm X ε i j| ≤ bound * (Real.sqrt ε)⁻¹ := by
  rw [nrm, abs_mul, abs_of_nonneg (invStd_nonneg X ε i)]
  exact mul_le_mul hbound (invStd_le_inv_sqrt hε X i)
    (invStd_nonneg X ε i) ((abs_nonneg _).trans hbound)

/--
Radius obtained from centered-input bounds `u` and input-direction bounds `d`.

The first two terms bound the direction and its row mean, and the third bounds the projection
along the normalized input. All sums run over the row's feature axis.
-/
def derivativeRadius {n : Nat} (ε : ℝ) (u d : Fin n → ℝ) (j : Fin n) : ℝ :=
  let t := (Real.sqrt ε)⁻¹
  t * (d j + (∑ k, d k) / n +
    (u j * t) * ((∑ k, d k * (u k * t)) / n))

/-- For positive `ε`, `|nrmJVP X ε dX i j|` is at most `derivativeRadius ε u d j` when `u` bounds
the centered row and `d` the direction coordinatewise. -/
theorem abs_nrmJVP_le {ε : ℝ} (hε : 0 < ε)
    (X dX : Vec (Shape.size [m, n])) (i : Fin m)
    (u d : Fin n → ℝ)
    (hu : ∀ k, |centered X i k| ≤ u k)
    (hd : ∀ k, |dX (idxMN (m := m) (n := n) i k)| ≤ d k)
    (j : Fin n) :
    |nrmJVP X ε dX i j| ≤ derivativeRadius ε u d j := by
  let t := (Real.sqrt ε)⁻¹
  have ht : 0 ≤ t := inv_nonneg.mpr (Real.sqrt_nonneg ε)
  have hu0 (k) : 0 ≤ u k := (abs_nonneg _).trans (hu k)
  have hd0 (k) : 0 ≤ d k := (abs_nonneg _).trans (hd k)
  have hnorm (k) : |nrm X ε i k| ≤ u k * t :=
    abs_nrm_le hε X i k (hu k)
  have hmean := abs_rowMean_le dX i d hd
  have hproduct :
      |(∑ k, dX (idxMN (m := m) (n := n) i k) * nrm X ε i k) / n| ≤
        (∑ k, d k * (u k * t)) / n :=
    abs_sum_div_le _ _ fun k => by
      rw [abs_mul]
      exact mul_le_mul (hd k) (hnorm k) (abs_nonneg _) (hd0 k)
  have hbracket :
      |dX (idxMN (m := m) (n := n) i j) - 𝔼 k, dX (idxMN (m := m) (n := n) i k) -
          nrm X ε i j *
            ((∑ k, dX (idxMN (m := m) (n := n) i k) * nrm X ε i k) / n)| ≤
        d j + (∑ k, d k) / n + (u j * t) * ((∑ k, d k * (u k * t)) / n) := by
    refine (abs_sub _ _).trans (add_le_add ?_ ?_)
    · exact (abs_sub _ _).trans (add_le_add (hd j) hmean)
    · rw [abs_mul]
      exact mul_le_mul (hnorm j) hproduct (abs_nonneg _) (mul_nonneg (hu0 j) ht)
  rw [nrmJVP, abs_mul, abs_of_nonneg (invStd_nonneg X ε i)]
  exact mul_le_mul (invStd_le_inv_sqrt hε X i) hbracket
    (abs_nonneg _) ht

end RowNorm

namespace LayerNorm

variable {m n : Nat}

/--
Coordinate bound for the full LayerNorm derivative, including both affine parameter directions,
for `0 < n` and `0 < ε`.

The bound on the normalized-input differential is multiplied by `|gamma|`, the scale direction
contributes the normalized-input bound times `|dgamma|`, and the bias direction contributes
`|dbeta|`. No hypothesis constrains `gamma` or `beta`.
-/
theorem abs_fderiv_forwardVec_le (hn : 0 < n)
    {ε : ℝ} (hε : 0 < ε) (q dq : CtxVec (ctx m n))
    (i : Fin m) (j : Fin n) (u d : Fin n → ℝ)
    (hu : ∀ k, |RowNorm.centered (valX q) i k| ≤ u k)
    (hd : ∀ k, |valX dq (idxMN (m := m) (n := n) i k)| ≤ d k) :
    |fderiv ℝ (forwardVec ε) q dq (idxMN (m := m) (n := n) i j)| ≤
      RowNorm.derivativeRadius ε u d j * |valGamma q j| +
        (u j * (Real.sqrt ε)⁻¹) * |valGamma dq j| + |valBeta dq j| := by
  rw [fderiv_forwardVec_idxMN hn hε]
  refine (abs_add_le _ _).trans (add_le_add ?_ (le_refl |valBeta dq j|))
  refine (abs_add_le _ _).trans (add_le_add ?_ ?_)
  · rw [abs_mul]
    exact mul_le_mul_of_nonneg_right
      (RowNorm.abs_nrmJVP_le hε (valX q) (valX dq) i u d hu hd j) (abs_nonneg _)
  · rw [abs_mul]
    exact mul_le_mul_of_nonneg_right
      (RowNorm.abs_nrm_le hε (valX q) i j (hu j)) (abs_nonneg _)

end LayerNorm

end

end TorchLean
