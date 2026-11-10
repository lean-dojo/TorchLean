/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.InnerProductSpace.Calculus
public import TorchLean.NN.Tape.RealNode.Matrix
public import TorchLean.Tensor.Statistics.Real
import Mathlib.Analysis.Calculus.FDeriv.WithLp

/-!
# Calculus of row normalization

LayerNorm normalizes every row of a `[seqLen, embedDim]` matrix, and the flattened form of
BatchNorm normalizes every row of a `[channels, positions]` matrix. Both apply the map

`x ↦ (x - 𝔼 x) / sqrt (var x + ε)`

to each row, where the row mean is Mathlib's `𝔼` (`Finset.expect`) and the row variance is the
population variance `TorchLean.var`. For `0 < ε` the map is differentiable on flattened matrices,
and for nonempty rows its derivative is the closed form

`dx ↦ invStd * (dx - 𝔼 dx - xhat * 𝔼 (dx * xhat))`

used by the normalization JVPs.
-/

@[expose] public section

-- In this file simp leaves out `size_cons`. It would rewrite the sizes inside the `Fin` and `Vec`
-- types before the `WithLp.toLp`, projection and sum lemmas match.
attribute [-simp] TorchLean.Shape.size_cons

open TorchLean.Shape (size)

namespace TorchLean
namespace RowNorm

open Matmul

open scoped BigOperators

noncomputable section

variable {m n : Nat}

/-- Centered entry `(i, j)`. -/
def centered (X : Vec (Shape.size [m, n])) (i : Fin m) (j : Fin n) : ℝ :=
  X (idxMN (m := m) (n := n) i j) - 𝔼 k, X (idxMN (m := m) (n := n) i k)

/-- Inverse standard deviation `1 / sqrt (var + ε)` of row `i`. -/
def invStd (X : Vec (Shape.size [m, n])) (ε : ℝ) (i : Fin m) : ℝ :=
  (Real.sqrt (TorchLean.var (fun j => X (idxMN (m := m) (n := n) i j)) + ε))⁻¹

/-- Normalized entry `(i, j)`. -/
def nrm (X : Vec (Shape.size [m, n])) (ε : ℝ) (i : Fin m) (j : Fin n) : ℝ :=
  centered X i j * invStd X ε i

/-- Closed-form differential of the normalized entry `(i, j)` in the direction `dX`. -/
def nrmJVP (X : Vec (Shape.size [m, n])) (ε : ℝ) (dX : Vec (Shape.size [m, n]))
    (i : Fin m) (j : Fin n) :
    ℝ :=
  invStd X ε i *
    (dX (idxMN (m := m) (n := n) i j) - 𝔼 k, dX (idxMN (m := m) (n := n) i k) -
      nrm X ε i j * ((∑ k : Fin n, dX (idxMN (m := m) (n := n) i k) * nrm X ε i k) / n))

/-- The variance of row `i` plus a positive `ε` is positive. -/
theorem rowVar_add_pos {ε : ℝ} (hε : 0 < ε) (X : Vec (Shape.size [m, n])) (i : Fin m) :
    0 < TorchLean.var (fun j => X (idxMN (m := m) (n := n) i j)) + ε :=
  add_pos_of_nonneg_of_pos (var_nonneg _) hε

/-- The centered row is orthogonal to constants, so `∑ c_j (d_j - a) = ∑ c_j d_j`. -/
theorem sum_centered_mul_sub (X : Vec (Shape.size [m, n])) (i : Fin m)
    (d : Fin n → ℝ) (a : ℝ) :
    ∑ j : Fin n, centered X i j * (d j - a) = ∑ j : Fin n, centered X i j * d j := by
  have h0 : ∑ j : Fin n, centered X i j = 0 :=
    Fintype.sum_balance fun j => X (idxMN (m := m) (n := n) i j)
  simp only [mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul, h0, zero_mul, sub_zero]

/-! ## Derivative building blocks -/

/-- Derivative of the row mean. -/
def meanD (i : Fin m) : Vec (Shape.size [m, n]) →L[ℝ] ℝ :=
  ((n : ℝ)⁻¹) • ∑ j : Fin n, EuclideanSpace.proj (𝕜 := ℝ) (idxMN (m := m) (n := n) i j)

/--
The row-mean derivative is the row mean of the perturbation, since the mean is already linear.
-/
@[simp] theorem meanD_apply (i : Fin m) (dX : Vec (Shape.size [m, n])) :
    meanD (m := m) (n := n) i dX = 𝔼 j, dX (idxMN (m := m) (n := n) i j) := by
  simp [meanD, Fintype.expect_eq_sum_div_card, div_eq_inv_mul]

/-- Derivative of a centered entry. -/
def centD (i : Fin m) (j : Fin n) : Vec (Shape.size [m, n]) →L[ℝ] ℝ :=
  EuclideanSpace.proj (𝕜 := ℝ) (idxMN (m := m) (n := n) i j) - meanD (m := m) (n := n) i

/-- The centering derivative maps `dX` to its entry `(i, j)` minus the mean of its row `i`.

Centering is linear too, so it is its own derivative. The nonlinearity of row normalization enters
only through the variance and the square root below. -/
@[simp] theorem centD_apply (i : Fin m) (j : Fin n) (dX : Vec (Shape.size [m, n])) :
    centD (m := m) (n := n) i j dX =
      dX (idxMN (m := m) (n := n) i j) - 𝔼 k, dX (idxMN (m := m) (n := n) i k) := by
  simp [centD]

/-- Derivative of the row variance at `X`. -/
def varD (X : Vec (Shape.size [m, n])) (i : Fin m) : Vec (Shape.size [m, n]) →L[ℝ] ℝ :=
  ((n : ℝ)⁻¹) •
    ∑ k : Fin n,
      (centered X i k • centD (m := m) (n := n) i k + centered X i k • centD (m := m) (n := n) i k)

/-- The variance differential is `2 * mean (centered X * dX)`: the row mean of `dX` drops out
because the centered row sums to zero. -/
theorem varD_apply (X : Vec (Shape.size [m, n])) (i : Fin m)
    (dX : Vec (Shape.size [m, n])) :
    varD X i dX = 2 * (∑ k : Fin n, centered X i k * dX (idxMN (m := m) (n := n) i k)) / n := by
  have hsum := sum_centered_mul_sub X i (fun k => dX (idxMN (m := m) (n := n) i k))
    (𝔼 k, dX (idxMN (m := m) (n := n) i k))
  simp only at hsum
  simp only [varD, smul_apply, sum_apply, add_apply, centD_apply, smul_eq_mul]
  simp only [← two_mul]
  rw [← Finset.mul_sum, hsum]
  ring

/-- Derivative of `sqrt (var + ε)` at `X`. -/
def sqrtD (X : Vec (Shape.size [m, n])) (ε : ℝ) (i : Fin m) :
    Vec (Shape.size [m, n]) →L[ℝ] ℝ :=
  (1 / (2 * Real.sqrt (TorchLean.var (fun j => X (idxMN (m := m) (n := n) i j)) + ε))) • varD X i

/-- Derivative of the inverse standard deviation at `X`. -/
def invD (X : Vec (Shape.size [m, n])) (ε : ℝ) (i : Fin m) :
    Vec (Shape.size [m, n]) →L[ℝ] ℝ :=
  (-(Real.sqrt (TorchLean.var (fun j => X (idxMN (m := m) (n := n) i j)) + ε) ^ 2)⁻¹) • sqrtD X ε i

/-- Derivative of the normalized entry `(i, j)` at `X`. -/
def nrmD (X : Vec (Shape.size [m, n])) (ε : ℝ) (i : Fin m) (j : Fin n) :
    Vec (Shape.size [m, n]) →L[ℝ] ℝ :=
  centered X i j • invD X ε i + invStd X ε i • centD (m := m) (n := n) i j

/-- The row mean has derivative `meanD` at every point. -/
theorem hasFDerivAt_rowMean (X : Vec (Shape.size [m, n])) (i : Fin m) :
    HasFDerivAt (fun Y : Vec (Shape.size [m, n]) => 𝔼 j, Y (idxMN (m := m) (n := n) i j))
      (meanD (m := m) (n := n) i) X := by
  have h := (HasFDerivAt.sum (u := Finset.univ)
    (fun j _ => PiLp.hasFDerivAt_apply (𝕜 := ℝ) 2 X (idxMN (m := m) (n := n) i j))).mul_const
    ((n : ℝ)⁻¹)
  simpa [meanD, Fintype.expect_eq_sum_div_card, div_eq_mul_inv] using h

/-- A centered entry has derivative `centD` at every point. -/
theorem hasFDerivAt_centered (X : Vec (Shape.size [m, n])) (i : Fin m) (j : Fin n) :
    HasFDerivAt (fun Y : Vec (Shape.size [m, n]) => centered Y i j)
      (centD (m := m) (n := n) i j) X :=
  (PiLp.hasFDerivAt_apply (𝕜 := ℝ) 2 X _).sub (hasFDerivAt_rowMean X i)

/-- The row variance has derivative `varD X i` at `X`. -/
theorem hasFDerivAt_rowVar (X : Vec (Shape.size [m, n])) (i : Fin m) :
    HasFDerivAt (fun Y : Vec (Shape.size [m, n]) =>
        TorchLean.var fun j => Y (idxMN (m := m) (n := n) i j))
      (varD X i) X := by
  have h := (HasFDerivAt.sum (u := Finset.univ)
    (fun k _ => (hasFDerivAt_centered X i k).mul (hasFDerivAt_centered X i k))).mul_const
    ((n : ℝ)⁻¹)
  have hfun : (fun Y : Vec (Shape.size [m, n]) =>
        TorchLean.var fun j => Y (idxMN (m := m) (n := n) i j)) =
      fun Y => (∑ k : Fin n, centered Y i k * centered Y i k) * (n : ℝ)⁻¹ := by
    funext Y
    change 𝔼 k, centered Y i k * centered Y i k = _
    rw [Fintype.expect_eq_sum_div_card, Fintype.card_fin, div_eq_mul_inv]
  rw [hfun]
  simpa [varD] using h

/-- Derivative of the inverse standard deviation, for positive `ε`. -/
theorem hasFDerivAt_invStd {ε : ℝ} (hε : 0 < ε) (X : Vec (Shape.size [m, n])) (i : Fin m) :
    HasFDerivAt (fun Y : Vec (Shape.size [m, n]) => invStd Y ε i) (invD X ε i) X := by
  have hpos := rowVar_add_pos hε X i
  have hs := Real.sqrt_pos.2 hpos
  have hsqrt : HasFDerivAt
      (fun Y : Vec (Shape.size [m, n]) =>
        Real.sqrt (TorchLean.var (fun j => Y (idxMN (m := m) (n := n) i j)) + ε))
      (sqrtD X ε i) X :=
    ((hasFDerivAt_rowVar X i).add_const ε).sqrt hpos.ne'
  exact (hasDerivAt_inv hs.ne').comp_hasFDerivAt X hsqrt

/-- The normalized entry is differentiable for positive `ε`. -/
theorem hasFDerivAt_nrm {ε : ℝ} (hε : 0 < ε) (X : Vec (Shape.size [m, n])) (i : Fin m)
    (j : Fin n) :
    HasFDerivAt (fun Y : Vec (Shape.size [m, n]) => nrm Y ε i j) (nrmD X ε i j) X :=
  (hasFDerivAt_centered X i j).mul (hasFDerivAt_invStd hε X i)

/-- The derivative of the normalized entry is the closed-form JVP. -/
theorem nrmD_apply (hn : 0 < n) {ε : ℝ} (hε : 0 < ε) (X : Vec (Shape.size [m, n])) (i : Fin m)
    (j : Fin n) (dX : Vec (Shape.size [m, n])) :
    nrmD X ε i j dX = nrmJVP X ε dX i j := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  have hs := Real.sqrt_pos.2 (rowVar_add_pos hε X i)
  have hsum :
      ∑ k : Fin n, dX (idxMN (m := m) (n := n) i k) * nrm X ε i k =
        (∑ k : Fin n, centered X i k * dX (idxMN (m := m) (n := n) i k)) * invStd X ε i := by
    rw [Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro k _
    simp only [nrm]
    ring
  simp only [nrmD, invD, sqrtD, add_apply, smul_apply, smul_eq_mul, centD_apply, varD_apply,
    nrmJVP]
  rw [hsum]
  simp only [nrm, invStd]
  field_simp
  ring

/-! ## Whole-matrix normalization -/

/-- Row index of a flattened matrix coordinate. -/
def rowOf (ip : Fin (Shape.size [m, n])) : Fin m :=
  ip.divNat (m := m) (n := Shape.size [n])

/-- Column index of a flattened matrix coordinate. -/
def colOf (ip : Fin (Shape.size [m, n])) : Fin n :=
  Fin.cast (Shape.size_vector n) (ip.modNat (m := m) (n := Shape.size [n]))

/-- Every flattened coordinate is `idxMN` of its row and column. -/
theorem idxMN_rowOf_colOf (ip : Fin (Shape.size [m, n])) :
    idxMN (m := m) (n := n) (rowOf ip) (colOf ip) = ip :=
  idxMN_divNat_modNat ip

/-- Row of the flattened coordinate `idxMN i j`. -/
theorem rowOf_idxMN (i : Fin m) (j : Fin n) : rowOf (idxMN (m := m) (n := n) i j) = i := by
  apply Fin.ext
  change ((Fin.cast (Shape.size_vector n).symm j).val + Shape.size [n] * i.val) /
      Shape.size [n] = i.val
  rw [Fin.val_cast, Shape.size_vector, Nat.add_mul_div_left _ _ j.pos, Nat.div_eq_of_lt j.isLt,
    Nat.zero_add]

/-- Column of the flattened coordinate `idxMN i j`. -/
theorem colOf_idxMN (i : Fin m) (j : Fin n) : colOf (idxMN (m := m) (n := n) i j) = j := by
  apply Fin.ext
  change ((Fin.cast (Shape.size_vector n).symm j).val + Shape.size [n] * i.val) %
      Shape.size [n] = j.val
  rw [Fin.val_cast, Shape.size_vector, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt j.isLt]

/-- The tape row-mean map computes the mean `𝔼 j, x (idxMN i j)` of each row `i`. -/
theorem rowMeanCLM_eq (x : Vec (Shape.size [m, n])) (i : Fin m) :
    Tape.RealNode.MatrixLinear.rowMeanCLM (m := m) (n := n) x i =
      𝔼 j, x (idxMN (m := m) (n := n) i j) := by
  show ((1 : ℝ) / (n : ℝ)) * ∑ j : Fin (Shape.size [n]), x (finProdFinEquiv (i, j)) = _
  rw [Fintype.expect_eq_sum_div_card, Fintype.card_fin, one_div, div_eq_inv_mul]
  congr 1
  exact (Fin.sum_congr' (fun j => x (finProdFinEquiv (i, j))) (Shape.size_vector n).symm).symm

/-- The tape row-broadcast map at `idxMN i j` reads entry `i`. -/
theorem broadcastRowCLM_idxMN (v : Vec m) (i : Fin m) (j : Fin n) :
    Tape.RealNode.MatrixLinear.broadcastRowCLM (m := m) (n := n) v (idxMN (m := m) (n := n) i j) =
      v i := by
  show v (rowOf (idxMN (m := m) (n := n) i j)) = v i
  rw [rowOf_idxMN]

/-- The tape column-broadcast map at `idxMN i j` reads entry `j`. -/
theorem broadcastColCLM_idxMN (v : Vec n) (i : Fin m) (j : Fin n) :
    Tape.RealNode.MatrixLinear.broadcastColCLM (m := m) (n := n) v (idxMN (m := m) (n := n) i j) =
      v j := by
  show v (colOf (idxMN (m := m) (n := n) i j)) = v j
  rw [colOf_idxMN]

/-- Normalize every row of a flattened matrix. -/
def nrmVec (X : Vec (Shape.size [m, n])) (ε : ℝ) : Vec (Shape.size [m, n]) :=
  WithLp.toLp 2 fun ip => nrm X ε (rowOf ip) (colOf ip)

/-- Coordinate `ip` of the normalized matrix is the normalized entry at its row and column. -/
@[simp] theorem nrmVec_apply (X : Vec (Shape.size [m, n])) (ε : ℝ)
    (ip : Fin (Shape.size [m, n])) :
    nrmVec X ε ip = nrm X ε (rowOf ip) (colOf ip) := by
  simp [nrmVec]

/-- Closed-form JVP of `nrmVec` at `X`, packaged as a continuous linear map. -/
def nrmJVPCLM (X : Vec (Shape.size [m, n])) (ε : ℝ) :
    Vec (Shape.size [m, n]) →L[ℝ] Vec (Shape.size [m, n]) :=
  let fLin : Vec (Shape.size [m, n]) →ₗ[ℝ] Vec (Shape.size [m, n]) :=
    { toFun := fun dX => WithLp.toLp 2 fun ip => nrmJVP X ε dX (rowOf ip) (colOf ip)
      map_add' := by
        intro a b
        apply PiLp.ext
        intro ip
        simp only [PiLp.add_apply, nrmJVP, Finset.expect_add_distrib, Finset.sum_add_distrib,
          add_mul, add_div]
        ring
      map_smul' := by
        intro r a
        apply PiLp.ext
        intro ip
        simp only [PiLp.smul_apply, smul_eq_mul, RingHom.id_apply, nrmJVP,
          ← Finset.mul_expect, mul_assoc, ← Finset.mul_sum]
        ring }
  { toLinearMap := fLin, cont := LinearMap.continuous_of_finiteDimensional fLin }

/-- The bundled JVP agrees with the closed-form `nrmJVP` coordinatewise. -/
@[simp] theorem nrmJVPCLM_apply (X : Vec (Shape.size [m, n])) (ε : ℝ)
    (dX : Vec (Shape.size [m, n])) (ip : Fin (Shape.size [m, n])) :
    nrmJVPCLM X ε dX ip = nrmJVP X ε dX (rowOf ip) (colOf ip) := by
  simp [nrmJVPCLM]

/-- Row normalization is differentiable on a nonempty feature axis, with the closed-form JVP
as derivative. The coordinatewise chain-rule calculation agrees with that JVP because positive
epsilon keeps each row's denominator nonzero. -/
theorem hasFDerivAt_nrmVec (hn : 0 < n) {ε : ℝ} (hε : 0 < ε) (X : Vec (Shape.size [m, n])) :
    HasFDerivAt (fun Y : Vec (Shape.size [m, n]) => nrmVec Y ε) (nrmJVPCLM X ε) X := by
  rw [← hasFDerivWithinAt_univ, hasFDerivWithinAt_euclidean]
  intro ip
  rw [hasFDerivWithinAt_univ]
  have h := hasFDerivAt_nrm hε X (rowOf ip) (colOf ip)
  have hfun : (fun Y : Vec (Shape.size [m, n]) => nrmVec Y ε ip) =
      fun Y : Vec (Shape.size [m, n]) => nrm Y ε (rowOf ip) (colOf ip) := by
    funext Y
    exact nrmVec_apply Y ε ip
  rw [hfun]
  refine h.congr_fderiv ?_
  ext dX
  simp [nrmD_apply hn hε]

end

end RowNorm
end TorchLean
