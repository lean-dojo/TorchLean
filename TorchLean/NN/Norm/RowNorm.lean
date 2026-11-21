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
population variance `TorchLean.var`. At a matrix whose rows have a single entry or each a nonzero
sum of variance and `ε`, the map has the closed-form derivative

`dx ↦ invStd * (dx - 𝔼 dx - xhat * 𝔼 (dx * xhat))`

used by the normalization JVPs (`hasFDerivAt_nrm_of_ne`, `hasFDerivAt_nrmVec`). Where the sum is
negative, the normalized row is zero nearby, since the real square root of a negative number is
zero, and a row with a single entry is zero once centered. Every `0 < ε` meets the condition
(`rowVar_add_pos`).
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

/-- If row `i` of `X` has a positive sum of its variance and `ε`, the inverse standard deviation of
that row has derivative `invD` at `X`. -/
theorem hasFDerivAt_invStd {ε : ℝ} (X : Vec (Shape.size [m, n])) (i : Fin m)
    (hpos : 0 < TorchLean.var (fun j => X (idxMN (m := m) (n := n) i j)) + ε) :
    HasFDerivAt (fun Y : Vec (Shape.size [m, n]) => invStd Y ε i) (invD X ε i) X :=
  (hasDerivAt_inv (Real.sqrt_pos.2 hpos).ne').comp_hasFDerivAt X
    (((hasFDerivAt_rowVar X i).add_const ε).sqrt hpos.ne')

/-- If row `i` of `X` has a positive sum of its variance and `ε`, the normalized entry `(i, j)` has
derivative `nrmD` at `X`. -/
theorem hasFDerivAt_nrm {ε : ℝ} (X : Vec (Shape.size [m, n])) (i : Fin m) (j : Fin n)
    (hpos : 0 < TorchLean.var (fun k => X (idxMN (m := m) (n := n) i k)) + ε) :
    HasFDerivAt (fun Y : Vec (Shape.size [m, n]) => nrm Y ε i j) (nrmD X ε i j) X :=
  (hasFDerivAt_centered X i j).mul (hasFDerivAt_invStd X i hpos)

/-- If row `i` of `X` has a positive sum of its variance and `ε`, the derivative `nrmD` of the
normalized entry `(i, j)` at `X` is the closed-form JVP `nrmJVP`. -/
theorem nrmD_apply {ε : ℝ} (X : Vec (Shape.size [m, n])) (i : Fin m) (j : Fin n)
    (hpos : 0 < TorchLean.var (fun k => X (idxMN (m := m) (n := n) i k)) + ε)
    (dX : Vec (Shape.size [m, n])) :
    nrmD X ε i j dX = nrmJVP X ε dX i j := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast j.pos.ne'
  have hs := Real.sqrt_pos.2 hpos
  have hsum :
      ∑ k : Fin n, dX (idxMN (m := m) (n := n) i k) * nrm X ε i k =
        (∑ k : Fin n, centered X i k * dX (idxMN (m := m) (n := n) i k)) * invStd X ε i := by
    rw [Finset.sum_mul]
    exact Finset.sum_congr rfl fun k _ => by simp only [nrm]; ring
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

/-- If the rows have a single entry or row `i` of `X` has a nonzero sum of its variance and `ε`,
the normalized entry `(i, j)` has the closed-form JVP as derivative at `X`. -/
theorem hasFDerivAt_nrm_of_ne {ε : ℝ} (X : Vec (Shape.size [m, n])) (i : Fin m) (j : Fin n)
    (h : n = 1 ∨ TorchLean.var (fun k => X (idxMN (m := m) (n := n) i k)) + ε ≠ 0) :
    HasFDerivAt (fun Y : Vec (Shape.size [m, n]) => nrm Y ε i j)
      ((EuclideanSpace.proj (𝕜 := ℝ) (idxMN (m := m) (n := n) i j) :
        Vec (Shape.size [m, n]) →L[ℝ] ℝ).comp (nrmJVPCLM X ε)) X := by
  rcases h with rfl | hne
  · have hc : ∀ Y : Vec (Shape.size [m, 1]), centered Y i j = 0 := fun Y => by
      rw [centered, Finset.expect_congr rfl fun k _ =>
        congrArg (fun k => Y (idxMN (m := m) (n := 1) i k)) (Subsingleton.elim k j),
        Fintype.expect_const, sub_self]
    have hf : (fun Y : Vec (Shape.size [m, 1]) => nrm Y ε i j) = fun _ => 0 :=
      funext fun Y => by rw [nrm, hc, zero_mul]
    rw [hf]
    refine (hasFDerivAt_const 0 X).congr_fderiv (ContinuousLinearMap.ext fun dX => ?_)
    have hd := hc dX
    rw [centered] at hd
    simp only [_root_.zero_apply, ContinuousLinearMap.comp_apply, PiLp.proj_apply, nrmJVPCLM_apply,
      rowOf_idxMN, colOf_idxMN, nrmJVP, hd, nrm, hc, zero_mul, sub_zero, mul_zero]
  · rcases hne.lt_or_gt with hneg | hpos
    · have hz : ∀ Y : Vec (Shape.size [m, n]),
          TorchLean.var (fun k => Y (idxMN (m := m) (n := n) i k)) + ε < 0 → invStd Y ε i = 0 :=
        fun Y hY => by rw [invStd, Real.sqrt_eq_zero_of_nonpos hY.le, inv_zero]
      have hev : ∀ᶠ Y in nhds X,
          TorchLean.var (fun k => Y (idxMN (m := m) (n := n) i k)) + ε < 0 :=
        ((hasFDerivAt_rowVar X i).continuousAt.add continuousAt_const).eventually_lt
          continuousAt_const hneg
      refine ((hasFDerivAt_const (0 : ℝ) X).congr_of_eventuallyEq
        (hev.mono fun Y hY => ?_)).congr_fderiv (ContinuousLinearMap.ext fun dX => ?_)
      · show nrm Y ε i j = 0
        rw [nrm, hz Y hY, mul_zero]
      · simp only [_root_.zero_apply, ContinuousLinearMap.comp_apply, PiLp.proj_apply,
          nrmJVPCLM_apply, rowOf_idxMN, colOf_idxMN, nrmJVP, hz X hneg, zero_mul]
    · refine (hasFDerivAt_nrm X i j hpos).congr_fderiv (ContinuousLinearMap.ext fun dX => ?_)
      rw [nrmD_apply X i j hpos]
      simp only [ContinuousLinearMap.comp_apply, PiLp.proj_apply, nrmJVPCLM_apply, rowOf_idxMN,
        colOf_idxMN]

/-- If the rows have a single entry or every row of `X` has a nonzero sum of its variance and `ε`,
row normalization has the closed-form JVP `nrmJVPCLM` as derivative at `X`. -/
theorem hasFDerivAt_nrmVec {ε : ℝ} (X : Vec (Shape.size [m, n]))
    (h : n = 1 ∨ ∀ i : Fin m, TorchLean.var (fun k => X (idxMN (m := m) (n := n) i k)) + ε ≠ 0) :
    HasFDerivAt (fun Y : Vec (Shape.size [m, n]) => nrmVec Y ε) (nrmJVPCLM X ε) X := by
  rw [← hasFDerivWithinAt_univ, hasFDerivWithinAt_euclidean]
  intro ip
  rw [hasFDerivWithinAt_univ]
  have hfun : (fun Y : Vec (Shape.size [m, n]) => nrmVec Y ε ip) =
      fun Y => nrm Y ε (rowOf ip) (colOf ip) :=
    funext fun Y => nrmVec_apply Y ε ip
  rw [hfun]
  have hd := hasFDerivAt_nrm_of_ne X (rowOf ip) (colOf ip) (h.imp_right fun h => h (rowOf ip))
  rwa [idxMN_rowOf_colOf] at hd

end

end RowNorm
end TorchLean
