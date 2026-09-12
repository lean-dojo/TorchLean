/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Nodes.Matrix


/-!
# Calculus of row normalization

LayerNorm normalizes every row of a `[seqLen, embedDim]` matrix; the flattened form of BatchNorm
normalizes every row of a `[channels, positions]` matrix. Both share the map

`x ↦ (x - mean(x)) / sqrt (var(x) + ε)`

applied to each row, where `mean` and `var` are the population statistics of that row. This file
proves that map is differentiable on flattened matrices (for `0 < ε`) and identifies its
derivative with the closed form

`dx ↦ invStd * (dx - mean(dx) - xhat * mean(dx * xhat))`

used by the normalization JVPs.
-/

@[expose] public section

namespace Proofs
namespace Autograd
namespace RowNorm

open TapeNodes.Matmul

open scoped BigOperators

noncomputable section

variable {m n : Nat}

/-- Mean of row `i` of a flattened `m × n` matrix. -/
def rowMean (X : Vec (matSize m n)) (i : Fin m) : ℝ :=
  (∑ j : Fin n, X (idxMN (m := m) (n := n) i j)) / n

/-- Centered entry `(i, j)`. -/
def centered (X : Vec (matSize m n)) (i : Fin m) (j : Fin n) : ℝ :=
  X (idxMN (m := m) (n := n) i j) - rowMean X i

/-- Population variance of row `i`. -/
def rowVar (X : Vec (matSize m n)) (i : Fin m) : ℝ :=
  (∑ j : Fin n, centered X i j * centered X i j) / n

/-- Inverse standard deviation `1 / sqrt (var + ε)` of row `i`. -/
def invStd (X : Vec (matSize m n)) (ε : ℝ) (i : Fin m) : ℝ :=
  (Real.sqrt (rowVar X i + ε))⁻¹

/-- Normalized entry `(i, j)`. -/
def nrm (X : Vec (matSize m n)) (ε : ℝ) (i : Fin m) (j : Fin n) : ℝ :=
  centered X i j * invStd X ε i

/-- Closed-form differential of the normalized entry `(i, j)` in the direction `dX`. -/
def nrmJvp (X : Vec (matSize m n)) (ε : ℝ) (dX : Vec (matSize m n)) (i : Fin m) (j : Fin n) :
    ℝ :=
  invStd X ε i *
    (dX (idxMN (m := m) (n := n) i j) - rowMean dX i -
      nrm X ε i j * ((∑ k : Fin n, dX (idxMN (m := m) (n := n) i k) * nrm X ε i k) / n))

/-- The row variance is a mean of squares. -/
theorem rowVar_nonneg (X : Vec (matSize m n)) (i : Fin m) : 0 ≤ rowVar X i :=
  div_nonneg (Finset.sum_nonneg fun _ _ => mul_self_nonneg _) (Nat.cast_nonneg n)

/-- `var + ε` is positive when `ε` is. -/
theorem rowVar_add_pos {ε : ℝ} (hε : 0 < ε) (X : Vec (matSize m n)) (i : Fin m) :
    0 < rowVar X i + ε :=
  add_pos_of_nonneg_of_pos (rowVar_nonneg X i) hε

/-- The centered entries of a row sum to zero. -/
theorem sum_centered (hn : 0 < n) (X : Vec (matSize m n)) (i : Fin m) :
    ∑ j : Fin n, centered X i j = 0 := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  simp only [centered, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul, rowMean]
  field_simp
  ring

/-- The centered row is orthogonal to constants, so `∑ c_j (d_j - a) = ∑ c_j d_j`. -/
theorem sum_centered_mul_sub (hn : 0 < n) (X : Vec (matSize m n)) (i : Fin m)
    (d : Fin n → ℝ) (a : ℝ) :
    ∑ j : Fin n, centered X i j * (d j - a) = ∑ j : Fin n, centered X i j * d j := by
  simp only [mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul, sum_centered hn X i, zero_mul,
    sub_zero]

/-! ## Derivative building blocks -/

/-- Coordinate projection as a continuous linear functional on flattened matrices. -/
def projCLM (k : Fin (matSize m n)) : Vec (matSize m n) →L[ℝ] ℝ :=
  EuclideanSpace.proj (𝕜 := ℝ) (ι := Fin (matSize m n)) k

/-- The projection functional reads coordinate `k`. -/
@[simp] theorem projCLM_apply (k : Fin (matSize m n)) (v : Vec (matSize m n)) :
    projCLM (m := m) (n := n) k v = v k := rfl

/-- Derivative of the row mean. -/
def meanD (i : Fin m) : Vec (matSize m n) →L[ℝ] ℝ :=
  ((n : ℝ)⁻¹) • ∑ j : Fin n, projCLM (m := m) (n := n) (idxMN (m := m) (n := n) i j)

/--
The row-mean derivative is the row mean of the perturbation, since the mean is already linear.
-/
@[simp] theorem meanD_apply (i : Fin m) (dX : Vec (matSize m n)) :
    meanD (m := m) (n := n) i dX = rowMean dX i := by
  simp [meanD, rowMean, div_eq_inv_mul]

/-- Derivative of a centered entry. -/
def centD (i : Fin m) (j : Fin n) : Vec (matSize m n) →L[ℝ] ℝ :=
  projCLM (m := m) (n := n) (idxMN (m := m) (n := n) i j) - meanD (m := m) (n := n) i

/-- The centering derivative subtracts the perturbation's own row mean.

Centering is linear too, so its derivative is itself; the nonlinearity in row normalization enters
only
through the variance and the square root below. -/
@[simp] theorem centD_apply (i : Fin m) (j : Fin n) (dX : Vec (matSize m n)) :
    centD (m := m) (n := n) i j dX = dX (idxMN (m := m) (n := n) i j) - rowMean dX i := by
  simp [centD]

/-- Derivative of the row variance at `X`. -/
def varD (X : Vec (matSize m n)) (i : Fin m) : Vec (matSize m n) →L[ℝ] ℝ :=
  ((n : ℝ)⁻¹) •
    ∑ k : Fin n,
      (centered X i k • centD (m := m) (n := n) i k + centered X i k • centD (m := m) (n := n) i k)

/-- The variance differential only sees the centered direction. -/
theorem varD_apply (hn : 0 < n) (X : Vec (matSize m n)) (i : Fin m) (dX : Vec (matSize m n)) :
    varD X i dX = 2 * (∑ k : Fin n, centered X i k * dX (idxMN (m := m) (n := n) i k)) / n := by
  have hsum := sum_centered_mul_sub hn X i (fun k => dX (idxMN (m := m) (n := n) i k))
    (rowMean dX i)
  simp only at hsum
  simp only [varD, smul_apply, sum_apply, add_apply, centD_apply, smul_eq_mul]
  simp only [← two_mul]
  rw [← Finset.mul_sum, hsum]
  ring

/-- Derivative of `sqrt (var + ε)` at `X`. -/
def sqrtD (X : Vec (matSize m n)) (ε : ℝ) (i : Fin m) : Vec (matSize m n) →L[ℝ] ℝ :=
  (1 / (2 * Real.sqrt (rowVar X i + ε))) • varD X i

/-- Derivative of the inverse standard deviation at `X`. -/
def invD (X : Vec (matSize m n)) (ε : ℝ) (i : Fin m) : Vec (matSize m n) →L[ℝ] ℝ :=
  (-(Real.sqrt (rowVar X i + ε) ^ 2)⁻¹) • sqrtD X ε i

/-- Derivative of the normalized entry `(i, j)` at `X`. -/
def nrmD (X : Vec (matSize m n)) (ε : ℝ) (i : Fin m) (j : Fin n) : Vec (matSize m n) →L[ℝ] ℝ :=
  centered X i j • invD X ε i + invStd X ε i • centD (m := m) (n := n) i j

/-- Coordinates of a flattened matrix are differentiable. -/
theorem hasFDerivAt_coord (X : Vec (matSize m n)) (k : Fin (matSize m n)) :
    HasFDerivAt (fun Y : Vec (matSize m n) => Y k) (projCLM (m := m) (n := n) k) X :=
  (projCLM (m := m) (n := n) k).hasFDerivAt

/-- Derivative of the row mean. -/
theorem hasFDerivAt_rowMean (X : Vec (matSize m n)) (i : Fin m) :
    HasFDerivAt (fun Y : Vec (matSize m n) => rowMean Y i) (meanD (m := m) (n := n) i) X := by
  have h := (HasFDerivAt.sum (u := Finset.univ)
    (fun j _ => hasFDerivAt_coord X (idxMN (m := m) (n := n) i j))).mul_const ((n : ℝ)⁻¹)
  simpa [rowMean, meanD, div_eq_mul_inv] using h

/-- Derivative of a centered entry. -/
theorem hasFDerivAt_centered (X : Vec (matSize m n)) (i : Fin m) (j : Fin n) :
    HasFDerivAt (fun Y : Vec (matSize m n) => centered Y i j) (centD (m := m) (n := n) i j) X :=
  (hasFDerivAt_coord X _).sub (hasFDerivAt_rowMean X i)

/-- Derivative of the row variance. -/
theorem hasFDerivAt_rowVar (X : Vec (matSize m n)) (i : Fin m) :
    HasFDerivAt (fun Y : Vec (matSize m n) => rowVar Y i) (varD X i) X := by
  have h := (HasFDerivAt.sum (u := Finset.univ)
    (fun k _ => (hasFDerivAt_centered X i k).mul (hasFDerivAt_centered X i k))).mul_const
    ((n : ℝ)⁻¹)
  simpa [rowVar, varD, div_eq_mul_inv] using h

/-- Derivative of the inverse standard deviation, for positive `ε`. -/
theorem hasFDerivAt_invStd {ε : ℝ} (hε : 0 < ε) (X : Vec (matSize m n)) (i : Fin m) :
    HasFDerivAt (fun Y : Vec (matSize m n) => invStd Y ε i) (invD X ε i) X := by
  have hpos : 0 < rowVar X i + ε := rowVar_add_pos hε X i
  have hs : 0 < Real.sqrt (rowVar X i + ε) := Real.sqrt_pos.2 hpos
  have hsqrt : HasFDerivAt (fun Y : Vec (matSize m n) => Real.sqrt (rowVar Y i + ε))
      (sqrtD X ε i) X :=
    ((hasFDerivAt_rowVar X i).add_const ε).sqrt hpos.ne'
  exact (hasDerivAt_inv hs.ne').comp_hasFDerivAt X hsqrt

/-- The normalized entry is differentiable for positive `ε`. -/
theorem hasFDerivAt_nrm {ε : ℝ} (hε : 0 < ε) (X : Vec (matSize m n)) (i : Fin m) (j : Fin n) :
    HasFDerivAt (fun Y : Vec (matSize m n) => nrm Y ε i j) (nrmD X ε i j) X :=
  (hasFDerivAt_centered X i j).mul (hasFDerivAt_invStd hε X i)

/-- The derivative of the normalized entry is the closed-form JVP. -/
theorem nrmD_apply (hn : 0 < n) {ε : ℝ} (hε : 0 < ε) (X : Vec (matSize m n)) (i : Fin m)
    (j : Fin n) (dX : Vec (matSize m n)) :
    nrmD X ε i j dX = nrmJvp X ε dX i j := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  have hs : 0 < Real.sqrt (rowVar X i + ε) := Real.sqrt_pos.2 (rowVar_add_pos hε X i)
  have hsum :
      ∑ k : Fin n, dX (idxMN (m := m) (n := n) i k) * nrm X ε i k =
        (∑ k : Fin n, centered X i k * dX (idxMN (m := m) (n := n) i k)) * invStd X ε i := by
    rw [Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro k _
    simp only [nrm]
    ring
  simp only [nrmD, invD, sqrtD, add_apply, smul_apply, smul_eq_mul, centD_apply, varD_apply hn,
    nrmJvp]
  rw [hsum]
  simp only [nrm, invStd]
  field_simp
  ring

/-! ## Whole-matrix normalization -/

/-- Row index of a flattened matrix coordinate. -/
def rowOf (ip : Fin (matSize m n)) : Fin m :=
  ip.divNat (m := m) (n := vecSize n)

/-- Column index of a flattened matrix coordinate. -/
def colOf (ip : Fin (matSize m n)) : Fin n :=
  Fin.cast (vecSize_eq n) (ip.modNat (m := m) (n := vecSize n))

/-- Every flattened coordinate is `idxMN` of its row and column. -/
theorem idxMN_rowOf_colOf (ip : Fin (matSize m n)) :
    idxMN (m := m) (n := n) (rowOf ip) (colOf ip) = ip := by
  apply Fin.ext
  change (ip.modNat (m := m) (n := vecSize n)).val +
      vecSize n * (ip.divNat (m := m) (n := vecSize n)).val = ip.val
  exact Nat.mod_add_div _ _

/-- Row of the flattened coordinate `idxMN i j`. -/
theorem rowOf_idxMN (i : Fin m) (j : Fin n) : rowOf (idxMN (m := m) (n := n) i j) = i := by
  apply Fin.ext
  change ((Fin.cast (vecSize_eq n).symm j).val + vecSize n * i.val) / vecSize n = i.val
  rw [Fin.val_cast, vecSize_eq, Nat.add_mul_div_left _ _ j.pos, Nat.div_eq_of_lt j.isLt,
    Nat.zero_add]

/-- Column of the flattened coordinate `idxMN i j`. -/
theorem colOf_idxMN (i : Fin m) (j : Fin n) : colOf (idxMN (m := m) (n := n) i j) = j := by
  apply Fin.ext
  change ((Fin.cast (vecSize_eq n).symm j).val + vecSize n * i.val) % vecSize n = j.val
  rw [Fin.val_cast, vecSize_eq, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt j.isLt]

/-- Reindex a sum over `Fin (vecSize n)` as a sum over `Fin n`. -/
theorem sum_vecSize (f : Fin (vecSize n) → ℝ) :
    ∑ j : Fin (vecSize n), f j = ∑ j : Fin n, f (Fin.cast (vecSize_eq n).symm j) :=
  (Fintype.sum_equiv (finCongr (vecSize_eq n).symm) _ _ (fun _ => rfl)).symm

/-- The tape row-mean map computes `rowMean`. -/
theorem rowMeanCLM_eq (x : Vec (matSize m n)) (i : Fin m) :
    TapeNodes.MatrixLinear.rowMeanCLM (m := m) (n := n) x i = rowMean x i := by
  show ((1 : ℝ) / (n : ℝ)) * ∑ j : Fin (vecSize n), x (finProdFinEquiv (i, j)) = rowMean x i
  rw [rowMean, one_div, div_eq_inv_mul]
  congr 1
  exact Fintype.sum_equiv (finCongr (vecSize_eq n)) _ _ (fun _ => rfl)

/-- The tape row-broadcast map at `idxMN i j` reads entry `i`. -/
theorem broadcastRowCLM_idxMN (v : Vec m) (i : Fin m) (j : Fin n) :
    TapeNodes.MatrixLinear.broadcastRowCLM (m := m) (n := n) v (idxMN (m := m) (n := n) i j) =
      v i := by
  show v (rowOf (idxMN (m := m) (n := n) i j)) = v i
  rw [rowOf_idxMN]

/-- The tape column-broadcast map at `idxMN i j` reads entry `j`. -/
theorem broadcastColCLM_idxMN (v : Vec n) (i : Fin m) (j : Fin n) :
    TapeNodes.MatrixLinear.broadcastColCLM (m := m) (n := n) v (idxMN (m := m) (n := n) i j) =
      v j := by
  show v (colOf (idxMN (m := m) (n := n) i j)) = v j
  rw [colOf_idxMN]

/-- Normalize every row of a flattened matrix. -/
def nrmVec (X : Vec (matSize m n)) (ε : ℝ) : Vec (matSize m n) :=
  vecOfFun (n := matSize m n) fun ip => nrm X ε (rowOf ip) (colOf ip)

/-- Coordinate `ip` of the normalized matrix is the normalized entry at its row and column. -/
@[simp] theorem nrmVec_apply (X : Vec (matSize m n)) (ε : ℝ) (ip : Fin (matSize m n)) :
    nrmVec X ε ip = nrm X ε (rowOf ip) (colOf ip) := by
  simp [nrmVec]

/-- Closed-form JVP of `nrmVec` at `X`, packaged as a continuous linear map. -/
def nrmJvpCLM (X : Vec (matSize m n)) (ε : ℝ) : Vec (matSize m n) →L[ℝ] Vec (matSize m n) := by
  classical
  let fLin : Vec (matSize m n) →ₗ[ℝ] Vec (matSize m n) :=
    { toFun := fun dX => vecOfFun (n := matSize m n) fun ip => nrmJvp X ε dX (rowOf ip) (colOf ip)
      map_add' := by
        intro a b
        apply PiLp.ext
        intro ip
        simp only [vecOfFun_ofLp, PiLp.add_apply, nrmJvp, rowMean, Finset.sum_add_distrib,
          add_mul, add_div]
        ring
      map_smul' := by
        intro r a
        apply PiLp.ext
        intro ip
        simp only [vecOfFun_ofLp, PiLp.smul_apply, smul_eq_mul, RingHom.id_apply, nrmJvp,
          rowMean, mul_assoc, ← Finset.mul_sum]
        ring }
  exact { toLinearMap := fLin, cont := LinearMap.continuous_of_finiteDimensional fLin }

/-- The bundled JVP agrees with the closed-form `nrmJvp` coordinatewise.

Writing the derivative down in closed form and then proving `HasFDerivAt` against it, rather than
deriving it compositionally, is what keeps the `ε` guard visible: the formula is only the derivative
because `ε > 0` keeps the denominator away from zero. -/
@[simp] theorem nrmJvpCLM_apply (X : Vec (matSize m n)) (ε : ℝ) (dX : Vec (matSize m n))
    (ip : Fin (matSize m n)) :
    nrmJvpCLM X ε dX ip = nrmJvp X ε dX (rowOf ip) (colOf ip) := by
  simp [nrmJvpCLM]

/-- Row normalization of a flattened matrix is differentiable for positive `ε`, with the
closed-form JVP as derivative. -/
theorem hasFDerivAt_nrmVec (hn : 0 < n) {ε : ℝ} (hε : 0 < ε) (X : Vec (matSize m n)) :
    HasFDerivAt (fun Y : Vec (matSize m n) => nrmVec Y ε) (nrmJvpCLM X ε) X := by
  rw [← hasFDerivWithinAt_univ, hasFDerivWithinAt_euclidean]
  intro ip
  rw [hasFDerivWithinAt_univ]
  have h := hasFDerivAt_nrm hε X (rowOf ip) (colOf ip)
  have hfun : (fun Y : Vec (matSize m n) => nrmVec Y ε ip) =
      fun Y : Vec (matSize m n) => nrm Y ε (rowOf ip) (colOf ip) := by
    funext Y
    exact nrmVec_apply Y ε ip
  rw [hfun]
  refine h.congr_fderiv ?_
  ext dX
  simp [nrmD_apply hn hε]

end

end RowNorm
end Autograd
end Proofs
