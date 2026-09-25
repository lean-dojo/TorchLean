/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Ops.Norm.LayerNormBounds

/-!
# Mixed LayerNorm derivatives

We differentiate the actual row differential, allowing both the input and its left direction
to vary. The latter variation supplies the upstream mixed derivative in the chain rule.
Centered input and direction bounds then enclose the result using only positive epsilon.
The affine scale is fixed; its absolute value multiplies the final radius.
-/

@[expose] public section

namespace Proofs.Autograd.RowNorm

open TorchLean TapeNodes.Matmul
open scoped BigOperators

noncomputable section

variable {m n : Nat}

/-- Variance differential written entirely in centered coordinates. -/
def varianceFirst (X A : Vec (matSize m n)) (i : Fin m) : ℝ :=
  2 * (∑ k, centered X i k * centered A i k) / n

/-- Mixed variance differential, including the upstream mixed direction `C`. -/
def varianceMixed (X A B C : Vec (matSize m n)) (i : Fin m) : ℝ :=
  2 * (∑ k, (centered A i k * centered B i k + centered X i k * centered C i k)) / n

/-- Inverse-standard-deviation differential in centered coordinates. -/
def inverseFirst (X A : Vec (matSize m n)) (ε : ℝ) (i : Fin m) : ℝ :=
  -(1 / 2 : ℝ) * invStd X ε i ^ 3 * varianceFirst X A i

/-- Mixed inverse-standard-deviation differential. -/
def inverseMixed (X A B C : Vec (matSize m n)) (ε : ℝ) (i : Fin m) : ℝ :=
  (3 / 4 : ℝ) * invStd X ε i ^ 5 * varianceFirst X A i * varianceFirst X B i -
    (1 / 2 : ℝ) * invStd X ε i ^ 3 * varianceMixed X A B C i

/-- The mixed row chain rule: the Hessian term and the input's mixed differential. -/
def normalizedMixed (X A B C : Vec (matSize m n)) (ε : ℝ)
    (i : Fin m) (j : Fin n) : ℝ :=
  centered C i j * invStd X ε i +
    centered A i j * inverseFirst X B ε i +
    centered B i j * inverseFirst X A ε i +
    centered X i j * inverseMixed X A B C ε i

theorem varD_apply_centered (X A : Vec (matSize m n)) (i : Fin m) :
    varD X i A = varianceFirst X A i := by
  simp only [varD, smul_apply, sum_apply, add_apply, centD_apply, smul_eq_mul,
    ← two_mul, ← Finset.mul_sum, varianceFirst, centered]
  ring

theorem invD_apply_power {ε : ℝ} (hε : 0 < ε)
    (X A : Vec (matSize m n)) (i : Fin m) :
    invD X ε i A = inverseFirst X A ε i := by
  have hs := (Real.sqrt_pos.mpr (rowVar_add_pos hε X i)).ne'
  simp only [invD, sqrtD, smul_apply, smul_eq_mul, varD_apply_centered,
    inverseFirst, invStd]
  field_simp [hs]

/-- The existing Fréchet differential agrees with the centered product-rule formula. -/
theorem nrmD_apply_centered {ε : ℝ} (hε : 0 < ε)
    (X A : Vec (matSize m n)) (i : Fin m) (j : Fin n) :
    nrmD X ε i j A =
      centered A i j * invStd X ε i + centered X i j * inverseFirst X A ε i := by
  simp only [nrmD, add_apply, smul_apply, smul_eq_mul, invD_apply_power hε,
    centD_apply, centered]
  ring

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

private theorem exists_nrmD_comp_derivative {ε : ℝ} (hε : 0 < ε)
    (X A : E → Vec (matSize m n)) (p : E)
    (DX DA : E →L[ℝ] Vec (matSize m n))
    (hX : HasFDerivAt X DX p) (hA : HasFDerivAt A DA p) (i : Fin m) (j : Fin n) :
    ∃ D : E →L[ℝ] ℝ,
      HasFDerivAt (fun q => nrmD (X q) ε i j (A q)) D p ∧
        ∀ v, D v = normalizedMixed (X p) (A p) (DX v) (DA v) ε i j := by
  have hz (k : Fin n) := (hasFDerivAt_centered (X p) i k).comp p hX
  have ha (k : Fin n) := (hasFDerivAt_centered (A p) i k).comp p hA
  have hr := (hasFDerivAt_invStd hε (X p) i).comp p hX
  have hq := ((HasFDerivAt.sum (u := Finset.univ)
    (fun k _ => (hz k).mul (ha k))).const_mul 2).mul_const ((n : ℝ)⁻¹)
  simp only [← div_eq_mul_inv] at hq
  have hrl := (((hr.pow 3).mul hq).const_mul (-(1 / 2 : ℝ)))
  have hout := ((ha j).mul hr).add ((hz j).mul hrl)
  have heq :
      (fun q => centered (A q) i j * invStd (X q) ε i +
        centered (X q) i j *
          (-(1 / 2 : ℝ) * (invStd (X q) ε i ^ 3 *
            (2 * (∑ k, centered (X q) i k * centered (A q) i k) / n)))) =
        fun q => nrmD (X q) ε i j (A q) := by
    funext q
    rw [nrmD_apply_centered hε]
    simp only [inverseFirst, varianceFirst]
    ring
  simp only [Function.comp_def, Pi.mul_apply, Finset.sum_apply] at hout
  have hout' := hout.congr_of_eventuallyEq
    (Filter.Eventually.of_forall fun q => (congrFun heq q).symm)
  refine ⟨_, hout', fun v => ?_⟩
  simp only [add_apply, smul_apply, sum_apply, ContinuousLinearMap.comp_apply,
    smul_eq_mul, nsmul_eq_mul, centD_apply, invD_apply_power hε]
  have hsum :
      (∑ k, (centered (X p) i k * centered (DA v) i k +
        centered (A p) i k * centered (DX v) i k)) =
      ∑ k, (centered (A p) i k * centered (DX v) i k +
        centered (X p) i k * centered (DA v) i k) := by
    apply Finset.sum_congr rfl
    intro k _
    ring
  simp only [normalizedMixed, inverseFirst, inverseMixed, varianceFirst, varianceMixed]
  simp only [centered] at hsum ⊢
  rw [hsum]
  ring

/-- Differentiating the actual row differential gives the full mixed chain formula. -/
theorem fderiv_nrmD_comp {ε : ℝ} (hε : 0 < ε)
    (X A : E → Vec (matSize m n)) (p : E)
    (DX DA : E →L[ℝ] Vec (matSize m n))
    (hX : HasFDerivAt X DX p) (hA : HasFDerivAt A DA p)
    (i : Fin m) (j : Fin n) (v : E) :
    fderiv ℝ (fun q => nrmD (X q) ε i j (A q)) p v =
      normalizedMixed (X p) (A p) (DX v) (DA v) ε i j := by
  obtain ⟨D, hD, hformula⟩ := exists_nrmD_comp_derivative hε X A p DX DA hX hA i j
  rw [hD.fderiv]
  exact hformula v

/-- The Hessian of a normalized row entry is the mixed formula with zero upstream term. -/
theorem fderiv_fderiv_nrm {ε : ℝ} (hε : 0 < ε)
    (X A B : Vec (matSize m n)) (i : Fin m) (j : Fin n) :
    fderiv ℝ (fun Y => fderiv ℝ (fun Z => nrm Z ε i j) Y A) X B =
      normalizedMixed X A B 0 ε i j := by
  have hfun : (fun Y => fderiv ℝ (fun Z => nrm Z ε i j) Y A) =
      fun Y => nrmD Y ε i j A := by
    funext Y
    rw [(hasFDerivAt_nrm hε Y i j).fderiv]
  rw [hfun]
  exact fderiv_nrmD_comp hε id (fun _ => A) X (ContinuousLinearMap.id ℝ _)
    0 (hasFDerivAt_id X) (hasFDerivAt_const A X) i j B

/-- The actual affine row output has the same mixed derivative, multiplied by its fixed scale. -/
theorem fderiv_fderiv_affine_nrm_comp {ε : ℝ} (hε : 0 < ε)
    (X : E → Vec (matSize m n)) (p left right : E)
    (hX : ∀ q, DifferentiableAt ℝ X q)
    (DA : E →L[ℝ] Vec (matSize m n))
    (hA : HasFDerivAt (fun q => fderiv ℝ X q left) DA p)
    (i : Fin m) (j : Fin n) (gamma beta : ℝ) :
    fderiv ℝ
        (fun q => fderiv ℝ (fun y => gamma * nrm (X y) ε i j + beta) q left) p right =
      gamma *
        normalizedMixed (X p) (fderiv ℝ X p left) (fderiv ℝ X p right) (DA right) ε i j := by
  obtain ⟨D, hD, hformula⟩ :=
    exists_nrmD_comp_derivative hε X (fun q => fderiv ℝ X q left) p
      (fderiv ℝ X p) DA (hX p).hasFDerivAt hA i j
  have hfun :
      (fun q => fderiv ℝ (fun y => gamma * nrm (X y) ε i j + beta) q left) =
        fun q => gamma * nrmD (X q) ε i j (fderiv ℝ X q left) := by
    funext q
    have hcomp : HasFDerivAt (fun y => gamma * nrm (X y) ε i j + beta)
        (gamma • (nrmD (X q) ε i j).comp (fderiv ℝ X q)) q :=
      (((hasFDerivAt_nrm hε (X q) i j).comp q
        (hX q).hasFDerivAt).const_mul gamma).add_const beta
    exact congrArg (fun D : E →L[ℝ] ℝ => D left) hcomp.fderiv
  rw [hfun, (hD.const_mul gamma).fderiv]
  change gamma * D right = _
  rw [hformula]

/-- Mixed chain rule for an actual composed row, with fixed left direction `u`: the case
`gamma = 1`, `beta = 0` of `fderiv_fderiv_affine_nrm_comp`. -/
theorem fderiv_fderiv_nrm_comp {ε : ℝ} (hε : 0 < ε)
    (X : E → Vec (matSize m n)) (p u v : E)
    (hX : ∀ q, DifferentiableAt ℝ X q)
    (DA : E →L[ℝ] Vec (matSize m n))
    (hA : HasFDerivAt (fun q => fderiv ℝ X q u) DA p) (i : Fin m) (j : Fin n) :
    fderiv ℝ (fun q => fderiv ℝ (fun y => nrm (X y) ε i j) q u) p v =
      normalizedMixed (X p) (fderiv ℝ X p u) (fderiv ℝ X p v) (DA v) ε i j := by
  simpa only [one_mul, add_zero] using fderiv_fderiv_affine_nrm_comp hε X p u v hX DA hA i j 1 0

/-- A bound for `qL = 2 mean(z*a)` from centered magnitudes. -/
def varianceFirstRadius (u a : Fin n → ℝ) : ℝ :=
  2 * (∑ k, u k * a k) / n

/-- A bound for `qLR = 2 mean(a*b + z*c)` from centered magnitudes. -/
def varianceMixedRadius (u a b c : Fin n → ℝ) : ℝ :=
  2 * (∑ k, (a k * b k + u k * c k)) / n

/-- Radius for the inverse-standard-deviation differential, using only epsilon. -/
def inverseFirstRadius (ε : ℝ) (u a : Fin n → ℝ) : ℝ :=
  (1 / 2 : ℝ) * (Real.sqrt ε)⁻¹ ^ 3 * varianceFirstRadius u a

/-- Radius for the mixed inverse-standard-deviation differential. -/
def inverseMixedRadius (ε : ℝ) (u a b c : Fin n → ℝ) : ℝ :=
  (3 / 4 : ℝ) * (Real.sqrt ε)⁻¹ ^ 5 *
      varianceFirstRadius u a * varianceFirstRadius u b +
    (1 / 2 : ℝ) * (Real.sqrt ε)⁻¹ ^ 3 * varianceMixedRadius u a b c

/-- First row radius for bounds on already centered input and direction. -/
def centeredFirstRadius (ε : ℝ) (u a : Fin n → ℝ) (j : Fin n) : ℝ :=
  a j * (Real.sqrt ε)⁻¹ + u j * inverseFirstRadius ε u a

/-- Mixed row radius for centered input, two first directions, and one mixed direction. -/
def centeredMixedRadius (ε : ℝ) (u a b c : Fin n → ℝ) (j : Fin n) : ℝ :=
  c j * (Real.sqrt ε)⁻¹ + a j * inverseFirstRadius ε u b +
    b j * inverseFirstRadius ε u a + u j * inverseMixedRadius ε u a b c

private theorem abs_mul_bound {x y a b : ℝ} (hx : |x| ≤ a) (hy : |y| ≤ b) :
    |x * y| ≤ a * b := by
  rw [abs_mul]
  exact mul_le_mul hx hy (abs_nonneg _) ((abs_nonneg _).trans hx)

private theorem abs_add_bound {x y a b : ℝ} (hx : |x| ≤ a) (hy : |y| ≤ b) :
    |x + y| ≤ a + b :=
  (abs_add_le _ _).trans (add_le_add hx hy)

theorem abs_varianceFirst_le (X A : Vec (matSize m n)) (i : Fin m)
    (u a : Fin n → ℝ) (hu : ∀ k, |centered X i k| ≤ u k)
    (ha : ∀ k, |centered A i k| ≤ a k) :
    |varianceFirst X A i| ≤ varianceFirstRadius u a := by
  have h := abs_sum_div_le _ _ (fun k => abs_mul_bound (hu k) (ha k))
  simpa only [varianceFirst, varianceFirstRadius, mul_div_assoc, abs_mul,
    abs_of_pos (by norm_num : (0 : ℝ) < 2)] using
    mul_le_mul_of_nonneg_left h (by norm_num : (0 : ℝ) ≤ 2)

theorem abs_varianceMixed_le (X A B C : Vec (matSize m n)) (i : Fin m)
    (u a b c : Fin n → ℝ) (hu : ∀ k, |centered X i k| ≤ u k)
    (ha : ∀ k, |centered A i k| ≤ a k) (hb : ∀ k, |centered B i k| ≤ b k)
    (hc : ∀ k, |centered C i k| ≤ c k) :
    |varianceMixed X A B C i| ≤ varianceMixedRadius u a b c := by
  have h := abs_sum_div_le _ _ fun k =>
    abs_add_bound (abs_mul_bound (ha k) (hb k)) (abs_mul_bound (hu k) (hc k))
  simpa only [varianceMixed, varianceMixedRadius, mul_div_assoc, abs_mul,
    abs_of_pos (by norm_num : (0 : ℝ) < 2)] using
    mul_le_mul_of_nonneg_left h (by norm_num : (0 : ℝ) ≤ 2)

theorem abs_inverseFirst_le {ε : ℝ} (hε : 0 < ε)
    (X A : Vec (matSize m n)) (i : Fin m)
    (u a : Fin n → ℝ) (hu : ∀ k, |centered X i k| ≤ u k)
    (ha : ∀ k, |centered A i k| ≤ a k) :
    |inverseFirst X A ε i| ≤ inverseFirstRadius ε u a := by
  have hr : |invStd X ε i ^ 3| ≤ (Real.sqrt ε)⁻¹ ^ 3 := by
    rw [abs_pow, abs_of_nonneg (invStd_nonneg X ε i)]
    exact pow_le_pow_left₀ (invStd_nonneg X ε i) (invStd_le_inv_sqrt hε X i) 3
  exact abs_mul_bound
    (abs_mul_bound (by norm_num : |-(1 / 2 : ℝ)| ≤ (1 / 2 : ℝ)) hr)
    (abs_varianceFirst_le X A i u a hu ha)

theorem abs_inverseMixed_le {ε : ℝ} (hε : 0 < ε)
    (X A B C : Vec (matSize m n)) (i : Fin m)
    (u a b c : Fin n → ℝ) (hu : ∀ k, |centered X i k| ≤ u k)
    (ha : ∀ k, |centered A i k| ≤ a k) (hb : ∀ k, |centered B i k| ≤ b k)
    (hc : ∀ k, |centered C i k| ≤ c k) :
    |inverseMixed X A B C ε i| ≤ inverseMixedRadius ε u a b c := by
  have hr (power : Nat) : |invStd X ε i ^ power| ≤ (Real.sqrt ε)⁻¹ ^ power := by
    rw [abs_pow, abs_of_nonneg (invStd_nonneg X ε i)]
    exact pow_le_pow_left₀ (invStd_nonneg X ε i) (invStd_le_inv_sqrt hε X i) power
  have hfirst := abs_mul_bound
    (abs_mul_bound
      (abs_mul_bound (by norm_num : |(3 / 4 : ℝ)| ≤ (3 / 4 : ℝ)) (hr 5))
      (abs_varianceFirst_le X A i u a hu ha))
    (abs_varianceFirst_le X B i u b hu hb)
  have hsecond := abs_mul_bound
    (abs_mul_bound (by norm_num : |(1 / 2 : ℝ)| ≤ (1 / 2 : ℝ)) (hr 3))
    (abs_varianceMixed_le X A B C i u a b c hu ha hb hc)
  simpa only [inverseMixed, inverseMixedRadius, sub_eq_add_neg, abs_neg] using
    abs_add_bound hfirst (show |-(1 / 2 * invStd X ε i ^ 3 *
      varianceMixed X A B C i)| ≤
        1 / 2 * (Real.sqrt ε)⁻¹ ^ 3 * varianceMixedRadius u a b c by
      simpa only [abs_neg] using hsecond)

/-- The first differential is bounded from centered data, without a derivative-bound premise. -/
theorem abs_nrmD_le_centered {ε : ℝ} (hε : 0 < ε)
    (X A : Vec (matSize m n)) (i : Fin m) (j : Fin n)
    (u a : Fin n → ℝ) (hu : ∀ k, |centered X i k| ≤ u k)
    (ha : ∀ k, |centered A i k| ≤ a k) :
    |nrmD X ε i j A| ≤ centeredFirstRadius ε u a j := by
  rw [nrmD_apply_centered hε]
  exact abs_add_bound
    (abs_mul_bound (ha j) (by
      rw [abs_of_nonneg (invStd_nonneg X ε i)]
      exact invStd_le_inv_sqrt hε X i))
    (abs_mul_bound (hu j) (abs_inverseFirst_le hε X A i u a hu ha))

/-- Centered magnitudes enclose all four terms of the analytic mixed chain formula. -/
theorem abs_normalizedMixed_le {ε : ℝ} (hε : 0 < ε)
    (X A B C : Vec (matSize m n)) (i : Fin m) (j : Fin n)
    (u a b c : Fin n → ℝ) (hu : ∀ k, |centered X i k| ≤ u k)
    (ha : ∀ k, |centered A i k| ≤ a k) (hb : ∀ k, |centered B i k| ≤ b k)
    (hc : ∀ k, |centered C i k| ≤ c k) :
    |normalizedMixed X A B C ε i j| ≤ centeredMixedRadius ε u a b c j := by
  exact abs_add_bound
    (abs_add_bound
      (abs_add_bound
        (abs_mul_bound (hc j) (by
          rw [abs_of_nonneg (invStd_nonneg X ε i)]
          exact invStd_le_inv_sqrt hε X i))
        (abs_mul_bound (ha j) (abs_inverseFirst_le hε X B i u b hu hb)))
      (abs_mul_bound (hb j) (abs_inverseFirst_le hε X A i u a hu ha)))
    (abs_mul_bound (hu j) (abs_inverseMixed_le hε X A B C i u a b c hu ha hb hc))

/-- Enclosure of the actual mixed derivative of a composed row with fixed affine scale. -/
theorem fderiv_fderiv_affine_nrm_comp_mem_Icc {ε : ℝ} (hε : 0 < ε)
    (X : E → Vec (matSize m n)) (p left right : E)
    (hX : ∀ q, DifferentiableAt ℝ X q)
    (DA : E →L[ℝ] Vec (matSize m n))
    (hA : HasFDerivAt (fun q => fderiv ℝ X q left) DA p)
    (i : Fin m) (j : Fin n) (gamma beta : ℝ) (u a b c : Fin n → ℝ)
    (hu : ∀ k, |centered (X p) i k| ≤ u k)
    (ha : ∀ k, |centered (fderiv ℝ X p left) i k| ≤ a k)
    (hb : ∀ k, |centered (fderiv ℝ X p right) i k| ≤ b k)
    (hc : ∀ k, |centered (DA right) i k| ≤ c k) :
    fderiv ℝ
        (fun q => fderiv ℝ (fun y => gamma * nrm (X y) ε i j + beta) q left) p right ∈
      Set.Icc (-(|gamma| * centeredMixedRadius ε u a b c j))
        (|gamma| * centeredMixedRadius ε u a b c j) := by
  rw [fderiv_fderiv_affine_nrm_comp hε X p left right hX DA hA]
  exact abs_le.mp (abs_mul_bound (le_refl |gamma|)
    (abs_normalizedMixed_le hε _ _ _ _ i j u a b c hu ha hb hc))

end

end Proofs.Autograd.RowNorm
