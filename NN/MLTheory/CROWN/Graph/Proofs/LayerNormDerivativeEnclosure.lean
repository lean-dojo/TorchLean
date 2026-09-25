/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.LayerNormDerivatives
public import NN.MLTheory.CROWN.Proofs.LayerNormEnclosure
public import NN.Proofs.Autograd.Tape.Ops.Norm.LayerNormSecondBounds

/-!
# Enclosure by the directed LayerNorm derivative sequence

Successful row evaluation bounds the real first and mixed differentials. The proof follows
the executable centering, mean, reciprocal, and directed arithmetic operations, then applies
the analytic row derivative bounds. Stored affine parameters and epsilon remain fixed.
-/

public section

namespace NN.MLTheory.CROWN.Graph

open Spec TorchLean TorchLean.Tensor
open NN.MLTheory.CROWN BoundOps LayerNormDirected
open _root_.Proofs.Autograd _root_.Proofs.Autograd.RowNorm
open _root_.Proofs.Autograd.TapeNodes.Matmul
open scoped BigOperators

noncomputable section

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]
variable [NonlinearBoundOps α] [LawfulNonlinearBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

omit [NonlinearBoundOps α] [LawfulNonlinearBoundOps α] in
/-- The executable endpoint magnitude bounds every real value between its endpoints. -/
theorem abs_le_derivativeMagnitude (hzero : value (0 : α) = 0)
    {bounds : α × α} {x : ℝ}
    (hx : value bounds.1 ≤ x ∧ x ≤ value bounds.2) :
    |x| ≤ value (derivativeMagnitude bounds) := by
  rw [derivativeMagnitude, value_max2]
  apply abs_le.mpr
  refine ⟨?_, hx.2.trans (le_max_right _ _)⟩
  have hneg : -x ≤ value (subUp 0 bounds.1) := by
    simpa only [hzero, zero_sub] using
      (sub_le_sub_left hx.1 (value (0 : α))).trans (LawfulBoundOps.le_subUp _ _)
  exact neg_le.mp (hneg.trans (le_max_left _ _))

/-- Successful centering bounds the real centered row, including the directed mean error. -/
theorem centeredDerivativeRadii?_encloses
    (hzero : value (0 : α) = 0) (hone : value (1 : α) = 1)
    {n : Nat} (hn : 0 < n) (bounds : Fin n → α × α) (f : Fin n → ℝ)
    (hb : ∀ j, value (bounds j).1 ≤ f j ∧ f j ≤ value (bounds j).2)
    {radii : Fin n → α} (hout : centeredDerivativeRadii? bounds = some radii) :
    ∀ j, |f j - (∑ k, f k) / n| ≤ value (radii j) := by
  unfold centeredDerivativeRadii? at hout
  obtain ⟨_, _, hout⟩ := Option.bind_eq_some_iff.mp hout
  obtain ⟨⟨meanLo, meanHi⟩, hmean, hout⟩ := Option.bind_eq_some_iff.mp hout
  have hm := directedRowMean?_encloses hzero hone hn bounds f hb hmean
  have hpoint := Internal.traverseFin_eq_some_iff.mp hout
  intro j
  obtain ⟨centered, hcentered, hj⟩ := Option.bind_eq_some_iff.mp (hpoint j)
  have heq := checkedFiniteBounds?_eq_of_eq_some hcentered
  subst centered
  have hjEq := Option.some.inj hj
  rw [← hjEq]
  exact abs_le_derivativeMagnitude hzero (sub_encloses (hb j) hm)

/-- The successful radius mean bounds the exact mean of any enclosed nonnegative row. -/
theorem meanDerivativeRadius?_encloses
    (hzero : value (0 : α) = 0) (hone : value (1 : α) = 1)
    {n : Nat} (hn : 0 < n) (radii : Fin n → α) (f : Fin n → ℝ)
    (hf : ∀ j, 0 ≤ f j ∧ f j ≤ value (radii j))
    {result : α} (hout : meanDerivativeRadius? radii = some result) :
    0 ≤ (∑ j, f j) / n ∧ (∑ j, f j) / n ≤ value result := by
  unfold meanDerivativeRadius? at hout
  obtain ⟨⟨lo, hi⟩, hmean, hout⟩ := Option.bind_eq_some_iff.mp hout
  have heq := Option.some.inj hout
  subst result
  refine ⟨div_nonneg (Finset.sum_nonneg fun j _ => (hf j).1) (Nat.cast_nonneg n), ?_⟩
  exact (directedRowMean?_encloses hzero hone hn _ f
    (fun j => by simpa only [hzero] using hf j) hmean).2

omit [NonlinearBoundOps α] [LawfulNonlinearBoundOps α] in
private theorem nonneg_mulUp {a b : α} {x y : ℝ}
    (hx : 0 ≤ x ∧ x ≤ value a) (hy : 0 ≤ y ∧ y ≤ value b) :
    0 ≤ x * y ∧ x * y ≤ value (mulUp a b) :=
  ⟨mul_nonneg hx.1 hy.1,
    (mul_le_mul hx.2 hy.2 hy.1 (hx.1.trans hx.2)).trans (LawfulBoundOps.le_mulUp _ _)⟩

omit [NonlinearBoundOps α] [LawfulNonlinearBoundOps α] in
private theorem nonneg_addUp {a b : α} {x y : ℝ}
    (hx : 0 ≤ x ∧ x ≤ value a) (hy : 0 ≤ y ∧ y ≤ value b) :
    0 ≤ x + y ∧ x + y ≤ value (addUp a b) :=
  ⟨add_nonneg hx.1 hy.1, (add_le_add hx.2 hy.2).trans (LawfulBoundOps.le_addUp _ _)⟩

omit [NonlinearBoundOps α] [LawfulNonlinearBoundOps α] in
private theorem symmetric_encloses (hzero : value (0 : α) = 0)
    {radius : α} {x : ℝ} (hx : |x| ≤ value radius) :
    value (subDown 0 radius) ≤ x ∧ x ≤ value radius := by
  refine ⟨?_, (abs_le.mp hx).2⟩
  have hlo := LawfulBoundOps.subDown_le (0 : α) radius
  rw [hzero, zero_sub] at hlo
  exact hlo.trans (abs_le.mp hx).1

omit [LawfulNonlinearBoundOps α] in
/-- A successful derivative row has strictly positive interpreted epsilon. -/
theorem layerNormDerivativeRow?_epsilon_pos
    (hzero : value (0 : α) = 0) {n : Nat}
    (input left right mixed : Fin n → α × α) (gamma : Tensor α [n]) (epsilon : α)
    {result : (Tensor α [n] × Tensor α [n]) × (Tensor α [n] × Tensor α [n])}
    (hout : layerNormDerivativeRow? input left right mixed gamma epsilon = some result) :
    0 < value epsilon := by
  unfold layerNormDerivativeRow? at hout
  obtain ⟨_, _, hout⟩ := Option.bind_eq_some_iff.mp hout
  split at hout
  · contradiction
  · rename_i h
    have hpos : (0 : α) < epsilon := by simpa using h
    simpa only [hzero] using (LawfulBoundOps.lt_iff 0 epsilon).mp hpos

/--
The actual directed row sequence encloses the analytic first and mixed differentials.

Only scalar operation laws, literal interpretations, the four upstream row enclosures,
and a successful return are assumed. Every intermediate radius is bounded in this proof.
-/
theorem layerNormDerivativeRow?_encloses
    (hzero : value (0 : α) = 0) (hone : value (1 : α) = 1)
    (htwo : value (2 : α) = 2) (hthree : value (3 : α) = 3)
    (hfour : value (4 : α) = 4) {m n : Nat} (hn : 0 < n)
    (input left right mixed : Fin n → α × α) (gamma : Tensor α [n]) (epsilon : α)
    (X A B C : Vec (matSize m n)) (i : Fin m)
    (hx : ∀ j, value (input j).1 ≤ X (idxMN i j) ∧ X (idxMN i j) ≤ value (input j).2)
    (ha : ∀ j, value (left j).1 ≤ A (idxMN i j) ∧ A (idxMN i j) ≤ value (left j).2)
    (hb : ∀ j, value (right j).1 ≤ B (idxMN i j) ∧ B (idxMN i j) ≤ value (right j).2)
    (hc : ∀ j, value (mixed j).1 ≤ C (idxMN i j) ∧ C (idxMN i j) ≤ value (mixed j).2)
    {firstLo firstHi mixedLo mixedHi : Tensor α [n]}
    (hout : layerNormDerivativeRow? input left right mixed gamma epsilon =
      some ((firstLo, firstHi), (mixedLo, mixedHi))) :
    ∀ j,
      (value (firstLo.getScalar j) ≤
          value (gamma.getScalar j) * nrmD X (value epsilon) i j A ∧
        value (gamma.getScalar j) * nrmD X (value epsilon) i j A ≤
          value (firstHi.getScalar j)) ∧
      (value (mixedLo.getScalar j) ≤
          value (gamma.getScalar j) * normalizedMixed X A B C (value epsilon) i j ∧
        value (gamma.getScalar j) * normalizedMixed X A B C (value epsilon) i j ≤
          value (mixedHi.getScalar j)) := by
  have hε := layerNormDerivativeRow?_epsilon_pos hzero input left right mixed gamma epsilon hout
  unfold layerNormDerivativeRow? at hout
  obtain ⟨_, _, hout⟩ := Option.bind_eq_some_iff.mp hout
  split at hout
  · contradiction
  · obtain ⟨⟨rootLo, rootHi⟩, hroot, hout⟩ := Option.bind_eq_some_iff.mp hout
    obtain ⟨_, _, hout⟩ := Option.bind_eq_some_iff.mp hout
    split at hout
    · contradiction
    · obtain ⟨⟨reciprocalLo, t⟩, hreciprocal, hout⟩ := Option.bind_eq_some_iff.mp hout
      obtain ⟨_, _, hout⟩ := Option.bind_eq_some_iff.mp hout
      have hrootBounds := LawfulNonlinearBoundOps.sqrtBounds_enclosure hroot le_rfl le_rfl
      have htBounds := LawfulNonlinearBoundOps.divBounds_enclosure hreciprocal
        (le_of_eq hone) (le_of_eq hone.symm) hrootBounds.1 hrootBounds.2
      have ht : 0 ≤ (Real.sqrt (value epsilon))⁻¹ ∧
          (Real.sqrt (value epsilon))⁻¹ ≤ value t :=
        ⟨inv_nonneg.mpr (Real.sqrt_nonneg _), by simpa only [one_div] using htBounds.2⟩
      have ht3 := nonneg_mulUp (nonneg_mulUp ht ht) ht
      have ht5 := nonneg_mulUp (nonneg_mulUp ht3 ht) ht
      simp only [← pow_succ, ← pow_two] at ht3 ht5
      obtain ⟨u, huResult, hout⟩ := Option.bind_eq_some_iff.mp hout
      obtain ⟨a, haResult, hout⟩ := Option.bind_eq_some_iff.mp hout
      obtain ⟨b, hbResult, hout⟩ := Option.bind_eq_some_iff.mp hout
      obtain ⟨c, hcResult, hout⟩ := Option.bind_eq_some_iff.mp hout
      have hu := centeredDerivativeRadii?_encloses hzero hone hn input _ hx huResult
      have hca := centeredDerivativeRadii?_encloses hzero hone hn left _ ha haResult
      have hcb := centeredDerivativeRadii?_encloses hzero hone hn right _ hb hbResult
      have hcc := centeredDerivativeRadii?_encloses hzero hone hn mixed _ hc hcResult
      change ∀ j, |centered X i j| ≤ value (u j) at hu
      change ∀ j, |centered A i j| ≤ value (a j) at hca
      change ∀ j, |centered B i j| ≤ value (b j) at hcb
      change ∀ j, |centered C i j| ≤ value (c j) at hcc
      have hu0 j : 0 ≤ value (u j) ∧ value (u j) ≤ value (u j) :=
        ⟨(abs_nonneg _).trans (hu j), le_rfl⟩
      have ha0 j : 0 ≤ value (a j) ∧ value (a j) ≤ value (a j) :=
        ⟨(abs_nonneg _).trans (hca j), le_rfl⟩
      have hb0 j : 0 ≤ value (b j) ∧ value (b j) ≤ value (b j) :=
        ⟨(abs_nonneg _).trans (hcb j), le_rfl⟩
      have hc0 j : 0 ≤ value (c j) ∧ value (c j) ≤ value (c j) :=
        ⟨(abs_nonneg _).trans (hcc j), le_rfl⟩
      have htwoBounds : 0 ≤ (2 : ℝ) ∧ 2 ≤ value (2 : α) := by
        rw [htwo]
        norm_num
      obtain ⟨qLeft, hqLeft, hout⟩ := Option.bind_eq_some_iff.mp hout
      obtain ⟨qRight, hqRight, hout⟩ := Option.bind_eq_some_iff.mp hout
      obtain ⟨qMixed, hqMixed, hout⟩ := Option.bind_eq_some_iff.mp hout
      have hqL := meanDerivativeRadius?_encloses hzero hone hn _ _
        (fun j => nonneg_mulUp htwoBounds (nonneg_mulUp (hu0 j) (ha0 j))) hqLeft
      have hqR := meanDerivativeRadius?_encloses hzero hone hn _ _
        (fun j => nonneg_mulUp htwoBounds (nonneg_mulUp (hu0 j) (hb0 j))) hqRight
      have hqM := meanDerivativeRadius?_encloses hzero hone hn _ _
        (fun j => nonneg_mulUp htwoBounds
          (nonneg_addUp (nonneg_mulUp (ha0 j) (hb0 j))
            (nonneg_mulUp (hu0 j) (hc0 j)))) hqMixed
      rw [← Finset.mul_sum] at hqL hqR hqM
      change 0 ≤ varianceFirstRadius (fun j => value (u j)) (fun j => value (a j)) ∧
        varianceFirstRadius (fun j => value (u j)) (fun j => value (a j)) ≤ value qLeft at hqL
      change 0 ≤ varianceFirstRadius (fun j => value (u j)) (fun j => value (b j)) ∧
        varianceFirstRadius (fun j => value (u j)) (fun j => value (b j)) ≤ value qRight at hqR
      change 0 ≤ varianceMixedRadius (fun j => value (u j)) (fun j => value (a j))
          (fun j => value (b j)) (fun j => value (c j)) ∧
        varianceMixedRadius (fun j => value (u j)) (fun j => value (a j))
          (fun j => value (b j)) (fun j => value (c j)) ≤ value qMixed at hqM
      obtain ⟨⟨halfLo, halfHi⟩, hhalf, hout⟩ := Option.bind_eq_some_iff.mp hout
      obtain ⟨⟨threeQuartersLo, threeQuartersHi⟩, hthreeQuarters, hout⟩ :=
        Option.bind_eq_some_iff.mp hout
      have hhalfBounds := LawfulNonlinearBoundOps.divBounds_enclosure hhalf
        (le_of_eq hone) (le_of_eq hone.symm) (le_of_eq htwo) (le_of_eq htwo.symm)
      have hthreeQuartersBounds := LawfulNonlinearBoundOps.divBounds_enclosure hthreeQuarters
        (le_of_eq hthree) (le_of_eq hthree.symm) (le_of_eq hfour) (le_of_eq hfour.symm)
      have hhalfUpper : 0 ≤ (1 / 2 : ℝ) ∧ (1 / 2 : ℝ) ≤ value halfHi :=
        ⟨by norm_num, hhalfBounds.2⟩
      have hthreeQuartersUpper : 0 ≤ (3 / 4 : ℝ) ∧ (3 / 4 : ℝ) ≤ value threeQuartersHi :=
        ⟨by norm_num, hthreeQuartersBounds.2⟩
      have hrL := nonneg_mulUp (nonneg_mulUp hhalfUpper ht3) hqL
      have hrR := nonneg_mulUp (nonneg_mulUp hhalfUpper ht3) hqR
      have hrM := nonneg_addUp
        (nonneg_mulUp (nonneg_mulUp (nonneg_mulUp hthreeQuartersUpper ht5) hqL) hqR)
        (nonneg_mulUp (nonneg_mulUp hhalfUpper ht3) hqM)
      obtain ⟨radii, hradii, hout⟩ := Option.bind_eq_some_iff.mp hout
      have hpoint := Internal.traverseFin_eq_some_iff.mp hradii
      have houtEq := Option.some.inj hout
      cases houtEq
      intro j
      simp only [Tensor.getScalar_ofFn]
      have hj := hpoint j
      obtain ⟨scale, hscale, hj⟩ := Option.bind_eq_some_iff.mp hj
      have hscaleEq := checkedFiniteBounds?_eq_of_eq_some hscale
      subst scale
      have hs := abs_le_derivativeMagnitude hzero
        (bounds := (gamma.getScalar j, gamma.getScalar j))
        (show value (gamma.getScalar j) ≤ value (gamma.getScalar j) ∧
          value (gamma.getScalar j) ≤ value (gamma.getScalar j) from ⟨le_rfl, le_rfl⟩)
      have hs0 := And.intro (abs_nonneg (value (gamma.getScalar j))) hs
      have hfirst := nonneg_mulUp hs0
        (nonneg_addUp (nonneg_mulUp (ha0 j) ht) (nonneg_mulUp (hu0 j) hrL))
      have hsecond := nonneg_mulUp hs0
        (nonneg_addUp
          (nonneg_addUp (nonneg_mulUp (hc0 j) ht) (nonneg_mulUp (ha0 j) hrR))
          (nonneg_addUp (nonneg_mulUp (hb0 j) hrL) (nonneg_mulUp (hu0 j) hrM)))
      have hfirstAnalytic := mul_le_mul_of_nonneg_left
        (abs_nrmD_le_centered hε X A i j _ _ hu hca) (abs_nonneg (value (gamma.getScalar j)))
      have hsecondAnalytic := mul_le_mul_of_nonneg_left
        (abs_normalizedMixed_le hε X A B C i j _ _ _ _ hu hca hcb hcc)
        (abs_nonneg (value (gamma.getScalar j)))
      have hfirstAbs := hfirstAnalytic.trans hfirst.2
      have hsecondUpper := hsecond.2
      simp only [← add_assoc] at hsecondUpper
      have hsecondAbs := hsecondAnalytic.trans hsecondUpper
      rw [← abs_mul] at hfirstAbs hsecondAbs
      obtain ⟨firstBounds, hfirstBounds, hj⟩ := Option.bind_eq_some_iff.mp hj
      obtain ⟨secondBounds, hsecondBounds, hj⟩ := Option.bind_eq_some_iff.mp hj
      have hfirstEq := checkedFiniteBounds?_eq_of_eq_some hfirstBounds
      have hsecondEq := checkedFiniteBounds?_eq_of_eq_some hsecondBounds
      subst firstBounds
      subst secondBounds
      have hjEq := Option.some.inj hj
      rw [← hjEq]
      exact ⟨symmetric_encloses hzero hfirstAbs, symmetric_encloses hzero hsecondAbs⟩

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/--
Successful derivative evaluation contains the actual derivatives of the composed affine row.

The input intervals enclose the upstream value and its actual first and mixed derivatives.
Differentiability of the left differential supplies the mixed chain rule. No premise assumes
a bound on either output derivative or on any computed intermediate radius.
-/
theorem layerNormDerivativeRow?_encloses_fderiv
    (hzero : value (0 : α) = 0) (hone : value (1 : α) = 1)
    (htwo : value (2 : α) = 2) (hthree : value (3 : α) = 3)
    (hfour : value (4 : α) = 4) {m n : Nat} (hn : 0 < n)
    (input left right mixed : Fin n → α × α) (gamma : Tensor α [n]) (epsilon : α)
    (X : E → Vec (matSize m n)) (p v w : E)
    (hX : ∀ q, DifferentiableAt ℝ X q)
    (DA : E →L[ℝ] Vec (matSize m n))
    (hA : HasFDerivAt (fun q => fderiv ℝ X q v) DA p)
    (i : Fin m) (beta : Fin n → ℝ)
    (hx : ∀ j, value (input j).1 ≤ X p (idxMN i j) ∧
      X p (idxMN i j) ≤ value (input j).2)
    (ha : ∀ j, value (left j).1 ≤ fderiv ℝ X p v (idxMN i j) ∧
      fderiv ℝ X p v (idxMN i j) ≤ value (left j).2)
    (hb : ∀ j, value (right j).1 ≤ fderiv ℝ X p w (idxMN i j) ∧
      fderiv ℝ X p w (idxMN i j) ≤ value (right j).2)
    (hc : ∀ j, value (mixed j).1 ≤ DA w (idxMN i j) ∧
      DA w (idxMN i j) ≤ value (mixed j).2)
    {firstLo firstHi mixedLo mixedHi : Tensor α [n]}
    (hout : layerNormDerivativeRow? input left right mixed gamma epsilon =
      some ((firstLo, firstHi), (mixedLo, mixedHi))) :
    ∀ j, let y := fun q => value (gamma.getScalar j) * nrm (X q) (value epsilon) i j + beta j
      (value (firstLo.getScalar j) ≤ fderiv ℝ y p v ∧
        fderiv ℝ y p v ≤ value (firstHi.getScalar j)) ∧
      (value (mixedLo.getScalar j) ≤ fderiv ℝ (fun q => fderiv ℝ y q v) p w ∧
        fderiv ℝ (fun q => fderiv ℝ y q v) p w ≤ value (mixedHi.getScalar j)) := by
  have hε := layerNormDerivativeRow?_epsilon_pos hzero input left right mixed gamma epsilon hout
  have hrow := layerNormDerivativeRow?_encloses hzero hone htwo hthree hfour hn
    input left right mixed gamma epsilon (X p) (fderiv ℝ X p v) (fderiv ℝ X p w)
    (DA w) i hx ha hb hc hout
  intro j
  dsimp only
  have hfirst :
      fderiv ℝ (fun q => value (gamma.getScalar j) * nrm (X q) (value epsilon) i j + beta j) p v =
        value (gamma.getScalar j) * nrmD (X p) (value epsilon) i j (fderiv ℝ X p v) := by
    have hd : HasFDerivAt
        (fun q => value (gamma.getScalar j) * nrm (X q) (value epsilon) i j + beta j)
        (value (gamma.getScalar j) •
          (nrmD (X p) (value epsilon) i j).comp (fderiv ℝ X p)) p :=
      (((hasFDerivAt_nrm hε (X p) i j).comp p (hX p).hasFDerivAt).const_mul
        (value (gamma.getScalar j))).add_const (beta j)
    exact congrArg (fun D : E →L[ℝ] ℝ => D v) hd.fderiv
  rw [hfirst, fderiv_fderiv_affine_nrm_comp hε X p v w hX DA hA]
  exact hrow j

/-- Exact real endpoints need no additional scalar or literal interpretation assumptions. -/
theorem layerNormDerivativeRow?_encloses_fderiv_real
    {m n : Nat} (hn : 0 < n)
    (input left right mixed : Fin n → ℝ × ℝ) (gamma : Tensor ℝ [n]) (epsilon : ℝ)
    (X : E → Vec (matSize m n)) (p v w : E)
    (hX : ∀ q, DifferentiableAt ℝ X q)
    (DA : E →L[ℝ] Vec (matSize m n))
    (hA : HasFDerivAt (fun q => fderiv ℝ X q v) DA p)
    (i : Fin m) (beta : Fin n → ℝ)
    (hx : ∀ j, (input j).1 ≤ X p (idxMN i j) ∧ X p (idxMN i j) ≤ (input j).2)
    (ha : ∀ j, (left j).1 ≤ fderiv ℝ X p v (idxMN i j) ∧
      fderiv ℝ X p v (idxMN i j) ≤ (left j).2)
    (hb : ∀ j, (right j).1 ≤ fderiv ℝ X p w (idxMN i j) ∧
      fderiv ℝ X p w (idxMN i j) ≤ (right j).2)
    (hc : ∀ j, (mixed j).1 ≤ DA w (idxMN i j) ∧ DA w (idxMN i j) ≤ (mixed j).2)
    {firstLo firstHi mixedLo mixedHi : Tensor ℝ [n]}
    (hout : layerNormDerivativeRow? input left right mixed gamma epsilon =
      some ((firstLo, firstHi), (mixedLo, mixedHi))) :
    ∀ j, let y := fun q => gamma.getScalar j * nrm (X q) epsilon i j + beta j
      (firstLo.getScalar j ≤ fderiv ℝ y p v ∧ fderiv ℝ y p v ≤ firstHi.getScalar j) ∧
      (mixedLo.getScalar j ≤ fderiv ℝ (fun q => fderiv ℝ y q v) p w ∧
        fderiv ℝ (fun q => fderiv ℝ y q v) p w ≤ mixedHi.getScalar j) :=
  layerNormDerivativeRow?_encloses_fderiv (α := ℝ) rfl rfl rfl rfl rfl hn
    input left right mixed gamma epsilon X p v w hX DA hA i beta hx ha hb hc hout

end

end NN.MLTheory.CROWN.Graph
