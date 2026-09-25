/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.LayerNormDirected
public import NN.MLTheory.CROWN.Extras.IntervalLemmas
public import NN.Proofs.Autograd.Tape.Ops.Norm.MatrixEntries

/-!
# Enclosure by the directed LayerNorm sequence

The successful executable row transfer encloses the corresponding row of `Spec.layerNorm`.
The proof follows its directed sum and count folds, both centering stages, variance, square
root, division, and stored affine parameters. It uses the scalar operation contracts and
the existing entrywise formula for the real specification.
-/

public section

namespace NN.MLTheory.CROWN.Graph.LayerNormDirected

open Spec TorchLean TorchLean.Tensor
open NN.MLTheory.CROWN BoundOps
open NN.MLTheory.CROWN.IntervalLemmas (value_max2)
open _root_.Proofs.Autograd.Norm
open scoped BigOperators

noncomputable section

variable {α : Type} [Context α]

/-- A successful finite-bounds check returns its input endpoints unchanged. -/
theorem checkedFiniteBounds?_eq_of_eq_some {input output : α × α}
    (h : checkedFiniteBounds? input = some output) : input = output := by
  unfold checkedFiniteBounds? at h
  split at h
  · exact Option.some.inj h
  · contradiction

/-- Validation after a scalar transfer preserves the successful transfer result. -/
theorem checkedFiniteBounds?_bind_eq_some {action : Option (α × α)} {output : α × α}
    (h : (action >>= checkedFiniteBounds?) = some output) : action = some output := by
  obtain ⟨result, hresult, hcheck⟩ := Option.bind_eq_some_iff.mp h
  exact hresult.trans (congrArg some (checkedFiniteBounds?_eq_of_eq_some hcheck))

variable [TorchLean.Storage α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- The executable four-accumulator fold encloses both the sum and the exact count. -/
theorem sum_count_fold_encloses {ι : Type}
    (bounds : ι → α × α) (f : ι → ℝ)
    (hb : ∀ i, value (bounds i).1 ≤ f i ∧ f i ≤ value (bounds i).2)
    (indices : List ι) (acc : α × α × α × α) (sum count : ℝ)
    (hacc : value acc.1 ≤ sum ∧ sum ≤ value acc.2.1 ∧
      value acc.2.2.1 ≤ count ∧ count ≤ value acc.2.2.2) :
    let result := indices.foldl (fun (lo, hi, countLo, countHi) i =>
      (addDown lo (bounds i).1, addUp hi (bounds i).2,
       addDown countLo 1, addUp countHi 1)) acc
    value result.1 ≤ indices.foldl (fun s i => s + f i) sum ∧
      indices.foldl (fun s i => s + f i) sum ≤ value result.2.1 ∧
      value result.2.2.1 ≤ indices.foldl (fun c _ => c + 1) count ∧
      indices.foldl (fun c _ => c + 1) count ≤ value result.2.2.2 := by
  induction indices generalizing acc sum count with
  | nil => exact hacc
  | cons i indices ih =>
    apply ih
    refine ⟨(LawfulBoundOps.addDown_le _ _).trans (add_le_add hacc.1 (hb i).1),
      (add_le_add hacc.2.1 (hb i).2).trans (LawfulBoundOps.le_addUp _ _), ?_, ?_⟩
    · calc
        value (addDown acc.2.2.1 1) ≤ value acc.2.2.1 + value (1 : α) :=
          LawfulBoundOps.addDown_le _ _
        _ = value acc.2.2.1 + 1 := by rw [(LawfulBoundOps.toReal_one (α := α))]
        _ ≤ count + 1 := add_le_add hacc.2.2.1 le_rfl
    · calc
        count + 1 ≤ value acc.2.2.2 + 1 := add_le_add hacc.2.2.2 le_rfl
        _ = value acc.2.2.2 + value (1 : α) := by rw [(LawfulBoundOps.toReal_one (α := α))]
        _ ≤ value (addUp acc.2.2.2 1) := LawfulBoundOps.le_addUp _ _

/-- The denominator is the exact natural count, enclosed by the directed fold of ones. -/
theorem directedRowMean?_encloses [NonlinearBoundOps α] [LawfulNonlinearBoundOps α]
    {n : Nat}
    (hn : 0 < n) (bounds : Fin n → α × α) (f : Fin n → ℝ)
    (hb : ∀ i, value (bounds i).1 ≤ f i ∧ f i ≤ value (bounds i).2)
    {outLo outHi : α} (hout : directedRowMean? bounds = some (outLo, outHi)) :
    value outLo ≤ (∑ i, f i) / n ∧ (∑ i, f i) / n ≤ value outHi := by
  have hfold := sum_count_fold_encloses bounds f hb (List.finRange n)
    (0, 0, 0, 0) 0 0 (by simp [(LawfulBoundOps.toReal_zero (α := α))])
  simp only [List.finRange_foldl_add_eq_finset_sum, Finset.sum_const,
    Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one] at hfold
  simp only [directedRowMean?, hn.ne', ↓reduceIte] at hout
  obtain ⟨_, _, hout⟩ := Option.bind_eq_some_iff.mp hout
  obtain ⟨_, _, hout⟩ := Option.bind_eq_some_iff.mp hout
  exact LawfulNonlinearBoundOps.divBounds_enclosure
    (checkedFiniteBounds?_bind_eq_some hout)
    hfold.1 hfold.2.1 hfold.2.2.1 hfold.2.2.2

/-- Subtracting two enclosing intervals uses opposite endpoints for the subtrahend. -/
theorem sub_encloses {aLo aHi bLo bHi : α} {a b : ℝ}
    (ha : value aLo ≤ a ∧ a ≤ value aHi)
    (hb : value bLo ≤ b ∧ b ≤ value bHi) :
    value (subDown aLo bHi) ≤ a - b ∧ a - b ≤ value (subUp aHi bLo) :=
  ⟨(LawfulBoundOps.subDown_le _ _).trans (sub_le_sub ha.1 hb.2),
    (sub_le_sub ha.2 hb.1).trans (LawfulBoundOps.le_subUp _ _)⟩

/-- Clamping both endpoints at zero encloses the clamped real value. -/
theorem max_zero_encloses
    {lo hi : α} {x : ℝ} (h : value lo ≤ x ∧ x ≤ value hi) :
    value (max2 lo 0) ≤ max x 0 ∧ max x 0 ≤ value (max2 hi 0) := by
  simpa only [value_max2, (LawfulBoundOps.toReal_zero (α := α))] using
    And.intro (max_le_max h.1 le_rfl) (max_le_max h.2 le_rfl)

/--
Every successful directed row transfer encloses its row of the real LayerNorm specification.

The literal hypotheses supply the interpretation of zero and one, which the scalar operation
contracts do not otherwise specify. Gamma, beta, and epsilon are interpreted stored values.
The statement assumes only the input enclosure and the existing scalar operation laws.
-/
theorem directedLayerNormRow?_encloses [NonlinearBoundOps α]
    [LawfulNonlinearBoundOps α]
    {m n : Nat} (hm : 0 < m) (hn : 0 < n)
    (lo hi gamma beta : Tensor α [n]) (epsilon : α)
    (x : Tensor ℝ [m, n]) (row : Fin m)
    (hx : ∀ j, value (lo.getScalar j) ≤ Spec.get2 x row j ∧
      Spec.get2 x row j ≤ value (hi.getScalar j))
    {outLo outHi : Tensor α [n]}
    (hout : directedLayerNormRow? lo hi gamma beta epsilon = some (outLo, outHi)) :
    ∀ j,
      let y := Spec.get2 (Spec.layerNorm x
        (Tensor.ofFn fun k => value (gamma.getScalar k))
        (Tensor.ofFn fun k => value (beta.getScalar k)) hm hn (value epsilon)) row j
      value (outLo.getScalar j) ≤ y ∧ y ≤ value (outHi.getScalar j) := by
  unfold directedLayerNormRow? at hout
  obtain ⟨_, _, hout⟩ := Option.bind_eq_some_iff.mp hout
  split at hout
  · contradiction
  · obtain ⟨_, _, hout⟩ := Option.bind_eq_some_iff.mp hout
    obtain ⟨⟨meanLo, meanHi⟩, hmean, hout⟩ := Option.bind_eq_some_iff.mp hout
    have hmeanBounds := directedRowMean?_encloses hn _ _ hx hmean
    change value meanLo ≤ rowMeanE x row ∧ rowMeanE x row ≤ value meanHi at hmeanBounds
    dsimp only at hout
    set centeredLo := Tensor.ofFn (fun j => subDown (lo.getScalar j) meanHi)
    set centeredHi := Tensor.ofFn (fun j => subUp (hi.getScalar j) meanLo)
    have hcentered (j : Fin n) :
        value (centeredLo.getScalar j) ≤ Spec.get2 x row j - rowMeanE x row ∧
          Spec.get2 x row j - rowMeanE x row ≤ value (centeredHi.getScalar j) := by
      simpa only [centeredLo, centeredHi, Tensor.getScalar_ofFn] using
        sub_encloses (hx j) hmeanBounds
    obtain ⟨_, _, hout⟩ := Option.bind_eq_some_iff.mp hout
    obtain ⟨⟨centerMeanLo, centerMeanHi⟩, hcenterMean, hout⟩ :=
      Option.bind_eq_some_iff.mp hout
    have hcenterMeanBounds := directedRowMean?_encloses hn _ _ hcentered hcenterMean
    rw [sum_sub_rowMeanE hn, zero_div] at hcenterMeanBounds
    dsimp only at hout
    set recenteredLo :=
      Tensor.ofFn (fun j => subDown (centeredLo.getScalar j) centerMeanHi)
    set recenteredHi :=
      Tensor.ofFn (fun j => subUp (centeredHi.getScalar j) centerMeanLo)
    have hrecentered (j : Fin n) :
        value (recenteredLo.getScalar j) ≤ Spec.get2 x row j - rowMeanE x row ∧
          Spec.get2 x row j - rowMeanE x row ≤ value (recenteredHi.getScalar j) := by
      simpa only [recenteredLo, recenteredHi, Tensor.getScalar_ofFn, sub_zero] using
        sub_encloses (hcentered j) hcenterMeanBounds
    obtain ⟨_, _, hout⟩ := Option.bind_eq_some_iff.mp hout
    obtain ⟨⟨varianceLo, varianceHi⟩, hvariance, hout⟩ :=
      Option.bind_eq_some_iff.mp hout
    have hsquared (j : Fin n) :=
      square_encloses (hrecentered j).1 (hrecentered j).2
    have hvarianceBounds := directedRowMean?_encloses hn _ _
      (fun j => by simpa only [Tensor.getScalar_ofFn] using hsquared j) hvariance
    change value varianceLo ≤ rowVarE x row ∧ rowVarE x row ≤ value varianceHi
      at hvarianceBounds
    have hclamped := max_zero_encloses hvarianceBounds
    rw [max_eq_left (rowVarE_nonneg x row)] at hclamped
    have hstabilized := shift_encloses (bias := epsilon) hclamped.1 hclamped.2
    obtain ⟨stabilized, hstabilizedCheck, hout⟩ := Option.bind_eq_some_iff.mp hout
    have hstabilizedEq := checkedFiniteBounds?_eq_of_eq_some hstabilizedCheck
    subst stabilized
    obtain ⟨⟨denominatorLo, denominatorHi⟩, hdenominator, hout⟩ :=
      Option.bind_eq_some_iff.mp hout
    have hsqrtInput := max_zero_encloses hstabilized
    have hdenominatorBounds := LawfulNonlinearBoundOps.sqrtBounds_enclosure
      (checkedFiniteBounds?_bind_eq_some hdenominator) hsqrtInput.1 hsqrtInput.2
    split at hout
    · contradiction
    · obtain ⟨bounds, hbounds, hout⟩ := Option.bind_eq_some_iff.mp hout
      have hpoint := Internal.traverseFin_eq_some_iff.mp hbounds
      have houtEq := Option.some.inj hout
      cases houtEq
      intro j
      dsimp only
      rw [get2_layerNorm]
      simp only [Tensor.getScalar_ofFn]
      have hj := hpoint j
      obtain ⟨⟨lower, upper⟩, hdivide, hj⟩ := Option.bind_eq_some_iff.mp hj
      have hquotient := LawfulNonlinearBoundOps.divBounds_enclosure
        (checkedFiniteBounds?_bind_eq_some hdivide)
        (hcentered j).1 (hcentered j).2 hdenominatorBounds.1 hdenominatorBounds.2
      have hscaled := scale_encloses (scale := gamma.getScalar j)
        hquotient.1 hquotient.2
      obtain ⟨scaled, hscaledCheck, hj⟩ := Option.bind_eq_some_iff.mp hj
      have hscaledEq := checkedFiniteBounds?_eq_of_eq_some hscaledCheck
      subst scaled
      have hshifted := shift_encloses (bias := beta.getScalar j) hscaled.1 hscaled.2
      have hfinal := checkedFiniteBounds?_eq_of_eq_some hj
      simpa only [← hfinal] using hshifted

/-- Exact real endpoints instantiate the sequence theorem without literal hypotheses. -/
theorem directedLayerNormRow?_encloses_real {m n : Nat} (hm : 0 < m) (hn : 0 < n)
    (lo hi gamma beta : Tensor ℝ [n]) (epsilon : ℝ)
    (x : Tensor ℝ [m, n]) (row : Fin m)
    (hx : ∀ j, lo.getScalar j ≤ Spec.get2 x row j ∧
      Spec.get2 x row j ≤ hi.getScalar j)
    {outLo outHi : Tensor ℝ [n]}
    (hout : directedLayerNormRow? lo hi gamma beta epsilon = some (outLo, outHi)) :
    ∀ j, outLo.getScalar j ≤ Spec.get2 (Spec.layerNorm x gamma beta hm hn epsilon) row j ∧
      Spec.get2 (Spec.layerNorm x gamma beta hm hn epsilon) row j ≤ outHi.getScalar j := by
  simpa only [LawfulBoundOps.toReal, id_eq, Tensor.ofFn_getScalar] using
    directedLayerNormRow?_encloses (α := ℝ) hm hn lo hi gamma beta epsilon x row hx hout

end

end NN.MLTheory.CROWN.Graph.LayerNormDirected
