/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.List
public import TorchLean.Verify.CROWN.LayerNorm.Directed
public import TorchLean.Verify.CROWN.BoundOps.Interval
public import TorchLean.NN.Norm.MatrixEntries

/-!
# Enclosure by the directed LayerNorm sequence

A successful executable row transfer encloses the matching row of `Tensor.layerNorm` over the last
axis (`layerNormRow?_encloses`). The proof follows the transfer's directed sum and count
folds, both centering stages, the variance, the square root, the division and the stored affine
parameters. Each step uses a scalar operation contract, and the real side is read through the
entrywise formula `get2_layerNorm`. A normalized entry has magnitude at most the square root of the
row width (`layerNorm_abs_le_sqrt`).
-/

public section

namespace TorchLean.CROWN.Graph

open TorchLean.Tensor
open BoundOps
open LawfulBoundOps (toReal_maximum)
open Normalization
open scoped BigOperators

noncomputable section

/-- For a nonempty row and a nonnegative `epsilon`, a centered row entry divided by its stabilized
standard deviation has magnitude at most the square root of the row width. A zero denominator gives
the quotient zero under Lean's real division. -/
theorem layerNorm_abs_le_sqrt {m n : Nat} (hn : 0 < n)
    (x : Tensor ℝ [m, n]) (epsilon : ℝ) (hepsilon : 0 ≤ epsilon)
    (i : Fin m) (j : Fin n) :
    |(Tensor.get2 x i j - 𝔼 k, Tensor.get2 x i k) /
        Real.sqrt (max (TorchLean.var (Tensor.get2 x i) + epsilon) 0)| ≤
      Real.sqrt n := by
  have hnreal : (0 : ℝ) < n := by exact_mod_cast hn
  have hentry :
      (Tensor.get2 x i j - 𝔼 k, Tensor.get2 x i k) ^ 2 ≤
        (n : ℝ) * TorchLean.var (Tensor.get2 x i) := by
    have hs := Finset.single_le_sum
      (fun k (_ : k ∈ (Finset.univ : Finset (Fin n))) =>
        mul_self_nonneg (Tensor.get2 x i k - 𝔼 l, Tensor.get2 x i l))
      (Finset.mem_univ j)
    simpa only [TorchLean.var, Fintype.expect_eq_sum_div_card, Fintype.card_fin,
      mul_div_cancel₀ _ hnreal.ne', pow_two] using hs
  have ht : TorchLean.var (Tensor.get2 x i) ≤
      max (TorchLean.var (Tensor.get2 x i) + epsilon) 0 :=
    (le_add_of_nonneg_right hepsilon).trans (le_max_left _ _)
  have hsquare := hentry.trans (mul_le_mul_of_nonneg_left ht hnreal.le)
  have habs := Real.abs_le_sqrt hsquare
  rw [Real.sqrt_mul hnreal.le] at habs
  by_cases hz : Real.sqrt (max (TorchLean.var (Tensor.get2 x i) + epsilon) 0) = 0
  · simp [hz, Real.sqrt_nonneg]
  · have hp : 0 < Real.sqrt (max (TorchLean.var (Tensor.get2 x i) + epsilon) 0) :=
      lt_of_le_of_ne (Real.sqrt_nonneg _) (Ne.symm hz)
    rw [abs_div, abs_of_nonneg (Real.sqrt_nonneg _)]
    exact (div_le_iff₀ hp).mpr habs

variable {α : Type} [Context α]

/-- When a scalar transfer followed by `finiteBounds?` succeeds, the transfer alone returned
the same bounds. -/
theorem finiteBounds?_bind_eq_some {action : Option (α × α)} {output : α × α}
    (h : (action >>= finiteBounds?) = some output) : action = some output := by
  obtain ⟨result, hresult, hcheck⟩ := Option.bind_eq_some_iff.mp h
  exact hresult.trans (congrArg some (finiteBounds?_eq_of_eq_some hcheck))

variable [Storage α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- If the accumulators enclose `sum` and `count`, the executable four-accumulator fold over
`indices` encloses `sum` plus every `f i` and `count` plus the number of indices. -/
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

/-- For a nonempty row whose bounds enclose `f`, a successful `rowMean?` encloses the mean
`(∑ i, f i) / n`. The real mean divides by the exact count `n`, which the directed fold of ones
encloses. -/
theorem rowMean?_encloses [NonlinearBoundOps α] [LawfulNonlinearBoundOps α]
    {n : Nat}
    (hn : 0 < n) (bounds : Fin n → α × α) (f : Fin n → ℝ)
    (hb : ∀ i, value (bounds i).1 ≤ f i ∧ f i ≤ value (bounds i).2)
    {outLo outHi : α} (hout : rowMean? bounds = some (outLo, outHi)) :
    value outLo ≤ (∑ i, f i) / n ∧ (∑ i, f i) / n ≤ value outHi := by
  have hfold := sum_count_fold_encloses bounds f hb (List.finRange n)
    (0, 0, 0, 0) 0 0 (by simp [(LawfulBoundOps.toReal_zero (α := α))])
  simp only [List.finRange_foldl_add_eq_add_sum, zero_add, Finset.sum_const,
    Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one] at hfold
  simp only [rowMean?, hn.ne', ↓reduceIte] at hout
  obtain ⟨_, _, hout⟩ := Option.bind_eq_some_iff.mp hout
  obtain ⟨_, _, hout⟩ := Option.bind_eq_some_iff.mp hout
  exact LawfulNonlinearBoundOps.divBounds_enclosure
    (finiteBounds?_bind_eq_some hout)
    hfold.1 hfold.2.1 hfold.2.2.1 hfold.2.2.2

/-- The lower endpoint `aLo - bHi` rounded down and the upper endpoint `aHi - bLo` rounded up
enclose `a - b`. -/
theorem sub_encloses {aLo aHi bLo bHi : α} {a b : ℝ}
    (ha : value aLo ≤ a ∧ a ≤ value aHi)
    (hb : value bLo ≤ b ∧ b ≤ value bHi) :
    value (subDown aLo bHi) ≤ a - b ∧ a - b ≤ value (subUp aHi bLo) :=
  ⟨(LawfulBoundOps.subDown_le _ _).trans (sub_le_sub ha.1 hb.2),
    (sub_le_sub ha.2 hb.1).trans (LawfulBoundOps.le_subUp _ _)⟩

/-- Clamping both endpoints at zero encloses the clamped real value. -/
theorem max_zero_encloses
    {lo hi : α} {x : ℝ} (h : value lo ≤ x ∧ x ≤ value hi) :
    value (Context.maximum lo 0) ≤ max x 0 ∧ max x 0 ≤ value (Context.maximum hi 0) := by
  simpa only [toReal_maximum, (LawfulBoundOps.toReal_zero (α := α))] using
    And.intro (max_le_max h.1 le_rfl) (max_le_max h.2 le_rfl)

/--
For a nonempty row whose bounds `lo` and `hi` enclose row `row` of `x`, a successful
`layerNormRow?` encloses that row of the real `Tensor.layerNorm` over the last axis.

The real side reads `gamma`, `beta` and `epsilon` through `LawfulBoundOps.toReal`, and the
`LawfulBoundOps` laws supply the exact interpretation of zero and one. The statement assumes only
the input enclosure and the scalar operation laws.
-/
theorem layerNormRow?_encloses [NonlinearBoundOps α]
    [LawfulNonlinearBoundOps α]
    {m n : Nat} (hn : 0 < n)
    (lo hi gamma beta : Tensor α [n]) (epsilon : α)
    (x : Tensor ℝ [m, n]) (row : Fin m)
    (hx : ∀ j, value (lo.getScalar j) ≤ Tensor.get2 x row j ∧
      Tensor.get2 x row j ≤ value (hi.getScalar j))
    {outLo outHi : Tensor α [n]}
    (hout : layerNormRow? lo hi gamma beta epsilon = some (outLo, outHi)) :
    ∀ j,
      let y := Tensor.get2 (Tensor.layerNorm x [⟨1, Nat.one_lt_two⟩]
        (Tensor.ofFn fun k : Fin n => value (gamma.getScalar k))
        (Tensor.ofFn fun k : Fin n => value (beta.getScalar k)) (value epsilon)) row j
      value (outLo.getScalar j) ≤ y ∧ y ≤ value (outHi.getScalar j) := by
  unfold layerNormRow? at hout
  obtain ⟨_, _, hout⟩ := Option.bind_eq_some_iff.mp hout
  split at hout
  · contradiction
  · obtain ⟨_, _, hout⟩ := Option.bind_eq_some_iff.mp hout
    obtain ⟨⟨meanLo, meanHi⟩, hmean, hout⟩ := Option.bind_eq_some_iff.mp hout
    have hE (f : Fin n → ℝ) : (∑ k, f k) / n = 𝔼 k, f k := by
      rw [Fintype.expect_eq_sum_div_card, Fintype.card_fin]
    have hmeanBounds := rowMean?_encloses hn _ _ hx hmean
    rw [hE] at hmeanBounds
    dsimp only at hout
    set centeredLo := Tensor.ofFn (fun j => subDown (lo.getScalar j) meanHi)
    set centeredHi := Tensor.ofFn (fun j => subUp (hi.getScalar j) meanLo)
    have hcentered (j : Fin n) :
        value (centeredLo.getScalar j) ≤ Tensor.get2 x row j - 𝔼 k, Tensor.get2 x row k ∧
          Tensor.get2 x row j - 𝔼 k, Tensor.get2 x row k ≤ value (centeredHi.getScalar j) := by
      simpa only [centeredLo, centeredHi, Tensor.getScalar_ofFn] using
        sub_encloses (hx j) hmeanBounds
    obtain ⟨_, _, hout⟩ := Option.bind_eq_some_iff.mp hout
    obtain ⟨⟨centerMeanLo, centerMeanHi⟩, hcenterMean, hout⟩ :=
      Option.bind_eq_some_iff.mp hout
    have hcenterMeanBounds := rowMean?_encloses hn _ _ hcentered hcenterMean
    have h0 : ∑ j, (Tensor.get2 x row j - 𝔼 k, Tensor.get2 x row k) = 0 :=
      Fintype.sum_balance (Tensor.get2 x row)
    rw [h0, zero_div] at hcenterMeanBounds
    dsimp only at hout
    set recenteredLo :=
      Tensor.ofFn (fun j => subDown (centeredLo.getScalar j) centerMeanHi)
    set recenteredHi :=
      Tensor.ofFn (fun j => subUp (centeredHi.getScalar j) centerMeanLo)
    have hrecentered (j : Fin n) :
        value (recenteredLo.getScalar j) ≤ Tensor.get2 x row j - 𝔼 k, Tensor.get2 x row k ∧
          Tensor.get2 x row j - 𝔼 k, Tensor.get2 x row k ≤ value (recenteredHi.getScalar j) := by
      simpa only [recenteredLo, recenteredHi, Tensor.getScalar_ofFn, sub_zero] using
        sub_encloses (hcentered j) hcenterMeanBounds
    obtain ⟨_, _, hout⟩ := Option.bind_eq_some_iff.mp hout
    obtain ⟨⟨varianceLo, varianceHi⟩, hvariance, hout⟩ :=
      Option.bind_eq_some_iff.mp hout
    have hsquared (j : Fin n) :=
      square_encloses (hrecentered j).1 (hrecentered j).2
    have hvarianceBounds := rowMean?_encloses hn _ _
      (fun j => by simpa only [Tensor.getScalar_ofFn, squareBounds] using hsquared j) hvariance
    rw [hE] at hvarianceBounds
    change value varianceLo ≤ TorchLean.var (Tensor.get2 x row) ∧
      TorchLean.var (Tensor.get2 x row) ≤ value varianceHi at hvarianceBounds
    have hclamped := max_zero_encloses hvarianceBounds
    rw [max_eq_left (var_nonneg (Tensor.get2 x row))] at hclamped
    have hstabilized := shift_encloses (bias := epsilon) hclamped.1 hclamped.2
    obtain ⟨stabilized, hstabilizedCheck, hout⟩ := Option.bind_eq_some_iff.mp hout
    have hstabilizedEq := finiteBounds?_eq_of_eq_some hstabilizedCheck
    subst stabilized
    obtain ⟨⟨denominatorLo, denominatorHi⟩, hdenominator, hout⟩ :=
      Option.bind_eq_some_iff.mp hout
    have hsqrtInput := max_zero_encloses hstabilized
    have hdenominatorBounds := LawfulNonlinearBoundOps.sqrtBounds_enclosure
      (finiteBounds?_bind_eq_some hdenominator) hsqrtInput.1 hsqrtInput.2
    split at hout
    · contradiction
    · obtain ⟨bounds, hbounds, hout⟩ := Option.bind_eq_some_iff.mp hout
      have hpoint := Tensor.sequenceFinM_get_of_eq_some hbounds
      have houtEq := Option.some.inj hout
      cases houtEq
      intro j
      dsimp only
      have hl := get2_layerNorm hn (value epsilon) x
        (Tensor.ofFn fun k : Fin n => value (gamma.getScalar k))
        (Tensor.ofFn fun k : Fin n => value (beta.getScalar k)) row j
      simp only [Tensor.getScalar_ofFn] at hl
      erw [hl]
      simp only [Tensor.getScalar_ofFn]
      have hj := hpoint j
      obtain ⟨⟨lower, upper⟩, hdivide, hj⟩ := Option.bind_eq_some_iff.mp hj
      have hquotient := LawfulNonlinearBoundOps.divBounds_enclosure
        (finiteBounds?_bind_eq_some hdivide)
        (hcentered j).1 (hcentered j).2 hdenominatorBounds.1 hdenominatorBounds.2
      have hscaled := scale_encloses (scale := gamma.getScalar j)
        hquotient.1 hquotient.2
      obtain ⟨scaled, hscaledCheck, hj⟩ := Option.bind_eq_some_iff.mp hj
      have hscaledEq := finiteBounds?_eq_of_eq_some hscaledCheck
      subst scaled
      have hshifted := shift_encloses (bias := beta.getScalar j) hscaled.1 hscaled.2
      have hfinal := finiteBounds?_eq_of_eq_some hj
      simpa only [← hfinal] using hshifted

end

end TorchLean.CROWN.Graph
