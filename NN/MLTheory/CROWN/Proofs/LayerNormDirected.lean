/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.Enclosure
public import NN.Proofs.Tensor.Basic.Folds

/-!
# Directed LayerNorm arithmetic

LayerNorm's variance stage squares a centered interval. An interval crossing zero has lower
square bound zero; otherwise the smaller endpoint square is a lower bound. The upper bound
is the larger endpoint square. The endpoint products must be rounded in the corresponding
direction before either selection.

These facts use the existing real interpretation of directed arithmetic. They compare the
computed endpoints with exact real operations, without assuming that ordinary backend
addition or multiplication is exact.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.LayerNormDirected

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN
open scoped BigOperators

noncomputable section

variable {α : Type} [TorchLean.Storage α] [Context α] [BoundOps α]
  [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- Endpoint minimum selects the smaller interpreted real value. -/
theorem value_min2 (a b : α) :
    value (BoundOps.min2 a b) = min (value a) (value b) := by
  by_cases h : b < a
  · have hreal := (LawfulBoundOps.lt_iff b a).mp h
    simp [BoundOps.min2, h, min_eq_right hreal.le]
  · have hreal : value a ≤ value b :=
      le_of_not_gt fun hlt => h ((LawfulBoundOps.lt_iff b a).mpr hlt)
    simp [BoundOps.min2, h, min_eq_left hreal]

/-- Endpoint maximum selects the larger interpreted real value. -/
theorem value_max2 (a b : α) :
    value (BoundOps.max2 a b) = max (value a) (value b) := by
  by_cases h : b < a
  · have hreal := (LawfulBoundOps.lt_iff b a).mp h
    simp [BoundOps.max2, h, max_eq_left hreal.le]
  · have hreal : value a ≤ value b :=
      le_of_not_gt fun hlt => h ((LawfulBoundOps.lt_iff b a).mpr hlt)
    simp [BoundOps.max2, h, max_eq_right hreal]

/-- A scalar square lies below the larger square of any enclosing interval's endpoints. -/
theorem square_le_max {lo x hi : ℝ} (hlo : lo ≤ x) (hhi : x ≤ hi) :
    x * x ≤ max (lo * lo) (hi * hi) := by
  by_cases hx : 0 ≤ x
  · exact (mul_self_le_mul_self hx hhi).trans (le_max_right _ _)
  · have hx' : 0 ≤ -x := neg_nonneg.mpr (le_of_not_ge hx)
    have h := mul_self_le_mul_self hx' (neg_le_neg hlo)
    have hsquare : x * x ≤ lo * lo := by simpa only [neg_mul_neg] using h
    exact hsquare.trans (le_max_left _ _)

/--
The directed square interval used by the LayerNorm variance stage encloses every input square.

The zero interpretation is stated explicitly because the base directed-arithmetic class
contains order and operation laws, but does not specify the interpretation of scalar literals.
-/
theorem square_encloses (hzero : value (0 : α) = 0)
    {lo hi : α} {x : ℝ} (hlo : value lo ≤ x) (hhi : x ≤ value hi) :
    value
        (if !(decide (0 < lo)) && !(decide (hi < 0)) then (0 : α)
         else BoundOps.min2 (BoundOps.mulDown lo lo) (BoundOps.mulDown hi hi)) ≤ x * x ∧
      x * x ≤ value
        (BoundOps.max2 (BoundOps.mulUp lo lo) (BoundOps.mulUp hi hi)) := by
  constructor
  · by_cases hl : (0 : α) < lo
    · have hlreal : 0 < value lo := by
        simpa only [hzero] using (LawfulBoundOps.lt_iff 0 lo).mp hl
      have hselect :
          (if !(decide (0 < lo)) && !(decide (hi < 0)) then (0 : α)
           else BoundOps.min2 (BoundOps.mulDown lo lo) (BoundOps.mulDown hi hi)) =
            BoundOps.min2 (BoundOps.mulDown lo lo) (BoundOps.mulDown hi hi) := by
        simp [hl]
      rw [hselect, value_min2]
      exact (min_le_left _ _).trans
        ((LawfulBoundOps.mulDown_le lo lo).trans (mul_self_le_mul_self hlreal.le hlo))
    · by_cases hh : hi < (0 : α)
      · have hhreal : value hi < 0 := by
          simpa only [hzero] using (LawfulBoundOps.lt_iff hi 0).mp hh
        have hselect :
            (if !(decide (0 < lo)) && !(decide (hi < 0)) then (0 : α)
             else BoundOps.min2 (BoundOps.mulDown lo lo) (BoundOps.mulDown hi hi)) =
              BoundOps.min2 (BoundOps.mulDown lo lo) (BoundOps.mulDown hi hi) := by
          simp [hh]
        rw [hselect, value_min2]
        have hsquare : value hi * value hi ≤ x * x := by
          have h := mul_self_le_mul_self (neg_nonneg.mpr hhreal.le) (neg_le_neg hhi)
          simpa only [neg_mul_neg] using h
        exact (min_le_right _ _).trans ((LawfulBoundOps.mulDown_le hi hi).trans hsquare)
      · simpa [hl, hh, hzero] using mul_self_nonneg x
  · rw [value_max2]
    exact (square_le_max hlo hhi).trans
      (max_le_max (LawfulBoundOps.le_mulUp lo lo) (LawfulBoundOps.le_mulUp hi hi))

/-- Multiplication by an exact stored scale respects both endpoint orders, including negative
scales. The scale itself is interpreted exactly; only endpoint products are directed. -/
theorem scale_encloses {lo hi scale : α} {x : ℝ}
    (hlo : value lo ≤ x) (hhi : x ≤ value hi) :
    value (BoundOps.min2 (BoundOps.mulDown lo scale) (BoundOps.mulDown hi scale)) ≤
        x * value scale ∧
      x * value scale ≤
        value (BoundOps.max2 (BoundOps.mulUp lo scale) (BoundOps.mulUp hi scale)) := by
  rw [value_min2, value_max2]
  by_cases hs : 0 ≤ value scale
  · exact ⟨(min_le_left _ _).trans
        ((LawfulBoundOps.mulDown_le lo scale).trans (mul_le_mul_of_nonneg_right hlo hs)),
      ((mul_le_mul_of_nonneg_right hhi hs).trans
        (LawfulBoundOps.le_mulUp hi scale)).trans (le_max_right _ _)⟩
  · have hs' : value scale ≤ 0 := le_of_not_ge hs
    exact ⟨(min_le_right _ _).trans
        ((LawfulBoundOps.mulDown_le hi scale).trans (mul_le_mul_of_nonpos_right hhi hs')),
      ((mul_le_mul_of_nonpos_right hlo hs').trans
        (LawfulBoundOps.le_mulUp lo scale)).trans (le_max_left _ _)⟩

/-- Directed endpoint addition encloses the affine bias shift. -/
theorem shift_encloses {lo hi bias : α} {x : ℝ}
    (hlo : value lo ≤ x) (hhi : x ≤ value hi) :
    value (BoundOps.addDown lo bias) ≤ x + value bias ∧
      x + value bias ≤ value (BoundOps.addUp hi bias) :=
  ⟨(LawfulBoundOps.addDown_le lo bias).trans (add_le_add hlo (le_refl (value bias))),
    (add_le_add hhi (le_refl (value bias))).trans (LawfulBoundOps.le_addUp hi bias)⟩

end

end NN.MLTheory.CROWN.Graph.LayerNormDirected
