/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.BoundOps.Lawful
public import TorchLean.Verify.CROWN.BoundOps.Interval

/-!
# Directed LayerNorm arithmetic

LayerNorm's variance stage squares a centered interval. Its lower square bound is zero when the
interval contains zero and the smaller endpoint square otherwise, and its upper bound is the larger
endpoint square. Each endpoint product is rounded down for the lower bound and up for the upper
bound before the minimum or maximum is taken.

The lemmas read directed arithmetic through `LawfulBoundOps.toReal` and compare the computed
endpoints with exact real operations, without assuming that backend addition or multiplication is
exact. They cover the variance square, the stored scale and the bias shift.
-/

@[expose] public section

namespace TorchLean.CROWN.Graph

open LawfulBoundOps (toReal_minimum toReal_maximum)
open CROWN (square_le_max)

noncomputable section

variable {α : Type} [Storage α] [Context α] [BoundOps α]
  [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/--
The directed square interval of the LayerNorm variance stage encloses `x * x` for every `x` in the
input interval.

The `LawfulBoundOps.toReal_zero` law supplies the exact interpretation of zero.
-/
theorem square_encloses
    {lo hi : α} {x : ℝ} (hlo : value lo ≤ x) (hhi : x ≤ value hi) :
    value
        (if !(decide (0 < lo)) && !(decide (hi < 0)) then (0 : α)
         else Context.minimum (BoundOps.mulDown lo lo) (BoundOps.mulDown hi hi)) ≤ x * x ∧
      x * x ≤ value
        (Context.maximum (BoundOps.mulUp lo lo) (BoundOps.mulUp hi hi)) := by
  constructor
  · by_cases hl : (0 : α) < lo
    · have hlreal : 0 < value lo := by
        simpa only [(LawfulBoundOps.toReal_zero (α := α))] using (LawfulBoundOps.lt_iff 0 lo).mp hl
      have hselect :
          (if !(decide (0 < lo)) && !(decide (hi < 0)) then (0 : α)
           else Context.minimum (BoundOps.mulDown lo lo) (BoundOps.mulDown hi hi)) =
            Context.minimum (BoundOps.mulDown lo lo) (BoundOps.mulDown hi hi) := by
        simp [hl]
      rw [hselect, toReal_minimum]
      exact (min_le_left _ _).trans
        ((LawfulBoundOps.mulDown_le lo lo).trans (mul_self_le_mul_self hlreal.le hlo))
    · by_cases hh : hi < (0 : α)
      · have hhreal : value hi < 0 := by
          simpa only [LawfulBoundOps.toReal_zero (α := α)] using
            (LawfulBoundOps.lt_iff hi 0).mp hh
        have hselect :
            (if !(decide (0 < lo)) && !(decide (hi < 0)) then (0 : α)
             else Context.minimum (BoundOps.mulDown lo lo) (BoundOps.mulDown hi hi)) =
              Context.minimum (BoundOps.mulDown lo lo) (BoundOps.mulDown hi hi) := by
          simp [hh]
        rw [hselect, toReal_minimum]
        have hsquare : value hi * value hi ≤ x * x := by
          have h := mul_self_le_mul_self (neg_nonneg.mpr hhreal.le) (neg_le_neg hhi)
          simpa only [neg_mul_neg] using h
        exact (min_le_right _ _).trans ((LawfulBoundOps.mulDown_le hi hi).trans hsquare)
      · simpa [hl, hh, (LawfulBoundOps.toReal_zero (α := α))] using mul_self_nonneg x
  · rw [toReal_maximum]
    exact (square_le_max hlo hhi).trans
      (max_le_max (LawfulBoundOps.le_mulUp lo lo) (LawfulBoundOps.le_mulUp hi hi))

/-- The smaller rounded-down endpoint product and the larger rounded-up one enclose `x` times the
stored scale, for a scale of either sign. The scale is read exactly, and only the endpoint products
are rounded. -/
theorem scale_encloses {lo hi scale : α} {x : ℝ}
    (hlo : value lo ≤ x) (hhi : x ≤ value hi) :
    value (Context.minimum (BoundOps.mulDown lo scale) (BoundOps.mulDown hi scale)) ≤
        x * value scale ∧
      x * value scale ≤
        value (Context.maximum (BoundOps.mulUp lo scale) (BoundOps.mulUp hi scale)) := by
  rw [toReal_minimum, toReal_maximum]
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

end TorchLean.CROWN.Graph
