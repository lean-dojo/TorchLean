/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Basic.Core

/-!
# Coordinate laws for checked slices

A slice embeds its output coordinates into its source coordinates. Its backward operation copies
coefficients along this embedding and fills the remaining source coordinates with zero.
-/

@[expose] public section

namespace TorchLean.Tensor

open Spec

/-- Distinct output coordinates of a checked slice read distinct source coordinates. -/
theorem sliceAxisRangeCoordinate_injective :
    (axis : Nat) → {shape : Shape} → [_hAxis : Shape.AxisInBounds axis shape] →
      (start count : Nat) → (hRange : start + count ≤ Shape.axisSize shape axis) →
      Function.Injective (sliceAxisRangeCoordinate axis start count hRange)
  | 0, .dim n rest, _, start, count, hRange => by
      rintro ⟨i, c⟩ ⟨j, d⟩ h
      apply Prod.ext
      · apply Fin.ext
        have hIndex := congrArg (fun p : (Shape.dim n rest).Coord => p.1.val) h
        change start + i.val = start + j.val at hIndex
        change i.val = j.val
        exact Nat.add_left_cancel hIndex
      · exact congrArg (fun p : (Shape.dim n rest).Coord => p.2) h
  | axis + 1, .dim n rest, hAxis, start, count, hRange => by
      let innerAxis : Shape.AxisInBounds axis rest := ⟨by
        have := hAxis.proof
        simp only [Shape.rank] at this
        omega⟩
      let := innerAxis
      have hRange' : start + count ≤ Shape.axisSize rest axis := by
        simpa only [Shape.axisSize_succ] using hRange
      rintro ⟨i, c⟩ ⟨j, d⟩ h
      apply Prod.ext
      · exact congrArg (fun p : (Shape.dim n rest).Coord => p.1) h
      · exact @sliceAxisRangeCoordinate_injective axis rest innerAxis start count hRange'
          c d (congrArg (fun p : (Shape.dim n rest).Coord => p.2) h)

/-- Reading the backward slice at a selected source coordinate recovers its coefficient. -/
theorem sliceAxisRangeBackwardSpec_apply_coordinate
    {α : Type} [Storage α] [Zero α] :
    (axis : Nat) → {shape : Shape} → [_hAxis : Shape.AxisInBounds axis shape] →
      (start count : Nat) → (hRange : start + count ≤ Shape.axisSize shape axis) →
      (value : Tensor α (shape.replaceAxis axis count)) →
      (c : (shape.replaceAxis axis count).Coord) →
      sliceAxisRangeBackwardSpec axis start count hRange value
          (sliceAxisRangeCoordinate axis start count hRange c) = value c
  | 0, .dim _ _, _, start, count, hRange, value, ⟨i, c⟩ => by
      have hBefore : ¬ start + i.val < start := by omega
      have hInside : start + i.val < start + count := Nat.add_lt_add_left i.isLt start
      simp only [Shape.replaceAxis] at value
      simp only [Shape.replaceAxis, sliceAxisRangeCoordinate, sliceAxisRangeBackwardSpec,
        Tensor.dim,
        Internal.Rep.stack_apply, hBefore, hInside, ↓reduceDIte,
        Spec.get, Tensor.unstack, Internal.Rep.unstack_apply]
      congr 1
      apply Prod.ext
      · apply Fin.ext
        change start + i.val - start = i.val
        omega
      · rfl
  | axis + 1, .dim _ rest, hAxis, start, count, hRange, value, ⟨i, c⟩ => by
      let innerAxis : Shape.AxisInBounds axis rest := ⟨by
        have := hAxis.proof
        simp only [Shape.rank] at this
        omega⟩
      let := innerAxis
      have hRange' : start + count ≤ Shape.axisSize rest axis := by
        simpa only [Shape.axisSize_succ] using hRange
      simp only [Shape.replaceAxis, sliceAxisRangeCoordinate, sliceAxisRangeBackwardSpec,
        Tensor.dim, Internal.Rep.stack_apply, Spec.get]
      exact (@sliceAxisRangeBackwardSpec_apply_coordinate α _ _ axis rest innerAxis
        start count hRange' (Tensor.unstack value i) c).trans
          (Internal.Rep.unstack_apply value i c)

/-- A backward slice is zero at every source coordinate outside the selected range. -/
theorem sliceAxisRangeBackwardSpec_apply_eq_zero
    {α : Type} [Storage α] [Zero α] :
    (axis : Nat) → {shape : Shape} → [_hAxis : Shape.AxisInBounds axis shape] →
      (start count : Nat) → (hRange : start + count ≤ Shape.axisSize shape axis) →
      (value : Tensor α (shape.replaceAxis axis count)) → (c : shape.Coord) →
      c ∉ Set.range (sliceAxisRangeCoordinate axis start count hRange) →
      sliceAxisRangeBackwardSpec axis start count hRange value c = 0
  | 0, .dim _ rest, _, start, count, _, value, ⟨i, c⟩, hOutside => by
      by_cases hBefore : i.val < start
      · simp [sliceAxisRangeBackwardSpec, Tensor.dim, Internal.Rep.stack_apply, hBefore,
          Tensor.full, Internal.Rep.const_apply]
      · by_cases hInside : i.val < start + count
        · exfalso
          apply hOutside
          refine ⟨(⟨i.val - start, by omega⟩, c), ?_⟩
          apply Prod.ext
          · apply Fin.ext
            simp only [sliceAxisRangeCoordinate]
            omega
          · rfl
        · simp [sliceAxisRangeBackwardSpec, Tensor.dim, Internal.Rep.stack_apply, hBefore,
            hInside, Tensor.full, Internal.Rep.const_apply]
  | axis + 1, .dim _ rest, hAxis, start, count, hRange, value, ⟨i, c⟩, hOutside => by
      let innerAxis : Shape.AxisInBounds axis rest := ⟨by
        have := hAxis.proof
        simp only [Shape.rank] at this
        omega⟩
      let := innerAxis
      have hRange' : start + count ≤ Shape.axisSize rest axis := by
        simpa only [Shape.axisSize_succ] using hRange
      have hInner : c ∉ Set.range (sliceAxisRangeCoordinate axis start count hRange') := by
        rintro ⟨d, hd⟩
        apply hOutside
        exact ⟨(i, d), congrArg (fun e => (i, e)) hd⟩
      simpa only [sliceAxisRangeBackwardSpec, Tensor.dim, Internal.Rep.stack_apply,
        Spec.get, Tensor.unstack, Internal.Rep.unstack_apply] using
        @sliceAxisRangeBackwardSpec_apply_eq_zero α _ _ axis rest innerAxis
          start count hRange' (Tensor.unstack value i) c hInner

end TorchLean.Tensor
