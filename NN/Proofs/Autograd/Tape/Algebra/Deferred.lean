/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Algebra.Context

/-!
Deferred uniform updates retain the order of scalar operations. Repeated updates can stop at a
checked fixed point; the proof uses decidable equality, with no additive identity assumption.
-/

@[expose] public section

namespace Proofs.Autograd.Algebra

namespace Deferred

/-- Apply the same operation a specified number of times, in evaluation order. -/
def iterateUpdates {β : Type} (step : β → β) : Nat → β → β
  | 0, value => value
  | count + 1, value => iterateUpdates step count (step value)

@[simp] theorem iterateUpdates_zero {β : Type} (step : β → β) (value : β) :
    iterateUpdates step 0 value = value := rfl

theorem iterateUpdates_succ {β : Type} (step : β → β) (count : Nat) (value : β) :
    iterateUpdates step (count + 1) value = step (iterateUpdates step count value) := by
  induction count generalizing value with
  | zero => rfl
  | succ count ih => exact ih (step value)

theorem iterateUpdates_fixed {β : Type} (step : β → β) (count : Nat) (value : β)
    (fixed : step value = value) : iterateUpdates step count value = value := by
  induction count with
  | zero => rfl
  | succ count ih => simpa only [iterateUpdates, fixed] using ih

/-- Stop repeating an operation only after checking that its actual result is a fixed point. -/
def repeatStable {β : Type} [DecidableEq β] (step : β → β) : Nat → β → β
  | 0, value => value
  | count + 1, value =>
      let next := step value
      if next = value then next else repeatStable step count next

/-- Fixed-point compression preserves every finite sequence of repeated operations. -/
theorem repeatStable_eq {β : Type} [DecidableEq β] (step : β → β)
    (count : Nat) (value : β) : repeatStable step count value = iterateUpdates step count value := by
  induction count generalizing value with
  | zero => rfl
  | succ count ih =>
      simp only [repeatStable, iterateUpdates]
      split
      next fixed => simp only [fixed, iterateUpdates_fixed step count value fixed]
      next _ => exact ih (step value)

/-- Consecutive occurrences of the same uniform contribution share one run. -/
abbrev History (α : Type) := List (α × Nat)

/-- Append one update to a history stored with its newest run first. -/
def push {α : Type} [DecidableEq α] (history : History α) (value : α) : History α :=
  match history with
  | [] => [(value, 1)]
  | (previous, count) :: rest =>
      if previous = value then (previous, count + 1) :: rest
      else (value, 1) :: history

/-- Apply the most recent `count` updates, in their original chronological order. -/
def applyRecent {α β : Type} (step : α → β → β) :
    History α → Nat → β → β
  | [], _, value => value
  | (uniform, runLength) :: rest, count, value =>
      iterateUpdates (step uniform) (min count runLength)
        (applyRecent step rest (count - runLength) value)

@[simp] theorem applyRecent_zero {α β : Type} (step : α → β → β)
    (history : History α) (value : β) : applyRecent step history 0 value = value := by
  induction history with
  | nil => rfl
  | cons entry rest ih =>
      rcases entry with ⟨uniform, count⟩
      simpa only [applyRecent, Nat.zero_min, Nat.zero_sub, iterateUpdates_zero] using ih

/-- Extending the history performs exactly one more update, after all previous updates. -/
theorem applyRecent_push {α β : Type} [DecidableEq α] (step : α → β → β)
    (history : History α) (uniform : α) (count : Nat) (value : β) :
    applyRecent step (push history uniform) (count + 1) value =
      step uniform (applyRecent step history count value) := by
  cases history with
  | nil => simp [push, applyRecent, iterateUpdates]
  | cons entry rest =>
      obtain ⟨previous, runLength⟩ := entry
      simp only [push]
      split
      next same =>
        subst previous
        have lengths : min (count + 1) (runLength + 1) = min count runLength + 1 := by omega
        simp only [applyRecent, Nat.add_sub_add_right, lengths, iterateUpdates_succ]
      next _ =>
        simp only [applyRecent, Nat.add_sub_cancel, Nat.min_eq_right (by omega : 1 ≤ count + 1),
          iterateUpdates]

/-- Execute a deferred history with checked fixed-point compression in each run. -/
def applyRecentStable {α β : Type} [DecidableEq β] (step : α → β → β) :
    History α → Nat → β → β
  | [], _, value => value
  | (uniform, runLength) :: rest, count, value =>
      repeatStable (step uniform) (min count runLength)
        (applyRecentStable step rest (count - runLength) value)

/-- Compressed history execution preserves the complete uncompressed update sequence. -/
theorem applyRecentStable_eq {α β : Type} [DecidableEq β] (step : α → β → β)
    (history : History α) (count : Nat) (value : β) :
    applyRecentStable step history count value = applyRecent step history count value := by
  induction history generalizing count with
  | nil => rfl
  | cons entry rest ih =>
      rcases entry with ⟨uniform, runLength⟩
      simp only [applyRecentStable, applyRecent, ih, repeatStable_eq]

end Deferred

end Proofs.Autograd.Algebra
