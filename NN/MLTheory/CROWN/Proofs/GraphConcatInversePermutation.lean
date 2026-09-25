/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.OpContracts

/-!
# Inverting the front-axis permutation

The checked inverse-permutation algorithm fills an optional index table and then reads it
in axis order. Its loop invariant identifies each written entry by its inverse position.
Specializing the invariant to a front-axis rotation gives the permutation used to restore
an arbitrary concat axis after concatenating along the leading axis.
-/

public section

set_option autoImplicit false
set_option warningAsError true

namespace NN.MLTheory.CROWN.Graph

open NN.IR.OpContracts

attribute [local simp] Bind.bind Pure.pure Except.bind Except.pure

private theorem inverseTable_update (rank count : Nat) (inverse : Nat → Nat)
    (table : Array (Option Nat)) (axis : Nat)
    (hsize : table.size = rank) (haxis : axis < rank)
    (hindex : inverse axis = count)
    (hinjective : ∀ other < rank, inverse other = count → other = axis)
    (htable : ∀ other < rank,
      table[other]! = if inverse other < count then some (inverse other) else none) :
    ∀ other < rank, (table.set! axis (some count))[other]! =
      if inverse other < count + 1 then some (inverse other) else none := by
  intro other hother
  by_cases heq : axis = other
  · subst other
    rw [Array.getElem!_set!_self _ _ _ (by omega), hindex,
      ite_eq_left (Nat.lt_succ_self count)]
  · rw [Array.getElem!_set!_ne _ _ _ _ heq, htable other hother]
    have hne : inverse other ≠ count := by
      intro heqIndex
      exact heq (hinjective other hother heqIndex).symm
    have hlt : (inverse other < count + 1) = (inverse other < count) := by
      apply propext
      omega
    simp only [hlt]

-- The table contains precisely the inverse positions already visited by the fill loop.
private theorem inverseEntries_of_index (rank : Nat) (inverse : Nat → Nat)
    (hinjective : ∀ a < rank, ∀ b < rank, inverse a = inverse b → a = b)
    (step : Nat → Array (Option Nat) × Nat →
      Except String (ForInStep (Array (Option Nat) × Nat)))
    (hstep : ∀ table count axis, axis < rank → table[axis]! = none →
      step axis (table, count) = .ok (.yield (table.set! axis (some count), count + 1)))
    (items : List Nat) (count : Nat) (table : Array (Option Nat))
    (hentries : ∀ index (hi : index < items.length),
      items[index] < rank ∧ inverse items[index] = count + index)
    (hsize : table.size = rank)
    (htable : ∀ axis < rank,
      table[axis]! = if inverse axis < count then some (inverse axis) else none) :
    ∃ result : Array (Option Nat),
      forIn items (table, count) step = .ok (result, count + items.length) ∧
      result.size = rank ∧
      ∀ axis < rank, result[axis]! =
        if inverse axis < count + items.length then some (inverse axis) else none := by
  induction items generalizing count table with
  | nil => exact ⟨table, rfl, hsize, htable⟩
  | cons axis items ih =>
      have haxis := hentries 0 (by simp)
      simp only [List.getElem_cons_zero, Nat.add_zero] at haxis
      have hnone : table[axis]! = none := by
        rw [htable axis haxis.1, haxis.2]
        simp
      have hnext := inverseTable_update rank count inverse table axis hsize haxis.1
        haxis.2 (fun other hother heq => hinjective other hother axis haxis.1
          (heq.trans haxis.2.symm)) htable
      have htail : ∀ index (hi : index < items.length),
          items[index] < rank ∧ inverse items[index] = count + 1 + index := by
        intro index hi
        have h := hentries (index + 1) (by simpa using hi)
        change items[index] < rank ∧ inverse items[index] = count + (index + 1) at h
        exact ⟨h.1, h.2.trans (by omega)⟩
      obtain ⟨result, hrun, hrsize, hrtable⟩ :=
        ih (count + 1) (table.set! axis (some count)) htail
          (by rw [Array.size_set!, hsize]) hnext
      have hcount : count + 1 + items.length = count + (axis :: items).length := by
        simp only [List.length_cons]
        omega
      rw [hcount] at hrun hrtable
      refine ⟨result, ?_, hrsize, hrtable⟩
      rw [List.forIn_cons, hstep table count axis haxis.1 hnone]
      exact hrun

-- A complete table is collected in increasing axis order.
private theorem collect_inverseEntries (table : Array (Option Nat)) (inverse : Nat → Nat)
    (missing : Nat → String) (start count : Nat) (out : Array Nat)
    (hread : ∀ axis ∈ List.range' start count, table[axis]! = some (inverse axis)) :
    forIn (List.range' start count) out
        (fun axis out => match table[axis]! with
          | some index =>
              (pure (ForInStep.yield (out.push index)) : Except String (ForInStep (Array Nat)))
          | none => .error (missing axis)) =
      .ok (out ++ ((List.range' start count).map inverse).toArray) := by
  induction count generalizing start out with
  | zero => simp
  | succ count ih =>
      have hfirst : table[start]! = some (inverse start) := by
        apply hread
        simp
      rw [List.range'_succ, List.forIn_cons, hfirst]
      change forIn (m := Except String) (List.range' (start + 1) count)
        (out.push (inverse start)) _ = _
      rw [ih (start + 1) (out.push (inverse start)) (by
        intro axis haxis
        apply hread
        simp only [List.range'_succ, List.mem_cons]
        exact Or.inr haxis)]
      simp only [List.map_cons]
      rw [List.toArray_cons, Array.push_eq_append, Array.append_assoc]

private theorem inversePerm_eq_of_inverse (perm : Array Nat) (inverse : Nat → Nat)
    (hinjective : ∀ a < perm.size, ∀ b < perm.size, inverse a = inverse b → a = b)
    (hbound : ∀ index (hi : index < perm.size), perm[index] < perm.size)
    (hindex : ∀ index (hi : index < perm.size), inverse perm[index] = index)
    (hinverse : ∀ axis < perm.size, inverse axis < perm.size) :
    inversePerm perm = .ok ((List.range perm.size).map inverse).toArray := by
  let step : Nat → Array (Option Nat) × Nat →
      Except String (ForInStep (Array (Option Nat) × Nat)) := fun axis state => do
    unless axis < perm.size do
      throw s!"permute: axis {axis} out of range for rank {perm.size} in {repr perm}"
    if state.1[axis]!.isSome then
      throw s!"permute: duplicate axis {axis} in {repr perm}"
    pure (.yield (state.1.set! axis (some state.2), state.2 + 1))
  have hstep : ∀ table count axis, axis < perm.size → table[axis]! = none →
      step axis (table, count) = .ok (.yield (table.set! axis (some count), count + 1)) := by
    intro table count axis haxis hnone
    simp [step, haxis, hnone]
  obtain ⟨table, hrun, _, hread⟩ := inverseEntries_of_index perm.size inverse hinjective
    step hstep perm.toList 0 (Array.replicate perm.size none)
    (by
      intro index hi
      simpa using And.intro (hbound index hi) (hindex index hi))
    (by simp)
    (by
      intro axis haxis
      simp [getElem!_pos, haxis])
  simp only [Array.length_toList, Nat.zero_add] at hrun hread
  have hall : ∀ axis < perm.size, table[axis]! = some (inverse axis) := by
    intro axis haxis
    simpa only [hinverse axis haxis, ↓reduceIte] using hread axis haxis
  unfold inversePerm
  change (forIn (m := Except String) perm
    (Array.replicate perm.size none, 0) step >>= _) = _
  rw [← Array.forIn_toList, hrun]
  change (forIn (m := Except String) [:perm.size] (#[] : Array Nat) _ >>= pure) = _
  rw [bind_pure]
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size,
    Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  have hcollect := collect_inverseEntries table inverse
    (fun axis => s!"permute: missing axis {axis} in {repr perm}") 0 perm.size #[]
    (by
      intro axis haxis
      apply hall
      simpa [List.mem_range'] using haxis)
  simp only [← List.range_eq_range', Array.empty_append] at hcollect
  simp only [← List.range_eq_range']
  rw [← hcollect]
  apply congrArg (forIn (m := Except String) (List.range perm.size) (#[] : Array Nat))
  funext axis out
  cases table[axis]! <;> rfl

/-- Inverting a front-axis rotation restores the original order of the preceding axes. -/
theorem inversePerm_front (axis trailing : Nat) :
    inversePerm (axis :: (List.range axis ++ List.range' (axis + 1) trailing)).toArray =
      .ok (List.range' 1 axis ++ [0] ++ List.range' (axis + 1) trailing).toArray := by
  let inverse : Nat → Nat := fun value =>
    if value < axis then value + 1 else if value = axis then 0 else value
  let perm := (axis :: (List.range axis ++ List.range' (axis + 1) trailing)).toArray
  have hsize : perm.size = axis + 1 + trailing := by simp [perm]; omega
  have hzero : inverse axis = 0 := by simp [inverse]
  have hlow : (List.range axis).map inverse = List.range' 1 axis := by
    rw [List.range'_eq_map_range]
    apply List.map_congr_left
    intro value hvalue
    have hvalue := List.mem_range.mp hvalue
    simp [inverse, hvalue, Nat.add_comm]
  have hhigh : (List.range' (axis + 1) trailing).map inverse =
      List.range' (axis + 1) trailing := by
    calc
      _ = (List.range' (axis + 1) trailing).map id := by
        apply List.map_congr_left
        intro value hvalue
        have hvalue := List.mem_range'.mp hvalue
        have hlo : ¬ value < axis := by omega
        have hne : value ≠ axis := by omega
        simp [inverse, hlo, hne]
      _ = _ := List.map_id _
  have hrange : List.range (axis + 1 + trailing) =
      List.range axis ++ axis :: List.range' (axis + 1) trailing := by
    have h := List.range'_append_1 (s := 0) (m := axis) (n := trailing + 1)
    simpa [List.range'_succ, ← List.range_eq_range', Nat.add_assoc,
      Nat.add_comm, Nat.add_left_comm] using h.symm
  have hmap : perm.toList.map inverse = List.range perm.size := by
    simp only [perm, List.map_cons, List.map_append,
      hzero, hlow, hhigh]
    rw [show (axis :: (List.range axis ++ List.range' (axis + 1) trailing)).toArray.size =
      axis + 1 + trailing from hsize]
    have hjoin : List.range' 1 axis ++ List.range' (axis + 1) trailing =
        List.range' 1 (axis + trailing) := by
      simpa only [Nat.add_comm axis 1] using
        (List.range'_append_1 (s := 1) (m := axis) (n := trailing))
    rw [hjoin, List.range_eq_range']
    rw [show axis + 1 + trailing = (axis + trailing) + 1 by omega, List.range'_succ]
  have hresult := inversePerm_eq_of_inverse perm inverse
    (by
      intro a ha b hb heq
      dsimp only [inverse] at heq
      split_ifs at heq <;> omega)
    (by
      intro index hi
      have hmem : perm[index] ∈ axis ::
          (List.range axis ++ List.range' (axis + 1) trailing) :=
        Array.getElem_mem_toList hi
      simp only [List.mem_cons, List.mem_append,
        List.mem_range, List.mem_range'] at hmem
      rw [hsize]
      rcases hmem with h | h | h <;> omega)
    (by
      intro index hi
      have h := congrArg (fun entries : List Nat => entries[index]?) hmap
      simpa only [List.getElem?_map,
        List.getElem?_eq_getElem (show index < perm.toList.length from hi),
        List.getElem?_range hi, Option.map_some, Option.some.injEq,
        Array.getElem_toList] using h)
    (by
      intro value hvalue
      rw [hsize] at hvalue ⊢
      dsimp only [inverse]
      split_ifs <;> omega)
  rw [hsize, hrange, List.map_append, List.map_cons, hlow, hzero, hhigh] at hresult
  simpa only [perm, List.append_assoc, List.singleton_append] using hresult

end NN.MLTheory.CROWN.Graph
