/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import all NN.MLTheory.CROWN.Proofs.GraphConcatPermutation
public import NN.IR.Semantics

/-!
# The checked adjacent-swap planner

The planner fixes one requested axis at a time. Its processed prefix agrees with the requested
permutation, and its recorded swaps reproduce its current axis list.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open Spec

attribute [local simp] Bind.bind Pure.pure Except.bind Except.pure

theorem swapAdjacentAxes_perm (axes : List Nat) (depth : Nat) :
    (Shape.swapAdjacentAxes axes depth).Perm axes := by
  induction depth generalizing axes with
  | zero =>
      cases axes with
      | nil => exact .refl _
      | cons a axes =>
          cases axes with
          | nil => exact .refl _
          | cons b axes => exact .swap _ _ _
  | succ depth ih =>
      cases axes with
      | nil => exact .refl _
      | cons a axes =>
          cases axes with
          | nil => exact .refl _
          | cons b axes => exact (ih (b :: axes)).cons a

theorem swapAdjacentAxes_map (axes : List Nat) (depth : Nat) (f : Nat → Nat) :
    Shape.swapAdjacentAxes (axes.map f) depth =
      (Shape.swapAdjacentAxes axes depth).map f := by
  induction depth generalizing axes with
  | zero => cases axes with
    | nil => rfl
    | cons a axes => cases axes <;> rfl
  | succ depth ih => cases axes with
    | nil => rfl
    | cons a axes =>
        cases axes with
        | nil => rfl
        | cons b axes => exact congrArg (f a :: ·) (ih (b :: axes))

theorem swapFold_perm (depths axes : List Nat) :
    (depths.foldl Shape.swapAdjacentAxes axes).Perm axes := by
  induction depths generalizing axes with
  | nil => exact .refl _
  | cons depth depths ih =>
      exact (ih _).trans (swapAdjacentAxes_perm axes depth)

theorem swapFold_map (depths axes : List Nat) (f : Nat → Nat) :
    depths.foldl Shape.swapAdjacentAxes (axes.map f) =
      (depths.foldl Shape.swapAdjacentAxes axes).map f := by
  induction depths generalizing axes with
  | nil => rfl
  | cons depth depths ih =>
      simp only [List.foldl_cons, swapAdjacentAxes_map, ih]

theorem swapFold_reverse (depths axes : List Nat) :
    depths.reverse.foldl Shape.swapAdjacentAxes
      (depths.foldl Shape.swapAdjacentAxes axes) = axes := by
  simpa only [applyAdjacentSwaps_eq_fold] using
    Shape.applyAdjacentSwaps_reverse axes depths

private theorem swapAdjacentAxes_append (leading axes : List Nat) (depth : Nat) :
    Shape.swapAdjacentAxes (leading ++ axes) (leading.length + depth) =
      leading ++ Shape.swapAdjacentAxes axes depth := by
  simp only [← swapAdjacentAtDepth_eq_swapAdjacentAxes]
  induction leading with
  | nil => simp
  | cons a leading ih =>
      simpa only [List.cons_append, List.length_cons, Nat.succ_add,
        Shape.swapAdjacentAtDepth] using congrArg (a :: ·) ih

private theorem swapFold_append (leading axes depths : List Nat) :
    (depths.map (leading.length + ·)).foldl Shape.swapAdjacentAxes (leading ++ axes) =
      leading ++ depths.foldl Shape.swapAdjacentAxes axes := by
  induction depths generalizing axes with
  | nil => rfl
  | cons depth depths ih =>
      simp only [List.map_cons, List.foldl_cons, swapAdjacentAxes_append, ih]

private theorem bubbleFold_at (leading middle trailing : List Nat) (target : Nat) :
    (List.range' leading.length middle.length).reverse.foldl Shape.swapAdjacentAxes
        (leading ++ middle ++ target :: trailing) =
      leading ++ target :: (middle ++ trailing) := by
  rw [List.range'_eq_map_range, ← List.map_reverse, List.append_assoc,
    swapFold_append, swapAdjacentAxes_reverse_range]

private theorem bubbleFold_take (axes : List Nat) (i j : Nat)
    (hj : j < axes.length) (hij : i ≤ j) :
    ((List.range' i (j - i)).reverse.foldl Shape.swapAdjacentAxes axes).take (i + 1) =
      axes.take i ++ [axes[j]] := by
  let leading := axes.take i
  let middle := (axes.drop i).take (j - i)
  let trailing := axes.drop (j + 1)
  have hleading : leading.length = i := by simp [leading]; omega
  have hmiddle : middle.length = j - i := by
    simp only [middle, List.length_take, List.length_drop]
    omega
  have hsplit : leading ++ middle ++ axes[j] :: trailing = axes := by
    change axes.take i ++ (axes.drop i).take (j - i) ++ axes[j] ::
      axes.drop (j + 1) = axes
    rw [← List.take_add, Nat.add_sub_of_le hij, ← List.drop_eq_getElem_cons hj,
      List.take_append_drop]
  have h := bubbleFold_at leading middle trailing axes[j]
  rw [hsplit, hleading, hmiddle] at h
  rw [h, List.take_append]
  simp only [hleading, Nat.add_sub_cancel_left, List.take_succ_cons, List.take_zero,
    List.take_of_length_le (by omega : leading.length ≤ i + 1)]
  rfl

private def plannerStep (perm : Array Nat) (i : Nat)
    (state : List Nat × List Nat) : Except String (ForInStep (List Nat × List Nat)) := do
  match perm[i]? with
  | none => throw s!"permute: internal error: missing perm[{i}]"
  | some target =>
      match state.1.findIdx? (· == target) with
      | none =>
          throw s!"permute: internal error: target axis {target} not in current axes {state.1}"
      | some j =>
          let result ← forIn (m := Except String) ({} : Lean.Loop)
            (state.1, state.2, j) fun _ current =>
              if current.2.2 > i then
                pure (.yield (Shape.swapAdjacentAxes current.1 (current.2.2 - 1),
                  (current.2.2 - 1) :: current.2.1, current.2.2 - 1))
              else pure (.done current)
          pure (.yield (result.1, result.2.1))

private def PlannerInvariant (perm : Array Nat) (i : Nat)
    (state : List Nat × List Nat) : Prop :=
  state.1.length = perm.size ∧ state.1.take i = perm.toList.take i ∧
    state.1 = state.2.reverse.foldl Shape.swapAdjacentAxes (List.range perm.size)

private theorem plannerStep_invariant (perm : Array Nat) (hnodup : perm.toList.Nodup)
    (i : Nat) (hi : i < perm.size) (state : List Nat × List Nat)
    (hstate : PlannerInvariant perm i state)
    (out : ForInStep (List Nat × List Nat))
    (hrun : plannerStep perm i state = .ok out) :
    ∃ next, out = .yield next ∧ PlannerInvariant perm (i + 1) next := by
  rcases state with ⟨axes, saved⟩
  rcases hstate with ⟨hlen, htake, hsaved⟩
  unfold plannerStep at hrun
  rw [getElem?_pos perm i hi] at hrun
  dsimp only at hrun
  cases hfind : axes.findIdx? (· == perm[i]) with
  | none => simp only [hfind, NN.IR.throw_eq_error, reduceCtorEq] at hrun
  | some j =>
      obtain ⟨hj, htarget, _⟩ := List.findIdx?_eq_some_iff_getElem.mp hfind
      simp only [beq_iff_eq] at htarget
      have hij : i ≤ j := by
        by_contra h
        have hji : j < i := by omega
        have hpj : j < perm.toList.length := by simpa using Nat.lt_trans hji hi
        have hsame : axes[j] = perm.toList[j] := by
          have h := congrArg (fun xs : List Nat => xs[j]?) htake
          simpa only [List.getElem?_take, hji, ite_true,
            getElem?_pos axes j hj, getElem?_pos perm.toList j hpj,
            Option.some.injEq] using h
        have hne := hnodup.getElem_inj_iff
          (i := j) (j := i) (hi := by simpa using Nat.lt_trans hji hi)
          (hj := by simpa using hi)
        have heq : perm.toList[j] = perm.toList[i] := hsame.symm.trans htarget
        have := hne.mp heq
        omega
      have hloop := bubbleLoop_eq_fold i (j - i) axes saved
      rw [Nat.add_sub_of_le hij, swapFold_pair] at hloop
      simp only [hfind] at hrun
      rw [hloop] at hrun
      simp only [Bind.bind, Except.bind, Pure.pure, Except.pure, Except.ok.injEq] at hrun
      subst out
      refine ⟨_, rfl, ?_, ?_, ?_⟩
      · exact (swapFold_perm _ axes).length_eq.trans hlen
      · rw [bubbleFold_take axes i j hj hij, htake, List.take_add_one]
        simp only [getElem?_pos perm.toList i (by simpa using hi), Array.getElem_toList,
          Option.toList_some, htarget]
      · simp only [List.reverse_append, List.reverse_reverse, List.foldl_append,
          ← hsaved]

private theorem plannerRun_invariant (perm : Array Nat) (hnodup : perm.toList.Nodup)
    (start count : Nat) (hcount : start + count = perm.size)
    (state result : List Nat × List Nat)
    (hstate : PlannerInvariant perm start state)
    (hrun : forIn (List.range' start count) state (plannerStep perm) = .ok result) :
    PlannerInvariant perm perm.size result := by
  induction count generalizing start state with
  | zero =>
      simp only [List.range'_zero, List.forIn_nil, Pure.pure, Except.pure,
        Except.ok.injEq] at hrun
      subst result
      simpa only [Nat.add_zero] using hcount ▸ hstate
  | succ count ih =>
      rw [List.range'_succ, List.forIn_cons] at hrun
      cases hs : plannerStep perm start state with
      | error e => simp only [hs, Bind.bind, Except.bind, reduceCtorEq] at hrun
      | ok out =>
          obtain ⟨next, rfl, hnext⟩ :=
            plannerStep_invariant perm hnodup start (by omega) state hstate out hs
          simp only [hs, Bind.bind, Except.bind] at hrun
          exact ih (start + 1) (by omega) next hnext hrun

/-- Every successful checked plan realizes the entire requested axis ordering. -/
theorem swapDepthsForPerm_realizes (perm : Array Nat) (rank : Nat)
    (hlen : perm.size = rank) (hnodup : perm.toList.Nodup)
    (swaps : Array Nat) (hplan : NN.IR.Graph.swapDepthsForPerm perm rank = .ok swaps) :
    swaps.toList.foldl Shape.swapAdjacentAxes (List.range rank) = perm.toList := by
  subst rank
  unfold NN.IR.Graph.swapDepthsForPerm at hplan
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size,
    Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one, ← List.range_eq_range'] at hplan
  change (do
    let result ← forIn (List.range perm.size) (List.range perm.size, [])
      (plannerStep perm)
    pure result.2.reverse.toArray) = .ok swaps at hplan
  cases hrun : forIn (List.range perm.size) (List.range perm.size, [])
      (plannerStep perm) with
  | error e => simp only [hrun, Bind.bind, Except.bind, reduceCtorEq] at hplan
  | ok result =>
      simp only [hrun, Bind.bind, Except.bind, Pure.pure, Except.pure,
        Except.ok.injEq] at hplan
      subst swaps
      have hi : PlannerInvariant perm 0 (List.range perm.size, []) := by
        simp [PlannerInvariant]
      have h := plannerRun_invariant perm hnodup 0 perm.size (by omega) _ _ hi
        (by simpa only [← List.range_eq_range'] using hrun)
      rcases h with ⟨hsize, htake, hsaved⟩
      simpa only [List.toList_toArray, ← hsaved,
        List.take_of_length_le (le_of_eq hsize),
        List.take_of_length_le (le_of_eq perm.length_toList)] using htake

end NN.MLTheory.CROWN.Graph
