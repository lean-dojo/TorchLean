/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphConcatTensor
public import NN.IR.Semantics

/-!
# Adjacent swaps for arbitrary-axis concatenation

The concat evaluator moves the selected axis to the front, concatenates there, and moves it
back. The proofs below connect its adjacent swaps to the independent coordinate map for
tensor concatenation.
-/

public section

namespace NN.MLTheory.CROWN.Graph

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor.Internal (Rep Coord)

attribute [local simp] Bind.bind Pure.pure Except.bind Except.pure

private theorem bubbleLoop_eq_fold (axis count : Nat) (axes swaps : List Nat) :
    forIn (m := Except String) ({} : Lean.Loop) (axes, swaps, axis + count)
        (fun _ state =>
          if state.2.2 > axis then
            pure (.yield (Shape.swapAdjacentAxes state.1 (state.2.2 - 1),
              (state.2.2 - 1) :: state.2.1, state.2.2 - 1))
          else pure (.done state)) =
      .ok (
        let out := (List.range' axis count).reverse.foldl
          (fun state depth => (Shape.swapAdjacentAxes state.1 depth, depth :: state.2))
          (axes, swaps)
        (out.1, out.2, axis)) := by
  induction count generalizing axes swaps with
  | zero =>
      change Lean.Loop.forIn _ _ _ = _
      rw [Lean.Loop.forIn_eq_of_monadTail]
      simp
  | succ count ih =>
      change Lean.Loop.forIn _ _ _ = _
      rw [Lean.Loop.forIn_eq_of_monadTail]
      have hgt : axis + (count + 1) > axis := by omega
      have hsub : axis + (count + 1) - 1 = axis + count := by omega
      simp only [hgt, ↓reduceIte, hsub, pure_bind]
      have h := ih (Shape.swapAdjacentAxes axes (axis + count)) ((axis + count) :: swaps)
      change Lean.Loop.forIn _ _ _ = _ at h
      simpa only [List.range'_1_concat, List.reverse_append, List.reverse_singleton,
        List.singleton_append, List.foldl_cons] using h

private theorem swapAdjacentAxes_append_length
    (leading trailing : List Nat) (left right : Nat) :
    Shape.swapAdjacentAxes (leading ++ left :: right :: trailing) leading.length =
      leading ++ right :: left :: trailing := by
  induction leading with
  | nil => rfl
  | cons head leading ih =>
      simpa [Shape.swapAdjacentAxes] using congrArg (head :: ·) ih

/-- Moving an axis left through its preceding axes preserves their relative order. -/
theorem swapAdjacentAxes_reverse_range (leading trailing : List Nat) (axis : Nat) :
    (List.range leading.length).reverse.foldl Shape.swapAdjacentAxes
        (leading ++ axis :: trailing) =
      axis :: (leading ++ trailing) := by
  induction leading using List.reverseRecOn generalizing trailing with
  | nil => rfl
  | append_singleton leading last ih =>
      simp only [List.length_append, List.length_singleton, List.range_succ,
        List.reverse_append, List.reverse_singleton, List.singleton_append,
        List.foldl_cons, List.append_assoc, List.singleton_append]
      rw [swapAdjacentAxes_append_length]
      simpa only [List.append_assoc, List.singleton_append] using ih (last :: trailing)

private theorem swapFold_pair (depths axes swaps : List Nat) :
    depths.foldl
        (fun state depth => (Shape.swapAdjacentAxes state.1 depth, depth :: state.2))
        (axes, swaps) =
      (depths.foldl Shape.swapAdjacentAxes axes, depths.reverse ++ swaps) := by
  induction depths generalizing axes swaps with
  | nil => rfl
  | cons depth depths ih =>
      simpa [List.foldl_cons, List.reverse_cons, List.append_assoc] using
        ih (Shape.swapAdjacentAxes axes depth) (depth :: swaps)

private theorem bubbleLoop_front (leading trailing : List Nat) (axis : Nat)
    (swaps : List Nat) :
    forIn (m := Except String) ({} : Lean.Loop)
        (leading ++ axis :: trailing, swaps, leading.length)
        (fun _ state =>
          if state.2.2 > 0 then
            pure (.yield (Shape.swapAdjacentAxes state.1 (state.2.2 - 1),
              (state.2.2 - 1) :: state.2.1, state.2.2 - 1))
          else pure (.done state)) =
      .ok (axis :: (leading ++ trailing), List.range leading.length ++ swaps, 0) := by
  have h := bubbleLoop_eq_fold 0 leading.length (leading ++ axis :: trailing) swaps
  rw [← List.range_eq_range', swapFold_pair, swapAdjacentAxes_reverse_range,
    List.reverse_reverse] at h
  simpa only [Nat.zero_add] using h

private theorem findIdx?_getElem_of_nodup (axes : List Nat) (hnodup : axes.Nodup)
    (index : Nat) (hindex : index < axes.length) :
    axes.findIdx? (· == axes[index]) = some index := by
  apply List.findIdx?_eq_some_iff_getElem.mpr
  refine ⟨hindex, by simp, ?_⟩
  intro earlier hearlier
  simpa only [beq_iff_eq, hnodup.getElem_inj_iff] using Nat.ne_of_lt hearlier

private theorem forIn_unchanged {α β : Type} (items : List α) (state : β)
    (step : α → β → Except String (ForInStep β))
    (hstep : ∀ item ∈ items, step item state = .ok (.yield state)) :
    forIn items state step = .ok state := by
  induction items with
  | nil => rfl
  | cons item items ih =>
      rw [List.forIn_cons, hstep item (by simp)]
      simpa using ih (fun value hvalue => hstep value (by simp [hvalue]))

private theorem range_split_axis (axis trailing : Nat) :
    List.range (axis + 1 + trailing) =
      List.range axis ++ axis :: List.range' (axis + 1) trailing := by
  rw [List.range_eq_range']
  have h := List.range'_append_1 (s := 0) (m := axis) (n := trailing + 1)
  simpa [List.range'_succ, ← List.range_eq_range', Nat.add_assoc,
    Nat.add_comm, Nat.add_left_comm] using h.symm

private theorem filter_range_ne_axis (axis trailing : Nat) :
    (List.range (axis + 1 + trailing)).filter (· != axis) =
      List.range axis ++ List.range' (axis + 1) trailing := by
  have hleft : (List.range axis).filter (· != axis) = List.range axis := by
    apply List.filter_eq_self.mpr
    intro index hindex
    simpa using Nat.ne_of_lt (List.mem_range.mp hindex)
  have hright : (List.range' (axis + 1) trailing).filter (· != axis) =
      List.range' (axis + 1) trailing := by
    apply List.filter_eq_self.mpr
    intro index hindex
    have := List.mem_range'.mp hindex
    simp only [bne_iff_ne]
    omega
  simp only [range_split_axis, List.filter_append, List.filter_cons, bne_self_eq_false,
    Bool.false_eq_true, ↓reduceIte, hleft, hright]

/-- The public front-axis contract returns the structural rotation used by the swap proof. -/
theorem permMoveAxisToFront_append (leading trailing : Shape) (length : Nat) :
    NN.IR.OpContracts.permMoveAxisToFront leading.length (leading ++ length :: trailing) =
      .ok (leading.length ::
        (List.range leading.length ++
          List.range' (leading.length + 1) trailing.length)).toArray := by
  have hrank : (leading ++ length :: trailing).rank =
      leading.length + 1 + trailing.length := by
    simp only [Shape.rank_eq_length, List.length_append, List.length_cons]
    omega
  have haxis : leading.length < leading.length + 1 + trailing.length := by omega
  simp only [NN.IR.OpContracts.permMoveAxisToFront, NN.IR.OpContracts.checkAxisValid,
    hrank, haxis, ↓reduceIte, Bind.bind, Pure.pure, Except.bind, Except.pure]
  congr 1
  apply Array.toList_inj.mp
  simp only [Array.toList_append, Array.toList_filter,
    Array.toList_range, filter_range_ne_axis, List.singleton_append]

private theorem forIn_advances {β : Type} (start count : Nat) (tail : List Nat)
    (states : Nat → β)
    (step : Nat → β → Except String (ForInStep β))
    (hstep : ∀ index ∈ List.range' start count,
      step index (states index) = .ok (.yield (states (index + 1)))) :
    forIn (List.range' start count ++ tail) (states start) step =
      forIn tail (states (start + count)) step := by
  induction count generalizing start with
  | zero => rfl
  | succ count ih =>
      rw [List.range'_succ, List.cons_append, List.forIn_cons,
        hstep start (by simp [List.range'_succ])]
      simp only [Bind.bind, Except.bind]
      simpa only [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
        ih (start + 1) (fun index hindex =>
          hstep index (by simpa only [List.range'_succ, List.mem_cons] using Or.inr hindex))

private theorem findIdx?_after_zero (count : Nat) (tail : List Nat) :
    (List.range' 1 count ++ 0 :: (count + 1) :: tail).findIdx? (· == count + 1) =
      some (count + 1) := by
  apply List.findIdx?_eq_some_iff_getElem.mpr
  refine ⟨by simp, ?_, ?_⟩
  · rw [List.getElem_append_right (by simp)]
    simp
  · intro earlier hearlier
    by_cases hbefore : earlier < count
    · rw [List.getElem_append_left (by simpa using hbefore)]
      simp only [List.getElem_range', Nat.one_mul, beq_iff_eq]
      omega
    · have heq : earlier = count := by omega
      subst earlier
      rw [List.getElem_append_right (by simp)]
      simp

private theorem range_one_rotate_nodup (count trailing : Nat) :
    (List.range' 1 count ++ 0 :: List.range' (count + 1) trailing).Nodup := by
  apply List.nodup_middle.mpr
  have horder : 0 :: (List.range' 1 count ++ List.range' (count + 1) trailing) =
      List.range (count + 1 + trailing) := by
    rw [show count + 1 = 1 + count by omega, List.range'_append_1,
      List.range_eq_range', show 1 + count + trailing = (count + trailing) + 1 by omega,
      List.range'_succ]
  rw [horder]
  exact List.nodup_range

/-- The actual permutation planner moves one selected axis to the front by adjacent swaps. -/
theorem swapDepthsForPerm_front (axis trailing : Nat) :
    NN.IR.Graph.swapDepthsForPerm
        (axis :: (List.range axis ++ List.range' (axis + 1) trailing)).toArray
        (axis + 1 + trailing) =
      .ok (List.range axis).reverse.toArray := by
  have hnodup :
      (axis :: (List.range axis ++ List.range' (axis + 1) trailing)).Nodup := by
    apply List.nodup_middle.mp
    rw [← range_split_axis]
    exact List.nodup_range
  have hfind :
      (List.range (axis + 1 + trailing)).findIdx? (· == axis) = some axis := by
    apply List.findIdx?_eq_some_iff_getElem.mpr
    refine ⟨by simp; omega, by simp, ?_⟩
    intro earlier hearlier
    simpa using Nat.ne_of_lt hearlier
  unfold NN.IR.Graph.swapDepthsForPerm
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size,
    Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  rw [show axis + 1 + trailing = (axis + trailing) + 1 by omega,
    List.range'_succ, List.forIn_cons]
  simp only [List.getElem?_toArray, List.getElem?_cons_zero,
    show axis + trailing + 1 = axis + 1 + trailing by omega, hfind]
  rw [range_split_axis]
  have hbubble := bubbleLoop_front (List.range axis) (List.range' (axis + 1) trailing)
    axis []
  simp only [List.length_range, List.append_nil] at hbubble
  rw [hbubble]
  simp only [Bind.bind, Except.bind, Pure.pure, Except.pure]
  rw [forIn_unchanged]
  intro index hindex
  have hi : index < (axis :: (List.range axis ++
      List.range' (axis + 1) trailing)).length := by
    have := List.mem_range'.mp hindex
    simp only [List.length_cons, List.length_append, List.length_range,
      List.length_range']
    omega
  simp only [List.getElem?_eq_getElem hi]
  rw [findIdx?_getElem_of_nodup _ hnodup index hi]
  dsimp only
  have hloop := bubbleLoop_eq_fold index 0
    (axis :: (List.range axis ++ List.range' (axis + 1) trailing)) (List.range axis)
  simp only [Nat.add_zero, Pure.pure, Except.pure, List.range'_zero,
    List.reverse_nil, List.foldl_nil] at hloop
  rw [hloop]

/-- The inverse rotation moves the leading axis right, one adjacent swap at each position. -/
theorem swapDepthsForPerm_back (axis trailing : Nat) :
    NN.IR.Graph.swapDepthsForPerm
        (List.range' 1 axis ++ 0 :: List.range' (axis + 1) trailing).toArray
        (axis + 1 + trailing) =
      .ok (List.range axis).toArray := by
  let perm := List.range' 1 axis ++ 0 :: List.range' (axis + 1) trailing
  let states (index : Nat) : List Nat × List Nat :=
    (List.range' 1 index ++ 0 :: List.range' (index + 1) (axis - index + trailing),
      (List.range index).reverse)
  let step (index : Nat) (state : List Nat × List Nat) :
      Except String (ForInStep (List Nat × List Nat)) := do
    match perm.toArray[index]? with
    | none => throw s!"permute: internal error: missing perm[{index}]"
    | some target =>
        match state.1.findIdx? (· == target) with
        | none =>
            throw s!"permute: internal error: target axis {target} not in current axes {state.1}"
        | some position =>
            let result ← forIn ({} : Lean.Loop) (state.1, state.2, position)
              (fun _ current =>
                if current.2.2 > index then
                  pure (.yield (Shape.swapAdjacentAxes current.1 (current.2.2 - 1),
                    (current.2.2 - 1) :: current.2.1, current.2.2 - 1))
                else pure (.done current))
            pure (.yield (result.1, result.2.1))
  have hadvance : ∀ index ∈ List.range' 0 axis,
      step index (states index) = .ok (.yield (states (index + 1))) := by
    intro index hindex
    have hi : index < axis := by
      have := List.mem_range'.mp hindex
      omega
    have hget : perm[index]? = some (index + 1) := by
      dsimp only [perm]
      rw [List.getElem?_append_left (by simpa using hi)]
      simpa only [Nat.one_mul, Nat.add_comm] using
        (List.getElem?_range' (s := 1) (step := 1) hi)
    have hcount : axis - index + trailing = (axis - (index + 1) + trailing) + 1 := by
      omega
    simp only [step, states, List.getElem?_toArray, hget]
    rw [hcount, List.range'_succ, findIdx?_after_zero]
    dsimp only
    have hloop := bubbleLoop_eq_fold index 1
      (List.range' 1 index ++ 0 :: (index + 1) ::
        List.range' (index + 1 + 1) (axis - (index + 1) + trailing))
      (List.range index).reverse
    simp only [List.range'_succ, List.range'_zero, List.reverse_cons, List.reverse_nil,
      List.nil_append, List.foldl_cons, List.foldl_nil] at hloop
    rw [hloop]
    simp only [Bind.bind, Pure.pure, Except.bind, Except.pure]
    have hswap := swapAdjacentAxes_append_length (List.range' 1 index)
      (List.range' (index + 1 + 1) (axis - (index + 1) + trailing)) 0 (index + 1)
    simp only [List.length_range'] at hswap
    rw [hswap]
    simp [List.range'_1_concat, List.range_succ, List.reverse_append, List.append_assoc,
      Nat.add_comm]
  have hrest :
      forIn (List.range' axis (1 + trailing)) (states axis) step = .ok (states axis) := by
    apply forIn_unchanged
    intro index hindex
    have hi : index < perm.length := by
      have := List.mem_range'.mp hindex
      simp only [perm, List.length_append, List.length_range', List.length_cons]
      omega
    simp only [step, states, Nat.sub_self, Nat.zero_add, List.getElem?_toArray,
      List.getElem?_eq_getElem hi]
    rw [findIdx?_getElem_of_nodup _ (range_one_rotate_nodup axis trailing) index hi]
    dsimp only
    have hloop := bubbleLoop_eq_fold index 0 perm (List.range axis).reverse
    simp only [Nat.add_zero, List.range'_zero, List.reverse_nil, List.foldl_nil, perm] at hloop
    rw [hloop]
    rfl
  have hwhole := forIn_advances 0 axis (List.range' axis (1 + trailing))
    states step hadvance
  have hrange : List.range' 0 (axis + 1 + trailing) =
      List.range' 0 axis ++ List.range' axis (1 + trailing) := by
    simpa only [Nat.zero_add, Nat.add_assoc] using
      (List.range'_append_1 (s := 0) (m := axis) (n := 1 + trailing)).symm
  have hstart : (List.range (axis + 1 + trailing), ([] : List Nat)) = states 0 := by
    change (List.range (axis + 1 + trailing), []) =
      (0 :: List.range' 1 (axis + trailing), [])
    rw [List.range_eq_range', show axis + 1 + trailing = (axis + trailing) + 1 by omega,
      List.range'_succ]
  unfold NN.IR.Graph.swapDepthsForPerm
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size,
    Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  change (do
    let result ← forIn (List.range' 0 (axis + 1 + trailing))
      (List.range (axis + 1 + trailing), []) step
    pure result.2.reverse.toArray) = .ok (List.range axis).toArray
  rw [hrange, hstart, hwhole, Nat.zero_add, hrest]
  simp [states]

/-- Move the selected axis to the front by its independent coordinate map. -/
@[expose] def concatAxisToFront {α : Type} [Storage α] (leading trailing : Shape)
    {length : Nat} (tensor : Tensor α (leading ++ length :: trailing)) :
    Tensor α (length :: (leading ++ trailing)) :=
  Rep.ofFn fun coordinate =>
    let rest := Coord.appendEquiv leading trailing coordinate.2
    tensor ((Coord.appendEquiv leading (length :: trailing)).symm
      (rest.1, coordinate.1, rest.2))

/-- Moving the selected axis to the front commutes with family concatenation. -/
theorem concatAxisToFront_concatenateAxes {α : Type} [Storage α]
    (leading trailing : Shape) (lengths : List Nat)
    (values : (parent : Fin lengths.length) →
      Tensor α (leading ++ lengths.get parent :: trailing)) :
    concatAxisToFront leading trailing (Rep.concatenateAxes leading trailing lengths values) =
      Rep.concatenateAxes [] (leading ++ trailing) lengths
        (fun parent => concatAxisToFront leading trailing (values parent)) := by
  apply Rep.ext
  intro coordinate
  simp only [concatAxisToFront, Rep.concatenateAxes, Rep.get_ofFn]
  let rest := Coord.appendEquiv leading trailing coordinate.2
  let joined := (Coord.appendEquiv leading (lengths.sum :: trailing)).symm
    (rest.1, coordinate.1, rest.2)
  let separated := Coord.appendEquiv leading (lengths.sum :: trailing) joined
  let segment := (Tensor.Internal.segmentIndexEquiv lengths).symm coordinate.1
  change
    values ((Tensor.Internal.segmentIndexEquiv lengths).symm separated.2.1).1
        ((Coord.appendEquiv leading
          (lengths.get ((Tensor.Internal.segmentIndexEquiv lengths).symm
            separated.2.1).1 :: trailing)).symm
          (separated.1, ((Tensor.Internal.segmentIndexEquiv lengths).symm
            separated.2.1).2, separated.2.2)) =
      values segment.1 ((Coord.appendEquiv leading (lengths.get segment.1 :: trailing)).symm
        (rest.1, segment.2, rest.2))
  exact congrArg
    (fun parts : Coord leading × (Fin lengths.sum × Coord trailing) =>
      let selected := (Tensor.Internal.segmentIndexEquiv lengths).symm parts.2.1
      values selected.1
        ((Coord.appendEquiv leading (lengths.get selected.1 :: trailing)).symm
          (parts.1, selected.2, parts.2.2)))
    ((Coord.appendEquiv leading (lengths.sum :: trailing)).apply_symm_apply
      (rest.1, coordinate.1, rest.2))

private theorem swapFold_dim {α : Type} [Storage α] {n : Nat}
    (depths : List Nat) {source target : Shape}
    (values : Fin n → Tensor α source) (results : Fin n → Tensor α target)
    (hshape : source.applyAdjacentSwaps depths = target)
    (hvalues : ∀ index, depths.foldl SomeTensor.swapAdjacentAtDepth
        (SomeTensor.ofTensor (values index)) = SomeTensor.ofTensor (results index)) :
    (depths.map Nat.succ).foldl SomeTensor.swapAdjacentAtDepth
        (SomeTensor.ofTensor (Tensor.dim values)) =
      SomeTensor.ofTensor (Tensor.dim results) := by
  induction depths generalizing source with
  | nil =>
      change source = target at hshape
      subst target
      have h : values = results := by
        funext index
        exact eq_of_heq (SomeTensor.mk.inj (hvalues index)).2
      subst results
      rfl
  | cons depth depths ih =>
      simp only [List.map_cons, List.foldl_cons]
      simpa only [SomeTensor.swapAdjacentAtDepth, Tensor.swapAdjacentAxes,
        SomeTensor.ofTensor, Tensor.unstack_dim] using
        ih (fun index => Tensor.swapAdjacentAxes (values index) depth)
          hshape hvalues

private theorem concatAxisToFront_cons {α : Type} [Storage α]
    (head : Nat) (leading trailing : Shape) {length : Nat}
    (tensor : Tensor α ((head :: leading) ++ length :: trailing)) :
    concatAxisToFront (head :: leading) trailing tensor =
      Tensor.swapAdjacentAxes
        (Tensor.dim fun index =>
          concatAxisToFront leading trailing (Tensor.unstack tensor index)) 0 := by
  apply Rep.ext
  intro coordinate
  simp only [concatAxisToFront, Rep.get_ofFn, Tensor.swapAdjacentAxes_apply,
    Tensor.Internal.swapAdjacentAxesCoordinate, Tensor.dim, Tensor.unstack,
    Rep.stack_apply, Rep.unstack_apply]
  rfl

private theorem swapAdjacentAtDepth_eq_swapAdjacentAxes (shape : Shape) (depth : Nat) :
    shape.swapAdjacentAtDepth depth = Shape.swapAdjacentAxes shape depth := by
  induction depth generalizing shape with
  | zero =>
      cases shape with
      | scalar => rfl
      | dim head shape => cases shape <;> rfl
  | succ depth ih =>
      cases shape with
      | scalar => rfl
      | dim head shape =>
          cases shape with
          | scalar => cases depth <;> rfl
          | dim next shape => exact congrArg (head :: ·) (ih (next :: shape))

private theorem applyAdjacentSwaps_eq_fold (shape : Shape) (depths : List Nat) :
    shape.applyAdjacentSwaps depths = depths.foldl Shape.swapAdjacentAxes shape := by
  induction depths generalizing shape with
  | nil => rfl
  | cons depth depths ih =>
      simpa only [Shape.applyAdjacentSwaps, List.foldl_cons,
        swapAdjacentAtDepth_eq_swapAdjacentAxes] using ih (shape.swapAdjacentAtDepth depth)

private theorem swapFold_front {α : Type} [Storage α]
    (leading trailing : Shape) {length : Nat}
    (tensor : Tensor α (leading ++ length :: trailing)) :
    (List.range leading.length).reverse.foldl SomeTensor.swapAdjacentAtDepth
        (SomeTensor.ofTensor tensor) =
      SomeTensor.ofTensor (concatAxisToFront leading trailing tensor) := by
  induction leading with
  | scalar =>
      simp only [List.length_nil, List.range_zero, List.reverse_nil, List.foldl_nil]
      congr 1
      apply Rep.ext
      intro coordinate
      simp [concatAxisToFront, Coord.appendEquiv]
  | dim head leading ih =>
      have hdepths : (List.range (head :: leading).length).reverse =
          (List.range leading.length).reverse.map Nat.succ ++ [0] := by
        simp only [List.length_cons, List.range_succ_eq_map, List.reverse_cons,
          List.map_reverse]
      rw [hdepths, List.foldl_append]
      have hshape :
          (leading ++ length :: trailing).applyAdjacentSwaps
              (List.range leading.length).reverse =
            length :: (leading ++ trailing) := by
        rw [applyAdjacentSwaps_eq_fold, swapAdjacentAxes_reverse_range]
      have hfold := swapFold_dim (List.range leading.length).reverse
        (Tensor.unstack tensor)
        (fun index => concatAxisToFront leading trailing (Tensor.unstack tensor index))
        hshape (fun index => ih (Tensor.unstack tensor index))
      rw [Tensor.dim_unstack] at hfold
      rw [hfold]
      change SomeTensor.ofTensor (Tensor.swapAdjacentAxes _ 0) =
        SomeTensor.ofTensor (concatAxisToFront (head :: leading) trailing tensor)
      rw [concatAxisToFront_cons]

private theorem swapAdjacentAtDepth_involutive {α : Type} [Storage α]
    (value : SomeTensor α) (depth : Nat) :
    (value.swapAdjacentAtDepth depth).swapAdjacentAtDepth depth = value := by
  rcases value with ⟨shape, tensor⟩
  induction depth generalizing shape with
  | zero =>
      cases shape with
      | scalar => rfl
      | dim head shape =>
          cases shape with
          | scalar => rfl
          | dim next shape =>
              change SomeTensor.ofTensor _ = SomeTensor.ofTensor tensor
              congr 1
              apply Rep.ext
              intro coordinate
              simp only [Tensor.swapAdjacentAxes_apply,
                Tensor.Internal.swapAdjacentAxesCoordinate]
  | succ depth ih =>
      cases shape with
      | scalar => rfl
      | dim head shape =>
          have hshape : shape.applyAdjacentSwaps [depth, depth] = shape := by
            simp [Shape.applyAdjacentSwaps]
          have hfold := swapFold_dim [depth, depth]
            (Tensor.unstack tensor) (Tensor.unstack tensor) hshape
            (fun index => ih shape (Tensor.unstack tensor index))
          simpa only [List.map_cons, List.map_nil, List.foldl_cons, List.foldl_nil,
            Tensor.dim_unstack, SomeTensor.ofTensor] using hfold

private theorem swapFold_reverse {α : Type} [Storage α]
    (depths : List Nat) (value : SomeTensor α) :
    depths.reverse.foldl SomeTensor.swapAdjacentAtDepth
        (depths.foldl SomeTensor.swapAdjacentAtDepth value) = value := by
  induction depths generalizing value with
  | nil => rfl
  | cons depth depths ih =>
      simp only [List.reverse_cons, List.foldl_cons, List.foldl_append,
        List.foldl_nil, ih, swapAdjacentAtDepth_involutive]

private theorem swapFold_back {α : Type} [Storage α]
    (leading trailing : Shape) {length : Nat}
    (tensor : Tensor α (leading ++ length :: trailing)) :
    (List.range leading.length).foldl SomeTensor.swapAdjacentAtDepth
        (SomeTensor.ofTensor (concatAxisToFront leading trailing tensor)) =
      SomeTensor.ofTensor tensor := by
  have h := swapFold_reverse (List.range leading.length).reverse (SomeTensor.ofTensor tensor)
  simpa only [List.reverse_reverse, swapFold_front] using h

/-- A successful runtime front permutation reads exactly the independently rotated coordinates. -/
theorem permuteSomeTensor_front_of_ok {α : Type} [Storage α] [Context α]
    (leading trailing : Shape) {length : Nat}
    (tensor : Tensor α (leading ++ length :: trailing)) (result : SomeTensor α)
    (heval : NN.IR.Graph.permuteSomeTensor (SomeTensor.ofTensor tensor)
      (leading.length ::
        (List.range leading.length ++ List.range' (leading.length + 1) trailing.length)).toArray =
      .ok result) :
    result = SomeTensor.ofTensor (concatAxisToFront leading trailing tensor) := by
  have hrank : (leading ++ length :: trailing).rank =
      leading.length + 1 + trailing.length := by
    simp only [Shape.rank_eq_length, List.length_append, List.length_cons]
    omega
  have hplan := swapDepthsForPerm_front leading.length trailing.length
  cases hperm : Shape.permute? (leading ++ length :: trailing)
      (leading.length ::
        (List.range leading.length ++ List.range' (leading.length + 1) trailing.length)) with
  | none =>
      simp [NN.IR.Graph.permuteSomeTensor, hperm, NN.IR.throw_eq_error] at heval
  | some shape =>
      simp only [NN.IR.Graph.permuteSomeTensor, SomeTensor.shape_ofTensor,
        hperm, hrank, hplan, Bind.bind, Pure.pure, Except.bind,
        Except.pure, List.foldl_toArray, Except.ok.injEq] at heval
      exact heval.symm.trans (swapFold_front leading trailing tensor)

/-- A successful inverse runtime permutation restores a tensor after its front rotation. -/
theorem permuteSomeTensor_back_of_ok {α : Type} [Storage α] [Context α]
    (leading trailing : Shape) {length : Nat}
    (tensor : Tensor α (leading ++ length :: trailing)) (result : SomeTensor α)
    (heval : NN.IR.Graph.permuteSomeTensor
      (SomeTensor.ofTensor (concatAxisToFront leading trailing tensor))
      (List.range' 1 leading.length ++
        0 :: List.range' (leading.length + 1) trailing.length).toArray = .ok result) :
    result = SomeTensor.ofTensor tensor := by
  have hrank : Shape.rank (length :: (leading ++ trailing)) =
      leading.length + 1 + trailing.length := by
    simp only [Shape.rank_eq_length, List.length_cons, List.length_append]
    omega
  have hplan := swapDepthsForPerm_back leading.length trailing.length
  cases hperm : Shape.permute? (length :: (leading ++ trailing))
      (List.range' 1 leading.length ++ 0 :: List.range' (leading.length + 1) trailing.length) with
  | none =>
      simp [NN.IR.Graph.permuteSomeTensor, hperm, NN.IR.throw_eq_error] at heval
  | some shape =>
      simp only [NN.IR.Graph.permuteSomeTensor, SomeTensor.shape_ofTensor,
        hperm, hrank, hplan, Bind.bind, Pure.pure, Except.bind,
        Except.pure, List.foldl_toArray, Except.ok.injEq] at heval
      exact heval.symm.trans (swapFold_back leading trailing tensor)

end NN.MLTheory.CROWN.Graph
