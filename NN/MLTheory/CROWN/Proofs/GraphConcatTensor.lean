/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape.ConcatSlice
public import NN.Tensor.Internal.Laws.PackIndex

/-!
# Tensor concatenation by families and binary folds

The family coordinate map places each parent occurrence in its consecutive axis segment.
These identities connect that map to the binary tensor primitive used by the IR evaluator.
-/

public section

namespace TorchLean.Tensor.Internal

private theorem segmentIndexEquiv_cons_zero_val (length : Nat) (lengths : List Nat)
    (index : Fin length) :
    (segmentIndexEquiv (length :: lengths) ⟨0, index⟩).val = index.val := by
  rw [segmentIndexEquiv_val (length :: lengths) ⟨0, index⟩]
  change (∑ j : Fin 0, (length :: lengths).get ⟨j.val, by omega⟩) + index.val = _
  simp

private theorem segmentIndexEquiv_cons_succ_val (length : Nat) (lengths : List Nat)
    (parent : Fin lengths.length) (index : Fin (lengths.get parent)) :
    (segmentIndexEquiv (length :: lengths) ⟨parent.succ, index⟩).val =
      length + (segmentIndexEquiv lengths ⟨parent, index⟩).val := by
  have h := segmentIndexEquiv_val (length :: lengths) ⟨parent.succ, index⟩
  simpa [segmentIndexEquiv_val, Fin.sum_univ_succ, Nat.add_assoc] using h

private theorem concatenateAxes_apply_coordinate {α : Type} [Storage α]
    (leading trailing : Shape) (lengths : List Nat)
    (values : (parent : Fin lengths.length) →
      Rep α (leading ++ lengths.get parent :: trailing))
    (parent : Fin lengths.length)
    (source : Coord (leading ++ lengths.get parent :: trailing)) :
    Rep.concatenateAxes leading trailing lengths values
        (Rep.concatenateAxesCoordinateEquiv leading trailing lengths ⟨parent, source⟩) =
      values parent source := by
  rw [Rep.concatenateAxes, Rep.get_ofFn]
  change
    (fun coordinate : (p : Fin lengths.length) ×
        Coord (leading ++ lengths.get p :: trailing) =>
      values coordinate.1 coordinate.2)
      ((Rep.concatenateAxesCoordinateEquiv leading trailing lengths).symm
        (Rep.concatenateAxesCoordinateEquiv leading trailing lengths ⟨parent, source⟩)) = _
  rw [Equiv.symm_apply_apply]

private theorem concatenateAxesCoordinateEquiv_apply_append
    (leading trailing : Shape) (lengths : List Nat) (parent : Fin lengths.length)
    (front : Coord leading) (index : Fin (lengths.get parent)) (back : Coord trailing) :
    Rep.concatenateAxesCoordinateEquiv leading trailing lengths
        ⟨parent, (Coord.appendEquiv leading (lengths.get parent :: trailing)).symm
          (front, index, back)⟩ =
      (Coord.appendEquiv leading (lengths.sum :: trailing)).symm
        (front, segmentIndexEquiv lengths ⟨parent, index⟩, back) := by
  change
    (Coord.appendEquiv leading (lengths.sum :: trailing)).symm
      ((Coord.appendEquiv leading (lengths.get parent :: trailing)
          ((Coord.appendEquiv leading (lengths.get parent :: trailing)).symm
            (front, index, back))).1,
        segmentIndexEquiv lengths ⟨parent,
          (Coord.appendEquiv leading (lengths.get parent :: trailing)
            ((Coord.appendEquiv leading (lengths.get parent :: trailing)).symm
              (front, index, back))).2.1⟩,
        (Coord.appendEquiv leading (lengths.get parent :: trailing)
          ((Coord.appendEquiv leading (lengths.get parent :: trailing)).symm
            (front, index, back))).2.2) = _
  rw [Equiv.apply_symm_apply]

/-- A family concatenation separates into its first occurrence and the remaining family. -/
theorem Rep.concatenateAxes_cons {α : Type} [Storage α]
    (leading trailing : Shape) (length : Nat) (lengths : List Nat)
    (values : (parent : Fin (length :: lengths).length) →
      Rep α (leading ++ (length :: lengths).get parent :: trailing)) :
    Rep.concatenateAxes leading trailing (length :: lengths) values =
      Rep.concatenateAxis leading trailing (values 0)
        (Rep.concatenateAxes leading trailing lengths (fun parent => values parent.succ)) := by
  apply Rep.ext
  intro coordinate
  obtain ⟨⟨parent, source⟩, rfl⟩ :=
    (Rep.concatenateAxesCoordinateEquiv leading trailing (length :: lengths)).surjective
      coordinate
  rw [concatenateAxes_apply_coordinate]
  revert source
  refine Fin.cases ?_ (fun parent => ?_) parent
  · intro source
    obtain ⟨⟨front, index, back⟩, rfl⟩ :=
      (Coord.appendEquiv leading (length :: trailing)).symm.surjective source
    erw [concatenateAxesCoordinateEquiv_apply_append]
    change _ = Rep.concatenateAxis leading trailing
      (leftLength := length) (rightLength := lengths.sum) (values 0)
      (Rep.concatenateAxes leading trailing lengths (fun p => values p.succ))
      ((Coord.appendEquiv leading ((length + lengths.sum) :: trailing)).symm
        (front, segmentIndexEquiv (length :: lengths) ⟨0, index⟩, back))
    simp only [Rep.concatenateAxis, Rep.get_ofFn, Equiv.apply_symm_apply]
    have hzero := segmentIndexEquiv_cons_zero_val length lengths index
    split
    · exact congrArg (fun localIndex : Fin length =>
        values 0 ((Coord.appendEquiv leading (length :: trailing)).symm
          (front, localIndex, back))) (Fin.ext hzero.symm)
    · rename_i hnot
      exact (hnot (lt_of_eq_of_lt hzero index.isLt)).elim
  · intro source
    obtain ⟨⟨front, index, back⟩, rfl⟩ :=
      (Coord.appendEquiv leading (lengths.get parent :: trailing)).symm.surjective source
    erw [concatenateAxesCoordinateEquiv_apply_append]
    change _ = Rep.concatenateAxis leading trailing
      (leftLength := length) (rightLength := lengths.sum) (values 0)
      (Rep.concatenateAxes leading trailing lengths (fun p => values p.succ))
      ((Coord.appendEquiv leading ((length + lengths.sum) :: trailing)).symm
        (front, segmentIndexEquiv (length :: lengths) ⟨parent.succ, index⟩, back))
    have hnot : ¬ length + (segmentIndexEquiv lengths ⟨parent, index⟩).val < length := by omega
    simp only [Rep.concatenateAxis, Rep.get_ofFn, Equiv.apply_symm_apply,
      segmentIndexEquiv_cons_succ_val, hnot, ↓reduceDIte,
      Nat.add_sub_cancel_left]
    have h :=
      (concatenateAxes_apply_coordinate leading trailing lengths (fun p => values p.succ)
        parent ((Coord.appendEquiv leading (lengths.get parent :: trailing)).symm
          (front, index, back))).symm
    erw [concatenateAxesCoordinateEquiv_apply_append] at h
    exact h

/-- Appending an empty axis segment preserves every coordinate of the left tensor. -/
theorem Rep.concatenateAxis_zero_right {α : Type} [Storage α]
    (leading trailing : Shape) {length : Nat}
    (left : Rep α (leading ++ length :: trailing))
    (right : Rep α (leading ++ 0 :: trailing)) :
    Rep.concatenateAxis leading trailing left right = left := by
  apply Rep.ext
  intro coordinate
  simp [Rep.concatenateAxis]

/-- Concatenation of one occurrence preserves its tensor. -/
theorem Rep.concatenateAxes_singleton {α : Type} [Storage α]
    (leading trailing : Shape) (length : Nat)
    (values : (parent : Fin [length].length) →
      Rep α (leading ++ [length].get parent :: trailing)) :
    Rep.concatenateAxes leading trailing [length] values = values 0 := by
  have h := Rep.concatenateAxes_cons leading trailing length [] values
  exact h.trans (Rep.concatenateAxis_zero_right leading trailing (values 0) _)

private theorem cast_append_coordinate (leading trailing : Shape)
    {source target : Nat} (h : source = target)
    (front : Coord leading) (index : Fin source) (suffix : Coord trailing) :
    (congrArg (fun length => leading ++ length :: trailing) h ▸
        (Coord.appendEquiv leading (source :: trailing)).symm (front, index, suffix)) =
      (Coord.appendEquiv leading (target :: trailing)).symm
        (front, Fin.cast h index, suffix) := by
  subst target
  rfl

private theorem castShape_apply {α : Type} [Storage α]
    {source target : Shape} (h : source = target)
    (tensor : Rep α source) (coordinate : Coord target) :
    Rep.castShape h tensor coordinate = tensor (h.symm ▸ coordinate) := by
  subst target
  rfl

/-- Binary concatenation is associative, including zero-length segments. -/
theorem Rep.concatenateAxis_assoc {α : Type} [Storage α]
    (leading trailing : Shape) {first second third : Nat}
    (a : Rep α (leading ++ first :: trailing))
    (b : Rep α (leading ++ second :: trailing))
    (c : Rep α (leading ++ third :: trailing)) :
    Rep.castShape
        (congrArg (fun length => leading ++ length :: trailing)
          (Nat.add_assoc first second third))
        (Rep.concatenateAxis leading trailing (Rep.concatenateAxis leading trailing a b) c) =
      Rep.concatenateAxis leading trailing a (Rep.concatenateAxis leading trailing b c) := by
  apply Rep.ext
  intro coordinate
  obtain ⟨⟨front, index, suffix⟩, rfl⟩ :=
    (Coord.appendEquiv leading ((first + (second + third)) :: trailing)).symm.surjective
      coordinate
  rw [castShape_apply,
    cast_append_coordinate leading trailing (Nat.add_assoc first second third).symm]
  by_cases hfirst : index.val < first
  · have hpair : index.val < first + second := by omega
    simp [Rep.concatenateAxis, hfirst, hpair]
  · by_cases hpair : index.val < first + second
    · have hsecond : index.val - first < second := by omega
      simp [Rep.concatenateAxis, hfirst, hpair, hsecond]
    · have hsecond : ¬ index.val - first < second := by omega
      simp [Rep.concatenateAxis, hfirst, hpair, hsecond, Nat.sub_sub]

end TorchLean.Tensor.Internal
