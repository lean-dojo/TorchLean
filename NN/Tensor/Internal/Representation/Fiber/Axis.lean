/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Representation.Coordinate
public import NN.Tensor.Internal.Representation.Basic.Pointwise -- shake: keep
public import Mathlib.Data.Fintype.BigOperators -- shake: keep

/-!
# Axis-selection fibers

This module describes preimages of coordinate maps and proves the cardinality
of fibers induced by selecting a duplicate-free tuple of axes.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

universe u v

/-- The preimage of `y` under `f`, represented as a finite subtype when the
domain is finite. -/
abbrev Fiber {ι : Type u} {κ : Type v} (f : ι → κ) (y : κ) :=
  {x : ι // f x = y}

namespace AxisTuple

/--
The fiber of selecting `source` axes from a duplicate-free `target` tuple is
equivalent to a tuple over the axes of `target` absent from `source`.

This is the structural statement behind both repeat multiplicities and
reduction/contraction fibers. It includes the empty complement, which gives a
singleton fiber, and complements containing a zero-length axis, which give an
empty fiber.
-/
noncomputable def selectFiberEquiv {ι : Type u} [BEq ι] [LawfulBEq ι]
    {length : ι → Nat} {source target : List ι}
    (hSource : source.Nodup) (hTarget : target.Nodup)
    (hSourceTarget : ∀ axis, axis ∈ source → axis ∈ target)
    (sourceCoordinate : AxisTuple length source) :
    Fiber (select (length := length) hSourceTarget) sourceCoordinate ≃
      AxisTuple length
        (target.filter fun axis => !source.contains axis) := by
  classical
  let introducedAxes :=
    target.filter fun axis => !source.contains axis
  have hIntroduced : introducedAxes.Nodup := hTarget.filter _
  have hIntroducedTarget :
      ∀ axis, axis ∈ introducedAxes → axis ∈ target :=
    fun _ hAxis => (List.mem_filter.mp hAxis).1
  let sourceByAxis :
      AxisTuple length source ≃
        ((axis : {axis // axis ∈ source}) → Fin (length axis.1)) :=
    (hSource.getEquiv source).piCongr fun _ => Equiv.refl _
  let targetByAxis :
      AxisTuple length target ≃
        ((axis : {axis // axis ∈ target}) → Fin (length axis.1)) :=
    (hTarget.getEquiv target).piCongr fun _ => Equiv.refl _
  let introducedByAxis :
      AxisTuple length introducedAxes ≃
        ((axis : {axis // axis ∈ introducedAxes}) → Fin (length axis.1)) :=
    (hIntroduced.getEquiv introducedAxes).piCongr fun _ => Equiv.refl _
  let retainedAxisEquiv :
      {axis // axis ∈ source} ≃
        {axis : {axis // axis ∈ target} // axis.1 ∈ source} :=
    { toFun := fun axis =>
        ⟨⟨axis.1, hSourceTarget axis.1 axis.2⟩, axis.2⟩
      invFun := fun axis => ⟨axis.1.1, axis.2⟩
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  let introducedAxisEquiv :
      {axis // axis ∈ introducedAxes} ≃
        {axis : {axis // axis ∈ target} // axis.1 ∉ source} :=
    { toFun := fun axis =>
        ⟨⟨axis.1, (List.mem_filter.mp axis.2).1⟩,
          by simpa using (List.mem_filter.mp axis.2).2⟩
      invFun := fun axis =>
        ⟨axis.1.1,
          List.mem_filter.mpr ⟨axis.1.2, by simpa using axis.2⟩⟩
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  let splitByMembership :
      ((axis : {axis // axis ∈ target}) → Fin (length axis.1)) ≃
        (((axis : {axis : {axis // axis ∈ target} // axis.1 ∈ source}) →
            Fin (length axis.1.1)) ×
          ((axis : {axis : {axis // axis ∈ target} // axis.1 ∉ source}) →
            Fin (length axis.1.1))) :=
    Equiv.piEquivPiSubtypeProd
      (fun axis : {axis // axis ∈ target} => axis.1 ∈ source)
      (fun axis => Fin (length axis.1))
  let retainedTupleEquiv :
      ((axis : {axis : {axis // axis ∈ target} // axis.1 ∈ source}) →
          Fin (length axis.1.1)) ≃
        AxisTuple length source :=
    ((retainedAxisEquiv.piCongr fun _ => Equiv.refl _).symm).trans
      sourceByAxis.symm
  let introducedTupleEquiv :
      ((axis : {axis : {axis // axis ∈ target} // axis.1 ∉ source}) →
          Fin (length axis.1.1)) ≃
        AxisTuple length introducedAxes :=
    ((introducedAxisEquiv.piCongr fun _ => Equiv.refl _).symm).trans
      introducedByAxis.symm
  let splitTuple :
      AxisTuple length target ≃
        AxisTuple length source × AxisTuple length introducedAxes :=
    targetByAxis |>.trans <|
      splitByMembership |>.trans <|
        Equiv.prodCongr retainedTupleEquiv introducedTupleEquiv
  have splitTuple_fst (coordinate : AxisTuple length target) :
      (splitTuple coordinate).1 = select hSourceTarget coordinate := by
    apply sourceByAxis.injective
    funext sourceAxis
    obtain ⟨sourceIndex, rfl⟩ :=
      (hSource.getEquiv source).surjective sourceAxis
    let targetIndex : Fin target.length :=
      ⟨target.idxOf (source.get sourceIndex),
        List.idxOf_lt_length_iff.mpr
          (hSourceTarget _ (List.get_mem source sourceIndex))⟩
    have hTargetAxis :
        (hTarget.getEquiv target) targetIndex =
          ⟨((hSource.getEquiv source) sourceIndex).1,
            hSourceTarget _
              ((hSource.getEquiv source) sourceIndex).2⟩ := by
      apply Subtype.ext
      exact List.idxOf_get targetIndex.isLt
    have hTargetValue :
        ((targetByAxis coordinate)
          ⟨((hSource.getEquiv source) sourceIndex).1,
            hSourceTarget _
              ((hSource.getEquiv source) sourceIndex).2⟩).val =
          (coordinate targetIndex).val := by
      have hApply :=
        @Equiv.piCongr_apply_apply
          (Fin target.length)
          {axis // axis ∈ target}
          (fun targetPosition =>
            Fin (length (target.get targetPosition)))
          (fun axis => Fin (length axis.1))
          (hTarget.getEquiv target)
          (fun targetPosition : Fin target.length =>
            Equiv.refl (Fin (length (target.get targetPosition))))
          coordinate targetIndex
      have hValueAtIndex :
          ((targetByAxis coordinate)
              ((hTarget.getEquiv target) targetIndex)).val =
            (coordinate targetIndex).val := by
        calc
          ((targetByAxis coordinate)
                ((hTarget.getEquiv target) targetIndex)).val =
              ((Equiv.refl (Fin (length (target.get targetIndex))))
                (coordinate targetIndex)).val :=
            congrArg Fin.val hApply
          _ = (coordinate targetIndex).val := rfl
      have hAxisValue :=
        congrArg
          (fun axis : {axis // axis ∈ target} =>
            ((targetByAxis coordinate) axis).val)
          hTargetAxis
      exact hAxisValue.symm.trans hValueAtIndex
    have hSelectedValue :
        ((sourceByAxis (select hSourceTarget coordinate))
          ((hSource.getEquiv source) sourceIndex)).val =
          (coordinate targetIndex).val := by
      have hApply :=
        @Equiv.piCongr_apply_apply
          (Fin source.length)
          {axis // axis ∈ source}
          (fun sourcePosition =>
            Fin (length (source.get sourcePosition)))
          (fun axis => Fin (length axis.1))
          (hSource.getEquiv source)
          (fun sourcePosition : Fin source.length =>
            Equiv.refl (Fin (length (source.get sourcePosition))))
          (select hSourceTarget coordinate) sourceIndex
      have hSelectValue :
          ((select hSourceTarget coordinate) sourceIndex).val =
            (coordinate targetIndex).val := by
        simp [select, targetIndex]
      simpa only [sourceByAxis] using
        (congrArg Fin.val hApply).trans hSelectValue
    apply Fin.ext
    simp only [List.get_eq_getElem, Equiv.coe_fn_mk, Equiv.trans_apply,
      Equiv.piEquivPiSubtypeProd_apply, Equiv.prodCongr_apply, Equiv.coe_trans,
      Equiv.coe_piCongr_symm, Equiv.refl_symm, Equiv.refl_apply, Prod.map_apply,
      Function.comp_apply, Equiv.apply_symm_apply, sourceByAxis, splitTuple,
      splitByMembership, retainedTupleEquiv, retainedAxisEquiv]
    exact hTargetValue.trans hSelectedValue.symm
  have splitTuple_snd (coordinate : AxisTuple length target) :
      (splitTuple coordinate).2 = select hIntroducedTarget coordinate := by
    apply introducedByAxis.injective
    funext introducedAxis
    obtain ⟨introducedIndex, rfl⟩ :=
      (hIntroduced.getEquiv introducedAxes).surjective introducedAxis
    let targetIndex : Fin target.length :=
      ⟨target.idxOf (introducedAxes.get introducedIndex),
        List.idxOf_lt_length_iff.mpr
          (hIntroducedTarget _
            (List.get_mem introducedAxes introducedIndex))⟩
    have hTargetAxis :
        (hTarget.getEquiv target) targetIndex =
          ⟨((hIntroduced.getEquiv introducedAxes) introducedIndex).1,
            hIntroducedTarget _
              ((hIntroduced.getEquiv introducedAxes) introducedIndex).2⟩ := by
      apply Subtype.ext
      exact List.idxOf_get targetIndex.isLt
    have hTargetValue :
        ((targetByAxis coordinate)
          ⟨((hIntroduced.getEquiv introducedAxes) introducedIndex).1,
            hIntroducedTarget _
              ((hIntroduced.getEquiv introducedAxes) introducedIndex).2⟩).val =
          (coordinate targetIndex).val := by
      have hApply :=
        @Equiv.piCongr_apply_apply
          (Fin target.length)
          {axis // axis ∈ target}
          (fun targetPosition =>
            Fin (length (target.get targetPosition)))
          (fun axis => Fin (length axis.1))
          (hTarget.getEquiv target)
          (fun targetPosition : Fin target.length =>
            Equiv.refl (Fin (length (target.get targetPosition))))
          coordinate targetIndex
      have hValueAtIndex :
          ((targetByAxis coordinate)
              ((hTarget.getEquiv target) targetIndex)).val =
            (coordinate targetIndex).val := by
        calc
          ((targetByAxis coordinate)
                ((hTarget.getEquiv target) targetIndex)).val =
              ((Equiv.refl (Fin (length (target.get targetIndex))))
                (coordinate targetIndex)).val :=
            congrArg Fin.val hApply
          _ = (coordinate targetIndex).val := rfl
      have hAxisValue :=
        congrArg
          (fun axis : {axis // axis ∈ target} =>
            ((targetByAxis coordinate) axis).val)
          hTargetAxis
      exact hAxisValue.symm.trans hValueAtIndex
    have hSelectedValue :
        ((introducedByAxis (select hIntroducedTarget coordinate))
          ((hIntroduced.getEquiv introducedAxes) introducedIndex)).val =
          (coordinate targetIndex).val := by
      have hApply :=
        @Equiv.piCongr_apply_apply
          (Fin introducedAxes.length)
          {axis // axis ∈ introducedAxes}
          (fun introducedPosition =>
            Fin (length (introducedAxes.get introducedPosition)))
          (fun axis => Fin (length axis.1))
          (hIntroduced.getEquiv introducedAxes)
          (fun introducedPosition : Fin introducedAxes.length =>
            Equiv.refl
              (Fin (length (introducedAxes.get introducedPosition))))
          (select hIntroducedTarget coordinate) introducedIndex
      have hSelectValue :
          ((select hIntroducedTarget coordinate) introducedIndex).val =
            (coordinate targetIndex).val := by
        simp [select, targetIndex]
      simpa only [introducedByAxis] using
        (congrArg Fin.val hApply).trans hSelectValue
    apply Fin.ext
    simp only [List.get_eq_getElem, Equiv.coe_fn_mk, Equiv.trans_apply,
      Equiv.piEquivPiSubtypeProd_apply, Equiv.prodCongr_apply, Equiv.coe_trans,
      Equiv.coe_piCongr_symm, Equiv.refl_symm, Equiv.refl_apply, Prod.map_apply,
      Function.comp_apply, Equiv.apply_symm_apply, introducedByAxis, splitTuple,
      splitByMembership, introducedTupleEquiv, introducedAxisEquiv]
    exact hTargetValue.trans hSelectedValue.symm
  let fiberEquiv :
      Fiber (select (length := length) hSourceTarget) sourceCoordinate ≃
        AxisTuple length introducedAxes :=
    { toFun := fun coordinate => select hIntroducedTarget coordinate.1
      invFun := fun introducedCoordinate =>
        ⟨splitTuple.symm (sourceCoordinate, introducedCoordinate), by
          rw [← splitTuple_fst]
          simp⟩
      left_inv := fun coordinate => by
        apply Subtype.ext
        apply splitTuple.injective
        apply Prod.ext
        · simp only [Equiv.apply_symm_apply]
          exact ((splitTuple_fst coordinate.1).trans coordinate.2).symm
        · simp only [Equiv.apply_symm_apply]
          exact (splitTuple_snd coordinate.1).symm
      right_inv := fun introducedCoordinate => by
        change
          select hIntroducedTarget
              (splitTuple.symm (sourceCoordinate, introducedCoordinate)) =
            introducedCoordinate
        rw [← splitTuple_snd]
        simp }
  exact fiberEquiv

/--
The free tuple returned by `selectFiberEquiv` consists of the target
coordinates whose axes are absent from the selected source.
-/
@[simp] theorem selectFiberEquiv_apply {ι : Type u} [BEq ι] [LawfulBEq ι]
    {length : ι → Nat} {source target : List ι}
    (hSource : source.Nodup) (hTarget : target.Nodup)
    (hSourceTarget : ∀ axis, axis ∈ source → axis ∈ target)
    (sourceCoordinate : AxisTuple length source)
    (coordinate :
      Fiber (select (length := length) hSourceTarget) sourceCoordinate) :
    selectFiberEquiv hSource hTarget hSourceTarget sourceCoordinate coordinate =
      select (fun _ hAxis => (List.mem_filter.mp hAxis).1) coordinate.1 := by
  rfl

/--
The fiber of tuple selection has one free coordinate for every target axis
absent from the source.

Consequently, its cardinality is the product of the introduced axis lengths.
The empty product is one, while any introduced zero-length axis makes the
fiber empty.
-/
theorem select_fiber_card {ι : Type u} [BEq ι] [LawfulBEq ι]
    {length : ι → Nat} {source target : List ι}
    (hSource : source.Nodup) (hTarget : target.Nodup)
    (hSourceTarget : ∀ axis, axis ∈ source → axis ∈ target)
    (sourceCoordinate : AxisTuple length source) :
    Fintype.card
        (Fiber (select (length := length) hSourceTarget) sourceCoordinate) =
      ((target.filter fun axis => !source.contains axis).map length).prod := by
  classical
  let introducedAxes :=
    target.filter fun axis => !source.contains axis
  calc
    Fintype.card
          (Fiber (select (length := length) hSourceTarget) sourceCoordinate) =
        Fintype.card (AxisTuple length introducedAxes) :=
      Fintype.card_congr
        (selectFiberEquiv hSource hTarget hSourceTarget sourceCoordinate)
    _ = Fintype.card (Coord (introducedAxes.map length)) :=
      (Fintype.card_congr (coordEquiv length introducedAxes)).symm
    _ = Shape.size (introducedAxes.map length) :=
      Coord.card (introducedAxes.map length)
    _ = (introducedAxes.map length).prod :=
      Shape.size_eq_prod (introducedAxes.map length)
    _ = ((target.filter fun axis => !source.contains axis).map length).prod :=
      rfl

end AxisTuple

end TorchLean.Tensor.Internal
