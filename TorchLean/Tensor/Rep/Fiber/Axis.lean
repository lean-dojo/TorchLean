/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.AxisTuple
public import TorchLean.Tensor.Rep.Pointwise -- shake: keep
public import Mathlib.Data.Fintype.BigOperators -- shake: keep

/-!
# Axis-selection fibers

A `Fiber` is the preimage of one point under a coordinate map. When a duplicate-free tuple of
source axes is selected out of a duplicate-free target tuple, each fiber is equivalent to the
tuples over the remaining target axes, so its cardinality is the product of those axis lengths.

The fiber equivalence is noncomputable and serves proofs. The computable `AxisTuple.merge`
rebuilds a target tuple from its source coordinates and its remaining coordinates, so reduction and
einsum lowering reconstruct input coordinates without running the equivalence to enumerate a fiber.
`AxisTuple.coordFiberEquiv` carries the fiber equivalence through equivalences of tensor coordinates
with axis tuples, which gives the reduction and contraction fibers.
-/

@[expose] public section

namespace TorchLean.Einops

universe u v

/-- The preimage of `y` under `f`, as a subtype of the domain. It is finite whenever the domain
is. -/
abbrev Fiber {ι : Type u} {κ : Type v} (f : ι → κ) (y : κ) :=
  {x : ι // f x = y}

namespace AxisTuple

open Shape (Coord)

/--
For duplicate-free `source` and `target` with every source axis in `target`, the fiber of
`select` over a source coordinate is equivalent to the tuples over the target axes absent from
`source`.

Repeat multiplicities, reduction fibers and contraction fibers all rest on this equivalence. An
empty complement gives a singleton fiber, and a complement containing a zero-length axis gives an
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
  -- Indexing by axis membership lets mathlib split the coordinate function
  -- into retained and free axes. The projection proofs reconnect that split
  -- to the ordered, first-occurrence lookup used by `select`.
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
A fiber of tuple selection has as many elements as the product of the lengths of the target axes
absent from the source.

The empty product is one, and any such zero-length axis makes the fiber empty.
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

/-- Every target axis occurs in `source` or among the target axes absent from `source`. -/
theorem mem_append_filter {ι : Type u} [BEq ι] [LawfulBEq ι]
    {source target : List ι} :
    ∀ axis, axis ∈ target →
      axis ∈ source ++ target.filter fun axis => !source.contains axis := by
  intro axis hTarget
  by_cases hSource : axis ∈ source
  · exact List.mem_append_left _ hSource
  · apply List.mem_append_right
    simp [hTarget, hSource]

/--
Rebuild a target tuple from its coordinates on `source` and its coordinates on the target axes
absent from `source`.

The source coordinates come first only in the intermediate appended tuple, and selection restores
the target axis order. Reductions and contractions recover a complete input coordinate this way
from an output coordinate and one assignment of the removed axes.
-/
@[inline] def merge {ι : Type u} [BEq ι] [LawfulBEq ι] {length : ι → Nat}
    {target : List ι} (source : List ι)
    (sourceCoordinate : AxisTuple length source)
    (restCoordinate :
      AxisTuple length (target.filter fun axis => !source.contains axis)) :
    AxisTuple length target :=
  select mem_append_filter (append source sourceCoordinate restCoordinate)

/--
For a duplicate-free `source`, selecting the source axes from a merged tuple recovers the source
coordinates.
-/
theorem select_merge_left {ι : Type u} [BEq ι] [LawfulBEq ι]
    {length : ι → Nat} {source target : List ι}
    (hSource : source.Nodup)
    (hSourceTarget : ∀ axis, axis ∈ source → axis ∈ target)
    (sourceCoordinate : AxisTuple length source)
    (restCoordinate :
      AxisTuple length (target.filter fun axis => !source.contains axis)) :
    select hSourceTarget (merge source sourceCoordinate restCoordinate) =
      sourceCoordinate := by
  rw [merge, select_comp]
  exact select_append_left hSource sourceCoordinate restCoordinate

/--
For a duplicate-free `target`, selecting the target axes absent from `source` from a merged tuple
recovers their supplied coordinates.
-/
theorem select_merge_right {ι : Type u} [BEq ι] [LawfulBEq ι]
    {length : ι → Nat} {source target : List ι}
    (hTarget : target.Nodup)
    (sourceCoordinate : AxisTuple length source)
    (restCoordinate :
      AxisTuple length (target.filter fun axis => !source.contains axis)) :
    select (fun _ hAxis => (List.mem_filter.mp hAxis).1)
        (merge source sourceCoordinate restCoordinate) =
      restCoordinate := by
  rw [merge, select_comp]
  exact select_append_right (hTarget.filter _)
    (fun axis hSource hRest => by
      have hAbsent : axis ∉ source := by
        simpa using (List.mem_filter.mp hRest).2
      exact hAbsent hSource)
    sourceCoordinate restCoordinate

/--
Reading `select` through equivalences of two coordinate types with axis tuples keeps its fibers:
the fiber over a source coordinate is equivalent to the tuples over the target axes absent from
`source`.

Reduction fibers of checked transforms and contraction fibers of checked einsums both have this
form.
-/
noncomputable def coordFiberEquiv {ι : Type u} [BEq ι] [LawfulBEq ι] {α β : Type*}
    {length : ι → Nat} {source target : List ι}
    (sourceEquiv : β ≃ AxisTuple length source)
    (targetEquiv : α ≃ AxisTuple length target)
    (hSource : source.Nodup) (hTarget : target.Nodup)
    (hSourceTarget : ∀ axis, axis ∈ source → axis ∈ target)
    (sourceCoordinate : β) :
    Fiber
        (fun targetCoordinate =>
          sourceEquiv.symm (select hSourceTarget (targetEquiv targetCoordinate)))
        sourceCoordinate ≃
      AxisTuple length (target.filter fun axis => !source.contains axis) :=
  let selectFiber :
      Fiber
          (fun targetCoordinate =>
            sourceEquiv.symm (select hSourceTarget (targetEquiv targetCoordinate)))
          sourceCoordinate ≃
        Fiber (select (length := length) hSourceTarget) (sourceEquiv sourceCoordinate) :=
    targetEquiv.subtypeEquiv fun _ => sourceEquiv.symm_apply_eq
  selectFiber.trans <|
    selectFiberEquiv hSource hTarget hSourceTarget (sourceEquiv sourceCoordinate)

end AxisTuple

end TorchLean.Einops
