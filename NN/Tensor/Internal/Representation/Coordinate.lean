/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Representation.Shape

/-!
# Elementary-axis coordinates

`AxisTuple length axes` assigns a bounded coordinate to every axis occurrence
in a finite axis list. It is equivalent to `Coord (axes.map length)`, but its
named indexing makes permutation, projection, and replication maps direct.

The equivalence is recursive in the axis list. Its first tuple entry is the
outermost tensor coordinate, matching `Coord` and its row-major linearization.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

universe u

/-- A bounded coordinate for each axis in a finite axis list. -/
abbrev AxisTuple {ι : Type u} (length : ι → Nat) (axes : List ι) :=
  (i : Fin axes.length) → Fin (length (axes.get i))

namespace AxisTuple

/--
Row-major coordinates of an elementary shape are equivalent to a dependent
tuple indexed by the corresponding axis list.
-/
def coordEquiv {ι : Type u} (length : ι → Nat) :
    (axes : List ι) → Coord (axes.map length) ≃ AxisTuple length axes
  | [] =>
      { toFun := fun _ index => Fin.elim0 index
        invFun := fun _ => PUnit.unit
        left_inv := fun coordinate => by
          cases coordinate
          rfl
        right_inv := fun coordinate => by
          funext index
          exact Fin.elim0 index }
  | axis :: axes =>
      (Equiv.prodCongr (Equiv.refl (Fin (length axis)))
          (coordEquiv length axes)).trans <|
        Fin.consEquiv fun index =>
          Fin (length ((axis :: axes).get index))

/--
The tensor shape obtained by multiplying the elementary axes in every
top-level group. Empty groups contribute a unit dimension.
-/
def groupedShape {ι : Type u} (length : ι → Nat) (groups : List (List ι)) :
    Shape :=
  groups.map fun group => (group.map length).prod

/--
Grouping adjacent elementary axes does not change the number of tensor
entries.
-/
theorem size_groupedShape {ι : Type u} (length : ι → Nat)
    (groups : List (List ι)) :
    Shape.size (groupedShape length groups) =
      Shape.size (groups.flatten.map length) := by
  induction groups with
  | nil => rfl
  | cons group groups ih =>
      simp only [groupedShape, List.map_cons, Shape.size_cons, List.flatten_cons,
        List.map_append]
      rw [Shape.size_append]
      change
        (group.map length).prod * Shape.size (groupedShape length groups) =
          Shape.size (group.map length) * Shape.size (groups.flatten.map length)
      rw [ih, Shape.size_eq_prod (group.map length)]

/--
Coordinates of a grouped tensor shape are equivalent to one bounded
coordinate for every elementary axis.

Both sides use row-major order, so grouping and ungrouping are represented by
an equality of flat finite-index spaces rather than an arbitrary bijection.
-/
def groupedCoordEquiv {ι : Type u} (length : ι → Nat)
    (groups : List (List ι)) :
    Coord (groupedShape length groups) ≃ AxisTuple length groups.flatten :=
  (((Coord.equivFin (groupedShape length groups)).trans <|
      finCongr (size_groupedShape length groups)).trans <|
    (Coord.equivFin (groups.flatten.map length)).symm).trans <|
      coordEquiv length groups.flatten

/--
Convert coordinates from a shape certified to equal a grouped elementary-axis
shape. This keeps dependent shape transport in one reusable definition.
-/
def groupedCoordEquivOfEq {ι : Type u} {shape : Shape}
    (length : ι → Nat) (groups : List (List ι))
    (hShape : groupedShape length groups = shape) :
    Coord shape ≃ AxisTuple length groups.flatten := by
  rw [← hShape]
  exact groupedCoordEquiv length groups

/--
A shape equal to a grouped elementary-axis shape has the same number of
entries as the corresponding ungrouped shape.
-/
theorem size_eq_elementary_of_groupedShape_eq {ι : Type u} {shape : Shape}
    (length : ι → Nat) (groups : List (List ι))
    (hShape : groupedShape length groups = shape) :
    Shape.size shape = Shape.size (groups.flatten.map length) := by
  rw [← hShape]
  exact size_groupedShape length groups

/--
Join coordinates over two consecutive axis lists.

The left coordinates remain first, followed by the right coordinates. This
is the computational operation used to combine retained and reduced axes
without constructing an intermediate tensor.
-/
def append {ι : Type u} {length : ι → Nat} {right : List ι} :
    (left : List ι) →
      AxisTuple length left →
      AxisTuple length right →
      AxisTuple length (left ++ right)
  | [], _, rightCoordinate => rightCoordinate
  | _ :: left, leftCoordinate, rightCoordinate =>
      Fin.cons (leftCoordinate 0)
        (append left (fun index => leftCoordinate index.succ) rightCoordinate)

/-- Looking up a left-axis position in an appended tuple returns the left coordinate. -/
theorem append_left_val {ι : Type u} {length : ι → Nat}
    {left right : List ι}
    (leftCoordinate : AxisTuple length left)
    (rightCoordinate : AxisTuple length right)
    (index : Fin left.length) :
    ((append left leftCoordinate rightCoordinate)
      ⟨index.val, by
        simpa only [List.length_append] using
          index.isLt.trans_le (Nat.le_add_right left.length right.length)⟩).val =
      (leftCoordinate index).val := by
  induction left with
  | nil => exact Fin.elim0 index
  | cons _ left induction =>
      refine Fin.cases ?_ (fun tailIndex => ?_) index
      · rfl
      · exact
          induction (fun currentIndex => leftCoordinate currentIndex.succ)
            tailIndex

/-- Looking up a shifted right-axis position returns the right coordinate. -/
theorem append_right_val {ι : Type u} {length : ι → Nat}
    {left right : List ι}
    (leftCoordinate : AxisTuple length left)
    (rightCoordinate : AxisTuple length right)
    (index : Fin right.length) :
    ((append left leftCoordinate rightCoordinate)
      ⟨left.length + index.val, by
        simpa only [List.length_append] using
          Nat.add_lt_add_left index.isLt left.length⟩).val =
      (rightCoordinate index).val := by
  induction left with
  | nil =>
      let shiftedIndex : Fin right.length :=
        ⟨0 + index.val, by simpa only [Nat.zero_add] using index.isLt⟩
      have hShiftedIndex : shiftedIndex = index := by
        apply Fin.ext
        exact Nat.zero_add index.val
      change (rightCoordinate shiftedIndex).val = (rightCoordinate index).val
      rw [hShiftedIndex]
  | cons axis left induction =>
      let recursiveIndex : Fin (left ++ right).length :=
        ⟨left.length + index.val, by
          simpa only [List.length_append] using
            Nat.add_lt_add_left index.isLt left.length⟩
      let outerIndex : Fin ((axis :: left) ++ right).length :=
        ⟨(axis :: left).length + index.val, by
          simpa only [List.length_append] using
            Nat.add_lt_add_left index.isLt (axis :: left).length⟩
      have hOuterIndex : outerIndex = recursiveIndex.succ := by
        apply Fin.ext
        change left.length + 1 + index.val = left.length + index.val + 1
        exact Nat.add_right_comm left.length 1 index.val
      change
        ((append (axis :: left) leftCoordinate rightCoordinate)
          outerIndex).val =
          (rightCoordinate index).val
      rw [hOuterIndex]
      simp only [append]
      exact induction (fun currentIndex => leftCoordinate currentIndex.succ)

/--
Select and reorder a tuple along inclusion of one finite axis list in
another. If an axis appears more than once in the target, its first
occurrence is selected; checked einops plans rule out that ambiguity.
-/
def select {ι : Type u} [BEq ι] [LawfulBEq ι] {length : ι → Nat}
    {source target : List ι}
    (h : ∀ axis, axis ∈ source → axis ∈ target)
    (coordinate : AxisTuple length target) : AxisTuple length source :=
  fun index =>
    let axis := source.get index
    let present := h axis (List.get_mem source index)
    let targetIndex : Fin target.length :=
      ⟨target.idxOf axis, List.idxOf_lt_length_iff.mpr present⟩
    Fin.cast (congrArg length (List.idxOf_get targetIndex.isLt))
      (coordinate targetIndex)

/-- Selecting the left axes from appended coordinates recovers the left tuple. -/
theorem select_append_left {ι : Type u} [BEq ι] [LawfulBEq ι]
    {length : ι → Nat} {left right : List ι}
    (hLeft : left.Nodup)
    (leftCoordinate : AxisTuple length left)
    (rightCoordinate : AxisTuple length right) :
    select (fun _ hAxis => List.mem_append_left right hAxis)
        (append left leftCoordinate rightCoordinate) =
      leftCoordinate := by
  funext index
  apply Fin.ext
  simp only [select, Fin.val_cast]
  let selectedIndex : Fin (left ++ right).length :=
    ⟨(left ++ right).idxOf (left.get index),
      List.idxOf_lt_length_iff.mpr
        (List.mem_append_left right (List.get_mem left index))⟩
  let appendedIndex : Fin (left ++ right).length :=
    ⟨index.val, by
      simpa only [List.length_append] using
        index.isLt.trans_le (Nat.le_add_right left.length right.length)⟩
  have hSelectedIndex : selectedIndex = appendedIndex := by
    apply Fin.ext
    change (left ++ right).idxOf (left.get index) = index.val
    rw [List.idxOf_append_of_mem (List.get_mem left index)]
    exact List.get_idxOf hLeft index
  change
    ((append left leftCoordinate rightCoordinate) selectedIndex).val =
      (leftCoordinate index).val
  rw [hSelectedIndex]
  exact append_left_val leftCoordinate rightCoordinate index

/--
Selecting the right axes from appended coordinates recovers the right tuple
when the two axis lists are disjoint.
-/
theorem select_append_right {ι : Type u} [BEq ι] [LawfulBEq ι]
    {length : ι → Nat} {left right : List ι}
    (hRight : right.Nodup)
    (hDisjoint : ∀ axis, axis ∈ left → axis ∉ right)
    (leftCoordinate : AxisTuple length left)
    (rightCoordinate : AxisTuple length right) :
    select (fun _ hAxis => List.mem_append_right left hAxis)
        (append left leftCoordinate rightCoordinate) =
      rightCoordinate := by
  funext index
  apply Fin.ext
  simp only [select, Fin.val_cast]
  have hNotLeft : right.get index ∉ left := by
    intro hInLeft
    exact hDisjoint (right.get index) hInLeft (List.get_mem right index)
  let selectedIndex : Fin (left ++ right).length :=
    ⟨(left ++ right).idxOf (right.get index),
      List.idxOf_lt_length_iff.mpr
        (List.mem_append_right left (List.get_mem right index))⟩
  let appendedIndex : Fin (left ++ right).length :=
    ⟨left.length + index.val, by
      simpa only [List.length_append] using
        Nat.add_lt_add_left index.isLt left.length⟩
  have hSelectedIndex : selectedIndex = appendedIndex := by
    apply Fin.ext
    change (left ++ right).idxOf (right.get index) =
      left.length + index.val
    rw [List.idxOf_append_of_notMem hNotLeft]
    exact congrArg (left.length + ·) (List.get_idxOf hRight index)
  change
    ((append left leftCoordinate rightCoordinate) selectedIndex).val =
      (rightCoordinate index).val
  rw [hSelectedIndex]
  exact append_right_val leftCoordinate rightCoordinate index

/-- Selecting a duplicate-free tuple along its identity inclusion does nothing. -/
@[simp] theorem select_self {ι : Type u} [BEq ι] [LawfulBEq ι]
    {length : ι → Nat} {axes : List ι} (hAxes : axes.Nodup)
    (coordinate : AxisTuple length axes) :
    select (source := axes) (target := axes) (fun _ h => h) coordinate =
      coordinate := by
  funext index
  let selectedIndex : Fin axes.length :=
    ⟨axes.idxOf (axes.get index),
      List.idxOf_lt_length_iff.mpr (List.get_mem axes index)⟩
  have hIndex : selectedIndex = index := by
    apply Fin.ext
    exact List.get_idxOf hAxes index
  apply Fin.ext
  change (coordinate selectedIndex).val = (coordinate index).val
  exact congrArg (fun i => (coordinate i).val) hIndex

/--
Successive axis selections compose to direct selection from the largest
tuple.

No duplicate-free hypothesis is needed: every selection consistently uses
the first occurrence returned by `idxOf`, and the intermediate lookup
recovers the same axis before the final lookup.
-/
theorem select_comp {ι : Type u} [BEq ι] [LawfulBEq ι]
    {length : ι → Nat} {source middle target : List ι}
    (hSourceMiddle : ∀ axis, axis ∈ source → axis ∈ middle)
    (hMiddleTarget : ∀ axis, axis ∈ middle → axis ∈ target)
    (coordinate : AxisTuple length target) :
    select hSourceMiddle (select hMiddleTarget coordinate) =
      select
        (fun axis hAxis => hMiddleTarget axis (hSourceMiddle axis hAxis))
        coordinate := by
  funext sourceIndex
  let middleIndex : Fin middle.length :=
    ⟨middle.idxOf (source.get sourceIndex),
      List.idxOf_lt_length_iff.mpr
        (hSourceMiddle _ (List.get_mem source sourceIndex))⟩
  let selectedTargetIndex : Fin target.length :=
    ⟨target.idxOf (middle.get middleIndex),
      List.idxOf_lt_length_iff.mpr
        (hMiddleTarget _ (List.get_mem middle middleIndex))⟩
  let directTargetIndex : Fin target.length :=
    ⟨target.idxOf (source.get sourceIndex),
      List.idxOf_lt_length_iff.mpr
        (hMiddleTarget _
          (hSourceMiddle _ (List.get_mem source sourceIndex)))⟩
  have hMiddleAxis :
      middle.get middleIndex = source.get sourceIndex :=
    List.idxOf_get middleIndex.isLt
  have hTargetIndex : selectedTargetIndex = directTargetIndex := by
    apply Fin.ext
    change target.idxOf (middle.get middleIndex) =
      target.idxOf (source.get sourceIndex)
    rw [hMiddleAxis]
  apply Fin.ext
  change (coordinate selectedTargetIndex).val =
    (coordinate directTargetIndex).val
  exact congrArg (fun index => (coordinate index).val) hTargetIndex

/--
Selecting from `target` to `source` and then back to `target` recovers the
original tuple when `target` has no duplicate axes.

The source list need not be duplicate-free for this direction: `idxOf`
chooses one source occurrence, and the reverse selection identifies its axis
with the unique target occurrence.
-/
theorem select_select {ι : Type u} [BEq ι] [LawfulBEq ι]
    {length : ι → Nat} {source target : List ι}
    (hTarget : target.Nodup)
    (hSourceTarget : ∀ axis, axis ∈ source → axis ∈ target)
    (hTargetSource : ∀ axis, axis ∈ target → axis ∈ source)
    (coordinate : AxisTuple length target) :
    select hTargetSource (select hSourceTarget coordinate) = coordinate := by
  funext targetIndex
  let sourceIndex : Fin source.length :=
    ⟨source.idxOf (target.get targetIndex),
      List.idxOf_lt_length_iff.mpr
        (hTargetSource _ (List.get_mem target targetIndex))⟩
  let selectedTargetIndex : Fin target.length :=
    ⟨target.idxOf (source.get sourceIndex),
      List.idxOf_lt_length_iff.mpr
        (hSourceTarget _ (List.get_mem source sourceIndex))⟩
  have hSourceIndex :
      source.get sourceIndex = target.get targetIndex :=
    List.idxOf_get sourceIndex.isLt
  have hTargetIndex : selectedTargetIndex = targetIndex := by
    apply Fin.ext
    change target.idxOf (source.get sourceIndex) = targetIndex.val
    rw [hSourceIndex]
    exact List.get_idxOf hTarget targetIndex
  apply Fin.ext
  change (coordinate selectedTargetIndex).val = (coordinate targetIndex).val
  exact congrArg (fun index => (coordinate index).val) hTargetIndex

/--
Reordering between duplicate-free axis lists containing the same axes is an
equivalence of bounded coordinate tuples.
-/
def selectEquiv {ι : Type u} [BEq ι] [LawfulBEq ι]
    {length : ι → Nat} {source target : List ι}
    (hSource : source.Nodup) (hTarget : target.Nodup)
    (hSourceTarget : ∀ axis, axis ∈ source → axis ∈ target)
    (hTargetSource : ∀ axis, axis ∈ target → axis ∈ source) :
    AxisTuple length target ≃ AxisTuple length source where
  toFun := select hSourceTarget
  invFun := select hTargetSource
  left_inv := select_select hTarget hSourceTarget hTargetSource
  right_inv := select_select hSource hTargetSource hSourceTarget

end AxisTuple

end TorchLean.Tensor.Internal
