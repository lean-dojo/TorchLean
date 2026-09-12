/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Representation.Coordinate
public import NN.Tensor.Internal.Representation.Segment
import Mathlib.Tactic.Ring.RingNF
import Mathlib.Tactic.Ring -- shake: keep

/-!
# Row-major composition laws

These laws connect structural coordinate composition with the flat row-major
indices used by native lowering. They are independent of any einops operation.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

/--
The row-major values of appended named-axis coordinates are the values of the
left coordinate followed by the values of the right coordinate.
-/
theorem AxisTuple.values_append {ι : Type*} {length : ι → Nat} :
    ∀ {left right : List ι}
      (leftCoordinate : AxisTuple length left)
      (rightCoordinate : AxisTuple length right),
      List.ofFn (fun index =>
          ((AxisTuple.append left leftCoordinate rightCoordinate) index).val) =
        List.ofFn (fun index => (leftCoordinate index).val) ++
          List.ofFn (fun index => (rightCoordinate index).val) := by
  intro left right leftCoordinate rightCoordinate
  let leftValues := fun index => (leftCoordinate index).val
  let rightValues := fun index => (rightCoordinate index).val
  let castIndex :
      Fin (left.length + right.length) → Fin (left ++ right).length :=
    Fin.cast (by simp only [List.length_append])
  rw [List.ofFn_congr
    (by simp only [List.length_append] :
      (left ++ right).length = left.length + right.length)
    (fun index =>
      ((AxisTuple.append left leftCoordinate rightCoordinate) index).val)]
  change
    List.ofFn (fun index =>
      ((AxisTuple.append left leftCoordinate rightCoordinate)
        (castIndex index)).val) =
      List.ofFn leftValues ++ List.ofFn rightValues
  have hAppend :
      (fun index =>
          ((AxisTuple.append left leftCoordinate rightCoordinate)
            (castIndex index)).val) =
        Fin.append leftValues rightValues := by
    funext index
    refine Fin.addCases (motive := fun index =>
      ((AxisTuple.append left leftCoordinate rightCoordinate)
          (castIndex index)).val =
        Fin.append leftValues rightValues index) ?_ ?_ index
    · intro leftIndex
      rw [Fin.append_left]
      let appendedIndex : Fin (left ++ right).length :=
        ⟨leftIndex.val, by
          simpa only [List.length_append] using
            leftIndex.isLt.trans_le
              (Nat.le_add_right left.length right.length)⟩
      have hIndex :
          castIndex (Fin.castAdd right.length leftIndex) = appendedIndex := by
        apply Fin.ext
        rfl
      rw [hIndex]
      exact AxisTuple.append_left_val
        leftCoordinate rightCoordinate leftIndex
    · intro rightIndex
      rw [Fin.append_right]
      let appendedIndex : Fin (left ++ right).length :=
        ⟨left.length + rightIndex.val, by
          simpa only [List.length_append] using
            Nat.add_lt_add_left rightIndex.isLt left.length⟩
      have hIndex :
          castIndex (Fin.natAdd left.length rightIndex) = appendedIndex := by
        apply Fin.ext
        rfl
      rw [hIndex]
      exact AxisTuple.append_right_val
        leftCoordinate rightCoordinate rightIndex
  rw [hAppend, List.ofFn_fin_append]

/--
Appending two coordinates places the right coordinate in the low-order
row-major digits and the left coordinate in the high-order digits.
-/
theorem Coord.linearize_appendEquiv_symm_val :
    ∀ (left right : Shape)
      (leftCoordinate : Coord left)
      (rightCoordinate : Coord right),
      (Coord.linearize
        ((Coord.appendEquiv left right).symm
          (leftCoordinate, rightCoordinate))).val =
        (Coord.linearize rightCoordinate).val +
          Shape.size right * (Coord.linearize leftCoordinate).val := by
  intro left
  induction left with
  | nil =>
      intro right leftCoordinate rightCoordinate
      cases leftCoordinate
      rfl
  | cons dimension left inductionHypothesis =>
      intro right leftCoordinate rightCoordinate
      obtain ⟨headCoordinate, tailCoordinate⟩ := leftCoordinate
      change
        (Coord.linearize
          (s := dimension :: (left ++ right))
          (headCoordinate,
            (Coord.appendEquiv left right).symm
              (tailCoordinate, rightCoordinate))).val =
          (Coord.linearize rightCoordinate).val +
            Shape.size right *
              (Coord.linearize
                (s := dimension :: left)
                (headCoordinate, tailCoordinate)).val
      rw [Coord.linearize_cons_val, Coord.linearize_cons_val,
        inductionHypothesis right tailCoordinate rightCoordinate,
        Shape.size_append]
      ring

/--
Linearizing appended named-axis coordinates has the same row-major formula as
appending their ordinary coordinate representations.
-/
theorem AxisTuple.linearize_append_val {ι : Type*}
    {length : ι → Nat} :
    ∀ (left right : List ι)
      (leftCoordinate : AxisTuple length left)
      (rightCoordinate : AxisTuple length right),
      (Coord.linearize
        ((AxisTuple.coordEquiv length (left ++ right)).symm <|
          AxisTuple.append left leftCoordinate rightCoordinate)).val =
        (Coord.linearize
          ((AxisTuple.coordEquiv length right).symm rightCoordinate)).val +
          Shape.size (right.map length) *
            (Coord.linearize
              ((AxisTuple.coordEquiv length left).symm leftCoordinate)).val := by
  intro left
  induction left with
  | nil =>
      intro right leftCoordinate rightCoordinate
      have hLeftCoordinate :
          leftCoordinate = AxisTuple.coordEquiv length [] PUnit.unit := by
        apply (AxisTuple.coordEquiv length []).symm.injective
        rw [Equiv.symm_apply_apply]
        cases (AxisTuple.coordEquiv length []).symm leftCoordinate
        rfl
      rw [hLeftCoordinate]
      rfl
  | cons axis left inductionHypothesis =>
      intro right leftCoordinate rightCoordinate
      let coordinate :=
        (AxisTuple.coordEquiv length (axis :: left)).symm leftCoordinate
      have hLeftCoordinate :
          leftCoordinate =
            AxisTuple.coordEquiv length (axis :: left) coordinate := by
        exact (Equiv.apply_symm_apply _ _).symm
      rw [hLeftCoordinate]
      obtain ⟨headCoordinate, tailCoordinate⟩ := coordinate
      have hAppend :
          (AxisTuple.coordEquiv length (axis :: (left ++ right))).symm
              (AxisTuple.append (axis :: left)
                (AxisTuple.coordEquiv length (axis :: left)
                  (headCoordinate, tailCoordinate))
                rightCoordinate) =
            (headCoordinate,
              (AxisTuple.coordEquiv length (left ++ right)).symm <|
                AxisTuple.append left
                  (AxisTuple.coordEquiv length left tailCoordinate)
                  rightCoordinate) := by
        apply
          (AxisTuple.coordEquiv length
            (axis :: (left ++ right))).injective
        rw [Equiv.apply_symm_apply]
        funext index
        refine Fin.cases ?_ (fun tailIndex => ?_) index
        · rfl
        · change
            (AxisTuple.append left
              (AxisTuple.coordEquiv length left tailCoordinate)
              rightCoordinate) tailIndex =
            (AxisTuple.coordEquiv length (left ++ right)
              ((AxisTuple.coordEquiv length (left ++ right)).symm <|
                AxisTuple.append left
                  (AxisTuple.coordEquiv length left tailCoordinate)
                  rightCoordinate)) tailIndex
          rw [Equiv.apply_symm_apply]
      have hAppend' :
          (AxisTuple.coordEquiv length ((axis :: left) ++ right)).symm
              (AxisTuple.append (axis :: left)
                (AxisTuple.coordEquiv length (axis :: left)
                  (headCoordinate, tailCoordinate))
                rightCoordinate) =
            (headCoordinate,
              (AxisTuple.coordEquiv length (left ++ right)).symm <|
                AxisTuple.append left
                  (AxisTuple.coordEquiv length left tailCoordinate)
                  rightCoordinate) := by
        simpa only [List.cons_append] using hAppend
      rw [hAppend', Equiv.symm_apply_apply]
      change
        (Coord.linearize
          (s := length axis :: (left ++ right).map length)
          (headCoordinate,
            (AxisTuple.coordEquiv length (left ++ right)).symm <|
              AxisTuple.append left
                (AxisTuple.coordEquiv length left tailCoordinate)
                rightCoordinate)).val =
          (Coord.linearize
            ((AxisTuple.coordEquiv length right).symm
              rightCoordinate)).val +
            Shape.size (right.map length) *
              (Coord.linearize
                (s := length axis :: left.map length)
                (headCoordinate, tailCoordinate)).val
      rw [Coord.linearize_cons_val, Coord.linearize_cons_val,
        inductionHypothesis right
          (AxisTuple.coordEquiv length left tailCoordinate)
          rightCoordinate, List.map_append, Shape.size_append]
      rw [Equiv.symm_apply_apply]
      ring

end TorchLean.Tensor.Internal
