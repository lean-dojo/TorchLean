/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.FlatBox.Flatten
public import TorchLean.Tensor.Movement
public import TorchLean.Tensor.Movement.SpecialForms
public import TorchLean.Tensor.Rep.Segment
public import TorchLean.Tensor.Rep.Traversal.Sequence
public import TorchLean.Tensor.ShapeChange.Flatten

/-!
# Concatenation along an axis and the flat concat layout

Three facts connect shaped concatenation with the CROWN concat layout.

- `cat_eq_concatenateAxes`: concatenating along `dim` (`Tensor.cat`) is the concatenation of the
  family along the axis after the axes before `dim` (`Rep.concatenateAxes`), the form the layout
  reads.
- `ConcatLayout.flatten_concatenateAxes`: flattening that concatenation gives the layout's flat
  concatenation `ConcatLayout.concat`.
- `concatFlatValues?_flatten`: the checked flat concatenation of a layout accepts every shaped
  family with that layout and returns its flattened concatenation.
-/

@[expose] public section

namespace TorchLean.CROWN.Graph

open _root_.TorchLean.Tensor

/-- Suppose `xs i` has extent `sizes i` along axis `dim` and the extents of `s` elsewhere, the
extents `sizes` are listed by `lengths`, and replacing axis `dim` of `s` by any `n` gives
`leading ++ n :: trailing` with `leading` of length `dim`. Then `Tensor.cat dim sizes xs` is, as a
`SomeTensor`, the concatenation `Rep.concatenateAxes leading trailing lengths` of the family along
the axis after `leading`. -/
theorem cat_eq_concatenateAxes {α : Type} [Storage α] {s : Shape} (dim : Fin s.rank) {k : Nat}
    (sizes : Fin k → Nat) (xs : (i : Fin k) → Tensor α (s.set dim (sizes i)))
    (out : Shape) (hout : out = s.set dim (∑ i, sizes i)) (leading trailing : Shape)
    (lengths : List Nat) (hk : k = lengths.length)
    (hsizes : ∀ i, sizes i = lengths.get (i.cast hk))
    (hs : ∀ n, s.set dim n = leading ++ n :: trailing) (hlead : leading.length = dim) :
    (⟨out, Tensor.cat dim sizes xs out hout⟩ : SomeTensor α) =
      ⟨_, Tensor.Rep.concatenateAxes leading trailing lengths fun i =>
        Tensor.castShape (xs (i.cast hk.symm)) ((hs _).trans (by rw [hsizes]; rfl))⟩ := by
  have hdim : dim.val < s.length := dim.isLt.trans_eq s.rank_eq_length
  obtain ⟨rfl, rfl⟩ : leading = s.take dim ∧ trailing = s.drop (dim + 1) := by
    obtain ⟨h₁, h₂⟩ := List.append_inj ((hs 0).symm.trans (List.set_eq_take_cons_drop 0 hdim))
      (by simp [hlead, hdim.le])
    exact ⟨h₁, (List.cons.inj h₂).2⟩
  obtain rfl : lengths = List.ofFn sizes :=
    List.ext_get (by simp [hk]) fun n h₁ h₂ => by rw [List.get_ofFn, hsizes]; rfl
  rw [Tensor.cat]
  exact SomeTensor.ofTensor_castShape _ _

namespace ConcatLayout

/-- Flattening shaped concatenation gives exactly the CROWN layout's flat concatenation. -/
theorem flatten_concatenateAxes {α : Type} [Storage α] (layout : ConcatLayout)
    (values : (parent : Fin layout.lengths.length) → Tensor α (layout.parentShape parent)) :
    Tensor.reshape
        (Tensor.Rep.concatenateAxes layout.leading layout.trailing layout.lengths
          values) [_] (Shape.size_vector _).symm =
      layout.concat (fun parent => Tensor.reshape (values parent) [_]
          (Shape.size_vector _).symm) := by
  apply Tensor.ext_vector
  intro index
  rw [concat]
  erw [Tensor.getScalar_ofFn]
  rw [Tensor.getScalar_reshape_equivFin, Tensor.getScalar_reshape_equivFin]
  simp only [Tensor.Rep.concatenateAxes, Tensor.Rep.get_ofFn]
  let parentEquiv (parent : Fin layout.lengths.length) :=
    Shape.Coord.equivFin (layout.parentShape parent)
  let source :=
    (Shape.Coord.concatEquiv
      layout.leading layout.trailing layout.lengths).symm
      ((Shape.Coord.equivFin layout.outputShape).symm index)
  change values source.1 source.2 =
    values source.1 ((parentEquiv source.1).symm (parentEquiv source.1 source.2))
  exact (congrArg (values source.1)
    ((parentEquiv source.1).symm_apply_apply source.2)).symm

end ConcatLayout

/-- Checked flat concatenation accepts every shaped family with the layout's parent shapes and
returns the flattened `Rep.concatenateAxes` of the family. -/
theorem concatFlatValues?_flatten {α : Type} [Storage α] [Context α]
    (layout : ConcatLayout)
    (values : (parent : Fin layout.lengths.length) → Tensor α (layout.parentShape parent)) :
    concatFlatValues? layout
        (Array.ofFn fun parent =>
          { n := (layout.parentShape parent).size,
            v := Tensor.reshape (values parent) [_] (Shape.size_vector _).symm }) =
      some
        { n := layout.outputShape.size
          v := Tensor.reshape
              (Tensor.Rep.concatenateAxes layout.leading layout.trailing layout.lengths
              values) [_] (Shape.size_vector _).symm } := by
  have hsequence (f : Fin layout.lengths.length → FlatTensor α) :
      Tensor.sequenceFinM (fun parent => some (f parent)) = some f :=
    Tensor.sequenceFinM_pure f
  simp only [concatFlatValues?, Array.size_ofFn, beq_self_eq_true,
    ↓reduceIte, Array.getElem?_ofFn, Fin.isLt]
  erw [hsequence]
  simp [← ConcatLayout.flatten_concatenateAxes]
  rfl

end TorchLean.CROWN.Graph
