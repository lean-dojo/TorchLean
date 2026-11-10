/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.MixedRadix
public import TorchLean.Tensor.Einops.Semantics.Pack
import Mathlib.Tactic.Ring -- shake: keep
public import TorchLean.Tensor.Einops.Equivalence.Lowering -- shake: keep

/-!
# Row-major packing indices

The laws here reduce the dependent coordinate equivalence behind `pack` and `unpack` to flat
row-major arithmetic. Native lowering runs the compact formulas, and correctness is stated
against the independent coordinate semantics.
-/

@[expose] public section

namespace TorchLean

open scoped BigOperators
open Shape (Coord)

/--
The concatenated segment index is the local index plus the lengths of all
preceding segments.
-/
theorem Shape.Coord.segmentEquiv_val (lengths : List Nat)
    (coordinate :
      (segment : Fin lengths.length) × Fin (lengths.get segment)) :
    (segmentEquiv lengths coordinate).val =
      (∑ index : Fin coordinate.1,
          lengths.get (Fin.castLE coordinate.1.isLt.le index)) +
        coordinate.2.val := by
  simp [segmentEquiv, finSigmaFinEquiv_apply]

/--
Concatenating one segment into a packed axis has the expected row-major flat
index: trailing coordinates are least significant, followed by the packed
axis and then the leading coordinates.
-/
theorem Shape.Coord.linearize_concatEquiv_val
    (leadingShape trailingShape : Shape) (lengths : List Nat)
    (component : Fin lengths.length)
    (coordinate :
      Coord (leadingShape ++ lengths.get component :: trailingShape)) :
    let separated :=
      Coord.appendEquiv leadingShape
        (lengths.get component :: trailingShape) coordinate
    (Coord.linearize
      (concatEquiv
        leadingShape trailingShape lengths ⟨component, coordinate⟩)).val =
      (Coord.linearize separated.2.2).val +
        Shape.size trailingShape *
          ((∑ index : Fin component,
              lengths.get (Fin.castLE component.isLt.le index)) +
            separated.2.1.val +
            lengths.sum * (Coord.linearize separated.1).val) := by
  dsimp
  let separated :=
    Coord.appendEquiv leadingShape
      (lengths.get component :: trailingShape) coordinate
  change
    (Coord.linearize
      ((Coord.appendEquiv leadingShape
        (lengths.sum :: trailingShape)).symm
          (separated.1,
            (segmentEquiv lengths
              ⟨component, separated.2.1⟩,
              separated.2.2)))).val =
      (Coord.linearize separated.2.2).val +
        Shape.size trailingShape *
          ((∑ index : Fin component,
              lengths.get (Fin.castLE component.isLt.le index)) +
            separated.2.1.val +
            lengths.sum * (Coord.linearize separated.1).val)
  rw [Coord.linearize_appendEquiv_symm_val,
    Coord.linearize_cons_val, segmentEquiv_val]
  simp only [Shape.size_cons]
  ring

/--
The checked component-to-packed equivalence has a compact row-major formula.
-/
theorem Einops.Pack.packedCoordEquiv_linearize_val
    (checked : Einops.Pack)
    (component : Fin checked.inputShapes.length)
    (coordinate : Coord (checked.inputShapes.get component)) :
    let segment := checked.segmentEquiv component
    let segmentCoordinate :=
      (Coord.reshapeEquiv
        (checked.input_size_eq_segment component)).symm coordinate
    let separated :=
      Coord.appendEquiv checked.leadingShape
        (checked.segmentLengths.get segment :: checked.trailingShape)
        segmentCoordinate
    (Coord.linearize
      (checked.packedCoordEquiv ⟨component, coordinate⟩)).val =
      (Coord.linearize separated.2.2).val +
        Shape.size checked.trailingShape *
          ((∑ index : Fin segment,
              checked.segmentLengths.get
                (Fin.castLE segment.isLt.le index)) +
            separated.2.1.val +
            checked.packedAxisLength *
              (Coord.linearize separated.1).val) := by
  dsimp
  change
    (Coord.linearize
      (Coord.concatEquiv checked.leadingShape
        checked.trailingShape checked.segmentLengths
          ⟨checked.segmentEquiv component,
            (Coord.reshapeEquiv
              (checked.input_size_eq_segment component)).symm
                coordinate⟩)).val = _
  rw [Coord.linearize_concatEquiv_val]
  rfl

namespace Einops.Pack

open Tensor (Rep)

/-- Total packed-axis length occupied by components before `component`. -/
def componentOffset (checked : Pack)
    (component : Fin checked.inputShapes.length) : Nat :=
  let segment := checked.segmentEquiv component
  ∑ index : Fin segment,
    checked.segmentLengths.get
      (Fin.castLE segment.isLt.le index)

/--
Summing the segments before a component is the ordinary list-prefix sum.

A concrete checked plan can evaluate this prefix once during compilation, so generated code
never traverses a finite sum for each scalar.
-/
theorem componentOffset_eq_sum_take (checked : Pack)
    (component : Fin checked.inputShapes.length) :
    componentOffset checked component =
      (checked.segmentLengths.take component.val).sum := by
  let segment := checked.segmentEquiv component
  have hLength :
      (checked.segmentLengths.take segment.val).length = segment.val :=
    List.length_take_of_le segment.isLt.le
  unfold componentOffset
  calc
    _ = ∑ index : Fin (checked.segmentLengths.take segment.val).length,
        (checked.segmentLengths.take segment.val)[index.val] := by
      refine Fintype.sum_equiv (finCongr hLength.symm) _ _ ?_
      intro index
      simp only [List.get_eq_getElem, finCongr_apply_coe, List.getElem_take, Fin.val_castLE]
    _ = _ := by
      simpa only [segment, checked.segmentEquiv_val] using
        Fin.sum_univ_getElem (checked.segmentLengths.take segment.val)

/--
Compute a packed flat index directly from one component's flat index.

The trailing coordinates occupy the low-order digits, the component's star
region occupies the next digits, and the shared leading coordinates occupy
the high-order digits. `offset` is the total length of preceding components.

This is an unbounded arithmetic formula. Checked-plan and native-lowering
theorems below supply the bounds needed before reading a tensor buffer.
-/
def unpackDirect
    (starShape trailingShape : Shape)
    (packedAxisLength offset inputIndex : Nat) : Nat :=
  let trailingSize := Shape.size trailingShape
  let segmentLength := Shape.size starShape
  inputIndex % trailingSize +
    trailingSize *
      (offset +
        (inputIndex / trailingSize) % segmentLength +
        packedAxisLength *
          (inputIndex / (trailingSize * segmentLength)))

/--
Within one component row, unpack's direct mixed-radix index is a contiguous
slice of the corresponding packed row.

Native block unpack relies on this fact, with `row` selecting the shared leading coordinate and
`column` ranging over the component's whole star-and-trailing block.
-/
theorem unpackDirect_row_column
    (segmentLength trailingSize packedAxisLength offset row column : Nat)
    (hTrailing : 0 < trailingSize)
    (hColumn : column < segmentLength * trailingSize) :
    unpackDirect [segmentLength] [trailingSize]
        packedAxisLength offset
        (row * (segmentLength * trailingSize) + column) =
      row * (packedAxisLength * trailingSize) +
        offset * trailingSize + column := by
  have hInputDivTrailing :
      (row * (segmentLength * trailingSize) + column) / trailingSize =
        row * segmentLength + column / trailingSize := by
    simpa [Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm] using
      Nat.mul_add_div hTrailing (row * segmentLength) column
  have hInputModTrailing :
      (row * (segmentLength * trailingSize) + column) % trailingSize =
        column % trailingSize := by
    simpa [Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm] using
      Nat.mul_add_mod_self_left trailingSize
        (row * segmentLength) column
  have hColumnDiv : column / trailingSize < segmentLength := by
    apply Nat.div_lt_of_lt_mul
    simpa [Nat.mul_comm] using hColumn
  have hInputDivTrailingMod :
      ((row * (segmentLength * trailingSize) + column) / trailingSize) %
          segmentLength =
        column / trailingSize := by
    rw [hInputDivTrailing]
    convert Nat.add_mul_mod_self_left
      (column / trailingSize) segmentLength row using 1 <;>
      simp [Nat.mod_eq_of_lt hColumnDiv, Nat.add_comm]
  have hInputDivBlock :
      (row * (segmentLength * trailingSize) + column) /
          (trailingSize * segmentLength) =
        row := by
    apply Nat.div_eq_of_lt_le
    · calc
        row * (trailingSize * segmentLength) =
            row * (segmentLength * trailingSize) := by
          rw [Nat.mul_comm trailingSize segmentLength]
        _ ≤ row * (segmentLength * trailingSize) + column :=
          Nat.le_add_right _ _
    · have hColumn' : column < trailingSize * segmentLength := by
        simpa [Nat.mul_comm] using hColumn
      calc
        row * (segmentLength * trailingSize) + column <
            row * (trailingSize * segmentLength) +
              trailingSize * segmentLength := by
          simpa [Nat.mul_comm] using
            Nat.add_lt_add_left hColumn'
              (row * (trailingSize * segmentLength))
        _ = (row + 1) * (trailingSize * segmentLength) := by
          rw [Nat.add_mul]
          simp
  simp only [unpackDirect, Shape.size, List.prod_singleton]
  rw [hInputModTrailing, hInputDivTrailingMod, hInputDivBlock]
  have hColumnDecomposition :=
    Nat.mod_add_div column trailingSize
  conv_rhs =>
    rw [← hColumnDecomposition]
  simp only [Nat.mul_add]
  ac_rfl

/--
Compute one component-local flat index from a packed flat index.

The packed-axis position selects a segment. Subtracting that segment's
compile-time prefix recovers the local star coordinate, while the leading and
trailing coordinates retain their row-major positions.

The formula alone does not check segment membership: subtraction is truncated
natural subtraction. The inverse and buffer-bound laws require the selected
segment's lower and upper bounds explicitly.
-/
def packDirect
    (starShape trailingShape : Shape)
    (packedAxisLength offset outputIndex : Nat) : Nat :=
  let trailingSize := Shape.size trailingShape
  let segmentLength := Shape.size starShape
  outputIndex % trailingSize +
    trailingSize *
      (((outputIndex / trailingSize) % packedAxisLength - offset) +
        segmentLength *
          (outputIndex / (trailingSize * packedAxisLength)))

/--
The direct component index lies inside the selected component buffer.

The selected packed-axis interval bounds the local star coordinate, while the
packed output bound controls the leading coordinate. Applying the row-major
encoding bound twice then accounts for the trailing coordinate.
-/
theorem packDirect_lt
    (starShape trailingShape : Shape)
    (packedAxisLength offset outputIndex leadingSize : Nat)
    (hTrailing : 0 < Shape.size trailingShape)
    (hOutput :
      outputIndex <
        leadingSize * (Shape.size trailingShape * packedAxisLength))
    (hLower :
      offset ≤
        (outputIndex / Shape.size trailingShape) % packedAxisLength)
    (hUpper :
      (outputIndex / Shape.size trailingShape) % packedAxisLength <
        offset + Shape.size starShape) :
    packDirect starShape trailingShape packedAxisLength
        offset outputIndex <
      Shape.size trailingShape * Shape.size starShape * leadingSize := by
  let trailingSize := Shape.size trailingShape
  let segmentLength := Shape.size starShape
  let position := (outputIndex / trailingSize) % packedAxisLength
  let localIndex := position - offset
  let leadingIndex := outputIndex / (trailingSize * packedAxisLength)
  have encode_lt {remainder digit radix digitCount : Nat}
      (hRemainder : remainder < radix) (hDigit : digit < digitCount) :
      remainder + radix * digit < radix * digitCount := by
    calc
      remainder + radix * digit < radix + radix * digit :=
        Nat.add_lt_add_right hRemainder _
      _ = radix * (digit + 1) := by
        simp [Nat.mul_succ, Nat.add_comm]
      _ ≤ radix * digitCount :=
        Nat.mul_le_mul_left radix (Nat.succ_le_iff.mpr hDigit)
  have hLeading : leadingIndex < leadingSize := by
    dsimp only [leadingIndex, trailingSize]
    apply Nat.div_lt_of_lt_mul
    simpa [Nat.mul_comm] using hOutput
  have hLocal : localIndex < segmentLength := by
    dsimp only [localIndex, position, trailingSize, segmentLength]
    omega
  have hInner :
      localIndex + segmentLength * leadingIndex <
        segmentLength * leadingSize :=
    encode_lt hLocal hLeading
  have hRemainder : outputIndex % trailingSize < trailingSize :=
    Nat.mod_lt _ hTrailing
  have hResult := encode_lt hRemainder hInner
  simpa [packDirect, trailingSize, segmentLength,
    position, localIndex, leadingIndex, Nat.mul_assoc] using hResult

/--
The direct component index respects equality of its compile-time segment
data.

Native pack lowering specializes the shapes, packed length, and component
offset to literals while retaining the checked expression as its proof
reference.
-/
theorem packDirect_congr
    {starShape starShape' trailingShape trailingShape' : Shape}
    {packedAxisLength packedAxisLength' offset offset' outputIndex : Nat}
    (hStarShape : starShape = starShape')
    (hTrailingShape : trailingShape = trailingShape')
    (hPackedAxisLength : packedAxisLength = packedAxisLength')
    (hOffset : offset = offset') :
    packDirect starShape trailingShape
        packedAxisLength offset outputIndex =
      packDirect starShape' trailingShape'
        packedAxisLength' offset' outputIndex := by
  subst starShape'
  subst trailingShape'
  subst packedAxisLength'
  subst offset'
  rfl

/--
The direct component index is a left inverse of the verified unpack index
inside the selected packed-axis segment.

This theorem is independent of component count. Native pack lowering uses one
instance per generated segment branch.
-/
theorem unpackDirect_packDirect
    (starShape trailingShape : Shape)
    (packedAxisLength offset outputIndex : Nat)
    (hTrailing : 0 < Shape.size trailingShape)
    (hLower :
      offset ≤
        (outputIndex / Shape.size trailingShape) % packedAxisLength)
    (hUpper :
      (outputIndex / Shape.size trailingShape) % packedAxisLength <
        offset + Shape.size starShape) :
    unpackDirect starShape trailingShape packedAxisLength offset
        (packDirect starShape trailingShape
          packedAxisLength offset outputIndex) =
      outputIndex := by
  let trailingSize := Shape.size trailingShape
  let segmentLength := Shape.size starShape
  let trailing := outputIndex % trailingSize
  let position := (outputIndex / trailingSize) % packedAxisLength
  let leading := outputIndex / (trailingSize * packedAxisLength)
  let localIndex := position - offset
  have hTrailingLt : trailing < trailingSize :=
    Nat.mod_lt _ hTrailing
  have hLocalLt : localIndex < segmentLength := by
    dsimp only [localIndex, position, trailingSize, segmentLength]
    omega
  have hPosition : offset + localIndex = position := by
    dsimp only [localIndex, position, trailingSize]
    omega
  have hOutput :
      outputIndex =
        trailing +
          trailingSize * (position + packedAxisLength * leading) := by
    have hOuter :
        outputIndex / trailingSize =
          position + packedAxisLength * leading := by
      dsimp only [position, leading]
      rw [← Nat.div_div_eq_div_mul]
      exact (Nat.mod_add_div (outputIndex / trailingSize)
        packedAxisLength).symm
    calc
      outputIndex =
          outputIndex % trailingSize +
            trailingSize * (outputIndex / trailingSize) := by
        exact (Nat.mod_add_div outputIndex trailingSize).symm
      _ = trailing +
          trailingSize * (position + packedAxisLength * leading) := by
        rw [hOuter]
  have hInput :
      packDirect starShape trailingShape
          packedAxisLength offset outputIndex =
        trailing +
          trailingSize * (localIndex + segmentLength * leading) := by
    rfl
  have hInputMod :
      packDirect starShape trailingShape
          packedAxisLength offset outputIndex % trailingSize =
        trailing := by
    rw [hInput, MixedRadix.mod_encode _ _ _ hTrailingLt]
  have hInputDiv :
      packDirect starShape trailingShape
          packedAxisLength offset outputIndex / trailingSize =
        localIndex + segmentLength * leading := by
    rw [hInput, MixedRadix.div_encode _ _ _ hTrailingLt]
  have hInputInnerMod :
      (packDirect starShape trailingShape
          packedAxisLength offset outputIndex / trailingSize) %
          segmentLength =
        localIndex := by
    rw [hInputDiv, MixedRadix.mod_encode _ _ _ hLocalLt]
  have hInputInnerDiv :
      packDirect starShape trailingShape
          packedAxisLength offset outputIndex /
          (trailingSize * segmentLength) =
        leading := by
    rw [← Nat.div_div_eq_div_mul, hInputDiv,
      MixedRadix.div_encode _ _ _ hLocalLt]
  simp only [unpackDirect]
  change
    packDirect starShape trailingShape
          packedAxisLength offset outputIndex % trailingSize +
        trailingSize *
          (offset +
            packDirect starShape trailingShape
                packedAxisLength offset outputIndex /
                trailingSize % segmentLength +
            packedAxisLength *
              (packDirect starShape trailingShape
                  packedAxisLength offset outputIndex /
                  (trailingSize * segmentLength))) =
      outputIndex
  rw [hInputMod, hInputInnerMod, hInputInnerDiv, hPosition]
  exact hOutput.symm

/--
The direct unpack index respects equality of its compile-time segment data.

With it, elaboration specializes checked shapes, lengths and offsets to literals without
unfolding the arithmetic program in generated proofs.
-/
theorem unpackDirect_congr
    {starShape starShape' trailingShape trailingShape' : Shape}
    {packedAxisLength packedAxisLength' offset offset' inputIndex : Nat}
    (hStarShape : starShape = starShape')
    (hTrailingShape : trailingShape = trailingShape')
    (hPackedAxisLength : packedAxisLength = packedAxisLength')
    (hOffset : offset = offset') :
    unpackDirect starShape trailingShape
        packedAxisLength offset inputIndex =
      unpackDirect starShape' trailingShape'
        packedAxisLength' offset' inputIndex := by
  subst starShape'
  subst trailingShape'
  subst packedAxisLength'
  subst offset'
  rfl

/--
Compute the packed flat index corresponding to one component flat index.

The definition is intentionally operation-level rather than compiler-level:
it remains valid for symbolic shapes and is independently related to the
checked coordinate semantics below.
-/
def unpackIndex (checked : Pack)
    (component : Fin checked.inputShapes.length)
    (inputIndex : Fin (Shape.size (checked.inputShapes.get component))) :
    Nat :=
  let coordinate := Coord.unlinearize inputIndex
  let segment := checked.segmentEquiv component
  let segmentCoordinate :=
    (Coord.reshapeEquiv
      (checked.input_size_eq_segment component)).symm coordinate
  let separated :=
    Coord.appendEquiv checked.leadingShape
      (checked.segmentLengths.get segment :: checked.trailingShape)
      segmentCoordinate
  (Coord.linearize separated.2.2).val +
    Shape.size checked.trailingShape *
      (componentOffset checked component +
        separated.2.1.val +
        checked.packedAxisLength *
          (Coord.linearize separated.1).val)

/--
The direct mixed-radix formula agrees with the coordinate-based compact
unpack index.
-/
theorem unpackDirect_eq (checked : Pack)
    (component : Fin checked.inputShapes.length)
    (inputIndex : Fin (Shape.size (checked.inputShapes.get component))) :
    unpackDirect
        (checked.starShape component) checked.trailingShape
        checked.packedAxisLength
        (componentOffset checked component)
        inputIndex.val =
      unpackIndex checked component inputIndex := by
  dsimp only [unpackDirect, unpackIndex]
  let coordinate := Coord.unlinearize inputIndex
  let segment := checked.segmentEquiv component
  let segmentCoordinate :=
    (Coord.reshapeEquiv
      (checked.input_size_eq_segment component)).symm coordinate
  let separated :=
    Coord.appendEquiv checked.leadingShape
      (checked.segmentLengths.get segment :: checked.trailingShape)
      segmentCoordinate
  let leadingIndex := (Coord.linearize separated.1).val
  let localIndex := separated.2.1.val
  let trailingIndex := (Coord.linearize separated.2.2).val
  let trailingSize := Shape.size checked.trailingShape
  let segmentLength := checked.segmentLengths.get segment
  have hTrailing : trailingIndex < trailingSize :=
    (Coord.linearize separated.2.2).isLt
  have hLocal : localIndex < segmentLength :=
    separated.2.1.isLt
  have hSegmentLength :
      segmentLength = Shape.size (checked.starShape component) := by
    exact checked.segmentLengths_segmentEquiv component
  have hInputIndex :
      inputIndex.val =
        trailingIndex +
          trailingSize * (localIndex + segmentLength * leadingIndex) := by
    have hReshape :
        (Coord.linearize segmentCoordinate).val = inputIndex.val := by
      dsimp only [segmentCoordinate, coordinate]
      rw [← Coord.reshapeEquiv_symm]
      rw [Coord.linearize_reshapeEquiv_val]
      exact congrArg Fin.val (Coord.linearize_unlinearize inputIndex)
    have hAppend :
        (Coord.linearize segmentCoordinate).val =
          (Coord.linearize separated.2).val +
            Shape.size (segmentLength :: checked.trailingShape) *
              (Coord.linearize separated.1).val := by
      have hSeparated :
          (Coord.appendEquiv checked.leadingShape
              (segmentLength :: checked.trailingShape)).symm separated =
            segmentCoordinate := by
        dsimp only [separated]
        exact Equiv.symm_apply_apply _ segmentCoordinate
      rw [← hSeparated]
      exact Coord.linearize_appendEquiv_symm_val
        checked.leadingShape
        (segmentLength :: checked.trailingShape)
        separated.1 separated.2
    have hCons :
        (Coord.linearize separated.2).val =
          trailingIndex + trailingSize * localIndex := by
      exact Coord.linearize_cons_val separated.2.1 separated.2.2
    calc
      inputIndex.val = (Coord.linearize segmentCoordinate).val :=
        hReshape.symm
      _ = (Coord.linearize separated.2).val +
          Shape.size (segmentLength :: checked.trailingShape) *
            (Coord.linearize separated.1).val :=
        hAppend
      _ = trailingIndex +
          trailingSize * (localIndex + segmentLength * leadingIndex) := by
        rw [hCons, Shape.size_cons]
        dsimp only [leadingIndex, localIndex, trailingIndex, trailingSize]
        ring
  have hMod :
      inputIndex.val % trailingSize = trailingIndex := by
    rw [hInputIndex, MixedRadix.mod_encode _ _ _ hTrailing]
  have hDiv :
      inputIndex.val / trailingSize =
        localIndex + segmentLength * leadingIndex := by
    rw [hInputIndex, MixedRadix.div_encode _ _ _ hTrailing]
  have hInnerMod :
      (inputIndex.val / trailingSize) % segmentLength = localIndex := by
    rw [hDiv, MixedRadix.mod_encode _ _ _ hLocal]
  have hInnerDiv :
      inputIndex.val / (trailingSize * segmentLength) = leadingIndex := by
    rw [← Nat.div_div_eq_div_mul, hDiv,
      MixedRadix.div_encode _ _ _ hLocal]
  change
    inputIndex.val % trailingSize +
        trailingSize *
          (componentOffset checked component +
            inputIndex.val / trailingSize %
              Shape.size (checked.starShape component) +
            checked.packedAxisLength *
              (inputIndex.val /
                (trailingSize *
                  Shape.size (checked.starShape component)))) =
      trailingIndex +
        trailingSize *
          (componentOffset checked component +
            localIndex +
            checked.packedAxisLength * leadingIndex)
  rw [← hSegmentLength, hMod, hInnerMod, hInnerDiv]

/--
The compact unpack index is exactly the flat index selected by the independent
packed-coordinate equivalence.
-/
theorem unpackIndex_eq (checked : Pack)
    (component : Fin checked.inputShapes.length)
    (inputIndex : Fin (Shape.size (checked.inputShapes.get component))) :
    unpackIndex checked component inputIndex =
      (Coord.linearize
        (checked.packedCoordEquiv
          ⟨component, Coord.unlinearize inputIndex⟩)).val := by
  rw [checked.packedCoordEquiv_linearize_val]
  rfl

/-- The compact unpack index is always inside the packed output buffer. -/
theorem unpackIndex_lt (checked : Pack)
    (component : Fin checked.inputShapes.length)
    (inputIndex : Fin (Shape.size (checked.inputShapes.get component))) :
    unpackIndex checked component inputIndex <
      Shape.size checked.output := by
  rw [unpackIndex_eq]
  exact Fin.isLt _

/--
The compact unpack index as a bounded flat map into the packed tensor.
-/
def unpackFlat (checked : Pack)
    (component : Fin checked.inputShapes.length)
    (inputIndex : Fin (Shape.size (checked.inputShapes.get component))) :
    Fin (Shape.size checked.output) :=
  ⟨unpackIndex checked component inputIndex,
    unpackIndex_lt checked component inputIndex⟩

/--
The compact unpack map is the row-major form of the independent coordinate
embedding.
-/
theorem unpackFlat_eq (checked : Pack)
    (component : Fin checked.inputShapes.length)
    (inputIndex : Fin (Shape.size (checked.inputShapes.get component))) :
    unpackFlat checked component inputIndex =
      Coord.linearize
        (checked.packedCoordEquiv
          ⟨component, Coord.unlinearize inputIndex⟩) := by
  apply Fin.ext
  exact unpackIndex_eq checked component inputIndex

/-- Each unpacked component is the flat pullback of the packed tensor along `unpackFlat`. -/
theorem denoteUnpack_eq_pullFlat {α : Type*}
    [storage : Storage α]
    (checked : Pack)
    (packedTensor : @OutputTensor checked α storage)
    (component : Fin checked.inputShapes.length) :
    @Semantics.denoteUnpack α storage checked packedTensor component =
      @Rep.pullFlat α storage checked.output
        (checked.inputShapes.get component)
        (unpackFlat checked component) packedTensor := by
  change
    Rep.pull
        (fun inputCoordinate =>
          checked.packedCoordEquiv ⟨component, inputCoordinate⟩)
        packedTensor =
      Rep.pullFlat (unpackFlat checked component) packedTensor
  symm
  exact Rep.pullFlat_eq_pull
    (unpackFlat checked component)
    (fun inputCoordinate =>
      checked.packedCoordEquiv ⟨component, inputCoordinate⟩)
    (unpackFlat_eq checked component) packedTensor

/--
A bounded component-local index obtained from the direct inverse formula maps
back to the packed output index that selected it.

The component interval hypotheses are exactly the facts established by one
branch of native pack's generated segment dispatcher.
-/
theorem unpackFlat_eq_of_packDirect
    (checked : Pack)
    (component : Fin checked.inputShapes.length)
    (inputIndex : Fin (Shape.size (checked.inputShapes.get component)))
    (outputIndex : Fin (Shape.size checked.output))
    (hValue :
      inputIndex.val =
        packDirect
          (checked.starShape component) checked.trailingShape
          checked.packedAxisLength (componentOffset checked component)
          outputIndex.val)
    (hTrailing : 0 < Shape.size checked.trailingShape)
    (hLower :
      componentOffset checked component ≤
        (outputIndex.val / Shape.size checked.trailingShape) %
          checked.packedAxisLength)
    (hUpper :
      (outputIndex.val / Shape.size checked.trailingShape) %
          checked.packedAxisLength <
        componentOffset checked component +
          Shape.size (checked.starShape component)) :
    unpackFlat checked component inputIndex = outputIndex := by
  apply Fin.ext
  change unpackIndex checked component inputIndex = outputIndex.val
  rw [← unpackDirect_eq checked component inputIndex, hValue]
  exact unpackDirect_packDirect
    (checked.starShape component) checked.trailingShape
    checked.packedAxisLength (componentOffset checked component)
    outputIndex.val hTrailing hLower hUpper

/--
Specialize the inverse pack index through literal segment metadata.

Native lowering computes shapes, packed length, and the component prefix once
during elaboration. This theorem transports those literal facts to the
checked plan before invoking the general inverse law.
-/
theorem unpackFlat_eq_of_packDirect'
    (checked : Pack)
    (component : Fin checked.inputShapes.length)
    (inputIndex : Fin (Shape.size (checked.inputShapes.get component)))
    (outputIndex : Fin (Shape.size checked.output))
    (starShape trailingShape : Shape)
    (packedAxisLength offset : Nat)
    (hStarShape : starShape = checked.starShape component)
    (hTrailingShape : trailingShape = checked.trailingShape)
    (hPackedAxisLength : packedAxisLength = checked.packedAxisLength)
    (hOffset : offset = componentOffset checked component)
    (hValue :
      inputIndex.val =
        packDirect starShape trailingShape
          packedAxisLength offset outputIndex.val)
    (hTrailing : 0 < Shape.size trailingShape)
    (hLower :
      offset ≤
        (outputIndex.val / Shape.size trailingShape) % packedAxisLength)
    (hUpper :
      (outputIndex.val / Shape.size trailingShape) % packedAxisLength <
        offset + Shape.size starShape) :
    unpackFlat checked component inputIndex = outputIndex := by
  apply unpackFlat_eq_of_packDirect
  · exact hValue.trans <|
      packDirect_congr hStarShape hTrailingShape
        hPackedAxisLength hOffset
  · simpa only [← hTrailingShape] using hTrailing
  · simpa only [← hTrailingShape, ← hPackedAxisLength, ← hOffset]
      using hLower
  · simpa only [← hStarShape, ← hTrailingShape,
      ← hPackedAxisLength, ← hOffset] using hUpper

/--
Reading a packed output through a component index is the inverse of the
verified unpack flat map.

Native pack lowering uses this theorem after selecting one concrete segment:
it only needs to prove that the generated component index maps back to the
current packed output index.
-/
theorem getFlat_denotePack_of_eq {α : Type*}
    [storage : Storage α]
    (checked : Pack)
    (inputTensors : @InputTensors checked α storage)
    (component : Fin checked.inputShapes.length)
    (inputIndex : Fin (Shape.size (checked.inputShapes.get component)))
    (outputIndex : Fin (Shape.size checked.output))
    (hIndex : unpackFlat checked component inputIndex = outputIndex) :
    (inputTensors component).getFlat inputIndex =
      (@Semantics.denotePack α storage checked inputTensors).getFlat
        outputIndex := by
  have hRoundTrip :=
    congrArg (fun tensor => tensor.getFlat inputIndex) <|
      congrFun (Semantics.denoteUnpack_denotePack checked inputTensors)
        component
  rw [denoteUnpack_eq_pullFlat,
    Rep.getFlat_pullFlat] at hRoundTrip
  rw [hIndex] at hRoundTrip
  exact hRoundTrip.symm

end Einops.Pack

end TorchLean
