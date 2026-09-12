/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Representation.Basic.Core
public import NN.Tensor.Internal.Elab.Native.Loop -- shake: keep
public import NN.Tensor.Internal.Representation.Basic.Pointwise -- shake: keep
public import NN.Tensor.Internal.Representation.Basic.Reindex -- shake: keep
public import NN.Tensor.Internal.Representation.Basic.Traversal -- shake: keep
public import Mathlib.Data.List.FinRange -- shake: keep

/-!
# Certified native array slices

Contiguous tensor blocks can be copied with `Array.foldl` rather than
recomputing a source index for every scalar. These helpers retain
`Array.ofFn` as their independent semantic reference.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal.Elab.Impl

universe u

/-!
Contiguous copying reuses `TorchLean.appendArraySlice` and
`appendArraySlice_eq_append_extract` from the storage layer.
-/

/-- Flatten one row and column index into a rectangular row-major index. -/
def rectangularIndex (rowCount rowLength : Nat)
    (row : Fin rowCount) (column : Fin rowLength) :
    Fin (rowCount * rowLength) :=
  ⟨row * rowLength + column, by
    calc
      row * rowLength + column < row * rowLength + rowLength :=
        Nat.add_lt_add_left column.isLt _
      _ = (row + 1) * rowLength := by
        rw [Nat.add_mul, Nat.one_mul]
      _ ≤ rowCount * rowLength :=
        Nat.mul_le_mul_right rowLength
          (Nat.succ_le_iff.mpr row.isLt)⟩

/--
Appending fixed-width rows in finite-index order reconstructs their flattened
finite function.
-/
theorem fin_foldl_append_rows_eq_array_ofFn
    {α : Type u} (rowCount rowLength : Nat)
    (values : Fin (rowCount * rowLength) → α) :
    Fin.foldl rowCount
        (fun output row =>
          output ++ Array.ofFn fun column : Fin rowLength =>
            values (rectangularIndex rowCount rowLength row column))
        (Array.emptyWithCapacity (rowCount * rowLength)) =
      Array.ofFn values := by
  rw [Fin.foldl_eq_foldl_finRange]
  apply Array.toList_inj.mp
  calc
    (List.foldl
        (fun output row =>
          output ++ Array.ofFn fun column : Fin rowLength =>
            values (rectangularIndex rowCount rowLength row column))
        (Array.emptyWithCapacity (rowCount * rowLength))
        (List.finRange rowCount)).toList =
      List.foldl
        (fun output row =>
          output ++ List.ofFn fun column : Fin rowLength =>
            values (rectangularIndex rowCount rowLength row column))
        []
        (List.finRange rowCount) := by
          symm
          apply List.foldl_hom Array.toList
          intro output row
          simp only [Array.toList_append, Array.toList_ofFn]
    _ = (List.map
          (fun row =>
            List.ofFn fun column : Fin rowLength =>
              values (rectangularIndex rowCount rowLength row column))
          (List.finRange rowCount)).flatten := by
          rw [List.foldl_append_eq_append]
          exact List.nil_append _
    _ = List.ofFn values := by
          rw [List.ofFn_mul, List.ofFn_eq_map]
          apply congrArg List.flatten
          apply List.map_congr_left
          intro row hRow
          apply congrArg List.ofFn
          funext column
          apply congrArg values
          apply Fin.ext
          rfl
    _ = (Array.ofFn values).toList :=
      Array.toList_ofFn.symm

/--
Build an array by appending one fixed-width source slice for each row.

The executable path uses an unboxed outer counter and the runtime array slice
fold. Source range proofs remain confined to the correctness theorem.
-/
@[inline] def nativeArrayOfSlices
    {α : Type u} (rowCount rowLength sourceStride sourceOffset : Nat)
    (rowBound : USize) (hRowBound : rowBound.toNat = rowCount)
    (source : Array α) : Array α :=
  nativeFinFoldl rowCount rowBound hRowBound
    (fun output row _ =>
      let start := row.toNat * sourceStride + sourceOffset
      appendArraySlice source start (start + rowLength) output)
    (Array.emptyWithCapacity (rowCount * rowLength))

/--
The native slice builder equals `Array.ofFn` when every extracted row agrees
with the corresponding finite-function row.
-/
theorem nativeArrayOfSlices_eq_array_ofFn
    {α : Type u} (rowCount rowLength sourceStride sourceOffset : Nat)
    (rowBound : USize) (hRowBound : rowBound.toNat = rowCount)
    (source : Array α) (values : Fin (rowCount * rowLength) → α)
    (hRanges :
      ∀ row : Fin rowCount,
        row * sourceStride + sourceOffset + rowLength ≤ source.size)
    (hValues :
      ∀ (row : Fin rowCount) (column : Fin rowLength),
        source[row * sourceStride + sourceOffset + column]'(by
          have hRange := hRanges row
          have hColumn := column.isLt
          omega) =
        values (rectangularIndex rowCount rowLength row column)) :
    nativeArrayOfSlices rowCount rowLength sourceStride sourceOffset
        rowBound hRowBound source =
      Array.ofFn values := by
  rw [nativeArrayOfSlices]
  rw [nativeFinFoldl_eq_fin_foldl_of_eq rowCount rowBound hRowBound
    (fun output row _ =>
      let start := row.toNat * sourceStride + sourceOffset
      appendArraySlice source start (start + rowLength) output)
    (fun output row =>
      output ++ Array.ofFn fun column : Fin rowLength =>
        values (rectangularIndex rowCount rowLength row column))
    (Array.emptyWithCapacity (rowCount * rowLength)) (by
      intro output row hRow
      rw [appendArraySlice_eq_append_extract]
      apply congrArg (output ++ ·)
      apply Array.ext
      · rw [Array.size_extract,
          Nat.min_eq_left (hRanges ⟨row.toNat, hRow⟩),
          Array.size_ofFn]
        change
          row.toNat * sourceStride + sourceOffset + rowLength -
              (row.toNat * sourceStride + sourceOffset) =
            rowLength
        omega
      · intro column hColumnLeft hColumnRight
        simp only [Array.getElem_extract, Array.getElem_ofFn]
        exact hValues ⟨row.toNat, hRow⟩
          ⟨column, by simpa only [Array.size_ofFn] using hColumnRight⟩)]
  exact fin_foldl_append_rows_eq_array_ofFn rowCount rowLength values

/-- The native slice builder produces the requested flattened size. -/
theorem nativeArrayOfSlices_size
    {α : Type u} (rowCount rowLength sourceStride sourceOffset : Nat)
    (rowBound : USize) (hRowBound : rowBound.toNat = rowCount)
    (source : Array α) (values : Fin (rowCount * rowLength) → α)
    (hRanges :
      ∀ row : Fin rowCount,
        row * sourceStride + sourceOffset + rowLength ≤ source.size)
    (hValues :
      ∀ (row : Fin rowCount) (column : Fin rowLength),
        source[row * sourceStride + sourceOffset + column]'(by
          have hRange := hRanges row
          have hColumn := column.isLt
          omega) =
        values (rectangularIndex rowCount rowLength row column)) :
    (nativeArrayOfSlices rowCount rowLength sourceStride sourceOffset
      rowBound hRowBound source).size =
      rowCount * rowLength := by
  rw [nativeArrayOfSlices_eq_array_ofFn
    rowCount rowLength sourceStride sourceOffset rowBound hRowBound
    source values hRanges hValues]
  · exact Array.size_ofFn

/--
Build a physical buffer by appending one fixed-width source slice per row.

For `Float`, `appendSlice` traverses the unboxed `FloatArray` directly.
-/
@[inline] def nativeBufferOfSlices
    {α : Type u} [storage : Storage α]
    (rowCount rowLength sourceStride sourceOffset : Nat)
    (rowBound : USize) (hRowBound : rowBound.toNat = rowCount)
    (source : storage.Buffer) : storage.Buffer :=
  nativeFinFoldl rowCount rowBound hRowBound
    (fun output row _ =>
      let start := row.toNat * sourceStride + sourceOffset
      storage.appendSlice source start (start + rowLength) output)
    (storage.emptyWithCapacity (rowCount * rowLength))

/-- Observing a finite physical slice fold commutes with every row append. -/
private theorem toArray_finFoldl_appendSlices
    {α : Type u} [storage : Storage α]
    (rowCount rowLength sourceStride sourceOffset : Nat)
    (source initial : storage.Buffer) :
    storage.toArray
        (Fin.foldl rowCount
          (fun output row =>
            let start := row * sourceStride + sourceOffset
            storage.appendSlice source start (start + rowLength) output)
          initial) =
      Fin.foldl rowCount
        (fun output row =>
          let start := row * sourceStride + sourceOffset
          output ++ (storage.toArray source).extract
            start (start + rowLength))
        (storage.toArray initial) := by
  rw [Fin.foldl_eq_foldl_finRange, Fin.foldl_eq_foldl_finRange]
  symm
  apply List.foldl_hom storage.toArray
  intro output row
  exact (storage.toArray_appendSlice source
    (row * sourceStride + sourceOffset)
    (row * sourceStride + sourceOffset + rowLength) output).symm

/--
The physical slice builder observes as `Array.ofFn` when each copied source
row agrees with the corresponding semantic row.
-/
theorem nativeBufferOfSlices_toArray
    {α : Type u} [storage : Storage α]
    (rowCount rowLength sourceStride sourceOffset : Nat)
    (rowBound : USize) (hRowBound : rowBound.toNat = rowCount)
    (source : storage.Buffer) (values : Fin (rowCount * rowLength) → α)
    (hRanges :
      ∀ row : Fin rowCount,
        row * sourceStride + sourceOffset + rowLength ≤ storage.size source)
    (hValues :
      ∀ (row : Fin rowCount) (column : Fin rowLength),
        storage.get source
            (row * sourceStride + sourceOffset + column) (by
              have hRange := hRanges row
              have hColumn := column.isLt
              omega) =
          values (rectangularIndex rowCount rowLength row column)) :
    storage.toArray
        (nativeBufferOfSlices rowCount rowLength sourceStride sourceOffset
          rowBound hRowBound source) =
      Array.ofFn values := by
  rw [nativeBufferOfSlices]
  rw [nativeFinFoldl_eq_fin_foldl_of_eq rowCount rowBound hRowBound
    (fun output row _ =>
      storage.appendSlice source
        (row.toNat * sourceStride + sourceOffset)
        (row.toNat * sourceStride + sourceOffset + rowLength) output)
    (fun output row =>
      storage.appendSlice source
        (row * sourceStride + sourceOffset)
        (row * sourceStride + sourceOffset + rowLength) output)
    (storage.emptyWithCapacity (rowCount * rowLength)) (by
      intros
      rfl)]
  rw [toArray_finFoldl_appendSlices]
  rw [storage.toArray_emptyWithCapacity]
  have hRows :
      (fun output (row : Fin rowCount) =>
        output ++ (storage.toArray source).extract
          (row * sourceStride + sourceOffset)
          (row * sourceStride + sourceOffset + rowLength)) =
      (fun output (row : Fin rowCount) =>
        output ++ Array.ofFn fun column : Fin rowLength =>
          values (rectangularIndex rowCount rowLength row column)) := by
    funext output row
    apply congrArg (output ++ ·)
    apply Array.ext
    · rw [Array.size_extract,
        Nat.min_eq_left (by
          rw [storage.toArray_size]
          exact hRanges row),
        Array.size_ofFn]
      omega
    · intro column hColumnLeft hColumnRight
      simp only [Array.getElem_extract, Array.getElem_ofFn]
      have hBuffer :
          row * sourceStride + sourceOffset + column <
            storage.size source := by
        have hRange := hRanges row
        have hColumn : column < rowLength := by
          simpa only [Array.size_ofFn] using hColumnRight
        omega
      rw [storage.toArray_get source
        (row * sourceStride + sourceOffset + column) hBuffer]
      exact hValues row
        ⟨column, by simpa only [Array.size_ofFn] using hColumnRight⟩
  rw [hRows]
  exact fin_foldl_append_rows_eq_array_ofFn rowCount rowLength values

/-- The physical slice builder produces exactly the requested scalar count. -/
theorem nativeBufferOfSlices_size
    {α : Type u} [storage : Storage α]
    (rowCount rowLength sourceStride sourceOffset : Nat)
    (rowBound : USize) (hRowBound : rowBound.toNat = rowCount)
    (source : storage.Buffer) (values : Fin (rowCount * rowLength) → α)
    (hRanges :
      ∀ row : Fin rowCount,
        row * sourceStride + sourceOffset + rowLength ≤ storage.size source)
    (hValues :
      ∀ (row : Fin rowCount) (column : Fin rowLength),
        storage.get source
            (row * sourceStride + sourceOffset + column) (by
              have hRange := hRanges row
              have hColumn := column.isLt
              omega) =
          values (rectangularIndex rowCount rowLength row column)) :
    storage.size
        (nativeBufferOfSlices rowCount rowLength sourceStride sourceOffset
          rowBound hRowBound source) =
      rowCount * rowLength := by
  rw [← storage.toArray_size,
    nativeBufferOfSlices_toArray rowCount rowLength sourceStride
      sourceOffset rowBound hRowBound source values hRanges hValues,
    Array.size_ofFn]

/--
Build a shaped tensor by copying a fixed-width source slice for each row.

The shape equality and row certificates erase, leaving only the source array,
four static index constants, and the native outer loop.
-/
@[inline] def nativeTensorOfSlices
    {α : Type u} [storage : Storage α]
    {shape sourceShape : Shape}
    (rowCount rowLength sourceStride sourceOffset : Nat)
    (hShapeSize : rowCount * rowLength = Shape.size shape)
    (rowBound : USize) (hRowBound : rowBound.toNat = rowCount)
    (source : Rep α sourceShape)
    (values : Fin (Shape.size shape) → α)
    (hRanges :
      ∀ row : Fin rowCount,
        row * sourceStride + sourceOffset + rowLength ≤
          Shape.size sourceShape)
    (hValues :
      ∀ (row : Fin rowCount) (column : Fin rowLength),
        source.getFlat
            ⟨row * sourceStride + sourceOffset + column, by
              have hRange := hRanges row
              have hColumn := column.isLt
              omega⟩ =
          values (Fin.cast hShapeSize
            (rectangularIndex rowCount rowLength row column))) :
    Rep α shape :=
  let rectangularValues : Fin (rowCount * rowLength) → α :=
    fun index => values (Fin.cast hShapeSize index)
  let hBufferRanges :
      ∀ row : Fin rowCount,
        row * sourceStride + sourceOffset + rowLength ≤
          storage.size source.buffer :=
    fun row => by
      rw [source.size_eq]
      exact hRanges row
  let hBufferValues :
      ∀ (row : Fin rowCount) (column : Fin rowLength),
        storage.get source.buffer
            (row * sourceStride + sourceOffset + column) (by
          have hRange := hBufferRanges row
          have hColumn := column.isLt
          omega) =
        rectangularValues
          (rectangularIndex rowCount rowLength row column) :=
    fun row column => by
      exact hValues row column
  Rep.mk
    (nativeBufferOfSlices rowCount rowLength sourceStride sourceOffset
      rowBound hRowBound source.buffer)
    (by
      rw [nativeBufferOfSlices_size rowCount rowLength sourceStride
        sourceOffset rowBound hRowBound source.buffer rectangularValues
        hBufferRanges]
      · exact hShapeSize
      · exact hBufferValues)

/--
The native tensor slice builder equals its ordinary finite-function tensor.
-/
theorem nativeTensorOfSlices_correct
    {α : Type u} [storage : Storage α]
    {shape sourceShape : Shape}
    (rowCount rowLength sourceStride sourceOffset : Nat)
    (hShapeSize : rowCount * rowLength = Shape.size shape)
    (rowBound : USize) (hRowBound : rowBound.toNat = rowCount)
    (source : Rep α sourceShape)
    (values : Fin (Shape.size shape) → α)
    (hRanges :
      ∀ row : Fin rowCount,
        row * sourceStride + sourceOffset + rowLength ≤
          Shape.size sourceShape)
    (hValues :
      ∀ (row : Fin rowCount) (column : Fin rowLength),
        source.getFlat
            ⟨row * sourceStride + sourceOffset + column, by
              have hRange := hRanges row
              have hColumn := column.isLt
              omega⟩ =
          values (Fin.cast hShapeSize
            (rectangularIndex rowCount rowLength row column))) :
    nativeTensorOfSlices rowCount rowLength sourceStride sourceOffset
        hShapeSize rowBound hRowBound source values hRanges hValues =
      Rep.ofFlatFn values := by
  apply Rep.mk_eq_ofFlatFn values
  let rectangularValues : Fin (rowCount * rowLength) → α :=
    fun index => values (Fin.cast hShapeSize index)
  let hBufferRanges :
      ∀ row : Fin rowCount,
        row * sourceStride + sourceOffset + rowLength ≤
          storage.size source.buffer :=
    fun row => by
      rw [source.size_eq]
      exact hRanges row
  let hBufferValues :
      ∀ (row : Fin rowCount) (column : Fin rowLength),
        storage.get source.buffer
            (row * sourceStride + sourceOffset + column) (by
          have hRange := hBufferRanges row
          have hColumn := column.isLt
          omega) =
        rectangularValues
          (rectangularIndex rowCount rowLength row column) :=
    fun row column => by
      exact hValues row column
  rw [nativeBufferOfSlices_toArray rowCount rowLength sourceStride
    sourceOffset rowBound hRowBound source.buffer rectangularValues
    hBufferRanges hBufferValues]
  apply Array.ext
  · simpa only [Array.size_ofFn] using hShapeSize
  · intro index hRectangular hShape
    simp only [Array.getElem_ofFn, rectangularValues]
    apply congrArg values
    apply Fin.ext
    rfl

end TorchLean.Tensor.Internal.Elab.Impl
