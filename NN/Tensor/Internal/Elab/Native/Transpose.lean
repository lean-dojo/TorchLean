/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Representation.Basic.Reindex
public import NN.Tensor.Internal.Elab.Native.Slice -- shake: keep

/-!
# Certified native two-dimensional transpose

This module gives the common two-axis `rearrange` permutation tiled native
implementations for packed `FloatArray` and ordinary polymorphic `Array`
storage. Their proof-visible definitions are ordinary row-major `Array.ofFn`
terms; compiled execution uses native tiled loops.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal.Elab.Impl

universe u

/-- The quotient/remainder transpose address is inside the source rectangle. -/
theorem transpose2DIndexValue_lt
    (rows columns index : Nat) (hIndex : index < columns * rows) :
    index / rows + columns * (index % rows) < rows * columns := by
  have hRows : 0 < rows := by
    have hProduct : 0 < columns * rows := Nat.zero_lt_of_lt hIndex
    exact Nat.pos_of_mul_pos_right (Nat.mul_comm columns rows ▸ hProduct)
  have hColumn : index / rows < columns :=
    Nat.div_lt_of_lt_mul (Nat.mul_comm columns rows ▸ hIndex)
  have hRow : index % rows < rows := Nat.mod_lt _ hRows
  calc
    index / rows + columns * (index % rows) <
        columns + columns * (index % rows) :=
      Nat.add_lt_add_right hColumn _
    _ = columns * (index % rows + 1) := by
      rw [Nat.mul_succ, Nat.add_comm]
    _ ≤ columns * rows :=
      Nat.mul_le_mul_left columns (Nat.succ_le_iff.mpr hRow)
    _ = rows * columns := Nat.mul_comm columns rows

/-- Source index selected by a row-major two-dimensional transpose. -/
def transpose2DIndex (rows columns : Nat) :
    Fin (columns * rows) → Fin (rows * columns) := fun outputIndex =>
  ⟨outputIndex.val / rows + columns * (outputIndex.val % rows),
    transpose2DIndexValue_lt rows columns outputIndex.val outputIndex.isLt⟩

/-- Shape-indexed form of `transpose2DIndex` used by tensor pullbacks. -/
def transpose2DShapeIndex (rows columns : Nat) :
    Fin (Shape.size [columns, rows]) →
      Fin (Shape.size [rows, columns]) := fun outputIndex =>
  ⟨outputIndex.val / rows + columns * (outputIndex.val % rows), by
    simpa [Shape.size] using
      transpose2DIndexValue_lt rows columns outputIndex.val
        (by simpa [Shape.size] using outputIndex.isLt)⟩

/--
Proof-visible model of the packed two-dimensional transpose.

The source-size certificate is erased. It supplies the bounds needed by the
ordinary finite-array model used in correctness proofs.
-/
def floatBufferTranspose2D
    (source : @& FloatArray) (rows columns : @& Nat)
    (hSize : source.data.size = rows * columns) : FloatArray :=
  ⟨Array.ofFn fun outputIndex : Fin (columns * rows) =>
    source.data[(transpose2DIndex rows columns outputIndex).val]'(by
      rw [hSize]
      exact (transpose2DIndex rows columns outputIndex).isLt)⟩

/--
Proof-visible model of the ordinary-array two-dimensional transpose.

The native implementation transfers retained object pointers directly. This
model states the same operation solely in terms of ordinary array indexing.
-/
def arrayBufferTranspose2D
    {α : Type u} (source : Array α) (rows columns : @& Nat)
    (hSize : source.size = rows * columns) : Array α :=
  Array.ofFn fun outputIndex : Fin (columns * rows) =>
    source[(transpose2DIndex rows columns outputIndex).val]'(by
      rw [hSize]
      exact (transpose2DIndex rows columns outputIndex).isLt)

/-- Native packed two-dimensional transpose, compiled to one tiled C loop. -/
@[extern "torchlean_float_array_transpose2d"]
def floatBufferTranspose2DNative
    (source : @& FloatArray) (rows columns : @& Nat)
    (hSize : source.data.size = rows * columns) : FloatArray :=
  floatBufferTranspose2D source rows columns hSize

/-- Native ordinary-array two-dimensional transpose, compiled to one tiled C loop. -/
@[extern "torchlean_array_transpose2d"]
def arrayBufferTranspose2DNative
    {α : Type u} (source : Array α) (rows columns : @& Nat)
    (hSize : source.size = rows * columns) : Array α :=
  arrayBufferTranspose2D source rows columns hSize

/--
Use the tiled native packed transpose only in generated code.
-/
@[csimp] theorem floatBufferTranspose2D_eq_native :
    @floatBufferTranspose2D = @floatBufferTranspose2DNative := rfl

/--
Use the tiled native ordinary-array transpose only in generated code.
-/
@[csimp] theorem arrayBufferTranspose2D_eq_native :
    @arrayBufferTranspose2D = @arrayBufferTranspose2DNative := rfl

/-- The packed transpose has exactly the transposed scalar count. -/
theorem floatBufferTranspose2D_size
    (source : FloatArray) (rows columns : Nat)
    (hSize : source.data.size = rows * columns) :
    (floatBufferTranspose2D source rows columns hSize).data.size =
      columns * rows := by
  exact Array.size_ofFn

/-- The ordinary-array transpose has exactly the transposed scalar count. -/
theorem arrayBufferTranspose2D_size
    {α : Type u} (source : Array α) (rows columns : Nat)
    (hSize : source.size = rows * columns) :
    (arrayBufferTranspose2D source rows columns hSize).size =
      columns * rows := by
  exact Array.size_ofFn

/--
Transpose a rank-two packed floating-point tensor without scalar callbacks.
-/
@[inline] def nativeFloatTranspose2D
    (rows columns : Nat)
    (source : @Rep Float [rows, columns] instFloatStorage) :
    @Rep Float [columns, rows] instFloatStorage :=
  let hSourceSize : source.buffer.data.size = rows * columns := by
    calc
      source.buffer.data.size =
          @Storage.size Float instFloatStorage source.buffer :=
        instFloatStorage.toArray_size source.buffer
      _ = rows * columns := by
        simpa [Shape.size] using source.size_eq
  Rep.mk
    (floatBufferTranspose2D source.buffer rows columns hSourceSize)
    (by
      calc
        @Storage.size Float instFloatStorage
            (floatBufferTranspose2D source.buffer rows columns hSourceSize) =
            (floatBufferTranspose2D source.buffer rows columns
              hSourceSize).data.size :=
          (instFloatStorage.toArray_size
            (floatBufferTranspose2D source.buffer rows columns
              hSourceSize)).symm
        _ = columns * rows :=
          floatBufferTranspose2D_size source.buffer rows columns hSourceSize
        _ = Shape.size [columns, rows] := by simp [Shape.size])

/--
Transpose a rank-two ordinary-array tensor without scalar callbacks.
-/
@[inline] def nativeArrayTranspose2D
    {α : Type u} (rows columns : Nat)
    (source : @Rep α [rows, columns] (instArrayStorage α)) :
    @Rep α [columns, rows] (instArrayStorage α) :=
  let hSourceSize : source.buffer.size = rows * columns := by
    have hSize := source.size_eq
    change source.buffer.size = Shape.size [rows, columns] at hSize
    simpa [Shape.size] using hSize
  Rep.mk
    (arrayBufferTranspose2D source.buffer rows columns hSourceSize)
    (by
      change
        (arrayBufferTranspose2D source.buffer rows columns hSourceSize).size =
          Shape.size [columns, rows]
      simpa [Shape.size] using
        arrayBufferTranspose2D_size source.buffer rows columns hSourceSize)

/-- The native rank-two transpose implements the ordinary flat pullback. -/
theorem nativeFloatTranspose2D_correct
    (rows columns : Nat)
    (source : @Rep Float [rows, columns] instFloatStorage) :
    nativeFloatTranspose2D rows columns source =
      Rep.pullFlat (transpose2DShapeIndex rows columns) source := by
  let hSourceSize : source.buffer.data.size = rows * columns := by
    calc
      source.buffer.data.size =
          @Storage.size Float instFloatStorage source.buffer :=
        instFloatStorage.toArray_size source.buffer
      _ = rows * columns := by
        simpa [Shape.size] using source.size_eq
  unfold nativeFloatTranspose2D Rep.pullFlat
  apply (@Rep.mk_eq_ofFlatFn Float instFloatStorage)
  change
    (floatBufferTranspose2D source.buffer rows columns hSourceSize).data =
      Array.ofFn fun outputIndex =>
        source.getFlat (transpose2DShapeIndex rows columns outputIndex)
  apply Array.ext
  · simp [floatBufferTranspose2D]
  · intro index hLeft hRight
    simp only [floatBufferTranspose2D, Array.getElem_ofFn] at hLeft hRight ⊢
    let outputIndex : Fin (columns * rows) :=
      ⟨index, by
        simpa [Shape.size] using hRight⟩
    let sourceIndex := transpose2DIndex rows columns outputIndex
    let shapeOutputIndex : Fin (Shape.size [columns, rows]) :=
      ⟨index, by simpa only [Array.size_ofFn] using hRight⟩
    let shapeSourceIndex :=
      transpose2DShapeIndex rows columns shapeOutputIndex
    have hSourceIndex :
        shapeSourceIndex.val = sourceIndex.val := by
      rfl
    have hShapeBuffer :
        shapeSourceIndex.val <
          @Storage.size Float instFloatStorage source.buffer := by
      rw [source.size_eq]
      exact shapeSourceIndex.isLt
    have hShapeArray :
        shapeSourceIndex.val <
          (instFloatStorage.toArray source.buffer).size := by
      rw [instFloatStorage.toArray_size]
      exact hShapeBuffer
    have hObserved :=
      instFloatStorage.toArray_get source.buffer shapeSourceIndex.val
        hShapeBuffer hShapeArray
    change
      source.buffer.data[shapeSourceIndex.val]'hShapeArray =
        @Storage.get Float instFloatStorage source.buffer
          shapeSourceIndex.val hShapeBuffer at hObserved
    change source.buffer.data[sourceIndex.val]'_ =
      source.getFlat shapeSourceIndex
    unfold Rep.getFlat
    simpa only [hSourceIndex] using hObserved

/-- The native ordinary-array transpose implements the flat pullback. -/
theorem nativeArrayTranspose2D_correct
    {α : Type u} (rows columns : Nat)
    (source : @Rep α [rows, columns] (instArrayStorage α)) :
    nativeArrayTranspose2D rows columns source =
      Rep.pullFlat (transpose2DShapeIndex rows columns) source := by
  let hSourceSize : source.buffer.size = rows * columns := by
    have hSize := source.size_eq
    change source.buffer.size = Shape.size [rows, columns] at hSize
    simpa [Shape.size] using hSize
  unfold nativeArrayTranspose2D Rep.pullFlat
  apply (@Rep.mk_eq_ofFlatFn α (instArrayStorage α))
  change
    arrayBufferTranspose2D source.buffer rows columns hSourceSize =
      Array.ofFn fun outputIndex =>
        source.getFlat (transpose2DShapeIndex rows columns outputIndex)
  apply Array.ext
  · simp [arrayBufferTranspose2D]
  · intro index hLeft hRight
    simp only [arrayBufferTranspose2D, Array.getElem_ofFn] at hLeft hRight ⊢
    rfl

end TorchLean.Tensor.Internal.Elab.Impl
