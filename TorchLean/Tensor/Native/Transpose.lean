/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Native.Slice -- shake: keep

/-!
# Certified native two-dimensional transpose

This module holds the native lowering of the two-axis `rearrange` swap from `[rows, columns]` to
`[columns, rows]`. `transpose2D` runs the storage's own `transpose2D`: packed `FloatArray`
and ordinary `Array` tensors call tiled C loops, and every other storage gathers each output entry
from its source position. `transpose2D_correct` proves the result is the flat pullback along
`transpose2DIndex`.

The proofs concern the Lean models in `Storage`. The C loops are a foreign-code boundary listed in
`docs/TRUST_BOUNDARIES.md`.
-/

@[expose] public section

namespace TorchLean.Native

open Tensor (Rep)

universe u

/-- Source index selected by a row-major two-dimensional transpose. -/
def transpose2DIndex (rows columns : Nat) :
    Fin (Shape.size [columns, rows]) →
      Fin (Shape.size [rows, columns]) := fun outputIndex =>
  ⟨outputIndex.val / rows + columns * (outputIndex.val % rows), by
    simpa [Shape.size] using
      transpose2DIndexValue_lt rows columns outputIndex.val
        (by simpa [Shape.size] using outputIndex.isLt)⟩

/--
A tensor whose buffer observes as the transpose model of its source's array is
the flat transpose pullback of that source.
-/
theorem mk_eq_pullFlat_transpose2D
    {α : Type u} [storage : Storage α] (rows columns : Nat)
    (source : Rep α [rows, columns]) (buffer : storage.Buffer)
    (hSize : storage.size buffer = Shape.size [columns, rows])
    (hData : storage.toArray buffer =
      arrayBufferTranspose2D source.data rows columns
        (by simp)) :
    Rep.mk buffer hSize = Rep.pullFlat (transpose2DIndex rows columns) source := by
  unfold Rep.pullFlat
  apply Rep.mk_eq_ofFlatFn
  rw [hData]
  apply Array.ext
  · simp [arrayBufferTranspose2D]
  · intro index _ hIndex
    simp only [arrayBufferTranspose2D, Array.getElem_ofFn]
    exact Rep.data_getFlat source
      (transpose2DIndex rows columns ⟨index, by simpa using hIndex⟩)

/--
Transpose a rank-two tensor with its storage's `transpose2D`.

`bound` is the output size as a platform-native loop bound, as `hBound` states.
-/
@[inline] def transpose2D
    {α : Type u} [storage : Storage α] (rows columns : Nat) (bound : USize)
    (hBound : bound.toNat = columns * rows) (source : Rep α [rows, columns]) :
    Rep α [columns, rows] :=
  Rep.mk
    (storage.transpose2D source.buffer rows columns bound hBound
      (by simpa [Shape.size] using source.size_eq))
    (by
      rw [← storage.toArray_size, storage.toArray_transpose2D, arrayBufferTranspose2D_size]
      simp [Shape.size])

/-- The native rank-two transpose is the flat pullback along `transpose2DIndex`. -/
theorem transpose2D_correct
    {α : Type u} [storage : Storage α] (rows columns : Nat) (bound : USize)
    (hBound : bound.toNat = columns * rows) (source : Rep α [rows, columns]) :
    transpose2D rows columns bound hBound source =
      Rep.pullFlat (transpose2DIndex rows columns) source :=
  mk_eq_pullFlat_transpose2D rows columns source _ _
    (storage.toArray_transpose2D source.buffer rows columns bound hBound _)

end TorchLean.Native
