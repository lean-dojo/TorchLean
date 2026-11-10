/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import TorchLean.Tensor.Einops.Elab.Einsum
public meta import TorchLean.Tensor.Einops.Elab.Pack
public meta import TorchLean.Tensor.Literal
public meta import TorchLean.Tensor.Einops.Elab.Transform
public meta import TorchLean.Tensor.Tactic.Einops.Report
public import TorchLean.Tensor.Einops.Elab.Syntax
public import TorchLean.Tensor.Literal
public import TorchLean.Tensor.Einops.Lowering.Pack.Index
public import TorchLean.Tensor.Einops.Lowering.Pack
public import TorchLean.Tensor.Einops.Lowering.TransformFusion
public import TorchLean.Tensor.Native.Construction
public import TorchLean.Tensor.Native.Transpose
public import TorchLean.Tensor.Operations
public import TorchLean.Tensor.Tactic.Init

/-!
# Tensor operations through the C kernels

The packed `Float` arithmetic of `csrc/cpu/torchlean_tensor.c`, the tagged entry point
`torchlean_float_array_binary` and the fused `UInt8` and `Float` additions, gives the bits of
the host's scalar IEEE operations, signed zeros, infinities and NaNs included. `rearrange` keeps
every element through the tiled transposes `torchlean_float_array_transpose2d` and
`torchlean_array_transpose2d`, also on empty shapes with `2 ^ 60` rows and after 1000 square
transposes, and a leading `expand` copies its source with `torchlean_float_array_append_slice`.
-/

@[expose] public section

namespace TorchLean.Test.Tensor

open TorchLean.Tensor

/-- Fail with `label` unless `condition` holds. -/
def expect (label : String) (condition : Bool) : IO Unit := do
  unless condition do
    throw <| IO.userError s!"tensor operation check failed: {label}"

/-- The `Float` matrix `[[1, 2], [3, 4]]`, transposed by `checkPackedTransposes`. -/
def floatMatrix : Tensor Float [2, 2] :=
  [[1.0, 2.0], [3.0, 4.0]]

/-- Packed native loops must retain scalar IEEE behavior, including signed zeros and NaNs. -/
def checkNativeArithmetic : IO Unit := do
  let positiveZero := Float.ofBits 0x0000000000000000
  let negativeZero := Float.ofBits 0x8000000000000000
  let positiveInfinity := Float.ofBits 0x7ff0000000000000
  let negativeInfinity := Float.ofBits 0xfff0000000000000
  let quietNaN := Float.ofBits 0x7ff8000000000042
  let specialLeft : Tensor Float [6] :=
    [positiveZero, negativeZero, positiveInfinity, negativeInfinity,
      quietNaN, 1.5]
  let specialRight : Tensor Float [6] :=
    [negativeZero, negativeZero, 1.0, positiveInfinity, 2.0, -0.5]
  let expectedSpecialBits :=
    #[positiveZero + negativeZero, negativeZero + negativeZero,
      positiveInfinity + 1.0, negativeInfinity + positiveInfinity,
      quietNaN + 2.0, 1.5 + -0.5].map Float.toBits
  let actualSpecialBits :=
    (Tensor.to (Tensor.add specialLeft specialRight : Tensor Float [6]) (Array Float)).map
      Float.toBits
  expect "native Float addition preserves scalar IEEE bit behavior"
    (actualSpecialBits == expectedSpecialBits)
  expect "Float tensor `+` on one element type has the scalar IEEE bits"
    ((Tensor.to (specialLeft + specialRight) (Array Float)).map Float.toBits ==
      expectedSpecialBits)

  let expectedSubBits :=
    #[positiveZero - negativeZero, negativeZero - negativeZero,
      positiveInfinity - 1.0, negativeInfinity - positiveInfinity,
      quietNaN - 2.0, 1.5 - -0.5].map Float.toBits
  let actualSubBits :=
    (Tensor.to (Tensor.sub specialLeft specialRight : Tensor Float [6]) (Array Float)).map
      Float.toBits
  expect "native Float subtraction preserves scalar IEEE bit behavior"
    (actualSubBits == expectedSubBits)
  expect "Float tensor `-` on one element type has the scalar IEEE bits"
    ((Tensor.to (specialLeft - specialRight) (Array Float)).map Float.toBits ==
      expectedSubBits)
  let expectedMulBits :=
    #[positiveZero * negativeZero, negativeZero * negativeZero,
      positiveInfinity * 1.0, negativeInfinity * positiveInfinity,
      quietNaN * 2.0, 1.5 * -0.5].map Float.toBits
  let actualMulBits :=
    (Tensor.to (Tensor.mul specialLeft specialRight : Tensor Float [6]) (Array Float)).map
      Float.toBits
  expect "native Float multiplication preserves scalar IEEE bit behavior"
    (actualMulBits == expectedMulBits)
  expect "Float tensor `*` on one element type has the scalar IEEE bits"
    ((Tensor.to (specialLeft * specialRight) (Array Float)).map Float.toBits ==
      expectedMulBits)
  let expectedDivBits :=
    #[positiveZero / negativeZero, negativeZero / negativeZero,
      positiveInfinity / 1.0, negativeInfinity / positiveInfinity,
      quietNaN / 2.0, 1.5 / -0.5].map Float.toBits
  let actualDivBits :=
    (Tensor.to (Tensor.div specialLeft specialRight : Tensor Float [6]) (Array Float)).map
      Float.toBits
  expect "native Float division preserves scalar IEEE bit behavior"
    (actualDivBits == expectedDivBits)
  expect "Float tensor `/` on one element type has the scalar IEEE bits"
    ((Tensor.to (specialLeft / specialRight) (Array Float)).map Float.toBits ==
      expectedDivBits)

  let mixedBytes : Tensor UInt8 [6] := [0, 1, 2, 3, 4, 255]
  let mixedFloats : Tensor Float [6] :=
    [negativeZero, positiveInfinity, negativeInfinity, quietNaN, -4.0, 0.5]
  let expectedByteFloatBits :=
    #[(0 : UInt8).toFloat + negativeZero,
      (1 : UInt8).toFloat + positiveInfinity,
      (2 : UInt8).toFloat + negativeInfinity,
      (3 : UInt8).toFloat + quietNaN,
      (4 : UInt8).toFloat + -4.0,
      (255 : UInt8).toFloat + 0.5].map Float.toBits
  let actualByteFloatBits :=
    (Tensor.to (mixedBytes + mixedFloats) (Array Float)).map Float.toBits
  expect "native UInt8 + Float preserves promoted scalar IEEE bit behavior"
    (actualByteFloatBits == expectedByteFloatBits)
  let expectedFloatByteBits :=
    #[negativeZero + (0 : UInt8).toFloat,
      positiveInfinity + (1 : UInt8).toFloat,
      negativeInfinity + (2 : UInt8).toFloat,
      quietNaN + (3 : UInt8).toFloat,
      -4.0 + (4 : UInt8).toFloat,
      0.5 + (255 : UInt8).toFloat].map Float.toBits
  let actualFloatByteBits :=
    (Tensor.to (mixedFloats + mixedBytes) (Array Float)).map Float.toBits
  expect "native Float + UInt8 preserves promoted scalar IEEE bit behavior"
    (actualFloatByteBits == expectedFloatByteBits)

/-- `rearrange` transposes square, rectangular and empty `Float` matrices, rectangular `UInt8` and
`Float32` ones, and `Float` and `Nat` matrices with `2 ^ 60` rows and no column. -/
def checkPackedTransposes : IO Unit := do
  let transposed : Tensor Float [2, 2] :=
    rearrange floatMatrix "row column -> column row"
  expect "rearrange materializes row-major transpose"
    (transposed.to (Array Float) == #[1.0, 3.0, 2.0, 4.0])

  let rectangular : Tensor Float [2, 3] :=
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
  let rectangularTranspose : Tensor Float [3, 2] :=
    rearrange rectangular "row column -> column row"
  expect "packed rearrange transposes rectangular matrices"
    (rectangularTranspose.to (Array Float) ==
      #[1.0, 4.0, 2.0, 5.0, 3.0, 6.0])

  let emptyRows : Tensor Float [0, 3] :=
    Tensor.generateFlat [0, 3] fun _ => 0.0
  let emptyRowsTranspose : Tensor Float [3, 0] :=
    rearrange emptyRows "row column -> column row"
  expect "packed rearrange preserves empty rectangular tensors"
    (emptyRowsTranspose.to (Array Float)).isEmpty

  -- A zero column count must not traverse the potentially enormous row axis.
  let emptyColumns : Tensor Float [2 ^ 60, 0] :=
    Tensor.generateFlat [2 ^ 60, 0] fun _ => 0.0
  let emptyColumnsTranspose : Tensor Float [0, 2 ^ 60] :=
    rearrange emptyColumns "row column -> column row"
  expect "packed transpose skips enormous empty row axes"
    (emptyColumnsTranspose.to (Array Float)).isEmpty
  let emptyBoxedColumns : Tensor Nat [2 ^ 60, 0] :=
    Tensor.generateFlat [2 ^ 60, 0] fun _ => 0
  let emptyBoxedTranspose : Tensor Nat [0, 2 ^ 60] :=
    rearrange emptyBoxedColumns "row column -> column row"
  expect "boxed transpose skips enormous empty row axes"
    (emptyBoxedTranspose.to (Array Nat)).isEmpty

  let bytesMatrix : Tensor UInt8 [2, 3] :=
    [[1, 2, 3], [4, 5, 6]]
  let bytesTranspose : Tensor UInt8 [3, 2] :=
    rearrange bytesMatrix "row column -> column row"
  expect "generic packed-byte rearrange remains correct"
    (bytesTranspose.to (Array UInt8) == #[1, 4, 2, 5, 3, 6])

  let binary32Input : Tensor Float32 [2, 3] :=
    Tensor.generateFlat [2, 3] fun index =>
      (Float.ofNat (index + 1)).toFloat32
  let binary32Transposed : Tensor Float32 [3, 2] :=
    rearrange binary32Input "row column -> column row"
  expect "Float32 rearrange remains correct"
    ((binary32Transposed.to (Array Float32)).map (Float32.toFloat ·) ==
      #[1.0, 4.0, 2.0, 5.0, 3.0, 6.0])

/-- `rearrange` keeps every entry of boxed `Rat` and `Complex Float` matrices through one transpose
and 1000 transposes of a `16 × 16` matrix, and of a `Complex Float` matrix through a round trip. -/
def checkBoxedTransposes : IO Unit := do
  let rationalMatrix : Tensor Rat [2, 3] :=
    [[1, 2, 3], [4, 5, 6]]
  let rationalTranspose : Tensor Rat [3, 2] :=
    rearrange rationalMatrix "row column -> column row"
  expect "exact rational rearrange remains correct"
    (rationalTranspose.to (Array Rat) == #[1, 4, 2, 5, 3, 6])

  let complexRectangular : Tensor (Complex Float) [2, 3] :=
    Tensor.generateFlat [2, 3] fun index =>
      { re := Float.ofNat (index + 1), im := Float.ofNat (index + 7) }
  let complexTranspose : Tensor (Complex Float) [3, 2] :=
    rearrange complexRectangular "row column -> column row"
  expect "boxed complex rearrange remains correct"
    (complexTranspose.to (Array (Complex Float)) ==
      #[{ re := 1.0, im := 7.0 }, { re := 4.0, im := 10.0 },
        { re := 2.0, im := 8.0 }, { re := 5.0, im := 11.0 },
        { re := 3.0, im := 9.0 }, { re := 6.0, im := 12.0 }])

  let complexRoundTrip : Tensor (Complex Float) [2, 3] :=
    rearrange
      (rearrange complexRectangular "row column -> column row")
      "column row -> row column"
  expect "owned rectangular complex transposes transfer every element safely"
    (complexRoundTrip.to (Array (Complex Float)) ==
      complexRectangular.to (Array (Complex Float)))

  let mut complexSquare : Tensor (Complex Float) [16, 16] :=
    Tensor.generateFlat [16, 16] fun index =>
      { re := Float.ofNat index, im := Float.ofNat (index + 1) }
  for _ in [:1000] do
    complexSquare :=
      rearrange complexSquare "row column -> column row"
  expect "repeated owned square complex transposes preserve all values"
    (complexSquare.to (Array (Complex Float)) ==
      Array.ofFn (n := 16 * 16) fun index =>
        { re := Float.ofNat index, im := Float.ofNat (index + 1) })

  let mut rationalSquare : Tensor Rat [16, 16] :=
    Tensor.generateFlat [16, 16] fun index =>
      Rat.ofInt (Int.ofNat (index % 31) - 15)
  for _ in [:1000] do
    rationalSquare :=
      rearrange rationalSquare "row column -> column row"
  expect "repeated owned square rational transposes preserve exact values"
    (rationalSquare.to (Array Rat) ==
      Array.ofFn (n := 16 * 16) fun index =>
        Rat.ofInt (Int.ofNat (index % 31) - 15))

/-- `expand` repeats a `Float` vector along one or two leading axes and along the grouped axis
`(batch row)`, and a `UInt8` vector along one leading axis. -/
def checkLeadingRepeat : IO Unit := do
  let vector : Tensor Float [2] := [1.0, 2.0]
  let repeated : Tensor Float [3, 2] :=
    expand vector "column -> row column" with row := 3
  expect "repeat materializes every requested row"
    (repeated.to (Array Float) == #[1.0, 2.0, 1.0, 2.0, 1.0, 2.0])

  let repeatedLeadingAxes : Tensor Float [2, 3, 2] :=
    expand vector "column -> batch row column"
      with batch := 2, row := 3
  expect "repeat copies a source buffer across multiple leading axes"
    (repeatedLeadingAxes.to (Array Float) ==
      #[1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
        1.0, 2.0, 1.0, 2.0, 1.0, 2.0])

  let repeatedComposite : Tensor Float [6, 2] :=
    expand vector "column -> (batch row) column"
      with batch := 2, row := 3
  expect "repeat preserves contiguous copies through composite output axes"
    (repeatedComposite.to (Array Float) ==
      #[1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
        1.0, 2.0, 1.0, 2.0, 1.0, 2.0])

  let repeatedBytes : Tensor UInt8 [3, 2] :=
    expand ([1, 2] : Tensor UInt8 [2])
      "column -> row column" with row := 3
  expect "leading repeat uses packed byte storage"
    (repeatedBytes.to (Array UInt8) == #[1, 2, 1, 2, 1, 2])

/-- The packed arithmetic, the transposes and the leading repeat. -/
def run : IO Unit := do
  checkNativeArithmetic
  checkPackedTransposes
  checkBoxedTransposes
  checkLeadingRepeat
  IO.println "  public tensor operations: passed"

end TorchLean.Test.Tensor
