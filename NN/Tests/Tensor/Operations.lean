/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor

/-!
# Public Tensor Operation Regression Tests

These runtime checks exercise representative static-shape operations through
the public `TorchLean.Tensor` API. They complement the semantic theorems by
ensuring that the compiled contiguous-buffer path returns the expected values.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat (Binary)

namespace NN.Tests.Tensor.Operations

open TorchLean
open TorchLean.Tensor

def floatScalarLiteral : Tensor Float [] := 1.25
def rationalScalarLiteral : Tensor Rat [] := 3

def expect (label : String) (condition : Bool) : IO Unit := do
  unless condition do
    throw <| IO.userError s!"tensor operation check failed: {label}"

def floatMatrix : Tensor Float [2, 2] :=
  [[1.0, 2.0], [3.0, 4.0]]

def floatRight : Tensor Float [2, 2] :=
  [[5.0, 6.0], [7.0, 8.0]]

/-- Two contracted axes whose stride costs favor a different traversal. -/
def orderedContraction (weights : Tensor Float [2, 2]) (values : Tensor Float [2, 2, 2]) :
    Tensor Float [2] :=
  einsum weights, values "a b, b a c -> c"

/-- A one-axis contraction large enough to select four-lane output tiling. -/
def tiledContraction (weights : Tensor Float [9]) (values : Tensor Float [9, 4]) :
    Tensor Float [4] :=
  einsum weights, values "a, a c -> c"

/-- Cancellation detects reordered contraction axes and reassociated tile accumulators. -/
def checkOrderedContraction : IO Unit := do
  -- Reordering contracted axes changes each result from 1 to 2.
  let weights : Tensor Float [2, 2] := Tensor.full [2, 2] 1.0
  let values : Tensor Float [2, 2, 2] :=
    [[[1.0e16, 1.0e16], [-1.0e16, -1.0e16]], [[1.0, 1.0], [1.0, 1.0]]]
  expect "Float einsum preserves contraction-axis order through cancellation"
    ((orderedContraction weights values).to (Array Float) == #[1.0, 1.0])
  -- Each lane folds [big, 1, -big, 1, 0, ...] to 1, without reassociation.
  let tiledWeights : Tensor Float [9] := Tensor.full [9] 1.0
  let tiledValues : Tensor Float [9, 4] :=
    Tensor.generateFlat [9, 4] fun index =>
      match index / 4 with
      | 0 => 1.0e16
      | 1 | 3 => 1.0
      | 2 => -1.0e16
      | _ => 0.0
  expect "tiled Float einsum preserves accumulation order through cancellation"
    ((tiledContraction tiledWeights tiledValues).to (Array Float) == Array.replicate 4 1.0)


/-- Empty output axes skip scalar indexing and traversal, even after a huge leading axis. -/
def emptyEinsumOutput : Tensor Nat [2^40, 0] :=
  einsum (Tensor.full [2^40, 0] (7 : Nat)) "a b -> a b"

/-- Symbolic leading extents retain the same empty-output behavior. -/
def symbolicEmptyEinsumOutput (n : Nat) : Tensor Nat [n, 0] :=
  einsum (Tensor.full [n, 0] (7 : Nat)) "a b -> a b"

/-- A zero inner contraction axis eliminates the entire contraction loop nest. -/
def emptyEinsumContraction : Tensor Float [2] :=
  einsum (Tensor.full [2, 2^40, 0] (7.0 : Float)) "a b c -> a"

/-- Cholesky reconstruction and ridge residuals exercise the shared strict array recurrences. -/
def checkFactorizationResiduals : IO Unit := do
  for diagonal in [1.0 / 0.0, -1.0 / 0.0, 0.0 / 0.0, -0.0, 0.0, 2.0] do
    let factor : Fin 1 → Fin 1 → Float := fun _ _ => diagonal
    let rhs : Fin 1 → Float := fun _ => 1.0
    let reference := Spec.triSolveUpperFn (fun i k => factor k i)
      (Spec.triSolveLowerFn factor rhs)
    let actual := Spec.cholSolveFn factor rhs
    expect "strict Cholesky solve retains exceptional full-dot behavior"
      ((actual 0).toBits == (reference 0).toBits ||
        ((actual 0).isNaN && (reference 0).isNaN))
  for n in [0, 1, 2, 3, 5] do
    let kernel : Tensor Float [n, n] := Tensor.matrix fun i j =>
      if i.val == j.val then 3.0 else 0.25
    let target : Tensor Float [n] := Tensor.ofFn fun i => (i.val + 1).toFloat
    let factor := Tensor.cholesky kernel
    let reconstructed := Tensor.matmul factor (Tensor.swapAdjacentAxes factor 0)
    let solution := Tensor.solveRidge kernel 0.1 target
    let regularized := kernel + Tensor.scale (Tensor.identity n) 0.1
    let recovered := Tensor.matvec regularized solution
    for i in List.finRange n do
      expect s!"ridge residual at {n}:{i.val}"
        (Float.abs (recovered.getScalar i - target.getScalar i) < 1e-12)
      for j in List.finRange n do
        expect s!"Cholesky reconstruction at {n}:{i.val},{j.val}"
          (Float.abs (Spec.get2 reconstructed i j - Spec.get2 kernel i j) < 1e-12)

/-- Invalid columns must not alter the active-basis policy or contaminate unused tensor slots. -/
def checkExceptionalWideQR : IO Unit := do
  let nan : Float := 0.0 / 0.0
  let firstInvalid : Tensor Float [2, 3] := [[nan, 1.0, 0.0], [0.0, 0.0, 1.0]]
  let firstFactors := Tensor.qr firstInvalid
  expect "invalid first column leaves room for later independent columns"
    (Tensor.maxAbsDiff firstFactors.q (Tensor.identity 2) == 0)
  expect "invalid first column has no active projection coefficients"
    (Tensor.maxAbsDiff firstFactors.r
      ([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] : Tensor Float [2, 3]) == 0)
  let lastInvalid : Tensor Float [2, 3] := [[1.0, 0.0, nan], [0.0, 1.0, 0.0]]
  let lastFactors := Tensor.qr lastInvalid
  expect "invalid later column preserves the completed basis"
    (Tensor.maxAbsDiff lastFactors.q (Tensor.identity 2) == 0)
  expect "active projections preserve IEEE invalid products"
    (lastFactors.r[0][2].isNaN && lastFactors.r[1][2].isNaN)

/-- Reduced QR keeps independent columns after dependent columns and handles empty axes. -/
def checkReducedQR : IO Unit := do
  checkExceptionalWideQR
  let wide : Tensor Float [2, 3] :=
    [[1.0, 2.0, 0.0],
     [0.0, 0.0, 1.0]]
  let wideFactors := Tensor.qr wide
  let wideQ : Tensor Float [2, 2] := wideFactors.q
  let wideR : Tensor Float [2, 3] := wideFactors.r
  let wideReconstructed : Tensor Float [2, 3] :=
    einsum wideQ, wideR
      "row contracted, contracted column -> row column"
  expect "wide reduced QR reconstructs after an early dependent column"
    (Tensor.maxAbsDiff wide wideReconstructed < 1e-6)
  let wideQtq : Tensor Float [2, 2] :=
    einsum wideQ, wideQ
      "contracted row, contracted column -> row column"
  expect "wide reduced QR retains a complete row-space basis"
    (Tensor.maxAbsDiff wideQtq (Tensor.identity 2) < 1e-6)

  let deficient : Tensor Float [3, 2] :=
    [[1.0, 2.0],
     [2.0, 4.0],
     [3.0, 6.0]]
  let deficientFactors := Tensor.qr deficient
  let deficientReconstructed : Tensor Float [3, 2] :=
    einsum deficientFactors.q, deficientFactors.r
      "row contracted, contracted column -> row column"
  expect "rank-deficient QR still reconstructs its input"
    (Tensor.maxAbsDiff deficient deficientReconstructed < 1e-6)
  let deficientSecondColumn : Tensor Float [3] :=
    rearrange deficientFactors.q "row column -> column row" |>.get 1
  expect "rank-deficient QR emits a zero dependent column"
    (Tensor.maxAbs deficientSecondColumn < 1e-6)

  let noRows : Tensor Float [0, 3] := Tensor.zeros [0, 3]
  let noRowsFactors := Tensor.qr noRows
  expect "zero-row QR has an empty Q buffer"
    ((Tensor.to noRowsFactors.q (Array Float)).isEmpty)
  expect "zero-row QR has an empty R buffer"
    ((Tensor.to noRowsFactors.r (Array Float)).isEmpty)

  let noColumns : Tensor Float [3, 0] := Tensor.zeros [3, 0]
  let noColumnsFactors := Tensor.qr noColumns
  expect "zero-column QR has an empty Q buffer"
    ((Tensor.to noColumnsFactors.q (Array Float)).isEmpty)
  expect "zero-column QR has an empty R buffer"
    ((Tensor.to noColumnsFactors.r (Array Float)).isEmpty)

/-- Error reductions must not turn exceptional entries into an apparent numerical match. -/
def checkErrorReductions : IO Unit := do
  let left : Tensor Float [2, 2] := [[1.0, -3.0], [2.0, 4.0]]
  let right : Tensor Float [2, 2] := [[0.0, -1.0], [5.0, 4.0]]
  expect "maximum tensor difference" (Tensor.maxAbsDiff left right == 3.0)
  expect "squared tensor difference" (Tensor.sum (Tensor.square (left - right)) == 14.0)
  expect "empty maximum absolute value"
    (Tensor.maxAbs (Tensor.zeros (α := Float) [0]) == 0.0)
  let nan : Float := 0.0 / 0.0
  for invalid in ([ [nan, 1.0], [1.0, nan] ] : List (Tensor Float [2])) do
    expect "NaN survives either reduction position" (Tensor.maxAbs invalid).isNaN
  let infinity : Float := 1.0 / 0.0
  let infinite : Tensor Float [1] := [infinity]
  expect "infinite mismatch remains infinite" ((Tensor.maxAbs infinite) == infinity)
  expect "equal infinities are an undefined difference"
    (Tensor.maxAbsDiff infinite infinite).isNaN
  let native : Tensor Float32 [2] := [1.0, -3.0]
  expect "binary32 maximum absolute value" (Tensor.maxAbs native == 3.0)
  let reference : Tensor (Binary 8 23) [2] :=
    [(1 : Binary 8 23), (Binary.canonicalNaN : Binary 8 23)]
  expect "reference binary32 NaN survives maximum"
    (Binary.isNaN (Tensor.maxAbs reference))
  let integral : Tensor Int [2] := [-3, 2]
  expect "ordered tensor fold" (Tensor.foldl (fun acc x => 10 * acc + x) 0 integral == -28)

/-- Huge empty axes must avoid traversal; literals must select the intended scalar storage. -/
def checkConstruction : IO Unit := do
  checkErrorReductions
  expect "empty einsum output" emptyEinsumOutput.data.isEmpty
  expect "symbolic empty einsum output" (symbolicEmptyEinsumOutput (2^40)).data.isEmpty
  expect "empty einsum contraction" (emptyEinsumContraction.data == #[0.0, 0.0])
  checkFactorizationResiduals
  checkReducedQR
  expect "decimal literals construct scalar Float tensors"
    (floatScalarLiteral.item == 1.25)
  expect "integer literals construct scalar exact tensors"
    (rationalScalarLiteral.item == 3)

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
    (Tensor.to (specialLeft + specialRight) (Array Float)).map Float.toBits
  expect "native Float addition preserves scalar IEEE bit behavior"
    (actualSpecialBits == expectedSpecialBits)

  let expectedSubBits :=
    #[positiveZero - negativeZero, negativeZero - negativeZero,
      positiveInfinity - 1.0, negativeInfinity - positiveInfinity,
      quietNaN - 2.0, 1.5 - -0.5].map Float.toBits
  let actualSubBits :=
    (Tensor.to (specialLeft - specialRight) (Array Float)).map Float.toBits
  expect "native Float subtraction preserves scalar IEEE bit behavior"
    (actualSubBits == expectedSubBits)
  let expectedMulBits :=
    #[positiveZero * negativeZero, negativeZero * negativeZero,
      positiveInfinity * 1.0, negativeInfinity * positiveInfinity,
      quietNaN * 2.0, 1.5 * -0.5].map Float.toBits
  let actualMulBits :=
    (Tensor.to (specialLeft * specialRight) (Array Float)).map Float.toBits
  expect "native Float multiplication preserves scalar IEEE bit behavior"
    (actualMulBits == expectedMulBits)
  let expectedDivBits :=
    #[positiveZero / negativeZero, negativeZero / negativeZero,
      positiveInfinity / 1.0, negativeInfinity / positiveInfinity,
      quietNaN / 2.0, 1.5 / -0.5].map Float.toBits
  let actualDivBits :=
    (Tensor.to (specialLeft / specialRight) (Array Float)).map Float.toBits
  expect "native Float division preserves scalar IEEE bit behavior"
    (actualDivBits == expectedDivBits)

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

def checkIndexingAndWindows : IO Unit := do
  let firstRow : Tensor Float [2] := floatMatrix[0]
  expect "natural-number tensor indexing selects the outer slice"
    (firstRow.to (Array Float) == #[1.0, 2.0])
  expect "natural-number vector indexing returns the scalar"
    (firstRow[1] == 2.0)

  let reshaped : Tensor Float [2, 2] :=
    (Tensor.from (#[1.0, 2.0, 3.0, 4.0] : Array Float)).reshape [2, 2]
  expect "literal reshape infers its element-count proof"
    (reshaped.to (Array Float) == #[1.0, 2.0, 3.0, 4.0])

  let sliceSource : Tensor Float [2, 3, 4] :=
    Tensor.generateFlat [2, 3, 4] Float.ofNat
  let middleChannels : Tensor Float [2, 1, 4] :=
    Tensor.sliceAxisRangeSpec 1 sliceSource 1 1 (by decide)
  expect "checked inner-axis slices preserve nonzero source offsets"
    (middleChannels.to (Array Float) ==
      #[4.0, 5.0, 6.0, 7.0, 16.0, 17.0, 18.0, 19.0])

  -- The CIFAR crop path must preserve channel and row strides after both axis truncations.
  let imageBatch : Tensor Float [1, 3, 32, 32] :=
    Tensor.generateFlat [1, 3, 32, 32] Float.ofNat
  let croppedRows : Tensor Float [1, 3, 8, 32] :=
    Tensor.take imageBatch 2 8
  let cropped : Tensor Float [1, 3, 8, 8] :=
    Tensor.take croppedRows 3 8
  let expectedCrop : Array Float :=
    Array.ofFn (n := 1 * 3 * 8 * 8) fun index =>
      let channel := index.val / 64
      let withinChannel := index.val % 64
      Float.ofNat
        (channel * 1024 + (withinChannel / 8) * 32 + withinChannel % 8)
  expect "CIFAR-style crops preserve channel and row offsets"
    (cropped.to (Array Float) == expectedCrop)

  let floatWindow : Tensor Float [4] :=
    Tensor.window ([10.0, 20.0, 30.0, 40.0] : Tensor Float [4]) 4 2 (-1.0)
  expect "tensor windows apply offsets and right padding"
    (floatWindow.to (Array Float) == #[30.0, 40.0, -1.0, -1.0])

  let byteWindow : Tensor UInt8 [3] :=
    Tensor.window ([10, 20, 30, 40] : Tensor UInt8 [4]) 3 1 0
  expect "tensor windows preserve packed byte storage"
    (byteWindow.to (Array UInt8) == #[20, 30, 40])

  let rationalWindow : Tensor Rat [3] :=
    Tensor.window ([1, 2] : Tensor Rat [2]) 3 0 (-1)
  expect "tensor windows preserve exact scalar storage"
    (rationalWindow.to (Array Rat) == #[1, 2, -1])

  let complexPad : Complex Float := { re := -1.0, im := 1.0 }
  let complexWindow : Tensor (Complex Float) [3] :=
    Tensor.window
      ([{ re := 1.0, im := 2.0 }, { re := 3.0, im := 4.0 }] : Tensor (Complex Float) [2])
      3 1 complexPad
  expect "tensor windows preserve boxed complex storage"
    (complexWindow.to (Array (Complex Float)) ==
      #[{ re := 3.0, im := 4.0 }, complexPad, complexPad])

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

  let complexRoundTripSource : Tensor (Complex Float) [2, 3] :=
    Tensor.generateFlat [2, 3] fun index =>
      { re := Float.ofNat (index + 1), im := Float.ofNat (index + 7) }
  let complexRoundTrip : Tensor (Complex Float) [2, 3] :=
    rearrange
      (rearrange complexRoundTripSource "row column -> column row")
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

def checkReductionsAndRepeat : IO Unit := do
  let rowSums : Tensor Float [2] :=
    reduce floatMatrix "row column -> row" by sum
  expect "sum reduction"
    (rowSums.to (Array Float) == #[3.0, 7.0])

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

  let repeatedInterleaved : Tensor Float [2, 3] :=
    expand vector "column -> column copy" with copy := 3
  expect "interleaved repeat retains its general gather layout"
    (repeatedInterleaved.to (Array Float) ==
      #[1.0, 1.0, 1.0, 2.0, 2.0, 2.0])

  let repeatedBytes : Tensor UInt8 [3, 2] :=
    expand ([1, 2] : Tensor UInt8 [2])
      "column -> row column" with row := 3
  expect "leading repeat uses packed byte storage"
    (repeatedBytes.to (Array UInt8) == #[1, 2, 1, 2, 1, 2])

  let product : Tensor Float [2, 2] :=
    einsum floatMatrix, floatRight
      "row contracted, contracted column -> row column"
  expect "einsum matrix contraction"
    (product.to (Array Float) == #[19.0, 22.0, 43.0, 50.0])

  checkOrderedContraction

def checkMixedPacking : IO Unit := do
  let bytes : Tensor UInt8 [2] := [1, 2]
  let offsets : Tensor Float [2] := [0.5, 1.5]
  let mixedOuter : Tensor Float [2, 2] :=
    einsum bytes, offsets "row, column -> row column"
  expect "einsum promotes mixed scalar inputs"
    (mixedOuter.to (Array Float) == #[0.5, 1.5, 1.0, 3.0])

  let mixedPacked := pack bytes, offsets "*"
  expect "pack promotes mixed scalar inputs"
    (Tensor.to mixedPacked.tensor (Array Float) == #[1.0, 2.0, 0.5, 1.5])

  let packedSource : Tensor Float [1, 2] := [[3.0, 4.0]]
  let packed := pack offsets, packedSource "*"
  expect "pack values"
    (Tensor.to packed.tensor (Array Float) == #[0.5, 1.5, 3.0, 4.0])
  expect "pack metadata"
    (packed.shapes == ([[2], [1, 2]] : List Shape))

  let unpacked := unpack packed "*"
  let unpackedVector : Tensor Float [2] :=
    unpacked ⟨0, by decide⟩
  let unpackedMatrix : Tensor Float [1, 2] :=
    unpacked ⟨1, by decide⟩
  expect "unpack consumes pack metadata for the vector component"
    (Tensor.to unpackedVector (Array Float) == #[0.5, 1.5])
  expect "unpack consumes pack metadata for the matrix component"
    (Tensor.to unpackedMatrix (Array Float) == #[3.0, 4.0])

  expect "parse_shape reports named dimensions"
    (parse_shape floatMatrix "row column" ==
      [("row", 2), ("column", 2)])

/-- Exercise compiled construction, packed storage, indexing, and scalar-generic operations. -/
def run : IO Unit := do
  checkConstruction
  checkNativeArithmetic
  checkIndexingAndWindows
  checkPackedTransposes
  checkBoxedTransposes
  checkReductionsAndRepeat
  checkMixedPacking
  IO.println "  public tensor operations: passed"

end NN.Tests.Tensor.Operations
