/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tests.MLTheory.Utils
public import NN.Runtime.Autograd.IRExec
public import NN.Spec.Core.Context.Rational

/-!
# Broadcast and Vector Matmul Regressions

These tests compare independently specified results with IR inference, reference evaluation,
IBP, nodewise CROWN, and backward objectives. Rational checks exercise coefficient accumulation
when several output batches share one input coordinate. Floating checks exercise directed
products and sums.
-/

public section

namespace NN.Tests.MLTheory.BinaryMatmul

open Spec TorchLean
open NN.IR
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open scoped Spec.RationalAlgebraic

private instance : BoundOps Rat where
  addDown := (· + ·)
  addUp := (· + ·)
  subDown := (· - ·)
  subUp := (· - ·)
  mulDown := (· * ·)
  mulUp := (· * ·)
  supportsExactAffineReassociation := true

private def require (condition : Bool) (message : String) : IO Unit :=
  unless condition do throw <| IO.userError s!"binary matmul: {message}"

private def graph (left right output : Shape) : NN.IR.Graph :=
  { nodes := #[
    { id := 0, kind := .input, parents := #[], outShape := left },
    { id := 1, kind := .const right, parents := #[], outShape := right },
    { id := 2, kind := .matmul, parents := #[0, 1], outShape := output }] }

private def checkCase {α : Type} [TorchLean.Storage α] [Context α]
    [BoundOps α] [NonlinearBoundOps α] [DecidableLE α]
    {leftShape rightShape : Shape}
    (label : String) (left : Tensor α leftShape) (right : Tensor α rightShape)
    (outputShape : Shape) (expected : Array α) : IO Unit := do
  let .ok inferred := OpContracts.inferMatmulOutShape leftShape rightShape
    | throw <| IO.userError s!"{label}: shape inference failed"
  require (inferred == outputShape) s!"{label}: inferred shape"
  let g := graph leftShape rightShape outputShape
  require g.checkShapes.isOk s!"{label}: graph shape check"
  let .ok result := NN.IR.Graph.evalAt g ({} : Payload α) (SomeTensor.ofTensor left)
      #[SomeTensor.ofTensor left, SomeTensor.ofTensor right] 2
    | throw <| IO.userError s!"{label}: IR evaluation failed"
  require (result.shape == outputShape) s!"{label}: evaluated shape"
  require (Tensor.to result.tensor (Array α) == expected) s!"{label}: evaluated values"
  let payload : Payload α :=
    { const? := fun id => if id = 1 then
        some ⟨rightShape.size, Tensor.flattenSpec right⟩ else none }
  let .ok lowered := Runtime.Autograd.IRExec.lowerToForwardGraph g payload
    | throw <| IO.userError s!"{label}: lowering failed"
  if h : leftShape = lowered.inShape then
    let some value := (lowered.denoteAll (Tensor.castShape left h))[2]?
      | throw <| IO.userError s!"{label}: missing lowered output"
    require (value.shape == outputShape) s!"{label}: lowered shape"
    require (Tensor.to value.tensor (Array α) == expected) s!"{label}: lowered values"
  else
    throw <| IO.userError s!"{label}: lowered input shape"
  let leftBox := FlatBox.ofTensor (Tensor.flattenSpec left)
  let rightValue : FlatTensor α := ⟨rightShape.size, Tensor.flattenSpec right⟩
  let ps : ParamStore α :=
    { inputBoxes := Std.HashMap.emptyWithCapacity.insert 0 leftBox
      constVals := Std.HashMap.emptyWithCapacity.insert 1 rightValue }
  let ibp := runIBP g ps
  let some box := ibp[2]?.join
    | throw <| IO.userError s!"{label}: missing IBP box"
  require (box.dim == expected.size) s!"{label}: IBP dimension"
  for i in [:expected.size] do
    require (decide (getAtOrZero box.lo [i] ≤ expected[i]!) &&
      decide (expected[i]! ≤ getAtOrZero box.hi [i])) s!"{label}: IBP coordinate {i}"
  let ctx : AffineCtx := { inputId := 0, inputDim := leftShape.size }
  let .ok crownBox := evalCROWNOutputBox? (runCROWN g ps ctx ibp) leftBox 2 leftShape.size
    | throw <| IO.userError s!"{label}: missing nodewise CROWN box"
  require (crownBox.dim == expected.size) s!"{label}: CROWN dimension"
  for i in [:expected.size] do
    require (decide (getAtOrZero crownBox.lo [i] ≤ expected[i]!) &&
      decide (expected[i]! ≤ getAtOrZero crownBox.hi [i])) s!"{label}: CROWN coordinate {i}"
  let objective : FlatTensor α :=
    ⟨outputShape.size, Tensor.full [outputShape.size] 1⟩
  let .ok bound := backwardObjectiveBox? g ps ctx ibp leftBox 2 objective
    | throw <| IO.userError s!"{label}: missing backward objective"
  let target := expected.foldl (· + ·) 0
  require (decide (getAtOrZero bound.lo [0] ≤ target) &&
    decide (target ≤ getAtOrZero bound.hi [0])) s!"{label}: backward enclosure"

private def checkShapesAndValues : IO Unit := do
  checkCase "dot" ([1, 2, 3] : Tensor Rat [3]) ([4, 5, 6] : Tensor Rat [3])
    [] #[32]
  checkCase "vector matrix" ([1, 2] : Tensor Rat [2])
    ([[1, 2, 3], [4, 5, 6]] : Tensor Rat [2, 3]) [3] #[9, 12, 15]
  checkCase "matrix vector" ([[1, 2], [3, 4]] : Tensor Rat [2, 2])
    ([2, 3] : Tensor Rat [2]) [2] #[8, 18]
  checkCase "vector broadcast"
    ([1, 2] : Tensor Rat [2])
    ([[[1, 0], [0, 1]], [[2, 0], [0, 3]]] : Tensor Rat [2, 2, 2])
    [2, 2] #[1, 2, 2, 6]
  checkCase "matrix broadcast"
    ([[1, 2], [3, 4]] : Tensor Rat [2, 2])
    ([[[1], [2]], [[3], [4]], [[5], [6]]] : Tensor Rat [3, 2, 1])
    [3, 2, 1] #[5, 11, 11, 25, 17, 39]
  let left : Tensor Rat [2, 1, 2, 2] := [[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]]
  let right : Tensor Rat [1, 3, 2, 2] :=
    [[[[1, 0], [0, 1]], [[2, 0], [0, 3]], [[1, 1], [1, -1]]]]
  checkCase "two-sided broadcast" left right [2, 3, 2, 2]
    #[1, 2, 3, 4, 2, 6, 6, 12, 3, -1, 7, -1,
      5, 6, 7, 8, 10, 18, 14, 24, 11, -1, 15, -1]
  checkCase "empty dot" (Tensor.full (α := Rat) [0] 0) (Tensor.full [0] 0) [] #[0]
  checkCase "empty contraction" (Tensor.full (α := Rat) [2, 0] 0)
    (Tensor.full [0, 3] 0) [2, 3] #[0, 0, 0, 0, 0, 0]
  checkCase "zero batch broadcast" (Tensor.full (α := Rat) [1, 1, 2] 0)
    (Tensor.full [0, 2, 1] 0) [0, 1, 1] #[]
  checkCase "empty rows" (Tensor.full (α := Rat) [0, 2] 0)
    (Tensor.full [2, 3] 0) [0, 3] #[]
  checkCase "Float vector broadcast"
    ([1, 2] : Tensor Float [2])
    ([[[1, 0], [0, 1]], [[2, 0], [0, 3]]] : Tensor Float [2, 2, 2])
    [2, 2] #[1, 2, 2, 6]
  checkCase "Float32 matrix vector"
    ([[1, 2], [3, 4]] : Tensor Float32 [2, 2])
    ([2, 3] : Tensor Float32 [2]) [2] #[8, 18]

/-- Shared input coefficients must receive contributions from every broadcast output batch. -/
private def checkBackwardCoefficients : IO Unit := do
  let right : Tensor Rat [2, 2, 2] := [[[1, 0], [0, 1]], [[2, 0], [0, 3]]]
  let leftBox : FlatBox Rat := { dim := 2, lo := [-1, -2], hi := [1, 2] }
  let ps : ParamStore Rat :=
    { inputBoxes := Std.HashMap.emptyWithCapacity.insert 0 leftBox
      constVals := Std.HashMap.emptyWithCapacity.insert 1
        ⟨8, Tensor.flattenSpec right⟩ }
  for parents in [#[0, 1], #[1, 0]] do
    let g : NN.IR.Graph := { nodes := #[
      { id := 0, kind := .input, parents := #[], outShape := [2] },
      { id := 1, kind := .const [2, 2, 2], parents := #[], outShape := [2, 2, 2] },
      { id := 2, kind := .matmul, parents := parents, outShape := [2, 2] }] }
    let some bounds := runCROWNBackwardObjective g ps
        { inputId := 0, inputDim := 2 } (runIBP g ps) 2 ⟨4, [1, 1, 1, 1]⟩
      | throw <| IO.userError "broadcast coefficient transfer failed"
    for aff in [bounds.loAff, bounds.hiAff] do
      require (getAtOrZero aff.A [0, 0] == 3 && getAtOrZero aff.A [0, 1] == 4)
        "broadcast coefficients were overwritten instead of summed"
      require (getAtOrZero aff.c [0] == 0) "unexpected broadcast constant"

private def checkBilinearBox : IO Unit := do
  let left : FlatBox Rat := { dim := 2, lo := [-2, 1], hi := [3, 4] }
  let right : FlatBox Rat := { dim := 2, lo := [-5, -3], hi := [2, 6] }
  let some bound := ibpBinaryMatmul? [2] [2] left right
    | throw <| IO.userError "missing mixed-sign bilinear box"
  require (getAtOrZero bound.lo [0] == -27 && getAtOrZero bound.hi [0] == 34)
    "mixed-sign bilinear extrema"

private def checkRejections : IO Unit := do
  for (left, right) in [([], [2]), ([2], []), ([2], [3]), ([2, 3, 4], [5, 4, 6])] do
    require (!(OpContracts.inferMatmulOutShape left right).isOk) "invalid shape accepted"
  let box : FlatBox Float := FlatBox.ofTensor ([1, 2] : Tensor Float [2])
  require (ibpBinaryMatmul? [3] [3] box box).isNone "wrong flat operand size accepted"
  let g := graph [2] [2] [1]
  let ps : ParamStore Float :=
    { inputBoxes := Std.HashMap.emptyWithCapacity.insert 0 box
      constVals := Std.HashMap.emptyWithCapacity.insert 1 ⟨2, [1, 2]⟩ }
  require (!crownGraphSemanticsSupported g ps) "unsqueezed dot output accepted"

private def checkRationalBounds (label : String) (lower upper : Option Rat)
    (expected : Rat) : IO Unit := do
  let some lo := lower | throw <| IO.userError s!"{label}: nonfinite lower bound"
  let some hi := upper | throw <| IO.userError s!"{label}: nonfinite upper bound"
  require (decide (lo ≤ expected) && decide (expected ≤ hi)) label

/-- Endpoint products are rounded before summation; compare with exact rational multiplication. -/
private def checkDirectedProducts : IO Unit := do
  let x : Float := 1 + 1 / 134217728
  let y : Float := 1 - 1 / 134217728
  let a : FlatBox Float := FlatBox.ofTensor (Tensor.full [1] x)
  let b : FlatBox Float := FlatBox.ofTensor (Tensor.full [1] y)
  let some bound := ibpBinaryMatmul? [1] [1] a b
    | throw <| IO.userError "missing directed dot"
  let exact : Rat := (1 + 1 / 134217728) * (1 - 1 / 134217728)
  let product := intervalMul x x y y
  checkRationalBounds "Float endpoint product enclosure"
    (FloatLib.Floats.ExecFloat.Binary.toRat? (FloatLib.Floats.ExecFloat.Binary.ofFloat product.1))
    (FloatLib.Floats.ExecFloat.Binary.toRat? (FloatLib.Floats.ExecFloat.Binary.ofFloat product.2))
    exact
  let lo := getAtOrZero bound.lo [0]
  let hi := getAtOrZero bound.hi [0]
  checkRationalBounds "rounded product excluded exact result"
    (FloatLib.Floats.ExecFloat.Binary.toRat? (FloatLib.Floats.ExecFloat.Binary.ofFloat lo))
    (FloatLib.Floats.ExecFloat.Binary.toRat? (FloatLib.Floats.ExecFloat.Binary.ofFloat hi)) exact
  let squared := boxSquare a
  checkRationalBounds "rounded square excluded exact result"
    (FloatLib.Floats.ExecFloat.Binary.toRat?
      (FloatLib.Floats.ExecFloat.Binary.ofFloat (getAtOrZero squared.lo [0])))
    (FloatLib.Floats.ExecFloat.Binary.toRat?
      (FloatLib.Floats.ExecFloat.Binary.ofFloat (getAtOrZero squared.hi [0])))
    ((1 + 1 / 134217728) ^ 2)
  let x32 : Float32 := 1 + 1 / 8192
  let y32 : Float32 := 1 - 1 / 8192
  let a32 : FlatBox Float32 := FlatBox.ofTensor (Tensor.full [1] x32)
  let b32 : FlatBox Float32 := FlatBox.ofTensor (Tensor.full [1] y32)
  let some bound32 := ibpBinaryMatmul? [1] [1] a32 b32
    | throw <| IO.userError "missing directed Float32 dot"
  let exact32 : Rat := (1 + 1 / 8192) * (1 - 1 / 8192)
  let product32 := intervalMul x32 x32 y32 y32
  checkRationalBounds "Float32 endpoint product enclosure"
    (FloatLib.Floats.ExecFloat.Binary.toRat?
      (FloatLib.Floats.ExecFloat.Binary.ofFloat32 product32.1))
    (FloatLib.Floats.ExecFloat.Binary.toRat?
      (FloatLib.Floats.ExecFloat.Binary.ofFloat32 product32.2)) exact32
  checkRationalBounds "Float32 product enclosure"
    (FloatLib.Floats.ExecFloat.Binary.toRat?
      (FloatLib.Floats.ExecFloat.Binary.ofFloat32 (getAtOrZero bound32.lo [0])))
    (FloatLib.Floats.ExecFloat.Binary.toRat?
      (FloatLib.Floats.ExecFloat.Binary.ofFloat32 (getAtOrZero bound32.hi [0]))) exact32
  let squared32 := boxSquare a32
  checkRationalBounds "Float32 square enclosure"
    (FloatLib.Floats.ExecFloat.Binary.toRat?
      (FloatLib.Floats.ExecFloat.Binary.ofFloat32 (getAtOrZero squared32.lo [0])))
    (FloatLib.Floats.ExecFloat.Binary.toRat?
      (FloatLib.Floats.ExecFloat.Binary.ofFloat32 (getAtOrZero squared32.hi [0])))
    ((1 + 1 / 8192) ^ 2)
  let small : Float := 1 / 18014398509481984
  let values : Tensor Float [129] :=
    Tensor.ofFn fun i => if i.val = 0 then 1 else small
  let ones : Tensor Float [129] := Tensor.full [129] 1
  let some sum := ibpBinaryMatmul? [129] [129]
      (FlatBox.ofTensor values) (FlatBox.ofTensor ones)
    | throw <| IO.userError "missing directed sum"
  let exactSum := 1 + 128 * small
  require (getAtOrZero sum.lo [0] ≤ exactSum && exactSum ≤ getAtOrZero sum.hi [0])
    "lost small summands"

def run : IO Unit := do
  checkShapesAndValues
  checkBackwardCoefficients
  checkBilinearBox
  checkRejections
  checkDirectedProducts
  IO.println "binary matmul: shapes, values, enclosures, and broadcast coefficients passed"

end NN.Tests.MLTheory.BinaryMatmul
