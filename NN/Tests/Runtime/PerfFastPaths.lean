/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Layers.Recurrent
public import NN.Runtime.Autograd.Torch.Core.BackwardOptim
public import NN.Runtime.Autograd.Torch.Core.Trainer.EagerOps
public import NN.Spec.Core.Sequence
public import NN.IR.Semantics
public import NN.Tensor.ShapeErasure
public import NN.Runtime.Autograd.Torch.TypedGraphSession.GraphOps

/-!
Regressions for the runtime fast paths.

Each check compares a compiled fast path against a reference bit for bit: the arithmetic
coordinate index maps, the IR broadcast matmul, shape erasure of tensor packs, typed graph value
reads, slice copies for `unstack` and `stack`, the push-based `mapAccumRight` loop, eager tape
recording after a failed operation, the stacked recurrent output, the balanced leading-axis map,
parent-gated linear backward, and the scaled division gradient.
-/

public section

namespace NN.Tests.Runtime.PerfFastPaths

open Spec TorchLean Runtime.Autograd

private def check (label : String) (condition : Bool) : IO Unit := do
  unless condition do throw <| IO.userError s!"runtime fast paths: {label}"

/-- Bitwise array equality, so a signed zero or a NaN payload change is a failure. -/
private def sameBits (left right : Array Float) : Bool :=
  left.size == right.size &&
    (List.range left.size).all fun i => left[i]!.toBits == right[i]!.toBits

private def floats {s : Shape} (x : Tensor Float s) : Array Float :=
  Tensor.to x (Array Float)

private def naturals {s : Shape} (x : Tensor Nat s) : Array Nat :=
  Tensor.to x (Array Nat)

/-- Entries with distinct values, signed zeros, and a NaN, so a misplaced copy is visible. -/
private def sample (s : Shape) : Tensor Float s :=
  TorchLean.Tensor.Internal.Rep.ofFlatFn fun index =>
    match index.val % 7 with
    | 0 => -0.0
    | 3 => (0.0 : Float) / 0.0
    | _ => index.val.toFloat * 0.37 - 5.0

private def naturalSample (s : Shape) : Tensor Nat s :=
  TorchLean.Tensor.Internal.Rep.ofFlatFn fun index => 3 * index.val + 1

/-- `unstack` and `stack` agree with their coordinate definitions on packed and boxed storage. -/
def checkSlices : IO Unit := do
  let x := sample [4, 2, 3]
  for i in List.finRange 4 do
    let fast := TorchLean.Tensor.Internal.Rep.unstack x i
    let reference : Tensor Float [2, 3] :=
      TorchLean.Tensor.Internal.Rep.ofFn fun coordinate => x (i, coordinate)
    check s!"unstack Float row {i}" (sameBits (floats fast) (floats reference))
  let n := naturalSample [3, 5]
  for i in List.finRange 3 do
    let fast := TorchLean.Tensor.Internal.Rep.unstack n i
    let reference : Tensor Nat [5] :=
      TorchLean.Tensor.Internal.Rep.ofFn fun coordinate => n (i, coordinate)
    check s!"unstack Nat row {i}" (naturals fast == naturals reference)
  let empty := sample [3, 0]
  check "unstack empty tail" ((floats (TorchLean.Tensor.Internal.Rep.unstack empty 1)).size == 0)
  let components : Fin 5 → Tensor Float [2, 3] := fun i =>
    TorchLean.Tensor.Internal.Rep.ofFlatFn fun index =>
      if index.val == 2 then -0.0 else (i.val * 10 + index.val).toFloat
  let stacked := TorchLean.Tensor.Internal.Rep.stack components
  let reference : Tensor Float [5, 2, 3] :=
    TorchLean.Tensor.Internal.Rep.ofFn fun coordinate => components coordinate.1 coordinate.2
  check "stack Float" (sameBits (floats stacked) (floats reference))
  let natComponents : Fin 4 → Tensor Nat [3] := fun i =>
    TorchLean.Tensor.Internal.Rep.ofFlatFn fun index => i.val * 100 + index.val
  let natStacked := TorchLean.Tensor.Internal.Rep.stack natComponents
  let natReference : Tensor Nat [4, 3] :=
    TorchLean.Tensor.Internal.Rep.ofFn fun coordinate =>
      natComponents coordinate.1 coordinate.2
  check "stack Nat" (naturals natStacked == naturals natReference)
  let emptyStack := TorchLean.Tensor.Internal.Rep.stack fun (_ : Fin 6) => sample [0]
  check "stack empty tail" ((floats emptyStack).size == 0)

/-- `mapAccumRight` visits indices from right to left and returns results in index order. -/
def checkMapAccumRight : IO Unit := do
  let n := 9
  let (visited, results) := Sequence.mapAccumRight n ([] : List Nat) fun i state =>
    (state ++ [i.val], i.val * i.val + state.length)
  check "mapAccumRight visit order" (visited == (List.range n).reverse)
  let expected := (List.range n).toArray.map fun i => i * i + (n - 1 - i)
  check "mapAccumRight results" (naturals results == expected)
  let (state, nothing) := Sequence.mapAccumRight 0 (7 : Nat) fun _ state => (state + 1, (0 : Nat))
  check "mapAccumRight empty" (state == 7 && (naturals nothing).size == 0)

/-- A failed eager operation leaves the tape as it was and later gradients stay correct. -/
def checkEagerRecording : IO Unit := do
  let session ← Torch.Internal.EagerSession.new (α := Float)
  let x ← session.input (sh := Shape.scalar) (Tensor.scalar (2.0 : Float)) (requiresGrad := true)
  let y ← session.mul x x
  let before := (← session.tape.get).nodes.size
  let negative ← session.scale y (-1.0)
  let afterScale := (← session.tape.get).nodes.size
  let failed ← try
      let _ ← session.log negative
      pure false
    catch _ => pure true
  check "log of a negative input fails" failed
  check "failed op keeps the tape" ((← session.tape.get).nodes.size == afterScale)
  check "scale appended one node" (afterScale == before + 1)
  let z ← session.add y x
  let gradients ← session.backwardScalarDenseAll z
  let dx ← Torch.Internal.EagerSession.grad gradients x
  check "gradient after a failed op" (dx.item == 5.0)

/-- Stacking recurrent step outputs matches one `writeLeading` scatter per step. -/
def checkUnrollLeading : IO Unit := do
  let rows := 7
  let width := 3
  let run (stacked : Bool) : IO (Array Float × Array Float) := do
    let session ← Torch.Internal.EagerSession.new (α := Float)
    let xs ← session.input (sh := [rows, width]) (sample [rows, width]) (requiresGrad := true)
    let action : Torch.Internal.EagerM Float (Torch.TensorRef Float [rows, width]) := do
      let base ← Model.const (m := Torch.Internal.EagerM Float) (α := Float)
        (Tensor.full [rows, width] (0.0 : Float))
      let initial ← Model.const (m := Torch.Internal.EagerM Float) (α := Float)
        (Tensor.full [width] (0.25 : Float))
      let step := fun (t : Fin rows) (previous : Torch.TensorRef Float [width]) => do
        let input ← Model.select (m := Torch.Internal.EagerM Float) (α := Float)
          (s := [rows, width]) 0 xs t
        let sum ← Model.add (m := Torch.Internal.EagerM Float) (α := Float) input previous
        let hidden ← Model.tanh (m := Torch.Internal.EagerM Float) (α := Float) sum
        pure (hidden, hidden)
      if stacked then
        Model.Layers.Internal.unrollLeading (m := Torch.Internal.EagerM Float) (α := Float)
          base initial step
      else
        let (_, output) ← (List.finRange rows).foldlM (init := (initial, base))
          fun (previous, output) t => do
            let (next, value) ← step t previous
            let output ← Model.Layers.Internal.writeLeading
              (m := Torch.Internal.EagerM Float) (α := Float) output value t
            pure (next, output)
        pure output
    let output ← action session
    let total ← (Torch.Ops.sum (m := Torch.Internal.EagerM Float) (α := Float) output) session
    let gradients ← session.backwardScalarDenseAll total
    let dxs ← Torch.Internal.EagerSession.grad gradients xs
    pure (floats (← session.getValue output), floats dxs)
  let (fastValue, fastGrad) ← run true
  let (referenceValue, referenceGrad) ← run false
  check "unrollLeading forward" (sameBits fastValue referenceValue)
  check "unrollLeading backward" (sameBits fastGrad referenceGrad)

/-- The balanced leading-axis map applies the row operation to every row, in order. -/
def checkMapLeading : IO Unit := do
  let rows := 11
  let session ← Torch.Internal.EagerSession.new (α := Float)
  let weightValue := sample [2, 3]
  let biasValue := sample [2]
  let inputValue := sample [rows, 3]
  let weight ← session.input (sh := [2, 3]) weightValue (requiresGrad := true)
  let bias ← session.input (sh := [2]) biasValue (requiresGrad := true)
  let input ← session.input (sh := [rows, 3]) inputValue (requiresGrad := true)
  let output ← (Model.linearEach (m := Torch.Internal.EagerM Float) (α := Float)
    (leadingShape := [rows]) weight bias input) session
  let value ← session.getValue output
  let layer : Spec.LinearSpec Float 3 2 := { weights := weightValue, bias := biasValue }
  for i in List.finRange rows do
    let expected := Spec.linearSpec layer (Spec.get inputValue i)
    check s!"mapLeading row {i}" (sameBits (floats (Spec.get value i)) (floats expected))

/-- Linear backward skips parents without gradients and keeps the others unchanged. -/
def checkLinearGating : IO Unit := do
  let gradientsFor (inputGrad : Bool) : IO (Array Float × Array Float × Array Float) := do
    let session ← Torch.Internal.EagerSession.new (α := Float)
    let weight ← session.input (sh := [2, 3]) (sample [2, 3]) (requiresGrad := true)
    let bias ← session.input (sh := [2]) (sample [2]) (requiresGrad := true)
    let input ← session.input (sh := [3]) (sample [3]) (requiresGrad := inputGrad)
    let output ← session.linear weight bias input
    let total ← session.sum output
    let gradients ← session.backwardScalarDenseAll total
    let dw ← Torch.Internal.EagerSession.grad gradients weight
    let db ← Torch.Internal.EagerSession.grad gradients bias
    let dx ← Torch.Internal.EagerSession.grad gradients input
    pure (floats dw, floats db, floats dx)
  let (dwAll, dbAll, dxAll) ← gradientsFor true
  let (dwData, dbData, dxData) ← gradientsFor false
  check "linear weight gradient" (sameBits dwAll dwData)
  check "linear bias gradient" (sameBits dbAll dbData)
  check "linear input gradient is computed" (dxAll.any (· != 0.0))
  check "linear data input has no gradient" (dxData.all (· == 0.0))

/-- The divisor gradient stays finite and nonzero when `b * b` would overflow. -/
def checkDivisionGradient : IO Unit := do
  let t0 : Tape Float := Tape.empty
  let (t1, aId) := Tape.leaf (t := t0) (Tensor.scalar (1.0e200 : Float))
  let (t2, bId) := Tape.leaf (t := t1) (Tensor.scalar (1.0e160 : Float))
  let .ok (t3, yId) := Tape.div (t := t2) (s := Shape.scalar) aId bId
    | throw <| IO.userError "runtime fast paths: division did not record"
  let .ok gradients := Tape.backwardScalar (t := t3) yId
    | throw <| IO.userError "runtime fast paths: division backward failed"
  let some db := gradients[bId]?
    | throw <| IO.userError "runtime fast paths: missing divisor gradient"
  if h : db.shape = Shape.scalar then
    let value := (db.cast h).item
    check s!"divisor gradient {value}" (value.isFinite && value < 0.0 &&
      Float.abs (value + 1.0e-120) ≤ 1.0e-133)
  else
    check "divisor gradient shape" false

/-- Every coordinate of a shape in row-major order, built by nested enumeration. -/
def rowMajorCoords : (s : List Nat) → List (TorchLean.Tensor.Internal.Coord s)
  | [] => [PUnit.unit]
  | n :: s => (List.finRange n).flatMap fun i => (rowMajorCoords s).map fun rest => (i, rest)

/--
The compiled index maps, `linearize`, `unlinearize` and `equivFin`, agree with an independent
row-major enumeration of every coordinate.
-/
def checkCoordIndex : IO Unit := do
  let shapes : List (List Nat) := [[], [3], [3, 4, 5], [1, 5, 1], [4, 0, 2], [2, 3, 2, 3]]
  for s in shapes do
    let size := TorchLean.Tensor.Internal.Shape.size s
    let coords := rowMajorCoords s
    check s!"coordinate count {s}" (coords.length == size)
    for (coordinate, i) in coords.zipIdx do
      if h : i < size then
        let index : Fin size := ⟨i, h⟩
        let equiv := TorchLean.Tensor.Internal.Coord.equivFin s
        check s!"unlinearize {s} {i}"
          (decide (TorchLean.Tensor.Internal.Coord.unlinearize index = coordinate) &&
            decide (equiv.symm index = coordinate))
        check s!"linearize {s} {i}"
          ((TorchLean.Tensor.Internal.Coord.linearize coordinate).val == i &&
            (equiv coordinate).val == i)

/--
The IR broadcast matmul fallback agrees bit for bit with a direct triple loop that sums the
products in contraction order.
-/
def checkBroadcastMatmul : IO Unit := do
  let rows := 2
  let inner := 3
  let cols := 2
  for (leftBatch, rightBatch) in [(2, 1), (1, 2)] do
    let batch := max leftBatch rightBatch
    let dims : NN.IR.OpContracts.MatmulDims :=
      { leading := [batch], rows, inner, cols
        leftShape := [leftBatch, rows, inner], rightShape := [rightBatch, inner, cols]
        leftLeading := [leftBatch], rightLeading := [rightBatch] }
    let leftValues : Array Float :=
      (Array.range (leftBatch * rows * inner)).map fun i => Float.sin (i.toFloat + 0.25)
    let rightValues : Array Float :=
      (Array.range (rightBatch * inner * cols)).map fun i => Float.cos (i.toFloat * 0.7)
    let left : Tensor Float dims.leftShape :=
      TorchLean.Tensor.Internal.Rep.ofFlatFn fun i => leftValues[i.val]!
    let right : Tensor Float dims.rightShape :=
      TorchLean.Tensor.Internal.Rep.ofFlatFn fun i => rightValues[i.val]!
    let output := (NN.IR.Graph.matmulWithDims (α := Float) dims left right).to (Array Float)
    let mut expected : Array Float := #[]
    for b in [0:batch] do
      for i in [0:rows] do
        for j in [0:cols] do
          let lb := if leftBatch = 1 then 0 else b
          let rb := if rightBatch = 1 then 0 else b
          let mut acc : Float := 0
          for k in [0:inner] do
            acc := acc + leftValues[(lb * rows + i) * inner + k]! *
              rightValues[(rb * inner + k) * cols + j]!
          expected := expected.push acc
    check s!"broadcast matmul {leftBatch}x{rightBatch}"
      (output.map Float.toBits == expected.map Float.toBits)

/-- Shape erasure keeps pack order, shapes, values, and the requested index numbering. -/
def checkShapeErasure : IO Unit := do
  let first : Tensor Float [2] := Tensor.ofFn fun i => i.val.toFloat + 1.0
  let second : Tensor Float Shape.scalar := Tensor.scalar 5.0
  let third : Tensor Float [1, 3] := TorchLean.Tensor.Internal.Rep.ofFlatFn fun i =>
    i.val.toFloat + 10.0
  let pack := TensorPack.cons first (.cons second (.cons third .nil))
  let erased := pack.toShapeErasedArray
  let indexed := pack.toIndexedShapeErasedArray 4
  let shapes : List Shape := [[2], Shape.scalar, [1, 3]]
  let values : List (Array Float) := [#[1.0, 2.0], #[5.0], #[10.0, 11.0, 12.0]]
  check "shape erasure shapes" (erased.toList.map (·.shape) == shapes)
  check "shape erasure values" (erased.toList.map (·.tensor.to (Array Float)) == values)
  check "indexed shape erasure"
    (indexed.toList.map (fun entry => (entry.1, entry.2.shape)) ==
      [(4, [2]), (5, Shape.scalar), (6, [1, 3])])

/-- Typed graph reads see nodes recorded after an earlier read and a reset session. -/
def checkTypedGraphValueCache : IO Unit := do
  let session ← Torch.Internal.TypedGraphSession.new (α := Float)
  let x ← session.input (sh := [2]) (Tensor.ofFn fun i => i.val.toFloat + 1.0)
  let y ← session.add x x
  check "typed graph first read" ((← session.getValue y).to (Array Float) == #[2.0, 4.0])
  let z ← session.add y x
  check "typed graph read after append" ((← session.getValue z).to (Array Float) == #[3.0, 6.0])
  check "typed graph earlier node" ((← session.getValue y).to (Array Float) == #[2.0, 4.0])
  session.resetTape
  let w ← session.input (sh := [2]) (Tensor.ofFn fun i => i.val.toFloat * 10.0)
  let v ← session.add w w
  check "typed graph read after reset" ((← session.getValue v).to (Array Float) == #[0.0, 20.0])

def run : IO Unit := do
  checkCoordIndex
  checkBroadcastMatmul
  checkShapeErasure
  checkTypedGraphValueCache
  checkSlices
  checkMapAccumRight
  checkEagerRecording
  checkUnrollLeading
  checkMapLeading
  checkLinearGating
  checkDivisionGradient
  IO.println "  runtime fast paths: ok"

end NN.Tests.Runtime.PerfFastPaths
