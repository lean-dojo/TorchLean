/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TypedGraph
public import NN.Runtime.Autograd.Torch.Core.TypedGraph
public import NN.Spec.Core.FloatInstances

/-!
Canonical `Float32.toBits` comparisons against the original dense Tape engine. These observations
distinguish signed zeros but canonicalize NaNs, so they do not test raw NaN sign or payload
identity.
Branch cancellation distinguishes reverse graph order from creation order; the repeated-parent case
distinguishes adding a complete local VJP from sequentially scattering its two terms. Singular and
custom VJPs exercise nodes with zero incoming cotangent, and input/intermediate seeds exercise the
full dense interface.
Single-output differentiation separately checks that disconnected overflow and singular VJPs are
not run, including configured scalars whose storage equality must compare encodings.
-/

public section

open Spec TorchLean Runtime.Autograd
open Runtime.Autograd.TypedGraph
open Proofs.Autograd.Algebra

namespace TypedGraphSemanticRegression

private instance : Inhabited (Spec.SomeTensor Float32) :=
  ⟨Spec.SomeTensor.ofTensor (Tensor.scalar (0.0 : Float32))⟩

private def expect (label : String) (condition : Bool) : IO Unit := do
  unless condition do throw <| IO.userError label

/-- Observe each scalar through `Float32.toBits`, which canonicalizes NaNs. -/
private def bits (value : Spec.SomeTensor Float32) : Array UInt32 :=
  (Storage.toArray value.tensor.buffer).map Float32.toBits

private def compare (label : String) (actual expected : Array (Spec.SomeTensor Float32)) :
    IO Unit := do
  expect (label ++ ": gradient count") (actual.size == expected.size)
  for i in [:actual.size] do
    let left := actual[i]!
    let right := expected[i]!
    expect s!"{label}: shape at {i}" (left.shape == right.shape)
    expect s!"{label}: tensor {i}: got {bits left}, expected {bits right}"
      (bits left == bits right)

private def checkSeed {Γ ss : List Shape} (label : String) (compiled : Compiled Float32 Γ ss)
    (reference : Tape (Spec.SomeTensor Float32)) (seed : TorchLean.TensorPack Float32 (Γ ++ ss)) :
    IO (Array (Spec.SomeTensor Float32)) := do
  let expected ← IO.ofExcept (reference.backwardDenseFrom seed.toShapeErasedArray)
  let actual := compiled.backwardDenseFrom seed
  compare label actual expected
  for _ in [:3] do
    compare (label ++ ": repeat") (compiled.backwardDenseFrom seed) actual
  pure actual

private def branch : GraphM.M Float32 [[4]] (GraphM.Var [4]) := do
  let x ← GraphM.arg 0 [4]
  let a ← GraphM.scale x 1.0
  let b ← GraphM.scale x 16777216.0
  let c ← GraphM.scale x (-16777216.0)
  let ab ← GraphM.add a b
  GraphM.add ab c

private def checkBranch : IO Unit := do
  let (output, state) ← IO.ofExcept (GraphM.run branch)
  let inputs : TorchLean.TensorPack Float32 [[4]] :=
    .cons (Tensor.full [4] 1.0) .nil
  let compiled ← IO.ofExcept (compileChecked state.data inputs ())
  let reference := Graph.lowerGraphDataToTape state.data inputs ()
  compare "branch forward" compiled.context.values reference.2.toShapeErasedArray
  compare "Tape adapter forward" (compiled.toTape.nodes.map (·.value))
    reference.2.toShapeErasedArray
  let idx ← IO.ofExcept (GraphM.mkIdx (Γ := [[4]]) state.nodeShapes output)
  let seed := TensorPack.single idx (Tensor.full [4] (1.0 : Float32))
  let gradients ← checkSeed "branch" compiled reference.1 seed
  expect "reverse branch order gives 1; creation order would give 0"
    (bits gradients[0]! == Array.replicate 4 (1.0 : Float32).toBits)
  let graph : Torch.TypedGraph Float32 [[4]] [4] :=
    { nodeShapes := state.nodeShapes, data := state.data, output := idx }
  let (checkedGradients, checkedValue) ← IO.ofExcept
    (graph.vjpChecked inputs () (Tensor.full [4] 1.0))
  compare "maintained checked API" checkedGradients.toShapeErasedArray
    (gradients.extract 0 1)
  compare "maintained pure API"
    (Torch.TypedGraph.vjp graph inputs (Tensor.full [4] 1.0)).toShapeErasedArray
    (gradients.extract 0 1)
  compare "maintained checked output" #[Spec.SomeTensor.ofTensor checkedValue]
    #[Spec.SomeTensor.ofTensor (Proofs.getIdx reference.2 idx)]
  let inputIdx ← IO.ofExcept
    (GraphM.mkIdx (Γ := [[4]]) state.nodeShapes (GraphM.Var.mk (s := [4]) 0))
  let special : Tensor Float32 [4] :=
    Tensor.Internal.Rep.ofArray
      #[Float32.ofBits 0x80000000, Float32.ofBits 1, -1.0, 16777216.0] (by decide)
  let _ ← checkSeed "input seed and signed zero" compiled reference.1
    (TensorPack.single inputIdx special)
  let intermediate ← IO.ofExcept
    (GraphM.mkIdx (Γ := [[4]]) state.nodeShapes (GraphM.Var.mk (s := [4]) 2))
  let _ ← checkSeed "intermediate seed" compiled reference.1
    (TensorPack.single intermediate special)

private def checkRepeatedParent : IO Unit := do
  let program : GraphM.M Float32 [[4]] (GraphM.Var [4]) := do
    let x ← GraphM.arg 0 [4]
    GraphM.add x x
  let (output, state) ← IO.ofExcept (GraphM.run program)
  let inputs : TorchLean.TensorPack Float32 [[4]] := .cons (Tensor.full [4] 1.0) .nil
  let compiled ← IO.ofExcept (compileChecked state.data inputs ())
  let reference := Graph.lowerGraphDataToTape state.data inputs ()
  let outputIdx ← IO.ofExcept (GraphM.mkIdx (Γ := [[4]]) state.nodeShapes output)
  let inputIdx ← IO.ofExcept
    (GraphM.mkIdx (Γ := [[4]]) state.nodeShapes (GraphM.Var.mk (s := [4]) 0))
  let seed := TorchLean.TensorPack.add
    (TensorPack.single inputIdx (Tensor.full [4] (16777216.0 : Float32)))
    (TensorPack.single outputIdx (Tensor.full [4] (1.0 : Float32)))
  let gradients ← checkSeed "repeated parent" compiled reference.1 seed
  expect "local VJP sums before adding to a seeded parent"
    (bits gradients[0]! == Array.replicate 4 (16777218.0 : Float32).toBits)
  let cancellation : GraphM.M Float32 [[4]] (GraphM.Var [4]) := do
    let x ← GraphM.arg 0 [4]
    let xx ← GraphM.mul x x
    let difference ← GraphM.sub xx xx
    GraphM.add difference x
  let (output, state) ← IO.ofExcept (GraphM.run cancellation)
  let input : Tensor Float32 [4] := Tensor.Internal.Rep.ofArray
    #[Float32.ofBits 0x80000000, Float32.ofBits 1, -3.0, 0.5] (by decide)
  let inputs : TorchLean.TensorPack Float32 [[4]] := .cons input .nil
  let compiled ← IO.ofExcept (compileChecked state.data inputs ())
  let reference := Graph.lowerGraphDataToTape state.data inputs ()
  compare "repeated multiply/subtract forward" compiled.context.values
    reference.2.toShapeErasedArray
  let idx ← IO.ofExcept (GraphM.mkIdx (Γ := [[4]]) state.nodeShapes output)
  let _ ← checkSeed "repeated multiply/subtract" compiled reference.1
    (TensorPack.single idx (Tensor.full [4] (1.0 : Float32)))

private def checkZeroCotangent : IO Unit := do
  let program : GraphM.M Float32 [[4]] (GraphM.Var [4]) := do
    let x ← GraphM.arg 0 [4]
    let _ ← GraphM.inv x
    GraphM.relu x
  let (output, state) ← IO.ofExcept (GraphM.run program)
  let inputs : TorchLean.TensorPack Float32 [[4]] := .cons (Tensor.full [4] 0.0) .nil
  let compiled ← IO.ofExcept (compileChecked state.data inputs ())
  let reference := Graph.lowerGraphDataToTape state.data inputs ()
  compare "singular forward" compiled.context.values reference.2.toShapeErasedArray
  let idx ← IO.ofExcept (GraphM.mkIdx (Γ := [[4]]) state.nodeShapes output)
  let gradients ← checkSeed "zero cotangent singular VJP" compiled reference.1
    (TensorPack.single idx (Tensor.full [4] (1.0 : Float32)))
  expect "the disconnected singular VJP is evaluated"
    ((Storage.toArray gradients[0]!.tensor.buffer).all Float32.isNaN)

private def checkCustomDense : IO Unit := do
  let node : NodeData Float32 Unit [[4]] [4] :=
    { forward := fun ctx _ => Proofs.getIdx ctx ⟨0, rfl⟩
      jvp := fun _ ctx _ => Proofs.getIdx ctx ⟨0, rfl⟩
      vjp := fun _ _ _ => .cons (Tensor.full [4] 3.0) .nil }
  let graph : Proofs.Autograd.Algebra.GraphData Float32 Unit [[4]] [[4]] := .snoc .nil node
  let inputs : TorchLean.TensorPack Float32 [[4]] := .cons (Tensor.full [4] 2.0) .nil
  let compiled ← IO.ofExcept (compileChecked graph inputs ())
  let reference := Graph.lowerGraphDataToTape graph inputs ()
  let gradients ← checkSeed "custom dense VJP at zero cotangent" compiled reference.1
    (TorchLean.TensorPack.zero (α := Float32) (ss := [[4], [4]]))
  expect "custom VJP contribution is retained"
    (bits gradients[0]! == Array.replicate 4 (3.0 : Float32).toBits)
  let output : Proofs.Idx ([[4]] ++ [[4]]) [4] := ⟨⟨1, by decide⟩, rfl⟩
  let gradients := compiled.backwardDenseAllFrom output (Tensor.full [4] 0.0)
  expect "a reached custom VJP is evaluated even at zero cotangent"
    (bits gradients[0]! == Array.replicate 4 (3.0 : Float32).toBits)

private def checkDisconnected : IO Unit := do
  let program : GraphM.M Float32 [[4]] (GraphM.Var [4]) := do
    let x ← GraphM.arg 0 [4]
    let _ ← GraphM.exp x
    GraphM.scale x 2.0
  let graph ← IO.ofExcept (Torch.lowerToTypedGraph program)
  let inputs : TorchLean.TensorPack Float32 [[4]] := .cons (Tensor.full [4] 1000.0) .nil
  let expected := #[Spec.SomeTensor.ofTensor (Tensor.full [4] (2.0 : Float32))]
  let seed : Tensor Float32 [4] := Tensor.full [4] 1.0
  compare "unused overflowing exp: pure VJP"
    (Torch.TypedGraph.vjp graph inputs seed).toShapeErasedArray expected
  let (gradient, output) ← IO.ofExcept (graph.vjpChecked inputs () seed)
  compare "unused overflowing exp: checked VJP" gradient.toShapeErasedArray expected
  expect "unused overflowing exp: forward output" (bits (Spec.SomeTensor.ofTensor output) ==
    Array.replicate 4 (2000.0 : Float32).toBits)
  -- Binary16's numerical equality identifies signed zeros and rejects NaN equality. The saved
  -- support program instead needs the decidable equality of FloatLib's encoded carrier.
  let program : GraphM.M (FloatLib.Floats.ExecFloat.Binary 5 10) [[1]] (GraphM.Var [1]) := do
    let x ← GraphM.arg 0 [1]
    let _ ← GraphM.inv x
    GraphM.scale x 2
  let graph ← IO.ofExcept (Torch.lowerToTypedGraph program)
  let inputs : TorchLean.TensorPack (FloatLib.Floats.ExecFloat.Binary 5 10) [[1]] :=
    .cons (Tensor.full [1] 0) .nil
  let seed : Tensor (FloatLib.Floats.ExecFloat.Binary 5 10) [1] := Tensor.full [1] 1
  let pureGradient := Torch.TypedGraph.vjp graph inputs seed
  let (checkedGradient, _) ← IO.ofExcept (graph.vjpChecked inputs () seed)
  for gradient in #[pureGradient, checkedGradient] do
    let value := (Proofs.getIdx gradient ⟨0, rfl⟩).getScalar ⟨0, by decide⟩
    expect "configured scalar: disconnected reciprocal does not produce NaN" (value == 2)

private def checkValidation : IO Unit := do
  let program : GraphM.M Float32 [[4]] (GraphM.Var [4]) := do
    let x ← GraphM.arg 0 [4]
    GraphM.log x
  let (_, state) ← IO.ofExcept (GraphM.run program)
  for value in #[0.0, -1.0, Float32.ofBits 0x7fc00000] do
    let inputs : TorchLean.TensorPack Float32 [[4]] := .cons (Tensor.full [4] value) .nil
    match compileChecked state.data inputs (), lowerToTapeChecked state.data inputs () with
    | .error actual, .error expected => expect "validation error text" (actual == expected)
    | _, _ => throw <| IO.userError "invalid logarithm must fail in both lowering paths"
  let inputs : TorchLean.TensorPack Float32 [[4]] := .cons (Tensor.full [4] 2.0) .nil
  let compiled ← IO.ofExcept (compileChecked state.data inputs ())
  let (_, expected) ← IO.ofExcept (lowerToTapeChecked state.data inputs ())
  compare "valid logarithm forward" compiled.context.values expected.toShapeErasedArray

def run : IO Unit := do
  checkBranch
  checkRepeatedParent
  checkZeroCotangent
  checkCustomDense
  checkDisconnected
  checkValidation
  IO.println "TypedGraph semantic regressions passed: canonical FP32 toBits \
    (including signed zeros), branch order, repeated parents, \
    input/intermediate seeds, zero-cotangent singular/custom VJPs, validation, repeated backward, \
    public checked/pure APIs, disconnected FP32/configured VJPs"

end TypedGraphSemanticRegression
