/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TypedGraph
public import NN.Runtime.Autograd.Torch.Core.TypedGraph
public import NN.Runtime.Autograd.Torch.TypedGraphSession.Autograd
public import NN.Runtime.Autograd.Torch.TypedGraphSession.GraphOps

/-!
Canonical `Float32.toBits` comparisons against the original dense Tape engine. These observations
distinguish signed zeros but canonicalize NaNs, so they do not test raw NaN sign or payload identity.
Branch cancellation distinguishes reverse graph order from creation order; the repeated-parent case
distinguishes adding a complete local VJP from sequentially scattering its two terms. Singular and
custom VJPs exercise nodes with zero incoming cotangent, and input/intermediate seeds exercise the
full dense interface.
-/

public section

open Spec TorchLean Runtime.Autograd
open Runtime.Autograd.TypedGraph
open Proofs.Autograd.Algebra

namespace TypedGraphScalingRegression

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
    (reference : Tape Float32) (seed : TorchLean.TensorPack Float32 (Γ ++ ss)) :
    IO (Array (Spec.SomeTensor Float32)) := do
  let expected ← okOrThrow (reference.backwardDenseFrom seed.toShapeErasedArray)
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
  let (output, state) ← okOrThrow (GraphM.run branch)
  let inputs : TorchLean.TensorPack Float32 [[4]] :=
    .cons (Tensor.full [4] 1.0) .nil
  let compiled ← okOrThrow (compileChecked state.data inputs ())
  let reference := Graph.lowerGraphDataToTape state.data inputs ()
  compare "branch forward" compiled.context.values reference.2.toShapeErasedArray
  compare "Tape adapter forward" (compiled.toTape.nodes.map (·.value))
    reference.2.toShapeErasedArray
  let idx ← okOrThrow (GraphM.mkIdx (_α := Float32) (Γ := [[4]]) state.nodeShapes output)
  let seed := TensorPack.single idx (Tensor.full [4] (1.0 : Float32))
  let gradients ← checkSeed "branch" compiled reference.1 seed
  expect "reverse branch order gives 1; creation order would give 0"
    (bits gradients[0]! == Array.replicate 4 (1.0 : Float32).toBits)
  let graph : Torch.TypedGraph Float32 [[4]] [4] :=
    { nodeShapes := state.nodeShapes, data := state.data, output := idx }
  let (checkedGradients, checkedValue) ← okOrThrow
    (graph.vjpChecked inputs () (Tensor.full [4] 1.0))
  compare "maintained checked API" checkedGradients.toShapeErasedArray
    (gradients.extract 0 1)
  compare "maintained pure API"
    (Torch.TypedGraph.vjpWithSeed graph inputs (Tensor.full [4] 1.0)).toShapeErasedArray
    (gradients.extract 0 1)
  compare "maintained checked output" #[Spec.SomeTensor.ofTensor checkedValue]
    #[Spec.SomeTensor.ofTensor (Proofs.getIdx reference.2 idx)]
  let inputIdx ← okOrThrow
    (GraphM.mkIdx (_α := Float32) (Γ := [[4]]) state.nodeShapes (GraphM.Var.mk (s := [4]) 0))
  let special : Tensor Float32 [4] :=
    Tensor.Internal.Rep.ofArray
      #[Float32.ofBits 0x80000000, Float32.ofBits 1, -1.0, 16777216.0] (by decide)
  let _ ← checkSeed "input seed and signed zero" compiled reference.1
    (TensorPack.single inputIdx special)
  let intermediate ← okOrThrow
    (GraphM.mkIdx (_α := Float32) (Γ := [[4]]) state.nodeShapes (GraphM.Var.mk (s := [4]) 2))
  let _ ← checkSeed "intermediate seed" compiled reference.1 (TensorPack.single intermediate special)

private def checkRepeatedParent : IO Unit := do
  let program : GraphM.M Float32 [[4]] (GraphM.Var [4]) := do
    let x ← GraphM.arg 0 [4]
    GraphM.add x x
  let (output, state) ← okOrThrow (GraphM.run program)
  let inputs : TorchLean.TensorPack Float32 [[4]] := .cons (Tensor.full [4] 1.0) .nil
  let compiled ← okOrThrow (compileChecked state.data inputs ())
  let reference := Graph.lowerGraphDataToTape state.data inputs ()
  let outputIdx ← okOrThrow (GraphM.mkIdx (_α := Float32) (Γ := [[4]]) state.nodeShapes output)
  let inputIdx ← okOrThrow
    (GraphM.mkIdx (_α := Float32) (Γ := [[4]]) state.nodeShapes (GraphM.Var.mk (s := [4]) 0))
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
  let (output, state) ← okOrThrow (GraphM.run cancellation)
  let input : Tensor Float32 [4] := Tensor.Internal.Rep.ofArray
    #[Float32.ofBits 0x80000000, Float32.ofBits 1, -3.0, 0.5] (by decide)
  let inputs : TorchLean.TensorPack Float32 [[4]] := .cons input .nil
  let compiled ← okOrThrow (compileChecked state.data inputs ())
  let reference := Graph.lowerGraphDataToTape state.data inputs ()
  compare "repeated multiply/subtract forward" compiled.context.values reference.2.toShapeErasedArray
  let idx ← okOrThrow (GraphM.mkIdx (_α := Float32) (Γ := [[4]]) state.nodeShapes output)
  let _ ← checkSeed "repeated multiply/subtract" compiled reference.1
    (TensorPack.single idx (Tensor.full [4] (1.0 : Float32)))

private def checkZeroCotangent : IO Unit := do
  let program : GraphM.M Float32 [[4]] (GraphM.Var [4]) := do
    let x ← GraphM.arg 0 [4]
    let _ ← GraphM.inv x
    GraphM.relu x
  let (output, state) ← okOrThrow (GraphM.run program)
  let inputs : TorchLean.TensorPack Float32 [[4]] := .cons (Tensor.full [4] 0.0) .nil
  let compiled ← okOrThrow (compileChecked state.data inputs ())
  let reference := Graph.lowerGraphDataToTape state.data inputs ()
  compare "singular forward" compiled.context.values reference.2.toShapeErasedArray
  let idx ← okOrThrow (GraphM.mkIdx (_α := Float32) (Γ := [[4]]) state.nodeShapes output)
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
  let compiled ← okOrThrow (compileChecked graph inputs ())
  let reference := Graph.lowerGraphDataToTape graph inputs ()
  let gradients ← checkSeed "custom dense VJP at zero cotangent" compiled reference.1
    (TorchLean.TensorPack.zero (α := Float32) (ss := [[4], [4]]))
  expect "custom VJP contribution is retained"
    (bits gradients[0]! == Array.replicate 4 (3.0 : Float32).toBits)

private def checkValidation : IO Unit := do
  let program : GraphM.M Float32 [[4]] (GraphM.Var [4]) := do
    let x ← GraphM.arg 0 [4]
    GraphM.log x
  let (_, state) ← okOrThrow (GraphM.run program)
  for value in #[0.0, -1.0, Float32.ofBits 0x7fc00000] do
    let inputs : TorchLean.TensorPack Float32 [[4]] := .cons (Tensor.full [4] value) .nil
    match compileChecked state.data inputs (), lowerToTapeChecked state.data inputs () with
    | .error actual, .error expected => expect "validation error text" (actual == expected)
    | _, _ => throw <| IO.userError "invalid logarithm must fail in both lowering paths"
  let inputs : TorchLean.TensorPack Float32 [[4]] := .cons (Tensor.full [4] 2.0) .nil
  let compiled ← okOrThrow (compileChecked state.data inputs ())
  let (_, expected) ← okOrThrow (lowerToTapeChecked state.data inputs ())
  compare "valid logarithm forward" compiled.context.values expected.toShapeErasedArray

private def checkSession : IO Unit := do
  let session ← Torch.Internal.TypedGraphSession.new (α := Float32)
  let x ← session.input (Tensor.full [4] (1.0 : Float32)) (requiresGrad := true)
  let frozen ← session.input (Tensor.full [4] (2.0 : Float32)) (requiresGrad := false)
  let repeated ← session.add x x
  let output ← session.add repeated frozen
  let state ← session.state.get
  let reference ← okOrThrow (Torch.Internal.TypedGraphSession.lowerTape state)
  let seed : Tensor Float32 [4] := Tensor.full [4] 1.0
  let idx ← okOrThrow (Torch.Internal.TypedGraphSession.mkIdxOrThrow
    (_α := Float32) (Γ := state.Γ) (ss := state.ss) output.id [4])
  let expected ← okOrThrow (backwardDenseAllFrom reference idx seed)
  let actual ← session.backwardDenseAll output seed
  compare "normal session with frozen input" actual expected
  compare "normal session cached forward" #[Spec.SomeTensor.ofTensor (← session.getValue output)]
    #[Spec.SomeTensor.ofTensor (Tensor.full [4] (4.0 : Float32))]
  let frozenSeed : Tensor Float32 [4] := Tensor.Internal.Rep.ofArray
    #[Float32.ofBits 0x80000000, Float32.ofBits 1, -2.0, 16777216.0] (by decide)
  let frozenIdx ← okOrThrow (Torch.Internal.TypedGraphSession.mkIdxOrThrow
    (_α := Float32) (Γ := state.Γ) (ss := state.ss) frozen.id [4])
  let expected ← okOrThrow (backwardDenseAllFrom reference frozenIdx frozenSeed)
  let actual ← session.backwardDenseAll frozen frozenSeed
  compare "explicit frozen leaf seed" actual expected
  expect "the frozen leaf retains the seed's Float32.toBits observations"
    (bits actual[frozen.id]! == (Storage.toArray frozenSeed.buffer).map Float32.toBits)

def run : IO Unit := do
  checkBranch
  checkRepeatedParent
  checkZeroCotangent
  checkCustomDense
  checkValidation
  checkSession
  IO.println "TypedGraph scaling regressions passed: canonical FP32 toBits (including signed zeros), \
    branch order, repeated parents, \
    input/intermediate seeds, zero-cotangent singular/custom VJPs, validation, repeated backward, \
    public checked/pure APIs, session frozen leaves"

end TypedGraphScalingRegression
