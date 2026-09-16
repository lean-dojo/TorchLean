/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tests.Runtime.Floats.Utils
public import NN.API.Seeded
public import NN.Runtime.Autograd.Model.Session.Autograd
public import NN.Runtime.Autograd.Model.Session.ShapeIndex

/-!
# TorchLeanIndexShapeCheck

Runtime checks for TorchLean indexing and shape helpers over floats.

These checks focus on shape-manipulating ops and indexing helpers that are easy to break when
refactoring tensor APIs.
-/

@[expose] public section

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Tests.Utils
open Tests.Floats.Utils

namespace Tests
namespace Floats
namespace TorchLeanIndexShapeCheck

/-- A task head preserves every leading axis rather than treating only axis zero as a batch. -/
def multiLeadingClassifier :
    TorchLean.nn.Sequential [2, 3, 4, 5] [2, 3, 7] :=
  TorchLean.nn.build 0 <|
    TorchLean.nn.heads.classifier
      (batchShape := [2, 3]) (featureShape := [4, 5]) 7

/-- Check that prefix flattening preserves every leading slice and row-major suffix order. -/
def checkFlattenThenTake : IO Unit := do
  let x : Tensor Float [2, 2, 2, 2] :=
    Tensor.generate [2, 2, 2, 2] fun coordinate =>
      Float.ofNat (100 * coordinate.getD 0 0 + 10 * coordinate.getD 1 0 +
        2 * coordinate.getD 2 0 + coordinate.getD 3 0)
  let y : Tensor Float [2, 2, 3] :=
    TorchLean.Tensor.flattenThenTake [2, 2] 3 (by decide) x
  for i in List.finRange 2 do
    for j in List.finRange 2 do
      let row := Spec.get (Spec.get y i) j
      for k in List.finRange 3 do
        assertApprox s!"flattenThenTake[{i.val},{j.val},{k.val}]"
          (vecVal row k) (Float.ofNat (100 * i.val + 10 * j.val + k.val))

def gradSelect (execution : Runtime.Autograd.Torch.ExecutionMode) :
    IO (Tensor Float [3]) := do
  let sess ← Runtime.Autograd.Model.Session.new (α := Float)
    (options := { execution := execution })
  let xVal : Tensor Float [3] := [1.0, 2.0, 3.0]
  let x : Runtime.Autograd.Torch.TensorRef Float [3] ←
    Runtime.Autograd.Model.Session.input sess xVal
      (name := some "x") (requiresGrad := true)
  let y : Runtime.Autograd.Torch.TensorRef Float Shape.scalar ←
    Runtime.Autograd.Model.Session.select
      (α := Float) (shape := Shape.ofList [3]) sess 0 x ⟨1, by decide⟩
  let grads ← Runtime.Autograd.Model.Session.backwardScalarDenseAll sess y
  Runtime.Autograd.Model.Session.grad sess grads x

def gradIndexSelectVector (execution : Runtime.Autograd.Torch.ExecutionMode) :
    IO (Tensor Float [3]) := do
  let sess ← Runtime.Autograd.Model.Session.new (α := Float)
    (options := { execution := execution })
  let xVal : Tensor Float [3] := [1.0, 2.0, 3.0]
  let indices : Fin 3 → Fin 3 :=
    ![⟨2, by decide⟩, ⟨0, by decide⟩, ⟨2, by decide⟩]
  let idx : Tensor (Fin 3) [3] := Tensor.ofFn indices
  let x : Runtime.Autograd.Torch.TensorRef Float [3] ←
    Runtime.Autograd.Model.Session.input sess xVal
      (name := some "x") (requiresGrad := true)
  let y ← Runtime.Autograd.Model.Session.indexSelect
    (α := Float) (shape := Shape.ofList [3]) sess 0 3 x idx
  let total ← Runtime.Autograd.Model.Session.sum sess (sh := [3]) y
  let grads ← Runtime.Autograd.Model.Session.backwardScalarDenseAll sess total
  Runtime.Autograd.Model.Session.grad sess grads x

def gradIndexSelectRows (execution : Runtime.Autograd.Torch.ExecutionMode) :
    IO (Tensor Float [3, 2]) :=
  do
  let sess ← Runtime.Autograd.Model.Session.new (α := Float)
    (options := { execution := execution })
  let xVal : Tensor Float [3, 2] :=
    Tensor.generate [3, 2] fun coordinate =>
      Float.ofNat (coordinate.getD 0 0 * 10 + coordinate.getD 1 0 + 1)
  let idx : Tensor (Fin 3) [2] :=
    Tensor.ofFn (fun _ => ⟨2, by decide⟩)
  let x : Runtime.Autograd.Torch.TensorRef Float [3, 2] ←
    Runtime.Autograd.Model.Session.input sess xVal
      (name := some "x") (requiresGrad := true)
  let y ← Runtime.Autograd.Model.Session.indexSelect
    (α := Float) (shape := Shape.ofList [3, 2]) sess 0 2 x idx
  let total ← Runtime.Autograd.Model.Session.sum sess (sh := [2, 2]) y
  let grads ← Runtime.Autograd.Model.Session.backwardScalarDenseAll sess total
  Runtime.Autograd.Model.Session.grad sess grads x

def gradBroadcastScalar (execution : Runtime.Autograd.Torch.ExecutionMode) : IO Float := do
  let sess ← Runtime.Autograd.Model.Session.new (α := Float)
    (options := { execution := execution })
  let sVal : Tensor Float Shape.scalar := Tensor.scalar 2.0
  let sRef ←
    Runtime.Autograd.Model.Session.input sess sVal (name := some "s") (requiresGrad := true)
  let cb : Shape.CanBroadcastTo Shape.scalar [4] :=
    Shape.CanBroadcastTo.scalarTo [4]
  let v ← Runtime.Autograd.Model.Session.broadcastTo sess
    (sh1 := Shape.scalar) (sh2 := [4]) cb sRef
  let total ← Runtime.Autograd.Model.Session.sum sess (sh := [4]) v
  let grads ← Runtime.Autograd.Model.Session.backwardScalarDenseAll sess total
  let dsT : Tensor Float Shape.scalar ← Runtime.Autograd.Model.Session.grad
    sess (sh := Shape.scalar) grads sRef
  pure (scalarVal dsT)

def gradReshapeMat (execution : Runtime.Autograd.Torch.ExecutionMode) :
    IO (Tensor Float [2, 3]) := do
  let sess ← Runtime.Autograd.Model.Session.new (α := Float)
    (options := { execution := execution })
  let xVal : Tensor Float [2, 3] :=
    Tensor.generate [2, 3] fun coordinate =>
      Float.ofNat (coordinate.getD 0 0 * 10 + coordinate.getD 1 0)
  let x : Runtime.Autograd.Torch.TensorRef Float [2, 3] ←
    Runtime.Autograd.Model.Session.input sess xVal
      (name := some "x") (requiresGrad := true)
  let h : Spec.Shape.size [2, 3] = Spec.Shape.size [6] := by decide
  let y ← Runtime.Autograd.Model.Session.reshape sess
    (sh1 := [2, 3]) (sh2 := [6]) x
    h
  let total ← Runtime.Autograd.Model.Session.sum sess (sh := [6]) y
  let grads ← Runtime.Autograd.Model.Session.backwardScalarDenseAll sess total
  Runtime.Autograd.Model.Session.grad sess grads x

def gradTransposeMat (execution : Runtime.Autograd.Torch.ExecutionMode) :
    IO (Tensor Float [2, 3]) :=
  do
  let sess ← Runtime.Autograd.Model.Session.new (α := Float)
    (options := { execution := execution })
  let xVal : Tensor Float [2, 3] :=
    Tensor.generate [2, 3] fun coordinate =>
      Float.ofNat (coordinate.getD 0 0 + coordinate.getD 1 0 + 1)
  let x : Runtime.Autograd.Torch.TensorRef Float [2, 3] ←
    Runtime.Autograd.Model.Session.input sess xVal
      (name := some "x") (requiresGrad := true)
  let xt ← Runtime.Autograd.Model.Session.swapAdjacentAtDepth sess 0 x
  let total ← Runtime.Autograd.Model.Session.sum sess (sh := [3, 2]) xt
  let grads ← Runtime.Autograd.Model.Session.backwardScalarDenseAll sess total
  Runtime.Autograd.Model.Session.grad sess grads x

def gradReduceMeanVec (execution : Runtime.Autograd.Torch.ExecutionMode) :
    IO (Tensor Float [3]) := do
  let sess ← Runtime.Autograd.Model.Session.new (α := Float)
    (options := { execution := execution })
  let xVal : Tensor Float [3] := [1.0, 2.0, 3.0]
  let x : Runtime.Autograd.Torch.TensorRef Float [3] ←
    Runtime.Autograd.Model.Session.input sess xVal
      (name := some "x") (requiresGrad := true)
  let m : Runtime.Autograd.Torch.TensorRef Float Shape.scalar ←
    Runtime.Autograd.Model.Session.reduceMean
      (α := Float) (sh := Shape.ofList [3]) sess 0 x
  let grads ← Runtime.Autograd.Model.Session.backwardScalarDenseAll sess m
  Runtime.Autograd.Model.Session.grad sess grads x

def gradScatterAddVec (execution : Runtime.Autograd.Torch.ExecutionMode) :
    IO (Tensor Float [3] × Float) :=
  do
  let sess ← Runtime.Autograd.Model.Session.new (α := Float)
    (options := { execution := execution })
  let xVal : Tensor Float [3] := [1.0, 2.0, 3.0]
  let vVal : Tensor Float [1] := [5.0]
  let idx : Tensor (Fin 3) [1] := Tensor.ofFn (fun _ => ⟨2, by decide⟩)
  let x : Runtime.Autograd.Torch.TensorRef Float [3] ←
    Runtime.Autograd.Model.Session.input sess xVal
      (name := some "x") (requiresGrad := true)
  let v : Runtime.Autograd.Torch.TensorRef Float [1] ←
    Runtime.Autograd.Model.Session.input sess vVal
      (name := some "v") (requiresGrad := true)
  let y ← Runtime.Autograd.Model.Session.scatterAdd
    (α := Float) (shape := Shape.ofList [3]) sess 0 1 x v idx
  let total ← Runtime.Autograd.Model.Session.sum sess (sh := [3]) y
  let grads ← Runtime.Autograd.Model.Session.backwardScalarDenseAll sess total
  let dx ← Runtime.Autograd.Model.Session.grad sess grads x
  let dvT : Tensor Float [1] ← Runtime.Autograd.Model.Session.grad
    sess (sh := [1]) grads v
  pure (dx, vecVal dvT ⟨0, by decide⟩)

/-- A scalar objective whose labels are bounded by the class count in their element type. -/
def boundedLabelObjective :
    Runtime.Autograd.Model.Module.ObjectiveDef (Fin 3) [[2, 3]] [] [[2]] where
  initState := .cons (Tensor.full [2, 3] 0.0) .nil
  loss := fun {α} => by
    intro _ _
    exact fun {m} _ _ =>
      fun logits => fun labels =>
        let logitsIndexed : Runtime.Autograd.Model.RefTy m α
            (Shape.concat [2] [3]) := by simpa using logits
        let labelsIndexed : Runtime.Autograd.Torch.DataRef (m := m) (α := α) (Fin 3)
            (Shape.concat [2] []) := by simpa using labels
        TorchLean.Loss.crossEntropy (m := m) (α := α)
          1 rfl logitsIndexed labelsIndexed

/-- Check that bounded labels pass through eager and typed-graph scalar objectives unchanged. -/
def boundedLabelLoss (execution : Runtime.Autograd.Torch.ExecutionMode) : IO Float := do
  let module ← TorchLean.Module.instantiate
    boundedLabelObjective { execution := execution } (α := Float)
  let logits : Tensor Float [2, 3] :=
    Tensor.generate [2, 3] fun coordinate =>
      if coordinate.getD 0 0 = coordinate.getD 1 0 then 2.0 else 0.0
  TorchLean.Module.Objective.setState module <|
    TorchLean.nn.State.empty.push logits
  let labels : Tensor (Fin 3) [2] :=
    Tensor.ofFn fun row => if row = 0 then ⟨0, by decide⟩ else ⟨2, by decide⟩
  let loss ← TorchLean.Module.Objective.loss module
    TorchLean.Arguments.empty
    (TorchLean.Arguments.empty.push labels)
  pure loss.item

/-- Check generic non-differentiable inputs under both execution modes. -/
def checkBoundedLabelObjective : IO Unit := do
  let eagerLoss ← boundedLabelLoss .eager
  let graphLoss ← boundedLabelLoss .typedGraph
  assertApprox "bounded-label loss eager/typed-graph" eagerLoss graphLoss

/-- Two invalid axes must not silently become an identity permutation. -/
def checkTransposeAxes : IO Unit := do
  let session ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float)
  let input ← session.input ([1.0, 2.0] : Tensor Float [2])
  for (first, second) in [(1, 1), (9, 10), (0, 9)] do
    let result ← Runtime.Autograd.Model.F.transpose (α := Float) (s := [2])
      (m := Runtime.Autograd.Torch.Internal.EagerM Float)
      (sOut := [2]) first second input session
    unless result.isNone do
      throw <| IO.userError "functional transpose accepted an out-of-range axis"
  let valid ← Runtime.Autograd.Model.F.transpose (α := Float) (s := [2])
    (m := Runtime.Autograd.Torch.Internal.EagerM Float) (sOut := [2]) 0 0 input session
  unless valid.isSome do
    throw <| IO.userError "functional transpose rejected the same valid axis"

def run : IO Unit := do
  IO.println "torchlean_index_shape_check: begin"
  checkTransposeAxes

  let gE ← gradSelect .eager
  let gC ← gradSelect .typedGraph
  for i in List.finRange 3 do
    assertApprox s!"gather grad[{i.val}] eager/typed-graph" (vecVal gE i) (vecVal gC i)
  for i in List.finRange 3 do
    assertApprox s!"gather grad[{i.val}] expected" (vecVal gE i) (if i.val = 1 then 1.0 else 0.0)

  let gvE ← gradIndexSelectVector .eager
  let gvC ← gradIndexSelectVector .typedGraph
  for i in List.finRange 3 do
    assertApprox s!"indexSelect vector dx[{i.val}] eager/typed-graph"
      (vecVal gvE i) (vecVal gvC i)
  assertApprox "indexSelect vector dx[0] expected" (vecVal gvE ⟨0, by decide⟩) 1.0
  assertApprox "indexSelect vector dx[1] expected" (vecVal gvE ⟨1, by decide⟩) 0.0
  assertApprox "indexSelect vector dx[2] expected" (vecVal gvE ⟨2, by decide⟩) 2.0

  let grE ← gradIndexSelectRows .eager
  let grC ← gradIndexSelectRows .typedGraph
  for i in List.finRange 3 do
    for j in List.finRange 2 do
      assertApprox s!"indexSelect rows dx[{i.val},{j.val}] eager/typed-graph"
        (matVal grE i j) (matVal grC i j)
  for j in List.finRange 2 do
    assertApprox s!"indexSelect rows dx[0,{j.val}] expected" (matVal grE ⟨0, by decide⟩ j) 0.0
    assertApprox s!"indexSelect rows dx[1,{j.val}] expected" (matVal grE ⟨1, by decide⟩ j) 0.0
    assertApprox s!"indexSelect rows dx[2,{j.val}] expected" (matVal grE ⟨2, by decide⟩ j) 2.0

  let bsE ← gradBroadcastScalar .eager
  let bsC ← gradBroadcastScalar .typedGraph
  assertApprox "broadcast scalar grad eager/typed-graph" bsE bsC
  assertApprox "broadcast scalar grad expected" bsE 4.0

  let rE ← gradReshapeMat .eager
  let rC ← gradReshapeMat .typedGraph
  for i in List.finRange 2 do
    for j in List.finRange 3 do
      assertApprox s!"reshape grad[{i.val},{j.val}] eager/typed-graph" (matVal rE i j)
        (matVal rC i j)
      assertApprox s!"reshape grad[{i.val},{j.val}] expected" (matVal rE i j) 1.0

  let tE ← gradTransposeMat .eager
  let tC ← gradTransposeMat .typedGraph
  for i in List.finRange 2 do
    for j in List.finRange 3 do
      assertApprox s!"transpose grad[{i.val},{j.val}] eager/typed-graph" (matVal tE i j)
        (matVal tC i j)
      assertApprox s!"transpose grad[{i.val},{j.val}] expected" (matVal tE i j) 1.0

  let mE ← gradReduceMeanVec .eager
  let mC ← gradReduceMeanVec .typedGraph
  for i in List.finRange 3 do
    assertApprox s!"reduce_mean grad[{i.val}] eager/typed-graph" (vecVal mE i) (vecVal mC i)
    assertApprox s!"reduce_mean grad[{i.val}] expected" (vecVal mE i) (1.0 / 3.0) 1e-6

  let (sxE, svE) ← gradScatterAddVec .eager
  let (sxC, svC) ← gradScatterAddVec .typedGraph
  for i in List.finRange 3 do
    assertApprox s!"scatter_add_vec dx[{i.val}] eager/typed-graph" (vecVal sxE i) (vecVal sxC i)
    assertApprox s!"scatter_add_vec dx[{i.val}] expected" (vecVal sxE i) 1.0
  assertApprox "scatter_add_vec dv eager/typed-graph" svE svC
  assertApprox "scatter_add_vec dv expected" svE 1.0

  checkFlattenThenTake
  checkBoundedLabelObjective

  IO.println "torchlean_index_shape_check: ok"

end TorchLeanIndexShapeCheck
end Floats
end Tests
