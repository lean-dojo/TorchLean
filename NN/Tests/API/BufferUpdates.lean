/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API

/-!
# Stateful Forward Tests

Composition wrappers report mutable buffers exactly. Runtime forwards advance dropout streams and
update BatchNorm from the activations used by that same forward, including inside residual blocks.
-/

@[expose] public section

namespace NN.Tests.API.BufferUpdates

open TorchLean

def expect (tag : String) (expected actual : Bool) : IO Unit := do
  unless actual == expected do
    throw <| IO.userError
      s!"buffer-update detection failed: {tag} (expected {expected}, got {actual})"

def hasUpdates {σ τ : Spec.Shape} (model : nn.Sequential σ τ) : Bool :=
  Runtime.Autograd.Model.Layers.Seq.hasBufferUpdates model

def affine : nn.Sequential [1] [1] :=
  nn.build 0 (nn.linear 1 1)

def normalized : nn.Sequential [1, 1, 1] [1, 1, 1] :=
  nn.Sequential.fromLayer <| Runtime.Autograd.Model.Layers.batchNorm 1 1 [1] (by decide)

def expectClose (tag : String) (expected actual : Float) : IO Unit := do
  unless Float.abs (expected - actual) < 0.00001 do
    throw <| IO.userError s!"{tag}: expected {expected}, got {actual}"

def frozenObjective : Module.ObjectiveDefinition Unit [[]] [] :=
  { initState := .cons (Tensor.scalar (2.0 : Float)) .nil
    requiresGrad := #[false]
    loss := fun {_α} _ _ => fun {_m} _ _ => fun p => Runtime.mul p p }

def checkFrozenObjective : IO Unit := do
  for execution in [Runtime.ExecutionMode.eager, .typedGraph] do
    let objective ← Module.instantiate frozenObjective { execution } (α := Float)
    let (gradient, value) ← objective.grad Arguments.empty Arguments.empty (value := true)
    expectClose "frozen objective value" 4.0 value.item
    expectClose "frozen objective gradient" 0.0 (gradient.get ⟨0, by decide⟩).item

def dropout : nn.Sequential [1, 64] [1, 64] := nn.build 0 (nn.dropout 0.5)

def normalization (momentum : Float) : nn.Sequential [1, 64] [1, 64] :=
  nn.build 0 (nn.batchNorm [64] (channels := 1) (momentum := momentum))

def accumulator : nn.Sequential [1] [1] := nn.Sequential.fromLayer
  { kind := "Accumulator"
    stateShapes := [[]]
    initState := .cons (Tensor.scalar 0.0) .nil
    requiresGrad := #[false]
    updateBuffers := some fun mode {_α} _ _ state input => do
      if mode == .train then
        pure (.cons (Tensor.scalar (state.head.item + input.getScalar ⟨0, by decide⟩)) .nil)
      else pure state
    forward := fun _ {_α} _ _ {_m} _ _ => fun _ input =>
      (pure input : _m (Runtime.Autograd.Model.RefTy _m _α [1])) }

def sharedBufferObjective : Module.ObjectiveDefinition Unit [[]] [[1]] :=
  { initState := .cons (Tensor.scalar 0.0) .nil
    requiresGrad := #[false]
    loss := fun {_α} _ _ => fun {_m} _ _ => fun state input => (do
      let refs := Runtime.Autograd.Torch.RefList.cons state .nil
      let first ← Runtime.Autograd.Model.Layers.Seq.forwardState
        (m := _m) accumulator .train refs input
      let second ← Runtime.Autograd.Model.Layers.Seq.forwardState
        (m := _m) accumulator .train refs first
      Runtime.sum second : _m (Runtime.Autograd.Model.RefTy _m _α [])) }

def checkSharedBuffer : IO Unit := do
  for execution in [Runtime.ExecutionMode.eager, .typedGraph] do
    let objective ← Module.instantiate sharedBufferObjective { execution } (α := Float)
    let inputs := Arguments.Internal.fromTensorPack <|
      TensorPack.singleton (Tensor.full [1] 2.0)
    let _ ← objective.loss inputs Arguments.empty
    expectClose "reused layer accumulates both updates" 4.0
      ((← objective.state).get ⟨0, by decide⟩).item

def checkLiveForward : IO Unit := do
  let module ← nn.Module.instantiate dropout (α := Float)
  let input : Tensor Float [1, 64] := Tensor.ones [1, 64]
  let first ← module.forward input
  let _ ← module.forward (mode := some .eval) input
  let second ← module.forward input
  expect "live dropout advances" false (first.to (Array Float) == second.to (Array Float))
  let fresh ← nn.Module.instantiate dropout (α := Float)
  let _ ← fresh.forward input
  expect "prediction preserves dropout stream" true
    (second.to (Array Float) == (← fresh.forward input).to (Array Float))
  let model := nn.residual (normalization 0.5)
  let module ← nn.Module.instantiate model (α := Float)
  let input : Tensor Float [1, 64] := Tensor.full [1, 64] 5.0
  let _ ← module.forward input
  let state ← module.state
  expectClose "nested BatchNorm updates once" 2.5
    ((state.get ⟨2, by decide⟩).to (Array Float))[0]!
  let _ ← module.forward (mode := some .eval) (Tensor.full [1, 64] 9.0)
  expectClose "prediction preserves running mean" 2.5
    (((← module.state).get ⟨2, by decide⟩).to (Array Float))[0]!
  let runner ← Trainer.Internal.Runner.instantiate dropout .mse (α := Float)
  let first ← runner.forward (Tensor.ones [1, 64])
  let second ← runner.forward (Tensor.ones [1, 64])
  expect "runner dropout advances" false (first.to (Array Float) == second.to (Array Float))

def checkStochasticBuffers (device : NN.Backend.Device := .cpu) : IO Unit := do
  let combined := nn.compose dropout (normalization 1.0)
  let sample : Sample.Supervised Float [1, 64] [1, 64] :=
    { input := Tensor.ones [1, 64], target := Tensor.zeros [1, 64] }
  let executions := if device == .cpu then
      [Runtime.ExecutionMode.eager, .typedGraph]
    else [Runtime.ExecutionMode.eager]
  for execution in executions do
    let reference ← Trainer.Internal.Runner.instantiate dropout .mse
      { execution, device } (α := Float)
    let runner ← Trainer.Internal.Runner.instantiate combined .mse
      { execution, device } (α := Float)
    let referenceStep ← reference.stepper (optim.sgd { learningRate := 0 })
    let step ← runner.stepper (optim.sgd { learningRate := 0 })
    for _ in [:4] do
      let loss ← referenceStep.step sample (loss := true)
      let _ ← step.step sample
      let state ← runner.state
      -- The dropout output is 0 or 2, so its squared mean is twice its mean.
      expectClose "BatchNorm sees the actual dropout draw" (loss / 2.0)
        ((state.get ⟨3, by decide⟩).to (Array Float))[0]!

def checkLossModes : IO Unit := do
  for execution in [Runtime.ExecutionMode.eager, .typedGraph] do
    let runner ← Trainer.Internal.Runner.instantiate (normalization 0.5) .mse
      { execution } (α := Float)
    let sample (value : Float) : Sample.Supervised Float [1, 64] [1, 64] :=
      { input := Tensor.full [1, 64] value, target := Tensor.zeros [1, 64] }
    let _ ← runner.loss (mode := some .train) (sample 5.0)
    expectClose "training loss updates running mean" 2.5
      (((← runner.state).get ⟨2, by decide⟩).to (Array Float))[0]!
    let _ ← runner.loss (mode := some .eval) (sample 9.0)
    expectClose "evaluation loss preserves running mean" 2.5
      (((← runner.state).get ⟨2, by decide⟩).to (Array Float))[0]!

def checkMappedBuffers : IO Unit := do
  let model := nn.mapLeading [2] (nn.residual (normalization 0.5))
  for execution in [Runtime.ExecutionMode.eager, .typedGraph] do
    let runner ← Trainer.Internal.Runner.instantiate model .mse
      { execution } (α := Float)
    let _ ← runner.loss (mode := some .train)
      { input := Tensor.full [2, 1, 64] 5.0, target := Tensor.zeros [2, 1, 64] }
    let state ← runner.state
    if h : 2 < (nn.stateShapes model).length then
      expectClose "mapped residual updates once per sample" 3.75
        ((state.get ⟨2, h⟩).to (Array Float))[0]!
    else
      throw <| IO.userError "mapped BatchNorm state is missing its running mean"

def run : IO Unit := do
  expect "affine" false (hasUpdates affine)
  expect "stateless residual" false (hasUpdates (nn.residual affine))
  expect "stateless branches" false (hasUpdates (nn.addBranches affine affine))
  expect "BatchNorm" true (hasUpdates normalized)
  expect "stateful residual" true (hasUpdates (nn.residual normalized))
  expect "stateful branch" true (hasUpdates (nn.addBranches normalized normalized))
  checkFrozenObjective
  checkSharedBuffer
  checkLiveForward
  checkStochasticBuffers (device := .cpu)
  checkLossModes
  checkMappedBuffers

end NN.Tests.API.BufferUpdates
