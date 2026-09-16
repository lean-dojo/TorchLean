/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Checkpoint
public import NN.API.Trainer.Results
public import NN.API.Trainer.Run
public import NN.API.Verification.Execution
public import NN.API.Trainer.Runner -- shake: keep

/-!
# Training Sessions

`trainer.train` owns the optimizer loop. A `Session` is the same trainer with the loop handed to
the caller:

```lean
let session ← trainer.open
for step in [0:steps] do
  let loss ← session.step (sampleAt step)
  if step % 100 = 0 then IO.println s!"step {step}: loss={loss}"
let after ← session.eval data
let trained ← session.finish { before, after }
```

Opening a session instantiates the model under the trainer's runtime settings and binds the
trainer's optimizer. `step` and `stepBatch` apply one update and return the loss;
`update` and `updateBatch` apply the same update without reading the loss; `predict`,
`loss`, and `eval` run the current parameters in evaluation mode; `state`, `save`, and `load`
read or replace the parameters; `finish` snapshots the current state as an ordinary `Result`.

Every public signature uses `Float`. The run executes in the binary32 scalar selected by
`RunConfig.arithmetic`, and values are converted at the boundary. `Trainer.train` is implemented
as `open`, a loop, and `finish`.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

/--
A trainer with its model instantiated and the optimizer loop owned by the caller.

Obtain one with `trainer.open`. The session type is indexed by the trainer it runs, so the state
layout `nn.stateShapes trainer.model` is available to `state` and `load`. Mode handling is
implicit: updates run stateful layers in training mode, while `predict`, `loss`, and `eval` run
them in evaluation mode.
-/
structure Session {σ τ : Shape} (trainer : TorchLean.Trainer σ τ) where
  private mk ::
  private stepBatchImpl : Array (Sample.Supervised Float σ τ) → IO Float
  private updateImpl : Array (Sample.Supervised Float σ τ) → IO Unit
  private stepsImpl : IO Nat
  private lossImpl : Sample.Supervised Float σ τ → IO Float
  private predictImpl : Tensor Float σ → IO (Tensor Float τ)
  private stateImpl : IO (nn.State Float (nn.stateShapes trainer.model))
  private setStateImpl : nn.State Float (nn.stateShapes trainer.model) → IO Unit
  private finishImpl : Training.LossProgress Float → IO (Result σ τ)

namespace Session

namespace Internal

/-- Construct a session at the trainer implementation boundary. -/
opaque create {σ τ : Shape} (trainer : TorchLean.Trainer σ τ)
    (stepBatch : Array (Sample.Supervised Float σ τ) → IO Float)
    (update : Array (Sample.Supervised Float σ τ) → IO Unit)
    (steps : IO Nat)
    (loss : Sample.Supervised Float σ τ → IO Float)
    (predict : Tensor Float σ → IO (Tensor Float τ))
    (state : IO (nn.State Float (nn.stateShapes trainer.model)))
    (setState : nn.State Float (nn.stateShapes trainer.model) → IO Unit)
    (finish : Training.LossProgress Float → IO (Result σ τ)) :
    Session trainer :=
  ⟨stepBatch, update, steps, loss, predict, state, setState, finish⟩

/-- Replace the live parameters and buffers from host `Float` tensors. -/
opaque setState {σ τ : Shape} {trainer : TorchLean.Trainer σ τ}
    (session : Session trainer)
    (state : nn.State Float (nn.stateShapes trainer.model)) : IO Unit :=
  session.setStateImpl state

end Internal

/-- The trainer whose model, objective, and runtime settings this session runs. -/
def trainer {σ τ : Shape} {trainer : TorchLean.Trainer σ τ} (_ : Session trainer) :
    TorchLean.Trainer σ τ :=
  trainer

/--
Apply one optimizer update on a single sample and return its loss.

Example:
```lean
-- One update per call, returning that sample's loss. Batching several samples into a single
-- update is `stepBatch`, not repeated `step` calls.
def descend (trainer : TorchLean.Trainer [2] [1]) : IO (Trainer.Result [2] [1]) := do
  let session ← trainer.open
  let sample : Sample.Supervised Float [2] [1] := { input := [1.0, 0.0], target := [1.0] }
  let before ← session.loss sample
  for _ in List.range 100 do
    let _ ← session.step sample
  let after ← session.loss sample
  session.finish { before := before, after := after }
```
-/
opaque step {σ τ : Shape} {trainer : TorchLean.Trainer σ τ} (session : Session trainer)
    (sample : Sample.Supervised Float σ τ) : IO Float :=
  session.stepBatchImpl #[sample]

/--
Apply one optimizer update on a nonempty batch and return the mean loss.

The per-sample gradients are averaged at the same parameter point, so the batch counts as a
single optimizer step. An empty batch is rejected.
-/
opaque stepBatch {σ τ : Shape} {trainer : TorchLean.Trainer σ τ}
    (session : Session trainer) (batch : Array (Sample.Supervised Float σ τ)) : IO Float :=
  session.stepBatchImpl batch

/--
Apply one optimizer update without returning the loss.

Use this for unlogged steps in custom loops. Native CUDA optimizers keep gradients and moments
on device and avoid the loss readback performed by `step`.
-/
opaque update {σ τ : Shape} {trainer : TorchLean.Trainer σ τ}
    (session : Session trainer) (sample : Sample.Supervised Float σ τ) : IO Unit :=
  session.updateImpl #[sample]

/--
Apply one mean-gradient update to a nonempty batch without returning the loss.

This shares the optimizer history and step counter used by `step`, `stepBatch`, and `update`.
-/
opaque updateBatch {σ τ : Shape} {trainer : TorchLean.Trainer σ τ}
    (session : Session trainer) (batch : Array (Sample.Supervised Float σ τ)) : IO Unit :=
  session.updateImpl batch

/-- Number of optimizer updates applied so far. -/
opaque steps {σ τ : Shape} {trainer : TorchLean.Trainer σ τ} (session : Session trainer) : IO Nat :=
  session.stepsImpl

/-- Loss of one sample under the current parameters in evaluation mode. -/
opaque loss {σ τ : Shape} {trainer : TorchLean.Trainer σ τ}
    (session : Session trainer) (sample : Sample.Supervised Float σ τ) : IO Float :=
  session.lossImpl sample

/-- Run one input through the current parameters in evaluation mode. -/
opaque predict {σ τ : Shape} {trainer : TorchLean.Trainer σ τ}
    (session : Session trainer) (input : Tensor Float σ) : IO (Tensor Float τ) :=
  session.predictImpl input

/-- Read the current parameters and buffers as host `Float` tensors. -/
opaque state {σ τ : Shape} {trainer : TorchLean.Trainer σ τ} (session : Session trainer) :
    IO (nn.State Float (nn.stateShapes trainer.model)) :=
  session.stateImpl

/--
Snapshot the current parameters as a trained `Result`.

The report records the number of updates applied so far, the runtime arithmetic, and the losses
supplied by the caller, typically from `eval` before and after the loop. Later session updates or
checkpoint loads do not change the result's parameters, predictions, or verification.

Example:
```lean
-- Keep a model snapshot with the losses measured for it. The session can continue training
-- without changing this result.
def package {trainer : TorchLean.Trainer [2] [1]}
    (session : Trainer.Session trainer) (before after : Float) :
    IO (Trainer.Result [2] [1]) :=
  session.finish { before := before, after := after }
```
-/
opaque finish {σ τ : Shape} {trainer : TorchLean.Trainer σ τ}
    (session : Session trainer) (loss : Training.LossProgress Float) : IO (Result σ τ) :=
  session.finishImpl loss

/-- Run several inputs through the current parameters in evaluation mode. -/
def predictMany {σ τ : Shape} {batch : Nat} {trainer : TorchLean.Trainer σ τ}
    (session : Session trainer) (inputs : Tensor Float (σ.prependDim batch)) :
    IO (Tensor Float (τ.prependDim batch)) :=
  Tensor.stackLeadingM fun index => session.predict inputs[index]

/-- Mean loss over a finite sample stream in evaluation mode; `0` for an empty stream. -/
def meanLoss {σ τ : Shape} {trainer : TorchLean.Trainer σ τ}
    (session : Session trainer)
    (samples : Data.SampleStream (Sample.Supervised Float σ τ)) : IO Float := do
  if samples.isEmpty then
    pure 0
  else
    let mut total : Float := 0
    for h : i in [0:samples.size] do
      have hi : i < samples.size := h.2.1
      total := total + (← session.loss (samples.get ⟨i, hi⟩))
    pure (total / samples.size.toFloat)

/-- Mean loss over a dataset in evaluation mode. -/
def eval {σ τ : Shape} {trainer : TorchLean.Trainer σ τ} (session : Session trainer)
    (data : Dataset σ τ) : IO Float := do
  session.meanLoss (← data.materialize (α := Float))

/-- Write the current parameters with `Checkpoint.State.save`. -/
def save {σ τ : Shape} {trainer : TorchLean.Trainer σ τ} (session : Session trainer)
    (path : System.FilePath) : IO Unit := do
  Checkpoint.State.save trainer.model (← session.state) path

/--
Replace the current model state from a checkpoint written by `save` or `Result.save`.

Optimizer history and the completed-step count are retained. Open a new session before loading
when training should restart with a fresh optimizer and schedule.
-/
def load {σ τ : Shape} {trainer : TorchLean.Trainer σ τ} (session : Session trainer)
    (path : System.FilePath) : IO Unit := do
  Internal.setState session (← Checkpoint.State.load trainer.model path)

end Session

namespace Internal

/--
Build a session for one runtime scalar.

The scalar must support host readback and the verifier's bound arithmetic so that the finished
result can predict, save, and verify.

`open` calls this at two concrete scalars, and without `nospecialize` the compiler generates a
specialized copy of the whole runner and stepper stack for each of them. That accounted for 14 of
the 22 seconds this module used to take. Session setup runs once per training run, so the generic
version costs nothing that matters.
-/
@[nospecialize]
def openSession {σ τ : Shape} {α : Type}
    [TorchLean.Storage α] [Context α] [ToString α]
    [Runtime.FromFloat α] [Runtime.TensorTransfer α]
    [NN.MLTheory.CROWN.BoundOps α] [NN.MLTheory.CROWN.NonlinearBoundOps α]
    (trainer : TorchLean.Trainer σ τ)
    (scheduler : Option Scheduler.Config) : IO (Session trainer) := do
  let runner ← Runner.instantiate trainer.model trainer.objective
    trainer.runtime.executionSettings (α := α)
  let stepper ← runner.stepper trainer.runtime.optimizer scheduler
  let cast (sample : Sample.Supervised Float σ τ) : Sample.Supervised α σ τ :=
    Sample.map (Tensor.map (Runtime.ofFloat (α := α))) (Tensor.map (Runtime.ofFloat (α := α)))
      sample
  let predict (input : Tensor Float σ) : IO (Tensor Float τ) := do
    Runtime.readFloatTensor (← runner.predict (Tensor.map (Runtime.ofFloat (α := α)) input))
  let readState := runner.floatState
  let finish (loss : Training.LossProgress Float) : IO (Result σ τ) := do
    let steps ← stepper.steps
    let frozenState ← runner.state
    let frozenFloatPack ← Runner.readFloatPack (nn.State.Internal.toTensorPack frozenState)
    let frozenFloatState := nn.State.Internal.fromTensorPack frozenFloatPack
    let frozenPredict ←
      if trainer.runtime.device == .cuda then do
        let parameters ← Runtime.Autograd.Torch.ParamList.ofPack
          (nn.State.Internal.toTensorPack frozenState)
        let evaluator ← Runtime.Autograd.Model.Module.Evaluator.withState
          (β := Unit) (stateShapes := nn.stateShapes trainer.model) (inputShapes := [σ])
          (dataInputShapes := []) (nn.forward trainer.model (α := α))
          trainer.runtime.executionSettings parameters
        pure fun input => do
          let output ← evaluator.run
            (TensorPack.singleton (Tensor.map (Runtime.ofFloat (α := α)) input)) .nil
          Runtime.readFloatTensor output
      else do
        let graph ← nn.lowerToTypedGraph trainer.model (α := α) (mode := .eval)
        pure fun input =>
          Runtime.readFloatTensor <| nn.TypedGraphModel.forward graph frozenState
            (Tensor.map (Runtime.ofFloat (α := α)) input)
    pure <| Result.Internal.create
      { steps := steps, loss := loss, arithmetic := trainer.runtime.arithmetic }
      (nn.stateShapes trainer.model)
      (pure frozenFloatState)
      (fun path => Checkpoint.State.save trainer.model frozenFloatState path)
      frozenPredict
      (Verification.Internal.forState trainer frozenState)
  pure <| Session.Internal.create trainer
    (fun batch => do
      Runtime.Autograd.Torch.TensorTransfer.toFloat (← stepper.stepBatch (batch.map cast)))
    (fun batch => stepper.update (batch.map cast))
    stepper.steps
    (fun sample => do
      Runtime.Autograd.Torch.TensorTransfer.toFloat
        (← runner.sampleLossWithMode .eval (cast sample)))
    predict
    readState
    runner.setFloatState
    finish

end Internal

/--
Instantiate the model under the trainer's runtime settings and hand the optimizer loop to the
caller.

The trainer's optimizer is bound immediately; `scheduler` optionally adjusts its learning rate by
completed step. Optimizer and scheduler settings must remain valid after binary32 conversion and
are checked before the model is instantiated. CUDA execution requires `.native` arithmetic, and
`.complex` arithmetic is not supported by supervised training.

Example:
```lean
-- A session holds the instantiated model and the bound optimizer. `trainer.train` is exactly this
-- call, a loop of `step`, and a `finish`, so opening a session is how you take that loop over.
def stepOnce (trainer : TorchLean.Trainer [2] [1]) : IO Float := do
  let session ← trainer.open
  session.step { input := [1.0, 0.0], target := [1.0] }
```
-/
def «open» {σ τ : Shape} (trainer : TorchLean.Trainer σ τ)
    (scheduler : Option Scheduler.Config := none) : IO (Session trainer) := do
  match trainer.runtime.optimizer.validateFloat32 with
  | .ok () => pure ()
  | .error message => throw <| IO.userError message
  match scheduler with
  | some schedule =>
      match Scheduler.validateFloat32 schedule with
      | .ok () => pure ()
      | .error message => throw <| IO.userError message
  | none => pure ()
  if trainer.runtime.executionSettings.usesCuda && trainer.runtime.arithmetic != .native then
    throw <| IO.userError
      "TorchLean.Trainer: CUDA execution currently requires --arithmetic native"
  match trainer.runtime.arithmetic with
  | .native => Internal.openSession (α := Float32) trainer scheduler
  | .ieee =>
      Internal.openSession
        (α := FloatLib.Floats.ExecFloat.Binary (exponentBits := 8) (fractionBits := 23))
        trainer scheduler
  | .complex =>
      throw <| IO.userError <|
        "TorchLean.Trainer: supervised training supports real arithmetic; " ++
          "complex arithmetic requires an explicit complex-valued training API"

end Trainer

end TorchLean
