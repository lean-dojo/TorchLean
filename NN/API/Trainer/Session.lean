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
  let loss ← session.step (sampleAt step) (loss := true)
  if step % 100 = 0 then IO.println s!"step {step}: loss={loss}"
let after ← session.eval data
let trained ← session.finish { before, after }
```

Opening a session instantiates the model under the trainer's runtime settings and binds the
trainer's optimizer. `step` applies one update; set `loss := true` to read its loss. `predict`,
`loss`, and `eval` run the current parameters in evaluation mode; `state`, `save`, and `load`
read or replace the parameters; `finish` snapshots the current state as an ordinary `Result`.

`trainer.open` uses `Float` at the boundary and executes in the binary32 scalar selected by
`RunConfig.arithmetic`. `trainer.openTyped (α := α)` retains the selected scalar through inputs,
state, predictions, losses, reports, and checkpoints on CPU. Both use the same session and snapshot
implementation. `Trainer.train` is implemented as `open`, a loop, and `finish`.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

/--
A trainer with its model instantiated and the optimizer loop owned by the caller.

Obtain one with `trainer.open` or `trainer.openTyped`. The session type is indexed by the trainer it
runs, so the state layout `nn.stateShapes trainer.model` is available to `state` and `load`. Updates
run stateful layers in training mode, while `predict`, `loss`, and `eval` use evaluation mode.
-/
structure Session {σ τ : Shape} (trainer : TorchLean.Trainer σ τ)
    (α : Type := Float) [Storage α] where
  private mk ::
  private stepWithLossImpl : Array (Sample.Supervised α σ τ) → IO α
  private stepImpl : Array (Sample.Supervised α σ τ) → IO Unit
  private stepsImpl : IO Nat
  private lossImpl : Sample.Supervised α σ τ → IO α
  private predictImpl : Tensor α σ → IO (Tensor α τ)
  private stateImpl : IO (nn.State α (nn.stateShapes trainer.model))
  private setStateImpl : nn.State α (nn.stateShapes trainer.model) → IO Unit
  private finishImpl : Training.LossProgress α → IO (Result σ τ α)

namespace Session

variable {α : Type} [Storage α]

namespace Internal

/-- Construct a session at the trainer implementation boundary. -/
opaque create {σ τ : Shape} (trainer : TorchLean.Trainer σ τ)
    (stepWithLoss : Array (Sample.Supervised α σ τ) → IO α)
    (step : Array (Sample.Supervised α σ τ) → IO Unit)
    (steps : IO Nat)
    (loss : Sample.Supervised α σ τ → IO α)
    (predict : Tensor α σ → IO (Tensor α τ))
    (state : IO (nn.State α (nn.stateShapes trainer.model)))
    (setState : nn.State α (nn.stateShapes trainer.model) → IO Unit)
    (finish : Training.LossProgress α → IO (Result σ τ α)) :
    Session trainer α :=
  ⟨stepWithLoss, step, steps, loss, predict, state, setState, finish⟩

/-- Replace the live parameters and buffers in the session's scalar. -/
opaque setState {σ τ : Shape} {trainer : TorchLean.Trainer σ τ}
    (session : Session trainer α)
    (state : nn.State α (nn.stateShapes trainer.model)) : IO Unit :=
  session.setStateImpl state

end Internal

/-- The trainer whose model, objective, and runtime settings this session runs. -/
def trainer {σ τ : Shape} {trainer : TorchLean.Trainer σ τ} (_ : Session trainer α) :
    TorchLean.Trainer σ τ :=
  trainer

/--
Apply one optimizer update, optionally returning the loss.

Pass one sample, or use `batch := true` with a nonempty array. A batch averages
per-sample gradients at the same parameter point and counts as one update. `loss := true`
returns the mean loss from those forwards; the default avoids reading the loss back.

Example:
```lean
-- One update per call. Set `batch := true` to average several samples in one update.
def descend (trainer : TorchLean.Trainer [2] [1]) : IO (Trainer.Result [2] [1]) := do
  let session ← trainer.open
  let sample : Sample.Supervised Float [2] [1] := { input := [1.0, 0.0], target := [1.0] }
  let before ← session.loss sample
  for _ in List.range 100 do
    session.step sample
  let after ← session.loss sample
  session.finish { before := before, after := after }
```
-/
opaque step {σ τ : Shape} {trainer : TorchLean.Trainer σ τ} {Input : Type}
    (session : Session trainer α) (sample : Input) (batch : Bool := false) (loss : Bool := false)
    [Trainer.Internal.BatchInput (Sample.Supervised α σ τ)
      (Array (Sample.Supervised α σ τ)) batch Input] :
    IO (match loss with | false => Unit | true => α) := by
  have inputType := Trainer.Internal.BatchInput.type_eq (single := Sample.Supervised α σ τ)
    (many := Array (Sample.Supervised α σ τ)) (batch := batch)
  subst Input
  let samples : Array (Sample.Supervised α σ τ) := by
    cases batch with
    | false => exact #[sample]
    | true => exact sample
  cases loss with
  | false => exact session.stepImpl samples
  | true => exact session.stepWithLossImpl samples

/-- Number of optimizer updates applied so far. -/
opaque steps {σ τ : Shape} {trainer : TorchLean.Trainer σ τ}
    (session : Session trainer α) : IO Nat :=
  session.stepsImpl

namespace Internal

/-- Evaluate a lazy sample stream in index order, returning zero for an empty stream. -/
opaque evaluate [Context α] {σ τ : Shape} {trainer : TorchLean.Trainer σ τ}
    (session : Session trainer α)
    (samples : Data.SampleStream (Sample.Supervised α σ τ)) : IO α := do
  if samples.isEmpty then
    pure 0
  else
    let mut total : α := 0
    for h : i in [0:samples.size] do
      have hi : i < samples.size := h.2.1
      total := total + (← session.lossImpl (samples.get ⟨i, hi⟩))
    pure (total / (samples.size : α))

end Internal

/--
Evaluation-mode loss under the current parameters.

`batch := true` takes a lazy sample stream and returns its ordered mean, or zero when empty.
-/
opaque loss [Context α] {σ τ : Shape} {trainer : TorchLean.Trainer σ τ} {Input : Type}
    (session : Session trainer α) (sample : Input) (batch : Bool := false)
    [Trainer.Internal.BatchInput (Sample.Supervised α σ τ)
      (Data.SampleStream (Sample.Supervised α σ τ)) batch Input] : IO α := by
  have inputType := Trainer.Internal.BatchInput.type_eq (single := Sample.Supervised α σ τ)
    (many := Data.SampleStream (Sample.Supervised α σ τ)) (batch := batch)
  subst Input
  cases batch with
  | false => exact session.lossImpl sample
  | true => exact Internal.evaluate session sample

/--
Evaluation-mode prediction under the current parameters.

With `batch := true`, map over the leading axis of length `batchSize`, including an empty axis.
-/
opaque predict {σ τ inputShape : Shape} {trainer : TorchLean.Trainer σ τ}
    (session : Session trainer α) (input : Tensor α inputShape)
    (batch : Bool := false) (batchSize : Nat := 1)
    [Trainer.Internal.BatchInput
      (Tensor α σ) (Tensor α (σ.prependDim batchSize)) batch (Tensor α inputShape)] :
    IO (Tensor α (match batch with | false => τ | true => τ.prependDim batchSize)) := by
  have inputType := Trainer.Internal.BatchInput.type_eq
    (single := Tensor α σ) (many := Tensor α (σ.prependDim batchSize)) (batch := batch)
  cases batch with
  | false => exact session.predictImpl (inputType.mp input)
  | true =>
      change IO (Tensor α (τ.prependDim batchSize))
      let input : Tensor α (σ.prependDim batchSize) := inputType.mp input
      exact Tensor.stackLeadingM fun index => session.predictImpl input[index]

/-- Read the current parameters and buffers in the session's scalar. -/
opaque state {σ τ : Shape} {trainer : TorchLean.Trainer σ τ} (session : Session trainer α) :
    IO (nn.State α (nn.stateShapes trainer.model)) :=
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
    (session : Session trainer α) (loss : Training.LossProgress α) : IO (Result σ τ α) :=
  session.finishImpl loss

/-- Mean loss over a dataset in evaluation mode. -/
def eval [Context α] [Runtime.FromFloat α] {σ τ : Shape} {trainer : TorchLean.Trainer σ τ}
    (session : Session trainer α)
    (data : Dataset σ τ) : IO α := do
  session.loss (← data.materialize (α := α)) (batch := true)

/-- Write the current parameters with `Checkpoint.State.save`. -/
def save [Checkpoint.Encoding α] {σ τ : Shape} {trainer : TorchLean.Trainer σ τ}
    (session : Session trainer α)
    (path : System.FilePath) : IO Unit := do
  Checkpoint.State.save trainer.model (← session.state) path

/--
Replace the current model state from a checkpoint written by `save` or `Result.save`.

Optimizer history and the completed-step count are retained. Open a new session before loading
when training should restart with a fresh optimizer and schedule.
-/
def load [Checkpoint.Encoding α] {σ τ : Shape} {trainer : TorchLean.Trainer σ τ}
    (session : Session trainer α)
    (path : System.FilePath) : IO Unit := do
  Internal.setState session (← Checkpoint.State.load trainer.model path)

end Session

namespace Internal

/-- Read a state pack using the opening path's scalar-preserving or binary32 readback operation. -/
def readStatePack {α β : Type} [Storage α] [Storage β]
    (readTensor : {shape : Shape} → Tensor α shape → IO (Tensor β shape)) :
    {shapes : List Shape} → TensorPack α shapes → IO (TensorPack β shapes)
  | [], .nil => pure .nil
  | _ :: _, .cons tensor rest => do
      let value ← readTensor tensor
      let values ← readStatePack readTensor rest
      pure (.cons value values)

/--
Build a session for a runtime scalar `α` and a public scalar `β`.

The opening path supplies the conversions and an optional verifier. Typed opening uses identity
conversions, preserving inputs, losses, and state without requiring bound-arithmetic instances.

`nospecialize` keeps this setup routine from generating a copy of the runner and stepper stack
for each runtime scalar. A session performs this setup once and reuses the bound optimizer.
-/
@[nospecialize]
def openSession {σ τ : Shape} {α β : Type}
    [TorchLean.Storage α] [Context α]
    [Runtime.FromFloat α] [Runtime.TensorTransfer α]
    [TorchLean.Storage β] [Checkpoint.Encoding β]
    (trainer : TorchLean.Trainer σ τ)
    (scheduler : Option Scheduler.Config)
    (toRuntime : β → α)
    (readTensor : {shape : Shape} → Tensor α shape → IO (Tensor β shape))
    (readScalar : α → IO β)
    (initialState? : Option (nn.State α (nn.stateShapes trainer.model)) := none)
    (scalarFormat? : Option String := none)
    (verify? : Option (nn.State α (nn.stateShapes trainer.model) →
      Tensor Float σ → Float → Verification.Norm → Verification.Property →
      Verification.Algorithm → IO Verification.Report) := none) :
    IO (Session trainer β) := do
  let runner ← Runner.instantiate trainer.model trainer.objective
    trainer.runtime.executionSettings (α := α) (initialState? := initialState?)
  let stepper ← runner.stepper trainer.runtime.optimizer scheduler
  let cast (sample : Sample.Supervised β σ τ) : Sample.Supervised α σ τ :=
    Sample.map (Tensor.map toRuntime) (Tensor.map toRuntime) sample
  let predict (input : Tensor β σ) : IO (Tensor β τ) := do
    readTensor (← runner.forward (Tensor.map toRuntime input) (mode := some .eval))
  let readState : IO (nn.State β (nn.stateShapes trainer.model)) := do
    let values ← readStatePack readTensor (nn.State.Internal.toTensorPack (← runner.state))
    pure (nn.State.Internal.fromTensorPack values)
  let finish (loss : Training.LossProgress β) : IO (Result σ τ β) := do
    let steps ← stepper.steps
    let frozenState ← runner.state
    let frozenPack ← readStatePack readTensor (nn.State.Internal.toTensorPack frozenState)
    let frozenPublicState := nn.State.Internal.fromTensorPack frozenPack
    let parameters ← Runtime.Autograd.Torch.ParamList.ofPack
      (nn.State.Internal.toTensorPack frozenState)
    let evaluator ← Runtime.Autograd.Model.Module.Evaluator.withState
      (β := Unit) (stateShapes := nn.stateShapes trainer.model) (inputShapes := [σ])
      (dataInputShapes := []) (nn.forward trainer.model (α := α) (mode := .eval))
      trainer.runtime.executionSettings parameters
    let frozenPredict := fun input => do
      let output ← evaluator.run (TensorPack.singleton (Tensor.map toRuntime input)) .nil
      readTensor output
    pure <| Result.Internal.create
      { steps := steps, loss := loss, arithmetic := trainer.runtime.arithmetic,
        scalarFormat? := scalarFormat? }
      (nn.stateShapes trainer.model)
      (pure frozenPublicState)
      (fun path => Checkpoint.State.save trainer.model frozenPublicState path)
      frozenPredict
      (verify?.map fun verify => verify frozenState)
  pure <| Session.Internal.create trainer
    (fun batch => do
      readScalar (← stepper.step (batch.map cast) (batch := true) (loss := true)))
    (fun batch => stepper.step (batch := true) (batch.map cast))
    stepper.steps
    (fun sample => do
      readScalar (← runner.loss (cast sample) (mode := some .eval)))
    predict
    readState
    (fun state => runner.setState (state.map (Tensor.map toRuntime)))
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
  session.step { input := [1.0, 0.0], target := [1.0] } (loss := true)
```
-/
def «open» {σ τ : Shape} (trainer : TorchLean.Trainer σ τ)
    (scheduler : Option Scheduler.Config := none)
    (initialState? : Option (nn.State Float (nn.stateShapes trainer.model)) := none) :
    IO (Session trainer) := do
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
  | .native =>
      Internal.openSession (α := Float32) trainer scheduler
        (Runtime.ofFloat (α := Float32)) Runtime.readFloatTensor
        Runtime.Autograd.Torch.TensorTransfer.toFloat
        (initialState?.map fun state => state.map (Tensor.map (Runtime.ofFloat (α := Float32))))
        (verify? := some (Verification.Internal.forState trainer))
  | .ieee =>
      Internal.openSession
        (α := FloatLib.Floats.ExecFloat.Binary (exponentBits := 8) (fractionBits := 23))
        trainer scheduler Runtime.ofFloat Runtime.readFloatTensor
        Runtime.Autograd.Torch.TensorTransfer.toFloat
        (initialState?.map fun state => state.map (Tensor.map Runtime.ofFloat))
        (verify? := some (Verification.Internal.forState trainer))
  | .complex =>
      throw <| IO.userError <|
        "TorchLean.Trainer: supervised training supports real arithmetic; " ++
          "complex arithmetic requires an explicit complex-valued training API"

/--
Open a CPU session whose inputs, state, predictions, and losses retain the scalar `α`.

`initialState?` supplies exact typed parameters and buffers. Without it, the model's existing seeded
`Float` initializers are converted into `α`; they do not acquire additional random precision.
Optimizer and scheduler coefficients are also configured in `Float` and converted into `α`.

The explicit scalar takes precedence over `RunConfig.arithmetic`. This path uses the maintained CPU
runtime and rejects other devices and custom backend profiles before instantiating state. It does
not narrow tensors for native kernels. Checkpoints use `Checkpoint.Encoding α`; finished results
carry no verifier, so `Result.verify` reports that verification is unavailable.
-/
def openTyped {σ τ : Shape} (trainer : TorchLean.Trainer σ τ)
    (α : Type) [Storage α] [Context α] [Runtime.FromFloat α]
    [Runtime.TensorTransfer α] [Checkpoint.Encoding α]
    (scheduler : Option Scheduler.Config := none)
    (initialState? : Option (nn.State α (nn.stateShapes trainer.model)) := none) :
    IO (Session trainer α) := do
  unless trainer.runtime.device == .cpu do
    throw <| IO.userError
      "TorchLean.Trainer.openTyped: typed sessions require CPU; CUDA/custom devices are unsupported"
  if trainer.runtime.backendProfile?.isSome then
    throw <| IO.userError
      "TorchLean.Trainer.openTyped: custom backend profiles are unsupported for typed sessions"
  Internal.openSession trainer scheduler id (fun tensor => pure tensor) pure
    initialState? (scalarFormat? := some (Checkpoint.Encoding.format (α := α)))

end Trainer

end TorchLean
