/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Checkpoint
public import TorchLean.NN.Trainer.Results
public import TorchLean.NN.Trainer.Run
public import TorchLean.NN.Trainer.Dataset
public import TorchLean.NN.Sequential.Execution
public import TorchLean.NN.IR.Lowering
public import TorchLean.NN.Trainer.Runner -- shake: keep

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
trainer's optimizer. `step` applies one update and, with `loss := true`, returns its loss.
`predict`, `loss` and `eval` run the current parameters in evaluation mode. `state`, `save` and
`load` read or replace the parameters, and `finish` snapshots the current state as a `Result`.

`trainer.open` uses `Float` at the boundary and executes in the binary32 scalar selected by
`RunConfig.arithmetic`. `trainer.openTyped (α := α)` keeps the selected scalar through inputs,
state, predictions, losses, reports and checkpoints on CPU. Both share one session and snapshot
implementation, and `Trainer.train` itself is built from `open`, a loop of `step` and `finish`.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

/--
A trainer with its model instantiated and the optimizer loop owned by the caller.

Obtain one with `trainer.open` or `trainer.openTyped`. The session type is indexed by the trainer it
runs, so the state layout `trainer.model.stateShapes` is available to `state` and `load`. Updates
run stateful layers in training mode, while `predict`, `loss`, and `eval` use evaluation mode.
-/
structure Session {σ τ : Shape} (trainer : TorchLean.Trainer σ τ)
    (α : Type := Float) [Storage α] where
  private mk ::
  private updateWithLoss : Array (Sample.Supervised α σ τ) → IO α
  private update : Array (Sample.Supervised α σ τ) → IO Unit
  private updateCount : IO Nat
  private sampleLoss : Sample.Supervised α σ τ → IO α
  private forward : Tensor α σ → IO (Tensor α τ)
  private readState : IO (TensorPack α (nn.Sequential.stateShapes trainer.model))
  private writeState : TensorPack α (nn.Sequential.stateShapes trainer.model) → IO Unit
  private snapshot : Training.LossProgress α → IO (Result σ τ α)

namespace Session

variable {α : Type} [Storage α]

/-- Construct a session at the trainer implementation boundary. -/
opaque create {σ τ : Shape} (trainer : TorchLean.Trainer σ τ)
    (stepWithLoss : Array (Sample.Supervised α σ τ) → IO α)
    (step : Array (Sample.Supervised α σ τ) → IO Unit)
    (steps : IO Nat)
    (loss : Sample.Supervised α σ τ → IO α)
    (predict : Tensor α σ → IO (Tensor α τ))
    (state : IO (TensorPack α (nn.Sequential.stateShapes trainer.model)))
    (setState : TensorPack α (nn.Sequential.stateShapes trainer.model) → IO Unit)
    (finish : Training.LossProgress α → IO (Result σ τ α)) :
    Session trainer α :=
  ⟨stepWithLoss, step, steps, loss, predict, state, setState, finish⟩

/-- Replace the live parameters and buffers in the session's scalar. -/
opaque setState {σ τ : Shape} {trainer : TorchLean.Trainer σ τ}
    (session : Session trainer α)
    (state : TensorPack α (nn.Sequential.stateShapes trainer.model)) : IO Unit :=
  session.writeState state

/--
Apply one optimizer update, optionally returning the loss.

Pass one sample, or use `batch := true` with a nonempty array. A batch averages per-sample
gradients at the same parameter point and counts as one update. `loss := true` returns the mean
loss of those forward passes, and the default skips reading the loss back.

Example:
```lean
-- 100 updates on one sample, with its evaluation-mode loss measured before and after them.
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
    [BatchInput (Sample.Supervised α σ τ)
      (Array (Sample.Supervised α σ τ)) batch Input] :
    IO (match loss with | false => Unit | true => α) := by
  have inputType := BatchInput.type_eq (single := Sample.Supervised α σ τ)
    (many := Array (Sample.Supervised α σ τ)) (batch := batch)
  subst Input
  let samples : Array (Sample.Supervised α σ τ) := by
    cases batch with
    | false => exact #[sample]
    | true => exact sample
  cases loss with
  | false => exact session.update samples
  | true => exact session.updateWithLoss samples

/-- Number of optimizer updates applied so far. -/
opaque steps {σ τ : Shape} {trainer : TorchLean.Trainer σ τ}
    (session : Session trainer α) : IO Nat :=
  session.updateCount

/-- Evaluate samples in index order and return the `Trainer.Mean.value` of their losses,
the mean that a batch update reports, so the sample count is never rounded into `α`. -/
opaque meanLoss [Context α] {σ τ : Shape} {trainer : TorchLean.Trainer σ τ}
    (session : Session trainer α)
    (samples : Data.SampleStream (Sample.Supervised α σ τ)) : IO α := do
  if h : 0 < samples.size then
    let count := samples.size
    let mut loss := Trainer.Mean.single count (← session.sampleLoss (samples.get ⟨0, h⟩))
    for h : index in [1:samples.size] do
      loss := loss.push count (← session.sampleLoss (samples.get ⟨index, h.2.1⟩))
    return loss.value count
  else
    pure 0

/--
Evaluation-mode loss under the current parameters.

`batch := true` takes a lazy sample stream and returns its ordered mean, or zero when empty.
-/
opaque loss [Context α] {σ τ : Shape} {trainer : TorchLean.Trainer σ τ} {Input : Type}
    (session : Session trainer α) (sample : Input) (batch : Bool := false)
    [BatchInput (Sample.Supervised α σ τ)
      (Data.SampleStream (Sample.Supervised α σ τ)) batch Input] : IO α := by
  have inputType := BatchInput.type_eq (single := Sample.Supervised α σ τ)
    (many := Data.SampleStream (Sample.Supervised α σ τ)) (batch := batch)
  subst Input
  cases batch with
  | false => exact session.sampleLoss sample
  | true => exact meanLoss session sample

/--
Evaluation-mode prediction under the current parameters.

With `batch := true`, map over the leading axis of length `batchSize`, including an empty axis.
-/
opaque predict {σ τ inputShape : Shape} {trainer : TorchLean.Trainer σ τ}
    (session : Session trainer α) (input : Tensor α inputShape)
    (batch : Bool := false) (batchSize : Nat := 1)
    [BatchInput
      (Tensor α σ) (Tensor α (Shape.dim batchSize σ)) batch (Tensor α inputShape)] :
    IO (Tensor α (match batch with | false => τ | true => Shape.dim batchSize τ)) :=
  Trainer.predictWith session.forward input (batch := batch) (batchSize := batchSize)

/-- Read the current parameters and buffers in the session's scalar. -/
opaque state {σ τ : Shape} {trainer : TorchLean.Trainer σ τ} (session : Session trainer α) :
    IO (TensorPack α (nn.Sequential.stateShapes trainer.model)) :=
  session.readState

/--
Snapshot the current parameters as a trained `Result`.

The report records the number of updates applied so far, the runtime arithmetic and the losses
supplied by the caller, typically from `eval` before and after the loop. Later session updates or
checkpoint loads do not change the result's parameters, predictions or verification.

Example:
```lean
-- `finish` measures nothing itself, and the report keeps the `before` and `after` passed in.
def package {trainer : TorchLean.Trainer [2] [1]}
    (session : Trainer.Session trainer) (before after : Float) :
    IO (Trainer.Result [2] [1]) :=
  session.finish { before := before, after := after }
```
-/
opaque finish {σ τ : Shape} {trainer : TorchLean.Trainer σ τ}
    (session : Session trainer α) (loss : Training.LossProgress α) : IO (Result σ τ α) :=
  session.snapshot loss

/-- Mean loss over a dataset in evaluation mode. -/
def eval [Context α] [FromFloat α] {σ τ : Shape} {trainer : TorchLean.Trainer σ τ}
    (session : Session trainer α)
    (data : Dataset σ τ) : IO α := do
  session.loss (← data.materialize (α := α)) (batch := true)

/-- Write the current parameters with `Checkpoint.State.save`. -/
def save [Checkpoint.Encoding α] {σ τ : Shape} {trainer : TorchLean.Trainer σ τ}
    (session : Session trainer α)
    (path : System.FilePath) : IO Unit := do
  Checkpoint.State.save (← session.state) path

/--
Replace the current model state from a checkpoint written by `save` or `Result.save`.

The optimizer history and the completed-step count are kept. To restart training with a fresh
optimizer and schedule, load the checkpoint into a freshly opened session.
-/
def load [Checkpoint.Encoding α] {σ τ : Shape} {trainer : TorchLean.Trainer σ τ}
    (session : Session trainer α)
    (path : System.FilePath) : IO Unit := do
  setState session (← Checkpoint.State.load (nn.Sequential.stateShapes trainer.model) path)

end Session

/--
The model of `trainer` with the parameters and buffers `state`, lowered to the verifier IR
(`IR.lowerForward`, `LoweredModel.toImported`) once, on first use.
-/
def lowerState {σ τ : Shape} {α : Type} [TorchLean.Storage α] [Context α]
    (trainer : TorchLean.Trainer σ τ)
    (state : TensorPack α (nn.Sequential.stateShapes trainer.model)) :
    Thunk (Except String (IR.Imported α)) :=
  Thunk.mk fun _ => do
    let lowered ← IR.lowerForward
      (nn.Sequential.forward trainer.model (α := α)) state
    pure lowered.toImported

/--
Build a session for a runtime scalar `α` and a public scalar `β`.

The opening path supplies the conversions and optionally the lowering of the trained state for
`trained.verify`. Typed opening uses identity conversions, preserving inputs, losses, and state
without requiring bound-arithmetic instances.

`nospecialize` keeps this setup routine from generating a copy of the runner and stepper stack
for each runtime scalar. A session performs this setup once and reuses the bound optimizer.
-/
@[nospecialize]
def openSession {σ τ : Shape} {α β : Type}
    [TorchLean.Storage α] [Context α]
    [FromFloat α] [TensorTransfer α]
    [TorchLean.Storage β] [Checkpoint.Encoding β]
    (trainer : TorchLean.Trainer σ τ)
    (scheduler : Option Scheduler.Config)
    (toRuntime : β → α)
    (readTensor : {shape : Shape} → Tensor α shape → IO (Tensor β shape))
    (readScalar : α → IO β)
    (initialState? : Option (TensorPack α (nn.Sequential.stateShapes trainer.model)) := none)
    (scalarFormat? : Option String := none)
    (lower? :
      Option (TensorPack α (nn.Sequential.stateShapes trainer.model) → LoweredModel) := none) :
    IO (Session trainer β) := do
  let runner ← Runner.instantiate trainer.model trainer.objective
    trainer.runtime.toConfig (α := α) (initialState? := initialState?)
  let stepper ← runner.stepper trainer.runtime.optimizer scheduler
  let cast (sample : Sample.Supervised β σ τ) : Sample.Supervised α σ τ :=
    Sample.map (Tensor.map toRuntime) (Tensor.map toRuntime) sample
  let predict (input : Tensor β σ) : IO (Tensor β τ) := do
    readTensor (← runner.forward (Tensor.map toRuntime input) (mode := .eval))
  let readState : IO (TensorPack β (nn.Sequential.stateShapes trainer.model)) := do
    Pack.mapM readTensor (← runner.state)
  let finish (loss : Training.LossProgress β) : IO (Result σ τ β) := do
    let steps ← stepper.steps
    let frozenState ← runner.state
    let frozenPublicState ← Pack.mapM readTensor frozenState
    let parameters ← ParamList.ofPack frozenState
    let evaluator ← Evaluator.create
      (β := Unit) (stateShapes := nn.Sequential.stateShapes trainer.model) (inputShapes := [σ])
      (dataInputShapes := []) (nn.Sequential.forward trainer.model (α := α) (mode := .eval))
      trainer.runtime.toConfig parameters
    let frozenPredict := fun input => do
      let output ← evaluator.run (Pack.singleton (Tensor.map toRuntime input)) .nil
      readTensor output
    pure <| Result.create
      { steps := steps, loss := loss, arithmetic := trainer.runtime.arithmetic,
        scalarFormat? := scalarFormat? }
      (nn.Sequential.stateShapes trainer.model)
      (pure frozenPublicState)
      (fun path => Checkpoint.State.save frozenPublicState path)
      frozenPredict
      (lower?.map fun lower => lower frozenState)
  pure <| Session.create trainer
    (fun batch => do
      readScalar (← stepper.step (batch.map cast) (batch := true) (loss := true)))
    (fun batch => stepper.step (batch := true) (batch.map cast))
    stepper.steps
    (fun sample => do
      readScalar (← runner.loss (cast sample) (mode := .eval)))
    predict
    readState
    (fun state => runner.setState (state.map (Tensor.map toRuntime)))
    finish

/--
Instantiate the model under the trainer's runtime settings and hand the optimizer loop to the
caller.

The trainer's optimizer is bound at once, and `scheduler`, when given, sets its learning rate by
completed step. Optimizer and scheduler settings must stay valid once rounded to binary32, which
`open` checks before instantiating the model. CUDA execution requires `native` arithmetic, and
supervised training rejects `complex` arithmetic.

Example:
```lean
-- One update on a single sample, returning the training-mode loss of that update.
def stepOnce (trainer : TorchLean.Trainer [2] [1]) : IO Float := do
  let session ← trainer.open
  session.step { input := [1.0, 0.0], target := [1.0] } (loss := true)
```
-/
def «open» {σ τ : Shape} (trainer : TorchLean.Trainer σ τ)
    (scheduler : Option Scheduler.Config := none)
    (initialState? : Option (TensorPack Float (nn.Sequential.stateShapes trainer.model)) := none) :
    IO (Session trainer) := do
  IO.ofExcept
    (trainer.runtime.optimizer.validateFor (α := Float32) (scalarName := "binary32"))
  match scheduler with
  | some schedule =>
      IO.ofExcept <| schedule.validate (round := fun value => value.toFloat32.toFloat)
  | none => pure ()
  if trainer.runtime.toConfig.usesCUDA && trainer.runtime.arithmetic != .native then
    throw <| IO.userError
      "TorchLean.Trainer: CUDA execution requires --arithmetic native"
  match trainer.runtime.arithmetic with
  | .native =>
      openSession (α := Float32) trainer scheduler
        (ofFloat (α := Float32)) TensorTransfer.readFloatTensor
        TensorTransfer.toFloat
        (initialState?.map fun state => state.map (Tensor.map (ofFloat (α := Float32))))
        (lower? := some fun state => .native (lowerState trainer state))
  | .ieee =>
      openSession
        (α := FloatLib.Floats.ExecFloat.Binary (exponentBits := 8) (fractionBits := 23))
        trainer scheduler ofFloat TensorTransfer.readFloatTensor
        TensorTransfer.toFloat
        (initialState?.map fun state => state.map (Tensor.map ofFloat))
        (lower? := some fun state => .ieee (lowerState trainer state))
  | .complex =>
      throw <| IO.userError <|
        "TorchLean.Trainer: supervised training supports real arithmetic; " ++
          "complex arithmetic requires an explicit complex-valued training API"

/--
Open a CPU session whose inputs, state, predictions and losses keep the scalar `α`.

`initialState?` supplies exact typed parameters and buffers. Without it, the model's seeded
`Float` initializers are converted into `α`, so a wider `α` gets no random digits beyond `Float`
precision. Optimizer and scheduler coefficients are configured in `Float` and converted into `α`.

The scalar argument takes precedence over `RunConfig.arithmetic`. `openTyped` runs on the CPU
runtime, rejects other devices and custom backend profiles before instantiating state, and never
narrows tensors for native kernels. Checkpoints use `Checkpoint.Encoding α`. Finished results
carry no lowered model, so `Result.verify` reports that verification is unavailable.
-/
def openTyped {σ τ : Shape} (trainer : TorchLean.Trainer σ τ)
    (α : Type) [Storage α] [Context α] [FromFloat α]
    [TensorTransfer α] [Checkpoint.Encoding α]
    (scheduler : Option Scheduler.Config := none)
    (initialState? : Option (TensorPack α (nn.Sequential.stateShapes trainer.model)) := none) :
    IO (Session trainer α) := do
  unless trainer.runtime.device == .cpu do
    throw <| IO.userError
      "TorchLean.Trainer.openTyped: typed sessions require CPU; CUDA/custom devices are unsupported"
  if trainer.runtime.backendProfile?.isSome then
    throw <| IO.userError
      "TorchLean.Trainer.openTyped: custom backend profiles are unsupported for typed sessions"
  openSession trainer scheduler id (fun tensor => pure tensor) pure
    initialState? (scalarFormat? := some (Checkpoint.Encoding.format (α := α)))

end Trainer

end TorchLean
