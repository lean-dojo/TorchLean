# Trainer

`Trainer` is TorchLean's ordinary model-training API. It owns five choices:

1. a checked `nn` model,
2. a training objective,
3. an optimizer,
4. runtime settings, and
5. an initialization seed.

Application code imports:

```lean
import NN.API
open TorchLean
```

## Normal Lifecycle

```lean
def model :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]

def trainer :=
  Trainer.new model
    { objective := .mse
      optimizer := optim.adam { learningRate := 0.03 }
      device := .cpu
      execution := .eager
      seed := 2026 }

def data := Data.fromTensors inputs targets

def main : IO Unit := do
  let before ← trainer.predict probe
  let trained ← trainer.train data
    { steps := 200, samplesPerStep := 16, logEvery := 25 }
  trained.printSummary
  let after ← trained.predict probe
  trained.save "model.state"
```

The values have distinct roles:

| Value | Meaning |
| --- | --- |
| `nn.Builder (nn.Sequential input output)` | Architecture plus seeded initialization. |
| `Trainer input output` | Checked model plus objective and runtime choices; no data consumed yet. |
| `Trainer.Dataset input target` | Finite supervised samples with typed sample shapes. |
| `Trainer.TrainOptions` | Per-call choices: required `steps`, `samplesPerStep`, logging, files. |
| `Trainer.Result input output` | Trained state behind `predict`, `state`, `save`, and `report`. |
| `Trainer.Report` | Step count, runtime arithmetic, and before/after `Float` losses. |

Use `trainer.predict` before training and `trained.predict` after training. Training never mutates
the immutable model definition.

Give inline prediction literals their tensor type, for example
`trained.predict ([0.25, -0.75] : Tensor Float [2])`. Tensor variables already carry their shape.

The completed run is readable without positional tuples or implementation terms:

```lean
trained.report.loss.before
trained.report.loss.after
trained.report.steps
trained.report.arithmetic
```

## Precision

`trainer.train` and `trainer.open` use `Tensor Float` at their data and result boundaries and train
in binary32. `arithmetic` selects Lean's `Float32` for `.native` or FloatLib's configured
`ExecFloat.Binary` with 8 exponent bits and 23 fraction bits for `.ieee`.
Inputs are converted into that scalar when a dataset is
materialized; predictions, losses, and `trained.state` are read back to `Float`, which is exact for
binary32 values. `trained.summary` prints `arithmetic=... scalar=...` so a log always shows what
ran.

For a configured FloatLib scalar `α`, use
`trainer.openTyped (α := α) (initialState? := some state)`. Inputs, predictions, losses, parameters,
and the finished `Trainer.Result input output α` retain that scalar. Sessions support eager and
graph CPU execution; CUDA and custom backend profiles are rejected. `session.save`, `session.load`,
and `trained.save` preserve the scalar's exact checkpoint encoding. `finish` captures a state
snapshot, so later session updates do not change the result.

Construct typed state and samples directly from literals or rationals when extra precision matters.
Without `initialState?`, seeded model initialization still starts from stored `Float` values.
Optimizer and scheduler settings also start as `Float`; `nn.sgdStep` accepts a coefficient in `α`
when that precision is needed. Typed results do not provide a verifier.

## Steps And Samples Per Step

`steps` counts optimizer updates and has no default. `samplesPerStep` counts dataset items whose
gradients are accumulated, one after another, into a single update; it is not a vectorized
minibatch. For vectorized batching give the model a leading batch axis and build the dataset with
`Data.batch`, whose items are already tensor minibatches, keeping `samplesPerStep := 1`.

Each training forward updates running buffers from the same activations used for its gradients.
For example, BatchNorm after dropout sees that forward's mask, not a separately replayed mask.
This also holds inside residual and parallel blocks. Prediction uses evaluation mode and leaves
the buffers unchanged.

Eager sessions advance their seeded random stream between forwards. Typed-graph training replays
the draws recorded in its fixed graph; it does not share the eager session's random schedule.

## Saving And Restoring

```lean
let trained ← trainer.train data { steps := 200 }
trained.save "model.state"
let state ← trained.state
let restored ← trainer.load "model.state" data
```

`trained.state` is `nn.State Float trained.stateShapes`, the layout of the trained model accepted by
`Checkpoint.State.save` and `Checkpoint.State.load`. `trainer.load` instantiates the model under the
trainer's runtime, restores the state, and evaluates `data` once so that the returned result has a
meaningful report (`steps = 0`). To train from a file, pass
`loadCheckpoint? := some path` in `TrainOptions`. These files contain parameters and persistent
model buffers; they do not contain optimizer moments, schedule position, sample position, or random
generator state. Training from a file starts a fresh optimizer and schedule at step zero.

`session.load path` replaces model state in an existing session while retaining its optimizer
history and completed-step count. Open a new session before loading to start a fresh history.

## Objectives

`Trainer.Config.objective` determines how predictions and targets become a scalar loss:

- `.mse` uses mean squared error;
- `.oneHotCrossEntropy axis` uses one-hot targets along a statically valid class axis;
- `.custom loss` accepts a checked scalar loss program.

The model output shape and dataset target shape must agree. Indexed class labels should be checked
and converted at the data boundary rather than passed to the one-hot objective accidentally.

## Runtime Choices

`Trainer.RunConfig` carries:

- `optimizer`;
- `arithmetic`;
- `execution`;
- `device`;
- optional backend-contract selection and reporting.

`Trainer.Config` extends `RunConfig` with `objective` and `seed`, so the same field names work in
`Trainer.new` and in a stored `RunConfig`. These settings change how the same checked model runs.
They do not create separate model types or separate training methods. Command-line parsing of the
runtime flags lives in `NN.API.CLI.Trainer`, shared by application commands and examples.

## Driving The Loop Yourself

`trainer.train` owns the optimizer loop. When the program must own it instead, open the same
trainer as a session:

```lean
let session ← trainer.open (scheduler := some decay)
for step in [0:steps] do
  let loss ← session.step (sampleAt step) (loss := true)
  if step % 100 = 0 then IO.println s!"step {step}: loss={loss}"
let after ← session.eval data
let trained ← session.finish { before, after }
trained.save "model.state"
```

| Method | Meaning |
| --- | --- |
| `trainer.open (scheduler := none)` | Instantiate the model and bind the trainer's optimizer. |
| `trainer.openTyped (α := α) (initialState? := some state)` | Open a CPU session with typed data and exact supplied state. |
| `session.step sample` | One optimizer update without loss readback. |
| `session.step samples (batch := true) (loss := true)` | One mean-gradient update from a nonempty array, returning its mean loss. |
| `session.steps` | Updates applied so far. |
| `session.predict input` | Evaluation-mode prediction. |
| `session.predict inputs (batch := true) (batchSize := n)` | Evaluation-mode predictions along a leading axis of length `n`. |
| `session.loss sample` | Evaluation-mode loss. |
| `session.loss samples (batch := true)` | Ordered mean over a lazy `Data.SampleStream`, or zero when empty. |
| `session.eval data` | Evaluation-mode mean loss over a `Dataset`. |
| `session.state`, `session.save path`, `session.load path` | Read, write, or replace parameters. |
| `session.finish { before, after }` | Snapshot the current state as a `Trainer.Result`. |

With `open`, values cross the boundary as `Float` and execution uses the trainer's binary32 scalar.
With `openTyped`, these same methods retain `α`.
Mode handling is implicit: updates run stateful layers in training mode, predictions and losses in
evaluation mode. `trainer.train`, `trainer.load`, `trainer.trainStream`, and
`trainer.trainAlternating` are all written as `open`, a loop, and `finish`.

## When Not To Use Trainer

Use direct autograd when the goal is a derivative value rather than parameter updates:

```lean
autograd.grad function input (value := true)
autograd.model.grad model loss state input target (value := true)
```

Import `NN.API.Trainer.FixedSample` only for benchmark or diagnostic loops that repeatedly update
on one sample. It is not a second beginner training API.

Runnable starting point:

```bash
lake exe torchlean quickstart_mlp --steps 20
```
