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
    { objective := .meanSquaredError
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

The completed run is readable without positional tuples or implementation terms:

```lean
trained.report.loss.before
trained.report.loss.after
trained.report.steps
trained.report.arithmetic
```

## Precision

The supervised `Trainer` uses `Tensor Float` at its data and result boundaries and trains in
binary32. `arithmetic` selects Lean's `Float32` for `.native` or FloatLib's configured
`ExecFloat.Binary` with 8 exponent bits and 23 fraction bits for `.ieee`.
Inputs are converted into that scalar when a dataset is
materialized; predictions, losses, and `trained.state` are read back to `Float`, which is exact for
binary32 values. `trained.summary` prints `arithmetic=... scalar=...` so a log always shows what
ran.

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

- `.meanSquaredError` uses mean squared error;
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
  let loss ← session.step (sampleAt step)
  if step % 100 = 0 then IO.println s!"step {step}: loss={loss}"
let after ← session.eval data
let trained ← session.finish { before, after }
trained.save "model.state"
```

| Method | Meaning |
| --- | --- |
| `trainer.open (scheduler := none)` | Instantiate the model and bind the trainer's optimizer. |
| `session.step sample`, `session.stepBatch batch` | One optimizer update; returns the loss. |
| `session.update sample`, `session.updateBatch batch` | One update without loss readback. |
| `session.steps` | Updates applied so far. |
| `session.predict input`, `session.predictMany inputs` | Evaluation-mode predictions. |
| `session.loss sample`, `session.meanLoss samples` | Evaluation-mode losses. |
| `session.eval data` | Evaluation-mode mean loss over a `Dataset`. |
| `session.state`, `session.save path`, `session.load path` | Read, write, or replace parameters. |
| `session.finish { before, after }` | Snapshot the current state as a `Trainer.Result`. |

Every value crosses the boundary as `Float`; the session runs in the trainer's binary32 scalar.
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
