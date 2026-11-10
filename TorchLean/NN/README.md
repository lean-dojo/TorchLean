# `TorchLean/NN`

The NN layer of TorchLean holds the application API (`nn` models, `Trainer`, `Data`, `optim`,
`autograd`), the autograd tape and the runtime that runs a model, and the graph IR with its import
and export. It also holds the model, dynamics and RL specifications and the theorems about all of
them. It builds on the [tensor layer](../Tensor/README.md) and the [CUDA layer](../CUDA/README.md),
and the [verification layer](../Verify/README.md) builds on it. User-facing names live in the
`TorchLean` namespace.

There is no import for the whole layer. A file imports the modules it uses, such as
`TorchLean.NN.Trainer.Train` for training code or `TorchLean.NN.IR.Graph` for the IR.
`scripts/lake.sh build TorchLean.NN` builds the layer.

## Layout

The directories are topical: a runtime module, its specification and its theorems sit side by side.

- Models and layers: [`Models/`](Models/README.md) (model specifications and constructors with
  their theorems), `Builder/` (the seeded `nn.Builder` constructors), `Layer/`, `Sequential/`,
  `Module/`, `Init/`, `Attention/`, `Conv/`, `Norm/`, `Softmax/` and `Functional/` (Fourier,
  spectral and selective-scan operators).
- Training and data: `Trainer/`, `ScalarTrainer/`, `Checkpoint/`, `CLI/`, `Training/`, `Text/`
  (tokenizers, GPT-2 BPE and generation) and [`Data/`](Data/README.md): the CSV and NPY readers
  (`Data/CSV.lean`, `Data/NPY.lean`) and the trainer's finite sample streams (`SampleStream`).
- Differentiation: [`Autograd/`](Autograd/README.md) (the `autograd` and `autograd.model`
  transforms), `Tape/`, `TypedGraph/`, `Dual/` (dual numbers and jets), `FDeriv/`, `Gradients/`,
  `Op/` and `Ops/`.
- Execution: `Session/`, `LibTorch/`, `TensorTransfer/`, `Random/`, `Backend/` (backend capsules
  with their contracts and trust level, the planner and its reports) and
  [`Kernel/`](Kernel/README.md): scalar and whole-tensor functions run on the CPU or GPU, custom
  tensor operations, and resident execution of IR graphs.
- Interchange: [`IR/`](IR/README.md) (the graph interchange representation, its import,
  evaluation and lowering) and [`PyTorch/`](PyTorch/README.md) (state-dict and IR-to-PyTorch
  bridges used for round-trip checks and external model exchange).
- Optimizers: [`Optimizer/`](Optimizer/README.md).
- Theory: `LearningTheory/`, `Lipschitz/`, `Loss/`, `Robustness/`, `StateSpace/` and
  `UniversalApproximation/`.
- `RL/`: reinforcement-learning specifications, Gymnasium sessions, typed environments, PPO and
  DQN helpers, rollouts, boundary checks and their theorems.
- Tools: [`Tactic/`](Tactic/README.md) (the `autograd`, `op_deriv` and `converges` tactics) and
  [`Widget/`](Widget/README.md) (infoview panels).

Beside the directories sit 48 top-level files, such as `Trainer.lean`, `Tape.lean` and
`StateSpace.lean` next to their directories, and single-file topics such as `Dynamics.lean`,
`VJPOp.lean`, `Json.lean`, `JAX.lean`, `MoE.lean` and `NormalCDF.lean`.

## Application API

| Concept | Public API | Use it for |
| --- | --- | --- |
| Typed values | `Tensor α shape` | Store homogeneous, row-major data whose shape is in its type. |
| Models | `nn` | Build checked layers and architectures. |
| Data | `Data`, `Trainer.Dataset` | Load, validate, batch, or generate supervised samples. |
| Training | `Trainer` | Configure an objective and optimizer, train, predict, and report results. |
| Optimizers | `optim` | Select optimizer behavior and hyperparameters. |
| Differentiation | `autograd`, `autograd.model` | Compute derivatives without a training loop. |
| Runtime choices | `Trainer.RunConfig` | Select arithmetic, device, and backend reporting. |
| Configured precision | `ExecFloat.Binary` | Pick exponent and fraction widths on the CPU. |

The pieces form one lifecycle:

```text
Tensor -> nn model -> Trainer -> Trainer.Result
             |
             +-> autograd for direct derivative queries
```

The modules that declare the names in the table above are the canonical application interface,
and tensor-only code needs only modules of the tensor layer.

### First training program

```lean
import TorchLean.NN.Data.Dataset
import TorchLean.NN.Trainer.Constructor
import TorchLean.NN.Trainer.Train

open TorchLean
open Trainer.Objective (mse)

def model :=
  nn.Sequential![
    nn.linear 2 8,
    nn.op .relu,
    nn.linear 8 1
  ]

def inputs : Tensor Float [4, 2] :=
  [[0, 0], [0, 1], [1, 0], [1, 1]]

def targets : Tensor Float [4, 1] :=
  [[0], [1], [1], [0]]

def data := Data.fromTensors inputs targets

def trainer :=
  Trainer.create model
    { objective := mse
      optimizer := optim.adam { lr := 0.03 }
      seed := 2026 }

def main : IO Unit := do
  let trained ← trainer.train data
    { steps := 20, samplesPerStep := 4, logEvery := 5 }
  trained.printSummary
  let prediction ← trained.predict ([0.25, -0.75] : Tensor Float [2])
  IO.println (reprStr prediction)
  trained.save "model.state"
```

The model definition is immutable. `Trainer.create` attaches the objective and runtime choices, and
`trainer.train` returns a result holding the trained state behind `trained.predict`,
`trained.state` and `trained.save`. `trainer.load path data` restores a saved state.

The runnable learning path starts here, and `TorchLeanExamples` has many more programs:

```bash
scripts/lake.sh exe torchlean quickstart_tensors
scripts/lake.sh exe torchlean quickstart_autograd
scripts/lake.sh exe torchlean quickstart_mlp --steps 20
```

### Tensors

`Tensor α [dims...]` has one element type and one statically known shape:

```lean
def vector : Tensor Float [3] := [1, 2, 3]
def matrix : Tensor Float [2, 2] := [[1, 2], [3, 4]]
```

Numerical inputs and outputs are tensors, including tensors whose dimensions are computed at
runtime. Arrays and lists appear at file-decoding and interoperability boundaries: convert them
there and validate the shape. Application code never chooses among internal tensor
representations.

Use `TensorPack α shapes` for model parameters and buffers and for the inputs of a multi-input
program. Each entry is a tensor with its own shape, so a state is not a single homogeneous tensor.
It is the tensor pack `TensorPack α shapes`: `.empty`, `.singleton`, `.pair`, `.fill` and
`.zero` build one, `snoc` adds a tensor at the end, `get` reads an entry and `set` replaces it.

### Naming and dot syntax

The public API uses names that describe the boundary being crossed:

- `Tensor.from source` imports an in-memory value using its intrinsic shape.
- `Tensor.to tensor TargetType` exports to a requested container type.
- `Tensor.load path` checks an NPY or numeric CSV file against the expected tensor type.
- Supervised samples expose `sample.input` and `sample.target`.
- Model configurations expose `inputShape` and `outputShape`, keeping `input` and `output` for
  tensor values.
- Intermediate shapes say what they describe, as in `hiddenShape`, `patchTokenShape` and
  `embeddingShape`. A layer's window `Window` gives the spatial extents alone, without batch
  or channel dimensions, through `convOutputSpatial`, `transposedOutputSpatial` and
  `poolOutputSpatial`.
- Model and autograd results get descriptive names such as `gradient`, `inputGradient` and
  `lossValue`.

Recursive representation constructors, such as tensor shape nodes and tensor-pack cons cells,
belong to implementation code. Application examples use literals, records and named operations,
and bind tuple results to descriptive local names right away.

Device and arithmetic choices are constructors written with a leading dot, such as `.cpu`,
`.cuda`, `.native` and `.ieee`, and Lean finds them in the type of the field. Other choices need
their namespace opened, as with `open Trainer.Objective (mse)` above. A method call such as
`trainer.train` is different: there the dot selects an operation on the value `trainer`.

For causal transformers, `tokenShape`, `vocabularyShape` and `embeddingShape` name the three input
representations. ViT's encoder uses `patchGrid` for spatial extents, `patchShape` for patch
features, `patchTokenShape` before adding a class token and `outputShape` for the encoded
sequence. To raise a parser error without an executable-name prefix, use Lean's `IO.ofExcept`.
`CLI.orThrow` adds the prefix when the message lacks it.

Stateful objects use `state` to read their current tensors and `setState` to replace them in
memory. `save` and `load` are reserved for checkpoints and external data, as in:

```lean
let weights : Tensor Float [4, 8] ← Tensor.load "weights.npy"
```

### Models and modules

| Object | Meaning |
| --- | --- |
| `nn.Layer` | One immutable checked layer definition. |
| `nn.Sequential` | One immutable checked model definition. |
| `nn.Builder` | A model definition waiting for a deterministic initialization seed. |
| `nn.Module` | Live parameter/buffer storage plus train/eval mode for manual execution. |
| `Trainer` | The normal application wrapper around a model, objective, optimizer, and runtime. |

Use `Trainer` for ordinary training and prediction. Instantiate `nn.Module` only when a program
needs mutable state or train/eval control outside the trainer lifecycle.

Every layer constructor (`nn.linear`, `nn.conv`, `nn.mlp`, `nn.transformerStack` and so on)
is a single `nn.Builder`-valued function, with no separate variant that takes a seed. Compose
builders with `nn.Sequential![...]`, fix the seeds with `nn.build seed builder`, or let `Trainer`
draw them. Combinators over built models (`nn.residual`, `nn.addBranches`, `nn.concatBranches`,
`nn.mapLeading`) are plain functions on `nn.Sequential`.

### Autograd

Use direct autograd when the goal is a derivative value rather than parameter updates, and choose
by what owns the differentiable state:

```lean
let (gradient, value) ← autograd.grad loss x (value := true)
```

uses a tensor function with no model parameters.

```lean
let state := nn.initialState model
let (gradient, lossValue) ←
  autograd.model.grad model lossFn state input target (value := true)
```

differentiates a model loss once and returns the model-state gradient with its loss value.
`Trainer` uses model autograd internally and adds updates, batching, optimizer state, device
selection and reporting. Use `autograd.model.vjp` only when an output gradient must also be pulled
back to the model input.

### Focused imports

| Need | Import |
| --- | --- |
| Caller-driven optimizer loops | `trainer.open` (in `TorchLean.NN.Trainer.Session`) |
| Train and check model robustness | `TorchLean.Verify.Trainer` |
| Verifier graph lowering and inspection | `TorchLean.Verify.Lowering` |
| Typed tensors/models with selected binary precision | `TorchLean.NN.Sequential.Training` |
| Classical model reference definitions | `TorchLean.NN.Models.<Model>` (for example `KNN`) |
| The optimizer update equations | `TorchLean.NN.Optimizer.Update` |
| Specifications, proofs, and backend internals | the module that declares each name |

## `Trainer`

`Trainer` is TorchLean's ordinary model-training API. It owns five choices: a checked `nn` model,
a training objective, an optimizer, runtime settings and an initialization seed. The values have
distinct roles:

| Value | Meaning |
| --- | --- |
| `nn.Builder (nn.Sequential input output)` | Architecture plus seeded initialization. |
| `Trainer input output` | Checked model plus objective and runtime choices, before any data. |
| `Trainer.Dataset input target` | Finite supervised samples with typed sample shapes. |
| `Trainer.TrainOptions` | Per-call choices: required `steps`, `samplesPerStep`, logging, files. |
| `Trainer.Result input output` | Trained state behind `predict`, `state`, `save`, and `report`. |
| `Trainer.Report` | Step count, runtime arithmetic, and before/after `Float` losses. |

Use `trainer.predict` before training and `trained.predict` after training. Give inline
prediction literals their tensor type, for example
`trained.predict ([0.25, -0.75] : Tensor Float [2])`. Tensor variables already carry their shape.

The completed run is readable without positional tuples or implementation terms:

```lean
trained.report.loss.before
trained.report.loss.after
trained.report.steps
trained.report.arithmetic
```

### Execution choices

`Trainer.RunConfig` keeps the execution choices of one model: `optimizer`, `arithmetic` (the
binary32 implementation, `.native` or `.ieee`), `device` (the storage and kernel target, `.cpu`
or `.cuda`), `seed` and an optional backend-contract selection, with `showBackend` reporting the
accepted backend capsules.
`Trainer.Config` extends `RunConfig` with `objective`, so the runtime field names work in
`Trainer.create` and in a stored `RunConfig`:

```lean
let trainer :=
  Trainer.create model
    { objective := mse
      arithmetic := .ieee
      optimizer := optim.adam { lr := 0.001 } }
let trained ← trainer.train data { steps := 200 }
let prediction ← trained.predict input
```

These settings change how the same checked model runs. They don't create separate model types or
separate training methods. Command-line parsing of the runtime flags lives in `CLI/Trainer.lean`,
shared by application commands and examples.

Training opens one session and records each sample's operations on a fresh eager tape. Typed
graphs belong to the transform API: `autograd` and `autograd.model` record them for gradients,
JVPs, VJPs and HVPs. A typed graph is neither compiled to native code nor the verification IR. The
device and the provider are separate runtime settings, and the external bridges exchange data or
run selected operations without changing the model's API.

The guide's [backend chapter](https://lean-dojo.github.io/TorchLean/blueprint/Runtime___-Autograd___-and-Interop/Inside-The-Backend-Planner/)
covers providers, capsules, VJPs, fallbacks and assurance policies in detail, and its
[GPU chapter](https://lean-dojo.github.io/TorchLean/blueprint/Floating-Point-and-Native-Boundaries/From-A-Tensor-Operation-To-A-GPU-Kernel/)
covers native CUDA dispatch and its checks.

### Precision

`trainer.train` and `trainer.open` use `Tensor Float` at their data and result boundaries and train
in binary32. `arithmetic` selects Lean's `Float32` for `.native` or FloatLib's configured
`ExecFloat.Binary` with 8 exponent bits and 23 fraction bits for `.ieee`. Inputs are converted into
that scalar as each sample is used, and predictions, losses and `trained.state` are read back to
`Float`, which is exact for binary32 values. `trained.summary` includes `arithmetic=... scalar=...`
and `trained.printSummary` prints it, so a log shows what ran.

Use `FloatLib.Floats.ExecFloat.Binary (exponentBits := 15) (fractionBits := 112)` as the scalar
of a `Tensor` and `TensorPack` to keep binary128 precision. FloatLib checks the format's width and
bias constraints in the type. Construct constants from exact rationals, decimal literals or
FloatLib's decimal parser, because a conversion through `Float` can discard the extra bits.

Lower the architecture with `nn.Sequential.lowerToTypedGraph model (α := Scalar)`, then call
`nn.TypedGraphModel.forward`, `jvp` or `vjp` with typed state and inputs.
`TorchLeanExamples/Quickstart/TypedTraining.lean` runs SGD on an affine model from the coefficient
`1 + 2^-100`, which binary64 can't represent, and prints its predictions as exact rationals.
Typed graphs at a configured scalar run in CPU software arithmetic.

For a configured FloatLib scalar `α`, use
`trainer.openTyped (α := α) (initialState? := some state)`. Inputs, predictions, losses, parameters
and the finished `Trainer.Result input output α` keep that scalar. Typed sessions run on the CPU
and reject CUDA and custom backend profiles. Configured formats run on the GPU through `Kernel`
custom computations instead. `session.save`, `session.load` and `trained.save` preserve the
scalar's exact checkpoint encoding. `finish` captures a state snapshot, so later session updates
don't change the result.

Construct typed state and samples from literals or rationals when extra precision matters.
Without `initialState?`, seeded model initialization starts from stored `Float` values. Optimizer
and scheduler settings start as `Float` too, so a wide scalar can't recover bits discarded there.
`nn.sgdStep` accepts a coefficient in `α` when the coefficient itself needs that precision. Typed
results don't provide a verifier.

On IEEE formats, `exp`, `log`, `log1p`, `expm1` and `Context.expMul` are FloatLib's correctly
rounded functions. Exact comparisons without floating-point approximation decide the rounding of
`tan`, `exp2`, `log2`, `log10`, `asinh`, `acosh` and `atanh` and of the pointwise `asin`, `acos`,
`atan` and `atan2` (`TorchLean.Tensor.ConfiguredBinary.Elementary` and `InverseTrigonometric`). In
an IEEE format, at a finite input in its domain with a finite result, each is the real function
rounded once to nearest even (`TorchLean.Verify.RuntimeApprox.Elementary` and
`InverseTrigonometric`). On other formats, `exp`, `log` and `Context.expMul` are deterministic
approximations and `log1p` and `expm1` use compensated defaults. Those, and `sin`, `cos`, `sinh`,
`cosh` and `tanh` in every format, carry no accuracy theorem.

The default safeguard `Context.defaultEpsilon` rounds `1/1000000` and falls back to the smallest
positive subnormal when that rounds to zero. In a tiny format the fallback can be large, so choose
tolerances for the problem at hand. Normalization rounds the exact rational `1/100000` separately.
That value is nonzero in binary16 but has no fallback, so a format where it rounds to zero needs a
positive epsilon of its own.

For three exponent bits and two fraction bits, `nn.layerNorm (width := 2) (eps := (1 / 16 : Rat))`
uses a representable positive value. Set `eps` before lowering the model to a typed graph. Otherwise
the default can round to zero and produce NaNs on constant inputs, because model validation checks
only that the rational setting is positive and never checks the converted scalar. The spec-level
`Tensor.layerNorm` takes `epsilon` in the selected scalar type.

### Steps and samples per step

`steps` counts optimizer updates and has no default. `samplesPerStep` counts the dataset items
whose gradients are accumulated, one after another, into a single update, which is different from
a vectorized minibatch. For vectorized batching, give the model a leading batch axis, build the
dataset with `Data.batch` (whose items are tensor minibatches) and keep `samplesPerStep := 1`.

Each training forward updates running buffers from the same activations used for its gradients. For
example, BatchNorm after dropout sees the mask drawn in that same forward. The same holds inside
residual and parallel blocks. Prediction uses evaluation mode and leaves the buffers unchanged. A
session advances its seeded random stream between forwards.

### Saving and restoring

```lean
let trained ← trainer.train data { steps := 200 }
trained.save "model.state"
let state ← trained.state
let restored ← trainer.load "model.state" data
```

`trained.state` reads a `TensorPack Float trained.stateShapes`, the layout of the trained model that
`Checkpoint.State.save` and `Checkpoint.State.load` accept. `trainer.load` instantiates the model
under the trainer's runtime, restores the state and evaluates `data` once, so the returned result
has a meaningful report (`steps = 0`). To train from a file, pass `loadCheckpoint? := some path` in
`TrainOptions`. The files hold parameters and persistent model buffers. They don't hold optimizer
moments, schedule position, sample position or random generator state, so training from a file
starts a fresh optimizer and schedule at step zero.

`session.load path` replaces model state in an existing session while retaining its optimizer
history and completed-step count. Open another session before loading to start a fresh history.

### Objectives

`Trainer.Config.objective` determines how predictions and targets become a scalar loss:

- `Trainer.Objective.mse` uses mean squared error.
- `Trainer.Objective.oneHotCrossEntropy axis` uses one-hot targets along the chosen class axis.
- `Trainer.Objective.custom loss` accepts a checked scalar loss program.

Open the constructors you need, for example `open Trainer.Objective (mse)`, to use the shorter name.
The model output shape and dataset target shape must agree. Check indexed class labels and convert
them to one-hot at the data boundary, so the one-hot objective never sees raw indices.

### Learning-rate schedules

Pass `scheduler := some schedule` in `TrainOptions`, or when opening a session. The schedule uses
the number of completed optimizer updates: the first update reads `schedule.rate 0`. Its arithmetic
uses `Float`, and each rate is converted to the optimizer's scalar before the update.

`Trainer.Scheduler` (in `Trainer/Scheduler.lean`) provides constant, step and exponential decay,
linear warm-up, cosine decay, cyclic triangles, one-cycle schedules, and warm-up followed by cosine
decay. For example:

```lean
open Trainer.Scheduler

def decay : Config := .warmupCosine 0.001 0.00001 100 1000
def cycling : Config := .cyclic 0.0001 0.001 100 (cycleDecay := 0.5)
def onePass : Config := .oneCycle 0.0001 0.001 0.00001 100 1000
```

`linearWarmup start peak steps` reaches the peak at `steps` and holds it. `cosine base min steps`
holds the minimum after `steps`, and `periodic := true` repeats a full cosine every `2 * steps`.
`cyclic` repeats equal-length linear rises and falls and scales the amplitude by `cycleDecay` after
each cycle. `oneCycle` reaches its peak at `riseSteps`, reaches its final rate at `totalSteps` and
holds that final rate. Its default interpolation is cosine, and `annealing := Annealing.linear`
selects linear phases. The integer phase boundaries differ from PyTorch's fractional OneCycleLR
convention. Only learning rates change: the schedules don't cycle momentum or Adam's beta
coefficients.

Sessions validate rates, endpoint ordering, phase lengths and the largest rate after scalar
conversion before attaching a schedule. Zero-duration linear warm-up selects the peak immediately,
and zero-duration cosine selects the minimum. Cyclic half-cycles and both one-cycle phases must be
positive. `warmupCosine` starts at a positive first warm-up rate, `peak / warmupSteps`, while
`linearWarmup` starts at `start`.

### Driving the loop yourself

`trainer.train` owns the optimizer loop. When the program must own it, open the same trainer as a
session:

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
| `trainer.openTyped (α := α) (initialState? := some state)` | Typed CPU session at exact `state`. |
| `session.step sample` | One optimizer update without loss readback. |
| `session.step samples (batch := true) (loss := true)` | Mean-gradient update and its mean loss. |
| `session.steps` | Updates applied so far. |
| `session.predict input` | Evaluation-mode prediction. |
| `session.predict inputs (batch := true) (batchSize := n)` | Predictions along a leading axis. |
| `session.loss sample` | Evaluation-mode loss. |
| `session.loss samples (batch := true)` | Ordered mean of a lazy `Data.SampleStream`, 0 if empty. |
| `session.eval data` | Evaluation-mode mean loss over a `Dataset`. |
| `session.state`, `session.save path`, `session.load path` | Read, write, or replace parameters. |
| `session.finish { before, after }` | Snapshot the current state as a `Trainer.Result`. |

With `open`, values cross the boundary as `Float` and execution uses the trainer's binary32 scalar.
With `openTyped`, the same methods keep `α`. `session.step` with `batch := true` needs a nonempty
array. Mode handling is automatic: updates run stateful layers in training mode, predictions and
losses in evaluation mode. `trainer.train`, `trainer.trainStream` and `trainer.trainAlternating`
are written as `open`, a loop and `finish`, while `trainer.load` opens a session, loads the file,
evaluates and finishes.

## Runtime

A training command goes through five steps.

1. A model is built from typed tensors, layers, parameters and a loss.
2. The trainer picks the arithmetic semantics and the device, CPU or CUDA.
3. The autograd engine records each operation, a registered operator of `Op`, with enough local
   data for the backward pass.
4. The optimizer updates the parameter list with the equations that the optimizer theorems reason
   about.
5. Optional exporters write logs, graphs, weights, predictions or certificate inputs that other
   TorchLean modules read.

The runtime stays tied to the specifications and proofs, so the same model can run as ordinary
Lean code, lower into the graph IR and appear in a verification statement. The pieces:

- The eager reverse-mode tape (`Tape/`, `LibTorch/`): `Tape.apply` records one registered operator
  on the CPU tape through its rule `Op.rule?`, and `LibTorch.Tape.apply` records it on the CUDA
  tape through its entry in `LibTorch.Kernel.ofOp`.
- Typed SSA graphs (`TypedGraph/`), recorded by the `autograd` transforms (gradients, JVPs, VJPs,
  HVPs) and used by the autograd proofs.
- The TorchLean-native runtime that the trainer uses (`Layer/`, `Module/`, `Functional/`,
  `Loss.lean`, `Checkpoint/`), with its layer functions, modules, losses, metrics and training
  glue. Its model type `nn.Sequential` has the pure denotation `Sequential.eval`, the
  evaluation-mode output for given parameters and buffers under the `Ops Id` instance of
  `Ops/Pure.lean`.
- `Batch.lean`: the shared recursion for mapping an operation over the outer axis of a shape-typed
  reference.
- The eager session and its tape on the CPU or CUDA (`Session/`), the scalar trainer
  (`ScalarTrainer/`), and the `Ops` instances for the tape and the typed-graph recorder (`Ops/`).
  `Session.apply` records a registered operator with one value output
  (`Op ins (.value sh :: rest)`) on the session's tape, under the capsule that the session's
  profile selects for it.
- `BatchInput.lean`: instances that give sample literals their type once the `batch` flag of a
  training or loss function is chosen, without copying data.
- Graph artifacts: `IR/Loaded.lean` loads the capture documents that the PyTorch and JAX captures
  write, and `IR/Artifact.lean` is the `torchlean.ir` JSON of typed operator programs with the
  words of their constants.
- `JAX.lean`: the Jaxpr importer, with its Python adapter `python/torchlean/jaxpr.py`. Capturing a
  computation needs JAX, and loading its document needs none.
- `Json.lean`: the JSON readers that the artifact importers, certificate checkers, training logs
  and RL bridges share, and the writers of rounding directions and operator records.
- `Training.lean`: training-log records shared by examples, plots, widgets, and command-line runs.

`nn.functional.dropout` draws its uniform sample `Op.randUniform` through `Ops.applyRandom`, which
replaces the operator's key with the interpreter's next key under `seed`. In eager mode
`Session.applyRandom` (`Session/Apply.lean`) derives that key with `Random.keyOf seed invocation`
from the session's random-operation counter, and `Session.takeRandomKey` (`Session/Random.lean`)
advances the counter once the call's checks pass, so two calls with the same seed draw different
samples. The samplers are in `TorchLean/Tensor/Random.lean`.

## Optimizers

`Optimizer/Update.lean` holds the optimizers' update equations as pure functions on typed tensors:
one state record and one update per optimizer, for a single parameter tensor. Their public names
are in `optim`. Each state record holds the fields its update reads. `AdamW.State` extends
`Adam.State` with the decay coefficient, and `Muon.State` extends `MomentumSGD.State` with the
orthogonalizer.

The SGD, momentum SGD, projected SGD and Muon updates need only `Add`, `Sub` and `Mul` on the
scalar, with no order and no square root. The Muon orthogonalizer and the GaLore projector bring
their own maps, and every `init` needs at most `Zero`, `One` and `Mul`. The adaptive updates
(AdaGrad, RMSProp, Adam, AdamW, Adadelta) take `Context α` and a decidable `>` for their square
roots and divisions. The steps for adding an optimizer, with its `TensorOptimizer` laws, are in
the [`Optimizer` README](Optimizer/README.md).

The optimizer theorems prove that the executable update follows the named spec. The pattern is:

1. define a pure update equation over tensor parameters and optimizer state,
2. package it as a shape-polymorphic `TensorOptimizer`,
3. state a one-step `StepLaw`,
4. prove reusable stream facts such as `runSteps_append` and
   `runSteps_eq`,
5. add optimizer-specific laws only where the rule has real structure.

`Optimizer/TensorOptimizer.lean` packages the seven first-order optimizers (SGD, momentum SGD,
Adagrad, RMSProp, Adam, AdamW, Adadelta) and Muon as `TensorOptimizer`s with the step and stream
laws. Two optimizers have more structure:

- **Muon.** Muon is an optimizer with an orthogonalization backend. The proof separates the
  momentum buffer recurrence, the backend output used as the update direction, and the parameter
  update equation. A backend can provide an exact certificate such as $Q^\mathsf{T}Q=I$, or an
  approximate certificate bounding $Q^\mathsf{T}Q-I$ entrywise. `Optimizer/Muon.lean` and
  `Optimizer/Muon/` hold the orthogonalizer contracts, the certificates for one executable Muon
  step, the QR/Gram-Schmidt backend with its exact certificates and the residual-checked
  Newton-Schulz backend. `TorchLeanExamples/DeepDives/Muon.lean` shows how a downstream proof
  consumes the packaged exact or approximate certificate.
- **GaLore.** GaLore is projected-gradient machinery in the runtime. `GaLore.Projector` pairs a
  `project` map with a `lift` map, and `identityProjector` leaves gradients unchanged. GaLore's
  state is a learning rate and a projector, and the projected SGD update in `Optimizer/Update.lean`
  subtracts the lifted projection of the gradient, scaled by the learning rate. With the identity
  projector it gives the parameters of SGD (`GaLore.update_params_eq_sgd`).

The Newton-Schulz polynomial iteration is executable. The theorems about the direction it returns
assume a bound on its Gram residual, or start from an exactly column-orthogonal matrix (at a
Newton-Schulz fixed point in `Optimizer/Muon/NewtonSchulz.lean`, over `ℝ` in
`Optimizer/Muon/QR.lean`).

The convergence theorems are exact `ℝ` statements. For strongly convex gradient descent,
`Optimizer/GD/Convergence.lean` states the contraction with its `q^k` factor
(`dist_sq_iterate_le_of_q_nonneg`), proves convergence when `q < 1`
(`tendsto_iterate_of_q_lt_one`) and derives the step-size form
(`dist_sq_iterate_le_of_step_size`). Applying them to executable Float32 operations, CUDA kernels
or a particular trained model needs a separate refinement argument with floating-point error
accounting.

```lean
import TorchLean.NN.Optimizer.GD.Convergence
```

## ML theory

The theory files hold the mathematical claims about TorchLean's executable ML objects. A file may
define the mathematical object (a Lipschitz-continuity predicate, a diffusion loss, a
uniform-stability predicate, a differential-privacy budget) or prove that an update rule or a
definition has a mathematical property under stated hypotheses. The `Float` diagnostics in
`Robustness/Empirical.lean` compute observed quantities on finite samples instead: they give
evidence about the samples they ran on, while a theorem gives a claim under its hypotheses.

CROWN and LiRPA bound propagation, its certificates and their soundness theorems live in
[TorchLean/Verify/CROWN](../Verify/CROWN/README.md), the floating-point approximation theorems in
[TorchLean/Verify/UniversalApproximation](../Verify/UniversalApproximation/), and the
Lipschitz and margin robustness theorems in
[TorchLean/Verify/Robustness](../Verify/Robustness/README.md).

### Generative models

The generative theorems sit with their models in `Models/Diffusion/`, `Models/VAE/`,
`Models/VQVAE/` and `Models/GAN/`:

- zero-step identities for the DDPM, DDIM and probability-flow samplers, Lipschitz and contraction
  bounds for the Euler and DDIM steps, and the agreement of the diffusion API's schedule and
  training sample with the image specification,
- over `ℝ`, the closed form, nonnegativity and zero case of the VAE KL term, β-monotonicity of the
  VAE loss, and the law of the scalar Gaussian reparameterization,
- VQ-VAE loss identities and the existence of a nearest code in a nonempty codebook,
- the LSGAN zero-loss cases.

The Gaussian law of the forward noising is proved in
`Models/Diffusion/ForwardProcess/Gaussian.lean`.

### Approximation, Hopfield and state-space theory

`UniversalApproximation.lean` builds a two-layer ReLU MLP within `ε` of a Lipschitz function on
`[a, b]`, with hidden width `⌈2L(b - a)/ε⌉ + 1`, and `UniversalApproximation/Rate.lean` gives its
rates. `UniversalApproximation/StoneWeierstrass.lean` proves that the coordinate polynomials are
dense in `C(K, ℝ)` for a compact `K` of `Tensor ℝ [n]` inputs. The other files of
`UniversalApproximation/` join the two through exact affine and ridge bridges and a product
approximation on boxes, so every continuous function on a compact set is uniformly approximable by
two-layer ReLU MLPs (`relu_compact`).

`Models/Hopfield/` proves the classical energy argument over `ℝ`: with symmetric, zero-diagonal
weights an asynchronous update never raises the energy, and cyclic sweeps reach a fixed point
within `2^n` sweeps. `StateSpace/` proves that affine selective-scan summaries compose
associatively over a semiring and agree with the left-to-right recurrence, and that S4/Mamba-style
runners are prefix-causal. The scan theorem says nothing about a reordered floating-point scan, and
a native kernel needs its own refinement argument.

### Learning theory

`LearningTheory/` holds `Stability.lean` and `DifferentialPrivacy.lean`, and the perturbation
predicates are in `Robustness.lean`. Their theorems live over exact real numbers.

The real ridge theorem in `TorchLeanExamples/DeepDives/Stability/RidgeRegression1D.lean` assumes
positive regularization and bounded training, replacement, and test examples. Its bound is
`4 * X^2 * Y^2 * (λ + X^2)^2 / (λ^3 * N)`, which decays as `1/N` with the other parameters fixed.
The bridge in the same file relates the fit's evaluation in an IEEE binary format to semantics that
round after each primitive, under a `FiniteEval` hypothesis. It doesn't carry the real stability
bound over to floating-point training.

The privacy layer proves monotonicity in `ε` and `δ` and measurable post-processing, and gives no
privacy guarantee for a concrete training procedure.

## Autograd, operator and backend proofs

Ordinary model code doesn't import the proof modules. Import one when a file proves a fact about
the specifications, the autograd tape, a graph translation or the backend planner. For the
executed backward pass of a lowered graph:

```lean
import TorchLean.NN.Tape.Lowering.BackwardDenseGraph
```

Tensor proofs and the shared list and real-function lemmas are in
[TorchLean/Tensor](../Tensor/README.md), runtime approximation bounds in
[TorchLean/Verify/RuntimeApprox](../Verify/RuntimeApprox/README.md), and ODE corridor proofs in
[TorchLeanExamples/Verification/ODE](../../TorchLeanExamples/Verification/ODE/README.md).

### Autograd

`AdjointOp.lean` and `Tape/AdjointOp.lean` prove the adjointness law
`⟪JVP(x, dx), δ⟫ = ⟪dx, VJP(x, δ)⟫` for selected operator specs, and `Tape/Graph.lean` proves
reverse-mode accumulation over a tape with shared values sound given that law at each node.
`Tape/Lowering.lean` proves that `GraphData.toTape` (in `TorchLean.NN.TypedGraph`) builds a `Tape`
holding the values of `GraphData.eval`. `Tape/Lowering/BackwardDense.lean` shows that, over a
scalar whose zero is an additive identity (`AddZeroClass`), the executed sweep
`Tape.backwardDenseAll` agrees with the proved sweep `Tape.backwardDenseFrom` on every
`ZeroPreserving` tape (`backwardDenseAll_eq_backwardDenseFrom`). `BackwardDenseGraph.lean`
discharges that hypothesis for tapes lowered over a commutative semiring
(`Graph.zeroPreserving_toTape`) and states the corollaries `Graph.backwardDenseAll_toTape` and,
over `ℝ`, `Graph.backwardDenseAll_eq_fderiv`. The second says that, on a tape lowered from a
proof-carrying graph, the input block that `backwardDenseAll` returns is the adjoint of the Fréchet
derivative of the forward map applied to the seed. `BackwardLeaves.lean` and `BackwardSnoc.lean`
hold the `nil` case and the inductive step shared by the `Graph` and `GraphData` backward-link
theorems of `BackwardGraph.lean`.

No runtime path sweeps a lowered tape. The eager trainer runs `backwardDenseAll` on its session
tape, whose CPU nodes `Tape.leaf` and `Tape.apply` record. `Tape/Apply/Correctness.lean` states
once, for every registered operator, what `Tape.apply` records on the CPU tape. The tape gets the
rule's forward value at the next node id and a backward that is the backward of the rule prepared
with every input flagged, keeping the contributions to parents that need a gradient
(`apply_backward_eq_filter`). Replacing the recorded backward by that unfiltered one leaves the
reverse sweeps unchanged (`apply_backwardDense`, `apply_backwardDenseAll`,
`apply_backwardDenseFrom`), and each way the call fails is a theorem with its error.

`LayerNorm.eager_backward_eq_fderiv` in `Norm/LayerNorm/Eager.lean` proves that, over `ℝ` with a
positive epsilon, the backward of a recorded layer normalization of a matrix returns the adjoint
of its Fréchet derivative at the parents that need a gradient. The typed-graph pullback
`vjpChecked` is covered end to end by `TypedGraph.WithData.vjpChecked_eq_fderiv` in
`TypedGraph/WithData/Checked.lean`. Over `ℝ`, a successful call returns the adjoint of the Fréchet
derivative of the selected output when a proof-carrying graph describes its operation data and is
differentiable at the input and the values it reaches. The theorems are about the exact tape model
at its carrier, with no `Float` rounding or CUDA.

Analytic derivative facts sit under `FDeriv/`, `Softmax/`, `Attention/`, `Norm/` and `Conv/`.
`Softmax/Correctness.lean` gives `Tensor.hasFDerivAt_softmax`, `FDerivOp.softmax`,
`softmaxBackward_eq_fderiv` and the log-softmax analogues, and attention has
`Attention.backprop_eq_fderiv`. For normalization, `LayerNorm.jvp_backward_adjoint` shows that the
LayerNorm backward is adjoint to its JVP at every real `ε`, while the derivative theorems
`fderiv_eq_jvp`, `adjoint_fderiv_eq_backward`, `BatchNorm.hasFDerivAt_forwardVec` and
`BatchNorm.fderiv_forwardVec_eq_jvp` assume `0 < ε`. `Gradients/` holds smaller gradient facts for
layers and activations.

### Operators

`Op/Arms.lean` bridges the arms of `Op.rule?` to the operator specifications and to the
reductions, index selection, scatter-add and concatenation written out. The certificates about the
arms are registered with `op_cert`. `Op/Certificates.lean` has the real adjointness records and the
derivatives of `pad` and `contract`. `Op/Pointwise.lean` has the derivatives of the pointwise
operators and of binary arithmetic, `max` and `min` at every shape. `Op/BatchNorm.lean` and
`Op/LayerNorm.lean` have the joint derivatives of `batchNormRunning` and `layerNorm`, and
`Op/Softmax.lean` has the derivatives of `softmax` and `logSoftmax` along any axis.

### Analysis, models and text

`Softmax.lean`, `Loss/` and `Lipschitz/` hold analytic facts for softmax, normalization, clamped
probability losses and Lipschitz-style statements. `Attention/` proves facts about attention masks
and weights and the permutation equivariance of self-attention, `Models/Classifier.lean` the
evaluation of a linear classifier, and `MoE.lean` mixture-of-experts routing.
`Models/Diffusion/ForwardProcess/Gaussian.lean`, `Random/Range.lean` and `NormalCDF.lean` cover
Gaussian forward noising, seeded random tensor support and GELU's normal CDF. In
`Text/GPT2BPE/Merge.lean`, the merge passes of `Text.GPT2BPE.encodeFragment` preserve the
concatenated byte-escaped text.

### Backend

The backend proofs concern the Lean planner and the declared contracts it selects. In a successful
plan, every kernel belongs to an operator node of the graph, and its capsule is offered for that
operator by an available implementation and admissible under the policy (`Graph.plan_sound`,
`planOp_sound`). Coalescing keeps every node id, operator, and complete capsule, in order and with
repeats, so a grouped acceptance proof covers every node of its source plan. These proofs do not
establish correctness of native kernels, numerical algorithms, or the FFI.

## Specifications

The specifications are built from the tensor and layer definitions of
[`TorchLean/Tensor`](../Tensor/README.md). Like the tensor specs, they are pure reference
definitions that runtime and verification code point back to, written over `ℝ` and `Tensor ℝ s`.

`VJPOp.lean` holds the spec-level reverse-mode building blocks (`VJPOp`) used by runtime AD
wrappers and proof files. `Models/` holds model compositions such as MLP, CNN, Transformer,
Seq2Seq, GNN, linear/logistic regression, gradient boosted trees, HMM/GMM/PCA, and state-space
models, next to the ResNet, ViT and U-Net constructors, and the diffusion and latent-variable
objective specs sit with their models. Import the module a file uses, for example
`TorchLean.NN.Models.MLP` or `TorchLean.NN.RL.FiniteMDP`.

### Dynamical systems

`Dynamics.lean` holds discrete-time dynamical systems. A `Dynamics.System s` is a pure transition
`step` on the real tensors `Tensor ℝ s`, a `Dynamics.DrivenSystem s u` also reads an input of
shape `u`, and iterating the step gives the trajectories. The file defines the iterates and
trajectories of a driven system (`DrivenSystem.iterate`, `trajectory`), `DrivenSystem.freeze`,
which fixes the input, and the predicates `System.IsLyapunovStable` and
`System.IsAsymptoticallyStable`. The predicates measure states with a supplied size function and
assume no norm laws, so relating them to metric stability needs those laws as separate hypotheses.

`StateSpace.lean` defines `DiagonalSSM`, the channelwise recurrence $h'=A\odot h+B\odot x$ with
readout $y=C\odot h+D\odot x$ over any scalar backend, its `scan` over an input sequence, and its
view as a driven system over `ℝ` (`DiagonalSSM.drivenSystem`).

The specs are pure: they don't fix a CUDA kernel, simulator or training loop. `Models/Mamba/` and
`Models/S4.lean` build on `DiagonalSSM`, and the DDIM and probability-flow Euler samplers wrap
their updates as systems (`ImageDDIM.stepSystem`, `pfODEEulerSystem`). Use these specs when the
claim is about repeated application of a transition rule:

```text
state₀
  -> step state₀
  -> step (step state₀)
  -> trajectory n
```

`Dynamics.lean` proves the algebra of iterates and trajectories, and the theorems about the
behavior of particular systems sit with them. `Models/Diffusion/Samplers.lean` carries Lipschitz
and contraction bounds from a sampler update to its system
(`ImageDDIM.stepSystem_contracts_of_step_contracts`), and `StateSpace/Scan.lean` proves that the
`scan` of a `DiagonalSSM` agrees with `Tensor.selectiveScan` (`DiagonalSSM.scan_eq_selectiveScan`).

## Reinforcement learning

`RL.lean` and `RL/` hold TorchLean's pure reinforcement-learning semantics: what an environment, a
return, a Bellman backup, an advantage estimate and the Bellman operator of a policy mean before
any runtime collector or optimizer is involved. The same directory holds the runtime that
implements or approximates them (Gymnasium sessions, rollout buffers, PPO and DQN) and the
theorems about both.

### Specifications

- `RL.lean`: one-step Bellman backups, TD residuals, discounted returns and GAE.
- `RL/Environment.lean`: a pure Gymnasium-style environment (`Env`) whose `reset` and `stepGym`
  pass the hidden state as a value, and `SafeEnv`, an environment with an invariant that valid
  steps preserve.
- `RL/FiniteMDP.lean`: deterministic finite discounted MDPs over `Fin n` states and actions.
- `RL/StochasticMDP.lean`: finite stochastic discounted MDPs with tensor transition rows.
- `RL/MDP.lean`: discounted MDPs on measurable state and action spaces, using mathlib Markov
  kernels.

The deterministic model is `RL.FiniteMDP`, and the stochastic models are `RL.StochasticMDP` and
`RL.MDP`. Both finite models use the tensor-backed `RL.ValueFunction`, while
`RL.MDP.ValueFunction` is a real-valued function on a measurable state space. Each model's Bellman
operators live in its namespace. Keeping the models separate avoids forcing every RL development
into the heaviest probability-theory abstraction.

The PPO rollout code takes its returns and GAE advantages from `RL.lean`, and adds minibatches,
clipping, logging and runtime tensors on top of those idealized returns and advantages. A
Gymnasium adapter talks to Python, while the environment interface gives the state transition,
reward and termination condition as pure functions.

### What is proved

The RL proofs separate the pieces that executable examples tend to blur:

- a Lean-native environment versus a Gymnasium subprocess,
- transition records versus external observations,
- replay-buffer structural invariants,
- MDP and finite stochastic MDP facts,
- checked-runtime bridges for floating-point rollouts, in `TorchLean/Verify/RuntimeApprox/RL`.

Gymnasium is an external environment. The theorems are about the boundary object once an external
observation has been parsed, checked and admitted on the TorchLean side. The facts about the
definitions above and their runtime versions:

- `RL/Backup.lean`: `getScalar_returnsFromAdvantages` and the one-step backup facts
  `discountedBackup_mono` and `discountedBackup_abs_sub_le`.
- `RL/Environment/Safety.lean`: the lengths of state traces and rollouts (`size_states`,
  `size_rollout`), and that a `SafeEnv` keeps its invariant along valid action paths
  (`evolve_safe`, `states_safe`).
- `RL/MDP/Bellman.lean`: the Bellman contraction bounds, and the fixed-point and value-iteration
  theorems for any operator that contracts the sup distance on bounded measurable value functions
  (`RL.MDP.BddContraction`).
- `RL/FiniteMDP/Bellman.lean` and `RL/StochasticMDP/Bellman.lean`: the same theorems for the two
  finite models, derived from `RL/MDP/Bellman.lean` through `toMarkov` and `SupContraction.toBdd`.
- `TorchLean/Verify/RuntimeApprox/RL`: the rounding chains of `discountedBackup` and
  `tdResidual` in any configured IEEE format when the intermediates are finite, as in a checked
  run that returns `.ok`.

None of these theorems covers a full training run.

## Evidence and trust boundaries

Each kind of evidence says something different.

- Executable examples (`TorchLeanExamples`, `scripts/lake.sh exe torchlean ...`): the command runs,
  uses the intended backend, and produces the expected artifact shape.
- Runtime tests (`TorchLeanTest`): the implementation agrees with LibTorch's CUDA kernels,
  PyTorch, ONNX models, native floating point and the files outside tools write.
- Formal proofs (this directory): a Lean theorem establishes a mathematical property of the
  specification, translation, bound, or checker.
- Certificate checks (`TorchLean/Verify`): an external or generated artifact is parsed and checked
  against a Lean side condition. The checker can be proved sound even when the artifact producer is
  not trusted.

The checkers and soundness theorems of `TorchLean/Verify` build on facts proved here, such as the
softmax and Lipschitz bounds in `Softmax.lean` and `Lipschitz/`, the activation derivatives in
`Gradients/`, and the softmax derivatives in `Softmax/`.

Native CUDA, LibTorch, the PyTorch exporters and Gymnasium are named external boundaries. A claim
that depends on one of them cites that boundary, or a checker or theorem that discharges it. The
LibTorch buffers and foreign functions are in
[`TorchLean/CUDA/LibTorch`](../CUDA/LibTorch/README.md), and the binary32 contracts for their
results are in [`TorchLean/Verify/LibTorch`](../Verify/LibTorch/README.md). The
[trust boundaries](../../docs/TRUST_BOUNDARIES.md) page separates checker acceptance, soundness
theorems and native execution evidence.

## Adding code

Put a file in the topical directory of its main object, next to the runtime code and specs it
talks about. Add a proof here when examples or checkers can reuse it. Other code goes elsewhere:

- a runnable training script: `TorchLeanExamples`,
- a one-off command test: `TorchLeanTest`,
- a parsed artifact checker, a CLI, or a theorem about CROWN, rounding bounds or floating-point
  execution: `TorchLean/Verify`,
- a tensor or layer denotation, or a tensor fact: `TorchLean/Tensor`.

Each theorem family you add gets a short module docstring that says:

- what semantic object it defines or proves facts about, and what is proved,
- whether the scalar world is exact `ℝ`, executable IEEE-style, `Float`, or another scalar model,
- which runtime or external producer assumptions it leaves open,
- which example or checker exercises the theorem, if any.

## References

- Hopfield networks: Hopfield, "Neural networks and physical systems with emergent collective
  computational abilities," PNAS 1982.
- Graded-response Hopfield networks: Hopfield, "Neurons with graded response have collective
  computational properties like those of two-state neurons," PNAS 1984.
- S4: Gu, Goel, and Ré, "Efficiently Modeling Long Sequences with Structured State
  Spaces," ICLR 2022.
- Mamba: Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces," COLM 2024.
- Stability: Bousquet and Elisseeff, "Stability and Generalization," JMLR 2002.
- Privacy: Dwork and Roth, *The Algorithmic Foundations of Differential Privacy*, 2014.
- IEEE floating point: IEEE Std 754-2019, with Goldberg (1991), "What Every Computer Scientist
  Should Know About Floating-Point Arithmetic," and Muller et al., *Handbook of Floating-Point
  Arithmetic*.
- Bellman, *Dynamic Programming* (1957).
- Puterman, *Markov Decision Processes* (1994).
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed.).
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  (2015).
- Gymnasium API reference: <https://gymnasium.farama.org/>.
- TorchRL documentation: <https://pytorch.org/rl/>.
