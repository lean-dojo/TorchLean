# TorchLean Application API

Most programs need:

```lean
import NN.API
open TorchLean
```

Import `NN` only when the same file also uses specification, proof, verification, or backend
internals.

## Concept Map

| Concept | Public API | Use it for |
| --- | --- | --- |
| Typed values | `Tensor α shape` | Store a homogeneous, row-major tensor whose shape is in its type. |
| Models | `nn` | Build checked layers and architectures. |
| Data | `Data`, `Trainer.Dataset` | Load, validate, batch, or generate supervised samples. |
| Training | `Trainer` | Configure an objective and optimizer, train, predict, and report results. |
| Optimizers | `optim` | Select optimizer behavior and hyperparameters. |
| Differentiation | `autograd`, `autograd.model` | Compute derivatives directly without running a training loop. |
| Runtime choices | `Runtime` | Select arithmetic semantics, execution mode, device, and backend reporting. |

These are one lifecycle, not competing tensor or model systems:

```text
Tensor -> nn model -> Trainer -> Trainer.Result
             |
             +-> autograd for direct derivative queries
```

## Stability Boundary

`NN.API` is the canonical import for application code. `NN.Tensor` is the smaller tensor-only
import, while verification, proof, and backend internals remain focused imports. Redundant import
routes are removed instead of re-exported under old names; the update guide records the required
source changes.

`NN/Tests/API/PublicSurface.lean` imports only `NN.API` and compiles representative tensor, model,
data, trainer, autograd, text, self-supervised, and reinforcement-learning usage. The external
example regression checks the same umbrella independently, and the repository linter prevents
removed routes from being restored.

## First Training Program

```lean
import NN.API

open TorchLean

def model :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]

def inputs : Tensor Float [4, 2] :=
  [[0, 0], [0, 1], [1, 0], [1, 1]]

def targets : Tensor Float [4, 1] :=
  [[0], [1], [1], [0]]

def data := Data.fromTensors inputs targets

def trainer :=
  Trainer.new model
    { objective := .meanSquaredError
      optimizer := optim.adam { learningRate := 0.03 }
      seed := 2026 }

def main : IO Unit := do
  let trained ← trainer.train data
    { steps := 20, samplesPerStep := 4, logEvery := 5 }
  trained.printSummary
  let prediction ← trained.predict ([0.25, -0.75] : Tensor Float [2])
  IO.println (reprStr prediction)
  trained.save "model.state"
```

The model definition is immutable. `Trainer.new` attaches the objective and runtime choices.
`trainer.train` returns a result retaining the trained state for prediction, `trained.state`, and
`trained.save`; `trainer.load path data` restores a saved state.

`steps` is required. `samplesPerStep` accumulates gradients from that many dataset items per
update; vectorized minibatches come from `Data.batch` and a model with an explicit batch axis.
Although the types say `Tensor Float`, training runs in binary32: `Float32` under `.native`
arithmetic and `IEEE32Exec` under `.ieee`. The result's summary names the scalar that ran.

## Tensors

`Tensor α [dims...]` has one element type and one statically known shape:

```lean
def vector : Tensor Float [3] := [1, 2, 3]
def matrix : Tensor Float [2, 2] := [[1, 2], [3, 4]]
```

Use `Array α` while a length is known only at runtime. Convert or reshape only after validating the
boundary. Ordinary application code always receives `Tensor`; it does not choose among internal
tensor representations.

`TensorPack α shapes` is reserved for a statically heterogeneous collection of tensor shapes, such
as model parameters and buffers. `Spec.SomeTensor α` is a runtime shape-erased value used by
evaluators. Neither replaces `Tensor` in application code.

## Naming And Dot Syntax

The public API uses names that describe the boundary being crossed:

- `Tensor.from source` imports an in-memory value using its intrinsic shape;
- `Tensor.to tensor TargetType` exports to a requested container type;
- `Tensor.load path` checks an NPY or numeric CSV file against the expected tensor type;
- supervised samples expose `sample.input` and `sample.target`;
- model and autograd operations use direct names such as `gradient`, `inputGradient`, and
  `lossValue`;
- completed training reports expose `report.loss.before` and `report.loss.after`.

Recursive representation constructors such as tensor shape nodes and tensor-pack cons cells belong
to implementation code. Application examples use literals, records, and named operations instead,
and tuple results are immediately bound to descriptive local names.
Type-directed choices such as `.relu`, `.train`, `.cuda`, and `.mean` remain concise because their
expected type already determines what the case means.

Stateful objects use `state` to read their current tensors and `setState` to replace them in
memory. `save` and `load` are reserved for checkpoint and external-data boundaries; for example:

```lean
let weights : Tensor Float [4, 8] ← Tensor.load "weights.npy"
```

## Models And Modules

The important objects are:

| Object | Meaning |
| --- | --- |
| `nn.Layer` | One immutable checked layer definition. |
| `nn.Sequential` | One immutable checked model definition. |
| `nn.Builder` | A model definition that still needs a deterministic initialization seed. |
| `nn.Module` | Live parameter/buffer storage plus train/eval mode for manual execution. |
| `Trainer` | The normal application wrapper around a model, objective, optimizer, and runtime. |

Use `Trainer` for ordinary training and prediction. Instantiate `nn.Module` directly only when a
program needs explicit mutable state or train/eval control without the trainer lifecycle.

Every layer constructor (`nn.linear`, `nn.conv`, `nn.mlp`, `nn.transformerEncoderStack`, and so on)
is a single `nn.Builder`-valued function; there is no separate explicit-seed constructor. Compose
builders with `nn.Sequential![...]`, fix the seeds with `nn.build seed builder`, or let `Trainer`
draw them. Combinators over already-built models (`nn.residual`, `nn.addBranches`,
`nn.concatBranches`, `nn.mapLeading`) are plain functions on `nn.Sequential`.

## Autograd

Choose by what owns the differentiable state:

```lean
let (gradient, value) ← autograd.grad loss x (value := true)
```

uses a tensor function with no model parameters.

```lean
let state := autograd.model.initialState model
let (gradient, lossValue) ←
  autograd.model.grad model lossFn state input target (value := true)
```

differentiates a model loss once and returns the model-state gradient with its loss value.
`Trainer` uses model autograd internally and adds updates, batching, optimizer state, device
selection, and reporting. Use `autograd.model.vjp` only when an explicit output gradient must also
be pulled back to the model input.

## Execution Choices

`Trainer.RunConfig` keeps execution choices on the same model:

- `arithmetic` selects the binary32 scalar (`Float32` for `.native`, `IEEE32Exec` for `.ieee`);
- `execution` selects eager tape execution or reusable typed-graph execution;
- `device` selects the storage and kernel target;
- `showBackend` reports accepted backend capsules.

Typed graph execution records reusable forward/JVP/VJP structure. It is not native-code
compilation and it is not the verification IR.

## Focused Advanced Imports

Advanced tools remain explicit:

| Need | Import |
| --- | --- |
| Caller-driven optimizer loops | `trainer.open` (in `NN.API.Trainer`) |
| Fixed-sample benchmark/diagnostic loops | `NN.API.Trainer.FixedSample` |
| Train and check model robustness | `NN.API.Verification` |
| Explicit verifier graph lowering and inspection | `NN.API.Verification.Lowering` |
| Full specifications, proofs, and backend internals | `NN` |

Start with the runnable learning path:

```bash
lake exe torchlean quickstart_tensors
lake exe torchlean quickstart_autograd
lake exe torchlean quickstart_mlp --steps 20
```

Detailed subsystem guides:

- `NN/API/Autograd/README.md`
- `NN/API/Data/README.md`
- `NN/API/Trainer/README.md`
- `NN/Examples/README.md`
