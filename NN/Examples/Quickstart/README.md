# Quickstart

The quickstart is one short path through the public TorchLean API:

```lean
import NN.API
open TorchLean
```

## Files

| File | Role |
| --- | --- |
| `TensorBasics.lean` | Literals, shapes, element types, conversion, reshape. |
| `AutogradBasics.lean` | `autograd.grad` for tensor functions, `autograd.model.grad` for models. |
| `SimpleMlpTrain.lean` | Model, dataset, `Trainer.new`, `trainer.train`, prediction. |
| `Proofs.lean` | Shapes, loss derivatives with `autograd`, and quadratic convergence with `converges`. |
| `Widgets.lean` | Optional editor panels for tensors and training logs. |
| `Common.lean` | Training quickstart flag parsing; not part of the API. |

For explicit scalar formats, continue with `Precision.lean` and `TypedTraining.lean`. Open either
file in Lean and evaluate its `run` definition with `#eval run` inside the example's namespace.
The latter keeps one typed graph across SGD steps; it does not rebuild the graph for each gradient.

The first three are commands. They run on CPU with in-memory data:

```bash
scripts/lake.sh exe torchlean quickstart_tensors
scripts/lake.sh exe torchlean quickstart_autograd
scripts/lake.sh exe torchlean quickstart_mlp --steps 20
```

The wrapper uses the pinned Lean toolchain and keeps CPU/CUDA build profiles separate.
`Proofs.lean` builds with `scripts/lake.sh build NN.Examples.Quickstart.Proofs`;
`Widgets.lean` is meant to be opened in the editor.

Read them in the order of the table.

## Train an MLP

`SimpleMlpTrain.lean` trains a two-layer network:

```lean
let trainer := Trainer.new model
  { objective := .mse
    optimizer := optim.adam { learningRate := 0.03 }
    seed := 0 }
let trained ← trainer.train data { steps := 200, logEvery := 25 }
trained.printSummary
```

The training quickstart accepts `--steps N`, `--seed S`, and the runtime flags
`--arithmetic native|ieee`, `--execution eager|typed-graph`, `--device cpu|cuda`, and
`--show-backend`. Training runs in binary32
(`Float32` or FloatLib’s `ExecFloat.Binary 8 23`) even though the tensors in the signatures
are `Tensor Float`;
the summary line prints which scalar ran. The defaults require neither a GPU nor downloads.

## Autograd Mental Model

Use direct autograd when a derivative is the answer. Use `Trainer` when updated model parameters
are the answer.

Top-level `autograd` transforms have no model parameters:

```lean
let (gradient, value) ← autograd.grad loss x (value := true)
```

`autograd.model` differentiates through a model:

```lean
let state := autograd.model.initialState model
let (gradient, lossValue) ←
  autograd.model.grad model lossFn state x target (value := true)
IO.println s!"loss = {lossValue}"
IO.println s!"gradient = {reprStr gradient}"
```

`Trainer` uses this machinery internally and adds parameter updates, optimizer state, gradient
accumulation, device selection, logging, and checkpoints. Application training code normally uses
`Trainer`; direct autograd calls are for custom differentiation and analysis. Use
`autograd.model.vjp` when an explicit output gradient should also be pulled back to the model input.

Advanced transforms are isolated in `NN/Examples/DeepDives/AutogradTransforms.lean`.
The public operation map is in `NN/API/Autograd/README.md`.

## Proving a derivative

`autograd.grad` computes a gradient. The `autograd` tactic proves a derivative statement using
mathlib. These are different tasks, despite sharing a name:

```lean
import NN.Tactic.Autograd

example (x : ℝ) : HasDerivAt (fun y : ℝ => Real.exp (y * y))
    (2 * x * Real.exp (x * x)) x := by
  autograd
```

`Proofs.lean` also defines a new smooth penalty, proves its derivative, and registers that rule
with `@[local autograd]` for use in a composition. It includes a logarithm example whose nonzero
hypothesis must be supplied. A numerical gradient alone does not establish either theorem.

For input derivatives of a model, use `autograd.model.derivative`. Repeating a direction computes
a higher derivative; different directions compute mixed derivatives. The deep dive runs both:

```bash
scripts/lake.sh exe torchlean autograd_transforms
```

`autograd.model.derivativeVjp` then differentiates through such a derivative with respect to
parameters and inputs. The [PINN example](../Models/Operators/Pinn.lean) uses it for training.

## Continue From Here

| Goal | Command or directory |
| --- | --- |
| CSV and NPY data | `scripts/lake.sh exe torchlean data_csv --help`, `NN/Examples/Data/` |
| CNN training | `scripts/lake.sh exe torchlean cnn --help`, `NN/Examples/Models/Vision/` |
| Larger models | `NN/Examples/Models/` |
| Verification | `scripts/lake.sh exe verify -- list`, `NN/Examples/Verification/` |
| PyTorch exchange | `NN/Examples/Interop/PyTorch/` |
| Runtime internals | `NN/Examples/DeepDives/` |
