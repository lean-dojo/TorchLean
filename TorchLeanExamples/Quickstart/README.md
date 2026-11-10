# Quickstart

The quickstart is one short path through the public TorchLean API. These modules provide its
tensors, models, autograd, datasets, and training:

```lean
import TorchLean.NN.Autograd.Function
import TorchLean.NN.Autograd.Model
import TorchLean.NN.Data.Dataset
import TorchLean.NN.Trainer.Constructor
import TorchLean.NN.Trainer.Train
open TorchLean
```

## Files

| File | Role |
| --- | --- |
| `Tensors.lean` | Literals, shapes, element types, conversion, reshape. |
| `Autograd.lean` | `autograd.grad` for tensor functions, `autograd.model.grad` for models. |
| `MLP.lean` | Model, dataset, `Trainer.create`, `trainer.train`, prediction. |
| `TorchLeanExamples/Quickstart.lean` | Shapes and loss derivatives with `autograd`. |

To fix the scalar format in the types, continue with `TypedTraining.lean`, whose `run` example uses
binary128. Open it in Lean and evaluate `#eval run` inside the example's namespace. It keeps one
typed graph across SGD steps and does not rebuild it for each gradient.

The first three are commands. They run on CPU with in-memory data:

```bash
scripts/lake.sh exe torchlean quickstart_tensors
scripts/lake.sh exe torchlean quickstart_autograd
scripts/lake.sh exe torchlean quickstart_mlp --steps 20
```

The wrapper uses the pinned Lean toolchain and keeps CPU/CUDA build profiles separate.
`TorchLeanExamples/Quickstart.lean` builds with
`scripts/lake.sh build TorchLeanExamples.Quickstart`.
For optional editor panels, open the [widget gallery](../DeepDives/WidgetGallery.lean).

Read them in the order of the table.

## Train an MLP

`MLP.lean` trains a two-layer network:

```lean
let trainer := Trainer.create model
  { objective := mse
    optimizer := optim.adam { lr := 0.03 }
    seed := 0 }
let trained ← trainer.train data { steps := 200, logEvery := 25 }
trained.printSummary
```

The training quickstart accepts `--steps N`, `--seed S`, and the runtime flags
`--arithmetic native|ieee`, `--device cpu|cuda`, and `--show-backend`. Training runs in binary32
(`Float32` or FloatLib's `Binary32`) even though the signatures say `Tensor Float`,
and the summary line prints which scalar ran. The defaults need neither a GPU nor downloads.

## Autograd Mental Model

Call autograd when a derivative is the answer, and `Trainer` when updated model parameters are
the answer.

Top-level `autograd` transforms have no model parameters:

```lean
let (gradient, value) ← autograd.grad loss x (value := true)
```

`autograd.model` differentiates through a model:

```lean
let state := nn.initialState model
let (gradient, lossValue) ←
  autograd.model.grad model lossFn state x target (value := true)
IO.println s!"loss = {lossValue}"
IO.println s!"gradient = {reprStr gradient}"
```

`Trainer` uses this machinery and adds parameter updates, optimizer state, gradient
accumulation, device selection, logging, and checkpoints. Application training code normally uses
`Trainer`, and autograd calls serve custom differentiation and analysis. `autograd.model.vjp`
pulls a given output gradient back to both the model state and the input in one reverse pass.

Advanced transforms are isolated in `TorchLeanExamples/DeepDives/AutogradTransforms.lean`.
The public operation map is in `TorchLean/NN/Autograd/README.md`.

## Proving a derivative

`autograd.grad` computes a gradient. The `autograd` tactic proves a derivative statement using
mathlib. These are different tasks, despite sharing a name:

```lean
import TorchLean.NN.Tactic.Autograd

example (x : ℝ) : HasDerivAt (fun y : ℝ => Real.exp (y * y))
    (2 * x * Real.exp (x * x)) x := by
  autograd
```

`TorchLeanExamples/Quickstart.lean` also defines a smooth penalty, `smoothPenalty`, proves its
derivative, and registers that rule with `@[local autograd]`. A numerical gradient alone does not
establish such a derivative.

For input derivatives of a model, use `autograd.model.derivative`. Repeating a direction computes
a higher derivative, and different directions compute mixed derivatives. The deep dive runs both:

```bash
scripts/lake.sh exe torchlean autograd_transforms
```

`autograd.model.derivativeVJP` then differentiates through such a derivative with respect to
parameters and inputs. The [PINN example](../Models/Operators/PINNPoisson.lean) uses it for
training.

## Continue From Here

| Goal | Command or directory |
| --- | --- |
| CSV and NPY data | `scripts/lake.sh exe torchlean data_csv --help`, `TorchLeanExamples/Data/` |
| CNN training | `scripts/lake.sh exe torchlean cnn --help`, `TorchLeanExamples/Models/Vision/` |
| Larger models | `TorchLeanExamples/Models/` |
| Verification | `scripts/lake.sh exe verify -- list`, `TorchLeanExamples/Verification/` |
| PyTorch exchange | `TorchLeanExamples/Interop/PyTorch/` |
| Runtime internals | `TorchLeanExamples/DeepDives/` |
