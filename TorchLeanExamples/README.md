# TorchLean Examples

Start with three CPU examples that need no external data:

```bash
scripts/lake.sh exe torchlean quickstart_tensors
scripts/lake.sh exe torchlean quickstart_autograd
scripts/lake.sh exe torchlean quickstart_mlp --steps 20
```

The first prints tensor operations, the second differentiates a tensor function and a model loss,
and the third trains a two-layer regression model. The [quickstart guide](Quickstart/README.md)
explains the model, dataset, training call, and prediction.

Learning the application API starts with these commands. Verification is a separate step for
supported model fragments:

```text
Tensor values
    ↓
nn model or differentiable tensor function
    ↓
autograd computes gradients
    ↓
Trainer applies optimizer updates
    ↓
verification checks a stated property
```

Examples hold runnable demonstrations, worked proofs, and their input fixtures. Reusable tensor
operations, model adapters, and training APIs belong in the library. Numerical values stay in
tensors, while collections describe commands, graph structure, or external file formats.
`scripts/lake.sh build TorchLeanExamples` checks the proof-only modules, which print no training
result. Each subdirectory README gives the command, the required inputs, and what success means.

## Where To Read

| Path | Purpose |
| --- | --- |
| `Quickstart/` | The learning path: tensors, autograd, training, proofs, plus optional widgets. |
| `Models/` | Maintained model commands grouped by supervised, vision, sequence, generative, operator, and RL workloads. |
| `Data/` | CSV/NPY loading and dataset boundaries. |
| `Transcendentals.lean` | A custom differentiable scalar function and the theorem behind its gradient. |
| `Verification/` | Runnable property checks and small certificate artifacts. |
| `Interop/` | JAX captures replayed through the IR, and PyTorch state dicts read into the ordinary `nn` models. |
| `Tree/` | Folds, hylomorphisms and unfolds over sample trees with their certificates, and the `tree_bench` benchmark. |
| `DeepDives/` | Advanced autograd, float semantics, graph IR, and runtime internals. |
| `BugZoo/` | Small checked case studies of ML failure modes. |
| `DeepDives/Muon.lean` | Proofs about one Muon step: direction, momentum buffer and updated parameters. |

## Public Usage

Ordinary application examples import modules of the application API. These five provide every
concept in the table below:

```lean
import TorchLean.NN.Autograd.Function
import TorchLean.NN.Autograd.Model
import TorchLean.NN.Data.Dataset
import TorchLean.NN.Trainer.Constructor
import TorchLean.NN.Trainer.Train
open TorchLean
```

The main concepts are:

| Concept | Public entry point |
| --- | --- |
| Shaped values | `Tensor α shape` |
| Models and layers | `nn` |
| Tensor-function differentiation | `autograd` |
| Model-state differentiation | `autograd.model` |
| Datasets | `Data` |
| Training | `Trainer` |
| Optimizers | `optim` |

Files that import runtime, specification, or proof modules outside the application API teach
internals, formal semantics, or trust boundaries, and ordinary training code should not use them as
a template.

Derivative proofs import `TorchLean.NN.Tactic.Autograd` and check their formulas with `by autograd`.
The tactic shares its name with the `autograd` runtime namespace above, which computes gradients.
The [proof quickstart](Quickstart.lean) proves scalar loss derivatives with it, and the
[autograd deep dive](DeepDives/AutogradTransforms.lean) connects recorded execution to mathlib at
arbitrary derivative order. The [derivative tactic guide](../TorchLean/NN/Tactic/README.md)
explains custom rules. The [tensor tactic guide](../TorchLean/Tensor/Tactic/README.md) explains
the difference between tensor expression syntax and proof automation.

## Commands

The command registries list the runnable commands:

```bash
scripts/lake.sh exe torchlean --help
scripts/lake.sh exe torchlean --list
scripts/lake.sh exe verify -- list
```

Representative next steps:

```bash
scripts/lake.sh exe torchlean data_csv --help
scripts/lake.sh exe torchlean mlp --help
scripts/lake.sh -K cuda=true exe torchlean cnn --device cuda --help
scripts/lake.sh exe torchlean pytorch_roundtrip --help
```

Build every example module with:

```bash
scripts/lake.sh build TorchLeanExamples
```
