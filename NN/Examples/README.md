# TorchLean Examples

Start with these CPU examples; they need no external data:

```bash
scripts/lake.sh exe torchlean quickstart_tensors
scripts/lake.sh exe torchlean quickstart_autograd
scripts/lake.sh exe torchlean quickstart_mlp --steps 20
```

The first prints tensor operations, the second compares gradients, and the third trains a
two-layer regression model. The [quickstart guide](Quickstart/README.md) explains the model,
dataset, training call, and prediction.

These are the entry points for learning the application API. Verification is a separate step
for supported model fragments:

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

Examples contain runnable demonstrations, worked proofs, and their input fixtures. Reusable
tensor operations, model adapters, and training APIs belong in the library. Numerical values
stay in tensors; collections describe commands, graph structure, or external file formats.
Proof-only modules are checked by `scripts/lake.sh build NNExamples`; they do not print a training
result. Subdirectory READMEs give the command, required inputs, and meaning of success.

## Where To Read

| Directory | Purpose |
| --- | --- |
| `Quickstart/` | The learning path: tensors, autograd, training, proofs, plus optional widgets. |
| `Models/` | Maintained model commands grouped by supervised, vision, sequence, generative, operator, and RL workloads. |
| `Data/` | CSV/NPY loading and dataset boundaries. |
| `Functional/` | Executable checks for differentiable scalar operations. |
| `Factorization/` | Cholesky and QR examples with positive and negative controls. |
| `Verification/` | Runnable property checks and small certificate artifacts. |
| `Interop/PyTorch/` | Explicit PyTorch import/export boundaries. |
| `DeepDives/` | Advanced autograd, float semantics, graph IR, and runtime internals. |
| `BugZoo/` | Small checked reproductions of ML failure modes. |
| `Optimization/` | Advanced proofs about one optimizer update direction; not a training tutorial. |

## Public Usage

Ordinary application examples use:

```lean
import NN.API
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

Files that import `NN.Runtime`, `NN.Spec`, or `NN.Proofs` are intentionally teaching internals,
formal semantics, or trust boundaries. They are not the template for ordinary training code.

## Commands

Use the command registries instead of searching the directory tree:

```bash
scripts/lake.sh exe torchlean --help
scripts/lake.sh exe torchlean --list
scripts/lake.sh exe verify -- list
```

Representative next steps:

```bash
scripts/lake.sh exe torchlean data_csv --help
scripts/lake.sh exe torchlean mlp --help
scripts/lake.sh exe torchlean factorizations
scripts/lake.sh exe torchlean transcendentals
scripts/lake.sh -K cuda=true exe torchlean cnn --device cuda --help
scripts/lake.sh exe torchlean pytorch_roundtrip --help
```

Build the complete maintained example surface with:

```bash
scripts/lake.sh build NNExamples
```
