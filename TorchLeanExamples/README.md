# TorchLean Examples

I wrote these examples to be changed, not only read. The small ones make it easy to try a tensor
operation or alter a model and see what Lean accepts. The larger ones exercise the same APIs on
training, imported graphs, numerical failures, and verification problems where the details matter.

If TorchLean is new to you, this is the shortest useful first pass:

```bash
scripts/lake.sh exe torchlean quickstart_tensors
scripts/lake.sh exe torchlean quickstart_autograd
scripts/lake.sh exe torchlean quickstart_mlp --steps 20
```

All three use the CPU and in-memory inputs. The first command works with shaped tensors, the second
differentiates a tensor function and a model loss, and the third trains a two-layer regression
model. Their source is short enough to keep beside the Infoview while you experiment. The
[quickstart guide](Quickstart/README.md) explains the model, dataset, training call, and prediction.

After that, follow the question you care about rather than a fixed sequence:

| Path | Purpose |
| --- | --- |
| [`Quickstart/`](Quickstart/README.md) | Tensors, autograd, training, and a first proof. |
| [`Models/`](Models/README.md) | Supervised, vision, sequence, generative, operator-learning, and RL workloads. |
| [`Data/`](Data/README.md) | CSV/NPY loading and dataset boundaries. |
| `Transcendentals.lean` | A custom differentiable scalar function and the theorem behind its gradient. |
| [`Verification/`](Verification/README.md) | Runnable property checks, proofs, and small certificate artifacts. |
| [`Interop/`](Interop/README.md) | JAX graphs and PyTorch state dictionaries crossing into TorchLean. |
| `Tree/` | Folds, hylomorphisms and unfolds over sample trees with their certificates, and the `tree_bench` benchmark. |
| [`DeepDives/`](DeepDives/README.md) | Advanced autograd, floating-point semantics, graph IR, and runtime internals. |
| [`BugZoo/`](BugZoo/README.md) | Small checked case studies of ML failure modes. |
| `DeepDives/Muon.lean` | Proofs about one Muon step: direction, momentum buffer and updated parameters. |

Some files are programs, some are proofs, and several deliberately show the boundary between the
two. A successful training run is evidence about that run. A theorem is checked when its module
builds. The verification READMEs say exactly what each certificate checker accepts and what that
acceptance does not establish.

Reusable operations and runtimes stay under `TorchLean/`; this directory keeps the examples and
their small fixtures. When an example grows a reusable API, the API moves into the library and the
example remains as a compact tour.

## Finding And Building Examples

The command registries list the runnable commands:

```bash
scripts/lake.sh exe torchlean --help
scripts/lake.sh exe torchlean --list
scripts/lake.sh exe verify -- list
```

Every command has its own help. For example:

```bash
scripts/lake.sh exe torchlean data_csv --help
scripts/lake.sh exe torchlean mlp --help
scripts/lake.sh -Kcuda=true exe torchlean cnn --device cuda --help
scripts/lake.sh exe torchlean pytorch_roundtrip --help
```

To check every example module, including the proof-only files:

```bash
scripts/lake.sh build TorchLeanExamples
```
