<h1 align="center">
  <img src="home_page/assets/media/brand/torchlean-logo.png" alt="TorchLean logo" width="88" align="center">
  Formalizing Neural Networks in Lean
</h1>

TorchLean is a Lean 4 framework for writing, running, inspecting, and verifying
neural-network programs. It provides typed tensors and model APIs, a shared graph
IR, runtime/autograd support, finite-precision semantics, certificate checkers,
CUDA/runtime boundaries, and examples across modern ML and scientific ML.

The project is organized around one discipline: keep the object of the claim
visible. A classifier, FNO, PINN residual, optimizer step, imported checkpoint,
or verifier certificate should not become a loose collection of scripts once it
starts running. TorchLean gives those objects Lean names, executable paths, graph
representations, and theorem or checker APIs where the current library has
support.

## Quickstart

```bash
git clone https://github.com/lean-dojo/TorchLean.git
cd TorchLean
lake build
lake exe torchlean quickstart_mlp --cpu --steps 10 --dtype float32 --backend eager
lake exe torchlean quickstart_mlp --cpu --steps 10 --dtype float --backend eager

# Optional CUDA run (requires CUDA toolkit, LibTorch, and an NVIDIA GPU):
#   https://pytorch.org/get-started/locally/ → LibTorch → extract to ./libtorch
#   or set TORCHLEAN_LIBTORCH_HOME=/path/to/libtorch
lake build -K cuda=true
lake exe -K cuda=true torchlean mlp --cuda --fast-kernels --steps 1000
```

The first quickstart uses TorchLean's executable IEEE-style Float32 scalar. The
second uses Lean's builtin `Float` runtime path. The CUDA command uses the
native GPU runtime path and checks that the CUDA backend is available. Theorem
statements that mention CUDA cite the native-runtime boundary in
`TRUST_BOUNDARIES.md` instead of treating a kernel launch as Lean proof evidence.

CUDA builds require [LibTorch](https://pytorch.org/get-started/locally/) in addition to the NVIDIA
CUDA toolkit. Pick the **LibTorch** download (not Pip) with a CUDA version that matches your
toolkit. Extract it to `./libtorch` at the repo root, or point the build at it with
`TORCHLEAN_LIBTORCH_HOME` or `-K libtorch_home=/path/to/libtorch`.

On `lake build -K cuda=true`, TorchLean looks for `include/torch/torch.h` under those locations.
If LibTorch is missing, the build stops with download instructions. When found, the resolved path
is written to `.lake/build/libtorch.path`.

TorchLean is pinned by `lean-toolchain` and currently builds with
`leanprover/lean4:v4.31.0`.

The public code shape is:

```lean
import NN
open TorchLean

def model :=
  nn.Sequential![
    nn.Linear 2 8,
    nn.ReLU,
    nn.Linear 8 1
  ]

def xs : Tensor.T Float (shape![4, 2]) :=
  tensorND! [4, 2] [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]

def ys : Tensor.T Float (shape![4, 1]) :=
  tensorND! [4, 1] [0.2, 1.0, 1.0, 1.8]

def data : Trainer.Dataset (Shape.vec 2) (Shape.vec 1) :=
  Data.tensorDataset xs ys

def trainOnce : IO Unit := do
  let trainer :=
    Trainer.new model
      { task := .regression
        optimizer := optim.sgd { lr := 0.05 }
        backend := .compiled
        dtype := .float32 }
  let initialPrediction ← trainer.predict (tensorND! [2] [0.5, -0.25])
  IO.println s!"initial={Tensor.pretty initialPrediction}"
  let trained ← trainer.train data { steps := 200, batchSize := 16, logEvery := 25 }
  trained.printSummary
```

## First Things To Try

```bash
lake exe torchlean --help
lake exe verify --help
lake exe verify -- torchlean-ibp
```

For the maintained examples:

```bash
lake build NN.Examples.Zoo
```

The public API is centered around `Trainer.new`. A trainer owns the model, task, optimizer, runtime
backend, scalar mode, and seed. It can run one prediction with `trainer.predict`, run training with
`trainer.train`, and return a trained handle whose `predict`, `predictBatch`, and `verify` methods
reuse the trained parameters. Verification commands use the same idea at the artifact level: the
CLI names the graph, certificate, dataset, or external producer boundary being checked.

Backend selection is an option on the same model. Eager execution, compiled graph execution, CUDA
runtime paths, and selected external forward-kernel providers are not separate public meanings of
the network. When a fast runtime supplies a value, TorchLean still records the graph, tape, payload,
or checker object that later statements refer to. If a path cannot preserve that relationship for
training, it belongs behind an explicit runtime boundary or fallback.

The maintained command set currently includes quickstarts, supervised models, CNN/ViT, GPT-style
text, Mamba, diffusion, FNO Burgers, PPO/DQN, PyTorch round trips, data loaders, floating-point
deep dives, GraphSpec, BugZoo, and verification workflows such as `torchlean-ibp`,
`torchlean-crown-ops`, `abcrown-leaf`, `pinn-cert`, `pinn-dataset-check`, `digits-train-certify`,
and `vnncomp-mnistfc`.

## Documentation

- Project site: <https://lean-dojo.github.io/TorchLean/>
- Guide: <https://lean-dojo.github.io/TorchLean/blueprint/>
- Examples: <https://lean-dojo.github.io/TorchLean/examples/>
- Module graph: <https://lean-dojo.github.io/TorchLean/graphs/>
- Updates and recent validation notes: <https://lean-dojo.github.io/TorchLean/updates/>
- Paper: [*TorchLean: Formalizing Neural Networks in Lean*](https://arxiv.org/abs/2602.22631)
  (arXiv:2602.22631)

For a first pass, read the guide and examples together: the guide explains the objects, and the
examples show the commands that produce them. Use the source tree for exact declarations and the
graph page for module ownership.

## Use TorchLean From Another Lean Project

TorchLean is a normal Lake package. You can depend on the Git repository directly:

```lean
require TorchLean from git "https://github.com/lean-dojo/TorchLean.git" @ "main"
```

Then run:

```bash
lake update
lake exe cache get
lake build
```

Use `import NN` for model, data, training, verification, and proof workflows. Focused imports such
as `NN.API.*`, `NN.GraphSpec.*`, `NN.Runtime.*`, or `NN.Proofs.*` are for files that deliberately
work inside one subsystem.

Downstream model and training files should start from:

```lean
import NN
open TorchLean
```

For local development against a checkout, use a path dependency instead:

```lean
require TorchLean from "../TorchLean"
```

## Repository Map

- `NN.lean`: canonical umbrella import for model, tensor, data, training, verification,
  and proof workflows.
- `NN/API`: subsystem APIs behind `import NN`.
- `NN/Spec`: mathematical tensor, layer, model, and dynamical-system definitions.
- `NN/Runtime`: executable autograd, optimizers, training loops, CUDA boundary,
  PyTorch import/export, and RL runtime support.
- `NN/IR` and `NN/GraphSpec`: graph IR, graph semantics, and typed architecture
  descriptions.
- `NN/Proofs`: tensor algebra, selected autograd correctness theorems, analytic derivatives,
  runtime approximation, and bridge proofs.
- `NN/Floats`: finite-precision models, IEEE-style executable semantics,
  NeuralFloat formats, and error-bound infrastructure.
- `NN/MLTheory`: learning theory, robustness, CROWN/LiRPA, generative objectives,
  optimization theory, and related proof layers.
- `NN/Verification`: certificate checkers and CLI workflows.
- `NN/Examples`: quickstarts, model zoo commands, widgets, bundled verification assets,
  and interoperability workflows.
- `blueprint/TorchLeanBlueprint/Guide`: source for the public guide.
- `home_page`: project website sources.

## Current Capabilities

The same rule applies across the tree: name the object, name the artifact, and name the boundary.

- **Training and runtime.** `Trainer.new` supports supervised tasks, scalar/backend choices,
  trained handles, prediction, logs, typed step streams, generated or file-backed batches, and
  optional CUDA-backed Float32 runtime paths. Public code should look like one trainer with selected
  backend options, not one public forward method per backend.
- **Graphs and compiler fragments.** TorchLean models can be lowered to a shared IR. A first-order
  forward fragment has Lean-side compiler-correctness theorems, and coverage grows operation by
  operation. GraphSpec describes typed architectures above the lower-level op-tagged DAG consumed by
  runtime, widgets, exporters, and verification passes.
- **Verification.** The repository includes IBP/CROWN-style graph checks, α/β-CROWN-style artifact
  replay, robustness workflows, VNN-COMP-style MNIST checks, PINN residual checks, ODE corridors,
  spline certificates, and 3D geometry projection certificates. External producers write compact
  artifacts; Lean parses the artifact, checks the stated predicate, and records any remaining
  producer assumptions.
- **ML theory.** The theory layer covers CROWN/LiRPA objects, optimizer laws, Muon
  orthogonalization certificates, learning-theory examples, generative objective identities,
  self-supervised-learning algebra, approximation theorems, and floating-point bridges. Optimizers
  use a generic `TensorOptimizer`/`StepSpec` interface so new update rules can share stream laws
  before proving optimizer-specific facts.
- **Scientific ML.** The FNO Burgers and PINN/ODE paths show how numerical artifacts can be carried
  back into Lean checks. External simulators and optimizers remain named producers; Lean owns the
  artifact schemas, residual predicates, dataset checks, and certificate replay statements that sit
  at the verification boundary.

## What To Cite For A Claim

| Claim shape | Where to look |
| --- | --- |
| "This model runs and trains" | `NN/Examples`, `NN/Runtime`, command output, and regression tests. |
| "This graph has a checked bound/certificate" | `NN/Verification`, `NN/MLTheory/CROWN`, and the artifact schema named by the command. |
| "This compilation/evaluation fragment preserves meaning" | `NN/Verification/TorchLean/Proved` and the theorem imported through `NN.Entrypoint.Verification`. |
| "This optimizer update follows the intended equation" | `NN/Runtime/Optim` for executable equations and `NN/MLTheory/Optimization` for reusable laws. |
| "This finite-precision statement has a formal model" | `NN/Floats`, `NN/Proofs/RuntimeApprox`, and the relevant bridge hypotheses. |
| "This CUDA/PyTorch/ATen/Julia/Gymnasium path was used" | `TRUST_BOUNDARIES.md`, the runtime module docstring, and the command or artifact provenance. |

## Correctness and Boundaries

For correctness claims, trust assumptions, and third-party tooling:

- `TRUST_BOUNDARIES.md`
- `AI_USAGE.md`
- `THIRD_PARTY_NOTICES.md`
- `CONTRIBUTING.md`

Lean proofs, executable checkers, Lake builds, tests, and explicit
trust-boundary documentation are the source of truth for what is proved,
checked, or assumed.

## Paper

TorchLean is described in the arXiv preprint
[TorchLean: Formalizing Neural Networks in Lean](https://arxiv.org/abs/2602.22631).

## License

TorchLean is released under the MIT License. See `LICENSE`.
