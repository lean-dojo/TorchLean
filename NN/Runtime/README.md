# NN/Runtime

`NN/Runtime` is the part of TorchLean that actually runs a model.

The files here turn typed tensors and parameter lists into executable training loops, prediction
calls, autograd tapes, typed graph executions, CUDA launches, PyTorch round trips, and
reinforcement-learning rollouts. The runtime is deliberately tied back to the spec and proof layers:
the same model should be runnable as ordinary Lean code, lowerable into the graph IR, and usable as
the object of a later verification statement.

Most downstream code should not import modules from this directory directly. Prefer:

* `import NN` for ordinary model and training code,
* `import NN.API.Runtime` when you are extending the runtime subsystem itself, or
* `import NN.Runtime` when you need the broad executable umbrella.

`NN/Runtime.lean` is the executable subsystem umbrella. User-facing code goes through the
`TorchLean` API, while runtime implementation files use focused subsystem imports.

## How A Run Moves

An ordinary training command follows this shape.

1. A model is built from typed tensors, layers, parameters, and a loss.
2. The public trainer chooses arithmetic semantics and an execution path, such as native eager,
   native typed graph, or a CUDA-backed run.
3. The autograd engine records the operations that need gradients and stores enough local data for
   the backward pass.
4. The optimizer updates the parameter list using the same equations that appear in the optimizer
   theory files.
5. Optional exporters write logs, graphs, weights, predictions, or certificate inputs that other
   TorchLean modules can inspect.

That is why TorchLean has runtime tests and proof modules. The tests check that the executable path
still runs on real data and real devices. The proof modules state and prove mathematical facts about
the specifications, graph translations, interval bounds, floating-point envelopes, optimizer
updates, and verification checkers.

## Execution Paths

| Area | Role |
| --- | --- |
| `Autograd/Engine` | The small eager reverse-mode tape, closest to the local backward rules for primitive tensor operations. |
| `Autograd/TypedGraph` | Typed SSA graph execution that runs through the same runtime values instead of becoming a detached interpreter. |
| `Autograd/Model` | The TorchLean-native runtime used by the trainer: layer functions, modules, sessions, losses, metrics, training glue, and the `Runtime.Autograd.Model.Layers.Seq` model type. Random helpers are abbreviations for `Spec.Random` in `NN/Spec/Core/Random.lean`. |
| `Autograd/IRExec` | Checked lowering from `NN.IR.Graph` to a forward-only shape-indexed `ForwardGraph`, with its correctness proofs under `IRExec/Correctness`. |
| `Autograd/LeadingAxis.lean` | The shared recursion for mapping an operation over the outer axis of a shape-typed reference. |
| `Autograd/Torch` | Lower-level imperative sessions used for PyTorch interop and typed graph recording. |
| `Autograd/Train` | Deterministic datasets, step streams, loaders, losses, training loops, evaluation helpers, and optimizer integration. |
| `Optim` | Executable optimizer equations and scheduler utilities. Public optimizer names are re-exported through `TorchLean.optim`. |
| `PyTorch` | State-dict, Torch export, ONNX, and IR-to-PyTorch bridges used for round-trip checks and external model exchange. |
| `RL` | Gymnasium sessions, typed environments, PPO/DQN helpers, rollouts, and boundary checks for reinforcement learning examples. |
| `External` | Small process helpers for executable integrations that deliberately leave Lean's kernel. |
| `Training` | Training-log records shared by examples, plots, widgets, and command-line runs. |

## Execution Choices

The public API should read like one model with different execution choices, not like several
competing APIs. In ordinary code the user builds a trainer once and then selects the execution
strategy, arithmetic semantics, and device:

```lean
let trainer :=
  Trainer.new model
    { objective := .meanSquaredError
      execution := .typedGraph
      arithmetic := .ieee
      optimizer := optim.adam { learningRate := 0.001 } }
let trained ← trainer.train data { steps := 200 }
let prediction ← trained.predict input
```

Eager mode executes through the dynamic tape. Typed graph mode records and reuses a typed SSA graph.
Device and provider choices are separate runtime settings; external bridges exchange data or run
selected operations without changing the public model API.

The guide's [backend chapter](https://lean-dojo.github.io/TorchLean/blueprint/Runtime___-Autograd___-and-Interop/Inside-The-Backend-Planner/)
owns the detailed provider, capsule, VJP, fallback, and assurance-policy account. The
[GPU chapter](https://lean-dojo.github.io/TorchLean/blueprint/Floating-Point-and-Native-Boundaries/From-A-Tensor-Operation-To-A-GPU-Kernel/)
covers native CUDA dispatch and checks.

## Tests, Checkers, And Proofs

Runtime evidence and proof evidence are different, and both have a role.

| Evidence | Where it lives | What it says |
| --- | --- | --- |
| Executable examples | `NN/Examples`, `lake exe torchlean ...` | The command runs, uses the intended backend, and produces the expected artifact shape. |
| Runtime tests | `NN/Tests/Runtime` | The implementation agrees with closed forms, cross-backend checks, saved fixtures, or regression expectations. |
| Formal proofs | `NN/Proofs`, `NN/MLTheory`, `NN/Verification/Builtin/Proved` | A Lean theorem establishes a mathematical property of the specification, translation, bound, or checker. |
| Certificate checks | `NN/Verification` | An external or generated artifact is parsed and checked against a Lean side condition. The checker can be proved sound even when the artifact producer is not trusted. |

## Trust Boundaries

Native CUDA, LibTorch, PyTorch exporters, Julia, and Gymnasium remain named external boundaries.
Claims that depend on them should cite the corresponding boundary or a checker or theorem that
discharges it.

Theorems and checkers live under `NN.Proofs.*`, `NN.MLTheory.*`, and `NN.Verification.*`; this
directory supplies the executable objects they refer to.
