/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# `NN.Runtime.Autograd`: Runtime Autograd Overview

This directory is the runtime execution layer for TorchLean's automatic differentiation.
It contains three closely related "ways to run" the same differentiable programs:

1. **Eager tape (dynamic DAG)**: record a runtime tape during the forward pass, then run a
   reverse-mode loop over that tape to accumulate gradients.
2. **Proof-compiled graph (typed SSA/DAG)**: record a *well-typed* IR (`GraphData`), compile it
   to the same runtime tape, then reuse the same reverse-mode loop.
3. **Imperative sessions**: wrap either backend behind an API that feels closer to PyTorch
   (`TensorRef` objects, `backward`, `step`, etc.).

The key architectural idea is that we keep a small, explicit, pure core (the tape and its
reverse-mode loop), then build convenience layers on top (imperative sessions, training helpers,
optimizers). This makes it easier to:

- reason about correctness at the IR level,
- connect compiled execution to proofs, and
- still offer familiar session-style ergonomics for executable training code.

## Where to look

- Eager tape engine (pure core):
  - `NN/Runtime/Autograd/Engine/Core.lean`
  - `NN/Runtime/Autograd/Engine/TapeM.lean` (StateT convenience wrapper)
  - `NN/Runtime/Autograd/Engine/FastKernels.lean` (optional runtime-only speedups)
- Proof-compiled execution path:
  - `NN/Runtime/Autograd/Compiled.lean` (runtime umbrella)
  - `NN/Runtime/Autograd/Compiled/Core.lean`
  - `NN/Runtime/Autograd/Compiled/GraphM.lean` (authoring DSL for `GraphData`)
  - `NN/Runtime/Autograd/Compiled/IRExec.lean` (shared `NN.IR.Graph` forward execution bridge)
- PyTorch-style imperative front-end:
  - `NN/Runtime/Autograd/Torch/Core.lean`
  - `NN/Runtime/Autograd/Torch/Utils.lean`
  - `NN/Runtime/Autograd/Torch/LinkedSession.lean` (records proved IR, runs compiled tape)
- Unified user-facing API:
  - `NN/Runtime/Autograd/TorchLean.lean` (umbrella import plus re-exports)
  - `NN/Runtime/Autograd/TorchLean/Session.lean` (one API, eager or compiled backend)
- Training helpers (datasets, logging, optimizers, trainer):
  - `NN/Runtime/Autograd/Train/*`
- Runtime utilities:
  - `NN/Runtime/Autograd/Utils.lean` (small umbrella for executable training scripts and tests)

## Connection to Proofs

The runtime files define executable behavior. Proof modules under `NN.Proofs.Autograd.*` state and
prove facts about the same tape/IR vocabulary, including semantic equivalence of compiled IR
execution and reusable tape-node soundness lemmas. CUDA and foreign-process bridges are checked by
contracts and tests at this layer, while their external implementations remain outside Lean's
trusted kernel.

## References / citations

- PyTorch `torch.autograd` docs:
  https://pytorch.org/docs/stable/autograd.html
- PyTorch "Autograd mechanics" note (good mental model of dynamic graph construction):
  https://pytorch.org/docs/stable/notes/autograd.html
- `torch.nn.functional.mse_loss` (mean reduction semantics):
  https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html
- `micrograd` (small reverse-mode AD engine, useful for intuition):
  https://github.com/karpathy/micrograd
-/

@[expose] public section
