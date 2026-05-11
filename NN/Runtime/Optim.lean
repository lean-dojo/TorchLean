/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Optim.GradientUtils
public import NN.Runtime.Optim.Optimizers
public import NN.Runtime.Optim.Schedulers

/-!
# Runtime Optim

`NN.Runtime.Optim` is the small umbrella for TorchLean's reusable optimizer math.

This subsystem intentionally contains only pure, tensor-level pieces:

- `NN.Runtime.Optim.Optimizers` defines per-parameter update equations such as SGD, Adam,
  AdamW, AdaGrad, RMSProp, Adadelta, GaLore-style projected SGD, and Muon-style updates.
- `NN.Runtime.Optim.GradientUtils` re-exports the canonical spec-layer gradient clipping and
  norm helpers under names that read naturally in optimizer code.
- `NN.Runtime.Optim.Schedulers` defines deterministic learning-rate schedule state machines.

What this file does **not** contain:

- heterogeneous parameter-list handling, optimizer handles, or training-loop mutation;
- the public `optim.sgd` / `optim.adam` API facade; or
- CUDA / PyTorch fused optimizer kernels.

Those are separate on purpose. The high-level runtime bridge in
`NN.Runtime.Autograd.TorchLean.Optim` lifts these pure single-tensor equations to parameter lists,
and `NN.API.Runtime` exposes user-facing optimizer configs. Keeping this layer pure gives proofs,
tests, and runtime code one shared source of truth for the actual update formulas.
-/

@[expose] public section
