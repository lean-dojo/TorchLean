/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Optim.Optimizers
public import NN.Runtime.Optim.Schedulers

/-!
# Runtime Optim

`NN.Runtime.Optim` is the small umbrella for TorchLean's reusable optimizer math.

This subsystem contains pure, tensor-level pieces:

- `NN.Runtime.Optim.Optimizers` defines per-parameter update equations such as SGD, Adam,
  AdamW, AdaGrad, RMSProp, Adadelta, GaLore-style projected SGD, and Muon-style updates.
- `NN.Runtime.Optim.Schedulers` defines deterministic learning-rate schedule state machines.

What this file does **not** contain:

- heterogeneous parameter-list handling, bound optimizer state, or training-loop mutation;
- the public `optim.sgd` / `optim.adam` API; or
- CUDA / PyTorch fused optimizer kernels.

Those are separate on purpose. The high-level runtime bridge in
`NN.Runtime.Autograd.Model.Optim` lifts these pure single-tensor equations to parameter lists,
and `NN.API.Runtime` exposes user-facing optimizer configs. Proofs, tests, and runtime code all use
these pure update formulas.
-/

@[expose] public section
