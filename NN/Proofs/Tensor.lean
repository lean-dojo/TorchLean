/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Algebra
public import NN.Proofs.Tensor.AxisAdjoint
public import NN.Proofs.Tensor.AxisLinear
public import NN.Proofs.Tensor.Basic

/-!
# Tensor Proofs

Stable umbrella for TorchLean's tensor proof layer.

The folder is split by role:

- `NN.Proofs.Tensor.Algebra` contains backend-generic algebra over semirings. It is the right import
  for autograd soundness proofs that should not commit to `ℝ`.
- `NN.Proofs.Tensor.Basic` contains the real-valued tensor toolkit used by analysis,
  Lipschitz, normalization, attention, and model-level proofs.

Use this umbrella from public entrypoints and CI. Import the leaf modules directly when a proof
should keep its dependencies narrow.
-/

@[expose] public section
