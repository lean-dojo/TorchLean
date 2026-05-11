/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.Core.SpecApprox
public import NN.Proofs.RuntimeApprox.Core.Tolerance

/-!
# Runtime Approximation Core

Backend-independent predicates and tolerance budgets for runtime-to-spec approximation.

This layer provides:
- `ApproxTol`, the absolute/relative/slack tolerance object used for “close enough” claims;
- scalar approximation predicates over `ℝ`;
- tensor/context approximation predicates that compare runtime tensors after mapping them into the
  real-valued spec world.

It deliberately contains no rounding model, no FP32 specialization, and no graph semantics. Those
are layered on top by `NN.Proofs.RuntimeApprox.Rounding`, `NN.Proofs.RuntimeApprox.Graph`,
`NN.Proofs.RuntimeApprox.NF`, and `NN.Proofs.RuntimeApprox.FP32`.
-/

@[expose] public section

