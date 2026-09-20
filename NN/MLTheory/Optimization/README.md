# Optimization Theory

This folder contains proof layer optimizer and first-order optimization material. Runtime update
equations live in `NN/Runtime/Optim`; this folder packages those equations into reusable theorem
interfaces and proves facts about convergence, fallback cases, and optimizer extension points.

## Files

- `FirstOrder.lean`: relations between executable first-order update rules.
- `StronglyConvexGD.lean`: gradient-descent facts for strongly convex objectives.
- `SmoothStrongConvexBridge.lean`: from `StrongConvexOn` or first-order strong convexity of `f` to
  strong monotonicity of `∇ f`, and a linear-convergence corollary that additionally assumes the
  gradient is Lipschitz. The Lipschitz-gradient half is not derived from smoothness of `f` here.
- `GDLinearConvergence.lean`: linear-convergence statements for gradient descent under the stated
  hypotheses.
- `OptimizerLaws.lean`: a generic `TensorOptimizer` interface over runtime optimizers, plus
  compositional step-stream laws.
- `Muon.lean`: proof contracts for Muon-style orthogonalized momentum, including exact and
  approximate column-Gram conditions on matrix update directions.

## Muon and GaLore-Style Updates

Muon is represented as momentum plus an explicit orthogonalizer backend. The runtime update can use
an identity orthogonalizer, an exact orthogonalizer, or a future optimized backend. The proof layer
states what must be true of the backend output, for example exact $Q^\mathsf{T}Q=I$ or an entrywise
bound on the Gram residual.

GaLore-style code is treated as projected-gradient structure. The important theorem-level fallback
is that the projected update reduces to ordinary SGD when the projector is the identity. That gives
future projection backends a clean boundary: they may optimize memory or rank structure, but the
surrounding update semantics already have a checked baseline case.

## Adding A New Optimizer

The intended path is:

1. implement the pure state/update equation in `NN/Runtime/Optim/Optimizers.lean`;
2. expose a public configuration helper if the optimizer should be user-facing;
3. package the update as a `TensorOptimizer` in `OptimizerLaws.lean` when it participates in generic
   step-stream reasoning;
4. prove optimizer-specific relations or analytic results in this folder;
5. document any backend or approximation assumption explicitly.

Tests can show that a trainer runs. The files here are where reusable mathematical claims about the
update rule should live.
