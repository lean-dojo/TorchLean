# Optimization Theory

This folder contains proof layer optimizer and first-order optimization material. Runtime update
equations live in `NN/Runtime/Optim`; this folder packages those equations into reusable theorem
interfaces and proves facts about convergence, fallback cases, and optimizer extension points.

## Files

- `FirstOrder.lean`: relations between executable first-order update rules (zero-decay AdamW
  agrees with Adam over `ℝ`).
- `GDLinearConvergence.lean`: the one-step squared-norm contraction of `x ↦ x - η g x` under
  strong monotonicity and a Lipschitz bound on `g`.
- `StronglyConvexGD.lean`: the contraction factor `q`, its step-size conditions, the iterated and
  limiting convergence statements, and the scalar quadratic warm-up (`ScalarGD`).
- `SmoothStrongConvexBridge.lean`: from `StrongConvexOn` or first-order strong convexity of `f` to
  strong monotonicity of `∇ f`, and linear-convergence corollaries that additionally assume the
  gradient is Lipschitz. The Lipschitz-gradient half is not derived from smoothness of `f` here.
- `OptimizerLaws.lean`: a generic `TensorOptimizer` interface over runtime optimizers, plus
  compositional step-stream laws.
- `Muon.lean`: umbrella for the Muon proof layer, which is split into
  - `Muon/Core.lean`: orthogonalizer contracts (exact and entrywise-approximate column Gram) and
    the checked backend records;
  - `Muon/Certificates.lean`: step certificates tying a backend contract to one executable Muon
    update;
  - `Muon/NewtonSchulz.lean`: the Newton-Schulz polynomial backend, its residual-checked
    packaging, and the fixed-point exact backend;
  - `Muon/QR.lean`: the real-valued QR backend with positive-pivot certificates, together with the
    real-valued Newton-Schulz facts at an exactly column-orthogonal matrix.

## Muon and GaLore-Style Updates

Muon is represented as momentum plus an explicit orthogonalizer backend. The runtime update can use
an identity orthogonalizer, an exact orthogonalizer, or a future optimized backend. The proof layer
states what must be true of the backend output, for example exact $Q^\mathsf{T}Q=I$ or an entrywise
bound on the Gram residual.

GaLore-style code is treated as projected-gradient structure: `NN/Runtime/Optim/Optimizers.lean`
defines the `GaLore.Projector` record, the identity projector, and the projected-SGD update. No
theorem about that update is proved in this folder yet. The intended first theorem-level fact is
the fallback that the projected update reduces to ordinary SGD when the projector is the identity;
until it is stated and proved, the identity projector is a runtime default rather than a checked
baseline case.

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
