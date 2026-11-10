# Optimizers

This folder holds TorchLean's optimizer updates and the proofs about them. `Update.lean` writes
SGD, momentum SGD, AdaGrad, RMSProp, Adam, AdamW, Adadelta, GaLore-style projected SGD and
Muon-style momentum as pure functions on `Tensor α s`. `Config.lean` and `Constructor.lean` give
the `optim` namespace (`optim.sgd`, `optim.adam` and the rest) that the trainer takes. The runtime
wrappers that keep one optimizer state per parameter are one level up, in
`TorchLean/NN/Optimizer.lean`. The proof modules relate two updates, state the step-stream laws of
the generic `TensorOptimizer` interface, prove convergence of gradient descent, and give the
backend contracts of Muon. The subfolders `GD/` and `Muon/` hold the convergence and Muon proofs.

## Files

- `Update.lean`: the state record and update equation of each optimizer, which return the next
  state and parameters.
- `Config.lean`: optimizer algorithms and hyperparameters shared by the trainer and runtime APIs.
- `Constructor.lean`: the `optim` configurations, `Optimizer.validateFor`, and the runtime-only
  constructors `optim.muon` and `optim.galore.sgd`, which take a backend from the caller.
- `AdamW.lean`: projected SGD with the identity projector agrees with SGD at every scalar
  type, and zero-decay AdamW agrees with Adam over `ℝ`.
- `GD.lean`: the one-step squared-norm contraction of `x ↦ x - η g x` under
  strong monotonicity and a Lipschitz bound on `g`.
- `GD/Convergence.lean`: the contraction factor `q`, its step-size conditions, the iterated and
  limiting convergence statements, and the scalar quadratic warm-up (`ScalarGD`).
- `GD/StrongConvexity.lean`: from `StrongConvexOn` or first-order strong convexity of `f` to
  strong monotonicity of `∇ f`, and linear-convergence corollaries that take a Lipschitz bound on
  the gradient as a hypothesis.
- `TensorOptimizer.lean`: a generic `TensorOptimizer` interface over runtime optimizers, plus
  compositional step-stream laws.
- `Muon.lean` and `Muon/`: the Muon proof layer, in four modules:
  - `Muon.lean`: orthogonalizer contracts (exact and entrywise-approximate column Gram) and
    the checked backend records.
  - `Muon/Certificates.lean`: step certificates tying a backend contract to one executable Muon
    update.
  - `Muon/NewtonSchulz.lean`: the Newton-Schulz polynomial backend, its residual-checked
    packaging, and the fixed-point exact backend.
  - `Muon/QR.lean`: the real-valued QR backend with positive-pivot certificates, together with the
    real-valued Newton-Schulz facts at an exactly column-orthogonal matrix.

## Muon and GaLore-Style Updates

Muon is momentum SGD plus an orthogonalization backend, a `Muon.Orthogonalizer` field of the
runtime state that turns the momentum buffer into the update direction. The runtime update takes
any such backend: the identity, the Newton-Schulz polynomial of `Muon/NewtonSchulz.lean`, or the
`Q` factor of the real QR specification (`Muon/QR.lean`). The proof layer states what the backend
output must satisfy, for example exact $Q^\mathsf{T}Q=I$ or an entrywise bound on the Gram residual.

`TorchLean/NN/Optimizer/Update.lean` treats GaLore as projected gradient descent: it defines the
`GaLore.Projector` record, the identity projector `GaLore.identityProjector` and the projected-SGD
update. `GaLore.update_params_eq_sgd` (`AdamW.lean`) proves that the update with
the identity projector gives the parameters of ordinary SGD. No other theorem about the projected
update is proved here.

## Adding An Optimizer

An optimizer joins this layer in five steps:

1. Implement the pure state and update equation in `TorchLean/NN/Optimizer/Update.lean`.
2. If it should be user-facing, add a public constructor in
   `TorchLean/NN/Optimizer/Constructor.lean` or a configuration record in
   `TorchLean/NN/Optimizer/Config.lean`.
3. Package the update as a `TensorOptimizer` in `TensorOptimizer.lean` when it takes part in
   generic step-stream reasoning.
4. Prove optimizer-specific relations or analytic results in this folder.
5. Document every backend or approximation assumption.
