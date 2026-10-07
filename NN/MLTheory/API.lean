/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

-- These public re-exports define the theory import surface; retain them when auditing imports.
public import NN.MLTheory.CROWN.Proofs.Distillation
public import NN.MLTheory.CROWN.Proofs.LayerNormDirected
public import NN.MLTheory.CROWN.Proofs.SoundnessProofs
public import NN.MLTheory.Generative.Diffusion
public import NN.MLTheory.Generative.Latent
public import NN.MLTheory.LearningTheory
public import NN.MLTheory.Optimization.FirstOrder
public import NN.MLTheory.Optimization.OptimizerLaws
public import NN.MLTheory.Optimization.SmoothStrongConvexBridge
public import NN.MLTheory.Proofs
public import NN.MLTheory.SelfSupervised

/-!
# `NN.MLTheory.API`

This is the recommended entrypoint for TorchLean's formal “ML theory” layer.

It collects specifications, executable checkers, and theorems into a single import. Import a focused
submodule when you only need one topic.

## Optimization theory

The optimization layer has three levels:

- executable optimizer equations over `TorchLean.Tensor`s;
- exact `ℝ` convergence theorems for gradient-descent-style operators;
- a calculus bridge from strong convexity to strong monotonicity of `∇f`.

The tensor/runtime and real-analysis layers are deliberately separate. To use a convergence theorem
for a concrete model, a model-specific bridge still has to identify the runtime gradient with the
mathematical operator and account for floating-point error.

## Self-supervised objectives

The SSL theory modules formalize a finite predictive-view objective algebra:

- MAE is predictive-view SSL with identity/pixel targets;
- JEPA is predictive-view SSL with latent target representations;
- VICReg and Barlow-style terms are reusable geometry/non-collapse guards; and
- masked/context-target prediction can be read as finite view-graph energy.

The concrete Euclidean layer also proves that positive-edge alignment energy is nonnegative, that
fully collapsed embeddings can still obtain zero alignment energy, and that a positive
variance-floor guard assigns positive objective value to collapsed representations in nonzero
dimension.

API training helpers can use these objectives with masked/reconstruction or joint-embedding targets
and an MLP, CNN, ViT, Mamba block, or custom model.

## Verified-network integration

This entrypoint includes CROWN soundness proofs and finite-precision approximation bounds.
For graph-level forward and backward rounding-error bounds, import `NN.Proofs.RuntimeApprox`.

Notes:

- Import `NN.MLTheory.CROWN.Lyapunov.Certificate` for Lyapunov certificate semantics. Tactic
  frontends for external certificate tooling also need explicit imports.
- This module does not define additional convenience APIs; those belong in `NN.Runtime` or
  `NN.Examples` rather than the theory layer.
-/

@[expose] public section
