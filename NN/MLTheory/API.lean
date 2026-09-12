/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.SpecialFunctions.Bernstein
public import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
public import Mathlib.Data.Nat.Factorial.DoubleFactorial
-- Read as an umbrella: this file names no constant from the modules below, so `lake shake` sees
-- them as redundant. They stay because being importable under one name is the point, and because
-- `NN.MLTheory.API` is how the theory layer reaches a typecheck target at all.
public import NN.MLTheory.CROWN.Extras.AlphaConfig
public import NN.MLTheory.CROWN.Operators
public import NN.MLTheory.CROWN.Proofs.Distillation
public import NN.MLTheory.CROWN.Proofs.LayerNormDirected
public import NN.MLTheory.CROWN.Proofs.SoundnessProofs
public import NN.MLTheory.Generative.Diffusion
public import NN.MLTheory.Generative.Latent
public import NN.MLTheory.LearningTheory
public import NN.MLTheory.Optimization.FirstOrder
public import NN.MLTheory.Optimization.GDLinearConvergence
public import NN.MLTheory.Optimization.OptimizerLaws
public import NN.MLTheory.Optimization.SmoothStrongConvexBridge
public import NN.MLTheory.Optimization.StronglyConvexGD
public import NN.MLTheory.Proofs
public import NN.MLTheory.SelfSupervised

/-!
Public entry point for machine-learning theory modules.

The imports collected here expose optimization, robustness, generative-model, Lyapunov, and
self-supervised results without requiring users to know the internal proof layout.
-/

/-!
# `NN.MLTheory.API`

This is the recommended entrypoint for TorchLean's formal “ML theory” layer.

It collects the core specifications, executable checkers, and theorems into a single import. The
subdirectories still contain focused implementation modules, but users should not need separate
top-level umbrellas such as `NN.MLTheory.Optimization` or `NN.MLTheory.SelfSupervised`.

## Optimization theory

The optimization layer has three levels:
- executable optimizer equations over TorchLean `TorchLean.Tensor`s;
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

These are objective semantics, not special-purpose layers: API training helpers can feed the same
masked/reconstruction or joint-embedding targets into an MLP, CNN, ViT, Mamba block, or custom
model.

## Verified-network integration

This entrypoint also imports the CROWN soundness layer and the NeuralFloat error-bound layer so
verification statements and floating-point error statements compile together through one surface.

Notes:
- The tactic frontends for external certificate tooling are kept out of this umbrella; import them
  explicitly when needed. The Lyapunov certificate semantics are re-exported here as part of the
  current `NN.MLTheory` surface.
- This module does not define additional convenience APIs; those belong in `NN.Runtime` or
  `NN.Examples` rather than the theory layer.
-/

@[expose] public section
