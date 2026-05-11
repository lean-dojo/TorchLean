/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Generative.Diffusion
public import NN.Spec.Generative.Latent

/-!
# Spec generative models

Umbrella import for generative-model specification utilities.

The diffusion chapter defines noising processes, reverse samplers, and PF-ODE-style dynamics. The
latent chapter provides reusable VAE/VQ-VAE-style latent helpers. Both are written as pure
scalar-polymorphic specs so runtime and proof layers can share the same equations.
-/

@[expose] public section

