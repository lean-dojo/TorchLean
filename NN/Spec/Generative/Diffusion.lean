/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Generative.Diffusion.Core
public import NN.Spec.Generative.Diffusion.Schedule
public import NN.Spec.Generative.Diffusion.ForwardProcess
public import NN.Spec.Generative.Diffusion.ReverseDDPM
public import NN.Spec.Generative.Diffusion.ReverseDDIM
public import NN.Spec.Generative.Diffusion.PFODE
public import NN.Spec.Generative.Diffusion.Loss

/-!
# Diffusion / flow specs (umbrella)

This is the curated public entrypoint for TorchLean's diffusion / flow spec layer.

It re-exports:

- a discrete VP schedule + forward noising (`qSample`),
- reverse samplers (DDPM and deterministic DDIM), and
- a continuous-time VP schedule + probability-flow ODE drift (`pfOdeRhs`).

All specs are scalar-polymorphic (`Context α`) so the same definitions can be reused for:

- runtime execution (`Float`, `IEEE32Exec`, `NeuralFloat`, …),
- proofs (`ℝ`), and
- verification backends (interval scalars, etc.).
-/

@[expose] public section

