/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Generative.Diffusion.ForwardGaussian
public import NN.MLTheory.Generative.Diffusion.ImageDDIM
public import NN.MLTheory.Generative.Diffusion.Samplers

/-!
# Diffusion theory

This entrypoint collects the diffusion-theory facts that connect TorchLean's executable sampler
specifications to the mathematical language used in diffusion and score-based generative modeling.

This entrypoint collects:
- `ForwardGaussian`: a mathlib-backed result showing that affine forward noising of a standard
  Gaussian remains Gaussian.
- `Samplers`: proved boundary, dynamics-adapter, and Euler-stability facts for DDPM, DDIM, and
  probability-flow samplers.
- `ImageDDIM`: equalities connecting the public image helpers to their coefficient indexing,
  denominator floor, and clipped reconstruction specifications.

Probabilistic claims and executable sampler claims stay separate. The spec layer defines the noising
and reverse-update functions; this theory layer records the mathematical facts we can prove cleanly
about those definitions.
-/

@[expose] public section
