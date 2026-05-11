/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Generative.Latent.VAE
public import NN.MLTheory.Generative.Latent.VQVAE
public import NN.MLTheory.Generative.Latent.GAN
public import NN.MLTheory.Generative.Latent.Objective

/-!
# Latent generative model theory

This entrypoint collects the proved theory facts for TorchLean's latent generative model specs:

- VAE reparameterization and β-VAE objective decomposition;
- VQ-VAE codebook lookup and loss decomposition; and
- LSGAN generator/discriminator composition facts.
- shared weighted-objective algebra connecting continuous-latent, discrete-latent, and adversarial
  objectives.

The executable model equations and the heavier probabilistic/game assumptions stay separate. These
files prove the stable rewrite and optimization facts that examples, verifiers, and theory modules
can use without unfolding the model specs by hand.
-/

@[expose] public section
