/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Models.Generative.Autoencoder
public import NN.Examples.Models.Generative.Mae
public import NN.Examples.Models.Generative.Vae
public import NN.Examples.Models.Generative.VqVae
public import NN.Examples.Models.Generative.Gan
public import NN.Examples.Models.Generative.Diffusion

/-!
# Generative Model Examples

Runnable generative and self-supervised examples. Theory-level objective semantics live under
`NN.MLTheory`; these files are where we actually load data, train for a few steps, and write logs or
image artifacts.

`Diffusion` is one public command with typed branches for ImageNet64 (default) and CIFAR-10.
-/

@[expose] public section
