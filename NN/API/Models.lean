/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Models.KAN
public import NN.API.Models.Cnn
public import NN.API.Models.ResNet
public import NN.API.Models.Unet
public import NN.API.Models.Vit
public import NN.API.Models.Recurrent
public import NN.API.Models.CausalTransformer
public import NN.API.Models.Mamba
public import NN.API.Models.Generative
public import NN.API.Models.SelfSupervised
public import NN.API.Models.Diffusion
public import NN.API.Models.FNO
public import NN.API.Models.PPO

/-!
# Neural Architectures

Reusable executable model constructors and configuration records.

Individual files under `NN/API/Models/*` own the implementation of each architecture family.
Examples should import this API layer, then add only dataset loading, CLI parsing, and reporting.
Mathematical model specifications remain available through the focused `NN.Spec.Models` import.
-/
