/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models.Autoencoder
public import NN.Spec.Module.Core

/-!
# Autoencoder as an `Spec.Module`

The autoencoder spec model defines the forward pass and its VJP pieces.
This file adds the `Spec.Module` wrapper so it can be composed with other modules and exported.
-/

@[expose] public section


namespace Spec.Module

open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- Autoencoder module specification following `Spec.Module`. -/
def autoencoder {inputDim hiddenDim : Nat} (m : AutoencoderSpec α inputDim hiddenDim) :
  Spec.Module α ([inputDim]) ([inputDim]) :=
{
  forward := autoencoderForwardSpec m,
  kind := "Autoencoder",
  pythonExpr :=
    let activation :=
      match m.activation with
      | .relu => "nn.ReLU()"
      | .gelu => "nn.GELU(approximate='tanh')"
      | .silu => "nn.SiLU()"
      | .tanh => "nn.Tanh()"
      | .sigmoid => "nn.Sigmoid()"
    (s!"nn.Sequential(nn.Linear({inputDim}, {hiddenDim}), " ++
        s!"{activation}, " ++
        s!"nn.Linear({hiddenDim}, {inputDim}))")
}

end Spec.Module
