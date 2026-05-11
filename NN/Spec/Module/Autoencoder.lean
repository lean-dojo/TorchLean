/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models.Autoencoder
public import NN.Spec.Module.SpecModule

/-!
# Autoencoder as an `NNModuleSpec`

The autoencoder spec model defines the forward pass and its VJP pieces.
This file adds the `NNModuleSpec` wrapper so it can be composed with other modules and exported.
-/

@[expose] public section


namespace Spec

open Tensor
open ModSpec

variable {α : Type} [Context α]

/-- Small helper used by exporters to render a PyTorch-style activation string. -/
def autoencoderActivationToPyTorch (activation_type : String) : String :=
  match activation_type with
  | "relu" => "nn.ReLU()"
  | "sigmoid" => "nn.Sigmoid()"
  | "tanh" => "nn.Tanh()"
  | _ => "nn.Identity()"

/-- Autoencoder module specification following `NNModuleSpec`. -/
def AutoencoderModuleSpec {inputDim hiddenDim : Nat} (m : AutoencoderSpec α inputDim hiddenDim) :
  NNModuleSpec α (.dim inputDim .scalar) (.dim inputDim .scalar) :=
{
  forward := autoencoderForwardSpec m,
  kind := "Autoencoder",
  export_func := {
    toPyTorch :=
      (s!"nn.Sequential(nn.Linear({inputDim}, {hiddenDim}), " ++
        s!"{autoencoderActivationToPyTorch m.activation_type}, " ++
        s!"nn.Linear({hiddenDim}, {inputDim}))"),
    dimensions := (inputDim, inputDim)
  }
}

end Spec
