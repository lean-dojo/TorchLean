/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models.Resnet
public import NN.Spec.Module.SpecModule

/-!
# ResNet as an `NNModuleSpec`

The ResNet spec model provides `ResNetSpec.forward` and related gradients.
This file adds a module wrapper for composition and export tooling.
-/

@[expose] public section


namespace Spec

open Tensor
open ModSpec

variable {α : Type} [Context α]

/-- Wrap `ResNetSpec.forward` as an `NNModuleSpec`. -/
def ResNetModuleSpec {cfg : ResNetConfig} {inputChannels numClasses inH inW : Nat}
  (h1 : inputChannels ≠ 0) (h2 : numClasses ≠ 0) (hCfg : cfg.WF)
  (resnet : ResNetSpec cfg α inputChannels numClasses h1 h2 hCfg)
  (h3 : inH ≠ 0) (h4 : inW ≠ 0) :
  NNModuleSpec α (.dim inputChannels (.dim inH (.dim inW .scalar))) (.dim numClasses .scalar) :=
{
  forward := fun x => ResNetSpec.forward h1 h2 hCfg resnet x h3 h4,
  kind := "ResNet",
  export_func := {
    toPyTorch := "UnsupportedLayer(\"ResNet\", \"torchvision.models.resnet\")",
    dimensions := (inputChannels, numClasses)
  }
}

end Spec
