/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models.Gmm
public import NN.Spec.Module.SpecModule

/-!
# Gaussian mixture model (GMM) as an `NNModuleSpec`

The GMM model spec defines the math for mixture weights, means, and covariances.
This file adds a small `NNModuleSpec` wrapper so the GMM can be composed/exported.
-/

@[expose] public section


namespace Spec

open Tensor
open ModSpec

variable {α : Type} [Context α]

/-- GMM module specification following `NNModuleSpec`. -/
def GMMModuleSpec {nComponents nFeatures : Nat} (m : GMMSpec α nComponents nFeatures) :
  NNModuleSpec α (.dim nFeatures .scalar) (.dim nComponents .scalar) :=
{
  forward := gmmForwardSpec m,
  kind := "GMM",
  export_func := {
    toPyTorch := "UnsupportedLayer(\"GMM\", \"torch.distributions.MixtureSameFamily\")",
    dimensions := (nFeatures, nComponents)
  }
}

end Spec

