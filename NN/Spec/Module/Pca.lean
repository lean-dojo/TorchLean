/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models.Pca
public import NN.Spec.Module.SpecModule

/-!
# PCA as an `NNModuleSpec`

The PCA spec model defines a projection `y = (x - mean) · componentsᵀ`.
This file provides the `NNModuleSpec` wrapper used for composition and export.
-/

@[expose] public section


namespace Spec

open Tensor
open ModSpec

variable {α : Type} [Context α]

/-- PCA module specification following `NNModuleSpec`. -/
def PCAModuleSpec {inDim outDim : Nat} (m : PCASpec α inDim outDim) :
  NNModuleSpec α (.dim inDim .scalar) (.dim outDim .scalar) :=
{
  forward := pcaForwardSpec m,
  kind := "PCA",
  export_func := {
    toPyTorch := s!"nn.Linear({inDim}, {outDim}, bias=False)",
    dimensions := (inDim, outDim)
  }
}

end Spec

