/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models.GradientBoostedTrees
public import NN.Spec.Module.SpecModule

/-!
# Gradient boosted trees as an `NNModuleSpec`

The model spec defines the ensemble prediction function. This file adds the `NNModuleSpec` wrapper
for composition and export.
-/

@[expose] public section


namespace Spec

open Tensor
open ModSpec

variable {α : Type} [Context α]

/-- Gradient boosted trees as an `NNModuleSpec`. -/
def GradientBoostedTreesModuleSpec {nTrees maxDepth nFeatures : Nat}
  (model : GradientBoostedTreesSpec α nTrees maxDepth) :
  NNModuleSpec α (.dim nFeatures .scalar) .scalar :=
{
  forward := fun x => gradientBoostedTreesForwardSpec model x,
  kind := "GradientBoostedTrees",
  export_func := {
    toPyTorch :=
      "UnsupportedLayer(\"GradientBoostedTrees\", "
        ++ "\"sklearn.ensemble.GradientBoostingRegressor\")",
    dimensions := (nFeatures, 1)
  }
}

end Spec
