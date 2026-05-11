/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models.LinearRegression
public import NN.Spec.Module.SpecModule

/-!
# Linear regression as an `NNModuleSpec`

The spec model file (`NN/Spec/Models/LinearRegression.lean`) defines the math: forward and
backward/VJP pieces.

This file provides the small `NNModuleSpec` wrapper so linear regression can be composed via
`SpecChain` and recognized by export tooling (PyTorch string renderings, dimension metadata, etc.).
-/

@[expose] public section


namespace Spec

open Tensor
open ModSpec

variable {α : Type} [Context α]

/-- Package a fixed-parameter linear regression as a `NNModuleSpec`.

PyTorch analogy: `nn.Linear(inDim, 1)`.
-/
def LinearRegressionModuleSpec {inDim : Nat}
  (model : LinearRegressionSpec α inDim) :
  NNModuleSpec α (.dim inDim .scalar) .scalar :=
{
  forward := fun x => linearRegressionForwardSpec model x,
  kind := "LinearRegression",
  export_func := {
    toPyTorch := s!"nn.Linear({inDim}, 1)",
    dimensions := (inDim, 1)
  }
}

end Spec

