/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.GlobalPooling
public import NN.Spec.Module.SpecModule

/-!
# Global pooling as `NNModuleSpec`s

Global pooling is a common "bridge" from convolutional feature maps to a classifier head:
it reduces the spatial dimensions and keeps only the channel axis.

This file wraps the flat global pooling specs from `NN/Spec/Layers/GlobalPooling.lean` as
`NNModuleSpec`s so they can be composed with `SpecChain` (and recognized by export tooling).

We focus on the flat variants because most model definitions here use them directly:

- `GlobalAvgPool2DFlatModuleSpec` returns a length-`inC` vector.
- `GlobalMaxPool2DFlatModuleSpec` returns a length-`inC` vector.
-/

@[expose] public section


namespace Spec

open Tensor
open ModSpec

variable {α : Type} [Context α]

/-- Global average pooling (flattened): `(C,H,W) -> (C)`. -/
def GlobalAvgPool2DFlatModuleSpec {inC inH inW : Nat}
  (hH : inH ≠ 0) (hW : inW ≠ 0)
  (layer : GlobalAvgPool2DSpec := {}) :
  NNModuleSpec α
    (.dim inC (.dim inH (.dim inW .scalar)))
    (.dim inC .scalar) :=
{ forward := fun x =>
    globalAvgPool2dFlatSpec (α := α) (inC := inC) (inH := inH) (inW := inW) hH hW layer x
  kind := "GlobalAvgPool2DFlat"
  export_func := {
    toPyTorch := "nn.AdaptiveAvgPool2d((1, 1)) + flatten"
    dimensions := (inC, inC)
  } }

/-- Global max pooling (flattened): `(C,H,W) -> (C)`. -/
def GlobalMaxPool2DFlatModuleSpec {inC inH inW : Nat}
  (hH : inH ≠ 0) (hW : inW ≠ 0)
  (layer : GlobalMaxPool2DSpec := {}) :
  NNModuleSpec α
    (.dim inC (.dim inH (.dim inW .scalar)))
    (.dim inC .scalar) :=
{ forward := fun x =>
    globalMaxPool2dFlatSpec (α := α) (inC := inC) (inH := inH) (inW := inW) hH hW layer x
  kind := "GlobalMaxPool2DFlat"
  export_func := {
    toPyTorch := "nn.AdaptiveMaxPool2d((1, 1)) + flatten"
    dimensions := (inC, inC)
  } }

end Spec
