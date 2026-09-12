/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Conv
public import NN.Spec.Module.Core
public import NN.Tensor.Conversion

/-!
# Convolution Modules

Convolution modules are parameterized by vectors of spatial extents. The same definitions cover
one-dimensional sequences, images, volumes, and higher-rank spatial data.
-/

@[expose] public section

namespace Spec.Module

open TorchLean TorchLean.Tensor

/-- Wrap an arbitrary-rank channels-first convolution as a `Spec.Module`. -/
def conv {α : Type} [TorchLean.Storage α] [Context α]
    {d inC outC : Nat} {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (m : ConvSpec d inC outC kernel stride padding α) :
    Spec.Module α
      (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat))))
      (Shape.ofList (outC :: Tensor.to
        (convOutSpatial inSpatial kernel stride padding) (List Nat))) :=
  { forward := convSpec m
    kind := "Conv"
    pythonExpr := "nn.Conv(...)" }

/-- Wrap an arbitrary-rank channels-first transposed convolution as a `Spec.Module`. -/
def convTranspose {α : Type} [TorchLean.Storage α] [Context α]
    {d inC outC : Nat} {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (m : ConvTransposeSpec d inC outC kernel stride padding α) :
    Spec.Module α
      (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat))))
      (Shape.ofList (outC :: Tensor.to
        (convTransposeOutSpatial inSpatial kernel stride padding) (List Nat))) :=
  { forward := convTransposeSpec m
    kind := "ConvTranspose"
    pythonExpr := "nn.ConvTranspose(...)" }

end Spec.Module
