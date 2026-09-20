/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Module.Core
public import NN.Tensor.Conversion
public import NN.Spec.Layers.Pooling.Spatial

/-!
# Pooling Modules

The wrappers in this file preserve a leading channel dimension and pool over an arbitrary vector
of spatial dimensions. Padding, stride, and window extents are independent on every axis.
-/

@[expose] public section


namespace Spec.Module
open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-- Wrap arbitrary-rank channels-first max pooling as a `Spec.Module`. -/
def maxPool {d C : Nat} {inSpatial kernel stride padding : TorchLean.Tensor Nat [d]}
    {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
    {hStride : ∀ i : Fin d, stride.getScalar i ≠ 0}
    (m : MaxPoolSpec d kernel stride padding hKernel hStride) :
    Spec.Module α
      (Shape.ofList (C :: (Tensor.to inSpatial (List Nat))))
      (Shape.ofList (C :: Tensor.to
        (poolOutSpatialPad inSpatial kernel stride padding) (List Nat))) :=
  { forward := maxPoolSpec m
    kind := "MaxPool"
    pythonExpr := "nn.MaxPool(...)" }

/-- Wrap arbitrary-rank channels-first average pooling as a `Spec.Module`. -/
def avgPool {d C : Nat} {inSpatial kernel stride padding : TorchLean.Tensor Nat [d]}
    {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
    {hStride : ∀ i : Fin d, stride.getScalar i ≠ 0}
    (m : AvgPoolSpec d kernel stride padding hKernel hStride) :
    Spec.Module α
      (Shape.ofList (C :: (Tensor.to inSpatial (List Nat))))
      (Shape.ofList (C :: Tensor.to
        (poolOutSpatialPad inSpatial kernel stride padding) (List Nat))) :=
  { forward := avgPoolSpec m
    kind := "AvgPool"
    pythonExpr := "nn.AvgPool(...)" }

end Spec.Module
