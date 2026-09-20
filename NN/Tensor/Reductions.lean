/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor.Operations

/-!
# Public Tensor Reductions

Shape-polymorphic scalar reductions for ordinary tensor programs. Empty means use denominator one
(`meanDenominator`), the same totalized convention as `meanSpec` and the loss layer.
-/

@[expose] public section

namespace TorchLean.Tensor

/-- Traverse entries in row-major order without converting the buffer to an array. -/
@[inline] def foldl {α β : Type} [TorchLean.Storage α] {shape : Shape}
    (step : β → α → β) (initial : β) (tensor : Tensor α shape) : β :=
  Internal.Rep.foldl step initial tensor

/-- Largest absolute entry, or zero for an empty tensor. Self-unequal values are propagated:
for IEEE scalars this prevents a NaN from being hidden by the scalar `max` operation. -/
def maxAbs {α : Type} [TorchLean.Storage α] [MathFunctions α] [BEq α] [Max α] [Zero α]
    {shape : Shape} (tensor : Tensor α shape) : α :=
  foldl (fun largest value =>
    if !(largest == largest) then largest
    else if !(value == value) then value
    else max largest (MathFunctions.abs value)) 0 tensor

/-- Maximum absolute entrywise difference, with the exceptional-value policy of `maxAbs`. -/
def maxAbsDiff {α : Type} [TorchLean.Storage α] [MathFunctions α] [BEq α]
    [Max α] [Zero α] [Sub α] {shape : Shape} (actual expected : Tensor α shape) : α :=
  maxAbs (actual - expected)

/-- Add every tensor entry in row-major order. Public spelling of `sumSpec`. -/
abbrev sum {α : Type} [TorchLean.Storage α] [Add α] [Zero α]
    {shape : Shape} (tensor : Tensor α shape) : α :=
  sumSpec tensor

/-- Arithmetic mean of all entries, using denominator one for an empty tensor. -/
def mean {α : Type} [TorchLean.Storage α] [Add α] [Zero α] [Div α] [NatCast α]
    {shape : Shape} (tensor : Tensor α shape) : α :=
  tensor.sum / (meanDenominator shape : α)

/-- Under a full scalar `Context`, the public mean is the specification mean. -/
theorem mean_eq_meanSpec {α : Type} [TorchLean.Storage α] [Context α]
    {shape : Shape} (tensor : Tensor α shape) :
    mean tensor = meanSpec tensor :=
  rfl

/-- Mean squared difference between two equally shaped tensors. -/
def meanSquaredError {α : Type} [TorchLean.Storage α]
    [Add α] [Sub α] [Mul α] [Div α] [Zero α] [NatCast α]
    {shape : Shape} (predicted target : Tensor α shape) : α :=
  mean (square (predicted - target))

end TorchLean.Tensor
