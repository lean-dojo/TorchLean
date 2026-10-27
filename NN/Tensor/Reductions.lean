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

/--
Traverse entries in row-major order without converting the buffer to an array.

The accumulator may have a different type from the entries and does not require a `Storage`
instance. For a configured binary tensor, we can accumulate in native binary32 without allocating
a converted tensor:

```lean
let total := tensor.foldl
  (fun (total : Float32) value => total + FloatLib.Floats.ExecFloat.Binary.toFloat32 value) 0
let average := total / (Tensor.meanDenominator shape : Float32)
```

The conversion, additions, and final division use the chosen accumulator's arithmetic. Widening
does not make those operations exact or prevent overflow for every input.
-/
@[inline] def foldl {α β : Type} [TorchLean.Storage α] {shape : Shape}
    (step : β → α → β) (initial : β) (tensor : Tensor α shape) : β :=
  Internal.Rep.foldl step initial tensor

/-- Maximum absolute entrywise difference, with the exceptional-value policy of `maxAbs`. -/
def maxAbsDiff {α : Type} [TorchLean.Storage α] [MathFunctions α] [BEq α]
    [Max α] [Zero α] [Sub α] {shape : Shape} (actual expected : Tensor α shape) : α :=
  maxAbs (actual - expected)

/-- Add every tensor entry in row-major order. Public spelling of `sumSpec`. -/
abbrev sum {α : Type} [TorchLean.Storage α] [Add α] [Zero α]
    {shape : Shape} (tensor : Tensor α shape) : α :=
  sumSpec tensor

/--
Arithmetic mean of all entries, using denominator one for an empty tensor.

The sum and element-count conversion both use the tensor's scalar type. Either can overflow in a
low-precision format even when the exact mean is representable. To select a wider accumulator,
convert the entries with `Tensor.map` before calling `mean`, or convert them inside `Tensor.foldl`
to accumulate in one pass. The latter also avoids allocating a wider tensor.
-/
def mean {α : Type} [TorchLean.Storage α] [Add α] [Zero α] [Div α] [NatCast α]
    {shape : Shape} (tensor : Tensor α shape) : α :=
  meanSpec tensor

/-- Under a full scalar `Context`, the public mean is the specification mean. -/
theorem mean_eq_meanSpec {α : Type} [TorchLean.Storage α] [Context α]
    {shape : Shape} (tensor : Tensor α shape) :
    mean tensor = meanSpec tensor :=
  rfl

end TorchLean.Tensor
