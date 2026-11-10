/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Operations

/-!
# Public Tensor Reductions

`Tensor.maxAbsDiff` is the largest absolute entrywise difference. `Tensor.sum`, `Tensor.mean`
and `Tensor.meanSquaredError` are in `TorchLean/Tensor/Reduce.lean`, and the row-major fold
`Tensor.foldl` is in `TorchLean/Tensor/Rep.lean`.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean.Tensor

/-- Maximum absolute entrywise difference, with the exceptional-value policy of `maxAbs`. -/
def maxAbsDiff {α : Type} [TorchLean.Storage α] [MathFunctions α] [BEq α]
    [Max α] [Zero α] [Sub α] {shape : Shape} (actual expected : Tensor α shape) : α :=
  maxAbs (actual - expected)

end TorchLean.Tensor
