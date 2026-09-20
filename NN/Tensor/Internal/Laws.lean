/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Laws.Equivalence
public import NN.Tensor.Internal.Laws.MixedRadix
public import NN.Tensor.Internal.Laws.PackIndex
public import NN.Tensor.Internal.Laws.ReductionIndex
public import NN.Tensor.Internal.Laws.RowMajor

/-!
# Tensor-pattern laws

Equivalence, row-major indexing, mixed-radix, pack, reduction, and matrix
theorems for checked and lowered operations.
-/
