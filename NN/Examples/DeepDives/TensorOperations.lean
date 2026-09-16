/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Widgets

/-!
# Tensor Operations

This example continues after `NN.Examples.Quickstart.TensorBasics`. It shows indexing, `einsum`,
batch axes, reshaping, elementwise operations, and editor widgets on the same shape-indexed tensor
type used by models and specifications.

Build with `lake build NN.Examples.DeepDives.TensorOperations`, then inspect the
`#tensor_view` commands in the editor. This module has no command-line entry point.
-/

@[expose] public section

namespace NN.Examples.DeepDives.TensorOperations

open TorchLean
open TorchLean.Tensor

/-! ## Construction and indexing -/

/-- A matrix literal. Its element type and both dimensions are visible in the type. -/
def matrix : Tensor Float [2, 3] :=
  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

/-- Selecting one coordinate on the leading axis returns a tensor of the remaining shape. -/
def firstRow : Tensor Float [3] :=
  matrix[0]

#tensor_view matrix
#tensor_view firstRow
#tensor_stats_view matrix

/-! ## Linear algebra -/

/-- A `4 × 3` parameter tensor. -/
def weights : Tensor Float [4, 3] :=
  Tensor.full [4, 3] 0.1

/-- A three-component input vector. -/
def input : Tensor Float [3] :=
  [1.0, 2.0, 3.0]

/-- Matrix multiplication preserves the output dimension in the result type. -/
def linearOutput : Tensor Float [4] :=
  einsum weights, input "output input, input -> output"

#tensor_view linearOutput
#tensor_stats_view linearOutput

/-! ## Batches and reshaping -/

/-- Two `2 × 3` samples stacked along a leading batch axis. -/
def batch : Tensor Float [2, 2, 3] :=
  [
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
  ]

/-- Indexing a rank-three tensor drops the outer axis, giving a `[2, 3]` matrix. -/
def firstSample : Tensor Float [2, 3] :=
  batch[0]

/-- The second batch element, same shape. -/
def secondSample : Tensor Float [2, 3] :=
  batch[1]

#tensor_view batch
#tensor_view firstSample
#tensor_view secondSample

/-- A flat tensor whose values will be viewed at another shape. -/
def flatValues : Tensor Float [6] :=
  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

/-- Reshape a vector without changing row-major element order. -/
def reshapedMatrix : Tensor Float [2, 3] :=
  flatValues.reshape [2, 3]

#tensor_view reshapedMatrix

/-! ## Shape-preserving transformations -/

/-- A constant matrix of twos for the elementwise addition example. -/
def tensorA : Tensor Float [2, 2] :=
  Tensor.full [2, 2] 2.0

/-- A constant matrix of threes; adding it to `tensorA` produces a matrix of fives. -/
def tensorB : Tensor Float [2, 2] :=
  Tensor.full [2, 2] 3.0

/--
Ordinary `+` on tensors is elementwise, and the shapes must already agree.

There is no implicit broadcasting here: a shape mismatch is a type error, not a runtime surprise.
-/
def elementwiseSum : Tensor Float [2, 2] :=
  tensorA + tensorB

/-- A pointwise transformation has the same shape discipline as a gradient buffer. -/
def doubledOutput : Tensor Float [4] :=
  Tensor.map (2.0 * ·) linearOutput

#eval elementwiseSum
#tensor_view elementwiseSum
#tensor_view doubledOutput

end NN.Examples.DeepDives.TensorOperations
