/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Widget.Tensor
public import TorchLean.Tensor.Einops.Elab.Einsum
public import TorchLean.Tensor.Literal
public import TorchLean.Tensor.Operations
meta import TorchLean.Tensor.Einops.Elab.Einsum
meta import TorchLean.Tensor.Literal
meta import TorchLean.Tensor.Operations

/-!
# Tensor Operations

A second tensor example, to read after `TorchLeanExamples.Quickstart.Tensors`: indexing,
`einsum`, batch axes, reshaping, elementwise operations and editor widgets, on the same
shape-indexed tensor type that models and specifications use.

The module has no command-line entry point. Build it with
`scripts/lake.sh build TorchLeanExamples.DeepDives.TensorOperations`, then open it in the editor and
put the cursor on a `#tensor_view` or `#tensor_stats_view` command to see its panel.
-/

@[expose] public section

namespace TorchLean.Examples.TensorOperations

open TorchLean.Tensor

/-! ## Construction and indexing -/

/-- A matrix literal, whose type shows its element type and both dimensions. -/
def matrix : Tensor Float [2, 3] :=
  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

/-- Selecting one coordinate on the leading axis returns a tensor of the remaining shape. -/
def firstRow : Tensor Float [3] :=
  matrix[0]

#tensor_view matrix
#tensor_view firstRow
#tensor_stats_view matrix

/-! ## Linear algebra -/

/-- A `4 × 3` weight matrix with every entry `0.1`. -/
def weights : Tensor Float [4, 3] :=
  Tensor.full [4, 3] 0.1

/-- A three-component input vector. -/
def input : Tensor Float [3] :=
  [1.0, 2.0, 3.0]

/-- Matrix-vector product through `einsum`, with the output dimension `4` in the result type. -/
def linear : Tensor Float [4] :=
  einsum weights, input "output input, input -> output"

#tensor_view linear
#tensor_stats_view linear

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

/-- The second batch element, also a `[2, 3]` matrix. -/
def secondSample : Tensor Float [2, 3] :=
  batch[1]

#tensor_view batch
#tensor_view firstSample
#tensor_view secondSample

/-- A flat tensor whose values will be viewed at another shape. -/
def flat : Tensor Float [6] :=
  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

/-- Reshape a vector without changing row-major element order. -/
def reshaped : Tensor Float [2, 3] :=
  flat.reshape [2, 3]

#tensor_view reshaped

/-! ## Shape-preserving transformations -/

/-- A constant matrix of twos for the elementwise addition example. -/
def a : Tensor Float [2, 2] :=
  Tensor.full [2, 2] 2.0

/-- A constant matrix of threes, so `a + b` is a matrix of fives. -/
def b : Tensor Float [2, 2] :=
  Tensor.full [2, 2] 3.0

/--
Elementwise addition with ordinary `+`, which needs both operands at the same shape.

`+` does no implicit broadcasting, so mismatched shapes are a type error, caught when the file
elaborates.
-/
def sum : Tensor Float [2, 2] :=
  a + b

/-- `Tensor.map` doubles every entry of `linear` and keeps its shape `[4]`. -/
def doubled : Tensor Float [4] :=
  Tensor.map (2.0 * ·) linear

#eval sum
#tensor_view sum
#tensor_view doubled

end TorchLean.Examples.TensorOperations
