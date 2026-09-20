/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor.Internal.Interface
public import NN.Tensor.Conversion
public import NN.Tensor.Constructors
public import NN.Tensor.Coordinate
public import NN.Tensor.LinearAlgebra
public import NN.Tensor.Operations
public import NN.Tensor.Packing
public import NN.Tensor.Reductions
public import NN.Tensor.Storage

/-!
# Tensors

Canonical public import for the complete `TorchLean.Tensor` surface. It
includes construction and ordinary tensor operations together with the
verified pattern language, lowering, runtime, laws, and tactics.

Runtime tapes that need heterogeneous tensor packs import `NN.Tensor.ShapeErasure` explicitly.
That boundary is intentionally absent from this application-facing umbrella.
-/
