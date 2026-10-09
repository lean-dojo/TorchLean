/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Representation.Fiber.Aggregation
public import NN.Tensor.Internal.Representation.Fiber.Axis
public import NN.Tensor.Internal.Representation.Fiber.Basic
public import NN.Tensor.Internal.Representation.Fiber.Differential
public import NN.Tensor.Internal.Representation.Fiber.Mean
public import NN.Tensor.Internal.Representation.Fiber.Product
public import NN.Tensor.Internal.Representation.Fiber.Tie

/-!
# Finite fibers and tensor aggregation

A fiber groups the input coordinates that contribute to one output coordinate.
For a row sum, it contains the entries of that row. The same grouping describes
which inputs receive each output gradient during backward.

This module exports the grouping and cardinality lemmas, together with forward
and reverse identities for sum, product, mean, and min/max reductions. The
order-independent results require the stated algebraic laws; ordered
floating-point reductions use their separate semantics.
-/
