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

This facade exposes coordinate-fiber geometry, order-independent tensor
aggregation, and proved differential and reverse-mode identities for sum,
product, mean, and attained-value reductions.
-/
