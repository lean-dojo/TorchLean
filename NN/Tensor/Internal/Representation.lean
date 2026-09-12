/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Representation.Basic
public import NN.Tensor.Internal.Representation.Coordinate
public import NN.Tensor.Internal.Representation.Fiber
public import NN.Tensor.Internal.Representation.Promotion
public import NN.Tensor.Internal.Representation.Reduction
public import NN.Tensor.Internal.Representation.Segment
public import NN.Tensor.Internal.Representation.Vector

/-!
# Contiguous Tensor Core

Internal row-major storage, coordinates, finite fibers, reductions, segments,
and vector equivalences used to implement the public `TorchLean.Tensor`.
-/
