/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Core
public import NN.Spec.Core.Tensor.Constructors
public import NN.Spec.Core.Tensor.Linalg
public import NN.Spec.Core.Tensor.Vec

/-!
# Tensor API

Umbrella import for the small tensor subchapter:
- `Core`: the shape-indexed tensor datatype and accessors;
- `Constructors`: total tensor builders;
- `Linalg`: matrix/vector primitives; and
- `Vec`: the vector-specialized `Tensor α (.dim n .scalar)` interface.

Elementwise operations, reductions, broadcasting, flattening, and shape-changing helpers live one
level up in `NN.Spec.Core.TensorOps` and `NN.Spec.Core.TensorReductionShape`.
-/

@[expose] public section
