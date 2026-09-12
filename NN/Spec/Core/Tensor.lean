/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- The `export Spec (...)` block below is the point of this file, so its imports have to stay and
-- so does every downstream import of it: reaching the leaves directly loses the aliases.
module -- shake: keep-downstream

public import NN.Spec.Core.Tensor.Constructors -- shake: keep
public import NN.Spec.Core.Tensor.Core -- shake: keep
public import NN.Spec.Core.Tensor.Linalg -- shake: keep

/-!
# Tensor

Umbrella module for the core tensor API.

Most downstream modules import this umbrella. The concrete tensor definitions live in focused files:
- `NN.Spec.Core.Tensor.Core`          (datatype + accessors)
- `NN.Spec.Core.Tensor.Constructors`  (total builders)
- `NN.Spec.Core.Tensor.Linalg`        (matrix and vector operations)

Elementwise ops and reductions remain in:
- `NN.Spec.Core.TensorOps`
- `NN.Spec.Core.TensorReductionShape`
-/

@[expose] public section


namespace TorchLean.Tensor

-- Expose the semantic accessors and constructors through the canonical tensor API.
export Spec (shapeOf getSpec get get2 getAtOrZero finZero getHead getTail
  tensorCast replicate
  sliceRangeSpec
  singleton padLeft
  identityTensorSpec matMulSpec matVecMulSpec vecMatMulSpec outerProductSpec)

end TorchLean.Tensor
