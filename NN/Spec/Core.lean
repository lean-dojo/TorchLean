/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Complex
public import NN.Spec.Core.Context
public import NN.Spec.Core.Scalar
public import NN.Spec.Core.Sequence
public import NN.Spec.Core.Shape
public import NN.Spec.Core.Tensor
public import NN.Spec.Core.TensorArray
public import NN.Spec.Core.TensorBridge
public import NN.Spec.Core.TensorGrad
public import NN.Spec.Core.TensorOps
public import NN.Spec.Core.TensorReductionShape
public import NN.Spec.Core.Utils

/-!
# Spec core

Umbrella import for TorchLean's core specification layer: scalar contexts, shape-indexed tensors,
runtime/list bridges, reductions, and gradient utilities.

These modules are intentionally pure. Runtime backends and verifier pipelines reuse the same
definitions instead of maintaining parallel meanings for tensor operations.
-/

@[expose] public section
