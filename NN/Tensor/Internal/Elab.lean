/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public meta import NN.Tensor.Internal.Elab.Einsum
public meta import NN.Tensor.Internal.Elab.Pack
public meta import NN.Tensor.Internal.Elab.TensorLiteral
public meta import NN.Tensor.Internal.Elab.Transform
public import NN.Tensor.Internal.Runtime -- shake: keep
import NN.Spec.Core.Tensor

/-!
# Tensor Pattern Elaborators

Public import boundary for the direct `rearrange`, `expand`, `reduce`,
`einsum`, `pack`, `unpack`, and `parse_shape` term forms. The forms are
scoped syntax activated by `open TorchLean.Tensor`.
-/
