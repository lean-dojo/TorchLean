/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Lowering.Einsum.Planning
public import NN.Tensor.Internal.Lowering.Pack
public import NN.Tensor.Internal.Lowering.Rearrange
public import NN.Tensor.Internal.Lowering.Reduce.View
public import NN.Tensor.Internal.Lowering.Repeat
public import NN.Tensor.Internal.Lowering.TransformFusion

/-!
# Verified operation lowering

Executable tensor programs, planning data, and correctness theorems relating
lowered operations to their independent coordinate semantics.
-/
