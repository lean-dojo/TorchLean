/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Semantics.Einsum
public import NN.Tensor.Internal.Semantics.Pack
public import NN.Tensor.Internal.Semantics.Transform

/-!
# Operation semantics

Independent coordinate denotations for every checked operation family.
These definitions specify observable behavior without executing lowering or
native compiler code.
-/
