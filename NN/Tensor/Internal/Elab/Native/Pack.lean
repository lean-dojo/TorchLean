/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public meta import NN.Tensor.Internal.Elab.Native.Pack.Dispatch
public meta import NN.Tensor.Internal.Elab.Native.Pack.Unpack
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Data.Finset.Attr
import Mathlib.Tactic.SetLike

/-!
# Certified native pack and unpack

This facade exposes the independent native unpack specialization and native
pack dispatch compilers.
-/
