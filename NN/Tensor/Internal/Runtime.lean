/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.NormNum.Inv
import Mathlib.Tactic.NormNum.Pow
import Mathlib.Tactic.Positivity.Finset
public import NN.Tensor.Internal.Elab.Einsum.Loop -- shake: keep
public import NN.Tensor.Internal.Elab.Einsum.Parallel -- shake: keep
public import NN.Tensor.Internal.Elab.Einsum.Tiling -- shake: keep
public import NN.Tensor.Internal.Elab.Native.Reduce -- shake: keep
public import NN.Tensor.Internal.Elab.Native.Pointwise -- shake: keep
public import NN.Tensor.Internal.Elab.Native.Slice -- shake: keep
public import NN.Tensor.Internal.Elab.Native.Tensor -- shake: keep
public import NN.Tensor.Internal.Elab.Native.Transpose -- shake: keep

/-!
# Certified native runtime

Runtime declarations referenced by generated tensor-pattern terms. The
elaborator and planners remain meta-level implementation modules; this facade
contains the executable loops, builders, and their semantic correctness
theorems.
-/
