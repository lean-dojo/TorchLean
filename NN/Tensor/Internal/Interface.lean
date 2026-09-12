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
public meta import NN.Tensor.Internal.Elab -- shake: keep
public meta import NN.Tensor.Internal.Tactic -- shake: keep
public import NN.Tensor.Internal.Language -- shake: keep
public import NN.Tensor.Internal.Laws -- shake: keep
public import NN.Tensor.Internal.Lowering -- shake: keep
public import NN.Tensor.Internal.Runtime -- shake: keep
public import NN.Tensor.Internal.Semantics -- shake: keep
public import NN.Tensor.Internal.Representation -- shake: keep

/-!
# Tensor Internal Interface

The literal operation syntax, tactics, verified semantics, lowering laws, and
native runtime support behind ordinary `TorchLean.Tensor` programs. The
literal tensor-pattern term forms are scoped syntax; write `open TorchLean.Tensor`
to use them.
-/
