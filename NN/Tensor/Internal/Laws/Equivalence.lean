/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Laws.Equivalence.Index
public import NN.Tensor.Internal.Laws.Equivalence.Lowering
public import NN.Tensor.Internal.Laws.Equivalence.Plan
public import NN.Tensor.Internal.Laws.Equivalence.Semantics

/-!
# Rearrangement equivalence laws

This stable import collects compact row-major index theory, checked-plan
comparison, semantic equivalence, and compiler-level congruence.
-/
