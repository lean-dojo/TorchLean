/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tactic.Verify
public import NN.Runtime.Autograd.IRExec.Correctness.SemanticEquivalence
public import NN.Tensor.Internal.Lowering.Einsum

/-!
# Verification rules for program lowering

Import this module explicitly when proving a lowering preserves semantics. The generic `verify`
tactic stays independent of the IR correctness development.

The graph rule needs both successful lowering and `NoRawLog`. The einsum rule compares an already
checked plan with its tensor semantics. Neither rule certifies the parser, native compilation,
or an external kernel; those require separate evidence.
-/

public meta section

attribute [verify]
  Runtime.Autograd.IRExec.denoteAll_eq_of_lowerToForwardGraph
  TorchLean.Tensor.Internal.Lowering.einsumTensor_correct
  TorchLean.Tensor.Internal.Lowering.einsumTensorKernel_correct
