/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Correctness.Ops.Activations
public import NN.Runtime.Autograd.IRExec.Correctness.Ops.Concat
public import NN.Runtime.Autograd.IRExec.Correctness.Ops.Constants
public import NN.Runtime.Autograd.IRExec.Correctness.Ops.Convolution
public import NN.Runtime.Autograd.IRExec.Correctness.Ops.Elementwise
public import NN.Runtime.Autograd.IRExec.Correctness.Ops.LinearAlgebra
public import NN.Runtime.Autograd.IRExec.Correctness.Ops.Loss
public import NN.Runtime.Autograd.IRExec.Correctness.Ops.Normalization
public import NN.Runtime.Autograd.IRExec.Correctness.Ops.Permutation
public import NN.Runtime.Autograd.IRExec.Correctness.Ops.Pooling
public import NN.Runtime.Autograd.IRExec.Correctness.Ops.Random
public import NN.Runtime.Autograd.IRExec.Correctness.Ops.Reductions
public import NN.Runtime.Autograd.IRExec.Correctness.Ops.Structural
public import NN.Runtime.Autograd.IRExec.Correctness.Ops.Unary

/-!
# Operator Correctness

Per-operator correctness lemmas for the IR-to-forward-executor lowering.

This module is an index. Import it when you want the checked lowering-step lemmas without the
recursive semantic equivalence theorem.

The imported files follow the operator families used by the IR. Each proof has the same shape:
unfold the lowering branch, normalize `Except` control flow, compare dependent shapes, and show that
the lowered `ForwardNode` appends the same `Spec.SomeTensor` as the IR evaluator. Keeping these
families separate makes the proof obligations local and keeps incremental builds predictable.

The remaining proof engineering is to factor the repeated one-parent/two-parent boilerplate into
reusable helper lemmas and keep individual branches focused on their semantic equation.
-/

@[expose] public section
