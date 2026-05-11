/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.Ops.Activations
public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.Ops.Constants
public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.Ops.Elementwise
public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.Ops.LinearAlgebra
public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.Ops.Loss
public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.Ops.Normalization
public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.Ops.Permutation
public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.Ops.Pooling
public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.Ops.Random
public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.Ops.Reductions
public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.Ops.Structural
public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.Ops.Unary

/-!
# Operator Correctness

Per-operator correctness lemmas for the IR → compiled runtime bridge.

This module is an index. Import it when you want the checked compiler-step lemmas without the
recursive semantic equivalence theorem.

The imported files are intentionally split by operator family. A single all-in-one proof file would
be hard to build and harder to review: Lean has to unfold compiler branches, normalize `Except`
control flow, compare dependent shapes, and prove that the compiled `GraphData` node appends the
same `DVal` as the IR evaluator. Keeping each family separate makes incremental builds and local
debugging much more pleasant.

The remaining proof engineering is to factor the repeated one-parent/two-parent boilerplate into
reusable helper lemmas and keep individual branches focused on their semantic equation.
-/

@[expose] public section
