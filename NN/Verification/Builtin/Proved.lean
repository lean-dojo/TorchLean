/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Builtin.Proved.Syntax
public import NN.Verification.Builtin.Proved.Lowering
public import NN.Verification.Builtin.Proved.Correctness

/-!
# Verified Forward Fragment

A first-order TorchLean forward language, its lowering into the verifier IR, and the checked theorem
that evaluation of the lowered IR agrees with the source evaluator.
-/
