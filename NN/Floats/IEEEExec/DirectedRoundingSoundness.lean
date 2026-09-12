/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.DirectedRoundingSoundness.Division
public import NN.Floats.IEEEExec.DirectedRoundingSoundness.Positive
public import NN.Floats.IEEEExec.DirectedRoundingSoundness.SignedOps

/-!
Directed-rounding soundness entry point.

The submodules prove that executable lower/upper rounding operations enclose the corresponding
real operations, which is the boundary required by interval and LiRPA proofs.
-/
