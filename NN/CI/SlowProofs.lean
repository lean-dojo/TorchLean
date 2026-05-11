/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Compiled.IRExec.Correctness

/-!
# Slow Proof CI Target

This module collects proof-heavy targets that we want CI to check regularly but
do not want to put on the normal development critical path.

The main target here is compiled IR execution correctness. Keeping it
as a named CI import makes the proof surface explicit without forcing every
developer build to elaborate the same targets.
-/

@[expose] public section
