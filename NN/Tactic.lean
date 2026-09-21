/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tactic.Autograd
public import NN.Tactic.Converges
public import NN.Tactic.Einops
public import NN.Tactic.Except
public import NN.Tactic.Verify

/-!
# TorchLean tactics

* `autograd` proves derivative formulas and registered autograd certificates.
* `converges` proves supported convergence and rate bounds from their hypotheses.
* `einops` proves supported tensor transformation identities.
* `except_cases` extracts successful steps from checked computations.
* `verify` combines registered soundness theorems with kernel-checked evidence.

The `?` variants explain the proof or tensor transformation. Import individual tactic modules when
the other domains are not needed. `NN.Tactic.Verify.Lowering` separately loads graph-lowering
correctness rules. Differential tests live in `NN.Testing.Command`, not in this proof collection.
-/
