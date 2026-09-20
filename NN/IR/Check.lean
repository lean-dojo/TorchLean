/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Infer

/-!
# IR Validation

Validation helpers for `NN.IR.Graph`.

There are two validation levels:
- `Graph.checkWellFormed` lives in `NN.IR.Graph` and checks only graph structure: id discipline,
  parent arity, and topological ordering.
- `Graph.checkShapes` lives in `NN.IR.Infer` and checks declared `Node.outShape`s
  against the shared shape inference rules from `NN.IR.Infer`.

`Infer.nodeOutShape` states the per-operation shape rules, and `Graph.checkShapes` validates
graph nodes through those rules.
-/

@[expose] public section


namespace NN.IR

namespace Graph

/-!
## Prop-level well-formedness wrappers

The IR layer primarily exposes executable checkers (`checkWellFormed`, `checkShapes`) because
that is what compilers/backends/exporters need: either accept a graph or produce a readable error.

For proofs it is often nicer to assume a proposition instead of carrying around `Except String`:
`WellFormed g` means the checker succeeds, and similarly for `WellShaped g`.

These wrappers keep the executable checker names and proposition-level assumptions in the same
module.
-/

/-- Core structural well-formedness (ids, arity, topo order), as a proposition. -/
def WellFormed (g : Graph) : Prop :=
  g.checkWellFormed = .ok ()

/-- Shape-consistency well-formedness (`WellFormed` + extra shape/axis checks), as a proposition. -/
def WellShaped (g : Graph) : Prop :=
  g.checkShapes = .ok ()

end Graph

end NN.IR
