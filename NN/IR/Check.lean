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
- `Graph.checkShapes` lives here and checks the declared `Node.outShape`s against the shared shape
  inference rules from `NN.IR.Infer`.

Shape logic has **one source of truth**: `Infer.inferNodeOutShape` states the per-op shape rules,
and `Graph.checkShapes` validates graph nodes through those rules.
-/

@[expose] public section


namespace NN.IR

namespace Graph

/--
Validate declared IR shapes using the central inference rules.

This wrapper keeps the public checker name aligned with `Graph.checkInferredShapes`, so callers,
widgets, and backends all use the same per-op shape contracts.
-/
def checkShapes (g : Graph) : Except String Unit := do
  g.checkInferredShapes

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

/-- `WellFormed g` is just the proposition that `checkWellFormed` returns `.ok ()`. -/
theorem wellFormed_iff (g : Graph) : g.WellFormed ↔ g.checkWellFormed = .ok () := Iff.rfl
/-- `WellShaped g` is just the proposition that `checkShapes` returns `.ok ()`. -/
theorem wellShaped_iff (g : Graph) : g.WellShaped ↔ g.checkShapes = .ok () := Iff.rfl

end Graph

end NN.IR
