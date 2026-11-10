/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.Graph

/-!
# Imported operator graphs

`Imported` is a graph of registered operators together with its payload and the reference of each
value of the source computation (`ids`). The typed graphs of `TorchLean.NN.IR.OpGraph` build one
(`TorchLean.IR.OpGraph.toOp`), and the lowering of a recorded model extends it
(`TorchLean.IR.LoweredModel`).
-/

@[expose] public section

namespace TorchLean.IR

/-- A graph of registered operators with its payload and the reference of each source value. -/
structure Imported (α : Type) [Storage α] where
  /-- The operator graph. -/
  graph : Graph
  /-- Its constants and custom programs. -/
  payload : Payload α
  /-- The reference of source value `j`. -/
  ids : Array Ref

end TorchLean.IR
