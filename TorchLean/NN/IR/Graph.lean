/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Codec
public import TorchLean.Tensor.Op.Run
public import TorchLean.CUDA.Kernel.Program

/-!
# Graphs of registered operators

A node of `TorchLean.IR.Graph` is a graph input at a port, a constant of the payload at a port, or a
registered operator (`TorchLean.Op`, with its ports forgotten in the type: `Operator`) with one
reference per input port. A reference (`Ref`) names a node and one of its outputs, so an operator
with several outputs (`sort`, `topk`) is one node, and index and Boolean values flow between nodes
at their ports. A node's id is its position in `nodes`. The payload (`Payload`) holds, by node id,
the constant of each constant node (a value, index or Boolean tensor at its port: parameters,
masks, index tensors) and the checked program of each `Op.custom` node. The ports a node produces
are derived from it (`Node.outs`), so there is no declared shape to check against the operator.
Equality of operators is decided through their wire records (`Op.instDecidableEqErased`), and
operators print as their records.

Evaluation is in `TorchLean.NN.IR.Eval` and typed programs (graph nodes, conversions and
registered calls with one scalar per step) in `TorchLean.NN.IR.Program`. Their JSON artifact,
the format `torchlean.ir`, is in `TorchLean.NN.IR.Artifact`.
-/

@[expose] public section

namespace TorchLean.IR

/-- A registered operator with its ports forgotten in the type. -/
abbrev Operator := Σ ins outs, Op ins outs

instance : Repr Operator where
  reprPrec op p := reprPrec (Op.encode op) p

/-- Output `out` of node `node`. -/
structure Ref where
  /-- The node, by its position. -/
  node : Nat
  /-- The output position. -/
  out : Nat := 0
  deriving DecidableEq, Repr, Inhabited

/-- A node: a graph input, a constant of the payload, or a registered operator with one reference
per input port, in port order. The payload holds the checked program of an `Op.custom` node. -/
inductive Node where
  /-- Graph input `index`, read at `port`. -/
  | input (index : Nat) (port : Port)
  /-- The payload's constant at the node's id, read at `port`. -/
  | const (port : Port)
  /-- Operator `op` on the outputs `inputs` name. -/
  | op (op : Operator) (inputs : Array Ref)
  deriving DecidableEq, Repr

/-- A graph: its nodes in evaluation order, a node's id being its position, and its outputs. -/
structure Graph where
  /-- The nodes. -/
  nodes : Array Node
  /-- The outputs. -/
  outputs : Array Ref
  deriving DecidableEq, Repr

/-- The constants and custom programs of a graph, by node id. -/
structure Payload (α : Type) [Storage α] where
  /-- The constant of a constant node. -/
  const? : Nat → Option (Op.Value α) := fun _ => none
  /-- The checked program of a custom node. -/
  custom? : Nat → Option (Kernel.Program α) := fun _ => none

/-- The ports a node produces. -/
def Node.outs : Node → List Port
  | .input _ p | .const p => [p]
  | .op ⟨_, outs, _⟩ _ => outs

end TorchLean.IR
