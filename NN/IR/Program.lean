/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Check
public import NN.IR.DType
public import FloatLib.Floats.Formats.BinaryInterchange.Rounding.Mode

/-!
# Typed interchange programs

`Program` attaches element types and an input/output interface to the existing
`Graph`. Shapes and dependencies remain in that graph; there is no second node
language. Ordinary operators have homogeneous signatures, so a frontend must
make its promotions explicit. Casts and registered heterogeneous primitives use
the existing custom-node shape contract and a separate element-type signature.

The custom node's name is for display. Its typed primitive entry, indexed by node
id, determines its meaning. An unregistered custom operation is rejected rather
than inferred from a spelling. Interpreters must supply the implementation of
every primitive they execute.

Type checking establishes consistency of the declared program, not equivalence
with a frontend or correctness of an external numerical backend.
-/

@[expose] public section

namespace NN.IR

open Spec

/-- Typed extension of a custom node; shape contracts remain in `OpKind.custom`. -/
inductive Primitive where
  /-- Convert one tensor to the node's result type, with an explicit rounding direction. -/
  | cast (rounding : FloatLib.Floats.Formats.BinaryInterchange.Model.IEEERoundingMode :=
      .nearestEven)
  /-- A registered operation with an explicit, possibly heterogeneous element-type signature. -/
  | call (name : Lean.Name) (inputs : Array DType) (output : DType)
  deriving DecidableEq, Repr

/-- One canonical graph together with its SSA type assignment and external interface. -/
structure Program where
  /-- Shared graph syntax, including all operation shape attributes. -/
  graph : Graph
  /-- Element type of every node result, indexed by node id. -/
  dtypes : Array DType
  /-- All runtime input nodes, in argument order. -/
  inputs : Array Nat
  /-- Designated result nodes, in return order. -/
  outputs : Array Nat
  /-- Typed bodies for custom nodes. Empty means no registrations; otherwise aligned with nodes. -/
  primitives : Array (Option Primitive) := #[]
  deriving Repr

namespace Program

/-- Recover a result's declared element type, reporting an invalid SSA index. -/
def dtype (program : Program) (id : Nat) : Except String DType :=
  match program.dtypes[id]? with
  | some dtype => .ok dtype
  | none => .error s!"IR types: node {id} has no declared element type"

/-- Typed primitive attached to a node. An empty table represents no registrations. -/
def primitive? (program : Program) (id : Nat) : Option Primitive :=
  (program.primitives[id]?).join

/-- Check one node's element-type rule; graph validation separately checks shape and arity. -/
def checkNodeTypes (program : Program) (node : Node) : Except String Unit := do
  let output ← program.dtype node.id
  output.check
  let inputs ← node.parents.mapM program.dtype
  match node.kind, program.primitive? node.id with
  | .custom _ shapes shape, some (.cast _) =>
      unless inputs.size == 1 do
        throw s!"IR types: cast node {node.id} needs exactly one operand"
      unless shapes = [shape] && shape = node.outShape do
        throw s!"IR types: cast node {node.id} must preserve its operand shape"
  | .custom _ _ _, some (.call name signature result) =>
      if name.isAnonymous then
        throw s!"IR types: node {node.id} has an anonymous primitive registration"
      unless inputs = signature do
        throw s!"IR types: node {node.id}: {name} expects {repr signature}, got {repr inputs}"
      unless output = result do
        throw s!"IR types: node {node.id}: {name} returns {result}, not {output}"
  | .custom name _ _, none =>
      throw s!"IR types: custom node {node.id} ({name}) has no registered type rule"
  | _, some _ =>
      throw s!"IR types: node {node.id}: a primitive registration requires a custom node"
  | _, none =>
      for input in inputs do
        unless input = output do
          throw <| s!"IR types: node {node.id} ({node.kind.tag}): " ++
            s!"operand {input} differs from result {output}; insert an explicit cast"

/-- Check ids, shapes, element types, primitive signatures, and the input/output interface. -/
def check (program : Program) : Except String Unit := do
  program.graph.checkShapes
  unless program.dtypes.size == program.graph.nodes.size do
    throw "IR types: the type assignment must contain one entry per node"
  unless program.primitives.isEmpty || program.primitives.size == program.graph.nodes.size do
    throw "IR types: primitive registrations must be empty or contain one entry per node"
  for node in program.graph.nodes do
    program.checkNodeTypes node
  let mut inputs := Array.replicate program.graph.nodes.size false
  for id in program.inputs do
    let node ← program.graph.getNode id
    unless node.kind = .input do
      throw s!"IR interface: argument {id} is not an input node"
    if inputs[id]?.getD false then
      throw s!"IR interface: repeated input node {id}"
    inputs := inputs.set! id true
  for node in program.graph.nodes do
    if node.kind = .input && !inputs[node.id]?.getD false then
      throw s!"IR interface: input node {node.id} is missing from the argument list"
  if program.outputs.isEmpty then
    throw "IR interface: a program must return at least one tensor"
  for id in program.outputs do
    let _ ← program.graph.getNode id

/-- Passing the typed checker supplies the shape hypothesis used by graph semantics. -/
theorem check_wellShaped {program : Program} (h : program.check = .ok ()) :
    program.graph.WellShaped := by
  cases hShape : program.graph.checkShapes with
  | error message => simp [check, Bind.bind, Except.bind, hShape] at h
  | ok result =>
      cases result
      exact hShape

/-- Embed a scalar-polymorphic graph without changing its operations or dependencies. -/
def ofGraph (graph : Graph) (outputs : Array Nat) : Program :=
  { graph := graph
    dtypes := Array.replicate graph.nodes.size .context
    inputs := (graph.nodes.filter (fun node => node.kind == .input)).map (·.id)
    outputs := outputs }

/-- Append a shape-preserving cast to a typed program and return its result id.

The target type is the result's entry in the typing context; the rounding direction
is the primitive's explicit attribute. Existing custom-node shape checks remain applicable.
-/
def cast (program : Program) (input : Nat) (target : DType)
    (rounding : FloatLib.Floats.Formats.BinaryInterchange.Model.IEEERoundingMode :=
      .nearestEven) : Except String (Program × Nat) := do
  unless program.dtypes.size == program.graph.nodes.size do
    throw "IR builder: incomplete type assignment"
  unless program.primitives.isEmpty || program.primitives.size == program.graph.nodes.size do
    throw "IR builder: incomplete primitive table"
  target.check
  let parent ← program.graph.getNode input
  let id := program.graph.nodes.size
  let registrations := if program.primitives.isEmpty then
    Array.replicate id none else program.primitives
  let node : Node :=
    { id := id, parents := #[input], kind := .custom "cast" [parent.outShape] parent.outShape,
      outShape := parent.outShape }
  let graph : Graph := { nodes := program.graph.nodes.push node }
  let result : Program :=
    { graph := graph
      dtypes := program.dtypes.push target
      inputs := program.inputs
      outputs := program.outputs
      primitives := registrations.push (some (.cast rounding)) }
  pure (result, id)

end Program

end NN.IR
