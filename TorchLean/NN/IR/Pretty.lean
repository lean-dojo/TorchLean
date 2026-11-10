/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.Graph

/-!
# Printing graphs of registered operators

`Node.prettyLine id n` renders node `n` at id `id` on one line: a graph input with its index and
port, a constant with its port, or an operator by its wire tag with the references it reads and
the ports it produces. `Graph.toDot` emits GraphViz DOT: a box per node, an edge from a node to each
node reading one of its outputs (labelled with the output's position when it is not the first),
and the graph's outputs marked. Neither is a serialization format:
`TorchLean.NN.IR.Artifact` writes the `torchlean.ir` artifact.

References:
- GraphViz DOT language: https://graphviz.org/doc/info/lang.html
-/

@[expose] public section

namespace TorchLean

/-- A port as its shape, preceded by its element sort unless it holds values. -/
def Port.pretty : Port → String
  | ⟨.value, s⟩ => toString s
  | ⟨.frozen, s⟩ => s!"frozen {s}"
  | ⟨.detached, s⟩ => s!"detached {s}"
  | ⟨.index bounds, s⟩ => s!"index {bounds} {s}"
  | ⟨.bool, s⟩ => s!"bool {s}"
  | ⟨.integer signed width, s⟩ =>
      s!"{if signed then "int" else "uint"}{width} {s}"

end TorchLean

namespace TorchLean.IR

/-- The operation of a node: `input`, `const`, or its operator's wire tag. -/
def Node.tag : Node → String
  | .input .. => "input"
  | .const _ => "const"
  | .op o _ => (Op.encode o).tag

/-- A reference as its node id, followed by the output's position when it is not the first. -/
def Ref.pretty (r : Ref) : String :=
  if r.out = 0 then toString r.node else s!"{r.node}.{r.out}"

/-- The ports a node produces: the port of a node with one output, else the list of ports. -/
def Node.outsPretty (n : Node) : String :=
  match n.outs with
  | [p] => p.pretty
  | ps => "[" ++ ", ".intercalate (ps.map Port.pretty) ++ "]"

/-- Node `n` at id `id` on one line, for logs and diagnostics. -/
def Node.prettyLine (id : Nat) (n : Node) : String :=
  match n with
  | .input index _ => s!"{id}: input {index} out={n.outsPretty}"
  | .const _ => s!"{id}: const out={n.outsPretty}"
  | .op _ refs =>
      s!"{id}: {n.tag} inputs=[{", ".intercalate (refs.toList.map Ref.pretty)}] \
        out={n.outsPretty}"

/-- Escape a string for a double-quoted DOT label. -/
def dotEscape (s : String) : String :=
  s.replace "\\" "\\\\" |>.replace "\"" "\\\"" |>.replace "\n" "\\n"

/-- The default DOT label of node `n` at id `id`: its id and operation, then its output ports. -/
def Node.dotLabel (id : Nat) (n : Node) : String :=
  s!"{id}: {n.tag}\n{n.outsPretty}"

/-- GraphViz DOT of the graph: a box per node labelled `label id n`, with the extra attributes
`attrs id n` when they are not empty, an edge from a node to each node reading one of its outputs
(labelled with the output's position when it is not `0`), and a double border on the nodes the
graph outputs name. -/
def Graph.toDot (g : Graph) (label : Nat → Node → String := Node.dotLabel)
    (attrs : Nat → Node → String := fun _ _ => "") : String :=
  let header := "digraph IR {\n  rankdir=LR;\n  node [shape=box, fontsize=10];\n"
  let output (id : Nat) : Bool := g.outputs.any (·.node == id)
  let nodes := g.nodes.zipIdx.toList.map fun (n, id) =>
    let extra := attrs id n
    let style := if output id then ", peripheries=2" else ""
    s!"  n{id} [label=\"{dotEscape (label id n)}\"{style}\
      {if extra.isEmpty then "" else ", " ++ extra}];"
  let edges := g.nodes.zipIdx.toList.flatMap fun (n, id) =>
    match n with
    | .op _ refs => refs.toList.map fun r =>
        if r.out = 0 then s!"  n{r.node} -> n{id};"
        else s!"  n{r.node} -> n{id} [label=\"{r.out}\"];"
    | _ => []
  header ++ "\n".intercalate (nodes ++ edges) ++ "\n}\n"

end TorchLean.IR
