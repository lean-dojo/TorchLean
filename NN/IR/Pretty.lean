/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Graph

/-!
# IR Pretty Printing

Pretty-printing utilities for `NN.IR.Graph`.

We use these functions when:

- a compiler pass produces a malformed graph and we want a clear, compact dump,
- a verifier rejects a graph and we want to see the exact node/parent structure,
- we want a quick visualization of a graph in GraphViz.

This module is intentionally **not** a stable serialization format. The IR itself evolves (new ops,
new payload conventions, new invariants), and we want the freedom to change the pretty output
without treating it as part of the public API.

PyTorch analogy:
- `pretty` is like printing an FX graph as a node list,
- `toDot` is like emitting a visualization of the dataflow graph.

References:
- GraphViz DOT language: https://graphviz.org/doc/info/lang.html
- PyTorch FX (for the mental model of an op-tagged graph): https://pytorch.org/docs/stable/fx.html
-/

@[expose] public section


namespace NN.IR

open Spec

namespace Node

/-!
Render a parent id list in a compact form.

`parents` in TorchLean's IR are **data dependencies**: an edge `p -> n` means "node `n` consumes
the value produced by node `p`."

We keep this concise because it shows up in logs; for more context you typically want the whole
graph listing (or the `.dot` output).
-/
/-- One-line rendering of a node (useful for logs). -/
def prettyLine (n : Node) : String :=
  let parentsString (ps : List Nat) : String :=
    "[" ++ String.intercalate ", " (ps.map toString) ++ "]"
  s!"{n.id}: {n.kind.describe} parents={parentsString n.parents} out={Shape.pretty n.outShape}"

end Node

namespace Graph

/-! ## Plain text rendering -/

/--
Render the graph as a simple line-per-node listing.

This assumes the usual IR invariant that nodes are in id/topological order, but it does not enforce
it; if you need validation, run `Graph.checkWellFormed` / `Graph.checkShapes` first.
-/
def pretty (g : Graph) : String :=
  let joinLines (xs : List String) : String :=
    String.intercalate "\n" xs
  joinLines <| (g.nodes.toList.map (fun n => n.prettyLine))

/-! ## GraphViz `.dot` rendering -/

/--
Emit a GraphViz `.dot` representation.

Edges point from a node to its consumers (i.e. from parent to child), matching the “dataflow”
picture most ML frameworks use.

Usage:
- `IO.FS.writeFile "g.dot" (Graph.toDot g)`
- `dot -Tpng g.dot -o g.png`
-/
def toDot (g : Graph) : String :=
  let joinLines (xs : List String) : String :=
    String.intercalate "\n" xs
  let dotNodeLabel (n : Node) : String :=
    -- Keep labels short; shapes get long quickly.
    --
    -- In DOT, `\n` inside a quoted label produces a line break. We rely on that to keep each node
    -- readable without making the graph extremely wide.
    s!"{n.id}: {n.kind.describe}\\nout={Shape.pretty n.outShape}"
  let dotEscape (s : String) : String :=
    -- Dot-quoted string escaping.
    --
    -- We escape:
    -- - backslashes (so `\n` in labels stays meaningful),
    -- - quotes (since we emit `label=\"...\"`),
    -- - and newlines (DOT expects `\n` inside quoted labels).
    s.replace "\\" "\\\\" |>.replace "\"" "\\\"" |>.replace "\n" "\\n"
  let header := "digraph IR {\n  rankdir=LR;\n  node [shape=box, fontsize=10];\n"
  let nodes :=
    g.nodes.toList.map (fun n =>
      s!"  n{n.id} [label=\"{dotEscape (dotNodeLabel n)}\"];")
  let edges :=
    g.nodes.toList.foldl (init := []) (fun acc n =>
      acc ++ n.parents.map (fun p => s!"  n{p} -> n{n.id};"))
  let footer := "}\n"
  header ++ joinLines (nodes ++ edges) ++ "\n" ++ footer

end Graph

end NN.IR
