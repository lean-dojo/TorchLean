/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.Graph
public meta import TorchLean.NN.Widget.IR.Graph
public import TorchLean.NN.Widget.IR.Graph
import ProofWidgets.Component.HtmlDisplay

/-!
# IR rewrite viewer

`#graph_rewrite_view before, after` shows two graphs of registered operators side by side and
compares their nodes position by position, along with their outputs. A row reads `same` when the
two nodes are equal as data: the same operator with the same attributes, the same references and
the same ports. The diff table and the change count cover at most 400 positions.
-/

public meta section

open scoped ProofWidgets.Jsx

namespace TorchLean.Widget

open IR
open UI

/-- One aligned row of a two-graph diff, holding whichever side has a node at this position. -/
private structure DiffRow where
  /-- Array position, the node's id. -/
  index : Nat
  /-- Node at this position in the left graph, if it has one. -/
  left? : Option Node
  /-- Node at this position in the right graph, if it has one. -/
  right? : Option Node

/-- One row per position of the longer graph, at most `maxNodes` of them. -/
private def diffRows (g₁ g₂ : Graph) (maxNodes : Nat := 400) : Array DiffRow :=
  let n := min (max g₁.nodes.size g₂.nodes.size) maxNodes
  (Array.range n).map (fun i =>
    { index := i, left? := g₁.nodes[i]?, right? := g₂.nodes[i]? })

/-- Text of a node in the diff table: its pretty-printed line and, for an operator, the wire record
with every attribute. -/
private def nodeSig (id : Nat) (n : Node) : String :=
  match n with
  | .op o _ => s!"{n.prettyLine id} {reprStr o}"
  | _ => n.prettyLine id

/-- Render one diff row, badged `same`, `changed`, `added`, or `removed`. -/
private def diffRowHtml (r : DiffRow) : ProofWidgets.Html :=
  let status :=
    match r.left?, r.right? with
    | none, none => warnBadge "missing"
    | some _, none => warnBadge "removed"
    | none, some _ => warnBadge "added"
    | some a, some b => if a = b then okBadge "same" else warnBadge "changed"
  let leftS := match r.left? with | none => "<missing>" | some n => nodeSig r.index n
  let rightS := match r.right? with | none => "<missing>" | some n => nodeSig r.index n
  ;
  <tr>
    <td style={json% {"padding": "6px 8px", "border-bottom":
      "1px solid rgba(127,127,127,0.18)"}}>{monospace (toString r.index)}</td>
    <td style={json% {"padding": "6px 8px", "border-bottom":
      "1px solid rgba(127,127,127,0.18)"}}>{status}</td>
    <td style={json% {"padding": "6px 8px", "border-bottom":
      "1px solid rgba(127,127,127,0.18)"}}>{monospace leftS}</td>
    <td style={json% {"padding": "6px 8px", "border-bottom":
      "1px solid rgba(127,127,127,0.18)"}}>{monospace rightS}</td>
  </tr>

/-- The `#graph_rewrite_view` panel: node counts, the number of changed rows and an outputs badge,
the two graphs side by side in `irHtml`, and a collapsed diff table with one row per position. -/
def graphRewriteHtml (g₁ g₂ : Graph) : ProofWidgets.Html :=
  let rows := diffRows g₁ g₂
  let sameCount :=
    rows.foldl (fun acc r =>
      match r.left?, r.right? with
      | some a, some b => if a = b then acc + 1 else acc
      | _, _ => acc) 0
  let changedCount := rows.size - sameCount
  ;
  <div style={json% {"display": "grid", "grid-template-columns": "1fr", "gap": "10px"}}>
    <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap"}}>
      {pill "Graph rewrite"} {pill s!"left={g₁.nodes.size} nodes"}
      {pill s!"right={g₂.nodes.size} nodes"}
      {pill s!"diffNodes={rows.size}"} {pill s!"changed in shown rows={changedCount}"}
      {if g₁.outputs = g₂.outputs then okBadge "outputs same" else warnBadge "outputs changed"}
    </div>
    <div style={json% {"display": "grid", "grid-template-columns": "1fr 1fr", "gap": "10px"}}>
      <div>
        <div style={json% {"margin-bottom": "6px"}}>{pill "before"}</div>
        {irHtml g₁}
      </div>
      <div>
        <div style={json% {"margin-bottom": "6px"}}>{pill "after"}</div>
        {irHtml g₂}
      </div>
    </div>
    <details «open»={false}>
      <summary>{.text "Diff table (per array position)"}</summary>
      <div style={json% {"margin-top": "8px", "overflow": "auto", "max-height": "420px",
        "border": "1px solid var(--vscode-panel-border, #e5e5e5)", "border-radius": "10px"}}>
        <table style={json% {"border-collapse": "collapse", "width": "100%"}}>
          <thead>
            <tr>
              <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
                "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "position"}</th>
              <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
                "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "status"}</th>
              <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
                "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "before"}</th>
              <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
                "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "after"}</th>
            </tr>
          </thead>
          <tbody>
            {... rows.map diffRowHtml}
          </tbody>
        </table>
      </div>
    </details>
  </div>

/-!
## Command
-/

/-- `#graph_rewrite_view g1, g2` shows the `graphRewriteHtml` comparison of the graph `g1` with
its rewrite `g2`. -/
macro "#graph_rewrite_view " g1:term ", " g2:term : command =>
  UI.canonicalCommand <$> `(#html (graphRewriteHtml $g1 $g2))

end TorchLean.Widget
