/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import NN.IR.Pretty
public meta import NN.Widgets.IR.Graph
public meta import NN.Widgets.Core.UI
public meta import ProofWidgets.Component.HtmlDisplay
public meta import ProofWidgets.Demos.Macro

/-!
# GraphRewrite

Graph rewrite / diff viewer.

TorchLean’s IR is designed to support compilation and optimization passes. When working on those
passes, the most useful debugging UI is "before/after":
- render both graphs,
- show which nodes changed (op kind / parents / shapes),
- and make it obvious when a pass accidentally changes shapes or dependencies.

Main command:
- `#graph_rewrite_view before, after`

## Main definitions

- `diffRows`: align nodes by id for before/after comparison.
- `graphRewriteHtml`: side-by-side graph panels plus id-wise diff table.
- `#graph_rewrite_view`: command entry point.

## Implementation notes

- We compare by node id intentionally; for compiler/debug workflows this is usually the first
  question ("what changed at node i?").
- We surface op/parents/out-shape signatures because those capture most semantic rewrite mistakes.
- Keeping full "before" and "after" graph panels next to the diff table makes quick visual checks
  easier than reading only textual diffs.

## References

- [ProofWidgets](https://github.com/leanprover-community/ProofWidgets4)
- [GraphViz DOT language](https://graphviz.org/doc/info/lang.html)
- [Lean community documentation style](https://leanprover-community.github.io/contribute/doc.html)

## Tags

graph-rewrite, ir, compiler, diff, proofwidgets
-/

public meta section

open scoped ProofWidgets.Jsx

namespace NN.Widgets

open NN.IR
open UI

private structure DiffRow where
  id : Nat
  left? : Option Node
  right? : Option Node

/-- Build aligned per-id rows for two graphs, clipped to `maxNodes`. -/
private def diffRows (g₁ g₂ : Graph) (maxNodes : Nat := 400) : Array DiffRow :=
  let n := min (max g₁.size g₂.size) maxNodes
  (Array.range n).map (fun i =>
    { id := i, left? := g₁.nodes[i]?, right? := g₂.nodes[i]? })

/-- Produce a compact structural signature for change detection. -/
private def nodeSig (n : Node) : String :=
  s!"{n.kind.tag} parents={n.parents} out={Spec.Shape.pretty n.outShape}"

private def diffRowHtml (r : DiffRow) : ProofWidgets.Html :=
  let status :=
    match r.left?, r.right? with
    | none, none => warnBadge "missing"
    | some _, none => warnBadge "removed"
    | none, some _ => warnBadge "added"
    | some a, some b =>
        if a.kind.tag = b.kind.tag ∧ a.parents = b.parents ∧ a.outShape = b.outShape then
          okBadge "same"
        else
          warnBadge "changed"
  let leftS := match r.left? with | none => "<missing>" | some n => nodeSig n
  let rightS := match r.right? with | none => "<missing>" | some n => nodeSig n
  ;
  <tr>
    <td style={json% {"padding": "6px 8px", "border-bottom":
      "1px solid rgba(127,127,127,0.18)"}}>{monospace (toString r.id)}</td>
    <td style={json% {"padding": "6px 8px", "border-bottom":
      "1px solid rgba(127,127,127,0.18)"}}>{status}</td>
    <td style={json% {"padding": "6px 8px", "border-bottom":
      "1px solid rgba(127,127,127,0.18)"}}>{monospace leftS}</td>
    <td style={json% {"padding": "6px 8px", "border-bottom":
      "1px solid rgba(127,127,127,0.18)"}}>{monospace rightS}</td>
  </tr>

/-- Render a side-by-side graph diff panel plus a per-node "changed/same" table. -/
def graphRewriteHtml (g₁ g₂ : Graph) : ProofWidgets.Html :=
  let rows := diffRows g₁ g₂
  let sameCount :=
    rows.foldl (fun acc r =>
      match r.left?, r.right? with
      | some a, some b =>
          if a.kind.tag = b.kind.tag ∧ a.parents = b.parents ∧ a.outShape = b.outShape then acc + 1
            else acc
      | _, _ => acc) 0
  let changedCount := rows.size - sameCount
  ;
  <div style={json% {"display": "grid", "grid-template-columns": "1fr", "gap": "10px"}}>
    <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap"}}>
      {pill "Graph rewrite"} {pill s!"left={g₁.size} nodes"} {pill s!"right={g₂.size} nodes"}
      {pill s!"diffNodes={rows.size}"} {pill s!"changed≈{changedCount}"}
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
      <summary>{.text "Diff table (per node id)"}</summary>
      <div style={json% {"margin-top": "8px", "overflow": "auto", "max-height": "420px",
        "border": "1px solid var(--vscode-panel-border, #e5e5e5)", "border-radius": "10px"}}>
        <table style={json% {"border-collapse": "collapse", "width": "100%"}}>
          <thead>
            <tr>
              <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
                "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "id"}</th>
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

syntax (name := graphRewriteViewCmd) "#graph_rewrite_view " term ", " term : command

macro "#graph_rewrite_view " g1:term ", " g2:term : command =>
  Lean.TSyntax.mkInfoCanonical <$> `(#html (graphRewriteHtml $g1 $g2))

end NN.Widgets
