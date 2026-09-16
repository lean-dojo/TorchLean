/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import NN.IR.Pretty
public import Mathlib.Data.Finset.Attr
import Mathlib.Tactic.Finiteness.Attr
import Mathlib.Tactic.SetLike
public meta import NN.Widgets.Core.UI
public meta import ProofWidgets.Component.HtmlDisplay
public meta import NN.Spec.Core.Shape -- shake: keep

/-!
# Widgets IR

IR graph viewer widget (for debugging / teaching).

This module defines `#ir_view g`, which renders an `NN.IR.Graph` as an interactive HTML panel with:
- a well-formedness check result,
- a per-node expandable view,
- and a DOT snippet you can paste into GraphViz if needed.

Like other widgets, this is an inspection surface for examples and pass debugging. Proof scripts
should cite the underlying graph definitions and theorems directly.

## Main definitions

- `irHtml`: render checks, node details, and DOT text for one `NN.IR.Graph`.
- `#ir_view`: command frontend for interactive infoview inspection.
-/

public meta section

open scoped ProofWidgets.Jsx

namespace NN.Widgets

open NN.IR
open UI

/-- Report `Graph.checkWellFormed` as a badge, with the rejection message when it fails. -/
private def wfHtml (g : Graph) : ProofWidgets.Html :=
  match Graph.checkWellFormed g with
  | .ok () =>
      <div>{okBadge "ok"} {pill "Graph.checkWellFormed"}</div>
  | .error msg =>
      <div>
        {errBadge "error"} {pill "Graph.checkWellFormed"}
        <pre style={json% {"white-space": "pre-wrap", "margin-top": "6px"}}>{.text msg}</pre>
      </div>

/-- Render one expandable node entry with id/op/shape/parents metadata. -/
private def nodeDetails (n : Node) : ProofWidgets.Html :=
  <details style={json% {"margin": "6px 0"}}>
    <summary>{monospace (Node.prettyLine n)}</summary>
    <div style={json% {"margin-top": "6px", "padding-left": "8px"}}>
      <div>{pill s!"id={n.id}"} {pill s!"kind={n.kind.tag}"} {pill
        s!"out={Spec.Shape.pretty n.outShape}"}</div>
      <div style={json% {"margin-top": "6px"}}>
        <div><b>parents:</b> {monospace (reprStr n.parents)}</div>
      </div>
    </div>
  </details>

/-- Render an IR graph as a rich HTML panel. -/
def irHtml (g : Graph) (maxDotChars : Nat := 6000) : ProofWidgets.Html :=
  let nodes : Array ProofWidgets.Html := g.nodes.map nodeDetails;
  let dot := Graph.toDot g;
  let dotPreview :=
    if dot.length <= maxDotChars then
      dot
    else
      -- `String.take` returns a slice, so convert to a `String` before appending.
      (dot.take maxDotChars).toString ++ "\n... (clipped)";
  <div style={json% {
    "padding": "10px",
    "border": "1px solid var(--vscode-panel-border, #e5e5e5)",
    "border-radius": "10px",
    "background": "var(--vscode-editor-background, transparent)",
    "color": "var(--vscode-editor-foreground, inherit)"
  }}>
    <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap", "margin-bottom":
      "10px"}}>
      {pill s!"nodes={g.size}"} {pill "NN.IR.Graph"} {pill "viewer"}
    </div>
    {wfHtml g}
    <details «open»={true} style={json% {"margin-top": "10px"}}>
      <summary>{.text "Nodes"}</summary>
      <div style={json% {"margin-top": "6px"}}>
        {...nodes}
      </div>
    </details>
    <details style={json% {"margin-top": "10px"}}>
      <summary>{.text "GraphViz DOT (paste into `dot`)"}</summary>
      <pre style={json% {
        "white-space": "pre",
        "overflow-x": "auto",
        "margin-top": "6px",
        "padding": "8px",
        "border-radius": "8px",
        "border": "1px solid var(--vscode-panel-border, #e5e5e5)",
        "background": "var(--vscode-textCodeBlock-background, rgba(127,127,127,0.12))"
      }}>{.text dotPreview}</pre>
    </details>
  </div>

/-!
## Commands
-/

syntax (name := irViewCmd) "#ir_view " term : command

macro "#ir_view " g:term : command =>
  UI.canonicalCommand <$> `(#html (irHtml $g))

end NN.Widgets
