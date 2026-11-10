/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import TorchLean.NN.IR.Pretty
public meta import TorchLean.NN.IR.Eval
public import TorchLean.NN.IR.Graph
public import Mathlib.Data.Finset.Attr
import Mathlib.Tactic.Finiteness.Attr
import Mathlib.Tactic.SetLike
public meta import TorchLean.NN.Widget.UI
public import TorchLean.NN.Widget.UI
import ProofWidgets.Component.HtmlDisplay
public import TorchLean.Tensor.Tactic.Init

/-!
# IR graph viewer

`#ir_view g` shows a graph of registered operators (`TorchLean.IR.Graph`): the result of the
structural check `Graph.check`, an expandable entry for every node, the graph's outputs and its
GraphViz DOT text from `Graph.toDot`. Only the DOT text is clipped, after `maxDotChars` characters
(6000 by default).
-/

public meta section

open scoped ProofWidgets.Jsx

namespace TorchLean.Widget

open IR
open UI

/-- Report `Graph.check` as a badge, with the rejection message when it fails. -/
private def checkHtml (g : Graph) : ProofWidgets.Html :=
  match g.check with
  | .ok () =>
      <div>{okBadge "ok"} {pill "Graph.check"}</div>
  | .error msg =>
      <div>
        {errBadge "error"} {pill "Graph.check"}
        <pre style={json% {"white-space": "pre-wrap", "margin-top": "6px"}}>{.text msg}</pre>
      </div>

/-- References as a bracketed list. -/
private def refsText (refs : Array Ref) : String :=
  "[" ++ ", ".intercalate (refs.toList.map Ref.pretty) ++ "]"

/-- One expandable node entry: its line, id, operation and output ports, then the references and
wire record of an operator, the index of a graph input or a note for a payload constant. -/
private def nodeDetails (id : Nat) (n : Node) : ProofWidgets.Html :=
  <details style={json% {"margin": "6px 0"}}>
    <summary>{monospace (n.prettyLine id)}</summary>
    <div style={json% {"margin-top": "6px", "padding-left": "8px"}}>
      <div>{pill s!"id={id}"} {pill s!"op={n.tag}"} {pill s!"out={n.outsPretty}"}</div>
      {match n with
        | .op o refs =>
            <div style={json% {"margin-top": "6px"}}>
              <div><b>inputs:</b> {monospace (refsText refs)}</div>
              <div><b>record:</b> {monospace (reprStr o)}</div>
            </div>
        | .input index _ =>
            <div style={json% {"margin-top": "6px"}}><b>graph input:</b> {monospace
              (toString index)}</div>
        | .const _ =>
            <div style={json% {"margin-top": "6px"}}>{.text "the payload's constant"}</div>}
    </div>
  </details>

/-- The `#ir_view` panel: a header with the node count and the output references, the
`Graph.check` result, one entry per node and the DOT text, clipped after `maxDotChars`
characters. -/
def irHtml (g : Graph) (maxDotChars : Nat := 6000) : ProofWidgets.Html :=
  let nodes : Array ProofWidgets.Html := g.nodes.zipIdx.map fun (n, id) => nodeDetails id n;
  let dot := g.toDot;
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
      {pill s!"nodes={g.nodes.size}"} {pill s!"outputs={refsText g.outputs}"}
      {pill "TorchLean.NN.IR.Graph"} {pill "viewer"}
    </div>
    {checkHtml g}
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

/-- `#ir_view g` shows the `irHtml` panel of the graph `g`. -/
macro "#ir_view " g:term : command =>
  UI.canonicalCommand <$> `(#html (irHtml $g))

end TorchLean.Widget
