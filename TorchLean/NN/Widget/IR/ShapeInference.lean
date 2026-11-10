/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import TorchLean.NN.IR.Eval
public meta import TorchLean.NN.IR.Pretty
public import TorchLean.NN.IR.Graph
public meta import TorchLean.NN.Widget.UI
public import TorchLean.NN.Widget.UI
import ProofWidgets.Component.HtmlDisplay
public import TorchLean.Tensor.Tactic.Init

/-!
# IR shape inference

`#shape_infer_view g` shows a table with one row per node of a graph of registered operators. Each
row gives the ports its operator expects, the ports its references name, the ports it produces and
the result of `Node.check`, which checks each node against the ports of the nodes before it. A
header lists the graph's outputs with their ports. Output ports follow from each node's operator
(`Node.outs`), so a shape error shows up as a reference whose port the operator's input does not
accept.
-/

public meta section

open scoped ProofWidgets.Jsx

namespace TorchLean.Widget

open IR
open UI

/-- One row of the shape table: what the node's operator expects against what its references
name. -/
private structure Row where
  /-- Node id. -/
  id : Nat
  /-- Operation, as `Node.tag` names it. -/
  op : String
  /-- The references the node reads. -/
  inputs : String
  /-- The input ports of the node's operator. -/
  expects : String
  /-- The ports of the outputs its references name. -/
  reads : String
  /-- The ports the node produces. -/
  produces : String
  /-- The message of `Node.check`, when the check fails. -/
  err? : Option String

/-- One row per node, each checked against the ports of the nodes before it. -/
private def rows (g : Graph) : Array Row :=
  g.nodes.zipIdx.map fun (n, i) =>
    let (inputs, expects, reads) := match n with
      | .op ⟨ins, _, _⟩ refs =>
          (toString (refs.toList.map Ref.pretty), toString (ins.map Port.pretty),
            toString (refs.toList.map fun r =>
              match g.portBefore i r with
              | some p => p.pretty
              | none => "?"))
      | _ => ("[]", "[]", "[]")
    { id := i, op := n.tag, inputs, expects, reads, produces := n.outsPretty
      err? := match n.check (g.portBefore i) i with
        | .ok () => none
        | .error msg => some msg }

/-- Render one table row, with an `ok` or `error` badge and the message of a failed check. -/
private def rowHtml (r : Row) : ProofWidgets.Html :=
  let status :=
    match r.err? with
    | none => okBadge "ok"
    | some _ => errBadge "error"
  ;
  <tr>
    <td style={json% {"padding": "6px 8px", "border-bottom": "1px solid rgba(127,127,127,0.18)"}}>
      {monospace (toString r.id)}
    </td>
    <td style={json% {"padding": "6px 8px", "border-bottom": "1px solid rgba(127,127,127,0.18)"}}>
      {monospace r.op}
    </td>
    <td style={json% {"padding": "6px 8px", "border-bottom": "1px solid rgba(127,127,127,0.18)"}}>
      {monospace r.inputs}
    </td>
    <td style={json% {"padding": "6px 8px", "border-bottom": "1px solid rgba(127,127,127,0.18)"}}>
      {monospace r.expects}
    </td>
    <td style={json% {"padding": "6px 8px", "border-bottom": "1px solid rgba(127,127,127,0.18)"}}>
      {monospace r.reads}
    </td>
    <td style={json% {"padding": "6px 8px", "border-bottom": "1px solid rgba(127,127,127,0.18)"}}>
      {monospace r.produces}
    </td>
    <td style={json% {"padding": "6px 8px", "border-bottom": "1px solid rgba(127,127,127,0.18)"}}>
      {status}
      {match r.err? with
        | none => ProofWidgets.Html.text ""
        | some msg =>
            <span style={json% {"margin-left": "8px", "opacity": 0.9}}>{monospace msg}</span>}
    </td>
  </tr>

/-- Render the `#shape_infer_view` panel: a header with the graph's outputs and their ports, then
the per-node table of the ports each operator expects, reads and produces. -/
def shapeInferHtml (g : Graph) : ProofWidgets.Html :=
  let outputs := g.outputs.toList.map fun r =>
    match g.portBefore g.nodes.size r with
    | some p => s!"{r.pretty}: {p.pretty}"
    | none => s!"{r.pretty}: ?"
  let outputsOk := g.outputs.all fun r => (g.portBefore g.nodes.size r).isSome
  ;
  <div style={json% {
    "padding": "10px",
    "border": "1px solid var(--vscode-panel-border, #e5e5e5)",
    "border-radius": "10px",
    "background": "var(--vscode-editor-background, transparent)",
    "color": "var(--vscode-editor-foreground, inherit)"
  }}>
    <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap", "margin-bottom":
      "10px"}}>
      {pill "IR shape inference"} {pill s!"nodes={g.nodes.size}"}
      {pill s!"outputs={outputs}"}
      {if outputsOk then okBadge "outputs ok" else errBadge "missing output"}
    </div>
    <div style={json% {"overflow": "auto", "max-height": "420px",
      "border": "1px solid var(--vscode-panel-border, #e5e5e5)", "border-radius": "10px"}}>
      <table style={json% {"border-collapse": "collapse", "width": "100%"}}>
        <thead>
          <tr>
            <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
              "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "id"}</th>
            <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
              "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "op"}</th>
            <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
              "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "inputs"}</th>
            <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
              "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "expects"}</th>
            <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
              "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "reads"}</th>
            <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
              "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "produces"}</th>
            <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
              "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "status"}</th>
          </tr>
        </thead>
        <tbody>
          {... (rows g).map rowHtml}
        </tbody>
      </table>
    </div>
  </div>

/-!
## Command
-/

/-- `#shape_infer_view g` shows the `shapeInferHtml` panel of the graph `g`. -/
macro "#shape_infer_view " g:term : command =>
  UI.canonicalCommand <$> `(#html (shapeInferHtml $g))

end TorchLean.Widget
