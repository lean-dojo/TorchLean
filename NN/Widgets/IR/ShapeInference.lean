/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import NN.IR.Infer
public meta import NN.IR.Pretty
public meta import NN.Widgets.Core.UI
public meta import ProofWidgets.Component.HtmlDisplay
public meta import ProofWidgets.Demos.Macro

/-!
# ShapeInfer

Shape inference / checking viewer for `NN.IR.Graph`.

TorchLean’s IR carries a declared `outShape` at each node. In practice, there are two common
debugging questions when working with graphs:

- "Do the declared shapes match what the op semantics would infer from parent shapes?"
- "Where exactly did shape propagation fail, and what parent shapes caused it?"

This module provides a small infoview panel that answers those questions in a compact table.

Main command:
- `#shape_infer_view g`

## Main definitions

- `inferRows`: perform sequential shape inference and collect row diagnostics.
- `shapeInferHtml`: render declared vs inferred shapes in a status table.
- `#shape_infer_view`: command entry point.

## Implementation notes

- A per-node row model exposes inference and mismatch details directly.
- Inference stops after the first hard failure, and later rows are marked as "stopped"; this
  mirrors real debugging sessions where one upstream error explains many downstream unknowns.
- Status badges distinguish "mismatch" (inferred but different) from "error" (could not infer).

## References

- [ProofWidgets](https://github.com/leanprover-community/ProofWidgets4)
- [Lean community documentation style](https://leanprover-community.github.io/contribute/doc.html)

## Tags

shape-inference, ir, diagnostics, proofwidgets
-/

public meta section

open scoped ProofWidgets.Jsx

namespace NN.Widgets

open _root_.Spec
open NN.IR
open UI

private structure Row where
  id : Nat
  op : String
  parents : List Nat
  declared : Shape
  inferred? : Option Shape
  err? : Option String

/-- Infer node output shapes left-to-right and record mismatches/errors as rows. -/
private def inferRows (g : Graph) : Except String (Array Row) := do
  g.checkWellFormed
  let mut inferred : Array Shape := #[]
  let mut rows : Array Row := #[]
  let mut stopped : Bool := false
  for i in [0:g.nodes.size] do
    let n ← g.getNode i
    if stopped then
      rows := rows.push {
        id := i, op := n.kind.tag, parents := n.parents, declared := n.outShape
        inferred? := none, err? := some "stopped (prior inference error)"
      }
    else
      let parentShapes := n.parents.map (fun pid => inferred[pid]!)
      match Infer.inferNodeOutShape n parentShapes with
      | .ok out =>
          let err? :=
            if out = n.outShape then none
            else
              some (s!"outShape mismatch: inferred={Spec.Shape.pretty out}, " ++
                s!"declared={Spec.Shape.pretty n.outShape}")
          rows := rows.push {
            id := i, op := n.kind.tag, parents := n.parents, declared := n.outShape
            inferred? := some out, err? := err?
          }
          inferred := inferred.push out
      | .error msg =>
          rows := rows.push {
            id := i, op := n.kind.tag, parents := n.parents, declared := n.outShape
            inferred? := none, err? := some msg
          }
          stopped := true
  pure rows

/-- Render one inference diagnostic row. -/
private def rowHtml (r : Row) : ProofWidgets.Html :=
  let inferredS := match r.inferred? with | none => "?" | some s => Spec.Shape.pretty s
  let status :=
    match r.err? with
    | none => okBadge "ok"
    | some _ =>
        if r.inferred?.isSome then warnBadge "mismatch" else errBadge "error"
  ;
  <tr>
    <td style={json% {"padding": "6px 8px", "border-bottom": "1px solid rgba(127,127,127,0.18)"}}>
      {monospace (toString r.id)}
    </td>
    <td style={json% {"padding": "6px 8px", "border-bottom": "1px solid rgba(127,127,127,0.18)"}}>
      {monospace r.op}
    </td>
    <td style={json% {"padding": "6px 8px", "border-bottom": "1px solid rgba(127,127,127,0.18)"}}>
      {monospace (toString r.parents)}
    </td>
    <td style={json% {"padding": "6px 8px", "border-bottom": "1px solid rgba(127,127,127,0.18)"}}>
      {monospace (Spec.Shape.pretty r.declared)}
    </td>
    <td style={json% {"padding": "6px 8px", "border-bottom": "1px solid rgba(127,127,127,0.18)"}}>
      {monospace inferredS}
    </td>
    <td style={json% {"padding": "6px 8px", "border-bottom": "1px solid rgba(127,127,127,0.18)"}}>
      {status}
      {match r.err? with
        | none => ProofWidgets.Html.text ""
        | some msg =>
            <span style={json% {"margin-left": "8px", "opacity": 0.9}}>{monospace msg}</span>}
    </td>
  </tr>

/-- Render a per-node table comparing declared shapes vs inferred shapes for an IR graph. -/
def shapeInferHtml (g : Graph) : ProofWidgets.Html :=
  match inferRows g with
  | .error msg =>
      <div style={json% {"padding": "10px"}}>
        {errBadge "IR malformed"} <span style={json% {"margin-left": "8px"}}>{monospace msg}</span>
      </div>
  | .ok rows =>
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
                  "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "parents"}</th>
                <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
                  "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "declared"}</th>
                <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
                  "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "inferred"}</th>
                <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
                  "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "status"}</th>
              </tr>
            </thead>
            <tbody>
              {... rows.map rowHtml}
            </tbody>
          </table>
        </div>
      </div>

/-!
## Command
-/

syntax (name := shapeInferViewCmd) "#shape_infer_view " term : command

macro "#shape_infer_view " g:term : command =>
  Lean.TSyntax.mkInfoCanonical <$> `(#html (shapeInferHtml $g))

end NN.Widgets
