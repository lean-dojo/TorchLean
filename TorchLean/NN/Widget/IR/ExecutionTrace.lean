/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import TorchLean.NN.IR.Eval
public meta import TorchLean.NN.IR.Pretty
public meta import TorchLean.NN.Widget.Tensor
public import TorchLean.NN.Widget.Tensor
import ProofWidgets.Component.HtmlDisplay
public import TorchLean.NN.IR.Eval

/-!
# IR execution trace

`#ir_exec_trace_view g, inputs` evaluates the nodes of a graph of registered operators in order
with `Node.eval` and shows the outputs of every node up to the first evaluation error. The inputs
form a `List (TorchLean.Op.Value α)`. The two-argument form passes the empty payload, so a graph
with constant or custom nodes needs `#ir_exec_trace_view g, payload, inputs` with an
`TorchLean.IR.Payload α` that holds them.

The panel reports the structural check `Graph.check` on its own line and runs the trace whether or
not the check passes.
-/

public meta section

open scoped ProofWidgets.Jsx

namespace TorchLean.Widget

open IR
open UI

/-- Turn an `Except` check into a badge, showing the message when the check failed. -/
private def checkBadge (name : String) (r : Except String Unit) : ProofWidgets.Html :=
  match r with
  | .ok _ => <span>{okBadge name}</span>
  | .error msg => <span>{warnBadge name} <span style={json% {"margin-left": "6px"}}>{monospace
    msg}</span></span>

/-- Result of a step-by-step graph execution: the outputs produced, and where it stopped. -/
private structure Trace (α : Type) [Storage α] where
  /-- Outputs of the nodes that ran, in node order. -/
  outs : Array (List (Op.Value α))
  /-- Node index and message of the first failure, `none` when the whole graph ran. -/
  failedAt? : Option (Nat × String)

/-- Execute a graph node by node, recording outputs until the first error. -/
private def execTrace
    {α : Type} [Storage α] [Context α] [Op.Capabilities α]
    (g : Graph) (payload : Payload α) (inputs : List (Op.Value α)) : Trace α :=
  let rec go (i : Nat) (outs : Array (List (Op.Value α))) : Trace α :=
    if h : i < g.nodes.size then
      match g.nodes[i].eval payload inputs outs i with
      | .ok vs => go (i + 1) (outs.push vs)
      | .error msg => { outs, failedAt? := some (i, msg) }
    else
      { outs, failedAt? := none }
  go 0 #[]

/-- Render one output as a tensor preview when its port carries scalars (a value, frozen or
detached port) and as a badge naming the port otherwise. -/
private def valueHtml {α : Type} [Storage α] [ToString α] :
    Op.Value α → ProofWidgets.Html
  | ⟨⟨.value, s⟩, x⟩ =>
      packedTensorHtml (α := α) ⟨s, x⟩ (maxRows := 10) (maxCols := 12) (maxElems := 64)
  | ⟨⟨.frozen, s⟩, x⟩ =>
      packedTensorHtml (α := α) ⟨s, x⟩ (maxRows := 10) (maxCols := 12) (maxElems := 64)
  | ⟨⟨.detached, s⟩, x⟩ =>
      packedTensorHtml (α := α) ⟨s, x⟩ (maxRows := 10) (maxCols := 12) (maxElems := 64)
  | ⟨p, _⟩ => pill s!"value at {p.pretty}"

/-- The expandable row of node `i`: its operation, references and output ports, a status badge
(`ok`, `port mismatch` or `not executed`) and the outputs it produced. -/
private def nodeRowHtml {α : Type} [Storage α] [ToString α]
    (g : Graph) (i : Nat) (outs? : Option (List (Op.Value α))) : ProofWidgets.Html :=
  let n? := g.nodes[i]?
  let op := match n? with | none => "<missing>" | some n => n.tag
  let inputs := match n? with
    | some (.op _ refs) => "[" ++ ", ".intercalate (refs.toList.map Ref.pretty) ++ "]"
    | _ => "[]"
  let declared := match n? with | none => "?" | some n => n.outsPretty
  let status : ProofWidgets.Html :=
    match outs? with
    | none => warnBadge "not executed"
    | some vs =>
        match n? with
        | none => okBadge "ok"
        | some n => if vs.map (·.1) = n.outs then okBadge "ok" else warnBadge "port mismatch"
  ;
  <details style={json% {"margin": "6px 0"}}>
    <summary>
      {monospace s!"{i}: {op}"} {pill s!"inputs={inputs}"} {pill s!"out={declared}"} {status}
    </summary>
    {match outs? with
      | none => ProofWidgets.Html.text ""
      | some vs =>
          <div style={json% {"margin-top": "8px", "padding-left": "10px"}}>
            {... vs.toArray.map (valueHtml (α := α))}
          </div>}
  </details>

/-- Run `g` on `inputs` node by node and render the trace: the node count and input ports, the
`Graph.check` badge, the evaluation status and one expandable row per node. -/
def irExecTraceHtml
    {α : Type} [Storage α] [Context α] [Op.Capabilities α] [ToString α]
    (g : Graph) (payload : Payload α) (inputs : List (Op.Value α)) :
    ProofWidgets.Html :=
  let checked := g.check
  let tr := execTrace (α := α) g payload inputs
  let outs : Array (Option (List (Op.Value α))) :=
    (Array.range g.nodes.size).map (fun i => tr.outs[i]?)
  let inputPorts := "[" ++ ", ".intercalate (inputs.map fun v => v.1.pretty) ++ "]"
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
      {pill "IR exec trace"} {pill s!"nodes={g.nodes.size}"} {pill s!"inputs={inputPorts}"}
    </div>
    <div style={json% {"display": "grid", "grid-template-columns": "1fr", "gap": "6px",
      "margin-bottom": "10px"}}>
      <div>{checkBadge "Graph.check" checked}</div>
      {match tr.failedAt? with
        | none => <div>{okBadge "eval ok"} {pill s!"computed={tr.outs.size}"}</div>
        | some (i, msg) =>
            <div>
              {errBadge s!"eval failed at node {i}"} <span style={json% {"margin-left":
                "8px"}}>{monospace msg}</span>
              <span style={json% {"margin-left": "8px"}}>{pill s!"computed={tr.outs.size}"}</span>
            </div>}
    </div>
    <details «open»={true}>
      <summary>{.text "Trace (expand nodes)"}</summary>
      <div style={json% {"margin-top": "8px"}}>
        {... (Array.range g.nodes.size).map (fun i => nodeRowHtml (α := α) g i outs[i]!)}
      </div>
    </details>
  </div>

/-!
## Commands
-/

/-- `#ir_exec_trace_view g, inputs` shows the `irExecTraceHtml` trace of `g` on `inputs` with the
empty payload. -/
macro "#ir_exec_trace_view " g:term ", " inputs:term : command =>
  UI.canonicalCommand <$> `(#html (irExecTraceHtml $g {} $inputs))

/-- `#ir_exec_trace_view g, payload, inputs` shows the `irExecTraceHtml` trace of `g` on `inputs`
with `payload` holding the data of constant and custom nodes. -/
macro "#ir_exec_trace_view " g:term ", " payload:term ", " inputs:term : command =>
  UI.canonicalCommand <$> `(#html (irExecTraceHtml $g $payload $inputs))

end TorchLean.Widget
