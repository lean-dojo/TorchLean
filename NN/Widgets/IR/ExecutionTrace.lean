/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import NN.IR.Semantics
public meta import NN.Widgets.Core.Tensor
public meta import NN.IR.Check -- shake: keep
public meta import NN.IR.Pretty -- shake: keep
public import NN.Widgets.Core.Tensor
import ProofWidgets.Component.HtmlDisplay
public import NN.IR.Semantics

/-!
# IR execution trace

`#ir_exec_trace_view g, input` evaluates nodes in order and displays intermediate values until
the first evaluation error. The input is `Spec.SomeTensor α`; an explicit `NN.IR.Payload α`
can be supplied with `#ir_exec_trace_view g, payload, input` instead of the empty default.

`irExecTraceHtml` reports well-formedness and shape checks separately from evaluation. A failed
check does not prevent the panel from attempting a trace. Displaying a result does not prove it.
-/

public meta section

open scoped ProofWidgets.Jsx

namespace NN.Widgets

open _root_.Spec _root_.TorchLean
open NN.IR
open Runtime
open UI

/-- Turn an `Except` check into a badge, showing the message when the check failed. -/
private def checkBadge (name : String) (r : Except String Unit) : ProofWidgets.Html :=
  match r with
  | .ok _ => <span>{okBadge name}</span>
  | .error msg => <span>{warnBadge name} <span style={json% {"margin-left": "6px"}}>{monospace
    msg}</span></span>

/-- Result of a step-by-step graph execution: the values produced, and where it stopped. -/
private structure Trace (α : Type) [TorchLean.Storage α] [Context α] where
  /-- Values produced by the nodes that ran, in node order. -/
  vals : Array (Spec.SomeTensor α)
  /-- Node index and message of the first failure, `none` when the whole graph ran. -/
  failedAt? : Option (Nat × String)

/-- Execute a graph step-by-step, recording values until the first error. -/
private def execTrace
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : Graph) (payload : Payload α) (input : Spec.SomeTensor α) : Trace α :=
  let rec go (i : Nat) (vals : Array (Spec.SomeTensor α)) : Trace α :=
    if i < g.nodes.size then
      match Graph.evalAt (α := α) (g := g) (payload := payload) (input := input) (vals := vals) (i
        := i) with
      | .ok v => go (i + 1) (vals.push v)
      | .error msg => { vals := vals, failedAt? := some (i, msg) }
    else
      { vals := vals, failedAt? := none }
  go 0 #[]

/-- One table row per IR node: op tag, parents, declared shape, and the value it produced. -/
private def nodeRowHtml {α : Type} [TorchLean.Storage α] [Context α] [ToString α]
    (g : Graph) (i : Nat) (v? : Option (Spec.SomeTensor α)) : ProofWidgets.Html :=
  let n? := g.nodes[i]?
  let op := match n? with | none => "<missing>" | some n => n.kind.tag
  let parents := match n? with | none => #[] | some n => n.parents
  let declared := match n? with | none => "?" | some n => Shape.pretty n.outShape
  let status : ProofWidgets.Html :=
    match v? with
    | none => warnBadge "not executed"
    | some v =>
        match n? with
        | none => okBadge "ok"
        | some n =>
            if v.1 = n.outShape then okBadge "ok" else warnBadge "shape mismatch"
  ;
  <details style={json% {"margin": "6px 0"}}>
    <summary>
      {monospace s!"{i}: {op}"} {pill s!"parents={parents}"} {pill s!"declared={declared}"} {status}
    </summary>
    {match v? with
      | none => ProofWidgets.Html.text ""
      | some v =>
          <div style={json% {"margin-top": "8px", "padding-left": "10px"}}>
            {packedTensorHtml (α := α) v (maxRows := 10) (maxCols := 12) (maxElems := 64)}
          </div>}
  </details>

/-- Render an "execute and show intermediates" panel for an IR graph and a single input. -/
def irExecTraceHtml
    {α : Type} [TorchLean.Storage α] [Context α] [ToString α]
    (g : Graph) (payload : Payload α) (input : Spec.SomeTensor α) : ProofWidgets.Html :=
  let wf := g.checkWellFormed
  let sh := g.checkShapes
  let tr := execTrace (α := α) (g := g) (payload := payload) (input := input)
  let values : Array (Option (Spec.SomeTensor α)) :=
    (Array.range g.nodes.size).map (fun i => tr.vals[i]?)
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
      {pill "IR exec trace"} {pill s!"nodes={g.nodes.size}"} {pill
        s!"inputShape={Shape.pretty input.shape}"}
    </div>
    <div style={json% {"display": "grid", "grid-template-columns": "1fr", "gap": "6px",
      "margin-bottom": "10px"}}>
      <div>{checkBadge "checkWellFormed" wf}</div>
      <div>{checkBadge "checkShapes" sh}</div>
      {match tr.failedAt? with
        | none => <div>{okBadge "eval ok"} {pill s!"computed={tr.vals.size}"}</div>
        | some (i, msg) =>
            <div>
              {errBadge s!"eval failed at node {i}"} <span style={json% {"margin-left":
                "8px"}}>{monospace msg}</span>
              <span style={json% {"margin-left": "8px"}}>{pill s!"computed={tr.vals.size}"}</span>
            </div>}
    </div>
    <details «open»={true}>
      <summary>{.text "Trace (expand nodes)"}</summary>
      <div style={json% {"margin-top": "8px"}}>
        {... (Array.range g.nodes.size).map (fun i => nodeRowHtml (α := α) (g := g) i values[i]! )}
      </div>
    </details>
  </div>

/-!
## Commands
-/

syntax (name := irExecTraceViewCmd1) "#ir_exec_trace_view " term ", " term : command
syntax (name := irExecTraceViewCmd2) "#ir_exec_trace_view " term ", " term ", " term : command

macro "#ir_exec_trace_view " g:term ", " input:term : command =>
  UI.canonicalCommand <$> `(#html (irExecTraceHtml $g {} $input))

macro "#ir_exec_trace_view " g:term ", " payload:term ", " input:term : command =>
  UI.canonicalCommand <$> `(#html (irExecTraceHtml $g $payload $input))

end NN.Widgets
