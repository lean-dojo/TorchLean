/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import NN.IR.Check
public meta import NN.IR.Pretty
public meta import NN.IR.Semantics
public meta import NN.Runtime.Context
public meta import NN.Widgets.Core.Tensor
public meta import NN.Widgets.Core.UI
public meta import ProofWidgets.Component.HtmlDisplay
public meta import ProofWidgets.Demos.Macro

/-!
# IRExecTrace

IR execution trace viewer (step-by-step evaluation).

TorchLean’s IR semantics (`NN.IR.Semantics`) can evaluate a graph and either:
- return the full table of intermediate values, or
- stop at the first failure (missing payload, shape mismatch, etc.).

When debugging, it is often more helpful to see:
- which nodes ran successfully,
- the intermediate tensor values (for small graphs),
- and the exact node id where evaluation stopped.

Main commands:
- `#ir_exec_trace_view g, input` where `input : Runtime.AnyTensor α` (payload defaults to empty)
- `#ir_exec_trace_view g, payload, input` where `payload : NN.IR.Payload α`

## Main definitions

- `execTrace`: execute nodes left-to-right and capture the first failure point.
- `irExecTraceHtml`: render checks, status badges, and per-node intermediate values.
- `#ir_exec_trace_view`: command entry point with optional payload.

## Implementation notes

- A stop-at-first-failure trace is the default debugging mode for IR execution.
- We intentionally display shape checks next to runtime trace data to reduce context switching.
- This viewer favors explicit node ids and parent links so failures are easy to localize.

## References

- [ProofWidgets](https://github.com/leanprover-community/ProofWidgets4)
- [GraphViz DOT language](https://graphviz.org/doc/info/lang.html)
- [Lean community documentation style](https://leanprover-community.github.io/contribute/doc.html)

## Tags

ir, execution-trace, debugging, semantics, proofwidgets
-/

public meta section

open scoped ProofWidgets.Jsx

namespace NN.Widgets

open _root_.Spec
open NN.IR
open Runtime
open UI

private def checkBadge (name : String) (r : Except String Unit) : ProofWidgets.Html :=
  match r with
  | .ok _ => <span>{okBadge name}</span>
  | .error msg => <span>{warnBadge name} <span style={json% {"margin-left": "6px"}}>{monospace
    msg}</span></span>

private structure Trace (α : Type) [Context α] where
  vals : Array (DVal α)
  failedAt? : Option (Nat × String)

/-- Execute a graph step-by-step, recording values until the first error. -/
private def execTrace
    {α : Type} [Context α] [Inhabited α] [DecidableEq Shape]
    (g : Graph) (payload : Payload α) (input : Runtime.AnyTensor α) : Trace α :=
  let inputD : DVal α := DVal.mk (α := α) input.s input.t
  let rec go (i : Nat) (vals : Array (DVal α)) : Trace α :=
    if i < g.nodes.size then
      match Graph.evalAt (α := α) (g := g) (payload := payload) (input := inputD) (vals := vals) (i
        := i) with
      | .ok v => go (i + 1) (vals.push v)
      | .error msg => { vals := vals, failedAt? := some (i, msg) }
    else
      { vals := vals, failedAt? := none }
  go 0 #[]

/-- Convert a semantic-domain value into a runtime shape-erased tensor wrapper. -/
private def dvToAny {α : Type} [Context α] (v : DVal α) : Runtime.AnyTensor α :=
  { s := v.1, t := v.2 }

private def nodeRowHtml {α : Type} [Context α] [ToString α]
    (g : Graph) (i : Nat) (v? : Option (DVal α)) : ProofWidgets.Html :=
  let n? := g.nodes[i]?
  let op := match n? with | none => "<missing>" | some n => n.kind.tag
  let parents := match n? with | none => [] | some n => n.parents
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
          let any := dvToAny (α := α) v
          ;
          <div style={json% {"margin-top": "8px", "padding-left": "10px"}}>
            {anyTensorHtml (α := α) any (maxRows := 10) (maxCols := 12) (maxElems := 64)}
          </div>}
  </details>

/-- Render an "execute and show intermediates" panel for an IR graph and a single input. -/
def irExecTraceHtml
    {α : Type} [Context α] [Inhabited α] [DecidableEq Shape] [ToString α]
    (g : Graph) (payload : Payload α) (input : Runtime.AnyTensor α) : ProofWidgets.Html :=
  let wf := g.checkWellFormed
  let sh := g.checkShapes
  let tr := execTrace (α := α) (g := g) (payload := payload) (input := input)
  let values : Array (Option (DVal α)) :=
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
        s!"inputShape={Shape.pretty input.s}"}
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
  Lean.TSyntax.mkInfoCanonical <$> `(#html (irExecTraceHtml $g {} $input))

macro "#ir_exec_trace_view " g:term ", " payload:term ", " input:term : command =>
  Lean.TSyntax.mkInfoCanonical <$> `(#html (irExecTraceHtml $g $payload $input))

end NN.Widgets
