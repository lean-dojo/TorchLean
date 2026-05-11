/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import NN.Runtime.Autograd.Engine.Core
public meta import NN.Widgets.Core.Tensor
public meta import NN.Widgets.Core.UI
public meta import ProofWidgets.Component.HtmlDisplay
public meta import ProofWidgets.Demos.Macro

/-!
# Autograd

Autograd widgets (tapes + gradients).

This module provides infoview panels for TorchLean’s eager-mode autograd tape:

- `#tape_view t` renders the tape nodes, showing parents, stored forward values, and a DOT snippet.
- `#tape_grads_view t, outId` runs reverse-mode from a scalar output (`outId`) and shows which node
  ids received gradients, plus small previews.

## Main definitions

- `tapeHtml`: inspect a tape's nodes and edge structure.
- `tapeGradsHtml`: run scalar reverse-mode and show gradient coverage/results.
- `tapeTraceHtml`: step-by-step reverse-pass trace with per-parent VJP contributions.
- `#tape_view`, `#tape_grads_view`, `#tape_trace_view`: command frontends.

## Implementation notes

- These widgets are intended for debugging and teaching, not proof scripts.
- DOT output is clipped for large tapes to keep infoview rendering responsive.
- Reverse traces use `seed = 1` at the output node, matching `loss.backward()` intuition.

## References

- [Backpropagation and reverse-mode AD](https://en.wikipedia.org/wiki/Automatic_differentiation)
- [GraphViz DOT language](https://graphviz.org/doc/info/lang.html)
- [ProofWidgets](https://github.com/leanprover-community/ProofWidgets4)
- [Lean community documentation style](https://leanprover-community.github.io/contribute/doc.html)

## Tags

autograd, reverse-mode, gradients, tape, widgets
-/

public meta section

open scoped ProofWidgets.Jsx

namespace NN.Widgets

open _root_.Spec
open Runtime
open Runtime.Autograd
open UI

/-- Pretty-print a tensor shape for badge/table display. -/
private def dimsString (s : Shape) : String :=
  Spec.Shape.pretty s

/-- Build an uncolored DOT view of the tape graph. -/
private def tapeDot {α : Type} (t : Tape α) : String :=
  let header := "digraph Tape {\n  rankdir=LR;\n  node [shape=box,fontname=\"monospace\"];\n"
  let pairs : List (Nat × Node α) := List.zip (List.range t.nodes.size) t.nodes.toList
  let nodes : List String :=
    pairs.map (fun (i, n) =>
      let name := n.name.getD ""
      let label := if name = "" then s!"{i}" else s!"{i}: {name}"
      s!"  n{i} [label=\"{label}\"];")
  let edges : List String :=
    pairs.foldl (fun acc (i, n) =>
      acc ++ (n.parents.map (fun p => s!"  n{p} -> n{i};"))) []
  header ++ String.intercalate "\n" (nodes ++ edges) ++ "\n}\n"

/-- Build a colored DOT view where output/gradient coverage is highlighted. -/
private def tapeDotColored {α : Type} (t : Tape α) (outId : Nat)
    (hasGrad : Nat → Bool) : String :=
  let header :=
    "digraph Tape {\n" ++
    "  rankdir=LR;\n" ++
    "  node [shape=box,fontname=\"monospace\",style=filled,fillcolor=\"#f7f7f7\"];\n"
  let pairs : List (Nat × Node α) := List.zip (List.range t.nodes.size) t.nodes.toList
  let nodes : List String :=
    pairs.map (fun (i, n) =>
      let name := n.name.getD ""
      let labelBase := if name = "" then s!"{i}" else s!"{i}: {name}"
      let label := escapeDotLabel labelBase
      let fill :=
        if i = outId then
          "#ffdddd"
        else if hasGrad i then
          "#ddffea"
        else if n.requires_grad then
          "#fff3cc"
        else
          "#f7f7f7"
      s!"  n{i} [label=\"{label}\", fillcolor=\"{fill}\"];")
  let edges : List String :=
    pairs.foldl (fun acc (i, n) =>
      acc ++ (n.parents.map (fun p => s!"  n{p} -> n{i};"))) []
  header ++ String.intercalate "\n" (nodes ++ edges) ++ "\n}\n"

/-- Render one tape node with metadata and forward tensor preview. -/
private def nodeHtml {α : Type} [ToString α] (id : Nat) (n : Node α) : ProofWidgets.Html :=
  <details style={json% {"margin": "6px 0"}}>
    <summary>
      {monospace (toString id ++ ": " ++ n.name.getD "(unnamed)")} {pill
        s!"requires_grad={n.requires_grad}"} {pill s!"parents={n.parents}"}
      {pill s!"shape={dimsString n.value.s}"}
    </summary>
    <div style={json% {"margin-top": "8px", "padding-left": "10px"}}>
      {anyTensorHtml (α := α) n.value}
    </div>
  </details>

/-- Render a tape as an HTML panel (nodes + DOT). -/
def tapeHtml {α : Type} [ToString α] (t : Tape α) (maxDotChars : Nat := 6000) : ProofWidgets.Html :=
  let dot := tapeDot t
  let dotPreview :=
    if dot.length <= maxDotChars then dot
    else (dot.take maxDotChars).toString ++ "\n... (clipped)";
  <div style={json% {
    "padding": "10px",
    "border": "1px solid var(--vscode-panel-border, #e5e5e5)",
    "border-radius": "10px",
    "background": "var(--vscode-editor-background, transparent)",
    "color": "var(--vscode-editor-foreground, inherit)"
  }}>
    <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap", "margin-bottom":
      "10px"}}>
      {pill "Autograd.Tape"} {pill s!"nodes={t.nodes.size}"} {pill "viewer"}
    </div>
    <details «open»={true}>
      <summary>{.text "Nodes"}</summary>
      <div style={json% {"margin-top": "8px"}}>
        {... (t.nodes.mapIdx (fun i n => nodeHtml (α := α) i n))}
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

/-- Render per-node gradient tensors from a completed backward pass. -/
private def gradsTableHtml {α : Type} [ToString α] (t : Tape α)
    (grads : Std.HashMap Nat (Runtime.AnyTensor α)) : ProofWidgets.Html :=
  let entries : List (Nat × Runtime.AnyTensor α) := grads.toList;
  <div style={json% {"margin-top": "10px"}}>
    <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap", "margin-bottom":
      "8px"}}>
      {pill s!"grads={entries.length}"} {pill "nodeId -> dL/d(node)"}
    </div>
    <div>
      {... (entries.toArray).map (fun (id, g) =>
        <details style={json% {"margin": "6px 0"}}>
          <summary>
            {monospace s!"node {id}"} {pill s!"shape={dimsString g.s}"}
            {match t.getNode? id with
              | none => errBadge "missing node"
              | some n => okBadge (n.name.getD "(unnamed)")}
          </summary>
          <div style={json% {"margin-top": "8px", "padding-left": "10px"}}>
            {anyTensorHtml (α := α) g}
          </div>
        </details>)}
    </div>
  </div>

/-- Render a clipped preview for a list of node ids. -/
private def listPreviewNat (maxElems : Nat) (xs : List Nat) : String :=
  let head := xs.take maxElems
  let clipped : Bool := decide (xs.length > maxElems)
  "[" ++ String.intercalate ", " (head.map toString) ++ (if clipped then ", ..." else "") ++ "]"

/-- Summarize gradient coverage over nodes that require gradients. -/
private def gradCoverageHtml {α : Type} (t : Tape α) (outId : Nat)
    (grads : Std.HashMap Nat (Runtime.AnyTensor α)) : ProofWidgets.Html :=
  let pairs : List (Nat × Node α) := List.zip (List.range t.nodes.size) t.nodes.toList
  let req : List Nat := pairs.filter (fun (_, n) => n.requires_grad) |>.map (·.1)
  let leaf : List Nat := pairs.filter (fun (_, n) => n.parents.isEmpty) |>.map (·.1)
  let hasGrad (i : Nat) : Bool := (grads.get? i).isSome
  let got : List Nat := req.filter (fun i => hasGrad i)
  let missing : List Nat := req.filter (fun i => !hasGrad i)
  let missingLeaves : List Nat := leaf.filter (fun i => (t.getNode? i).map (·.requires_grad) |>.getD
    false) |>.filter (fun i => !hasGrad i)
  <details «open»={false}>
    <summary>
      {pill "Grad coverage"} {pill s!"outId={outId}"} {pill s!"requires_grad={req.length}"} {pill
        s!"gotGrad={got.length}"} {pill s!"missing={missing.length}"}
    </summary>
    <div style={json% {"margin-top": "8px", "display": "grid", "grid-template-columns": "1fr",
      "gap": "6px"}}>
      <div>{pill "missing ids"} {monospace (listPreviewNat 30 missing)}</div>
      <div>{pill "missing leaves"} {monospace (listPreviewNat 30 missingLeaves)}</div>
        <div style={json% {"opacity": 0.8}}>
          {.text
          ("Tip: missing grads usually means a `requires_grad=false` break, " ++
            "a disconnected tape, or choosing a non-scalar output id.")}
        </div>
      </div>
    </details>

/-- Render gradients from a scalar output id (like `loss.backward()`). -/
def tapeGradsHtml {α : Type} [ToString α] [Add α] [One α] [DecidableEq Shape]
    (t : Tape α) (outId : Nat) : ProofWidgets.Html :=
  match Tape.backwardScalar (α := α) (t := t) outId with
  | .ok grads =>
      let hasGrad (i : Nat) : Bool := (grads.get? i).isSome
      let dot := tapeDotColored (α := α) t outId hasGrad
      let dotPreview :=
        if dot.length <= 6000 then
          dot
        else
          (dot.take 6000).toString ++ "\n... (clipped)";
      <div style={json% {"display": "grid", "grid-template-columns": "1fr", "gap": "10px"}}>
        {tapeHtml (α := α) t}
        {gradCoverageHtml (α := α) t outId grads}
        <details style={json% {"margin-top": "0"}} «open»={false}>
          <summary>
            {pill "GraphViz DOT"} {pill "colored by grads"} {pill s!"outId={outId}"} {pill
              s!"gradNodes={grads.size}"}
          </summary>
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
        {gradsTableHtml (α := α) t grads}
      </div>
  | .error msg =>
      <div style={json% {"display": "grid", "grid-template-columns": "1fr", "gap": "10px"}}>
        {tapeHtml (α := α) t}
        <div style={json% {"padding": "10px", "border":
          "1px solid var(--vscode-panel-border, #e5e5e5)", "border-radius": "10px"}}>
          {errBadge "backwardScalar failed"} {monospace msg}
        </div>
      </div>

/-!
## Reverse-Pass Trace

`#tape_grads_view` answers "which nodes received a gradient?".

When you need to understand *why* a gradient was produced (or why it is missing), it is useful to
see the reverse traversal itself:
- the upstream cotangent at each node, and
- the per-parent contributions returned by the node’s local VJP rule.

This viewer runs reverse-mode and renders a step-by-step trace in reverse id order.
-/

/-- Render a reverse-pass trace for a tape, starting from a scalar output node `outId`. -/
def tapeTraceHtml {α : Type} [ToString α] [Add α] [One α] [DecidableEq Shape]
    (t : Tape α) (outId : Nat) : ProofWidgets.Html :=
  let seed : Runtime.AnyTensor α := AnyTensor.mk (Tensor.scalar (1 : α))
  match Tape.backwardDense (α := α) (t := t) outId seed with
  | .error msg =>
      <div style={json% {"display": "grid", "grid-template-columns": "1fr", "gap": "10px"}}>
        {tapeHtml (α := α) t}
        <div style={json% {"padding": "10px", "border":
          "1px solid var(--vscode-panel-border, #e5e5e5)", "border-radius": "10px"}}>
          {errBadge "backwardDense failed"} {monospace msg}
        </div>
      </div>
  | .ok dense =>
      let getGrad (id : Nat) : Option (Runtime.AnyTensor α) :=
        match dense[id]? with
        | none => none
        | some g? => g?
      let ids : List Nat := (List.range t.nodes.size).reverse
      let stepHtml (id : Nat) : ProofWidgets.Html :=
        match t.getNode? id with
        | none =>
            <div style={json% {"margin": "6px 0"}}>{errBadge s!"missing node {id}"}</div>
        | some node =>
            let g? := getGrad id
            let status : ProofWidgets.Html :=
              match g? with
              | none =>
                  if node.requires_grad then warnBadge "no upstream grad" else pill
                    "requires_grad=false"
              | some _ => okBadge "has upstream grad"
            ;
            <details style={json% {"margin": "6px 0"}}>
              <summary>
                {monospace s!"{id}: {node.name.getD "(unnamed)"}"} {pill s!"parents={node.parents}"}
                  {pill s!"shape={dimsString node.value.s}"} {status}
              </summary>
              <div style={json% {"margin-top": "8px", "padding-left": "10px", "display": "grid",
                "grid-template-columns": "1fr", "gap": "10px"}}>
                <div>
                  {pill "forward value"}
                  <div style={json% {"margin-top": "6px"}}>{anyTensorHtml (α := α) node.value}</div>
                </div>
                {match g? with
                  | none => ProofWidgets.Html.text ""
                  | some g =>
                      <div>
                        {pill "upstream dL/dy"}
                        <div style={json% {"margin-top": "6px"}}>{anyTensorHtml (α := α) g}</div>
                      </div>}
                {match g? with
                  | none => ProofWidgets.Html.text ""
                  | some g =>
                      match node.backward g with
                      | .error msg =>
                          <div>{errBadge "VJP failed"} <span style={json% {"margin-left":
                            "8px"}}>{monospace msg}</span></div>
                      | .ok contribs =>
                          let contribHtml : Array ProofWidgets.Html :=
                            contribs.toArray.map (fun (pid, pg) =>
                              <details style={json% {"margin": "6px 0"}}>
                                <summary>{monospace s!"parent {pid}"} {pill
                                  s!"shape={dimsString pg.s}"}</summary>
                                <div style={json% {"margin-top": "6px", "padding-left": "10px"}}>
                                  {anyTensorHtml (α := α) pg}
                                </div>
                              </details>)
                          ;
                          <div>
                            <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap",
                              "align-items": "center"}}>
                              {pill s!"contribs={contribs.length}"} {pill "pid -> dL/d(parent)"}
                            </div>
                            <div style={json% {"margin-top": "8px"}}>
                              {... contribHtml}
                            </div>
                          </div>}
              </div>
            </details>
      let steps : Array ProofWidgets.Html := ids.toArray.map stepHtml
      ;
      <div style={json% {"display": "grid", "grid-template-columns": "1fr", "gap": "10px"}}>
        {tapeHtml (α := α) t}
        <div style={json% {
          "padding": "10px",
          "border": "1px solid var(--vscode-panel-border, #e5e5e5)",
          "border-radius": "10px",
          "background": "var(--vscode-editor-background, transparent)"
        }}>
          <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap", "margin-bottom":
            "10px"}}>
            {pill "Reverse-pass trace"} {pill s!"outId={outId}"} {pill s!"nodes={t.nodes.size}"}
              {pill "seed=1 (scalar)"}
          </div>
          <div style={json% {"opacity": 0.85, "margin-bottom": "10px"}}>
            {.text
              ("Each step shows the upstream gradient at that node and the parent contributions " ++
                "produced by the node’s VJP rule.")}
          </div>
          <details «open»={true}>
            <summary>{.text "Steps (reverse id order)"}</summary>
            <div style={json% {"margin-top": "8px"}}>
              {... steps}
            </div>
          </details>
        </div>
      </div>

/-!
## Commands
-/

syntax (name := tapeViewCmd) "#tape_view " term : command
-- Use an explicit delimiter so the first term can't greedily parse as an application
-- that swallows the second argument.
syntax (name := tapeGradsViewCmd) "#tape_grads_view " term ", " term : command
syntax (name := tapeTraceViewCmd) "#tape_trace_view " term ", " term : command

macro "#tape_view " t:term : command =>
  Lean.TSyntax.mkInfoCanonical <$> `(#html (tapeHtml $t))

macro "#tape_grads_view " t:term ", " outId:term : command =>
  Lean.TSyntax.mkInfoCanonical <$> `(#html (tapeGradsHtml $t $outId))

macro "#tape_trace_view " t:term ", " outId:term : command =>
  Lean.TSyntax.mkInfoCanonical <$> `(#html (tapeTraceHtml $t $outId))

end NN.Widgets
