/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import NN.IR.Pretty
public meta import NN.Widgets.Core.Tensor
public meta import NN.MLTheory.CROWN.Graph -- shake: keep
public import NN.Widgets.Core.Tensor
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import NN.Spec.Core.Tensor
import ProofWidgets.Component.HtmlDisplay

/-!
# Verification

Verification widgets (bounds / certificates).

This module provides small infoview panels for *bound propagation* artifacts, aimed at debugging
and teaching rather than proofs:

- `#crown_view g, st` shows a per-node table for a `CROWN.graph.PropState`, including optional IBP
  boxes and optional affine forms.

The panel makes it easy to inspect:
- which nodes got bounds,
- the shapes and flattened dimensions that the propagation engine believes it is operating on,
- and small previews of the vectors/matrices involved.

## Main definitions

- `crownPropHtml`: interactive per-node state viewer for CROWN/IBP propagation.
- `boundsTightnessHtml`: interval-width diagnostic panel ($\mathrm{hi}-\mathrm{lo}$) per node.
- `#crown_view g, st`: command form for `crownPropHtml`.
- `#bounds_tightness_view g, st`: command form for `boundsTightnessHtml`.
-/

public meta section

open scoped ProofWidgets.Jsx

namespace NN.Widgets

open _root_.Spec _root_.TorchLean
open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN
open UI

/-- Show a flat interval box as its two endpoint vectors side by side. -/
private def flatBoxHtml {α : Type} [TorchLean.Storage α] [Context α] [ToString α] (b : FlatBox α) :
    ProofWidgets.Html :=
  <div style={json% {
    "display": "grid",
    "grid-template-columns": "1fr 1fr",
    "gap": "10px",
    "margin-top": "8px"
  }}>
    <div>
      <div style={json% {"margin-bottom": "6px"}}>{pill "lo"}</div>
      {tensorHtml (α := α) (s := .dim b.dim .scalar) b.lo (maxRows := 12) (maxCols := 16) (maxElems
        := 64)}
    </div>
    <div>
      <div style={json% {"margin-bottom": "6px"}}>{pill "hi"}</div>
      {tensorHtml (α := α) (s := .dim b.dim .scalar) b.hi (maxRows := 12) (maxCols := 16) (maxElems
        := 64)}
    </div>
  </div>

/-- Show an affine form as two collapsed panels, the coefficient matrix and the offset. -/
private def affineVecHtml {α : Type} [TorchLean.Storage α] [ToString α] {inDim outDim : Nat}
    (aff : NN.MLTheory.CROWN.AffineVec α inDim outDim) : ProofWidgets.Html :=
  <div style={json% {"display": "grid", "grid-template-columns": "1fr", "gap": "10px", "margin-top":
    "8px"}}>
    <details «open»={false}>
      <summary>
        {pill s!"A : {outDim}×{inDim}"} {pill "coeffs"}
      </summary>
      <div style={json% {"margin-top": "8px"}}>
        {tensorHtml (α := α) (s := .dim outDim (.dim inDim .scalar)) aff.A (maxRows := 10) (maxCols
          := 12) (maxElems := 64)}
      </div>
    </details>
    <details «open»={false}>
      <summary>
        {pill s!"c : {outDim}"} {pill "offset"}
      </summary>
      <div style={json% {"margin-top": "8px"}}>
        {tensorHtml (α := α) (s := .dim outDim .scalar) aff.c (maxRows := 12) (maxCols := 16)
          (maxElems := 64)}
      </div>
    </details>
  </div>

/-- One row of the CROWN panel: the IR node's pretty line plus whatever state the run recorded. -/
private def nodeStateHtml {α : Type} [TorchLean.Storage α] [Context α] [ToString α]
    (nid : Nat) (n? : Option Node) (st? : Option (NodeState α)) : ProofWidgets.Html :=
  let summaryLine : String :=
    match n? with
    | none => s!"{nid}: <missing IR node>"
    | some n => s!"{nid}: {NN.IR.Node.prettyLine n}"
  let statusBadges : ProofWidgets.Html :=
    match st? with
    | none => warnBadge "no state"
    | some st =>
        let ibpB :=
          match st.ibp? with
          | none => warnBadge "IBP: none"
          | some _ => okBadge "IBP: some"
        ;
        let affB :=
          match st.aff? with
          | none => warnBadge "Affine: none"
          | some _ => okBadge "Affine: some"
        ;
        <span
          style={json% {"display": "inline-flex", "gap": "6px", "flex-wrap": "wrap"}}
        >{ibpB} {affB}</span>
  ;
  <details style={json% {"margin": "6px 0"}}>
    <summary>{monospace summaryLine}
      <span style={json% {"margin-left": "8px"}}>{statusBadges}</span>
    </summary>
    <div style={json% {"margin-top": "8px", "padding-left": "10px"}}>
      {match st? with
        | none =>
            <div style={json% {"opacity": 0.8}}>{.text
              "No bound state available for this node id."}</div>
        | some st =>
            <div>
              <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap"}}>
                {pill s!"shape={Shape.pretty st.shape}"}
                {pill s!"flatDim={Spec.Shape.size st.shape}"}
              </div>
              {match st.ibp? with
                | none => ProofWidgets.Html.text ""
                | some b =>
                    <details style={json% {"margin-top": "10px"}} «open»={false}>
                      <summary>{pill s!"IBP box (dim={b.dim})"}</summary>
                      {flatBoxHtml (α := α) b}
                    </details>}
              {match st.aff? with
                | none => ProofWidgets.Html.text ""
                | some a =>
                    <details style={json% {"margin-top": "10px"}} «open»={false}>
                      <summary>{pill s!"Affine form (in={a.inDim}, out={a.outDim})"}</summary>
                      {affineVecHtml (α := α) (inDim := a.inDim) (outDim := a.outDim) a.aff}
                    </details>}
            </div>}
    </div>
  </details>

/-- Render a short array preview, clipping with `...` when the array is long. -/
private def arrayPreview {α : Type} [ToString α] (maxElems : Nat) (xs : Array α) : String :=
  let head := xs.extract 0 (min maxElems xs.size)
  let body := head.foldl (fun acc x =>
    if acc.isEmpty then toString x else acc ++ ", " ++ toString x) ""
  let suffix := if xs.size > maxElems then ", ..." else ""
  "[" ++ body ++ suffix ++ "]"

/-- Produce a compact one-line preview of a flat interval box. -/
private def flatBoxPreview {α : Type} [TorchLean.Storage α] [Context α] [ToString α]
    (b : FlatBox α) (maxElems : Nat := 4) : String :=
  let lo := Tensor.to b.lo (Array α)
  let hi := Tensor.to b.hi (Array α)
  s!"lo={arrayPreview (α := α) maxElems lo}, hi={arrayPreview (α := α) maxElems hi}"

/-- Build a DOT graph for compact visualization of state coverage across nodes. -/
private def crownDot {α : Type} [TorchLean.Storage α] [Context α] [ToString α] (g : Graph)
    (ps : PropState α) (maxNodes : Nat := 400) : String :=
  let nG := g.size
  let nS := ps.states.size
  let n := min (max nG nS) maxNodes
  let header :=
    "digraph IR {\n" ++
    "  rankdir=LR;\n" ++
    "  node [shape=box,fontname=\"monospace\",style=filled,fillcolor=\"#f7f7f7\"];\n"
  let nodes : List String :=
    (List.range n).map (fun nid =>
      let n? := g.nodes[nid]?
      let st? := ps.states[nid]?
      let label :=
        match n?, st? with
        | some n, some st =>
            let base := s!"{nid}: {n.kind.tag}\nshape={Shape.pretty st.shape}"
            match st.ibp? with
            | none => base
            | some b => base ++ s!"\n{flatBoxPreview (α := α) b 3}"
        | some n, none =>
            s!"{nid}: {n.kind.tag}\n(no state)"
        | none, some st =>
            s!"{nid}: <missing node>\nshape={Shape.pretty st.shape}"
        | none, none =>
            s!"{nid}: <missing node>\n(no state)"
      let fill :=
        match st? with
        | none => "#fff3cc"
        | some st =>
            match st.ibp?, st.aff? with
            | none, none => "#fff3cc"
            | some _, none => "#ddffea"
            | none, some _ => "#dde9ff"
            | some _, some _ => "#d7fff6"
      s!"  n{nid} [label=\"{escapeDotLabel label}\", fillcolor=\"{fill}\"];")
  let edges : List String :=
    (List.range n).flatMap (fun nid =>
      match g.nodes[nid]? with
      | none => []
      | some nd => (nd.parents.map (fun p => s!"  n{p} -> n{nid};")).toList)
  header ++ String.intercalate "\n" (nodes ++ edges) ++ "\n}\n"

/-- Render a compact table summary of per-node state availability and previews. -/
private def compactTableHtml {α : Type} [TorchLean.Storage α] [Context α] [ToString α]
    (g : Graph) (ps : PropState α) (maxNodes : Nat := 200) : ProofWidgets.Html :=
  let nG := g.size
  let nS := ps.states.size
  let n := max nG nS
  let ids := List.range (min n maxNodes)
  let clipped : Bool := decide (n > maxNodes);
  <div style={json% {"overflow": "auto", "max-height": "360px", "border":
    "1px solid var(--vscode-panel-border, #e5e5e5)", "border-radius": "10px"}}>
    <table style={json% {"border-collapse": "collapse", "width": "100%"}}>
      <thead>
        <tr>
          <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
            "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "id"}</th>
          <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
            "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "op"}</th>
          <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
            "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "shape"}</th>
          <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
            "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "IBP/Aff"}</th>
          <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
            "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "preview"}</th>
        </tr>
      </thead>
      <tbody>
        {... ids.toArray.map (fun nid =>
          let n? := g.nodes[nid]?
          let st? := ps.states[nid]?
          let op := match n? with | none => "<missing>" | some n => n.kind.tag
          let shape := match st? with | none => "?" | some st => Shape.pretty st.shape
          let flags :=
            match st? with
            | none => "none"
            | some st =>
                let ibp := match st.ibp? with | none => "IBP:none" | some _ => "IBP:some"
                let aff := match st.aff? with | none => "Aff:none" | some _ => "Aff:some"
                ibp ++ " " ++ aff
          let preview :=
            match st? with
            | none => ""
            | some st =>
                match st.ibp? with
                | none => ""
                | some b => flatBoxPreview (α := α) b 3;
          <tr>
            <td style={json% {"padding": "6px 8px", "border-bottom":
              "1px solid rgba(127,127,127,0.18)"}}>{monospace (toString nid)}</td>
            <td style={json% {"padding": "6px 8px", "border-bottom":
              "1px solid rgba(127,127,127,0.18)"}}>{monospace op}</td>
            <td style={json% {"padding": "6px 8px", "border-bottom":
              "1px solid rgba(127,127,127,0.18)"}}>{monospace shape}</td>
            <td style={json% {"padding": "6px 8px", "border-bottom":
              "1px solid rgba(127,127,127,0.18)"}}>{monospace flags}</td>
            <td style={json% {"padding": "6px 8px", "border-bottom":
              "1px solid rgba(127,127,127,0.18)", "white-space": "nowrap"}}>{monospace preview}</td>
          </tr>)}
      </tbody>
    </table>
    {if clipped then
      <div style={json% {"padding": "6px 8px", "opacity": 0.7}}>{.text "... (more nodes)"}</div>
     else
      ProofWidgets.Html.text ""}
  </div>

/-- Render a `CROWN.graph.PropState` as a per-node HTML panel. -/
def crownPropHtml {α : Type} [TorchLean.Storage α] [Context α] [ToString α] (g : Graph)
    (ps : PropState α) (maxNodes : Nat := 200) : ProofWidgets.Html :=
  let nG := g.size
  let nS := ps.states.size
  let n := max nG nS
  let clipped : Bool := decide (n > maxNodes)
  let ids := List.range (min n maxNodes);
  let dot := crownDot (α := α) g ps (maxNodes := maxNodes)
  let dotPreview :=
    if dot.length <= 6000 then
      dot
    else
      (dot.take 6000).toString ++ "\n... (clipped)";
  <div style={json% {
    "padding": "10px",
    "border": "1px solid var(--vscode-panel-border, #e5e5e5)",
    "border-radius": "10px",
    "background": "var(--vscode-editor-background, transparent)",
    "color": "var(--vscode-editor-foreground, inherit)"
  }}>
    <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap", "margin-bottom":
      "10px"}}>
      {pill "CROWN / IBP state"} {pill s!"graphNodes={nG}"} {pill s!"stateNodes={nS}"}
      {pill s!"inputId={ps.inputId}"} {pill s!"inputDim={ps.inputDim}"}
      {if nG = nS then okBadge "sizes match" else warnBadge "size mismatch"}
      {if clipped then warnBadge s!"clipped to {maxNodes}" else ProofWidgets.Html.text ""}
    </div>
    <div style={json% {"opacity": 0.85, "margin-bottom": "10px"}}>
      {.text ("This viewer displays optional IBP boxes and affine forms per IR node id. " ++
          "Their presence is diagnostic state, not a checked certificate.")}
    </div>
    <details «open»={false} style={json% {"margin-bottom": "10px"}}>
      <summary>{.text "Compact summary (table)"}</summary>
      <div style={json% {"margin-top": "8px"}}>
        {compactTableHtml (α := α) g ps (maxNodes := maxNodes)}
      </div>
    </details>
    <details «open»={false} style={json% {"margin-bottom": "10px"}}>
      <summary>{.text "GraphViz DOT (colored by state coverage)"}</summary>
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
    <details «open»={true}>
      <summary>{.text "Per-node states"}</summary>
      <div style={json% {"margin-top": "8px"}}>
        {... ids.toArray.map (fun nid =>
          let n? := g.nodes[nid]?
          let st? := ps.states[nid]?
          nodeStateHtml (α := α) nid n? st?)}
        {if clipped then
          <div style={json% {"margin-top": "8px", "opacity": 0.7}}>{.text "... (more nodes)"}</div>
         else
          ProofWidgets.Html.text ""}
      </div>
    </details>
  </div>

/-!
## Bounds Tightness

When IBP boxes exist, a very fast diagnostic for "where are my bounds blowing up?" is to look at
interval widths $\mathrm{hi}-\mathrm{lo}$ node-by-node.

This viewer computes width summaries per node and highlights missing IBP coverage.
-/

/-- Compute `(min, max, mean)` for a nonempty array. -/
private def arrayStats {α : Type} [TorchLean.Storage α] [Context α] (xs : Array α) :
    Option (α × α × α) :=
  xs[0]?.map fun x =>
    let rest := xs.extract 1 xs.size
    let mn := rest.foldl (fun acc y => min acc y) x
    let mx := rest.foldl (fun acc y => max acc y) x
    let sum := rest.foldl (fun acc y => acc + y) x
    let mean := sum / (↑xs.size : α)
    (mn, mx, mean)

/-- Per-component width `hi - lo` of a box, the quantity the diagnostic panel ranks nodes by. -/
private def flatBoxWidthTensor {α : Type} [TorchLean.Storage α] [Context α] (b : FlatBox α) :
    Tensor α [b.dim] :=
  Tensor.subSpec (α := α) b.hi b.lo

/-- Render a per-node diagnostic panel summarizing IBP interval widths
$\mathrm{hi}-\mathrm{lo}$. -/
def boundsTightnessHtml {α : Type} [TorchLean.Storage α] [Context α] [ToString α]
    (g : Graph) (ps : PropState α) (maxNodes : Nat := 200) : ProofWidgets.Html :=
  let nG := g.size
  let nS := ps.states.size
  let n := max nG nS
  let ids := List.range (min n maxNodes)
  let clipped : Bool := decide (n > maxNodes)
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
      {pill "IBP width diagnostic"} {pill s!"graphNodes={nG}"} {pill s!"stateNodes={nS}"}
      {pill "width = hi - lo"} {if clipped then warnBadge s!"clipped to {maxNodes}" else
        ProofWidgets.Html.text ""}
    </div>
    <div style={json% {"opacity": 0.85, "margin-bottom": "10px"}}>
      {.text
        ("Use this to spot nodes where IBP intervals become very wide " ++
          "(a common source of conservative certificates).")}
    </div>
    <div style={json% {"overflow": "auto", "max-height": "520px",
      "border": "1px solid var(--vscode-panel-border, #e5e5e5)", "border-radius": "10px"}}>
      <table style={json% {"border-collapse": "collapse", "width": "100%"}}>
        <thead>
          <tr>
            <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
              "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "id"}</th>
            <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
              "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "op"}</th>
            <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
              "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "shape"}</th>
            <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
              "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "dim"}</th>
            <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
              "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "maxWidth"}</th>
            <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
              "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "meanWidth"}</th>
            <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
              "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "details"}</th>
          </tr>
        </thead>
        <tbody>
          {... ids.toArray.map (fun nid =>
            let n? := g.nodes[nid]?
            let st? := ps.states[nid]?
            let op := match n? with | none => "<missing>" | some n => n.kind.tag
            let shape := match st? with | none => "?" | some st => Shape.pretty st.shape
            match st? with
            | none =>
                <tr>
                  <td style={json% {"padding": "6px 8px", "border-bottom":
                    "1px solid rgba(127,127,127,0.18)"}}>{monospace (toString nid)}</td>
                  <td style={json% {"padding": "6px 8px", "border-bottom":
                    "1px solid rgba(127,127,127,0.18)"}}>{monospace op}</td>
                  <td style={json% {"padding": "6px 8px", "border-bottom":
                    "1px solid rgba(127,127,127,0.18)"}}>{monospace shape}</td>
                  <td style={json% {"padding": "6px 8px", "border-bottom":
                    "1px solid rgba(127,127,127,0.18)"}}>{monospace "?"}</td>
                  <td style={json% {"padding": "6px 8px", "border-bottom":
                    "1px solid rgba(127,127,127,0.18)"}}>{warnBadge "no state"}</td>
                  <td style={json% {"padding": "6px 8px", "border-bottom":
                    "1px solid rgba(127,127,127,0.18)"}}>{ProofWidgets.Html.text ""}</td>
                  <td style={json% {"padding": "6px 8px", "border-bottom":
                    "1px solid rgba(127,127,127,0.18)"}}>{ProofWidgets.Html.text ""}</td>
                </tr>
            | some st =>
                match st.ibp? with
                | none =>
                    <tr>
                      <td style={json% {"padding": "6px 8px", "border-bottom":
                        "1px solid rgba(127,127,127,0.18)"}}>{monospace (toString nid)}</td>
                      <td style={json% {"padding": "6px 8px", "border-bottom":
                        "1px solid rgba(127,127,127,0.18)"}}>{monospace op}</td>
                      <td style={json% {"padding": "6px 8px", "border-bottom":
                        "1px solid rgba(127,127,127,0.18)"}}>{monospace shape}</td>
                      <td style={json% {"padding": "6px 8px", "border-bottom":
                        "1px solid rgba(127,127,127,0.18)"}}>{monospace "?"}</td>
                      <td style={json% {"padding": "6px 8px", "border-bottom":
                        "1px solid rgba(127,127,127,0.18)"}}>{warnBadge "IBP:none"}</td>
                      <td style={json% {"padding": "6px 8px", "border-bottom":
                        "1px solid rgba(127,127,127,0.18)"}}>{ProofWidgets.Html.text ""}</td>
                      <td style={json% {"padding": "6px 8px", "border-bottom":
                        "1px solid rgba(127,127,127,0.18)"}}>{ProofWidgets.Html.text ""}</td>
                    </tr>
                | some b =>
                    let wT := flatBoxWidthTensor (α := α) b
                    let ws : Array α := Tensor.to wT (Array α)
                    let stats := arrayStats (α := α) ws
                    let mxS := match stats with | none => "?" | some s => toString s.2.1
                    let meanS := match stats with | none => "?" | some s => toString s.2.2
                    ;
                    <tr>
                      <td style={json% {"padding": "6px 8px", "border-bottom":
                        "1px solid rgba(127,127,127,0.18)"}}>{monospace (toString nid)}</td>
                      <td style={json% {"padding": "6px 8px", "border-bottom":
                        "1px solid rgba(127,127,127,0.18)"}}>{monospace op}</td>
                      <td style={json% {"padding": "6px 8px", "border-bottom":
                        "1px solid rgba(127,127,127,0.18)"}}>{monospace shape}</td>
                      <td style={json% {"padding": "6px 8px", "border-bottom":
                        "1px solid rgba(127,127,127,0.18)"}}>{monospace (toString b.dim)}</td>
                      <td style={json% {"padding": "6px 8px", "border-bottom":
                        "1px solid rgba(127,127,127,0.18)"}}>{monospace mxS}</td>
                      <td style={json% {"padding": "6px 8px", "border-bottom":
                        "1px solid rgba(127,127,127,0.18)"}}>{monospace meanS}</td>
                      <td style={json% {"padding": "6px 8px", "border-bottom":
                        "1px solid rgba(127,127,127,0.18)"}}>
                        <details «open»={false}>
                          <summary>{pill "width vector"}</summary>
                          <div style={json% {"margin-top": "6px"}}>
                            {tensorHtml (α := α) (s := .dim b.dim .scalar) wT (maxRows := 12)
                              (maxCols := 16) (maxElems := 96)}
                          </div>
                        </details>
                      </td>
                    </tr>
          )}
        </tbody>
      </table>
    </div>
  </div>

/-!
## Commands
-/

-- Use an explicit delimiter so the first term can't greedily parse as an application
-- that swallows the second argument.
syntax (name := crownViewCmd) "#crown_view " term ", " term : command

macro "#crown_view " g:term ", " ps:term : command =>
  UI.canonicalCommand <$> `(#html (crownPropHtml $g $ps))

syntax (name := boundsTightnessViewCmd) "#bounds_tightness_view " term ", " term : command

macro "#bounds_tightness_view " g:term ", " ps:term : command =>
  UI.canonicalCommand <$> `(#html (boundsTightnessHtml $g $ps))

end NN.Widgets
