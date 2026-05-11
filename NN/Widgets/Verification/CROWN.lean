/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import NN.IR.Pretty
public meta import NN.MLTheory.CROWN.Graph
public meta import NN.Widgets.Core.Tensor
public meta import NN.Widgets.Core.UI
public meta import ProofWidgets.Component.HtmlDisplay
public meta import ProofWidgets.Demos.Macro

/-!
# Verification

Verification widgets (bounds / certificates).

This module provides small infoview panels for *bound propagation* artifacts, aimed at debugging
and teaching rather than proofs:

- `#crown_view g, st` shows a per-node table for a `CROWN.graph.PropState`, including optional IBP
  boxes and optional affine forms.

The goal is to make it easy to sanity-check:
- which nodes got bounds,
- the shapes and flattened dimensions that the propagation engine believes it is operating on,
- and small previews of the vectors/matrices involved.

## Main definitions

- `crownPropHtml`: interactive per-node state viewer for CROWN/IBP propagation.
- `boundsTightnessHtml`: interval-width diagnostic panel (`hi - lo`) per node.
- `#crown_view g, st`: command form for `crownPropHtml`.
- `#bounds_tightness_view g, st`: command form for `boundsTightnessHtml`.

## Implementation notes

- We keep this viewer "pure HTML" using ProofWidgets `#html` with no custom JS.
- The same node ids are used for IR nodes and propagated states, so mismatches are easy to spot.
- DOT text is intentionally clipped for large graphs to keep infoview rendering responsive.

## References

- [CROWN / LiRPA style bound propagation](https://arxiv.org/abs/1811.00866)
- [GraphViz DOT language](https://graphviz.org/doc/info/lang.html)
- [ProofWidgets](https://github.com/leanprover-community/ProofWidgets4)
- [Lean community documentation style](https://leanprover-community.github.io/contribute/doc.html)

## Tags

verification, crown, ibp, bounds, certificates, widgets
-/

public meta section

open scoped ProofWidgets.Jsx

namespace NN.Widgets

open _root_.Spec
open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN
open UI

private def flatBoxHtml {α : Type} [Context α] [ToString α] (b : FlatBox α) : ProofWidgets.Html :=
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

private def affineVecHtml {α : Type} [ToString α] {inDim outDim : Nat}
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

private def nodeStateHtml {α : Type} [Context α] [ToString α]
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
                {pill s!"shape={Shape.pretty st.shape}"} {pill s!"flatDim={Shape.size st.shape}"}
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

/-- Render a short list preview, clipping with `...` when the list is long. -/
private def listPreview {α : Type} [ToString α] (maxElems : Nat) (xs : List α) : String :=
  let head := xs.take maxElems
  let clipped : Bool := decide (xs.length > maxElems)
  "[" ++ String.intercalate ", " (head.map toString) ++ (if clipped then ", ..." else "") ++ "]"

/-- Produce a compact one-line preview of a flat interval box. -/
private def flatBoxPreview {α : Type} [Context α] [ToString α] (b : FlatBox α) (maxElems : Nat := 4)
  : String :=
  let lo := toList (α := α) (s := .dim b.dim .scalar) b.lo
  let hi := toList (α := α) (s := .dim b.dim .scalar) b.hi
  s!"lo={listPreview (α := α) maxElems lo}, hi={listPreview (α := α) maxElems hi}"

/-- Build a DOT graph for quick visualization of state coverage across nodes. -/
private def crownDot {α : Type} [Context α] [ToString α] (g : Graph) (ps : PropState α)
    (maxNodes : Nat := 400) : String :=
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
            let base := s!"{nid}: {n.kind.tag}\\nshape={Shape.pretty st.shape}"
            match st.ibp? with
            | none => base
            | some b => base ++ s!"\\n{flatBoxPreview (α := α) b 3}"
        | some n, none =>
            s!"{nid}: {n.kind.tag}\\n(no state)"
        | none, some st =>
            s!"{nid}: <missing node>\\nshape={Shape.pretty st.shape}"
        | none, none =>
            s!"{nid}: <missing node>\\n(no state)"
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
    (List.range n).foldl (fun acc nid =>
      match g.nodes[nid]? with
      | none => acc
      | some nd => acc ++ (nd.parents.map (fun p => s!"  n{p} -> n{nid};"))) []
  header ++ String.intercalate "\n" (nodes ++ edges) ++ "\n}\n"

/-- Render a compact table summary of per-node state availability and previews. -/
private def compactTableHtml {α : Type} [Context α] [ToString α]
    (g : Graph) (ps : PropState α) (maxNodes : Nat := 200) : ProofWidgets.Html :=
  let nG := g.size
  let nS := ps.states.size
  let n := max nG nS
  let ids := (List.range n).take maxNodes
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
def crownPropHtml {α : Type} [Context α] [ToString α] (g : Graph) (ps : PropState α)
    (maxNodes : Nat := 200) : ProofWidgets.Html :=
  let nG := g.size
  let nS := ps.states.size
  let n := max nG nS
  let clipped : Bool := decide (n > maxNodes)
  let ids := (List.range n).take maxNodes;
  let dot := crownDot (α := α) g ps
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
      {.text "This viewer displays optional IBP boxes and affine forms per IR node id."}
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
interval widths `hi - lo` node-by-node.

This viewer computes width summaries per node and highlights missing IBP coverage.
-/

/-- Compute `(min, max, mean)` for a nonempty list. -/
private def listStats {α : Type} [Context α] (xs : List α) : Option (α × α × α) :=
  match xs with
  | [] => none
  | x :: rest =>
      let mn := rest.foldl (fun acc y => min acc y) x
      let mx := rest.foldl (fun acc y => max acc y) x
      let sum := rest.foldl (fun acc y => acc + y) x
      let mean := sum / (↑((x :: rest).length) : α)
      some (mn, mx, mean)

private def flatBoxWidthTensor {α : Type} [Context α] (b : FlatBox α) : Tensor α (.dim b.dim
  .scalar) :=
  -- Width per flattened component.
  Tensor.subSpec (α := α) b.hi b.lo

/-- Render a per-node diagnostic panel summarizing IBP interval widths (`hi - lo`). -/
def boundsTightnessHtml {α : Type} [Context α] [ToString α]
    (g : Graph) (ps : PropState α) (maxNodes : Nat := 200) : ProofWidgets.Html :=
  let nG := g.size
  let nS := ps.states.size
  let n := max nG nS
  let ids := (List.range n).take maxNodes
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
                    let ws : List α := toList (α := α) (s := .dim b.dim .scalar) wT
                    let stats := listStats (α := α) ws
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
  Lean.TSyntax.mkInfoCanonical <$> `(#html (crownPropHtml $g $ps))

syntax (name := boundsTightnessViewCmd) "#bounds_tightness_view " term ", " term : command

macro "#bounds_tightness_view " g:term ", " ps:term : command =>
  Lean.TSyntax.mkInfoCanonical <$> `(#html (boundsTightnessHtml $g $ps))

end NN.Widgets
