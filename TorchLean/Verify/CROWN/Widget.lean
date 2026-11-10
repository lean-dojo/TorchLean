/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import TorchLean.NN.IR.Pretty
public meta import TorchLean.NN.Widget.Tensor
public meta import TorchLean.Verify.CROWN.Graph
public import TorchLean.NN.Widget.Tensor
import ProofWidgets.Component.HtmlDisplay
public import TorchLean.Verify.CROWN.FlatBox

/-!
# CROWN Widgets

Infoview panels for bound propagation over a graph of registered operators (`TorchLean.IR.Graph`),
for debugging and teaching. `#crown_view g, boxes, affine` shows a per-node table of the interval
boxes of `TorchLean.CROWN.Op.runIBP` and the affine bounds of `TorchLean.CROWN.Op.runCROWN`,
each optional per node (`#crown_view g, boxes` without affine bounds). The table gives the shapes
and flattened dimensions of every node and small previews of its vectors and matrices.

## Main definitions

- `crownPropHtml`: interactive per-node viewer of IBP boxes and CROWN affine bounds.
- `boundsTightnessHtml`: interval-width diagnostic panel ($\mathrm{hi}-\mathrm{lo}$) per node.
- `#crown_view g, boxes, affine` (or `g, boxes`): command form for `crownPropHtml`.
- `#bounds_tightness_view g, boxes`: command form for `boundsTightnessHtml`.
-/

public meta section

open scoped ProofWidgets.Jsx

namespace TorchLean.Widget

open CROWN.Graph
open CROWN
open IR
open UI

/-- Show a flat interval box as its two endpoint vectors side by side. -/
private def flatBoxHtml {α : Type} [Storage α] [Context α] [ToString α] (b : FlatBox α) :
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
private def affineVecHtml {α : Type} [Storage α] [ToString α] {inDim outDim : Nat}
    (aff : Linear α inDim outDim) : ProofWidgets.Html :=
  <div style={json% {"display": "grid", "grid-template-columns": "1fr", "gap": "10px", "margin-top":
    "8px"}}>
    <details «open»={false}>
      <summary>
        {pill s!"A : {outDim}×{inDim}"} {pill "coeffs"}
      </summary>
      <div style={json% {"margin-top": "8px"}}>
        {tensorHtml (α := α) (s := .dim outDim (.dim inDim .scalar)) aff.weights (maxRows := 10)
          (maxCols := 12) (maxElems := 64)}
      </div>
    </details>
    <details «open»={false}>
      <summary>
        {pill s!"c : {outDim}"} {pill "offset"}
      </summary>
      <div style={json% {"margin-top": "8px"}}>
        {tensorHtml (α := α) (s := .dim outDim .scalar) aff.bias (maxRows := 12) (maxCols := 16)
          (maxElems := 64)}
      </div>
    </details>
  </div>

/-- One collapsible entry of the per-node list: the node's line with IBP and affine status badges,
then its output shapes, its box and its lower and upper affine bounds when they exist. -/
private def nodeStateHtml {α : Type} [Storage α] [Context α] [ToString α]
    (nid : Nat) (n? : Option Node) (box? : Option (FlatBox α))
    (aff? : Option (FlatAffineBounds α)) : ProofWidgets.Html :=
  let summaryLine : String :=
    match n? with
    | none => s!"{nid}: <missing IR node>"
    | some n => n.prettyLine nid
  let statusBadges : ProofWidgets.Html :=
    let ibpB :=
      match box? with
      | none => warnBadge "IBP: none"
      | some _ => okBadge "IBP: some"
    ;
    let affB :=
      match aff? with
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
      {match box?, aff? with
        | none, none =>
            <div style={json% {"opacity": 0.8}}>{.text
              "No bound state available for this node id."}</div>
        | _, _ =>
            <div>
              {match n? with
                | none => ProofWidgets.Html.text ""
                | some n =>
                    <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap"}}>
                      {pill s!"out={n.outsPretty}"}
                    </div>}
              {match box? with
                | none => ProofWidgets.Html.text ""
                | some b =>
                    <details style={json% {"margin-top": "10px"}} «open»={false}>
                      <summary>{pill s!"IBP box (dim={b.dim})"}</summary>
                      {flatBoxHtml (α := α) b}
                    </details>}
              {match aff? with
                | none => ProofWidgets.Html.text ""
                | some a =>
                    <details style={json% {"margin-top": "10px"}} «open»={false}>
                      <summary>{pill s!"Affine bounds (in={a.inDim}, out={a.outDim})"}</summary>
                      <div style={json% {"margin-top": "8px"}}>{pill "lower"}</div>
                      {affineVecHtml (α := α) (inDim := a.inDim) (outDim := a.outDim) a.loAff}
                      <div style={json% {"margin-top": "8px"}}>{pill "upper"}</div>
                      {affineVecHtml (α := α) (inDim := a.inDim) (outDim := a.outDim) a.hiAff}
                    </details>}
            </div>}
    </div>
  </details>

/-- One-line preview `lo=[...], hi=[...]` of a box: the first `maxElems` entries of each endpoint,
read without converting the full buffers, followed by `...` when the box has more. -/
private def flatBoxPreview {α : Type} [Storage α] [Context α] [ToString α]
    (b : FlatBox α) (maxElems : Nat := 4) : String :=
  let preview (t : Tensor α (.dim b.dim .scalar)) : String :=
    let entries := (prefixEntries t maxElems).toList.map toString
    let entries := if b.dim > maxElems then entries ++ ["..."] else entries
    "[" ++ String.intercalate ", " entries ++ "]"
  s!"lo={preview b.lo}, hi={preview b.hi}"

/-- GraphViz DOT of the graph (`Graph.toDot`). A node's fill colour tells whether it has a box,
affine bounds, both or neither, and the label of a node with a box ends with a three-entry preview
of it. -/
private def crownDot {α : Type} [Storage α] [Context α] [ToString α] (g : Graph)
    (boxes : Array (Option (FlatBox α))) (affine : Array (Option (FlatAffineBounds α))) :
    String :=
  let label (nid : Nat) (n : Node) : String :=
    match boxes[nid]?.join with
    | none => Node.dotLabel nid n
    | some b => Node.dotLabel nid n ++ s!"\n{flatBoxPreview (α := α) b 3}"
  let fill (nid : Nat) (_ : Node) : String :=
    let colour :=
      match boxes[nid]?.join, affine[nid]?.join with
      | none, none => "#fff3cc"
      | some _, none => "#ddffea"
      | none, some _ => "#dde9ff"
      | some _, some _ => "#d7fff6"
    s!"style=filled, fillcolor=\"{colour}\""
  g.toDot (label := label) (attrs := fill)

/-- A table with one row per node id, up to `maxNodes`: the operation, the output shapes, whether a
box and affine bounds exist, and a three-entry box preview. -/
private def compactTableHtml {α : Type} [Storage α] [Context α] [ToString α]
    (g : Graph) (boxes : Array (Option (FlatBox α))) (affine : Array (Option (FlatAffineBounds α)))
    (maxNodes : Nat := 200) : ProofWidgets.Html :=
  let n := max g.nodes.size (max boxes.size affine.size)
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
            "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "out"}</th>
          <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
            "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "IBP/Aff"}</th>
          <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
            "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "preview"}</th>
        </tr>
      </thead>
      <tbody>
        {... ids.toArray.map (fun nid =>
          let n? := g.nodes[nid]?
          let box? := boxes[nid]?.join
          let aff? := affine[nid]?.join
          let op := match n? with | none => "<missing>" | some n => n.tag
          let out := match n? with | none => "?" | some n => n.outsPretty
          let ibp := match box? with | none => "IBP:none" | some _ => "IBP:some"
          let aff := match aff? with | none => "Aff:none" | some _ => "Aff:some"
          let preview := match box? with | none => "" | some b => flatBoxPreview (α := α) b 3;
          <tr>
            <td style={json% {"padding": "6px 8px", "border-bottom":
              "1px solid rgba(127,127,127,0.18)"}}>{monospace (toString nid)}</td>
            <td style={json% {"padding": "6px 8px", "border-bottom":
              "1px solid rgba(127,127,127,0.18)"}}>{monospace op}</td>
            <td style={json% {"padding": "6px 8px", "border-bottom":
              "1px solid rgba(127,127,127,0.18)"}}>{monospace out}</td>
            <td style={json% {"padding": "6px 8px", "border-bottom":
              "1px solid rgba(127,127,127,0.18)"}}>{monospace (ibp ++ " " ++ aff)}</td>
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

/-- Render the boxes of `TorchLean.CROWN.Op.runIBP` and the affine bounds of
`TorchLean.CROWN.Op.runCROWN` over a graph of registered operators as an HTML panel. It holds
the array sizes, a summary table, the DOT source (cut at 6000 characters) and an entry for each of
the first `maxNodes` node ids. -/
def crownPropHtml {α : Type} [Storage α] [Context α] [ToString α] (g : Graph)
    (boxes : Array (Option (FlatBox α))) (affine : Array (Option (FlatAffineBounds α)) := #[])
    (maxNodes : Nat := 200) : ProofWidgets.Html :=
  let nG := g.nodes.size
  let n := max nG (max boxes.size affine.size)
  let clipped : Bool := decide (n > maxNodes)
  let ids := List.range (min n maxNodes)
  let inDims := (affine.filterMap id).map (·.inDim)
  let dot := crownDot (α := α) g boxes affine
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
      {pill "CROWN / IBP state"} {pill s!"graphNodes={nG}"} {pill s!"boxes={boxes.size}"}
      {pill s!"affine={affine.size}"}
      {match inDims[0]? with
        | none => ProofWidgets.Html.text ""
        | some d => pill s!"inputDim={d}"}
      {if boxes.size = nG then okBadge "sizes match" else warnBadge "size mismatch"}
      {if clipped then warnBadge s!"clipped to {maxNodes}" else ProofWidgets.Html.text ""}
    </div>
    <div style={json% {"opacity": 0.85, "margin-bottom": "10px"}}>
      {.text ("This viewer displays optional IBP boxes and affine bounds per graph node id. " ++
          "Their presence is diagnostic state, not a checked certificate.")}
    </div>
    <details «open»={false} style={json% {"margin-bottom": "10px"}}>
      <summary>{.text "Compact summary (table)"}</summary>
      <div style={json% {"margin-top": "8px"}}>
        {compactTableHtml (α := α) g boxes affine (maxNodes := maxNodes)}
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
          nodeStateHtml (α := α) nid g.nodes[nid]? boxes[nid]?.join affine[nid]?.join)}
        {if clipped then
          <div style={json% {"margin-top": "8px", "opacity": 0.7}}>{.text "... (more nodes)"}</div>
         else
          ProofWidgets.Html.text ""}
      </div>
    </details>
  </div>

/-!
## Bounds Tightness

The interval widths $\mathrm{hi}-\mathrm{lo}$ show node by node where IBP bounds blow up. The
tightness panel lists the largest and the mean width of each node's box, with the width vector in a
collapsed panel, and flags the nodes without a box.
-/

/-- `(min, max, mean)` of an array, or `none` when it is empty. -/
private def arrayStats {α : Type} [Storage α] [Context α] (xs : Array α) :
    Option (α × α × α) :=
  xs[0]?.map fun x =>
    let rest := xs.extract 1 xs.size
    let mn := rest.foldl (fun acc y => min acc y) x
    let mx := rest.foldl (fun acc y => max acc y) x
    let sum := rest.foldl (fun acc y => acc + y) x
    let mean := sum / (↑xs.size : α)
    (mn, mx, mean)

/-- Coordinatewise width `hi - lo` of a box, which the tightness panel summarizes per node. -/
private def flatBoxWidthTensor {α : Type} [Storage α] [Context α] (b : FlatBox α) :
    Tensor α [b.dim] :=
  b.hi - b.lo

/-- Render the tightness panel of `boxes` over `g`: one row per node id up to `maxNodes` with the
box dimension and the largest and mean width $\mathrm{hi}-\mathrm{lo}$, or a flag when the node has
no box. -/
def boundsTightnessHtml {α : Type} [Storage α] [Context α] [ToString α]
    (g : Graph) (boxes : Array (Option (FlatBox α))) (maxNodes : Nat := 200) :
    ProofWidgets.Html :=
  let nG := g.nodes.size
  let nS := boxes.size
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
      {pill "IBP width diagnostic"} {pill s!"graphNodes={nG}"} {pill s!"boxes={nS}"}
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
              "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "out"}</th>
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
            let op := match n? with | none => "<missing>" | some n => n.tag
            let shape := match n? with | none => "?" | some n => n.outsPretty
            match boxes[nid]? with
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
            | some box? =>
                match box? with
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

/-- `#crown_view g, boxes` shows `crownPropHtml g boxes` in the infoview: the boxes `boxes`
over `g`, with no affine bounds. -/
-- The `, ` delimiter keeps the first term from parsing as an application of the second.
macro "#crown_view " g:term ", " boxes:term : command =>
  UI.canonicalCommand <$> `(#html (crownPropHtml $g $boxes))

/-- `#crown_view g, boxes, affine` shows `crownPropHtml g boxes affine` in the infoview:
the boxes `boxes` and the affine bounds `affine` over `g`. -/
macro "#crown_view " g:term ", " boxes:term ", " affine:term : command =>
  UI.canonicalCommand <$> `(#html (crownPropHtml $g $boxes $affine))

/-- `#bounds_tightness_view g, boxes` shows `boundsTightnessHtml g boxes` in the infoview:
the widths of the boxes `boxes` over `g`. -/
macro "#bounds_tightness_view " g:term ", " boxes:term : command =>
  UI.canonicalCommand <$> `(#html (boundsTightnessHtml $g $boxes))

end TorchLean.Widget
