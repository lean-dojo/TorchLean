/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import NN.Runtime.Context
public meta import NN.Spec.Core.Tensor
public meta import NN.Spec.Core.Utils
public meta import NN.Widgets.Core.UI
public meta import ProofWidgets.Component.HtmlDisplay
public meta import ProofWidgets.Demos.Macro

/-!
# Tensor

Tensor inspection widgets for the Lean infoview.

This module defines a `#tensor_view t` command that displays a small tensor as a rich HTML panel in
the infoview. It is designed for:
- examples,
- inspecting runtime output,
- teaching/exposition in the manual.

It is **not** intended to be used inside proofs, and it is kept out of TorchLean’s default build
surface (you must explicitly import `NN.Entrypoint.Widgets` or a concrete widget module such as
`NN.Widgets.Core.Tensor`).

Implementation note:
We build on ProofWidgets’ `#html` command (which ships with mathlib’s dependency set) rather than
introducing any custom JavaScript or external build step.

## Main definitions

- `tensorHtml`: shape-aware renderer for typed tensors.
- `anyTensorHtml`: same renderer for runtime shape-erased tensors.
- `tensorStatsHtml`: compact scalar summary (min/max/mean/norms).
- `#tensor_view`, `#anytensor_view`, `#tensor_stats_view`: command entry points.

## Implementation notes

- Table rendering for small tensors plus clipped previews for large ones balances
  readability/performance tradeoff in infoview.
- Element rendering is class-based (`TensorElemView`) so backends can customize display without
  forking widget logic.
- We cap recursion depth for higher-rank tensors to avoid overwhelming nested expansions.

## References

- [ProofWidgets](https://github.com/leanprover-community/ProofWidgets4)
- [Lean community documentation style](https://leanprover-community.github.io/contribute/doc.html)

## Tags

tensor, visualization, stats, proofwidgets, inspection
-/

public meta section

open scoped ProofWidgets.Jsx

namespace NN.Widgets

open _root_.Spec
open _root_.Spec.Tensor
open UI

/--
Element renderer for `#tensor_view`.

The tensor widget is used across the library, so we keep element rendering customizable:
- the default instance uses `ToString`,
- specialized instances can add tooltips (e.g. float32 bits), units, or compact formatting.
-/
class TensorElemView (α : Type) where
  render : α → ProofWidgets.Html

/-- Default element renderer for `#tensor_view`, using `ToString`. -/
instance {α : Type} [ToString α] : TensorElemView α :=
  ⟨fun x => monospace (toString x)⟩

namespace TensorInternal

/-- Join strings with a separator. -/
def join (sep : String) (xs : List String) : String :=
  String.intercalate sep xs

/-- Render shape dimensions as a bracketed list string. -/
def dimsString (s : Shape) : String :=
  match Shape.toList s with
  | [] => "[]"
  | ds => "[" ++ join ", " (ds.map toString) ++ "]"

def fmtMaybe (x : Option String) : ProofWidgets.Html :=
  match x with
  | none => <span style={json% {"opacity": 0.7}}>(none)</span>
  | some s => monospace s

/-- Render a 1D tensor as a clipped single-row table. -/
def renderVector {α : Type} [TensorElemView α] (maxCols : Nat) {n : Nat}
    (t : Tensor α (.dim n .scalar)) : ProofWidgets.Html :=
  let xs := (toList (s := .dim n .scalar) t).take maxCols;
  let clipped : Bool := decide (n > maxCols);
  <div>
    <div style={json% {"margin-bottom": "6px"}}>
      {pill s!"vector len={n}"} {pill s!"showing={xs.length}"} {pill s!"clipped={clipped}"}
    </div>
    <div style={json% {"overflow-x": "auto"}}>
      <table style={json% {"border-collapse": "collapse"}}>
        <tbody>
          <tr>
            {... xs.toArray.map (fun x =>
              <td style={json% {"border": "1px solid #ddd", "padding": "4px 6px", "text-align":
                "right"}}>
                {TensorElemView.render x}
              </td>)}
            {if clipped then
              <td style={json% {"border": "1px solid #ddd", "padding": "4px 6px", "opacity": 0.7}}>
                ...
              </td>
             else
              ProofWidgets.Html.text ""}
          </tr>
        </tbody>
      </table>
    </div>
  </div>

/-- Render a 2D tensor as a clipped grid table. -/
def renderMatrix {α : Type} [TensorElemView α] (maxRows maxCols : Nat) {n m : Nat}
    (t : Tensor α (.dim n (.dim m .scalar))) : ProofWidgets.Html :=
  let rows :=
    (List.finRange n).take maxRows |>.map (fun i =>
      let row : Tensor α (.dim m .scalar) := getAtSpec t i
      (toList (s := .dim m .scalar) row).take maxCols);
  let clippedRows : Bool := decide (n > maxRows);
  let clippedCols : Bool := decide (m > maxCols);
  <div>
    <div style={json% {"margin-bottom": "6px"}}>
      {pill s!"matrix {n}×{m}"} {pill s!"rows={rows.length}"} {pill s!"clippedRows={clippedRows}"}
        {pill s!"clippedCols={clippedCols}"}
    </div>
    <div style={json% {"overflow": "auto", "max-height": "420px"}}>
      <table style={json% {"border-collapse": "collapse"}}>
        <tbody>
          {... rows.toArray.map (fun row =>
            <tr>
              {... row.toArray.map (fun x =>
                <td style={json% {"border": "1px solid #ddd", "padding": "4px 6px", "text-align":
                  "right"}}>
                  {TensorElemView.render x}
                </td>)}
              {if clippedCols then
                <td style={json% {"border": "1px solid #ddd", "padding": "4px 6px", "opacity":
                  0.7}}>
                  ...
                </td>
               else
                ProofWidgets.Html.text ""}
            </tr>)}
        </tbody>
      </table>
      {if clippedRows then
        <div style={json% {"padding": "6px", "opacity": 0.7}}>
          ... (more rows)
        </div>
       else
        ProofWidgets.Html.text ""}
    </div>
  </div>

def renderFlatPreview {α : Type} [ToString α] (maxElems : Nat) {s : Shape}
    (t : Tensor α s) : ProofWidgets.Html :=
  let xs := (toList (s := s) t);
  -- `prefix` is a Lean keyword (notation command), so we avoid it as a local name.
  let head := xs.take maxElems;
  let clipped : Bool := decide (xs.length > maxElems);
  let preview :=
    if head.isEmpty then
      "(empty?)"
    else
      "[" ++ join ", " (head.map toString) ++ (if clipped then ", ..." else "") ++ "]";
  <details style={json% {"margin-top": "10px"}}>
    <summary>{.text s!"Flat preview (first {maxElems})"}</summary>
    <div style={json% {"margin-top": "6px"}}>
      {monospace preview}
    </div>
  </details>

end TensorInternal

/--
Render a tensor as HTML.

For small vectors/matrices, we render an actual table; otherwise we show a compact pretty string
plus a flat preview.
-/
def tensorHtml {α : Type} [ToString α] {s : Shape} (t : Tensor α s)
    (maxRows : Nat := 16) (maxCols : Nat := 16) (maxElems : Nat := 64) : ProofWidgets.Html :=
  -- Core renderer: only depends on a depth budget, so it can recurse on higher-rank tensors without
  -- dumping an enormous nested pretty-printer view by default.
  let rec tensorHtmlRec {s : Shape} (t : Tensor α s)
      (depth : Nat) : ProofWidgets.Html :=
    match depth with
    | 0 =>
        <div>
          <details «open»={false}>
            <summary>{.text "Pretty (nested)"}</summary>
            <pre style={json% {"white-space": "pre-wrap", "margin-top": "6px"}}>
              {.text (Spec.pretty (α := α) (s := s) t)}
            </pre>
          </details>
          {TensorInternal.renderFlatPreview (α := α) (s := s) maxElems t}
        </div>
    | depth + 1 =>
        match s, t with
        | .scalar, Tensor.scalar x =>
            <div>
              {pill "scalar"} {TensorElemView.render x}
            </div>
        | .dim n .scalar, Tensor.dim f =>
            TensorInternal.renderVector (α := α) (n := n) maxCols (t := Tensor.dim f)
        | .dim n (.dim m .scalar), Tensor.dim f =>
            TensorInternal.renderMatrix (α := α) (n := n) (m := m) maxRows maxCols (t := Tensor.dim
              f)
        | .dim n s', Tensor.dim f =>
            -- Higher rank: show a few slices along the outer dimension, recursively.
            let maxSlices : Nat := 6;
            let idxs := (List.finRange n).take maxSlices;
            let clipped : Bool := decide (n > maxSlices);
            <div>
              <details «open»={true}>
                <summary>
                      {pill s!"slices(axis0)={idxs.length}"} {pill s!"clipped={clipped}"} {pill
                        s!"sliceShape={TensorInternal.dimsString s'}"}
                    </summary>
                <div style={json% {"margin-top": "8px"}}>
                  {... idxs.toArray.map (fun i =>
                    let slice : Tensor α s' := getAtSpec (Tensor.dim f) i;
                    <details style={json% {"margin": "8px 0"}}>
                      <summary>{.text s!"[{i.1}]"}</summary>
                      <div style={json% {"margin-top": "6px", "padding-left": "8px"}}>
                        {tensorHtmlRec (s := s') slice depth}
                      </div>
                    </details>)}
                  {if clipped then
                    <div style={json% {"opacity": 0.7}}>{.text "... (more slices)"}</div>
                   else
                    ProofWidgets.Html.text ""}
                </div>
              </details>
              {TensorInternal.renderFlatPreview (α := α) (s := .dim n s') maxElems (Tensor.dim f)}
            </div>
        | _, _ =>
            -- Should be unreachable (shape index mismatch), but keep a robust fallback.
            <div>
              <details «open»={false}>
                <summary>{.text "Pretty (nested)"}</summary>
                <pre style={json% {"white-space": "pre-wrap", "margin-top": "6px"}}>
                  {.text (Spec.pretty (α := α) (s := s) t)}
                </pre>
              </details>
              {TensorInternal.renderFlatPreview (α := α) (s := s) maxElems t}
            </div>

  let header :=
    <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap", "margin-bottom":
      "8px"}}>
      {pill s!"shape={TensorInternal.dimsString s}"} {pill s!"rank={Shape.rank s}"} {pill
        s!"size={Shape.size s}"}
    </div>;
  -- Default to `ToString` element rendering, but allow specialized renderers via
  -- `TensorElemView` instances (when the caller imports them).
  let body := tensorHtmlRec (s := s) (t := t) (depth := 2);
  <div style={json% {
    "padding": "10px",
    "border": "1px solid var(--vscode-panel-border, #e5e5e5)",
    "border-radius": "10px",
    "background": "var(--vscode-editor-background, transparent)",
    "color": "var(--vscode-editor-foreground, inherit)"
  }}>
    {header}
    {body}
  </div>

/-!
## Runtime Wrappers
-/

/-- Render a `Runtime.AnyTensor` (shape-erased runtime tensor) with the same UI as `tensorHtml`. -/
def anyTensorHtml {α : Type} [ToString α] (v : Runtime.AnyTensor α)
    (maxRows : Nat := 16) (maxCols : Nat := 16) (maxElems : Nat := 64) : ProofWidgets.Html :=
  tensorHtml (α := α) (s := v.s) v.t (maxRows := maxRows) (maxCols := maxCols) (maxElems :=
    maxElems)

/-!
## Stats

For small tensors, it is often helpful to quickly sanity-check numeric ranges without expanding
every element. This widget computes simple scalar summaries (min/max/mean/norms).

Main command:
- `#tensor_stats_view t`
-/

namespace TensorInternal

def abs' {α : Type} [Context α] (x : α) : α :=
  MathFunctions.abs (α := α) x

def sqrt' {α : Type} [Context α] (x : α) : α :=
  MathFunctions.sqrt (α := α) x

def tensorStatsHtml {α : Type} [Context α] [ToString α] {s : Shape} (t : Tensor α s) :
  ProofWidgets.Html :=
  let xs : List α := Spec.toList (α := α) (s := s) t
  match xs with
  | [] =>
      <div style={json% {"padding": "10px"}}>
        {pill "Tensor stats"} {pill "empty tensor"} {pill s!"shape={dimsString s}"}
      </div>
  | x :: rest =>
      let n : Nat := (x :: rest).length
      let mn := rest.foldl (fun acc y => min acc y) x
      let mx := rest.foldl (fun acc y => max acc y) x
      let sum := rest.foldl (fun acc y => acc + y) x
      let mean := sum / (↑n : α)
      let absmax := rest.foldl (fun acc y => max acc (abs' (α := α) y)) (abs' (α := α) x)
      let l1 := rest.foldl (fun acc y => acc + abs' (α := α) y) (abs' (α := α) x)
      let sqsum := rest.foldl (fun acc y => acc + (y * y)) (x * x)
      let l2 := sqrt' (α := α) sqsum
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
          {pill "Tensor stats"} {pill s!"shape={dimsString s}"} {pill s!"size={Shape.size s}"}
        </div>
        <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap"}}>
          {pill s!"min={toString mn}"}
          {pill s!"max={toString mx}"}
          {pill s!"mean={toString mean}"}
          {pill s!"absmax={toString absmax}"}
          {pill s!"l1={toString l1}"}
          {pill s!"l2={toString l2}"}
        </div>
      </div>

end TensorInternal

/-- Render simple scalar summary statistics (min/max/mean/norms) for a tensor as HTML. -/
def tensorStatsHtml {α : Type} [Context α] [ToString α] {s : Shape} (t : Tensor α s) :
  ProofWidgets.Html :=
  TensorInternal.tensorStatsHtml (α := α) (s := s) t

/-!
## Commands
-/

syntax (name := tensorViewCmd) "#tensor_view " term : command
syntax (name := anyTensorViewCmd) "#anytensor_view " term : command
syntax (name := tensorStatsViewCmd) "#tensor_stats_view " term : command

macro "#tensor_view " t:term : command =>
  -- Ensure the widget is attached to a canonical syntax node.
  Lean.TSyntax.mkInfoCanonical <$> `(#html (tensorHtml $t))

macro "#anytensor_view " v:term : command =>
  Lean.TSyntax.mkInfoCanonical <$> `(#html (anyTensorHtml $v))

macro "#tensor_stats_view " t:term : command =>
  Lean.TSyntax.mkInfoCanonical <$> `(#html (tensorStatsHtml $t))

end NN.Widgets
