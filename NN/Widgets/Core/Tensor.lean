/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import NN.Spec.Core.Tensor.SomeTensor
public meta import NN.Tensor.Conversion
import Mathlib.Tactic.Bound.Init
public import NN.Spec.Core.Tensor.Core
public meta import NN.Widgets.Core.UI
public meta import ProofWidgets.Component.HtmlDisplay
public meta import ProofWidgets.Demos.Macro
public meta import NN.Spec.Core.Tensor -- shake: keep

/-!
# Tensor

Tensor inspection widgets for the Lean infoview.

This module defines a `#tensor_view t` command that displays a small tensor as a rich HTML panel in
the infoview. It is designed for:
- examples,
- inspecting runtime output,
- teaching/exposition in the manual.

It is **not** intended to be used inside proofs, and it is kept out of TorchLean’s default build
surface (you must explicitly import `NN.Widgets` or a concrete widget module such as
`NN.Widgets.Core.Tensor`).

Implementation note:
We build on ProofWidgets’ `#html` command (which ships with mathlib’s dependency set) rather than
introducing any custom JavaScript or external build step.

## Main definitions

- `tensorHtml`: shape-aware renderer for typed tensors.
- `packedTensorHtml`: the same renderer for shape-erased tensors.
- `tensorStatsHtml`: compact scalar summary (min/max/mean/norms).
- `#tensor_view`, `#anytensor_view`, `#tensor_stats_view`: command entry points.
-/

public meta section

open scoped ProofWidgets.Jsx

namespace NN.Widgets

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
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

/-- Read only the requested prefix, without converting the full backing buffer. -/
def prefixEntries {α : Type} [Storage α] {s : Shape}
    (tensor : Tensor α s) (limit : Nat) : Array α :=
  Array.ofFn (fun i : Fin (min (Spec.Shape.size s) limit) =>
    tensor.getFlat ⟨i.val, by
      simpa only [Spec.Shape.size_eq_prod, TorchLean.Tensor.Internal.Shape.size_eq_prod] using
        Nat.lt_of_lt_of_le i.isLt (Nat.min_le_left (Spec.Shape.size s) limit)⟩)

/-- Render a 1D tensor as a clipped single-row table. -/
def renderVector {α : Type} [TorchLean.Storage α] [TensorElemView α] (maxCols : Nat) {n : Nat}
    (t : Tensor α [n]) : ProofWidgets.Html :=
  let xs := prefixEntries t maxCols;
  let clipped : Bool := decide (n > maxCols);
  <div>
    <div style={json% {"margin-bottom": "6px"}}>
      {pill s!"rank one, size={n}"} {pill s!"showing={xs.size}"} {pill s!"clipped={clipped}"}
    </div>
    <div style={json% {"overflow-x": "auto"}}>
      <table style={json% {"border-collapse": "collapse"}}>
        <tbody>
          <tr>
            {... xs.map (fun x =>
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
def renderMatrix {α : Type} [TorchLean.Storage α] [TensorElemView α]
    (maxRows maxCols : Nat) {n m : Nat}
    (t : Tensor α [n, m]) : ProofWidgets.Html :=
  let rows :=
    (Array.finRange (min n maxRows)).map (fun i =>
      (⟨i.val, Nat.lt_of_lt_of_le i.isLt (Nat.min_le_left _ _)⟩ : Fin n)) |>.map (fun i =>
      let row : Tensor α [m] := get t i
      prefixEntries row maxCols);
  let clippedRows : Bool := decide (n > maxRows);
  let clippedCols : Bool := decide (m > maxCols);
  <div>
    <div style={json% {"margin-bottom": "6px"}}>
      {pill s!"matrix {n}×{m}"} {pill s!"rows={rows.size}"} {pill s!"clippedRows={clippedRows}"}
        {pill s!"clippedCols={clippedCols}"}
    </div>
    <div style={json% {"overflow": "auto", "max-height": "420px"}}>
      <table style={json% {"border-collapse": "collapse"}}>
        <tbody>
          {... rows.map (fun row =>
            <tr>
              {... row.map (fun x =>
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

/-- First `maxElems` entries of a tensor in row-major order, with a marker when there are more.

Truncating matters here: a widget that tried to print a full tensor would hang the editor on
anything of realistic size. -/
def renderFlatPreview {α : Type} [TorchLean.Storage α] [ToString α] (maxElems : Nat) {s : Shape}
    (t : Tensor α s) : ProofWidgets.Html :=
  let head := prefixEntries t maxElems;
  let clipped : Bool := decide (Spec.Shape.size s > maxElems);
  let preview :=
    if head.isEmpty then
      "[]"
    else
      "[" ++ String.intercalate ", " (head.toList.map toString) ++
        (if clipped then ", ..." else "") ++ "]";
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
def tensorHtml {α : Type} [TorchLean.Storage α] [ToString α] [TensorElemView α]
    {s : Shape} (t : Tensor α s)
    (maxRows : Nat := 16) (maxCols : Nat := 16) (maxElems : Nat := 64) : ProofWidgets.Html :=
  -- Core renderer: only depends on a depth budget, so it can recurse on higher-rank tensors without
  -- dumping an enormous nested pretty-printer view by default.
  let rec tensorHtmlRec {s : Shape} (t : Tensor α s)
      (depth : Nat) : ProofWidgets.Html :=
    match depth with
    | 0 => TensorInternal.renderFlatPreview (α := α) (s := s) maxElems t
    | depth + 1 =>
        match s with
        | .scalar =>
            <div>
              {pill "scalar"} {TensorElemView.render (Tensor.item t)}
            </div>
        | .dim n .scalar =>
            TensorInternal.renderVector (α := α) (n := n) maxCols t
        | .dim n (.dim m .scalar) =>
            TensorInternal.renderMatrix (α := α) (n := n) (m := m) maxRows maxCols t
        | .dim n s' =>
            -- Higher rank: show a few slices along the outer dimension, recursively.
            let maxSlices : Nat := 6;
            let idxs := (Array.finRange (min n maxSlices)).map (fun i =>
              (⟨i.val, Nat.lt_of_lt_of_le i.isLt (Nat.min_le_left _ _)⟩ : Fin n));
            let clipped : Bool := decide (n > maxSlices);
            <div>
              <details «open»={true}>
                <summary>
                      {pill s!"leading slices={idxs.size}"} {pill s!"clipped={clipped}"} {pill
                        s!"sliceShape={Shape.pretty s'}"}
                    </summary>
                <div style={json% {"margin-top": "8px"}}>
                  {... idxs.map (fun i =>
                    let slice : Tensor α s' := Tensor.unstack t i;
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
              {TensorInternal.renderFlatPreview (α := α) (s := .dim n s') maxElems t}
            </div>

  let header :=
    <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap", "margin-bottom":
      "8px"}}>
      {pill s!"shape={Shape.pretty s}"} {pill s!"rank={Spec.Shape.rank s}"} {pill
        s!"size={Spec.Shape.size s}"}
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

/-- Render a `Spec.SomeTensor` with the same UI as `tensorHtml`. -/
def packedTensorHtml {α : Type} [TorchLean.Storage α] [ToString α] [TensorElemView α]
    (v : Spec.SomeTensor α)
    (maxRows : Nat := 16) (maxCols : Nat := 16) (maxElems : Nat := 64) : ProofWidgets.Html :=
  tensorHtml (α := α) (s := v.shape) v.tensor (maxRows := maxRows) (maxCols := maxCols) (maxElems :=
    maxElems)

/-!
## Stats

For small tensors, it is often helpful to inspect numeric ranges without expanding
every element. This widget computes simple scalar summaries (min/max/mean/norms).

Main command:
- `#tensor_stats_view t`
-/

namespace TensorInternal

/-- Numeric summary panel: shape, size, extrema, mean, and the three usual norms. -/
def tensorStatsHtml {α : Type} [TorchLean.Storage α] [Context α] [ToString α] {s : Shape}
    (t : Tensor α s) : ProofWidgets.Html :=
  let xs : Array α := Tensor.to t (Array α)
  match xs[0]? with
  | none =>
      <div style={json% {"padding": "10px"}}>
        {pill "Tensor stats"} {pill "empty tensor"} {pill s!"shape={Shape.pretty s}"}
      </div>
  | some x =>
      let rest := xs.extract 1 xs.size
      let n : Nat := xs.size
      let mn := rest.foldl (fun acc y => min acc y) x
      let mx := rest.foldl (fun acc y => max acc y) x
      let sum := rest.foldl (fun acc y => acc + y) x
      let mean := sum / (↑n : α)
      let magnitude : α → α := MathFunctions.abs (α := α)
      let absmax := rest.foldl (fun acc y => max acc (magnitude y)) (magnitude x)
      let l1 := rest.foldl (fun acc y => acc + magnitude y) (magnitude x)
      let sqsum := rest.foldl (fun acc y => acc + (y * y)) (x * x)
      let l2 := MathFunctions.sqrt (α := α) sqsum
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
          {pill "Tensor stats"} {pill s!"shape={Shape.pretty s}"}
          {pill s!"size={Spec.Shape.size s}"}
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
def tensorStatsHtml {α : Type} [TorchLean.Storage α] [Context α] [ToString α] {s : Shape}
    (t : Tensor α s) : ProofWidgets.Html :=
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
  Lean.TSyntax.mkInfoCanonical <$> `(#html (packedTensorHtml $v))

macro "#tensor_stats_view " t:term : command =>
  Lean.TSyntax.mkInfoCanonical <$> `(#html (tensorStatsHtml $t))

end NN.Widgets
