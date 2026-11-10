/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import TorchLean.Tensor.SomeTensor
public meta import TorchLean.Tensor.Conversion
public import TorchLean.Tensor.Datatype
public meta import TorchLean.NN.Widget.UI
public import TorchLean.NN.Widget.UI
import ProofWidgets.Component.HtmlDisplay
public import Mathlib.Algebra.Order.Algebra
public import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# Tensor

Infoview widgets for inspecting tensor values, built on ProofWidgets' `#html` command. Import
`TorchLean.NN.Widget.Tensor` to use them. The previews read bounded prefixes and leading
slices, while the statistics panel scans the whole tensor.

## Main definitions

- `tensorHtml`: shape-aware renderer for typed tensors.
- `packedTensorHtml`: the same renderer for shape-erased tensors.
- `tensorStatsHtml`: compact scalar summary (min/max/mean/norms).
- `#tensor_view`, `#anytensor_view`, `#tensor_stats_view`: command entry points.
-/

open TorchLean.Shape (rank size size_eq_prod)

public meta section

open scoped ProofWidgets.Jsx
open FloatLib.Numerics (MathFunctions)

namespace TorchLean.Widget

open _root_.TorchLean.Tensor
open UI

/--
How `#tensor_view` renders a single entry. Instances for particular scalar types can add tooltips,
units or compact formatting.
-/
class TensorElemView (α : Type) where
  /-- HTML for one entry. -/
  render : α → ProofWidgets.Html

/-- Default entry renderer: `toString x` in monospace. -/
instance {α : Type} [ToString α] : TensorElemView α :=
  ⟨fun x => monospace (toString x)⟩

/-- Read only the requested prefix, without converting the full backing buffer. -/
def prefixEntries {α : Type} [Storage α] {s : Shape}
    (tensor : Tensor α s) (limit : Nat) : Array α :=
  Array.ofFn (fun i : Fin (min (Shape.size s) limit) =>
    tensor.getFlat ⟨i.val, by
      simpa only [Shape.size_eq_prod] using
        Nat.lt_of_lt_of_le i.isLt (Nat.min_le_left (Shape.size s) limit)⟩)

/-- A rank-one tensor as a single-row table of at most `maxCols` cells, with an ellipsis cell when
the vector is longer. -/
def renderVector {α : Type} [Storage α] [TensorElemView α] (maxCols : Nat) {n : Nat}
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

/-- A matrix as a table of at most `maxRows` by `maxCols` cells, marking clipped rows and
columns. -/
def renderMatrix {α : Type} [Storage α] [TensorElemView α]
    (maxRows maxCols : Nat) {n m : Nat}
    (t : Tensor α [n, m]) : ProofWidgets.Html :=
  let rows :=
    (Array.finRange (min n maxRows)).map (fun i =>
      (⟨i.val, Nat.lt_of_lt_of_le i.isLt (Nat.min_le_left _ _)⟩ : Fin n)) |>.map (fun i =>
      let row : Tensor α [m] := Tensor.unstack t i
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
def renderFlatPreview {α : Type} [Storage α] [ToString α] (maxElems : Nat) {s : Shape}
    (t : Tensor α s) : ProofWidgets.Html :=
  let head := prefixEntries t maxElems;
  let clipped : Bool := decide (Shape.size s > maxElems);
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

/--
Render a tensor as an HTML panel headed by its shape, rank and size.

A scalar shows its value, vectors and matrices become clipped tables, and a tensor of higher rank
shows up to six leading slices, each rendered the same way, followed by a flat preview of its
first `maxElems` entries. Slicing stops after two levels, and deeper slices appear as flat previews.
-/
def tensorHtml {α : Type} [Storage α] [ToString α] [TensorElemView α]
    {s : Shape} (t : Tensor α s)
    (maxRows : Nat := 16) (maxCols : Nat := 16) (maxElems : Nat := 64) : ProofWidgets.Html :=
  -- Recurse on the depth budget so both slice count and nesting stay bounded.
  let rec go {s : Shape} (t : Tensor α s)
      (depth : Nat) : ProofWidgets.Html :=
    match depth with
    | 0 => renderFlatPreview (α := α) (s := s) maxElems t
    | depth + 1 =>
        match s with
        | .scalar =>
            <div>
              {pill "scalar"} {TensorElemView.render (Tensor.item t)}
            </div>
        | .dim n .scalar =>
            renderVector (α := α) (n := n) maxCols t
        | .dim n (.dim m .scalar) =>
            renderMatrix (α := α) (n := n) (m := m) maxRows maxCols t
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
                        s!"sliceShape={s'}"}
                    </summary>
                <div style={json% {"margin-top": "8px"}}>
                  {... idxs.map (fun i =>
                    let slice : Tensor α s' := Tensor.unstack t i;
                    <details style={json% {"margin": "8px 0"}}>
                      <summary>{.text s!"[{i.1}]"}</summary>
                      <div style={json% {"margin-top": "6px", "padding-left": "8px"}}>
                        {go (s := s') slice depth}
                      </div>
                    </details>)}
                  {if clipped then
                    <div style={json% {"opacity": 0.7}}>{.text "... (more slices)"}</div>
                   else
                    ProofWidgets.Html.text ""}
                </div>
              </details>
              {renderFlatPreview (α := α) (s := .dim n s') maxElems t}
            </div>

  let header :=
    <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap", "margin-bottom":
      "8px"}}>
      {pill s!"shape={s}"} {pill s!"rank={Shape.rank s}"} {pill
        s!"size={Shape.size s}"}
    </div>;
  let body := go (s := s) (t := t) (depth := 2);
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

/-- Render a `SomeTensor` with the same UI as `tensorHtml`. -/
def packedTensorHtml {α : Type} [Storage α] [ToString α] [TensorElemView α]
    (v : SomeTensor α)
    (maxRows : Nat := 16) (maxCols : Nat := 16) (maxElems : Nat := 64) : ProofWidgets.Html :=
  tensorHtml (α := α) (s := v.shape) v.tensor (maxRows := maxRows) (maxCols := maxCols) (maxElems :=
    maxElems)

/-!
## Stats

`#tensor_stats_view t` materializes every entry of `t` and computes the extrema, mean and norms
with the scalar's `Context` operations.
-/

/-- Summary panel with the shape, size, min, max, mean, absmax, l1 and l2 norms of a tensor, or
an "empty tensor" badge when it has no entries. -/
def tensorStatsHtml {α : Type} [Storage α] [Context α] [ToString α] {s : Shape}
    (t : Tensor α s) : ProofWidgets.Html :=
  let xs : Array α := Tensor.to t (Array α)
  match xs[0]? with
  | none =>
      <div style={json% {"padding": "10px"}}>
        {pill "Tensor stats"} {pill "empty tensor"} {pill s!"shape={s}"}
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
          {pill "Tensor stats"} {pill s!"shape={s}"}
          {pill s!"size={Shape.size s}"}
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

/-!
## Commands
-/

/-- `#tensor_view t` shows the `tensorHtml` panel of the tensor `t`. -/
macro "#tensor_view " t:term : command =>
  -- Mark the generated `#html` command canonical so the infoview shows the panel at this command.
  UI.canonicalCommand <$> `(#html (tensorHtml $t))

/-- `#anytensor_view v` shows the `packedTensorHtml` panel of the `SomeTensor` `v`. -/
macro "#anytensor_view " v:term : command =>
  UI.canonicalCommand <$> `(#html (packedTensorHtml $v))

/-- `#tensor_stats_view t` shows the `tensorStatsHtml` summary of the tensor `t`. -/
macro "#tensor_stats_view " t:term : command =>
  UI.canonicalCommand <$> `(#html (tensorStatsHtml $t))

end TorchLean.Widget
