/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Training.Log
public meta import NN.Widgets.Core.Tensor
public meta import NN.Widgets.Core.UI
public meta import ProofWidgets.Component.HtmlDisplay
public meta import ProofWidgets.Demos.Macro

/-!
# Training

Training/testing loop visualizations (logs, curves, and small reports).

TorchLean’s core runtime and specs are purely mathematical; "training loops" are just repeated
application of an update rule. In practice, the first thing you want when debugging training is:

- a loss curve (did it decrease? did it blow up?),
- a few scalar metrics (accuracy, learning rate, gradient norm),
- a compact “last N steps” table.

This module provides a *pure* log viewer (no JS): inline SVG sparklines + HTML tables.

Main command:
- `#train_log_view log` renders a `TrainLog`.

Optional testing command:
- `#confusion_view labels, cm` renders a confusion matrix for classification eval.

## Main definitions

- `trainLogHtml`: render scalar metric series and recent-step tables.
- `confusionHtml`: render confusion matrix + per-class precision/recall.
- `#train_log_view` / `#train_log_file_view`: in-memory and file-backed entry points.
- `#confusion_view`: quick classifier diagnostics.

## Implementation notes

- Inline SVG sparklines are lightweight and render quickly in the infoview.
- We intentionally avoid custom JavaScript so widget files stay easy to import and review.
- For saved logs, we render an error panel instead of throwing hard failures; this keeps viewer
  files robust even when artifacts are missing.

## References

- [ProofWidgets](https://github.com/leanprover-community/ProofWidgets4)
- [scikit-learn confusion matrix terminology](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
- [Lean community documentation style](https://leanprover-community.github.io/contribute/doc.html)

## Tags

training, logs, metrics, confusion-matrix, proofwidgets
-/

namespace NN.Widgets

public meta section

open scoped ProofWidgets.Jsx

open _root_.Runtime.Training
open UI

/-!
## Curves
-/

/-- Compute `(min, max)` for a nonempty float array. -/
private def arrayMinMax (xs : Array Float) : Option (Float × Float) :=
  if xs.size = 0 then
    none
  else
    let init := (xs[0]!, xs[0]!)
    let (lo, hi) :=
      (Array.range xs.size).foldl (fun (acc : Float × Float) i =>
        let x := xs[i]!
        -- Use `<=` / `>=` to avoid the JSX parser confusing `< ident` with a tag start.
        let lo := if x <= acc.1 then x else acc.1
        let hi := if x >= acc.2 then x else acc.2
        (lo, hi)) init
    some (lo, hi)

private def floatClamp01 (x : Float) : Float :=
  if x <= 0.0 then 0.0 else if x >= 1.0 then 1.0 else x

/-- Convert a float series into SVG polyline points. -/
private def sparklinePoints (w h : Nat) (xs : Array Float) : String :=
  let wF : Float := Float.ofNat (max 1 (w - 1))
  let hF : Float := Float.ofNat (max 1 (h - 1))
  match arrayMinMax xs with
  | none => ""
  | some (lo, hi) =>
      let denom0 := hi - lo
      -- Avoid division by a near-zero range (and avoid needing `DecidableEq Float`).
      let tiny : Float := (1e-30 : Float)
      let denom := if Float.abs denom0 <= tiny then 1.0 else denom0
      let n := xs.size
      let pts : List String :=
        (List.range n).map (fun i =>
          let x := xs[i]!
          let t : Float := if n ≤ 1 then 0.0 else Float.ofNat i / Float.ofNat (n - 1)
          let y01 := floatClamp01 ((x - lo) / denom)
          let xf := t * wF
          let yf := (1.0 - y01) * hF
          -- SVG coordinates are floats; keep strings short.
          s!"{xf},{yf}")
      String.intercalate " " pts

/-- Render a small inline SVG sparkline for a metric series. -/
private def sparklineSvg (xs : Array Float) (stroke : String) (w : Nat := 240) (h : Nat := 52) :
  ProofWidgets.Html :=
  let pts := sparklinePoints w h xs;
  <svg
    width={toString w}
    height={toString h}
    viewBox={s!"0 0 {w} {h}"}
    style={json% {
      "display": "block",
      "border": "1px solid var(--vscode-panel-border, #e5e5e5)",
      "border-radius": "10px",
      "background": "var(--vscode-editor-background, transparent)"
    }}>
    <polyline
      fill="none"
      stroke={stroke}
      strokeWidth="2"
      points={pts} />
  </svg>

private def last? (xs : Array Float) : Option Float :=
  if xs.size = 0 then none else some xs[xs.size - 1]!

private def seriesMinMax (xs : Array Float) : Option (Float × Float) :=
  arrayMinMax xs

private def seriesStartEnd (xs : Array Float) : Option (Float × Float) :=
  if xs.size = 0 then none else some (xs[0]!, xs[xs.size - 1]!)

private def fmt2 (x : Float) : String :=
  -- Keep it short; we're in a UI badge.
  let s := toString x
  if s.length <= 10 then s else (s.take 10).toString

/-- Render a `TrainLog` (metric series + recent steps) as an infoview HTML panel. -/
def trainLogHtml (log : _root_.Runtime.Training.TrainLog) (maxRows : Nat := 20) : ProofWidgets.Html
  :=
  let sCount := log.series.size
  if h : sCount = 0 then
    <div style={json% {"padding": "10px"}}>{warnBadge "no series"} {.text
      "TrainLog has no metric series."}</div>
  else
    let first : _root_.Runtime.Training.Series :=
      log.series[0]'(by
        -- `0 < log.series.size` since we are in the `else` branch (`sCount ≠ 0`).
        simpa [sCount] using Nat.pos_of_ne_zero h)
    let n : Nat :=
      -- Use the shortest series length as the effective length.
      log.series.foldl (fun acc s => min acc s.values.size) first.values.size
    let steps : Array Nat :=
      if log.steps.size = n then log.steps
      else (Array.range n).map (fun i => i)
    let rows : Array Nat := (Array.range n).reverse.take maxRows |>.reverse;
    <div style={json% {
      "padding": "10px",
      "border": "1px solid var(--vscode-panel-border, #e5e5e5)",
      "border-radius": "10px",
      "background": "var(--vscode-editor-background, transparent)",
      "color": "var(--vscode-editor-foreground, inherit)"
    }}>
      <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap", "margin-bottom":
        "10px"}}>
        {pill log.title} {pill s!"steps={n}"} {pill s!"series={sCount}"}
        {if log.steps.size = n || log.steps.size = 0 then ProofWidgets.Html.text "" else warnBadge
          "steps length mismatch"}
      </div>

      {if log.notes.isEmpty then ProofWidgets.Html.text "" else
        <details «open»={false} style={json% {"margin-bottom": "10px"}}>
          <summary>{.text "Notes"}</summary>
          <div style={json% {"margin-top": "8px", "display": "grid", "grid-template-columns": "1fr",
            "gap": "6px"}}>
            {... log.notes.map (fun s => <div>{monospace s}</div>)}
          </div>
        </details>}

      <div style={json% {"display": "grid", "grid-template-columns": "1fr", "gap": "10px"}}>
        {... log.series.map (fun s =>
          let lastS := match last? s.values with | none => "(none)" | some v => toString v;
          let se := seriesStartEnd s.values
          let mm := seriesMinMax s.values
          let startS := match se with | none => "?" | some p => fmt2 p.1
          let endS := match se with | none => "?" | some p => fmt2 p.2
          let minS := match mm with | none => "?" | some p => fmt2 p.1
          let maxS := match mm with | none => "?" | some p => fmt2 p.2
          let deltaS :=
            match se with
            | none => "?"
            | some p => fmt2 (p.2 - p.1);
          <div>
            <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap", "align-items":
              "center", "margin-bottom": "6px"}}>
              {pill s.name} {pill s!"n={s.values.size}"} {pill s!"start={startS}"} {pill
                s!"end={endS}"} {pill s!"Δ={deltaS}"} {pill s!"min={minS}"} {pill s!"max={maxS}"}
            </div>
            {sparklineSvg s.values s.color}
          </div>)}
      </div>

      <details «open»={false} style={json% {"margin-top": "10px"}}>
        <summary>{.text s!"Last {maxRows} steps (table)"}</summary>
        <div style={json% {"margin-top": "8px", "overflow": "auto", "max-height": "360px",
          "border": "1px solid var(--vscode-panel-border, #e5e5e5)", "border-radius": "10px"}}>
          <table style={json% {"border-collapse": "collapse", "width": "100%"}}>
            <thead>
              <tr>
                <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
                  "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "step"}</th>
                {... log.series.map (fun s =>
                  <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
                    "1px solid var(--vscode-panel-border, #e5e5e5)"}}>
                    {monospace s.name}
                  </th>)}
              </tr>
            </thead>
            <tbody>
              {... rows.map (fun i =>
                <tr>
                  <td style={json% {"padding": "6px 8px", "border-bottom":
                    "1px solid rgba(127,127,127,0.18)"}}>
                    {monospace (toString steps[i]!)}
                  </td>
                  {... log.series.map (fun s =>
                    let v := if i < s.values.size then toString s.values[i]! else "?";
                    <td style={json% {"padding": "6px 8px", "border-bottom":
                      "1px solid rgba(127,127,127,0.18)"}}>
                      {monospace v}
                    </td>)}
                </tr>)}
            </tbody>
          </table>
        </div>
      </details>
    </div>

/-!
## Confusion Matrix
-/

/-- Render a `ConfusionMatrix` (with optional label clipping) as an infoview HTML panel. -/
def confusionHtml (labels : Array String) (cm : _root_.Runtime.Training.ConfusionMatrix) (maxLabels
  : Nat := 40) : ProofWidgets.Html :=
  let n := min labels.size cm.counts.size
  let clipped := decide (labels.size > maxLabels) || decide (cm.counts.size > maxLabels)
  let ids := (Array.range (min n maxLabels));
  let totals : Array Nat :=
    ids.map (fun i =>
      let row := cm.counts[i]!
      (ids.foldl (fun acc j =>
        let v := if j < row.size then row[j]! else 0
        acc + v) 0))
  let colTotals : Array Nat :=
    ids.map (fun j =>
      ids.foldl (fun acc i =>
        let row := cm.counts[i]!
        let v := if j < row.size then row[j]! else 0
        acc + v) 0)
  let correct : Nat :=
    ids.foldl (fun acc i =>
      let row := cm.counts[i]!
      let v := if i < row.size then row[i]! else 0
      acc + v) 0
  let total : Nat := totals.foldl (· + ·) 0
  let accF : Float :=
    if total = 0 then 0.0 else (Float.ofNat correct) / (Float.ofNat total)
  let accPct : String := fmt2 (100.0 * accF);
  <div style={json% {
    "padding": "10px",
    "border": "1px solid var(--vscode-panel-border, #e5e5e5)",
    "border-radius": "10px",
    "background": "var(--vscode-editor-background, transparent)"
  }}>
    <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap", "margin-bottom":
      "10px"}}>
      {pill "Confusion matrix"} {pill s!"classes={n}"} {pill s!"acc={accPct}%"} {pill
        s!"correct={correct}"} {pill s!"total={total}"}
      {if clipped then warnBadge s!"clipped to {maxLabels}" else ProofWidgets.Html.text ""}
    </div>
    <div style={json% {"overflow": "auto", "max-height": "420px",
      "border": "1px solid var(--vscode-panel-border, #e5e5e5)", "border-radius": "10px"}}>
      <table style={json% {"border-collapse": "collapse"}}>
        <thead>
          <tr>
            <th style={json% {"position": "sticky", "left": "0", "background":
              "var(--vscode-editor-background, #fff)",
              "padding": "6px 8px", "border-bottom":
                "1px solid var(--vscode-panel-border, #e5e5e5)"}}>
              {.text "true\\pred"}
            </th>
            {... ids.map (fun j =>
              <th style={json% {"padding": "6px 8px", "border-bottom":
                "1px solid var(--vscode-panel-border, #e5e5e5)"}}>
                {monospace labels[j]!}
              </th>)}
          </tr>
        </thead>
        <tbody>
          {... ids.map (fun i =>
            <tr>
              <th style={json% {"position": "sticky", "left": "0", "background":
                "var(--vscode-editor-background, #fff)",
                "padding": "6px 8px", "border-bottom": "1px solid rgba(127,127,127,0.18)"}}>
                {monospace labels[i]!}
              </th>
              {... ids.map (fun j =>
                let row := cm.counts[i]!
                let v := if j < row.size then row[j]! else 0
                if i = j then
                  <td style={json% {"padding": "6px 8px", "border-bottom":
                    "1px solid rgba(127,127,127,0.18)", "background": "rgba(0, 200, 120, 0.14)"}}>
                    {monospace (toString v)}
                  </td>
                else
                  <td style={json% {"padding": "6px 8px", "border-bottom":
                    "1px solid rgba(127,127,127,0.18)"}}>
                    {monospace (toString v)}
                  </td>)}
            </tr>)}
        </tbody>
      </table>
    </div>
    <details «open»={false} style={json% {"margin-top": "10px"}}>
      <summary>{.text "Per-class precision/recall"}</summary>
      <div style={json% {"margin-top": "8px", "overflow": "auto", "border":
        "1px solid var(--vscode-panel-border, #e5e5e5)", "border-radius": "10px"}}>
        <table style={json% {"border-collapse": "collapse", "width": "100%"}}>
          <thead>
            <tr>
              <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
                "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "class"}</th>
              <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
                "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "support"}</th>
              <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
                "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "precision"}</th>
              <th style={json% {"text-align": "left", "padding": "6px 8px", "border-bottom":
                "1px solid var(--vscode-panel-border, #e5e5e5)"}}>{.text "recall"}</th>
            </tr>
          </thead>
          <tbody>
            {... ids.map (fun i =>
              let row := cm.counts[i]!;
              let tp : Nat := if i + 1 <= row.size then row[i]! else 0;
              let sup : Nat := totals[i]!;
              let pred : Nat := colTotals[i]!;
              let precF : Float := if pred = 0 then 0.0 else (Float.ofNat tp) / (Float.ofNat pred);
              let recF : Float := if sup = 0 then 0.0 else (Float.ofNat tp) / (Float.ofNat sup);
              <tr>
                <td style={json% {"padding": "6px 8px", "border-bottom":
                  "1px solid rgba(127,127,127,0.18)"}}>{monospace labels[i]!}</td>
                <td style={json% {"padding": "6px 8px", "border-bottom":
                  "1px solid rgba(127,127,127,0.18)"}}>{monospace (toString sup)}</td>
                <td style={json% {"padding": "6px 8px", "border-bottom":
                  "1px solid rgba(127,127,127,0.18)"}}>{monospace s!"{fmt2 (100.0 * precF)}%"}</td>
                <td style={json% {"padding": "6px 8px", "border-bottom":
                  "1px solid rgba(127,127,127,0.18)"}}>{monospace s!"{fmt2 (100.0 * recF)}%"}</td>
              </tr>)}
          </tbody>
        </table>
      </div>
    </details>
  </div>

/-!
## Commands
-/

/--
Render a `Runtime.Training.TrainLog` value directly in the infoview.

This is the in-memory (non-IO) variant. For executables that write JSON logs to disk, see
`#train_log_file_view`.
-/
syntax (name := trainLogViewCmd) "#train_log_view " term : command

macro "#train_log_view " log:term : command =>
  Lean.TSyntax.mkInfoCanonical <$> `(#html (trainLogHtml $log))

/-!
`TrainLog` is pure data, but many executables write logs to disk.

This command reads a saved JSON log (written by `Runtime.Training.TrainLog.writeJson`) and
renders it using the same viewer as `#train_log_view`.
-/
/--
Read a saved `Runtime.Training.TrainLog` JSON file and render it in the infoview.

The expected JSON schema is the one produced by `Runtime.Training.TrainLog.writeJson` and
TorchLean's executable training examples (for example PPO examples under `NN/Examples/Models/*`).

When the file is missing or malformed, this command renders an error panel instead of failing the
build, so widget-view files stay safe to import.
-/
syntax (name := trainLogFileViewCmd) "#train_log_file_view " term : command

macro "#train_log_file_view " path:term : command =>
  Lean.TSyntax.mkInfoCanonical <$> `(#html (do
    let p : System.FilePath := $path
    try
      let log ← _root_.Runtime.Training.TrainLog.readJson p
      pure (trainLogHtml log)
    catch e =>
      pure <|
        <div style={json% {"padding": "10px"}}>
          {warnBadge "train_log_file_view"}
          <div style={json% {"margin-top": "8px"}}>
            {.text "Could not read a TrainLog JSON file at: "}
            {monospace p.toString}
          </div>
          <div style={json% {"margin-top": "6px", "opacity": "0.9"}}>
            {.text "Tip: this file is usually produced by a TorchLean executable training run. "}
            {.text "Run the matching `lake exe ...` command (often with `-- --log <path>`), "}
            {.text "or pass an absolute path here."}
          </div>
          <div style={json% {"margin-top": "6px"}}>
            {monospace (toString e)}
          </div>
        </div>))

/--
Render a confusion matrix report in the infoview.

This is a small viewer for `Runtime.Training.ConfusionMatrix` plus an aligned array of class
labels.
-/
syntax (name := confusionViewCmd) "#confusion_view " term ", " term : command

macro "#confusion_view " labels:term ", " cm:term : command =>
  Lean.TSyntax.mkInfoCanonical <$> `(#html (confusionHtml $labels $cm))

end
end NN.Widgets
