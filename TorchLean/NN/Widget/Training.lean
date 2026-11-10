/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import TorchLean.NN.Training
public meta import TorchLean.NN.Widget.Tensor -- shake: keep
public import TorchLean.NN.Widget.UI
import TorchLean.NN.Training
import ProofWidgets.Component.HtmlDisplay
public import Mathlib.Algebra.Order.Algebra
public import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# Training

Infoview panels for training and evaluation runs. `#train_log_view log` renders a `TrainLog`
value, and `#train_log_file_view path` reads one from a JSON file first. The panel draws every
metric series of the log, such as the loss, accuracy, learning rate or gradient norm, as an inline
SVG sparkline. Badges give the first and last values of each series, their difference, and the
minimum and maximum, so a glance shows whether the loss fell or blew up. A collapsed table lists
the values of the last steps. For a classifier's evaluation, `#confusion_view labels, cm` renders
a confusion matrix with per-class precision and recall. The panels are HTML and SVG, with no
custom JavaScript.

## Main definitions

- `trainLogHtml`: render scalar metric series and recent-step tables.
- `confusionHtml`: render confusion matrix + per-class precision/recall.
- `#train_log_view` / `#train_log_file_view`: in-memory and file-backed entry points.
- `#confusion_view`: classifier diagnostics for a label/confusion-matrix pair.
-/

namespace TorchLean.Widget

public meta section

open scoped ProofWidgets.Jsx

open Training
open UI

/-!
## Curves
-/

/-- Range of the finite samples, so neither infinite values nor unreported metrics (which read
`NaN`) set the axes. -/
private def arrayMinMax (xs : Array Float) : Option (Float × Float) :=
  xs.foldl (fun bounds x =>
    if x.isNaN || x.isInf then bounds else
    match bounds with
    | none => some (x, x)
    | some (lo, hi) => some (min lo x, max hi x)) none

/-- Clamp to `[0, 1]`, which keeps a sparkline point inside its viewport. -/
private def floatClamp01 (x : Float) : Float :=
  if x <= 0.0 then 0.0 else if x >= 1.0 then 1.0 else x

/-- Build the SVG path data for a sparkline, with a gap at every unreported (`NaN`) or infinite
value. -/
private def sparklinePath (w h : Nat) (xs : Array Float) : String := Id.run do
  let wF := Float.ofNat (max 1 (w - 1))
  let hF := Float.ofNat (max 1 (h - 1))
  let some (lo, hi) := arrayMinMax xs | return ""
  -- Scale first so two finite endpoints cannot overflow when subtracted.
  let scale := max 1.0 (max lo.abs hi.abs)
  let range := hi / scale - lo / scale
  let mut connected := false
  let mut commands : Array String := #[]
  for i in [:xs.size] do
    let x := xs[i]!
    if x.isNaN || x.isInf then
      connected := false
    else
      let t := if xs.size ≤ 1 then 0.0 else Float.ofNat i / Float.ofNat (xs.size - 1)
      let y := if range == 0.0 then 0.0 else floatClamp01 ((x / scale - lo / scale) / range)
      let command := if connected then "L" else "M"
      commands := commands.push s!"{command}{t * wF},{(1.0 - y) * hF}"
      connected := true
  return String.intercalate " " commands.toList

/-- Render a small inline SVG sparkline for a metric series. -/
private def sparklineSvg (xs : Array Float) (stroke : String) (w : Nat := 240) (h : Nat := 52) :
  ProofWidgets.Html :=
  let pts := sparklinePath w h xs;
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
    <path
      fill="none"
      stroke={stroke}
      strokeWidth="2"
      d={pts} />
  </svg>

/-- First and last value of a metric series, which is how the panel reports improvement. -/
private def seriesStartEnd (xs : Array Float) : Option (Float × Float) :=
  if xs.size = 0 then none else some (xs[0]!, xs[xs.size - 1]!)

/-- Render a `TrainLog` as an infoview panel: its notes, a sparkline per metric series with badges
for the first and last values, their difference and the range, and a collapsed table of the last
`maxRows` steps. Warning badges flag a log with no series and arrays of unequal length, and
missing table entries show as `?`. -/
def trainLogHtml (log : Training.TrainLog) (maxRows : Nat := 20) : ProofWidgets.Html
  :=
  let sCount := log.series.size
  let n := log.series.foldl (fun acc series => max acc series.values.size) log.steps.size
  let steps := (Array.range n).map (fun i => log.steps[i]?.getD i)
  let rows := (Array.range (min n maxRows)).map (fun i => n - min n maxRows + i);
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
      {if sCount == 0 then warnBadge "no metric series" else ProofWidgets.Html.text ""}
      {if log.steps.size = n || log.steps.size = 0 then ProofWidgets.Html.text "" else warnBadge
        "steps length mismatch"}
      {if log.series.any (fun series => series.values.size != n) then
        warnBadge "series length mismatch; missing values shown as ?"
        else ProofWidgets.Html.text ""}
    </div>

    {if log.notes.isEmpty then ProofWidgets.Html.text "" else
      <details «open»={false} style={json% {"margin-bottom": "10px"}}>
        <summary>{.text "Notes"}</summary>
        <div style={json% {"margin-top": "8px", "display": "grid", "grid-template-columns": "1fr",
          "gap": "6px"}}>
          {... log.notes.map (fun s =>
            <pre style={json% {"white-space": "pre-wrap", "overflow-wrap": "anywhere",
              "margin": "0"}}>{.text s}</pre>)}
        </div>
      </details>}

    <div style={json% {"display": "grid", "grid-template-columns": "1fr", "gap": "10px"}}>
      {... log.series.map (fun s =>
        let se := seriesStartEnd s.values
        let mm := arrayMinMax s.values
        let startS := match se with | none => "?" | some p => toString p.1
        let endS := match se with | none => "?" | some p => toString p.2
        let minS := match mm with | none => "?" | some p => toString p.1
        let maxS := match mm with | none => "?" | some p => toString p.2
        let deltaS :=
          match se with
          | none => "?"
          | some p => toString (p.2 - p.1);
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

/-- Render a `ConfusionMatrix` as an infoview panel: the accuracy, the matrix for the first
`maxLabels` classes with true classes as rows and the diagonal shaded, and a collapsed table of
per-class support, precision and recall. A matrix that is not square, or whose class count differs
from the number of labels, gives an error badge. -/
def confusionHtml (labels : Array String) (cm : Training.ConfusionMatrix) (maxLabels
  : Nat := 40) : ProofWidgets.Html :=
  let n := cm.counts.size
  let result := do
    unless labels.size == n do
      throw "confusion matrix labels must match its class count"
    cm.classStatistics
  match result with
  | .error message => <div>{errBadge message}</div>
  | .ok stats => Id.run do
    let clipped := n > maxLabels
    let ids := Array.range (min n maxLabels)
    let totals := stats.support
    let colTotals := stats.predicted
    let correct := stats.correct
    let total := stats.total
    let accPct := if total == 0 then "n/a" else
      s!"{100.0 * Float.ofNat correct / Float.ofNat total}%"
    return <div style={json% {
      "padding": "10px",
      "border": "1px solid var(--vscode-panel-border, #e5e5e5)",
      "border-radius": "10px",
      "background": "var(--vscode-editor-background, transparent)"
    }}>
      <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap", "margin-bottom":
        "10px"}}>
        {pill "Confusion matrix"} {pill s!"classes={n}"} {pill s!"acc={accPct}"} {pill
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
                let precision := if pred = 0 then "n/a" else
                s!"{100.0 * Float.ofNat tp / Float.ofNat pred}%";
                let recallRate := if sup = 0 then "n/a" else
                s!"{100.0 * Float.ofNat tp / Float.ofNat sup}%";
                <tr>
                  <td style={json% {"padding": "6px 8px", "border-bottom":
                    "1px solid rgba(127,127,127,0.18)"}}>{monospace labels[i]!}</td>
                  <td style={json% {"padding": "6px 8px", "border-bottom":
                    "1px solid rgba(127,127,127,0.18)"}}>{monospace (toString sup)}</td>
                  <td style={json% {"padding": "6px 8px", "border-bottom":
                    "1px solid rgba(127,127,127,0.18)"}}>{monospace precision}</td>
                  <td style={json% {"padding": "6px 8px", "border-bottom":
                    "1px solid rgba(127,127,127,0.18)"}}>{monospace recallRate}</td>
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
Render a `Training.TrainLog` value from memory in the infoview, without any IO.

JSON logs that an executable wrote to disk are read by `#train_log_file_view`.
-/
macro "#train_log_view " log:term : command =>
  UI.canonicalCommand <$> `(#html (trainLogHtml $log))

/--
Read a saved `Training.TrainLog` JSON file and render it in the infoview.

The JSON schema is the one `Training.TrainLog.writeJson` writes, which TorchLean's
executable training examples use (the PPO examples under `TorchLeanExamples/Models/RL`, for
instance).

A missing or malformed file gives an error panel and the build goes on, so widget-view files stay
safe to import.
-/
macro "#train_log_file_view " path:term : command =>
  UI.canonicalCommand <$> `(#html (do
    let p : System.FilePath := $path
    try
      let log ← Training.TrainLog.readJson p
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
            {.text "Run the matching `scripts/lake.sh exe torchlean ... --log <path>` command, "}
            {.text "or pass an absolute path here."}
          </div>
          <div style={json% {"margin-top": "6px"}}>
            {monospace (toString e)}
          </div>
        </div>))

/--
Render a confusion matrix report in the infoview from an array of class labels and a
`Training.ConfusionMatrix` over the same classes, in the same order.
-/
macro "#confusion_view " labels:term ", " cm:term : command =>
  UI.canonicalCommand <$> `(#html (confusionHtml $labels $cm))

end
end TorchLean.Widget
