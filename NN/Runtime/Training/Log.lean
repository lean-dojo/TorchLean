/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Lean.Data.Json.Parser
public import Lean.Data.Json.Printer

/-!
# Training Logs

Pure training-log data plus the stable JSON codec used by examples, widgets, and CLI artifacts.

The data structures in this file are deliberately small:
- `Series` is one named scalar metric, such as loss or accuracy;
- `Curve` is the convenient builder for one metric over time;
- `MetricHistory` is the convenient builder for several aligned metrics; and
- `TrainLog` is the persisted/viewed artifact shared by JSON IO and widgets.

For W&B-style workflows, `ExperimentLog` adds run metadata, config entries, tags, and artifact
references while still lowering to the same stable `TrainLog` JSON that TorchLean widgets already
know how to render. This gives examples one clean local artifact format and a straightforward bridge
to hosted trackers.

We keep JSON support in the same module as the data model so there is one canonical logging API:
`Runtime.Training.TrainLog.writeJson` / `readJson`. Widgets live in
`NN.Widgets.Runtime.Training` and render these logs in the infoview.
-/

@[expose] public section

namespace Runtime.Training

open Lean
open Json

/-!
## Core Artifact Model

TorchLean logs are intentionally local-first. The shape mirrors the small useful subset of hosted
experiment trackers such as Weights & Biases:
- a run has project/name/id metadata,
- config records hyperparameters and data choices,
- metric history records scalar values over steps,
- artifacts point to files produced by the run.

The data stays simple and serializes to ordinary JSON, so examples can write it without any network
service or background process.
-/

/-- A string-valued hyperparameter/config entry attached to a run. -/
structure ConfigEntry where
  /-- Config key, for example `"lr"`, `"optimizer"`, or `"dataset"`. -/
  key : String
  /-- Human-readable value. Keep this stringly-typed so JSON artifacts stay tool-agnostic. -/
  value : String
  deriving Inhabited

/-- A file artifact produced or consumed by a training run. -/
structure Artifact where
  /-- Short artifact name, for example `"checkpoint"` or `"predictions_csv"`. -/
  name : String
  /-- Local path to the artifact. External trackers may use this path when uploading artifacts. -/
  path : System.FilePath
  /-- Optional kind, such as `"model"`, `"plot"`, `"dataset"`, or `"report"`. -/
  kind : String := "file"
  /-- Optional human-readable description. -/
  description : String := ""
  deriving Inhabited

/--
Run metadata for a local experiment.

This is deliberately close to the mental model of W&B (`project`, `name`, `group`, `tags`, config),
but it remains a plain Lean value and does not perform network IO.
-/
structure RunInfo where
  /-- Project or collection name. Examples usually use `"torchlean"`. -/
  project : String := "torchlean"
  /-- Display name for this run. Defaults to the resulting `TrainLog.title` if left empty. -/
  name : String := ""
  /-- Stable run identifier, useful when several artifacts belong to one run. -/
  runId : String := ""
  /-- Optional group/sweep name. -/
  group : String := ""
  /-- Tags for filtering/comparing runs. -/
  tags : Array String := #[]
  /-- String-valued hyperparameters and data/runtime choices. -/
  config : Array ConfigEntry := #[]
  deriving Inhabited

/-- A named scalar metric series over training steps (e.g. loss/accuracy/LR). -/
structure Series where
  /-- Display name (e.g. `"loss"` or `"val_acc"`). -/
  name : String
  /-- Scalar values over steps. -/
  values : Array Float
  /-- CSS color hint used by viewers. -/
  color : String := "#0a7"
  deriving Inhabited

/-- A small multi-series training log (curves + optional notes). -/
structure TrainLog where
  /-- Title shown at the top of viewers. -/
  title : String := "Training"
  /--
  Optional step indices.

  If empty, viewers typically use `0..(n-1)`.
  -/
  steps : Array Nat := #[]
  /-- One or more metric series (should have compatible lengths). -/
  series : Array Series := #[]
  /-- Free-form notes (hyperparameters, dataset, run id, etc.). -/
  notes : Array String := #[]

namespace TrainLog

/--
Build a two-point loss log from an initial and final scalar loss.

This is useful for any training routine that records a baseline loss at step `0` and another
loss after `steps` updates. More detailed loops should use `Curve` or `MetricHistory` below.
-/
def beforeAfterLoss (title : String) (steps : Nat) (loss0 loss1 : Float)
    (notes : Array String := #[]) (color : String := "#4e79a7") : TrainLog :=
  { title := title
    steps := #[0, steps]
    series := #[
      { name := "loss", values := #[loss0, loss1], color := color }
    ]
    notes := notes }

end TrainLog

/--
 A single scalar curve over discrete training steps.

Use this when a training routine records one scalar repeatedly, such as training loss, validation
loss, learning rate, or reward. Convert it to `TrainLog` for JSON persistence and widgets.

We intentionally store curves as `Array`s (not tensors):
- curve lengths are runtime-dependent,
- curves are persisted as JSON for widgets, and
- typed tensors are reserved for fixed-shape model inputs/outputs.
-/
structure Curve where
  /-- Step indices (e.g. update number). -/
  steps : Array Nat := #[]
  /-- Scalar metric values aligned with `steps`. -/
  values : Array Float := #[]
  deriving Inhabited

namespace Curve

/-- Append one point `(step, value)` to a curve. -/
def push (c : Curve) (step : Nat) (value : Float) : Curve :=
  { steps := c.steps.push step, values := c.values.push value }

/--
Convert a single curve into a `TrainLog` with one series.

This matches the expectations of TorchLean's widgets (`#train_log_file_view`).
-/
def toTrainLog (c : Curve) (title : String) (seriesName : String)
    (color : String := "#4e79a7") (notes : Array String := #[]) : TrainLog :=
  { title := title
    steps := c.steps
    series := #[
      { name := seriesName, values := c.values, color := color }
    ]
    notes := notes }

end Curve

/-!
## Multi-metric histories

`Curve` is perfect for a single scalar, but many training runs record several aligned metrics:
train loss, validation loss, accuracy, learning rate, reward, and so on. `MetricHistory` stores
that common table-shaped history and converts it to the stable `TrainLog` artifact consumed by
JSON IO and widgets.
-/

/-- A multi-series scalar history aligned by step. -/
structure MetricHistory where
  /-- Step indices shared by every series. -/
  steps : Array Nat := #[]
  /-- Metric series. Each push appends one value to each series. -/
  series : Array Series := #[]
  deriving Inhabited

namespace MetricHistory

/-- Construct an empty history from `(metric name, color)` pairs. -/
def empty (metrics : Array (String × String)) : MetricHistory :=
  { steps := #[]
    series := metrics.map (fun (name, color) =>
      { name := name, values := #[], color := color }) }

/--
Append one row of metric values.

If `values.size` does not match `series.size`, the update is ignored.  This keeps the function
total and avoids manufacturing malformed logs; callers that want a hard error can check sizes
before calling.
-/
def push (h : MetricHistory) (step : Nat) (values : Array Float) : MetricHistory :=
  if values.size = h.series.size then
    { h with
      steps := h.steps.push step
      series := h.series.zipIdx.map (fun (s, i) =>
        { s with values := s.values.push (values[i]!) }) }
  else
    h

/-- Convert a multi-metric history into a stable `TrainLog` artifact. -/
def toTrainLog (h : MetricHistory) (title : String) (notes : Array String := #[]) : TrainLog :=
  { title := title, steps := h.steps, series := h.series, notes := notes }

end MetricHistory

/-- Confusion matrix for classification evaluation. -/
structure ConfusionMatrix where
  /-- Row-major counts: `counts[i][j]` means “true class i predicted as j”. -/
  counts : Array (Array Nat)

/--
A W&B-shaped local experiment artifact.

`ExperimentLog` is the richer authoring object. Use `toTrainLog` before writing JSON or rendering in
widgets. The conversion stores metadata as structured notes so existing `TrainLog` consumers keep
working without a second widget protocol.
-/
structure ExperimentLog where
  /-- Run metadata and config. -/
  run : RunInfo := {}
  /-- Aligned scalar metrics over time. -/
  history : MetricHistory := {}
  /-- Files produced or consumed by this run. -/
  artifacts : Array Artifact := #[]
  /-- Additional free-form notes. -/
  notes : Array String := #[]
  deriving Inhabited

/-!
## Experiment-Tracker Helpers
-/

namespace RunInfo

/-- Render run metadata and config as stable notes inside a `TrainLog`. -/
def toNotes (run : RunInfo) : Array String :=
  let base :=
    #[ s!"project={run.project}" ]
  let base := if run.name.isEmpty then base else base.push s!"run_name={run.name}"
  let base := if run.runId.isEmpty then base else base.push s!"run_id={run.runId}"
  let base := if run.group.isEmpty then base else base.push s!"group={run.group}"
  let base :=
    if run.tags.isEmpty then base
    else
      let tagString := String.intercalate "," run.tags.toList
      base.push s!"tags={tagString}"
  run.config.foldl (fun acc kv => acc.push s!"config.{kv.key}={kv.value}") base

end RunInfo

namespace Artifact

/-- Render an artifact reference as a stable note inside a `TrainLog`. -/
def toNote (a : Artifact) : String :=
  let base := s!"artifact.{a.name}={a.path}"
  let base := if a.kind.isEmpty then base else base ++ s!" kind={a.kind}"
  if a.description.isEmpty then base else base ++ s!" desc={a.description}"

end Artifact

namespace ExperimentLog

/-- Start a local experiment with optional metadata. -/
def init (run : RunInfo := {}) (metrics : Array (String × String) := #[]) : ExperimentLog :=
  { run := run, history := MetricHistory.empty metrics }

/-- Append one scalar metric. New metric names are added lazily with a default color. -/
def log (e : ExperimentLog) (step : Nat) (name : String) (value : Float)
    (color : String := "#4e79a7") : ExperimentLog :=
  let idx? := e.history.series.findIdx? (fun s => s.name = name)
  match idx? with
  | some i =>
      let series := e.history.series.modify i (fun s =>
        { s with values := s.values.push value })
      { e with history := { steps := e.history.steps.push step, series := series } }
  | none =>
      let series := e.history.series.push { name := name, values := #[value], color := color }
      { e with history := { steps := e.history.steps.push step, series := series } }

/-- Append one aligned row of metrics to an experiment. -/
def logRow (e : ExperimentLog) (step : Nat) (values : Array Float) : ExperimentLog :=
  { e with history := e.history.push step values }

/-- Attach a local file artifact to the run. -/
def addArtifact (e : ExperimentLog) (artifact : Artifact) : ExperimentLog :=
  { e with artifacts := e.artifacts.push artifact }

/-- Convert a rich experiment object into the stable widget/JSON `TrainLog` artifact. -/
def toTrainLog (e : ExperimentLog) (title : String := "") : TrainLog :=
  let title := if title.isEmpty then
    if e.run.name.isEmpty then "Training" else e.run.name
  else
    title
  let notes := e.run.toNotes ++ e.notes ++ e.artifacts.map Artifact.toNote
  { e.history.toTrainLog title notes with notes := notes }

end ExperimentLog

/-!
## Logging Destinations

Examples should make logging explicit. `LogDestination.disabled` is the local equivalent of
`wandb disabled`; `LogDestination.json path` writes the standard TorchLean JSON artifact.
-/

/-- Where a training routine should send its log artifact. -/
inductive LogDestination where
  /-- Do not write a log artifact. Useful for lightweight tests or CI runs. -/
  | disabled
  /-- Write a local JSON artifact to the given path. -/
  | json (path : System.FilePath)
  deriving Inhabited, Repr

namespace LogDestination

/-- Is logging enabled for this destination? -/
def isEnabled : LogDestination → Bool
  | .disabled => false
  | .json _ => true

/-- Return the JSON path when this destination writes one. -/
def path? : LogDestination → Option System.FilePath
  | .disabled => none
  | .json path => some path

/-- Return the JSON path, falling back to `defaultPath` when logging is disabled. -/
def pathD (dest : LogDestination) (defaultPath : System.FilePath) : System.FilePath :=
  dest.path?.getD defaultPath

/--
Parse a CLI logging value.

Accepted disabled values are `false`, `off`, `none`, `no`, and `disabled`. Any other value is
treated as a JSON path.
-/
def parseValue (raw : String) : LogDestination :=
  let lower := raw.trimAscii.toString.toLower
  if lower = "false" || lower = "off" || lower = "none" || lower = "no" ||
      lower = "disabled" then
    .disabled
  else
    .json (System.FilePath.mk raw)

/-- Parse an optional CLI value, using the default JSON path when no value is supplied. -/
def parse? (defaultPath : System.FilePath) (raw? : Option String) : LogDestination :=
  match raw? with
  | none => .json defaultPath
  | some raw => parseValue raw

end LogDestination

/-!
## JSON Codec

Training logs are often produced by executable examples and then rendered later by widgets. The
codec below is intentionally stable and human-readable:
- finite floats are JSON numbers;
- non-finite floats are string sentinels (`"NaN"`, `"Infinity"`, `"-Infinity"`);
- missing optional fields fall back to the same defaults as the Lean structures.

This is not intended to be a high-throughput experiment tracker. It is a small artifact format that
keeps examples, tests, widgets, and external tooling speaking the same language.
-/

namespace JsonCodec

/-! ### Primitive arrays -/

/-- Encode a `Float` as JSON: a number when finite, otherwise Lean's standard string sentinel. -/
def floatToJson (x : Float) : Json :=
  match JsonNumber.fromFloat? x with
  | .inr n => .num n
  | .inl s => .str s

/--
Decode a `Float` from JSON.

We accept JSON numbers, Lean's non-finite string sentinels, and numeric strings for backward
compatibility with generated artifacts.
-/
def floatOfJsonE (field : String) (j : Json) : Except String Float :=
  match j with
  | .num n => .ok n.toFloat
  | .str s =>
      if s = "NaN" then
        .ok (0.0 / 0.0)
      else if s = "Infinity" then
        .ok (1.0 / 0.0)
      else if s = "-Infinity" then
        .ok (-1.0 / 0.0)
      else
        match Json.parse s with
        | .ok (.num n) => .ok n.toFloat
        | _ => .error s!"Metric JSON: `{field}` expected a number, got string `{s}`."
  | _ => .error s!"Metric JSON: `{field}` expected a number."

/-- Encode natural-number arrays used by logs and lightweight runtime artifacts. -/
def natArrayToJson (xs : Array Nat) : Json :=
  .arr (xs.map (fun n => Json.num (JsonNumber.fromNat n)))

/-- Decode natural-number arrays with field-aware errors. -/
def natArrayOfJsonE (field : String) (j : Json) : Except String (Array Nat) := do
  let xs ←
    match j.getArr? with
    | .ok xs => pure xs
    | .error e => throw s!"Metric/artifact JSON: `{field}` expected an array: {e}"
  xs.mapM (fun v =>
    match v.getNat? with
    | .ok n => pure n
    | .error e => throw s!"Metric/artifact JSON: `{field}` expected Nat entries: {e}")

/-- Encode string-note arrays used by logs and lightweight runtime artifacts. -/
def stringArrayToJson (xs : Array String) : Json :=
  .arr (xs.map (fun s => .str s))

/-- Decode string-note arrays with field-aware errors. -/
def stringArrayOfJsonE (field : String) (j : Json) : Except String (Array String) := do
  let xs ←
    match j.getArr? with
    | .ok xs => pure xs
    | .error e => throw s!"Metric/artifact JSON: `{field}` expected an array: {e}"
  xs.mapM (fun v =>
    match v.getStr? with
    | .ok s => pure s
    | .error e => throw s!"Metric/artifact JSON: `{field}` expected String entries: {e}")

/-- Encode a float-valued metric series. -/
def floatArrayToJson (xs : Array Float) : Json :=
  .arr (xs.map floatToJson)

/-- Decode a float-valued metric series with field-aware errors. -/
def floatArrayOfJsonE (field : String) (j : Json) : Except String (Array Float) := do
  let xs ←
    match j.getArr? with
    | .ok xs => pure xs
    | .error e => throw s!"TrainLog JSON: `{field}` expected an array: {e}"
  xs.mapM (floatOfJsonE (field := field))

end JsonCodec

namespace Series

/-- JSON encoding for one named metric series. -/
def toJson (s : Series) : Json :=
  Json.mkObj
    [ ("name", .str s.name)
    , ("values", JsonCodec.floatArrayToJson s.values)
    , ("color", .str s.color)
    ]

/-- JSON decoding for one named metric series. -/
def ofJsonE (j : Json) : Except String Series := do
  let o ←
    match j.getObj? with
    | .ok o => pure o
    | .error e => throw s!"TrainLog JSON: series expected object: {e}"

  let nameJ ←
    match o.get? "name" with
    | some v => pure v
    | none => throw "TrainLog JSON: series missing field `name`."
  let valuesJ ←
    match o.get? "values" with
    | some v => pure v
    | none => throw "TrainLog JSON: series missing field `values`."
  let colorJ := (o.get? "color").getD (.str "#0a7")

  let name ←
    match nameJ.getStr? with
    | .ok s => pure s
    | .error e => throw s!"TrainLog JSON: series.name expected String: {e}"
  let values ← JsonCodec.floatArrayOfJsonE (field := s!"series({name}).values") valuesJ
  let color ←
    match colorJ.getStr? with
    | .ok s => pure s
    | .error e => throw s!"TrainLog JSON: series.color expected String: {e}"

  pure { name := name, values := values, color := color }

end Series

namespace TrainLog

/-- JSON encoding for the stable `TrainLog` artifact. -/
def toJson (log : TrainLog) : Json :=
  Json.mkObj
    [ ("title", .str log.title)
    , ("steps", JsonCodec.natArrayToJson log.steps)
    , ("series", .arr (log.series.map Series.toJson))
    , ("notes", JsonCodec.stringArrayToJson log.notes)
    ]

/-- JSON decoding for the stable `TrainLog` artifact. -/
def ofJsonE (j : Json) : Except String TrainLog := do
  let o ←
    match j.getObj? with
    | .ok o => pure o
    | .error e => throw s!"TrainLog JSON: train log expected object: {e}"

  let title : String :=
    match (o.get? "title") with
    | some (.str s) => s
    | some _ => "Training"
    | none => "Training"

  let steps : Array Nat :=
    match o.get? "steps" with
    | none => #[]
    | some v =>
        match JsonCodec.natArrayOfJsonE (field := "steps") v with
        | Except.ok xs => xs
        | Except.error _ => #[]

  let seriesJ := (o.get? "series").getD (.arr #[])
  let seriesArr ←
    match seriesJ.getArr? with
    | .ok xs => pure xs
    | .error e => throw s!"TrainLog JSON: `series` expected array: {e}"
  let series ← seriesArr.mapM Series.ofJsonE

  let notes : Array String :=
    match o.get? "notes" with
    | none => #[]
    | some v =>
        match JsonCodec.stringArrayOfJsonE (field := "notes") v with
        | Except.ok xs => xs
        | Except.error _ => #[]

  pure { title := title, steps := steps, series := series, notes := notes }

/-- Write a `TrainLog` as JSON to disk, creating parent directories if needed. -/
def writeJson (path : System.FilePath) (log : TrainLog) (pretty : Bool := true) : IO Unit := do
  match path.parent with
  | some parent => IO.FS.createDirAll parent
  | none => pure ()
  let j := toJson log
  let s := if pretty then Json.pretty j else Json.compress j
  IO.FS.writeFile path (s ++ "\n")

/-- Read a `TrainLog` from a JSON file. -/
def readJson (path : System.FilePath) : IO TrainLog := do
  let s ← IO.FS.readFile path
  let j ←
    match Json.parse s with
    | .ok j => pure j
    | .error e => throw <| IO.userError s!"TrainLog JSON: parse error: {e}"
  match ofJsonE j with
  | .ok log => pure log
  | .error e => throw <| IO.userError e

end TrainLog

namespace ConfusionMatrix

/-- JSON encoding for a classification confusion matrix. -/
def toJson (cm : ConfusionMatrix) : Json :=
  .arr <| cm.counts.map (fun row => JsonCodec.natArrayToJson row)

/-- JSON decoding for a classification confusion matrix. -/
def ofJsonE (j : Json) : Except String ConfusionMatrix := do
  let rows ←
    match j.getArr? with
    | .ok xs => pure xs
    | .error e => throw s!"TrainLog JSON: confusion matrix expected array: {e}"
  let counts ← rows.mapM (fun r => JsonCodec.natArrayOfJsonE (field := "confusion.counts") r)
  pure { counts := counts }

/-- Write a confusion matrix as JSON to disk, creating parent directories if needed. -/
def writeJson (path : System.FilePath) (cm : ConfusionMatrix) (pretty : Bool := true) : IO Unit := do
  match path.parent with
  | some parent => IO.FS.createDirAll parent
  | none => pure ()
  let j := toJson cm
  let s := if pretty then Json.pretty j else Json.compress j
  IO.FS.writeFile path (s ++ "\n")

/-- Read a confusion matrix from a JSON file. -/
def readJson (path : System.FilePath) : IO ConfusionMatrix := do
  let s ← IO.FS.readFile path
  let j ←
    match Json.parse s with
    | .ok j => pure j
    | .error e => throw <| IO.userError s!"TrainLog JSON: parse error: {e}"
  match ofJsonE j with
  | .ok cm => pure cm
  | .error e => throw <| IO.userError e

end ConfusionMatrix

namespace LogDestination

/-- Write a `TrainLog` to this destination. Disabled destinations are a no-op. -/
def writeTrainLog (dest : LogDestination) (log : TrainLog) (pretty : Bool := true) : IO Unit := do
  match dest with
  | .disabled => pure ()
  | .json path => TrainLog.writeJson path log pretty

/-- Write an `ExperimentLog` to this destination. Disabled destinations are a no-op. -/
def writeExperimentLog (dest : LogDestination) (log : ExperimentLog)
    (title : String := "") (pretty : Bool := true) : IO Unit := do
  writeTrainLog dest (log.toTrainLog title) pretty

end LogDestination

end Runtime.Training
