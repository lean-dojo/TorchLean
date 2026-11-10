/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Lean.Data.Json.FromToJson.Basic
public import Lean.Data.Json.Printer
public import TorchLean.NN.Json

/-!
# Training Logs

Pure training-log data plus the stable JSON codec used by examples, widgets, and CLI artifacts.

Four compact structures carry the data:
- `Series` is one named scalar metric, such as loss or accuracy,
- `Curve` is the convenient builder for one metric over time,
- `MetricHistory` is the convenient builder for several aligned metrics, and
- `TrainLog` is the artifact that JSON IO writes and widgets display.

For W&B-style workflows, `ExperimentLog` adds run metadata, config entries, tags, and artifact
references, and it lowers to the same `TrainLog` JSON that the TorchLean widgets render. Examples
get one clean local artifact format and a straightforward bridge to hosted trackers.

Metric values are `Float` on purpose. A logged loss or accuracy is a reporting scalar that has
already left the generic `α` optimizer layer: it is converted once at the call site and then only
compared, plotted, and written as a JSON number. Generalising the log over `α` would push a
`ToJson`/`FromJson` obligation onto every scalar type for no gain in precision.

We keep JSON support in the same module as the data model so there is one canonical logging API:
`TorchLean.Training.TrainLog.writeJson` / `readJson`. Widgets live in
`TorchLean.NN.Widget.Training` and render these logs in the infoview.
-/

@[expose] public section

namespace TorchLean.Training

open Lean
open Lean.Json

/-!
## Core Artifact Model

TorchLean logs are local artifacts with a schema that mirrors the useful scalar-metric subset of
hosted experiment trackers such as Weights & Biases:
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
  /-- Display value, kept as a string so that JSON artifacts stay tool-agnostic. -/
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
  /-- Optional description, appended to the artifact's note as `desc=…`. -/
  description : String := ""
  deriving Inhabited

/--
Run metadata for a local experiment.

The fields line up with the common tracker vocabulary (`project`, `name`, `group`, `tags`, config),
but the value is plain Lean data and performs no network IO.
-/
structure RunInfo where
  /-- Project or collection name. Examples usually use `"torchlean"`. -/
  project : String := "torchlean"
  /-- Display name for this run, which `ExperimentLog.toTrainLog` uses as the log title when it is
  given none. -/
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
Build a two-point loss log from scalar losses measured before and after training.

It suits any training routine that records a baseline loss at step `0` and another loss after
`steps` updates. Loops that record more points use `Curve` or `MetricHistory` below.
-/
def lossComparison (title : String) (steps : Nat) (lossBefore lossAfter : Float)
    (notes : Array String := #[]) (color : String := "#4e79a7") : TrainLog :=
  { title := title
    steps := #[0, steps]
    series := #[
      { name := "loss", values := #[lossBefore, lossAfter], color := color }
    ]
    notes := notes }

end TrainLog

/--
A single scalar curve over discrete training steps.

Use it when a training routine records one scalar repeatedly, such as training loss, validation
loss, learning rate, or reward, and convert it to `TrainLog` for JSON persistence and widgets.
Curves are `Array`s because their lengths are known only at run time and they are persisted as
JSON for widgets, while typed tensors are reserved for fixed-shape model inputs and outputs.
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
Convert a single curve into a `TrainLog` with one series, the form that TorchLean's widgets
(`#train_log_file_view`) display.
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

A `Curve` holds a single scalar, but many training runs record several aligned metrics: train loss,
validation loss, accuracy, learning rate, reward, and so on. `MetricHistory` stores that
table-shaped history and converts it to the `TrainLog` artifact read by JSON IO and widgets.
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

If `values.size` does not match `series.size`, the update is ignored. Callers that want a hard
error can check sizes before calling.
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
widgets. The conversion stores the metadata as structured notes, so every `TrainLog` consumer can
read it without a second widget protocol.
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

/-- Render run metadata and config as `key=value` notes for a `TrainLog`, such as
`project=torchlean` and `config.lr=…`. An empty name, run id, group or tag list is omitted. -/
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

/-- Render an artifact reference as one `TrainLog` note: `artifact.name=path`, followed by its kind
and description when they are not empty. -/
def toNote (a : Artifact) : String :=
  let base := s!"artifact.{a.name}={a.path}"
  let base := if a.kind.isEmpty then base else base ++ s!" kind={a.kind}"
  if a.description.isEmpty then base else base ++ s!" desc={a.description}"

end Artifact

namespace ExperimentLog

/-- Start a local experiment with optional metadata. -/
def init (run : RunInfo := {}) (metrics : Array (String × String) := #[]) : ExperimentLog :=
  { run := run, history := MetricHistory.empty metrics }

/-- Record a scalar at a step. Consecutive writes at the same step share one row, and writing the
same metric again replaces its value. Unreported metrics read `NaN`, including earlier rows when a
metric is first introduced. -/
def log (e : ExperimentLog) (step : Nat) (name : String) (value : Float)
    (color : String := "#4e79a7") : ExperimentLog := Id.run do
  let missing := Float.ofBits 0x7ff8000000000000
  let sameStep := e.history.steps.back? == some step
  let steps := if sameStep then e.history.steps else e.history.steps.push step
  let row := steps.size - 1
  let series := if sameStep then e.history.series else
    e.history.series.map (fun s => { s with values := s.values.push missing })
  let series := match series.findIdx? (fun s => s.name = name) with
    | some i => series.modify i (fun s => { s with values := s.values.set! row value })
    | none => series.push
        { name, color, values := (Array.replicate steps.size missing).set! row value }
  return { e with history := { steps, series } }

/-- Convert a rich experiment object into the stable widget/JSON `TrainLog` artifact. -/
def toTrainLog (e : ExperimentLog) (title : String := "") : TrainLog :=
  let title := if title.isEmpty then
    if e.run.name.isEmpty then "Training" else e.run.name
  else
    title
  let notes := e.run.toNotes ++ e.notes ++ e.artifacts.map Artifact.toNote
  e.history.toTrainLog title notes

end ExperimentLog

/-!
## Logging Destinations

Examples should say where their logs go. `LogDestination.disabled` is the local equivalent of
`wandb disabled`, and `LogDestination.jsonFile path` writes the standard TorchLean JSON artifact.
-/

/-- Where a training routine should send its log artifact. -/
inductive LogDestination where
  /-- Do not write a log artifact. Useful for tests, CI, and runs where metrics are printed only. -/
  | disabled
  /-- Write a local JSON artifact to the given path. -/
  | jsonFile (path : System.FilePath)
  deriving Inhabited, Repr, BEq

instance : ToString LogDestination where
  toString
    | .disabled => "disabled"
    | .jsonFile path => path.toString

namespace LogDestination

/-- Whether this destination writes a log artifact. -/
def isEnabled : LogDestination → Bool
  | .disabled => false
  | .jsonFile _ => true

/-- Return the JSON path when this destination writes one. -/
def path? : LogDestination → Option System.FilePath
  | .disabled => none
  | .jsonFile path => some path

/--
Parse a CLI logging value.

The values `false`, `off`, `none`, `no` and `disabled` turn logging off, in any letter case and
with surrounding whitespace ignored. Any other value is taken unchanged as a JSON path.
-/
def parse (raw : String) : LogDestination :=
  let lower := raw.trimAscii.toString.toLower
  if lower = "false" || lower = "off" || lower = "none" || lower = "no" ||
      lower = "disabled" then
    .disabled
  else
    .jsonFile (System.FilePath.mk raw)

/-- Resolve an optional CLI value, preserving `default` when no override is supplied. -/
def resolve (default : LogDestination) (value? : Option String) : LogDestination :=
  match value? with
  | none => default
  | some raw => parse raw

end LogDestination

/-!
## JSON Codec

Training logs are often produced by executable examples and rendered later by widgets. The codec
is stable and easy to inspect:
- finite floats are JSON numbers,
- non-finite floats are string sentinels (`"NaN"`, `"Infinity"`, `"-Infinity"`), and
- missing optional fields fall back to the same defaults as the Lean structures, and a field of
  the wrong type is an error.

The decoders read through `Json`, whose `Json.float` also reads the sentinels. Examples,
tests, widgets and external tools all read and write this one format.
-/

namespace JsonCodec

/-! ### Primitive arrays -/

/-- Encode finite metric values without losing precision to the default six-decimal formatter:
when Lean's JSON number does not read back to the same `Float`, the exact decimal expansion of the
binary value is written. Non-finite values use Lean's string sentinels, and both signed zeros
serialize as JSON `0`.
-/
def floatToJson (x : Float) : Json :=
  match JsonNumber.fromFloat? x with
  | .inl s => .str s
  | .inr short =>
      if short.toFloat == x then
        .num short
      else
        match x.toModel.unpack with
        | .finite sign mantissa exponent _ =>
            -- Convert m * 2^e exactly: a negative exponent gives m * 5^(-e) / 10^(-e).
            let (digits, places) := match exponent with
              | .ofNat e => (mantissa * 2 ^ e, 0)
              | .negSucc e => (mantissa * 5 ^ (e + 1), e + 1)
            let signed : Int := if sign == .negative then -(Int.ofNat digits) else Int.ofNat digits
            .num ⟨signed, places⟩
        | _ => .num short

/-- Encode a float-valued metric series. -/
def floatArrayToJson (xs : Array Float) : Json :=
  .arr (xs.map floatToJson)

/-- Create parent directories, then encode and write a newline-terminated JSON artifact. -/
def writeFile {α : Type} (path : System.FilePath) (encode : α → Json)
    (value : α) (pretty : Bool) : IO Unit := do
  match path.parent with
  | some parent => IO.FS.createDirAll parent
  | none => pure ()
  let j := encode value
  let s := if pretty then Json.pretty j else Json.compress j
  IO.FS.writeFile path (s ++ "\n")

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
  let ctx := "TrainLog JSON: series"
  let o ← Json.object ctx j
  let nameJ ← Json.field ctx "name" o
  let valuesJ ← Json.field ctx "values" o
  let name ← Json.string s!"{ctx}.name" nameJ
  let values ← (← Json.array s!"{ctx}({name}).values" valuesJ).mapM
    (Json.float s!"{ctx}({name}).values")
  let color ← Json.string s!"{ctx}.color" ((o.get? "color").getD (.str "#0a7"))
  pure { name := name, values := values, color := color }

end Series

namespace TrainLog

/-- JSON encoding for the stable `TrainLog` artifact. -/
def toJson (log : TrainLog) : Json :=
  Json.mkObj
    [ ("title", .str log.title)
    , ("steps", Lean.toJson log.steps)
    , ("series", .arr (log.series.map Series.toJson))
    , ("notes", Lean.toJson log.notes)
    ]

/-- JSON decoding for the stable `TrainLog` artifact. A missing `title`, `steps`, `series` or
`notes` field falls back to its default, while a field of the wrong type or a malformed series is
an error. -/
def ofJsonE (j : Json) : Except String TrainLog := do
  let o ← Json.object "TrainLog JSON" j
  let fieldOr (key : String) (fallback : Json) : Json := (o.get? key).getD fallback
  let title ← Json.string "TrainLog JSON: title" (fieldOr "title" (.str "Training"))
  let steps ← Json.nats "TrainLog JSON: steps" (fieldOr "steps" (.arr #[]))
  let series ← (← Json.array "TrainLog JSON: series" (fieldOr "series" (.arr #[]))).mapM
    Series.ofJsonE
  let notes ← (← Json.array "TrainLog JSON: notes" (fieldOr "notes" (.arr #[]))).mapM
    (Json.string "TrainLog JSON: notes")
  pure { title := title, steps := steps, series := series, notes := notes }

/-- Write a `TrainLog` as JSON to disk, creating parent directories if needed. -/
def writeJson (path : System.FilePath) (log : TrainLog) (pretty : Bool := true) : IO Unit :=
  JsonCodec.writeFile path toJson log pretty

/-- Read a `TrainLog` from a JSON file. -/
def readJson (path : System.FilePath) : IO TrainLog := do
  IO.ofExcept (ofJsonE (← Json.readFile path))

end TrainLog

namespace ConfusionMatrix

/-- Counts over every class, independent of any display limits. -/
structure Statistics where
  /-- Number of true examples per class (row totals). -/
  support : Array Nat
  /-- Number of predictions per class (column totals). -/
  predicted : Array Nat
  /-- Number of correctly classified examples. -/
  correct : Nat
  /-- Total number of classified examples. -/
  total : Nat

/-- Summarize a square confusion matrix, rejecting malformed row lengths. -/
def classStatistics (cm : ConfusionMatrix) : Except String Statistics := do
  let n := cm.counts.size
  unless cm.counts.all (fun row => row.size == n) do
    throw "confusion matrix must be square"
  let support := cm.counts.map (fun row => row.foldl (· + ·) 0)
  let predicted := (Array.range n).map (fun j =>
    cm.counts.foldl (fun count row => count + row[j]!) 0)
  let correct := (Array.range n).foldl (fun count i => count + (cm.counts[i]!)[i]!) 0
  return { support, predicted, correct, total := support.foldl (· + ·) 0 }

end ConfusionMatrix

namespace LogDestination

/-- Write a `TrainLog` to this destination. Disabled destinations are a no-op. -/
def writeTrainLog (dest : LogDestination) (log : TrainLog) (pretty : Bool := true) : IO Unit := do
  match dest with
  | .disabled => pure ()
  | .jsonFile path => TrainLog.writeJson path log pretty

end LogDestination

end TorchLean.Training
