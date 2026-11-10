/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Lean.Data.Json.Basic
public import TorchLean.NN.Training

/-!
# GridWorld Run Artifacts

The small GridWorld JSON artifacts of TorchLean's RL workflows and widgets. A `Diff β` stores
before/after snapshots for a fixed `width × height` GridWorld, and a `Codec β` names the
artifact and encodes, decodes and validates its snapshots:

- `Codec.policy`: greedy action maps, row-major arrays of action indices `0..3`.
- `Codec.path`: episode trajectories, arrays of `(row, col)` positions inside the grid.

These files are small run artifacts for visualization and regression checks, not a general RL
dataset format. A cell `(row, col)` is written as the two-element array `[row, col]`, which keeps a
long path compact and easy to read. The GridWorld widgets read these files and draw the two
snapshots side by side. The decoders read through `Json`, as the training-log decoders do,
so both report malformed JSON the same way.
-/

@[expose] public section

namespace TorchLean.Examples.GridWorld

open Lean
open Lean.Json

/-!
## Positions
-/

/-- Encode a GridWorld position `(row, col)` as JSON `[row, col]`. -/
def posToJson (p : Nat × Nat) : Json :=
  .arr #[ Json.num (JsonNumber.fromNat p.1), Json.num (JsonNumber.fromNat p.2) ]

/-- Decode the JSON pair `[row, col]` of natural numbers into the position `(row, col)`. Error
messages include `ctx`. -/
def posOfJsonE (ctx : String) (j : Json) : Except String (Nat × Nat) := do
  match ← Json.nats ctx j with
  | #[r, c] => pure (r, c)
  | _ => Json.typeError ctx "a pair [row, col]" j

/-- Encode an array of GridWorld positions as a JSON array of `[row, col]` pairs. -/
def posArrayToJson (xs : Array (Nat × Nat)) : Json :=
  .arr (xs.map posToJson)

/-- Decode an array of GridWorld positions from JSON. Error messages include `ctx`. -/
def posArrayOfJsonE (ctx : String) (j : Json) : Except String (Array (Nat × Nat)) := do
  (← Json.array ctx j).mapM (posOfJsonE ctx)

/-!
## Before/after snapshots
-/

/--
Before/after snapshots for a fixed `width × height` GridWorld: greedy policies (`Diff Nat`, with
`Codec.policy`) or rollout paths (`Diff (Nat × Nat)`, with `Codec.path`).
-/
structure Diff (β : Type) where
  /-- Number of grid columns. -/
  width : Nat
  /-- Number of grid rows. -/
  height : Nat
  /-- The earlier snapshot. -/
  before : Array β
  /-- The later snapshot. -/
  after : Array β
  /-- Free-form notes stored with the artifact. -/
  notes : Array String := #[]
  deriving Inhabited

/--
The JSON format of one kind of `Diff`: the label of its error messages, the codec of one snapshot,
and the check that readers and writers run.
-/
structure Codec (β : Type) where
  /-- The artifact kind in error messages (`GridWorld <label> artifact: ...`). -/
  label : String
  /-- Encode one snapshot. -/
  encode : Array β → Json
  /-- Decode one snapshot. Error messages include the string argument. -/
  decode : String → Json → Except String (Array β)
  /-- Check a record before it is written and after it is read. -/
  validate : Diff β → Except String Unit

variable {β : Type}

/-- Encode `p` as a JSON object with fields `width`, `height`, `before`, `after` and `notes`. -/
def Diff.toJson (codec : Codec β) (p : Diff β) : Json :=
  Json.mkObj
    [ ("width", Json.num (JsonNumber.fromNat p.width))
    , ("height", Json.num (JsonNumber.fromNat p.height))
    , ("before", codec.encode p.before)
    , ("after", codec.encode p.after)
    , ("notes", Lean.toJson p.notes)
    ]

/--
Decode and validate a `Diff` from a JSON object. The fields `width`, `height`, `before` and `after`
are required, and a missing `notes` field decodes as no notes but a malformed one is rejected.
-/
def Diff.ofJsonE (codec : Codec β) (j : Json) : Except String (Diff β) := do
  let ctx := s!"GridWorld {codec.label} artifact"
  let o ← Json.object ctx j

  let widthJ ← Json.field ctx "width" o
  let heightJ ← Json.field ctx "height" o
  let beforeJ ← Json.field ctx "before" o
  let afterJ ← Json.field ctx "after" o

  let width ← Json.nat s!"{ctx}: width" widthJ
  let height ← Json.nat s!"{ctx}: height" heightJ
  let before ← codec.decode s!"{ctx}: before" beforeJ
  let after ← codec.decode s!"{ctx}: after" afterJ
  let notes ← (← Json.array s!"{ctx}: notes" ((o.get? "notes").getD (.arr #[]))).mapM
    (Json.string s!"{ctx}: notes")

  let p : Diff β :=
    { width := width, height := height, before := before, after := after, notes := notes }
  codec.validate p
  pure p

/--
Validate `p` and write it as JSON to `path`, creating parent directories if needed. The file is
pretty-printed unless `pretty := false`.
-/
def Diff.writeJson (codec : Codec β) (path : System.FilePath) (p : Diff β) (pretty : Bool := true) :
    IO Unit := do
  IO.ofExcept (codec.validate p)
  Training.JsonCodec.writeFile path (toJson codec) p pretty

/-- Read a `Diff` from the JSON file at `path` and validate it. -/
def Diff.readJson (codec : Codec β) (path : System.FilePath) : IO (Diff β) := do
  IO.ofExcept (ofJsonE codec (← Json.readFile path))

namespace Codec

/--
Greedy policy snapshots: one action per cell in row-major order, cell `(row, col)` at index
`row * width + col`, with `0..3` meaning up, down, left and right. The check requires
`width * height` actions in each snapshot, every one below `4`.
-/
def policy : Codec Nat where
  label := "policy"
  encode := Lean.toJson
  decode := Json.nats
  validate p := do
    let expected := p.width * p.height
    if p.before.size != expected then
      throw s!"GridWorld policy artifact: `before` expected length {expected}, got {p.before.size}."
    if p.after.size != expected then
      throw s!"GridWorld policy artifact: `after` expected length {expected}, got {p.after.size}."
    if !(p.before.all (fun a => a < 4)) then
      throw "GridWorld policy artifact: `before` contains an out-of-range action (expected 0..3)."
    if !(p.after.all (fun a => a < 4)) then
      throw "GridWorld policy artifact: `after` contains an out-of-range action (expected 0..3)."

/--
Rollout path snapshots: arrays of positions `(row, col)`. The check requires `row < height` and
`col < width` for every position.
-/
def path : Codec (Nat × Nat) where
  label := "path"
  encode := posArrayToJson
  decode := posArrayOfJsonE
  validate p := do
    let inBounds (pos : Nat × Nat) : Bool :=
      pos.1 < p.height && pos.2 < p.width
    if !(p.before.all inBounds) then
      throw "GridWorld path artifact: `before` contains an out-of-bounds position."
    if !(p.after.all inBounds) then
      throw "GridWorld path artifact: `after` contains an out-of-bounds position."

end Codec

end TorchLean.Examples.GridWorld
