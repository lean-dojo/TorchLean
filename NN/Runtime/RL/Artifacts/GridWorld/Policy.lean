/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Artifacts.GridWorld.Position

/-!
# GridWorld Policy-Difference Artifacts

A `PolicyDiff` stores before/after greedy action maps for a fixed GridWorld. These files are small
run artifacts for visualization and regression checks, not a general RL dataset format.
-/

@[expose] public section

namespace Runtime.RL.Artifacts.GridWorld

open Lean
open Json
open Runtime.Training.JsonCodec

/-!
## Policy snapshots
-/

/--
Before/after greedy policy snapshots for a fixed `width × height` GridWorld.

`before` and `after` are flattened row-major arrays of action indices (0..3).
-/
structure PolicyDiff where
  width : Nat
  height : Nat
  before : Array Nat
  after : Array Nat
  notes : Array String := #[]
  deriving Inhabited

namespace PolicyDiff

/--
Validate a `PolicyDiff` record (lengths and action ranges).

This is used defensively by IO readers/writers and widgets; it is scoped to IO
specification layer for policies.
-/
def validateE (p : PolicyDiff) : Except String Unit := do
  let expected := p.width * p.height
  if p.before.size != expected then
    throw s!"GridWorld policy artifact: `before` expected length {expected}, got {p.before.size}."
  if p.after.size != expected then
    throw s!"GridWorld policy artifact: `after` expected length {expected}, got {p.after.size}."
  if !(p.before.all (fun a => a < 4)) then
    throw "GridWorld policy artifact: `before` contains an out-of-range action (expected 0..3)."
  if !(p.after.all (fun a => a < 4)) then
    throw "GridWorld policy artifact: `after` contains an out-of-range action (expected 0..3)."

/-- JSON encoding for `PolicyDiff`. -/
def toJson (p : PolicyDiff) : Json :=
  Json.mkObj
    [ ("width", Json.num (JsonNumber.fromNat p.width))
    , ("height", Json.num (JsonNumber.fromNat p.height))
    , ("before", natArrayToJson p.before)
    , ("after", natArrayToJson p.after)
    , ("notes", stringArrayToJson p.notes)
    ]

/-- JSON decoding for `PolicyDiff`. -/
def ofJsonE (j : Json) : Except String PolicyDiff := do
  let o ←
    match j.getObj? with
    | .ok o => pure o
    | .error e => throw s!"GridWorld policy artifact: expected object: {e}"

  let widthJ ←
    match o.get? "width" with
    | some v => pure v
    | none => throw "GridWorld policy artifact: missing field `width`."
  let heightJ ←
    match o.get? "height" with
    | some v => pure v
    | none => throw "GridWorld policy artifact: missing field `height`."
  let beforeJ ←
    match o.get? "before" with
    | some v => pure v
    | none => throw "GridWorld policy artifact: missing field `before`."
  let afterJ ←
    match o.get? "after" with
    | some v => pure v
    | none => throw "GridWorld policy artifact: missing field `after`."
  let notesJ := (o.get? "notes").getD (.arr #[])

  let width ←
    match widthJ.getNat? with
    | .ok n => pure n
    | .error e => throw s!"GridWorld policy artifact: width expected Nat: {e}"
  let height ←
    match heightJ.getNat? with
    | .ok n => pure n
    | .error e => throw s!"GridWorld policy artifact: height expected Nat: {e}"
  let before ← natArrayOfJsonE (field := "before") beforeJ
  let after ← natArrayOfJsonE (field := "after") afterJ
  let notes :=
    match stringArrayOfJsonE (field := "notes") notesJ with
    | Except.ok xs => xs
    | Except.error _ => #[]

  let p : PolicyDiff := { width := width, height := height, before := before, after := after, notes := notes }
  validateE p
  pure p

/--
Write a `PolicyDiff` JSON file to disk (creating parent directories if needed).
-/
def writeJson (path : System.FilePath) (p : PolicyDiff) (pretty : Bool := true) : IO Unit := do
  match validateE p with
  | .error e => throw <| IO.userError e
  | .ok () =>
      match path.parent with
      | some parent => IO.FS.createDirAll parent
      | none => pure ()
      let j := toJson p
      let s := if pretty then Json.pretty j else Json.compress j
      IO.FS.writeFile path (s ++ "\n")

/-- Read a `PolicyDiff` from a JSON file. -/
def readJson (path : System.FilePath) : IO PolicyDiff := do
  let s ← IO.FS.readFile path
  let j ←
    match Json.parse s with
    | .ok j => pure j
    | .error e => throw <| IO.userError s!"GridWorld policy artifact: parse error: {e}"
  match ofJsonE j with
  | .ok p => pure p
  | .error e => throw <| IO.userError e

end PolicyDiff


end Runtime.RL.Artifacts.GridWorld
