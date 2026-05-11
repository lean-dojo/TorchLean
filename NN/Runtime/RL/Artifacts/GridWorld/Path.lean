/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Artifacts.GridWorld.Position

/-!
# GridWorld Path-Difference Artifacts

A `PathDiff` stores before/after episode trajectories for a fixed GridWorld. It uses the shared
position codec and validates that every recorded `(row, col)` stays inside the declared grid.
-/

@[expose] public section

namespace Runtime.RL.Artifacts.GridWorld

open Lean
open Json
open Runtime.Training.JsonCodec

/-!
## Rollout paths
-/

/--
Before/after episode path snapshots for a fixed `width × height` GridWorld.

Each position is stored as a pair `(row, col)` with `row < height` and `col < width`.
-/
structure PathDiff where
  width : Nat
  height : Nat
  before : Array (Nat × Nat)
  after : Array (Nat × Nat)
  notes : Array String := #[]
  deriving Inhabited

namespace PathDiff

/--
Validate a `PathDiff` record (positions are in bounds).
-/
def validateE (p : PathDiff) : Except String Unit := do
  let inBounds (pos : Nat × Nat) : Bool :=
    pos.1 < p.height && pos.2 < p.width
  if !(p.before.all inBounds) then
    throw "GridWorld path artifact: `before` contains an out-of-bounds position."
  if !(p.after.all inBounds) then
    throw "GridWorld path artifact: `after` contains an out-of-bounds position."

/-- JSON encoding for `PathDiff`. -/
def toJson (p : PathDiff) : Json :=
  Json.mkObj
    [ ("width", Json.num (JsonNumber.fromNat p.width))
    , ("height", Json.num (JsonNumber.fromNat p.height))
    , ("before", posArrayToJson p.before)
    , ("after", posArrayToJson p.after)
    , ("notes", stringArrayToJson p.notes)
    ]

/-- JSON decoding for `PathDiff`. -/
def ofJsonE (j : Json) : Except String PathDiff := do
  let o ←
    match j.getObj? with
    | .ok o => pure o
    | .error e => throw s!"GridWorld path artifact: expected object: {e}"

  let widthJ ←
    match o.get? "width" with
    | some v => pure v
    | none => throw "GridWorld path artifact: missing field `width`."
  let heightJ ←
    match o.get? "height" with
    | some v => pure v
    | none => throw "GridWorld path artifact: missing field `height`."
  let beforeJ ←
    match o.get? "before" with
    | some v => pure v
    | none => throw "GridWorld path artifact: missing field `before`."
  let afterJ ←
    match o.get? "after" with
    | some v => pure v
    | none => throw "GridWorld path artifact: missing field `after`."
  let notesJ := (o.get? "notes").getD (.arr #[])

  let width ←
    match widthJ.getNat? with
    | .ok n => pure n
    | .error e => throw s!"GridWorld path artifact: width expected Nat: {e}"
  let height ←
    match heightJ.getNat? with
    | .ok n => pure n
    | .error e => throw s!"GridWorld path artifact: height expected Nat: {e}"
  let before ← posArrayOfJsonE (field := "before") beforeJ
  let after ← posArrayOfJsonE (field := "after") afterJ
  let notes :=
    match stringArrayOfJsonE (field := "notes") notesJ with
    | Except.ok xs => xs
    | Except.error _ => #[]

  let p : PathDiff := { width := width, height := height, before := before, after := after, notes := notes }
  validateE p
  pure p

/--
Write a `PathDiff` JSON file to disk (creating parent directories if needed).
-/
def writeJson (path : System.FilePath) (p : PathDiff) (pretty : Bool := true) : IO Unit := do
  match validateE p with
  | .error e => throw <| IO.userError e
  | .ok () =>
      match path.parent with
      | some parent => IO.FS.createDirAll parent
      | none => pure ()
      let j := toJson p
      let s := if pretty then Json.pretty j else Json.compress j
      IO.FS.writeFile path (s ++ "\n")

/-- Read a `PathDiff` from a JSON file. -/
def readJson (path : System.FilePath) : IO PathDiff := do
  let s ← IO.FS.readFile path
  let j ←
    match Json.parse s with
    | .ok j => pure j
    | .error e => throw <| IO.userError s!"GridWorld path artifact: parse error: {e}"
  match ofJsonE j with
  | .ok p => pure p
  | .error e => throw <| IO.userError e

end PathDiff

end Runtime.RL.Artifacts.GridWorld
