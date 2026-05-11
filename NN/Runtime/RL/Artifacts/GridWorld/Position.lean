/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Training.Log

/-!
# GridWorld Artifact Positions

Shared JSON encoding for GridWorld coordinates used by policy/path visualization artifacts. Positions
are stored as `[row, col]` pairs because that format is compact, human-readable, and easy for the
widgets to consume.
-/

@[expose] public section

namespace Runtime.RL.Artifacts.GridWorld

open Lean
open Json
open Runtime.Training.JsonCodec

/-- Encode a GridWorld position `(row, col)` as JSON `[row, col]`. -/
def posToJson (p : Nat × Nat) : Json :=
  .arr #[ Json.num (JsonNumber.fromNat p.1), Json.num (JsonNumber.fromNat p.2) ]

/--
Decode a GridWorld position `(row, col)` from JSON `[row, col]`.

The `field` argument is only used for nicer error messages.
-/
def posOfJsonE (field : String) (j : Json) : Except String (Nat × Nat) := do
  let xs ←
    match j.getArr? with
    | .ok xs => pure xs
    | .error e => throw s!"GridWorld artifact JSON: `{field}` expected an array: {e}"
  if xs.size != 2 then
    throw s!"GridWorld artifact JSON: `{field}` expected a pair [row, col]."
  let r ←
    match xs[0]!.getNat? with
    | .ok n => pure n
    | .error e => throw s!"GridWorld artifact JSON: `{field}[0]` expected Nat: {e}"
  let c ←
    match xs[1]!.getNat? with
    | .ok n => pure n
    | .error e => throw s!"GridWorld artifact JSON: `{field}[1]` expected Nat: {e}"
  pure (r, c)

/-- Encode an array of GridWorld positions as a JSON array of `[row, col]` pairs. -/
def posArrayToJson (xs : Array (Nat × Nat)) : Json :=
  .arr (xs.map posToJson)

/--
Decode an array of GridWorld positions from JSON, using `field` only for nicer errors.
-/
def posArrayOfJsonE (field : String) (j : Json) : Except String (Array (Nat × Nat)) := do
  let xs ←
    match j.getArr? with
    | .ok xs => pure xs
    | .error e => throw s!"GridWorld artifact JSON: `{field}` expected an array: {e}"
  xs.mapM (posOfJsonE (field := field))


end Runtime.RL.Artifacts.GridWorld
