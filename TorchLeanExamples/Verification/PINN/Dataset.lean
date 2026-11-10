/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Json

/-!
# PINN Dataset Helpers

Reusable parsing and interval diagnostics for dataset-backed PINN checks.
-/

@[expose] public section

namespace TorchLean.Examples.PINN

open Lean
open Lean.Json

/-- A scalar PINN sample point `(x, y-or-t, u)`. -/
structure Sample where
  /-- First coordinate. -/
  x : Float
  /-- Second coordinate, read from `y` for spatial data or `t` for time data. -/
  yOrT : Float
  /-- Reference solution value at the point. -/
  u : Float
deriving Repr

/-- Whether `u` lies in `[lo - tol, hi + tol]`, the interval widened by `tol` at both endpoints. -/
def Dataset.containsWithTol (u lo hi tol : Float) : Bool :=
  (u ≥ lo - tol) && (u ≤ hi + tol)

/-- Read the second coordinate, accepting either `y` for 2D data or `t` for 1D-in-time data. -/
def Dataset.getSecondCoordinate (j : Json) : Except String Float := do
  let o ← Json.object "dataset point" j
  match Std.TreeMap.Raw.get? o "y" with
  | some y => Json.parseFiniteFloat "dataset point.y" y
  | none =>
      Json.parseFiniteFloat "dataset point.t" (← Json.field "dataset point" "t" o)

/-- Parse one dataset point as `(x, y-or-t, u)`. -/
def Dataset.parseSample (j : Json) : Except String Sample := do
  let o ← Json.object "dataset point" j
  let x ← Json.parseFiniteFloat "dataset point.x" (← Json.field "dataset point" "x" o)
  let yOrT ← getSecondCoordinate j
  let u ← Json.parseFiniteFloat "dataset point.u" (← Json.field "dataset point" "u" o)
  pure { x := x, yOrT := yOrT, u := u }

/-- Load one named dataset section into checked PINN sample points, reading a missing or `null`
section as empty. -/
def Dataset.loadSection (path : String) (sectionName : String) : IO (Array Sample) := do
  let j ← Json.readFile path
  let entries : Except String (Array Json) := do
    match (← Json.object "dataset" j).get? sectionName with
    | none | some .null => pure #[]
    | some value => Json.array s!"dataset.{sectionName}" value
  let arr ← match entries with
    | .ok a => pure a
    | .error msg => throw <| IO.userError s!"Dataset.{sectionName}: {msg}"
  let mut out : Array Sample := #[]
  for entry in arr do
    match parseSample entry with
    | .ok point => out := out.push point
    | .error msg => throw <| IO.userError s!"Dataset.{sectionName}: {msg}"
  pure out

end TorchLean.Examples.PINN
