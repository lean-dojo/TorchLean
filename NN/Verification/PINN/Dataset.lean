/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Util.Json

/-!
# PINN Dataset Helpers

Reusable parsing and interval diagnostics for dataset-backed PINN checks.
-/

@[expose] public section

namespace NN.Verification.PINN.Dataset

open Lean
open Json

/-- A scalar PINN sample point `(x, y-or-t, u)`. -/
structure Point where
  /-- First coordinate. -/
  x : Float
  /-- Second coordinate, read from `y` for spatial data or `t` for time data. -/
  yOrT : Float
  /-- Reference solution value at the point. -/
  u : Float
deriving Repr

/-- Absolute difference for Float diagnostics. -/
def absDiff (a b : Float) : Float :=
  if a ≥ b then a - b else b - a

/-- Check interval containment with a symmetric tolerance on the endpoints. -/
def containsWithTol (u lo hi tol : Float) : Bool :=
  (u ≥ lo - tol) && (u ≤ hi + tol)

/-- Read the second coordinate, accepting either `y` for 2D data or `t` for 1D-in-time data. -/
def getSecondCoordinate (j : Json) : Except String Float := do
  let o ← TorchLean.Json.expectObject "dataset point" j
  match Std.TreeMap.Raw.get? o "y" with
  | some _ => NN.Verification.Json.parseFieldFiniteFloat "dataset point" "y" j
  | none => NN.Verification.Json.parseFieldFiniteFloat "dataset point" "t" j

/-- Parse one dataset point as `(x, y-or-t, u)`. -/
def parsePoint (j : Json) : Except String Point := do
  let x ← NN.Verification.Json.parseFieldFiniteFloat "dataset point" "x" j
  let yOrT ← getSecondCoordinate j
  let u ← NN.Verification.Json.parseFieldFiniteFloat "dataset point" "u" j
  pure { x := x, yOrT := yOrT, u := u }

/-- Load one named dataset section into checked PINN sample points. -/
def loadSection (path : String) (sectionName : String) : IO (Array Point) := do
  let j ← NN.Verification.Json.readJsonFile path
  let arr ← match NN.Verification.Json.fieldArrayOrEmpty "dataset" sectionName j with
    | .ok a => pure a
    | .error msg => throw <| IO.userError s!"Dataset.{sectionName}: {msg}"
  let mut out : Array Point := #[]
  for entry in arr do
    match parsePoint entry with
    | .ok point => out := out.push point
    | .error msg => throw <| IO.userError s!"Dataset.{sectionName}: {msg}"
  pure out

end NN.Verification.PINN.Dataset
