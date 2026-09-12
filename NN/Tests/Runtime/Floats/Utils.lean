/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Lean.Data.Json
public import NN.Spec.Core.Tensor
public import NN.Tensor
public import NN.Core.ExternalProcess
public import NN.Tests.Utils
public import Std

/-!
# Floats Utils

Shared helpers for the Float runtime checks.

These helpers keep the curated test files focused on the checked behavior instead of re-declaring
the same tensor accessors and approximate equality checks.
-/

@[expose] public section


open Spec TorchLean
open TorchLean TorchLean.Tensor
open Lean

namespace Tests
namespace Floats
namespace Utils

/-!
`assertApprox` and `assertFinite` are not defined here. Both are in `Tests.Utils`, which the CUDA
suites can also reach, and this module's `1e-5` default moved there unchanged, so the `assertApprox`
calls in this directory mean exactly what they meant before. Files that `open Tests.Floats.Utils`
for the bare spelling now open `Tests.Utils` alongside it.
-/

/-- Approximate equality for same-length float arrays. -/
def assertArrayApprox (label : String) (got expected : Array Float) (tol : Float := 2e-5) :
    IO Unit := do
  unless got.size = expected.size do
    throw (IO.userError s!"{label}: length mismatch {got.size} vs {expected.size}")
  for i in [0:got.size] do
    Tests.Utils.assertApprox s!"{label}[{i}]" got[i]! expected[i]! tol

/-- Parse a JSON field containing a flat array of floats. -/
def jsonFloatArrayField (j : Json) (key : String) : Except String (Array Float) := do
  let arr ←
    match ← j.getObjVal? key with
    | .arr xs => pure xs
    | other => throw s!"field `{key}` was not an array: {other}"
  arr.mapM fun
    | .num n => pure n.toFloat
    | other => throw s!"field `{key}` contained non-number: {other}"

/-- Whether the active Python environment can import PyTorch. -/
def pythonHasTorch : IO Bool := do
  TorchLean.External.Process.pythonCanImport #["torch"]

/-- Read the scalar payload from a scalar tensor. -/
def scalarVal (t : Tensor Float Shape.scalar) : Float :=
  t.item

/-- Read one coordinate from a vector tensor. -/
def vecVal {n : Nat} (t : Tensor Float [n]) (i : Fin n) : Float :=
  Tensor.getScalar t i

/-- Read one coordinate from a matrix tensor. -/
def matVal {rows cols : Nat} (t : Tensor Float [rows, cols])
    (i : Fin rows) (j : Fin cols) : Float :=
  Tensor.get2 t i j

/-- Read one scalar coordinate from a tensor of arbitrary rank.

This helper is intentionally rank-neutral: tests pass the coordinates they are checking rather
than introducing layout-specific accessors. An invalid coordinate is a test-authoring error and
therefore fails immediately.
-/
def tensorVal {s : Shape} (t : Tensor Float s) (indices : List Nat) : Float :=
  match Spec.getSpec t indices with
  | some value => value
  | none => panic! s!"tensor coordinate {indices} is invalid for shape {repr s}"

end Utils
end Floats
end Tests
