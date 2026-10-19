/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor
public import NN.Core.ExternalProcess
public import NN.Tests.Utils

/-!
# Floats Utils

Shared helpers for the Float runtime checks.

These helpers share array comparisons, JSON decoding, and dependency checks.
-/

@[expose] public section


open Spec TorchLean
open TorchLean TorchLean.Tensor
open Lean

namespace Tests
namespace Floats
namespace Utils

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

/-- Check PyTorch availability, failing when required interop checks are enabled. -/
def pythonHasTorch : IO Bool := do
  Tests.Utils.checkInteropDependency "torch"
    (← TorchLean.External.Process.pythonCanImport #["torch"])

end Utils
end Floats
end Tests
