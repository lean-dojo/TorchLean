/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Subprocess
public import TorchLeanTest.Harness
public import TorchLean.Tensor.Operations
public import TorchLean.Tensor.Tactic.Init

/-!
# Array, JSON and Python helpers

Helpers shared by the `Float` runtime checks: `assertArrayApprox` compares arrays entry by entry,
`jsonFloatArrayField` reads a JSON array of numbers, and `pythonHasTorch` asks whether Python can
import PyTorch.
-/

@[expose] public section


open TorchLean.Tensor
open Lean

namespace TorchLean.Test
namespace Harness

/-- Throw unless both arrays have the same length and each pair of entries is finite and within
`tol`. -/
def assertArrayApprox (label : String) (got expected : Array Float) (tol : Float := 2e-5) :
    IO Unit := do
  unless got.size = expected.size do
    throw (IO.userError s!"{label}: length mismatch {got.size} vs {expected.size}")
  for i in [0:got.size] do
    assertApprox s!"{label}[{i}]" got[i]! expected[i]! tol

/-- Parse a JSON field containing a flat array of floats. -/
def jsonFloatArrayField (j : Json) (key : String) : Except String (Array Float) := do
  let arr ←
    match ← j.getObjVal? key with
    | .arr xs => pure xs
    | other => throw s!"field `{key}` was not an array: {other}"
  arr.mapM fun
    | .num n => pure n.toFloat
    | other => throw s!"field `{key}` contained non-number: {other}"

/-- Whether `python3` can import `torch`. A missing PyTorch is an error when
`TORCHLEAN_REQUIRE_INTEROP=1`. -/
def pythonHasTorch : IO Bool := do
  checkInteropDependency "torch"
    (← Subprocess.pythonCanImport #["torch"])

end Harness
end TorchLean.Test
