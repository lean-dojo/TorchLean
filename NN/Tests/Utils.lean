/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Std

/-!
# Shared Test Support

Finite floating-point assertions, optional-bound checks, interop dependency policy, and parameter
identifiers shared by the runtime suites. This module depends only on `Std` so each suite can import
it independently.
-/

@[expose] public section

namespace Tests
namespace Utils

/-- Fail on a missing Python dependency when CI requests `TORCHLEAN_REQUIRE_INTEROP=1`. -/
def checkInteropDependency (name : String) (available : Bool) : IO Bool := do
  if !available && (← IO.getEnv "TORCHLEAN_REQUIRE_INTEROP") == some "1" then
    throw <| IO.userError <|
      s!"required interop dependency {name} is unavailable in python3; " ++
      "install scripts/checks/requirements-interop.txt in the active Python environment"
  pure available

/-- Reject `NaN` and infinities. -/
def assertFinite (msg : String) (x : Float) : IO Unit := do
  if x.isNaN || x.isInf then
    throw <| IO.userError s!"{msg}: expected finite, got {x}"

/-- Compare finite values with an absolute tolerance; non-finite inputs always fail. -/
def assertApprox (msg : String) (x y : Float) (tol : Float := 1e-5) : IO Unit := do
  if x.isNaN || x.isInf || y.isNaN || y.isInf then
    throw <| IO.userError s!"{msg}: expected finite values, got {x} and {y}"
  if Float.abs (x - y) > tol then
    throw <| IO.userError s!"{msg}: got {x}, expected {y} (tol={tol})"

/-- Require an existing node whose optional bound is `none`; a missing index fails. -/
def assertNoBoundAt {β : Type} (label : String) (values : Array (Option β)) (id : Nat) : IO Unit :=
  match values[id]? with
  | some none => pure ()
  | _ => throw <| IO.userError s!"{label}: expected node {id} to have no bound"

/-- Check that node `id` of a bound array carries a bound. -/
def assertBoundAt {β : Type} (label : String) (values : Array (Option β)) (id : Nat) : IO Unit :=
  match values[id]? with
  | some (some _) => pure ()
  | _ => throw <| IO.userError s!"{label}: expected node {id} to have a bound"

/-- Parameter node identifiers shared by the floating-point and rational two-layer MLP tests. -/
structure ParamIds where
  /-- Tape id of the first layer's weight matrix. -/
  hiddenWeightId : Nat
  /-- Tape id of the first layer's bias vector. -/
  hiddenBiasId : Nat
  /-- Tape id of the second layer's weight matrix. -/
  outputWeightId : Nat
  /-- Tape id of the second layer's bias vector. -/
  outputBiasId : Nat

end Utils
end Tests
