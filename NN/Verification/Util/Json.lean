/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Json

/-!
# Json

Shared JSON helpers for TorchLean verification tools.

Many verification workflows consume small JSON “certificates” produced by Python tooling
(often PyTorch-based). This module centralizes:
- “shape” checks (object + field existence),
- simple scalar parsing (Nat/Float/Bool),
- small array helpers used across checkers.

These helpers take a strict stance: malformed certificates fail fast with contextual error
messages instead of silently defaulting.
-/

@[expose] public section


namespace NN.Verification.Json

open Lean
open Json

def fromExcept {α : Type} (x : Except String α) : IO α := do
  match x with
  | .ok a => pure a
  | .error e => throw <| IO.userError e

/--
Read and parse a JSON verification artifact from disk.

Use this at checker boundaries instead of repeating `IO.FS.readFile` and `Json.parse` in every
tool. The file path is included in parse errors.
-/
def readJsonFile (path : String) : IO Json :=
  NN.API.Json.parseFile (System.FilePath.mk path)

/-- Ensure a JSON value is an object. -/
def expectObj (j : Json) (ctx : String) : IO Json := do
  match NN.API.Json.expectObjE ctx j with
  | .ok _ => pure j
  | .error e => throw <| IO.userError e

/-- Extract a required field from a JSON object. -/
def expectField (j : Json) (k : String) (ctx : String) : IO Json := do
  match NN.API.Json.expectFieldE ctx k j with
  | .ok v => pure v
  | .error e => throw <| IO.userError e

/-- Read a JSON artifact and require the top-level value to be an object. -/
def readJsonObjectFile (path : String) (ctx : String := "top-level") : IO Json := do
  let j ← readJsonFile path
  expectObj j ctx

/-- Extract an optional field from a JSON object. -/
def optionalField? (j : Json) (k : String) (ctx : String) : IO (Option Json) := do
  let o ← fromExcept <| NN.API.Json.expectObjE ctx j
  pure <| Std.TreeMap.Raw.get? o k

/-- Parse a JSON string. -/
def expectString (j : Json) (ctx : String) : IO String :=
  fromExcept <| NN.API.Json.expectStringE ctx j

/-- Parse a JSON natural number. -/
def expectNat (j : Json) (ctx : String) : IO Nat :=
  fromExcept <| NN.API.Json.expectNatE ctx j

/-- Parse a JSON array. -/
def expectArray (j : Json) (ctx : String) : IO (Array Json) :=
  fromExcept <| NN.API.Json.expectArrayE ctx j

/-- Parse a `Nat` from a JSON number or decimal string. -/
def asNat? (j : Json) : Option Nat :=
  match NN.API.Json.expectNatE "Nat" j with
  | .ok n => some n
  | .error _ => none

/-- Parse a `Float` from a JSON number or a string containing a JSON number. -/
def asFloat? (j : Json) : Option Float :=
  match j with
  | .num n => some n.toFloat
  | .str s =>
      match Json.parse s with
      | .ok (.num n) => some n.toFloat
      | _ => none
  | _ => none

/-- Parse a JSON boolean. -/
def parseBool? (j : Json) : Option Bool :=
  match j with
  | .bool b => some b
  | _ => none

/-- Parse a JSON float. -/
def expectFloat (j : Json) (ctx : String) : IO Float := do
  match asFloat? j with
  | some x => pure x
  | none => throw <| IO.userError s!"{ctx}: expected float"

/-- Parse a JSON boolean. -/
def expectBool (j : Json) (ctx : String) : IO Bool := do
  match parseBool? j with
  | some b => pure b
  | none => throw <| IO.userError s!"{ctx}: expected boolean"

/-- Parse a JSON array of floats. -/
def parseFloatArray (j : Json) : Option (Array Float) :=
  match j with
  | .arr xs => xs.mapM asFloat?
  | _ => none

/-- Parse a JSON array of floats with contextual errors. -/
def expectFloatArray (j : Json) (ctx : String) : IO (Array Float) := do
  match parseFloatArray j with
  | some xs => pure xs
  | none => throw <| IO.userError s!"{ctx}: expected float array"

/-- Extract an object-valued field. -/
def expectFieldObj (j : Json) (k : String) (ctx : String) : IO Json := do
  let v ← expectField j k ctx
  expectObj v s!"{ctx}.{k}"

/-- Extract a string-valued field. -/
def expectFieldString (j : Json) (k : String) (ctx : String) : IO String := do
  expectString (← expectField j k ctx) s!"{ctx}.{k}"

/-- Extract a natural-number-valued field. -/
def expectFieldNat (j : Json) (k : String) (ctx : String) : IO Nat := do
  expectNat (← expectField j k ctx) s!"{ctx}.{k}"

/-- Extract an array-valued field. -/
def expectFieldArray (j : Json) (k : String) (ctx : String) : IO (Array Json) := do
  expectArray (← expectField j k ctx) s!"{ctx}.{k}"

/-- Extract a float-array-valued field. -/
def expectFieldFloatArray (j : Json) (k : String) (ctx : String) : IO (Array Float) := do
  expectFloatArray (← expectField j k ctx) s!"{ctx}.{k}"

/-- Extract an optional string-valued field. -/
def optionalFieldString? (j : Json) (k : String) (ctx : String) : IO (Option String) := do
  match ← optionalField? j k ctx with
  | none => pure none
  | some v => some <$> expectString v s!"{ctx}.{k}"

/-- Extract an optional natural-number-valued field. -/
def optionalFieldNat? (j : Json) (k : String) (ctx : String) : IO (Option Nat) := do
  match ← optionalField? j k ctx with
  | none => pure none
  | some v => some <$> expectNat v s!"{ctx}.{k}"

/-- Extract an optional boolean-valued field. -/
def optionalFieldBool? (j : Json) (k : String) (ctx : String) : IO (Option Bool) := do
  match ← optionalField? j k ctx with
  | none => pure none
  | some v => some <$> expectBool v s!"{ctx}.{k}"

/--
Require a top-level `format` field to match an expected artifact schema string.

This makes schema checks uniform across verification tools and keeps examples from hand-rolling
their own unsupported-format errors.
-/
def expectFormat (j : Json) (expected : String) (ctx : String := "top-level") : IO Unit := do
  let fmt ← expectFieldString j "format" ctx
  if fmt != expected then
    throw <| IO.userError s!"{ctx}.format: unsupported format `{fmt}` (expected `{expected}`)"

/-- Pointwise `all` on two float arrays of equal length. -/
def all2 (a b : Array Float) (p : Float → Float → Bool) : Bool :=
  if _h : a.size = b.size then
    (List.finRange a.size).all (fun i => p (a[i]!) (b[i]!))
  else
    false

/-- Pointwise `any` on two float arrays of equal length. -/
def any2 (a b : Array Float) (p : Float → Float → Bool) : Bool :=
  if _h : a.size = b.size then
    (List.finRange a.size).any (fun i => p (a[i]!) (b[i]!))
  else
    false

end NN.Verification.Json
