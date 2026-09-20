/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Lean.Data.Json.Parser

/-!
# JSON Parsing

Conservative JSON helpers shared by artifact loaders and verification tools.

The functions here fail fast with contextual messages. This module stays focused:
not a schema library, just the common parsing substrate for TorchLean JSON artifacts.
-/

@[expose] public section

namespace TorchLean.Json

open Lean
open Lean.Json

/-- Parse a JSON value as an object. -/
def expectObject (context : String) (value : Lean.Json) :
    Except String (Std.TreeMap.Raw String Lean.Json compare) := do
  match Lean.Json.getObj? value with
  | .ok object => pure object
  | .error message => throw s!"{context}: expected object ({message})"

/-- Extract a required field from a JSON object. -/
def expectField (context key : String) (value : Lean.Json) : Except String Lean.Json := do
  let object ← expectObject context value
  match Std.TreeMap.Raw.get? object key with
  | some fieldValue => pure fieldValue
  | none => throw s!"{context}: missing field `{key}`"

/-- Require a JSON string and report `context` in the error message on mismatch. -/
def expectString (context : String) (value : Lean.Json) : Except String String := do
  match Lean.Json.getStr? value with
  | .ok text => pure text
  | .error message => throw s!"{context}: expected string ({message})"

/-- Parse a JSON natural number, accepting either a JSON number or a decimal string. -/
def expectNat (context : String) (value : Lean.Json) : Except String Nat := do
  match Lean.Json.getNat? value with
  | .ok n => pure n
  | .error _ =>
      match value with
      | .str text =>
          match text.toNat? with
          | some n => pure n
          | none => throw s!"{context}: expected natural number"
      | _ => throw s!"{context}: expected natural number"

/-- Require a JSON array and return its entries. -/
def expectArray (context : String) (value : Lean.Json) : Except String (Array Lean.Json) := do
  match value with
  | .arr entries => pure entries
  | _ => throw s!"{context}: expected array"

/-- Read and parse a JSON file from disk. -/
def readFile (path : System.FilePath) : IO Lean.Json := do
  let contents ← IO.FS.readFile path
  match Lean.Json.parse contents with
  | .ok value => pure value
  | .error message => throw <| IO.userError s!"{path}: invalid JSON: {message}"

end TorchLean.Json
