/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Lean.Data.Json.Parser

/-!
# API Json

Small conservative JSON helpers shared by public artifact loaders and verification tools.

The functions here fail fast with contextual messages. This module is intentionally small:
not a schema library, just the common parsing substrate for TorchLean JSON artifacts.
-/

@[expose] public section

namespace NN
namespace API
namespace Json

open Lean
open Lean.Json

/-- Throw an `Except String` parse error. -/
def fail {α : Type} (msg : String) : Except String α :=
  throw msg

/-- Parse a JSON value as an object. -/
def expectObjE (ctx : String) (j : Lean.Json) :
    Except String (Std.TreeMap.Raw String Lean.Json compare) := do
  match Lean.Json.getObj? j with
  | .ok o => pure o
  | .error e => fail s!"{ctx}: expected object ({e})"

/-- Extract a required field from a JSON object. -/
def expectFieldE (ctx key : String) (j : Lean.Json) : Except String Lean.Json := do
  let o ← expectObjE ctx j
  match Std.TreeMap.Raw.get? o key with
  | some v => pure v
  | none => fail s!"{ctx}: missing field `{key}`"

/-- Parse a JSON string. -/
def expectStringE (ctx : String) (j : Lean.Json) : Except String String := do
  match Lean.Json.getStr? j with
  | .ok s => pure s
  | .error e => fail s!"{ctx}: expected string ({e})"

/-- Parse a JSON natural number, accepting either a JSON number or a decimal string. -/
def expectNatE (ctx : String) (j : Lean.Json) : Except String Nat := do
  match Lean.Json.getNat? j with
  | .ok n => pure n
  | .error _ =>
      match j with
      | .str s =>
          match s.toNat? with
          | some n => pure n
          | none => fail s!"{ctx}: expected natural number"
      | _ => fail s!"{ctx}: expected natural number"

/-- Parse a JSON array. -/
def expectArrayE (ctx : String) (j : Lean.Json) : Except String (Array Lean.Json) := do
  match j with
  | .arr xs => pure xs
  | _ => fail s!"{ctx}: expected array"

/-- Parse an optional JSON boolean field with a default. -/
def optionalBoolFieldE (ctx key : String) (default : Bool) (j : Lean.Json) :
    Except String Bool := do
  let o ← expectObjE ctx j
  match Std.TreeMap.Raw.get? o key with
  | none => pure default
  | some (.bool b) => pure b
  | some _ => fail s!"{ctx}.{key}: expected boolean"

/-- Parse a JSON array of natural numbers. -/
def expectNatArrayE (ctx : String) (j : Lean.Json) : Except String (Array Nat) := do
  let xs ← expectArrayE ctx j
  xs.mapIdxM (fun i x => expectNatE s!"{ctx}[{i}]" x)

/-- Parse a JSON file from disk. -/
def parseFile (path : System.FilePath) : IO Lean.Json := do
  let s ← IO.FS.readFile path
  match Lean.Json.parse s with
  | .ok j => pure j
  | .error e => throw <| IO.userError s!"{path}: invalid JSON: {e}"

end Json
end API
end NN
