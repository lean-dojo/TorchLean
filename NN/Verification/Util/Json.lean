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

/-- Turn an `Except String α` parser result into `IO`, preserving the parser's error text. -/
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
  TorchLean.Json.readFile (System.FilePath.mk path)

/-- Ensure a JSON value is an object. -/
def expectObject (j : Json) (ctx : String) : IO Json := do
  match TorchLean.Json.expectObject ctx j with
  | .ok _ => pure j
  | .error e => throw <| IO.userError e

/-- Extract a required field from a JSON object. -/
def expectField (j : Json) (k : String) (ctx : String) : IO Json := do
  match TorchLean.Json.expectField ctx k j with
  | .ok v => pure v
  | .error e => throw <| IO.userError e

/-- Read a JSON artifact and require the top-level value to be an object. -/
def readJsonObjectFile (path : String) (ctx : String := "top-level") : IO Json := do
  let j ← readJsonFile path
  expectObject j ctx

/-- Extract an optional field from a JSON object. -/
def optionalField? (j : Json) (k : String) (ctx : String) : IO (Option Json) := do
  let o ← fromExcept <| TorchLean.Json.expectObject ctx j
  pure <| Std.TreeMap.Raw.get? o k

/-- Require a JSON string in an `IO` parser, preserving contextual error messages. -/
def expectString (j : Json) (ctx : String) : IO String :=
  fromExcept <| TorchLean.Json.expectString ctx j

/-- Require a natural number, accepting either JSON numeric syntax or a decimal string. -/
def expectNat (j : Json) (ctx : String) : IO Nat :=
  fromExcept <| TorchLean.Json.expectNat ctx j

/-- Require a JSON array and return its entries. -/
def expectArray (j : Json) (ctx : String) : IO (Array Json) :=
  fromExcept <| TorchLean.Json.expectArray ctx j

/-- Parse a `Nat` from a JSON number or decimal string. -/
def asNat? (j : Json) : Option Nat :=
  match TorchLean.Json.expectNat "Nat" j with
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

/-- Parse a finite `Float` from a JSON number or a string containing a JSON number. -/
def asFiniteFloat? (j : Json) : Option Float := do
  let x ← asFloat? j
  if x.isFinite then
    some x
  else
    none

/-- Parse a floating-point value with contextual errors. -/
def parseFloat (ctx : String) (j : Json) : Except String Float :=
  match asFloat? j with
  | some x => pure x
  | none => throw s!"{ctx}: expected float"

/-- Parse a finite floating-point value with contextual errors. -/
def parseFiniteFloat (ctx : String) (j : Json) : Except String Float :=
  match asFiniteFloat? j with
  | some x => pure x
  | none => throw s!"{ctx}: expected finite float"

/-- Parse a JSON array of finite floats with contextual errors. -/
def parseFiniteFloatArray (ctx : String) (j : Json) : Except String (Array Float) := do
  let xs ← TorchLean.Json.expectArray ctx j
  xs.mapIdxM fun i x => parseFiniteFloat s!"{ctx}[{i}]" x

/-- A finite axis-aligned region parsed from a verification artifact. -/
structure BoxRegion where
  /-- Declared dimension, or the inferred endpoint-array length when `dim` is absent. -/
  dim : Nat
  /-- Lower coordinate bounds. -/
  lo : Array Float
  /-- Upper coordinate bounds. -/
  hi : Array Float
  deriving Repr

namespace BoxRegion

/-- Check the dimensional and ordering invariants of an axis-aligned region. -/
def validate (ctx : String) (region : BoxRegion) : Except String Unit := do
  unless region.lo.size = region.dim && region.hi.size = region.dim do
    throw <| s!"{ctx}: dimension {region.dim} does not match endpoint lengths " ++
      s!"{region.lo.size} and {region.hi.size}"
  for i in [0:region.dim] do
    let lo := region.lo[i]!
    let hi := region.hi[i]!
    unless lo.isFinite && hi.isFinite && lo <= hi do
      throw s!"{ctx}: invalid interval at coordinate {i}: [{lo}, {hi}]"

end BoxRegion

/--
Parse either `{lo, hi}` or `{center, eps}` notation for a finite axis-aligned region.

When `dim` is absent, it is inferred from the endpoint or center array. When present, it must be a
natural number equal to the resulting endpoint lengths. The parser also rejects negative radii,
non-finite values, incomplete schemas, and intervals whose lower endpoint exceeds the upper one.
Keeping these checks here gives certificate consumers one well-formed region type instead of
several subtly different parsers.
-/
def parseBoxRegion (ctx : String) (j : Json) : Except String BoxRegion := do
  let obj <- TorchLean.Json.expectObject ctx j
  let declaredDim? <- match Std.TreeMap.Raw.get? obj "dim" with
    | none => pure none
    | some dimJson => some <$> TorchLean.Json.expectNat s!"{ctx}.dim" dimJson
  let lo? := Std.TreeMap.Raw.get? obj "lo"
  let hi? := Std.TreeMap.Raw.get? obj "hi"
  let center? := Std.TreeMap.Raw.get? obj "center"
  let eps? := Std.TreeMap.Raw.get? obj "eps"
  let region <- match lo?, hi?, center?, eps? with
  | some loJson, some hiJson, none, none =>
      let lo ← parseFiniteFloatArray s!"{ctx}.lo" loJson
      let hi ← parseFiniteFloatArray s!"{ctx}.hi" hiJson
      pure { dim := declaredDim?.getD lo.size, lo, hi }
  | none, none, some centerJson, some epsJson =>
      let center ← parseFiniteFloatArray s!"{ctx}.center" centerJson
      let radius ← parseFiniteFloat s!"{ctx}.eps" epsJson
      pure
        { dim := declaredDim?.getD center.size
          lo := center.map (· - radius)
          hi := center.map (· + radius) }
  | some _, none, _, _ =>
      throw s!"{ctx}: field `lo` requires a matching `hi` field"
  | none, some _, _, _ =>
      throw s!"{ctx}: field `hi` requires a matching `lo` field"
  | some _, some _, _, _ =>
      throw s!"{ctx}: endpoint fields (`lo`, `hi`) cannot be combined with `center` or `eps`"
  | none, none, some _, none =>
      throw s!"{ctx}: field `center` requires a matching `eps` field"
  | none, none, none, some _ =>
      throw s!"{ctx}: field `eps` requires either `center` or endpoint fields"
  | none, none, none, none =>
      throw s!"{ctx}: expected either (`lo`, `hi`) or (`center`, `eps`)"
  region.validate ctx
  pure region

/-- Parse the exact endpoint schema `{lo, hi}`, with an optional matching `dim` field. -/
def parseEndpointBoxRegion (ctx : String) (j : Json) : Except String BoxRegion := do
  let obj ← TorchLean.Json.expectObject ctx j
  if (Std.TreeMap.Raw.get? obj "center").isSome || (Std.TreeMap.Raw.get? obj "eps").isSome then
    throw s!"{ctx}: expected endpoint fields (`lo`, `hi`), not `center` or `eps`"
  parseBoxRegion ctx j

/-- Parse a finite floating-point-valued field with contextual errors. -/
def parseFieldFiniteFloat (ctx key : String) (j : Json) : Except String Float := do
  parseFiniteFloat s!"{ctx}.{key}" (← TorchLean.Json.expectField ctx key j)

/-- Parse a string-valued field with contextual errors. -/
def parseFieldString (ctx key : String) (j : Json) : Except String String := do
  TorchLean.Json.expectString s!"{ctx}.{key}" (← TorchLean.Json.expectField ctx key j)

/-- Decode a JSON boolean if the value is exactly `true` or `false`. -/
def parseBool? (j : Json) : Option Bool :=
  match j with
  | .bool b => some b
  | _ => none

/-- Require a finite floating-point value, accepting JSON numbers and string-encoded numbers. -/
def expectFiniteFloat (j : Json) (ctx : String) : IO Float := do
  match asFiniteFloat? j with
  | some x => pure x
  | none => throw <| IO.userError s!"{ctx}: expected finite float"

/-- Require a JSON boolean and report `ctx` on mismatch. -/
def expectBool (j : Json) (ctx : String) : IO Bool := do
  match parseBool? j with
  | some b => pure b
  | none => throw <| IO.userError s!"{ctx}: expected boolean"

/-- Parse a JSON array of floats. -/
def parseFloatArray (j : Json) : Option (Array Float) :=
  match j with
  | .arr xs => xs.mapM asFloat?
  | _ => none

/-- Parse a JSON matrix represented as an array of float arrays. -/
def parseFloatMatrix (j : Json) : Option (Array (Array Float)) := do
  match j with
  | .arr rows => rows.mapM parseFloatArray
  | _ => none

/-- Parse a JSON array of finite floats with contextual errors. -/
def expectFiniteFloatArray (j : Json) (ctx : String) : IO (Array Float) := do
  let xs ← expectArray j ctx
  xs.mapIdxM fun i x => expectFiniteFloat x s!"{ctx}[{i}]"


/-- Parse a JSON matrix whose entries are all finite floats. -/
def expectFiniteFloatMatrix (j : Json) (ctx : String) : IO (Array (Array Float)) := do
  let rows ← expectArray j ctx
  rows.mapIdxM fun i row => expectFiniteFloatArray row s!"{ctx}[{i}]"

/-- Extract an object-valued field. -/
def expectFieldObject (j : Json) (k : String) (ctx : String) : IO Json := do
  let v ← expectField j k ctx
  expectObject v s!"{ctx}.{k}"

/-- Extract a string-valued field. -/
def expectFieldString (j : Json) (k : String) (ctx : String) : IO String := do
  expectString (← expectField j k ctx) s!"{ctx}.{k}"

/-- Extract a natural-number-valued field. -/
def expectFieldNat (j : Json) (k : String) (ctx : String) : IO Nat := do
  expectNat (← expectField j k ctx) s!"{ctx}.{k}"

/-- Extract an array-valued field. -/
def expectFieldArray (j : Json) (k : String) (ctx : String) : IO (Array Json) := do
  expectArray (← expectField j k ctx) s!"{ctx}.{k}"

/-- Extract an array field, using `#[]` when it is absent or null. -/
def fieldArrayOrEmpty (ctx key : String) (j : Json) : Except String (Array Json) := do
  let o ← TorchLean.Json.expectObject ctx j
  match Std.TreeMap.Raw.get? o key with
  | none => pure #[]
  | some .null => pure #[]
  | some (.arr xs) => pure xs
  | some _ => throw s!"{ctx}.{key}: expected array"

/-- Extract a finite-float-array-valued field. -/
def expectFieldFiniteFloatArray (j : Json) (k : String) (ctx : String) : IO (Array Float) := do
  expectFiniteFloatArray (← expectField j k ctx) s!"{ctx}.{k}"

/-- Extract an optional natural-number-valued field. -/
def optionalFieldNat? (j : Json) (k : String) (ctx : String) : IO (Option Nat) := do
  match ← optionalField? j k ctx with
  | none => pure none
  | some v => some <$> expectNat v s!"{ctx}.{k}"

/-- Extract an optional finite floating-point-valued field. -/
def optionalFieldFiniteFloat? (j : Json) (k : String) (ctx : String) : IO (Option Float) := do
  match ← optionalField? j k ctx with
  | none => pure none
  | some v => some <$> expectFiniteFloat v s!"{ctx}.{k}"

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

end NN.Verification.Json
