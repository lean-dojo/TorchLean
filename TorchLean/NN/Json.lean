/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Shape
public import FloatLib.Floats.Formats.BinaryInterchange.Rounding.Mode
public import Lean.Data.Json.FromToJson.Basic
public import Lean.Data.Json.Parser
import all Lean.Data.Json.Basic
import Std.Data.TreeMap.Raw.Lemmas
import all Lean.Data.Json.FromToJson.Basic

/-!
# JSON of artifacts

The JSON readers that the artifact importers, certificate checkers, training logs and RL bridges
share, and the JSON writer of rounding directions. A reader's error messages name its `ctx`
argument, the caller's prefix if any and the path of the value read, such as
`"RL boundary: transition.action"`.

`nat`, `string`, `array`, `object`, `field`, `nats`, `shape`, `shapes` and the `...Field` forms
read naturals, strings, arrays, objects, fields and shapes and name the JSON they rejected, and
`exactObject` reads an object with exactly the given keys. The three readers of naturals differ.
`nat` reads every JSON number whose value is a natural number (`3.0` too), `expectNat` a number
without decimal places or a decimal string, and `parseNatStrict` the naturals that a conversion to
binary64 and then to `UInt64` returns unchanged. `asFloat?` reads a float from a number or a
numeric string, `parseFiniteFloat` and its array and matrix forms require it finite,
`float` also reads the strings `NaN`, `Infinity` and `-Infinity`, and `floatField` reads a
number field. `readFile` reads and parses a JSON file. `rounding` reads the rounding direction of
a conversion, every `IEEERoundingMode` constructor by its name in `ofRounding`.

`rounding_ofRounding` proves that `rounding` returns what `ofRounding` wrote, and `nat_toJson`,
`nats_toJson` and `shape_toJson` prove the same for the Lean JSON of naturals and shapes. The round
trips are stated for JSON values, not for their text.
-/

@[expose] public section

namespace TorchLean.Json

open Lean
open FloatLib.Floats.Formats.BinaryInterchange

/-- A uniform "wrong JSON shape" error, so import failures all read the same way. -/
def typeError {α : Type} (ctx expected : String) (j : Json) : Except String α :=
  .error s!"JSON: {ctx}: expected {expected}, got {j}"

/-- Read a JSON number whose value is a nonnegative integer, such as `3` or `3.0`, as a natural
number. -/
def nat (ctx : String) : Json → Except String Nat
  | .num n =>
      let rec
        /-- `some (value / 10 ^ k)` for the second argument `k` when `10 ^ k` divides `value`, and
        `none` otherwise. -/
        integral (value : Nat) : Nat → Option Nat
          | 0 => some value
          | places + 1 =>
              if value = 0 then some 0
              else if value % 10 = 0 then integral (value / 10) places
              else none
      match if n.mantissa < 0 then none else integral n.mantissa.toNat n.exponent with
      | some value => .ok value
      | none => .error s!"JSON: {ctx}: expected natural number, got {n}"
  | j => typeError ctx "natural number" j

/-- Decode a JSON value as a string, reporting the importing context on mismatch. -/
def string (ctx : String) : Json → Except String String
  | .str s => .ok s
  | j => typeError ctx "string" j

/-- Decode a JSON value as an array, reporting the importing context on mismatch. -/
def array (ctx : String) : Json → Except String (Array Json)
  | .arr xs => .ok xs
  | j => typeError ctx "array" j

/-- Decode a JSON value as an object. -/
def object (ctx : String) : Json → Except String (Std.TreeMap.Raw String Json)
  | .obj o => .ok o
  | j => typeError ctx "object" j

/-- Read a required object field. -/
def field (ctx key : String) (o : Std.TreeMap.Raw String Json) : Except String Json :=
  match o.get? key with
  | some j => .ok j
  | none => .error s!"JSON: {ctx}: missing field `{key}`"

/-- Parse a JSON array of natural numbers. -/
def nats (ctx : String) (j : Json) : Except String (Array Nat) := do
  let xs ← array ctx j
  xs.mapM (nat ctx)

/--
Parse a shape encoded as a dimension list.

Examples:
- `[]` means scalar,
- `[4]` means `Shape.dim 4 Shape.scalar`,
- `[2, 3]` means `Shape.dim 2 (Shape.dim 3 Shape.scalar)`.
-/
def shape (ctx : String) (j : Json) : Except String Shape := do
  pure (← nats ctx j).toList

/-- Read a natural-number field of an object. -/
def natField (ctx key : String) (o : Std.TreeMap.Raw String Json) :
    Except String Nat := do
  nat s!"{ctx}.{key}" (← field ctx key o)

/-- Read a Boolean field of an object. -/
def boolField (ctx key : String) (o : Std.TreeMap.Raw String Json) :
    Except String Bool := do
  match ← field ctx key o with
  | .bool b => pure b
  | bad => typeError s!"{ctx}.{key}" "boolean" bad

/-- Read a shape-valued field of an object. -/
def shapeField (ctx key : String) (o : Std.TreeMap.Raw String Json) :
    Except String Shape := do
  shape s!"{ctx}.{key}" (← field ctx key o)

/-- Read a natural-number array field of an object. -/
def natArrayField (ctx key : String) (o : Std.TreeMap.Raw String Json) :
    Except String (Array Nat) := do
  nats s!"{ctx}.{key}" (← field ctx key o)

/-- Read a number field of an object as a `Float`. -/
def floatField (ctx key : String) (o : Std.TreeMap.Raw String Json) : Except String Float := do
  match ← field ctx key o with
  | .num value => pure value.toFloat
  | bad => typeError s!"{ctx}.{key}" "number" bad

/-- Parse a JSON array whose elements are shape arrays. -/
def shapes (ctx : String) (j : Json) : Except String (Array Shape) := do
  let xs ← array ctx j
  xs.mapM (shape ctx)

/-- Decode a JSON object whose keys are exactly the keys `keys`, which must be distinct. -/
def exactObject (ctx : String) (keys : List String) (json : Json) :
    Except String (Std.TreeMap.Raw String Json) := do
  let o ← object ctx json
  unless keys.Nodup && o.size = keys.length && keys.all o.contains do
    throw s!"IR import: {ctx}: expected exactly the keys {keys}, got {json.compress}"
  pure o

/-- Read a natural number from a JSON number without decimal places (`Json.getNat?`) or from a
decimal string (`String.toNat?`). -/
def expectNat (ctx : String) (j : Json) : Except String Nat :=
  match j.getNat?, j with
  | .ok n, _ => .ok n
  | _, .str text =>
      match text.toNat? with
      | some n => .ok n
      | none => typeError ctx "natural number" j
  | _, _ => typeError ctx "natural number" j

/-- Read a natural number from a JSON number whose value a conversion to binary64 and then to
`UInt64` returns unchanged: an integer below `2^64` that binary64 represents exactly, or
`2^64 - 1`, to which the conversion to `UInt64` saturates. -/
def parseNatStrict (ctx : String) (j : Json) : Except String Nat :=
  match j with
  | .num n =>
      let f := n.toFloat
      if f.isNaN || f.isInf then
        .error s!"JSON: {ctx}: expected integer JSON number, got {n}"
      else if f < 0 then
        .error s!"JSON: {ctx}: expected nonnegative integer, got {n}"
      else
        let u := f.toUInt64
        if (Float.ofNat u.toNat) == f &&
            n.normalize == (JsonNumber.fromNat u.toNat).normalize then
          .ok (u.toNat)
        else
          .error s!"JSON: {ctx}: expected integer JSON number, got {n}"
  | _ => typeError ctx "JSON number" j

/-- The `Float` of a JSON number, or of a string holding a JSON number. A negative zero reads as
`+0.0` (`Lean.JsonNumber` stores an `Int` mantissa). -/
def asFloat? : Json → Option Float
  | .num n => some n.toFloat
  | .str s =>
      match Json.parse s with
      | .ok (.num n) => some n.toFloat
      | _ => none
  | _ => none

/-- Read a finite `Float` from a JSON number or a string holding one (`asFloat?`). -/
def parseFiniteFloat (ctx : String) (j : Json) : Except String Float :=
  match (asFloat? j).filter Float.isFinite with
  | some x => .ok x
  | none => .error s!"JSON: {ctx}: expected finite float"

/-- Read an array of finite floats (`parseFiniteFloat`), naming each entry's index in errors. -/
def parseFiniteFloatArray (ctx : String) (j : Json) : Except String (Array Float) := do
  (← array ctx j).mapIdxM fun i x => parseFiniteFloat s!"{ctx}[{i}]" x

/-- Read an array of rows of finite floats (`parseFiniteFloatArray`). -/
def parseFiniteFloatMatrix (ctx : String) (j : Json) : Except String (Array (Array Float)) := do
  (← array ctx j).mapIdxM fun i row => parseFiniteFloatArray s!"{ctx}[{i}]" row

/-- Read a `Float` from a JSON number, a string holding a JSON number (`asFloat?`), or one of the
strings `"NaN"`, `"Infinity"` and `"-Infinity"`. -/
def float (ctx : String) (j : Json) : Except String Float :=
  match j, asFloat? j with
  | .str "NaN", _ => .ok (0.0 / 0.0)
  | .str "Infinity", _ => .ok (1.0 / 0.0)
  | .str "-Infinity", _ => .ok (-1.0 / 0.0)
  | _, some x => .ok x
  | _, none => typeError ctx "number" j

/-- Read and parse the JSON file at `path`. -/
def readFile (path : System.FilePath) : IO Json := do
  match Json.parse (← IO.FS.readFile path) with
  | .ok j => pure j
  | .error message => throw <| IO.userError s!"{path}: invalid JSON: {message}"

/-- The artifact name of a rounding direction: `nearestEven`, `nearestAway`, `towardZero`,
`towardPositiveInfinity` and `towardNegativeInfinity` for IEEE 754's rounding-direction attributes,
and `awayFromZero` for `away`, which is MPFR's `MPFR_RNDA` and has no IEEE 754 attribute. -/
def ofRounding : Model.IEEERoundingMode → Json
  | .nearest => "nearestEven"
  | .nearestAway => "nearestAway"
  | .towardZero => "towardZero"
  | .up => "towardPositiveInfinity"
  | .down => "towardNegativeInfinity"
  | .away => "awayFromZero"

/-- Read a rounding direction by its `ofRounding` name. Any other value fails, so an unknown
direction never falls back to a default. -/
def rounding (value : Json) : Except String Model.IEEERoundingMode := do
  match ← value.getStr? with
  | "nearestEven" => pure .nearest
  | "nearestAway" => pure .nearestAway
  | "towardZero" => pure .towardZero
  | "towardPositiveInfinity" => pure .up
  | "towardNegativeInfinity" => pure .down
  | "awayFromZero" => pure .away
  | name => throw s!"IR import: unsupported rounding direction {name}"

/-- Every rounding direction reads back from its name. -/
theorem rounding_ofRounding (mode : Model.IEEERoundingMode) :
    rounding (ofRounding mode) = .ok mode := by
  cases mode <;> rfl

/-- A field of an object built from a list with distinct keys is its value in the list. -/
theorem field_ofList {ctx k : String} {l : List (String × Json)} {v : Json}
    (distinct : l.Pairwise (fun a b => ¬ compare a.1 b.1 = .eq)) (mem : (k, v) ∈ l) :
    field ctx k (Std.TreeMap.Raw.ofList l compare) = .ok v := by
  unfold field
  rw [Std.TreeMap.Raw.get?_eq_getElem?,
    Std.TreeMap.Raw.getElem?_ofList_of_mem (k := k) (Std.ReflCmp.compare_self) distinct mem]

/-- An object built from a list with distinct keys has exactly the keys of the list. -/
theorem exactObject_ofList {ctx : String} {l : List (String × Json)} {keys : List String}
    (distinct : l.Pairwise (fun a b => ¬ compare a.1 b.1 = .eq)) (hkeys : l.map Prod.fst = keys) :
    exactObject ctx keys (.obj (Std.TreeMap.Raw.ofList l compare)) =
      .ok (Std.TreeMap.Raw.ofList l compare) := by
  subst hkeys
  have hnodup : (l.map Prod.fst).Nodup := by
    rw [List.Nodup, List.pairwise_map]
    exact distinct.imp fun h heq => h (by rw [heq]; exact Std.ReflCmp.compare_self)
  have hsize := Std.TreeMap.Raw.size_ofList distinct
  have hall : ((l.map Prod.fst).all fun k => (Std.TreeMap.Raw.ofList l compare).contains k) =
      true := by
    simp only [List.all_eq_true, Std.TreeMap.Raw.contains_ofList]
    intro k hk
    simpa using hk
  simp [exactObject, object, hnodup, hsize, hall, bind, Except.bind, pure, Except.pure]

/-- A natural number reads back from its JSON. -/
theorem nat_toJson (ctx : String) (n : Nat) : nat ctx (toJson n) = .ok n := by
  have h : ¬ ((n : Int) < 0) := by omega
  simp [nat, toJson, JsonNumber.fromNat, nat.integral, h]

/-- The JSON of a list is the array of its entries' JSON. -/
theorem toJson_list {β : Type} [ToJson β] (l : List β) :
    toJson l = Json.arr (l.map toJson).toArray := by
  simp [toJson, Lean.List.toJson, Lean.Array.toJson]

/-- A list of naturals reads back from its JSON. -/
theorem nats_toJson (ctx : String) (l : List Nat) : nats ctx (toJson l) = .ok l.toArray := by
  rw [toJson_list]
  simp only [nats, array, bind, Except.bind, List.mapM_toArray, List.mapM_map]
  rw [show nat ctx ∘ toJson = (pure ·) from funext (nat_toJson ctx), List.mapM_pure]
  simp [Functor.map, Except.map, pure, Except.pure]

/-- A shape reads back from the JSON of its extents. -/
theorem shape_toJson (ctx : String) (s : Shape) : shape ctx (toJson s) = .ok s := by
  simp only [shape, bind, Except.bind]
  rw [nats_toJson]
  simp [pure, Except.pure]

/-- A one-key object lists its one entry. -/
theorem toList_ofList_singleton (k : String) (v : Json) :
    (Std.TreeMap.Raw.ofList [(k, v)] compare).toList = [(k, v)] := rfl

end TorchLean.Json
