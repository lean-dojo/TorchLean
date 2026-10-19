/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.BinaryInterchange.Format.Catalog
public import FloatLib.Floats.Formats.DecimalInterchange.Format
public import Lean.Data.Json.FromToJson.Basic
public import Lean.Data.Name

/-!
# IR element types

An IR value has a tensor shape and an element type. These descriptors identify
the element representation without choosing a device, storage buffer, or kernel.
Binary and decimal descriptors reuse FloatLib's validated formats. A configured
binary128 or custom binary format is not a separate IR constructor.

`context` preserves scalar-polymorphic Lean graphs: the caller chooses their
element type through the existing `Storage` and `Context` instances. Captured
mixed-dtype programs use explicit descriptors instead.

Named types provide an extension point for a frontend or backend with another
representation. Describing a type does not supply its arithmetic implementation;
an interpreter must register the corresponding semantics or reject execution.
-/

@[expose] public section

namespace NN.IR

/-- Device-independent element representation of a tensor value. -/
inductive DType where
  /-- Scalar type supplied by a homogeneous Lean graph's caller. -/
  | context
  /-- Boolean values, distinct from numeric zero/one tensors. -/
  | bool
  /-- Signed or unsigned integers; `none` means unbounded, and a width must be positive. -/
  | integer (signed : Bool) (width : Option Nat := none)
  /-- A validated FloatLib binary format, including its bias and exceptional-value encoding. -/
  | binary (format : FloatLib.Floats.Formats.BinaryInterchange.FloatFormat)
  /-- A validated FloatLib decimal interchange format. -/
  | decimal (format : FloatLib.Floats.Formats.DecimalInterchange.Format)
  /-- A Posit Standard encoding of at least two bits. -/
  | posit (bits : Nat)
  /-- Exact rational values. -/
  | rational
  /-- Mathematical real values, for specification interpreters. -/
  | real
  /-- Real and imaginary components have the same explicitly described element type. -/
  | complex (component : DType)
  /-- An extension identified by a qualified name and static natural-number parameters. -/
  | named (name : Lean.Name) (parameters : Array Nat := #[])
  deriving DecidableEq, Repr, Inhabited

namespace DType

open FloatLib.Floats.Formats.BinaryInterchange (FloatFormat)

/-- IEEE binary16. -/
def float16 : DType := .binary FloatFormat.binary16

/-- Brain floating point with eight exponent bits and seven fraction bits. -/
def bfloat16 : DType := .binary FloatFormat.bfloat16

/-- IEEE binary32. -/
def float32 : DType := .binary FloatFormat.binary32

/-- IEEE binary64. -/
def float64 : DType := .binary FloatFormat.binary64

/-- Whether parameters not already validated by FloatLib form a usable descriptor. -/
def valid : DType → Bool
  | .integer _ (some width) => 0 < width
  | .posit bits => 2 ≤ bits
  | .complex component => component.valid
  | .named name _ => !name.isAnonymous
  | _ => true

/-- Check descriptor parameters before accepting a graph or registering an implementation. -/
def check (dtype : DType) : Except String Unit :=
  if dtype.valid then .ok ()
  else .error s!"invalid IR element type: {repr dtype}"

/-- Compact type spelling for errors and graph displays, including custom format parameters. -/
def describe : DType → String
  | .context => "context"
  | .bool => "bool"
  | .integer signed width =>
      let base := if signed then "int" else "uint"
      match width with
      | some bits => s!"{base}{bits}"
      | none => if signed then "Int" else "Nat"
  | .binary format =>
      if format = FloatFormat.binary16 then "float16"
      else if format = FloatFormat.bfloat16 then "bfloat16"
      else if format = FloatFormat.binary32 then "float32"
      else if format = FloatFormat.binary64 then "float64"
      else s!"binary({format.expWidth},{format.fracWidth}," ++
        s!"bias={format.exponentBias},encoding={repr format.encoding})"
  | .decimal format =>
      s!"decimal(declets={format.declets},exponentBits={format.exponentBits},bias={format.bias})"
  | .posit bits => s!"posit{bits}"
  | .rational => "Rat"
  | .real => "Real"
  | .complex component => s!"complex({component.describe})"
  | .named name parameters => s!"{name}{repr parameters}"

instance : ToString DType where
  toString := describe

namespace Internal

open Lean

/-- Preserve numeric and escaped name components, rather than parsing a display spelling. -/
def nameJson (name : Name) : Json :=
  .arr <| name.components.toArray.map fun part => match part with
    | .str _ text => toJson text
    | .num _ value => toJson value
    | .anonymous => .null

/-- Component arrays are lossless; qualified strings are also accepted from frontend producers. -/
def name (value : Json) : Except String Name := do
  match value with
  | .str text =>
      if text == "[anonymous]" then return .anonymous
      -- Keep Lean's identifier and escape rules, but bound recursion by the source length.
      -- The core string-to-name parser is partial and cannot reduce in an import proof.
      let rec parse : Nat → List Char → Name → Option Name
        | 0, _, _ => none
        | fuel + 1, chars, parent => do
            let (component, rest) ← match chars with
              | '«' :: chars =>
                  let (part, rest) := chars.span (· != '»')
                  match rest with
                  | '»' :: rest => some (Name.str parent (String.ofList part), rest)
                  | _ => none
              | char :: chars =>
                  if Lean.isIdFirst char then
                    let (part, rest) := chars.span Lean.isIdRest
                    some (Name.str parent (String.ofList (char :: part)), rest)
                  else if char.isDigit then
                    let (part, rest) := chars.span Char.isDigit
                    let number := (char :: part).foldl
                      (fun n digit => n * 10 + (digit.toNat - '0'.toNat)) 0
                    some (Name.num parent number, rest)
                  else none
              | [] => none
            match rest with
            | [] => some component
            | '.' :: rest => parse fuel rest component
            | _ => none
      let chars := text.toList
      match parse (chars.length + 1) chars .anonymous with
      | some name => pure name
      | none => throw s!"expected a `Name`, got '{value}'"
  | .arr parts =>
      parts.foldlM (init := Name.anonymous) fun parent part => do
        match part with
        | .str text => pure (.str parent text)
        | .num _ => pure (.num parent (← part.getNat?))
        | _ => throw "IR element type: a name component must be a string or natural number"
  | _ => throw "IR element type: expected a name string or component array"

mutual
  /-- JSON nesting depth, computed without the partial printer or proof-only `SizeOf` code. -/
  def depth : Json → Nat
    | .arr ⟨values⟩ => listDepth values + 1
    | .obj ⟨⟨fields⟩⟩ => treeDepth fields + 1
    | _ => 1
  termination_by value => sizeOf value

  /-- Maximum nesting among an array's entries. -/
  def listDepth : List Json → Nat
    | [] => 0
    | value :: values => max (depth value) (listDepth values)
  termination_by values => sizeOf values

  /-- Maximum nesting among an object's values; keys do not add a component level. -/
  def treeDepth : Std.DTreeMap.Internal.Impl String (fun _ => Json) → Nat
    | .leaf => 0
    | .inner _ _ value left right => max (depth value) (max (treeDepth left) (treeDepth right))
  termination_by fields => sizeOf fields
end

/-- Decode nested components with a budget derived from the JSON structure, not a fixed depth. -/
def read : Nat → Json → Except String DType
  | 0, _ => .error "IR element type: nesting exceeds the descriptor's structural depth"
  | fuel + 1, value => do
      match value with
      | .str "context" => pure .context
      | .str "float16" => pure float16
      | .str "bfloat16" => pure bfloat16
      | .str "float32" => pure float32
      | .str "float64" => pure float64
      | .str "float128" => pure (.binary FloatFormat.binary128)
      | .str "bool" => pure .bool
      | .str "rational" => pure .rational
      | .str "real" => pure .real
      | .obj _ =>
          match ← (← value.getObjVal? "kind").getStr? with
          | "binary" =>
              let exponent ← (← value.getObjVal? "exponent_bits").getNat?
              let fraction ← (← value.getObjVal? "fraction_bits").getNat?
              let bias ← (← value.getObjVal? "bias").getNat?
              let encoding ← match ← (← value.getObjVal? "encoding").getStr? with
                | "ieee" => pure FloatFormat.Encoding.ieee
                | "finiteMaxNaN" => pure .finiteMaxNaN
                | "finiteUnsignedZero" => pure .finiteUnsignedZero
                | "finite" => pure .finite
                | other => throw s!"IR element type: unknown binary encoding {other}"
              if he : 2 ≤ exponent then
                if hf : 0 < fraction then
                  if hb : 0 < bias then
                    if hm : bias ≤ encoding.maxFiniteExponent exponent then
                      pure (.binary ⟨exponent, he, fraction, hf, bias, encoding, hb, hm⟩)
                    else throw "IR element type: binary bias exceeds the finite exponent range"
                  else throw "IR element type: binary bias must be positive"
                else throw "IR element type: binary fraction width must be positive"
              else throw "IR element type: binary exponent width must be at least two"
          | "decimal" =>
              let declets ← (← value.getObjVal? "declets").getNat?
              let exponent ← (← value.getObjVal? "exponent_bits").getNat?
              let bias ← (← value.getObjVal? "bias").getInt?
              if h : 0 < exponent then
                pure (.decimal ⟨declets, exponent, bias, h⟩)
              else throw "IR element type: decimal exponent width must be positive"
          | "integer" =>
              let signed ← (← value.getObjVal? "signed").getBool?
              let width ← match value.getObjVal? "width" with
                | .error _ | .ok .null => pure none
                | .ok width => pure (some (← width.getNat?))
              pure (.integer signed width)
          | "posit" => pure (.posit (← (← value.getObjVal? "bits").getNat?))
          | "complex" => pure (.complex (← read fuel (← value.getObjVal? "component")))
          | "named" =>
              let identifier ← name (← value.getObjVal? "name")
              let parameters ← (← (← value.getObjVal? "parameters").getArr?).mapM Json.getNat?
              pure (.named identifier parameters)
          | other => throw s!"IR element type: unknown descriptor kind {other}"
      | _ => throw s!"IR element type: unsupported descriptor {value}"

end Internal

/-- Encode the complete element representation, including format bias and name components. -/
def toJson : DType → Lean.Json
  | .context => .str "context"
  | .bool => .str "bool"
  | .integer signed width => Lean.Json.mkObj [
      ("kind", .str "integer"), ("signed", Lean.toJson signed), ("width", Lean.toJson width)]
  | .binary format => Lean.Json.mkObj [
      ("kind", .str "binary"), ("exponent_bits", Lean.toJson format.expWidth),
      ("fraction_bits", Lean.toJson format.fracWidth), ("bias", Lean.toJson format.exponentBias),
      ("encoding", .str (match format.encoding with
        | .ieee => "ieee"
        | .finiteMaxNaN => "finiteMaxNaN"
        | .finiteUnsignedZero => "finiteUnsignedZero"
        | .finite => "finite"))]
  | .decimal format => Lean.Json.mkObj [
      ("kind", .str "decimal"), ("declets", Lean.toJson format.declets),
      ("exponent_bits", Lean.toJson format.exponentBits), ("bias", Lean.toJson format.bias)]
  | .posit bits => Lean.Json.mkObj [("kind", .str "posit"), ("bits", Lean.toJson bits)]
  | .rational => .str "rational"
  | .real => .str "real"
  | .complex component => Lean.Json.mkObj [
      ("kind", .str "complex"), ("component", component.toJson)]
  | .named name parameters => Lean.Json.mkObj [
      ("kind", .str "named"), ("name", Internal.nameJson name),
      ("parameters", Lean.toJson parameters)]

/-- Decode and validate a descriptor without selecting a numerical implementation. -/
def fromJson (value : Lean.Json) : Except String DType := do
  let dtype ← Internal.read (Internal.depth value + 1) value
  dtype.check
  pure dtype

/-- Successful decoding cannot introduce invalid element-type parameters. -/
theorem fromJson_valid {value : Lean.Json} {dtype : DType}
    (h : fromJson value = .ok dtype) : dtype.valid = true := by
  unfold fromJson at h
  cases hRead : Internal.read (Internal.depth value + 1) value with
  | error message => simp [hRead, Bind.bind, Except.bind] at h
  | ok parsed =>
      simp only [hRead, Bind.bind, Except.bind] at h
      cases hValid : parsed.valid with
      | false => simp [check, hValid] at h
      | true =>
          have hParsed : parsed = dtype := by
            simpa [check, hValid, Pure.pure, Except.pure, Bind.bind, Except.bind] using h
          cases hParsed
          exact hValid

instance : Lean.ToJson DType := ⟨toJson⟩
instance : Lean.FromJson DType := ⟨fromJson⟩

end DType

end NN.IR
