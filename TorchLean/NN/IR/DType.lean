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
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic
import all Lean.Data.Name

/-!
# IR element types

An IR value has a tensor shape and an element type. The `DType` descriptors identify the element
representation without choosing a device, storage buffer or kernel. Binary and decimal descriptors
reuse FloatLib's validated formats, so a configured binary128 or custom binary format is a `binary`
descriptor with no IR constructor of its own.

`context` keeps Lean graphs scalar-polymorphic: the caller chooses their element type through the
`Storage` and `Context` instances. Captured mixed-dtype programs name each element type with a
descriptor.

Named types are the extension point for a frontend or backend with another representation. A
descriptor carries no arithmetic, so an interpreter registers the semantics of a named type or
rejects execution.
-/

@[expose] public section

namespace TorchLean.IR

/-- Device-independent element representation of a tensor value. -/
inductive DType where
  /-- Scalar type supplied by a homogeneous Lean graph's caller. -/
  | context
  /-- Boolean values, distinct from numeric zero/one tensors. -/
  | bool
  /-- Signed or unsigned integers with a positive bit `width`, or unbounded when it is `none`. -/
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
  /-- Complex values whose real and imaginary components share the element type `component`. -/
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

/-- Whether an element type can be the scalar of a step of a typed program: a valid descriptor
other than a Boolean or an integer, which are port sorts there. -/
def isScalar (d : DType) : Bool :=
  d.valid && match d with
    | .bool | .integer .. => false
    | _ => true

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

section

open Lean

/-- Encode a name as the array of its string and numeric components, so that every component,
escaped or numeric, survives the round trip (`nameOfJson_nameJson`). -/
def nameJson (name : Name) : Json :=
  .arr <| name.components.toArray.map fun part => match part with
    | .str _ text => toJson text
    | .num _ value => toJson value
    | .anonymous => .null

/-- Decode a name from the component array that `nameJson` writes. -/
def nameOfJson (value : Json) : Except String Name := do
  match value with
  | .arr parts =>
      parts.foldlM (init := Name.anonymous) fun parent part => do
        match part with
        | .str text => pure (.str parent text)
        | .num _ => pure (.num parent (← part.getNat?))
        | _ => throw "IR element type: a name component must be a string or natural number"
  | _ => throw "IR element type: expected a name component array"

mutual
  /-- JSON nesting depth, computed without the partial printer or proof-only `SizeOf` code. -/
  def jsonDepth : Json → Nat
    | .arr ⟨values⟩ => jsonListDepth values + 1
    | .obj ⟨⟨fields⟩⟩ => jsonTreeDepth fields + 1
    | _ => 1
  termination_by value => sizeOf value

  /-- Maximum nesting among an array's entries. -/
  def jsonListDepth : List Json → Nat
    | [] => 0
    | value :: values => max (jsonDepth value) (jsonListDepth values)
  termination_by values => sizeOf values

  /-- Maximum nesting among an object's values. Keys add no level of their own. -/
  def jsonTreeDepth : Std.DTreeMap.Internal.Impl String (fun _ => Json) → Nat
    | .leaf => 0
    | .inner _ _ value left right =>
      max (jsonDepth value) (max (jsonTreeDepth left) (jsonTreeDepth right))
  termination_by fields => sizeOf fields
end

/-- The number of keys of a descriptor object of `kind`. -/
def keyCount : String → Nat
  | "binary" => 5
  | "decimal" => 4
  | "integer" | "named" => 3
  | "posit" | "complex" => 2
  | _ => 0

/-- The binary format of the given fields, checked as `FloatFormat` requires. -/
def binaryFormat (exponent fraction bias : Nat) (encoding : FloatFormat.Encoding) :
    Except String FloatFormat :=
  if he : 2 ≤ exponent then
    if hf : 0 < fraction then
      if hb : 0 < bias then
        if hm : bias ≤ encoding.maxFiniteExponent exponent then
          pure ⟨exponent, he, fraction, hf, bias, encoding, hb, hm⟩
        else throw "IR element type: binary bias exceeds the finite exponent range"
      else throw "IR element type: binary bias must be positive"
    else throw "IR element type: binary fraction width must be positive"
  else throw "IR element type: binary exponent width must be at least two"

/-- The fields of a format rebuild it. -/
theorem binaryFormat_fields (format : FloatFormat) :
    binaryFormat format.expWidth format.fracWidth format.exponentBias format.encoding =
      .ok format := by
  simp [binaryFormat, format.expWidth_ge_two, format.fracWidth_pos, format.exponentBias_pos,
    format.exponentBias_le_maxFinite, pure, Except.pure]

/-- Decode a descriptor, nested components included, within fuel that callers derive from the
JSON's depth. A descriptor object has exactly the keys its kind names. -/
def read : Nat → Json → Except String DType
  | 0, _ => .error "IR element type: nesting exceeds the descriptor's structural depth"
  | fuel + 1, value => do
      match value with
      | .str "context" => pure .context
      | .str "float16" => pure float16
      | .str "bfloat16" => pure bfloat16
      | .str "float32" => pure float32
      | .str "float64" => pure float64
      | .str "bool" => pure .bool
      | .str "rational" => pure .rational
      | .str "real" => pure .real
      | .obj fields =>
          let kind ← (← value.getObjVal? "kind").getStr?
          if fields.size != keyCount kind then
            throw s!"IR element type: a {kind} descriptor has exactly {keyCount kind} keys"
          match kind with
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
              pure (.binary (← binaryFormat exponent fraction bias encoding))
          | "decimal" =>
              let declets ← (← value.getObjVal? "declets").getNat?
              let exponent ← (← value.getObjVal? "exponent_bits").getNat?
              let bias ← (← value.getObjVal? "bias").getInt?
              if h : 0 < exponent then
                pure (.decimal ⟨declets, exponent, bias, h⟩)
              else throw "IR element type: decimal exponent width must be positive"
          | "integer" =>
              let signed ← (← value.getObjVal? "signed").getBool?
              let width ← match ← value.getObjVal? "width" with
                | .null => pure none
                | width => pure (some (← width.getNat?))
              pure (.integer signed width)
          | "posit" => pure (.posit (← (← value.getObjVal? "bits").getNat?))
          | "complex" => pure (.complex (← read fuel (← value.getObjVal? "component")))
          | "named" =>
              let identifier ← nameOfJson (← value.getObjVal? "name")
              let parameters ← (← (← value.getObjVal? "parameters").getArr?).mapM Json.getNat?
              pure (.named identifier parameters)
          | other => throw s!"IR element type: unknown descriptor kind {other}"
      | _ => throw s!"IR element type: unsupported descriptor {value}"

end

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
      ("kind", .str "named"), ("name", nameJson name),
      ("parameters", Lean.toJson parameters)]

section

open Lean

/-- A value of an object is less deeply nested than the object. -/
theorem jsonDepth_lt_of_get? {o : Std.TreeMap.Raw String Json compare} {k : String} {v : Json}
    (h : o.get? k = some v) : jsonDepth v < jsonDepth (.obj o) := by
  rcases o with ⟨⟨t⟩⟩
  have key : ∀ t : Std.DTreeMap.Internal.Impl String (fun _ => Json),
      Std.DTreeMap.Internal.Impl.Const.get? t k = some v → jsonDepth v ≤ jsonTreeDepth t := by
    intro t
    induction t with
    | leaf => simp [Std.DTreeMap.Internal.Impl.Const.get?]
    | inner _ k' v' l r ihl ihr =>
        intro hget
        simp only [Std.DTreeMap.Internal.Impl.Const.get?] at hget
        rw [jsonTreeDepth]
        split at hget
        · have := ihl hget; omega
        · have := ihr hget; omega
        · cases hget; omega
  have := key t h
  rw [jsonDepth]
  omega

/-- `nameOfJson` decodes the JSON written by `nameJson` to the same name. -/
theorem nameOfJson_nameJson (n : Name) : nameOfJson (nameJson n) = .ok n := by
  simp only [nameOfJson, nameJson, List.map_toArray, List.foldlM_toArray]
  induction n with
  | anonymous => rfl
  | str m s ih =>
      simp only [Name.components, Name.componentsRev, List.reverse_cons, List.map_append,
        List.foldlM_append] at ih ⊢
      rw [ih]
      rfl
  | num m k ih =>
      simp only [Name.components, Name.componentsRev, List.reverse_cons, List.map_append,
        List.foldlM_append] at ih ⊢
      rw [ih]
      rfl

/-- The structural reader returns the descriptor of its JSON at any fuel above its depth. -/
theorem read_toJson : (d : DType) → (fuel : Nat) → jsonDepth d.toJson ≤ fuel →
    read (fuel + 1) d.toJson = .ok d
  | .context, _, _ => rfl
  | .bool, _, _ => rfl
  | .rational, _, _ => rfl
  | .real, _, _ => rfl
  | .integer _ width, _, _ => by cases width <;> rfl
  | .posit _, _, _ => rfl
  | .binary format, _, _ => by
      have h := binaryFormat_fields format
      rcases format with ⟨e, he, f, hf, b, encoding, hb, hm⟩
      cases encoding <;>
      · show (binaryFormat e f b _ >>= fun x => pure (DType.binary x)) = _
        rw [h]
        rfl
  | .decimal ⟨declets, exponent, bias, h⟩, _, _ => by
      show (if h' : 0 < exponent then pure (DType.decimal ⟨declets, exponent, bias, h'⟩)
        else throw _) = _
      simp only [h, ↓reduceDIte]
      rfl
  | .complex component, fuel, hdepth => by
      have hlt : jsonDepth component.toJson < jsonDepth (DType.complex component).toJson :=
        jsonDepth_lt_of_get? (k := "component") rfl
      obtain ⟨fuel, rfl⟩ : ∃ f, fuel = f + 1 := ⟨fuel - 1, by omega⟩
      show (read (fuel + 1) component.toJson >>= fun x => pure (DType.complex x)) = _
      rw [read_toJson component fuel (by omega)]
      rfl
  | .named identifier parameters, _, _ => by
      have key : ∀ l : List Nat, (l.map Lean.toJson).mapM Json.getNat? = .ok l := by
        intro l
        induction l with
        | nil => rfl
        | cons x xs ih => simp only [List.map_cons, List.mapM_cons, ih]; rfl
      have hp : (parameters.map Lean.toJson).mapM Json.getNat? = .ok parameters := by
        rcases parameters with ⟨l⟩
        simp only [List.map_toArray, List.mapM_toArray, key]
        rfl
      show (nameOfJson (nameJson identifier) >>= fun identifier =>
        (parameters.map Lean.toJson).mapM Json.getNat? >>= fun parameters =>
          pure (DType.named identifier parameters)) = _
      rw [nameOfJson_nameJson, hp]
      rfl

end

/-- Decode a descriptor structurally, without validating it, refusing an object with a key its
kind does not name. -/
def parse (value : Lean.Json) : Except String DType :=
  read (jsonDepth value + 1) value

/-- Parsing the JSON of any descriptor returns it. -/
theorem parse_toJson (d : DType) : parse d.toJson = .ok d :=
  read_toJson d _ (Nat.le_refl _)

/-- Decode and validate a descriptor without selecting a numerical implementation. -/
def fromJson (value : Lean.Json) : Except String DType := do
  let dtype ← parse value
  dtype.check
  pure dtype

/-- Successful decoding cannot introduce invalid element-type parameters. -/
theorem fromJson_valid {value : Lean.Json} {dtype : DType}
    (h : fromJson value = .ok dtype) : dtype.valid = true := by
  unfold fromJson at h
  cases hRead : parse value with
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

end TorchLean.IR
