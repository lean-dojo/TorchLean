/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Json
public import TorchLean.NN.IR.Program
import all Lean.Data.Json.Basic
import Std.Data.TreeMap.Raw.Lemmas
import all Lean.Data.Json.FromToJson.Basic

/-!
# Operator graph artifacts (`torchlean.ir`)

An artifact (`Artifact`) is a typed program (`IR.Program`) with the words of its constants.
Its JSON is `{"format": "torchlean.ir", "nodes": [...], "outputs": [[2, 0]], "scalars": [...],
"constants": [...]}`, a step's id being its position in `nodes` and a reference `[node, out]`.
`scalars` holds one element type (`DType.toJson`) or null per step and `constants` one word array
or null per step. A step has one key naming its form. It is `{"input": {"index": 0, "port": P}}`,
`{"const": P}`, `{"op": {"record": R, "inputs": [[0, 0], [1, 0]]}}`, `{"cast": {"rounding":
"nearestEven", "source": [0, 0], "port": P}}` or `{"call": {"name": N, "inputs": [[0, 0]],
"signature": [T], "port": P}}`. The record `R` is the operator's wire record (`Op.encode`) as
`ofOpRecord` writes it, `{"version": 3, "tag": ..., "fields": [...]}` with the wire version
`Op.Wire.formatVersion`. A declared type `T` is `{"port": P, "scalar": D}`. A port is
`{"value": [2, 3]}` (or `frozen`, `detached`, `bool`) with its shape,
`{"index": {"bounds": [4], "shape": [2]}}` or `{"integer": {"signed": true, "width": 32,
"shape": [2]}}`. We read every object with exactly its keys.

`opRecord` reads a wire record from the JSON `ofOpRecord` writes, one object per `Op.Field`
naming its form (`opField` and `ofOpField`).

Decoding is structural: `Artifact.ofJson` checks the format name, decodes each record with
`Op.decode` and each element type with `DType.parse`, but checks no reference, port or word against
the steps. `Artifact.ofJson_toJson` proves that decoding the JSON of any artifact returns it, from
the round trips of the record JSON (`TorchLean.Json.opRecord_ofOpRecord`), `Op.decode_encode`,
`DType.parse_toJson` and `DType.nameOfJson_nameJson`. These round trips are about `Json` values,
so printing and parsing text lie outside them. `Artifact.load` checks an artifact under a runtime
and builds its payload: the constants from their words (`Typed.decode`), the custom programs and
calls from a registry (`Program.Env`). `Artifact.load_ok` proves that the program of an artifact it
accepts passes `Program.check` and its payload `Payload.check`.
-/

@[expose] public section

open Lean TorchLean.Json

namespace TorchLean.Json

/-- One field of a registered operator's codec record, from an object with exactly one key, the
field's form: `{"shape": [2, 3]}`, `{"shapes": [[2], []]}`, `{"nat": 1}`, `{"nats": [0, 2]}`,
`{"pairs": [[0, 1]]}`, `{"bool": true}`, `{"mask": null}` or `{"mask": [true, false]}`,
`{"name": "relu"}`, and `{"element": "scalar"}`, `{"element": "bool"}` or
`{"element": {"integer": {"signed": true, "width": 32}}}`. -/
def opField (json : Json) : Except String Op.Field := do
  let [(form, value)] := (← object "op field" json).toList
    | throw s!"IR import: an operator field has exactly one key, its form, not {json.compress}"
  match form with
  | "shape" => return .shape (← shape "op field" value)
  | "shapes" => return .shapes (← shapes "op field" value).toList
  | "nat" => return .nat (← nat "op field" value)
  | "nats" => return .nats (← nats "op field" value).toList
  | "pairs" =>
      let pairs ← (← array "op field" value).mapM fun pair => do
        let #[i, j] ← array "op field" pair
          | throw "IR import: an operator field pair has two entries"
        pure (← nat "op field" i, ← nat "op field" j)
      return .pairs pairs.toList
  | "bool" =>
      let .bool b := value | typeError "op field" "Boolean" value
      return .bool b
  | "mask" =>
      if value matches .null then return .mask none
      let bits ← (← array "op field" value).mapM fun bit => do
        let .bool b := bit | typeError "op field" "Boolean" bit
        pure b
      return .mask (some bits)
  | "name" => return .name (← string "op field" value)
  | "element" =>
      match value with
      | .str "scalar" => return .element .scalar
      | .str "bool" => return .element .bool
      | .str name => throw s!"IR import: an operator element of unknown form {name}"
      | _ =>
          let integer ← exactObject "op field.element" ["integer"] value
          let word ← exactObject "op field.element.integer" ["signed", "width"]
            (← field "op field.element" "integer" integer)
          return .element (.integer (← boolField "op field.element.integer" "signed" word)
            (← natField "op field.element.integer" "width" word))
  | form => throw s!"IR import: an operator field of unknown form {form}"

/-- The object `opField` reads for a field. -/
def ofOpField : Op.Field → Json
  | .shape value => Json.mkObj [("shape", toJson value)]
  | .shapes value => Json.mkObj [("shapes", toJson value)]
  | .nat value => Json.mkObj [("nat", toJson value)]
  | .nats value => Json.mkObj [("nats", toJson value)]
  | .pairs value => Json.mkObj [("pairs", toJson (value.map fun (i, j) => [i, j]))]
  | .bool value => Json.mkObj [("bool", toJson value)]
  | .mask none => Json.mkObj [("mask", Json.null)]
  | .mask (some value) => Json.mkObj [("mask", toJson value)]
  | .name value => Json.mkObj [("name", toJson value)]
  | .element .scalar => Json.mkObj [("element", "scalar")]
  | .element .bool => Json.mkObj [("element", "bool")]
  | .element (.integer signed width) => Json.mkObj [("element", Json.mkObj [("integer",
      Json.mkObj [("signed", toJson signed), ("width", toJson width)])])]

/-- A registered operator's codec record, `{"version": 3, "tag": "argmax", "fields": [...]}`. -/
def opRecord (json : Json) : Except String Op.Wire := do
  let o ← object "op" json
  let version ← nat "op.version" (← field "op" "version" o)
  let tag ← string "op.tag" (← field "op" "tag" o)
  let fields ← (← array "op.fields" (← field "op" "fields" o)).mapM opField
  pure ⟨version, tag, fields.toList⟩

/-- The object `opRecord` reads for a record. -/
def ofOpRecord (record : Op.Wire) : Json :=
  Json.mkObj [("version", toJson record.version), ("tag", toJson record.tag),
    ("fields", Json.arr (record.fields.map ofOpField).toArray)]

/-- A natural field reads back. -/
theorem opField_nat (n : Nat) : opField (ofOpField (.nat n)) = .ok (.nat n) := by
  unfold opField
  simp only [ofOpField, Json.mkObj, object, toList_ofList_singleton, bind, Except.bind]
  rw [nat_toJson]
  rfl

/-- A Boolean field reads back. -/
theorem opField_bool (b : Bool) : opField (ofOpField (.bool b)) = .ok (.bool b) := by
  unfold opField
  simp only [ofOpField, Json.mkObj, object, toList_ofList_singleton, bind, Except.bind]
  rfl

/-- A name field reads back. -/
theorem opField_name (s : String) : opField (ofOpField (.name s)) = .ok (.name s) := by
  unfold opField
  simp only [ofOpField, Json.mkObj, object, toList_ofList_singleton, bind, Except.bind]
  rfl

/-- A field of naturals reads back. -/
theorem opField_nats (l : List Nat) : opField (ofOpField (.nats l)) = .ok (.nats l) := by
  unfold opField
  simp only [ofOpField, Json.mkObj, object, toList_ofList_singleton, bind, Except.bind]
  rw [nats_toJson]
  simp [pure, Except.pure]

/-- A shape field reads back. -/
theorem opField_shape (s : Shape) : opField (ofOpField (.shape s)) = .ok (.shape s) := by
  unfold opField
  simp only [ofOpField, Json.mkObj, object, toList_ofList_singleton, bind, Except.bind, shape]
  rw [nats_toJson]
  simp [pure, Except.pure]

/-- A field of shapes reads back. -/
theorem opField_shapes (l : List Shape) : opField (ofOpField (.shapes l)) = .ok (.shapes l) := by
  unfold opField
  simp only [ofOpField, Json.mkObj, object, toList_ofList_singleton, bind, Except.bind,
    shapes, toJson_list, array, List.mapM_toArray, List.mapM_map]
  rw [show shape "op field" ∘ toJson = (pure ·) from
    funext (shape_toJson _), List.mapM_pure]
  simp [Functor.map, Except.map, pure, Except.pure]

/-- An absent mask reads back. -/
theorem opField_maskNone : opField (ofOpField (.mask none)) = .ok (.mask none) := by
  unfold opField
  simp only [ofOpField, Json.mkObj, object, toList_ofList_singleton, bind, Except.bind]
  rfl

/-- A field of pairs reads back. -/
theorem opField_pairs (l : List (Nat × Nat)) :
    opField (ofOpField (.pairs l)) = .ok (.pairs l) := by
  unfold opField
  dsimp only [ofOpField, Json.mkObj, object, toList_ofList_singleton, bind, Except.bind]
  rw [toJson_list]
  dsimp only [array]
  rw [List.mapM_toArray, List.mapM_map, List.mapM_map,
    congrArg (List.mapM · l) (funext fun x => (?_ : _ = pure x)), List.mapM_pure]
  · simp [Functor.map, Except.map, pure, Except.pure]
  · rw [Function.comp_apply, Function.comp_apply, toJson_list]
    rfl

/-- A mask reads back. -/
theorem opField_maskSome (a : Array Bool) :
    opField (ofOpField (.mask (some a))) = .ok (.mask (some a)) := by
  unfold opField
  rcases a with ⟨l⟩
  have ha : toJson (Array.mk l) = Json.arr (l.map toJson).toArray := by
    simp [toJson, Lean.Array.toJson]
  simp only [ofOpField, Json.mkObj, object, toList_ofList_singleton, bind, Except.bind, ha,
    array, List.mapM_toArray, List.mapM_map]
  rw [congrArg (List.mapM · l) (funext fun x => (?_ : _ = pure x)), List.mapM_pure]
  · simp [Functor.map, Except.map, pure, Except.pure]
  · cases x <;> rfl

/-- An element field reads back. -/
theorem opField_element (e : Op.Element) :
    opField (ofOpField (.element e)) = .ok (.element e) := by
  unfold opField
  cases e with
  | integer signed width =>
      have hinteger := exactObject_ofList (ctx := "op field.element") (keys := ["integer"])
        (l := [("integer", Json.mkObj [("signed", toJson signed), ("width", toJson width)])])
        (by simp) rfl
      have hword := exactObject_ofList (ctx := "op field.element.integer")
        (keys := ["signed", "width"]) (l := [("signed", toJson signed), ("width", toJson width)])
        (by simp) rfl
      have hfield : field "op field.element" "integer" (Std.TreeMap.Raw.ofList
          [("integer", Json.mkObj [("signed", toJson signed), ("width", toJson width)])] compare) =
          .ok (Json.mkObj [("signed", toJson signed), ("width", toJson width)]) :=
        field_ofList (by simp) (by simp)
      have hsigned : field "op field.element.integer" "signed" (Std.TreeMap.Raw.ofList
          [("signed", toJson signed), ("width", toJson width)] compare) = .ok (toJson signed) :=
        field_ofList (by simp) (by simp)
      have hwidth : field "op field.element.integer" "width" (Std.TreeMap.Raw.ofList
          [("signed", toJson signed), ("width", toJson width)] compare) = .ok (toJson width) :=
        field_ofList (by simp) (by simp)
      simp only [ofOpField, Json.mkObj, object, toList_ofList_singleton, bind, Except.bind] at *
      simp only [hinteger, hfield, hword, boolField, natField, hsigned, hwidth, bind, Except.bind,
        nat_toJson]
      rfl
  | _ =>
      simp only [ofOpField, Json.mkObj, object, toList_ofList_singleton, bind, Except.bind]
      rfl

/-- Every operator field reads back from the JSON `ofOpField` writes. -/
theorem opField_ofOpField (f : Op.Field) : opField (ofOpField f) = .ok f := by
  cases f with
  | shape s => exact opField_shape s
  | shapes l => exact opField_shapes l
  | nat n => exact opField_nat n
  | nats l => exact opField_nats l
  | pairs l => exact opField_pairs l
  | bool b => exact opField_bool b
  | mask m =>
      cases m with
      | none => exact opField_maskNone
      | some a => exact opField_maskSome a
  | name s => exact opField_name s
  | element e => exact opField_element e

/-- Every wire record reads back from the JSON `ofOpRecord` writes. -/
theorem opRecord_ofOpRecord (w : Op.Wire) : opRecord (ofOpRecord w) = .ok w := by
  rcases w with ⟨version, tag, fields⟩
  have hv : field "op" "version" (Std.TreeMap.Raw.ofList [("version", toJson version),
      ("tag", toJson tag), ("fields", Json.arr (fields.map ofOpField).toArray)] compare) =
      .ok (toJson version) :=
      field_ofList (by simp) (by simp)
  have ht : field "op" "tag" (Std.TreeMap.Raw.ofList [("version", toJson version),
      ("tag", toJson tag), ("fields", Json.arr (fields.map ofOpField).toArray)] compare) =
      .ok (toJson tag) :=
      field_ofList (by simp) (by simp)
  have hf : field "op" "fields" (Std.TreeMap.Raw.ofList [("version", toJson version),
      ("tag", toJson tag), ("fields", Json.arr (fields.map ofOpField).toArray)] compare) =
      .ok (Json.arr (fields.map ofOpField).toArray) :=
      field_ofList (by simp) (by simp)
  simp only [opRecord, ofOpRecord, Json.mkObj, object, bind, Except.bind, hv, ht, hf,
    nat_toJson, string, array, List.mapM_toArray, List.mapM_map]
  rw [show opField ∘ ofOpField = (pure ·) from funext opField_ofOpField, List.mapM_pure]
  simp [Functor.map, Except.map, pure, Except.pure, toJson]

end TorchLean.Json

namespace TorchLean.IR

/-- The JSON of a port, one key naming the element sort: `{"value": [2, 3]}` (or `frozen`,
`detached`, `bool`) with the shape, `{"index": {"bounds": [4], "shape": [2]}}` or
`{"integer": {"signed": true, "width": 32, "shape": [2]}}`. -/
def _root_.TorchLean.Port.toJson (p : Port) : Json :=
  match p.kind with
  | .value => Json.mkObj [("value", Lean.toJson p.shape)]
  | .frozen => Json.mkObj [("frozen", Lean.toJson p.shape)]
  | .detached => Json.mkObj [("detached", Lean.toJson p.shape)]
  | .bool => Json.mkObj [("bool", Lean.toJson p.shape)]
  | .index bounds => Json.mkObj [("index", Json.mkObj [("bounds", Lean.toJson bounds),
      ("shape", Lean.toJson p.shape)])]
  | .integer signed width => Json.mkObj [("integer", Json.mkObj [("signed", Json.bool signed),
      ("width", Lean.toJson width), ("shape", Lean.toJson p.shape)])]

/-- The port of the JSON `Port.toJson` writes. -/
def _root_.TorchLean.Port.ofJson (json : Json) : Except String Port := do
  let [(dtype, value)] := (← object "port" json).toList
    | throw s!"IR import: a port has exactly one key, its element sort, not {json.compress}"
  match dtype with
  | "value" => pure ⟨.value, ← shape "port" value⟩
  | "frozen" => pure ⟨.frozen, ← shape "port" value⟩
  | "detached" => pure ⟨.detached, ← shape "port" value⟩
  | "bool" => pure ⟨.bool, ← shape "port" value⟩
  | "index" =>
      let o ← exactObject "port" ["bounds", "shape"] value
      pure ⟨.index (← nats "port.bounds" (← field "port" "bounds" o)).toList,
        ← shape "port.shape" (← field "port" "shape" o)⟩
  | "integer" =>
      let o ← exactObject "port" ["signed", "width", "shape"] value
      let .bool signed ← field "port" "signed" o
        | throw s!"IR import: the signedness of an integer port is a Boolean"
      pure ⟨.integer signed (← nat "port.width" (← field "port" "width" o)),
        ← shape "port.shape" (← field "port" "shape" o)⟩
  | other => throw s!"IR import: unknown element sort {other}"

/-- Decoding the JSON of a port returns it. -/
theorem _root_.TorchLean.Port.ofJson_toJson (p : Port) : Port.ofJson (Port.toJson p) = .ok p := by
  rcases p with ⟨d, s⟩
  cases d with
  | index bounds =>
      have hb : field "port" "bounds" (Std.TreeMap.Raw.ofList [("bounds", Lean.toJson bounds),
          ("shape", Lean.toJson s)] compare) = .ok (Lean.toJson bounds) :=
      field_ofList (by simp) (by simp)
      have hs : field "port" "shape" (Std.TreeMap.Raw.ofList [("bounds", Lean.toJson bounds),
          ("shape", Lean.toJson s)] compare) = .ok (Lean.toJson s) :=
      field_ofList (by simp) (by simp)
      simp (disch := simp) only [Port.toJson, Port.ofJson, Json.mkObj, object,
        toList_ofList_singleton, bind, Except.bind, exactObject_ofList, hb, hs, nats_toJson,
        shape_toJson]
      simp [pure, Except.pure]
  | integer signed width =>
      have hb : field "port" "signed" (Std.TreeMap.Raw.ofList [("signed", Json.bool signed),
          ("width", Lean.toJson width), ("shape", Lean.toJson s)] compare) =
          .ok (Json.bool signed) :=
      field_ofList (by simp) (by simp)
      have hw : field "port" "width" (Std.TreeMap.Raw.ofList [("signed", Json.bool signed),
          ("width", Lean.toJson width), ("shape", Lean.toJson s)] compare) =
          .ok (Lean.toJson width) :=
      field_ofList (by simp) (by simp)
      have hs : field "port" "shape" (Std.TreeMap.Raw.ofList [("signed", Json.bool signed),
          ("width", Lean.toJson width), ("shape", Lean.toJson s)] compare) =
          .ok (Lean.toJson s) :=
      field_ofList (by simp) (by simp)
      simp (disch := simp) only [Port.toJson, Port.ofJson, Json.mkObj, object,
        toList_ofList_singleton, bind, Except.bind, exactObject_ofList, hb, hw, hs, nat_toJson,
        shape_toJson]
      simp [pure, Except.pure]
  | _ =>
      simp only [Port.toJson, Port.ofJson, Json.mkObj, object, toList_ofList_singleton, bind,
        Except.bind, shape_toJson]
      rfl

/-- The JSON of a reference: `[node, out]`. -/
def Ref.toJson (r : Ref) : Json := Json.arr #[Lean.toJson r.node, Lean.toJson r.out]

/-- The reference of the JSON `Ref.toJson` writes. -/
def Ref.ofJson (json : Json) : Except String Ref := do
  let xs ← array "ref" json
  if h : xs.size = 2 then
    pure ⟨← nat "ref" xs[0], ← nat "ref" xs[1]⟩
  else throw s!"IR import: a reference has two entries, not {json.compress}"

/-- Decoding the JSON of a reference returns it. -/
theorem Ref.ofJson_toJson (r : Ref) : Ref.ofJson r.toJson = .ok r := by
  simp only [Ref.toJson, Ref.ofJson, array, bind, Except.bind]
  simp [nat_toJson, pure, Except.pure]

/-- The JSON of a node, one key naming its form: `{"input": {"index": 0, "port": P}}`,
`{"const": P}`, or `{"op": {"record": R, "inputs": [[0, 0], [1, 0]]}}` with `R` the operator's
wire record (`ofOpRecord`). -/
def Node.toJson : Node → Json
  | .input index p => Json.mkObj [("input", Json.mkObj [("index", Lean.toJson index),
      ("port", Port.toJson p)])]
  | .const p => Json.mkObj [("const", Port.toJson p)]
  | .op o refs => Json.mkObj [("op", Json.mkObj [("record", ofOpRecord (Op.encode o)),
      ("inputs", Json.arr (refs.map Ref.toJson))])]

/-- The node of the JSON `Node.toJson` writes, its record decoded with `Op.decode`. -/
def Node.ofJson (json : Json) : Except String Node := do
  let [(form, value)] := (← object "node" json).toList
    | throw s!"IR import: a node has exactly one key, its form, not {json.compress}"
  match form with
  | "input" =>
      let o ← exactObject "node" ["index", "port"] value
      pure (.input (← nat "node.index" (← field "node" "index" o))
        (← Port.ofJson (← field "node" "port" o)))
  | "const" => pure (.const (← Port.ofJson value))
  | "op" =>
      let o ← exactObject "node" ["record", "inputs"] value
      let recordJson ← field "node" "record" o
      discard <| exactObject "op" ["version", "tag", "fields"] recordJson
      let record ← opRecord recordJson
      let some operator := Op.decode record
        | throw s!"IR import: the record {record.tag} does not decode"
      pure (.op operator (← (← array "node.inputs" (← field "node" "inputs" o)).mapM Ref.ofJson))
  | other => throw s!"IR import: unknown node form {other}"

/-- Decoding the encodings of an array returns the array when `dec` inverts `enc`. -/
theorem mapM_toJson {β : Type} {enc : β → Json} {dec : Json → Except String β}
    (h : ∀ x, dec (enc x) = .ok x) (xs : Array β) : (xs.map enc).mapM dec = .ok xs := by
  rw [Array.mapM_map, show dec ∘ enc = (pure ·) from funext h, Array.mapM_pure]
  simp [pure, Except.pure]

/-- Decoding the JSON of a node returns it. -/
theorem Node.ofJson_toJson (n : Node) : Node.ofJson n.toJson = .ok n := by
  cases n with
  | input index p =>
      have hi : field "node" "index" (Std.TreeMap.Raw.ofList [("index", Lean.toJson index),
          ("port", Port.toJson p)] compare) = .ok (Lean.toJson index) :=
      field_ofList (by simp) (by simp)
      have hp : field "node" "port" (Std.TreeMap.Raw.ofList [("index", Lean.toJson index),
          ("port", Port.toJson p)] compare) = .ok (Port.toJson p) :=
      field_ofList (by simp) (by simp)
      simp (disch := simp) only [Node.toJson, Node.ofJson, Json.mkObj, object,
        toList_ofList_singleton, bind, Except.bind, exactObject_ofList, hi, hp, nat_toJson,
        Port.ofJson_toJson]
      rfl
  | const p =>
      simp only [Node.toJson, Node.ofJson, Json.mkObj, object, toList_ofList_singleton, bind,
        Except.bind, Port.ofJson_toJson]
      rfl
  | op o refs =>
      have hr : field "node" "record" (Std.TreeMap.Raw.ofList
          [("record", ofOpRecord (Op.encode o)),
            ("inputs", Json.arr (refs.map Ref.toJson))] compare) =
          .ok (ofOpRecord (Op.encode o)) :=
      field_ofList (by simp) (by simp)
      have hin : field "node" "inputs" (Std.TreeMap.Raw.ofList
          [("record", ofOpRecord (Op.encode o)),
            ("inputs", Json.arr (refs.map Ref.toJson))] compare) =
          .ok (Json.arr (refs.map Ref.toJson)) :=
      field_ofList (by simp) (by simp)
      have hmap : (refs.map Ref.toJson).mapM Ref.ofJson = .ok refs :=
        mapM_toJson Ref.ofJson_toJson refs
      have hx : exactObject "op" ["version", "tag", "fields"]
          (ofOpRecord (Op.encode o)) = .ok _ := exactObject_ofList (by simp) rfl
      simp (disch := simp) only [Node.toJson, Node.ofJson, Json.mkObj, object,
        toList_ofList_singleton, bind, Except.bind, exactObject_ofList, hr, hin, hx,
        opRecord_ofOpRecord, Op.decode_encode, array, hmap]
      rfl

/-- The format name of artifacts. -/
def formatName : String := "torchlean.ir"

/-! ### Typed programs -/

/-- The JSON of a step's scalar: its element type, or null. -/
def scalarJson : Option DType → Json
  | some d => d.toJson
  | none => .null

/-- The scalar of the JSON `scalarJson` writes, its element type read structurally
(`DType.parse`). -/
def scalarOfJson : Json → Except String (Option DType)
  | .null => pure none
  | json => some <$> DType.parse json

/-- Decoding the JSON of a step's scalar returns it. -/
theorem scalarOfJson_scalarJson : (s : Option DType) → scalarOfJson (scalarJson s) = .ok s
  | none => rfl
  | some d => by
      have h : d.toJson ≠ .null := by cases d <;> simp [DType.toJson, Json.mkObj]
      simp only [scalarJson, scalarOfJson]
      rw [DType.parse_toJson]
      rfl

/-- The JSON of a typed port: `{"port": P, "scalar": D}`, the scalar null where it has none. -/
def Program.Typed.toJson (t : Program.Typed) : Json :=
  Json.mkObj [("port", Port.toJson t.port), ("scalar", scalarJson t.scalar)]

/-- The typed port of the JSON `Typed.toJson` writes. -/
def Program.Typed.ofJson (json : Json) : Except String Program.Typed := do
  let o ← exactObject "typed" ["port", "scalar"] json
  pure ⟨← Port.ofJson (← field "typed" "port" o), ← scalarOfJson (← field "typed" "scalar" o)⟩

/-- Decoding the JSON of a typed port returns it. -/
theorem Program.Typed.ofJson_toJson (t : Program.Typed) :
    Program.Typed.ofJson t.toJson = .ok t := by
  rcases t with ⟨port, scalar⟩
  have hp : field "typed" "port" (Std.TreeMap.Raw.ofList [("port", Port.toJson port),
      ("scalar", scalarJson scalar)] compare) = .ok (Port.toJson port) :=
    field_ofList (by simp) (by simp)
  have hs : field "typed" "scalar" (Std.TreeMap.Raw.ofList [("port", Port.toJson port),
      ("scalar", scalarJson scalar)] compare) = .ok (scalarJson scalar) :=
    field_ofList (by simp) (by simp)
  simp (disch := simp) only [Program.Typed.toJson, Program.Typed.ofJson, Json.mkObj,
    exactObject_ofList, bind, Except.bind, hp, hs,
    Port.ofJson_toJson, scalarOfJson_scalarJson]
  rfl

/-- The JSON of a step: an operator-graph node as `Node.toJson` writes it,
`{"cast": {"rounding": R, "source": [i, o], "port": P}}` with `R` one of the names of
`ofRounding`, or `{"call": {"name": N, "inputs": [[i, o]], "signature": [T], "port": P}}` with
`N` the name's components (`DType.nameJson`) and `T` the declared types. -/
def Program.Node.toJson : Program.Node → Json
  | .node n => n.toJson
  | .cast rounding source port => Json.mkObj [("cast", Json.mkObj [
      ("rounding", ofRounding rounding), ("source", source.toJson),
      ("port", Port.toJson port)])]
  | .call name args signature port => Json.mkObj [("call", Json.mkObj [
      ("name", DType.nameJson name), ("inputs", Json.arr (args.map Ref.toJson)),
      ("signature", Json.arr (signature.map Program.Typed.toJson).toArray),
      ("port", Port.toJson port)])]

/-- The step of the JSON `Program.Node.toJson` writes. -/
def Program.Node.ofJson (json : Json) : Except String Program.Node := do
  let [(form, value)] := (← object "node" json).toList
    | throw s!"IR import: a node has exactly one key, its form, not {json.compress}"
  match form with
  | "cast" =>
      let o ← exactObject "node" ["rounding", "source", "port"] value
      pure (.cast (← rounding (← field "node" "rounding" o))
        (← Ref.ofJson (← field "node" "source" o)) (← Port.ofJson (← field "node" "port" o)))
  | "call" =>
      let o ← exactObject "node" ["name", "inputs", "signature", "port"] value
      let nameJson ← field "node" "name" o
      discard <| array "node.name" nameJson
      let name ← DType.nameOfJson nameJson
      let args ← (← array "node.inputs" (← field "node" "inputs" o)).mapM Ref.ofJson
      let signature ← (← array "node.signature" (← field "node" "signature" o)).mapM
        Program.Typed.ofJson
      pure (.call name args signature.toList (← Port.ofJson (← field "node" "port" o)))
  | _ => .node <$> IR.Node.ofJson json

/-- On the JSON of an operator-graph node, `Program.Node.ofJson` is `IR.Node.ofJson` wrapped in
`.node`. -/
theorem Program.Node.ofJson_node (n : IR.Node) :
    Program.Node.ofJson n.toJson = .node <$> IR.Node.ofJson n.toJson := by
  cases n <;> rfl

/-- Decoding the JSON of a step returns it. -/
theorem Program.Node.ofJson_toJson (n : Program.Node) :
    Program.Node.ofJson n.toJson = .ok n := by
  cases n with
  | node n =>
      rw [show (Program.Node.node n).toJson = n.toJson from rfl, Program.Node.ofJson_node,
        IR.Node.ofJson_toJson]
      rfl
  | cast rounding source port =>
      have hr : field "node" "rounding" (Std.TreeMap.Raw.ofList [
          ("rounding", ofRounding rounding), ("source", source.toJson),
          ("port", Port.toJson port)] compare) = .ok (ofRounding rounding) :=
        field_ofList (by simp) (by simp)
      have hs : field "node" "source" (Std.TreeMap.Raw.ofList [
          ("rounding", ofRounding rounding), ("source", source.toJson),
          ("port", Port.toJson port)] compare) = .ok source.toJson :=
        field_ofList (by simp) (by simp)
      have hp : field "node" "port" (Std.TreeMap.Raw.ofList [
          ("rounding", ofRounding rounding), ("source", source.toJson),
          ("port", Port.toJson port)] compare) = .ok (Port.toJson port) :=
        field_ofList (by simp) (by simp)
      simp (disch := simp) only [Program.Node.toJson, Program.Node.ofJson, Json.mkObj, object,
        toList_ofList_singleton, exactObject_ofList, bind,
        Except.bind, hr, hs, hp, rounding_ofRounding, Ref.ofJson_toJson, Port.ofJson_toJson]
      rfl
  | call name args signature port =>
      have hn : field "node" "name" (Std.TreeMap.Raw.ofList [
          ("name", DType.nameJson name), ("inputs", Json.arr (args.map Ref.toJson)),
          ("signature", Json.arr (signature.map Program.Typed.toJson).toArray),
          ("port", Port.toJson port)] compare) = .ok (DType.nameJson name) :=
        field_ofList (by simp) (by simp)
      have hi : field "node" "inputs" (Std.TreeMap.Raw.ofList [
          ("name", DType.nameJson name), ("inputs", Json.arr (args.map Ref.toJson)),
          ("signature", Json.arr (signature.map Program.Typed.toJson).toArray),
          ("port", Port.toJson port)] compare) = .ok (Json.arr (args.map Ref.toJson)) :=
        field_ofList (by simp) (by simp)
      have hg : field "node" "signature" (Std.TreeMap.Raw.ofList [
          ("name", DType.nameJson name), ("inputs", Json.arr (args.map Ref.toJson)),
          ("signature", Json.arr (signature.map Program.Typed.toJson).toArray),
          ("port", Port.toJson port)] compare) =
          .ok (Json.arr (signature.map Program.Typed.toJson).toArray) :=
        field_ofList (by simp) (by simp)
      have hp : field "node" "port" (Std.TreeMap.Raw.ofList [
          ("name", DType.nameJson name), ("inputs", Json.arr (args.map Ref.toJson)),
          ("signature", Json.arr (signature.map Program.Typed.toJson).toArray),
          ("port", Port.toJson port)] compare) = .ok (Port.toJson port) :=
        field_ofList (by simp) (by simp)
      have hsig : (signature.map Program.Typed.toJson).toArray.mapM Program.Typed.ofJson =
          .ok signature.toArray := by
        rw [← List.map_toArray]
        exact mapM_toJson Program.Typed.ofJson_toJson signature.toArray
      simp (disch := simp) only [Program.Node.toJson, Program.Node.ofJson, Json.mkObj, object,
        toList_ofList_singleton, exactObject_ofList, bind,
        Except.bind, hn, hi, hg, hp, DType.nameOfJson_nameJson, array,
        mapM_toJson Ref.ofJson_toJson, hsig, Port.ofJson_toJson]
      rfl

/-- A typed program with the words of its constants, one entry per step: the words of a constant
step's tensor in row-major order (as `Typed.decode` reads them), none at every other step. -/
structure Artifact where
  /-- The program. -/
  program : Program
  /-- The words of each constant step. -/
  constants : Array (Option (Array Nat))
  deriving DecidableEq, Repr

/-- The JSON of a step's words: their array, or null. -/
def wordsJson : Option (Array Nat) → Json
  | some words => Lean.toJson words.toList
  | none => .null

/-- The words of the JSON `wordsJson` writes. -/
def wordsOfJson : Json → Except String (Option (Array Nat))
  | .null => pure none
  | json => some <$> nats "artifact.constants" json

/-- Decoding the JSON of a step's words returns them. -/
theorem wordsOfJson_wordsJson : (w : Option (Array Nat)) → wordsOfJson (wordsJson w) = .ok w
  | none => rfl
  | some words => by
      simp only [wordsJson, toJson_list, wordsOfJson]
      rw [← toJson_list, nats_toJson]
      rfl

/-- The JSON of an artifact: `{"format": "torchlean.ir", "nodes": [...], "outputs": [...],
"scalars": [...], "constants": [...]}`, with one scalar (`scalarJson`) and one word array or null
(`wordsJson`) per step. -/
def Artifact.toJson (a : Artifact) : Json :=
  Json.mkObj [("format", Json.str formatName),
    ("nodes", Json.arr (a.program.nodes.map Program.Node.toJson)),
    ("outputs", Json.arr (a.program.outputs.map Ref.toJson)),
    ("scalars", Json.arr (a.program.scalars.map scalarJson)),
    ("constants", Json.arr (a.constants.map wordsJson))]

/-- The artifact of the JSON `Artifact.toJson` writes. Decoding is structural: nothing is checked
against the steps (`Artifact.load` checks). -/
def Artifact.ofJson (json : Json) : Except String Artifact := do
  let o ← exactObject "artifact" ["format", "nodes", "outputs", "scalars", "constants"] json
  let name ← string "artifact.format" (← field "artifact" "format" o)
  unless name = formatName do throw s!"IR import: format {name}, not {formatName}"
  let nodes ← (← array "artifact.nodes" (← field "artifact" "nodes" o)).mapM Program.Node.ofJson
  let outputs ← (← array "artifact.outputs" (← field "artifact" "outputs" o)).mapM Ref.ofJson
  let scalars ← (← array "artifact.scalars" (← field "artifact" "scalars" o)).mapM scalarOfJson
  let constants ← (← array "artifact.constants" (← field "artifact" "constants" o)).mapM
    wordsOfJson
  pure ⟨⟨nodes, outputs, scalars⟩, constants⟩

/-- Decoding the JSON of any artifact returns it. -/
theorem Artifact.ofJson_toJson (a : Artifact) : Artifact.ofJson a.toJson = .ok a := by
  rcases a with ⟨⟨nodes, outputs, scalars⟩, constants⟩
  have hf : field "artifact" "format" (Std.TreeMap.Raw.ofList [("format", Json.str formatName),
      ("nodes", Json.arr (nodes.map Program.Node.toJson)),
      ("outputs", Json.arr (outputs.map Ref.toJson)),
      ("scalars", Json.arr (scalars.map scalarJson)),
      ("constants", Json.arr (constants.map wordsJson))] compare) = .ok (Json.str formatName) :=
    field_ofList (by simp) (by simp)
  have hn : field "artifact" "nodes" (Std.TreeMap.Raw.ofList [("format", Json.str formatName),
      ("nodes", Json.arr (nodes.map Program.Node.toJson)),
      ("outputs", Json.arr (outputs.map Ref.toJson)),
      ("scalars", Json.arr (scalars.map scalarJson)),
      ("constants", Json.arr (constants.map wordsJson))] compare) =
      .ok (Json.arr (nodes.map Program.Node.toJson)) :=
    field_ofList (by simp) (by simp)
  have ho : field "artifact" "outputs" (Std.TreeMap.Raw.ofList [("format", Json.str formatName),
      ("nodes", Json.arr (nodes.map Program.Node.toJson)),
      ("outputs", Json.arr (outputs.map Ref.toJson)),
      ("scalars", Json.arr (scalars.map scalarJson)),
      ("constants", Json.arr (constants.map wordsJson))] compare) =
      .ok (Json.arr (outputs.map Ref.toJson)) :=
    field_ofList (by simp) (by simp)
  have hs : field "artifact" "scalars" (Std.TreeMap.Raw.ofList [("format", Json.str formatName),
      ("nodes", Json.arr (nodes.map Program.Node.toJson)),
      ("outputs", Json.arr (outputs.map Ref.toJson)),
      ("scalars", Json.arr (scalars.map scalarJson)),
      ("constants", Json.arr (constants.map wordsJson))] compare) =
      .ok (Json.arr (scalars.map scalarJson)) :=
    field_ofList (by simp) (by simp)
  have hc : field "artifact" "constants" (Std.TreeMap.Raw.ofList [("format", Json.str formatName),
      ("nodes", Json.arr (nodes.map Program.Node.toJson)),
      ("outputs", Json.arr (outputs.map Ref.toJson)),
      ("scalars", Json.arr (scalars.map scalarJson)),
      ("constants", Json.arr (constants.map wordsJson))] compare) =
      .ok (Json.arr (constants.map wordsJson)) :=
    field_ofList (by simp) (by simp)
  simp (disch := simp) only [Artifact.toJson, Artifact.ofJson, Json.mkObj,
    exactObject_ofList, bind, Except.bind, hf, hn, ho, hs, hc,
    string, array, mapM_toJson Program.Node.ofJson_toJson,
    mapM_toJson Ref.ofJson_toJson, mapM_toJson scalarOfJson_scalarJson,
    mapM_toJson wordsOfJson_wordsJson]
  rfl

/-- The tensor of a typed port from its words in row-major order: interchange words that `R`
reads at a value, frozen or detached port, zero or one at a Boolean port, words below
`2 ^ width` at an integer port, and at an index port `bounds.length` components per entry
(component `k` of entry `j` at position `j * bounds.length + k`), each below its bound. -/
def Program.Typed.decode (R : Program.Machine) : (t : Program.Typed) → Array Nat →
    Except String (t.Carrier R)
  | ⟨⟨.value, s⟩, some d⟩, words | ⟨⟨.frozen, s⟩, some d⟩, words
  | ⟨⟨.detached, s⟩, some d⟩, words =>
      letI := R.storage d
      do
      unless words.size = s.size do
        throw "IR artifact: the word count does not match the tensor shape"
      Tensor.generateFlatM s fun j => R.read d words[j.val]!
  | ⟨⟨.value, _⟩, none⟩, _ | ⟨⟨.frozen, _⟩, none⟩, _ | ⟨⟨.detached, _⟩, none⟩, _ =>
      throw "IR artifact: a value port without a scalar"
  | ⟨⟨.bool, s⟩, _⟩, words => do
      unless words.size = s.size do
        throw "IR artifact: the word count does not match the tensor shape"
      Tensor.generateFlatM s fun j => match words[j.val]! with
        | 0 => pure false
        | 1 => pure true
        | _ => throw "IR artifact: a Boolean word is zero or one"
  | ⟨⟨.integer _ width, s⟩, _⟩, words => do
      unless words.size = s.size do
        throw "IR artifact: the word count does not match the tensor shape"
      Tensor.generateFlatM s fun j =>
        if h : words[j.val]! < 2 ^ width then pure (BitVec.ofNatLT _ h)
        else throw "IR artifact: a word exceeds its integer width"
  | ⟨⟨.index bounds, s⟩, _⟩, words => do
      unless words.size = s.size * bounds.length do
        throw "IR artifact: the component count does not match the index tensor"
      Tensor.generateFlatM s fun j =>
        Port.IndexTuple.ofComponents bounds fun k => (words[j.val * bounds.length + k]! : Int)

/-- The program of an artifact and its payload under `R`, the custom operations and calls taken
from `env`. The checks run in this order: (1) the program passes `Program.check R`, (2) there is
one word entry per step, and each constant step has words that `Typed.decode` reads at its type
and no other step has words. Then (3) each custom step has an operation registered under its name
with the step's scalar, input shapes and output shape, and (4) each call has an implementation
registered under its name with exactly the declared signature and result type. Last, (5) the
payload passes `Payload.check`. -/
def Artifact.load (R : Program.Machine) (env : Program.Env R) (a : Artifact) :
    Except String (Program × Program.Payload R) := do
  let p := a.program
  p.check R
  unless a.constants.size = p.nodes.size do
    throw s!"IR artifact: {a.constants.size} word entries for {p.nodes.size} nodes"
  let consts ← p.nodes.mapIdxM fun i n => do
    match n, a.constants[i]?.join with
    | .node (.const port), some words =>
        let t := Program.Typed.ofPort (p.scalar i) port
        let tensor ← (t.decode R words).mapError (s!"IR artifact: node {i}: " ++ ·)
        pure (some ⟨t, tensor⟩)
    | .node (.const _), none => throw s!"IR artifact: node {i}: a constant without words"
    | _, some _ => throw s!"IR artifact: node {i}: words for a node that reads no constant"
    | _, none => pure none
  let customs ← p.nodes.mapIdxM fun i n => do
    match n with
    | .node (.op ⟨_, _, .custom name inputs output⟩ _) =>
        let some c := env.custom name
          | throw s!"IR artifact: node {i}: no operation registered for the custom operator {name}"
        unless some c.scalar = p.scalar i && c.inputs = inputs && c.output = output do
          throw s!"IR artifact: node {i}: {name} is registered at another scalar or interface"
        pure (some c)
    | _ => pure none
  let calls ← p.nodes.mapIdxM fun i n => do
    match n with
    | .call name _ signature port =>
        let some c := env.call name
          | throw s!"IR artifact: node {i}: no implementation registered for {name}"
        unless c.inputs = signature && c.output = Program.Typed.ofPort (p.scalar i) port do
          throw s!"IR artifact: node {i}: {name} is registered with another signature"
        pure (some c)
    | _ => pure none
  let P : Program.Payload R := ⟨consts, customs, calls⟩
  P.check p
  pure (p, P)

/-- An artifact that `Artifact.load` accepts gives back its own program, which passes
`Program.check R`, with a payload that passes `Payload.check` on it. -/
theorem Artifact.load_ok {R : Program.Machine} {env : Program.Env R} {a : Artifact}
    {p : Program} {P : Program.Payload R} (h : a.load R env = .ok (p, P)) :
    p = a.program ∧ p.check R = .ok () ∧ P.check p = .ok () := by
  unfold Artifact.load at h
  obtain ⟨⟨⟩, hc, h⟩ := bind_eq_ok.1 h
  split at h
  · obtain ⟨consts, -, h⟩ := bind_eq_ok.1 h
    obtain ⟨customs, -, h⟩ := bind_eq_ok.1 h
    obtain ⟨calls, -, h⟩ := bind_eq_ok.1 h
    obtain ⟨⟨⟩, hP, h⟩ := bind_eq_ok.1 h
    cases h
    exact ⟨rfl, hc, hP⟩
  · obtain ⟨_, h, -⟩ := bind_eq_ok.1 h
    cases h

end TorchLean.IR
