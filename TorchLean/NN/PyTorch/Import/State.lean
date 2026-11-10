/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Pack
public import TorchLean.NN.IR.DType
public import TorchLean.NN.Checkpoint.Encoding
public import TorchLean.NN.Json
public import TorchLean.Tensor.Constructors
import all Lean.Data.Json.Basic
import Std.Data.TreeMap.Raw.Lemmas
import TorchLean.Tensor.Rep.Traversal.Sequence

/-!
# PyTorch state dicts

`TorchLean.PyTorch.state shapes names document` reads a PyTorch `state_dict` into
`TensorPack α shapes`. Tensor `i` of the state is the entry named `names[i]`, the name the PyTorch
module gives that parameter or buffer, such as `fc1.weight`. `TorchLean.PyTorch.stateDict names s`
writes the document that `python/torchlean/state.py` writes from `module.state_dict()`:

```
{"format": "torchlean.state",
 "entries": {"fc.bias": {"dtype": "float32", "shape": [2], "words": [1065353216, 2147483648]}}}
```

Here `1065353216` is the binary32 word of `1.0` and `2147483648` the word of `-0.0`.

The words are the unsigned words of `Checkpoint.Encoding α` in row-major order, so negative zeros,
subnormals and every NaN payload that `α` keeps come back unchanged. The dtype is PyTorch's name
(`float16`, `bfloat16`, `float32`, `float64`) for binary16, bfloat16, binary32 and binary64 words,
and the encoding's format tag for any other scalar type. Reading never converts a dtype, whereas
PyTorch's `load_state_dict` copies each tensor into the dtype of the module. Missing keys,
unexpected keys and shape mismatches are reported in the words of `load_state_dict`, and a dtype
mismatch in the form of its size-mismatch message.

The main declarations are `state`, `stateDict` and `dtype`, with the round trips
`state_stateDict`, for an encoding whose decoder inverts its encoder, and `state_stateDict_scalar`
for every `TorchLean.Kernel.Scalar`.
-/

public section

namespace TorchLean
namespace PyTorch

open Lean

/-- The format name of a state document, `torchlean.state`. -/
def stateFormat : String := "torchlean.state"

/-- The dtype name of the words of `α`: `float64` for `Float` and FloatLib's `Binary64`, `float32`
for `Float32` and `Binary32`, `float16` for `Binary16` and `bfloat16` for `BFloat16`, recognized by
their encoding's format tag, and that tag for every other `α`. No theorem states these six
names. -/
def dtype (α : Type) [Checkpoint.Encoding α] : String :=
  let tag := Checkpoint.Encoding.format (α := α)
  let named : List (String × IR.DType) :=
    [(Checkpoint.Encoding.format (α := Float), .float64),
      (Checkpoint.Encoding.format (α := Float32), .float32),
      (Checkpoint.Encoding.format (α := FloatLib.Floats.Binary16), .float16),
      (Checkpoint.Encoding.format (α := FloatLib.Floats.BFloat16), .bfloat16),
      (Checkpoint.Encoding.format (α := FloatLib.Floats.Binary32), .float32),
      (Checkpoint.Encoding.format (α := FloatLib.Floats.Binary64), .float64)]
  match List.lookup tag named with
  | some name => name.describe
  | none => tag

variable {α : Type} [Storage α] [Checkpoint.Encoding α]

/-- The entry of one tensor: its dtype, its shape and its words in row-major order. -/
private def entryJson {shape : Shape} (tensor : Tensor α shape) : Json :=
  Json.mkObj [("dtype", Json.str (dtype α)), ("shape", toJson shape),
    ("words", toJson ((Tensor.to tensor (List α)).map
      Checkpoint.Encoding.encode))]

/-- The entries of a tensor pack, the tensor at position `i` named `names[i]`. -/
private def entryList : {shapes : List Shape} → List String →
    TensorPack α shapes → List (String × Json)
  | _, [], _ => []
  | _, _ :: _, .nil => []
  | _, name :: names, .cons first rest => (name, entryJson first) :: entryList names rest

/-- The state dict of `s`, its tensor `i` named `names[i]`, as `python/torchlean/state.py` writes
it. `state_stateDict` reads it back when `names` has no repeats and one name per tensor. -/
def stateDict {shapes : List Shape} (names : List String) (s : TensorPack α shapes) : Json :=
  Json.mkObj [("format", Json.str stateFormat),
    ("entries", Json.mkObj (entryList names s))]

/-- The error of `Json.exactObject context keys json` when `json` is an object, in the form
of `load_state_dict`'s key errors: the keys expected twice, the missing keys and the unexpected
keys. The messages are checked by evaluation. -/
private def keyError (context : String) (keys : List String) (json : Json) (error : String) :
    String :=
  match json with
  | .obj object =>
    let quoted (names : List String) := ", ".intercalate (names.map fun name => s!"\"{name}\"")
    let repeated := (keys.filter fun key => keys.count key > 1).eraseDups
    let missing := keys.filter fun key => !object.contains key
    let unexpected := object.keys.filter fun key => !keys.contains key
    let parts :=
      (if repeated.isEmpty then [] else [s!"Repeated key(s) for {context}: {quoted repeated}."]) ++
      (if missing.isEmpty then [] else [s!"Missing key(s) in {context}: {quoted missing}."]) ++
      (if unexpected.isEmpty then [] else
        [s!"Unexpected key(s) in {context}: {quoted unexpected}."])
    if parts.isEmpty then error else "PyTorch state: " ++ " ".intercalate parts
  | _ => error

/-- `Json.exactObject` with the errors of `keyError`. -/
private def exactKeys (context : String) (keys : List String) (json : Json) :
    Except String (Std.TreeMap.Raw String Json) :=
  (Json.exactObject context keys json).mapError (keyError context keys json)

/-- Read the entry `name` as a tensor of shape `shape`. Its keys, its dtype `dtype α`, its shape and
its number of words are checked, and every word is decoded. -/
private def entryTensor (shape : Shape) (name : String) (json : Json) :
    Except String (Tensor α shape) := do
  let context := s!"PyTorch state entry {name}"
  let entry ← exactKeys s!"entry {name}" ["dtype", "shape", "words"] json
  let stored ← Json.string s!"{context}.dtype" (← Json.field context "dtype" entry)
  unless stored = dtype α do
    throw (s!"PyTorch state: dtype mismatch for {name}: copying a param with dtype {stored} " ++
      s!"from checkpoint, the dtype in current model is {dtype α}.")
  let storedShape ← Json.shapeField context "shape" entry
  unless storedShape = shape do
    throw (s!"PyTorch state: size mismatch for {name}: copying a param with shape " ++
      s!"torch.Size({storedShape}) from checkpoint, the shape in current model is " ++
      s!"torch.Size({shape}).")
  let words ← Json.natArrayField context "words" entry
  if h : words.size = shape.size then
    Tensor.generateFlatM shape fun index =>
      (Checkpoint.Encoding.decode
        (words[index.val]'(by rw [h]; exact index.isLt))).mapError
        fun error => s!"PyTorch state: {name}[{index.val}]: {error}"
  else
    throw (s!"PyTorch state: the entry {name} of shape torch.Size({shape}) has " ++
      s!"{words.size} word(s), expected {shape.size}.")

/-- Read the entries `names` of `entries` as tensors of the shapes `shapes`, in order. -/
private def entryTensors (entries : Std.TreeMap.Raw String Json) :
    (shapes : List Shape) → (names : List String) → names.length = shapes.length →
      Except String (TensorPack α shapes)
  | [], [], _ => pure .nil
  | shape :: shapes, name :: names, h => do
    let first ← entryTensor shape name (← Json.field "PyTorch state entries" name entries)
    let rest ← entryTensors entries shapes names (by simpa using h)
    pure (.cons first rest)
  | [], _ :: _, h => absurd h (by simp)
  | _ :: _, [], h => absurd h (by simp)

/-- Read a state document into `TensorPack α shapes`, tensor `i` from the entry `names[i]`. The
document must have exactly the entries `names`, each with the dtype `dtype α`, the shape
`shapes[i]` and one word per element, and every word must decode. -/
def state (shapes : List Shape) (names : List String) (json : Json) :
    Except String (TensorPack α shapes) :=
  if h : names.length = shapes.length then do
    let document ← exactKeys "the document" ["format", "entries"] json
    let stored ← Json.string "PyTorch state format"
      (← Json.field "PyTorch state" "format" document)
    unless stored = stateFormat do
      throw s!"PyTorch state: expected the format {stateFormat}, got {stored}."
    let entries ← exactKeys "state_dict" names
      (← Json.field "PyTorch state" "entries" document)
    entryTensors entries shapes names h
  else
    throw s!"PyTorch state: {names.length} name(s) for {shapes.length} tensor(s)."

/-! ## Round trip -/

omit [Checkpoint.Encoding α] in
/-- `generateFlatM` builds `tensor` from reads that return its elements in row-major order. -/
private theorem generateFlatM_ok {shape : Shape} (tensor : Tensor α shape)
    (f : Fin shape.size → Except String α)
    (hf : ∀ index : Fin (Shape.size shape),
      f index = .ok (Tensor.Rep.getFlat tensor index)) :
    Tensor.generateFlatM shape f = .ok tensor :=
  calc Tensor.generateFlatM shape f
      = Tensor.generateFlatM shape
          (pure ∘ Tensor.Rep.getFlat tensor) :=
        congrArg (Tensor.generateFlatM shape) (funext hf)
    _ = .ok tensor := by
        unfold Tensor.generateFlatM
        rw [Tensor.sequenceFinM_pure]
        exact congrArg Except.ok (Tensor.Rep.ofFlatFn_getFlat tensor)

/-- `exactKeys` returns an object built from a list with distinct keys `keys`. -/
private theorem exactKeys_ofList {context : String} {l : List (String × Json)}
    {keys : List String} (distinct : l.Pairwise (fun a b => ¬ compare a.1 b.1 = .eq))
    (hkeys : l.map Prod.fst = keys) :
    exactKeys context keys (.obj (Std.TreeMap.Raw.ofList l compare)) =
      .ok (Std.TreeMap.Raw.ofList l compare) := by
  rw [exactKeys, Json.exactObject_ofList distinct hkeys]
  rfl

/-- An entry written by `entryJson` reads back as its tensor when every word decodes to the
scalar it encodes. -/
private theorem entryTensor_entryJson
    (hword : ∀ x : α, Checkpoint.Encoding.decode
      (Checkpoint.Encoding.encode x) = (.ok x : Except String α))
    {shape : Shape} (name : String) (tensor : Tensor α shape) :
    entryTensor shape name (entryJson tensor) = .ok tensor := by
  have hkeys : ∀ (context : String) (d s w : Json), exactKeys context ["dtype", "shape", "words"]
      (.obj (Std.TreeMap.Raw.ofList [("dtype", d), ("shape", s), ("words", w)] compare)) =
      .ok (Std.TreeMap.Raw.ofList [("dtype", d), ("shape", s), ("words", w)] compare) :=
    fun _ _ _ _ => exactKeys_ofList (by simp) rfl
  have hd : ∀ (context : String) (d s w : Json), Json.field context "dtype"
      (Std.TreeMap.Raw.ofList [("dtype", d), ("shape", s), ("words", w)] compare) = .ok d :=
    fun _ _ _ _ => Json.field_ofList (by simp) (by simp)
  have hs : ∀ (context : String) (d s w : Json), Json.field context "shape"
      (Std.TreeMap.Raw.ofList [("dtype", d), ("shape", s), ("words", w)] compare) = .ok s :=
    fun _ _ _ _ => Json.field_ofList (by simp) (by simp)
  have hw : ∀ (context : String) (d s w : Json), Json.field context "words"
      (Std.TreeMap.Raw.ofList [("dtype", d), ("shape", s), ("words", w)] compare) = .ok w :=
    fun _ _ _ _ => Json.field_ofList (by simp) (by simp)
  simp [entryTensor, entryJson, Json.mkObj, hkeys, hd, hs, hw, Json.string,
    Json.shapeField, Json.natArrayField, Json.shape_toJson,
    Json.nats_toJson, hword, Except.mapError, bind,
    Except.bind]
  exact generateFlatM_ok tensor _ fun index => by simp

/-- The names of `entryList names pack` are `names` when there is one name per tensor. -/
private theorem entryList_keys : {shapes : List Shape} → (names : List String) →
    (pack : TensorPack α shapes) → names.length = shapes.length →
    (entryList names pack).map Prod.fst = names
  | _, [], _, _ => rfl
  | _, _ :: _, .nil, h => absurd h (by simp)
  | _, name :: names, .cons first rest, h => by
    simp only [entryList, List.map_cons, entryList_keys names rest (by simpa using h)]

/-- `entryTensors` reads back a tensor pack from an object whose fields are its entries. -/
private theorem entryTensors_entryList
    (hword : ∀ x : α, Checkpoint.Encoding.decode
      (Checkpoint.Encoding.encode x) = (.ok x : Except String α))
    (entries : Std.TreeMap.Raw String Json) :
    {shapes : List Shape} → (names : List String) →
    (pack : TensorPack α shapes) → (h : names.length = shapes.length) →
    (∀ pair ∈ entryList names pack,
      Json.field "PyTorch state entries" pair.1 entries = .ok pair.2) →
    entryTensors entries shapes names h = .ok pack
  | _, [], .nil, _, _ => rfl
  | _, name :: names, .cons first rest, h, hfield => by
    have hfirst : Json.field "PyTorch state entries" name entries =
        .ok (entryJson first) :=
      hfield (name, entryJson first) (by simp [entryList])
    have hrest := entryTensors_entryList hword entries names rest (by simpa using h)
      (fun pair hpair => hfield pair (by simp [entryList, hpair]))
    simp only [entryTensors, bind, Except.bind, hfirst, entryTensor_entryJson hword, hrest, pure,
      Except.pure]
  | _, [], .cons _ _, h, _ => absurd h (by simp)
  | _, _ :: _, .nil, h, _ => absurd h (by simp)

/-- Reading the state dict of `s` gives back `s` when `names` has no repeats, there is one name
per tensor and every word decodes to the scalar it encodes. -/
theorem state_stateDict
    (hword : ∀ x : α, Checkpoint.Encoding.decode
      (Checkpoint.Encoding.encode x) = (.ok x : Except String α))
    {shapes : List Shape} {names : List String} (hnodup : names.Nodup)
    (hlength : names.length = shapes.length) (s : TensorPack α shapes) :
    state shapes names (stateDict names s) = .ok s := by
  have hkeys := entryList_keys names s hlength
  have hdistinct : (entryList names s).Pairwise
      (fun a b => ¬ compare a.1 b.1 = .eq) := by
    have hnames := hnodup
    rw [← hkeys, List.Nodup, List.pairwise_map] at hnames
    exact hnames.imp fun hne heq => hne (Std.LawfulEqOrd.eq_of_compare heq)
  have htensors := entryTensors_entryList hword
    (Std.TreeMap.Raw.ofList (entryList names s) compare)
    names s hlength
    (fun ⟨_, _⟩ hpair => Json.field_ofList hdistinct hpair)
  have hdocument : ∀ f e : Json, exactKeys "the document" ["format", "entries"]
      (.obj (Std.TreeMap.Raw.ofList [("format", f), ("entries", e)] compare)) =
      .ok (Std.TreeMap.Raw.ofList [("format", f), ("entries", e)] compare) :=
    fun _ _ => exactKeys_ofList (by simp) rfl
  have hformat : ∀ f e : Json, Json.field "PyTorch state" "format"
      (Std.TreeMap.Raw.ofList [("format", f), ("entries", e)] compare) = .ok f :=
    fun _ _ => Json.field_ofList (by simp) (by simp)
  have hentries : ∀ f e : Json, Json.field "PyTorch state" "entries"
      (Std.TreeMap.Raw.ofList [("format", f), ("entries", e)] compare) = .ok e :=
    fun _ _ => Json.field_ofList (by simp) (by simp)
  rw [state, dite_eq_left hlength]
  simp [stateDict, Json.mkObj, hdocument, hformat, hentries, Json.string,
    exactKeys_ofList hdistinct hkeys, htensors, bind, Except.bind]

/-! ## Scalars of custom computations -/

section Scalar

variable {β : Type} [Kernel.Scalar β]

/-- Reading the state dict of a state of custom-computation scalars gives back the state when
`names` has no repeats and there is one name per tensor. -/
theorem state_stateDict_scalar [Storage β] {shapes : List Shape}
    {names : List String} (hnodup : names.Nodup) (hlength : names.length = shapes.length)
    (s : TensorPack β shapes) :
    state shapes names (stateDict names s) = .ok s :=
  state_stateDict Checkpoint.Encoding.decode_encode_scalar hnodup hlength s

end Scalar

end PyTorch
end TorchLean
