/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.Pointwise
public import TorchLean.NN.IR.Artifact

/-!
# The op table of the Python captures

`TorchLean.IR.table dtypes` is the JSON of the facts that the Python captures read from Lean. It
holds the format name, the wire version, the JSON of each element type of `dtypes` and the codes
that the codec writes for division rounding, GELU approximation, reductions, pools, scatters, square
root policies, bitcast sides and cast sources. It also holds the typed forms of the pointwise
operators (`Pointwise.form?`) between the element types of `dtypes`, their steps encoded at the
shape `[]`.

Python replaces every `{"shape": []}` field of a step by the shape of the node it lowers. We check
that replacement at the shape `[2, 3]` (`checkTable`) and do not prove it. `tableOfNames` reads
the element types from their names (`DType.ofName`), runs the check and returns the table. The
tool `verify ir --op-table` and the Python binding `op_table` call it on names given at run time.
Compiled code never applies `table` to a literal list, since Lean computes such a closed term when
the module is initialized.
-/

@[expose] public section

open Lean

namespace TorchLean.IR

/-- The element type named `name`: one that the JSON reader (`DType.fromJson`) reads from the
string, such as `float32` or `bool`, or for `int<w>` and `uint<w>` the integer type of width `w`.
It is refused unless it is valid and `DType.describe` gives `name` back. -/
def DType.ofName (name : String) : Except String DType := do
  let word (signed : Bool) (digits : List Char) : Except String DType :=
    match (String.ofList digits).toNat? with
    | some width => pure (.integer signed (some width))
    | none => throw s!"IR element type: no element type is named {name}"
  let d ← (match DType.fromJson (.str name), name.toList with
    | .ok d, _ => pure d
    | .error _, 'u' :: 'i' :: 'n' :: 't' :: digits => word false digits
    | .error _, 'i' :: 'n' :: 't' :: digits => word true digits
    | .error message, _ => throw message)
  if d.describe = name ∧ d.valid then pure d
  else throw s!"IR element type: no element type is named {name}"

/-- An element type read from a name is valid and describes as that name. -/
theorem DType.ofName_ok {name : String} {d : DType} (h : DType.ofName name = .ok d) :
    d.describe = name ∧ d.valid = true := by
  unfold DType.ofName at h
  obtain ⟨d', -, h⟩ := IR.bind_eq_ok.1 h
  split at h
  · next hd =>
      cases h
      exact hd
  · cases h

end TorchLean.IR

namespace TorchLean.IR

/-- The op table at the element types `dtypes`, each named by `DType.describe`, with the forms of
the pointwise operators at the shape `[]` (`Pointwise.form?`). No property of it is proved, and
`checkTable` checks the replacement of the shape `[]` in its steps. -/
def table (dtypes : List DType) : Json :=
  Json.mkObj [("format", Json.str formatName),
    ("version", Lean.toJson Op.Wire.formatVersion),
    ("dtypes", Json.mkObj (dtypes.map fun d => (d.describe, d.toJson))),
    ("codes", codes),
    ("pointwise", Json.mkObj (names.map fun name => (name, Json.arr
      ((PointwiseOp.all.filter (·.name == name)).map (entry dtypes)).toArray)))]
where
  /-- The codes that the codec (`TorchLean.Op.Wire`) writes for division rounding modes, GELU
  approximations, reductions, pools (the average counting padding), scatter reductions, square root
  policies, the sides of a bitcast and the sources of a cast. The cast code of an index below
  `bound` is `index` plus `bound` (`Op.Wire.castSourceCode_index`). -/
  codes : Json :=
    Json.mkObj [
      ("div", Json.mkObj [
        ("none", Lean.toJson (Op.Wire.divRoundingCode DivRounding.none)),
        ("trunc", Lean.toJson (Op.Wire.divRoundingCode DivRounding.trunc)),
        ("floor", Lean.toJson (Op.Wire.divRoundingCode DivRounding.floor))]),
      ("gelu", Json.mkObj [
        ("none", Lean.toJson
          (Op.Wire.geluApproximationCode GeluApproximation.none)),
        ("tanh", Lean.toJson
          (Op.Wire.geluApproximationCode GeluApproximation.tanh))]),
      ("reduce", Json.mkObj [
        ("sum", Lean.toJson (Op.Wire.reduceOpCode Tensor.ReduceOp.sum)),
        ("prod", Lean.toJson (Op.Wire.reduceOpCode Tensor.ReduceOp.prod)),
        ("max", Lean.toJson (Op.Wire.reduceOpCode Tensor.ReduceOp.max)),
        ("min", Lean.toJson (Op.Wire.reduceOpCode Tensor.ReduceOp.min))]),
      ("pool", Json.mkObj [
        ("max", Lean.toJson (Op.Wire.poolOpCode Tensor.PoolOp.max)),
        ("avg", Lean.toJson (Op.Wire.poolOpCode (Tensor.PoolOp.avg true)))]),
      ("scatter", Json.mkObj [
        ("replace", Lean.toJson
          (Op.Wire.scatterCode Tensor.ScatterReduce.replace)),
        ("sum", Lean.toJson (Op.Wire.scatterCode Tensor.ScatterReduce.sum)),
        ("prod", Lean.toJson (Op.Wire.scatterCode Tensor.ScatterReduce.prod)),
        ("max", Lean.toJson (Op.Wire.scatterCode Tensor.ScatterReduce.max)),
        ("min", Lean.toJson (Op.Wire.scatterCode Tensor.ScatterReduce.min))]),
      ("sqrt", Json.mkObj [
        ("raw", Lean.toJson (Op.Wire.sqrtPolicyCode Tensor.SqrtPolicy.raw)),
        ("clampNegative", Lean.toJson
          (Op.Wire.sqrtPolicyCode Tensor.SqrtPolicy.clampNegative))]),
      ("word", Json.mkObj [
        ("scalar", Lean.toJson (Op.Wire.wordCode Op.Word.scalar)),
        ("unsigned", Lean.toJson (Op.Wire.wordCode (Op.Word.integer false))),
        ("signed", Lean.toJson (Op.Wire.wordCode (Op.Word.integer true)))]),
      ("cast", Json.mkObj [
        ("bool", Lean.toJson (Op.Wire.castSourceCode Op.CastSource.bool)),
        ("index", Lean.toJson
          (Op.Wire.castSourceCode (Op.CastSource.index 0)))])]
  /-- The JSON of an argument of a step: `["operand", k]`, `["zero"]` or `["node", j]`. -/
  arg : Pointwise.Arg → Json
    | .operand k => Json.arr #[Json.str "operand", Lean.toJson k]
    | .zero => Json.arr #[Json.str "zero"]
    | .node j => Json.arr #[Json.str "node", Lean.toJson j]
  /-- The JSON of a form: the element type its steps run at, whether a step reads the zero tensor,
  and each step's wire record (`Json.ofOpRecord`) with its arguments. -/
  form (f : Pointwise.Fragment) (e : DType) : Json :=
    Json.mkObj [("at", Json.str e.describe), ("zero", Json.bool (Pointwise.Fragment.usesZero f)),
      ("steps", Json.arr (f.map fun (o, args) => Json.mkObj [
        ("record", Json.ofOpRecord (Op.encode o)),
        ("args", Json.arr (args.map arg).toArray)]).toArray)]
  /-- The entry of `op`: its keyword (`PointwiseOp.keyword`) or null, its arity and its forms by
  operand type and result type, without the pairs that have no form. -/
  entry (dtypes : List DType) (op : PointwiseOp) : Json :=
    Json.mkObj [
      ("keyword", match op.keyword with
        | some (key, value) => Json.arr #[Json.str key, Json.str value]
        | none => Json.null),
      ("arity", Lean.toJson op.arity),
      ("forms", Json.mkObj <| dtypes.filterMap fun operand =>
        let results := dtypes.filterMap fun d =>
          (Pointwise.form? op operand d []).map fun (f, e) => (d.describe, form f e)
        if results.isEmpty then none else some (operand.describe, Json.mkObj results))]
  /-- The names of the pointwise operators (`PointwiseOp.name`), each once, in the order of
  `PointwiseOp.all`. -/
  names : List String :=
    PointwiseOp.all.foldl
      (fun acc op => if acc.contains op.name then acc else acc ++ [op.name]) []

/-- Checks, for each pointwise operator and pair of element types of `dtypes`, that the form at
the shape `[2, 3]` is the form at `[]` with every field `.shape []` of its records replaced by
`.shape [2, 3]`. That is the replacement Python makes at a node's shape, checked when the table is
written and not proved. -/
def checkTable (dtypes : List DType) : Except String Unit :=
  PointwiseOp.all.forM fun op => dtypes.forM fun operand => dtypes.forM fun d =>
    if agrees (Pointwise.form? op operand d []) (Pointwise.form? op operand d [2, 3]) then
      pure ()
    else
      throw (s!"op table: the form of {op.describe} from {operand.describe} to {d.describe} " ++
        "at the shape [2, 3] is not its form at [] with the shape replaced")
where
  /-- The record `w` with every field `.shape []` replaced by `.shape [2, 3]`. -/
  replace (w : Op.Wire) : Op.Wire :=
    { w with fields := w.fields.map fun f => if f = .shape [] then .shape [2, 3] else f }
  /-- Whether the form at `[2, 3]` (the second) is the form at `[]` (the first) with the shape of
  its records replaced: the same element type, the same arguments and the replaced records. -/
  agrees : Option (Pointwise.Fragment × DType) → Option (Pointwise.Fragment × DType) → Bool
    | none, none => true
    | some (f, e), some (g, e') =>
        e == e' && f.map (·.2) == g.map (·.2) &&
          f.map (fun step => replace (Op.encode step.1)) ==
            g.map (fun step => Op.encode step.1)
    | _, _ => false

/-- The op table at the element types named `names` (`DType.ofName`), once `checkTable` passes on
them. It is refused when `names` is empty, as `verify ir --op-table` refuses it. -/
def tableOfNames (names : List String) : Except String Json :=
  if names.isEmpty then
    .error "op table: give at least one element type name"
  else do
    let dtypes ← names.mapM DType.ofName
    checkTable dtypes
    pure (table dtypes)

/-- A table that `tableOfNames` returns is `table` at the element types read from the names, and
`checkTable` passed on them. -/
theorem tableOfNames_ok {names : List String} {json : Json} (h : tableOfNames names = .ok json) :
    ∃ dtypes, names.mapM DType.ofName = .ok dtypes ∧ checkTable dtypes = .ok () ∧
      json = table dtypes := by
  unfold tableOfNames at h
  split at h
  · cases h
  obtain ⟨dtypes, hd, h⟩ := bind_eq_ok.1 h
  obtain ⟨⟨⟩, hc, h⟩ := bind_eq_ok.1 h
  cases h
  exact ⟨dtypes, hd, hc, rfl⟩

end TorchLean.IR
